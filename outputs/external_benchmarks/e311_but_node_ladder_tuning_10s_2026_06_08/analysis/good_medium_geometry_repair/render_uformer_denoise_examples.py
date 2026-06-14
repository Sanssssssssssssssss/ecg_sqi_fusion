"""Render UFormer denoising examples for the current N17043 experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_OUT = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
ANALYSIS_REPORT = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
GEOM_SCRIPT = ANALYSIS_OUT / "uformer_geometry_branch_experiment.py"
VARIANT_ID = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
CKPT_PATH = OUT_ROOT / "runs" / "uformer_geometry_nodecal" / "N17043_gm_probe" / f"{VARIANT_ID}_geom_tabular_dualbad_internal" / "ckpt_best.pt"
SIGNALS_PATH = ROOT / "outputs" / "external_benchmarks" / "e311_but_protocol_adaptation_2026_06_03" / "protocols" / "p1_current_10s_center" / "signals.npz"


def load_geom():
    spec = importlib.util.spec_from_file_location("uformer_geometry_branch_experiment", GEOM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GEOM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORT.mkdir(parents=True, exist_ok=True)
    geom = load_geom()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = geom.load_trained_model({"checkpoint": str(CKPT_PATH)}, device)
    model.eval()
    mean = np.asarray(ckpt["feature_train_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_train_std"], dtype=np.float32)
    columns = list(ckpt["feature_columns"])

    synth_png = render_synthetic(model, mean, std, columns, device)
    orig_png = render_original(model, mean, std, columns, device)
    report = ANALYSIS_REPORT / "uformer_denoise_examples_report.md"
    report.write_text(
        "\n".join(
            [
                "# UFormer Denoise Examples",
                "",
                "The current best classifier checkpoint is `fusion_mode=tabular_only`, so waveform features are not used for classification. These figures show the denoiser output for interpretation only.",
                "",
                f"![Synthetic denoise examples]({synth_png.as_posix()})",
                "",
                f"![Original denoise examples]({orig_png.as_posix()})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (ANALYSIS_OUT / report.name).write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print(report)
    print(synth_png)
    print(orig_png)


def render_synthetic(model, mean: np.ndarray, std: np.ndarray, columns: list[str], device: torch.device) -> Path:
    data_dir = OUT_ROOT / "synthetic_variants" / VARIANT_ID / "datasets"
    labels = pd.read_csv(data_dir / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    noisy = np.load(data_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    clean = np.load(data_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    feat = np.load(data_dir / "tabular_features.npz")["features"].astype(np.float32)
    rows = []
    for cls in ["good", "medium", "bad"]:
        subset = labels[(labels["split"].astype(str) == "test") & (labels["y_class"].astype(str) == cls)]
        if subset.empty:
            subset = labels[labels["y_class"].astype(str) == cls]
        take = subset.head(3)["idx"].astype(int).tolist()
        rows.extend(take)
    x = noisy[rows]
    c = clean[rows]
    tab = ((feat[rows] - mean) / std).astype(np.float32)
    den = denoise(model, x, tab, device)
    titles = [labels.loc[labels["idx"] == i, "y_class"].iloc[0] for i in rows]
    out = ANALYSIS_REPORT / "uformer_synthetic_noisy_denoised_clean_examples.png"
    plot_three(x, den, c, titles, out, "Synthetic: noisy vs UFormer denoised vs clean")
    copy = ANALYSIS_OUT / out.name
    copy.write_bytes(out.read_bytes())
    return out


def render_original(model, mean: np.ndarray, std: np.ndarray, columns: list[str], device: torch.device) -> Path:
    atlas = pd.read_csv(OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv")
    signals = np.load(SIGNALS_PATH)["X"].astype(np.float32)
    if signals.ndim == 3:
        sig = signals[:, 0, :]
    else:
        sig = signals
    for col in columns:
        if col not in atlas.columns:
            atlas[col] = 0.0
    raw_features = atlas[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    signal_rows = []
    feature_rows = []
    for cls in ["good", "medium", "bad"]:
        subset = atlas[(atlas["split"].astype(str) == "test") & (atlas["class_name"].astype(str) == cls)]
        if subset.empty:
            subset = atlas[atlas["class_name"].astype(str) == cls]
        chosen = subset.head(3)
        signal_rows.extend(chosen["idx"].astype(int).tolist())
        feature_rows.extend(chosen.index.astype(int).tolist())
    x = sig[signal_rows]
    tab = ((raw_features[feature_rows] - mean) / std).astype(np.float32)
    den = denoise(model, x, tab, device)
    titles = [atlas.iloc[i]["class_name"] for i in feature_rows]
    out = ANALYSIS_REPORT / "uformer_original_raw_denoised_examples.png"
    plot_two(x, den, titles, out, "Original BUT: raw vs UFormer denoised")
    copy = ANALYSIS_OUT / out.name
    copy.write_bytes(out.read_bytes())
    return out


@torch.no_grad()
def denoise(model, x: np.ndarray, tab: np.ndarray, device: torch.device) -> np.ndarray:
    tx = torch.from_numpy(x[:, None, :].astype(np.float32)).to(device)
    tt = torch.from_numpy(tab.astype(np.float32)).to(device)
    out = model(tx, tt)["denoise"]
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    if out.ndim == 3:
        out = out[:, 0, :]
    return out.astype(np.float32)


def plot_three(noisy: np.ndarray, den: np.ndarray, clean: np.ndarray, titles: list[str], out: Path, suptitle: str) -> None:
    t = np.arange(noisy.shape[1]) / 125.0
    n = len(titles)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.45 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, y0, y1, y2, title in zip(axes, noisy, den, clean, titles):
        ax.plot(t, y0, color="#9aa0a6", lw=0.8, alpha=0.75, label="noisy")
        ax.plot(t, y1, color="#1f77b4", lw=0.95, label="UFormer denoised")
        ax.plot(t, y2, color="#2ca02c", lw=0.8, alpha=0.75, label="clean")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Seconds")
    fig.suptitle(suptitle, y=0.995, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_two(raw: np.ndarray, den: np.ndarray, titles: list[str], out: Path, suptitle: str) -> None:
    t = np.arange(raw.shape[1]) / 125.0
    n = len(titles)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.45 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, y0, y1, title in zip(axes, raw, den, titles):
        ax.plot(t, y0, color="#9aa0a6", lw=0.8, alpha=0.75, label="raw")
        ax.plot(t, y1, color="#1f77b4", lw=0.95, label="UFormer denoised")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Seconds")
    fig.suptitle(suptitle, y=0.995, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
