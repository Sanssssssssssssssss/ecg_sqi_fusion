"""Visualize whether aligned PTB synthetic morphology matches clean BUT.

This is an external-only diagnostic helper. It compares waveform-computable
SQI/morphology distributions and representative 10s waveforms for clean BUT
and aligned PTB protocol variants.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "event_factorized_sqi_conformer" / "ptb_but_detector_failure_alignment_visuals"
ALIGNED_RUNNER = ANALYSIS_DIR / "run_event_factorized_cross_dataset_aligned_ptb.py"
BUT_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier"

FEATURES = [
    "qrs_visibility",
    "qrs_band_ratio",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "band_15_30",
    "band_30_45",
    "sqi_basSQI",
    "amplitude_entropy",
]
LABELS = ["good", "medium", "bad"]
SOURCE_ORDER = ["BUT train", "BUT test", "PTB v1", "PTB v6 aggressive", "PTB v7 mixed", "PTB v8 visibleQRS", "PTB v9 strongMed", "PTB v10 extremeMed", "PTB v11 styleReplay"]
COLORS = {
    "good": "#4F77BE",
    "medium": "#B69A25",
    "bad": "#C35D3E",
}
SOURCE_COLORS = {
    "BUT train": "#293241",
    "BUT test": "#68758D",
    "PTB v1": "#88B04B",
    "PTB v6 aggressive": "#CF6A87",
    "PTB v7 mixed": "#2A9D8F",
    "PTB v8 visibleQRS": "#6C63FF",
    "PTB v9 strongMed": "#E76F51",
    "PTB v10 extremeMed": "#B5179E",
    "PTB v11 styleReplay": "#0077B6",
}


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("aligned_runner", ALIGNED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {ALIGNED_RUNNER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_signals(path: Path) -> np.ndarray:
    z = np.load(path)
    key = "X" if "X" in z.files else "X_noisy" if "X_noisy" in z.files else z.files[0]
    x = np.asarray(z[key], dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    return x.astype(np.float32)


def robust_wave(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.nanmedian(x))
    q25, q75 = np.nanpercentile(x, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(x) + 1e-6)
    return np.clip((x - med) / scale, -4.5, 4.5).astype(np.float32)


def sample_rows(x: np.ndarray, atlas: pd.DataFrame, label: str, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(atlas["class_name"].astype(str).to_numpy() == label)
    if len(idx) <= n:
        return idx.tolist()
    primitives = quick_features(x[idx])
    arr = primitives.to_numpy(dtype=np.float32)
    center = np.nanmedian(arr, axis=0)
    spread = np.nanpercentile(arr, 75, axis=0) - np.nanpercentile(arr, 25, axis=0)
    spread = np.where(spread > 1e-6, spread, np.nanstd(arr, axis=0) + 1e-6)
    dist = np.sqrt(np.nanmean(((arr - center) / spread) ** 2, axis=1))
    order = np.argsort(dist)
    central = order[: max(n, int(0.75 * len(order)))]
    # Spread picks through the central band so the panel is representative,
    # not just ten nearly identical medoids.
    anchors = np.linspace(0, len(central) - 1, n).round().astype(int)
    picks = [int(idx[int(central[a])]) for a in anchors]
    rng.shuffle(picks)
    return picks


def quick_features(x: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in x:
        y = robust_wave(row)
        dy = np.diff(y, prepend=y[0])
        rows.append(
            {
                "mean_abs": float(np.mean(np.abs(y))),
                "std": float(np.std(y)),
                "diff95": float(np.percentile(np.abs(dy), 95)),
                "flatline": float(np.mean(np.abs(dy) < 0.005)),
            }
        )
    return pd.DataFrame(rows)


def load_source(name: str, protocol: Path, runner: Any, split: str | None = None) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    x = load_signals(protocol / "signals.npz")
    atlas = pd.read_csv(protocol / "original_region_atlas.csv")
    if split is not None:
        mask = atlas["split"].astype(str).eq(split).to_numpy()
        x = x[mask]
        atlas = atlas.loc[mask].reset_index(drop=True)
    if "class_name" not in atlas.columns and "y_class" in atlas.columns:
        atlas["class_name"] = atlas["y_class"].astype(str)
    primitives = runner.GEOM_FEATURES.compute_primitives(x)
    feats = primitives.copy()
    feats["class_name"] = atlas["class_name"].astype(str).to_numpy()
    feats["source"] = name
    if "detector_mix_block" in atlas.columns:
        feats["detector_mix_block"] = atlas["detector_mix_block"].astype(str).to_numpy()
    return x, atlas, feats


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source, label), group in features.groupby(["source", "class_name"], dropna=False):
        for feat in FEATURES:
            vals = pd.to_numeric(group[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "source": source,
                    "class_name": label,
                    "feature": feat,
                    "n": int(len(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "median": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                    "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                }
            )
    return pd.DataFrame(rows)


def plot_feature_medians(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(len(LABELS), len(FEATURES), figsize=(22, 8.3), sharex=False)
    for r, label in enumerate(LABELS):
        for c, feat in enumerate(FEATURES):
            ax = axes[r, c]
            view = summary[(summary["class_name"] == label) & (summary["feature"] == feat)].copy()
            view["source"] = pd.Categorical(view["source"], categories=SOURCE_ORDER, ordered=True)
            view = view.sort_values("source")
            xloc = np.arange(len(view))
            med = view["median"].to_numpy(dtype=float)
            lo = med - view["p10"].to_numpy(dtype=float)
            hi = view["p90"].to_numpy(dtype=float) - med
            colors = [SOURCE_COLORS.get(str(s), "#777777") for s in view["source"].astype(str)]
            ax.errorbar(xloc, med, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, alpha=0.55)
            ax.scatter(xloc, med, s=18, c=colors, zorder=3)
            if r == 0:
                ax.set_title(feat, fontsize=8.5)
            if c == 0:
                ax.set_ylabel(label, fontsize=10, color=COLORS[label], fontweight="bold")
            ax.set_xticks(xloc)
            ax.set_xticklabels([str(s).replace(" ", "\n") for s in view["source"]], rotation=0, fontsize=6)
            ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.55)
    fig.suptitle("BUT vs PTB waveform-computable SQI/morphology distribution", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.94, "Dots are medians; bars show p10-p90. Lower gap means synthetic morphology is closer to the clean BUT protocol.", fontsize=9, color="#5F667A")
    fig.subplots_adjust(top=0.89, left=0.05, right=0.995, bottom=0.09, wspace=0.28, hspace=0.55)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_medium_distributions(features: pd.DataFrame, out: Path) -> None:
    view = features[features["class_name"].astype(str).eq("medium")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    plot_feats = ["qrs_visibility", "qrs_band_ratio", "baseline_step", "non_qrs_diff_p95", "band_15_30", "sqi_basSQI"]
    for ax, feat in zip(axes.ravel(), plot_feats):
        for source in SOURCE_ORDER:
            vals = pd.to_numeric(view.loc[view["source"].eq(source), feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) < 5:
                continue
            sns.kdeplot(vals.astype(float), ax=ax, color=SOURCE_COLORS.get(source, "#777777"), linewidth=1.3, label=source, warn_singular=False)
        ax.set_title(f"medium / {feat}", fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Medium-class distribution: the main PTB/BUT alignment failure mode", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.86, left=0.07, right=0.98, bottom=0.08, wspace=0.25, hspace=0.34)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_contact(sources: dict[str, tuple[np.ndarray, pd.DataFrame]], out: Path, n_each: int = 10) -> pd.DataFrame:
    source_names = ["BUT train", "BUT test", "PTB v10 extremeMed", "PTB v11 styleReplay"]
    fig, axes = plt.subplots(len(source_names) * len(LABELS), n_each, figsize=(20, 14.2), sharex=True, sharey=True)
    t = np.linspace(0, 10, next(iter(sources.values()))[0].shape[1])
    rows: list[dict[str, Any]] = []
    row_idx = 0
    for source in source_names:
        x, atlas = sources[source]
        for label in LABELS:
            picks = sample_rows(x, atlas, label, n_each, seed=3100 + row_idx)
            for col, idx in enumerate(picks):
                ax = axes[row_idx, col]
                ax.plot(t, robust_wave(x[idx]), lw=0.62, color=COLORS[label])
                ax.set_ylim(-4.5, 4.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.45)
                if col == 0:
                    ax.set_ylabel(f"{source}\n{label}", rotation=0, ha="right", va="center", labelpad=44, fontsize=8, color=COLORS[label])
                rows.append({"source": source, "label": label, "rank": col + 1, "row_idx": int(idx)})
            row_idx += 1
    fig.suptitle("Representative 10s waveforms: BUT vs mixed detector-failure PTB", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.955, "Rows are source x label. All traces are robust-normalized per window.", fontsize=9, color="#5F667A")
    fig.subplots_adjust(top=0.925, left=0.09, right=0.995, bottom=0.03, wspace=0.04, hspace=0.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Arial", "sans-serif"]})
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    runner = load_runner()
    protocols = {
        "PTB v1": ANALYSIS_DIR / "event_xds_aligned_v1" / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}",
        "PTB v6 aggressive": ANALYSIS_DIR / "event_xds_aligned_v6_lowqrs_aggressive" / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}",
        "PTB v7 mixed": ANALYSIS_DIR / "event_xds_aligned_v7_mixed_detectorfail" / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}",
        "PTB v8 visibleQRS": ANALYSIS_DIR / "event_xds_aligned_v8_visibleqrs_detailmatch" / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}",
        "PTB v9 strongMed": ANALYSIS_DIR / "event_xds_aligned_v9_strong_visibleqrs_medium" / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}",
        "PTB v10 extremeMed": ANALYSIS_DIR / "event_xds_aligned_v10_butmedium_extreme" / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}",
        "PTB v11 styleReplay": ANALYSIS_DIR / "event_xds_aligned_v11_buttrain_style_replay" / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}",
    }

    sources_for_waves: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    feature_frames: list[pd.DataFrame] = []
    but_train_x, but_train_atlas, but_train_feats = load_source("BUT train", BUT_PROTOCOL, runner, split="train")
    but_test_x, but_test_atlas, but_test_feats = load_source("BUT test", BUT_PROTOCOL, runner, split="test")
    sources_for_waves["BUT train"] = (but_train_x, but_train_atlas)
    sources_for_waves["BUT test"] = (but_test_x, but_test_atlas)
    feature_frames.extend([but_train_feats, but_test_feats])
    for name, protocol in protocols.items():
        if not (protocol / "signals.npz").exists():
            raise FileNotFoundError(f"missing protocol for {name}: {protocol}")
        x, atlas, feats = load_source(name, protocol, runner, split=None)
        if name in {"PTB v7 mixed", "PTB v8 visibleQRS", "PTB v9 strongMed", "PTB v10 extremeMed", "PTB v11 styleReplay"}:
            sources_for_waves[name] = (x, atlas)
        feature_frames.append(feats)
    all_features = pd.concat(feature_frames, ignore_index=True)
    all_features.to_csv(REPORT_DIR / "ptb_but_alignment_features.csv", index=False)

    summary = feature_summary(all_features)
    summary.to_csv(REPORT_DIR / "ptb_but_alignment_feature_summary.csv", index=False)
    plot_feature_medians(summary, REPORT_DIR / "feature_median_p10_p90_by_class.png")
    plot_medium_distributions(all_features, REPORT_DIR / "medium_distribution_kde.png")
    selections = plot_waveform_contact(sources_for_waves, REPORT_DIR / "but_vs_ptb_v7_waveform_contact_sheet.png")
    selections.to_csv(REPORT_DIR / "waveform_contact_sheet_selection.csv", index=False)

    counts = (
        all_features.groupby(["source", "class_name"]).size().reset_index(name="n").sort_values(["source", "class_name"])
    )
    counts.to_csv(REPORT_DIR / "source_class_counts.csv", index=False)
    report = [
        "# PTB/BUT Detector-Failure Alignment Visuals",
        "",
        "These plots compare clean BUT train/test against old aligned PTB, aggressive low-QRS PTB, and the new mixed detector-failure PTB variant.",
        "",
        "## Class Counts",
        "",
        md_table(counts),
        "",
        "## Feature Distribution",
        "",
        f"![feature medians]({REPORT_DIR / 'feature_median_p10_p90_by_class.png'})",
        "",
        "## Medium-Class KDE",
        "",
        f"![medium kde]({REPORT_DIR / 'medium_distribution_kde.png'})",
        "",
        "## Representative Waveforms",
        "",
        f"![waveforms]({REPORT_DIR / 'but_vs_ptb_v7_waveform_contact_sheet.png'})",
        "",
        "## Files",
        "",
        f"- Feature rows: `{REPORT_DIR / 'ptb_but_alignment_features.csv'}`",
        f"- Feature summary: `{REPORT_DIR / 'ptb_but_alignment_feature_summary.csv'}`",
        f"- Waveform selection: `{REPORT_DIR / 'waveform_contact_sheet_selection.csv'}`",
    ]
    (REPORT_DIR / "ptb_but_detector_failure_alignment_visuals.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_DIR / "ptb_but_detector_failure_alignment_visuals.md"),
                "feature_summary": str(REPORT_DIR / "ptb_but_alignment_feature_summary.csv"),
                "figures": [
                    str(REPORT_DIR / "feature_median_p10_p90_by_class.png"),
                    str(REPORT_DIR / "medium_distribution_kde.png"),
                    str(REPORT_DIR / "but_vs_ptb_v7_waveform_contact_sheet.png"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
