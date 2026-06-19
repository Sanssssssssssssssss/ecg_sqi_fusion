from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


THIS = Path(__file__)
ANALYSIS_DIR = THIS.parent
OUT_ROOT = ANALYSIS_DIR.parent.parent
REPORT_DIR = Path(
    str(ANALYSIS_DIR).replace(
        "\\outputs\\external_benchmarks\\",
        "\\reports\\external_benchmarks\\",
    )
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ROOT = OUT_ROOT / "runs" / "waveform_geometry_student" / "N17043_gm_probe" / "search"
DEFAULT_CANDIDATES = [
    RUN_ROOT / "predtop20_sqiquery_boundary_pretrain" / "ckpt_best.pt",
    RUN_ROOT / "predtop20_sqiquery_balancedguard_pretrain" / "ckpt_best.pt",
    RUN_ROOT / "predtop20_sqiquery_badguardlite_pretrain" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_auxprimitive_boundary" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_auxprimitive_badguard" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_intermittentbad_balanced" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_intermittentbad_stress" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_intermitbad_auxmask_balanced" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_intermitbad_auxmask_stress" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_qrsdetector_auxmask_balanced" / "ckpt_best.pt",
    RUN_ROOT / "sqiquery_qrsdetector_auxmask_stress" / "ckpt_best.pt",
]

FOCUS_FEATURES = [
    "pc1",
    "pc2",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "qrs_visibility",
    "qrs_band_ratio",
    "template_corr",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "sqi_bSQI",
    "sqi_basSQI",
    "low_amp_ratio",
    "amplitude_entropy",
    "mean_abs",
]


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", THIS.with_name("run_waveform_geometry_student.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run_waveform_geometry_student.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(mod: Any, ckpt_path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt["candidate_config"])
    in_ch = 3 if str(cfg.get("channels", "robust3")) == "robust3" else 1
    model = mod.GeometryStudent(cfg, in_ch=in_ch).to(device)
    if ckpt.get("teacher_prototypes") is not None:
        model.set_teacher_prototypes(ckpt["teacher_prototypes"])
    if ckpt.get("waveform_atlas_prototypes") is not None:
        model.set_waveform_atlas(
            ckpt["waveform_atlas_prototypes"],
            ckpt["waveform_atlas_mean"],
            ckpt["waveform_atlas_std"],
        )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(mod: Any, model: Any, ds: Any, device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    aux_pred = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            out = model(x)
            probs.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
            aux_pred.append(out["aux_pred"].cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(aux_pred, axis=0)


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"].to_numpy()
    split = df["split"].astype(str).to_numpy()
    region = df["region"].astype(str).to_numpy()
    group = np.full(len(df), "other", dtype=object)
    group[(split == "test") & (y == 2) & (region == "outlier_low_confidence")] = "test_bad_outlier_stress"
    group[(split == "test") & (y == 2) & (region == "near_bad_boundary")] = "test_bad_core_nearboundary"
    group[(split != "test") & (y == 2) & (region == "right_bad_island")] = "non_test_right_bad_island"
    group[(split != "test") & (y == 2) & (region == "outlier_low_confidence")] = "non_test_bad_outlier"
    group[(split == "test") & (y != 2)] = "test_nonbad"
    df = df.copy()
    df["analysis_group"] = group
    return df


def summarize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby("analysis_group"):
        if group == "other" or sub.empty:
            continue
        row = {
            "group": group,
            "n": int(len(sub)),
            "bad_recall_raw": float((sub["pred_raw"] == 2).mean()) if (sub["y"] == 2).all() else np.nan,
            "bad_recall_badcal": float((sub["pred_badcal"] == 2).mean()) if (sub["y"] == 2).all() else np.nan,
            "bad_prob_mean": float(sub["p_bad"].mean()),
            "medium_prob_mean": float(sub["p_medium"].mean()),
            "good_prob_mean": float(sub["p_good"].mean()),
        }
        for name in features:
            if name in sub:
                row[f"{name}_median"] = float(sub[name].median())
                row[f"{name}_p10"] = float(sub[name].quantile(0.10))
                row[f"{name}_p90"] = float(sub[name].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def top_group_gaps(df: pd.DataFrame, features: list[str], a: str, b: str) -> pd.DataFrame:
    aa = df[df["analysis_group"] == a]
    bb = df[df["analysis_group"] == b]
    rows = []
    for name in features:
        if name not in df or aa.empty or bb.empty:
            continue
        av = aa[name].to_numpy(dtype=float)
        bv = bb[name].to_numpy(dtype=float)
        pooled = np.nanstd(np.concatenate([av, bv]))
        if not np.isfinite(pooled) or pooled < 1e-8:
            effect = np.nan
        else:
            effect = (np.nanmedian(av) - np.nanmedian(bv)) / pooled
        rows.append(
            {
                "feature": name,
                "group_a": a,
                "group_b": b,
                "median_a": float(np.nanmedian(av)),
                "median_b": float(np.nanmedian(bv)),
                "robust_effect": float(effect),
                "abs_effect": float(abs(effect)) if np.isfinite(effect) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_effect", ascending=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 18) -> str:
    if df.empty:
        return "_No rows._"
    cols = ["feature", "group_a", "group_b", "median_a", "median_b", "robust_effect"]
    view = df[[c for c in cols if c in df.columns]].head(max_rows).copy()
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def plot_waveform_examples(x: np.ndarray, df: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    groups = [
        ("test_bad_outlier_stress", "test bad stress"),
        ("test_bad_core_nearboundary", "test bad core"),
        ("non_test_right_bad_island", "non-test right island"),
        ("test_nonbad", "test non-bad"),
    ]
    picks = []
    for group, label in groups:
        sub = df[df["analysis_group"] == group].copy()
        if group == "test_bad_outlier_stress":
            sub = sub.sort_values("p_bad", ascending=True)
        elif group == "test_nonbad":
            sub = sub.sort_values("p_bad", ascending=False)
        else:
            sub = sub.sort_values("p_bad", ascending=False)
        for _, row in sub.head(3).iterrows():
            picks.append((int(row["idx"]), label, row))
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, max(3, 1.3 * len(picks))), sharex=True)
    if len(picks) == 1:
        axes = [axes]
    t = np.arange(x.shape[-1])
    for ax, (idx, label, row) in zip(axes, picks):
        wave = x[idx]
        offsets = np.arange(wave.shape[0]) * 5.0
        for ch in range(wave.shape[0]):
            ax.plot(t, wave[ch] + offsets[ch], lw=0.8, label=f"ch{ch}" if idx == picks[0][0] else None)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_title(
            f"idx={idx} split={row['split']} region={row['region']} y={int(row['y'])} "
            f"pred={int(row['pred_badcal'])} p_bad={row['p_bad']:.3f} "
            f"qrs_vis={row.get('qrs_visibility', np.nan):.3f} bas={row.get('sqi_basSQI', np.nan):.3f}",
            fontsize=8,
        )
    axes[0].legend(loc="upper right", ncol=3, fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_candidate(mod: Any, ckpt_path: Path, device: torch.device, original_ds: Any) -> dict[str, Any]:
    name = ckpt_path.parent.name
    model, ckpt = load_model(mod, ckpt_path, device)
    probs, aux_pred = predict(mod, model, original_ds, device)
    pred_raw = probs.argmax(axis=1)
    threshold = float(ckpt.get("bad_threshold_trainval", np.nan))
    pred_badcal = mod.ARCH.apply_bad_threshold(probs, threshold) if np.isfinite(threshold) else pred_raw

    df = pd.DataFrame(
        {
            "idx": np.arange(len(original_ds.y)),
            "split": original_ds.split.astype(str),
            "region": original_ds.region.astype(str),
            "y": original_ds.y.astype(int),
            "pred_raw": pred_raw.astype(int),
            "pred_badcal": pred_badcal.astype(int),
            "p_good": probs[:, 0],
            "p_medium": probs[:, 1],
            "p_bad": probs[:, 2],
        }
    )
    feature_names = list(mod.FEATURE_COLUMNS)
    for j, fname in enumerate(feature_names):
        df[fname] = original_ds.aux[:, j]
        df[f"pred_{fname}"] = aux_pred[:, j]
    df = add_group_columns(df)
    features = [f for f in FOCUS_FEATURES if f in df]
    out_dir = ANALYSIS_DIR / "original_bad_stress_failure" / name
    rep_dir = REPORT_DIR / "original_bad_stress_failure" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "original_bad_stress_rows.csv"
    df.to_csv(rows_path, index=False)
    summary = summarize_features(df, features)
    summary_path = out_dir / "original_bad_stress_group_summary.csv"
    summary.to_csv(summary_path, index=False)
    gaps = pd.concat(
        [
            top_group_gaps(df, features, "test_bad_outlier_stress", "test_bad_core_nearboundary"),
            top_group_gaps(df, features, "test_bad_outlier_stress", "non_test_right_bad_island"),
            top_group_gaps(df, features, "test_bad_outlier_stress", "test_nonbad"),
        ],
        ignore_index=True,
    )
    gaps_path = out_dir / "original_bad_stress_feature_gaps.csv"
    gaps.to_csv(gaps_path, index=False)
    fig_path = out_dir / "original_bad_stress_waveform_examples.png"
    plot_waveform_examples(original_ds.x, df, fig_path, f"{name}: original bad-stress failure examples")
    # Mirror compact report artifacts.
    summary.to_csv(rep_dir / summary_path.name, index=False)
    gaps.to_csv(rep_dir / gaps_path.name, index=False)
    if fig_path.exists():
        (rep_dir / fig_path.name).write_bytes(fig_path.read_bytes())

    test_stress = df[df["analysis_group"] == "test_bad_outlier_stress"]
    test_core = df[df["analysis_group"] == "test_bad_core_nearboundary"]
    notes = {
        "candidate": name,
        "checkpoint": str(ckpt_path),
        "n_test_bad_outlier_stress": int(len(test_stress)),
        "n_test_bad_core_nearboundary": int(len(test_core)),
        "bad_outlier_stress_recall_raw": float((test_stress["pred_raw"] == 2).mean()) if len(test_stress) else None,
        "bad_outlier_stress_recall_badcal": float((test_stress["pred_badcal"] == 2).mean()) if len(test_stress) else None,
        "bad_core_recall_raw": float((test_core["pred_raw"] == 2).mean()) if len(test_core) else None,
        "bad_core_recall_badcal": float((test_core["pred_badcal"] == 2).mean()) if len(test_core) else None,
        "top_gap_rows": gaps.head(12).to_dict(orient="records"),
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
        "gaps_csv": str(gaps_path),
        "figure": str(fig_path),
    }
    (out_dir / "original_bad_stress_summary.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    (rep_dir / "original_bad_stress_summary.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    lines = [
        f"# Original Bad-Stress Failure: {name}",
        "",
        "Post-hoc report-only analysis. Original BUT is not used for training or model selection.",
        "",
        f"- test bad outlier stress: n={notes['n_test_bad_outlier_stress']}, raw recall={notes['bad_outlier_stress_recall_raw']:.4f}, badcal recall={notes['bad_outlier_stress_recall_badcal']:.4f}",
        f"- test bad core/near-boundary: n={notes['n_test_bad_core_nearboundary']}, raw recall={notes['bad_core_recall_raw']:.4f}, badcal recall={notes['bad_core_recall_badcal']:.4f}",
        "",
        "## Top Feature Gaps",
        "",
        markdown_table(gaps, max_rows=18),
        "",
        f"![waveform examples]({fig_path})",
        "",
    ]
    (out_dir / "original_bad_stress_report.md").write_text("\n".join(lines), encoding="utf-8")
    (rep_dir / "original_bad_stress_report.md").write_text("\n".join(lines), encoding="utf-8")
    return notes


def main() -> None:
    mod = load_runner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = mod.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = mod.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = mod.ARCH.OriginalWaveDataset(norm, "robust3")
    summaries = []
    for ckpt in DEFAULT_CANDIDATES:
        if ckpt.exists():
            summaries.append(analyze_candidate(mod, ckpt, device, original_ds))
    out_path = ANALYSIS_DIR / "original_bad_stress_failure_summary.json"
    rep_path = REPORT_DIR / "original_bad_stress_failure_summary.json"
    payload = {"summaries": summaries}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rep_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
