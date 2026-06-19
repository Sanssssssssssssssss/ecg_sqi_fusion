"""Compare original-BUT errors exchanged between waveform-only candidates.

External-only diagnostic.  This script does not train or select with BUT.  It
uses saved report-only predictions to identify which BUT rows one waveform
Transformer fixes/regresses relative to another, then summarizes feature and
waveform patterns.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
ANALYSIS_DIR = Path(__file__).parent
RUNNER_PATH = ANALYSIS_DIR / "run_waveform_geometry_student.py"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
AUDIT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "original_candidate_error_audit"
ATLAS_PATH = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"

FEATURES = [
    "rms",
    "std",
    "ptp_p99_p01",
    "mean_abs",
    "low_amp_ratio",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "baseline_step",
    "diff_abs_p95",
    "detail_instability",
    "lf_ratio",
    "hf_ratio",
    "qrs_band_ratio",
    "spectral_entropy",
    "qrs_peak_count",
    "spurious_peak_density",
    "qrs_prom_median",
    "qrs_prom_p90",
    "qrs_slope_median",
    "periodicity",
    "qrs_visibility",
    "fatal_or_score",
    "medium_detail_unreliable_score",
    "sqi_iSQI",
    "sqi_bSQI",
    "sqi_pSQI",
    "sqi_sSQI",
    "sqi_kSQI",
    "sqi_fSQI",
    "sqi_basSQI",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "zero_crossing_rate",
    "diff_zero_crossing_rate",
    "sample_entropy_proxy",
    "amplitude_entropy",
    "higuchi_fd_proxy",
    "template_corr",
    "detector_count_std",
    "detector_agreement",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "band_0p3_1",
    "band_1_5",
    "band_5_15",
    "band_15_30",
    "band_30_45",
    "wavelet_e0",
    "wavelet_e1",
    "wavelet_e2",
    "wavelet_e3",
    "wavelet_e4",
    "pc1",
    "pc2",
    "pc3",
    "pca_margin",
    "knn_label_purity",
    "boundary_confidence",
    "region_confidence",
]


def load_runner():
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_predictions(candidate: str) -> pd.DataFrame:
    path = AUDIT_DIR / candidate / "original_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    keep = [
        "row",
        "split",
        "region",
        "true",
        "pred_raw",
        "pred_badcal",
        "correct_raw",
        "correct_badcal",
        "prob_good",
        "prob_medium",
        "prob_bad",
        "is_test",
        "is_bad_core",
        "is_bad_outlier",
    ]
    return df[keep].copy()


def join_atlas(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    atlas = pd.read_csv(ATLAS_PATH).reset_index(drop=True)
    base_name = next(iter(preds))
    base = preds[base_name].copy()
    if len(base) != len(atlas):
        raise RuntimeError(f"prediction/atlas row mismatch: {len(base)} vs {len(atlas)}")
    for col in FEATURES + ["record_id", "subject_id", "ambiguous_type", "original_region"]:
        if col in atlas.columns and col not in base.columns:
            base[col] = atlas[col].values
    for name, df in preds.items():
        if len(df) != len(base):
            raise RuntimeError(f"{name} length mismatch")
        for col in ["pred_raw", "correct_raw", "prob_good", "prob_medium", "prob_bad"]:
            base[f"{name}__{col}"] = df[col].values
    return base


def exchange_labels(df: pd.DataFrame, reference: str, candidate: str) -> pd.Series:
    ref_ok = df[f"{reference}__correct_raw"].astype(bool)
    cand_ok = df[f"{candidate}__correct_raw"].astype(bool)
    labels = np.full(len(df), "both_wrong", dtype=object)
    labels[ref_ok & cand_ok] = "both_correct"
    labels[~ref_ok & cand_ok] = "candidate_fixes_reference"
    labels[ref_ok & ~cand_ok] = "candidate_regresses_reference"
    return pd.Series(labels, index=df.index)


def summarize_exchange(df: pd.DataFrame, reference: str, candidate: str) -> pd.DataFrame:
    work = df[df["is_test"].astype(bool)].copy()
    work["exchange"] = exchange_labels(work, reference, candidate)
    rows = []
    for keys, sub in work.groupby(["exchange", "true", "region"], dropna=False):
        rows.append(
            {
                "exchange": keys[0],
                "true": keys[1],
                "region": keys[2],
                "n": len(sub),
                "ref_pred_top": sub[f"{reference}__pred_raw"].value_counts().idxmax() if len(sub) else "",
                "cand_pred_top": sub[f"{candidate}__pred_raw"].value_counts().idxmax() if len(sub) else "",
                "record_top": sub["record_id"].value_counts().idxmax() if len(sub) else "",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["exchange", "n"], ascending=[True, False])


def feature_gap(df: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series, label_a: str, label_b: str) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        if feat not in df.columns:
            continue
        a = pd.to_numeric(df.loc[mask_a, feat], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[mask_b, feat], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        pooled = np.sqrt((float(a.var(ddof=0)) + float(b.var(ddof=0))) / 2.0) + 1e-12
        rows.append(
            {
                "feature": feat,
                f"{label_a}_median": float(a.median()),
                f"{label_b}_median": float(b.median()),
                "median_delta_a_minus_b": float(a.median() - b.median()),
                "abs_standardized_delta": abs(float(a.mean() - b.mean()) / pooled),
                f"{label_a}_n": int(len(a)),
                f"{label_b}_n": int(len(b)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_standardized_delta", ascending=False)


def plot_waveform_panels(df: pd.DataFrame, reference: str, candidate: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    runner = load_runner()
    # Use the same normalized original waveform loader as the audit scripts.  The
    # channel choice only affects displayed channels, not the saved predictions.
    base_train = runner.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = runner.ARCH.FeatureNorm(mean=base_train.mean, std=base_train.std)
    ds = runner.ARCH.OriginalWaveDataset(norm, "robust3")

    work = df[df["is_test"].astype(bool)].copy()
    work["exchange"] = exchange_labels(work, reference, candidate)
    picks = []
    selectors = [
        ("fix bad outlier", "candidate_fixes_reference", "bad", "outlier_low_confidence"),
        ("fix medium", "candidate_fixes_reference", "medium", None),
        ("fix good", "candidate_fixes_reference", "good", None),
        ("regress bad", "candidate_regresses_reference", "bad", None),
        ("regress medium", "candidate_regresses_reference", "medium", None),
        ("regress good", "candidate_regresses_reference", "good", None),
    ]
    for title, ex, true, region in selectors:
        sub = work[(work["exchange"] == ex) & (work["true"] == true)]
        if region is not None:
            sub = sub[sub["region"] == region]
        if len(sub) == 0:
            continue
        # Use the most confident changed rows so the panel is not visually random.
        prob_cols = [f"{candidate}__prob_good", f"{candidate}__prob_medium", f"{candidate}__prob_bad"]
        sub = sub.assign(conf=sub[prob_cols].max(axis=1)).sort_values("conf", ascending=False)
        for _, row in sub.head(2).iterrows():
            picks.append((title, int(row["row"])))
    if not picks:
        out_path.with_suffix(".txt").write_text("No exchange rows selected.", encoding="utf-8")
        return
    cols = 4
    rows = int(math.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 2.6 * rows), dpi=170, squeeze=False)
    t = np.arange(ds.x.shape[-1])
    for ax, (title, idx) in zip(axes.ravel(), picks):
        x = ds.x[idx]
        scale = np.nanpercentile(np.abs(x), 95) + 1e-6
        for ch, color in enumerate(["#222222", "#2b8cbe", "#de2d26"][: x.shape[0]]):
            ax.plot(t, x[ch] / scale + ch * 2.0, lw=0.75, color=color, alpha=0.9)
        row = df.loc[df["row"] == idx].iloc[0]
        ax.set_title(
            f"{title}: {row['true']} | {reference}:{row[f'{reference}__pred_raw']} -> {candidate}:{row[f'{candidate}__pred_raw']}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(picks) :]:
        ax.axis("off")
    fig.suptitle(f"BUT original-test error exchange: {candidate} vs {reference}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def render_report(
    out_dir: Path,
    reference: str,
    candidates: list[str],
    joined: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    gaps: dict[str, pd.DataFrame],
) -> str:
    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Waveform Candidate Error Exchange",
        "",
        "Scope: BUT original test split only. This is report-only analysis; no BUT rows are used for training or selection.",
        "",
        f"Reference candidate: `{reference}`",
        "",
    ]
    for cand in candidates:
        work = joined[joined["is_test"].astype(bool)].copy()
        ex = exchange_labels(work, reference, cand)
        counts = ex.value_counts().to_dict()
        lines.extend(
            [
                f"## {cand}",
                "",
                "| exchange | n |",
                "|---|---:|",
            ]
        )
        for key in ["candidate_fixes_reference", "candidate_regresses_reference", "both_correct", "both_wrong"]:
            lines.append(f"| {key} | {int(counts.get(key, 0))} |")
        lines.extend(["", "Largest class/region exchanges:", ""])
        lines.append(md_table(summaries[cand].head(12)))
        lines.extend(["", "Top feature differences: fixed p20 errors vs regressed p20 correct rows", ""])
        lines.append(md_table(gaps[cand].head(15)))
        panel = out_dir / f"{cand}_exchange_waveforms.png"
        if panel.exists():
            lines.extend(["", f"![{cand} exchange waveforms]({panel})", ""])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="predtop20_sqiquery_subject111_impulsebad_dual_p20")
    parser.add_argument(
        "--candidates",
        default="p20_sqiquery_primctx_v5_light,p20_sqiquery_primctx_v5_badguard,predtop20_eventqrs_impulsebad_dual_p20_qrsheavy",
    )
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    names = [args.reference] + candidates
    preds = {name: load_predictions(name) for name in names}
    joined = join_atlas(preds)

    out_dir = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "waveform_candidate_exchange"
    out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_dir / "joined_original_predictions_with_atlas.csv", index=False)

    summaries: dict[str, pd.DataFrame] = {}
    gaps: dict[str, pd.DataFrame] = {}
    for cand in candidates:
        summaries[cand] = summarize_exchange(joined, args.reference, cand)
        summaries[cand].to_csv(out_dir / f"{cand}_exchange_summary.csv", index=False)
        test = joined[joined["is_test"].astype(bool)].copy()
        ex = exchange_labels(test, args.reference, cand)
        fix = ex == "candidate_fixes_reference"
        regress = ex == "candidate_regresses_reference"
        gaps[cand] = feature_gap(test, fix, regress, "fixes", "regresses")
        gaps[cand].to_csv(out_dir / f"{cand}_fix_vs_regress_feature_gap.csv", index=False)
        plot_waveform_panels(joined, args.reference, cand, out_dir / f"{cand}_exchange_waveforms.png")

    report = render_report(out_dir, args.reference, candidates, joined, summaries, gaps)
    (out_dir / "waveform_candidate_exchange_report.md").write_text(report, encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
