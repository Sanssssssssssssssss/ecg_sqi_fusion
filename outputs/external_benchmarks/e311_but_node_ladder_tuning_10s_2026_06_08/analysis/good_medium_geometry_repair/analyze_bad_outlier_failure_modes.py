"""Analyze which original bad outliers waveform models catch or miss.

Report-only.  Original BUT is never used for training or selection.  This script
uses original-test predictions plus teacher/primitive diagnostics to explain
why bad-outlier recall remains low.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()
BASE = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair"
RUNNER = BASE / "run_waveform_geometry_student.py"
HELPER = BASE / "analyze_primitive_domain_gap.py"
PRED_PATH = BASE / "waveform_bad_score_original_test_predictions.csv"
REPORT_DIR = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
FOCUS_CANDIDATE = "qrsbank_highamp_mix_patch_guard"
FOCUS_THRESHOLD = 0.03


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize_gap(df: pd.DataFrame, a: str, b: str, features: list[str]) -> pd.DataFrame:
    rows = []
    aa = df[df["bucket"] == a]
    bb = df[df["bucket"] == b]
    for feat in features:
        if feat not in df.columns or aa.empty or bb.empty:
            continue
        av = pd.to_numeric(aa[feat], errors="coerce").dropna().to_numpy(dtype=float)
        bv = pd.to_numeric(bb[feat], errors="coerce").dropna().to_numpy(dtype=float)
        if len(av) == 0 or len(bv) == 0:
            continue
        rows.append(
            {
                "feature": feat,
                f"{a}_mean": float(np.mean(av)),
                f"{b}_mean": float(np.mean(bv)),
                "abs_mean_gap": abs(float(np.mean(av)) - float(np.mean(bv))),
                f"{a}_p50": float(np.quantile(av, 0.50)),
                f"{b}_p50": float(np.quantile(bv, 0.50)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_mean_gap", ascending=False)


def main() -> None:
    mod = load_module(RUNNER, "waveform_geometry_student_runner")
    helper = load_module(HELPER, "primitive_gap_helper")
    train_ds = mod.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = mod.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original = mod.ARCH.OriginalWaveDataset(norm, "robust3")
    preds = pd.read_csv(PRED_PATH)
    preds = preds[preds["candidate"] == FOCUS_CANDIDATE].copy()
    if preds.empty:
        raise RuntimeError(f"candidate {FOCUS_CANDIDATE} not found in {PRED_PATH}")

    orig_stats = helper.collect_stats(mod, original)
    train_stats = helper.collect_stats(mod, train_ds)
    mu = train_stats.mean(axis=0)
    sigma = np.where(train_stats.std(axis=0) < 1e-6, 1.0, train_stats.std(axis=0))
    zstats = (orig_stats - mu[None, :]) / sigma[None, :]
    primitive_cols = helper.primitive_names(train_ds.x.shape[1])
    primitive_df = pd.DataFrame(zstats, columns=[f"prim_{x}" for x in primitive_cols])

    aux_df = pd.DataFrame(original.aux, columns=mod.FEATURE_COLUMNS)
    merged = pd.concat([preds.reset_index(drop=True), aux_df.iloc[preds["orig_row"].to_numpy()].reset_index(drop=True), primitive_df.iloc[preds["orig_row"].to_numpy()].reset_index(drop=True)], axis=1)
    pred_bad_focus = merged["prob_bad"].to_numpy(dtype=float) >= FOCUS_THRESHOLD
    is_bad_outlier = (merged["true"] == "bad") & (merged["region"] == "outlier_low_confidence")
    is_nonbad = merged["true"] != "bad"
    buckets = np.full(len(merged), "other", dtype=object)
    buckets[is_bad_outlier & pred_bad_focus] = "bad_outlier_caught"
    buckets[is_bad_outlier & ~pred_bad_focus] = "bad_outlier_missed"
    buckets[is_nonbad & pred_bad_focus] = "nonbad_false_bad"
    buckets[is_nonbad & ~pred_bad_focus] = "nonbad_kept"
    merged["bucket"] = buckets

    teacher_features = [
        "pc1",
        "pc2",
        "pc3",
        "pca_margin",
        "boundary_confidence",
        "knn_label_purity",
        "qrs_visibility",
        "detector_agreement",
        "baseline_step",
        "flatline_ratio",
        "non_qrs_diff_p95",
        "amplitude_entropy",
        "qrs_prom_p90",
        "qrs_band_ratio",
        "template_corr",
        "sqi_bSQI",
        "sqi_basSQI",
    ]
    primitive_features = [
        c
        for c in merged.columns
        if c.startswith("prim_ch") and any(k in c for k in ["qrs", "stress_baseline", "stress_detail", "diff_tail", "contact", "flat"])
    ]
    counts = merged["bucket"].value_counts().rename_axis("bucket").reset_index(name="n")
    teacher_gap = summarize_gap(merged, "bad_outlier_caught", "bad_outlier_missed", teacher_features)
    falsebad_gap = summarize_gap(merged, "bad_outlier_caught", "nonbad_false_bad", teacher_features)
    primitive_gap = summarize_gap(merged, "bad_outlier_caught", "bad_outlier_missed", primitive_features).head(40)
    falsebad_primitive_gap = summarize_gap(merged, "bad_outlier_caught", "nonbad_false_bad", primitive_features).head(40)

    for name, df in [
        ("bad_outlier_failure_bucket_counts.csv", counts),
        ("bad_outlier_caught_vs_missed_teacher_gap.csv", teacher_gap),
        ("bad_outlier_caught_vs_falsebad_teacher_gap.csv", falsebad_gap),
        ("bad_outlier_caught_vs_missed_primitive_gap.csv", primitive_gap),
        ("bad_outlier_caught_vs_falsebad_primitive_gap.csv", falsebad_primitive_gap),
    ]:
        out = BASE / name
        df.to_csv(out, index=False)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / name).write_bytes(out.read_bytes())

    lines = [
        "# Bad Outlier Failure Modes",
        "",
        f"Candidate `{FOCUS_CANDIDATE}`, report-only threshold `{FOCUS_THRESHOLD}`.",
        "",
        "## Bucket Counts",
        "",
        counts.to_csv(index=False),
        "",
        "## Caught vs Missed: Teacher/Geometry Features",
        "",
        teacher_gap.head(25).to_csv(index=False),
        "",
        "## Caught vs False-Bad: Teacher/Geometry Features",
        "",
        falsebad_gap.head(25).to_csv(index=False),
        "",
        "## Caught vs Missed: Waveform Primitive Features",
        "",
        primitive_gap.head(25).to_csv(index=False),
    ]
    report = "\n".join(lines)
    (BASE / "bad_outlier_failure_modes_report.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "bad_outlier_failure_modes_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
