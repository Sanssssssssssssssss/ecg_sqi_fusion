"""Compare augmented synthetic bad shells against original bad outliers.

This is report-only.  It does not train, select, or use original BUT for any
model choice.  The goal is to test whether a proposed bad augmentation actually
moves waveform-derived primitives toward the original bad-outlier bucket.
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
OUT_DIR = BASE
REPORT_DIR = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
CONFIGS = [
    "base_synthetic",
    "qrsbank_badexpert_badguard",
    "qrsbank_badstress_hardneg_balanced",
    "qrsbank_stattoken_badguard",
    "qrsbank_highamp_patch_balanced",
    "qrsbank_highamp_stattoken_badguard",
    "qrsbank_highamp_mix_patch_guard",
    "qrsbank_highamp_mix_stattoken_guard",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_dataset(mod: Any, cfg_name: str) -> Any:
    base = mod.ARCH.SyntheticWaveDataset("train", "robust3")
    if cfg_name == "base_synthetic":
        return base
    cfg = dict(mod.CANDIDATES[cfg_name])
    return mod.AugmentedSyntheticDataset(
        base,
        float(cfg.get("augment_strength", 0.0)),
        int(cfg.get("seed", 0)),
        float(cfg.get("bad_outlier_aug_strength", 0.0)),
        float(cfg.get("bad_stress_shell_strength", 0.0)),
        float(cfg.get("nonbad_hardneg_strength", 0.0)),
        float(cfg.get("bad_highamp_stress_strength", 0.0)),
        float(cfg.get("bad_highamp_prob", 1.0)),
    )


def zscore(values: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (values - mu[None, :]) / sigma[None, :]


def main() -> None:
    mod = load_module(RUNNER, "waveform_geometry_student_runner")
    helper = load_module(HELPER, "primitive_gap_helper")
    base_train = mod.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = mod.ARCH.FeatureNorm(mean=base_train.mean, std=base_train.std)
    original = mod.ARCH.OriginalWaveDataset(norm, "robust3")

    base_stats = helper.collect_stats(mod, base_train)
    mu = base_stats.mean(axis=0)
    sigma = np.where(base_stats.std(axis=0) < 1e-6, 1.0, base_stats.std(axis=0))
    orig_stats = zscore(helper.collect_stats(mod, original), mu, sigma)
    names = helper.primitive_names(base_train.x.shape[1])

    orig_test = original.split == "test"
    orig_bad_outlier_mask = (
        orig_test
        & (original.y == CLASS_TO_INT["bad"])
        & (original.region == "outlier_low_confidence")
    )
    orig_bad_outlier = orig_stats[orig_bad_outlier_mask]

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for cfg_name in CONFIGS:
        ds = make_dataset(mod, cfg_name)
        stats = zscore(helper.collect_stats(mod, ds), mu, sigma)
        bad_mask = ds.y == CLASS_TO_INT["bad"]
        synth_bad = stats[bad_mask]
        feature_rows = []
        for j, feature in enumerate(names):
            orig_values = orig_bad_outlier[:, j]
            synth_values = synth_bad[:, j]
            row = {
                "config": cfg_name,
                "feature": feature,
                "orig_bad_outlier_mean_z": float(np.mean(orig_values)),
                "synthetic_bad_mean_z": float(np.mean(synth_values)),
                "mean_gap": abs(float(np.mean(orig_values)) - float(np.mean(synth_values))),
                "ks_outlier_vs_synthetic_bad": helper.ks_stat(orig_values, synth_values),
            }
            rows.append(row)
            feature_rows.append(row)
        feature_df = pd.DataFrame(feature_rows)
        summary.append(
            {
                "config": cfg_name,
                "n_synthetic_bad": int(bad_mask.sum()),
                "mean_ks_all_features": float(feature_df["ks_outlier_vs_synthetic_bad"].mean()),
                "median_ks_all_features": float(feature_df["ks_outlier_vs_synthetic_bad"].median()),
                "mean_gap_all_features": float(feature_df["mean_gap"].mean()),
                "top25_mean_ks": float(feature_df.sort_values("ks_outlier_vs_synthetic_bad", ascending=False).head(25)["ks_outlier_vs_synthetic_bad"].mean()),
                "top25_mean_gap": float(feature_df.sort_values("mean_gap", ascending=False).head(25)["mean_gap"].mean()),
            }
        )

    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary).sort_values(["top25_mean_ks", "mean_ks_all_features"], ascending=True)
    detail_path = OUT_DIR / "augmented_bad_domain_gap_features.csv"
    summary_path = OUT_DIR / "augmented_bad_domain_gap_summary.csv"
    df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [detail_path, summary_path]:
        (REPORT_DIR / path.name).write_bytes(path.read_bytes())

    base_detail = df[df["config"] == "base_synthetic"].sort_values("ks_outlier_vs_synthetic_bad", ascending=False).head(20)
    top_features = base_detail["feature"].tolist()
    compact = df[df["feature"].isin(top_features)].pivot_table(
        index="feature",
        columns="config",
        values="ks_outlier_vs_synthetic_bad",
        aggfunc="first",
    ).reset_index()
    compact_path = OUT_DIR / "augmented_bad_domain_gap_topbase_ks.csv"
    compact.to_csv(compact_path, index=False)
    (REPORT_DIR / compact_path.name).write_bytes(compact_path.read_bytes())

    lines = [
        "# Augmented Bad Domain Gap",
        "",
        "Report-only diagnostic. Lower KS/gap means the augmented synthetic bad shell is closer to original bad-outlier waveform primitives.",
        "",
        "## Config Summary",
        "",
        summary_df.to_csv(index=False),
        "",
        "## Top Base-Gap Feature KS By Config",
        "",
        compact.to_csv(index=False),
        "",
        f"- Summary CSV: `{summary_path}`",
        f"- Feature CSV: `{detail_path}`",
        f"- Top-base compact CSV: `{compact_path}`",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "augmented_bad_domain_gap_report.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "augmented_bad_domain_gap_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
