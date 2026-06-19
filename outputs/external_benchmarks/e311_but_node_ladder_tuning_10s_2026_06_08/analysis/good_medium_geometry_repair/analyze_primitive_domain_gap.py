"""Waveform primitive domain-gap analysis for original bad outliers.

Compares waveform-derived primitive statistics for PTB synthetic labels against
original BUT buckets.  This is report-only and does not train or select models.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path.cwd().resolve()
RUNNER = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair" / "run_waveform_geometry_student.py"
OUT_DIR = RUNNER.parent
REPORT_DIR = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def primitive_names(channels: int) -> list[str]:
    wave = [
        "mean",
        "std",
        "mean_abs",
        "rms",
        "ptp",
        "diff_abs",
        "diff_rms",
        "flat",
        "low_amp",
        "zcr",
        "diff_tail10",
        "diff_tail02",
        "detail_short",
        "detail_long",
        "baseline_slope",
        "contact",
    ]
    qrs = [
        "qrs_top01",
        "qrs_top03",
        "qrs_top10",
        "qrs_prom_ratio",
        "qrs_detail_rms",
        "qrs_diff_top",
        "qrs_diff_ratio",
        "qrs_baseline_span",
        "qrs_baseline_slope",
        "qrs_flat",
        "qrs_low_amp",
        "qrs_mean_abs",
        "qrs_rms",
        "qrs_ptp",
        "qrs_zcr",
    ]
    stress = [
        "stress_flat005",
        "stress_flat015",
        "stress_flat030",
        "stress_low015",
        "stress_low050",
        "stress_low100",
        "stress_zcr",
        "stress_dzcr",
        "stress_flat_run101_max",
        "stress_flat_run251_max",
        "stress_low_run101_max",
        "stress_low_run251_max",
        "stress_baseline_span101",
        "stress_baseline_span251",
        "stress_baseline_slope251",
        "stress_baseline_curve251",
        "stress_detail_ratio31",
        "stress_detail_ratio101",
        "stress_diff_top01",
        "stress_diff_top10",
        "stress_ptp",
        "stress_top_abs10",
        "stress_flat_run101_mean",
        "stress_low_run101_mean",
    ]
    qrsbank = [
        "qrsbank_top005",
        "qrsbank_top02",
        "qrsbank_top10",
        "qrsbank_top_ratio",
        "qrsbank_local_top01",
        "qrsbank_local_top03",
        "qrsbank_peak_cv",
        "qrsbank_peak_density08",
        "qrsbank_peak_density12",
        "qrsbank_peak_density18",
        "qrsbank_no_peak251",
        "qrsbank_ac50",
        "qrsbank_ac100",
        "qrsbank_ac150",
        "qrsbank_ac250",
        "qrsbank_ac350",
        "qrsbank_high_rms",
        "qrsbank_high_mean_abs",
        "qrsbank_short_detail",
    ]
    base = wave + qrs + stress + qrsbank
    names: list[str] = []
    for ch in range(channels):
        names.extend([f"ch{ch}_{name}" for name in base])
    return names


@torch.no_grad()
def collect_stats(mod: Any, ds: Any) -> np.ndarray:
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    chunks = []
    for batch in loader:
        x = batch["x"].to(dtype=torch.float32)
        chunks.append(mod.primitive_waveform_stats(x, "qrs_enhanced").cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([a, b])))
    ca = np.searchsorted(a, values, side="right") / len(a)
    cb = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def main() -> None:
    mod = load_runner()
    train_ds = mod.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = mod.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = mod.ARCH.OriginalWaveDataset(norm, "robust3")
    synth = collect_stats(mod, train_ds)
    orig = collect_stats(mod, original_ds)
    mu = synth.mean(axis=0)
    sigma = np.where(synth.std(axis=0) < 1e-6, 1.0, synth.std(axis=0))
    zs = (synth - mu[None, :]) / sigma[None, :]
    zo = (orig - mu[None, :]) / sigma[None, :]
    synth_y = train_ds.y
    orig_test = original_ds.split == "test"
    orig_y = original_ds.y
    orig_region = original_ds.region
    masks = {
        "synthetic_good": synth_y == CLASS_TO_INT["good"],
        "synthetic_medium": synth_y == CLASS_TO_INT["medium"],
        "synthetic_bad": synth_y == CLASS_TO_INT["bad"],
        "synthetic_nonbad": synth_y != CLASS_TO_INT["bad"],
    }
    orig_masks = {
        "original_test_good": orig_test & (orig_y == CLASS_TO_INT["good"]),
        "original_test_medium": orig_test & (orig_y == CLASS_TO_INT["medium"]),
        "original_test_bad_core": orig_test & (orig_y == CLASS_TO_INT["bad"]) & (orig_region != "outlier_low_confidence"),
        "original_test_bad_outlier": orig_test & (orig_y == CLASS_TO_INT["bad"]) & (orig_region == "outlier_low_confidence"),
    }
    names = primitive_names(train_ds.x.shape[1])
    rows = []
    for j, name in enumerate(names):
        synth_bad = zs[masks["synthetic_bad"], j]
        synth_nonbad = zs[masks["synthetic_nonbad"], j]
        orig_bad_out = zo[orig_masks["original_test_bad_outlier"], j]
        orig_bad_core = zo[orig_masks["original_test_bad_core"], j]
        dist_bad = abs(float(np.mean(orig_bad_out)) - float(np.mean(synth_bad)))
        dist_nonbad = abs(float(np.mean(orig_bad_out)) - float(np.mean(synth_nonbad)))
        rows.append(
            {
                "feature": name,
                "orig_bad_outlier_mean_z": float(np.mean(orig_bad_out)),
                "orig_bad_outlier_p10_z": float(np.quantile(orig_bad_out, 0.10)),
                "orig_bad_outlier_p90_z": float(np.quantile(orig_bad_out, 0.90)),
                "orig_bad_core_mean_z": float(np.mean(orig_bad_core)),
                "synth_bad_mean_z": float(np.mean(synth_bad)),
                "synth_nonbad_mean_z": float(np.mean(synth_nonbad)),
                "ks_outlier_vs_synth_bad": ks_stat(orig_bad_out, synth_bad),
                "ks_outlier_vs_synth_nonbad": ks_stat(orig_bad_out, synth_nonbad),
                "closer_to_synth_nonbad": bool(dist_nonbad < dist_bad),
                "mean_gap_to_synth_bad": dist_bad,
                "mean_gap_to_synth_nonbad": dist_nonbad,
            }
        )
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "primitive_domain_gap_bad_outlier.csv"
    df.to_csv(out_csv, index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / out_csv.name).write_bytes(out_csv.read_bytes())
    top_ks = df.sort_values("ks_outlier_vs_synth_bad", ascending=False).head(25)
    top_nonbad = df[df["closer_to_synth_nonbad"]].sort_values("mean_gap_to_synth_bad", ascending=False).head(25)
    lines = [
        "# Primitive Domain Gap: Bad Outliers",
        "",
        "Report-only. Compares waveform-derived primitive stats, not external SQI inputs.",
        "",
        "## Largest Original Bad-Outlier vs Synthetic Bad KS",
        "",
        top_ks.to_csv(index=False),
        "",
        "## Features Where Original Bad-Outlier Is Closer To Synthetic Nonbad",
        "",
        top_nonbad.to_csv(index=False),
        "",
        f"- CSV: `{out_csv}`",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "primitive_domain_gap_bad_outlier_report.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "primitive_domain_gap_bad_outlier_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
