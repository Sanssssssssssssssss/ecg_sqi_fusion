"""Build a PTB-only medium-shell replay bank for the curated clean-BUT protocol.

The bank contains PTB synthetic train rows only.  BUT clean train rows are used
only as a feature-distribution reference, never as waveform training input.
This targets the observed clean/drop error where high-QRS/high-RMS BUT medium
rows are predicted as good by PTB-trained waveform models.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
RUNNER = ANALYSIS_DIR / "run_ptb_bad_alignment_cross_dataset.py"
OUT_DIR = ANALYSIS_DIR / "ptb_medium_shell_bank"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def robust_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0).astype(np.float32)
    scale = (np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    return center, scale


def main() -> None:
    runner = load_module("badalign_for_medium_shell_bank", RUNNER)
    train_norm = runner.BadAlignedSyntheticDualviewDataset("train", "none", 0.0, 0.0, 0.0, 0.0)
    policy = "margin_ge_5s_drop_outlier"
    clean_train = runner.DUAL.CleanButDataset(runner.DUAL.PROTOCOL_ROOT / policy, "train", train_norm.channel_stats, train_norm.feature_norm)

    target = clean_train.frame.copy()
    target["y_int"] = clean_train.y
    target = target[
        (target["y_int"] == runner.CLASS_TO_INT["medium"])
        & (target["qrs_visibility"].astype(float) >= 0.09)
        & (target["rms"].astype(float) >= 0.23)
    ].reset_index(drop=True)

    labels = pd.read_csv(runner.ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    signals = np.load(runner.ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    feat_npz = np.load(runner.ARCH.DATA_DIR / "tabular_features.npz")
    schema = json.loads((runner.ARCH.DATA_DIR / "tabular_feature_schema.json").read_text(encoding="utf-8"))
    feature_columns = list(schema["feature_columns"])

    match_cols = [
        "qrs_visibility",
        "qrs_band_ratio",
        "rms",
        "std",
        "ptp_p99_p01",
        "sqi_sSQI",
        "sqi_kSQI",
        "sqi_basSQI",
        "baseline_step",
        "flatline_ratio",
        "non_qrs_diff_p95",
        "diff_abs_p95",
        "band_15_30",
        "wavelet_e0",
        "wavelet_e2",
        "hjorth_complexity",
        "template_corr",
    ]
    match_cols = [c for c in match_cols if c in feature_columns and c in target.columns]
    feat_idx = [feature_columns.index(c) for c in match_cols]
    feat = pd.DataFrame(feat_npz["features"][:, feat_idx], columns=match_cols)
    source = pd.concat([labels.reset_index(drop=True), feat], axis=1)
    source = source[
        (source["split"].astype(str) == "train")
        & (source["y_class"].astype(str) == "medium")
        & (source["qrs_visibility"].astype(float) >= 0.09)
        & (source["rms"].astype(float) >= 0.18)
    ].reset_index(drop=True)

    center, scale = robust_scale(target[match_cols].to_numpy(dtype=np.float32))
    target_z = np.nan_to_num((target[match_cols].to_numpy(dtype=np.float32) - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    source_z = np.nan_to_num((source[match_cols].to_numpy(dtype=np.float32) - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(2026061905)
    n_take = min(2200, max(1, len(target)))
    picked_target = rng.choice(np.arange(len(target_z)), size=n_take, replace=True)
    selected: list[int] = []
    distances: list[float] = []
    for ti in picked_target:
        dist = np.mean((source_z - target_z[ti][None, :]) ** 2, axis=1)
        # Mild randomization among near neighbors avoids replaying only a few rows.
        k = min(12, len(dist))
        nn = np.argpartition(dist, k - 1)[:k]
        choice = int(rng.choice(nn))
        selected.append(choice)
        distances.append(float(dist[choice]))

    selected_source = source.iloc[selected].reset_index(drop=True)
    source_idx = selected_source["idx"].to_numpy(dtype=np.int64)
    bank_x = signals[source_idx].astype(np.float32)
    manifest = pd.DataFrame(
        {
            "source_idx": source_idx,
            "y_class": "medium",
            "stress": 0.0,
            "bank_role": "gm_visible_qrs_medium_shell",
            "target_but_idx": target.iloc[picked_target]["idx"].to_numpy(dtype=np.int64),
            "target_original_region": target.iloc[picked_target]["original_region"].astype(str).to_numpy(),
            "distance": np.asarray(distances, dtype=np.float32),
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "ptb_medium_shell_match_clean_drop_train_qrsrms"
    np.savez_compressed(OUT_DIR / f"{stem}_signals.npz", X=bank_x)
    manifest.to_csv(OUT_DIR / f"{stem}_manifest.csv", index=False)

    summary = {
        "policy": policy,
        "target_rows": int(len(target)),
        "source_rows": int(len(source)),
        "selected_rows": int(len(manifest)),
        "unique_source_rows": int(manifest["source_idx"].nunique()),
        "match_columns": match_cols,
        "signals": str(OUT_DIR / f"{stem}_signals.npz"),
        "manifest": str(OUT_DIR / f"{stem}_manifest.csv"),
        "distance_q50": float(np.quantile(distances, 0.50)),
        "distance_q90": float(np.quantile(distances, 0.90)),
    }
    (OUT_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
