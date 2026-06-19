"""Build PTB-only good/medium boundary replay bank for clean-BUT alignment.

The target definitions use only the curated clean-BUT train split.  The saved
signals are PTB synthetic train waveforms, with labels carried from PTB.  This
keeps inference waveform-only and makes the replay blocks interpretable:

* medium_visible_qrs_shell: medium that looks good by QRS/amplitude.
* medium_lowqrs_flat_detail: medium with low QRS/detail and mild flatness.
* good_lowenergy_entropy_protect: good rows that otherwise get eaten by medium.
"""

from __future__ import annotations

import importlib.util
import argparse
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
OUT_DIR = ANALYSIS_DIR / "ptb_gm_boundary_replay_bank"


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


def nearest_replay(
    source: pd.DataFrame,
    target: pd.DataFrame,
    cols: list[str],
    count: int,
    rng: np.random.Generator,
    neighbor_k: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    center, scale = robust_scale(target[cols].to_numpy(dtype=np.float32))
    sz = np.nan_to_num((source[cols].to_numpy(dtype=np.float32) - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    tz = np.nan_to_num((target[cols].to_numpy(dtype=np.float32) - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    picked_target = rng.choice(np.arange(len(target)), size=count, replace=True)
    selected: list[int] = []
    distances: list[float] = []
    for ti in picked_target:
        dist = np.mean((sz - tz[ti][None, :]) ** 2, axis=1)
        k = min(max(1, int(neighbor_k)), len(dist))
        nn = np.argpartition(dist, k - 1)[:k]
        choice = int(rng.choice(nn))
        selected.append(choice)
        distances.append(float(dist[choice]))
    return source.iloc[selected].reset_index(drop=True), picked_target.astype(np.int64), np.asarray(distances, dtype=np.float32)


def augment_medium_goodlike_overlap_bank(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep QRS visible but degrade quality cues around the good/medium boundary."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    t = np.arange(n, dtype=np.float32) / 125.0
    # Mild baseline and amplitude drift: visually still readable, but not
    # measurement-clean.
    y *= float(rng.uniform(0.82, 1.08))
    y += (
        float(rng.uniform(0.015, 0.055))
        * np.sin(2.0 * np.pi * float(rng.uniform(0.15, 0.75)) * t + float(rng.uniform(-np.pi, np.pi)))
    ).astype(np.float32)
    y += (float(rng.uniform(-0.035, 0.035)) * np.linspace(-1.0, 1.0, n, dtype=np.float32)).astype(np.float32)
    # Local measurement artifacts: short notches/bursts that reduce agreement
    # without turning the row into bad stress.
    for _ in range(int(rng.integers(2, 5))):
        width = int(rng.integers(10, 34))
        start = int(rng.integers(0, max(1, n - width)))
        if rng.random() < 0.55:
            y[start : start + width] += rng.normal(0.0, float(rng.uniform(0.012, 0.035)), size=width).astype(np.float32)
        else:
            y[start : start + width] *= float(rng.uniform(0.70, 0.92))
    # Sparse extra small peaks can confuse detectors while preserving a clear
    # dominant rhythm.
    for _ in range(int(rng.integers(1, 4))):
        center = int(rng.integers(8, max(9, n - 8)))
        half = int(rng.integers(3, 8))
        lo, hi = max(0, center - half), min(n, center + half + 1)
        z = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
        bump = np.exp(-8.0 * z * z).astype(np.float32)
        y[lo:hi] += float(rng.uniform(0.025, 0.075)) * bump * (1.0 if rng.random() < 0.5 else -1.0)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="margin_ge_5s_drop_outlier")
    parser.add_argument("--stem", type=str, default="ptb_gm_boundary_replay_clean_drop_train")
    parser.add_argument("--seed", type=int, default=2026061906)
    parser.add_argument("--neighbor-k", type=int, default=96)
    args = parser.parse_args()

    runner = load_module("badalign_for_gm_replay_bank", RUNNER)
    norm = runner.BadAlignedSyntheticDualviewDataset("train", "none", 0.0, 0.0, 0.0, 0.0, ptb_feature_source="waveform_primitives")
    policy = str(args.policy)
    clean_train = runner.DUAL.CleanButDataset(runner.DUAL.PROTOCOL_ROOT / policy, "train", norm.channel_stats, norm.feature_norm)
    target = clean_train.frame.copy()
    target["y_int"] = clean_train.y

    labels = pd.read_csv(runner.ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    signals = np.load(runner.ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    feature_columns = list(runner.FORMAL_FEATURE_COLUMNS)
    match_cols = [
        "qrs_visibility",
        "qrs_band_ratio",
        "qrs_prom_p90",
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
        "detector_agreement",
        "amplitude_entropy",
        "sample_entropy_proxy",
        "low_amp_ratio",
    ]
    match_cols = [c for c in match_cols if c in feature_columns and c in target.columns]
    features_all = runner.load_ptb_feature_matrix("waveform_primitives")
    feat = pd.DataFrame(features_all[:, [feature_columns.index(c) for c in match_cols]], columns=match_cols)
    source = pd.concat([labels.reset_index(drop=True), feat], axis=1)
    source = source[source["split"].astype(str).eq("train")].reset_index(drop=True)

    good = target[target["y_int"].eq(runner.CLASS_TO_INT["good"])]
    med = target[target["y_int"].eq(runner.CLASS_TO_INT["medium"])]
    src_good = source[source["y_class"].astype(str).eq("good")]
    src_med = source[source["y_class"].astype(str).eq("medium")]

    roles = [
        {
            "role": "medium_goodlike_overlap",
            "label": "medium",
            "count": 1100,
            "target": med[
                (med["original_region"].astype(str).eq("good_medium_overlap"))
                & (med["qrs_prom_p90"] >= med["qrs_prom_p90"].quantile(0.45))
                & (med["sqi_sSQI"] >= med["sqi_sSQI"].quantile(0.45))
                & (med["sample_entropy_proxy"] <= med["sample_entropy_proxy"].quantile(0.70))
                & (med["detector_agreement"] <= med["detector_agreement"].quantile(0.65))
            ],
            "source": src_med[
                (src_med["qrs_prom_p90"] >= src_med["qrs_prom_p90"].quantile(0.45))
                & (src_med["sqi_sSQI"] >= src_med["sqi_sSQI"].quantile(0.45))
                & (src_med["sample_entropy_proxy"] <= src_med["sample_entropy_proxy"].quantile(0.75))
                & (src_med["detector_agreement"] <= src_med["detector_agreement"].quantile(0.75))
            ],
        },
        {
            "role": "medium_visible_qrs_shell",
            "label": "medium",
            "count": 900,
            "target": med[(med["qrs_visibility"] >= 0.09) & (med["rms"] >= 0.23)],
            "source": src_med[(src_med["qrs_visibility"] >= 0.09) & (src_med["rms"] >= 0.18)],
        },
        {
            "role": "medium_high_sqi_qrsvisible",
            "label": "medium",
            "count": 900,
            "target": med[
                (med["original_region"].astype(str).isin(["good_medium_overlap", "outlier_low_confidence"]))
                & (med["qrs_visibility"] >= med["qrs_visibility"].quantile(0.70))
                & (med["qrs_prom_p90"] >= med["qrs_prom_p90"].quantile(0.70))
                & (med["sqi_sSQI"] >= med["sqi_sSQI"].quantile(0.70))
                & (med["sqi_kSQI"] >= med["sqi_kSQI"].quantile(0.70))
            ],
            "source": src_med[
                (src_med["qrs_visibility"] >= src_med["qrs_visibility"].quantile(0.70))
                & (src_med["qrs_prom_p90"] >= src_med["qrs_prom_p90"].quantile(0.65))
                & (src_med["sqi_sSQI"] >= src_med["sqi_sSQI"].quantile(0.60))
                & (src_med["sqi_kSQI"] >= src_med["sqi_kSQI"].quantile(0.60))
            ],
        },
        {
            "role": "medium_lowqrs_flat_detail",
            "label": "medium",
            "count": 500,
            "target": med[(med["qrs_visibility"] <= 0.14) & (med["flatline_ratio"] >= 0.08) & (med["diff_abs_p95"] <= 0.18)],
            "source": src_med[(src_med["qrs_visibility"] <= 0.20) & (src_med["flatline_ratio"] >= 0.04) & (src_med["diff_abs_p95"] <= 0.24)],
        },
        {
            "role": "good_lowenergy_entropy_protect",
            "label": "good",
            "count": 900,
            "target": good[
                (good["sqi_sSQI"] <= good["sqi_sSQI"].quantile(0.30))
                & (good["rms"] <= good["rms"].quantile(0.40))
                & (good["amplitude_entropy"] >= good["amplitude_entropy"].quantile(0.55))
            ],
            "source": src_good[
                (src_good["sqi_sSQI"] <= src_good["sqi_sSQI"].quantile(0.40))
                & (src_good["rms"] <= src_good["rms"].quantile(0.55))
                & (src_good["amplitude_entropy"] >= src_good["amplitude_entropy"].quantile(0.45))
            ],
        },
        {
            "role": "good_baseline_outlier_protect",
            "label": "good",
            "count": 900,
            "target": good[
                (good["original_region"].astype(str).isin(["outlier_low_confidence", "good_medium_overlap"]))
                & (good["baseline_step"] >= good["baseline_step"].quantile(0.65))
                & (good["sqi_basSQI"] <= good["sqi_basSQI"].quantile(0.45))
                & (good["qrs_visibility"] <= good["qrs_visibility"].quantile(0.55))
            ],
            "source": src_good[
                (src_good["baseline_step"] >= src_good["baseline_step"].quantile(0.60))
                & (src_good["sqi_basSQI"] <= src_good["sqi_basSQI"].quantile(0.55))
                & (src_good["qrs_visibility"] <= src_good["qrs_visibility"].quantile(0.65))
            ],
        },
    ]

    rng = np.random.default_rng(int(args.seed))
    manifest_parts: list[pd.DataFrame] = []
    x_parts: list[np.ndarray] = []
    summary_roles: list[dict[str, object]] = []
    for item in roles:
        role = str(item["role"])
        label = str(item["label"])
        role_target = item["target"].reset_index(drop=True)
        role_source = item["source"].reset_index(drop=True)
        count = min(int(item["count"]), max(1, len(role_target) * 2))
        if len(role_target) == 0:
            role_target = (med if label == "medium" else good).reset_index(drop=True)
        if len(role_source) == 0:
            role_source = (src_med if label == "medium" else src_good).reset_index(drop=True)
        if len(role_target) == 0 or len(role_source) == 0:
            raise RuntimeError(f"empty role target/source for {role}: target={len(role_target)} source={len(role_source)}")
        selected, target_pick, dist = nearest_replay(role_source, role_target, match_cols, count, rng, int(args.neighbor_k))
        source_idx = selected["idx"].to_numpy(dtype=np.int64)
        role_x = signals[source_idx].astype(np.float32).copy()
        if role == "medium_goodlike_overlap":
            role_x = np.stack([augment_medium_goodlike_overlap_bank(row, rng) for row in role_x]).astype(np.float32)
        x_parts.append(role_x)
        manifest_parts.append(
            pd.DataFrame(
                {
                    "source_idx": source_idx,
                    "y_class": label,
                    "stress": 0.0,
                    "bank_role": role,
                    "target_but_idx": role_target.iloc[target_pick]["idx"].to_numpy(dtype=np.int64),
                    "target_original_region": role_target.iloc[target_pick]["original_region"].astype(str).to_numpy(),
                    "distance": dist,
                }
            )
        )
        summary_roles.append(
            {
                "role": role,
                "label": label,
                "target_rows": int(len(role_target)),
                "source_rows": int(len(role_source)),
                "selected_rows": int(count),
                "unique_source_rows": int(pd.Series(source_idx).nunique()),
                "distance_q50": float(np.quantile(dist, 0.50)),
                "distance_q90": float(np.quantile(dist, 0.90)),
            }
        )

    bank_x = np.concatenate(x_parts, axis=0)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = str(args.stem)
    np.savez_compressed(OUT_DIR / f"{stem}_signals.npz", X=bank_x)
    manifest.to_csv(OUT_DIR / f"{stem}_manifest.csv", index=False)
    summary = {
        "policy": policy,
        "rows": int(len(manifest)),
        "class_counts": manifest["y_class"].value_counts().to_dict(),
        "role_counts": manifest["bank_role"].value_counts().to_dict(),
        "roles": summary_roles,
        "match_columns": match_cols,
        "neighbor_k": int(args.neighbor_k),
        "signals": str(OUT_DIR / f"{stem}_signals.npz"),
        "manifest": str(OUT_DIR / f"{stem}_manifest.csv"),
    }
    (OUT_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
