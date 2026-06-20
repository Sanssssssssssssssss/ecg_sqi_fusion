"""BUT-aligned PTB synthetic protocol and cross-dataset evaluation.

This script tests the current suspected root cause: the PTB synthetic class
geometry, especially bad/medium, is not aligned with the clean BUT protocol.

It builds a new PTB protocol by selecting PTB synthetic candidates that are
nearest to clean BUT *train-split* rows in waveform-computable SQI/physiology
features.  BUT test rows are not used for matching.  Model inference still uses
waveform-derived channels only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
BASE_SCRIPT = ANALYSIS_DIR / "run_event_factorized_cross_dataset_top3.py"

CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v1"
CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_report.md"
CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v1"
DEFAULT_BUT_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier"

POOL_FILES = [
    ("ptb_combo_boundary_bank", "ptb_combo_balanced2500_block_corebad_ctl10_gm_keepout_blocks_v3"),
    ("ptb_combo_boundary_bank", "ptb_combo_balanced2500_block_corebad_ctl10_gm_keepout_diverse_v2"),
    ("ptb_combo_boundary_bank", "ptb_combo_keepbad5s_controlled_block_gm_cvfold2_wavefact_v1"),
    ("ptb_gm_boundary_replay_bank", "ptb_gm_boundary_replay_keep_outlier_cvfold2_wavefact_blocks_v3"),
    ("ptb_medium_shell_bank", "ptb_medium_shell_match_clean_drop_train_qrsrms"),
    ("ptb_bad_waveform_feature_match", "ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb"),
]

MATCH_FEATURES = [
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "sqi_basSQI",
    "non_qrs_diff_p95",
    "qrs_band_ratio",
    "template_corr",
    "amplitude_entropy",
    "contact_loss_win_ratio",
    "rms",
    "mean_abs",
    "band_15_30",
    "band_30_45",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SCRIPT, "event_xds_top3_base_for_aligned")
EVENT = BASE.EVENT
GEOM_FEATURES = BASE.GEOM_FEATURES


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def configure_variant(variant: str) -> None:
    global CROSS_OUT, CROSS_REPORT, CROSS_RUN_DIR
    if variant == "v1":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v1"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v1"
    elif variant == "v2_morphcal":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v2_morphcal"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v2_morphcal_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v2_morphcal"
    elif variant == "v3_detectorfail":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v3_detectorfail"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v3_detectorfail_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v3_detectorfail"
    elif variant == "v4_peakdrop":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v4_peakdrop"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v4_peakdrop_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v4_peakdrop"
    elif variant == "v5_reject_detectorfail":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v5_reject_detectorfail"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v5_reject_detectorfail_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v5_reject_detectorfail"
    elif variant == "v6_lowqrs_aggressive":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v6_lowqrs_aggressive"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v6_lowqrs_aggressive_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v6_lowqrs_aggressive"
    elif variant == "v7_mixed_detectorfail":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v7_mixed_detectorfail"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v7_mixed_detectorfail_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v7_mixed_detectorfail"
    elif variant == "v8_visibleqrs_detailmatch":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v8_visibleqrs_detailmatch"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v8_visibleqrs_detailmatch_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v8_visibleqrs_detailmatch"
    elif variant == "v9_strong_visibleqrs_medium":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v9_strong_visibleqrs_medium"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v9_strong_visibleqrs_medium_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v9_strong_visibleqrs_medium"
    elif variant == "v10_butmedium_extreme":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v10_butmedium_extreme"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v10_butmedium_extreme_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v10_butmedium_extreme"
    elif variant == "v11_buttrain_style_replay":
        CROSS_OUT = ANALYSIS_DIR / "event_xds_aligned_v11_buttrain_style_replay"
        CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_aligned_ptb_v11_buttrain_style_replay_report.md"
        CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_aligned_v11_buttrain_style_replay"
    else:
        raise ValueError(f"unknown variant: {variant}")


def ensure_dirs() -> None:
    CROSS_OUT.mkdir(parents=True, exist_ok=True)
    CROSS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CROSS_RUN_DIR.mkdir(parents=True, exist_ok=True)


def load_pool() -> tuple[np.ndarray, pd.DataFrame]:
    xs: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    for folder_name, stem in POOL_FILES:
        folder = ANALYSIS_DIR / folder_name
        sig_path = folder / f"{stem}_signals.npz"
        manifest_path = folder / f"{stem}_manifest.csv"
        if not sig_path.exists() or not manifest_path.exists():
            continue
        x = np.load(sig_path)["X"].astype(np.float32)
        if x.ndim == 3:
            x = x[:, 0, :]
        manifest = pd.read_csv(manifest_path).reset_index(drop=True)
        if "y_class" not in manifest.columns and "class_name" in manifest.columns:
            manifest["y_class"] = manifest["class_name"].astype(str)
        if "y_class" not in manifest.columns:
            continue
        n = min(len(manifest), len(x))
        manifest = manifest.iloc[:n].copy()
        manifest["pool_file"] = stem
        manifest["pool_folder"] = folder_name
        manifest["pool_row"] = np.arange(n, dtype=np.int64)
        if "source_idx" not in manifest.columns:
            manifest["source_idx"] = manifest["pool_row"]
        xs.append(x[:n])
        frames.append(manifest)
    if not xs:
        raise RuntimeError("no PTB pool files found")
    x_all = np.concatenate(xs, axis=0).astype(np.float32)
    frame = pd.concat(frames, ignore_index=True)
    frame["pool_key"] = frame["pool_file"].astype(str) + ":" + frame["pool_row"].astype(str)
    # Remove exact duplicated pool-key rows if a bank was accidentally listed twice.
    keep = ~frame["pool_key"].duplicated()
    x_all = x_all[keep.to_numpy()]
    frame = frame.loc[keep].reset_index(drop=True)
    frame["pool_global_idx"] = np.arange(len(frame), dtype=np.int64)
    return x_all, frame


def split_groups_stratified(frame: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    group_col = frame["pool_file"].astype(str) + ":" + frame["source_idx"].astype(str)
    work = pd.DataFrame({"group": group_col, "y_class": frame["y_class"].astype(str)})
    major = work.groupby("group")["y_class"].agg(lambda s: s.value_counts().index[0]).reset_index(name="major_class")
    split_by_group: dict[str, str] = {}
    for cls in ["good", "medium", "bad"]:
        keys = major.loc[major["major_class"].eq(cls), "group"].to_numpy()
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        for key in keys[:n_train]:
            split_by_group[str(key)] = "train"
        for key in keys[n_train : n_train + n_val]:
            split_by_group[str(key)] = "val"
        for key in keys[n_train + n_val :]:
            split_by_group[str(key)] = "test"
    return group_col.map(split_by_group).fillna("train").to_numpy()


def feature_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else 0.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win) | 1)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), kernel, mode="same").astype(np.float32)


def calibrate_signals_to_but_train(
    x: np.ndarray,
    selected: pd.DataFrame,
    but_train: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """Move PTB synthetic raw morphology toward clean-BUT train feature medians.

    This is intentionally simple and interpretable: amplitude calibration,
    baseline step/ramp injection, detail-noise injection, and light QRS masking
    for low-QRS classes.  It uses only clean-BUT train split statistics.
    """
    rng = np.random.default_rng(int(seed) + 17043)
    targets: dict[str, dict[str, float]] = {}
    for cls in ["good", "medium", "bad"]:
        part = but_train.loc[but_train["class_name"].astype(str).eq(cls)].copy()
        targets[cls] = {}
        for col in ["rms", "mean_abs", "baseline_step", "qrs_visibility", "non_qrs_diff_p95", "band_15_30", "band_30_45"]:
            val = pd.to_numeric(part[col], errors="coerce").replace([np.inf, -np.inf], np.nan).median() if col in part.columns else np.nan
            targets[cls][col] = float(val) if np.isfinite(val) else 0.0
    out = np.asarray(x, dtype=np.float32).copy()
    n_samples = out.shape[1]
    t = np.linspace(-0.5, 0.5, n_samples, dtype=np.float32)
    for i, row in selected.reset_index(drop=True).iterrows():
        cls = str(row["y_class"])
        target = targets.get(cls, targets["medium"])
        sig = out[i].astype(np.float32)
        sig = sig - np.median(sig)
        cur_std = float(np.std(sig))
        if cur_std < 1e-6:
            cur_std = 1.0
        target_rms = max(float(target.get("rms", 0.2)), 0.03)
        target_rms *= float(np.clip(rng.lognormal(mean=0.0, sigma=0.08), 0.82, 1.22))
        sig = sig * (target_rms / cur_std)

        baseline_step = float(target.get("baseline_step", 0.0))
        qrs_visibility = float(target.get("qrs_visibility", 0.5))
        non_qrs_detail = float(target.get("non_qrs_diff_p95", 0.05))
        band_tail = float(target.get("band_15_30", 0.1) + 0.7 * target.get("band_30_45", 0.02))

        # Baseline movement is what separated BUT medium from PTB medium most.
        step_scale = {"good": 0.30, "medium": 0.58, "bad": 0.42}.get(cls, 0.45)
        step_amp = target_rms * step_scale * float(np.clip(baseline_step, 0.02, 0.90))
        if step_amp > 0:
            pos = int(rng.integers(max(20, n_samples // 5), max(21, 4 * n_samples // 5)))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            sig = sig + sign * step_amp * (np.arange(n_samples, dtype=np.float32) >= pos)
            sig = sig + sign * step_amp * 0.35 * t

        # Add detail artifacts in a controlled way; avoid making every bad pure white noise.
        noise = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        high = noise - moving_average_1d(noise, 41)
        fast = noise - moving_average_1d(noise, 9)
        detail_scale = {"good": 0.018, "medium": 0.055, "bad": 0.090}.get(cls, 0.05)
        detail_scale *= float(np.clip(0.7 + 2.8 * non_qrs_detail + 0.8 * band_tail, 0.7, 2.4))
        sig = sig + target_rms * detail_scale * (0.65 * high + 0.35 * fast)

        # Low-QRS BUT rows are not just high-frequency noisy; QRS is obscured.
        if qrs_visibility < 0.45 or cls == "bad":
            smooth = moving_average_1d(sig, 17 if cls == "medium" else 25)
            alpha = {"medium": 0.24, "bad": 0.36}.get(cls, 0.12)
            alpha *= float(np.clip((0.50 - qrs_visibility) / 0.35, 0.35, 1.25))
            sig = (1.0 - alpha) * sig + alpha * smooth

        # Final amplitude lock to the clean-BUT class scale.
        final_std = float(np.std(sig))
        if final_std > 1e-6:
            sig = sig * (target_rms / final_std)
        out[i] = sig.astype(np.float32)
    return out.astype(np.float32)


def detector_failure_signals(x: np.ndarray, selected: pd.DataFrame, seed: int) -> np.ndarray:
    """Add BUT-like detector-failure morphology missing from PTB banks.

    These rows are not just noisy ECG. They contain monotonic ramps, sigmoid
    steps, contact-like plateaus, and weak ECG content so the peak detector can
    fail or become unstable, matching the hard BUT good/medium/bad boundary.
    """
    rng = np.random.default_rng(int(seed) + 7110)
    out = np.asarray(x, dtype=np.float32).copy()
    n_samples = out.shape[1]
    grid = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)

    def robust(row: np.ndarray) -> np.ndarray:
        med = np.median(row)
        q75 = np.percentile(row, 75)
        q25 = np.percentile(row, 25)
        scale = (q75 - q25) / 1.349
        std = np.std(row)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = std if std > 1e-6 else 1.0
        return ((row - med) / scale).astype(np.float32)

    def rand_smooth(width: int = 151) -> np.ndarray:
        knots = rng.normal(0.0, 1.0, 10).astype(np.float32)
        slow = np.interp(np.linspace(0, len(knots) - 1, n_samples), np.arange(len(knots)), knots).astype(np.float32)
        slow = moving_average_1d(slow, width)
        return ((slow - slow.mean()) / (slow.std() + 1e-6)).astype(np.float32)

    def monotonic_artifact(cls: str) -> np.ndarray:
        pos = float(rng.uniform(-0.45, 0.45))
        width = float(rng.uniform(0.04, 0.18))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        ramp = sign * grid
        step = sign * np.tanh((grid - pos) / width).astype(np.float32)
        cumulative = np.cumsum(np.abs(rng.normal(0.0, 0.006, n_samples)).astype(np.float32))
        cumulative = (cumulative - cumulative.mean()) / (cumulative.std() + 1e-6)
        plateau = np.zeros(n_samples, dtype=np.float32)
        if rng.random() < 0.45:
            start = int(rng.integers(n_samples // 8, n_samples // 2))
            stop = min(n_samples, start + int(rng.integers(n_samples // 12, n_samples // 4)))
            plateau[start:stop] = sign
        scale = {"good": 0.75, "medium": 1.15, "bad": 0.95}.get(cls, 1.0)
        return (scale * (0.55 * ramp + 0.85 * step + 0.08 * cumulative + 0.35 * plateau)).astype(np.float32)

    def high_artifact() -> np.ndarray:
        noise = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        high = noise - moving_average_1d(noise, 9)
        slow = rand_smooth(101)
        return (0.70 * high / (high.std() + 1e-6) + 0.30 * slow).astype(np.float32)

    for i, row in selected.reset_index(drop=True).iterrows():
        cls = str(row["y_class"])
        z = robust(out[i])
        mono = monotonic_artifact(cls)
        if cls == "good":
            if rng.random() < 0.28:
                sig = 0.18 * z + mono + 0.03 * high_artifact()
            else:
                sig = 0.70 * z + 0.22 * mono + 0.03 * high_artifact()
        elif cls == "medium":
            if rng.random() < 0.58:
                sig = 0.04 * z + mono + 0.05 * high_artifact()
            else:
                sig = 0.20 * z + 0.65 * mono + 0.12 * high_artifact()
        elif cls == "bad":
            if rng.random() < 0.62:
                sig = 0.03 * z + 0.85 * mono + 0.22 * high_artifact()
            else:
                sig = 0.05 * z + 0.35 * rand_smooth(151) + 0.95 * high_artifact()
        else:
            sig = z
        out[i] = sig.astype(np.float32)
    return out.astype(np.float32)


def peakdrop_signals(x: np.ndarray, selected: pd.DataFrame, seed: int) -> np.ndarray:
    """Cleaner peak-drop coverage: mix original ECG with no-peak monotonic artifacts."""
    rng = np.random.default_rng(int(seed) + 7200)
    out = np.asarray(x, dtype=np.float32).copy()
    n_samples = out.shape[1]
    grid = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)

    def robust(row: np.ndarray) -> np.ndarray:
        med = np.median(row)
        q75 = np.percentile(row, 75)
        q25 = np.percentile(row, 25)
        scale = (q75 - q25) / 1.349
        std = np.std(row)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = std if std > 1e-6 else 1.0
        return ((row - med) / scale).astype(np.float32)

    def no_peak_shape() -> np.ndarray:
        sign = -1.0 if rng.random() < 0.5 else 1.0
        pos = float(rng.uniform(-0.45, 0.45))
        width = float(rng.uniform(0.05, 0.22))
        shape = sign * (0.60 * grid + 0.95 * np.tanh((grid - pos) / width).astype(np.float32))
        shape += sign * 0.08 * grid * grid
        return shape.astype(np.float32)

    def soft_detail() -> np.ndarray:
        noise = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        detail = noise - moving_average_1d(noise, 17)
        return (detail / (detail.std() + 1e-6)).astype(np.float32)

    for i, row in selected.reset_index(drop=True).iterrows():
        cls = str(row["y_class"])
        z = robust(out[i])
        peakdrop = no_peak_shape()
        if cls == "good":
            if rng.random() < 0.18:
                sig = 0.18 * z + 0.85 * peakdrop + 0.02 * soft_detail()
            else:
                sig = 0.86 * z + 0.12 * peakdrop + 0.02 * soft_detail()
        elif cls == "medium":
            r = rng.random()
            if r < 0.38:
                sig = 0.06 * z + 0.95 * peakdrop + 0.03 * soft_detail()
            elif r < 0.72:
                sig = 0.25 * z + 0.55 * peakdrop + 0.05 * soft_detail()
            else:
                sig = 0.55 * z + 0.20 * peakdrop + 0.06 * soft_detail()
        elif cls == "bad":
            r = rng.random()
            if r < 0.50:
                sig = 0.04 * z + 0.85 * peakdrop + 0.08 * soft_detail()
            elif r < 0.78:
                sig = 0.16 * z + 0.45 * peakdrop + 0.35 * soft_detail()
            else:
                sig = 0.05 * z + 0.95 * soft_detail()
        else:
            sig = z
        out[i] = sig.astype(np.float32)
    return out.astype(np.float32)


def rejection_detector_failure_signals(x: np.ndarray, selected: pd.DataFrame, seed: int, aggressive_lowqrs: bool = False) -> np.ndarray:
    """Generate multiple interpretable candidates and keep the closest to BUT feature bands."""
    rng = np.random.default_rng(int(seed) + 2025)
    x = np.asarray(x, dtype=np.float32)
    n_rows, n_samples = x.shape
    out = np.empty_like(x)
    grid = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
    time = np.linspace(0.0, 10.0, n_samples, dtype=np.float32)
    feature_cols = [
        "qrs_visibility",
        "qrs_band_ratio",
        "baseline_step",
        "non_qrs_diff_p95",
        "band_15_30",
        "band_30_45",
        "amplitude_entropy",
        "flatline_ratio",
        "template_corr",
    ]
    centers = {
        "good": np.asarray([0.55, 0.70, 0.32, 0.055, 0.22, 0.02, 0.62, 0.28, 0.68], dtype=np.float32),
        "medium": np.asarray([0.22, 0.55, 0.55, 0.11, 0.24, 0.04, 0.70, 0.09, 0.56], dtype=np.float32),
        "bad": np.asarray([0.18, 0.60, 0.16, 0.40, 0.55, 0.18, 0.88, 0.02, 0.20], dtype=np.float32),
    }
    scales = {
        "good": np.asarray([0.35, 0.55, 0.35, 0.05, 0.18, 0.03, 0.14, 0.20, 0.20], dtype=np.float32),
        "medium": np.asarray([0.22, 0.40, 0.35, 0.09, 0.16, 0.05, 0.12, 0.10, 0.22], dtype=np.float32),
        "bad": np.asarray([0.20, 0.45, 0.18, 0.22, 0.30, 0.16, 0.13, 0.04, 0.18], dtype=np.float32),
    }
    if aggressive_lowqrs:
        centers["medium"] = np.asarray([0.08, 0.18, 0.32, 0.10, 0.22, 0.04, 0.70, 0.10, 0.45], dtype=np.float32)
        centers["bad"] = np.asarray([0.06, 0.18, 0.12, 0.36, 0.45, 0.16, 0.86, 0.03, 0.18], dtype=np.float32)
        scales["medium"] = np.asarray([0.10, 0.18, 0.38, 0.09, 0.18, 0.06, 0.15, 0.12, 0.25], dtype=np.float32)
        scales["bad"] = np.asarray([0.10, 0.20, 0.20, 0.24, 0.30, 0.18, 0.14, 0.05, 0.20], dtype=np.float32)

    def robust(row: np.ndarray) -> np.ndarray:
        med = np.median(row)
        q75 = np.percentile(row, 75)
        q25 = np.percentile(row, 25)
        scale = (q75 - q25) / 1.349
        std = np.std(row)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = std if std > 1e-6 else 1.0
        return ((row - med) / scale).astype(np.float32)

    def standard(row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32)
        return ((row - np.median(row)) / (np.std(row) + 1e-6)).astype(np.float32)

    def random_walk() -> np.ndarray:
        inc = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        rw = np.cumsum(moving_average_1d(inc, 33)).astype(np.float32)
        return standard(rw)

    def piecewise_step() -> np.ndarray:
        sig = np.zeros(n_samples, dtype=np.float32)
        n_steps = int(rng.integers(1, 5))
        for _ in range(n_steps):
            pos = int(rng.integers(n_samples // 8, 7 * n_samples // 8))
            sig[pos:] += float(rng.normal(0.0, 1.0))
        sig += 0.10 * random_walk()
        return standard(sig)

    def low_sine() -> np.ndarray:
        freq = float(rng.uniform(0.35, 1.35))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        sig = np.sin(2.0 * np.pi * freq * time + phase).astype(np.float32)
        sig += 0.15 * np.sin(2.0 * np.pi * rng.uniform(1.5, 3.0) * time + phase / 2.0).astype(np.float32)
        return standard(sig)

    def high_detail() -> np.ndarray:
        noise = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        high = noise - moving_average_1d(noise, 9)
        mid = moving_average_1d(noise, 19) - moving_average_1d(noise, 75)
        return standard(0.65 * high + 0.35 * mid)

    def monotonic_step() -> np.ndarray:
        pos = float(rng.uniform(-0.45, 0.45))
        width = float(rng.uniform(0.006, 0.08))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return standard(sign * (0.55 * grid + np.tanh((grid - pos) / width).astype(np.float32)))

    def candidate_set(row: np.ndarray, cls: str) -> np.ndarray:
        z = robust(row)
        base = [
            z,
            0.70 * z + 0.25 * piecewise_step() + 0.04 * high_detail(),
            0.45 * z + 0.45 * low_sine() + 0.08 * high_detail(),
            0.25 * z + 0.65 * piecewise_step() + 0.08 * high_detail(),
            0.10 * z + 0.85 * piecewise_step(),
            0.08 * z + 0.80 * monotonic_step() + 0.08 * high_detail(),
            0.05 * z + 0.80 * low_sine() + 0.10 * high_detail(),
            high_detail(),
            0.20 * z + 0.35 * piecewise_step() + 0.55 * high_detail(),
            0.05 * z + 0.25 * piecewise_step() + 0.75 * high_detail(),
            0.02 * z + 0.98 * piecewise_step(),
            0.02 * z + 0.98 * monotonic_step(),
        ]
        if cls == "good":
            weights = [1.0, 1.0, 0.9, 0.35, 0.15, 0.12, 0.4, 0.05, 0.1, 0.02, 0.05, 0.05]
        elif cls == "medium":
            weights = [0.2, 0.65, 0.85, 1.0, 1.0, 0.95, 0.95, 0.25, 0.7, 0.35, 0.9, 0.8]
        else:
            weights = [0.08, 0.2, 0.15, 0.55, 0.6, 0.55, 0.25, 0.9, 1.0, 1.0, 0.55, 0.35]
        rows = [standard(sig) for sig, w in zip(base, weights) if rng.random() <= w]
        if not rows:
            rows = [standard(base[0])]
        return np.stack(rows).astype(np.float32)

    for start in range(0, n_rows, 128):
        end = min(n_rows, start + 128)
        cand_rows: list[np.ndarray] = []
        owners: list[int] = []
        classes: list[str] = []
        for local_i, idx in enumerate(range(start, end)):
            cls = str(selected.iloc[idx]["y_class"])
            cands = candidate_set(x[idx], cls)
            cand_rows.append(cands)
            owners.extend([idx] * len(cands))
            classes.extend([cls] * len(cands))
        cand_x = np.concatenate(cand_rows, axis=0)
        cand_features = GEOM_FEATURES.compute_primitives(cand_x)
        fmat = cand_features[feature_cols].to_numpy(dtype=np.float32)
        scores = np.zeros(len(cand_x), dtype=np.float32)
        for i, cls in enumerate(classes):
            diff = (fmat[i] - centers[cls]) / scales[cls]
            scores[i] = float(np.nanmean(diff * diff))
            # Penalize the exact failure we saw: high-QRS synthetic medium/bad.
            if cls in {"medium", "bad"}:
                qrs_band = fmat[i, feature_cols.index("qrs_band_ratio")]
                qrs_vis = fmat[i, feature_cols.index("qrs_visibility")]
                if aggressive_lowqrs:
                    scores[i] += 12.0 * max(0.0, qrs_band - 0.90)
                    scores[i] += 8.0 * max(0.0, qrs_vis - 0.45)
                    # Allow true no-peak rows; they are exactly missing from the PTB pool.
                    scores[i] -= 0.8 if (qrs_band < 0.35 and qrs_vis < 0.20) else 0.0
                else:
                    scores[i] += 4.0 * max(0.0, qrs_band - 1.15)
                    scores[i] += 3.0 * max(0.0, qrs_vis - 0.65)
            if cls == "good":
                scores[i] += 1.5 * max(0.0, 0.18 - fmat[i, feature_cols.index("qrs_visibility")])
        owner_arr = np.asarray(owners, dtype=np.int64)
        for idx in range(start, end):
            mask = owner_arr == idx
            best = np.flatnonzero(mask)[np.argmin(scores[mask])]
            out[idx] = cand_x[best]
    return out.astype(np.float32)


def mixed_detector_failure_signals(x: np.ndarray, selected: pd.DataFrame, seed: int) -> tuple[np.ndarray, pd.Series]:
    """Blend BUT-like detector-failure morphology into PTB without replacing the whole class.

    The earlier aggressive variant moved all medium/bad examples toward low-QRS
    detector-failure shapes. That made the synthetic source distribution too
    narrow and encouraged class shortcuts. This variant keeps the aligned PTB
    body as the anchor and only injects a controlled detector-failure slice.
    """
    rng = np.random.default_rng(int(seed) + 7707)
    generated = rejection_detector_failure_signals(x, selected, seed, aggressive_lowqrs=True)
    labels = selected["y_class"].astype(str).to_numpy()
    probs = np.zeros(len(selected), dtype=np.float32)
    probs[labels == "good"] = 0.06
    probs[labels == "medium"] = 0.46
    probs[labels == "bad"] = 0.42
    replace = rng.random(len(selected)) < probs

    out = np.asarray(x, dtype=np.float32).copy()
    out[replace] = generated[replace]
    block = pd.Series(np.where(replace, "butlike_detector_failure_mix", "aligned_ptb_body"), index=selected.index)
    return out.astype(np.float32), block


def visible_qrs_detailmatch_signals(
    x: np.ndarray,
    selected: pd.DataFrame,
    seed: int,
    strong_medium: bool = False,
    extreme_but_medium: bool = False,
) -> tuple[np.ndarray, pd.Series]:
    """Match clean BUT morphology: QRS remains visible while detail/baseline degrade.

    The clean retained BUT protocol's medium windows are not mostly no-QRS.
    They show strong QRS evidence plus elevated baseline/detail corruption.
    This generator therefore preserves or sharpens QRS-like peaks and injects
    non-QRS detail/baseline artifacts in a controlled class-specific mixture.
    """
    rng = np.random.default_rng(int(seed) + 8808)
    x = np.asarray(x, dtype=np.float32)
    out = x.copy()
    n_rows, n_samples = x.shape
    time = np.linspace(0.0, 10.0, n_samples, dtype=np.float32)

    def standard(row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32)
        return ((row - np.median(row)) / (np.std(row) + 1e-6)).astype(np.float32)

    def robust(row: np.ndarray) -> np.ndarray:
        med = np.median(row)
        q75 = np.percentile(row, 75)
        q25 = np.percentile(row, 25)
        scale = (q75 - q25) / 1.349
        std = np.std(row)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = std if std > 1e-6 else 1.0
        return ((row - med) / scale).astype(np.float32)

    def qrs_emphasis(z: np.ndarray) -> np.ndarray:
        absz = np.abs(z)
        thr = max(float(np.percentile(absz, 90)), float(np.mean(absz) + 0.45 * np.std(absz)))
        peak_idx: list[int] = []
        last = -9999
        min_gap = 26
        for j in range(2, n_samples - 2):
            if absz[j] < thr or j - last < min_gap:
                continue
            if absz[j] >= absz[j - 1] and absz[j] >= absz[j + 1]:
                peak_idx.append(j)
                last = j
        if not peak_idx:
            return np.zeros(n_samples, dtype=np.float32)
        grid = np.arange(n_samples, dtype=np.float32)
        pulse = np.zeros(n_samples, dtype=np.float32)
        width = float(rng.uniform(2.5, 5.0))
        for p in peak_idx[:18]:
            pulse += float(np.sign(z[p]) or 1.0) * np.exp(-0.5 * ((grid - float(p)) / width) ** 2).astype(np.float32)
        return standard(pulse)

    def baseline_steps() -> np.ndarray:
        sig = np.zeros(n_samples, dtype=np.float32)
        for _ in range(int(rng.integers(2, 6))):
            pos = int(rng.integers(n_samples // 12, 11 * n_samples // 12))
            amp = float(rng.normal(0.0, 1.0))
            width = float(rng.uniform(2.0, 18.0))
            grid = np.arange(n_samples, dtype=np.float32)
            sig += amp * (1.0 / (1.0 + np.exp(-(grid - pos) / width))).astype(np.float32)
        sig += 0.35 * np.sin(2.0 * np.pi * rng.uniform(0.15, 0.55) * time + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        return standard(sig)

    def detail_noise() -> np.ndarray:
        noise = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
        high = noise - moving_average_1d(noise, 11)
        mid = moving_average_1d(noise, 17) - moving_average_1d(noise, 83)
        burst = np.zeros(n_samples, dtype=np.float32)
        for _ in range(int(rng.integers(2, 5))):
            center = int(rng.integers(n_samples // 10, 9 * n_samples // 10))
            width = float(rng.uniform(18.0, 80.0))
            freq = float(rng.uniform(12.0, 32.0))
            env = np.exp(-0.5 * ((np.arange(n_samples, dtype=np.float32) - center) / width) ** 2)
            burst += env.astype(np.float32) * np.sin(2.0 * np.pi * freq * time + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        return standard(0.45 * high + 0.25 * mid + 0.30 * burst)

    labels = selected["y_class"].astype(str).to_numpy()
    block = np.full(n_rows, "aligned_ptb_body", dtype=object)
    for i in range(n_rows):
        cls = labels[i]
        z = robust(x[i])
        qrs = qrs_emphasis(z)
        base = baseline_steps()
        detail = detail_noise()
        if cls == "good":
            if rng.random() < 0.14:
                sig = 0.88 * z + 0.18 * qrs + 0.08 * base + 0.06 * detail
                block[i] = "visible_qrs_light_good_boundary"
            else:
                sig = z
        elif cls == "medium":
            if extreme_but_medium and rng.random() < 0.94:
                sig = 0.20 * z + 4.70 * qrs + 0.90 * base + 1.25 * detail
                block[i] = "extreme_butlike_visible_qrs_medium"
            elif strong_medium and rng.random() < 0.92:
                sig = 0.44 * z + 1.65 * qrs + 0.78 * base + 1.05 * detail
                block[i] = "strong_visible_qrs_medium_detail_baseline"
            elif rng.random() < 0.74:
                sig = 0.76 * z + 0.35 * qrs + 0.42 * base + 0.38 * detail
                block[i] = "visible_qrs_medium_detail_baseline"
            else:
                sig = z
        elif cls == "bad":
            if rng.random() < 0.62:
                sig = 0.56 * z + 0.16 * qrs + 0.38 * base + 0.88 * detail
                block[i] = "controlled_bad_visibleqrs_stress"
            else:
                sig = z
        else:
            sig = z
        out[i] = standard(sig)
    return out.astype(np.float32), pd.Series(block, index=selected.index)


def buttrain_style_replay_signals(x: np.ndarray, selected: pd.DataFrame, but_protocol: Path, seed: int) -> tuple[np.ndarray, pd.Series]:
    """Replay clean BUT train artifact morphology onto PTB rows without using BUT test.

    This is a diagnostic generator: it tests whether the remaining PTB->BUT gap
    is mainly caused by hand-written artifact synthesis failing to match real
    BUT morphology. It uses only the clean BUT train split as style/residual
    templates and keeps the selected PTB labels/splits.
    """
    rng = np.random.default_rng(int(seed) + 11111)
    x = np.asarray(x, dtype=np.float32)
    out = x.copy()

    zfile = np.load(but_protocol / "signals.npz")
    key = "X" if "X" in zfile.files else zfile.files[0]
    but_x = np.asarray(zfile[key], dtype=np.float32)
    if but_x.ndim == 3:
        but_x = but_x[:, 0, :]
    but_atlas = pd.read_csv(but_protocol / "original_region_atlas.csv")
    train_mask = but_atlas["split"].astype(str).eq("train").to_numpy()
    but_x = but_x[train_mask]
    but_atlas = but_atlas.loc[train_mask].reset_index(drop=True)
    by_class = {
        cls: np.flatnonzero(but_atlas["class_name"].astype(str).to_numpy() == cls)
        for cls in ["good", "medium", "bad"]
    }

    def standard(row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32)
        return ((row - np.median(row)) / (np.std(row) + 1e-6)).astype(np.float32)

    def robust(row: np.ndarray) -> np.ndarray:
        med = np.median(row)
        q75 = np.percentile(row, 75)
        q25 = np.percentile(row, 25)
        scale = (q75 - q25) / 1.349
        std = np.std(row)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = std if std > 1e-6 else 1.0
        return ((row - med) / scale).astype(np.float32)

    labels = selected["y_class"].astype(str).to_numpy()
    block = np.full(len(selected), "aligned_ptb_body", dtype=object)
    for i, cls in enumerate(labels):
        idx_pool = by_class.get(cls, np.asarray([], dtype=np.int64))
        if len(idx_pool) == 0:
            continue
        if cls == "good":
            use_style = rng.random() < 0.18
        elif cls == "medium":
            use_style = rng.random() < 0.82
        elif cls == "bad":
            use_style = rng.random() < 0.72
        else:
            use_style = False
        if not use_style:
            continue
        style = robust(but_x[int(rng.choice(idx_pool))])
        base = moving_average_1d(style, 251)
        detail = style - moving_average_1d(style, 21)
        mid = moving_average_1d(style, 17) - moving_average_1d(style, 83)
        z = robust(x[i])
        if cls == "good":
            sig = 0.78 * z + 0.24 * base + 0.14 * detail
            block[i] = "buttrain_light_good_style"
        elif cls == "medium":
            sig = 0.24 * z + 0.42 * moving_average_1d(z, 9) + 1.10 * base + 1.10 * detail + 0.45 * mid
            block[i] = "buttrain_medium_style_replay"
        else:
            sig = 0.15 * z + 0.95 * style + 0.55 * detail
            block[i] = "buttrain_bad_style_replay"
        out[i] = standard(sig)
    return out.astype(np.float32), pd.Series(block, index=selected.index)


def select_aligned_rows(
    pool_frame: pd.DataFrame,
    pool_features: pd.DataFrame,
    but_train: pd.DataFrame,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    selected: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for cls in ["good", "medium", "bad"]:
        pool_mask = pool_frame["y_class"].astype(str).eq(cls).to_numpy()
        target_mask = but_train["class_name"].astype(str).eq(cls).to_numpy()
        cand = pool_frame.loc[pool_mask].copy().reset_index(drop=True)
        cand_features = pool_features.loc[pool_mask].reset_index(drop=True)
        target = but_train.loc[target_mask].copy().reset_index(drop=True)
        if cand.empty or target.empty:
            raise RuntimeError(f"missing class for alignment: {cls}")
        cols = [c for c in MATCH_FEATURES if c in cand_features.columns or c in target.columns]
        tmat = feature_matrix(target, cols)
        cmat = feature_matrix(cand_features, cols)
        mu = np.nanmedian(tmat, axis=0)
        q75 = np.nanpercentile(tmat, 75, axis=0)
        q25 = np.nanpercentile(tmat, 25, axis=0)
        scale = np.where((q75 - q25) > 1e-6, (q75 - q25) / 1.349, np.nanstd(tmat, axis=0))
        scale = np.where(scale > 1e-6, scale, 1.0)
        tnorm = (tmat - mu[None, :]) / scale[None, :]
        cnorm = (cmat - mu[None, :]) / scale[None, :]
        nn = NearestNeighbors(n_neighbors=min(16, len(cnorm)), metric="euclidean").fit(cnorm)
        target_order = rng.permutation(len(tnorm))
        picked: list[int] = []
        used: set[int] = set()
        for tidx in np.resize(target_order, max(len(target_order), int(per_class) * 2)):
            neigh = nn.kneighbors(tnorm[int(tidx) : int(tidx) + 1], return_distance=False)[0]
            neigh = rng.permutation(neigh)
            for cidx in neigh:
                cidx = int(cidx)
                if cidx not in used:
                    used.add(cidx)
                    picked.append(cidx)
                    break
            if len(picked) >= int(per_class):
                break
        if len(picked) < int(per_class):
            # Fill with globally closest candidates if unique nearest matching was exhausted.
            cand_to_target = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(tnorm)
            dist, _ = cand_to_target.kneighbors(cnorm, return_distance=True)
            order = np.argsort(dist[:, 0])
            for cidx in order:
                cidx = int(cidx)
                if cidx not in used:
                    picked.append(cidx)
                    used.add(cidx)
                if len(picked) >= int(per_class):
                    break
        chosen = cand.iloc[picked[: int(per_class)]].copy()
        chosen["align_target_class"] = cls
        chosen["align_match_features"] = ",".join(cols)
        selected.append(chosen)
        audit_rows.append(
            {
                "class_name": cls,
                "target_rows": int(len(target)),
                "pool_rows": int(len(cand)),
                "selected_rows": int(len(chosen)),
                "unique_sources": int((chosen["pool_file"].astype(str) + ":" + chosen["source_idx"].astype(str)).nunique()),
            }
        )
    out = pd.concat(selected, ignore_index=True)
    out.attrs["audit_rows"] = audit_rows
    return out


def materialize_aligned_protocol(args: argparse.Namespace) -> Path:
    ensure_dirs()
    protocol = CROSS_OUT / f"protocol_ptb_buttrain_aligned_pc{int(args.per_class)}_s{int(args.seed)}"
    atlas_path = protocol / "original_region_atlas.csv"
    signals_path = protocol / "signals.npz"
    if atlas_path.exists() and signals_path.exists() and not bool(args.force_protocol):
        return protocol
    protocol.mkdir(parents=True, exist_ok=True)

    pool_x, pool_frame = load_pool()
    pool_primitives = GEOM_FEATURES.compute_primitives(pool_x)
    y_pool = pool_frame["y_class"].astype(str).map(EVENT.CLASS_TO_INT).to_numpy(dtype=np.int64)
    # Fit pool PCA only for diagnostic columns; classifier input remains waveform-derived channels.
    train_mask_pool = np.ones(len(pool_frame), dtype=bool)
    pool_state = GEOM_FEATURES.fit_feature_state(pool_primitives, y_pool, train_mask_pool)
    pool_features = GEOM_FEATURES.assemble_features(pool_primitives, pool_state)
    for col in pool_primitives.columns:
        if col not in pool_features.columns:
            pool_features[col] = pool_primitives[col]

    but_protocol = Path(args.but_protocol)
    but_atlas = pd.read_csv(but_protocol / "original_region_atlas.csv")
    but_train = but_atlas.loc[but_atlas["split"].astype(str).eq("train")].copy()
    selected = select_aligned_rows(pool_frame, pool_features, but_train, int(args.per_class), int(args.seed))
    audit_rows = selected.attrs.get("audit_rows", [])
    selected = selected.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
    selected["split"] = split_groups_stratified(selected, int(args.seed))
    pool_indices = selected["pool_global_idx"].astype(int).to_numpy()
    x_selected = pool_x[pool_indices].astype(np.float32)
    if str(args.variant) == "v2_morphcal":
        x_selected = calibrate_signals_to_but_train(x_selected, selected, but_train, int(args.seed))
    elif str(args.variant) == "v3_detectorfail":
        x_selected = detector_failure_signals(x_selected, selected, int(args.seed))
    elif str(args.variant) == "v4_peakdrop":
        x_selected = peakdrop_signals(x_selected, selected, int(args.seed))
    elif str(args.variant) == "v5_reject_detectorfail":
        x_selected = rejection_detector_failure_signals(x_selected, selected, int(args.seed))
    elif str(args.variant) == "v6_lowqrs_aggressive":
        x_selected = rejection_detector_failure_signals(x_selected, selected, int(args.seed), aggressive_lowqrs=True)
    elif str(args.variant) == "v7_mixed_detectorfail":
        x_selected, detector_mix_block = mixed_detector_failure_signals(x_selected, selected, int(args.seed))
        selected["detector_mix_block"] = detector_mix_block.to_numpy()
    elif str(args.variant) == "v8_visibleqrs_detailmatch":
        x_selected, detector_mix_block = visible_qrs_detailmatch_signals(x_selected, selected, int(args.seed))
        selected["detector_mix_block"] = detector_mix_block.to_numpy()
    elif str(args.variant) == "v9_strong_visibleqrs_medium":
        x_selected, detector_mix_block = visible_qrs_detailmatch_signals(x_selected, selected, int(args.seed), strong_medium=True)
        selected["detector_mix_block"] = detector_mix_block.to_numpy()
    elif str(args.variant) == "v10_butmedium_extreme":
        x_selected, detector_mix_block = visible_qrs_detailmatch_signals(x_selected, selected, int(args.seed), extreme_but_medium=True)
        selected["detector_mix_block"] = detector_mix_block.to_numpy()
    elif str(args.variant) == "v11_buttrain_style_replay":
        x_selected, detector_mix_block = buttrain_style_replay_signals(x_selected, selected, but_protocol, int(args.seed))
        selected["detector_mix_block"] = detector_mix_block.to_numpy()
    if str(args.variant) in {"v2_morphcal", "v3_detectorfail", "v4_peakdrop", "v5_reject_detectorfail", "v6_lowqrs_aggressive", "v7_mixed_detectorfail", "v8_visibleqrs_detailmatch", "v9_strong_visibleqrs_medium", "v10_butmedium_extreme", "v11_buttrain_style_replay"}:
        selected_primitives = GEOM_FEATURES.compute_primitives(x_selected)
        selected_y = selected["y_class"].astype(str).map(EVENT.CLASS_TO_INT).to_numpy(dtype=np.int64)
        selected_state = GEOM_FEATURES.fit_feature_state(selected_primitives, selected_y, selected["split"].astype(str).eq("train").to_numpy())
        sel_features = GEOM_FEATURES.assemble_features(selected_primitives, selected_state)
        for col in selected_primitives.columns:
            if col not in sel_features.columns:
                sel_features[col] = selected_primitives[col]
    else:
        sel_features = pool_features.iloc[pool_indices].reset_index(drop=True)
    x_save = x_selected[:, None, :].astype(np.float32)
    y = selected["y_class"].astype(str).map(EVENT.CLASS_TO_INT).to_numpy(dtype=np.int64)

    atlas = pd.DataFrame(
        {
            "idx": np.arange(len(selected), dtype=np.int64),
            "source_idx": pd.to_numeric(selected["source_idx"], errors="coerce").fillna(selected["pool_row"]).astype(int).to_numpy(),
            "split": selected["split"].astype(str).to_numpy(),
            "y": y,
            "class_name": selected["y_class"].astype(str).to_numpy(),
            "record_id": "ptb_align_" + selected["pool_file"].astype(str) + "_" + selected["source_idx"].astype(str),
            "subject_id": "ptb_align_" + selected["pool_file"].astype(str) + "_" + selected["source_idx"].astype(str),
            "dataset": "ptb_synthetic_buttrain_aligned",
            "original_region": selected.get("profile_block", selected.get("target_original_region", pd.Series(["ptb_aligned"] * len(selected)))).fillna("ptb_aligned").astype(str).to_numpy(),
            "bank_role": selected.get("bank_role", pd.Series([""] * len(selected))).fillna("").astype(str).to_numpy(),
            "profile_block": selected.get("profile_block", pd.Series([""] * len(selected))).fillna("").astype(str).to_numpy(),
            "detector_mix_block": selected.get("detector_mix_block", pd.Series(["aligned_ptb_body"] * len(selected))).fillna("aligned_ptb_body").astype(str).to_numpy(),
            "pool_file": selected["pool_file"].astype(str).to_numpy(),
            "pool_folder": selected["pool_folder"].astype(str).to_numpy(),
            "pool_row": selected["pool_row"].astype(int).to_numpy(),
            "alignment_target": "clean_but_train_only",
        }
    )
    for col in EVENT.FORMAL_FEATURE_COLUMNS:
        if col in sel_features.columns:
            atlas[col] = pd.to_numeric(sel_features[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        else:
            atlas[col] = 0.0

    np.savez_compressed(signals_path, X=x_save)
    atlas.to_csv(atlas_path, index=False)
    selected.to_csv(protocol / "metadata.csv", index=False)
    split_counts = atlas.groupby(["split", "class_name"]).size().reset_index(name="n")
    split_counts.to_csv(protocol / "split_counts.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(protocol / "alignment_audit.csv", index=False)
    pool_counts = selected.groupby(["y_class", "pool_file"]).size().reset_index(name="n")
    pool_counts.to_csv(protocol / "selected_pool_composition.csv", index=False)
    summary = {
        "created_at": now(),
        "n": int(len(atlas)),
        "per_class": int(args.per_class),
        "class_counts": atlas["class_name"].value_counts().to_dict(),
        "split_counts": split_counts.to_dict(orient="records"),
        "alignment_target": "clean BUT train split only; no clean BUT test rows used for matching",
        "match_features": MATCH_FEATURES,
        "pool_files": POOL_FILES,
        "but_protocol": str(but_protocol),
        "variant": str(args.variant),
        "morphology_calibration": str(args.variant) in {"v2_morphcal", "v3_detectorfail", "v4_peakdrop", "v5_reject_detectorfail", "v6_lowqrs_aggressive", "v7_mixed_detectorfail", "v8_visibleqrs_detailmatch", "v9_strong_visibleqrs_medium", "v10_butmedium_extreme", "v11_buttrain_style_replay"},
    }
    (protocol / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return protocol


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return BASE.summarize_metrics(metrics)


def write_report(summary: pd.DataFrame, metrics_path: Path, records_path: Path, recovery_path: Path, ptb_protocol: Path, but_protocol: Path) -> None:
    def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
        if df.empty:
            return "_empty_"
        view = df.head(max_rows).copy()
        cols = list(view.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    key = summary.loc[summary["bucket"].isin(["cross_test", "cross_all"])].copy()
    cols = [
        "direction",
        "candidate",
        "bucket",
        "runs",
        "acc",
        "macro_f1",
        "good_recall",
        "medium_recall",
        "bad_recall",
        "record_macro_supported_f1",
    ]
    cols = [c for c in cols if c in key.columns]
    report = [
        "# Event-Factorized Cross-Dataset Aligned-PTB Report",
        "",
        f"- Generated: {now()}",
        "- Scope: external-only experiment; no `src/sqi_pipeline` changes.",
        "- PTB protocol is selected to match clean BUT train-split waveform-computable feature distribution.",
        "- Clean BUT test rows are not used for alignment.",
        "- Formal model input remains waveform-derived channels only.",
        "",
        "## Protocols",
        "",
        f"- Aligned PTB protocol: `{ptb_protocol}`",
        f"- Clean BUT protocol: `{but_protocol}`",
        "",
        "## Cross-Dataset Summary",
        "",
        md_table(key[cols].sort_values(["direction", "bucket", "acc"], ascending=[True, True, False])),
        "",
        "## Output Files",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Summary: `{CROSS_OUT / 'aligned_cross_dataset_summary.csv'}`",
        f"- Record metrics: `{records_path}`",
        f"- Feature recovery: `{recovery_path}`",
        f"- Checkpoints/logs: `{CROSS_RUN_DIR}`",
    ]
    CROSS_REPORT.write_text("\n".join(report), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    ptb_protocol = materialize_aligned_protocol(args)
    but_protocol = Path(args.but_protocol)
    metric_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    seeds = [int(args.seed) + 1000 * i for i in range(int(args.seeds))]
    top3 = BASE.TOP3
    if args.candidates:
        allowed = set(args.candidates.split(","))
        top3 = [item for item in top3 if item[1] in allowed]
    # Repoint the base helper's run dir so aligned logs/checkpoints stay separate.
    BASE.CROSS_RUN_DIR = CROSS_RUN_DIR
    for seed_index, seed in enumerate(seeds):
        for stage, candidate in top3:
            cfg = EVENT.candidate_grid(stage, int(seed))[candidate]
            cfg["stage"] = stage
            for direction, train_protocol, test_protocol in [
                ("ptb_aligned_to_but", ptb_protocol, but_protocol),
                ("but_to_ptb_aligned", but_protocol, ptb_protocol),
            ]:
                res = BASE.train_cross(direction, candidate, dict(cfg), train_protocol, test_protocol, args, seed_index)
                metric_rows.extend(res["metrics"])
                record_rows.extend(res["records"])
                recovery_rows.extend(res["recovery"])
                metrics = pd.DataFrame(metric_rows)
                records = pd.DataFrame(record_rows)
                recovery = pd.DataFrame(recovery_rows)
                metrics_path = CROSS_OUT / "aligned_cross_dataset_metrics.csv"
                records_path = CROSS_OUT / "aligned_cross_dataset_record_metrics.csv"
                recovery_path = CROSS_OUT / "aligned_cross_dataset_feature_recovery.csv"
                metrics.to_csv(metrics_path, index=False)
                records.to_csv(records_path, index=False)
                recovery.to_csv(recovery_path, index=False)
                summary = summarize_metrics(metrics)
                summary.to_csv(CROSS_OUT / "aligned_cross_dataset_summary.csv", index=False)
                write_report(summary, metrics_path, records_path, recovery_path, ptb_protocol, but_protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="run", choices=["materialize_protocol", "run"])
    parser.add_argument(
        "--variant",
        type=str,
        default="v1",
        choices=["v1", "v2_morphcal", "v3_detectorfail", "v4_peakdrop", "v5_reject_detectorfail", "v6_lowqrs_aggressive", "v7_mixed_detectorfail", "v8_visibleqrs_detailmatch", "v9_strong_visibleqrs_medium", "v10_butmedium_extreme", "v11_buttrain_style_replay"],
    )
    parser.add_argument("--but-protocol", type=str, default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--force-protocol", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--record-balanced-sampler", action="store_true", default=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--max-cross-rows", type=int, default=0)
    parser.add_argument("--max-cross-all-rows", type=int, default=0)
    parser.add_argument("--candidates", type=str, default="")
    args = parser.parse_args()
    configure_variant(str(args.variant))
    ensure_dirs()
    if args.stage == "materialize_protocol":
        print(materialize_aligned_protocol(args))
    else:
        run(args)
        print(CROSS_REPORT)


if __name__ == "__main__":
    main()
