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
    if str(args.variant) in {"v2_morphcal", "v3_detectorfail", "v4_peakdrop"}:
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
        "morphology_calibration": str(args.variant) in {"v2_morphcal", "v3_detectorfail", "v4_peakdrop"},
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
    parser.add_argument("--variant", type=str, default="v1", choices=["v1", "v2_morphcal", "v3_detectorfail", "v4_peakdrop"])
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
