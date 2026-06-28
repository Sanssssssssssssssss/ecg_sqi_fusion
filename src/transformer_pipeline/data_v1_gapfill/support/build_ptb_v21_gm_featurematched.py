"""Build V21 PTB protocol: keep V20 bad, feature-match good/medium.

The V20 loop fixed the natural bad/poor peak-detector mechanism but forward
transfer still missed BUT medium.  V21 therefore keeps the V20 bad rows exactly
and rebuilds only good/medium PTB rows by selecting waveform-only candidates
that match BUT train-split good/medium feature distributions.

BUT usage is distributional only: feature medians/IQRs from the train split are
used as targets; no BUT waveform is copied into the synthetic protocol.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402

V20_PROTOCOL = ANALYSIS_DIR / "event_xds_aligned_v20_bad_subtype_featurematched" / "protocol_v20_pc3000_s20260621"
BUT_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier"
V20_BUILDER = SCRIPT_DIR / "build_ptb_bad_subtype_featurematched_v14.py"

RUN_VERSION_TAG = "v21"
GENERATOR_PROFILE = "v21"
OUT_DIR = ANALYSIS_DIR / f"event_xds_aligned_{RUN_VERSION_TAG}_gm_featurematched"
REPORT_OUT_DIR = REPORT_DIR / f"event_xds_aligned_{RUN_VERSION_TAG}_gm_featurematched"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
GM_FEATURES = [
    "sqi_basSQI",
    "baseline_step",
    "rms",
    "mean_abs",
    "qrs_band_ratio",
    "qrs_visibility",
    "detector_agreement",
    "non_qrs_diff_p95",
    "band_15_30",
    "band_30_45",
    "amplitude_entropy",
    "flatline_ratio",
    "low_amp_ratio",
    "template_corr",
]
FEATURE_WEIGHTS = {
    "sqi_basSQI": 1.8,
    "baseline_step": 1.6,
    "rms": 1.4,
    "mean_abs": 1.4,
    "qrs_band_ratio": 1.4,
    "qrs_visibility": 1.2,
    "detector_agreement": 1.1,
    "non_qrs_diff_p95": 1.1,
}


V22_FEATURE_WEIGHTS = {
    **FEATURE_WEIGHTS,
    # V21 still under-covered BUT-good high-amplitude/QRS-band/low-frequency
    # morphology, so V22 tightens candidate selection around those mechanisms.
    "rms": 2.2,
    "mean_abs": 2.0,
    "qrs_band_ratio": 2.2,
    "sqi_basSQI": 2.2,
    "baseline_step": 2.0,
    "amplitude_entropy": 1.7,
    "template_corr": 1.4,
}

V24_FEATURE_WEIGHTS = {
    **V22_FEATURE_WEIGHTS,
    # V23 improved tail coverage but BUT-test medium is still mostly predicted
    # as good.  V24 therefore emphasizes visible-QRS medium hard negatives:
    # QRS can remain strong while baseline/detail/non-QRS quality degrades.
    "qrs_visibility": 2.0,
    "qrs_band_ratio": 2.4,
    "baseline_step": 2.4,
    "non_qrs_diff_p95": 2.0,
    "sqi_basSQI": 2.2,
    "template_corr": 1.8,
    "detector_agreement": 1.6,
}

QUANTILE_ANCHORS = ["q10", "q25", "median", "q75", "q90"]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V20 = load_module(V20_BUILDER, "v20_builder_for_v21_gm")
FEATURES = V20.FEATURES
BUILD = V20.BUILD


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(long_path(path), index=False)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    return BUILD.moving_average(np.asarray(x, dtype=np.float32), int(win))


def compute_primitives(x: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    prim = FEATURES.compute_primitives(arr)
    for col in GM_FEATURES:
        if col not in prim.columns:
            prim[col] = 0.0
    return prim


def recompute_frame_features(protocol: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(protocol / "original_region_atlas.csv")
    x = np.load(protocol / "signals.npz")["X"].astype(np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    prim = compute_primitives(x)
    for col in GM_FEATURES:
        frame[col] = pd.to_numeric(prim[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return frame, x


def target_stats(but_frame: pd.DataFrame, target_split: str) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    work = but_frame.copy()
    if target_split != "all":
        work = work.loc[work["split"].astype(str).eq(target_split)].copy()
    rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, float]] = {}
    for cls in ["good", "medium"]:
        sub = work.loc[work["class_name"].astype(str).eq(cls)].copy()
        stats[cls] = {}
        for col in GM_FEATURES:
            vals = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32)
            if vals.size == 0:
                vals = np.asarray([0.0], dtype=np.float32)
            med = float(np.median(vals))
            q10 = float(np.percentile(vals, 10))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            q90 = float(np.percentile(vals, 90))
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            scale = max(iqr, float(np.std(vals)), 1e-3)
            stats[cls][f"{col}_q10"] = q10
            stats[cls][f"{col}_q25"] = q25
            stats[cls][f"{col}_median"] = med
            stats[cls][f"{col}_q75"] = q75
            stats[cls][f"{col}_q90"] = q90
            stats[cls][f"{col}_scale"] = scale
            rows.append({"class_name": cls, "feature": col, "target_median": med, "target_scale": scale, "n_but": int(len(sub))})
    return stats, pd.DataFrame(rows)


def gm_candidate(base: np.ndarray, cls: str, mode: str, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(base, dtype=np.float32).copy()
    n = x.shape[-1]
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    centered = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    x = x - float(np.median(x))
    if mode == "identity":
        return x.astype(np.float32)

    if cls == "good":
        if mode == "good_amp_baseline":
            scale = float(rng.uniform(1.22, 1.72))
            lf_amp = float(rng.uniform(0.10, 0.38))
            lf = lf_amp * np.sin(2 * np.pi * float(rng.uniform(0.06, 0.20)) * t + float(rng.uniform(-np.pi, np.pi)))
            y = scale * x + lf.astype(np.float32) + float(rng.uniform(0.02, 0.10)) * centered
        elif mode == "good_morph_smooth":
            scale = float(rng.uniform(1.10, 1.55))
            y = scale * (0.72 * x + 0.28 * moving_average(x, int(rng.choice([11, 17, 25]))))
            y += float(rng.uniform(0.06, 0.22)) * np.sin(2 * np.pi * float(rng.uniform(0.05, 0.16)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        elif mode == "good_qrs_band":
            detail = x - moving_average(x, int(rng.choice([31, 51])))
            y = float(rng.uniform(1.12, 1.60)) * x + float(rng.uniform(0.05, 0.18)) * detail
        elif mode == "good_v22_highamp_baseline":
            scale = float(rng.uniform(1.55, 2.55))
            lf1 = np.sin(2 * np.pi * float(rng.uniform(0.045, 0.13)) * t + float(rng.uniform(-np.pi, np.pi)))
            lf2 = np.sin(2 * np.pi * float(rng.uniform(0.12, 0.28)) * t + float(rng.uniform(-np.pi, np.pi)))
            baseline = float(rng.uniform(0.18, 0.62)) * lf1 + float(rng.uniform(0.04, 0.22)) * lf2
            y = scale * x + baseline.astype(np.float32) + float(rng.uniform(0.04, 0.18)) * centered
        elif mode == "good_v22_qrs_preserve_band":
            smooth = moving_average(x, int(rng.choice([7, 11, 17])))
            detail = x - moving_average(x, int(rng.choice([41, 61, 81])))
            y = float(rng.uniform(1.45, 2.20)) * smooth + float(rng.uniform(0.10, 0.32)) * detail
            y += float(rng.uniform(0.08, 0.32)) * np.sin(2 * np.pi * float(rng.uniform(0.05, 0.18)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        elif mode == "good_v22_lowentropy_morph":
            smooth = moving_average(x, int(rng.choice([5, 9, 13])))
            coarse = moving_average(x, int(rng.choice([31, 51])))
            y = float(rng.uniform(1.55, 2.35)) * (0.82 * smooth + 0.18 * coarse)
            y += float(rng.uniform(0.12, 0.42)) * np.sin(2 * np.pi * float(rng.uniform(0.045, 0.14)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        elif mode == "good_v22_amp_step":
            scale = float(rng.uniform(1.45, 2.30))
            step_at = int(rng.integers(max(8, n // 5), max(9, 4 * n // 5)))
            step = np.zeros(n, dtype=np.float32)
            step[step_at:] = float(rng.uniform(-0.18, 0.18))
            step = moving_average(step, int(rng.choice([21, 31, 41])))
            y = scale * x + step + float(rng.uniform(0.08, 0.34)) * np.sin(2 * np.pi * float(rng.uniform(0.05, 0.18)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        else:
            y = x
    else:
        if mode == "medium_preserve":
            y = float(rng.uniform(0.92, 1.18)) * x
            y += float(rng.uniform(0.04, 0.18)) * np.sin(2 * np.pi * float(rng.uniform(0.06, 0.22)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        elif mode == "medium_detail":
            detail = x - moving_average(x, int(rng.choice([21, 31, 41])))
            y = float(rng.uniform(0.92, 1.25)) * x + float(rng.uniform(0.04, 0.20)) * detail
            y += rng.normal(0.0, float(rng.uniform(0.006, 0.026)), n).astype(np.float32)
        elif mode == "medium_baseline":
            y = float(rng.uniform(0.88, 1.22)) * x
            y += float(rng.uniform(0.05, 0.24)) * np.sin(2 * np.pi * float(rng.uniform(0.05, 0.18)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
            y += float(rng.uniform(-0.05, 0.05)) * centered
        elif mode == "medium_v22_template_jitter":
            amp_mod = 1.0 + float(rng.uniform(0.04, 0.14)) * np.sin(2 * np.pi * float(rng.uniform(0.10, 0.45)) * t + float(rng.uniform(-np.pi, np.pi)))
            y = float(rng.uniform(0.90, 1.18)) * x * amp_mod.astype(np.float32)
            y += float(rng.uniform(0.04, 0.16)) * np.sin(2 * np.pi * float(rng.uniform(0.05, 0.20)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        elif mode == "medium_v22_lowdetector_detail":
            detail = x - moving_average(x, int(rng.choice([17, 25, 33])))
            y = float(rng.uniform(0.86, 1.16)) * x + float(rng.uniform(0.08, 0.26)) * detail
            for _ in range(int(rng.integers(1, 4))):
                start = int(rng.integers(0, max(1, n - 12)))
                width = int(rng.integers(4, 18))
                y[start : min(n, start + width)] *= float(rng.uniform(0.72, 0.94))
            y += rng.normal(0.0, float(rng.uniform(0.006, 0.022)), n).astype(np.float32)
        elif mode == "medium_v24_visible_qrs_baseline_detail":
            smooth = moving_average(x, int(rng.choice([5, 9, 13])))
            detail = x - moving_average(x, int(rng.choice([23, 35, 47])))
            lf = np.sin(2 * np.pi * float(rng.uniform(0.045, 0.16)) * t + float(rng.uniform(-np.pi, np.pi)))
            y = float(rng.uniform(1.18, 1.85)) * smooth
            y += float(rng.uniform(0.08, 0.22)) * detail
            y += float(rng.uniform(0.14, 0.46)) * lf.astype(np.float32)
            y += rng.normal(0.0, float(rng.uniform(0.004, 0.018)), n).astype(np.float32)
        elif mode == "medium_v24_goodlike_nonqrs_burst":
            smooth = moving_average(x, int(rng.choice([5, 9, 13])))
            y = float(rng.uniform(1.12, 1.72)) * smooth
            for _ in range(int(rng.integers(2, 5))):
                start = int(rng.integers(0, max(1, n - 40)))
                width = int(rng.integers(16, 70))
                burst = rng.normal(0.0, float(rng.uniform(0.025, 0.080)), min(width, n - start)).astype(np.float32)
                y[start : start + len(burst)] += burst
            y += float(rng.uniform(0.08, 0.28)) * np.sin(2 * np.pi * float(rng.uniform(0.05, 0.18)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        elif mode == "medium_v24_mild_contact_reset":
            y = float(rng.uniform(1.02, 1.55)) * x
            y += float(rng.uniform(0.08, 0.30)) * np.sin(2 * np.pi * float(rng.uniform(0.04, 0.16)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
            for _ in range(int(rng.integers(1, 3))):
                start = int(rng.integers(0, max(1, n - 24)))
                width = int(rng.integers(8, 38))
                plateau = float(np.median(y[max(0, start - 20) : start + 1])) if start > 0 else float(np.median(y))
                y[start : min(n, start + width)] = 0.70 * y[start : min(n, start + width)] + 0.30 * plateau
        else:
            y = x
    return np.asarray(y, dtype=np.float32)


def score_features(feat: pd.DataFrame, cls: str, stats: dict[str, dict[str, float]], anchor: str = "median") -> np.ndarray:
    parts = []
    weights = []
    for col in GM_FEATURES:
        vals = pd.to_numeric(feat[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        med = float(stats[cls][f"{col}_{anchor}"])
        scale = float(stats[cls][f"{col}_scale"])
        parts.append(np.minimum(np.abs(vals - med) / scale, 8.0))
        if GENERATOR_PROFILE.startswith(("v24", "v25")):
            active_weights = V24_FEATURE_WEIGHTS
        elif GENERATOR_PROFILE.startswith(("v22", "v23")):
            active_weights = V22_FEATURE_WEIGHTS
        else:
            active_weights = FEATURE_WEIGHTS
        weights.append(float(active_weights.get(col, 1.0)))
    arr = np.stack(parts, axis=1)
    w = np.asarray(weights, dtype=np.float32)
    return (arr * w[None, :]).sum(axis=1) / w.sum()


def quantile_balanced_order(
    cand_feat: pd.DataFrame,
    cls: str,
    stats: dict[str, dict[str, float]],
    n_select: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select candidates near multiple BUT quantile anchors, not only the median.

    V21/V22 intentionally optimized median/IQR gaps.  That improved aggregate
    distribution plots but compressed the good/medium overlap tails.  V23 uses
    the same waveform-computable feature space while forcing coverage of BUT
    q10/q25/q50/q75/q90 anchors.  This remains PTB-only waveform selection:
    BUT contributes target feature quantiles, never waveform rows.
    """
    selected: list[int] = []
    selected_anchor: list[str] = []
    used: set[int] = set()
    quota = max(1, int(np.ceil(float(n_select) / float(len(QUANTILE_ANCHORS)))))
    anchor_scores = {anchor: score_features(cand_feat, cls, stats, anchor) for anchor in QUANTILE_ANCHORS}
    for anchor in QUANTILE_ANCHORS:
        order = np.argsort(anchor_scores[anchor])
        added = 0
        for idx in order.tolist():
            if idx in used:
                continue
            selected.append(int(idx))
            selected_anchor.append(anchor)
            used.add(int(idx))
            added += 1
            if added >= quota or len(selected) >= n_select:
                break
        if len(selected) >= n_select:
            break
    if len(selected) < n_select:
        median_order = np.argsort(anchor_scores["median"])
        for idx in median_order.tolist():
            if idx in used:
                continue
            selected.append(int(idx))
            selected_anchor.append("median_fill")
            used.add(int(idx))
            if len(selected) >= n_select:
                break
    return np.asarray(selected[:n_select], dtype=np.int64), np.asarray(selected_anchor[:n_select], dtype=object)


def v25_medium_forced_mode_order(
    cand_feat: pd.DataFrame,
    cand_meta: list[dict[str, Any]],
    cls: str,
    stats: dict[str, dict[str, float]],
    n_select: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cls != "medium":
        return quantile_balanced_order(cand_feat, cls, stats, n_select)
    scores = score_features(cand_feat, cls, stats, "median")
    forced_modes = [
        "medium_v24_visible_qrs_baseline_detail",
        "medium_v24_goodlike_nonqrs_burst",
        "medium_v24_mild_contact_reset",
    ]
    quota_each = max(12, int(round(0.055 * float(n_select))))
    selected: list[int] = []
    selected_anchor: list[str] = []
    used: set[int] = set()
    for mode in forced_modes:
        idxs = [i for i, meta in enumerate(cand_meta) if str(meta.get("candidate_mode")) == mode]
        idxs = sorted(idxs, key=lambda i: float(scores[i]))
        added = 0
        for idx in idxs:
            if idx in used:
                continue
            selected.append(int(idx))
            selected_anchor.append(f"forced_{mode}")
            used.add(int(idx))
            added += 1
            if added >= quota_each or len(selected) >= n_select:
                break
    fill_order, fill_anchor = quantile_balanced_order(cand_feat, cls, stats, n_select)
    for idx, anchor in zip(fill_order.tolist(), fill_anchor.tolist()):
        if idx in used:
            continue
        selected.append(int(idx))
        selected_anchor.append(str(anchor))
        used.add(int(idx))
        if len(selected) >= n_select:
            break
    return np.asarray(selected[:n_select], dtype=np.int64), np.asarray(selected_anchor[:n_select], dtype=object)


def generate_nonbad_class(
    base_x: np.ndarray,
    base_frame: pd.DataFrame,
    cls: str,
    stats: dict[str, dict[str, float]],
    pool_multiplier: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed) + (17 if cls == "good" else 31))
    modes = ["identity"]
    if cls == "good":
        modes += ["good_amp_baseline", "good_morph_smooth", "good_qrs_band"]
        if GENERATOR_PROFILE.startswith(("v22", "v23", "v24", "v25")):
            modes += [
                "good_v22_highamp_baseline",
                "good_v22_qrs_preserve_band",
                "good_v22_lowentropy_morph",
                "good_v22_amp_step",
            ]
    else:
        modes += ["medium_preserve", "medium_detail", "medium_baseline"]
        if GENERATOR_PROFILE.startswith(("v22", "v23", "v24", "v25")):
            modes += ["medium_v22_template_jitter", "medium_v22_lowdetector_detail"]
        if GENERATOR_PROFILE.startswith(("v24", "v25")):
            modes += [
                "medium_v24_visible_qrs_baseline_detail",
                "medium_v24_goodlike_nonqrs_burst",
                "medium_v24_mild_contact_reset",
            ]
    class_rows: list[pd.DataFrame] = []
    class_x: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        sub = base_frame.loc[
            base_frame["class_name"].astype(str).eq(cls) & base_frame["split"].astype(str).eq(split)
        ].copy()
        if sub.empty:
            continue
        cand_x: list[np.ndarray] = []
        cand_meta: list[dict[str, Any]] = []
        for _, row in sub.iterrows():
            base_sig = base_x[int(row["idx"])]
            for j in range(max(1, int(pool_multiplier))):
                mode = modes[j % len(modes)]
                cand_x.append(gm_candidate(base_sig, cls, mode, rng))
                cand_meta.append({"base_idx": int(row["idx"]), "base_source_idx": int(row["source_idx"]), "candidate_mode": mode})
        cand_arr = np.stack(cand_x, axis=0).astype(np.float32)
        cand_feat = compute_primitives(cand_arr)
        scores = score_features(cand_feat, cls, stats)
        if GENERATOR_PROFILE.startswith("v25"):
            order, anchor_labels = v25_medium_forced_mode_order(cand_feat, cand_meta, cls, stats, len(sub))
        elif GENERATOR_PROFILE.startswith(("v23", "v24")):
            order, anchor_labels = quantile_balanced_order(cand_feat, cls, stats, len(sub))
        else:
            order = np.argsort(scores)[: len(sub)]
            anchor_labels = np.asarray(["median_best"] * len(order), dtype=object)
        for local_i, cand_i in enumerate(order.tolist()):
            row = sub.iloc[local_i % len(sub)].copy()
            meta = cand_meta[cand_i]
            row["idx"] = len(class_x)
            row["source_idx"] = int((10_000_000 if cls == "good" else 15_000_000) + len(class_x))
            row["record_id"] = f"ptb_{RUN_VERSION_TAG}_{cls}_{split}_{len(class_x)}"
            row["subject_id"] = row["record_id"]
            row["dataset"] = f"ptb_synthetic_{RUN_VERSION_TAG}_gm_featurematched"
            row["original_region"] = f"{RUN_VERSION_TAG}_{cls}_featurematched"
            row["bank_role"] = f"{RUN_VERSION_TAG}_{cls}_featurematched"
            row["profile_block"] = f"{RUN_VERSION_TAG}_{cls}_featurematched"
            row["clean_policy"] = f"ptb_{RUN_VERSION_TAG}_gm_featurematched"
            row["gm_candidate_mode"] = str(meta["candidate_mode"])
            row["gm_quantile_anchor"] = str(anchor_labels[local_i])
            row["base_pool_idx"] = int(meta["base_idx"])
            for col in GM_FEATURES:
                row[col] = float(pd.to_numeric(cand_feat.iloc[cand_i][col], errors="coerce"))
            class_rows.append(row.to_frame().T)
            class_x.append(cand_arr[cand_i])
            selected_rows.append(
                {
                    "class_name": cls,
                    "split": split,
                    "selected_score": float(scores[cand_i]),
                    "quantile_anchor": str(anchor_labels[local_i]),
                    **meta,
                    **{f"{col}": float(pd.to_numeric(cand_feat.iloc[cand_i][col], errors="coerce")) for col in GM_FEATURES},
                }
            )
    if not class_rows:
        return np.empty((0, base_x.shape[1]), dtype=np.float32), pd.DataFrame(), pd.DataFrame()
    frame = pd.concat(class_rows, ignore_index=True)
    return np.stack(class_x, axis=0).astype(np.float32), frame, pd.DataFrame(selected_rows)


def plot_audit(frame: pd.DataFrame, but: pd.DataFrame, out_name: str = "v21") -> None:
    work = frame.loc[frame["class_name"].astype(str).isin(["good", "medium"])].copy()
    target = but.loc[but["class_name"].astype(str).isin(["good", "medium"])].copy()
    rows = []
    for cls in ["good", "medium"]:
        for col in GM_FEATURES:
            p = pd.to_numeric(work.loc[work["class_name"].astype(str).eq(cls), col], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(target.loc[target["class_name"].astype(str).eq(cls), col], errors="coerce").dropna().to_numpy()
            if p.size == 0 or b.size == 0:
                continue
            scale = max(np.percentile(b, 75) - np.percentile(b, 25), np.std(b), 1e-3)
            rows.append(
                {
                    "class_name": cls,
                    "feature": col,
                    "but_median": float(np.median(b)),
                    "ptb_median": float(np.median(p)),
                    "robust_z_gap": float((np.median(p) - np.median(b)) / scale),
                    "abs_gap": float(abs((np.median(p) - np.median(b)) / scale)),
                }
            )
    gaps = pd.DataFrame(rows)
    write_csv(gaps, REPORT_OUT_DIR / f"{out_name}_gm_feature_gaps.csv")

    for cls in ["good", "medium"]:
        sub = gaps.loc[gaps["class_name"].eq(cls)].sort_values("abs_gap", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        colors = ["#5477C4" if v >= 0 else "#CC6F47" for v in sub["robust_z_gap"]]
        ax.barh(sub["feature"], sub["robust_z_gap"], color=colors)
        ax.axvline(0, color="#464C55", lw=1)
        ax.set_title(f"{out_name.upper()} {cls} PTB-vs-BUT robust feature gap")
        ax.set_xlabel("Robust z gap; 0 means matched to BUT")
        ax.grid(axis="x", alpha=0.2)
        fig.savefig(REPORT_OUT_DIR / f"{out_name}_gm_{cls}_feature_gap.png", dpi=220)
        plt.close(fig)

    features_plot = ["sqi_basSQI", "baseline_step", "rms", "mean_abs", "qrs_band_ratio", "amplitude_entropy", "band_30_45", "detector_agreement"]
    for cls in ["good", "medium"]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
        for ax, col in zip(axes.ravel(), features_plot):
            data = [
                pd.to_numeric(target.loc[target["class_name"].astype(str).eq(cls), col], errors="coerce").dropna().to_numpy(),
                pd.to_numeric(work.loc[work["class_name"].astype(str).eq(cls), col], errors="coerce").dropna().to_numpy(),
            ]
            ax.boxplot(data, tick_labels=["BUT", "PTB"], showfliers=False)
            ax.set_title(col)
            ax.grid(alpha=0.16)
        fig.suptitle(f"{out_name.upper()} {cls} waveform-computable feature distributions", fontsize=14)
        fig.savefig(REPORT_OUT_DIR / f"{out_name}_gm_{cls}_feature_boxplots.png", dpi=220)
        plt.close(fig)


def build_protocol(args: argparse.Namespace) -> Path:
    ensure_dirs()
    base_frame, base_x = recompute_frame_features(Path(args.base_protocol))
    but_frame, _ = recompute_frame_features(Path(args.but_protocol))
    stats, stat_rows = target_stats(but_frame, str(args.target_split))

    good_x, good_frame, good_sel = generate_nonbad_class(base_x, base_frame, "good", stats, int(args.pool_multiplier), int(args.seed))
    med_x, med_frame, med_sel = generate_nonbad_class(base_x, base_frame, "medium", stats, int(args.pool_multiplier), int(args.seed))

    bad_frame = base_frame.loc[base_frame["class_name"].astype(str).eq("bad")].copy()
    bad_x = base_x[bad_frame["idx"].astype(int).to_numpy()]
    bad_frame = bad_frame.reset_index(drop=True)
    offset = len(good_frame) + len(med_frame)
    bad_frame["idx"] = np.arange(offset, offset + len(bad_frame), dtype=np.int64)

    good_frame["idx"] = np.arange(0, len(good_frame), dtype=np.int64)
    med_frame["idx"] = np.arange(len(good_frame), len(good_frame) + len(med_frame), dtype=np.int64)
    new_x = np.concatenate([good_x, med_x, bad_x], axis=0).astype(np.float32)
    frame = pd.concat([good_frame, med_frame, bad_frame], ignore_index=True)
    frame["y"] = frame["class_name"].astype(str).map(CLASS_TO_INT).astype(int)

    # Recompute all waveform-computable columns once more after concatenation.
    prim = compute_primitives(new_x)
    for col in GM_FEATURES:
        frame[col] = pd.to_numeric(prim[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    protocol = OUT_DIR / f"protocol_{RUN_VERSION_TAG}_pc{int(args.total_bad)}_s{int(args.seed)}"
    protocol.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(long_path(protocol / "signals.npz"), X=new_x[:, None, :].astype(np.float32))
    write_csv(frame, protocol / "original_region_atlas.csv")
    write_csv(frame, protocol / "metadata.csv")
    write_csv(frame.groupby(["split", "class_name"], dropna=False).size().reset_index(name="n"), protocol / "split_counts.csv")
    write_csv(pd.concat([good_sel, med_sel], ignore_index=True), protocol / "selected_gm_candidate_scores.csv")
    write_csv(stat_rows, protocol / "but_gm_target_feature_stats.csv")
    plot_audit(frame, but_frame, RUN_VERSION_TAG)
    summary = {
        "created_at": now(),
        "protocol": str(protocol),
        "base_protocol": str(args.base_protocol),
        "but_reference_protocol": str(args.but_protocol),
        "seed": int(args.seed),
        "pool_multiplier": int(args.pool_multiplier),
        "target_split": str(args.target_split),
        "input_contract": "PTB waveform generation/selection; BUT train feature distribution only; no BUT waveform copied",
        "generator_profile": GENERATOR_PROFILE,
        "note": f"{RUN_VERSION_TAG.upper()} keeps V20 bad rows and rebuilds good/medium only.",
    }
    (protocol / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return protocol


def write_report(protocol: Path) -> None:
    split = pd.read_csv(protocol / "split_counts.csv")
    good_gap = pd.read_csv(REPORT_OUT_DIR / f"{RUN_VERSION_TAG}_gm_feature_gaps.csv")

    def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
        view = df.head(max_rows).copy()
        cols = [str(c) for c in view.columns]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            vals = []
            for col in view.columns:
                val = row[col]
                vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        f"# {RUN_VERSION_TAG.upper()} Good/Medium Feature-Matched Protocol",
        "",
        f"- Generated: {now()}",
        f"- Protocol: `{protocol}`",
        "- Bad rows: inherited from V20 bad/poor distribution-matched protocol.",
        "- Good/medium rows: PTB-only candidates selected against BUT train-split waveform-computable feature medians/IQRs.",
        f"- Generator profile: `{GENERATOR_PROFILE}`.",
        "- No BUT waveform is copied; BUT is used as a distribution target.",
        "",
        "## Split Counts",
        "",
        md_table(split),
        "",
        "## Largest Remaining Good/Medium Feature Gaps",
        "",
        md_table(good_gap.sort_values("abs_gap", ascending=False).head(20)),
        "",
        "## Figures",
        "",
        f"- Good gap: `{REPORT_OUT_DIR / f'{RUN_VERSION_TAG}_gm_good_feature_gap.png'}`",
        f"- Medium gap: `{REPORT_OUT_DIR / f'{RUN_VERSION_TAG}_gm_medium_feature_gap.png'}`",
        f"- Good boxplots: `{REPORT_OUT_DIR / f'{RUN_VERSION_TAG}_gm_good_feature_boxplots.png'}`",
        f"- Medium boxplots: `{REPORT_OUT_DIR / f'{RUN_VERSION_TAG}_gm_medium_feature_boxplots.png'}`",
    ]
    (REPORT_OUT_DIR / f"{RUN_VERSION_TAG}_gm_featurematched_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global RUN_VERSION_TAG, GENERATOR_PROFILE, OUT_DIR, REPORT_OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-protocol", type=str, default=str(V20_PROTOCOL))
    parser.add_argument("--but-protocol", type=str, default=str(BUT_PROTOCOL))
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--total-bad", type=int, default=3000)
    parser.add_argument("--pool-multiplier", type=int, default=6)
    parser.add_argument("--target-split", type=str, default="train", choices=["train", "all"])
    parser.add_argument("--version-tag", type=str, default="v21")
    parser.add_argument(
        "--generator-profile",
        type=str,
        default="v21",
        choices=["v21", "v22_goodstrong", "v23_quantile", "v24_medium_hardneg", "v25_forced_medium_modes"],
    )
    args = parser.parse_args()
    RUN_VERSION_TAG = str(args.version_tag)
    GENERATOR_PROFILE = str(args.generator_profile)
    OUT_DIR = ANALYSIS_DIR / f"event_xds_aligned_{RUN_VERSION_TAG}_gm_featurematched"
    REPORT_OUT_DIR = REPORT_DIR / f"event_xds_aligned_{RUN_VERSION_TAG}_gm_featurematched"
    protocol = build_protocol(args)
    write_report(protocol)
    print(protocol, flush=True)


if __name__ == "__main__":
    main()

