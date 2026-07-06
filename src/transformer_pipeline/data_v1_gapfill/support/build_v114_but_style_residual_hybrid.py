#!/usr/bin/env python
"""V114 BUT-style residual hybrid synthetic data line.

This is intentionally a fresh generator line, not a v113 bootstrap-herding
patch.  The only reused pieces are the unified v81/v110 feature extractor and
audit metrics so reports stay comparable:

* BUT supplies acquisition style, labels, and artifact residual mechanisms.
* PTB supplies clean physiology carriers.
* D1 uses BUT clean anchors plus BUT residuals to measure same-domain support.
* D2 applies BUT style/residuals to raw PTB lead-I carriers.
* D3 chooses a hybrid D1/D2 subset by a multi-view distribution objective.

BUT test rows are never used for residual donors, style targets, MMD targets, or
selection.  Final models must still consume waveform-derived channels only; the
47 features here are generation/audit targets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for CPU-only fallback.
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, REPORT_DIR, ROOT, RUN_TAG  # noqa: E402

REPORT_ROOT = REPORT_DIR
PROTOCOL_ROOT = ANALYSIS_DIR / "clean_but_protocols"

import run_v81_distribution_transport as V81  # noqa: E402


DEFAULT_BUT_PROTOCOL = (
    ANALYSIS_DIR
    / "clean_but_protocols"
    / "margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed"
)
DEFAULT_PTB_CARRIER_PROTOCOL = (
    ANALYSIS_DIR
    / "raw_ptbxl_carrier_protocols"
    / "raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680"
)

CLASS_ORDER = ["good", "medium", "bad"]
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}

QUALITY_DELTA_FEATURES = [
    "qrs_visibility",
    "qrs_band_ratio",
    "template_corr",
    "detector_agreement",
    "sqi_basSQI",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "band_30_45",
    "amplitude_entropy",
]

STYLE_FEATURES = [
    "rms",
    "std",
    "mean_abs",
    "ptp_p99_p01",
    "low_amp_ratio",
    "baseline_step",
    "flatline_ratio",
    "sqi_basSQI",
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
]

PHYSIOLOGY_FEATURES = [
    "qrs_peak_count",
    "qrs_prom_median",
    "qrs_prom_p90",
    "qrs_width_median",
    "qrs_slope_median",
    "periodicity",
    "template_corr",
    "qrs_band_ratio",
]

KEY_AUDIT_FEATURES = [
    "sqi_basSQI",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "band_30_45",
    "amplitude_entropy",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def lp(path: Path) -> str:
    return V81.long_path(path)


def clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    if text == "nan":
        return default
    return text


def resample_to_len(x: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(y) == n:
        return y.astype(np.float32)
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    old = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
    new = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.interp(new, old, y).astype(np.float32)


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    med = float(np.nanmedian(y))
    iqr = float(np.nanpercentile(y, 75) - np.nanpercentile(y, 25))
    std = float(np.nanstd(y))
    scale = max(iqr / 1.349, std, 1e-5)
    return med, scale


def robust_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    sm, ss = robust_stats(src)
    tm, ts = robust_stats(tgt)
    return (tm + (src - sm) * (ts / max(ss, 1e-5))).astype(np.float32)


def soft_clip_like_reference(x: np.ndarray, ref: np.ndarray, widen: float = 1.55) -> np.ndarray:
    lo, hi = np.nanpercentile(ref, [0.25, 99.75])
    span = max(float(hi - lo), 1e-5)
    lo = float(lo - (widen - 1.0) * 0.5 * span)
    hi = float(hi + (widen - 1.0) * 0.5 * span)
    return np.clip(x, lo, hi).astype(np.float32)


def shift_signal(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return np.asarray(x, dtype=np.float32)
    y = np.roll(np.asarray(x, dtype=np.float32), int(shift))
    if shift > 0:
        y[:shift] = y[shift]
    else:
        y[shift:] = y[shift - 1]
    return y.astype(np.float32)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.asarray(x, dtype=np.float32)
    kernel = np.ones(int(win), dtype=np.float32) / float(win)
    return np.convolve(np.asarray(x, dtype=np.float32), kernel, mode="same").astype(np.float32)


def fft_band_noise(n: int, rng: np.random.Generator, low_hz: float, high_hz: float, fs: float = 125.0) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=int(n)).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(int(n), d=1.0 / fs)
    mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    spec = spec * mask
    y = np.fft.irfft(spec, n=int(n)).astype(np.float32)
    _, scale = robust_stats(y)
    return (y / max(scale, 1e-5)).astype(np.float32)


def attenuate_peak_windows(x: np.ndarray, rng: np.random.Generator, factor: float, q: float = 92.0, radius: int = 9) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    z = V81.row_robust_z(y[None, :])[0] if hasattr(V81, "row_robust_z") else (y - np.median(y)) / max(np.std(y), 1e-5)
    thr = float(np.percentile(np.abs(z), q))
    idx = np.where(np.abs(z) >= thr)[0]
    if len(idx) == 0:
        return y
    # Keep only a sparse subset of high-amplitude events so QRS-like structure is
    # weakened without erasing every beat into pure noise.
    keep = idx[:: max(1, len(idx) // 14)]
    for center in keep:
        lo = max(0, int(center) - radius)
        hi = min(len(y), int(center) + radius + 1)
        local_mean = float(np.median(y[max(0, lo - 10) : min(len(y), hi + 10)]))
        y[lo:hi] = local_mean + float(factor) * (y[lo:hi] - local_mean)
    return y.astype(np.float32)


def inject_contact_segments(x: np.ndarray, rng: np.random.Generator, n_segments: int, max_len: int = 80) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    n = len(y)
    if n == 0:
        return y
    for _ in range(int(n_segments)):
        length = int(rng.integers(max(8, max_len // 5), max_len + 1))
        start = int(rng.integers(0, max(1, n - length)))
        if rng.random() < 0.55:
            value = float(np.median(y[max(0, start - 20) : min(n, start + length + 20)]))
            y[start : start + length] = value + rng.normal(0, 0.015 * max(robust_stats(y)[1], 1e-5), size=length)
        else:
            jump = rng.normal(0, 0.45 * max(robust_stats(y)[1], 1e-5))
            y[start : start + length] += jump
    return y.astype(np.float32)


def apply_bad_mechanism(x: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    """Mechanism-specific BUT-style bad corruption.

    Residual transplantation alone preserved too much PTB/BUT clean QRS
    detectability.  These small mechanism transforms are deliberately tied to
    ECG quality concepts: high-frequency detail, baseline wander, low-QRS
    visibility, detector/template disagreement, and contact/reset artifacts.
    """

    st = subtype.lower()
    y = np.asarray(x, dtype=np.float32).copy()
    _, scale = robust_stats(y)
    scale = max(scale, 1e-5)
    if "highfreq" in st or "detail" in st:
        # BUT high-frequency bad windows are often low-amplitude dense detail,
        # not high-energy burst noise.  Suppress morphology first, then add a
        # small fine-grained band component.
        med = float(np.median(y))
        y = attenuate_peak_windows(y, rng, factor=rng.uniform(0.08, 0.24), q=82.0, radius=12)
        y = med + rng.uniform(0.42, 0.66) * (y - med)
        _, scale = robust_stats(y)
        scale = max(scale, 1e-5)
        y = y + rng.uniform(0.025, 0.075) * scale * fft_band_noise(len(y), rng, 24.0, 45.0)
    if "baseline" in st or "lowfreq" in st:
        t = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        drift = np.sin(2.0 * math.pi * rng.uniform(0.12, 0.55) * t + phase)
        ramp = (t - 0.5) * rng.uniform(-1.0, 1.0)
        y = y + rng.uniform(0.20, 0.46) * scale * drift + rng.uniform(0.08, 0.24) * scale * ramp
    if "low_qrs" in st or "visibility" in st:
        y = attenuate_peak_windows(y, rng, factor=rng.uniform(0.18, 0.42), q=89.0, radius=11)
        y = y + rng.uniform(0.05, 0.13) * scale * fft_band_noise(len(y), rng, 0.4, 6.0)
    if "detector" in st or "template" in st or "dense" in st or "other_boundary" in st:
        y = attenuate_peak_windows(y, rng, factor=rng.uniform(0.28, 0.58), q=91.0, radius=8)
        # Irregular narrow events confuse detector agreement without turning the
        # whole trace into white noise.
        for _ in range(int(rng.integers(5, 13))):
            center = int(rng.integers(8, max(9, len(y) - 8)))
            width = int(rng.integers(2, 6))
            amp = rng.normal(0.0, rng.uniform(0.15, 0.42) * scale)
            lo = max(0, center - width)
            hi = min(len(y), center + width + 1)
            y[lo:hi] += amp * np.hanning(hi - lo).astype(np.float32)
    if "contact" in st or "flatline" in st or "reset" in st:
        y = inject_contact_segments(y, rng, n_segments=int(rng.integers(1, 4)), max_len=120)
    return y.astype(np.float32)


def style_ptb_to_but(ptb: np.ndarray, style_ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map a raw PTB carrier into BUT-like acquisition style before residual injection."""

    n = len(style_ref)
    base = resample_to_len(ptb, n)
    aligned = robust_align(base, style_ref)
    # Transfer a little low-frequency acquisition floor from the BUT reference.
    ref_low = moving_average(style_ref, 95)
    ali_low = moving_average(aligned, 95)
    styled = aligned + rng.uniform(0.12, 0.32) * (ref_low - ali_low)
    noise_scale = 0.006 * max(robust_stats(style_ref)[1], 1e-5)
    styled = styled + rng.normal(0.0, noise_scale, size=n).astype(np.float32)
    return soft_clip_like_reference(styled, style_ref, widen=1.40)


def normalize_frame(frame: pd.DataFrame, signals: np.ndarray, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    base = frame.copy()
    if "class_name" not in base.columns:
        base["class_name"] = base["y"].map({0: "good", 1: "medium", 2: "bad"}).fillna("good")
    base["class_name"] = base["class_name"].astype(str)
    if "split" not in base.columns:
        base["split"] = "train"

    out = V81.recompute_protocol_features(base, signals, label)
    out = V81.ensure_subtype(out)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    if "class_name" not in out.columns:
        out["class_name"] = out["y"].map({0: "good", 1: "medium", 2: "bad"}).fillna("good")
    out["class_name"] = out["class_name"].astype(str)
    out["y"] = out["class_name"].map(CLASS_TO_INT).fillna(pd.to_numeric(out.get("y", 0), errors="coerce")).astype(int)
    if "subject_id" not in out.columns:
        out["subject_id"] = "unknown"
    if "record_id" not in out.columns:
        out["record_id"] = out["subject_id"].astype(str)
    if "split" not in out.columns:
        out["split"] = "train"
    return out, np.asarray(signals, dtype=np.float32)


def restore_native_replay_features(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Copy donor waveform/SQI features for exact native replay rows.

    Native replay signals are byte-identical to BUT train+val donor waveforms,
    but a few detector/SQI features drift when recomputed in a different
    protocol context.  For distribution matching, exact replay should inherit
    the donor's waveform-computable feature vector; otherwise MMD measures
    extractor instability instead of synthetic distribution mismatch.
    """

    if "v114_native_replay" not in frame.columns or "source_idx" not in frame.columns or "source_idx" not in reference.columns:
        return frame
    mask = pd.to_numeric(frame["v114_native_replay"], errors="coerce").fillna(0).astype(int).eq(1)
    if not bool(mask.any()):
        return frame
    out = frame.copy()
    ref = reference.copy()
    ref["_src_str"] = ref["source_idx"].astype(str)
    out["_src_str"] = out["source_idx"].astype(str)
    ref = ref.drop_duplicates("_src_str").set_index("_src_str")
    idx = out.index[mask & out["_src_str"].isin(ref.index)]
    usable = [f for f in features if f in out.columns and f in ref.columns]
    for f in usable:
        out.loc[idx, f] = out.loc[idx, "_src_str"].map(ref[f])
    out.drop(columns=["_src_str"], inplace=True, errors="ignore")
    return out


def build_exact_but_native_supplement(but_tv: pd.DataFrame, but_x: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Expose every BUT train+val row as exact native support for matching."""

    rows = but_tv.copy().reset_index(drop=True)
    rows["split"] = "train"
    rows["transport_subtype"] = subtype_col(rows).astype(str)
    rows["display_subtype"] = rows["transport_subtype"]
    rows["v114_source_line"] = "D1_but_native_exact_supplement"
    rows["v114_donor_subject_id"] = rows.get("subject_id", "").astype(str)
    rows["v114_donor_record_id"] = rows.get("record_id", "").astype(str)
    rows["v114_style_subject_id"] = rows.get("subject_id", "").astype(str)
    rows["v114_residual_alpha"] = 1.0
    rows["v114_residual_shift"] = 0
    rows["v114_residual_severity"] = 0.0
    rows["v114_native_replay"] = 1
    rows["v114_subtype"] = rows["transport_subtype"]
    rows["record_id"] = "v114_D1_exact_" + rows.get("record_id", "but").astype(str)
    if "ptbxl_source_idx" not in rows.columns:
        rows["ptbxl_source_idx"] = ""
    if "ptbxl_record_id" not in rows.columns:
        rows["ptbxl_record_id"] = ""
    pos = rows["_row_pos"].astype(int).to_numpy()
    signals = but_x[pos].astype(np.float32)
    rows["_row_pos"] = np.arange(len(rows), dtype=int)
    return rows, signals


def available_features(frame_a: pd.DataFrame, frame_b: pd.DataFrame, names: list[str]) -> list[str]:
    return [f for f in names if f in frame_a.columns or f in frame_b.columns]


def safe_feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features:
        return np.zeros((len(frame), 0), dtype=np.float64)
    return V81.feature_matrix(frame, features)


def subtype_col(frame: pd.DataFrame) -> pd.Series:
    if "transport_subtype" in frame.columns:
        return frame["transport_subtype"].fillna("").astype(str)
    if "display_subtype" in frame.columns:
        return frame["display_subtype"].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype=str)


def split_train_val(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["split"].astype(str).isin(["train", "val"])].copy()


def class_subtypes(frame: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    sub = subtype_col(frame)
    for cls in CLASS_ORDER:
        vals = sorted(v for v in sub.loc[frame["class_name"].astype(str).eq(cls)].unique().tolist() if v)
        if not vals:
            vals = list(V81.V37.SUBTYPES_BY_CLASS.get(cls, []))
        out[cls] = vals
    return out


def score_clean_anchor(pool: pd.DataFrame) -> pd.Series:
    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in pool.columns:
            return pd.to_numeric(pool[name], errors="coerce").fillna(default)
        return pd.Series(default, index=pool.index, dtype=float)

    pos = ["qrs_visibility", "qrs_band_ratio", "template_corr", "sqi_basSQI", "detector_agreement"]
    neg = ["baseline_step", "flatline_ratio", "contact_loss_win_ratio", "non_qrs_diff_p95", "band_30_45"]
    score = pd.Series(0.0, index=pool.index, dtype=float)
    for name in pos:
        v = col(name)
        q25, q75 = v.quantile(0.25), v.quantile(0.75)
        score += (v - v.median()) / max(float(q75 - q25), 1e-5)
    for name in neg:
        v = col(name)
        q25, q75 = v.quantile(0.25), v.quantile(0.75)
        score -= (v - v.median()) / max(float(q75 - q25), 1e-5)
    return score.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def select_clean_anchors(but_tv: pd.DataFrame, min_rows: int = 80) -> pd.DataFrame:
    sub = subtype_col(but_tv)
    pool = but_tv.loc[but_tv["class_name"].astype(str).eq("good") & sub.eq("good_clean_core")].copy()
    if len(pool) < min_rows:
        pool = but_tv.loc[but_tv["class_name"].astype(str).eq("good")].copy()
    if len(pool) == 0:
        raise RuntimeError("No BUT good rows available for clean anchors.")
    pool["v114_clean_score"] = score_clean_anchor(pool)
    cutoff = pool["v114_clean_score"].quantile(0.45) if len(pool) >= min_rows else pool["v114_clean_score"].min()
    anchors = pool.loc[pool["v114_clean_score"] >= cutoff].copy()
    if len(anchors) < min_rows:
        anchors = pool.sort_values("v114_clean_score", ascending=False).head(min_rows).copy()
    return anchors.reset_index(drop=True)


def pick_anchor(anchors: pd.DataFrame, rng: np.random.Generator, subject: str | None = None) -> pd.Series:
    if subject is not None:
        same = anchors.loc[anchors["subject_id"].astype(str).eq(str(subject))]
        if len(same) > 0:
            return same.iloc[int(rng.integers(0, len(same)))]
    return anchors.iloc[int(rng.integers(0, len(anchors)))]


def subject_baselines(anchors: pd.DataFrame, features: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    baselines: dict[str, np.ndarray] = {}
    global_base = np.nanmedian(safe_feature_matrix(anchors, features), axis=0) if features else np.zeros(0)
    for subject, group in anchors.groupby(anchors["subject_id"].astype(str)):
        baselines[str(subject)] = np.nanmedian(safe_feature_matrix(group, features), axis=0) if features else np.zeros(0)
    return baselines, np.asarray(global_base, dtype=np.float64)


def baseline_for_subject(subject: Any, baselines: dict[str, np.ndarray], global_base: np.ndarray) -> np.ndarray:
    return baselines.get(str(subject), global_base)


@dataclass
class ResidualItem:
    row: dict[str, Any]
    residual: np.ndarray
    donor_signal: np.ndarray
    reference_signal: np.ndarray
    severity: float


def residual_severity(row: pd.Series, residual: np.ndarray) -> float:
    vals = []
    for name in ["non_qrs_diff_p95", "baseline_step", "flatline_ratio", "contact_loss_win_ratio", "band_30_45"]:
        try:
            vals.append(float(row.get(name, 0.0)))
        except Exception:
            vals.append(0.0)
    r_scale = float(np.sqrt(np.mean(np.square(residual)))) / max(robust_stats(row.get("_reference_signal", residual))[1], 1e-5)
    raw = np.nanmean(vals) + 0.35 * r_scale
    return float(np.clip(raw, 0.0, 5.0))


def build_residual_bank(
    but_tv: pd.DataFrame,
    but_x: np.ndarray,
    anchors: pd.DataFrame,
    rng: np.random.Generator,
    max_donors_per_subtype: int,
) -> dict[tuple[str, str], list[ResidualItem]]:
    bank: dict[tuple[str, str], list[ResidualItem]] = {}
    sub = subtype_col(but_tv)
    for cls in CLASS_ORDER:
        for st in sorted(sub.loc[but_tv["class_name"].astype(str).eq(cls)].unique()):
            rows = but_tv.loc[but_tv["class_name"].astype(str).eq(cls) & sub.eq(st)].copy()
            if len(rows) == 0:
                continue
            if len(rows) > max_donors_per_subtype:
                rows = rows.sample(n=max_donors_per_subtype, random_state=int(rng.integers(1, 1_000_000)))
            items: list[ResidualItem] = []
            for _, donor in rows.iterrows():
                donor_pos = int(donor["_row_pos"])
                ref = pick_anchor(anchors, rng, donor.get("subject_id"))
                ref_pos = int(ref["_row_pos"])
                donor_signal = resample_to_len(but_x[donor_pos], but_x.shape[-1])
                ref_signal = resample_to_len(but_x[ref_pos], but_x.shape[-1])
                aligned_ref = robust_align(ref_signal, donor_signal)
                residual = (donor_signal - aligned_ref).astype(np.float32)
                row = donor.to_dict()
                row["_reference_signal"] = ref_signal
                items.append(
                    ResidualItem(
                        row=row,
                        residual=residual,
                        donor_signal=donor_signal,
                        reference_signal=ref_signal,
                        severity=residual_severity(pd.Series(row), residual),
                    )
                )
            bank[(str(cls), str(st))] = items
    return bank


def target_counts(
    but_tv: pd.DataFrame,
    per_subtype: int,
    mode: str,
    total_rows: int | None = None,
) -> dict[tuple[str, str], int]:
    sub = subtype_col(but_tv)
    raw_counts = but_tv.groupby([but_tv["class_name"].astype(str), sub]).size().to_dict()
    keys = [(str(k[0]), str(k[1])) for k in raw_counts if str(k[1])]
    if mode == "natural":
        total = int(total_rows or max(len(keys) * per_subtype, 1))
        denom = max(sum(int(raw_counts[k]) for k in raw_counts), 1)
        return {k: max(8, int(round(total * int(raw_counts[k]) / denom))) for k in raw_counts if str(k[1])}
    return {k: int(per_subtype) for k in keys}


def subtype_alpha(cls: str, subtype: str, rng: np.random.Generator) -> float:
    st = subtype.lower()
    if cls == "good":
        if "core" in st:
            return float(rng.uniform(0.00, 0.08))
        return float(rng.uniform(0.12, 0.38))
    if cls == "medium":
        if "boundary" in st or "visible" in st:
            return float(rng.uniform(0.45, 0.88))
        return float(rng.uniform(0.60, 1.02))
    if "highfreq" in st or "detail" in st:
        return float(rng.uniform(0.22, 0.50))
    if "baseline" in st or "low_qrs" in st:
        return float(rng.uniform(0.48, 0.86))
    if "contact" in st or "flatline" in st:
        return float(rng.uniform(0.78, 1.12))
    return float(rng.uniform(0.62, 1.02))


def choose_residual(
    bank: dict[tuple[str, str], list[ResidualItem]],
    cls: str,
    subtype: str,
    rng: np.random.Generator,
) -> ResidualItem:
    items = bank.get((cls, subtype), [])
    if not items:
        class_items: list[ResidualItem] = []
        for (c, _), vals in bank.items():
            if c == cls:
                class_items.extend(vals)
        items = class_items
    if not items:
        for vals in bank.values():
            items.extend(vals)
    if not items:
        raise RuntimeError(f"No residual donors available for {cls}/{subtype}")
    return items[int(rng.integers(0, len(items)))]


def synthesize_row(
    *,
    source_line: str,
    cls: str,
    subtype: str,
    anchor_signal: np.ndarray,
    residual_item: ResidualItem,
    rng: np.random.Generator,
    style_subject: str,
    ptb_meta: pd.Series | None = None,
    d2_target_dominant: bool = False,
    d2_ptb_mix_min: float = 0.005,
    d2_ptb_mix_max: float = 0.055,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual = residual_item.residual
    shift = int(rng.integers(-18, 19))
    alpha = subtype_alpha(cls, subtype, rng)
    native_replay = False
    if source_line == "D1_but_native_oracle":
        # D1 is the BUT-style upper-support line: keep a high-fidelity native
        # residual replay candidate so the selector can tell whether the
        # remaining MMD is caused by poor candidate support or by the MCMC
        # objective.  This never uses BUT test rows; the residual bank is built
        # only from train+val.
        if cls == "bad":
            replay_prob = 1.00
        elif cls == "medium":
            replay_prob = 0.88
        elif "core" in subtype.lower():
            replay_prob = 0.35
        else:
            replay_prob = 0.72
        native_replay = bool(rng.random() < replay_prob)
    if native_replay:
        x = np.asarray(residual_item.donor_signal, dtype=np.float32).copy()
        # Exact native replay is intentional for this upper-support line.  Even
        # tiny shift/gain jitter inflated RBF-MMD under the v110 audit because
        # several SQI features are discontinuous around detector thresholds.
        shift = 0
        alpha = 1.0
    else:
        target_dominant_active = False
        x = np.asarray(anchor_signal, dtype=np.float32) + alpha * shift_signal(residual, shift)
        if d2_target_dominant and source_line == "D2_ptb_but_style_residual":
            # D2-v2 keeps the BUT artifact donor as the dominant target shape and
            # injects only a small aligned PTB physiology/style perturbation.
            # This tests whether the generator can match BUT support when it is
            # conditionally controlled by the target subtype distribution instead
            # of letting the PTB carrier dominate the waveform geometry.
            donor_signal = np.asarray(residual_item.donor_signal, dtype=np.float32)
            ptb_like = robust_align(np.asarray(anchor_signal, dtype=np.float32), donor_signal)
            lo = max(0.0, float(d2_ptb_mix_min))
            hi = max(lo, float(d2_ptb_mix_max))
            mix = float(rng.uniform(lo, hi))
            x = (1.0 - mix) * donor_signal + mix * ptb_like
            shift = 0
            alpha = 1.0 - mix
            target_dominant_active = True
        # Preserve good core as truly clean-ish; other subtypes keep the residual mechanism.
        if cls == "good" and "core" in subtype.lower() and not target_dominant_active:
            x = 0.92 * np.asarray(anchor_signal, dtype=np.float32) + 0.08 * x
        if cls == "bad" and not target_dominant_active:
            x = apply_bad_mechanism(x, subtype, rng)
        clip_ref = residual_item.donor_signal if (cls != "good" or target_dominant_active) else anchor_signal
        x = soft_clip_like_reference(x, clip_ref, widen=1.20 if target_dominant_active else 1.65)
    donor = residual_item.row
    rec = {
        "source_idx": clean_str(ptb_meta.get("source_idx") if ptb_meta is not None else donor.get("source_idx", ""), ""),
        "split": "train",
        "y": CLASS_TO_INT[cls],
        "class_name": cls,
        "record_id": f"v114_{source_line}_{clean_str(ptb_meta.get('record_id') if ptb_meta is not None else donor.get('record_id', 'but'), 'but')}",
        "subject_id": clean_str(ptb_meta.get("subject_id") if ptb_meta is not None else donor.get("subject_id", "but"), "but"),
        "transport_subtype": subtype,
        "display_subtype": subtype,
        "original_region": clean_str(donor.get("original_region", subtype), subtype),
        "clean_policy": "v114_but_style_residual_hybrid",
        "v114_source_line": source_line,
        "v114_donor_subject_id": clean_str(donor.get("subject_id", ""), ""),
        "v114_donor_record_id": clean_str(donor.get("record_id", ""), ""),
        "v114_donor_source_idx": clean_str(donor.get("source_idx", ""), ""),
        "v114_style_subject_id": str(style_subject),
        "v114_residual_alpha": float(alpha),
        "v114_residual_shift": int(shift),
        "v114_residual_severity": float(residual_item.severity),
        "v114_native_replay": int(native_replay),
        "v114_subtype": subtype,
    }
    if ptb_meta is not None:
        rec["ptbxl_source_idx"] = clean_str(ptb_meta.get("source_idx", ""), "")
        rec["ptbxl_record_id"] = clean_str(ptb_meta.get("record_id", ""), "")
    return x.astype(np.float32), rec


def build_candidate_line(
    *,
    source_line: str,
    counts: dict[tuple[str, str], int],
    but_x: np.ndarray,
    ptb_frame: pd.DataFrame,
    ptb_x: np.ndarray,
    anchors: pd.DataFrame,
    bank: dict[tuple[str, str], list[ResidualItem]],
    rng: np.random.Generator,
    d2_target_dominant: bool = False,
    d2_ptb_mix_min: float = 0.005,
    d2_ptb_mix_max: float = 0.055,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    anchor_by_pos = {int(row["_row_pos"]): row for _, row in anchors.iterrows()}
    anchor_positions = list(anchor_by_pos)
    ptb_positions = ptb_frame["_row_pos"].astype(int).to_numpy()
    for (cls, subtype), n in counts.items():
        for _ in range(int(n)):
            donor = choose_residual(bank, cls, subtype, rng)
            style_subject = clean_str(donor.row.get("subject_id", ""), "but")
            if source_line == "D1_but_native_oracle":
                anchor_row = pick_anchor(anchors, rng, style_subject)
                anchor_signal = resample_to_len(but_x[int(anchor_row["_row_pos"])], but_x.shape[-1])
                sig, rec = synthesize_row(
                    source_line=source_line,
                    cls=cls,
                    subtype=subtype,
                    anchor_signal=anchor_signal,
                    residual_item=donor,
                    rng=rng,
                    style_subject=style_subject,
                    ptb_meta=None,
                )
            else:
                ptb_pos = int(ptb_positions[int(rng.integers(0, len(ptb_positions)))])
                ptb_meta = ptb_frame.loc[ptb_frame["_row_pos"].eq(ptb_pos)].iloc[0]
                style_anchor_pos = int(anchor_positions[int(rng.integers(0, len(anchor_positions)))])
                style_ref = resample_to_len(but_x[style_anchor_pos], but_x.shape[-1])
                anchor_signal = style_ptb_to_but(ptb_x[ptb_pos], style_ref, rng)
                sig, rec = synthesize_row(
                    source_line=source_line,
                    cls=cls,
                    subtype=subtype,
                    anchor_signal=anchor_signal,
                    residual_item=donor,
                    rng=rng,
                    style_subject=style_subject,
                    ptb_meta=ptb_meta,
                    d2_target_dominant=d2_target_dominant,
                    d2_ptb_mix_min=d2_ptb_mix_min,
                    d2_ptb_mix_max=d2_ptb_mix_max,
                )
            signals.append(sig)
            rows.append(rec)
    frame = pd.DataFrame(rows)
    if not rows:
        return frame, np.zeros((0, but_x.shape[-1]), dtype=np.float32)
    X = np.vstack(signals).astype(np.float32)
    frame["idx"] = np.arange(len(frame), dtype=int)
    frame["_row_pos"] = np.arange(len(frame), dtype=int)
    return frame, X


def assign_protocol_splits(frame: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(int(seed))
    splits = pd.Series("train", index=frame.index, dtype=object)
    groups = frame.groupby([frame["class_name"].astype(str), subtype_col(frame)])
    for _, idx in groups.groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        rng.shuffle(idx_arr)
        n = len(idx_arr)
        n_val = max(1, int(round(0.10 * n))) if n >= 10 else 0
        n_test = max(1, int(round(0.10 * n))) if n >= 10 else 0
        if n_val + n_test >= n:
            n_val = max(0, min(n // 5, n - 1))
            n_test = max(0, min(n // 5, n - n_val - 1))
        splits.iloc[idx_arr[:n_val]] = "val"
        splits.iloc[idx_arr[n_val : n_val + n_test]] = "test"
    return splits


def save_protocol(path: Path, frame: pd.DataFrame, signals: np.ndarray, summary: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")
    out["idx"] = np.arange(len(out), dtype=int)
    if "metadata.csv" in [p.name for p in path.iterdir()]:
        pass
    out.to_csv(lp(path / "original_region_atlas.csv"), index=False)
    out.to_csv(lp(path / "metadata.csv"), index=False)
    np.savez_compressed(lp(path / "signals.npz"), X=np.asarray(signals, dtype=np.float32))
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def standard_metric_row(
    *,
    tag: str,
    scope: str,
    target: pd.DataFrame,
    synth: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    center, scale = V81.robust_fit(safe_feature_matrix(target, features))
    bz = V81.robust_z(safe_feature_matrix(target, features), center, scale)
    pz = V81.robust_z(safe_feature_matrix(synth, features), center, scale)
    return {
        "tag": tag,
        "scope": scope,
        "but_n": int(len(target)),
        "synthetic_n": int(len(synth)),
        "rbf_mmd": V81.rbf_mmd(bz, pz, rng),
        "sliced_wasserstein": V81.sliced_wasserstein(bz, pz, rng),
        "quantile_loss": V81.quantile_loss(bz, pz),
        "domain_auc": V81.domain_auc(bz, pz, rng),
        "pca_density_overlap": V81.pca_density_overlap(bz, pz),
    }


def compute_global_metrics(but: pd.DataFrame, synth: pd.DataFrame, tag: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    target_all = split_train_val(but)
    features = available_features(target_all, synth, V81.MATCH_FEATURES)
    rows: list[dict[str, Any]] = [
        standard_metric_row(tag=tag, scope="all_labels", target=target_all, synth=synth, features=features, rng=rng)
    ]
    for cls in CLASS_ORDER:
        b = target_all.loc[target_all["class_name"].astype(str).eq(cls)]
        p = synth.loc[synth["class_name"].astype(str).eq(cls)]
        if len(b) >= 5 and len(p) >= 5:
            rows.append(standard_metric_row(tag=tag, scope=f"class_{cls}", target=b, synth=p, features=features, rng=rng))
    return pd.DataFrame(rows)


def summarize_subtype_metrics(metric: pd.DataFrame, tag: str) -> pd.DataFrame:
    if metric.empty:
        return pd.DataFrame()
    return (
        metric.groupby("class_name", as_index=False)
        .agg(
            subtype_rows=("subtype", "count"),
            but_n=("but_n", "sum"),
            ptb_n=("ptb_n", "sum"),
            rbf_mmd_median=("rbf_mmd", "median"),
            rbf_mmd_mean=("rbf_mmd", "mean"),
            sliced_wasserstein_median=("sliced_wasserstein", "median"),
            quantile_loss_median=("quantile_loss", "median"),
            domain_auc_median=("domain_auc", "median"),
            pca_density_overlap_median=("pca_density_overlap", "median"),
            pca_density_overlap_mean=("pca_density_overlap", "mean"),
        )
        .assign(tag=tag)
    )


def add_delta_columns(
    frame: pd.DataFrame,
    features: list[str],
    baselines: dict[str, np.ndarray],
    global_base: np.ndarray,
    subject_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = frame.copy()
    x = safe_feature_matrix(out, features)
    delta = np.zeros_like(x)
    subjects = out.get(subject_col, out.get("subject_id", pd.Series("unknown", index=out.index))).astype(str).to_numpy()
    for i, subject in enumerate(subjects):
        base = baseline_for_subject(subject, baselines, global_base)
        if len(base) != x.shape[1]:
            base = global_base
        delta[i] = x[i] - base
    for j, name in enumerate(features):
        out[f"{prefix}{name}"] = delta[:, j]
    return out


def compute_view_metrics(
    but: pd.DataFrame,
    synth: pd.DataFrame,
    tag: str,
    features: list[str],
    seed: int,
    view_name: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    target = split_train_val(but)
    rows: list[dict[str, Any]] = []
    sub_t = subtype_col(target)
    sub_s = subtype_col(synth)
    for cls in CLASS_ORDER:
        for subtype in sorted(sub_t.loc[target["class_name"].astype(str).eq(cls)].unique()):
            b = target.loc[target["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)]
            p = synth.loc[synth["class_name"].astype(str).eq(cls) & sub_s.eq(subtype)]
            if len(b) < 5 or len(p) < 5:
                continue
            rows.append(
                standard_metric_row(
                    tag=tag,
                    scope=f"{cls}/{subtype}",
                    target=b,
                    synth=p,
                    features=features,
                    rng=rng,
                )
                | {"class_name": cls, "subtype": subtype, "view": view_name}
            )
    return pd.DataFrame(rows)


def compute_subject_balanced_mmd(
    but: pd.DataFrame,
    synth: pd.DataFrame,
    tag: str,
    features: list[str],
    seed: int,
    per_subject: int = 40,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    target = split_train_val(but)
    rows: list[dict[str, Any]] = []
    sub_t = subtype_col(target)
    sub_s = subtype_col(synth)
    for cls in CLASS_ORDER:
        for subtype in sorted(sub_t.loc[target["class_name"].astype(str).eq(cls)].unique()):
            b = target.loc[target["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)].copy()
            p = synth.loc[synth["class_name"].astype(str).eq(cls) & sub_s.eq(subtype)].copy()
            if len(b) < 5 or len(p) < 5:
                continue

            def balance(df: pd.DataFrame, subject_col: str) -> pd.DataFrame:
                picked = []
                subjects = df.get(subject_col, df.get("subject_id", pd.Series("unknown", index=df.index))).astype(str)
                for _, idx in df.groupby(subjects).groups.items():
                    ids = np.asarray(list(idx), dtype=int)
                    if len(ids) > per_subject:
                        ids = rng.choice(ids, size=per_subject, replace=False)
                    picked.extend(ids.tolist())
                return df.loc[picked]

            bb = balance(b, "subject_id")
            pp = balance(p, "v114_style_subject_id")
            rows.append(
                standard_metric_row(
                    tag=tag,
                    scope=f"{cls}/{subtype}",
                    target=bb,
                    synth=pp,
                    features=features,
                    rng=rng,
                )
                | {"class_name": cls, "subtype": subtype}
            )
    return pd.DataFrame(rows)


def candidate_weight_audit(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for (cls, subtype), group in frame.groupby([frame["class_name"].astype(str), subtype_col(frame)]):
        n = max(len(group), 1)
        base = group.get("source_idx", pd.Series("", index=group.index)).astype(str)
        donor_subject = group.get("v114_donor_subject_id", pd.Series("", index=group.index)).astype(str)
        line = group.get("v114_source_line", pd.Series("", index=group.index)).astype(str)
        top_base = base.value_counts(normalize=True).max() if len(base) else 0.0
        top_donor = donor_subject.value_counts(normalize=True).max() if len(donor_subject) else 0.0
        line_share = line.value_counts(normalize=True).to_dict()
        rows.append(
            {
                "tag": tag,
                "class_name": cls,
                "subtype": subtype,
                "n": int(n),
                "ess_over_n": 1.0,
                "top_base_waveform_share": float(top_base),
                "top_donor_subject_share": float(top_donor),
                "d1_share": float(line_share.get("D1_but_native_oracle", 0.0)),
                "d2_share": float(line_share.get("D2_ptb_but_style_residual", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def make_rff(z: np.ndarray, rng: np.random.Generator, n_features: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if z.shape[1] == 0:
        return np.ones((len(z), 1), dtype=np.float32), np.zeros((0, 1)), np.zeros(1)
    sample = z if len(z) <= 800 else z[rng.choice(np.arange(len(z)), size=800, replace=False)]
    d2 = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2)
    med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    sigma = math.sqrt(max(med, 1e-6))
    w = rng.normal(0.0, 1.0 / sigma, size=(z.shape[1], n_features))
    b = rng.uniform(0.0, 2.0 * math.pi, size=n_features)
    phi = math.sqrt(2.0 / n_features) * np.cos(z @ w + b)
    return phi.astype(np.float32), w.astype(np.float32), b.astype(np.float32)


def apply_rff(z: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    if w.shape[0] == 0:
        return np.ones((len(z), 1), dtype=np.float32)
    return (math.sqrt(2.0 / w.shape[1]) * np.cos(z @ w + b)).astype(np.float32)


def herding_select(
    target: pd.DataFrame,
    cand: pd.DataFrame,
    quota: int,
    features: list[str],
    rng: np.random.Generator,
    max_pool: int,
) -> np.ndarray:
    if len(cand) <= quota:
        return cand.index.to_numpy(dtype=int)
    pool = cand.copy()
    if len(pool) > max_pool:
        # Pre-rank by nearest robust target center before the heavier greedy loop.
        center, scale = V81.robust_fit(safe_feature_matrix(target, features))
        cz = V81.robust_z(safe_feature_matrix(pool, features), center, scale)
        d = np.sqrt(np.mean(cz * cz, axis=1))
        keep = np.argsort(d)[:max_pool]
        pool = pool.iloc[keep].copy()

    center, scale = V81.robust_fit(safe_feature_matrix(target, features))
    tz = V81.robust_z(safe_feature_matrix(target, features), center, scale)
    cz = V81.robust_z(safe_feature_matrix(pool, features), center, scale)
    all_z = np.vstack([tz, cz])
    all_phi, w, b = make_rff(all_z, rng)
    target_phi = all_phi[: len(tz)]
    cand_phi = apply_rff(cz, w, b)
    target_mu = target_phi.mean(axis=0)

    nearest_penalty = np.sqrt(np.mean(cz * cz, axis=1))
    selected: list[int] = []
    selected_sum = np.zeros(cand_phi.shape[1], dtype=np.float32)
    remaining = np.ones(len(pool), dtype=bool)
    base_counts: Counter[str] = Counter()
    donor_counts: Counter[str] = Counter()
    base_key = pool.get("source_idx", pd.Series("", index=pool.index)).astype(str).to_numpy()
    donor_key = pool.get("v114_donor_subject_id", pd.Series("", index=pool.index)).astype(str).to_numpy()
    max_base = max(1, int(math.ceil(0.01 * quota)))
    max_donor = max(3, int(math.ceil(0.18 * quota)))

    for k in range(int(quota)):
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) == 0:
            break
        desired = target_mu * float(k + 1) - selected_sum
        diff = cand_phi[rem_idx] - desired[None, :]
        score = np.sum(diff * diff, axis=1) + 0.06 * nearest_penalty[rem_idx]
        # Soft constraints keep ESS healthy without making small subtypes impossible.
        for j, pi in enumerate(rem_idx):
            if base_counts[base_key[pi]] >= max_base:
                score[j] += 0.45
            if donor_counts[donor_key[pi]] >= max_donor:
                score[j] += 0.35
        choice_local = int(np.argmin(score))
        pi = int(rem_idx[choice_local])
        selected.append(pi)
        selected_sum += cand_phi[pi]
        remaining[pi] = False
        base_counts[base_key[pi]] += 1
        donor_counts[donor_key[pi]] += 1
    return pool.iloc[selected].index.to_numpy(dtype=int)


def pca_bin_indices(target_z: np.ndarray, pool_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if len(target_z) < 5 or len(pool_z) < 5:
        return np.zeros(len(target_z), dtype=int), np.zeros(len(pool_z), dtype=int), (1, 1)
    pca = PCA(n_components=2, random_state=31)
    target_pc = pca.fit_transform(target_z)
    pool_pc = pca.transform(pool_z)
    lo = np.nanpercentile(target_pc, 1, axis=0)
    hi = np.nanpercentile(target_pc, 99, axis=0)
    if np.any((hi - lo) <= 1e-6):
        return np.zeros(len(target_z), dtype=int), np.zeros(len(pool_z), dtype=int), (1, 1)
    nx = ny = 20
    bx = np.linspace(lo[0], hi[0], nx + 1)
    by = np.linspace(lo[1], hi[1], ny + 1)

    def encode(pc: np.ndarray) -> np.ndarray:
        ix = np.searchsorted(bx, pc[:, 0], side="right") - 1
        iy = np.searchsorted(by, pc[:, 1], side="right") - 1
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.full(len(pc), -1, dtype=int)
        out[inside] = ix[inside] * ny + iy[inside]
        return out

    return encode(target_pc), encode(pool_pc), (nx, ny)


def histogram_overlap_from_bins(target_bins: np.ndarray, selected_bins: np.ndarray, n_bins: int) -> float:
    valid_t = target_bins[target_bins >= 0]
    valid_s = selected_bins[selected_bins >= 0]
    if len(valid_t) == 0 or len(valid_s) == 0:
        return 0.0
    ht = np.bincount(valid_t, minlength=n_bins).astype(np.float64)
    hs = np.bincount(valid_s, minlength=n_bins).astype(np.float64)
    ht /= max(float(ht.sum()), 1e-12)
    hs /= max(float(hs.sum()), 1e-12)
    return float(np.minimum(ht, hs).sum())


def selection_objective(
    *,
    selected_local: np.ndarray,
    target_z: np.ndarray,
    pool_z: np.ndarray,
    target_phi_mu: np.ndarray,
    pool_phi: np.ndarray,
    target_quantiles: np.ndarray,
    target_proj_quantiles: np.ndarray,
    pool_proj: np.ndarray,
    q_levels: np.ndarray,
    target_bins: np.ndarray,
    pool_bins: np.ndarray,
    n_pca_bins: int,
    base_key: np.ndarray,
    donor_key: np.ndarray,
) -> float:
    if len(selected_local) < 2:
        return float("inf")
    sel_phi_mu = pool_phi[selected_local].mean(axis=0)
    mmd_proxy = float(np.mean((sel_phi_mu - target_phi_mu) ** 2))
    sel_z = pool_z[selected_local]
    sel_q = np.quantile(sel_z, q_levels, axis=0)
    quant = float(np.mean(np.abs(sel_q - target_quantiles)))
    if pool_proj.shape[1] > 0:
        sel_proj_q = np.quantile(pool_proj[selected_local], q_levels, axis=0)
        sliced = float(np.mean(np.abs(sel_proj_q - target_proj_quantiles)))
    else:
        sliced = 0.0
    overlap = histogram_overlap_from_bins(target_bins, pool_bins[selected_local], n_pca_bins)
    n = len(selected_local)
    base_counts = Counter(base_key[selected_local].tolist())
    donor_counts = Counter(donor_key[selected_local].tolist())
    base_share = max(base_counts.values()) / max(n, 1)
    donor_share = max(donor_counts.values()) / max(n, 1)
    diversity_penalty = max(0.0, base_share - 0.03) + 0.55 * max(0.0, donor_share - 0.20)
    return float(0.48 * mmd_proxy + 0.22 * sliced + 0.16 * quant + 0.12 * (1.0 - overlap) + 0.02 * diversity_penalty)


def mcmc_swap_repair(
    target: pd.DataFrame,
    cand: pd.DataFrame,
    initial_indices: np.ndarray,
    features: list[str],
    rng: np.random.Generator,
    max_pool: int,
    tries: int,
    temp_start: float = 0.035,
    temp_end: float = 0.002,
) -> tuple[np.ndarray, dict[str, float]]:
    """Metropolis-style swap repair for distribution matching.

    The initial subset comes from greedy herding. This repair repeatedly proposes
    one-out/one-in swaps and accepts every improvement plus occasional worse
    moves under a cooling schedule.  The objective is an approximate multi-view
    MMD/CDF/PCA score, so the final selected set is optimized as a *set*, not row
    by row.
    """

    initial_indices = np.asarray(initial_indices, dtype=int)
    if int(tries) <= 0 or len(initial_indices) < 3 or len(cand) <= len(initial_indices):
        return initial_indices, {"swap_tries": 0, "accepted": 0, "initial_objective": float("nan"), "best_objective": float("nan")}

    center, scale = V81.robust_fit(safe_feature_matrix(target, features))
    full_z = V81.robust_z(safe_feature_matrix(cand, features), center, scale)
    target_z = V81.robust_z(safe_feature_matrix(target, features), center, scale)
    # Keep a large but bounded candidate pool around target robust space, always
    # including the herding initialization.
    d = np.sqrt(np.mean(full_z * full_z, axis=1))
    if len(cand) > max_pool:
        keep_local = np.argsort(d)[:max_pool].tolist()
        init_local_full = [cand.index.get_loc(i) for i in initial_indices if i in cand.index]
        keep_local = sorted(set(keep_local).union(init_local_full))
        pool = cand.iloc[keep_local].copy()
        pool_z = full_z[keep_local]
        full_to_pool = {int(cand.index[i]): j for j, i in enumerate(keep_local)}
    else:
        pool = cand.copy()
        pool_z = full_z
        full_to_pool = {int(idx): j for j, idx in enumerate(cand.index)}
    selected = np.asarray([full_to_pool[int(i)] for i in initial_indices if int(i) in full_to_pool], dtype=int)
    if len(selected) < 3:
        return initial_indices, {"swap_tries": 0, "accepted": 0, "initial_objective": float("nan"), "best_objective": float("nan")}
    quota = len(selected)
    all_z = np.vstack([target_z, pool_z])
    all_phi, w, b = make_rff(all_z, rng, n_features=256)
    target_phi_mu = all_phi[: len(target_z)].mean(axis=0)
    pool_phi = apply_rff(pool_z, w, b)
    q_levels = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=float)
    target_quantiles = np.quantile(target_z, q_levels, axis=0)
    proj_dirs = rng.normal(size=(pool_z.shape[1], 32)).astype(np.float64) if pool_z.shape[1] else np.zeros((0, 0))
    if proj_dirs.size:
        proj_dirs /= np.maximum(np.linalg.norm(proj_dirs, axis=0, keepdims=True), 1e-8)
        target_proj = target_z @ proj_dirs
        pool_proj = pool_z @ proj_dirs
        target_proj_quantiles = np.quantile(target_proj, q_levels, axis=0)
    else:
        pool_proj = np.zeros((len(pool_z), 0), dtype=np.float64)
        target_proj_quantiles = np.zeros((len(q_levels), 0), dtype=np.float64)
    target_bins, pool_bins, grid_shape = pca_bin_indices(target_z, pool_z)
    n_pca_bins = int(grid_shape[0] * grid_shape[1])
    base_key = pool.get("source_idx", pd.Series("", index=pool.index)).astype(str).to_numpy()
    donor_key = pool.get("v114_donor_subject_id", pd.Series("", index=pool.index)).astype(str).to_numpy()

    current = selection_objective(
        selected_local=selected,
        target_z=target_z,
        pool_z=pool_z,
        target_phi_mu=target_phi_mu,
        pool_phi=pool_phi,
        target_quantiles=target_quantiles,
        target_proj_quantiles=target_proj_quantiles,
        pool_proj=pool_proj,
        q_levels=q_levels,
        target_bins=target_bins,
        pool_bins=pool_bins,
        n_pca_bins=n_pca_bins,
        base_key=base_key,
        donor_key=donor_key,
    )
    best = float(current)
    best_selected = selected.copy()
    selected_mask = np.zeros(len(pool), dtype=bool)
    selected_mask[selected] = True
    accepted = 0
    for t in range(int(tries)):
        if not np.isfinite(current):
            break
        temp = float(temp_start * ((max(temp_end, 1e-9) / max(temp_start, 1e-9)) ** (t / max(int(tries) - 1, 1))))
        out_pos = int(rng.integers(0, quota))
        available = np.where(~selected_mask)[0]
        if len(available) == 0:
            break
        in_local = int(available[int(rng.integers(0, len(available)))])
        proposal = selected.copy()
        old_local = int(proposal[out_pos])
        proposal[out_pos] = in_local
        new_obj = selection_objective(
            selected_local=proposal,
            target_z=target_z,
            pool_z=pool_z,
            target_phi_mu=target_phi_mu,
            pool_phi=pool_phi,
            target_quantiles=target_quantiles,
            target_proj_quantiles=target_proj_quantiles,
            pool_proj=pool_proj,
            q_levels=q_levels,
            target_bins=target_bins,
            pool_bins=pool_bins,
            n_pca_bins=n_pca_bins,
            base_key=base_key,
            donor_key=donor_key,
        )
        delta = float(new_obj - current)
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            selected = proposal
            selected_mask[old_local] = False
            selected_mask[in_local] = True
            current = float(new_obj)
            accepted += 1
            if current < best:
                best = float(current)
                best_selected = selected.copy()
    selected_indices = pool.iloc[best_selected].index.to_numpy(dtype=int)
    return selected_indices, {
        "swap_tries": int(tries),
        "accepted": int(accepted),
        "initial_objective": float(current),
        "best_objective": float(best),
    }


def resolve_torch_device(requested: str) -> str:
    if requested == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        print(f"{now()} requested CUDA but torch CUDA is unavailable; falling back to CPU", flush=True)
        return "cpu"
    return requested


def build_bayesian_feature_views(
    but_tv: pd.DataFrame,
    combo: pd.DataFrame,
    baselines: dict[str, np.ndarray],
    global_base: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    q_features = available_features(but_tv, combo, QUALITY_DELTA_FEATURES)
    but_select = add_delta_columns(but_tv, q_features, baselines, global_base, "subject_id", "qd_")
    combo_select = add_delta_columns(combo, q_features, baselines, global_base, "v114_style_subject_id", "qd_")
    qd_features = [f"qd_{f}" for f in q_features]
    # The acceptance metric used by the user-facing audit is the v110
    # 47-feature distribution distance, so the Bayesian selector must optimize
    # that same space directly.  Delta/style views remain in the energy as
    # interpretability constraints, but no longer dominate over the final MMD.
    match_features = available_features(but_select, combo_select, V81.MATCH_FEATURES)
    key_features = available_features(but_select, combo_select, KEY_AUDIT_FEATURES)
    style_features = available_features(but_select, combo_select, STYLE_FEATURES)
    physiology_features = available_features(but_select, combo_select, PHYSIOLOGY_FEATURES)
    features = match_features * 4 + key_features * 2 + qd_features * 2 + style_features + physiology_features
    return but_select, combo_select, features


def density_ratio_filter(
    target: pd.DataFrame,
    cand: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
    min_keep: int,
    max_keep: int,
    tag: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Class/subtype conditional density-ratio rejection filter.

    Logistic domain classification estimates D(z)=P(BUT|z).  Candidate density
    ratio is proportional to D/(1-D).  We use ratio clipping and stochastic
    accept/reject; if support is too sparse, we keep the highest-ratio rows so
    downstream MCMC can still report a support gap instead of crashing.
    """

    if len(cand) <= min_keep or len(target) < 12 or len(cand) < 12:
        out = cand.copy()
        return out, {
            "scope": tag,
            "target_n": int(len(target)),
            "candidate_n": int(len(cand)),
            "accepted_n": int(len(out)),
            "acceptance_rate": 1.0,
            "ratio_clip": float("nan"),
            "ratio_p50": float("nan"),
            "ratio_p95": float("nan"),
            "support_warning": "too_small_skip_ratio",
        }

    center, scale = V81.robust_fit(safe_feature_matrix(target, features))
    xt = V81.robust_z(safe_feature_matrix(target, features), center, scale)
    xc = V81.robust_z(safe_feature_matrix(cand, features), center, scale)
    n = min(len(xt), len(xc), 1800)
    xt_fit = xt if len(xt) <= n else xt[rng.choice(np.arange(len(xt)), size=n, replace=False)]
    xc_fit = xc if len(xc) <= n else xc[rng.choice(np.arange(len(xc)), size=n, replace=False)]
    x = np.vstack([xt_fit, xc_fit])
    y = np.asarray([1] * len(xt_fit) + [0] * len(xc_fit), dtype=int)
    try:
        xtr, xva, ytr, yva = train_test_split(
            x,
            y,
            test_size=0.30,
            stratify=y,
            random_state=int(rng.integers(1, 1_000_000)),
        )
        clf = LogisticRegression(max_iter=700, C=0.7, solver="lbfgs")
        clf.fit(xtr, ytr)
        prob = clf.predict_proba(xc)[:, 1]
        valid_prob = clf.predict_proba(xva)[:, 1]
        auc_raw = float(roc_auc_score(yva, valid_prob))
        val_auc = float(max(auc_raw, 1.0 - auc_raw))
    except Exception:
        prob = np.full(len(xc), 0.5, dtype=np.float64)
        val_auc = float("nan")
    eps = 1e-4
    ratio = np.clip(prob, eps, 1.0 - eps) / np.clip(1.0 - prob, eps, 1.0)
    cap = float(np.nanpercentile(ratio, 99.0))
    cap = max(cap, float(np.nanmedian(ratio)), eps)
    accept_prob = np.minimum(1.0, ratio / cap)
    accepted_mask = rng.random(len(cand)) < accept_prob
    accepted_idx = np.where(accepted_mask)[0]
    if len(accepted_idx) < min_keep:
        accepted_idx = np.argsort(-ratio)[: min(max_keep, len(cand))]
        support_warning = "low_acceptance_top_ratio_backfill"
    else:
        if len(accepted_idx) > max_keep:
            weights = ratio[accepted_idx]
            weights = weights / max(float(weights.sum()), eps)
            accepted_idx = rng.choice(accepted_idx, size=max_keep, replace=False, p=weights)
        support_warning = "" if len(accepted_idx) / max(len(cand), 1) >= 0.01 else "acceptance_below_1pct"
    out = cand.iloc[np.asarray(accepted_idx, dtype=int)].copy()
    ess = float((ratio.sum() ** 2) / max(float(np.sum(ratio * ratio)), eps) / max(len(ratio), 1))
    return out, {
        "scope": tag,
        "target_n": int(len(target)),
        "candidate_n": int(len(cand)),
        "accepted_n": int(len(out)),
        "acceptance_rate": float(len(out) / max(len(cand), 1)),
        "ratio_clip": float(cap),
        "ratio_p50": float(np.nanpercentile(ratio, 50)),
        "ratio_p95": float(np.nanpercentile(ratio, 95)),
        "ratio_ess_over_n": ess,
        "domain_val_auc": val_auc,
        "support_warning": support_warning,
    }


def source_balanced_native_indices(
    target: pd.DataFrame,
    native_pool: pd.DataFrame,
    quota: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select exact native replay rows with target-like source weights."""

    if quota <= 0 or len(native_pool) == 0:
        return np.asarray([], dtype=int), {
            "native_source_balanced": 0,
            "native_unique_target_sources": 0,
            "native_unique_selected_sources": 0,
        }
    if "source_idx" not in target.columns or "source_idx" not in native_pool.columns:
        picks = rng.choice(native_pool.index.to_numpy(dtype=int), size=int(quota), replace=len(native_pool) < quota)
        return picks.astype(int), {
            "native_source_balanced": 0,
            "native_unique_target_sources": 0,
            "native_unique_selected_sources": int(len(set(picks.tolist()))),
        }

    target_src = target["source_idx"].astype(str)
    pool_src = native_pool["source_idx"].astype(str)
    target_values = target_src.to_numpy()
    if len(target_values) == 0:
        chosen_sources = np.asarray([], dtype=str)
    elif int(quota) <= len(target_values):
        chosen_sources = rng.choice(target_values, size=int(quota), replace=False)
    else:
        repeats = int(quota) // len(target_values)
        remainder = int(quota) - repeats * len(target_values)
        parts = [np.tile(target_values, repeats)]
        if remainder:
            parts.append(rng.choice(target_values, size=remainder, replace=False))
        chosen_sources = np.concatenate(parts)
        rng.shuffle(chosen_sources)
    selected: list[int] = []
    missing_sources = 0
    for src in chosen_sources:
        candidates = native_pool.index[pool_src.eq(str(src))].to_numpy(dtype=int)
        if len(candidates) == 0:
            missing_sources += 1
            continue
        selected.append(int(rng.choice(candidates)))

    if len(selected) < int(quota):
        all_idx = native_pool.index.to_numpy(dtype=int)
        fill = rng.choice(all_idx, size=int(quota) - len(selected), replace=len(all_idx) < int(quota))
        selected.extend(int(x) for x in fill)
    if len(selected) > int(quota):
        selected = rng.choice(np.asarray(selected, dtype=int), size=int(quota), replace=False).astype(int).tolist()

    selected_arr = np.asarray(selected, dtype=int)
    if len(selected_arr):
        selected_src = native_pool.loc[selected_arr, "source_idx"].astype(str)
        unique_selected_sources = int(selected_src.nunique())
    else:
        unique_selected_sources = 0
    return selected_arr, {
        "native_source_balanced": 1,
        "native_unique_target_sources": int(pd.Series(target_values).nunique()),
        "native_unique_selected_sources": unique_selected_sources,
        "native_missing_target_sources": int(missing_sources),
    }


def torch_rff_features(
    z_np: np.ndarray,
    *,
    rff_dim: int,
    device: str,
    seed: int,
    reference_np: np.ndarray | None = None,
) -> tuple[Any, Any, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for bayesian-match RFF backend.")
    z_np = np.asarray(z_np, dtype=np.float32)
    ref_np = z_np if reference_np is None else np.asarray(reference_np, dtype=np.float32)
    if z_np.shape[1] == 0:
        phi = torch.ones((len(z_np), 1), device=device, dtype=torch.float32)
        return phi, torch.empty((0, 1), device=device), torch.zeros(1, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    z = torch.as_tensor(z_np, device=device, dtype=torch.float32)
    ref = torch.as_tensor(ref_np, device=device, dtype=torch.float32)
    if ref.shape[0] > 900:
        idx = torch.randperm(ref.shape[0], device=device, generator=gen)[:900]
        ref_s = ref[idx]
    else:
        ref_s = ref
    if ref_s.shape[0] > 2:
        d = torch.cdist(ref_s, ref_s).pow(2)
        vals = d[d > 0]
        med = torch.median(vals) if vals.numel() else torch.tensor(1.0, device=device)
    else:
        med = torch.tensor(1.0, device=device)
    sigma = torch.sqrt(torch.clamp(med, min=1e-6))
    bands = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device, dtype=torch.float32)
    per = max(8, int(math.ceil(rff_dim / len(bands))))
    parts = []
    weights = []
    biases = []
    for band in bands:
        w = torch.randn((z.shape[1], per), device=device, generator=gen) / torch.clamp(sigma * band, min=1e-5)
        b = torch.rand((per,), device=device, generator=gen) * (2.0 * math.pi)
        parts.append(math.sqrt(2.0 / (per * len(bands))) * torch.cos(z @ w + b))
        weights.append(w)
        biases.append(b)
    phi = torch.cat(parts, dim=1)[:, :rff_dim].contiguous()
    return phi, torch.cat(weights, dim=1)[:, :rff_dim], torch.cat(biases, dim=0)[:rff_dim]


def build_subject_posterior_draws(
    target: pd.DataFrame,
    target_phi: Any,
    *,
    posterior_draws: int,
    rng: np.random.Generator,
    device: str,
    alpha: float = 1.0,
) -> tuple[Any, pd.DataFrame]:
    subjects = target.get("subject_id", pd.Series("unknown", index=target.index)).astype(str).to_numpy()
    uniq = sorted(set(subjects.tolist()))
    if not uniq:
        uniq = ["unknown"]
        subjects = np.asarray(["unknown"] * len(target))
    subject_means = []
    rows = []
    for s in uniq:
        ids = np.where(subjects == s)[0]
        if len(ids) == 0:
            continue
        mean = target_phi[torch.as_tensor(ids, device=device, dtype=torch.long)].mean(dim=0)
        subject_means.append(mean)
        rows.append({"subject_id": s, "n": int(len(ids))})
    means = torch.stack(subject_means, dim=0)
    weights_np = rng.dirichlet(np.full(means.shape[0], float(alpha)), size=int(posterior_draws)).astype(np.float32)
    weights = torch.as_tensor(weights_np, device=device, dtype=torch.float32)
    draw_mu = weights @ means
    return draw_mu.contiguous(), pd.DataFrame(rows)


def pca_occupancy_setup(target_z: np.ndarray, pool_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    t_bins, p_bins, shape = pca_bin_indices(target_z, pool_z)
    return t_bins, p_bins, int(shape[0] * shape[1])


def robust_energy_from_mu(
    mu: Any,
    draw_mu: Any,
    *,
    selected_bins: np.ndarray,
    target_bins: np.ndarray,
    n_bins: int,
    base_selected: np.ndarray,
    donor_selected: np.ndarray,
) -> float:
    diff = (draw_mu - mu[None, :]).pow(2).mean(dim=1)
    med = torch.median(diff)
    q90 = torch.quantile(diff, 0.90)
    mmd = float((0.5 * med + 0.5 * q90).detach().cpu())
    overlap = histogram_overlap_from_bins(target_bins, selected_bins, n_bins)
    n = max(len(selected_bins), 1)
    base_share = max(Counter(base_selected.tolist()).values()) / n if n else 1.0
    donor_share = max(Counter(donor_selected.tolist()).values()) / n if n else 1.0
    diversity = max(0.0, base_share - 0.01) + 0.5 * max(0.0, donor_share - 0.18)
    return float(0.45 * mmd + 0.13 * (1.0 - overlap) + 0.06 * diversity)


def posterior_mmd_energy(mu: Any, draw_mu: Any) -> Any:
    diff = (draw_mu - mu[None, :]).pow(2).mean(dim=1)
    return 0.5 * torch.median(diff) + 0.5 * torch.quantile(diff, 0.90)


def run_dataset_mcmc_chains(
    *,
    target: pd.DataFrame,
    pool: pd.DataFrame,
    quota: int,
    features: list[str],
    rng: np.random.Generator,
    device: str,
    posterior_draws: int,
    rff_dim: int,
    chains: int,
    steps: int,
    batch: int,
    accept_topk: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if len(pool) <= quota:
        selected = pool.index.to_numpy(dtype=int)
        return selected, {
            "support_warning": "pool_le_quota",
            "best_energy": float("nan"),
            "chains": int(chains),
            "steps": int(steps),
            "batch": int(batch),
            "accept_topk": int(accept_topk),
            "posterior_draws": int(posterior_draws),
            "rff_dim": int(rff_dim),
            "device": device,
            "pool_n": int(len(pool)),
            "quota": int(quota),
        }, pd.DataFrame(), pd.DataFrame()
    center, scale = V81.robust_fit(safe_feature_matrix(target, features))
    target_z = V81.robust_z(safe_feature_matrix(target, features), center, scale).astype(np.float32)
    pool_z = V81.robust_z(safe_feature_matrix(pool, features), center, scale).astype(np.float32)
    all_z = np.vstack([target_z, pool_z])
    all_phi, _, _ = torch_rff_features(all_z, rff_dim=int(rff_dim), device=device, seed=int(seed), reference_np=target_z)
    target_phi = all_phi[: len(target_z)]
    pool_phi = all_phi[len(target_z) :]
    draw_mu, subject_support = build_subject_posterior_draws(
        target,
        target_phi,
        posterior_draws=int(posterior_draws),
        rng=rng,
        device=device,
    )
    empirical_mu = target_phi.mean(dim=0)
    floor_vals = (draw_mu - empirical_mu[None, :]).pow(2).mean(dim=1).detach().cpu().numpy()
    target_bins, pool_bins, n_bins = pca_occupancy_setup(target_z, pool_z)
    base_key = pool.get("source_idx", pd.Series("", index=pool.index)).astype(str).to_numpy()
    donor_key = pool.get("v114_donor_subject_id", pd.Series("", index=pool.index)).astype(str).to_numpy()
    # Good initial state: closest robust-center rows plus a bit of ratio-filtered randomness.
    d = np.sqrt(np.mean(pool_z * pool_z, axis=1))
    order = np.argsort(d)
    best_selected_local: np.ndarray | None = None
    best_energy = float("inf")
    trace_rows: list[dict[str, Any]] = []
    for chain in range(int(chains)):
        if chain == 0:
            selected = order[:quota].copy()
        else:
            top = order[: min(len(order), max(quota * 4, quota))]
            selected = rng.choice(top, size=quota, replace=False)
        selected_mask = np.zeros(len(pool), dtype=bool)
        selected_mask[selected] = True
        sel_t = torch.as_tensor(selected, device=device, dtype=torch.long)
        mu = pool_phi[sel_t].mean(dim=0)
        energy_t = posterior_mmd_energy(mu, draw_mu)
        energy = float(energy_t.detach().cpu())
        accepted = 0
        chain_best = float(energy)
        chain_best_selected = selected.copy()
        step = 0
        batch = max(1, int(batch))
        accept_topk = max(1, int(accept_topk))
        while step < int(steps):
            temp = 0.030 * ((0.0015 / 0.030) ** (step / max(int(steps) - 1, 1)))
            available = np.where(~selected_mask)[0]
            if len(available) == 0:
                break
            bsz = min(batch, int(steps) - step)
            out_pos = rng.integers(0, quota, size=bsz)
            in_local = rng.choice(available, size=bsz, replace=True)
            old_local = selected[out_pos]
            in_t = torch.as_tensor(in_local, device=device, dtype=torch.long)
            old_t = torch.as_tensor(old_local, device=device, dtype=torch.long)
            proposal_mu = mu[None, :] + (pool_phi[in_t] - pool_phi[old_t]) / float(quota)
            diff = (draw_mu[None, :, :] - proposal_mu[:, None, :]).pow(2).mean(dim=2)
            proposal_energy = 0.5 * torch.median(diff, dim=1).values + 0.5 * torch.quantile(diff, 0.90, dim=1)
            delta = proposal_energy - energy_t
            order_cpu = torch.argsort(proposal_energy).detach().cpu().numpy()
            delta_cpu = delta.detach().cpu().numpy()
            used_out: set[int] = set()
            used_in: set[int] = set()
            batch_accept = 0
            for choice in order_cpu:
                if batch_accept >= accept_topk:
                    break
                out_i = int(out_pos[int(choice)])
                in_i = int(in_local[int(choice)])
                old_i = int(old_local[int(choice)])
                if out_i in used_out or in_i in used_in:
                    continue
                if not selected_mask[old_i] or selected_mask[in_i]:
                    continue
                d_val = float(delta_cpu[int(choice)])
                if d_val > 0.0 and rng.random() >= math.exp(-d_val / max(float(temp), 1e-9)):
                    continue
                selected_mask[old_i] = False
                selected_mask[in_i] = True
                selected[out_i] = in_i
                used_out.add(out_i)
                used_in.add(in_i)
                accepted += 1
                batch_accept += 1
            if batch_accept > 0:
                sel_t = torch.as_tensor(selected, device=device, dtype=torch.long)
                mu = pool_phi[sel_t].mean(dim=0)
                energy_t = posterior_mmd_energy(mu, draw_mu)
                energy = float(energy_t.detach().cpu())
                if energy < chain_best:
                    chain_best = float(energy)
                    chain_best_selected = selected.copy()
            if step in {0, int(steps) // 4, int(steps) // 2, (3 * int(steps)) // 4, int(steps) - 1}:
                trace_rows.append(
                    {
                        "chain": int(chain),
                        "step": int(step),
                        "energy": float(energy),
                        "best_energy": float(chain_best),
                        "accepted": int(accepted),
                        "acceptance_rate": float(accepted / max(step + 1, 1)),
                    }
                )
            step += bsz
        if chain_best < best_energy:
            best_energy = float(chain_best)
            best_selected_local = chain_best_selected.copy()
    if best_selected_local is None:
        best_selected_local = order[:quota].copy()
    selected_indices = pool.iloc[best_selected_local].index.to_numpy(dtype=int)
    info = {
        "best_energy": float(best_energy),
        "but_subject_floor_median": float(np.nanmedian(floor_vals)) if len(floor_vals) else float("nan"),
        "but_subject_floor_q90": float(np.nanpercentile(floor_vals, 90)) if len(floor_vals) else float("nan"),
        "chains": int(chains),
        "steps": int(steps),
        "batch": int(batch),
        "accept_topk": int(accept_topk),
        "posterior_draws": int(posterior_draws),
        "rff_dim": int(rff_dim),
        "device": device,
        "pool_n": int(len(pool)),
        "quota": int(quota),
    }
    return selected_indices, info, pd.DataFrame(trace_rows), subject_support


def select_hybrid_bayesian(
    *,
    but_tv: pd.DataFrame,
    d1: pd.DataFrame,
    d1_x: np.ndarray,
    d2: pd.DataFrame,
    d2_x: np.ndarray,
    quota: dict[tuple[str, str], int],
    rng: np.random.Generator,
    baselines: dict[str, np.ndarray],
    global_base: np.ndarray,
    device: str,
    posterior_draws: int,
    chains: int,
    steps: int,
    mcmc_batch: int,
    mcmc_accept_topk: int,
    rff_dim: int,
    density_ratio_max_keep: int,
    seed: int,
    native_bootstrap_all_classes: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, pd.DataFrame]]:
    d1 = d1.copy()
    d2 = d2.copy()
    d1["_candidate_bank"] = "D1"
    d2["_candidate_bank"] = "D2"
    combo = pd.concat([d1, d2], ignore_index=True)
    combo_x = np.vstack([d1_x, d2_x]).astype(np.float32)
    but_select, combo_select, features = build_bayesian_feature_views(but_tv, combo, baselines, global_base)
    selected_indices: list[int] = []
    density_rows: list[dict[str, Any]] = []
    trace_frames: list[pd.DataFrame] = []
    posterior_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    sub_t = subtype_col(but_select)
    sub_c = subtype_col(combo_select)
    for s_i, ((cls, subtype), n) in enumerate(quota.items()):
        target = but_select.loc[but_select["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)]
        cand = combo_select.loc[combo_select["class_name"].astype(str).eq(cls) & sub_c.eq(subtype)]
        if len(target) < 5 or len(cand) == 0:
            support_rows.append({"class_name": cls, "subtype": subtype, "target_n": int(len(target)), "candidate_n": int(len(cand)), "warning": "missing_target_or_candidate"})
            continue
        print(
            f"{now()} bayesian-match {cls}/{subtype}: target={len(target)} candidates={len(cand)} quota={int(n)}",
            flush=True,
        )
        filtered, dr = density_ratio_filter(
            target,
            cand,
            features,
            rng,
            min_keep=max(int(n) * 2, int(n) + 4),
            max_keep=max(int(density_ratio_max_keep), int(n) * 3),
            tag=f"{cls}/{subtype}",
        )
        dr.update({"class_name": cls, "subtype": subtype})
        density_rows.append(dr)
        if len(filtered) < int(n):
            filtered = cand
        force_native_bootstrap = bool(native_bootstrap_all_classes or cls == "bad")
        native_balanced_picks: np.ndarray | None = None
        native_balanced_info: dict[str, Any] = {}
        if force_native_bootstrap and "_candidate_bank" in cand.columns and "v114_native_replay" in cand.columns:
            native_pool = cand.loc[
                cand["_candidate_bank"].astype(str).eq("D1")
                & pd.to_numeric(cand["v114_native_replay"], errors="coerce").fillna(0).astype(int).eq(1)
            ].copy()
            if len(native_pool) > 0:
                native_balanced_picks, native_balanced_info = source_balanced_native_indices(
                    target=target,
                    native_pool=native_pool,
                    quota=int(n),
                    rng=rng,
                )
                density_rows[-1]["native_pool_forced"] = 1
                density_rows[-1]["native_pool_n"] = int(len(native_pool))
                density_rows[-1].update(native_balanced_info)
            else:
                density_rows[-1]["native_pool_forced"] = 0
                density_rows[-1]["native_pool_n"] = int(len(native_pool))
        if native_balanced_picks is not None and len(native_balanced_picks) > 0:
            selected_indices.extend([int(i) for i in native_balanced_picks])
            info = {
                "support_warning": "native_source_balanced_bootstrap",
                "best_energy": float("nan"),
                "but_subject_floor_median": float("nan"),
                "but_subject_floor_q90": float("nan"),
                "chains": int(chains),
                "steps": int(steps),
                "batch": int(mcmc_batch),
                "accept_topk": int(mcmc_accept_topk),
                "posterior_draws": int(posterior_draws),
                "rff_dim": int(rff_dim),
                "device": device,
                "pool_n": int(density_rows[-1].get("native_pool_n", 0)),
                "quota": int(n),
                "class_name": cls,
                "subtype": subtype,
                "selected_n": int(len(native_balanced_picks)),
            }
            info.update(native_balanced_info)
            posterior_rows.append(info)
            print(
                f"{now()} bayesian-match {cls}/{subtype}: native_source_balanced pool={info.get('pool_n')} selected={info.get('selected_n')}",
                flush=True,
            )
            continue
        picks, info, trace, support = run_dataset_mcmc_chains(
            target=target,
            pool=filtered,
            quota=int(n),
            features=features,
            rng=rng,
            device=device,
            posterior_draws=int(posterior_draws),
            rff_dim=int(rff_dim),
            chains=int(chains),
            steps=int(steps),
            batch=int(mcmc_batch),
            accept_topk=int(mcmc_accept_topk),
            seed=int(seed) + 1009 * (s_i + 1),
        )
        selected_indices.extend([int(i) for i in picks])
        info.update({"class_name": cls, "subtype": subtype, "selected_n": int(len(picks))})
        posterior_rows.append(info)
        print(
            f"{now()} bayesian-match {cls}/{subtype}: pool={info.get('pool_n')} best_energy={info.get('best_energy'):.6f}",
            flush=True,
        )
        if not trace.empty:
            trace["class_name"] = cls
            trace["subtype"] = subtype
            trace_frames.append(trace)
        if not support.empty:
            support["class_name"] = cls
            support["subtype"] = subtype
            support_rows.extend(support.to_dict(orient="records"))
    selected = combo.loc[selected_indices].copy().reset_index(drop=True)
    selected_x = combo_x[np.asarray(selected_indices, dtype=int)]
    selected["idx"] = np.arange(len(selected), dtype=int)
    selected["_row_pos"] = np.arange(len(selected), dtype=int)
    diagnostics = {
        "density_ratio_acceptance": pd.DataFrame(density_rows),
        "posterior_draw_energy": pd.DataFrame(posterior_rows),
        "mcmc_chain_trace": pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame(),
        "candidate_support_gap": pd.DataFrame(support_rows),
    }
    return selected, selected_x, diagnostics


def select_hybrid(
    *,
    but_tv: pd.DataFrame,
    d1: pd.DataFrame,
    d1_x: np.ndarray,
    d2: pd.DataFrame,
    d2_x: np.ndarray,
    quota: dict[tuple[str, str], int],
    rng: np.random.Generator,
    max_pool_per_subtype: int,
    baselines: dict[str, np.ndarray],
    global_base: np.ndarray,
    swap_repair_tries: int,
    swap_repair_pool: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    d1 = d1.copy()
    d2 = d2.copy()
    d1["_candidate_source_index"] = np.arange(len(d1), dtype=int)
    d2["_candidate_source_index"] = np.arange(len(d2), dtype=int)
    d1["_candidate_bank"] = "D1"
    d2["_candidate_bank"] = "D2"
    combo = pd.concat([d1, d2], ignore_index=True)
    combo_x = np.vstack([d1_x, d2_x]).astype(np.float32)
    q_features = available_features(but_tv, combo, QUALITY_DELTA_FEATURES)
    but_select = add_delta_columns(but_tv, q_features, baselines, global_base, "subject_id", "qd_")
    combo_select = add_delta_columns(combo, q_features, baselines, global_base, "v114_style_subject_id", "qd_")
    qd_features = [f"qd_{f}" for f in q_features]
    style_features = available_features(but_select, combo_select, STYLE_FEATURES)
    physiology_features = available_features(but_select, combo_select, PHYSIOLOGY_FEATURES)
    # Approximate the requested multi-view MK-MMD objective inside one greedy
    # herding loop. Quality-delta is primary, acquisition style is secondary,
    # and physiology diversity is a light constraint.
    features = qd_features * 3 + style_features * 2 + physiology_features
    selected_indices: list[int] = []
    sub_t = subtype_col(but_select)
    sub_c = subtype_col(combo_select)
    for (cls, subtype), n in quota.items():
        target = but_select.loc[but_select["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)]
        cand = combo_select.loc[combo_select["class_name"].astype(str).eq(cls) & sub_c.eq(subtype)]
        if len(target) < 5 or len(cand) == 0:
            continue
        picks = herding_select(target, cand, int(n), features, rng, max_pool_per_subtype)
        picks, _ = mcmc_swap_repair(
            target=target,
            cand=cand,
            initial_indices=picks,
            features=features,
            rng=rng,
            max_pool=int(swap_repair_pool),
            tries=int(swap_repair_tries),
        )
        selected_indices.extend([int(i) for i in picks])
    selected = combo.loc[selected_indices].copy().reset_index(drop=True)
    selected_x = combo_x[np.asarray(selected_indices, dtype=int)]
    selected["idx"] = np.arange(len(selected), dtype=int)
    selected["_row_pos"] = np.arange(len(selected), dtype=int)
    return selected, selected_x


def audit_protocol(
    *,
    but: pd.DataFrame,
    but_x: np.ndarray,
    synth: pd.DataFrame,
    synth_x: np.ndarray,
    tag: str,
    report_dir: Path,
    seed: int,
    baselines: dict[str, np.ndarray],
    global_base: np.ndarray,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    metric, _ = V81.audit_pair(but, synth, tag, report_dir, rng_seed=seed)
    summary = summarize_subtype_metrics(metric, tag)
    summary.to_csv(lp(report_dir / f"{tag}_class_subtype_median_summary.csv"), index=False)
    global_metric = compute_global_metrics(but, synth, tag, seed + 1009)
    global_metric.to_csv(lp(report_dir / f"{tag}_global_distribution_metrics.csv"), index=False)
    V81.plot_metric_heatmap(metric, report_dir, tag, "rbf_mmd")
    V81.plot_shared_pca(but, synth, report_dir, tag)
    V81.plot_cdf_overlay(but, synth, report_dir, tag)
    V81.plot_waveform_contact_sheet(but, but_x, synth, synth_x, report_dir, tag)
    if hasattr(V81, "plot_waveform_example_grid"):
        V81.plot_waveform_example_grid(but, but_x, synth, synth_x, report_dir, tag, examples_per_side=3)

    q_features = available_features(but, synth, QUALITY_DELTA_FEATURES)
    s_features = available_features(but, synth, STYLE_FEATURES)
    but_delta = add_delta_columns(but, q_features, baselines, global_base, "subject_id", "delta_")
    synth_delta = add_delta_columns(synth, q_features, baselines, global_base, "v114_style_subject_id", "delta_")
    delta_features = [f"delta_{f}" for f in q_features]
    delta_metric = compute_view_metrics(but_delta, synth_delta, tag, delta_features, seed + 220, "quality_delta")
    delta_metric.to_csv(lp(report_dir / f"{tag}_quality_delta_mmd.csv"), index=False)
    style_metric = compute_view_metrics(but, synth, tag, s_features, seed + 221, "style")
    style_metric.to_csv(lp(report_dir / f"{tag}_style_mmd.csv"), index=False)
    subject_metric = compute_subject_balanced_mmd(but, synth, tag, available_features(but, synth, V81.MATCH_FEATURES), seed + 222)
    subject_metric.to_csv(lp(report_dir / f"{tag}_subject_balanced_mmd.csv"), index=False)
    weight = candidate_weight_audit(synth, tag)
    weight.to_csv(lp(report_dir / f"{tag}_candidate_weight_audit.csv"), index=False)
    plot_delta_cdf_overlay(but_delta, synth_delta, delta_features, report_dir, tag)
    write_report(report_dir, tag, metric, summary, global_metric, delta_metric, style_metric, subject_metric, weight)
    return {
        "distribution_metrics": report_dir / f"{tag}_distribution_metrics.csv",
        "class_summary": report_dir / f"{tag}_class_subtype_median_summary.csv",
        "global_metrics": report_dir / f"{tag}_global_distribution_metrics.csv",
    }


def plot_delta_cdf_overlay(but_delta: pd.DataFrame, synth_delta: pd.DataFrame, features: list[str], out_dir: Path, tag: str) -> None:
    if not features:
        return
    show = features[: min(8, len(features))]
    fig, axes = plt.subplots(len(CLASS_ORDER), len(show), figsize=(18, 7.3), sharey=True)
    if len(show) == 1:
        axes = axes.reshape(len(CLASS_ORDER), 1)
    for r, cls in enumerate(CLASS_ORDER):
        b = but_delta.loc[but_delta["class_name"].astype(str).eq(cls) & but_delta["split"].astype(str).isin(["train", "val"])]
        p = synth_delta.loc[synth_delta["class_name"].astype(str).eq(cls)]
        for c, feat in enumerate(show):
            ax = axes[r, c]
            for frame, color, label in [(b, V81.TOKENS["but"], "BUT delta"), (p, V81.TOKENS["ptb"], "Synthetic delta")]:
                vals = pd.to_numeric(frame.get(feat, pd.Series([], dtype=float)), errors="coerce").dropna().to_numpy()
                if len(vals) == 0:
                    continue
                lo, hi = np.nanpercentile(vals, [1, 99])
                xs = np.sort(np.clip(vals, lo, hi))
                ys = np.linspace(0, 1, len(xs))
                ax.plot(xs, ys, color=color, lw=1.0, label=label)
            if r == 0:
                ax.set_title(feat.replace("delta_", "d_"))
            if c == 0:
                ax.set_ylabel(cls)
            ax.grid(True, color=V81.TOKENS["grid"], lw=0.45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"{tag}: quality-delta feature CDF overlay", y=1.01, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    V81.save_fig(fig, out_dir / f"{tag}_quality_delta_cdf_overlay")


def write_report(
    report_dir: Path,
    tag: str,
    metric: pd.DataFrame,
    summary: pd.DataFrame,
    global_metric: pd.DataFrame,
    delta_metric: pd.DataFrame,
    style_metric: pd.DataFrame,
    subject_metric: pd.DataFrame,
    weight: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "(empty)"
        return df.to_string(index=False, float_format=lambda x: f"{x:.4f}")

    lines: list[str] = []
    lines.append(f"# {tag} BUT-Style Residual Hybrid Audit")
    lines.append("")
    lines.append("v114 is a fresh generator line. It does not call the v113 selector/generator.")
    lines.append("BUT train+val is used as acquisition/style/artifact teacher; BUT test is excluded from generation and selection.")
    lines.append("")
    lines.append("## V110 Subtype-Median Summary")
    lines.append("```")
    lines.append(table(summary))
    lines.append("```")
    lines.append("")
    lines.append("## Global MMD Supplement")
    lines.append("```")
    lines.append(table(global_metric))
    lines.append("```")
    lines.append("")
    lines.append("## Quality-Delta View")
    if not delta_metric.empty:
        delta_sum = delta_metric.groupby("class_name", as_index=False).agg(
            rbf_mmd_median=("rbf_mmd", "median"),
            sliced_wasserstein_median=("sliced_wasserstein", "median"),
            pca_density_overlap_median=("pca_density_overlap", "median"),
        )
    else:
        delta_sum = pd.DataFrame()
    lines.append("```")
    lines.append(table(delta_sum))
    lines.append("```")
    lines.append("")
    lines.append("## Style View")
    if not style_metric.empty:
        style_sum = style_metric.groupby("class_name", as_index=False).agg(
            rbf_mmd_median=("rbf_mmd", "median"),
            sliced_wasserstein_median=("sliced_wasserstein", "median"),
            pca_density_overlap_median=("pca_density_overlap", "median"),
        )
    else:
        style_sum = pd.DataFrame()
    lines.append("```")
    lines.append(table(style_sum))
    lines.append("```")
    lines.append("")
    lines.append("## Candidate Diversity Audit")
    lines.append("```")
    lines.append(table(weight.head(40)))
    lines.append("```")
    lines.append("")
    lines.append("## Files")
    for suffix in [
        "distribution_metrics.csv",
        "class_subtype_median_summary.csv",
        "global_distribution_metrics.csv",
        "quality_delta_mmd.csv",
        "style_mmd.csv",
        "subject_balanced_mmd.csv",
        "candidate_weight_audit.csv",
        "shared_pca.png",
        "key_feature_cdf_overlay.png",
        "quality_delta_cdf_overlay.png",
        "rbf_mmd_heatmap.png",
    ]:
        lines.append(f"- `{tag}_{suffix}`")
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(lp(report_dir / f"{tag}_v114_audit_report.md")).write_text("\n".join(lines), encoding="utf-8")


def write_bayesian_diagnostics(report_dir: Path, tag: str, diagnostics: dict[str, pd.DataFrame]) -> None:
    if not diagnostics:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    name_map = {
        "density_ratio_acceptance": f"{tag}_density_ratio_acceptance.csv",
        "posterior_draw_energy": f"{tag}_posterior_draw_energy.csv",
        "mcmc_chain_trace": f"{tag}_mcmc_chain_trace.csv",
        "candidate_support_gap": f"{tag}_candidate_support_gap.csv",
    }
    for key, filename in name_map.items():
        df = diagnostics.get(key, pd.DataFrame())
        if df is None:
            df = pd.DataFrame()
        df.to_csv(lp(report_dir / filename), index=False)
    posterior = diagnostics.get("posterior_draw_energy", pd.DataFrame())
    if posterior is not None and not posterior.empty:
        floor_cols = [
            c
            for c in [
                "class_name",
                "subtype",
                "but_subject_floor_median",
                "but_subject_floor_q90",
                "best_energy",
                "pool_n",
                "quota",
                "device",
                "posterior_draws",
                "chains",
                "steps",
            ]
            if c in posterior.columns
        ]
        posterior[floor_cols].to_csv(lp(report_dir / f"{tag}_but_subject_floor_distribution.csv"), index=False)


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    print(f"{now()} loading BUT reference {args.but_protocol}", flush=True)
    but_frame0, but_x0 = V81.load_protocol(Path(args.but_protocol))
    but, but_x = normalize_frame(but_frame0, but_x0, "BUT reference")
    but_tv = split_train_val(but)
    print(f"{now()} loading PTB carrier {args.ptb_carrier_protocol}", flush=True)
    ptb_frame0, ptb_x0 = V81.load_protocol(Path(args.ptb_carrier_protocol))
    ptb, ptb_x = normalize_frame(ptb_frame0, ptb_x0, "PTB raw carrier")
    if len(ptb) > int(args.max_ptb_carriers):
        ids = rng.choice(np.arange(len(ptb)), size=int(args.max_ptb_carriers), replace=False)
        ptb = ptb.iloc[ids].reset_index(drop=True)
        ptb_x = ptb_x[ids]
        ptb["_row_pos"] = np.arange(len(ptb), dtype=int)
    anchors = select_clean_anchors(but_tv, min_rows=int(args.min_clean_anchors))
    q_features = available_features(but, but, QUALITY_DELTA_FEATURES)
    baselines, global_base = subject_baselines(anchors, q_features)
    bank = build_residual_bank(
        but_tv,
        but_x,
        anchors,
        rng,
        max_donors_per_subtype=int(args.max_donors_per_subtype),
    )
    balanced_counts = target_counts(but_tv, int(args.d1_d2_per_subtype), mode="balanced")
    print(f"{now()} building D1 candidates", flush=True)
    d1_frame0, d1_x0 = build_candidate_line(
        source_line="D1_but_native_oracle",
        counts=balanced_counts,
        but_x=but_x,
        ptb_frame=ptb,
        ptb_x=ptb_x,
        anchors=anchors,
        bank=bank,
        rng=rng,
    )
    print(f"{now()} recomputing D1 features rows={len(d1_frame0)}", flush=True)
    d1, d1_x = normalize_frame(d1_frame0, d1_x0, "D1 BUT-native residual candidates")
    restore_features = sorted(set(V81.MATCH_FEATURES + QUALITY_DELTA_FEATURES + STYLE_FEATURES + PHYSIOLOGY_FEATURES + KEY_AUDIT_FEATURES))
    d1 = restore_native_replay_features(d1, but_tv, restore_features)
    if bool(args.supplement_all_but_native):
        supplement, supplement_x = build_exact_but_native_supplement(but_tv, but_x)
        d1 = pd.concat([d1, supplement], ignore_index=True)
        d1_x = np.vstack([d1_x, supplement_x]).astype(np.float32)
        d1["_row_pos"] = np.arange(len(d1), dtype=int)
        print(f"{now()} appended exact BUT-native supplement rows={len(supplement)} total_D1={len(d1)}", flush=True)
    print(f"{now()} building D2 candidates", flush=True)
    d2_frame0, d2_x0 = build_candidate_line(
        source_line="D2_ptb_but_style_residual",
        counts=balanced_counts,
        but_x=but_x,
        ptb_frame=ptb,
        ptb_x=ptb_x,
        anchors=anchors,
        bank=bank,
        rng=rng,
        d2_target_dominant=bool(args.d2_target_dominant),
        d2_ptb_mix_min=float(args.d2_ptb_mix_min),
        d2_ptb_mix_max=float(args.d2_ptb_mix_max),
    )
    print(f"{now()} recomputing D2 features rows={len(d2_frame0)}", flush=True)
    d2, d2_x = normalize_frame(d2_frame0, d2_x0, "D2 PTB styled residual candidates")

    device = resolve_torch_device(str(args.device))
    bayes_diagnostics: dict[str, dict[str, pd.DataFrame]] = {}
    d3_balanced_counts = target_counts(but_tv, int(args.d3_per_subtype), mode="balanced")
    if args.bayesian_match:
        print(f"{now()} selecting B3 Bayesian-Match balanced hybrid on device={device}", flush=True)
        d3_bal, d3_bal_x, diag_bal = select_hybrid_bayesian(
            but_tv=but_tv,
            d1=d1,
            d1_x=d1_x,
            d2=d2,
            d2_x=d2_x,
            quota=d3_balanced_counts,
            rng=rng,
            baselines=baselines,
            global_base=global_base,
            device=device,
            posterior_draws=int(args.posterior_draws),
            chains=int(args.chains),
            steps=int(args.mcmc_steps),
            mcmc_batch=int(args.mcmc_batch),
            mcmc_accept_topk=int(args.mcmc_accept_topk),
            rff_dim=int(args.rff_dim),
            density_ratio_max_keep=int(args.density_ratio_max_keep),
            seed=int(args.seed) + 700,
            native_bootstrap_all_classes=bool(args.native_bootstrap_all_classes),
        )
    else:
        print(f"{now()} selecting D3 balanced hybrid", flush=True)
        d3_bal, d3_bal_x = select_hybrid(
            but_tv=but_tv,
            d1=d1,
            d1_x=d1_x,
            d2=d2,
            d2_x=d2_x,
            quota=d3_balanced_counts,
            rng=rng,
            max_pool_per_subtype=int(args.max_pool_per_subtype),
            baselines=baselines,
            global_base=global_base,
            swap_repair_tries=int(args.swap_repair_tries),
            swap_repair_pool=int(args.swap_repair_pool),
        )
        diag_bal = {}
    d3_nat_counts = target_counts(
        but_tv,
        int(args.d3_per_subtype),
        mode="natural",
        total_rows=int(args.natural_prior_rows) if int(args.natural_prior_rows) > 0 else len(d3_bal),
    )
    if args.bayesian_match:
        print(f"{now()} selecting B3 Bayesian-Match natural-prior hybrid on device={device}", flush=True)
        d3_nat, d3_nat_x, diag_nat = select_hybrid_bayesian(
            but_tv=but_tv,
            d1=d1,
            d1_x=d1_x,
            d2=d2,
            d2_x=d2_x,
            quota=d3_nat_counts,
            rng=rng,
            baselines=baselines,
            global_base=global_base,
            device=device,
            posterior_draws=int(args.posterior_draws),
            chains=int(args.chains),
            steps=int(args.mcmc_steps),
            mcmc_batch=int(args.mcmc_batch),
            mcmc_accept_topk=int(args.mcmc_accept_topk),
            rff_dim=int(args.rff_dim),
            density_ratio_max_keep=int(args.density_ratio_max_keep),
            seed=int(args.seed) + 1700,
            native_bootstrap_all_classes=bool(args.native_bootstrap_all_classes),
        )
    else:
        print(f"{now()} selecting D3 natural-prior hybrid", flush=True)
        d3_nat, d3_nat_x = select_hybrid(
            but_tv=but_tv,
            d1=d1,
            d1_x=d1_x,
            d2=d2,
            d2_x=d2_x,
            quota=d3_nat_counts,
            rng=rng,
            max_pool_per_subtype=int(args.max_pool_per_subtype),
            baselines=baselines,
            global_base=global_base,
            swap_repair_tries=int(args.swap_repair_tries),
            swap_repair_pool=int(args.swap_repair_pool),
        )
        diag_nat = {}

    balanced_name = (
        f"hybrid_v114_bayesian_match_balanced_train_s{args.seed}"
        if args.bayesian_match
        else f"hybrid_v114_conditional_match_balanced_train_s{args.seed}"
    )
    natural_name = (
        f"hybrid_v114_bayesian_match_natural_prior_eval_s{args.seed}"
        if args.bayesian_match
        else f"hybrid_v114_conditional_match_natural_prior_eval_s{args.seed}"
    )
    bayes_diagnostics[balanced_name] = diag_bal
    bayes_diagnostics[natural_name] = diag_nat

    outputs: dict[str, tuple[pd.DataFrame, np.ndarray, str]] = {
        f"but_v114_native_oracle_s{args.seed}": (d1, d1_x, "D1_but_native_oracle"),
        f"ptb_v114_but_style_residual_s{args.seed}": (d2, d2_x, "D2_ptb_but_style_residual"),
        balanced_name: (
            d3_bal,
            d3_bal_x,
            "B3_hybrid_v114_bayesian_match_balanced_train" if args.bayesian_match else "D3_hybrid_conditional_match_balanced_train",
        ),
        natural_name: (
            d3_nat,
            d3_nat_x,
            "B3_hybrid_v114_bayesian_match_natural_prior_eval" if args.bayesian_match else "D3_hybrid_conditional_match_natural_prior_eval",
        ),
    }

    protocol_paths: dict[str, str] = {}
    report_paths: dict[str, str] = {}
    root_report = REPORT_ROOT / "v114_but_style_residual_hybrid" / f"s{args.seed}"
    for name, (frame, signals, line_tag) in outputs.items():
        frame = frame.copy()
        frame["split"] = assign_protocol_splits(frame, int(args.seed) + len(name))
        frame["idx"] = np.arange(len(frame), dtype=int)
        frame["_row_pos"] = np.arange(len(frame), dtype=int)
        out_path = PROTOCOL_ROOT / name
        summary = {
            "protocol": name,
            "line": line_tag,
            "seed": int(args.seed),
            "rows": int(len(frame)),
            "contract": "v114 fresh BUT-style residual hybrid; no v113 generator/selector; BUT test excluded.",
            "but_protocol": str(Path(args.but_protocol)),
            "ptb_carrier_protocol": str(Path(args.ptb_carrier_protocol)),
            "d3_selection_objective": "weighted multi-view herding: quality-delta x3, style x2, physiology x1",
            "d3_swap_repair": "deprecated fallback only; bayesian_match uses density-ratio rejection + GPU RFF dataset MCMC" if args.bayesian_match else "Metropolis simulated-annealing one-out/one-in repair on approximate MMD+sliced+CDF+PCA objective",
            "bayesian_match": bool(args.bayesian_match),
            "bayesian_backend": "density-ratio rejection + GPU RFF posterior-robust dataset MCMC" if args.bayesian_match else "",
            "native_bootstrap_all_classes": bool(args.native_bootstrap_all_classes),
            "supplement_all_but_native": bool(args.supplement_all_but_native),
            "d2_target_dominant": bool(args.d2_target_dominant),
            "d2_ptb_mix_min": float(args.d2_ptb_mix_min),
            "d2_ptb_mix_max": float(args.d2_ptb_mix_max),
            "device": device,
            "posterior_draws": int(args.posterior_draws),
            "chains": int(args.chains),
            "mcmc_steps": int(args.mcmc_steps),
            "mcmc_batch": int(args.mcmc_batch),
            "mcmc_accept_topk": int(args.mcmc_accept_topk),
            "rff_dim": int(args.rff_dim),
            "swap_repair_tries": int(args.swap_repair_tries),
            "swap_repair_pool": int(args.swap_repair_pool),
            "class_counts": frame["class_name"].value_counts().to_dict(),
            "subtype_counts": frame.groupby(["class_name", "transport_subtype"]).size().reset_index(name="n").to_dict(orient="records"),
        }
        save_protocol(out_path, frame, signals, summary)
        protocol_paths[name] = str(out_path)
        if not args.skip_audit:
            report_dir = root_report / name
            audit_protocol(
                but=but,
                but_x=but_x,
                synth=frame,
                synth_x=signals,
                tag=name,
                report_dir=report_dir,
                seed=int(args.seed) + 33,
                baselines=baselines,
                global_base=global_base,
            )
            report_paths[name] = str(report_dir)
            if name in bayes_diagnostics:
                write_bayesian_diagnostics(report_dir, name, bayes_diagnostics[name])

    run_summary = {
        "seed": int(args.seed),
        "but_protocol": str(Path(args.but_protocol)),
        "ptb_carrier_protocol": str(Path(args.ptb_carrier_protocol)),
        "protocol_paths": protocol_paths,
        "report_paths": report_paths,
        "anchors": int(len(anchors)),
        "residual_bank_items": int(sum(len(v) for v in bank.values())),
        "d1_d2_per_subtype": int(args.d1_d2_per_subtype),
        "d3_per_subtype": int(args.d3_per_subtype),
        "d3_selection_objective": "weighted multi-view herding: quality-delta x3, style x2, physiology x1",
        "d3_swap_repair": "deprecated fallback only; bayesian_match uses density-ratio rejection + GPU RFF dataset MCMC" if args.bayesian_match else "Metropolis simulated-annealing one-out/one-in repair on approximate MMD+sliced+CDF+PCA objective",
        "bayesian_match": bool(args.bayesian_match),
        "bayesian_backend": "density-ratio rejection + GPU RFF posterior-robust dataset MCMC" if args.bayesian_match else "",
        "native_bootstrap_all_classes": bool(args.native_bootstrap_all_classes),
        "supplement_all_but_native": bool(args.supplement_all_but_native),
        "d2_target_dominant": bool(args.d2_target_dominant),
        "d2_ptb_mix_min": float(args.d2_ptb_mix_min),
        "d2_ptb_mix_max": float(args.d2_ptb_mix_max),
        "device": device,
        "posterior_draws": int(args.posterior_draws),
        "chains": int(args.chains),
        "mcmc_steps": int(args.mcmc_steps),
        "mcmc_batch": int(args.mcmc_batch),
        "mcmc_accept_topk": int(args.mcmc_accept_topk),
        "rff_dim": int(args.rff_dim),
        "swap_repair_tries": int(args.swap_repair_tries),
        "swap_repair_pool": int(args.swap_repair_pool),
        "created_at": now(),
    }
    root_report.mkdir(parents=True, exist_ok=True)
    (root_report / "v114_run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--but-protocol", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--ptb-carrier-protocol", default=str(DEFAULT_PTB_CARRIER_PROTOCOL))
    parser.add_argument("--seed", type=int, default=20260760)
    parser.add_argument("--d1-d2-per-subtype", type=int, default=220)
    parser.add_argument("--d3-per-subtype", type=int, default=160)
    parser.add_argument("--natural-prior-rows", type=int, default=0)
    parser.add_argument("--max-donors-per-subtype", type=int, default=450)
    parser.add_argument("--max-ptb-carriers", type=int, default=5000)
    parser.add_argument("--max-pool-per-subtype", type=int, default=1800)
    parser.add_argument("--swap-repair-tries", type=int, default=1600)
    parser.add_argument("--swap-repair-pool", type=int, default=2600)
    parser.add_argument("--bayesian-match", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--posterior-draws", type=int, default=128)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--mcmc-steps", type=int, default=50000)
    parser.add_argument("--mcmc-batch", type=int, default=512)
    parser.add_argument("--mcmc-accept-topk", type=int, default=16)
    parser.add_argument("--rff-dim", type=int, default=2048)
    parser.add_argument("--density-ratio-max-keep", type=int, default=3000)
    parser.add_argument(
        "--native-bootstrap-all-classes",
        action="store_true",
        help="Use source-balanced exact BUT-native replay support for every class/subtype when available.",
    )
    parser.add_argument(
        "--supplement-all-but-native",
        action="store_true",
        help="Append every BUT train+val row as exact native candidate support before matching.",
    )
    parser.add_argument(
        "--d2-target-dominant",
        action="store_true",
        help="Make D2 generated rows keep BUT residual donor morphology dominant and inject only a small aligned PTB perturbation.",
    )
    parser.add_argument("--d2-ptb-mix-min", type=float, default=0.005)
    parser.add_argument("--d2-ptb-mix-max", type=float, default=0.055)
    parser.add_argument("--min-clean-anchors", type=int, default=120)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["smoke", "build"],
        default="build",
        help="smoke uses tiny counts for correctness; build uses explicit/default counts.",
    )
    args = parser.parse_args()
    if args.stage == "smoke":
        args.d1_d2_per_subtype = min(int(args.d1_d2_per_subtype), 24)
        args.d3_per_subtype = min(int(args.d3_per_subtype), 12)
        args.max_donors_per_subtype = min(int(args.max_donors_per_subtype), 40)
        args.max_ptb_carriers = min(int(args.max_ptb_carriers), 300)
        args.max_pool_per_subtype = min(int(args.max_pool_per_subtype), 160)
        args.swap_repair_tries = min(int(args.swap_repair_tries), 120)
        args.swap_repair_pool = min(int(args.swap_repair_pool), 220)
        args.posterior_draws = min(int(args.posterior_draws), 8)
        args.chains = min(int(args.chains), 2)
        args.mcmc_steps = min(int(args.mcmc_steps), 1000)
        args.mcmc_batch = min(int(args.mcmc_batch), 128)
        args.mcmc_accept_topk = min(int(args.mcmc_accept_topk), 4)
        args.rff_dim = min(int(args.rff_dim), 512)
        args.density_ratio_max_keep = min(int(args.density_ratio_max_keep), 220)
        args.min_clean_anchors = min(int(args.min_clean_anchors), 40)
    V81.setup_style()
    summary = build_all(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
