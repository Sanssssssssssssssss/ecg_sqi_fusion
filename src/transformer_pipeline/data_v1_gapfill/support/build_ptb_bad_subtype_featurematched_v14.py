"""Build PTB bad-subtype protocol with feature-matched candidate selection.

This is the next step after v12/v13:

1. Keep the interpretable BUT bad subtype proportions.
2. Generate several PTB-only candidate waveforms for each bad subtype.
3. Score candidates against BUT subtype medians/IQRs using waveform-computable
   SQI/morphology features.
4. Keep the closest candidates and write a protocol-compatible synthetic bank.

BUT is used only as a feature-distribution target; no BUT waveform is copied.
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

V12_SCRIPT = SCRIPT_DIR / "build_ptb_bad_subtype_balanced_v2.py"
BASE_PROTOCOL = ANALYSIS_DIR / "event_xds_aligned_v11_buttrain_style_replay" / "protocol_ptb_buttrain_aligned_pc3000_s20260620"
BUT_REFERENCE_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_keep_outlier"
OUT_DIR = ANALYSIS_DIR / "event_xds_aligned_v14_bad_subtype_featurematched"
REPORT_OUT_DIR = REPORT_DIR / "event_xds_aligned_v14_bad_subtype_featurematched"
ACTIVE_VERSION = "v14"

MATCH_FEATURES = [
    "baseline_step",
    "qrs_visibility",
    "qrs_band_ratio",
    "detector_agreement",
    "non_qrs_diff_p95",
    "band_30_45",
    "flatline_ratio",
    "sqi_basSQI",
    "low_amp_ratio",
    "template_corr",
    "amplitude_entropy",
    "mean_abs",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILD = load_module(V12_SCRIPT, "ptb_bad_builder_v14_base")
SUB = BUILD.SUB
FEATURES = BUILD.FEATURES
CLASS_TO_INT = BUILD.CLASS_TO_INT
BAD_SUBTYPES = BUILD.BAD_SUBTYPES


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(long_path(path), "w", encoding="utf-8") as f:
        f.write(text)


def target_stats(but: pd.DataFrame) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    labels = [SUB.SUBTYPES[i] for i in SUB.subtype_labels(but)]
    bad = but.assign(target_bad_subtype=labels)
    bad = bad.loc[bad["class_name"].astype(str).eq("bad")].copy()
    rows = []
    stats: dict[str, dict[str, float]] = {}
    for subtype in BAD_SUBTYPES:
        sub = bad.loc[bad["target_bad_subtype"].astype(str).eq(subtype)].copy()
        if sub.empty:
            sub = bad
        stats[subtype] = {}
        for col in MATCH_FEATURES:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
            if vals.size == 0:
                vals = np.asarray([0.0], dtype=np.float32)
            med = float(np.median(vals))
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            scale = max(iqr, float(np.std(vals)), 1e-3)
            stats[subtype][f"{col}_median"] = med
            stats[subtype][f"{col}_scale"] = scale
            rows.append({"subtype": subtype, "feature": col, "target_median": med, "target_scale": scale, "n_but": int(len(sub))})
    return stats, pd.DataFrame(rows)


def target_stats_from_protocol(protocol: Path) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Build subtype targets after recomputing feature columns from signals."""

    frame = pd.read_csv(protocol / "original_region_atlas.csv")
    bad = frame.loc[frame["class_name"].astype(str).eq("bad")].copy()
    labels = [SUB.SUBTYPES[i] for i in SUB.subtype_labels(bad)]
    bad["target_bad_subtype"] = labels
    x = np.load(protocol / "signals.npz")["X"].astype(np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    rows = bad["idx"].astype(int).to_numpy()
    primitives = FEATURES.compute_primitives(x[rows])
    for col in MATCH_FEATURES:
        if col in primitives.columns:
            bad[col] = primitives[col].to_numpy(dtype=np.float32)
    return target_stats(bad)


def score_candidates(features: pd.DataFrame, subtype: str, stats: dict[str, dict[str, float]]) -> np.ndarray:
    parts = []
    for col in MATCH_FEATURES:
        vals = pd.to_numeric(features[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        med = float(stats[subtype][f"{col}_median"])
        scale = float(stats[subtype][f"{col}_scale"])
        parts.append(np.minimum(np.abs(vals - med) / scale, 8.0))
    score = np.mean(np.stack(parts, axis=1), axis=1)
    # Penalize the exact v12/v13 failure mode hard: too-visible QRS and too
    # strong derivative tails relative to the BUT subtype target.
    qrs_hi = np.maximum(0.0, pd.to_numeric(features["qrs_visibility"], errors="coerce").fillna(0.0).to_numpy() - stats[subtype]["qrs_visibility_median"])
    diff_hi = np.maximum(0.0, pd.to_numeric(features["non_qrs_diff_p95"], errors="coerce").fillna(0.0).to_numpy() - stats[subtype]["non_qrs_diff_p95_median"])
    band45_hi = np.maximum(0.0, pd.to_numeric(features["band_30_45"], errors="coerce").fillna(0.0).to_numpy() - stats[subtype]["band_30_45_median"])
    det_vals = pd.to_numeric(features["detector_agreement"], errors="coerce").fillna(0.0).to_numpy()
    score += 0.45 * np.minimum(qrs_hi / stats[subtype]["qrs_visibility_scale"], 8.0)
    score += 0.35 * np.minimum(diff_hi / stats[subtype]["non_qrs_diff_p95_scale"], 8.0)
    score += 1.20 * np.minimum(band45_hi / stats[subtype]["band_30_45_scale"], 8.0)
    if ACTIVE_VERSION in {"v19", "v20"}:
        det_gap = np.abs(det_vals - stats[subtype]["detector_agreement_median"])
        score += 1.75 * np.minimum(det_gap / stats[subtype]["detector_agreement_scale"], 8.0)
    return score.astype(np.float32)


def smooth_bad_candidate(base: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth/continuous bad candidates that avoid spiky pseudo-QRS.

    v12/v13 candidates were often rejected because local spikes made
    qrs_visibility artificially high.  These candidates mimic the BUT bad
    failure shape where the whole window is contaminated, so local peaks are
    less exceptional relative to the background.
    """

    x = BUILD.robust_norm(base)
    n = len(x)
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    centered = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    phase1 = float(rng.uniform(-np.pi, np.pi))
    phase2 = float(rng.uniform(-np.pi, np.pi))
    lf_amp = float(rng.uniform(0.55, 1.25))
    lf = lf_amp * np.sin(2.0 * np.pi * float(rng.uniform(0.08, 0.24)) * t + phase1).astype(np.float32)
    if subtype == "bad_highfreq_detail_noise":
        hf_freq = float(rng.uniform(30.0, 44.0))
        hf_amp = float(rng.uniform(0.08, 0.28))
    else:
        hf_freq = float(rng.uniform(12.0, 28.0))
        hf_amp = float(rng.uniform(0.015, 0.12))
    hf = hf_amp * np.sin(2.0 * np.pi * hf_freq * t + phase2).astype(np.float32)
    y = float(rng.uniform(0.03, 0.18)) * x + lf + hf

    if subtype == "bad_baseline_wander_lowfreq":
        y += float(rng.uniform(0.15, 0.55)) * centered
        y += float(rng.uniform(0.04, 0.18)) * (centered * centered - 0.33)
    elif subtype == "bad_contact_reset_flatline":
        y = float(rng.uniform(0.04, 0.16)) * x + 0.65 * lf + 0.35 * hf
        for _ in range(int(rng.integers(1, 4))):
            w = int(rng.integers(max(24, n // 45), max(52, n // 8)))
            s = int(rng.integers(0, max(1, n - w)))
            y[s : s + w] = float(rng.uniform(-0.10, 0.10)) + rng.normal(0.0, 0.006, w).astype(np.float32)
    elif subtype == "bad_low_qrs_visibility":
        y = float(rng.uniform(0.01, 0.08)) * BUILD.moving_average(x, int(rng.choice([31, 51, 71]))) + lf + 0.45 * hf
    elif subtype == "bad_highfreq_detail_noise":
        y = float(rng.uniform(0.02, 0.12)) * x + 0.45 * lf
        y += float(rng.uniform(0.18, 0.42)) * np.sin(2.0 * np.pi * float(rng.uniform(34.0, 44.0)) * t + phase2).astype(np.float32)
        y += rng.normal(0.0, float(rng.uniform(0.015, 0.050)), n).astype(np.float32)
    elif subtype == "bad_detector_template_disagree":
        y = float(rng.uniform(0.04, 0.16)) * x + 0.60 * lf + 0.65 * hf
        y = 0.70 * y + 0.30 * BUILD.moving_average(y, int(rng.choice([9, 15, 21])))
    elif subtype == "bad_dense_right_island":
        y = float(rng.uniform(0.02, 0.12)) * x + 0.75 * lf + 0.90 * hf
        y += rng.normal(0.0, float(rng.uniform(0.008, 0.030)), n).astype(np.float32)
    elif subtype == "bad_other_boundary":
        y = float(rng.uniform(0.02, 0.14)) * x + 0.70 * lf + float(rng.uniform(0.20, 0.70)) * hf
    else:
        raise ValueError(f"unknown subtype {subtype}")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p = float(np.percentile(np.abs(y), 99))
    if p > 1e-4:
        y = y / max(1.0, p / 1.05)
    return y.astype(np.float32)


def dense_stochastic_bad_candidate(base: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    """Dense noise-like candidates for BUT bad that has strong global artifact.

    These are not pure unstructured white noise only; they mix broad-band noise,
    high-frequency content, low-frequency drift, and optional weak morphology so
    feature matching can select windows near the target BUT distribution.
    """

    x = BUILD.robust_norm(base)
    n = len(x)
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    white = rng.normal(0.0, 1.0, n).astype(np.float32)
    brown = np.cumsum(rng.normal(0.0, 0.05, n)).astype(np.float32)
    brown = BUILD.robust_norm(brown)
    hf_min, hf_max = (30.0, 44.0) if subtype == "bad_highfreq_detail_noise" else (12.0, 28.0)
    hf = np.sin(2.0 * np.pi * float(rng.uniform(hf_min, hf_max)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
    lf = np.sin(2.0 * np.pi * float(rng.uniform(0.06, 0.28)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
    y = (
        float(rng.uniform(0.04, 0.20)) * x
        + float(rng.uniform(0.40, 0.95)) * white
        + float(rng.uniform(0.08, 0.35)) * brown
        + float(rng.uniform(0.03, 0.28) if subtype == "bad_highfreq_detail_noise" else rng.uniform(0.01, 0.10)) * hf
        + float(rng.uniform(0.02, 0.22)) * lf
    )
    if subtype in {"bad_baseline_wander_lowfreq", "bad_other_boundary"}:
        y += float(rng.uniform(0.15, 0.55)) * lf
    if subtype == "bad_highfreq_detail_noise":
        y += float(rng.uniform(0.10, 0.40)) * hf
    elif subtype == "bad_dense_right_island":
        y += float(rng.uniform(0.02, 0.10)) * hf
    if subtype == "bad_contact_reset_flatline":
        for _ in range(int(rng.integers(1, 4))):
            w = int(rng.integers(max(24, n // 50), max(60, n // 7)))
            s = int(rng.integers(0, max(1, n - w)))
            y[s : s + w] *= float(rng.uniform(0.02, 0.25))
    if subtype == "bad_low_qrs_visibility":
        y = 0.75 * y + 0.25 * BUILD.moving_average(y, int(rng.choice([9, 15, 21])))
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p = float(np.percentile(np.abs(y), 99))
    if p > 1e-4:
        y = y / max(1.0, p / 1.20)
    return y.astype(np.float32)


def morphology_preserving_bad_candidate(base: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    """Bad candidate that preserves QRS morphology while corrupting quality.

    BUT bad is not always pure noise.  Many windows still contain a visible QRS
    complex, but baseline/contact/detail artifacts make measurement unreliable.
    This family keeps the source morphology and adds interpretable quality
    degradation without flooding the 30-45 Hz band.
    """

    x = BUILD.robust_norm(base)
    n = len(x)
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    centered = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    drift = np.sin(2.0 * np.pi * float(rng.uniform(0.04, 0.20)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
    mid = np.sin(2.0 * np.pi * float(rng.uniform(8.0, 22.0)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
    y = (
        float(rng.uniform(0.38, 0.82)) * x
        + float(rng.uniform(0.12, 0.42)) * drift
        + float(rng.uniform(0.01, 0.08)) * mid
        + rng.normal(0.0, float(rng.uniform(0.006, 0.030)), n).astype(np.float32)
    )

    if subtype == "bad_baseline_wander_lowfreq":
        y += float(rng.uniform(0.20, 0.65)) * drift
        y += float(rng.uniform(0.08, 0.28)) * centered
    elif subtype == "bad_contact_reset_flatline":
        for _ in range(int(rng.integers(1, 4))):
            w = int(rng.integers(max(24, n // 55), max(54, n // 8)))
            s = int(rng.integers(0, max(1, n - w)))
            level = float(rng.uniform(-0.14, 0.14))
            y[s : s + w] = level + rng.normal(0.0, 0.004, w).astype(np.float32)
    elif subtype == "bad_low_qrs_visibility":
        y = (
            float(rng.uniform(0.18, 0.38)) * BUILD.moving_average(x, int(rng.choice([11, 21, 31])))
            + float(rng.uniform(0.35, 0.70)) * drift
            + float(rng.uniform(0.02, 0.08)) * mid
        )
    elif subtype == "bad_highfreq_detail_noise":
        hf = np.sin(2.0 * np.pi * float(rng.uniform(30.0, 42.0)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        y += float(rng.uniform(0.08, 0.22)) * hf
        y += rng.normal(0.0, float(rng.uniform(0.020, 0.055)), n).astype(np.float32)
    elif subtype == "bad_detector_template_disagree":
        # Keep QRS visible but alter local template consistency using mild
        # time-varying gain and sparse local disturbances.
        gain = 1.0 + float(rng.uniform(0.15, 0.45)) * drift
        y = y * gain
        for _ in range(int(rng.integers(2, 5))):
            w = int(rng.integers(max(12, n // 100), max(28, n // 30)))
            s = int(rng.integers(0, max(1, n - w)))
            y[s : s + w] += float(rng.uniform(-0.18, 0.18))
    elif subtype == "bad_dense_right_island":
        y += float(rng.uniform(0.10, 0.34)) * drift
        y += float(rng.uniform(0.04, 0.14)) * mid
    elif subtype == "bad_other_boundary":
        y += float(rng.uniform(0.12, 0.40)) * drift
        if rng.random() < 0.5:
            y += float(rng.uniform(0.05, 0.14)) * mid
    else:
        raise ValueError(f"unknown subtype {subtype}")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p = float(np.percentile(np.abs(y), 99))
    if p > 1e-4:
        y = y / max(1.0, p / 1.10)
    return y.astype(np.float32)


def rhythm_disrupted_bad_candidate(base: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    """Candidate with unreliable peak count but modest high-frequency content."""

    x = BUILD.robust_norm(base)
    n = len(x)
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    y = float(rng.uniform(0.22, 0.58)) * x
    drift = np.sin(2.0 * np.pi * float(rng.uniform(0.04, 0.18)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
    y += float(rng.uniform(0.12, 0.42)) * drift
    y += rng.normal(0.0, float(rng.uniform(0.006, 0.022)), n).astype(np.float32)

    mode = "many" if subtype not in {"bad_low_qrs_visibility", "bad_contact_reset_flatline"} else str(rng.choice(["many", "few"], p=[0.45, 0.55]))
    if mode == "many":
        pos = float(rng.uniform(0.10, 0.35))
        while pos < 9.85:
            c = int(round(pos / 10.0 * (n - 1)))
            w = int(rng.integers(max(8, n // 180), max(18, n // 55)))
            lo = max(0, c - 3 * w)
            hi = min(n, c + 3 * w + 1)
            grid = np.arange(lo, hi, dtype=np.float32) - float(c)
            pulse = np.exp(-0.5 * (grid / max(1.0, float(w))) ** 2).astype(np.float32)
            y[lo:hi] += float(rng.uniform(0.25, 0.80)) * float(rng.choice([-1.0, 1.0])) * pulse
            pos += float(rng.uniform(0.25, 0.48))
    else:
        y = float(rng.uniform(0.05, 0.20)) * BUILD.moving_average(x, int(rng.choice([21, 31, 51]))) + float(rng.uniform(0.35, 0.80)) * drift
        for _ in range(int(rng.integers(1, 4))):
            c = int(rng.integers(0, n))
            w = int(rng.integers(max(18, n // 120), max(42, n // 35)))
            lo = max(0, c - 3 * w)
            hi = min(n, c + 3 * w + 1)
            grid = np.arange(lo, hi, dtype=np.float32) - float(c)
            pulse = np.exp(-0.5 * (grid / max(1.0, float(w))) ** 2).astype(np.float32)
            y[lo:hi] += float(rng.uniform(0.15, 0.55)) * float(rng.choice([-1.0, 1.0])) * pulse

    if subtype == "bad_contact_reset_flatline":
        for _ in range(int(rng.integers(1, 3))):
            w = int(rng.integers(max(24, n // 50), max(58, n // 8)))
            s = int(rng.integers(0, max(1, n - w)))
            y[s : s + w] = float(rng.uniform(-0.10, 0.10)) + rng.normal(0.0, 0.004, w).astype(np.float32)
    elif subtype == "bad_baseline_wander_lowfreq":
        y += float(rng.uniform(0.12, 0.45)) * drift

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p = float(np.percentile(np.abs(y), 99))
    if p > 1e-4:
        y = y / max(1.0, p / 1.12)
    return y.astype(np.float32)


def detector_fail_preserve_bad_candidate(base: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    """Preserve visible morphology while forcing unreliable peak-count evidence.

    This targets the BUT pattern where detector_agreement recomputes near zero
    without requiring the entire window to become high-frequency white noise.
    It adds many broad low/mid-frequency pseudo-peaks or a few oversized wide
    pulses, then lets feature matching keep only candidates near the target.
    """

    x = BUILD.robust_norm(base)
    n = len(x)
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    drift = np.sin(2.0 * np.pi * float(rng.uniform(0.04, 0.18)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
    y = float(rng.uniform(0.36, 0.72)) * x + float(rng.uniform(0.08, 0.35)) * drift
    y += rng.normal(0.0, float(rng.uniform(0.004, 0.020)), n).astype(np.float32)

    if ACTIVE_VERSION == "v20":
        mode = "too_many"
    elif subtype in {"bad_contact_reset_flatline", "bad_low_qrs_visibility"}:
        mode = str(rng.choice(["too_many", "too_few"], p=[0.45, 0.55]))
    else:
        mode = str(rng.choice(["too_many", "too_few"], p=[0.72, 0.28]))

    if mode == "too_many":
        pos = float(rng.uniform(0.04, 0.16))
        while pos < 9.95:
            c = int(round(pos / 10.0 * (n - 1)))
            # Wide enough to avoid a pure high-frequency signature, narrow enough
            # to create many local maxima for the detector.
            if ACTIVE_VERSION == "v20":
                w = int(rng.integers(max(6, n // 240), max(14, n // 105)))
            else:
                w = int(rng.integers(max(5, n // 260), max(13, n // 90)))
            lo = max(0, c - 4 * w)
            hi = min(n, c + 4 * w + 1)
            grid = np.arange(lo, hi, dtype=np.float32) - float(c)
            pulse = np.exp(-0.5 * (grid / max(1.0, float(w))) ** 2).astype(np.float32)
            amp = float(rng.uniform(0.18, 0.46) if ACTIVE_VERSION == "v20" else rng.uniform(0.20, 0.58))
            y[lo:hi] += amp * float(rng.choice([-1.0, 1.0])) * pulse
            pos += float(rng.uniform(0.32, 0.44) if ACTIVE_VERSION == "v20" else rng.uniform(0.105, 0.185))
    else:
        y = float(rng.uniform(0.08, 0.26)) * BUILD.moving_average(x, int(rng.choice([21, 31, 51])))
        y += float(rng.uniform(0.35, 0.75)) * drift
        for _ in range(int(rng.integers(1, 3))):
            c = int(rng.integers(0, n))
            w = int(rng.integers(max(36, n // 70), max(88, n // 18)))
            lo = max(0, c - 3 * w)
            hi = min(n, c + 3 * w + 1)
            grid = np.arange(lo, hi, dtype=np.float32) - float(c)
            pulse = np.exp(-0.5 * (grid / max(1.0, float(w))) ** 2).astype(np.float32)
            y[lo:hi] += float(rng.uniform(0.20, 0.65)) * float(rng.choice([-1.0, 1.0])) * pulse

    if subtype == "bad_baseline_wander_lowfreq":
        y += float(rng.uniform(0.12, 0.42)) * drift
    elif subtype == "bad_contact_reset_flatline" and ACTIVE_VERSION != "v20":
        for _ in range(int(rng.integers(1, 3))):
            w = int(rng.integers(max(20, n // 60), max(54, n // 9)))
            s = int(rng.integers(0, max(1, n - w)))
            y[s : s + w] = float(rng.uniform(-0.08, 0.08)) + rng.normal(0.0, 0.004, w).astype(np.float32)
    elif subtype == "bad_highfreq_detail_noise":
        hf = np.sin(2.0 * np.pi * float(rng.uniform(28.0, 38.0)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        y += float(rng.uniform(0.04, 0.12)) * hf

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p = float(np.percentile(np.abs(y), 99))
    if p > 1e-4:
        y = y / max(1.0, p / 1.12)
    return y.astype(np.float32)


def generate_featurematched_bad(
    subtype: str,
    count: int,
    source_pool_x: np.ndarray,
    rng: np.random.Generator,
    stats: dict[str, dict[str, float]],
    pool_multiplier: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pool_n = max(int(count) * int(pool_multiplier), int(count) + 64)
    signals = []
    base_idx = []
    cand_variants = []
    if ACTIVE_VERSION == "v20":
        variants = [
            "detector_fail_preserve",
            "morph_preserving",
            "dense_stochastic",
            "rhythm_disrupted",
            "smooth",
            "v13_softmatch",
        ]
        probs = [0.50, 0.25, 0.12, 0.06, 0.04, 0.03]
    elif ACTIVE_VERSION == "v19":
        variants = [
            "detector_fail_preserve",
            "morph_preserving",
            "dense_stochastic",
            "rhythm_disrupted",
            "smooth",
            "v13_softmatch",
            "v12",
        ]
        probs = [0.38, 0.25, 0.16, 0.11, 0.06, 0.03, 0.01]
    elif ACTIVE_VERSION == "v18":
        variants = ["rhythm_disrupted", "morph_preserving", "dense_stochastic", "smooth", "v13_softmatch", "v12"]
        probs = [0.38, 0.36, 0.12, 0.08, 0.04, 0.02]
    elif ACTIVE_VERSION == "v17":
        variants = ["morph_preserving", "dense_stochastic", "smooth", "v13_softmatch", "v12"]
        probs = [0.54, 0.22, 0.14, 0.06, 0.04]
    else:
        variants = ["dense_stochastic", "smooth", "v13_softmatch", "v12"]
        probs = [0.55, 0.20, 0.15, 0.10]
    for _ in range(pool_n):
        src_i = int(rng.integers(0, len(source_pool_x)))
        variant = str(rng.choice(variants, p=probs))
        if variant == "detector_fail_preserve":
            y = detector_fail_preserve_bad_candidate(source_pool_x[src_i], subtype, rng)
        elif variant == "rhythm_disrupted":
            y = rhythm_disrupted_bad_candidate(source_pool_x[src_i], subtype, rng)
        elif variant == "morph_preserving":
            y = morphology_preserving_bad_candidate(source_pool_x[src_i], subtype, rng)
        elif variant == "smooth":
            y = smooth_bad_candidate(source_pool_x[src_i], subtype, rng)
        elif variant == "dense_stochastic":
            y = dense_stochastic_bad_candidate(source_pool_x[src_i], subtype, rng)
        else:
            y = BUILD.generate_bad_waveform(source_pool_x[src_i], subtype, rng, variant=variant)
        signals.append(y)
        base_idx.append(src_i)
        cand_variants.append(variant)
    cand_x = np.stack(signals, axis=0).astype(np.float32)
    feat = FEATURES.compute_primitives(cand_x)
    feat["candidate_variant"] = cand_variants
    feat["score"] = score_candidates(feat, subtype, stats)
    feat["base_pool_idx"] = np.asarray(base_idx, dtype=np.int64)
    order = np.argsort(feat["score"].to_numpy(dtype=np.float32), kind="mergesort")
    take = order[: int(count)]
    return cand_x[take], np.asarray(base_idx, dtype=np.int64)[take], feat.iloc[take].reset_index(drop=True)


def recompute_features(signals: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    return BUILD.recompute_features(signals, frame)


def make_protocol(seed: int, total_bad: int, pool_multiplier: int, target_mode: str, version: str) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    protocol_name = f"protocol_{version}_pc{int(total_bad)}_s{int(seed)}"
    out_protocol = OUT_DIR / protocol_name
    out_protocol.mkdir(parents=True, exist_ok=True)

    base_atlas = pd.read_csv(BASE_PROTOCOL / "original_region_atlas.csv")
    base_x = np.load(BASE_PROTOCOL / "signals.npz")["X"].astype(np.float32)
    if base_x.ndim == 3:
        base_x = base_x[:, 0, :]
    but = pd.read_csv(BUT_REFERENCE_PROTOCOL / "original_region_atlas.csv")
    stats, stat_rows = target_stats_from_protocol(BUT_REFERENCE_PROTOCOL) if target_mode == "recomputed" else target_stats(but)

    keep_nonbad = base_atlas.loc[base_atlas["class_name"].astype(str).isin(["good", "medium"])].copy()
    nonbad_x = base_x[keep_nonbad["idx"].astype(int).to_numpy()]
    source_pool_x = base_x[base_atlas.index.to_numpy()]
    target_counts = BUILD.target_bad_counts(but, total_bad)
    base_bad_template = base_atlas.loc[base_atlas["class_name"].astype(str).eq("bad")].iloc[0].copy()

    bad_rows: list[pd.Series] = []
    bad_signals: list[np.ndarray] = []
    selected_rows = []
    global_bad_i = 0
    for subtype, count in target_counts.items():
        cand_x, base_idx, selected_feat = generate_featurematched_bad(
            subtype, int(count), source_pool_x, rng, stats, int(pool_multiplier)
        )
        per_split = BUILD.split_counts(int(count), rng)
        splits = []
        for split, n in per_split.items():
            splits.extend([split] * int(n))
        rng.shuffle(splits)
        for j in range(int(count)):
            row = base_bad_template.copy()
            split = str(splits[j])
            row["split"] = split
            row["class_name"] = "bad"
            row["y"] = 2
            row["source_idx"] = int(20_000_000 + global_bad_i)
            row["record_id"] = f"ptbgen_{version}_{subtype}"
            row["subject_id"] = f"ptbgen_{version}_{subtype}"
            row["original_region"] = subtype
            row["ambiguous_type"] = subtype
            row["clean_policy"] = f"ptb_bad_subtype_featurematched_{version}"
            row["target_bad_subtype"] = subtype
            row["generated_bad_mode"] = subtype
            row["base_pool_idx"] = int(base_idx[j])
            bad_rows.append(row)
            bad_signals.append(cand_x[j])
            selected_rows.append(
                {
                    "subtype": subtype,
                    "split": split,
                    "selected_score": float(selected_feat.loc[j, "score"]),
                    "candidate_variant": str(selected_feat.loc[j, "candidate_variant"])
                    if "candidate_variant" in selected_feat.columns
                    else "",
                }
            )
            global_bad_i += 1

    new_x = np.concatenate([nonbad_x, np.stack(bad_signals, axis=0)], axis=0).astype(np.float32)
    new_frame = pd.concat([keep_nonbad, pd.DataFrame(bad_rows)], ignore_index=True)
    new_frame["idx"] = np.arange(len(new_frame), dtype=np.int64)
    new_frame["y"] = new_frame["class_name"].astype(str).map(CLASS_TO_INT).astype(int)
    new_frame = recompute_features(new_x, new_frame)
    subtype_labels = [SUB.SUBTYPES[i] for i in SUB.subtype_labels(new_frame)]
    new_frame["computed_subtype"] = subtype_labels
    bad_mask = new_frame["class_name"].astype(str).eq("bad")
    new_frame.loc[bad_mask, "computed_subtype"] = new_frame.loc[bad_mask, "target_bad_subtype"].astype(str)

    np.savez_compressed(long_path(out_protocol / "signals.npz"), X=new_x[:, None, :].astype(np.float32))
    write_csv(new_frame, out_protocol / "original_region_atlas.csv")
    write_csv(
        new_frame[["idx", "source_idx", "split", "class_name", "y", "record_id", "subject_id", "target_bad_subtype", "generated_bad_mode"]],
        out_protocol / "metadata.csv",
    )
    write_csv(new_frame.groupby(["split", "class_name"], dropna=False).size().reset_index(name="n"), out_protocol / "split_counts.csv")
    write_csv(
        new_frame.groupby(["split", "class_name", "computed_subtype"], dropna=False).size().reset_index(name="n"),
        out_protocol / "subtype_counts.csv",
    )
    selected_df = pd.DataFrame(selected_rows)
    write_csv(selected_df, out_protocol / "selected_candidate_scores.csv")
    write_csv(stat_rows, out_protocol / "but_bad_subtype_target_feature_stats.csv")
    summary = {
        "created_at": now(),
        "protocol": str(out_protocol),
        "base_protocol": str(BASE_PROTOCOL),
        "but_reference_protocol": str(BUT_REFERENCE_PROTOCOL),
        "seed": int(seed),
        "total_bad": int(total_bad),
        "pool_multiplier": int(pool_multiplier),
        "target_mode": str(target_mode),
        "target_bad_counts": target_counts,
        "match_features": MATCH_FEATURES,
        "input_contract": "PTB waveform generation; BUT subtype proportions and feature stats only, no BUT waveform copied",
    }
    write_text(out_protocol / "summary.json", json.dumps(summary, indent=2))
    return out_protocol, new_frame, selected_df, stat_rows


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    view = df.head(max_rows)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def plot_outputs(protocol: Path, frame: pd.DataFrame, selected: pd.DataFrame) -> None:
    x = np.load(protocol / "signals.npz")["X"][:, 0, :]
    subtype_counts = pd.read_csv(protocol / "subtype_counts.csv")
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    bad_counts = subtype_counts.loc[subtype_counts["class_name"].eq("bad")].groupby("computed_subtype")["n"].sum().sort_values(ascending=False)
    bad_counts.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_title("PTB v14 feature-matched bad subtype counts")
    ax.set_ylabel("windows")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(REPORT_OUT_DIR / "ptb_v14_bad_subtype_counts.png", dpi=220)
    plt.close(fig)

    bad = frame.loc[frame["class_name"].astype(str).eq("bad")].copy()
    fig, axes = plt.subplots(len(BAD_SUBTYPES), 4, figsize=(18, 18), sharex=True, constrained_layout=True)
    fs = 125.0
    rng = np.random.default_rng(20260621)
    for r, subtype in enumerate(BAD_SUBTYPES):
        sub = bad.loc[bad["target_bad_subtype"].astype(str).eq(subtype)]
        if sub.empty:
            continue
        take = sub.sample(n=min(4, len(sub)), random_state=int(rng.integers(0, 1_000_000)))
        for c, (_, row) in enumerate(take.iterrows()):
            ax = axes[r, c]
            sig = x[int(row["idx"])]
            t = np.arange(len(sig)) / fs
            smooth = BUILD.moving_average(sig, 41)
            ax.plot(t, sig, color="#a7adb5", lw=0.45, alpha=0.75)
            ax.plot(t, smooth, color="black", lw=0.9)
            ax.set_title(f"{subtype}\n{row['split']}", fontsize=8)
            ax.set_xlim(0, 10)
            ax.grid(alpha=0.12)
        axes[r, 0].set_ylabel(subtype.replace("bad_", "").replace("_", "\n"), fontsize=8)
    fig.suptitle("PTB v14 feature-matched bad subtype waveform examples", fontsize=15)
    fig.savefig(REPORT_OUT_DIR / "ptb_v14_bad_subtype_waveforms.png", dpi=220)
    plt.close(fig)

    but = pd.read_csv(BUT_REFERENCE_PROTOCOL / "original_region_atlas.csv")
    but_bad = but.loc[but["class_name"].astype(str).eq("bad")].copy()
    ptb_bad = bad.copy()
    rows = []
    for col in MATCH_FEATURES[:6]:
        for group, df in [("BUT keep bad", but_bad), ("PTB v14 bad", ptb_bad)]:
            rows.append({"feature": col, "group": group, "median": float(pd.to_numeric(df[col], errors="coerce").median())})
    med = pd.DataFrame(rows)
    med.to_csv(REPORT_OUT_DIR / "ptb_v14_vs_but_bad_feature_medians.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, col in zip(axes.ravel(), MATCH_FEATURES[:6]):
        data = [
            pd.to_numeric(but_bad[col], errors="coerce").dropna().to_numpy(),
            pd.to_numeric(ptb_bad[col], errors="coerce").dropna().to_numpy(),
        ]
        ax.boxplot(data, tick_labels=["BUT bad", "PTB v14 bad"], showfliers=False)
        ax.set_title(col)
        ax.grid(alpha=0.2)
    fig.suptitle("BUT bad vs PTB v14 feature-matched bad", fontsize=15)
    fig.savefig(REPORT_OUT_DIR / "ptb_v14_vs_but_bad_feature_distributions.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    selected.groupby("subtype")["selected_score"].median().sort_values(ascending=False).plot(kind="bar", ax=ax, color="#f58518")
    ax.set_title("PTB v14 selected candidate median feature-match score")
    ax.set_ylabel("lower is closer to BUT subtype")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(REPORT_OUT_DIR / "ptb_v14_selected_score_by_subtype.png", dpi=220)
    plt.close(fig)


def write_report(protocol: Path, frame: pd.DataFrame, selected: pd.DataFrame, version: str) -> None:
    split_counts = pd.read_csv(protocol / "split_counts.csv")
    subtype_counts = pd.read_csv(protocol / "subtype_counts.csv")
    med = pd.read_csv(REPORT_OUT_DIR / "ptb_v14_vs_but_bad_feature_medians.csv")
    lines = [
        f"# PTB Bad Subtype Feature-Matched Protocol {REPORT_OUT_DIR.name}",
        "",
        f"- Generated: {now()}",
        f"- Protocol: `{protocol}`",
        "- Good/medium rows: inherited from v11 PTB aligned protocol.",
        "- Bad rows: PTB-only generated candidates selected by interpretable feature distance to BUT subtype statistics.",
        "- v15 mode recomputes BUT target features from signals with the same extractor used for generated PTB rows.",
        "- BUT usage: subtype proportions and feature medians/IQRs only; no BUT waveform is copied.",
        "",
        "## Split Counts",
        "",
        md_table(split_counts),
        "",
        "## Subtype Counts",
        "",
        md_table(subtype_counts),
        "",
        "## Feature Median Sanity",
        "",
        md_table(med),
        "",
        "## Candidate Score Summary",
        "",
        md_table(selected.groupby("subtype")["selected_score"].describe().reset_index()),
        "",
        "## Figures",
        "",
        f"- Counts: `{REPORT_OUT_DIR / 'ptb_v14_bad_subtype_counts.png'}`",
        f"- Waveforms: `{REPORT_OUT_DIR / 'ptb_v14_bad_subtype_waveforms.png'}`",
        f"- Feature distributions: `{REPORT_OUT_DIR / 'ptb_v14_vs_but_bad_feature_distributions.png'}`",
        f"- Candidate scores: `{REPORT_OUT_DIR / 'ptb_v14_selected_score_by_subtype.png'}`",
    ]
    (REPORT_OUT_DIR / f"ptb_bad_subtype_featurematched_{version}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    global OUT_DIR, REPORT_OUT_DIR, ACTIVE_VERSION
    ACTIVE_VERSION = str(args.version)
    if str(args.version) in {"v15", "v16", "v17", "v18", "v19", "v20"}:
        OUT_DIR = ANALYSIS_DIR / f"event_xds_aligned_{args.version}_bad_subtype_featurematched"
        REPORT_OUT_DIR = REPORT_DIR / f"event_xds_aligned_{args.version}_bad_subtype_featurematched"
    ensure_dirs()
    version = str(args.version)
    protocol, frame, selected, _ = make_protocol(
        int(args.seed),
        int(args.total_bad),
        int(args.pool_multiplier),
        str(args.target_mode),
        version,
    )
    plot_outputs(protocol, frame, selected)
    write_report(protocol, frame, selected, version)
    print(protocol, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--total-bad", type=int, default=3000)
    parser.add_argument("--pool-multiplier", type=int, default=6)
    parser.add_argument("--target-mode", type=str, default="atlas", choices=["atlas", "recomputed"])
    parser.add_argument("--version", type=str, default="v14", choices=["v14", "v15", "v16", "v17", "v18", "v19", "v20"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

