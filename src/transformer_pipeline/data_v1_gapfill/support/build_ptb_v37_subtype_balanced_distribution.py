"""Build V37 subtype-balanced PTB synthetic protocol aligned to BUT.

V37 fixes the鍙ｅ緞 problem that earlier visualizations used drop-outlier BUT.
The target reference is the keep-outlier subtype-stratified BUT protocol.

Core idea:
1. Assign a small, interpretable subtype vocabulary for good/medium/bad.
2. Generate a large PTB-only candidate pool per subtype.
3. Recompute waveform-computable ECG quality features with one extractor.
4. Select candidates by subtype-conditional empirical matching to BUT
   train+val rows using nearest-neighbor herding in robust feature space.
5. Write a protocol compatible with the Event-Factorized SQI Conformer.

BUT waveforms are never copied. BUT test rows are never used for candidate
selection; they are only included in audit plots and natural-prior reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402

V21_BUILDER = SCRIPT_DIR / "build_ptb_v21_gm_featurematched.py"
V14_BUILDER = SCRIPT_DIR / "build_ptb_bad_subtype_featurematched_v14.py"
BASE_PROTOCOL = ANALYSIS_DIR / "event_xds_aligned_v11_buttrain_style_replay" / "protocol_ptb_buttrain_aligned_pc3000_s20260620"
DEFAULT_BUT_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_keep_outlier_subtype_stratified_seed20260621"
BUT_PROTOCOL = DEFAULT_BUT_PROTOCOL

VERSION = "v37_subtype_balanced_distribution"
OUT_DIR = ANALYSIS_DIR / f"event_xds_aligned_{VERSION}"
REPORT_OUT_DIR = REPORT_DIR / f"event_xds_aligned_{VERSION}"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CLASS_ORDER = ["good", "medium", "bad"]

GOOD_SUBTYPES = [
    "good_clean_core",
    "good_overlap_boundary",
    "good_isolated_low_purity",
    "good_mild_artifact_outlier",
    "good_hard_baseline_lowqrs",
]
MEDIUM_SUBTYPES = [
    "medium_clean_core",
    "medium_overlap_boundary",
    "medium_isolated_lowqrs",
    "medium_visible_qrs_detail",
    "medium_outlier_or_bad_boundary",
    "medium_hard_baseline_lowqrs",
]
BAD_SUBTYPES = [
    "bad_dense_right_island",
    "bad_detector_template_disagree",
    "bad_baseline_wander_lowfreq",
    "bad_contact_reset_flatline",
    "bad_low_qrs_visibility",
    "bad_highfreq_detail_noise",
    "bad_other_boundary",
]
SUBTYPES_BY_CLASS = {
    "good": GOOD_SUBTYPES,
    "medium": MEDIUM_SUBTYPES,
    "bad": BAD_SUBTYPES,
}

MATCH_FEATURES = [
    "raw_rms",
    "raw_ptp_p99_p01",
    "raw_diff_abs_p95",
    "rms",
    "std",
    "mean_abs",
    "ptp_p99_p01",
    "low_amp_ratio",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "baseline_step",
    "diff_abs_p95",
    "detail_instability",
    "qrs_band_ratio",
    "qrs_visibility",
    "qrs_prom_median",
    "qrs_prom_p90",
    "periodicity",
    "template_corr",
    "detector_agreement",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "band_0p3_1",
    "band_1_5",
    "band_5_15",
    "band_15_30",
    "band_30_45",
    "sqi_iSQI",
    "sqi_bSQI",
    "sqi_pSQI",
    "sqi_sSQI",
    "sqi_kSQI",
    "sqi_fSQI",
    "sqi_basSQI",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "sample_entropy_proxy",
    "amplitude_entropy",
    "wavelet_e0",
    "wavelet_e1",
    "wavelet_e2",
    "wavelet_e3",
    "wavelet_e4",
]

REP_FEATURES = [
    "pc1",
    "pc2",
    "pc3",
    "baseline_step",
    "flatline_ratio",
    "qrs_visibility",
    "qrs_band_ratio",
    "detector_agreement",
    "template_corr",
    "non_qrs_diff_p95",
    "band_30_45",
    "sqi_basSQI",
    "amplitude_entropy",
]

GM_MODE_MAP = {
    "good_clean_core": ["identity", "good_morph_smooth", "good_qrs_band"],
    "good_overlap_boundary": ["good_amp_baseline", "good_qrs_band", "good_v22_qrs_preserve_band"],
    "good_isolated_low_purity": ["good_v22_lowentropy_morph", "good_morph_smooth", "good_v22_qrs_preserve_band"],
    "good_mild_artifact_outlier": ["good_v22_highamp_baseline", "good_v22_amp_step", "good_amp_baseline"],
    "good_hard_baseline_lowqrs": ["good_v22_highamp_baseline", "good_v22_amp_step", "good_amp_baseline"],
    "medium_clean_core": ["medium_preserve", "identity"],
    "medium_overlap_boundary": ["medium_detail", "medium_v22_template_jitter", "medium_baseline"],
    "medium_isolated_lowqrs": ["medium_v22_lowdetector_detail", "medium_v22_template_jitter"],
    "medium_visible_qrs_detail": ["medium_detail", "medium_v22_template_jitter", "medium_baseline"],
    "medium_outlier_or_bad_boundary": ["medium_baseline", "medium_v22_lowdetector_detail", "medium_detail"],
    "medium_hard_baseline_lowqrs": ["medium_baseline", "medium_v22_lowdetector_detail", "medium_v22_template_jitter"],
}

BAD_VARIANTS = [
    "detector_fail_preserve",
    "morph_preserving",
    "dense_stochastic",
    "rhythm_disrupted",
    "smooth",
    "v13_softmatch",
]
BAD_VARIANT_PROBS = np.asarray([0.34, 0.25, 0.16, 0.10, 0.09, 0.06], dtype=np.float64)
BAD_VARIANT_PROBS = BAD_VARIANT_PROBS / BAD_VARIANT_PROBS.sum()
HARD_NONBAD_SUBTYPES = {"good_hard_baseline_lowqrs", "medium_hard_baseline_lowqrs"}
SOURCE_PREFILTER_FEATURES = [
    "raw_rms",
    "raw_ptp_p99_p01",
    "raw_diff_abs_p95",
    "qrs_visibility",
    "qrs_band_ratio",
    "qrs_prom_median",
    "qrs_prom_p90",
    "template_corr",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "sqi_basSQI",
    "non_qrs_diff_p95",
    "band_15_30",
    "band_30_45",
    "amplitude_entropy",
]

PALETTE = {
    "BUT train+val": "#2f5f9f",
    "BUT test": "#87a9d8",
    "PTB synthetic": "#cf7c1b",
}
BAD_FREQUENCY_PROFILE = "midband"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GM = load_module(V21_BUILDER, "gm_builder_for_v37")
BAD = load_module(V14_BUILDER, "bad_builder_for_v37")
FEATURES = GM.FEATURES
BUILD = BAD.BUILD


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def configure_version(version: str) -> None:
    global VERSION, OUT_DIR, REPORT_OUT_DIR
    VERSION = str(version)
    OUT_DIR = ANALYSIS_DIR / f"event_xds_aligned_{VERSION}"
    REPORT_OUT_DIR = REPORT_DIR / f"event_xds_aligned_{VERSION}"


def configure_base_protocol(path: str) -> None:
    global BASE_PROTOCOL
    BASE_PROTOCOL = Path(path)


def configure_bad_frequency_profile(profile: str) -> None:
    global BAD_FREQUENCY_PROFILE
    BAD_FREQUENCY_PROFILE = str(profile)


def configure_but_protocol(path: str) -> None:
    global BUT_PROTOCOL
    if str(path).strip():
        BUT_PROTOCOL = Path(path)


def bad_range(midband_range: tuple[float, float]) -> tuple[float, float]:
    if BAD_FREQUENCY_PROFILE == "legacy":
        return (18.0, 42.0)
    return midband_range


def version_starts(prefixes: tuple[str, ...]) -> bool:
    return any(VERSION.startswith(prefix) for prefix in prefixes)


def natural_nonbad_version() -> bool:
    return version_starts(("v69", "v70", "v71", "v72", "v73", "v74", "v75", "v76", "v78", "v79"))


def suppress_peak_train_version() -> bool:
    return version_starts(("v63", "v69", "v70", "v71", "v72", "v73", "v74", "v75", "v76", "v78", "v79"))


def natural_source_nonbad_version() -> bool:
    return version_starts(("v75", "v76", "v78", "v79"))


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


def setup_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, path_no_ext: Path, dpi: int = 260) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(long_path(path_no_ext.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(long_path(path_no_ext.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


def robust_trace(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    y = y - float(np.nanmedian(y))
    lo, hi = np.nanpercentile(y, [2, 98])
    scale = float(hi - lo)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (y / scale).astype(np.float32)


def recompute_primitives(x: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    prim = FEATURES.compute_primitives(arr)
    for col in MATCH_FEATURES:
        if col not in prim.columns:
            prim[col] = 0.0
    return prim


def recompute_full_features(signals: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    arr = np.asarray(signals, dtype=np.float32)
    full = BUILD.recompute_features(arr, frame)
    # BUILD.recompute_features and waveform primitives expose partly
    # overlapping SQI/geometry columns. Recompute primitive waveform facts for
    # both BUT and PTB so audit/generation never compares stale atlas columns
    # against missing synthetic columns filled with zero.
    prim = FEATURES.compute_primitives(arr)
    for col in prim.columns:
        full[col] = prim[col].to_numpy()
    for col in MATCH_FEATURES:
        if col not in full.columns:
            full[col] = 0.0
    # Raw envelope features must always be computed from the exact waveform
    # being audited/selected. Some upstream extractors expose similarly named
    # columns on different scales, which creates fake distribution gaps.
    return add_raw_waveform_stats(full, arr)


def add_raw_waveform_stats(frame: pd.DataFrame, signals: np.ndarray) -> pd.DataFrame:
    """Attach raw waveform envelope stats used by the waveform-only model.

    The full SQI feature extractor may normalize internally. The model sees
    the raw synthetic waveform channels, so distribution matching must also
    align raw RMS/PTP/local derivative envelopes.
    """

    out = frame.copy()
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    if "idx" in out.columns:
        idx = pd.to_numeric(out["idx"], errors="coerce").fillna(0).astype(int).to_numpy()
        if len(idx) == len(out) and idx.min(initial=0) >= 0 and idx.max(initial=0) < len(x):
            sig = x[idx]
        else:
            sig = x[: len(out)]
    else:
        sig = x[: len(out)]
    centered = sig - np.nanmedian(sig, axis=1, keepdims=True)
    out["raw_rms"] = np.sqrt(np.nanmean(centered * centered, axis=1)).astype(np.float32)
    out["raw_ptp_p99_p01"] = (
        np.nanpercentile(sig, 99, axis=1) - np.nanpercentile(sig, 1, axis=1)
    ).astype(np.float32)
    out["raw_diff_abs_p95"] = np.nanpercentile(np.abs(np.diff(sig, axis=1)), 95, axis=1).astype(np.float32)
    return out


def candidate_feature_frame(
    signals: np.ndarray,
    primitive_frame: pd.DataFrame,
    cls: str,
    use_full_features: bool,
) -> pd.DataFrame:
    """Return the feature table used for candidate selection.

Earlier V37-V43 builds selected candidates using waveform primitive
    features, then audited/trained on BUILD.recompute_features. V44 can use
    the same full feature extractor for BUT targets, PTB candidates, and final
    audit so the matching objective is not silently using a different feature
    scale.
    """

    frame = primitive_frame.copy().reset_index(drop=True)
    frame["idx"] = np.arange(len(frame), dtype=np.int64)
    frame["source_idx"] = np.arange(len(frame), dtype=np.int64)
    frame["split"] = "train"
    frame["class_name"] = str(cls)
    frame["y"] = int(CLASS_TO_INT[str(cls)])
    frame["record_id"] = f"candidate_{cls}"
    frame["subject_id"] = f"candidate_{cls}"
    if not bool(use_full_features):
        return add_raw_waveform_stats(frame, signals)
    meta_cols = list(frame.columns)
    full = recompute_full_features(np.asarray(signals, dtype=np.float32), frame)
    for col in meta_cols:
        if col not in full.columns:
            full[col] = frame[col].to_numpy()
    return add_raw_waveform_stats(full.reset_index(drop=True), signals)


def raw_or_feature_value(row: pd.Series, raw_col: str, feature_col: str, default: float = 0.0) -> float:
    raw = safe_row_value(row, raw_col, float("nan"))
    if np.isfinite(raw):
        return float(raw)
    return safe_row_value(row, feature_col, default)



def feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    cols: list[np.ndarray] = []
    for col in features:
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fill = float(vals.median()) if vals.notna().any() else 0.0
            cols.append(vals.fillna(fill).to_numpy(dtype=np.float32))
        else:
            cols.append(np.zeros(len(frame), dtype=np.float32))
    return np.stack(cols, axis=1).astype(np.float32)


def fit_robust_scaler(target: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = feature_matrix(target, features).astype(np.float64)
    center = np.nanmedian(x, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    scale = np.maximum(q75 - q25, np.nanstd(x, axis=0))
    scale = np.maximum(scale, 1e-4)
    return center.astype(np.float32), scale.astype(np.float32)


def zscore(frame: pd.DataFrame, features: list[str], center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = feature_matrix(frame, features)
    return ((x - center[None, :]) / scale[None, :]).astype(np.float32)


def assign_but_subtypes(frame: pd.DataFrame) -> pd.Series:
    cls = frame["class_name"].astype(str)
    region = frame.get("original_region", pd.Series([""] * len(frame))).fillna("").astype(str)
    amb = frame.get("ambiguous_type", pd.Series([""] * len(frame))).fillna("").astype(str)
    subtype = pd.Series("unknown", index=frame.index, dtype=object)

    good = cls.eq("good")
    baseline = pd.to_numeric(frame.get("baseline_step", pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
    qrs = pd.to_numeric(frame.get("qrs_visibility", pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
    det = pd.to_numeric(frame.get("detector_agreement", pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
    bas = pd.to_numeric(frame.get("sqi_basSQI", pd.Series([1.0] * len(frame))), errors="coerce").fillna(1.0)
    if good.any():
        good_hard = good & baseline.ge(float(baseline[good].quantile(0.70))) & (
            qrs.le(float(qrs[good].quantile(0.30)))
            | det.le(float(det[good].quantile(0.30)))
            | bas.le(float(bas[good].quantile(0.30)))
        )
    else:
        good_hard = pd.Series(False, index=frame.index)
    subtype.loc[good & region.eq("clean_core")] = "good_clean_core"
    subtype.loc[good & amb.eq("good_medium_boundary")] = "good_overlap_boundary"
    subtype.loc[good & amb.isin(["isolated_good", "good_medium_low_purity", "good_medium_negative_margin"])] = "good_isolated_low_purity"
    subtype.loc[good & region.eq("outlier_low_confidence")] = "good_mild_artifact_outlier"
    subtype.loc[good_hard] = "good_hard_baseline_lowqrs"
    subtype.loc[good & subtype.eq("unknown")] = "good_overlap_boundary"

    medium = cls.eq("medium")
    detail = pd.to_numeric(frame.get("non_qrs_diff_p95", pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
    med_qrs = float(qrs[medium].median()) if medium.any() else 0.0
    med_detail = float(detail[medium].median()) if medium.any() else 0.0
    if medium.any():
        medium_hard = medium & baseline.ge(float(baseline[medium].quantile(0.70))) & (
            qrs.le(float(qrs[medium].quantile(0.30)))
            | det.le(float(det[medium].quantile(0.30)))
            | bas.le(float(bas[medium].quantile(0.30)))
        )
    else:
        medium_hard = pd.Series(False, index=frame.index)
    subtype.loc[medium & region.eq("clean_core")] = "medium_clean_core"
    subtype.loc[medium & amb.eq("good_medium_boundary")] = "medium_overlap_boundary"
    subtype.loc[medium & amb.eq("isolated_medium")] = "medium_isolated_lowqrs"
    subtype.loc[medium & (qrs.ge(med_qrs) & detail.ge(med_detail))] = "medium_visible_qrs_detail"
    subtype.loc[medium & (region.eq("outlier_low_confidence") | region.eq("medium_bad_overlap") | amb.eq("medium_bad_boundary"))] = (
        "medium_outlier_or_bad_boundary"
    )
    subtype.loc[medium_hard] = "medium_hard_baseline_lowqrs"
    # Keep the explicit BUT isolated-medium bucket from being swallowed by the
    # broad outlier or hard-baseline buckets; this subtype is a visual target.
    subtype.loc[medium & amb.eq("isolated_medium")] = "medium_isolated_lowqrs"
    subtype.loc[medium & subtype.eq("unknown")] = "medium_overlap_boundary"

    bad = cls.eq("bad")
    if "subtype_for_split" in frame.columns:
        st = frame["subtype_for_split"].fillna("").astype(str)
        subtype.loc[bad & st.isin(BAD_SUBTYPES)] = st.loc[bad & st.isin(BAD_SUBTYPES)]
    subtype.loc[bad & subtype.eq("unknown")] = "bad_other_boundary"
    return subtype


def target_counts(total_per_class: int) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cls, subtypes in SUBTYPES_BY_CLASS.items():
        base = int(total_per_class) // len(subtypes)
        rem = int(total_per_class) - base * len(subtypes)
        counts = {s: base for s in subtypes}
        for s in subtypes[:rem]:
            counts[s] += 1
        out[cls] = counts
    return out


def split_assignments(count: int, rng: np.random.Generator) -> list[str]:
    n_train = int(round(0.70 * int(count)))
    n_val = int(round(0.15 * int(count)))
    n_test = int(count) - n_train - n_val
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng.shuffle(splits)
    return splits


def safe_row_value(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = float(row.get(col, default))
    except Exception:
        val = default
    if not np.isfinite(val):
        return float(default)
    return float(val)


def rescale_to_anchor(y: np.ndarray, anchor: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Match the subtype anchor amplitude envelope without copying BUT waveform."""

    out = np.asarray(y, dtype=np.float32).copy()
    out = out - float(np.nanmedian(out))
    c_rms = float(np.sqrt(np.mean(np.square(out)))) if out.size else 1.0
    c_ptp = float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)) if out.size else 1.0
    target_rms = max(raw_or_feature_value(anchor, "raw_rms", "rms", c_rms), 1e-5)
    target_ptp = max(raw_or_feature_value(anchor, "raw_ptp_p99_p01", "ptp_p99_p01", c_ptp), 1e-5)
    scale_rms = target_rms / max(c_rms, 1e-6)
    scale_ptp = target_ptp / max(c_ptp, 1e-6)
    mix = float(rng.uniform(0.55, 0.80))
    scale = mix * scale_rms + (1.0 - mix) * scale_ptp
    scale = float(np.clip(scale, 0.0015, 4.0))
    return (out * scale).astype(np.float32)


def add_lowfreq_component(n: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    f1 = float(rng.uniform(0.035, 0.22))
    f2 = float(rng.uniform(0.18, 0.65))
    phase1 = float(rng.uniform(-np.pi, np.pi))
    phase2 = float(rng.uniform(-np.pi, np.pi))
    drift = np.sin(2 * np.pi * f1 * t + phase1) + 0.35 * np.sin(2 * np.pi * f2 * t + phase2)
    return (float(amp) * drift).astype(np.float32)


def add_local_bursts(
    n: int,
    amp: float,
    rng: np.random.Generator,
    count: tuple[int, int] = (1, 4),
    freq_range: tuple[float, float] = (18.0, 42.0),
) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    n_bursts = int(rng.integers(count[0], count[1] + 1))
    for _ in range(n_bursts):
        width = int(rng.integers(max(12, n // 80), max(24, n // 8)))
        start = int(rng.integers(0, max(1, n - width)))
        local_n = min(width, n - start)
        t = np.linspace(0.0, local_n / 125.0, local_n, dtype=np.float32)
        freq = float(rng.uniform(float(freq_range[0]), float(freq_range[1])))
        carrier = np.sin(2.0 * np.pi * freq * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        window = np.hanning(local_n).astype(np.float32)
        out[start : start + local_n] += float(amp) * window * carrier
        out[start : start + local_n] += rng.normal(0.0, float(amp) * 0.25, local_n).astype(np.float32) * window
    return out


def suppress_excess_very_high_frequency(y: np.ndarray, target_band_30_45: float, rng: np.random.Generator) -> np.ndarray:
    """Reduce synthetic-only 30-45Hz dominance while preserving mid-band artifact texture."""

    x = np.asarray(y, dtype=np.float32)
    if target_band_30_45 > 0.12:
        return x
    low = BAD.BUILD.moving_average(x, int(rng.choice([3, 5, 7])))
    mid = x - BAD.BUILD.moving_average(x, int(rng.choice([19, 25, 31])))
    keep = float(np.clip(0.18 + 2.2 * target_band_30_45, 0.18, 0.55))
    # Keep some mid-frequency structure; remove the synthetic white-noise texture.
    out = (1.0 - keep) * low + keep * x + 0.18 * mid
    return out.astype(np.float32)


def attenuate_qrs_like_detail(y: np.ndarray, target_visibility: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(y, dtype=np.float32)
    win = int(rng.choice([15, 21, 31, 41]))
    low = BAD.BUILD.moving_average(x, win)
    detail = x - low
    # qrs_visibility in the BUT atlas is small for low-QRS/poor windows.
    gain = float(np.clip(0.20 + 1.25 * target_visibility, 0.05, 1.15))
    return (low + gain * detail).astype(np.float32)


def attenuate_qrs_hard_negative(y: np.ndarray, target_visibility: float, rng: np.random.Generator) -> np.ndarray:
    """Make a non-bad waveform QRS-poor without turning it into pure noise."""

    x = np.asarray(y, dtype=np.float32)
    win = int(rng.choice([21, 31, 41, 55]))
    low = BAD.BUILD.moving_average(x, win)
    detail = x - low
    gain = float(np.clip(0.04 + 0.55 * target_visibility, 0.025, 0.50))
    out = low + gain * detail
    # Keep a tiny amount of local morphology so the sample remains a boundary
    # non-bad example rather than an extreme bad outlier.
    out += add_local_bursts(
        x.shape[-1],
        amp=float(np.clip(0.025 + 0.025 * target_visibility, 0.012, 0.045)),
        rng=rng,
        count=(1, 2),
        freq_range=(7.0, 18.0),
    )
    return out.astype(np.float32)


def add_smooth_level_shifts(n: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Piecewise baseline drift with smooth transitions.

    This mimics long low-frequency/contact-like baseline movement without
    injecting the sharp, high-frequency pulses that make a window look like
    an extreme bad/noise sample.
    """

    n = int(n)
    if n <= 4:
        return np.zeros(max(n, 1), dtype=np.float32)
    n_segments = int(rng.integers(3, 7))
    cuts = np.sort(rng.choice(np.arange(max(8, n // 12), max(9, n - n // 12)), size=n_segments - 1, replace=False))
    edges = np.asarray([0, *cuts.tolist(), n], dtype=int)
    levels = np.cumsum(rng.normal(0.0, float(amp), size=n_segments).astype(np.float32))
    levels = levels - float(np.median(levels))
    out = np.zeros(n, dtype=np.float32)
    for i in range(n_segments):
        out[edges[i] : edges[i + 1]] = float(levels[i])
    smooth_width = int(rng.choice([31, 45, 61, 79, 95]))
    out = BAD.BUILD.moving_average(out, smooth_width)
    return out.astype(np.float32)


def add_soft_baseline_edges(n: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Baseline edges that raise baseline_step without creating contact loss.

    The BUT medium-hard windows often show readable beats riding on slow
    level changes. A tanh edge produces baseline movement visible to the
    primitive extractor while preserving non-flat waveform texture.
    """

    n = int(n)
    out = np.zeros(max(n, 1), dtype=np.float32)
    if n < 32 or float(amp) <= 0:
        return out
    for _ in range(int(rng.integers(1, 4))):
        center = int(rng.integers(max(16, n // 10), max(17, n - n // 10)))
        width = int(rng.integers(max(8, n // 160), max(14, n // 45)))
        level = float(rng.normal(0.0, float(amp)))
        t = (np.arange(n, dtype=np.float32) - float(center)) / max(float(width), 1.0)
        out += level * np.tanh(t).astype(np.float32)
    out = out - float(np.nanmedian(out))
    return out.astype(np.float32)


def add_visible_qrs_peak_train(y: np.ndarray, target_visibility: float, rng: np.random.Generator) -> np.ndarray:
    """Add a light, regular QRS-like peak train for visible-QRS medium rows."""

    out = np.asarray(y, dtype=np.float32).copy()
    n = out.shape[-1]
    if n < 64:
        return out
    abs_out = np.abs(out - float(np.nanmedian(out)))
    local_scale = float(np.nanpercentile(abs_out, 92)) + 1e-5
    strength = float(np.clip(0.18 + 0.10 * min(float(target_visibility), 2.0), 0.14, 0.42))
    amp = strength * local_scale
    count = int(np.clip(round(n / (125.0 * float(rng.uniform(0.76, 0.92)))), 9, 14))
    spacing = n / float(count + 1)
    polarity = 1.0 if rng.random() < 0.55 else -1.0
    for k in range(count):
        center = int(round((k + 1) * spacing + rng.normal(0.0, 0.045 * 125.0)))
        center = int(np.clip(center, 8, n - 9))
        width = int(rng.integers(5, 10))
        lo = max(0, center - width)
        hi = min(n, center + width + 1)
        t = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
        # Narrow positive/negative core with small shoulders is enough for the
        # primitive peak detector, without becoming an artificial spike storm.
        shape = np.exp(-0.5 * (t / 0.28) ** 2).astype(np.float32)
        shoulder = 0.20 * np.exp(-0.5 * ((np.abs(t) - 0.55) / 0.18) ** 2).astype(np.float32)
        out[lo:hi] += polarity * amp * (shape - shoulder)
    return out.astype(np.float32)


def make_lowqrs_baseline_boundary(
    y: np.ndarray,
    anchor: pd.Series,
    cls: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a non-bad hard boundary sample matching BUT false-bad morphology.

    In the unified extractor, these BUT rows usually still have visible QRS
    complexes. The hard part is baseline/detail morphology, not pure QRS
    erasure. Only anchors that are truly low-QRS get the aggressive path.
    """

    x = np.asarray(y, dtype=np.float32).reshape(-1)
    n = x.shape[-1]
    x = x - float(np.nanmedian(x))
    target_rms = max(raw_or_feature_value(anchor, "raw_rms", "rms", float(np.nanstd(x))), 1e-5)
    target_ptp = max(raw_or_feature_value(anchor, "raw_ptp_p99_p01", "ptp_p99_p01", float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1))), 1e-5)
    target_base = max(safe_row_value(anchor, "baseline_step", 0.0), 0.0)
    target_qrs = max(safe_row_value(anchor, "qrs_visibility", 0.0), 0.0)
    target_det = max(safe_row_value(anchor, "detector_agreement", 0.0), 0.0)
    target_bas = safe_row_value(anchor, "sqi_basSQI", 1.0)

    long = BAD.BUILD.moving_average(x, int(rng.choice([125, 155, 185])))
    ecg_detail = x - long
    mid = BAD.BUILD.moving_average(x - BAD.BUILD.moving_average(x, int(rng.choice([25, 35, 45]))), int(rng.choice([7, 9, 13])))
    low_qrs_anchor = target_qrs < 0.18 and target_det < 0.28
    if low_qrs_anchor:
        qrs_gain = float(np.clip(0.10 + 0.35 * target_qrs + 0.08 * target_det, 0.07, 0.32))
        base_ecg = 0.08 * long + qrs_gain * mid + 0.05 * ecg_detail
    elif cls == "medium":
        # BUT medium hard-boundary windows usually keep visible QRS energy;
        # the label comes from baseline/detail degradation around otherwise
        # readable beats. Keep those beat edges instead of making a low-QRS
        # bad-like sample.
        qrs_gain = float(np.clip(1.05 + 0.20 * min(target_qrs, 3.0) + 0.08 * target_det, 0.92, 1.68))
        base_ecg = qrs_gain * x + float(rng.uniform(0.10, 0.20)) * mid + float(rng.uniform(0.07, 0.13)) * ecg_detail
    else:
        qrs_gain = float(np.clip(0.70 + 0.12 * min(target_qrs, 2.0) + 0.10 * target_det, 0.62, 1.02))
        base_ecg = qrs_gain * x + float(rng.uniform(0.02, 0.08)) * mid + float(rng.uniform(0.02, 0.06)) * ecg_detail

    severity = float(np.clip((1.0 - target_bas) + 0.80 * target_base + max(0.0, 0.65 - target_det), 0.15, 2.4))
    if cls == "medium" and not low_qrs_anchor:
        baseline_amp = float(np.clip(0.11 * target_rms + 0.18 * target_ptp + 1.25 * target_base + 0.060 * severity, 0.025, 1.65))
    else:
        baseline_amp = float(np.clip(0.10 * target_rms + 0.14 * target_ptp + 0.90 * target_base + 0.070 * severity, 0.018, 1.45))
    baseline = add_smooth_level_shifts(n, baseline_amp, rng)
    baseline += add_lowfreq_component(n, 0.08 * target_rms + 0.42 * target_base + 0.035 * severity, rng)
    if cls == "medium" and not low_qrs_anchor:
        baseline += add_soft_baseline_edges(n, 0.35 * baseline_amp + 0.80 * target_base, rng)

    out = base_ecg + baseline
    # Subtle sensor wander/detail, not burst noise. This helps detector
    # disagreement without creating a bad_highfreq-like phenotype.
    wander = BAD.BUILD.moving_average(rng.normal(0.0, 1.0, n).astype(np.float32), int(rng.choice([41, 61, 81])))
    wander = wander / max(float(np.nanstd(wander)), 1e-6)
    out += float(np.clip(0.020 * target_rms + 0.030 * target_base, 0.002, 0.085)) * wander
    target_detail = max(raw_or_feature_value(anchor, "raw_diff_abs_p95", "non_qrs_diff_p95", 0.0), 0.0)
    if cls == "medium" and not low_qrs_anchor:
        out += add_local_bursts(
            n,
            amp=float(np.clip(0.045 * target_rms + 0.12 * target_detail + 0.012, 0.006, 0.090)),
            rng=rng,
            count=(2, 4),
            freq_range=(12.0, 30.0),
        )
        if not natural_nonbad_version():
            out = add_visible_qrs_peak_train(out, target_qrs, rng)
    elif target_detail > 0.035 and not low_qrs_anchor:
        out += add_local_bursts(
            n,
            amp=float(np.clip(0.05 * target_detail + 0.010 * target_rms, 0.003, 0.045)),
            rng=rng,
            count=(1, 2),
            freq_range=(9.0, 22.0),
        )
    out = out - float(np.nanmedian(out))

    # Match raw envelope with a conservative cap so the baseline dominates the
    # robust-z normalization, which is what lowers qrs_visibility.
    cur_rms = float(np.sqrt(np.mean(out * out))) if out.size else 1.0
    cur_ptp = float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)) if out.size else 1.0
    scale = 0.60 * (target_rms / max(cur_rms, 1e-6)) + 0.40 * (target_ptp / max(cur_ptp, 1e-6))
    upper_scale = 2.25 if cls == "medium" and not low_qrs_anchor else 1.85
    scale = float(np.clip(scale, 0.02, upper_scale))
    out = out * scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def apply_contact_or_reset(y: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32).copy()
    n = out.shape[-1]
    n_events = int(np.clip(round(2 + 6 * strength), 2, 7))
    for _ in range(n_events):
        width = int(rng.integers(max(16, n // 70), max(42, n // 6)))
        start = int(rng.integers(0, max(1, n - width)))
        stop = min(n, start + width)
        plateau = float(np.nanmedian(out[max(0, start - 24) : max(1, start)])) if start > 0 else float(np.nanmedian(out))
        mode = str(rng.choice(["plateau", "dropout", "flatline", "reset"], p=[0.30, 0.30, 0.25, 0.15]))
        if mode == "plateau":
            out[start:stop] = 0.15 * out[start:stop] + 0.85 * plateau
        elif mode == "dropout":
            out[start:stop] *= float(rng.uniform(0.0, 0.12))
        elif mode == "flatline":
            out[start:stop] = plateau + rng.normal(0.0, 0.004 * (np.nanstd(out) + 1e-5), stop - start).astype(np.float32)
        else:
            jump = float(rng.uniform(-0.8, 0.8)) * (np.nanpercentile(np.abs(out), 85) + 1e-5)
            out[start:stop] += jump
            out[stop:] -= 0.35 * jump
    return out.astype(np.float32)


def raw_diff95(y: np.ndarray) -> float:
    x = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size < 3:
        return 0.0
    return float(np.nanpercentile(np.abs(np.diff(x)), 95))


def limit_local_derivative_envelope(
    y: np.ndarray,
    target_diff: float,
    target_rms: float,
    target_ptp: float,
    rng: np.random.Generator,
    strength: float = 0.75,
) -> np.ndarray:
    """Compress needle-like local slopes toward the BUT subtype envelope.

    This keeps a waveform-only, physiology-compatible signal: the operation is
    equivalent to limiting implausibly sharp local transitions and restoring the
    gross RMS/PTP envelope, not injecting any external feature at inference.
    """

    out = np.asarray(y, dtype=np.float32).reshape(-1).copy()
    if out.size < 4:
        return out
    target_diff = max(float(target_diff), 0.0)
    target_rms = max(float(target_rms), 1e-5)
    target_ptp = max(float(target_ptp), 1e-5)
    cap = float(np.clip(1.08 * target_diff + 0.0045 * target_rms, 0.004, 0.16))
    for _ in range(4):
        cur = raw_diff95(out)
        if cur <= max(1.12 * target_diff, cap):
            break
        d = np.diff(out)
        d = np.clip(d, -cap, cap).astype(np.float32)
        rec = np.concatenate([[float(out[0])], float(out[0]) + np.cumsum(d, dtype=np.float32)]).astype(np.float32)
        rec = rec - float(np.nanmedian(rec))
        low = BAD.BUILD.moving_average(out, int(rng.choice([7, 9, 13])))
        out = (1.0 - strength) * out + 0.72 * strength * rec + 0.28 * strength * low
        out = out.astype(np.float32)
    out = out - float(np.nanmedian(out))
    cur_rms = float(np.sqrt(np.mean(out * out))) if out.size else 1.0
    cur_ptp = float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)) if out.size else 1.0
    scale = 0.72 * (target_rms / max(cur_rms, 1e-6)) + 0.28 * (target_ptp / max(cur_ptp, 1e-6))
    out = out * float(np.clip(scale, 0.05, 1.35))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def add_highpass_detail(y: np.ndarray, target_gap: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(y, dtype=np.float32).copy()
    n = x.shape[-1]
    noise = rng.normal(0.0, 1.0, n).astype(np.float32)
    noise = noise - BAD.BUILD.moving_average(noise, int(rng.choice([9, 13, 17])))
    scale = float(np.nanstd(noise))
    if not np.isfinite(scale) or scale < 1e-6:
        return x
    noise = noise / scale
    amp = float(np.clip(0.48 * target_gap, 0.004, 0.22))
    # Localize part of the detail so it behaves like natural artifact bursts
    # instead of global white-noise contamination.
    mask = np.zeros(n, dtype=np.float32)
    for _ in range(int(rng.integers(2, 5))):
        width = int(rng.integers(max(16, n // 80), max(32, n // 6)))
        start = int(rng.integers(0, max(1, n - width)))
        mask[start : start + width] = np.maximum(mask[start : start + width], np.hanning(width).astype(np.float32))
    mask = np.maximum(mask, float(rng.uniform(0.18, 0.34)))
    return (x + amp * mask * noise).astype(np.float32)


def match_raw_detail_envelope(y: np.ndarray, anchor: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Raise local derivative/detail without breaking the raw RMS/PTP envelope."""

    out = np.asarray(y, dtype=np.float32).copy()
    n = out.shape[-1]
    target = raw_or_feature_value(anchor, "raw_diff_abs_p95", "non_qrs_diff_p95", raw_diff95(out))
    target = max(float(target), 0.0)
    for _ in range(4):
        current = raw_diff95(out)
        if current >= 0.94 * target:
            break
        gap = float(np.clip(target - current, 0.0, 1.0))
        if gap <= 1e-4:
            break
        out += add_local_bursts(
            n,
            amp=float(np.clip(0.42 * gap, 0.004, 0.18)),
            rng=rng,
            count=(3, 7),
            freq_range=bad_range((12.0, 34.0)),
        )
        out = add_highpass_detail(out, gap, rng)
        # Re-match the gross envelope, but only partially so the local detail
        # does not get erased by amplitude normalization.
        gross = rescale_to_anchor(out, anchor, rng)
        out = (0.94 * out + 0.06 * gross).astype(np.float32)
    return out.astype(np.float32)


def make_saturated_lowband_bad(n: int, rng: np.random.Generator) -> np.ndarray:
    """Low-QRS, low-entropy block artifact for BUT highfreq-labeled bad rows.

    Under the unified extractor, the BUT rows carrying the old
    bad_highfreq_detail_noise subtype have very low 30-45Hz energy. They look
    like dense/saturated poor-quality blocks, not literal high-frequency white
    noise. A fast telegraph process captures that morphology better than a
    high-pass noise burst.
    """

    n = int(n)
    out = np.zeros(max(n, 1), dtype=np.float32)
    i = 0
    level = 0.0
    while i < n:
        width = int(rng.integers(1, 6))
        level += float(rng.normal(0.0, 1.0))
        level = float(np.clip(level, -2.2, 2.2))
        out[i : min(n, i + width)] = level
        i += width
    # A few dropout-like shelves keep it from becoming a stationary square
    # wave and lower QRS visibility without injecting white noise.
    for _ in range(int(rng.integers(2, 6))):
        width = int(rng.integers(max(10, n // 80), max(18, n // 10)))
        start = int(rng.integers(0, max(1, n - width)))
        out[start : start + width] *= float(rng.uniform(0.03, 0.28))
    out = BAD.BUILD.moving_average(out, int(rng.choice([1, 3])))
    out = out - float(np.nanmedian(out))
    scale = max(float(np.nanstd(out)), 1e-6)
    return (out / scale).astype(np.float32)


def force_bad_detector_failure(
    y: np.ndarray,
    anchor: pd.Series,
    subtype: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Push bad rows toward natural BUT detector/template failure.

    The BUT poor/bad buckets are often dense artifact: too many local peaks for
    a plausible 10s rhythm, not merely a smooth drift. This adds continuous
    mid/high-frequency texture after envelope matching so detector_agreement
    falls without relying on label-only atlas targets.
    """

    out = np.asarray(y, dtype=np.float32).copy()
    n = out.shape[-1]
    target_det = max(safe_row_value(anchor, "detector_agreement", 0.0), 0.0)
    if target_det > 0.30 and subtype not in {"bad_dense_right_island", "bad_highfreq_detail_noise"}:
        return out
    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        target_diff = max(safe_row_value(anchor, "non_qrs_diff_p95", raw_diff95(out)), 0.0)
    else:
        target_diff = max(raw_or_feature_value(anchor, "raw_diff_abs_p95", "non_qrs_diff_p95", raw_diff95(out)), 0.0)
    target_rms = max(raw_or_feature_value(anchor, "raw_rms", "rms", float(np.nanstd(out))), 1e-5)
    target_hf = max(safe_row_value(anchor, "band_30_45", 0.0), 0.0)

    if subtype in {"bad_contact_reset_flatline"}:
        # Contact/reset bad should stay event-like rather than dense-noise-like.
        return out

    stationary_subtypes = {
        "bad_dense_right_island",
        "bad_detector_template_disagree",
        "bad_baseline_wander_lowfreq",
        "bad_low_qrs_visibility",
        "bad_highfreq_detail_noise",
        "bad_other_boundary",
    }
    if subtype in stationary_subtypes:
        white = rng.normal(0.0, 1.0, n).astype(np.float32)
        mid = white - BAD.BUILD.moving_average(white, int(rng.choice([9, 11, 13, 15])))
        mid = BAD.BUILD.moving_average(mid, int(rng.choice([3, 5, 7])))

        # Natural BUT poor/bad windows are often dense, irregular mid-band
        # texture rather than synthetic 30-45Hz white noise. Smooth pulses add
        # many ambiguous detector peaks without dominating the very-high band.
        pulses = np.zeros(n, dtype=np.float32)
        n_pulses = int(rng.integers(28, 58))
        positions = rng.choice(np.arange(6, max(7, n - 6)), size=min(n_pulses, max(1, n - 12)), replace=False)
        for pos in positions.tolist():
            width = int(rng.integers(4, 14))
            start = max(0, int(pos) - width // 2)
            stop = min(n, start + width)
            pulse = np.hanning(max(2, stop - start)).astype(np.float32)
            sign = -1.0 if rng.random() < 0.5 else 1.0
            pulses[start:stop] += sign * pulse[: stop - start]
        pulses = BAD.BUILD.moving_average(pulses, int(rng.choice([3, 5])))

        slow = BAD.BUILD.moving_average(white, int(rng.choice([21, 31, 41])))
        white_hp = white - BAD.BUILD.moving_average(white, int(rng.choice([5, 7, 9])))
        if subtype == "bad_highfreq_detail_noise":
            texture = 0.70 * mid + 0.24 * pulses + 0.06 * white_hp
        elif subtype in {"bad_dense_right_island", "bad_detector_template_disagree", "bad_other_boundary"}:
            texture = 0.66 * mid + 0.24 * pulses + 0.10 * white_hp
        elif subtype in {"bad_baseline_wander_lowfreq", "bad_low_qrs_visibility"}:
            texture = 0.56 * mid + 0.22 * pulses + 0.08 * white_hp + 0.14 * slow
        else:
            texture = 0.64 * mid + 0.24 * pulses + 0.12 * white_hp
        texture = texture / max(float(np.nanstd(texture)), 1e-6)
        keep = 0.06 if subtype in {"bad_dense_right_island", "bad_detector_template_disagree", "bad_other_boundary"} else 0.10
        out = keep * out + texture
        if subtype == "bad_baseline_wander_lowfreq":
            out += add_lowfreq_component(n, 0.05 * target_rms + 0.08 * safe_row_value(anchor, "baseline_step", 0.0), rng)
        out = rescale_to_anchor(out, anchor, rng)
        out = match_raw_detail_envelope(out, anchor, rng)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    noise = rng.normal(0.0, 1.0, n).astype(np.float32)
    noise = noise - BAD.BUILD.moving_average(noise, int(rng.choice([7, 9, 11, 15])))
    noise = BAD.BUILD.moving_average(noise, int(rng.choice([1, 3, 5])))
    noise = noise / max(float(np.nanstd(noise)), 1e-6)

    # Add many short alternating pseudo-peaks. These are deliberately tiny in
    # raw amplitude but numerous enough to break R-peak count agreement.
    spikes = np.zeros(n, dtype=np.float32)
    n_spikes = int(rng.integers(40, 76))
    positions = rng.choice(np.arange(4, max(5, n - 5)), size=min(n_spikes, max(1, n - 10)), replace=False)
    for pos in positions.tolist():
        width = int(rng.integers(2, 7))
        start = max(0, int(pos) - width // 2)
        stop = min(n, start + width)
        pulse = np.hanning(max(2, stop - start)).astype(np.float32)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        spikes[start:stop] += sign * pulse[: stop - start]
    spikes = spikes / max(float(np.nanstd(spikes)), 1e-6)

    detail_amp = float(np.clip(0.50 * target_diff + 0.045 * target_rms + 0.045 * target_hf, 0.014, 0.34))
    if subtype in {"bad_dense_right_island", "bad_highfreq_detail_noise", "bad_other_boundary"}:
        out = 0.34 * out + detail_amp * (0.68 * noise + 0.32 * spikes)
    else:
        out = 0.56 * out + detail_amp * (0.50 * noise + 0.50 * spikes)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def naturalize_nonbad_morphology_v69(
    y: np.ndarray,
    anchor: pd.Series,
    cls: str,
    subtype: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reduce synthetic-only needle peaks/detail for natural BUT-like non-bad.

    Earlier good/medium generators could match some PCA targets while creating
    visually artificial R-spike trains. BUT good/medium rows usually keep QRS
    visible but not saturated, with slow natural baseline/amplitude variation
    and bounded non-QRS detail. This v69 post-conditioner uses only
    waveform-computable anchor scalars and never copies BUT waveforms.
    """

    out = np.asarray(y, dtype=np.float32).reshape(-1).copy()
    n = out.shape[-1]
    out = out - float(np.nanmedian(out))
    target_qrs = max(safe_row_value(anchor, "qrs_visibility", 0.0), 0.0)
    target_det = max(safe_row_value(anchor, "detector_agreement", 0.0), 0.0)
    target_bas = safe_row_value(anchor, "sqi_basSQI", 1.0)
    target_base = max(safe_row_value(anchor, "baseline_step", 0.0), 0.0)
    target_rms = max(raw_or_feature_value(anchor, "raw_rms", "rms", float(np.nanstd(out))), 1e-5)
    target_ptp = max(
        raw_or_feature_value(anchor, "raw_ptp_p99_p01", "ptp_p99_p01", float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1))),
        1e-5,
    )
    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        target_diff = max(safe_row_value(anchor, "non_qrs_diff_p95", raw_diff95(out)), 0.0)
    else:
        target_diff = max(raw_or_feature_value(anchor, "raw_diff_abs_p95", "non_qrs_diff_p95", raw_diff95(out)), 0.0)

    # Tame the tall, regular synthetic peak train first. A tiny moving-average
    # blend keeps QRS morphology visible but removes the "barcode" appearance.
    smooth = BAD.BUILD.moving_average(out, int(rng.choice([3, 5, 7])))
    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        if cls == "good":
            smooth_mix = float(np.clip(0.78 + 0.08 * max(0.0, 1.0 - target_bas), 0.72, 0.90))
        else:
            smooth_mix = float(np.clip(0.62 + 0.10 * max(0.0, 1.0 - target_bas), 0.56, 0.80))
    elif version_starts(("v70", "v71")):
        if cls == "good":
            smooth_mix = float(np.clip(0.58 + 0.12 * max(0.0, 1.0 - target_bas), 0.52, 0.76))
        else:
            smooth_mix = float(np.clip(0.44 + 0.10 * max(0.0, 1.0 - target_bas), 0.36, 0.62))
    elif cls == "good":
        smooth_mix = float(np.clip(0.28 + 0.10 * max(0.0, 1.0 - target_bas), 0.24, 0.42))
    else:
        smooth_mix = float(np.clip(0.18 + 0.08 * max(0.0, 1.0 - target_bas), 0.14, 0.32))
    out = (1.0 - smooth_mix) * out + smooth_mix * smooth

    # If the BUT anchor does not have saturated QRS visibility, suppress excess
    # detail instead of adding more peak train. Medium can retain more detail.
    low = BAD.BUILD.moving_average(out, int(rng.choice([15, 21, 31])))
    detail = out - low
    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        if cls == "good":
            detail_gain = float(np.clip(0.11 + 0.13 * min(target_qrs, 1.2) + 0.03 * target_det, 0.09, 0.38))
        else:
            detail_gain = float(np.clip(0.18 + 0.18 * min(target_qrs, 1.4) + 0.04 * target_det, 0.14, 0.55))
    elif version_starts(("v70", "v71")):
        if cls == "good":
            detail_gain = float(np.clip(0.20 + 0.28 * min(target_qrs, 1.2) + 0.05 * target_det, 0.18, 0.60))
        else:
            detail_gain = float(np.clip(0.34 + 0.26 * min(target_qrs, 1.4) + 0.05 * target_det, 0.28, 0.78))
    elif cls == "good":
        detail_gain = float(np.clip(0.52 + 0.22 * min(target_qrs, 1.4) + 0.10 * target_det, 0.50, 0.92))
    else:
        detail_gain = float(np.clip(0.68 + 0.18 * min(target_qrs, 1.4) + 0.08 * target_det, 0.62, 1.05))
    out = low + detail_gain * detail

    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        limit = float(np.clip((0.28 if cls == "good" else 0.38) * target_ptp, 0.014, 1.65))
        out = limit * np.tanh(out / max(limit, 1e-6))
    elif version_starts(("v70", "v71")):
        # Soft-limit narrow R-spike outliers before envelope rescaling. This
        # broadens PTB morphology toward the BUT examples without injecting any
        # external waveform content.
        limit = float(np.clip((0.46 if cls == "good" else 0.62) * target_ptp, 0.020, 2.4))
        out = limit * np.tanh(out / max(limit, 1e-6))

    # Keep natural slow variation but avoid turning good into a medium/bad
    # baseline artifact. This is the morphology users see in BUT examples.
    if version_starts(("v72", "v73", "v74", "v75", "v76")) and cls == "good":
        base_amp = float(np.clip(0.026 * target_rms + 0.36 * target_base, 0.004, 0.090))
        burst_amp = float(np.clip(0.010 * target_diff + 0.002 * target_rms, 0.000, 0.010))
    elif version_starts(("v72", "v73", "v74", "v75", "v76")):
        base_amp = float(np.clip(0.045 * target_rms + 0.48 * target_base, 0.006, 0.145))
        burst_amp = float(np.clip(0.020 * target_diff + 0.004 * target_rms, 0.000, 0.018))
    elif cls == "good":
        base_amp = float(np.clip(0.020 * target_rms + 0.18 * target_base, 0.001, 0.055))
        burst_amp = float(np.clip(0.020 * target_diff + 0.004 * target_rms, 0.000, 0.020))
    else:
        base_amp = float(np.clip(0.035 * target_rms + 0.28 * target_base, 0.004, 0.105))
        burst_amp = float(np.clip(0.060 * target_diff + 0.010 * target_rms, 0.002, 0.055))
    out += add_lowfreq_component(n, base_amp, rng)
    if burst_amp > 1e-4:
        out += add_local_bursts(n, burst_amp, rng, count=(1, 2) if cls == "good" else (1, 3), freq_range=(8.0, 22.0))

    # Iteratively pull derivative/detail envelope toward the BUT anchor. This
    # directly targets the CDF gaps visible in non_qrs_diff_p95 and entropy.
    for _ in range(5 if version_starts(("v72", "v73", "v74", "v75", "v76")) else 3):
        cur = raw_diff95(out)
        tolerance = 1.02 if version_starts(("v72", "v73", "v74", "v75", "v76")) else (1.05 if version_starts(("v70", "v71")) else 1.20)
        slack = 0.003 if version_starts(("v72", "v73", "v74", "v75", "v76")) else (0.006 if version_starts(("v70", "v71")) else 0.015)
        if cur <= max(tolerance * target_diff, target_diff + slack):
            break
        low = BAD.BUILD.moving_average(out, int(rng.choice([5, 7, 9])))
        if version_starts(("v72", "v73", "v74", "v75", "v76")):
            out = 0.22 * out + 0.78 * low
        elif version_starts(("v70", "v71")):
            out = 0.48 * out + 0.52 * low
        else:
            out = 0.70 * out + 0.30 * low

    out = out - float(np.nanmedian(out))
    cur_rms = float(np.sqrt(np.mean(out * out))) if out.size else 1.0
    cur_ptp = float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)) if out.size else 1.0
    scale = 0.70 * (target_rms / max(cur_rms, 1e-6)) + 0.30 * (target_ptp / max(cur_ptp, 1e-6))
    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        upper = 0.82 if cls == "good" else 0.98
    else:
        upper = 0.95 if version_starts(("v70", "v71")) and cls == "good" else (1.12 if version_starts(("v70", "v71")) else 1.45)
    out = out * float(np.clip(scale, 0.025, upper))
    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        out = limit_local_derivative_envelope(
            out,
            target_diff=target_diff,
            target_rms=target_rms,
            target_ptp=target_ptp,
            rng=rng,
            strength=0.90 if cls == "good" else 0.82,
        )
        out += add_smooth_level_shifts(
            n,
            amp=float(np.clip((0.030 if cls == "good" else 0.045) * target_rms + 0.42 * target_base, 0.002, 0.120)),
            rng=rng,
        )
        out = limit_local_derivative_envelope(
            out,
            target_diff=target_diff,
            target_rms=target_rms,
            target_ptp=target_ptp,
            rng=rng,
            strength=0.70 if cls == "good" else 0.62,
        )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def target_condition_waveform(
    y: np.ndarray,
    cls: str,
    subtype: str,
    anchor: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mechanism-conditioned transformation toward a BUT train/val feature anchor.

    The anchor contributes only waveform-computable scalar targets. No BUT signal
    samples are copied into synthetic rows.
    """

    out = np.asarray(y, dtype=np.float32).copy()
    n = out.shape[-1]
    out = out - float(np.nanmedian(out))
    target_rms = max(raw_or_feature_value(anchor, "raw_rms", "rms", 0.05), 1e-5)
    target_base = max(safe_row_value(anchor, "baseline_step", 0.0), 0.0)
    if version_starts(("v72", "v73", "v74", "v75", "v76")):
        target_detail = max(safe_row_value(anchor, "non_qrs_diff_p95", 0.0), 0.0)
    else:
        target_detail = max(raw_or_feature_value(anchor, "raw_diff_abs_p95", "non_qrs_diff_p95", 0.0), 0.0)
    target_hf = max(safe_row_value(anchor, "band_30_45", 0.0), 0.0)
    target_flat = max(safe_row_value(anchor, "flatline_ratio", 0.0), 0.0)
    target_contact = max(safe_row_value(anchor, "contact_loss_win_ratio", 0.0), 0.0)
    target_qrs = max(safe_row_value(anchor, "qrs_visibility", 0.0), 0.0)
    target_det = max(safe_row_value(anchor, "detector_agreement", 0.0), 0.0)
    target_bas = safe_row_value(anchor, "sqi_basSQI", 1.0)

    out = rescale_to_anchor(out, anchor, rng)
    amp_unit = max(target_rms, float(np.nanpercentile(np.abs(out), 75)), 1e-4)

    if subtype in HARD_NONBAD_SUBTYPES:
        out = make_lowqrs_baseline_boundary(out, anchor, cls, rng)
    elif subtype == "medium_visible_qrs_detail":
        out = 1.05 * out + 0.08 * (out - BAD.BUILD.moving_average(out, int(rng.choice([9, 13, 17]))))
        out += add_lowfreq_component(n, 0.10 * amp_unit + 0.35 * target_base, rng)
        if not suppress_peak_train_version():
            out += add_soft_baseline_edges(n, 0.10 * amp_unit + 0.45 * target_base, rng)
        out += add_local_bursts(n, 0.10 * amp_unit + 0.24 * target_detail, rng, count=(2, 4), freq_range=(12.0, 30.0))
        if not suppress_peak_train_version():
            out = add_visible_qrs_peak_train(out, target_qrs, rng)
    elif subtype in {"good_mild_artifact_outlier", "medium_overlap_boundary"}:
        if subtype in {"good_mild_artifact_outlier", "good_hard_baseline_lowqrs"} and (
            target_qrs < 0.45 or target_det < 0.32 or target_bas < 0.86
        ):
            out = attenuate_qrs_hard_negative(out, target_qrs, rng)
        base_gain = 0.75 if subtype == "good_hard_baseline_lowqrs" else 0.45
        out += add_lowfreq_component(n, 0.15 * amp_unit + base_gain * target_base, rng)
        out += add_local_bursts(n, 0.10 * amp_unit + 0.18 * target_detail, rng, count=(1, 2), freq_range=(12.0, 28.0))
    elif subtype in {"medium_isolated_lowqrs", "medium_outlier_or_bad_boundary", "medium_hard_baseline_lowqrs"}:
        if VERSION.startswith("v63") and subtype == "medium_outlier_or_bad_boundary":
            medium_lowqrs_condition = target_qrs < 0.45 or target_det < 0.34 or target_bas < 0.86
        else:
            medium_lowqrs_condition = (
                target_qrs < (0.45 if subtype == "medium_isolated_lowqrs" else 0.32)
                or target_det < (0.34 if subtype == "medium_isolated_lowqrs" else 0.25)
            )
        if medium_lowqrs_condition:
            out = attenuate_qrs_hard_negative(out, target_qrs, rng)
        else:
            out = 1.02 * out + 0.08 * (out - BAD.BUILD.moving_average(out, int(rng.choice([9, 13, 17]))))
        base_gain = 0.82 if subtype == "medium_hard_baseline_lowqrs" else 0.55
        out += add_lowfreq_component(n, 0.22 * amp_unit + base_gain * target_base, rng)
        if (
            subtype in {"medium_outlier_or_bad_boundary", "medium_hard_baseline_lowqrs"}
            and not medium_lowqrs_condition
            and not (VERSION.startswith("v63") and subtype == "medium_outlier_or_bad_boundary")
        ):
            out += add_soft_baseline_edges(n, 0.12 * amp_unit + 0.70 * target_base, rng)
        out += add_local_bursts(n, 0.12 * amp_unit + 0.16 * target_detail, rng, count=(1, 3), freq_range=(10.0, 26.0))
        if not medium_lowqrs_condition and not (VERSION.startswith("v63") and subtype == "medium_outlier_or_bad_boundary") and not natural_nonbad_version():
            out = add_visible_qrs_peak_train(out, target_qrs, rng)
    elif subtype == "bad_baseline_wander_lowfreq":
        out = 0.55 * out + add_lowfreq_component(n, 0.50 * amp_unit + 0.90 * target_base, rng)
    elif subtype == "bad_contact_reset_flatline":
        out = 0.45 * out + add_lowfreq_component(n, 0.25 * amp_unit + 0.45 * target_base, rng)
        out = apply_contact_or_reset(out, float(np.clip(target_flat + target_contact + 0.15, 0.1, 1.0)), rng)
    elif subtype == "bad_low_qrs_visibility":
        out = attenuate_qrs_like_detail(out, min(target_qrs, 0.08), rng)
        out = 0.55 * out + add_lowfreq_component(n, 0.20 * amp_unit + 0.35 * target_base, rng)
        out += add_local_bursts(n, 0.08 * amp_unit + 0.10 * target_detail, rng, count=(1, 2), freq_range=bad_range((8.0, 24.0)))
    elif subtype == "bad_highfreq_detail_noise":
        out = 0.65 * out + add_lowfreq_component(n, 0.10 * amp_unit + 0.22 * target_base, rng)
        if BAD_FREQUENCY_PROFILE == "legacy":
            out += add_local_bursts(n, 0.20 * amp_unit + 0.35 * (target_detail + target_hf), rng, count=(2, 5), freq_range=(18.0, 42.0))
        else:
            out += add_local_bursts(n, 0.12 * amp_unit + 0.20 * (target_detail + target_hf), rng, count=(2, 4), freq_range=(16.0, 32.0))
    elif subtype == "bad_detector_template_disagree":
        out = 0.70 * out
        if target_det < 0.35 or rng.random() < 0.75:
            out += add_local_bursts(n, 0.10 * amp_unit + 0.10 * target_detail, rng, count=(2, 4), freq_range=bad_range((9.0, 25.0)))
            # Short pseudo-peaks create detector disagreement while keeping morphology visible.
            for _ in range(int(rng.integers(2, 6))):
                pos = int(rng.integers(8, max(9, n - 8)))
                width = int(rng.integers(3, 10))
                stop = min(n, pos + width)
                pulse = np.hanning(stop - pos).astype(np.float32)
                out[pos:stop] += float(rng.uniform(0.4, 1.2)) * amp_unit * pulse
    elif subtype == "bad_dense_right_island":
        out = 0.35 * out + add_lowfreq_component(n, 0.20 * amp_unit + 0.40 * target_base, rng)
        if BAD_FREQUENCY_PROFILE == "legacy":
            out += add_local_bursts(n, 0.18 * amp_unit + 0.18 * target_detail, rng, count=(2, 4), freq_range=(18.0, 42.0))
        else:
            out += add_local_bursts(n, 0.14 * amp_unit + 0.14 * target_detail, rng, count=(2, 4), freq_range=(10.0, 28.0))
    elif subtype == "bad_other_boundary":
        out = 0.55 * out + add_lowfreq_component(n, 0.18 * amp_unit + 0.35 * target_base, rng)
        out += add_local_bursts(n, 0.10 * amp_unit + 0.12 * target_detail, rng, count=(1, 3), freq_range=bad_range((10.0, 26.0)))

    if cls == "bad" and BAD_FREQUENCY_PROFILE != "legacy":
        out = suppress_excess_very_high_frequency(out, target_hf, rng)
    out = rescale_to_anchor(out, anchor, rng)
    if subtype in HARD_NONBAD_SUBTYPES:
        out = make_lowqrs_baseline_boundary(out, anchor, cls, rng)
    if natural_nonbad_version() and cls in {"good", "medium"}:
        out = naturalize_nonbad_morphology_v69(out, anchor, cls, subtype, rng)
    if version_starts(("v76",)) and cls in {"good", "medium"}:
        out = natural_sqi_pull_v76(out, anchor, cls, rng)
    if cls == "bad":
        out = match_raw_detail_envelope(out, anchor, rng)
        out = force_bad_detector_failure(out, anchor, subtype, rng)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def natural_sqi_pull_v76(
    y: np.ndarray,
    anchor: pd.Series,
    cls: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pull non-bad candidates toward BUT-like SQI without copying waveforms.

    The v75 audit showed a stable failure mode: PTB synthetic good/medium still
    has too much local derivative and saturated QRS evidence. This correction
    is deliberately simple and physiological: soften sharp local slopes, reduce
    relative QRS detail, add smooth level/flatline-like patches, then restore the
    gross RMS/PTP envelope from the BUT subtype anchor.
    """

    out = np.asarray(y, dtype=np.float32).reshape(-1).copy()
    if out.size < 16:
        return out
    n = out.shape[-1]
    out = out - float(np.nanmedian(out))
    target_rms = max(raw_or_feature_value(anchor, "raw_rms", "rms", float(np.nanstd(out))), 1e-5)
    target_ptp = max(
        raw_or_feature_value(anchor, "raw_ptp_p99_p01", "ptp_p99_p01", float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1))),
        1e-5,
    )
    target_diff = max(safe_row_value(anchor, "non_qrs_diff_p95", raw_diff95(out)), 0.0)
    target_qrs = max(safe_row_value(anchor, "qrs_visibility", 0.0), 0.0)
    target_base = max(safe_row_value(anchor, "baseline_step", 0.0), 0.0)
    target_flat = max(safe_row_value(anchor, "flatline_ratio", 0.0), 0.0)

    # Reduce QRS/detail saturation first; this targets qrs_visibility and the
    # downstream non-QRS detail mask that fails when peaks are too sharp.
    out = attenuate_qrs_like_detail(out, 0.46 * target_qrs, rng)
    for _ in range(2 if cls == "good" else 3):
        low = BAD.BUILD.moving_average(out, int(rng.choice([5, 7, 9, 13])))
        mix = 0.58 if cls == "good" else 0.66
        out = ((1.0 - mix) * out + mix * low).astype(np.float32)

    out = limit_local_derivative_envelope(
        out,
        target_diff=float(np.clip(0.55 * target_diff, 0.002, 0.18)),
        target_rms=target_rms,
        target_ptp=target_ptp,
        rng=rng,
        strength=0.96 if cls == "good" else 0.93,
    )

    # BUT good/medium windows often have smoother baseline/level structure than
    # the PTB synthetic carrier. Add it gently, without high-frequency pulses.
    out += add_smooth_level_shifts(
        n,
        amp=float(np.clip((0.012 if cls == "good" else 0.018) * target_rms + 0.34 * target_base, 0.0015, 0.075)),
        rng=rng,
    )
    if target_flat > 0.02:
        for _ in range(int(rng.integers(1, 3))):
            width = int(rng.integers(max(12, n // 14), max(18, n // 5)))
            start = int(rng.integers(0, max(1, n - width)))
            stop = min(n, start + width)
            local = BAD.BUILD.moving_average(out, int(rng.choice([17, 25, 33])))
            blend = float(np.clip(0.18 + 0.9 * target_flat, 0.18, 0.55))
            out[start:stop] = ((1.0 - blend) * out[start:stop] + blend * local[start:stop]).astype(np.float32)

    out = out - float(np.nanmedian(out))
    cur_rms = float(np.sqrt(np.mean(out * out))) if out.size else 1.0
    cur_ptp = float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)) if out.size else 1.0
    scale = 0.68 * (target_rms / max(cur_rms, 1e-6)) + 0.32 * (target_ptp / max(cur_ptp, 1e-6))
    out = out * float(np.clip(scale, 0.04, 0.95 if cls == "good" else 1.05))
    out = limit_local_derivative_envelope(
        out,
        target_diff=float(np.clip(0.70 * target_diff, 0.002, 0.22)),
        target_rms=target_rms,
        target_ptp=target_ptp,
        rng=rng,
        strength=0.82 if cls == "good" else 0.78,
    )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_anchor(target: pd.DataFrame, rng: np.random.Generator) -> pd.Series | None:
    if target.empty:
        return None
    return target.iloc[int(rng.integers(0, len(target)))]


def source_conditioned_pool(
    source_pool: np.ndarray,
    source_feat: pd.DataFrame,
    target: pd.DataFrame,
    cls: str,
    subtype: str,
    pool_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Filter PTB source rows toward BUT subtype morphology before generation.

    V68-V70 showed that post-conditioning alone cannot remove all synthetic
    needle-peak morphology. V71 therefore constrains the source ECG pool using
    waveform-computable BUT train+val features before any augmentation is
    applied. BUT test rows are never used here.
    """

    audit: dict[str, Any] = {
        "source_filter_enabled": False,
        "source_pool_before": int(len(source_pool)),
        "source_pool_after": int(len(source_pool)),
        "source_score_median": float("nan"),
        "source_score_p90": float("nan"),
    }
    if not VERSION.startswith("v71") or target.empty or len(source_pool) < 16 or len(source_feat) != len(source_pool):
        return source_pool, source_feat.reset_index(drop=True), audit

    features = [c for c in SOURCE_PREFILTER_FEATURES if c in target.columns or c in source_feat.columns]
    if len(features) < 4:
        return source_pool, source_feat.reset_index(drop=True), audit

    center, scale = fit_robust_scaler(target, features)
    target_z = zscore(target, features, center, scale)
    source_z = zscore(source_feat, features, center, scale)
    target_med = np.nanmedian(target_z, axis=0)
    score = np.sqrt(np.nanmean(np.square(source_z - target_med[None, :]), axis=1))

    # Penalize the exact failure modes visible in the v68-v70 audits: saturated
    # QRS visibility, excessive local derivative/detail, and too-clean baseline
    # spectra for natural BUT good/medium.
    target_q75 = target[features].apply(pd.to_numeric, errors="coerce").quantile(0.75)
    target_q25 = target[features].apply(pd.to_numeric, errors="coerce").quantile(0.25)
    source_num = source_feat[features].apply(pd.to_numeric, errors="coerce")
    feature_index = {name: i for i, name in enumerate(features)}

    def asym_penalty(name: str, direction: str, weight: float) -> np.ndarray:
        if name not in feature_index:
            return np.zeros(len(source_feat), dtype=np.float64)
        vals = source_num[name].fillna(float(target_q75.get(name, 0.0))).to_numpy(dtype=np.float64)
        width = float(scale[feature_index[name]])
        if direction == "above":
            edge = float(target_q75.get(name, np.nan))
            raw = np.maximum(0.0, vals - edge) / max(width, 1e-4)
        else:
            edge = float(target_q25.get(name, np.nan))
            raw = np.maximum(0.0, edge - vals) / max(width, 1e-4)
        return float(weight) * raw

    if cls in {"good", "medium"}:
        score = score + asym_penalty("qrs_visibility", "above", 1.60)
        score = score + asym_penalty("qrs_band_ratio", "above", 0.90)
        score = score + asym_penalty("non_qrs_diff_p95", "above", 1.10)
        score = score + asym_penalty("raw_diff_abs_p95", "above", 0.90)
        score = score + asym_penalty("sqi_basSQI", "below", 0.70)
        if "hard_baseline_lowqrs" in subtype or "isolated_lowqrs" in subtype:
            score = score + asym_penalty("baseline_step", "below", 0.50)
            score = score + asym_penalty("detector_agreement", "above", 0.45)
    elif cls == "bad":
        score = score + asym_penalty("band_30_45", "above", 0.65)
        score = score + asym_penalty("qrs_visibility", "above", 0.45)
        if "baseline_wander" in subtype:
            score = score + asym_penalty("baseline_step", "below", 0.80)
        if "low_qrs" in subtype or "detector" in subtype:
            score = score + asym_penalty("detector_agreement", "above", 0.80)

    score = np.nan_to_num(score, nan=np.inf, posinf=np.inf, neginf=np.inf)
    order = np.argsort(score)
    # Keep enough source diversity for subtype-balanced generation while
    # removing obviously mismatched PTB morphology.
    keep_n = int(max(96, min(len(order), min(1200, max(160, int(math.sqrt(max(pool_n, 1)) * 10))))))
    keep = order[:keep_n]
    if len(keep) < min(16, len(order)):
        keep = order[: min(len(order), 16)]
    filtered_x = np.asarray(source_pool, dtype=np.float32)[keep]
    filtered_feat = source_feat.iloc[keep].reset_index(drop=True).copy()
    audit.update(
        {
            "source_filter_enabled": True,
            "source_pool_after": int(len(filtered_x)),
            "source_score_median": float(np.nanmedian(score[keep])),
            "source_score_p90": float(np.nanpercentile(score[keep], 90)),
        }
    )
    return filtered_x, filtered_feat, audit


def generate_gm_candidates(
    source_pool: np.ndarray,
    cls: str,
    subtype: str,
    pool_n: int,
    rng: np.random.Generator,
    target: pd.DataFrame | None = None,
    target_condition: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    modes = GM_MODE_MAP[subtype]
    signals: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for i in range(int(pool_n)):
        src_i = int(rng.integers(0, len(source_pool)))
        mode = str(rng.choice(modes))
        if natural_source_nonbad_version():
            y = np.asarray(source_pool[src_i], dtype=np.float32).reshape(-1).copy()
            mode = f"natural_source_{mode}"
        else:
            y = GM.gm_candidate(source_pool[src_i], cls, mode, rng).astype(np.float32)
        anchor_idx = -1
        if target_condition and target is not None and not target.empty:
            anchor_idx = int(rng.integers(0, len(target)))
            y = target_condition_waveform(y, cls, subtype, target.iloc[anchor_idx], rng)
        signals.append(y)
        rows.append({"candidate_mode": mode, "base_pool_idx": src_i, "candidate_local_idx": i, "target_anchor_idx": anchor_idx})
    x = np.stack(signals, axis=0).astype(np.float32)
    feat = recompute_primitives(x)
    for k, v in pd.DataFrame(rows).items():
        feat[k] = v.to_numpy()
    return x, feat


def generate_bad_candidate_one(base: np.ndarray, subtype: str, variant: str, rng: np.random.Generator) -> np.ndarray:
    if variant == "detector_fail_preserve":
        return BAD.detector_fail_preserve_bad_candidate(base, subtype, rng)
    if variant == "morph_preserving":
        return BAD.morphology_preserving_bad_candidate(base, subtype, rng)
    if variant == "dense_stochastic":
        return BAD.dense_stochastic_bad_candidate(base, subtype, rng)
    if variant == "rhythm_disrupted":
        return BAD.rhythm_disrupted_bad_candidate(base, subtype, rng)
    if variant == "smooth":
        return BAD.smooth_bad_candidate(base, subtype, rng)
    return BAD.BUILD.generate_bad_waveform(base, subtype, rng, variant=variant)


def generate_bad_candidates(
    source_pool: np.ndarray,
    subtype: str,
    pool_n: int,
    rng: np.random.Generator,
    target: pd.DataFrame | None = None,
    target_condition: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    signals: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for i in range(int(pool_n)):
        src_i = int(rng.integers(0, len(source_pool)))
        variant = str(rng.choice(BAD_VARIANTS, p=BAD_VARIANT_PROBS))
        y = generate_bad_candidate_one(source_pool[src_i], subtype, variant, rng).astype(np.float32)
        anchor_idx = -1
        if target_condition and target is not None and not target.empty:
            anchor_idx = int(rng.integers(0, len(target)))
            y = target_condition_waveform(y, "bad", subtype, target.iloc[anchor_idx], rng)
        signals.append(y)
        rows.append({"candidate_mode": variant, "base_pool_idx": src_i, "candidate_local_idx": i, "target_anchor_idx": anchor_idx})
    x = np.stack(signals, axis=0).astype(np.float32)
    feat = recompute_primitives(x)
    for k, v in pd.DataFrame(rows).items():
        feat[k] = v.to_numpy()
    return x, feat


def calibrate_pool_amplitude(
    cand_x: np.ndarray,
    target: pd.DataFrame,
    rng: np.random.Generator,
    cls: str | None = None,
    subtype: str | None = None,
) -> np.ndarray:
    """Scale candidates to the BUT subtype RMS/PTP envelope before scoring."""

    if len(cand_x) == 0 or target.empty:
        return cand_x
    if version_starts(("v79",)) and str(cls or "") == "bad":
        # v78 exposed a bad-specific scale bug: raw envelope columns made
        # synthetic bad 10-300 IQR too energetic in the formal SQI space used
        # for target matching.  For bad we calibrate directly in the formal
        # waveform/SQI envelope that defines the BUT reference distribution.
        rms_col = "rms"
        ptp_col = "ptp_p99_p01"
    else:
        rms_col = "raw_rms" if "raw_rms" in target.columns else "rms"
        ptp_col = "raw_ptp_p99_p01" if "raw_ptp_p99_p01" in target.columns else "ptp_p99_p01"
    t_rms = (
        pd.to_numeric(target.get(rms_col, pd.Series([], dtype=float)), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float32)
    )
    t_ptp = (
        pd.to_numeric(target.get(ptp_col, pd.Series([], dtype=float)), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float32)
    )
    if version_starts(("v79",)) and str(cls or "") == "bad":
        diff_col = "non_qrs_diff_p95"
    else:
        diff_col = "raw_diff_abs_p95" if "raw_diff_abs_p95" in target.columns else "non_qrs_diff_p95"
    t_diff = (
        pd.to_numeric(target.get(diff_col, pd.Series([], dtype=float)), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float32)
    )
    if t_rms.size == 0:
        return cand_x
    out = cand_x.astype(np.float32).copy()
    centered = out - np.median(out, axis=1, keepdims=True)
    c_rms = np.sqrt(np.mean(centered * centered, axis=1))
    c_ptp = np.percentile(out, 99, axis=1) - np.percentile(out, 1, axis=1)
    anchors_rms = rng.choice(t_rms, size=len(out), replace=t_rms.size < len(out))
    scale_rms = anchors_rms / np.maximum(c_rms, 1e-6)
    if t_ptp.size:
        anchors_ptp = rng.choice(t_ptp, size=len(out), replace=t_ptp.size < len(out))
        scale_ptp = anchors_ptp / np.maximum(c_ptp, 1e-6)
        scale = 0.65 * scale_rms + 0.35 * scale_ptp
    else:
        scale = scale_rms
    lo_clip = 0.003 if version_starts(("v79",)) and str(cls or "") == "bad" else 0.025
    scale = np.clip(scale, lo_clip, 1.25).astype(np.float32)
    out = centered * scale[:, None]
    if t_diff.size:
        anchors_diff = rng.choice(t_diff, size=len(out), replace=t_diff.size < len(out))
        for i in range(len(out)):
            if raw_diff95(out[i]) >= 0.82 * float(anchors_diff[i]):
                continue
            fake_anchor = pd.Series(
                {
                    "raw_rms": float(anchors_rms[i]),
                    "raw_ptp_p99_p01": float(anchors_ptp[i] if t_ptp.size else np.nan),
                    "raw_diff_abs_p95": float(anchors_diff[i]),
                }
            )
            out[i] = match_raw_detail_envelope(out[i], fake_anchor, rng)
    if subtype in HARD_NONBAD_SUBTYPES and not target.empty:
        anchor_idx = rng.choice(np.arange(len(target)), size=len(out), replace=len(target) < len(out))
        hard_cls = str(cls or "medium")
        for i, ai in enumerate(anchor_idx.tolist()):
            out[i] = make_lowqrs_baseline_boundary(out[i], target.iloc[int(ai)], hard_cls, rng)
    return out.astype(np.float32)


def empirical_herding_select(
    target: pd.DataFrame,
    cand_feat: pd.DataFrame,
    n_select: int,
    features: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    center, scale = fit_robust_scaler(target, features)
    target_z = zscore(target, features, center, scale)
    cand_z = zscore(cand_feat, features, center, scale)
    if len(target_z) == 0:
        raise RuntimeError("empty target subtype")
    anchor_idx = rng.choice(np.arange(len(target_z)), size=int(n_select), replace=len(target_z) < int(n_select))
    anchors = target_z[anchor_idx]
    nn = NearestNeighbors(n_neighbors=min(96, len(cand_z)), algorithm="auto", metric="euclidean")
    nn.fit(cand_z)
    dist, idx = nn.kneighbors(anchors, return_distance=True)
    selected: list[int] = []
    used: set[int] = set()
    match_distance: dict[int, float] = {}
    for row_d, row_i in zip(dist, idx):
        pick = None
        pick_d = None
        for d, i in zip(row_d.tolist(), row_i.tolist()):
            if int(i) not in used:
                pick = int(i)
                pick_d = float(d)
                break
        if pick is not None:
            used.add(pick)
            selected.append(pick)
            match_distance[pick] = float(pick_d)
        if len(selected) >= int(n_select):
            break
    if len(selected) < int(n_select):
        target_med = np.median(target_z, axis=0)
        score = np.sqrt(np.mean((cand_z - target_med[None, :]) ** 2, axis=1))
        order = np.argsort(score)
        for i in order.tolist():
            if int(i) in used:
                continue
            selected.append(int(i))
            match_distance[int(i)] = float(score[int(i)])
            used.add(int(i))
            if len(selected) >= int(n_select):
                break
    selected = selected[: int(n_select)]
    audit = cand_feat.iloc[selected].reset_index(drop=True).copy()
    audit["herding_distance"] = [match_distance.get(int(i), np.nan) for i in selected]
    return np.asarray(selected, dtype=np.int64), audit


def waveform_prototype_matrix(x: np.ndarray, n_points: int = 80) -> np.ndarray:
    """Compact morphology prototype for visual-like matching.

    This is not a model input.  It keeps the distribution matcher from choosing
    rows that match scalar SQI while looking unlike the BUT waveform examples.
    """

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    if arr.ndim == 1:
        arr = arr[None, :]
    med = np.nanmedian(arr, axis=1, keepdims=True)
    q25 = np.nanpercentile(arr, 25, axis=1, keepdims=True)
    q75 = np.nanpercentile(arr, 75, axis=1, keepdims=True)
    scale = np.maximum((q75 - q25) / 1.349, np.nanstd(arr, axis=1, keepdims=True))
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0)
    z = np.clip((arr - med) / scale, -6.0, 6.0)
    grid = np.linspace(0, z.shape[1] - 1, int(n_points))
    lo = np.floor(grid).astype(int)
    hi = np.minimum(lo + 1, z.shape[1] - 1)
    frac = (grid - lo)[None, :]
    proto = (1.0 - frac) * z[:, lo] + frac * z[:, hi]
    dz = np.diff(proto, axis=1, prepend=proto[:, :1])
    return np.concatenate([proto, 0.5 * dz], axis=1).astype(np.float32)


def balanced_target_anchor_indices(target_emb: np.ndarray, n_select: int, rng: np.random.Generator) -> np.ndarray:
    if len(target_emb) == 0:
        raise RuntimeError("empty target subtype")
    if int(n_select) <= len(target_emb):
        # Random without replacement preserves the empirical subtype density
        # better than only taking PCA-center points.
        return rng.choice(np.arange(len(target_emb)), size=int(n_select), replace=False).astype(np.int64)
    reps = int(np.ceil(float(n_select) / float(len(target_emb))))
    parts = [rng.permutation(len(target_emb)) for _ in range(reps)]
    return np.concatenate(parts)[: int(n_select)].astype(np.int64)


def distribution_ot_select(
    target: pd.DataFrame,
    target_x: np.ndarray,
    cand_feat: pd.DataFrame,
    cand_x: np.ndarray,
    n_select: int,
    features: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Approximate subtype-conditional OT matching in interpretable space.

    The matcher first embeds target/candidate rows with robust 47-feature
    z-scores, BUT-fitted PCA coordinates, and a low-dimensional waveform
    prototype. It then assigns target empirical anchors to candidate rows with a
    linear-sum solver on a nearest-neighbor shortlist.
    """

    center, scale = fit_robust_scaler(target, features)
    target_z = zscore(target, features, center, scale)
    cand_z = zscore(cand_feat, features, center, scale)
    if len(target_z) == 0 or len(cand_z) == 0:
        return empirical_herding_select(target, cand_feat, n_select, features, rng)

    pca = robust_pca_fit(feature_matrix(target, features))
    target_pc = robust_pca_transform(feature_matrix(target, features), pca)[:, :4]
    cand_pc = robust_pca_transform(feature_matrix(cand_feat, features), pca)[:, :4]
    pc_center = np.nanmedian(target_pc, axis=0)
    pc_scale = np.maximum(
        np.nanpercentile(target_pc, 75, axis=0) - np.nanpercentile(target_pc, 25, axis=0),
        np.nanstd(target_pc, axis=0),
    )
    pc_scale = np.maximum(pc_scale, 1e-4)
    target_pc_z = (target_pc - pc_center[None, :]) / pc_scale[None, :]
    cand_pc_z = (cand_pc - pc_center[None, :]) / pc_scale[None, :]

    target_proto = waveform_prototype_matrix(target_x)
    cand_proto = waveform_prototype_matrix(cand_x)
    proto_center = np.nanmedian(target_proto, axis=0)
    proto_scale = np.maximum(
        np.nanpercentile(target_proto, 75, axis=0) - np.nanpercentile(target_proto, 25, axis=0),
        np.nanstd(target_proto, axis=0),
    )
    proto_scale = np.maximum(proto_scale, 1e-4)
    target_proto_z = (target_proto - proto_center[None, :]) / proto_scale[None, :]
    cand_proto_z = (cand_proto - proto_center[None, :]) / proto_scale[None, :]

    feature_weight = np.ones(len(features), dtype=np.float32)
    for j, name in enumerate(features):
        if name in {"qrs_visibility", "detector_agreement", "sqi_basSQI", "baseline_step", "flatline_ratio"}:
            feature_weight[j] = 1.45
        elif name in {"non_qrs_diff_p95", "band_30_45", "amplitude_entropy", "template_corr", "qrs_band_ratio"}:
            feature_weight[j] = 1.25
    target_emb = np.concatenate(
        [
            target_z * feature_weight[None, :],
            0.90 * target_pc_z,
            0.35 * target_proto_z,
        ],
        axis=1,
    ).astype(np.float32)
    cand_emb = np.concatenate(
        [
            cand_z * feature_weight[None, :],
            0.90 * cand_pc_z,
            0.35 * cand_proto_z,
        ],
        axis=1,
    ).astype(np.float32)
    target_emb = np.nan_to_num(target_emb, nan=0.0, posinf=0.0, neginf=0.0)
    cand_emb = np.nan_to_num(cand_emb, nan=0.0, posinf=0.0, neginf=0.0)

    anchor_idx = balanced_target_anchor_indices(target_emb, int(n_select), rng)
    anchors = target_emb[anchor_idx]
    nn_k = min(len(cand_emb), max(128, min(384, int(n_select) * 3)))
    nn = NearestNeighbors(n_neighbors=nn_k, algorithm="auto", metric="euclidean")
    nn.fit(cand_emb)
    _, nn_idx = nn.kneighbors(anchors, return_distance=True)
    shortlist = np.unique(nn_idx.reshape(-1)).astype(np.int64)
    max_short = max(int(n_select) * 16, int(n_select) + 256)
    if len(shortlist) > max_short:
        # Keep the candidates that are nearest to at least one target anchor.
        min_d = np.full(len(shortlist), np.inf, dtype=np.float64)
        short_emb = cand_emb[shortlist]
        for start in range(0, len(anchors), 64):
            d = np.mean((anchors[start : start + 64, None, :] - short_emb[None, :, :]) ** 2, axis=2)
            min_d = np.minimum(min_d, d.min(axis=0))
        shortlist = shortlist[np.argsort(min_d)[:max_short]]

    short_emb = cand_emb[shortlist]
    costs = np.empty((len(anchors), len(shortlist)), dtype=np.float32)
    for start in range(0, len(anchors), 64):
        costs[start : start + 64] = np.mean(
            (anchors[start : start + 64, None, :] - short_emb[None, :, :]) ** 2,
            axis=2,
        )
    row_ind, col_ind = linear_sum_assignment(costs)
    selected = shortlist[col_ind].astype(np.int64)
    if len(selected) < int(n_select):
        used = set(int(i) for i in selected.tolist())
        nearest_order = np.argsort(costs.min(axis=0))
        fill: list[int] = []
        for col in nearest_order.tolist():
            cand_i = int(shortlist[col])
            if cand_i in used:
                continue
            fill.append(cand_i)
            used.add(cand_i)
            if len(selected) + len(fill) >= int(n_select):
                break
        if fill:
            selected = np.concatenate([selected, np.asarray(fill, dtype=np.int64)])
    selected = selected[: int(n_select)]
    audit = cand_feat.iloc[selected].reset_index(drop=True).copy()
    selected_cols = [int(np.flatnonzero(shortlist == int(i))[0]) for i in selected if int(i) in set(shortlist.tolist())]
    if selected_cols:
        audit["distribution_ot_cost"] = np.min(costs[:, selected_cols], axis=0)[: len(audit)]
    else:
        audit["distribution_ot_cost"] = np.nan
    audit["distribution_matcher"] = "feature_pca_waveform_ot"
    return selected.astype(np.int64), audit


def candidate_distribution_prefilter(
    target: pd.DataFrame,
    cand_x: np.ndarray,
    cand_feat: pd.DataFrame,
    n_select: int,
    cls: str,
    subtype: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Gate candidate pool to rows that can plausibly match BUT subtype tails."""

    audit: dict[str, Any] = {
        "candidate_prefilter_enabled": False,
        "candidate_prefilter_before": int(len(cand_feat)),
        "candidate_prefilter_after": int(len(cand_feat)),
        "candidate_prefilter_mode": "none",
        "candidate_prefilter_penalty_median": float("nan"),
    }
    if not version_starts(("v74", "v75", "v76", "v78", "v79")) or target.empty or len(cand_feat) == 0:
        return cand_x, cand_feat.reset_index(drop=True), audit

    target_num = target.apply(pd.to_numeric, errors="coerce")
    cand_num = cand_feat.apply(pd.to_numeric, errors="coerce")

    def bounds(name: str) -> tuple[float, float, float]:
        vals = pd.to_numeric(target.get(name, pd.Series([], dtype=float)), errors="coerce").dropna()
        if vals.empty:
            return float("nan"), float("nan"), 1.0
        q05 = float(vals.quantile(0.05))
        q95 = float(vals.quantile(0.95))
        iqr = float(vals.quantile(0.75) - vals.quantile(0.25))
        return q05, q95, max(iqr, float(vals.std()), 1e-4)

    rules: list[tuple[str, str, float]] = []
    if cls in {"good", "medium"}:
        rules = [
            ("qrs_visibility", "upper", 2.2),
            ("non_qrs_diff_p95", "upper", 2.5),
            ("sqi_basSQI", "lower", 1.2),
            ("flatline_ratio", "lower", 0.8),
            ("band_30_45", "upper", 0.8),
            ("amplitude_entropy", "upper", 0.7),
        ]
        if "baseline" in subtype or "hard" in subtype:
            rules.append(("baseline_step", "lower", 0.9))
    elif cls == "bad":
        rules = [
            ("band_30_45", "upper", 0.8),
            ("qrs_visibility", "upper", 1.0),
            ("detector_agreement", "upper", 0.8),
            ("non_qrs_diff_p95", "upper", 1.4),
        ]
    else:
        return cand_x, cand_feat.reset_index(drop=True), audit

    penalty = np.zeros(len(cand_feat), dtype=np.float64)
    mask = np.ones(len(cand_feat), dtype=bool)
    for name, side, weight in rules:
        if name not in cand_num.columns:
            continue
        lo, hi, scale = bounds(name)
        vals = cand_num[name].fillna(hi if side == "upper" else lo).to_numpy(dtype=np.float64)
        if side == "upper" and np.isfinite(hi):
            excess = np.maximum(0.0, vals - hi) / scale
            penalty += float(weight) * excess
            mask &= vals <= hi + 0.5 * scale
        elif side == "lower" and np.isfinite(lo):
            excess = np.maximum(0.0, lo - vals) / scale
            penalty += float(weight) * excess
            mask &= vals >= lo - 0.5 * scale

    min_keep = min(len(cand_feat), max(int(n_select), int(n_select) * 3))
    if int(mask.sum()) >= min_keep:
        keep = np.flatnonzero(mask)
        mode = "strict_range"
    else:
        keep_n = min(len(cand_feat), max(min_keep, int(n_select) + 96))
        keep = np.argsort(np.nan_to_num(penalty, nan=np.inf, posinf=np.inf, neginf=np.inf))[:keep_n]
        mode = "penalty_topk"
    filtered_x = np.asarray(cand_x, dtype=np.float32)[keep]
    filtered_feat = cand_feat.iloc[keep].reset_index(drop=True).copy()
    audit.update(
        {
            "candidate_prefilter_enabled": True,
            "candidate_prefilter_after": int(len(filtered_feat)),
            "candidate_prefilter_mode": mode,
            "candidate_prefilter_penalty_median": float(np.nanmedian(penalty[keep])) if len(keep) else float("nan"),
        }
    )
    return filtered_x, filtered_feat, audit


def robust_pca_fit(x: np.ndarray) -> dict[str, np.ndarray]:
    center = np.nanmedian(x, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    scale = np.maximum(q75 - q25, np.nanstd(x, axis=0))
    scale = np.maximum(scale, 1e-4)
    z = (x - center[None, :]) / scale[None, :]
    _, s, vt = np.linalg.svd(z.astype(np.float64), full_matrices=False)
    return {
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "components": vt[:3].astype(np.float32),
        "explained": ((s[:3] ** 2) / max(float(np.sum(s**2)), 1e-12)).astype(np.float32),
    }


def robust_pca_transform(x: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    z = (x - pca["center"][None, :]) / pca["scale"][None, :]
    return (z @ pca["components"].T).astype(np.float32)


def quantile_loss(a: np.ndarray, b: np.ndarray) -> float:
    qs = [5, 10, 25, 50, 75, 90, 95]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    qa = np.nanpercentile(a, qs, axis=0)
    qb = np.nanpercentile(b, qs, axis=0)
    scale = np.maximum(np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0), np.nanstd(a, axis=0))
    scale = np.maximum(scale, 1e-4)
    return float(np.nanmean(np.abs(qa - qb) / scale[None, :]))


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, seed: int, n_proj: int = 64) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    dim = a.shape[1]
    proj = rng.normal(size=(int(n_proj), dim)).astype(np.float32)
    proj /= np.maximum(np.linalg.norm(proj, axis=1, keepdims=True), 1e-8)
    vals: list[float] = []
    n = min(len(a), len(b), 600)
    ia = rng.choice(len(a), size=n, replace=len(a) < n)
    ib = rng.choice(len(b), size=n, replace=len(b) < n)
    aa = a[ia]
    bb = b[ib]
    for p in proj:
        vals.append(float(np.mean(np.abs(np.sort(aa @ p) - np.sort(bb @ p)))))
    return float(np.mean(vals))


def mmd_rbf(a: np.ndarray, b: np.ndarray, seed: int, max_n: int = 500) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    ia = rng.choice(len(a), size=min(max_n, len(a)), replace=False)
    ib = rng.choice(len(b), size=min(max_n, len(b)), replace=False)
    x = a[ia].astype(np.float64)
    y = b[ib].astype(np.float64)
    xy = np.vstack([x, y])
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / max(med, 1e-6)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)).mean()
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2)).mean()
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def discriminative_auc(a: np.ndarray, b: np.ndarray, seed: int) -> float:
    if len(a) < 8 or len(b) < 8:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    n = min(len(a), len(b), 800)
    ia = rng.choice(len(a), size=n, replace=len(a) < n)
    ib = rng.choice(len(b), size=n, replace=len(b) < n)
    x = np.vstack([a[ia], b[ib]])
    y = np.asarray([0] * n + [1] * n, dtype=int)
    perm = rng.permutation(len(y))
    x = x[perm]
    y = y[perm]
    cut = int(0.7 * len(y))
    if len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
        return float("nan")
    try:
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(x[:cut], y[:cut])
        score = clf.predict_proba(x[cut:])[:, 1]
        auc = float(roc_auc_score(y[cut:], score))
        return max(auc, 1.0 - auc)
    except Exception:
        return float("nan")


def pca_density_gap(a_pc: np.ndarray, b_pc: np.ndarray, bins: int = 28) -> float:
    if len(a_pc) == 0 or len(b_pc) == 0:
        return float("nan")
    lo = np.nanpercentile(np.vstack([a_pc[:, :2], b_pc[:, :2]]), 2, axis=0)
    hi = np.nanpercentile(np.vstack([a_pc[:, :2], b_pc[:, :2]]), 98, axis=0)
    if np.any(hi <= lo):
        return float("nan")
    ha, _, _ = np.histogram2d(a_pc[:, 0], a_pc[:, 1], bins=bins, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    hb, _, _ = np.histogram2d(b_pc[:, 0], b_pc[:, 1], bins=bins, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    ha = ha / max(float(ha.sum()), 1.0)
    hb = hb / max(float(hb.sum()), 1.0)
    return float(np.abs(ha - hb).sum() / 2.0)


def representative_indices(frame: pd.DataFrame, n: int) -> list[int]:
    if len(frame) == 0:
        return []
    if len(frame) <= n:
        return frame["idx"].astype(int).tolist()
    cols = [c for c in REP_FEATURES if c in frame.columns]
    if not cols:
        return frame.sample(n, random_state=17)["idx"].astype(int).tolist()
    vals = frame[cols].apply(pd.to_numeric, errors="coerce")
    med = vals.median()
    scale = (vals.quantile(0.75) - vals.quantile(0.25)).replace(0, np.nan)
    dist = np.sqrt(np.nanmean(np.square(((vals - med) / scale).to_numpy(dtype=float)), axis=1))
    order = np.argsort(np.nan_to_num(dist, nan=np.inf))
    return frame.iloc[order[:n]]["idx"].astype(int).tolist()


def make_waveform_sheet(
    frame: pd.DataFrame,
    signals: np.ndarray,
    dataset_label: str,
    cls: str,
    out_path: Path,
    max_cols: int = 4,
) -> None:
    sub = frame.loc[frame["class_name"].astype(str).eq(cls)].copy()
    subtypes = [s for s in SUBTYPES_BY_CLASS[cls] if s in set(sub["display_subtype"].astype(str))]
    if not subtypes:
        return
    rows = len(subtypes)
    fig, axes = plt.subplots(rows, max_cols, figsize=(9.2, max(2.2, rows * 1.05)), squeeze=False, sharex=True)
    t = np.arange(signals.shape[-1]) / 125.0
    for r, subtype in enumerate(subtypes):
        d = sub.loc[sub["display_subtype"].astype(str).eq(subtype)].copy()
        chosen = representative_indices(d, max_cols)
        for c in range(max_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c >= len(chosen):
                continue
            idx = int(chosen[c])
            ax.plot(t, robust_trace(signals[idx]), color="#333333", lw=0.7)
            if c == 0:
                ax.text(
                    -0.05,
                    0.50,
                    subtype.replace("_", "\n"),
                    transform=ax.transAxes,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=6,
                    color="#222222",
                )
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(-2.2, 2.2)
    fig.suptitle(f"{dataset_label} | {cls} subtype representative waveforms", fontsize=11)
    save_fig(fig, out_path, dpi=260)


def make_shared_pca_plot(
    but: pd.DataFrame,
    synth: pd.DataFrame,
    pca_features: list[str],
    out_path: Path,
) -> None:
    setup_plot_style()
    target = but.loc[but["split"].astype(str).isin(["train", "val"])].copy()
    pca = robust_pca_fit(feature_matrix(target, pca_features))
    but_pc = robust_pca_transform(feature_matrix(but, pca_features), pca)
    syn_pc = robust_pca_transform(feature_matrix(synth, pca_features), pca)
    but_plot = but.copy()
    syn_plot = synth.copy()
    but_plot[["pc1_shared", "pc2_shared", "pc3_shared"]] = but_pc
    syn_plot[["pc1_shared", "pc2_shared", "pc3_shared"]] = syn_pc
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharex=False, sharey=False)
    for ax, cls in zip(axes, CLASS_ORDER):
        b_train = but_plot.loc[but_plot["class_name"].eq(cls) & but_plot["split"].isin(["train", "val"])].sample(
            n=min(1600, int((but_plot["class_name"].eq(cls) & but_plot["split"].isin(["train", "val"])).sum())),
            random_state=13,
            replace=False,
        )
        b_test = but_plot.loc[but_plot["class_name"].eq(cls) & but_plot["split"].eq("test")].sample(
            n=min(800, int((but_plot["class_name"].eq(cls) & but_plot["split"].eq("test")).sum())),
            random_state=17,
            replace=False,
        )
        s = syn_plot.loc[syn_plot["class_name"].eq(cls)].sample(
            n=min(1600, int(syn_plot["class_name"].eq(cls).sum())),
            random_state=19,
            replace=False,
        )
        ax.scatter(b_train["pc1_shared"], b_train["pc2_shared"], s=5, alpha=0.22, c=PALETTE["BUT train+val"], label="BUT train+val")
        ax.scatter(b_test["pc1_shared"], b_test["pc2_shared"], s=5, alpha=0.25, c=PALETTE["BUT test"], label="BUT test")
        ax.scatter(s["pc1_shared"], s["pc2_shared"], s=5, alpha=0.30, c=PALETTE["PTB synthetic"], marker="x", linewidths=0.35, label="PTB synthetic")
        ax.set_title(cls)
        ax.set_xlabel("shared PC1")
        ax.set_ylabel("shared PC2")
        ax.grid(True, color="#e8e8e8", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3)
    fig.suptitle(f"{VERSION} PTB synthetic vs BUT keep-outlier geometry", y=1.17, fontsize=12)
    save_fig(fig, out_path, dpi=280)


def make_feature_gap_heatmap(metrics: pd.DataFrame, out_path: Path) -> None:
    if metrics.empty:
        return
    pivot = metrics.pivot_table(index="subtype", columns="metric", values="value", aggfunc="mean")
    cols = [c for c in ["median_abs_robust_gap", "quantile_loss", "sliced_wasserstein", "mmd_rbf", "pca_density_gap", "discriminative_auc"] if c in pivot.columns]
    data = pivot[cols].copy()
    fig, ax = plt.subplots(figsize=(8.8, max(3.0, 0.25 * len(data) + 1.5)))
    arr = data.to_numpy(dtype=float)
    vmax = np.nanpercentile(arr, 95) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(vmax, 1e-4))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels([str(s).replace("_", " ") for s in data.index], fontsize=6)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i,j]:.2f}", ha="center", va="center", fontsize=5, color="#222222")
    fig.colorbar(im, ax=ax, shrink=0.8, label="gap / separability")
    ax.set_title("Subtype distribution gap: BUT train+val vs selected PTB synthetic")
    save_fig(fig, out_path, dpi=260)


def audit_distribution(but: pd.DataFrame, synth: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = but.loc[but["split"].astype(str).isin(["train", "val"])].copy()
    pca = robust_pca_fit(feature_matrix(target, MATCH_FEATURES))
    for cls, subtypes in SUBTYPES_BY_CLASS.items():
        for subtype in subtypes:
            b = target.loc[target["display_subtype"].astype(str).eq(subtype)].copy()
            s = synth.loc[synth["display_subtype"].astype(str).eq(subtype)].copy()
            if b.empty or s.empty:
                continue
            center, scale = fit_robust_scaler(b, MATCH_FEATURES)
            bz = zscore(b, MATCH_FEATURES, center, scale)
            sz = zscore(s, MATCH_FEATURES, center, scale)
            bpc = robust_pca_transform(feature_matrix(b, MATCH_FEATURES), pca)
            spc = robust_pca_transform(feature_matrix(s, MATCH_FEATURES), pca)
            med_gap = float(np.nanmean(np.abs(np.nanmedian(bz, axis=0) - np.nanmedian(sz, axis=0))))
            vals = {
                "median_abs_robust_gap": med_gap,
                "quantile_loss": quantile_loss(bz, sz),
                "sliced_wasserstein": sliced_wasserstein(bz, sz, seed + len(rows)),
                "mmd_rbf": mmd_rbf(bz, sz, seed + len(rows)),
                "pca_density_gap": pca_density_gap(bpc, spc),
                "discriminative_auc": discriminative_auc(bz, sz, seed + len(rows)),
            }
            for metric, value in vals.items():
                rows.append(
                    {
                        "class_name": cls,
                        "subtype": subtype,
                        "metric": metric,
                        "value": value,
                        "but_n_trainval": int(len(b)),
                        "ptb_n": int(len(s)),
                    }
                )
    return pd.DataFrame(rows)


def subtype_count_tables(but: pd.DataFrame, synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    but_counts = but.groupby(["split", "class_name", "display_subtype"], dropna=False).size().reset_index(name="n")
    synth_counts = synth.groupby(["split", "class_name", "display_subtype"], dropna=False).size().reset_index(name="n")
    total = but.loc[but["split"].isin(["train", "val"]), "display_subtype"].value_counts().to_dict()
    but_counts["natural_trainval_share"] = but_counts["display_subtype"].map(
        lambda s: float(total.get(s, 0)) / max(1, sum(total.values()))
    )
    return but_counts, synth_counts


def make_protocol(args: argparse.Namespace) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(args.seed))
    out_protocol = OUT_DIR / f"protocol_{VERSION}_pc{int(args.total_per_class)}_s{int(args.seed)}"
    out_protocol.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    but = pd.read_csv(BUT_PROTOCOL / "original_region_atlas.csv", low_memory=False)
    but_display_subtype = assign_but_subtypes(but)
    but_x = np.load(BUT_PROTOCOL / "signals.npz")["X"].astype(np.float32)
    if but_x.ndim == 3:
        but_x = but_x[:, 0, :]
    if bool(args.recompute_but_features):
        but = recompute_full_features(but_x, but)
    but = add_raw_waveform_stats(but, but_x)
    but["_row_i"] = np.arange(len(but), dtype=np.int64)
    but["display_subtype"] = but_display_subtype.to_numpy()
    but["subtype_for_split"] = but["display_subtype"].astype(str)

    base = pd.read_csv(BASE_PROTOCOL / "original_region_atlas.csv", low_memory=False)
    base_x = np.load(BASE_PROTOCOL / "signals.npz")["X"].astype(np.float32)
    if base_x.ndim == 3:
        base_x = base_x[:, 0, :]
    base_local = base.reset_index(drop=True).copy()
    base_local["idx"] = np.arange(len(base_local), dtype=np.int64)
    if VERSION.startswith("v71") or bool(args.candidate_full_features):
        print(f"{now()} recomputing PTB source feature table for source-conditioned generation", flush=True)
        source_feat_all = recompute_full_features(base_x, base_local)
    else:
        source_feat_all = add_raw_waveform_stats(base_local, base_x)

    counts = target_counts(int(args.total_per_class))
    selected_signals: list[np.ndarray] = []
    selected_rows: list[pd.Series] = []
    candidate_audit_rows: list[pd.DataFrame] = []
    global_i = 0

    base_by_class = {
        cls: base.loc[base["class_name"].astype(str).eq(cls)].copy()
        for cls in CLASS_ORDER
    }
    source_pool_all = base_x[base["idx"].astype(int).to_numpy()]
    source_feat_all_by_idx = source_feat_all.iloc[base["idx"].astype(int).to_numpy()].reset_index(drop=True)

    for cls in CLASS_ORDER:
        cls_base = base_by_class[cls]
        if cls_base.empty:
            cls_base = base
        cls_idx = cls_base["idx"].astype(int).to_numpy()
        cls_pool = base_x[cls_idx]
        cls_source_feat = source_feat_all.iloc[cls_idx].reset_index(drop=True)
        template = cls_base.iloc[0].copy()
        for subtype, n_select in counts[cls].items():
            target = but.loc[
                but["display_subtype"].astype(str).eq(subtype) & but["split"].astype(str).isin(["train", "val"])
            ].copy()
            if target.empty:
                target = but.loc[but["class_name"].astype(str).eq(cls) & but["split"].astype(str).isin(["train", "val"])].copy()
            pool_n = max(int(n_select) * int(args.pool_multiplier), int(n_select) + 64)
            print(f"{now()} generating {cls}/{subtype}: target={n_select} pool={pool_n}", flush=True)
            if cls in {"good", "medium"}:
                active_pool, active_source_feat, source_audit = source_conditioned_pool(
                    cls_pool, cls_source_feat, target, cls, subtype, pool_n, rng
                )
                cand_x, cand_feat = generate_gm_candidates(
                    active_pool,
                    cls,
                    subtype,
                    pool_n,
                    rng,
                    target=target,
                    target_condition=bool(args.target_conditioned),
                )
            else:
                BAD.ACTIVE_VERSION = "v20"
                active_pool, active_source_feat, source_audit = source_conditioned_pool(
                    source_pool_all, source_feat_all_by_idx, target, cls, subtype, pool_n, rng
                )
                cand_x, cand_feat = generate_bad_candidates(
                    active_pool,
                    subtype,
                    pool_n,
                    rng,
                    target=target,
                    target_condition=bool(args.target_conditioned),
                )
            if bool(args.amplitude_calibrate):
                meta_cols = [c for c in ["candidate_mode", "base_pool_idx", "candidate_local_idx", "target_anchor_idx"] if c in cand_feat.columns]
                meta_keep = cand_feat[meta_cols].copy()
                cand_x = calibrate_pool_amplitude(cand_x, target, rng, cls=cls, subtype=subtype)
                cand_feat = recompute_primitives(cand_x)
                for col in meta_cols:
                    cand_feat[col] = meta_keep[col].to_numpy()
            cand_feat = candidate_feature_frame(cand_x, cand_feat, cls, bool(args.candidate_full_features))
            cand_x, cand_feat, candidate_prefilter_audit = candidate_distribution_prefilter(
                target, cand_x, cand_feat, int(n_select), cls, subtype
            )
            if version_starts(("v78", "v79")) and "_row_i" in target.columns:
                target_rows = target["_row_i"].astype(int).to_numpy()
                target_rows = target_rows[(target_rows >= 0) & (target_rows < len(but_x))]
                target_x = but_x[target_rows] if len(target_rows) else but_x[:0]
                take, audit = distribution_ot_select(
                    target,
                    target_x,
                    cand_feat,
                    cand_x,
                    int(n_select),
                    MATCH_FEATURES,
                    rng,
                )
            else:
                take, audit = empirical_herding_select(target, cand_feat, int(n_select), MATCH_FEATURES, rng)
            splits = split_assignments(int(n_select), rng)
            for j, cand_i in enumerate(take.tolist()):
                row = template.copy()
                row["idx"] = int(global_i)
                row["split"] = str(splits[j])
                row["class_name"] = cls
                row["y"] = CLASS_TO_INT[cls]
                row["source_idx"] = int(37_000_000 + global_i)
                row["record_id"] = f"ptbgen_v37_{subtype}"
                row["subject_id"] = f"ptbgen_v37_{subtype}"
                row["original_region"] = subtype
                row["ambiguous_type"] = subtype
                row["display_subtype"] = subtype
                row["subtype_id"] = subtype
                row["subtype_for_split"] = subtype
                row["clean_policy"] = "ptb_v37_subtype_balanced_distribution"
                row["generated_mode"] = str(cand_feat.loc[cand_i, "candidate_mode"])
                row["base_pool_idx"] = int(cand_feat.loc[cand_i, "base_pool_idx"])
                row["target_anchor_idx"] = int(cand_feat.loc[cand_i, "target_anchor_idx"]) if "target_anchor_idx" in cand_feat.columns else -1
                selected_rows.append(row)
                selected_signals.append(cand_x[cand_i].astype(np.float32))
                global_i += 1
            audit = audit.copy()
            audit["class_name"] = cls
            audit["target_subtype"] = subtype
            audit["n_target_but_trainval"] = int(len(target))
            for key, val in source_audit.items():
                audit[key] = val
            for key, val in candidate_prefilter_audit.items():
                audit[key] = val
            candidate_audit_rows.append(audit)

    x = np.stack(selected_signals, axis=0).astype(np.float32)
    frame = pd.DataFrame(selected_rows).reset_index(drop=True)
    frame["idx"] = np.arange(len(frame), dtype=np.int64)
    frame = recompute_full_features(x, frame)
    frame = add_raw_waveform_stats(frame, x)
    frame["display_subtype"] = frame["subtype_id"].astype(str)
    frame["subtype_for_split"] = frame["subtype_id"].astype(str)
    frame["clean_policy"] = VERSION
    write_csv(frame, out_protocol / "original_region_atlas.csv")
    write_csv(frame, out_protocol / "metadata.csv")
    np.savez_compressed(long_path(out_protocol / "signals.npz"), X=x)

    but_counts, synth_counts = subtype_count_tables(but, frame)
    write_csv(but_counts, out_protocol / "but_keepoutlier_subtype_counts.csv")
    write_csv(synth_counts, out_protocol / "ptb_v37_subtype_counts.csv")
    cand_audit = pd.concat(candidate_audit_rows, ignore_index=True) if candidate_audit_rows else pd.DataFrame()
    write_csv(cand_audit, out_protocol / "selected_candidate_scores.csv")
    metrics = audit_distribution(but, frame, int(args.seed))
    write_csv(metrics, out_protocol / "distribution_metrics.csv")

    summary = {
        "created_at": now(),
        "protocol": str(out_protocol),
        "version": VERSION,
        "base_protocol": str(BASE_PROTOCOL),
        "but_reference_protocol": str(BUT_PROTOCOL),
        "seed": int(args.seed),
        "total_per_class": int(args.total_per_class),
        "pool_multiplier": int(args.pool_multiplier),
        "amplitude_calibrate": bool(args.amplitude_calibrate),
        "target_conditioned": bool(args.target_conditioned),
        "recompute_but_features": bool(args.recompute_but_features),
        "candidate_full_features": bool(args.candidate_full_features),
        "bad_frequency_profile": BAD_FREQUENCY_PROFILE,
        "subtypes_by_class": SUBTYPES_BY_CLASS,
        "target_counts": counts,
        "match_features": MATCH_FEATURES,
        "input_contract": "PTB-only waveform generation; BUT train+val feature distributions only; BUT test not used for selection",
    }
    write_text(out_protocol / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return out_protocol, frame, but, metrics


def write_report(protocol: Path, frame: pd.DataFrame, but: pd.DataFrame, metrics: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
        if df.empty:
            return "_empty_"
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

    count = frame.groupby(["split", "class_name", "display_subtype"], dropna=False).size().reset_index(name="n")
    but_count = but.groupby(["split", "class_name", "display_subtype"], dropna=False).size().reset_index(name="n")
    metric_wide = metrics.pivot_table(
        index=["class_name", "subtype", "but_n_trainval", "ptb_n"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    metric_wide.columns = [str(c) for c in metric_wide.columns]
    lines = [
        f"# {VERSION} Subtype-Balanced PTB Synthetic Distribution Report",
        "",
        f"- Created: `{now()}`",
        f"- Protocol: `{protocol}`",
        f"- BUT reference: `{BUT_PROTOCOL}`",
        f"- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.",
        "",
        "## Key Outputs",
        "",
        f"- Shared PCA: `{REPORT_OUT_DIR / f'{VERSION}_shared_pca_but_vs_ptb.png'}`",
        f"- Feature gap heatmap: `{REPORT_OUT_DIR / f'{VERSION}_subtype_feature_gap_heatmap.png'}`",
        f"- PTB waveform sheets: `{REPORT_OUT_DIR / f'{VERSION}_ptb_synthetic_good_subtype_waveforms.png'}`, `{REPORT_OUT_DIR / f'{VERSION}_ptb_synthetic_medium_subtype_waveforms.png'}`, `{REPORT_OUT_DIR / f'{VERSION}_ptb_synthetic_bad_subtype_waveforms.png'}`",
        f"- BUT waveform sheets: `{REPORT_OUT_DIR / f'{VERSION}_but_keepoutlier_good_subtype_waveforms.png'}`, `{REPORT_OUT_DIR / f'{VERSION}_but_keepoutlier_medium_subtype_waveforms.png'}`, `{REPORT_OUT_DIR / f'{VERSION}_but_keepoutlier_bad_subtype_waveforms.png'}`",
        "",
        "## PTB Synthetic Counts",
        "",
        md_table(count, 120),
        "",
        "## BUT Keep-Outlier Counts",
        "",
        md_table(but_count, 120),
        "",
        "## Distribution Metrics",
        "",
        "Lower is better for robust gaps/MMD/Wasserstein/density gap. Discriminative AUC closer to 0.5 means synthetic and BUT are harder to tell apart.",
        "",
        md_table(metric_wide.sort_values(["class_name", "subtype"]), 120),
        "",
        "## Interpretation",
        "",
        "- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.",
        "- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.",
        "- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.",
    ]
    write_text(REPORT_OUT_DIR / f"{VERSION}_subtype_balanced_distribution_report.md", "\n".join(lines))


def make_audit_outputs(protocol: Path, frame: pd.DataFrame, but: pd.DataFrame, metrics: pd.DataFrame) -> None:
    setup_plot_style()
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_ptb = np.load(protocol / "signals.npz")["X"].astype(np.float32)
    if x_ptb.ndim == 3:
        x_ptb = x_ptb[:, 0, :]
    x_but = np.load(BUT_PROTOCOL / "signals.npz")["X"].astype(np.float32)
    if x_but.ndim == 3:
        x_but = x_but[:, 0, :]
    make_shared_pca_plot(but, frame, MATCH_FEATURES, REPORT_OUT_DIR / f"{VERSION}_shared_pca_but_vs_ptb")
    make_feature_gap_heatmap(metrics, REPORT_OUT_DIR / f"{VERSION}_subtype_feature_gap_heatmap")
    for cls in CLASS_ORDER:
        make_waveform_sheet(frame, x_ptb, f"PTB synthetic {VERSION}", cls, REPORT_OUT_DIR / f"{VERSION}_ptb_synthetic_{cls}_subtype_waveforms")
        make_waveform_sheet(but, x_but, "BUT keep-outlier", cls, REPORT_OUT_DIR / f"{VERSION}_but_keepoutlier_{cls}_subtype_waveforms")
    write_report(protocol, frame, but, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--total-per-class", type=int, default=3000)
    parser.add_argument("--pool-multiplier", type=int, default=30)
    parser.add_argument("--stage", type=str, default="all", choices=["build", "audit", "all"])
    parser.add_argument("--protocol", type=str, default="")
    parser.add_argument("--version", type=str, default=VERSION)
    parser.add_argument("--amplitude-calibrate", action="store_true", default=False)
    parser.add_argument("--target-conditioned", action="store_true", default=False)
    parser.add_argument("--candidate-full-features", action="store_true", default=False)
    parser.add_argument("--recompute-but-features", action="store_true", default=True)
    parser.add_argument("--no-recompute-but-features", dest="recompute_but_features", action="store_false")
    parser.add_argument("--bad-frequency-profile", type=str, default="midband", choices=["legacy", "midband"])
    parser.add_argument("--but-protocol", type=str, default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--base-protocol", type=str, default=str(BASE_PROTOCOL))
    args = parser.parse_args()
    configure_version(str(args.version))
    configure_bad_frequency_profile(str(args.bad_frequency_profile))
    configure_but_protocol(str(args.but_protocol))
    configure_base_protocol(str(args.base_protocol))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in {"build", "all"}:
        protocol, frame, but, metrics = make_protocol(args)
        make_audit_outputs(protocol, frame, but, metrics)
        print(f"{now()} wrote {protocol}", flush=True)
        return

    protocol = Path(args.protocol) if args.protocol else OUT_DIR / f"protocol_{VERSION}_pc{int(args.total_per_class)}_s{int(args.seed)}"
    frame = pd.read_csv(protocol / "original_region_atlas.csv", low_memory=False)
    but = pd.read_csv(BUT_PROTOCOL / "original_region_atlas.csv", low_memory=False)
    but_display_subtype = assign_but_subtypes(but)
    if bool(args.recompute_but_features):
        but_x = np.load(BUT_PROTOCOL / "signals.npz")["X"].astype(np.float32)
        if but_x.ndim == 3:
            but_x = but_x[:, 0, :]
        but = recompute_full_features(but_x, but)
        but = add_raw_waveform_stats(but, but_x)
    but["display_subtype"] = but_display_subtype.to_numpy()
    but["subtype_for_split"] = but["display_subtype"].astype(str)
    metrics = pd.read_csv(protocol / "distribution_metrics.csv")
    make_audit_outputs(protocol, frame, but, metrics)
    print(f"{now()} audited {protocol}", flush=True)


if __name__ == "__main__":
    main()

