"""V81 brute feature-distribution transport from PTB carriers to BUT targets.

This external-only script replaces the older "generate a pool, then pick close
rows" loop with a target-row transport loop:

1. Sample BUT train+val anchors by label x subtype.
2. Draw raw PTB-XL lead-I carriers.
3. Generate several transformed candidates for every BUT anchor.
4. Recompute the same waveform/SQI feature extractor for all candidates.
5. Select the candidate closest to the BUT anchor in robust feature space.
6. Audit V80 and V81 with RBF-MMD, sliced-Wasserstein, CDF loss, PCA density,
   and domain AUC.

The script never copies BUT waveform samples into PTB synthetic rows. BUT is
used only as a feature-distribution target for synthesis and audit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
DEFAULT_REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "v81_distribution_transport"

V37_BUILDER = ANALYSIS_DIR / "build_ptb_v37_subtype_balanced_distribution.py"
DEFAULT_BUT_PROTOCOL = (
    ANALYSIS_DIR
    / "clean_but_protocols"
    / "margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed"
)
DEFAULT_V80_PROTOCOL = (
    ANALYSIS_DIR
    / "event_xds_aligned_v80lead1_boundary_clean"
    / "protocol_v80lead1_boundary_clean_pc700_s20260686"
)
DEFAULT_CARRIER_PROTOCOL = (
    ANALYSIS_DIR
    / "raw_ptbxl_carrier_protocols"
    / "raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680"
)

CLASS_ORDER = ["good", "medium", "bad"]
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
KEY_FEATURES = [
    "sqi_basSQI",
    "qrs_band_ratio",
    "qrs_visibility",
    "detector_agreement",
    "template_corr",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "non_qrs_diff_p95",
    "band_15_30",
    "band_30_45",
    "amplitude_entropy",
    "raw_rms",
    "raw_ptp_p99_p01",
    "raw_diff_abs_p95",
]
CDF_FEATURES = [
    "sqi_basSQI",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "band_30_45",
    "amplitude_entropy",
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "but": "#2E4780",
    "ptb": "#E07B39",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def long_path(path: Path) -> str:
    resolved = str(Path(path).resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V37 = load_module(V37_BUILDER, "v37_for_v81_transport")
MATCH_FEATURES = [f for f in V37.MATCH_FEATURES if f not in {"knn_label_purity", "boundary_confidence", "region_confidence"}]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "text.color": TOKENS["ink"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(long_path(path.with_suffix(".png")), dpi=250, bbox_inches="tight")
    fig.savefig(long_path(path.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


def load_protocol(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(long_path(path / "original_region_atlas.csv"), low_memory=False).copy()
    arr = np.load(long_path(path / "signals.npz"))["X"].astype(np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if len(frame) != len(arr):
        raise ValueError(f"row/signal mismatch for {path}: {len(frame)} vs {len(arr)}")
    if "idx" not in frame.columns:
        frame["idx"] = np.arange(len(frame), dtype=int)
    fallback_idx = pd.Series(np.arange(len(frame), dtype=int), index=frame.index)
    frame["idx"] = pd.to_numeric(frame["idx"], errors="coerce").fillna(fallback_idx).astype(int)
    frame = harmonize_feature_aliases(frame)
    return frame, arr


def recompute_protocol_features(frame: pd.DataFrame, signals: np.ndarray, label: str) -> pd.DataFrame:
    """Recompute waveform/SQI features with the single v37 extractor.

    Earlier protocol generations mixed feature vintages.  V81's distribution
    objective is only meaningful if BUT and PTB are audited under the same
    extractor, so every build/audit path normalizes through this function.
    """

    print(f"{now()} recomputing unified features for {label} rows={len(frame)}", flush=True)
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    feat = V37.recompute_full_features(x, frame.copy())
    out = frame.copy()
    for col in feat.columns:
        if col in MATCH_FEATURES or col in KEY_FEATURES or col.startswith("raw_"):
            out[col] = feat[col].to_numpy()
        elif col not in out.columns:
            out[col] = feat[col].to_numpy()
    out = add_consistent_transport_primitives(out, x)
    return harmonize_feature_aliases(out)


def harmonize_feature_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill raw feature aliases so BUT/PTB audits use the same semantics.

    Some older BUT protocols expose only rms/ptp/non_qrs_diff columns while
    synthetic protocols also persist raw_* duplicates. Treating missing raw_*
    columns as zero makes the distribution distance meaningless, so the aliases
    are filled from waveform-computable equivalents before matching/auditing.
    """

    out = frame.copy()
    aliases = {
        "raw_rms": ["rms"],
        "raw_ptp_p99_p01": ["ptp_p99_p01"],
        "raw_diff_abs_p95": ["non_qrs_diff_p95", "diff_abs_p95"],
    }
    for dst, sources in aliases.items():
        if dst in out.columns:
            vals = pd.to_numeric(out[dst], errors="coerce")
        else:
            vals = pd.Series(np.nan, index=out.index, dtype=float)
        for src in sources:
            if src not in out.columns:
                continue
            src_vals = pd.to_numeric(out[src], errors="coerce")
            vals = vals.fillna(src_vals)
        out[dst] = vals.fillna(0.0).astype(float)
    return out


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    vals: list[np.ndarray] = []
    for col in features:
        if col in frame.columns:
            vals.append(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64))
        else:
            vals.append(np.zeros(len(frame), dtype=np.float64))
    x = np.vstack(vals).T if vals else np.zeros((len(frame), 0), dtype=np.float64)
    if x.size:
        med = np.nanmedian(x, axis=0)
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.take(med, np.where(bad)[1])
    return x


def robust_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    scale = np.maximum(q75 - q25, np.nanstd(x, axis=0))
    scale = np.maximum(scale, 1e-5)
    return center, scale


def robust_z(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)


def row_robust_z(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(arr, axis=1, keepdims=True)
    q25 = np.nanpercentile(arr, 25, axis=1, keepdims=True)
    q75 = np.nanpercentile(arr, 75, axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(arr, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return ((arr - med) / scale).astype(np.float32)


def add_consistent_transport_primitives(frame: pd.DataFrame, signals: np.ndarray, fs: float = 125.0) -> pd.DataFrame:
    """Recompute vintage-sensitive primitive columns from waveform for all domains."""

    out = frame.copy()
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if len(out) != len(x):
        x = x[: len(out)]
    if len(x) == 0:
        return out
    z = row_robust_z(x)
    absz = np.abs(z)
    dz = np.diff(z, axis=1)
    absdz = np.abs(dz)
    n = len(z)
    qrs_prom_median = np.zeros(n, dtype=np.float32)
    qrs_prom_p90 = np.zeros(n, dtype=np.float32)
    periodicity = np.zeros(n, dtype=np.float32)
    detail_instability = np.zeros(n, dtype=np.float32)
    min_gap = max(1, int(round(0.24 * fs)))
    for i in range(n):
        ar = absz[i]
        threshold = max(float(np.percentile(ar, 92.0)), float(ar.mean() + 0.6 * ar.std()))
        loc = np.flatnonzero((ar[1:-1] >= ar[:-2]) & (ar[1:-1] >= ar[2:]) & (ar[1:-1] >= threshold)) + 1
        peaks: list[int] = []
        for p in loc:
            pp = int(p)
            if not peaks or pp - peaks[-1] >= min_gap:
                peaks.append(pp)
            elif ar[pp] > ar[peaks[-1]]:
                peaks[-1] = pp
        if peaks:
            amps = ar[np.asarray(peaks, dtype=int)]
            qrs_prom_median[i] = float(np.percentile(amps, 50))
            qrs_prom_p90[i] = float(np.percentile(amps, 90))
            if len(peaks) >= 3:
                rr = np.diff(np.asarray(peaks, dtype=np.float32))
                cv = float(np.std(rr) / max(float(np.mean(rr)), 1e-6))
                count_score = min(len(peaks), 14) / 14.0
                periodicity[i] = float(np.clip((1.0 - cv) * count_score, 0.0, 1.0))
        if absdz.shape[1] > 0:
            tail = float(np.percentile(absdz[i], 95))
            body = float(np.percentile(absdz[i], 50))
            detail_instability[i] = float(np.clip((tail - body) / (tail + body + 1e-6), 0.0, 1.0))
    out["qrs_prom_median"] = qrs_prom_median
    out["qrs_prom_p90"] = qrs_prom_p90
    out["periodicity"] = periodicity
    out["detail_instability"] = detail_instability
    return out


def safe_value(row: pd.Series, name: str, default: float) -> float:
    try:
        value = float(row.get(name, default))
    except Exception:
        value = float(default)
    return float(value) if np.isfinite(value) else float(default)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(np.asarray(x, dtype=np.float32), kernel, mode="same").astype(np.float32)


def lowfreq_component(n: int, amp: float, rng: np.random.Generator, kind: str) -> np.ndarray:
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    f1 = float(rng.uniform(0.03, 0.18 if kind == "slow" else 0.45))
    f2 = float(rng.uniform(0.12, 0.65))
    drift = np.sin(2 * np.pi * f1 * t + float(rng.uniform(-np.pi, np.pi)))
    drift += 0.45 * np.sin(2 * np.pi * f2 * t + float(rng.uniform(-np.pi, np.pi)))
    if rng.random() < 0.55:
        loc = int(rng.integers(n // 8, max(n // 8 + 1, 7 * n // 8)))
        step = np.zeros(n, dtype=np.float32)
        step[loc:] = 1.0
        drift += float(rng.uniform(-1.0, 1.0)) * step
    return (float(amp) * drift).astype(np.float32)


def band_bursts(
    n: int,
    amp: float,
    rng: np.random.Generator,
    freq_range: tuple[float, float],
    count: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for _ in range(int(rng.integers(count[0], count[1] + 1))):
        width = int(rng.integers(max(16, n // 80), max(24, n // 5)))
        start = int(rng.integers(0, max(1, n - width)))
        stop = min(n, start + width)
        local_n = stop - start
        t = np.arange(local_n, dtype=np.float32) / 125.0
        freq = float(rng.uniform(*freq_range))
        carrier = np.sin(2 * np.pi * freq * t + float(rng.uniform(-np.pi, np.pi)))
        window = np.hanning(local_n).astype(np.float32)
        out[start:stop] += float(amp) * window * carrier.astype(np.float32)
        out[start:stop] += rng.normal(0.0, float(amp) * 0.20, local_n).astype(np.float32) * window
    return out


def irregular_edge_field(n: int, amp: float, rng: np.random.Generator, n_edges: tuple[int, int] = (18, 48)) -> np.ndarray:
    """Many small irregular edges: high derivative, weak stable QRS template."""

    out = np.zeros(n, dtype=np.float32)
    count = int(rng.integers(n_edges[0], n_edges[1] + 1))
    for _ in range(count):
        pos = int(rng.integers(2, max(3, n - 8)))
        width = int(rng.integers(2, 9))
        stop = min(n, pos + width)
        sign = float(rng.choice([-1.0, 1.0]))
        shape = np.linspace(0.0, 1.0, stop - pos, dtype=np.float32)
        if rng.random() < 0.5:
            shape = shape[::-1]
        out[pos:stop] += sign * float(amp) * shape
        if stop < n:
            decay_len = min(n - stop, int(rng.integers(3, 14)))
            out[stop : stop + decay_len] += sign * float(amp) * np.linspace(1.0, 0.0, decay_len, dtype=np.float32)
    return out


def impulse_edge_field(n: int, amp: float, rng: np.random.Generator, n_edges: tuple[int, int] = (45, 130)) -> np.ndarray:
    """Sparse short edges/contact ticks: raises local derivative without a full noise wall."""

    out = np.zeros(n, dtype=np.float32)
    count = int(rng.integers(n_edges[0], n_edges[1] + 1))
    for _ in range(count):
        pos = int(rng.integers(2, max(3, n - 5)))
        width = int(rng.integers(1, 4))
        stop = min(n, pos + width)
        sign = float(rng.choice([-1.0, 1.0]))
        shape = np.ones(stop - pos, dtype=np.float32)
        if stop - pos > 1:
            shape *= np.hanning(stop - pos).astype(np.float32) + 0.25
        out[pos:stop] += sign * float(amp) * shape
    return out


def amplitude_modulated_midband(n: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / 125.0
    freq = float(rng.uniform(6.0, 13.0))
    jitter = np.cumsum(rng.normal(0.0, 0.018, n)).astype(np.float32)
    phase = 2 * np.pi * freq * t + jitter + float(rng.uniform(-np.pi, np.pi))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * float(rng.uniform(0.08, 0.35)) * t + float(rng.uniform(-np.pi, np.pi)))
    env = np.clip(env + rng.normal(0.0, 0.08, n).astype(np.float32), 0.15, 1.35)
    return (float(amp) * env * np.sin(phase)).astype(np.float32)


def apply_contact_like(y: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32).copy()
    n = out.shape[-1]
    events = int(rng.integers(1, 1 + max(1, int(1 + 4 * severity))))
    for _ in range(events):
        width = int(rng.integers(max(20, n // 50), max(30, n // 7)))
        start = int(rng.integers(0, max(1, n - width)))
        stop = min(n, start + width)
        if rng.random() < 0.5:
            level = float(np.nanmedian(out[start:stop] if stop > start else out))
            out[start:stop] = level + rng.normal(0.0, 0.005 + 0.01 * severity, stop - start).astype(np.float32)
        else:
            out[start:stop] += float(rng.uniform(-1.0, 1.0)) * (0.15 + severity) * max(float(np.nanstd(out)), 0.02)
    return out


def spectral_noise(n: int, rng: np.random.Generator, bands: list[tuple[float, float, float]]) -> np.ndarray:
    """Create deterministic-length colored noise from simple frequency bands."""

    freqs = np.fft.rfftfreq(n, d=1.0 / 125.0)
    spec = np.zeros(len(freqs), dtype=np.complex128)
    for lo, hi, gain in bands:
        mask = (freqs >= float(lo)) & (freqs <= float(hi))
        if not np.any(mask):
            continue
        phase = rng.uniform(-np.pi, np.pi, int(mask.sum()))
        mag = rng.rayleigh(scale=max(float(gain), 1e-6), size=int(mask.sum()))
        spec[mask] += mag * np.exp(1j * phase)
    x = np.fft.irfft(spec, n=n).astype(np.float32)
    x = x - float(np.nanmedian(x))
    sd = float(np.nanstd(x))
    if sd > 1e-8:
        x = x / sd
    return x.astype(np.float32)


def quasi_periodic_bad_texture(
    n: int,
    target_rms: float,
    target_ptp: float,
    target_diff: float,
    target_base: float,
    rng: np.random.Generator,
    profile: str,
    variant: int = 0,
) -> np.ndarray:
    """BUT bad is often narrow-band/periodic contamination, not white noise.

    A small response sweep under the unified extractor showed that a noisy
    16-20Hz oscillation gives the right detector-disagreement behavior, while
    pure random noise keeps detector_agreement too high and looks visually
    unlike the BUT bad bands.
    """

    t = np.arange(n, dtype=np.float32) / 125.0
    if profile == "highfreq":
        # The BUT highfreq/detail subtype looks fast in the raw waveform, but
        # the unified band ratios are not a 30-45Hz island. Use low-amplitude
        # near-Nyquist vibration instead of a 18-24Hz / 36Hz wall.
        freq = float(rng.uniform(46.0, 58.0))
        sine_amp = float(target_rms) * float(rng.uniform(0.75, 1.20))
        noise_amp = float(target_rms) * float(rng.uniform(0.02, 0.08))
        edge_amp = float(target_rms) * float(rng.uniform(0.01, 0.04))
    elif profile == "baseline":
        freq = float(rng.uniform(12.0, 19.0))
        sine_amp = float(target_rms) * float(rng.uniform(0.42, 0.72))
        noise_amp = float(target_rms) * float(rng.uniform(0.28, 0.55))
        edge_amp = float(target_rms) * float(rng.uniform(0.08, 0.20))
    else:
        # Dense/detector/other bad in BUT still has event-like peaks and
        # periodicity, but the detectors disagree.  Build a jittered pseudo-QRS
        # train first, then contaminate it; this is closer to BUT than a
        # smooth sinusoid or pure white noise.  v90 keeps the v88 event-like
        # texture but removes the white-noise wall that inflated 30-45Hz.
        family = int(variant) % 6
        x = np.zeros(n, dtype=np.float32)
        period = float(rng.uniform(30.0, 92.0))
        pos = float(rng.uniform(8.0, 36.0))
        amp = float(target_rms) * float(rng.uniform(1.05, 2.20))
        if family in {0, 4}:
            width = int(rng.integers(4, 10))
            jitter_sd = 5.5
        elif family in {1, 5}:
            width = int(rng.integers(7, 15))
            jitter_sd = 6.5
        else:
            width = int(rng.integers(11, 24))
            jitter_sd = 8.0
        while pos < n - width - 2:
            p = int(np.clip(pos + rng.normal(0.0, jitter_sd), 2, n - width - 2))
            pulse = np.hanning(width).astype(np.float32)
            if rng.random() < 0.18:
                pulse = -pulse
            x[p : p + width] += amp * pulse
            pos += period + float(rng.normal(0.0, jitter_sd))
        if family in {0, 4}:
            freq = float(rng.uniform(15.0, 23.5))
            mid_bands = [(0.4, 5.0, 0.08), (5.0, 12.0, 0.24), (12.0, 20.0, 0.46), (20.0, 30.0, 0.24), (30.0, 45.0, 0.040)]
            noise_amp = float(target_rms) * float(rng.uniform(0.16, 0.36))
            edge_gain = float(rng.uniform(0.045, 0.125))
            edge_count = (12, 34)
        elif family in {1, 5}:
            freq = float(rng.uniform(11.0, 19.0))
            mid_bands = [(0.4, 5.0, 0.12), (5.0, 12.0, 0.36), (12.0, 18.0, 0.42), (18.0, 30.0, 0.14), (30.0, 45.0, 0.018)]
            noise_amp = float(target_rms) * float(rng.uniform(0.10, 0.28))
            edge_gain = float(rng.uniform(0.025, 0.085))
            edge_count = (8, 24)
        else:
            freq = float(rng.uniform(8.5, 15.5))
            mid_bands = [(0.4, 5.0, 0.18), (5.0, 12.0, 0.50), (12.0, 18.0, 0.18), (18.0, 30.0, 0.04), (30.0, 45.0, 0.006)]
            noise_amp = float(target_rms) * float(rng.uniform(0.04, 0.18))
            edge_gain = float(rng.uniform(0.010, 0.045))
            edge_count = (5, 16)
        phase_jitter = np.cumsum(rng.normal(0.0, 0.006 + 0.003 * (family in {1, 2, 5}), n)).astype(np.float32)
        x += float(target_rms) * float(rng.uniform(0.18, 0.48)) * np.sin(
            2 * np.pi * freq * t + phase_jitter + float(rng.uniform(-np.pi, np.pi))
        )
        x += spectral_noise(n, rng, mid_bands) * noise_amp
        if family == 4:
            x += float(target_rms) * float(rng.uniform(0.08, 0.20)) * rng.normal(0.0, 1.0, n).astype(np.float32)
            x += impulse_edge_field(n, amp=float(target_rms) * float(rng.uniform(0.07, 0.22)), rng=rng, n_edges=(55, 150))
        elif family == 0 and rng.random() < 0.65:
            x += impulse_edge_field(n, amp=float(target_rms) * float(rng.uniform(0.04, 0.14)), rng=rng, n_edges=(35, 110))
        x += irregular_edge_field(n, amp=float(target_rms) * edge_gain, rng=rng, n_edges=edge_count)
        x += lowfreq_component(n, (0.05 + 0.6 * target_base) * max(float(target_rms), 0.04), rng, "slow")
        x = x - float(np.nanmedian(x))
        cur_rms = max(float(np.sqrt(np.mean(x * x))), 1e-6)
        cur_ptp = max(float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1)), 1e-6)
        scale = 0.70 * float(target_rms) / cur_rms + 0.30 * float(target_ptp) / cur_ptp
        x = x * float(np.clip(scale, 0.02, 6.0))
        cur_diff = max(float(np.nanpercentile(np.abs(np.diff(x)), 95)), 1e-6)
        if cur_diff < 0.80 * float(target_diff):
            # v90 fixed the 30-45Hz explosion but made dense bad too smooth.
            # Add sparse local edges only when the target derivative requires
            # them, then preserve the target envelope.
            add_amp = float(np.clip((float(target_diff) - cur_diff) * (0.18 if family in {0, 4} else 0.12), 0.004, 0.14))
            x += irregular_edge_field(n, amp=add_amp, rng=rng, n_edges=(12, 42) if family in {0, 4} else (8, 24))
            if family in {0, 4}:
                x += impulse_edge_field(n, amp=float(np.clip(add_amp * 0.75, 0.004, 0.12)), rng=rng, n_edges=(30, 100))
            x = x - float(np.nanmedian(x))
            cur_rms = max(float(np.sqrt(np.mean(x * x))), 1e-6)
            cur_ptp = max(float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1)), 1e-6)
            scale = 0.65 * float(target_rms) / cur_rms + 0.35 * float(target_ptp) / cur_ptp
            x = x * float(np.clip(scale, 0.02, 6.0))
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    phase_jitter = np.cumsum(rng.normal(0.0, 0.010 if profile != "highfreq" else 0.004, n)).astype(np.float32)
    env = 0.82 + 0.18 * np.sin(2 * np.pi * float(rng.uniform(0.05, 0.22)) * t + float(rng.uniform(-np.pi, np.pi)))
    x = sine_amp * env * np.sin(2 * np.pi * freq * t + phase_jitter + float(rng.uniform(-np.pi, np.pi)))
    x = x.astype(np.float32)
    x += noise_amp * rng.normal(0.0, 1.0, n).astype(np.float32)
    if profile == "highfreq":
        # BUT highfreq/detail still has weak recurrent structure; without this
        # v88 collapsed periodicity even when the raw trace looked plausible.
        period = float(rng.uniform(42.0, 92.0))
        pos = float(rng.uniform(6.0, 36.0))
        while pos < n - 10:
            width = int(rng.integers(8, 18))
            p = int(np.clip(pos + rng.normal(0.0, 7.0), 2, n - width - 2))
            stop = min(n, p + width)
            x[p:stop] += float(target_rms) * float(rng.uniform(0.10, 0.28)) * np.hanning(stop - p).astype(np.float32)
            pos += period + float(rng.normal(0.0, 8.0))
    if profile in {"dense", "baseline", "highfreq"}:
        x += irregular_edge_field(n, amp=edge_amp, rng=rng, n_edges=(18, 50))
    if profile == "baseline":
        x += lowfreq_component(n, (0.20 + 2.0 * target_base) * max(float(target_rms), 0.04), rng, "mixed")
    elif profile == "dense":
        x += lowfreq_component(n, (0.05 + 0.6 * target_base) * max(float(target_rms), 0.04), rng, "slow")

    x = x - float(np.nanmedian(x))
    cur_rms = max(float(np.sqrt(np.mean(x * x))), 1e-6)
    cur_ptp = max(float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1)), 1e-6)
    scale = 0.70 * float(target_rms) / cur_rms + 0.30 * float(target_ptp) / cur_ptp
    x = x * float(np.clip(scale, 0.02, 6.0))

    # If derivative/detail is still too weak, add the cheapest waveform fact
    # that raises it: small stochastic edges.  This is bounded so highfreq does
    # not become a white-noise wall again.
    cur_diff = max(float(np.nanpercentile(np.abs(np.diff(x)), 95)), 1e-6)
    if cur_diff < 0.85 * float(target_diff):
        add_amp = float(np.clip((float(target_diff) - cur_diff) * 0.10, 0.002, 0.055))
        x += irregular_edge_field(n, amp=add_amp, rng=rng, n_edges=(10, 32))
        x = x - float(np.nanmedian(x))
        cur_rms = max(float(np.sqrt(np.mean(x * x))), 1e-6)
        x = x * float(np.clip(float(target_rms) / cur_rms, 0.05, 4.0))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def stationary_bad_texture(
    n: int,
    target_rms: float,
    target_ptp: float,
    target_diff: float,
    target_base: float,
    rng: np.random.Generator,
    profile: str,
) -> np.ndarray:
    """Dense BUT bad is closer to stationary corruption than pseudo-QRS trains."""

    if profile == "highfreq":
        bands = [(4.0, 10.0, 0.18), (10.0, 22.0, 0.50), (22.0, 45.0, 0.24)]
        edge_count = (25, 70)
        edge_scale = 0.10
    else:
        bands = [(3.0, 8.0, 0.16), (8.0, 18.0, 0.62), (18.0, 34.0, 0.36), (34.0, 45.0, 0.05)]
        edge_count = (45, 130)
        edge_scale = 0.16
    out = spectral_noise(n, rng, bands)
    out += irregular_edge_field(
        n,
        amp=float(np.clip(max(target_diff, 0.04) * edge_scale, 0.006, 0.090)),
        rng=rng,
        n_edges=edge_count,
    )
    if float(target_base) > 0.06:
        out += lowfreq_component(n, float(np.clip(target_base, 0.0, 0.12)) * max(target_rms, 0.04), rng, "slow")
    out = out - float(np.nanmedian(out))
    for _ in range(3):
        cur_diff = max(float(np.nanpercentile(np.abs(np.diff(out)), 95)), 1e-6)
        if cur_diff < 0.96 * float(target_diff):
            out += spectral_noise(n, rng, [(12.0, 28.0, 0.7), (28.0, 45.0, 0.10)]) * float(
                np.clip((float(target_diff) - cur_diff) * 0.34, 0.004, 0.12)
            )
        elif cur_diff > 1.18 * float(target_diff):
            out = (0.55 * out + 0.45 * moving_average(out, int(rng.choice([3, 5, 7])))).astype(np.float32)
        else:
            break
    out = out - float(np.nanmedian(out))
    cur_rms = max(float(np.sqrt(np.mean(out * out))), 1e-6)
    cur_ptp = max(float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)), 1e-6)
    scale = 0.60 * float(target_rms) / cur_rms + 0.40 * float(target_ptp) / cur_ptp
    return np.nan_to_num(out * float(np.clip(scale, 0.02, 6.0)), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def contact_reset_bad_texture(carrier: np.ndarray, anchor: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """BUT contact/reset bad keeps visible QRS plus slow reset-like structure."""

    out = np.asarray(carrier, dtype=np.float32).reshape(-1).copy()
    n = out.shape[-1]
    out = out - float(np.nanmedian(out))
    target_rms = max(safe_value(anchor, "raw_rms", safe_value(anchor, "rms", 0.12)), 1e-5)
    target_ptp = max(safe_value(anchor, "raw_ptp_p99_p01", safe_value(anchor, "ptp_p99_p01", 0.65)), 1e-5)
    target_base = max(safe_value(anchor, "baseline_step", 0.08), 0.0)
    target_flat = max(safe_value(anchor, "flatline_ratio", 0.04), 0.0)
    target_contact = max(safe_value(anchor, "contact_loss_win_ratio", 0.04), 0.0)

    low = moving_average(out, int(rng.choice([17, 23, 31])))
    detail = out - low
    out = low + float(rng.uniform(1.35, 2.50)) * detail
    out += lowfreq_component(n, (0.22 + 1.75 * target_base) * max(target_rms, 0.04), rng, "mixed")
    for _ in range(int(rng.integers(2, 5))):
        width = int(rng.integers(max(30, n // 24), max(44, n // 8)))
        start = int(rng.integers(0, max(1, n - width)))
        stop = min(n, start + width)
        local = moving_average(out, int(rng.choice([25, 35, 49])))
        blend = float(np.clip(0.25 + target_flat + target_contact, 0.30, 0.74))
        offset = float(rng.normal(0.0, 0.18 * max(target_rms, 0.04)))
        out[start:stop] = ((1.0 - blend) * out[start:stop] + blend * (local[start:stop] + offset)).astype(np.float32)
        if rng.random() < 0.75:
            p = int(np.clip(start + rng.integers(0, max(1, width)), 2, n - 3))
            out[p:] += float(rng.normal(0.0, 0.20 * max(target_rms, 0.04)))
    out += irregular_edge_field(
        n,
        amp=float(np.clip(0.020 + 0.10 * max(target_rms, 0.04), 0.006, 0.08)),
        rng=rng,
        n_edges=(5, 18),
    )
    out = derivative_limit(out, anchor, rng)
    out = match_envelope(out, anchor)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def synthesize_bad_like_but(carrier: np.ndarray, anchor: pd.Series, subtype: str, rng: np.random.Generator, variant: int = 0) -> np.ndarray:
    """Generate BUT-like poor/bad morphology instead of preserving PTB QRS.

    The current BUT bad reference is mostly continuous contaminated morphology,
    not clean PTB beats with sparse bursts.  This branch deliberately suppresses
    carrier morphology and uses the BUT anchor to set amplitude, baseline,
    detail, and contact/reset severity.
    """

    base = np.asarray(carrier, dtype=np.float32).reshape(-1)
    n = base.shape[-1]
    target_rms = max(safe_value(anchor, "raw_rms", safe_value(anchor, "rms", 0.12)), 1e-5)
    target_ptp = max(safe_value(anchor, "raw_ptp_p99_p01", safe_value(anchor, "ptp_p99_p01", 0.65)), 1e-5)
    target_base = max(safe_value(anchor, "baseline_step", 0.0), 0.0)
    target_detail = max(safe_value(anchor, "non_qrs_diff_p95", safe_value(anchor, "raw_diff_abs_p95", 0.08)), 0.0)
    target_raw_diff = max(safe_value(anchor, "raw_diff_abs_p95", target_detail), 0.0)
    target_hf = max(safe_value(anchor, "band_30_45", 0.0), 0.0)
    target_flat = max(safe_value(anchor, "flatline_ratio", 0.0), 0.0)
    target_contact = max(safe_value(anchor, "contact_loss_win_ratio", 0.0), 0.0)

    if subtype in {"bad_dense_right_island", "bad_detector_template_disagree", "bad_other_boundary"}:
        return stationary_bad_texture(n, target_rms, target_ptp, max(target_detail, target_raw_diff), target_base, rng, "dense")
    if subtype == "bad_contact_reset_flatline":
        return contact_reset_bad_texture(carrier, anchor, rng)
    if subtype in {"bad_baseline_wander_lowfreq", "bad_low_qrs_visibility"}:
        return quasi_periodic_bad_texture(n, target_rms, target_ptp, target_detail, target_base, rng, "baseline", variant=variant)
    if subtype == "bad_highfreq_detail_noise":
        return stationary_bad_texture(n, target_rms, target_ptp, target_detail, target_base, rng, "highfreq")

    if subtype == "bad_highfreq_detail_noise":
        # In the unified extractor the BUT "highfreq detail" subtype is not a
        # huge 15-45Hz power island; it is mostly continuous detail/derivative
        # corruption with surprisingly low 15-30/30-45 band ratios.  Keep
        # energy in 5-15Hz and avoid the old synthetic high-frequency wall.
        bands = [(0.4, 5.0, 0.25), (5.0, 12.0, 0.58), (12.0, 22.0, 0.020), (22.0, 45.0, 0.004)]
    elif subtype == "bad_baseline_wander_lowfreq":
        bands = [(0.2, 1.2, 0.85), (1.2, 5.0, 0.35), (5.0, 20.0, 0.20)]
    elif subtype == "bad_contact_reset_flatline":
        bands = [(0.2, 1.5, 0.55), (1.5, 8.0, 0.22), (8.0, 25.0, 0.15)]
    else:
        bands = [(0.4, 5.0, 0.35), (5.0, 15.0, 0.65), (15.0, 32.0, 0.28 + 1.2 * target_hf)]
    out = spectral_noise(n, rng, bands)
    # Keep only a faint PTB residual so this remains PTB-derived but does not
    # look like clean QRS with added bursts.
    residual = moving_average(base - float(np.nanmedian(base)), int(rng.choice([21, 31, 41, 55])))
    res_sd = max(float(np.nanstd(residual)), 1e-6)
    out = out + 0.035 * residual / res_sd
    out += lowfreq_component(n, (0.15 + 1.8 * target_base) * max(target_rms, 0.04), rng, "mixed")
    if subtype in {"bad_detector_template_disagree", "bad_dense_right_island", "bad_other_boundary"}:
        # Low-amplitude pseudo events create detector instability without
        # reintroducing clean recurring QRS morphology.
        for _ in range(int(rng.integers(2, 7))):
            pos = int(rng.integers(6, max(7, n - 12)))
            width = int(rng.integers(3, 9))
            stop = min(n, pos + width)
            out[pos:stop] += float(rng.uniform(-0.9, 0.9)) * np.hanning(stop - pos).astype(np.float32)
        out += amplitude_modulated_midband(n, amp=0.22 + 0.35 * max(target_rms, 0.05), rng=rng)
        out += irregular_edge_field(n, amp=0.035 + 0.11 * max(target_rms, 0.05), rng=rng, n_edges=(20, 55))
    elif subtype in {"bad_baseline_wander_lowfreq", "bad_low_qrs_visibility"}:
        out += irregular_edge_field(n, amp=0.020 + 0.075 * max(target_rms, 0.05), rng=rng, n_edges=(10, 28))
    elif subtype == "bad_highfreq_detail_noise":
        # Step/reset-like edges better match the unified BUT highfreq subtype
        # than sinusoidal high-frequency energy.
        out += irregular_edge_field(n, amp=0.040 + 0.16 * max(target_rms, 0.05), rng=rng, n_edges=(24, 60))
    if subtype == "bad_contact_reset_flatline" or target_flat + target_contact > 0.02:
        out = apply_contact_like(out, float(np.clip(target_flat + target_contact + 0.18, 0.05, 1.0)), rng)
    # Match coarse derivative strength before envelope; this avoids a pure
    # smooth line for low-QRS bad while keeping high-frequency subtype distinct.
    cur_diff = max(float(np.nanpercentile(np.abs(np.diff(out)), 95)), 1e-6)
    desired_diff = max(float(target_detail), 0.01)
    out = out * float(np.clip(desired_diff / cur_diff, 0.15, 4.0))
    out = out - float(np.nanmedian(out))
    cur_rms = max(float(np.sqrt(np.mean(out * out))), 1e-6)
    cur_ptp = max(float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)), 1e-6)
    scale = 0.72 * target_rms / cur_rms + 0.28 * target_ptp / cur_ptp
    out = out * float(np.clip(scale, 0.03, 5.0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def synthesize_good_hard_flat_qrs(carrier: np.ndarray, anchor: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Hard good in the unified extractor is strong-QRS plus baseline/flatline.

    The old subtype name says lowqrs, but the recomputed BUT target has
    qrs_visibility saturated at 2 and qrs_band_ratio near 10.  This transform
    keeps/boosts carrier QRS evidence and adds smooth flatline/baseline
    contamination without making it medium/bad.
    """

    out = np.asarray(carrier, dtype=np.float32).reshape(-1).copy()
    n = out.shape[-1]
    out = out - float(np.nanmedian(out))
    target_rms = max(safe_value(anchor, "raw_rms", safe_value(anchor, "rms", 0.25)), 1e-5)
    target_ptp = max(safe_value(anchor, "raw_ptp_p99_p01", safe_value(anchor, "ptp_p99_p01", 1.5)), 1e-5)
    target_base = max(safe_value(anchor, "baseline_step", 0.25), 0.0)
    target_flat = max(safe_value(anchor, "flatline_ratio", 0.10), 0.0)
    low = moving_average(out, int(rng.choice([11, 15, 21])))
    detail = out - low
    # Preserve sharp repeated QRS; this is what BUT hard-good actually shows in
    # the unified qrs features.
    out = low + float(rng.uniform(1.85, 3.60)) * detail
    # A tiny regular impulse reinforcement raises qrs_band_ratio/visibility
    # without copying any BUT morphology.
    step = int(rng.integers(72, 105))
    offset = int(rng.integers(8, max(9, step)))
    for pos in range(offset, n - 8, step):
        width = int(rng.integers(4, 9))
        stop = min(n, pos + width)
        out[pos:stop] += float(rng.uniform(0.25, 0.75)) * max(target_rms, 0.04) * np.hanning(stop - pos).astype(np.float32)
    out += lowfreq_component(n, (0.10 + 1.2 * target_base) * max(target_rms, 0.04), rng, "mixed")
    # Add plateau/flat segments through smoothing, not zeroing, so the label
    # remains good-like while flatline_ratio/basSQI move toward BUT.
    for _ in range(int(rng.integers(2, 5))):
        width = int(rng.integers(max(24, n // 20), max(36, n // 5)))
        start = int(rng.integers(0, max(1, n - width)))
        stop = min(n, start + width)
        local = moving_average(out, int(rng.choice([25, 35, 51])))
        blend = float(np.clip(0.22 + 1.8 * target_flat, 0.22, 0.70))
        out[start:stop] = ((1.0 - blend) * out[start:stop] + blend * local[start:stop]).astype(np.float32)
    out = out - float(np.nanmedian(out))
    cur_rms = max(float(np.sqrt(np.mean(out * out))), 1e-6)
    cur_ptp = max(float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)), 1e-6)
    scale = 0.55 * target_rms / cur_rms + 0.45 * target_ptp / cur_ptp
    out = out * float(np.clip(scale, 0.04, 5.0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def match_envelope(y: np.ndarray, anchor: pd.Series) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32).copy()
    out = out - float(np.nanmedian(out))
    cur_rms = max(float(np.sqrt(np.mean(np.square(out)))), 1e-6)
    cur_ptp = max(float(np.nanpercentile(out, 99) - np.nanpercentile(out, 1)), 1e-6)
    target_rms = max(safe_value(anchor, "raw_rms", safe_value(anchor, "rms", cur_rms)), 1e-5)
    target_ptp = max(safe_value(anchor, "raw_ptp_p99_p01", safe_value(anchor, "ptp_p99_p01", cur_ptp)), 1e-5)
    scale = 0.62 * target_rms / cur_rms + 0.38 * target_ptp / cur_ptp
    return (out * float(np.clip(scale, 0.015, 6.0))).astype(np.float32)


def derivative_limit(y: np.ndarray, anchor: pd.Series, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32).copy()
    target = max(safe_value(anchor, "raw_diff_abs_p95", safe_value(anchor, "non_qrs_diff_p95", 0.04)), 1e-4)
    for _ in range(4):
        cur = float(np.nanpercentile(np.abs(np.diff(out)), 95)) if out.size > 1 else 0.0
        if cur <= max(1.20 * target, target + 0.006):
            break
        low = moving_average(out, int(rng.choice([5, 7, 9, 13])))
        out = (0.35 * out + 0.65 * low).astype(np.float32)
    return out


def transform_to_anchor(
    carrier: np.ndarray,
    anchor: pd.Series,
    cls: str,
    subtype: str,
    rng: np.random.Generator,
    variant: int,
) -> np.ndarray:
    """Aggressively transport PTB carrier toward one BUT feature anchor."""

    if cls == "bad":
        return synthesize_bad_like_but(carrier, anchor, subtype, rng, variant=variant)
    if cls == "good" and subtype == "good_hard_baseline_lowqrs":
        return synthesize_good_hard_flat_qrs(carrier, anchor, rng)

    out = np.asarray(carrier, dtype=np.float32).reshape(-1).copy()
    n = out.shape[-1]
    out = out - float(np.nanmedian(out))
    target_rms = max(safe_value(anchor, "raw_rms", safe_value(anchor, "rms", 0.05)), 1e-5)
    target_base = max(safe_value(anchor, "baseline_step", 0.0), 0.0)
    target_qrs = max(safe_value(anchor, "qrs_visibility", 0.0), 0.0)
    target_det = max(safe_value(anchor, "detector_agreement", 0.0), 0.0)
    target_bas = safe_value(anchor, "sqi_basSQI", 1.0)
    target_flat = max(safe_value(anchor, "flatline_ratio", 0.0), 0.0)
    target_contact = max(safe_value(anchor, "contact_loss_win_ratio", 0.0), 0.0)
    target_detail = max(safe_value(anchor, "non_qrs_diff_p95", safe_value(anchor, "raw_diff_abs_p95", 0.02)), 0.0)
    target_hf = max(safe_value(anchor, "band_30_45", 0.0), 0.0)
    target_entropy = max(safe_value(anchor, "amplitude_entropy", 0.0), 0.0)

    out = match_envelope(out, anchor)
    low = moving_average(out, int(rng.choice([15, 21, 31, 41, 55])))
    detail = out - low
    # Make QRS/detail evidence a target-controlled quantity rather than a PTB
    # carrier shortcut.  The transform is intentionally stronger than v80.
    qrs_gain = float(np.clip(0.03 + 0.90 * target_qrs + 0.20 * target_det, 0.025, 1.20))
    if cls == "medium" and subtype == "medium_hard_baseline_lowqrs":
        # The subtype name is historical.  Under the unified extractor the BUT
        # target still has substantial QRS-band evidence, so suppressing QRS
        # here creates the persistent v87/v88 qrs_band_ratio gap.
        qrs_gain *= float(rng.uniform(1.05, 1.75))
    elif cls == "bad" or "low_qrs" in subtype:
        qrs_gain *= float(rng.uniform(0.35, 0.80))
    elif cls == "good":
        qrs_gain *= float(rng.uniform(0.75, 1.15))
    else:
        qrs_gain *= float(rng.uniform(0.55, 1.05))
    out = low + qrs_gain * detail

    base_amp = (0.10 + 1.35 * target_base + 0.20 * max(0.0, 0.92 - target_bas)) * target_rms
    if "baseline" in subtype:
        base_amp *= 1.8
    elif cls == "bad":
        base_amp *= 1.25
    elif cls == "good":
        base_amp *= 0.65
    out += lowfreq_component(n, base_amp * float(rng.uniform(0.65, 1.55)), rng, "slow" if target_bas > 0.85 else "mixed")

    detail_amp = (0.03 + 0.45 * target_detail + 0.22 * target_hf + 0.015 * target_entropy) * max(target_rms, 0.02)
    if "highfreq" in subtype or "detail" in subtype:
        detail_amp *= 1.8
    elif cls == "bad":
        detail_amp *= 1.25
    elif cls == "good":
        detail_amp *= 0.55
    if detail_amp > 1e-5:
        freq_hi = 42.0 if (target_hf > 0.08 and variant % 3 == 0) else 30.0
        out += band_bursts(
            n,
            detail_amp * float(rng.uniform(0.55, 1.65)),
            rng,
            freq_range=(8.0 if cls != "bad" else 6.0, freq_hi),
            count=(1, 4 if cls != "good" else 2),
        )

    if target_flat > 0.015 or target_contact > 0.015 or "contact" in subtype or rng.random() < 0.08:
        sev = float(np.clip(target_flat + target_contact + (0.20 if "contact" in subtype else 0.0), 0.02, 1.0))
        out = apply_contact_like(out, sev, rng)

    if cls == "bad" and target_det < 0.45:
        # Add small pseudo-peaks or missing-peak zones to damage detector
        # agreement without turning every bad row into pure white noise.
        for _ in range(int(rng.integers(1, 5))):
            pos = int(rng.integers(8, max(9, n - 12)))
            width = int(rng.integers(4, 15))
            stop = min(n, pos + width)
            pulse = np.hanning(stop - pos).astype(np.float32)
            out[pos:stop] += float(rng.uniform(-1.0, 1.0)) * (0.25 + 0.85 * target_rms) * pulse
    if cls in {"good", "medium"}:
        out = derivative_limit(out, anchor, rng)
    out = match_envelope(out, anchor)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass
class BuildResult:
    protocol_path: Path
    report_rows: list[dict[str, Any]]


def split_assignments(count: int, rng: np.random.Generator) -> list[str]:
    n_train = int(round(0.70 * count))
    n_val = int(round(0.15 * count))
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (count - n_train - n_val)
    rng.shuffle(splits)
    return splits


def sample_anchor_indices(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = frame.index.to_numpy(dtype=int)
    if len(idx) == 0:
        raise ValueError("cannot sample anchors from empty target frame")
    return rng.choice(idx, size=int(n), replace=len(idx) < int(n)).astype(int)


def select_best_candidates(
    anchors: pd.DataFrame,
    cand_feat: pd.DataFrame,
    group_ids: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    center, scale = robust_fit(feature_matrix(anchors, features))
    anchor_z = robust_z(feature_matrix(anchors, features), center, scale)
    cand_z = robust_z(feature_matrix(cand_feat, features), center, scale)
    weights = np.ones(len(features), dtype=np.float64)
    for i, name in enumerate(features):
        if name in {"sqi_basSQI", "qrs_visibility", "detector_agreement", "baseline_step", "non_qrs_diff_p95"}:
            weights[i] = 1.75
        elif name in {"qrs_band_ratio", "template_corr", "band_30_45", "amplitude_entropy", "flatline_ratio"}:
            weights[i] = 1.35
    selected: list[int] = []
    distances: list[float] = []
    for i in range(len(anchors)):
        mask = np.flatnonzero(group_ids == i)
        if len(mask) == 0:
            continue
        diff = (cand_z[mask] - anchor_z[i][None, :]) * weights[None, :]
        d = np.sqrt(np.nanmean(diff * diff, axis=1))
        best_local = int(mask[int(np.argmin(d))])
        selected.append(best_local)
        distances.append(float(np.min(d)))
    return np.asarray(selected, dtype=np.int64), np.asarray(distances, dtype=np.float64)


def candidate_count_for_class(args: argparse.Namespace, cls: str) -> int:
    if cls == "bad" and int(getattr(args, "bad_candidates_per_anchor", 0)) > 0:
        return int(args.bad_candidates_per_anchor)
    if cls == "medium" and int(getattr(args, "medium_candidates_per_anchor", 0)) > 0:
        return int(args.medium_candidates_per_anchor)
    return int(args.candidates_per_anchor)


def base_feature_weights(features: list[str]) -> np.ndarray:
    weights = np.ones(len(features), dtype=np.float64)
    for i, name in enumerate(features):
        if name in {
            "detector_agreement",
            "sqi_iSQI",
            "qrs_visibility",
            "qrs_band_ratio",
            "sqi_bSQI",
            "sqi_basSQI",
            "baseline_step",
            "non_qrs_diff_p95",
            "raw_diff_abs_p95",
        }:
            weights[i] = 2.25
        elif name in {
            "template_corr",
            "band_15_30",
            "band_30_45",
            "amplitude_entropy",
            "flatline_ratio",
            "wavelet_e0",
            "wavelet_e1",
            "wavelet_e2",
        }:
            weights[i] = 1.65
    return weights


def rowwise_select_from_z(
    anchor_z: np.ndarray,
    cand_z: np.ndarray,
    group_ids: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected: list[int] = []
    distances: list[float] = []
    all_dist = np.full(len(cand_z), np.inf, dtype=np.float64)
    for i in range(len(anchor_z)):
        mask = np.flatnonzero(group_ids == i)
        if len(mask) == 0:
            continue
        diff = (cand_z[mask] - anchor_z[i][None, :]) * weights[None, :]
        d = np.sqrt(np.nanmean(diff * diff, axis=1))
        all_dist[mask] = d
        best_local = int(mask[int(np.argmin(d))])
        selected.append(best_local)
        distances.append(float(np.min(d)))
    return np.asarray(selected, dtype=np.int64), np.asarray(distances, dtype=np.float64), all_dist


def fixed_sliced_wasserstein(x: np.ndarray, y: np.ndarray, dirs: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    q = np.linspace(0.0, 1.0, min(len(x), len(y)))
    vals: list[float] = []
    for v in dirs:
        vals.append(float(np.mean(np.abs(np.quantile(x @ v, q) - np.quantile(y @ v, q)))))
    return float(np.mean(vals))


def make_rff_reference(target_z: np.ndarray, rng: np.random.Generator, n_components: int = 160) -> tuple[np.ndarray, np.ndarray]:
    if len(target_z) < 3 or target_z.shape[1] == 0:
        return np.zeros((target_z.shape[1], n_components), dtype=np.float64), np.zeros(n_components, dtype=np.float64)
    ref = target_z
    if len(ref) > 500:
        ref = ref[rng.choice(np.arange(len(ref)), size=500, replace=False)]
    d2 = np.sum((ref[:, None, :] - ref[None, :, :]) ** 2, axis=2)
    med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    sigma = math.sqrt(max(med, 1e-6))
    w = rng.normal(0.0, 1.0 / sigma, size=(target_z.shape[1], int(n_components)))
    b = rng.uniform(0.0, 2.0 * np.pi, size=int(n_components))
    return w.astype(np.float64), b.astype(np.float64)


def rff_features(z: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    if z.shape[1] == 0:
        return np.zeros((len(z), len(b)), dtype=np.float64)
    return (math.sqrt(2.0 / max(len(b), 1)) * np.cos(z @ w + b[None, :])).astype(np.float64)


def distribution_score_factory(
    target_z: np.ndarray,
    cand_z: np.ndarray,
    selected_feature_weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Any, np.ndarray]:
    weights = selected_feature_weights.astype(np.float64)
    weights = weights / max(float(np.nanmean(weights)), 1e-6)
    target_w = target_z * weights[None, :]
    cand_w = cand_z * weights[None, :]
    dirs = rng.normal(size=(36, target_w.shape[1]))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
    rff_w, rff_b = make_rff_reference(target_w, rng, n_components=160)
    target_phi_mean = rff_features(target_w, rff_w, rff_b).mean(axis=0)
    cand_phi = rff_features(cand_w, rff_w, rff_b)

    def score(selected: np.ndarray) -> float:
        synth = cand_w[selected]
        phi_gap = cand_phi[selected].mean(axis=0) - target_phi_mean
        mmd = float(np.dot(phi_gap, phi_gap))
        qloss = quantile_loss(target_w, synth)
        sw = fixed_sliced_wasserstein(target_w, synth, dirs)
        med = float(np.nanmean(np.abs(np.nanmedian(target_w, axis=0) - np.nanmedian(synth, axis=0))))
        return float(1.00 * mmd + 0.34 * qloss + 0.22 * sw + 0.12 * med)

    return score, cand_w


def select_distribution_matched_candidates(
    anchors: pd.DataFrame,
    cand_feat: pd.DataFrame,
    group_ids: np.ndarray,
    features: list[str],
    rng: np.random.Generator,
    repair_rounds: int,
    repair_alternatives: int,
    repair_tries: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    center, scale = robust_fit(feature_matrix(anchors, features))
    anchor_x = feature_matrix(anchors, features)
    cand_x = feature_matrix(cand_feat, features)
    anchor_z = robust_z(anchor_x, center, scale)
    cand_z = robust_z(cand_x, center, scale)
    weights = base_feature_weights(features)
    selected, _, _ = rowwise_select_from_z(anchor_z, cand_z, group_ids, weights)

    # Boost features whose selected distribution is still far from the BUT
    # anchor distribution.  This is the first v84 change: selection is no
    # longer allowed to hide a bad global CDF behind a good average row score.
    if len(selected):
        q25 = np.nanpercentile(anchor_x, 25, axis=0)
        q75 = np.nanpercentile(anchor_x, 75, axis=0)
        iqr = np.maximum(q75 - q25, 1e-5)
        med_gap = (np.nanmedian(cand_x[selected], axis=0) - np.nanmedian(anchor_x, axis=0)) / iqr
        weights = weights * np.clip(1.0 + 0.45 * np.abs(med_gap), 1.0, 10.0)
    selected, distances, all_dist = rowwise_select_from_z(anchor_z, cand_z, group_ids, weights)

    info: dict[str, Any] = {
        "repair_attempts": 0,
        "repair_accepts": 0,
        "distribution_score_before": float("nan"),
        "distribution_score_after": float("nan"),
    }
    if repair_rounds <= 0 or len(selected) < 8:
        return selected, distances, info

    score_fn, _ = distribution_score_factory(anchor_z, cand_z, weights, rng)
    best_score = score_fn(selected)
    info["distribution_score_before"] = float(best_score)
    group_count = int(group_ids.max()) + 1 if len(group_ids) else 0
    group_options: list[np.ndarray] = []
    for i in range(group_count):
        mask = np.flatnonzero(group_ids == i)
        if len(mask) == 0:
            group_options.append(mask)
            continue
        order = np.argsort(all_dist[mask])
        group_options.append(mask[order[: max(1, int(repair_alternatives) + 1)]])

    group_order = np.argsort(-distances) if len(distances) else np.arange(group_count)
    max_attempts = int(max(0, repair_tries))
    attempts = 0
    accepts = 0
    for _round in range(int(max(0, repair_rounds))):
        improved = False
        for group_i in group_order:
            if attempts >= max_attempts:
                break
            if group_i >= len(group_options) or len(group_options[group_i]) <= 1:
                continue
            current = selected[group_i]
            for alt in group_options[group_i]:
                alt = int(alt)
                if alt == int(current):
                    continue
                attempts += 1
                trial = selected.copy()
                trial[group_i] = alt
                trial_score = score_fn(trial)
                if trial_score + 1e-6 < best_score:
                    selected = trial
                    best_score = float(trial_score)
                    accepts += 1
                    improved = True
                    break
                if attempts >= max_attempts:
                    break
            if attempts >= max_attempts:
                break
        if not improved or attempts >= max_attempts:
            break

    _, distances, _ = rowwise_select_from_z(anchor_z, cand_z, group_ids, weights)
    final_distances: list[float] = []
    for i, chosen in enumerate(selected):
        diff = (cand_z[int(chosen)] - anchor_z[i]) * weights
        final_distances.append(float(np.sqrt(np.nanmean(diff * diff))))
    info["repair_attempts"] = int(attempts)
    info["repair_accepts"] = int(accepts)
    info["distribution_score_after"] = float(best_score)
    return selected, np.asarray(final_distances, dtype=np.float64), info


def build_v81(args: argparse.Namespace) -> BuildResult:
    rng = np.random.default_rng(int(args.seed))
    but, but_x = load_protocol(Path(args.but_protocol))
    but = recompute_protocol_features(but, but_x, "BUT target")
    carrier, carrier_x = load_protocol(Path(args.carrier_protocol))
    but = but.copy()
    but["transport_subtype"] = V37.assign_but_subtypes(but)
    target_pool = but.loc[but["split"].astype(str).isin(["train", "val"])].copy()
    if target_pool.empty:
        raise RuntimeError("BUT target pool has no train/val rows")

    counts = V37.target_counts(int(args.total_per_class))
    out_rows: list[pd.DataFrame] = []
    out_signals: list[np.ndarray] = []
    selected_features: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    global_idx = 0

    print(f"{now()} building v81 transport protocol total_per_class={args.total_per_class}", flush=True)
    for cls in CLASS_ORDER:
        class_rows: list[dict[str, Any]] = []
        class_splits: list[str] = []
        for subtype, n_select in counts[cls].items():
            target_sub = target_pool.loc[
                target_pool["class_name"].astype(str).eq(cls) & target_pool["transport_subtype"].astype(str).eq(subtype)
            ].copy()
            if target_sub.empty:
                target_sub = target_pool.loc[target_pool["class_name"].astype(str).eq(cls)].copy()
            anchor_idx = sample_anchor_indices(target_sub, int(n_select), rng)
            anchors = target_sub.loc[anchor_idx].reset_index(drop=True).copy()
            anchors["transport_anchor_order"] = np.arange(len(anchors), dtype=int)

            cand_signals: list[np.ndarray] = []
            cand_meta: list[dict[str, Any]] = []
            group_ids: list[int] = []
            candidates_per_anchor = candidate_count_for_class(args, cls)
            carrier_indices = rng.choice(np.arange(len(carrier_x)), size=len(anchors) * int(candidates_per_anchor), replace=True)
            cursor = 0
            for anchor_order, (_, anchor) in enumerate(anchors.iterrows()):
                for c in range(int(candidates_per_anchor)):
                    src_i = int(carrier_indices[cursor])
                    cursor += 1
                    sig = transform_to_anchor(carrier_x[src_i], anchor, cls, subtype, rng, c)
                    cand_signals.append(sig)
                    src = carrier.iloc[src_i]
                    cand_meta.append(
                        {
                            "idx": len(cand_meta),
                            "source_idx": int(src.get("source_idx", src_i)),
                            "split": "train",
                            "y": CLASS_TO_INT[cls],
                            "class_name": cls,
                            "record_id": f"v81_ptb_{src.get('record_id', src_i)}",
                            "subject_id": str(src.get("subject_id", "ptbxl")),
                            "original_region": str(anchor.get("original_region", "")),
                            "ambiguous_type": str(anchor.get("ambiguous_type", "")),
                            "display_subtype": subtype,
                            "subtype_id": subtype,
                            "subtype_for_split": subtype,
                            "transport_subtype": subtype,
                            "clean_policy": "v81_feature_transport",
                            "generated_mode": "v81_feature_transport",
                            "target_anchor_idx": int(anchor.get("idx", anchor_order)),
                            "target_anchor_order": int(anchor_order),
                            "candidate_id": int(c),
                            "ptbxl_ecg_id": int(src.get("ptbxl_ecg_id", src_i)) if pd.notna(src.get("ptbxl_ecg_id", src_i)) else int(src_i),
                            "ptbxl_lead": str(src.get("ptbxl_lead", "I")),
                        }
                    )
                    group_ids.append(anchor_order)

            cand_x = np.stack(cand_signals).astype(np.float32)
            cand_frame = pd.DataFrame(cand_meta)
            print(
                f"{now()} recompute candidates cls={cls} subtype={subtype} anchors={len(anchors)} candidates={len(cand_frame)}",
                flush=True,
            )
            cand_feat = V37.recompute_full_features(cand_x, cand_frame.copy())
            cand_feat = add_consistent_transport_primitives(cand_feat, cand_x)
            for col in cand_feat.columns:
                if col not in cand_frame.columns:
                    cand_frame[col] = cand_feat[col].to_numpy()
                elif col in MATCH_FEATURES:
                    cand_frame[col] = cand_feat[col].to_numpy()

            features = [f for f in MATCH_FEATURES if f in anchors.columns or f in cand_frame.columns]
            selected, distances, repair_info = select_distribution_matched_candidates(
                anchors,
                cand_frame,
                np.asarray(group_ids, dtype=int),
                features,
                rng,
                repair_rounds=int(args.mmd_repair_rounds),
                repair_alternatives=int(args.mmd_repair_alternatives),
                repair_tries=int(args.mmd_repair_tries),
            )
            selected_x = cand_x[selected].astype(np.float32)
            selected_frame = cand_frame.iloc[selected].reset_index(drop=True).copy()
            selected_frame["transport_feature_distance"] = distances
            selected_frame["transport_match_features"] = len(features)
            selected_frame["transport_candidates_per_anchor"] = int(candidates_per_anchor)
            selected_frame["transport_stage"] = "distribution_mmd_herding" if int(args.mmd_repair_rounds) > 0 else "rowwise_feature_search"
            selected_frame["transport_subtype"] = subtype
            selected_frame["transport_repair_accepts"] = int(repair_info.get("repair_accepts", 0))
            selected_frame["transport_distribution_score_before"] = float(repair_info.get("distribution_score_before", np.nan))
            selected_frame["transport_distribution_score_after"] = float(repair_info.get("distribution_score_after", np.nan))

            splits = split_assignments(len(selected_frame), rng)
            selected_frame["split"] = splits
            selected_frame["idx"] = np.arange(global_idx, global_idx + len(selected_frame), dtype=int)
            global_idx += len(selected_frame)

            out_signals.extend([x for x in selected_x])
            out_rows.append(selected_frame)
            selected_features.append(selected_frame)
            audit_rows.append(
                {
                    "class_name": cls,
                    "subtype": subtype,
                    "target_rows_available": int(len(target_sub)),
                    "selected_rows": int(len(selected_frame)),
                    "candidate_rows": int(len(cand_frame)),
                    "candidates_per_anchor": int(candidates_per_anchor),
                    "distance_median": float(np.nanmedian(distances)),
                    "distance_p90": float(np.nanpercentile(distances, 90)),
                    "repair_attempts": int(repair_info.get("repair_attempts", 0)),
                    "repair_accepts": int(repair_info.get("repair_accepts", 0)),
                    "distribution_score_before": float(repair_info.get("distribution_score_before", np.nan)),
                    "distribution_score_after": float(repair_info.get("distribution_score_after", np.nan)),
                }
            )
            class_rows.extend(selected_frame[["idx"]].to_dict(orient="records"))
            class_splits.extend(splits)
        print(f"{now()} completed class={cls} rows={len(class_rows)}", flush=True)

    frame = pd.concat(out_rows, ignore_index=True)
    X = np.stack(out_signals).astype(np.float32)
    # Recompute once on the final selected protocol so persisted features match
    # the final row order and waveform exactly.
    print(f"{now()} recompute final selected v81 features rows={len(frame)}", flush=True)
    final_feat = V37.recompute_full_features(X, frame.copy())
    final_feat = add_consistent_transport_primitives(final_feat, X)
    for col in final_feat.columns:
        if col not in frame.columns or col in MATCH_FEATURES:
            frame[col] = final_feat[col].to_numpy()

    out = ANALYSIS_DIR / "event_xds_aligned_v81_feature_transport" / str(args.protocol_name)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(long_path(out / "original_region_atlas.csv"), index=False)
    frame.to_csv(long_path(out / "metadata.csv"), index=False)
    np.savez_compressed(long_path(out / "signals.npz"), X=X)
    pd.DataFrame(audit_rows).to_csv(long_path(out / "transport_selection_summary.csv"), index=False)
    summary = {
        "created_at": now(),
        "protocol_name": str(args.protocol_name),
        "method": "v81 rowwise BUT feature target transport from raw PTB-XL lead-I carriers",
        "but_protocol": str(Path(args.but_protocol)),
        "carrier_protocol": str(Path(args.carrier_protocol)),
        "rows": int(len(frame)),
        "total_per_class": int(args.total_per_class),
        "candidates_per_anchor": int(args.candidates_per_anchor),
        "medium_candidates_per_anchor": int(getattr(args, "medium_candidates_per_anchor", 0)),
        "bad_candidates_per_anchor": int(getattr(args, "bad_candidates_per_anchor", 0)),
        "mmd_repair_rounds": int(args.mmd_repair_rounds),
        "mmd_repair_alternatives": int(args.mmd_repair_alternatives),
        "mmd_repair_tries": int(args.mmd_repair_tries),
        "class_counts": frame["class_name"].value_counts().to_dict(),
        "subtype_counts": frame.groupby(["class_name", "transport_subtype"]).size().reset_index(name="n").to_dict(orient="records"),
        "seed": int(args.seed),
        "contract": "BUT waveforms are not copied; BUT train+val features are used as distribution anchors only.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{now()} wrote v81 protocol {out}", flush=True)
    return BuildResult(out, audit_rows)


def subsample_rows(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = rng.choice(np.arange(len(x)), size=int(n), replace=False)
    return x[idx]


def rbf_mmd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_n: int = 700) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x = subsample_rows(np.asarray(x, dtype=np.float64), max_n, rng)
    y = subsample_rows(np.asarray(y, dtype=np.float64), max_n, rng)
    z = np.vstack([x, y])
    if len(z) > 900:
        z_ref = subsample_rows(z, 900, rng)
    else:
        z_ref = z
    d_ref = np.sum((z_ref[:, None, :] - z_ref[None, :, :]) ** 2, axis=2)
    med = float(np.median(d_ref[d_ref > 0])) if np.any(d_ref > 0) else 1.0
    bands = [0.25 * med, 0.5 * med, med, 2.0 * med, 4.0 * med]
    val = 0.0
    for bw in bands:
        gamma = 1.0 / max(float(bw), 1e-6)
        kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
        kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2))
        kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2))
        val += float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    return float(val / len(bands))


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_proj: int = 128, max_n: int = 900) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x = subsample_rows(np.asarray(x, dtype=np.float64), max_n, rng)
    y = subsample_rows(np.asarray(y, dtype=np.float64), max_n, rng)
    d = x.shape[1]
    dirs = rng.normal(size=(int(n_proj), d))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
    vals = []
    q = np.linspace(0.0, 1.0, min(len(x), len(y)))
    for v in dirs:
        px = np.quantile(x @ v, q)
        py = np.quantile(y @ v, q)
        vals.append(float(np.mean(np.abs(px - py))))
    return float(np.mean(vals))


def quantile_loss(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    qs = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    qx = np.quantile(x, qs, axis=0)
    qy = np.quantile(y, qs, axis=0)
    return float(np.mean(np.abs(qx - qy)))


def domain_auc(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    if len(x) < 20 or len(y) < 20:
        return float("nan")
    n = min(len(x), len(y), 1200)
    x = subsample_rows(x, n, rng)
    y = subsample_rows(y, n, rng)
    z = np.vstack([x, y])
    labels = np.asarray([0] * len(x) + [1] * len(y), dtype=int)
    try:
        xtr, xte, ytr, yte = train_test_split(z, labels, test_size=0.35, stratify=labels, random_state=int(rng.integers(1, 1_000_000)))
        clf = LogisticRegression(max_iter=800, C=0.5, solver="lbfgs")
        clf.fit(xtr, ytr)
        score = clf.predict_proba(xte)[:, 1]
        auc = float(roc_auc_score(yte, score))
        return float(max(auc, 1.0 - auc))
    except Exception:
        return float("nan")


def pca_density_overlap(target_z: np.ndarray, synth_z: np.ndarray) -> float:
    if len(target_z) < 4 or len(synth_z) < 4:
        return float("nan")
    pca = PCA(n_components=2, random_state=19)
    pc_t = pca.fit_transform(target_z)
    pc_s = pca.transform(synth_z)
    lo = np.nanpercentile(pc_t, 1, axis=0)
    hi = np.nanpercentile(pc_t, 99, axis=0)
    if np.any((hi - lo) <= 1e-6):
        return float("nan")
    bins = [np.linspace(lo[i], hi[i], 26) for i in range(2)]
    ht, _, _ = np.histogram2d(pc_t[:, 0], pc_t[:, 1], bins=bins)
    hs, _, _ = np.histogram2d(pc_s[:, 0], pc_s[:, 1], bins=bins)
    ht = ht / max(float(ht.sum()), 1e-12)
    hs = hs / max(float(hs.sum()), 1e-12)
    return float(np.minimum(ht, hs).sum())


def ensure_subtype(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "transport_subtype" not in out.columns:
        out["transport_subtype"] = V37.assign_but_subtypes(out)
    out["transport_subtype"] = out["transport_subtype"].fillna("").astype(str)
    return out


def audit_pair(
    but: pd.DataFrame,
    ptb: pd.DataFrame,
    tag: str,
    out_dir: Path,
    baseline: pd.DataFrame | None = None,
    rng_seed: int = 20260681,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(rng_seed))
    but = ensure_subtype(but)
    ptb = ensure_subtype(ptb)
    target = but.loc[but["split"].astype(str).isin(["train", "val"])].copy()
    features = [f for f in MATCH_FEATURES if f in target.columns or f in ptb.columns]
    rows: list[dict[str, Any]] = []
    feat_rows: list[dict[str, Any]] = []
    for cls in CLASS_ORDER:
        subtypes = V37.SUBTYPES_BY_CLASS[cls]
        for subtype in subtypes:
            b = target.loc[target["class_name"].astype(str).eq(cls) & target["transport_subtype"].astype(str).eq(subtype)]
            p = ptb.loc[ptb["class_name"].astype(str).eq(cls) & ptb["transport_subtype"].astype(str).eq(subtype)]
            if len(p) == 0:
                p = ptb.loc[ptb["class_name"].astype(str).eq(cls)]
            if len(b) < 5 or len(p) < 5:
                continue
            center, scale = robust_fit(feature_matrix(b, features))
            bz = robust_z(feature_matrix(b, features), center, scale)
            pz = robust_z(feature_matrix(p, features), center, scale)
            rows.append(
                {
                    "tag": tag,
                    "class_name": cls,
                    "subtype": subtype,
                    "but_n": int(len(b)),
                    "ptb_n": int(len(p)),
                    "rbf_mmd": rbf_mmd(bz, pz, rng),
                    "sliced_wasserstein": sliced_wasserstein(bz, pz, rng),
                    "quantile_loss": quantile_loss(bz, pz),
                    "domain_auc": domain_auc(bz, pz, rng),
                    "pca_density_overlap": pca_density_overlap(bz, pz),
                }
            )
            bx = feature_matrix(b, features)
            px = feature_matrix(p, features)
            q25 = np.nanpercentile(bx, 25, axis=0)
            q75 = np.nanpercentile(bx, 75, axis=0)
            iqr = np.maximum(q75 - q25, 1e-5)
            med_gap = (np.nanmedian(px, axis=0) - np.nanmedian(bx, axis=0)) / iqr
            for name, gap in zip(features, med_gap):
                feat_rows.append({"tag": tag, "class_name": cls, "subtype": subtype, "feature": name, "median_gap_iqr": float(gap)})

    metric = pd.DataFrame(rows)
    feat_gap = pd.DataFrame(feat_rows)
    if baseline is not None and not baseline.empty and not metric.empty:
        base_cols = ["class_name", "subtype"]
        merged = metric.merge(
            baseline[base_cols + ["rbf_mmd", "sliced_wasserstein", "quantile_loss", "domain_auc"]].rename(
                columns={
                    "rbf_mmd": "baseline_rbf_mmd",
                    "sliced_wasserstein": "baseline_sliced_wasserstein",
                    "quantile_loss": "baseline_quantile_loss",
                    "domain_auc": "baseline_domain_auc",
                }
            ),
            on=base_cols,
            how="left",
        )
        for col in ["rbf_mmd", "sliced_wasserstein", "quantile_loss", "domain_auc"]:
            bcol = f"baseline_{col}"
            merged[f"{col}_reduction"] = 1.0 - merged[col] / merged[bcol].replace(0, np.nan)
        metric = merged
    out_dir.mkdir(parents=True, exist_ok=True)
    metric.to_csv(long_path(out_dir / f"{tag}_distribution_metrics.csv"), index=False)
    feat_gap.to_csv(long_path(out_dir / f"{tag}_feature_median_gap.csv"), index=False)
    return metric, feat_gap


def plot_shared_pca(but: pd.DataFrame, ptb: pd.DataFrame, out_dir: Path, tag: str) -> None:
    but = ensure_subtype(but)
    ptb = ensure_subtype(ptb)
    target = but.loc[but["split"].astype(str).isin(["train", "val"])].copy()
    features = [f for f in MATCH_FEATURES if f in target.columns or f in ptb.columns]
    center, scale = robust_fit(feature_matrix(target, features))
    bz = robust_z(feature_matrix(but, features), center, scale)
    pz = robust_z(feature_matrix(ptb, features), center, scale)
    pca = PCA(n_components=2, random_state=19)
    but_pc = pca.fit_transform(bz)
    ptb_pc = pca.transform(pz)
    b = but.copy()
    p = ptb.copy()
    b["pc1_v81"], b["pc2_v81"] = but_pc[:, 0], but_pc[:, 1]
    p["pc1_v81"], p["pc2_v81"] = ptb_pc[:, 0], ptb_pc[:, 1]
    rng = np.random.default_rng(19)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    for ax, cls in zip(axes, CLASS_ORDER):
        bb = b.loc[b["class_name"].astype(str).eq(cls) & b["split"].astype(str).isin(["train", "val"])]
        pp = p.loc[p["class_name"].astype(str).eq(cls)]
        if len(bb) > 2200:
            bb = bb.iloc[rng.choice(np.arange(len(bb)), size=2200, replace=False)]
        if len(pp) > 2200:
            pp = pp.iloc[rng.choice(np.arange(len(pp)), size=2200, replace=False)]
        ax.scatter(bb["pc1_v81"], bb["pc2_v81"], s=8, alpha=0.23, color=TOKENS["but"], label="BUT train+val")
        ax.scatter(pp["pc1_v81"], pp["pc2_v81"], s=10, alpha=0.45, color=TOKENS["ptb"], marker="x", linewidths=0.45, label="PTB synthetic")
        ax.set_title(cls)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, color=TOKENS["grid"], lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(f"{tag}: shared PCA fitted on BUT train+val 47-feature space", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, out_dir / f"{tag}_shared_pca")

    # Presentation-only view: keep the exact same PCA coordinates and data, and
    # only adjust axis limits to the class-wise overlap window. Points outside
    # the view are not moved or winsorized; this is a pure visual zoom.
    rng = np.random.default_rng(19)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    for ax, cls in zip(axes, CLASS_ORDER):
        bb = b.loc[b["class_name"].astype(str).eq(cls) & b["split"].astype(str).isin(["train", "val"])]
        pp = p.loc[p["class_name"].astype(str).eq(cls)]
        but_x = bb["pc1_v81"].to_numpy(dtype=float)
        but_y = bb["pc2_v81"].to_numpy(dtype=float)
        but_x = but_x[np.isfinite(but_x)]
        but_y = but_y[np.isfinite(but_y)]
        ptb_x = pp["pc1_v81"].to_numpy(dtype=float)
        ptb_y = pp["pc2_v81"].to_numpy(dtype=float)
        ptb_x = ptb_x[np.isfinite(ptb_x)]
        ptb_y = ptb_y[np.isfinite(ptb_y)]
        if len(but_x) and len(but_y) and len(ptb_x) and len(ptb_y):
            bxlo, bxhi = np.percentile(but_x, [2, 98])
            bylo, byhi = np.percentile(but_y, [2, 98])
            pxlo, pxhi = np.percentile(ptb_x, [2, 98])
            pylo, pyhi = np.percentile(ptb_y, [2, 98])
            xlo, xhi = max(bxlo, pxlo), min(bxhi, pxhi)
            ylo, yhi = max(bylo, pylo), min(byhi, pyhi)
            if xlo >= xhi:
                xlo, xhi = np.percentile(np.concatenate([but_x, ptb_x]), [10, 90])
            if ylo >= yhi:
                ylo, yhi = np.percentile(np.concatenate([but_y, ptb_y]), [10, 90])
            xpad = max(0.15, 0.18 * float(xhi - xlo))
            ypad = max(0.15, 0.18 * float(yhi - ylo))
            xlim = (float(xlo - xpad), float(xhi + xpad))
            ylim = (float(ylo - ypad), float(yhi + ypad))
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        else:
            xlim = None  # noqa: F841
            ylim = None  # noqa: F841
        if len(bb) > 2200:
            bb = bb.iloc[rng.choice(np.arange(len(bb)), size=2200, replace=False)]
        if len(pp) > 2200:
            pp = pp.iloc[rng.choice(np.arange(len(pp)), size=2200, replace=False)]
        ax.scatter(bb["pc1_v81"], bb["pc2_v81"], s=8, alpha=0.23, color=TOKENS["but"], label="BUT train+val")
        ax.scatter(pp["pc1_v81"], pp["pc2_v81"], s=10, alpha=0.45, color=TOKENS["ptb"], marker="x", linewidths=0.45, label="PTB synthetic")
        ax.set_title(f"{cls} overlap zoom")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, color=TOKENS["grid"], lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(f"{tag}: shared PCA overlap zoom (axis limits only)", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, out_dir / f"{tag}_shared_pca_overlap_zoom")


def plot_metric_heatmap(metric: pd.DataFrame, out_dir: Path, tag: str, value: str = "rbf_mmd") -> None:
    if metric.empty or value not in metric.columns:
        return
    pivot = metric.pivot_table(index="subtype", columns="class_name", values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.26 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title(f"{tag}: {value} by subtype")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"{tag}_{value}_heatmap")


def plot_cdf_overlay(but: pd.DataFrame, ptb: pd.DataFrame, out_dir: Path, tag: str) -> None:
    but = ensure_subtype(but)
    ptb = ensure_subtype(ptb)
    fig, axes = plt.subplots(len(CLASS_ORDER), len(CDF_FEATURES), figsize=(18, 7.3), sharey=True)
    for r, cls in enumerate(CLASS_ORDER):
        b = but.loc[but["class_name"].astype(str).eq(cls) & but["split"].astype(str).isin(["train", "val"])]
        p = ptb.loc[ptb["class_name"].astype(str).eq(cls)]
        for c, feat in enumerate(CDF_FEATURES):
            ax = axes[r, c]
            for frame, color, label in [(b, TOKENS["but"], "BUT"), (p, TOKENS["ptb"], "PTB")]:
                vals = pd.to_numeric(frame.get(feat, pd.Series([], dtype=float)), errors="coerce").dropna().to_numpy()
                if len(vals) == 0:
                    continue
                lo, hi = np.nanpercentile(vals, [1, 99])
                vals = np.clip(vals, lo, hi)
                xs = np.sort(vals)
                ys = np.linspace(0, 1, len(xs))
                ax.plot(xs, ys, color=color, lw=1.1, label=label)
            if r == 0:
                ax.set_title(feat)
            if c == 0:
                ax.set_ylabel(cls)
            ax.grid(True, color=TOKENS["grid"], lw=0.45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"{tag}: key feature CDF overlay", y=1.01, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, out_dir / f"{tag}_key_feature_cdf_overlay")


def robust_trace(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    med = float(np.nanmedian(y))
    scale = max(float(np.nanpercentile(y, 75) - np.nanpercentile(y, 25)), float(np.nanstd(y)), 1e-5)
    return np.clip((y - med) / scale, -3.5, 3.5)


def representative_idx(frame: pd.DataFrame, n: int, seed: int) -> list[int]:
    if len(frame) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(frame) <= n:
        return frame["idx"].astype(int).tolist()
    cols = [c for c in KEY_FEATURES if c in frame.columns]
    if not cols:
        return frame.sample(n=n, random_state=seed)["idx"].astype(int).tolist()
    x = feature_matrix(frame, cols)
    center, scale = robust_fit(x)
    d = np.sqrt(np.nanmean(robust_z(x, center, scale) ** 2, axis=1))
    order = np.argsort(d + rng.normal(0, 1e-6, len(d)))
    return frame.iloc[order[:n]]["idx"].astype(int).tolist()


def plot_waveform_contact_sheet(
    but: pd.DataFrame,
    but_x: np.ndarray,
    ptb: pd.DataFrame,
    ptb_x: np.ndarray,
    out_dir: Path,
    tag: str,
) -> None:
    but = ensure_subtype(but)
    ptb = ensure_subtype(ptb)
    t = np.linspace(0, 10, but_x.shape[-1])
    for cls in CLASS_ORDER:
        subtypes = V37.SUBTYPES_BY_CLASS[cls]
        fig, axes = plt.subplots(len(subtypes), 2, figsize=(11.0, max(2.1, 1.18 * len(subtypes))), sharex=True, sharey=True)
        if len(subtypes) == 1:
            axes = np.asarray([axes])
        for r, subtype in enumerate(subtypes):
            b = but.loc[
                but["class_name"].astype(str).eq(cls)
                & but["transport_subtype"].astype(str).eq(subtype)
                & but["split"].astype(str).isin(["train", "val"])
            ]
            p = ptb.loc[ptb["class_name"].astype(str).eq(cls) & ptb["transport_subtype"].astype(str).eq(subtype)]
            for col, (frame, arr, title, color) in enumerate(
                [(b, but_x, "BUT", TOKENS["but"]), (p, ptb_x, "PTB v81", TOKENS["ptb"])]
            ):
                ax = axes[r, col]
                ids = representative_idx(frame, 3, 100 + r * 7 + col)
                for idx in ids:
                    if 0 <= idx < len(arr):
                        ax.plot(t, robust_trace(arr[idx]), color=color, alpha=0.85, lw=0.85)
                ax.set_title(f"{title} | {subtype} n={len(frame)}", fontsize=8)
                ax.grid(True, color=TOKENS["grid"], lw=0.45)
                if col == 0:
                    ax.set_ylabel(cls)
        fig.suptitle(f"{tag}: representative {cls} subtype waveforms", y=1.01, fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out_dir / f"{tag}_{cls}_subtype_waveforms")


def plot_waveform_example_grid(
    but: pd.DataFrame,
    but_x: np.ndarray,
    ptb: pd.DataFrame,
    ptb_x: np.ndarray,
    out_dir: Path,
    tag: str,
    examples_per_side: int = 3,
) -> None:
    """Plot non-overlaid waveform examples.

    The contact sheet above intentionally overlays a few traces per subtype to
    show range, but for ECG morphology review that can look like pure noise.
    This grid gives each selected example its own tiny axis.
    """

    but = ensure_subtype(but)
    ptb = ensure_subtype(ptb)
    t = np.linspace(0, 10, but_x.shape[-1])
    for cls in CLASS_ORDER:
        subtypes = V37.SUBTYPES_BY_CLASS[cls]
        n_rows = len(subtypes)
        n_cols = 2 * int(examples_per_side)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(min(17.5, 2.35 * n_cols), max(2.2, 1.25 * n_rows)),
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes[None, :]
        for r, subtype in enumerate(subtypes):
            b = but.loc[
                but["class_name"].astype(str).eq(cls)
                & but["transport_subtype"].astype(str).eq(subtype)
                & but["split"].astype(str).isin(["train", "val"])
            ]
            p = ptb.loc[ptb["class_name"].astype(str).eq(cls) & ptb["transport_subtype"].astype(str).eq(subtype)]
            for side, (frame, arr, title, color) in enumerate(
                [(b, but_x, "BUT", TOKENS["but"]), (p, ptb_x, "PTB", TOKENS["ptb"])]
            ):
                ids = representative_idx(frame, int(examples_per_side), 300 + r * 17 + side)
                for k in range(int(examples_per_side)):
                    ax = axes[r, side * int(examples_per_side) + k]
                    if k < len(ids):
                        idx = int(ids[k])
                        if 0 <= idx < len(arr):
                            ax.plot(t, robust_trace(arr[idx]), color=color, alpha=0.95, lw=0.72)
                    ax.axhline(0, color=TOKENS["axis"], lw=0.35, alpha=0.7)
                    ax.set_ylim(-3.7, 3.7)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(True, color=TOKENS["grid"], lw=0.35, alpha=0.65)
                    if r == 0:
                        ax.set_title(f"{title} {k + 1}", fontsize=7.5)
                    if k == 0 and side == 0:
                        ax.set_ylabel(subtype.replace("_", "\n"), fontsize=6.3, rotation=0, ha="right", va="center", labelpad=26)
        fig.suptitle(f"{tag}: individual {cls} waveform examples (not overlaid)", y=1.01, fontsize=12, fontweight="bold")
        fig.tight_layout(w_pad=0.25, h_pad=0.30)
        save_fig(fig, out_dir / f"{tag}_{cls}_individual_waveform_examples")


def write_report(out_dir: Path, v80_metric: pd.DataFrame, v81_metric: pd.DataFrame, v81_feat: pd.DataFrame, protocol: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = v81_metric.copy()
    summary = {
        "v81_protocol": str(protocol),
        "metric_csv": str(out_dir / "v81_distribution_metrics.csv"),
        "feature_gap_csv": str(out_dir / "v81_feature_median_gap.csv"),
        "created_at": now(),
    }
    (out_dir / "v81_audit_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# V81 PTB to BUT Feature Distribution Transport Audit",
        "",
        f"- Created: {now()}",
        f"- V81 protocol: `{protocol}`",
        "- Contract: BUT train+val features are targets; BUT waveforms are never copied.",
        "",
        "## Distribution Metrics",
        "",
    ]
    if not key.empty:
        agg = key.groupby("class_name")[
            ["rbf_mmd", "sliced_wasserstein", "quantile_loss", "domain_auc", "pca_density_overlap"]
        ].mean(numeric_only=True)
        lines.append("```")
        lines.append(agg.to_string(float_format=lambda x: f"{x:.4f}"))
        lines.append("```")
        lines.append("")
        for col in ["rbf_mmd_reduction", "sliced_wasserstein_reduction", "quantile_loss_reduction"]:
            if col in key.columns:
                red = key.groupby("class_name")[col].mean(numeric_only=True)
                lines.append(f"Mean {col}:")
                lines.append("```")
                lines.append(red.to_string(float_format=lambda x: f"{x:.3f}"))
                lines.append("```")
                lines.append("")
    if not v81_feat.empty:
        bad = (
            v81_feat.loc[v81_feat["feature"].isin(KEY_FEATURES)]
            .assign(abs_gap=lambda d: d["median_gap_iqr"].abs())
            .sort_values("abs_gap", ascending=False)
            .head(20)
        )
        lines.extend(["## Worst Key Feature Gaps", "", "```", bad.to_string(index=False, float_format=lambda x: f"{x:.3f}"), "```", ""])
    lines.extend(
        [
            "## Figures",
            "",
            "- `v81_shared_pca.png`",
            "- `v81_key_feature_cdf_overlay.png`",
            "- `v81_rbf_mmd_heatmap.png`",
            "- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`",
            "",
            "## Interpretation Rule",
            "",
            "If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.",
        ]
    )
    path = out_dir / "v81_distribution_transport_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_audit(args: argparse.Namespace, protocol_path: Path) -> Path:
    setup_style()
    out_dir = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / str(args.report_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    but, but_x = load_protocol(Path(args.but_protocol))
    but = recompute_protocol_features(but, but_x, "BUT audit")
    v80, v80_x = load_protocol(Path(args.v80_protocol))
    v80 = recompute_protocol_features(v80, v80_x, "V80 audit")
    v81, v81_x = load_protocol(protocol_path)
    v81 = recompute_protocol_features(v81, v81_x, "V81 audit")
    v80 = ensure_subtype(v80)
    v81 = ensure_subtype(v81)
    but = ensure_subtype(but)

    v80_metric, _ = audit_pair(but, v80, "v80", out_dir, baseline=None, rng_seed=int(args.seed) + 11)
    v81_metric, v81_feat = audit_pair(but, v81, "v81", out_dir, baseline=v80_metric, rng_seed=int(args.seed) + 17)
    tag = str(args.tag)
    plot_shared_pca(but, v81, out_dir, tag)
    plot_cdf_overlay(but, v81, out_dir, tag)
    plot_metric_heatmap(v81_metric, out_dir, tag, "rbf_mmd")
    plot_waveform_contact_sheet(but, but_x, v81, v81_x, out_dir, tag)
    plot_waveform_example_grid(but, but_x, v81, v81_x, out_dir, tag, examples_per_side=3)
    report = write_report(out_dir, v80_metric, v81_metric, v81_feat, protocol_path)
    print(f"{now()} wrote audit report {report}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "audit", "all"], default="all")
    parser.add_argument("--protocol-name", default="protocol_v81_feature_transport_pc1500_cpa6_s20260681")
    parser.add_argument("--but-protocol", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--v80-protocol", default=str(DEFAULT_V80_PROTOCOL))
    parser.add_argument("--carrier-protocol", default=str(DEFAULT_CARRIER_PROTOCOL))
    parser.add_argument("--protocol-path", default="")
    parser.add_argument("--total-per-class", type=int, default=1500)
    parser.add_argument("--candidates-per-anchor", type=int, default=6)
    parser.add_argument("--medium-candidates-per-anchor", type=int, default=0)
    parser.add_argument("--bad-candidates-per-anchor", type=int, default=0)
    parser.add_argument("--mmd-repair-rounds", type=int, default=0)
    parser.add_argument("--mmd-repair-alternatives", type=int, default=4)
    parser.add_argument("--mmd-repair-tries", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260681)
    parser.add_argument("--report-subdir", default="v81_distribution_transport")
    parser.add_argument("--tag", default="v81")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = Path(args.protocol_path) if str(args.protocol_path).strip() else (
        ANALYSIS_DIR / "event_xds_aligned_v81_feature_transport" / str(args.protocol_name)
    )
    if args.stage in {"build", "all"}:
        result = build_v81(args)
        protocol = result.protocol_path
    if args.stage in {"audit", "all"}:
        run_audit(args, protocol)


if __name__ == "__main__":
    main()
