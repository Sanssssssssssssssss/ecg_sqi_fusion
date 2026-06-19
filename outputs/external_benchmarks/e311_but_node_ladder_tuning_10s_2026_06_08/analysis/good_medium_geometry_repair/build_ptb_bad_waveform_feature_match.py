"""Generate PTB bad waveforms matched to BUT bad after recomputing features.

The earlier 47-feature matcher only resampled existing PTB waveforms using the
feature sidecar.  This script changes the actual waveform: it generates a
candidate pool from PTB synthetic bad rows, recomputes the 39 waveform-computable
SQI/morphology primitive features on the generated waveforms, and selects a bank
whose recomputed feature distribution is close to BUT bad.

No BUT waveform enters training output.  BUT is only the target distribution for
this external diagnostic/domain-alignment artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
SYNTH_DATA_DIR = (
    OUT_ROOT
    / "synthetic_variants"
    / "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
    / "datasets"
)
BUT_SIGNALS = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_protocol_adaptation_2026_06_03"
    / "protocols"
    / "p1_current_10s_center"
    / "signals.npz"
)
BUT_ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
CLEAN_PROTOCOL_ROOT = ANALYSIS_DIR / "clean_but_protocols"
OUT_DIR = ANALYSIS_DIR / "ptb_bad_waveform_feature_match"
REPORT_OUT_DIR = REPORT_DIR / "ptb_bad_waveform_feature_match"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


UFORMER = load_module("uformer_geom_branch_for_bad_match", ANALYSIS_DIR / "uformer_geometry_branch_experiment.py")
PRIMITIVE_COLUMNS = list(json.loads((SYNTH_DATA_DIR / "tabular_feature_schema.json").read_text(encoding="utf-8"))["primitive_columns"])
# sqi_bSQI in this experiment is batch-percentile-normalized from
# baseline_step inside compute_primitives, so it is useful for reporting but
# unstable as a nearest-neighbor matching coordinate across different pools.
MATCH_EXCLUDE_COLUMNS = {"sqi_bSQI"}
MATCH_COLUMNS = [col for col in PRIMITIVE_COLUMNS if col not in MATCH_EXCLUDE_COLUMNS]


def augment_candidate(x: np.ndarray, rng: np.random.Generator, mode: str, strength: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[0]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    s = float(strength)
    if mode == "identity":
        return y
    if mode in {"cleanhf", "cleanhf_qrsdrop"}:
        # Clean-protocol BUT bad is mostly near-boundary bad, not the broad
        # record111 flat/dropout outlier. It has very high local derivative and
        # high-frequency detail while preserving enough ECG-like structure to
        # remain visually near the good/medium boundary.
        y = y - np.median(y)
        y = y / (np.percentile(np.abs(y), 90) + 1e-4)
        hf = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(6, 14))):
            freq = float(rng.uniform(32.0, 52.0))
            phase = float(rng.uniform(-np.pi, np.pi))
            amp = float(rng.uniform(0.16, 0.42) * s)
            hf += amp * np.sin(2.0 * np.pi * freq * np.linspace(0.0, 10.0, n, dtype=np.float32) + phase).astype(np.float32)
        # Alternating micro-steps raise diff zero-crossing without pushing all
        # power into the 15-30 Hz band.
        alt = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        alt = np.convolve(np.pad(alt, (1, 1), mode="edge"), np.array([0.18, 0.64, 0.18], dtype=np.float32), mode="valid")[:n]
        hf += float(rng.uniform(0.10, 0.30) * s) * alt
        for _ in range(int(rng.integers(8, 20))):
            width = int(rng.integers(4, 22))
            start = int(rng.integers(0, max(1, n - width)))
            local = np.arange(width, dtype=np.float32)
            freq = float(rng.uniform(0.32, 0.49))
            burst = np.sin(2.0 * np.pi * freq * local + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
            taper = np.hanning(max(3, width)).astype(np.float32)
            hf[start : start + width] += float(rng.uniform(0.12, 0.45) * s) * burst * taper
        if mode == "cleanhf_qrsdrop":
            # Attenuate several short beat-like neighborhoods so detector
            # agreement/QRS reliability drops without turning the whole row
            # into a trivial flatline.
            for _ in range(int(rng.integers(4, 10))):
                width = int(rng.integers(18, 48))
                start = int(rng.integers(0, max(1, n - width)))
                y[start : start + width] *= float(rng.uniform(0.10, 0.45))
        y = float(rng.uniform(0.28, 0.55)) * y + hf
        y += rng.normal(0.0, float(rng.uniform(0.010, 0.035) * s), size=n).astype(np.float32)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode == "rightisland_osc":
        # Clean BUT right-bad-island rows are dominated by narrow-band
        # oscillatory artifact around the QRS/detail band: high 15-30 Hz and
        # high zero-crossing/diff-zero-crossing, but not the very high 30-45 Hz
        # texture of near-boundary bad.
        y = y - np.median(y)
        y = y / (np.percentile(np.abs(y), 90) + 1e-4)
        tt = np.linspace(0.0, 10.0, n, dtype=np.float32)
        osc = np.zeros(n, dtype=np.float32)
        carrier = float(rng.uniform(18.0, 27.0))
        phase = float(rng.uniform(-np.pi, np.pi))
        osc += float(rng.uniform(0.85, 1.35) * s) * np.sin(2.0 * np.pi * carrier * tt + phase).astype(np.float32)
        osc += float(rng.uniform(0.18, 0.36) * s) * np.sin(2.0 * np.pi * float(rng.uniform(10.0, 16.0)) * tt + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        # Slow amplitude modulation keeps it ECG-window-like rather than a
        # perfectly stationary tone.
        env = 0.82 + 0.18 * np.sin(2.0 * np.pi * float(rng.uniform(0.12, 0.32)) * tt + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        osc *= env.astype(np.float32)
        # Sparse drop/phase reset chunks break detector agreement.
        for _ in range(int(rng.integers(3, 8))):
            width = int(rng.integers(18, 65))
            start = int(rng.integers(0, max(1, n - width)))
            osc[start : start + width] *= float(rng.uniform(0.12, 0.55))
        y = float(rng.uniform(0.04, 0.14)) * y + osc
        y += rng.normal(0.0, float(rng.uniform(0.004, 0.014) * s), size=n).astype(np.float32)
        y = y / (np.std(y) + 1e-4) * float(rng.uniform(0.92, 1.18))
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode == "flatstep_qrsvisible":
        # BUT record111-like stress is not just "more noise": it has large
        # baseline steps/contact-like flat regions, low detail/high-frequency
        # power, and still-visible but unreliable QRS-like peaks.  This mode is
        # deliberately different from high-zcr/rough bad.
        step_at = int(rng.integers(max(80, n // 8), max(100, 7 * n // 8)))
        offset = float(rng.uniform(0.55, 1.85) * s) * (1.0 if rng.random() < 0.5 else -1.0)
        y[step_at:] += offset
        drift = float(rng.uniform(0.25, 1.10) * s)
        y += drift * (0.42 * t + 0.22 * np.sin(np.pi * t + float(rng.uniform(-np.pi, np.pi))))
        # Strong smoothing lowers detail bands while preserving broad shape.
        kernel = int(rng.choice([31, 51, 71]))
        filt = np.ones(kernel, dtype=np.float32) / float(kernel)
        smooth = np.convolve(np.pad(y, (kernel // 2, kernel // 2), mode="edge"), filt, mode="valid")[:n]
        y = 0.38 * y + 0.62 * smooth
        for _ in range(int(rng.integers(2, 5))):
            width = int(rng.integers(max(60, n // 16), max(90, n // 3)))
            start = int(rng.integers(0, max(1, n - width)))
            end = start + width
            if rng.random() < 0.60:
                y[start:end] = float(rng.uniform(-0.10, 0.10)) + rng.normal(0.0, 0.004 * s, size=width).astype(np.float32)
            else:
                y[start:end] *= float(rng.uniform(0.03, 0.22))
        # Sparse visible peaks keep the row from becoming trivial flatline bad.
        for _ in range(int(rng.integers(5, 13))):
            center = int(rng.integers(10, max(11, n - 10)))
            width = int(rng.integers(4, 11))
            lo = max(0, center - width)
            hi = min(n, center + width + 1)
            local = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
            pulse = np.exp(-10.0 * local * local).astype(np.float32)
            y[lo:hi] += float(rng.uniform(0.08, 0.42) * s) * pulse * (1.0 if rng.random() < 0.55 else -1.0)
        if rng.random() < 0.55:
            y += rng.normal(0.0, float(rng.uniform(0.002, 0.014) * s), size=n).astype(np.float32)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode in {"baseline", "mixed", "stress", "record111"}:
        drift = float(rng.uniform(0.20, 1.45) * s)
        curve = float(rng.uniform(0.08, 0.65) * s)
        y += drift * (0.55 * t + 0.28 * np.sin(np.pi * t + float(rng.uniform(-np.pi, np.pi))))
        y += curve * (t * t - 0.35)
    if mode in {"dropout", "mixed", "stress", "record111"}:
        chunks = int(rng.integers(1, 4 if mode != "stress" else 6))
        for _ in range(chunks):
            width = int(rng.integers(max(20, n // 40), max(45, int(n * rng.uniform(0.06, 0.38)))))
            start = int(rng.integers(0, max(1, n - width)))
            end = start + width
            if rng.random() < 0.55:
                y[start:end] = float(rng.uniform(-0.08, 0.08)) + rng.normal(0.0, 0.006 * s, size=width).astype(np.float32)
            else:
                y[start:end] *= float(rng.uniform(0.02, 0.35))
    if mode in {"smooth", "mixed"}:
        kernel = int(rng.choice([15, 21, 31, 51]))
        filt = np.ones(kernel, dtype=np.float32) / float(kernel)
        smooth = np.convolve(np.pad(y, (kernel // 2, kernel // 2), mode="edge"), filt, mode="valid")[:n]
        mix = float(rng.uniform(0.18, 0.55) * min(1.0, 0.75 + 0.10 * s))
        y = (1.0 - mix) * y + mix * smooth
    if mode in {"reset", "mixed", "stress", "record111"}:
        for _ in range(int(rng.integers(1, 7))):
            center = int(rng.integers(8, max(9, n - 8)))
            width = int(rng.integers(3, 22))
            lo = max(0, center - width)
            hi = min(n, center + width + 1)
            local = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
            spike = np.exp(-16.0 * local * local).astype(np.float32)
            y[lo:hi] += float(rng.uniform(0.08, 0.75) * s) * spike * (1.0 if rng.random() < 0.5 else -1.0)
    if mode in {"detail", "highzcr", "rough", "mixed", "stress", "record111"}:
        hf_scale = float(rng.uniform(0.015, 0.115) * s)
        alt = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        alt = np.convolve(np.pad(alt, (1, 1), mode="edge"), np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="valid")[:n]
        y += hf_scale * alt
        bursts = int(rng.integers(2, 12 if mode != "highzcr" else 20))
        for _ in range(bursts):
            width = int(rng.integers(5, 45 if mode != "highzcr" else 28))
            start = int(rng.integers(0, max(1, n - width)))
            local = np.arange(width, dtype=np.float32)
            freq = float(rng.uniform(0.18, 0.49))
            burst = np.sin(2.0 * np.pi * freq * local + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
            taper = np.hanning(max(3, width)).astype(np.float32)
            y[start : start + width] += float(rng.uniform(0.025, 0.19) * s) * burst * taper
    if mode in {"qrs_confuse", "record111"}:
        # Add sparse pseudo-QRS proposals. This targets bad cases where the
        # detector sees visible but unreliable spike trains.
        count = int(rng.integers(4, 15))
        for _ in range(count):
            center = int(rng.integers(8, max(9, n - 8)))
            width = int(rng.integers(4, 13))
            lo = max(0, center - width)
            hi = min(n, center + width + 1)
            local = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
            pulse = (1.0 - np.abs(local)).clip(0, 1)
            y[lo:hi] += float(rng.uniform(0.10, 0.65) * s) * pulse * (1.0 if rng.random() < 0.55 else -1.0)
    if mode in {"rough", "stress", "record111"}:
        for _ in range(int(rng.integers(1, 6))):
            start = int(rng.integers(0, max(1, n - 90)))
            end = min(n, start + int(rng.integers(18, 105)))
            ramp = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
            y[start:end] += float(rng.uniform(-0.20, 0.20) * s) * ramp
    if mode != "identity" and rng.random() < 0.55:
        y += rng.normal(0.0, float(rng.uniform(0.002, 0.030) * s), size=n).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.sort(a.astype(np.float64))
    y = np.sort(b.astype(np.float64))
    grid = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / max(1, len(x))
    cdf_y = np.searchsorted(y, grid, side="right") / max(1, len(y))
    return float(np.max(np.abs(cdf_x - cdf_y)))


def robust_scale(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(target, axis=0).astype(np.float32)
    scale = (np.nanpercentile(target, 75, axis=0) - np.nanpercentile(target, 25, axis=0)).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    return center, scale


def nearest_select(cand_z: np.ndarray, target_z: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_pick = rng.choice(np.arange(target_z.shape[0]), size=int(count), replace=True)
    selected = np.empty(int(count), dtype=np.int64)
    distances = np.empty(int(count), dtype=np.float32)
    for out_i, target_i in enumerate(target_pick):
        dist = np.mean((cand_z - target_z[target_i][None, :]) ** 2, axis=1)
        best = int(np.argmin(dist))
        selected[out_i] = best
        distances[out_i] = float(np.sqrt(dist[best]))
    return selected, target_pick.astype(np.int64), distances


def nearest_select_diverse(
    cand_z: np.ndarray,
    target_z: np.ndarray,
    count: int,
    rng: np.random.Generator,
    cand_manifest: pd.DataFrame,
    top_k: int = 48,
    source_cap: int = 4,
    mode_cap_frac: float = 0.34,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbor selection with source/mode diversity guards.

    The plain nearest selector can collapse onto a handful of synthetic rows.
    For bad-outlier generation that is exactly the wrong failure mode: we need a
    morphology shell, not one repeated waveform trick.  This selector still uses
    the target feature geometry, but samples from the nearest local candidate
    neighborhood while capping repeated source windows and any single mode.
    """
    target_pick = rng.choice(np.arange(target_z.shape[0]), size=int(count), replace=True)
    source = cand_manifest["source_idx"].to_numpy(dtype=np.int64)
    mode = cand_manifest["mode"].astype(str).to_numpy()
    source_counts: dict[int, int] = {}
    mode_counts: dict[str, int] = {}
    mode_cap = max(1, int(round(float(count) * float(mode_cap_frac))))
    selected = np.empty(int(count), dtype=np.int64)
    distances = np.empty(int(count), dtype=np.float32)
    order_top_k = max(1, int(top_k))
    for out_i, target_i in enumerate(target_pick):
        dist = np.mean((cand_z - target_z[target_i][None, :]) ** 2, axis=1)
        order = np.argsort(dist)[:order_top_k]
        feasible: list[int] = []
        for cand_i in order:
            src = int(source[cand_i])
            md = str(mode[cand_i])
            if source_counts.get(src, 0) >= int(source_cap):
                continue
            if mode_counts.get(md, 0) >= mode_cap:
                continue
            feasible.append(int(cand_i))
        if not feasible:
            feasible = [int(i) for i in order[: min(8, len(order))]]
        feasible_arr = np.asarray(feasible, dtype=np.int64)
        local_dist = dist[feasible_arr]
        local_dist = local_dist - float(np.min(local_dist))
        weights = np.exp(-local_dist / (float(np.median(local_dist)) + 1e-6))
        weights = weights / float(np.sum(weights))
        chosen = int(rng.choice(feasible_arr, p=weights))
        selected[out_i] = chosen
        distances[out_i] = float(dist[chosen])
        source_counts[int(source[chosen])] = source_counts.get(int(source[chosen]), 0) + 1
        mode_counts[str(mode[chosen])] = mode_counts.get(str(mode[chosen]), 0) + 1
    return selected, target_pick.astype(np.int64), distances


def controlled_outlier_target_mask(target: pd.DataFrame, quantile: float) -> np.ndarray:
    """Keep all bad core rows plus the closest controlled outlier shell.

    We do not want to erase bad outliers, but the most extreme outliers are often
    pure contact/noise stress.  The controlled shell keeps outlier rows whose
    waveform-computable features remain nearest to the bad core distribution.
    """
    if "original_region" not in target.columns:
        return np.ones(len(target), dtype=bool)
    region = target["original_region"].astype(str)
    core_mask = region.ne("outlier_low_confidence").to_numpy()
    outlier_mask = region.eq("outlier_low_confidence").to_numpy()
    if not np.any(outlier_mask) or not np.any(core_mask):
        return np.ones(len(target), dtype=bool)
    preferred = [
        "rms",
        "std",
        "mean_abs",
        "ptp_p99_p01",
        "flatline_ratio",
        "contact_loss_win_ratio",
        "baseline_step",
        "diff_abs_p95",
        "detail_instability",
        "qrs_band_ratio",
        "qrs_visibility",
        "sqi_bSQI",
        "sqi_basSQI",
        "detector_agreement",
        "non_qrs_diff_p95",
        "band_0p3_1",
        "band_1_5",
        "band_5_15",
        "band_15_30",
        "band_30_45",
        "amplitude_entropy",
        "sample_entropy_proxy",
        "hjorth_activity",
        "hjorth_mobility",
    ]
    cols = [c for c in preferred if c in target.columns]
    if len(cols) < 6:
        return np.ones(len(target), dtype=bool)
    arr = target[cols].to_numpy(dtype=np.float32)
    center = np.nanmedian(arr[core_mask], axis=0)
    scale = np.nanpercentile(arr[core_mask], 75, axis=0) - np.nanpercentile(arr[core_mask], 25, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    z = np.nan_to_num((arr - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.mean(np.abs(z), axis=1)
    outlier_dist = dist[outlier_mask]
    threshold = float(np.quantile(outlier_dist, np.clip(float(quantile), 0.05, 1.0)))
    keep = core_mask.copy()
    keep |= outlier_mask & (dist <= threshold)
    return keep


def feature_ks_table(target: pd.DataFrame, base: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in PRIMITIVE_COLUMNS:
        for group, df in [("base_ptb_bad", base), ("matched_generated_ptb_bad", matched)]:
            rows.append(
                {
                    "feature": col,
                    "group": group,
                    "ks_vs_but_bad": ks_stat(target[col].to_numpy(), df[col].to_numpy()),
                    "target_median": float(np.nanmedian(target[col].to_numpy())),
                    "group_median": float(np.nanmedian(df[col].to_numpy())),
                    "target_q10": float(np.nanpercentile(target[col].to_numpy(), 10)),
                    "target_q90": float(np.nanpercentile(target[col].to_numpy(), 90)),
                    "group_q10": float(np.nanpercentile(df[col].to_numpy(), 10)),
                    "group_q90": float(np.nanpercentile(df[col].to_numpy(), 90)),
                }
            )
    return pd.DataFrame(rows)


def pca2(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:2].T).astype(np.float32)


def plot_pca(target: pd.DataFrame, base: pd.DataFrame, matched: pd.DataFrame, out_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    groups = [
        ("BUT bad target", target[PRIMITIVE_COLUMNS].to_numpy(dtype=np.float32), "#444444", 900),
        ("PTB bad base", base[PRIMITIVE_COLUMNS].to_numpy(dtype=np.float32), "#d95f02", 700),
        ("generated PTB bad matched", matched[PRIMITIVE_COLUMNS].to_numpy(dtype=np.float32), "#1b9e77", 700),
    ]
    z = pca2(np.concatenate([g[1] for g in groups], axis=0))
    offset = 0
    fig, ax = plt.subplots(figsize=(8, 6), dpi=170)
    for label, arr, color, max_n in groups:
        pts = z[offset : offset + len(arr)]
        offset += len(arr)
        if len(pts) > max_n:
            pts = pts[rng.choice(len(pts), size=max_n, replace=False)]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.35, label=label, color=color, edgecolors="none")
    ax.set_title("Generated PTB bad vs BUT bad in recomputed primitive feature space")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_waveforms(signals: np.ndarray, manifest: pd.DataFrame, out_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    take = rng.choice(len(signals), size=min(16, len(signals)), replace=False)
    t = np.linspace(0, 10, signals.shape[1], dtype=np.float32)
    fig, axes = plt.subplots(4, 4, figsize=(12, 7), dpi=160, sharex=True)
    for ax, idx in zip(axes.ravel(), take):
        ax.plot(t, signals[idx], lw=0.75, color="#1b9e77")
        ax.set_title(str(manifest.iloc[idx]["mode"]), fontsize=8)
        ax.set_yticks([])
        ax.grid(alpha=0.12)
    fig.suptitle("Selected generated PTB bad waveforms")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="but_bad_all",
        choices=[
            "but_bad_all",
            "but_bad_test",
            "clean_margin_ge_5s_drop_outlier_bad_train",
            "clean_margin_ge_5s_drop_outlier_bad_test",
            "clean_margin_ge_5s_drop_outlier_bad_all",
            "clean_margin_ge_5s_keep_outlier_bad_train",
            "clean_margin_ge_5s_keep_outlier_bad_test",
            "clean_margin_ge_5s_keep_outlier_bad_all",
            "clean_margin_ge_5s_keep_outlier_bad_train_controlled",
            "clean_margin_ge_5s_keep_outlier_bad_all_controlled",
        ],
    )
    parser.add_argument("--candidate-per-bad", type=int, default=12)
    parser.add_argument("--selected-count", type=int, default=1816)
    parser.add_argument("--selector", type=str, default="nearest", choices=["nearest", "nearest_diverse", "block_profile", "best_mode"])
    parser.add_argument("--mode-profile", type=str, default="all", choices=["all", "soft_controlled_bad"])
    parser.add_argument("--source-classes", type=str, default="bad")
    parser.add_argument("--controlled-outlier-quantile", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=2026061917)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    labels = pd.read_csv(SYNTH_DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    synth = np.load(SYNTH_DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    source_classes = [c.strip() for c in str(args.source_classes).split(",") if c.strip()]
    bad_rows = np.flatnonzero(labels["y_class"].astype(str).isin(source_classes).to_numpy() & labels["split"].astype(str).eq("train").to_numpy())
    bad_idx = labels.loc[bad_rows, "idx"].to_numpy(dtype=np.int64)
    source_class_by_idx = labels.set_index("idx")["y_class"].astype(str)
    base_bad = synth[bad_idx]

    if str(args.target).startswith("clean_margin_ge_5s_drop_outlier") or str(args.target).startswith("clean_margin_ge_5s_keep_outlier"):
        protocol_name = "margin_ge_5s_drop_outlier" if "drop_outlier" in str(args.target) else "margin_ge_5s_keep_outlier"
        protocol_dir = CLEAN_PROTOCOL_ROOT / protocol_name
        if not protocol_dir.exists():
            raise FileNotFoundError(f"clean BUT protocol not found: {protocol_dir}")
        clean_npz = np.load(protocol_dir / "signals.npz")
        clean_key = "X" if "X" in clean_npz.files else clean_npz.files[0]
        clean_x = np.asarray(clean_npz[clean_key], dtype=np.float32)
        if clean_x.ndim == 3:
            clean_raw = clean_x[:, 0, :]
        else:
            clean_raw = clean_x
        atlas = pd.read_csv(protocol_dir / "original_region_atlas.csv")
        target_mask = atlas["class_name"].astype(str).eq("bad")
        if str(args.target).endswith("_bad_train"):
            target_mask &= atlas["split"].astype(str).eq("train")
        elif str(args.target).endswith("_bad_test"):
            target_mask &= atlas["split"].astype(str).eq("test")
        target = atlas.loc[target_mask].reset_index(drop=True)
        if str(args.target).endswith("_controlled"):
            keep = controlled_outlier_target_mask(target, float(args.controlled_outlier_quantile))
            target = target.loc[keep].reset_index(drop=True)
        target_x = clean_raw[target["idx"].to_numpy(dtype=np.int64)]
    else:
        but_x = np.load(BUT_SIGNALS)["X"].astype(np.float32)
        atlas = pd.read_csv(BUT_ATLAS, usecols=["idx", "split", "class_name", "record_id", "original_region"])
        target_mask = atlas["class_name"].astype(str).eq("bad")
        if args.target == "but_bad_test":
            target_mask &= atlas["split"].astype(str).eq("test")
        target = atlas.loc[target_mask].reset_index(drop=True)
        target_x = but_x[target["idx"].to_numpy(dtype=np.int64), 0, :]

    print(f"compute target primitives: {len(target_x)} BUT bad rows", flush=True)
    target_feat = UFORMER.compute_primitives(target_x)[PRIMITIVE_COLUMNS]
    print(f"compute base primitives: {len(base_bad)} PTB train bad rows", flush=True)
    base_feat = UFORMER.compute_primitives(base_bad)[PRIMITIVE_COLUMNS]

    if str(args.mode_profile) == "soft_controlled_bad":
        modes = [
            "baseline",
            "dropout",
            "smooth",
            "reset",
            "detail",
            "rough",
            "qrs_confuse",
            "mixed",
            "record111",
            "flatstep_qrsvisible",
            "rightisland_osc",
        ]
        strengths = [0.35, 0.55, 0.75, 0.95, 1.15]
    else:
        modes = [
            "baseline",
            "dropout",
            "smooth",
            "reset",
            "detail",
            "highzcr",
            "rough",
            "qrs_confuse",
            "mixed",
            "record111",
            "stress",
            "flatstep_qrsvisible",
            "cleanhf",
            "cleanhf_qrsdrop",
            "rightisland_osc",
        ]
        strengths = [0.55, 0.85, 1.15, 1.45, 1.75]
    cand_signals: list[np.ndarray] = []
    cand_rows: list[dict[str, Any]] = []
    for row_no, src_idx in enumerate(bad_idx):
        raw = synth[src_idx]
        cand_signals.append(raw.copy())
        cand_rows.append({"source_idx": int(src_idx), "source_y_class": str(source_class_by_idx.loc[int(src_idx)]), "mode": "identity", "strength": 0.0})
        for _ in range(int(args.candidate_per_bad)):
            mode = str(rng.choice(modes))
            strength = float(rng.choice(strengths) * rng.uniform(0.85, 1.15))
            cand_signals.append(augment_candidate(raw, rng, mode, strength))
            cand_rows.append({"source_idx": int(src_idx), "source_y_class": str(source_class_by_idx.loc[int(src_idx)]), "mode": mode, "strength": strength})
    cand = np.stack(cand_signals).astype(np.float32)
    cand_manifest = pd.DataFrame(cand_rows)
    print(f"compute candidate primitives: {len(cand)} generated PTB bad rows", flush=True)
    cand_feat = UFORMER.compute_primitives(cand)[PRIMITIVE_COLUMNS]

    center, scale = robust_scale(target_feat[MATCH_COLUMNS].to_numpy(dtype=np.float32))
    cand_z = np.nan_to_num((cand_feat[MATCH_COLUMNS].to_numpy(dtype=np.float32) - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    target_z = np.nan_to_num((target_feat[MATCH_COLUMNS].to_numpy(dtype=np.float32) - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mode_rows: list[dict[str, Any]] = []
    cand_feat_with_mode = cand_feat.copy()
    cand_feat_with_mode["mode"] = cand_manifest["mode"].to_numpy()
    for mode, mode_df in cand_feat_with_mode.groupby("mode"):
        if len(mode_df) < 5:
            continue
        tmp = feature_ks_table(target_feat, base_feat, mode_df[PRIMITIVE_COLUMNS])
        mode_part = tmp.loc[tmp["group"].eq("matched_generated_ptb_bad")]
        mode_rows.append(
            {
                "mode": str(mode),
                "rows": int(len(mode_df)),
                "mean_ks": float(mode_part["ks_vs_but_bad"].mean()),
                "top10_mean_ks": float(mode_part.nlargest(10, "ks_vs_but_bad")["ks_vs_but_bad"].mean()),
                "zcr_ks": float(mode_part.loc[mode_part["feature"].eq("zero_crossing_rate"), "ks_vs_but_bad"].iloc[0]),
                "diff_zcr_ks": float(mode_part.loc[mode_part["feature"].eq("diff_zero_crossing_rate"), "ks_vs_but_bad"].iloc[0]),
                "band15_30_ks": float(mode_part.loc[mode_part["feature"].eq("band_15_30"), "ks_vs_but_bad"].iloc[0]),
                "non_qrs_diff_p95_ks": float(mode_part.loc[mode_part["feature"].eq("non_qrs_diff_p95"), "ks_vs_but_bad"].iloc[0]),
                "sqi_bSQI_ks": float(mode_part.loc[mode_part["feature"].eq("sqi_bSQI"), "ks_vs_but_bad"].iloc[0]),
                "flatline_ratio_ks": float(mode_part.loc[mode_part["feature"].eq("flatline_ratio"), "ks_vs_but_bad"].iloc[0]),
            }
        )
    mode_summary = pd.DataFrame(mode_rows).sort_values("mean_ks")
    if args.selector == "best_mode":
        best_mode = str(mode_summary.iloc[0]["mode"])
        pool = np.flatnonzero(cand_manifest["mode"].astype(str).to_numpy() == best_mode)
        if int(args.selected_count) > 0 and len(pool) > int(args.selected_count):
            selected = rng.choice(pool, size=int(args.selected_count), replace=False)
        else:
            selected = pool
        matched = cand[selected]
        matched_feat = cand_feat.iloc[selected].reset_index(drop=True)
        matched_manifest = cand_manifest.iloc[selected].reset_index(drop=True)
        target_pick = rng.choice(np.arange(len(target)), size=len(selected), replace=True)
        matched_manifest["target_but_idx"] = target.loc[target_pick, "idx"].to_numpy(dtype=np.int64)
        matched_manifest["target_record_id"] = target.loc[target_pick, "record_id"].to_numpy()
        matched_manifest["target_original_region"] = target.loc[target_pick, "original_region"].astype(str).to_numpy()
        matched_manifest["match_distance"] = np.nan
        matched_manifest["selector_best_mode"] = best_mode
    elif args.selector == "block_profile":
        source_cls = cand_manifest["source_y_class"].astype(str).to_numpy() if "source_y_class" in cand_manifest.columns else np.asarray(["bad"] * len(cand_manifest))
        mode_arr = cand_manifest["mode"].astype(str).to_numpy()
        profiles = [
            (
                "bad_core_guard",
                0.35,
                (source_cls == "bad")
                & np.isin(mode_arr, ["identity", "rightisland_osc", "dropout", "detail", "rough", "reset"]),
            ),
            (
                "bad_controlled_qrs_visible",
                0.30,
                (source_cls != "bad")
                & np.isin(mode_arr, ["flatstep_qrsvisible", "baseline", "qrs_confuse", "reset", "mixed"]),
            ),
            (
                "bad_controlled_contact",
                0.25,
                (source_cls != "bad")
                & np.isin(mode_arr, ["dropout", "record111", "mixed", "baseline", "smooth"]),
            ),
            (
                "bad_mild_noise_guard",
                0.10,
                (source_cls != "bad")
                & np.isin(mode_arr, ["detail", "rough", "smooth"]),
            ),
        ]
        picked: list[int] = []
        picked_blocks: list[str] = []
        for block_name, frac, block_mask in profiles:
            pool = np.flatnonzero(block_mask)
            want = int(round(float(args.selected_count) * float(frac)))
            if want <= 0 or len(pool) == 0:
                continue
            take = rng.choice(pool, size=min(want, len(pool)), replace=False)
            picked.extend([int(i) for i in take])
            picked_blocks.extend([block_name] * len(take))
        if len(picked) < int(args.selected_count):
            already = np.zeros(len(cand_manifest), dtype=bool)
            already[np.asarray(picked, dtype=np.int64)] = True
            fill_pool = np.flatnonzero(~already)
            fill = rng.choice(fill_pool, size=min(int(args.selected_count) - len(picked), len(fill_pool)), replace=False)
            picked.extend([int(i) for i in fill])
            picked_blocks.extend(["bad_profile_fill"] * len(fill))
        selected = np.asarray(picked[: int(args.selected_count)], dtype=np.int64)
        picked_blocks = picked_blocks[: int(args.selected_count)]
        matched = cand[selected]
        matched_feat = cand_feat.iloc[selected].reset_index(drop=True)
        matched_manifest = cand_manifest.iloc[selected].reset_index(drop=True)
        matched_manifest["profile_block"] = picked_blocks
        target_pick = rng.choice(np.arange(len(target)), size=len(selected), replace=True)
        matched_manifest["target_but_idx"] = target.loc[target_pick, "idx"].to_numpy(dtype=np.int64)
        matched_manifest["target_record_id"] = target.loc[target_pick, "record_id"].to_numpy()
        matched_manifest["target_original_region"] = target.loc[target_pick, "original_region"].astype(str).to_numpy()
        selected_z = cand_z[selected]
        picked_target_z = target_z[target_pick]
        matched_manifest["match_distance"] = np.mean((selected_z - picked_target_z) ** 2, axis=1).astype(np.float32)
        matched_manifest["selector_best_mode"] = ""
    elif args.selector == "nearest":
        selected, target_pick, distances = nearest_select(cand_z, target_z, int(args.selected_count), rng)
        matched = cand[selected]
        matched_feat = cand_feat.iloc[selected].reset_index(drop=True)
        matched_manifest = cand_manifest.iloc[selected].reset_index(drop=True)
        matched_manifest["target_but_idx"] = target.loc[target_pick, "idx"].to_numpy(dtype=np.int64)
        matched_manifest["target_record_id"] = target.loc[target_pick, "record_id"].to_numpy()
        matched_manifest["target_original_region"] = target.loc[target_pick, "original_region"].astype(str).to_numpy()
        matched_manifest["match_distance"] = distances
        matched_manifest["selector_best_mode"] = ""
    else:
        selected, target_pick, distances = nearest_select_diverse(cand_z, target_z, int(args.selected_count), rng, cand_manifest)
        matched = cand[selected]
        matched_feat = cand_feat.iloc[selected].reset_index(drop=True)
        matched_manifest = cand_manifest.iloc[selected].reset_index(drop=True)
        matched_manifest["target_but_idx"] = target.loc[target_pick, "idx"].to_numpy(dtype=np.int64)
        matched_manifest["target_record_id"] = target.loc[target_pick, "record_id"].to_numpy()
        matched_manifest["target_original_region"] = target.loc[target_pick, "original_region"].astype(str).to_numpy()
        matched_manifest["match_distance"] = distances
        matched_manifest["selector_best_mode"] = ""
    matched_manifest["source_split"] = "train"
    matched_manifest["y_class"] = "bad"
    matched_manifest["stress"] = 1.0
    if "profile_block" in matched_manifest.columns:
        matched_manifest["block"] = "bad_waveform_feature_match|" + matched_manifest["profile_block"].astype(str)
    else:
        matched_manifest["block"] = "bad_waveform_feature_match"

    dist = feature_ks_table(target_feat, base_feat, matched_feat)
    distance_values = pd.to_numeric(matched_manifest["match_distance"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    summary = {
        "target": args.target,
        "target_count": int(len(target_feat)),
        "base_train_bad_count": int(len(base_bad)),
        "source_classes": source_classes,
        "candidate_count": int(len(cand)),
        "matched_count": int(len(matched)),
        "unique_source_count": int(matched_manifest["source_idx"].nunique()),
        "base_mean_ks": float(dist.loc[dist["group"].eq("base_ptb_bad"), "ks_vs_but_bad"].mean()),
        "matched_mean_ks": float(dist.loc[dist["group"].eq("matched_generated_ptb_bad"), "ks_vs_but_bad"].mean()),
        "base_top10_mean_ks": float(dist.loc[dist["group"].eq("base_ptb_bad")].nlargest(10, "ks_vs_but_bad")["ks_vs_but_bad"].mean()),
        "matched_top10_mean_ks": float(dist.loc[dist["group"].eq("matched_generated_ptb_bad")].nlargest(10, "ks_vs_but_bad")["ks_vs_but_bad"].mean()),
        "distance_q50": float(np.percentile(distance_values, 50)) if len(distance_values) else None,
        "distance_q90": float(np.percentile(distance_values, 90)) if len(distance_values) else None,
        "selected_modes": matched_manifest["mode"].value_counts().to_dict(),
        "selected_source_classes": matched_manifest["source_y_class"].value_counts().to_dict() if "source_y_class" in matched_manifest.columns else {},
        "selected_profile_blocks": matched_manifest["profile_block"].value_counts().to_dict() if "profile_block" in matched_manifest.columns else {},
        "target_region_counts": matched_manifest["target_original_region"].value_counts().to_dict(),
        "best_modes_by_mean_ks": mode_summary.head(8).to_dict(orient="records"),
        "selector": args.selector,
        "mode_profile": str(args.mode_profile),
        "controlled_outlier_quantile": float(args.controlled_outlier_quantile),
        "match_columns": MATCH_COLUMNS,
        "match_exclude_columns": sorted(MATCH_EXCLUDE_COLUMNS),
    }

    target_stem = str(args.target)
    target_stem = target_stem.replace("clean_margin_ge_5s_drop_outlier_bad_train", "cleanbad5s_train")
    target_stem = target_stem.replace("clean_margin_ge_5s_drop_outlier_bad_test", "cleanbad5s_test")
    target_stem = target_stem.replace("clean_margin_ge_5s_drop_outlier_bad_all", "cleanbad5s_all")
    target_stem = target_stem.replace("clean_margin_ge_5s_keep_outlier_bad_train_controlled", "keepbad5s_train_controlled")
    target_stem = target_stem.replace("clean_margin_ge_5s_keep_outlier_bad_all_controlled", "keepbad5s_all_controlled")
    target_stem = target_stem.replace("clean_margin_ge_5s_keep_outlier_bad_train", "keepbad5s_train")
    target_stem = target_stem.replace("clean_margin_ge_5s_keep_outlier_bad_test", "keepbad5s_test")
    target_stem = target_stem.replace("clean_margin_ge_5s_keep_outlier_bad_all", "keepbad5s_all")
    profile_suffix = "" if str(args.mode_profile) == "all" else f"_{str(args.mode_profile)}"
    source_suffix = "" if str(args.source_classes) == "bad" else "_src" + str(args.source_classes).replace(",", "")
    ctrl_suffix = ""
    if "controlled" in str(args.target):
        ctrl_suffix = f"_q{str(args.controlled_outlier_quantile).replace('.', 'p')}"
    stem = f"ptb_bad_waveform_feature_match_{target_stem}_gen{int(args.candidate_per_bad)}_{args.selector}{profile_suffix}{source_suffix}{ctrl_suffix}"
    if len(stem) > 86:
        import hashlib

        stem = stem[:58] + "_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    np.savez_compressed(OUT_DIR / f"{stem}_signals.npz", X=matched)
    matched_manifest.to_csv(OUT_DIR / f"{stem}_manifest.csv", index=False)
    matched_feat.to_csv(OUT_DIR / f"{stem}_features.csv", index=False)
    dist.to_csv(OUT_DIR / f"{stem}_feature_ks.csv", index=False)
    mode_summary.to_csv(OUT_DIR / f"{stem}_mode_ks_summary.csv", index=False)
    (OUT_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pca_path = REPORT_OUT_DIR / f"{stem}_pca.png"
    wave_path = REPORT_OUT_DIR / f"{stem}_waveforms.png"
    plot_pca(target_feat, base_feat, matched_feat, pca_path, int(args.seed))
    plot_waveforms(matched, matched_manifest, wave_path, int(args.seed) + 17)

    worst = dist.loc[dist["group"].eq("matched_generated_ptb_bad")].sort_values("ks_vs_but_bad", ascending=False).head(12)
    report = [
        "# PTB Bad Waveform Feature Match",
        "",
        "Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.",
        "",
        "## Summary",
        "",
        f"- Target: `{args.target}` rows `{len(target_feat)}`",
        f"- Base PTB train bad rows: `{len(base_bad)}`",
        f"- Candidate generated rows: `{len(cand)}`",
        f"- Matched rows: `{len(matched)}` from `{summary['unique_source_count']}` unique PTB sources",
        f"- Mean KS base -> BUT bad: `{summary['base_mean_ks']:.4f}`",
        f"- Mean KS matched -> BUT bad: `{summary['matched_mean_ks']:.4f}`",
        f"- Top10 KS base -> BUT bad: `{summary['base_top10_mean_ks']:.4f}`",
        f"- Top10 KS matched -> BUT bad: `{summary['matched_top10_mean_ks']:.4f}`",
        f"- Selected modes: `{summary['selected_modes']}`",
        f"- Target region counts: `{summary['target_region_counts']}`",
        "",
        "## Remaining Largest Matched KS",
        "",
        "| feature | matched KS | target median | matched median |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in worst.iterrows():
        report.append(f"| {row['feature']} | {float(row['ks_vs_but_bad']):.4f} | {float(row['target_median']):.4f} | {float(row['group_median']):.4f} |")
    report.extend(["", "## Candidate Mode KS Summary", "", "| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for _, row in mode_summary.head(12).iterrows():
        report.append(
            f"| {row['mode']} | {int(row['rows'])} | {float(row['mean_ks']):.4f} | {float(row['top10_mean_ks']):.4f} | "
            f"{float(row['zcr_ks']):.4f} | {float(row['diff_zcr_ks']):.4f} | {float(row['band15_30_ks']):.4f} | "
            f"{float(row['non_qrs_diff_p95_ks']):.4f} | {float(row['sqi_bSQI_ks']):.4f} | {float(row['flatline_ratio_ks']):.4f} |"
        )
    report.extend(
        [
            "",
            "## Figures",
            "",
            f"![PCA]({pca_path})",
            "",
            f"![Waveforms]({wave_path})",
            "",
            "## Files",
            "",
            f"- Matched signals: `{OUT_DIR / (stem + '_signals.npz')}`",
            f"- Manifest: `{OUT_DIR / (stem + '_manifest.csv')}`",
            f"- Feature KS: `{OUT_DIR / (stem + '_feature_ks.csv')}`",
        ]
    )
    report_text = "\n".join(report)
    (OUT_DIR / f"{stem}_report.md").write_text(report_text, encoding="utf-8")
    (REPORT_OUT_DIR / f"{stem}_report.md").write_text(report_text, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
