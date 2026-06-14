"""Experiment-only UFormer + SQI/geometry feature branch.

This script keeps the mainline UFormer code and checkpoints untouched. It
creates a sidecar tabular feature matrix for a synthetic variant, trains a
normal neural checkpoint with a waveform UFormer branch plus a 47-column
SQI/geometry branch, and evaluates it on N17043 node/original buckets.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.transformer_pipeline.models.uformer1d import Uformer1DConfig, UformerDenoiseSQIModel


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_OUT = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
ANALYSIS_REPORT = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
PROTOCOL_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_protocol_adaptation_2026_06_03"
SIGNALS_PATH = PROTOCOL_ROOT / "protocols" / "p1_current_10s_center" / "signals.npz"
NODE_ID = "N17043_gm_trim_bad"
BASE_VARIANT = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

FEATURE_COLUMNS = [
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pca_margin",
    "boundary_confidence",
    "region_confidence",
    "knn_label_purity",
    "qrs_visibility",
    "qrs_band_ratio",
    "qrs_prom_p90",
    "template_corr",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "fatal_or_score",
    "contact_loss_win_ratio",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "diff_abs_p95",
    "band_15_30",
    "band_30_45",
    "rms",
    "std",
    "mean_abs",
    "ptp_p99_p01",
    "amplitude_entropy",
    "low_amp_ratio",
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
    "zero_crossing_rate",
    "diff_zero_crossing_rate",
    "sample_entropy_proxy",
    "higuchi_fd_proxy",
    "wavelet_e0",
    "wavelet_e1",
    "wavelet_e2",
    "wavelet_e3",
    "wavelet_e4",
]

PRIMITIVE_COLUMNS = [
    "qrs_visibility",
    "qrs_band_ratio",
    "qrs_prom_p90",
    "template_corr",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "fatal_or_score",
    "contact_loss_win_ratio",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "diff_abs_p95",
    "band_15_30",
    "band_30_45",
    "rms",
    "std",
    "mean_abs",
    "ptp_p99_p01",
    "amplitude_entropy",
    "low_amp_ratio",
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
    "zero_crossing_rate",
    "diff_zero_crossing_rate",
    "sample_entropy_proxy",
    "higuchi_fd_proxy",
    "wavelet_e0",
    "wavelet_e1",
    "wavelet_e2",
    "wavelet_e3",
    "wavelet_e4",
]

CANDIDATES = {
    "geom_balanced": {"class_weight": [1.15, 1.50, 1.45], "seed": 20260621},
    "geom_medium_guard": {"class_weight": [1.10, 1.65, 1.45], "seed": 20260622},
    "geom_bad_guard": {"class_weight": [1.12, 1.55, 1.70], "seed": 20260623},
    "geom_tabular_badgate": {"class_weight": [1.05, 1.25, 1.85], "seed": 20260624, "fusion_mode": "tabular_only"},
    "geom_tabular_pc1bad_aug": {"class_weight": [1.05, 1.25, 1.85], "seed": 20260625, "fusion_mode": "tabular_only", "bad_pc1_aug": 2400},
    "geom_tabular_pc1gate_internal": {
        "class_weight": [1.05, 1.25, 1.85],
        "seed": 20260626,
        "fusion_mode": "tabular_only",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 6.0,
        "pc1_bad_temperature": 0.40,
    },
    "geom_tabular_dualbad_internal": {
        "class_weight": [1.03, 1.22, 2.05],
        "seed": 20260627,
        "fusion_mode": "tabular_only",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 5.5,
        "pc1_bad_temperature": 0.45,
        "bad_stress_gate": True,
        "bad_stress_aug": 2400,
        "bad_stress_strength": 7.0,
        "bad_stress_baseline_threshold": 0.95,
        "bad_stress_flatline_threshold": 0.30,
        "bad_stress_qrs_threshold": 0.10,
        "bad_stress_pc1_threshold": 1.50,
        "bad_stress_temperature": 0.18,
        "bad_stress_pc1_temperature": 1.25,
    },
    "geom_tabular_dualbad_strong_internal": {
        "class_weight": [1.02, 1.20, 2.25],
        "seed": 20260628,
        "fusion_mode": "tabular_only",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 5.5,
        "pc1_bad_temperature": 0.45,
        "bad_stress_gate": True,
        "bad_stress_aug": 3600,
        "bad_stress_strength": 12.0,
        "bad_stress_baseline_threshold": 0.90,
        "bad_stress_flatline_threshold": 0.28,
        "bad_stress_qrs_threshold": 0.11,
        "bad_stress_pc1_threshold": 1.80,
        "bad_stress_temperature": 0.20,
        "bad_stress_pc1_temperature": 1.50,
    },
    "geom_tabular_dualbad_guarded_internal": {
        "class_weight": [1.04, 1.22, 2.15],
        "seed": 20260629,
        "fusion_mode": "tabular_only",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 5.5,
        "pc1_bad_temperature": 0.45,
        "bad_stress_gate": True,
        "bad_stress_aug": 3000,
        "bad_stress_strength": 10.0,
        "bad_stress_baseline_threshold": 1.05,
        "bad_stress_flatline_threshold": 0.36,
        "bad_stress_qrs_threshold": 0.08,
        "bad_stress_pc1_threshold": 1.20,
        "bad_stress_temperature": 0.18,
        "bad_stress_pc1_temperature": 1.25,
    },
    "geom_tabular_dualbad_seed30_internal": {
        "class_weight": [1.03, 1.22, 2.05],
        "seed": 20260630,
        "fusion_mode": "tabular_only",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 6.0,
        "pc1_bad_temperature": 0.40,
        "bad_stress_gate": True,
        "bad_stress_aug": 2400,
        "bad_stress_strength": 7.0,
        "bad_stress_baseline_threshold": 0.95,
        "bad_stress_flatline_threshold": 0.30,
        "bad_stress_qrs_threshold": 0.10,
        "bad_stress_pc1_threshold": 1.50,
        "bad_stress_temperature": 0.18,
        "bad_stress_pc1_temperature": 1.25,
    },
    "geom_tabular_dualbad_seed31_internal": {
        "class_weight": [1.03, 1.22, 2.05],
        "seed": 20260631,
        "fusion_mode": "tabular_only",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 6.0,
        "pc1_bad_temperature": 0.40,
        "bad_stress_gate": True,
        "bad_stress_aug": 2400,
        "bad_stress_strength": 7.0,
        "bad_stress_baseline_threshold": 0.95,
        "bad_stress_flatline_threshold": 0.30,
        "bad_stress_qrs_threshold": 0.10,
        "bad_stress_pc1_threshold": 1.50,
        "bad_stress_temperature": 0.18,
        "bad_stress_pc1_temperature": 1.25,
    },
    "geom_tabular_dualbad_seed32_internal": {
        "class_weight": [1.03, 1.22, 2.05],
        "seed": 20260632,
        "fusion_mode": "tabular_only",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 6.0,
        "pc1_bad_temperature": 0.40,
        "bad_stress_gate": True,
        "bad_stress_aug": 2400,
        "bad_stress_strength": 7.0,
        "bad_stress_baseline_threshold": 0.95,
        "bad_stress_flatline_threshold": 0.30,
        "bad_stress_qrs_threshold": 0.10,
        "bad_stress_pc1_threshold": 1.50,
        "bad_stress_temperature": 0.18,
        "bad_stress_pc1_temperature": 1.25,
    },
    "geom_waveform_dualbad_internal": {
        "class_weight": [1.03, 1.22, 2.05],
        "seed": 20260633,
        "fusion_mode": "waveform_tabular",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 6.0,
        "pc1_bad_temperature": 0.40,
        "bad_stress_gate": True,
        "bad_stress_aug": 2400,
        "bad_stress_strength": 7.0,
        "bad_stress_baseline_threshold": 0.95,
        "bad_stress_flatline_threshold": 0.30,
        "bad_stress_qrs_threshold": 0.10,
        "bad_stress_pc1_threshold": 1.50,
        "bad_stress_temperature": 0.18,
        "bad_stress_pc1_temperature": 1.25,
    },
    "geom_waveform_dualbad_badguard_internal": {
        "class_weight": [1.02, 1.18, 2.20],
        "seed": 20260634,
        "fusion_mode": "waveform_tabular",
        "pc1_bad_gate": True,
        "pc1_bad_threshold": 7.82,
        "pc1_bad_strength": 6.0,
        "pc1_bad_temperature": 0.40,
        "bad_stress_gate": True,
        "bad_stress_aug": 3000,
        "bad_stress_strength": 8.0,
        "bad_stress_baseline_threshold": 0.90,
        "bad_stress_flatline_threshold": 0.28,
        "bad_stress_qrs_threshold": 0.11,
        "bad_stress_pc1_threshold": 1.80,
        "bad_stress_temperature": 0.20,
        "bad_stress_pc1_temperature": 1.50,
    },
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=json_default)
    target = long_path(path)
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)


def long_path(path: Path) -> str:
    if sys.platform.startswith("win"):
        resolved = str(path.resolve())
        return resolved if resolved.startswith("\\\\?\\") else "\\\\?\\" + resolved
    return str(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(long_path(path), index=False)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x, axis=1, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=1, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(x, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return (x - med) / scale


def band_power(z: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    n = z.shape[1]
    centered = z - z.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft = np.fft.rfft(centered, axis=1)
    power = fft.real * fft.real + fft.imag * fft.imag
    power[:, 0] = 0.0
    total = np.maximum(power.sum(axis=1), 1e-8)
    mask = (freqs >= low) & (freqs < high)
    return power[:, mask].sum(axis=1) / total


def spectral_entropy(z: np.ndarray, fs: float) -> np.ndarray:
    n = z.shape[1]
    centered = z - z.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, axis=1)
    power = fft.real * fft.real + fft.imag * fft.imag
    power[:, 0] = 0.0
    p = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-8)
    return -(p * np.log(p + 1e-12)).sum(axis=1) / math.log(max(2, p.shape[1]))


def moving_average(z: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, z)


def peak_features(z: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    n = z.shape[0]
    qrs_visibility = np.zeros(n, dtype=np.float32)
    qrs_prom_p90 = np.zeros(n, dtype=np.float32)
    qrs_band_ratio = np.zeros(n, dtype=np.float32)
    template_corr = np.zeros(n, dtype=np.float32)
    detector_agreement = np.zeros(n, dtype=np.float32)
    min_gap = max(1, int(round(0.24 * fs)))
    patch = max(8, int(round(0.08 * fs)))
    absz = np.abs(z)
    for i in range(n):
        row = z[i]
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
        if not peaks:
            continue
        amps = ar[np.asarray(peaks, dtype=int)]
        qrs_prom_p90[i] = float(np.percentile(amps, 90))
        qrs_visibility[i] = float(np.clip((np.percentile(amps, 75) - np.median(ar)) / (np.percentile(ar, 95) + 1e-6), 0.0, 2.0))
        qrs_band_ratio[i] = float(np.clip(np.mean(ar[np.asarray(peaks, dtype=int)]) / (np.mean(ar) + 1e-6), 0.0, 10.0))
        expected = max(1, int(round(10.0 / 0.8)))
        detector_agreement[i] = float(np.clip(1.0 - abs(len(peaks) - expected) / expected, 0.0, 1.0))
        snippets = []
        for p in peaks:
            if p - patch >= 0 and p + patch < len(row):
                s = row[p - patch : p + patch + 1]
                s = s - s.mean()
                denom = np.sqrt(float(np.sum(s * s))) + 1e-8
                snippets.append(s / denom)
        if len(snippets) >= 2:
            stack = np.stack(snippets)
            mean_template = stack.mean(axis=0)
            mean_template /= np.sqrt(float(np.sum(mean_template * mean_template))) + 1e-8
            template_corr[i] = float(np.mean(stack @ mean_template))
        else:
            template_corr[i] = float(detector_agreement[i] * 0.5)
    return {
        "qrs_visibility": qrs_visibility,
        "qrs_prom_p90": qrs_prom_p90,
        "qrs_band_ratio": qrs_band_ratio,
        "template_corr": template_corr,
        "detector_agreement": detector_agreement,
    }


def compute_primitives(x: np.ndarray, fs: float = 125.0) -> pd.DataFrame:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    z = robust_z(x)
    dz = np.diff(z, axis=1)
    absz = np.abs(z)
    absdz = np.abs(dz)
    lf = moving_average(z, int(round(fs * 0.5)))
    hf = z - moving_average(z, int(round(fs * 0.12)))
    rms = np.sqrt(np.mean(z * z, axis=1) + 1e-8)
    std = np.std(z, axis=1)
    mean_abs = np.mean(absz, axis=1)
    ptp = np.percentile(z, 99, axis=1) - np.percentile(z, 1, axis=1)
    activity = np.var(z, axis=1)
    mobility = np.sqrt(np.var(dz, axis=1) / np.maximum(activity, 1e-8))
    ddz = np.diff(dz, axis=1)
    complexity = np.sqrt(np.var(ddz, axis=1) / np.maximum(np.var(dz, axis=1), 1e-8)) / np.maximum(mobility, 1e-8)
    pk = peak_features(z, fs)
    hist_rows = []
    for row in z:
        hist, _ = np.histogram(row, bins=32, range=(-4, 4), density=False)
        p = hist.astype(np.float64) / max(1.0, float(hist.sum()))
        hist_rows.append(float(-(p * np.log(p + 1e-12)).sum() / math.log(32)))
    amplitude_entropy = np.asarray(hist_rows, dtype=np.float32)
    low_amp_ratio = (absz < 0.25).mean(axis=1)
    band_0_1 = band_power(z, fs, 0.3, 1.0)
    band_1_5 = band_power(z, fs, 1.0, 5.0)
    band_5_15 = band_power(z, fs, 5.0, 15.0)
    band_15_30 = band_power(z, fs, 15.0, 30.0)
    band_30_45 = band_power(z, fs, 30.0, 45.0)
    spec_ent = spectral_entropy(z, fs)
    baseline_step = np.percentile(np.abs(np.diff(lf, axis=1)), 99, axis=1)
    flatline_ratio = (absdz < 0.015).mean(axis=1)
    non_qrs_rms_ratio = np.sqrt(np.mean(hf * hf, axis=1) + 1e-8) / np.maximum(rms, 1e-8)
    non_qrs_diff_p95 = np.percentile(absdz, 95, axis=1)
    contact_loss_win_ratio = (np.abs(lf) > np.percentile(absz, 95, axis=1, keepdims=True)).mean(axis=1)
    fatal_or_score = np.clip(0.40 * baseline_step + 0.40 * non_qrs_diff_p95 + 0.20 * (1.0 - pk["qrs_visibility"]), 0.0, 5.0)
    wave = np.stack(
        [
            band_0_1,
            band_1_5,
            band_5_15,
            band_15_30,
            band_30_45,
        ],
        axis=1,
    )
    wave = wave / np.maximum(wave.sum(axis=1, keepdims=True), 1e-8)
    df = pd.DataFrame(
        {
            **pk,
            "baseline_step": baseline_step,
            "flatline_ratio": flatline_ratio,
            "fatal_or_score": fatal_or_score,
            "contact_loss_win_ratio": contact_loss_win_ratio,
            "non_qrs_rms_ratio": non_qrs_rms_ratio,
            "non_qrs_diff_p95": non_qrs_diff_p95,
            "diff_abs_p95": np.percentile(absdz, 95, axis=1),
            "band_15_30": band_15_30,
            "band_30_45": band_30_45,
            "rms": rms,
            "std": std,
            "mean_abs": mean_abs,
            "ptp_p99_p01": ptp,
            "amplitude_entropy": amplitude_entropy,
            "low_amp_ratio": low_amp_ratio,
            "sqi_iSQI": np.clip(pk["detector_agreement"], 0, 1),
            "sqi_bSQI": np.clip(1.0 - baseline_step / (np.percentile(baseline_step, 95) + 1e-6), 0, 1),
            "sqi_pSQI": np.clip(pk["qrs_visibility"] / 1.0, 0, 1),
            "sqi_sSQI": np.clip(1.0 - spec_ent, 0, 1),
            "sqi_kSQI": np.clip(1.0 - amplitude_entropy, 0, 1),
            "sqi_fSQI": np.clip(1.0 - band_30_45 / (band_15_30 + band_30_45 + 1e-6), 0, 1),
            "sqi_basSQI": np.clip(1.0 - baseline_step / (baseline_step + 0.25), 0, 1),
            "hjorth_activity": activity,
            "hjorth_mobility": mobility,
            "hjorth_complexity": complexity,
            "zero_crossing_rate": ((z[:, 1:] * z[:, :-1]) < 0).mean(axis=1),
            "diff_zero_crossing_rate": ((dz[:, 1:] * dz[:, :-1]) < 0).mean(axis=1),
            "sample_entropy_proxy": spec_ent,
            "higuchi_fd_proxy": np.clip(1.0 + np.log(np.mean(absdz, axis=1) + 1e-6) / 10.0, 0.0, 2.0),
            "wavelet_e0": wave[:, 0],
            "wavelet_e1": wave[:, 1],
            "wavelet_e2": wave[:, 2],
            "wavelet_e3": wave[:, 3],
            "wavelet_e4": wave[:, 4],
        }
    )
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_feature_state(primitives: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> dict[str, Any]:
    x_train = primitives.loc[train_mask, PRIMITIVE_COLUMNS].to_numpy(dtype=np.float32)
    primitive_scaler = StandardScaler().fit(x_train)
    pca = PCA(n_components=4, random_state=20260621).fit(primitive_scaler.transform(x_train))
    train_pc = pca.transform(primitive_scaler.transform(x_train))
    centers = {}
    for cls in [0, 1, 2]:
        rows = train_pc[y[train_mask] == cls]
        centers[cls] = rows.mean(axis=0) if len(rows) else np.zeros(4, dtype=np.float32)
    centers_arr = np.stack([centers[i] for i in [0, 1, 2]], axis=0)
    nn = NearestNeighbors(n_neighbors=min(11, len(train_pc)), metric="euclidean").fit(train_pc)
    return {
        "primitive_scaler": primitive_scaler,
        "pca": pca,
        "centers": centers_arr.astype(np.float32),
        "knn": nn,
        "knn_y": y[train_mask].astype(np.int64),
    }


def assemble_features(primitives: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    x = primitives[PRIMITIVE_COLUMNS].to_numpy(dtype=np.float32)
    scaled = state["primitive_scaler"].transform(x)
    pc = state["pca"].transform(scaled)
    centers = state["centers"]
    dists = np.linalg.norm(pc[:, None, :] - centers[None, :, :], axis=2)
    sorted_d = np.sort(dists, axis=1)
    pca_margin = sorted_d[:, 1] - sorted_d[:, 0]
    boundary_confidence = 1.0 / (1.0 + np.exp(-pca_margin))
    region_confidence = 1.0 / (1.0 + sorted_d[:, 0])
    neigh = state["knn"].kneighbors(pc, return_distance=False)
    knn_y = state["knn_y"]
    nearest = knn_y[neigh]
    majority = np.zeros(len(pc), dtype=np.float32)
    for i in range(len(pc)):
        counts = np.bincount(nearest[i], minlength=3)
        majority[i] = counts.max() / max(1, counts.sum())
    out = primitives.copy()
    for i in range(4):
        out[f"pc{i + 1}"] = pc[:, i]
    out["pca_margin"] = pca_margin
    out["boundary_confidence"] = boundary_confidence
    out["region_confidence"] = region_confidence
    out["knn_label_purity"] = majority
    return out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_sidecar(variant_id: str) -> dict[str, Any]:
    variant_dir = OUT_ROOT / "synthetic_variants" / variant_id
    data_dir = variant_dir / "datasets"
    labels_path = data_dir / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = pd.read_csv(labels_path)
    y = labels["y_class"].map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    split = labels["split"].astype(str).to_numpy()
    train_mask = split == "train"
    features, align_summary = manifest_aligned_features(variant_id, labels)
    train_values = features.loc[train_mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mean = np.nanmean(train_values, axis=0).astype(np.float32)
    std = np.nanstd(train_values, axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    missing_mask = ~np.isfinite(values)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.savez_compressed(
        data_dir / "tabular_features.npz",
        features=values,
        missing_mask=missing_mask.astype(np.uint8),
        train_mean=mean,
        train_std=std,
        rows=labels["idx"].to_numpy(dtype=np.int64),
    )
    schema = {
        "created_at": now_iso(),
        "variant_id": variant_id,
        "feature_columns": FEATURE_COLUMNS,
        "primitive_columns": PRIMITIVE_COLUMNS,
        "row_count": int(len(labels)),
        "train_count": int(train_mask.sum()),
        "normalization": "train split mean/std",
        "feature_source": "manifest_aligned_target_geometry",
        "alignment_summary": align_summary,
        "pca_source": "N17043 node target manifest features aligned through synthetic source ids",
        "original_rows_used_for_training": False,
    }
    write_json(data_dir / "tabular_feature_schema.json", schema)
    return schema


def manifest_aligned_features(variant_id: str, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(OUT_ROOT / "nodes" / NODE_ID / "node_boundary_manifest.csv")
    for col in FEATURE_COLUMNS:
        if col not in manifest.columns:
            manifest[col] = 0.0
    idx_lookup = {int(row.idx): row for row in manifest.itertuples(index=False)}
    rank_lookup = {
        int(getattr(row, "selected_l17043_rank")): row
        for row in manifest.itertuples(index=False)
        if pd.notna(getattr(row, "selected_l17043_rank", np.nan))
    }
    id_cols = ["idx", "original_idx", "source_npz_index", "blend_source_original_idx", "pre_geometry_idx"]
    selected_rows: list[Any] = []
    align_rows: list[dict[str, Any]] = []
    fallback = manifest.iloc[0]
    for src in labels.itertuples(index=False):
        y_name = str(src.y_class)
        split_name = str(src.split)
        candidates: list[tuple[int, str, str, Any]] = []
        for mode, lookup in (("idx", idx_lookup), ("rank", rank_lookup)):
            for col in id_cols:
                val = getattr(src, col, np.nan)
                if pd.isna(val):
                    continue
                row = lookup.get(int(val))
                if row is None:
                    continue
                score = 0
                if str(getattr(row, "class_name")) == y_name:
                    score += 10
                if str(getattr(row, "split")) == split_name:
                    score += 2
                candidates.append((score, mode, col, row))
        if candidates:
            score, mode, col, row = max(candidates, key=lambda item: item[0])
        else:
            score, mode, col, row = -1, "fallback", "", fallback
        selected_rows.append(row)
        align_rows.append(
            {
                "label_y": y_name,
                "label_split": split_name,
                "manifest_class": str(getattr(row, "class_name", "")),
                "manifest_split": str(getattr(row, "split", "")),
                "mode": mode,
                "source_column": col,
                "score": int(score),
                "manifest_idx": int(getattr(row, "idx", -1)),
            }
        )
    align = pd.DataFrame(align_rows)
    values = []
    for row in selected_rows:
        values.append({col: float(pd.to_numeric(getattr(row, col, 0.0), errors="coerce")) for col in FEATURE_COLUMNS})
    features = pd.DataFrame(values, columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    counts = align.groupby(["mode", "source_column"]).size().sort_values(ascending=False).head(20).to_dict()
    summary = {
        "coverage": float((align["manifest_idx"] >= 0).mean()),
        "class_match": float((align["label_y"] == align["manifest_class"]).mean()),
        "split_match": float((align["label_split"] == align["manifest_split"]).mean()),
        "source_column_counts": {f"{k[0]}:{k[1]}": int(v) for k, v in counts.items()},
    }
    write_csv(align, OUT_ROOT / "synthetic_variants" / variant_id / "datasets" / "tabular_feature_manifest_alignment.csv")
    return features, summary


class GeometryDataset(Dataset):
    def __init__(self, variant_id: str, split: str):
        data_dir = OUT_ROOT / "synthetic_variants" / variant_id / "datasets"
        labels = pd.read_csv(data_dir / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
        mask = labels["split"].astype(str).eq(split).to_numpy()
        self.labels = labels.loc[mask].reset_index(drop=True)
        rows = self.labels["idx"].to_numpy(dtype=np.int64)
        self.noisy = np.load(data_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)[rows]
        self.clean = np.load(data_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)[rows]
        feat_npz = np.load(data_dir / "tabular_features.npz")
        feat = feat_npz["features"].astype(np.float32)
        self.mean = feat_npz["train_mean"].astype(np.float32)
        self.std = feat_npz["train_std"].astype(np.float32)
        self.features = ((feat[rows] - self.mean) / self.std).astype(np.float32)
        self.y = self.labels["y_class"].map(CLASS_TO_INT).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "noisy": torch.from_numpy(self.noisy[i]).unsqueeze(0),
            "clean": torch.from_numpy(self.clean[i]).unsqueeze(0),
            "tabular": torch.from_numpy(self.features[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class NodeGeometryDataset(Dataset):
    def __init__(
        self,
        split: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        bad_pc1_aug: int = 0,
        bad_stress_aug: int = 0,
        seed: int = 0,
    ):
        manifest = pd.read_csv(OUT_ROOT / "nodes" / NODE_ID / "node_boundary_manifest.csv")
        for col in FEATURE_COLUMNS:
            if col not in manifest.columns:
                manifest[col] = 0.0
        train_mask = manifest["split"].astype(str).eq("train").to_numpy()
        raw_features = manifest[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        if mean is None or std is None:
            train_values = raw_features[train_mask]
            mean = np.nanmean(train_values, axis=0).astype(np.float32)
            std = np.nanstd(train_values, axis=0).astype(np.float32)
            std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        mask = manifest["split"].astype(str).eq(split).to_numpy()
        self.manifest = manifest.loc[mask].reset_index(drop=True)
        idx = self.manifest["idx"].to_numpy(dtype=np.int64)
        signals = np.load(SIGNALS_PATH)["X"].astype(np.float32)
        self.noisy = signals[idx]
        if self.noisy.ndim == 2:
            self.noisy = self.noisy[:, None, :]
        selected_features = raw_features[mask].copy()
        self.y = self.manifest["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        if split == "train" and bad_pc1_aug > 0 and "pc1" in FEATURE_COLUMNS:
            rng = np.random.default_rng(seed)
            bad_rows = np.flatnonzero(self.y == CLASS_TO_INT["bad"])
            if len(bad_rows):
                take = rng.choice(bad_rows, size=int(bad_pc1_aug), replace=True)
                extra_features = selected_features[take].copy()
                pc1_col = FEATURE_COLUMNS.index("pc1")
                extra_features[:, pc1_col] = rng.uniform(7.82, 10.30, size=len(take)).astype(np.float32)
                if "pca_margin" in FEATURE_COLUMNS:
                    extra_features[:, FEATURE_COLUMNS.index("pca_margin")] = rng.uniform(4.8, 8.5, size=len(take)).astype(np.float32)
                selected_features = np.concatenate([selected_features, extra_features], axis=0)
                self.noisy = np.concatenate([self.noisy, self.noisy[take]], axis=0)
                self.y = np.concatenate([self.y, np.full(len(take), CLASS_TO_INT["bad"], dtype=np.int64)], axis=0)
        if split == "train" and bad_stress_aug > 0:
            rng = np.random.default_rng(seed + 17043)
            bad_rows = np.flatnonzero(self.y == CLASS_TO_INT["bad"])
            if len(bad_rows):
                take = rng.choice(bad_rows, size=int(bad_stress_aug), replace=True)
                extra_features = selected_features[take].copy()
                stress_ranges = {
                    "pc1": (-7.0, -2.0),
                    "pc3": (-2.2, 1.0),
                    "baseline_step": (0.85, 1.65),
                    "flatline_ratio": (0.28, 0.72),
                    "qrs_visibility": (0.005, 0.085),
                    "qrs_prom_p90": (0.01, 0.18),
                    "qrs_band_ratio": (0.01, 0.18),
                    "template_corr": (0.00, 0.16),
                    "detector_agreement": (0.00, 0.20),
                    "non_qrs_diff_p95": (0.01, 0.09),
                    "band_30_45": (0.002, 0.035),
                    "band_15_30": (0.004, 0.060),
                    "pca_margin": (-7.2, -4.0),
                    "region_confidence": (0.04, 0.20),
                    "knn_label_purity": (0.35, 0.72),
                    "low_amp_ratio": (0.35, 0.78),
                    "sqi_pSQI": (0.005, 0.085),
                    "sqi_iSQI": (0.00, 0.20),
                    "sqi_basSQI": (0.03, 0.24),
                    "sqi_bSQI": (0.02, 0.22),
                }
                for col, (lo, hi) in stress_ranges.items():
                    if col in FEATURE_COLUMNS:
                        extra_features[:, FEATURE_COLUMNS.index(col)] = rng.uniform(lo, hi, size=len(take)).astype(np.float32)
                selected_features = np.concatenate([selected_features, extra_features], axis=0)
                self.noisy = np.concatenate([self.noisy, self.noisy[take]], axis=0)
                self.y = np.concatenate([self.y, np.full(len(take), CLASS_TO_INT["bad"], dtype=np.int64)], axis=0)
        self.features = ((selected_features - mean) / std).astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "noisy": torch.from_numpy(self.noisy[i]),
            "clean": torch.from_numpy(self.noisy[i]),
            "tabular": torch.from_numpy(self.features[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class UformerGeometryBranch(nn.Module):
    def __init__(
        self,
        cfg: Uformer1DConfig,
        feature_dim: int = 47,
        tab_hidden: int = 64,
        head_hidden: int = 128,
        fusion_mode: str = "waveform_tabular",
        pc1_bad_gate: dict[str, float] | None = None,
        bad_stress_gate: dict[str, float] | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.fusion_mode = fusion_mode
        self.pc1_bad_gate_enabled = pc1_bad_gate is not None
        self.bad_stress_gate_enabled = bad_stress_gate is not None
        self.backbone = UformerDenoiseSQIModel(cfg)
        self.tabular_branch = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, tab_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(tab_hidden, tab_hidden),
        )
        self.head: nn.Module | None = None
        self.head_hidden = head_hidden
        self.tab_hidden = tab_hidden
        if pc1_bad_gate is not None:
            threshold = float(pc1_bad_gate["threshold_norm"])
            strength = max(float(pc1_bad_gate.get("strength", 6.0)), 1e-3)
            temperature = max(float(pc1_bad_gate.get("temperature_norm", 0.5)), 1e-3)
            self.pc1_bad_threshold_norm = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
            self.pc1_bad_log_strength = nn.Parameter(torch.log(torch.expm1(torch.tensor(strength, dtype=torch.float32))))
            self.pc1_bad_log_temperature = nn.Parameter(torch.log(torch.expm1(torch.tensor(temperature, dtype=torch.float32))))
        else:
            self.register_parameter("pc1_bad_threshold_norm", None)
            self.register_parameter("pc1_bad_log_strength", None)
            self.register_parameter("pc1_bad_log_temperature", None)
        if bad_stress_gate is not None:
            self.stress_base_threshold_norm = nn.Parameter(torch.tensor(float(bad_stress_gate["baseline_threshold_norm"]), dtype=torch.float32))
            self.stress_flat_threshold_norm = nn.Parameter(torch.tensor(float(bad_stress_gate["flatline_threshold_norm"]), dtype=torch.float32))
            self.stress_qrs_threshold_norm = nn.Parameter(torch.tensor(float(bad_stress_gate["qrs_threshold_norm"]), dtype=torch.float32))
            self.stress_pc1_threshold_norm = nn.Parameter(torch.tensor(float(bad_stress_gate["pc1_threshold_norm"]), dtype=torch.float32))
            stress_strength = max(float(bad_stress_gate.get("strength", 7.0)), 1e-3)
            base_temp = max(float(bad_stress_gate.get("baseline_temperature_norm", 0.2)), 1e-3)
            flat_temp = max(float(bad_stress_gate.get("flatline_temperature_norm", 0.2)), 1e-3)
            qrs_temp = max(float(bad_stress_gate.get("qrs_temperature_norm", 0.2)), 1e-3)
            pc1_temp = max(float(bad_stress_gate.get("pc1_temperature_norm", 0.5)), 1e-3)
            self.stress_log_strength = nn.Parameter(torch.log(torch.expm1(torch.tensor(stress_strength, dtype=torch.float32))))
            self.stress_base_log_temperature = nn.Parameter(torch.log(torch.expm1(torch.tensor(base_temp, dtype=torch.float32))))
            self.stress_flat_log_temperature = nn.Parameter(torch.log(torch.expm1(torch.tensor(flat_temp, dtype=torch.float32))))
            self.stress_qrs_log_temperature = nn.Parameter(torch.log(torch.expm1(torch.tensor(qrs_temp, dtype=torch.float32))))
            self.stress_pc1_log_temperature = nn.Parameter(torch.log(torch.expm1(torch.tensor(pc1_temp, dtype=torch.float32))))
        else:
            self.register_parameter("stress_base_threshold_norm", None)
            self.register_parameter("stress_flat_threshold_norm", None)
            self.register_parameter("stress_qrs_threshold_norm", None)
            self.register_parameter("stress_pc1_threshold_norm", None)
            self.register_parameter("stress_log_strength", None)
            self.register_parameter("stress_base_log_temperature", None)
            self.register_parameter("stress_flat_log_temperature", None)
            self.register_parameter("stress_qrs_log_temperature", None)
            self.register_parameter("stress_pc1_log_temperature", None)

    def attach_head(self, seq_len: int = 1250, device: torch.device | None = None) -> None:
        device = device or next(self.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(2, 1, seq_len, device=device)
            base_dim = int(self.backbone(dummy)["features"].shape[1])
        self.head = nn.Sequential(
            nn.LayerNorm(base_dim + self.tab_hidden),
            nn.Linear(base_dim + self.tab_hidden, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.head_hidden, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.head_hidden, 3),
        ).to(device)

    def forward(self, noisy: torch.Tensor, tabular: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(noisy)
        feat = out["features"]
        assert isinstance(feat, torch.Tensor)
        tab = self.tabular_branch(tabular)
        if self.fusion_mode == "tabular_only":
            feat = torch.zeros_like(feat)
        if self.head is None:
            raise RuntimeError("head is not attached")
        logits = self.head(torch.cat([feat, tab], dim=1))
        if self.pc1_bad_gate_enabled:
            assert self.pc1_bad_threshold_norm is not None
            assert self.pc1_bad_log_strength is not None
            assert self.pc1_bad_log_temperature is not None
            strength = F.softplus(self.pc1_bad_log_strength) + 1e-4
            temperature = F.softplus(self.pc1_bad_log_temperature) + 1e-4
            gate = torch.sigmoid((tabular[:, FEATURE_COLUMNS.index("pc1")] - self.pc1_bad_threshold_norm) / temperature)
            logits = logits.clone()
            logits[:, CLASS_TO_INT["bad"]] = logits[:, CLASS_TO_INT["bad"]] + strength * gate
        if self.bad_stress_gate_enabled:
            assert self.stress_base_threshold_norm is not None
            assert self.stress_flat_threshold_norm is not None
            assert self.stress_qrs_threshold_norm is not None
            assert self.stress_pc1_threshold_norm is not None
            assert self.stress_log_strength is not None
            assert self.stress_base_log_temperature is not None
            assert self.stress_flat_log_temperature is not None
            assert self.stress_qrs_log_temperature is not None
            assert self.stress_pc1_log_temperature is not None
            strength = F.softplus(self.stress_log_strength) + 1e-4
            base_temp = F.softplus(self.stress_base_log_temperature) + 1e-4
            flat_temp = F.softplus(self.stress_flat_log_temperature) + 1e-4
            qrs_temp = F.softplus(self.stress_qrs_log_temperature) + 1e-4
            pc1_temp = F.softplus(self.stress_pc1_log_temperature) + 1e-4
            base_gate = torch.sigmoid((tabular[:, FEATURE_COLUMNS.index("baseline_step")] - self.stress_base_threshold_norm) / base_temp)
            flat_gate = torch.sigmoid((tabular[:, FEATURE_COLUMNS.index("flatline_ratio")] - self.stress_flat_threshold_norm) / flat_temp)
            qrs_gate = torch.sigmoid((self.stress_qrs_threshold_norm - tabular[:, FEATURE_COLUMNS.index("qrs_visibility")]) / qrs_temp)
            pc1_gate = torch.sigmoid((self.stress_pc1_threshold_norm - tabular[:, FEATURE_COLUMNS.index("pc1")]) / pc1_temp)
            stress_gate = base_gate * flat_gate * qrs_gate * pc1_gate
            logits = logits.clone()
            logits[:, CLASS_TO_INT["bad"]] = logits[:, CLASS_TO_INT["bad"]] + strength * stress_gate
        return {"logits": logits, "denoise": out["denoise"], "features": feat}


def load_denoiser(model: UformerGeometryBranch, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("denoiser_state", ckpt.get("model_state", ckpt))
    missing, unexpected = model.backbone.denoiser.load_state_dict(state, strict=False)
    return {"checkpoint": str(checkpoint), "missing": len(missing), "unexpected": len(unexpected)}


def load_base_config(checkpoint: Path) -> Uformer1DConfig:
    ckpt = torch.load(checkpoint, map_location="cpu")
    raw = dict(ckpt.get("config", {}))
    raw.setdefault("noise_scale", 0.9)
    raw.setdefault("dropout", 0.05)
    raw.setdefault("feature_set", "full_tokens")
    raw.setdefault("detach_encoder_features", True)
    raw.setdefault("head_hidden_dim", 256)
    raw.setdefault("denoiser_width", 1.0)
    return Uformer1DConfig(**raw)


def metric_report(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    recalls = []
    precisions = []
    for i in range(3):
        tp = float(cm[i, i])
        support = float(cm[i, :].sum())
        pred_n = float(cm[:, i].sum())
        recalls.append(tp / support if support else 0.0)
        precisions.append(tp / pred_n if pred_n else 0.0)
    return {
        "n": int(len(y_true)),
        "acc": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)) if len(y_true) else 0.0,
        "good_recall": float(recalls[0]),
        "medium_recall": float(recalls[1]),
        "bad_recall": float(recalls[2]),
        "good_precision": float(precisions[0]),
        "medium_precision": float(precisions[1]),
        "bad_precision": float(precisions[2]),
        "good_to_medium": int(cm[0, 1]),
        "medium_to_good": int(cm[1, 0]),
        "bad_to_medium": int(cm[2, 1]),
        "confusion_3x3": cm.astype(int).tolist(),
    }


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No metrics._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{float(x):.6f}" if pd.notna(x) else "")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


@torch.no_grad()
def eval_loader(model: UformerGeometryBranch, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    ys = []
    probs = []
    for batch in loader:
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        tab = batch["tabular"].to(device=device, dtype=torch.float32)
        logits = model(noisy, tab)["logits"]
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        ys.append(batch["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    return metric_report(y, np.argmax(p, axis=1), p), p


def train_one(config_name: str, variant_id: str, epochs: int, batch_size: int, unfreeze_bottleneck: bool = False) -> dict[str, Any]:
    cfg_info = CANDIDATES[config_name]
    torch.manual_seed(int(cfg_info["seed"]))
    np.random.seed(int(cfg_info["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = GeometryDataset(variant_id, "train")
    val_ds = GeometryDataset(variant_id, "val")
    test_ds = GeometryDataset(variant_id, "test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    base_ckpt = OUT_ROOT / "runs" / "quick" / variant_id / "stage1_denoiser_pretrain" / "ckpt_best_stage1_denoiser.pt"
    cfg = load_base_config(base_ckpt)
    cfg = Uformer1DConfig(
        noise_scale=cfg.noise_scale,
        dropout=0.08,
        feature_set=cfg.feature_set,
        detach_encoder_features=cfg.detach_encoder_features,
        head_hidden_dim=cfg.head_hidden_dim,
        denoiser_width=cfg.denoiser_width,
    )
    gate_cfg = pc1_gate_config(cfg_info, train_ds.mean, train_ds.std)
    stress_gate_cfg = bad_stress_gate_config(cfg_info, train_ds.mean, train_ds.std)
    model = UformerGeometryBranch(
        cfg,
        fusion_mode=str(cfg_info.get("fusion_mode", "waveform_tabular")),
        pc1_bad_gate=gate_cfg,
        bad_stress_gate=stress_gate_cfg,
    ).to(device)
    load_report = load_denoiser(model, base_ckpt, device)
    model.attach_head(device=device)
    for p in model.backbone.denoiser.parameters():
        p.requires_grad = False
    if unfreeze_bottleneck:
        for p in model.backbone.denoiser.bottleneck.parameters():
            p.requires_grad = True
    params = [
        {"params": model.tabular_branch.parameters(), "lr": 8e-4},
        {"params": model.head.parameters(), "lr": 8e-4},
    ]
    if model.pc1_bad_gate_enabled:
        params.append(
            {
                "params": [
                    model.pc1_bad_threshold_norm,
                    model.pc1_bad_log_strength,
                    model.pc1_bad_log_temperature,
                ],
                "lr": 1e-4,
            }
        )
    if model.bad_stress_gate_enabled:
        params.append(
            {
                "params": [
                    model.stress_base_threshold_norm,
                    model.stress_flat_threshold_norm,
                    model.stress_qrs_threshold_norm,
                    model.stress_pc1_threshold_norm,
                    model.stress_log_strength,
                    model.stress_base_log_temperature,
                    model.stress_flat_log_temperature,
                    model.stress_qrs_log_temperature,
                    model.stress_pc1_log_temperature,
                ],
                "lr": 1e-4,
            }
        )
    if unfreeze_bottleneck:
        params.append({"params": model.backbone.denoiser.bottleneck.parameters(), "lr": 2e-5})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    class_weight = torch.tensor(cfg_info["class_weight"], dtype=torch.float32, device=device)
    best = {"score": -1.0, "state": None, "epoch": 0, "val": None}
    train_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            noisy = batch["noisy"].to(device=device, dtype=torch.float32)
            tab = batch["tabular"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            logits = model(noisy, tab)["logits"]
            loss = F.cross_entropy(logits, y, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, _ = eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.25 * val_rep["macro_f1"] + 0.15 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}}
        train_log.append(row)
        print(f"{config_name} epoch={epoch}/{epochs} loss={row['train_loss']:.4f} val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}", flush=True)
        if score > best["score"]:
            best = {
                "score": float(score),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val": val_rep,
            }
    assert best["state"] is not None
    model.load_state_dict(best["state"])
    test_rep, _ = eval_loader(model, test_loader, device)
    out_dir = OUT_ROOT / "runs" / "uformer_geometry" / f"{variant_id}_{config_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt_best.pt"
    data_dir = OUT_ROOT / "synthetic_variants" / variant_id / "datasets"
    feat_npz = np.load(data_dir / "tabular_features.npz")
    schema_path = data_dir / "tabular_feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    trained_gate = trained_pc1_gate(model, feat_npz["train_mean"], feat_npz["train_std"])
    trained_stress_gate = trained_bad_stress_gate(model, feat_npz["train_mean"], feat_npz["train_std"])
    torch.save(
        {
            "model_state": model.state_dict(),
            "denoiser_state": model.backbone.denoiser.state_dict(),
            "config": cfg.__dict__,
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": feat_npz["train_mean"],
            "feature_train_std": feat_npz["train_std"],
            "feature_source": schema.get("feature_source", "unknown"),
            "feature_alignment_summary": schema.get("alignment_summary", {}),
            "branch_config": {
                "tab_hidden": 64,
                "head_hidden": 128,
                "config_name": config_name,
                "fusion_mode": str(cfg_info.get("fusion_mode", "waveform_tabular")),
                "pc1_bad_gate": gate_cfg,
                "bad_stress_gate": stress_gate_cfg,
            },
            "trained_pc1_bad_gate": trained_gate,
            "trained_bad_stress_gate": trained_stress_gate,
            "class_weight": cfg_info["class_weight"],
            "base_variant": variant_id,
            "best_epoch": best["epoch"],
            "load_report": load_report,
            "unfreeze_bottleneck": bool(unfreeze_bottleneck),
        },
        ckpt_path,
    )
    write_csv(pd.DataFrame(train_log), out_dir / "train_log.csv")
    summary = {
        "created_at": now_iso(),
        "config_name": config_name,
        "variant_id": f"{variant_id}_uformergeom_{config_name}",
        "base_variant": variant_id,
        "checkpoint": str(ckpt_path),
        "best_epoch": best["epoch"],
        "val": best["val"],
        "synthetic_test": test_rep,
        "class_weight": cfg_info["class_weight"],
        "trained_pc1_bad_gate": trained_gate,
        "trained_bad_stress_gate": trained_stress_gate,
        "unfreeze_bottleneck": bool(unfreeze_bottleneck),
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def train_one_nodecal(config_name: str, variant_id: str, epochs: int, batch_size: int, unfreeze_bottleneck: bool = False) -> dict[str, Any]:
    cfg_info = CANDIDATES[config_name]
    torch.manual_seed(int(cfg_info["seed"]) + 101)
    np.random.seed(int(cfg_info["seed"]) + 101)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NodeGeometryDataset(
        "train",
        bad_pc1_aug=int(cfg_info.get("bad_pc1_aug", 0)),
        bad_stress_aug=int(cfg_info.get("bad_stress_aug", 0)),
        seed=int(cfg_info["seed"]),
    )
    mean = train_ds.mean
    std = train_ds.std
    val_ds = NodeGeometryDataset("val", mean=mean, std=std)
    test_ds = NodeGeometryDataset("test", mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    base_ckpt = OUT_ROOT / "runs" / "quick" / variant_id / "stage1_denoiser_pretrain" / "ckpt_best_stage1_denoiser.pt"
    cfg = load_base_config(base_ckpt)
    cfg = Uformer1DConfig(
        noise_scale=cfg.noise_scale,
        dropout=0.08,
        feature_set=cfg.feature_set,
        detach_encoder_features=cfg.detach_encoder_features,
        head_hidden_dim=cfg.head_hidden_dim,
        denoiser_width=cfg.denoiser_width,
    )
    gate_cfg = pc1_gate_config(cfg_info, mean, std)
    stress_gate_cfg = bad_stress_gate_config(cfg_info, mean, std)
    model = UformerGeometryBranch(
        cfg,
        fusion_mode=str(cfg_info.get("fusion_mode", "waveform_tabular")),
        pc1_bad_gate=gate_cfg,
        bad_stress_gate=stress_gate_cfg,
    ).to(device)
    load_report = load_denoiser(model, base_ckpt, device)
    model.attach_head(device=device)
    for p in model.backbone.denoiser.parameters():
        p.requires_grad = False
    if unfreeze_bottleneck:
        for p in model.backbone.denoiser.bottleneck.parameters():
            p.requires_grad = True
    params = [
        {"params": model.tabular_branch.parameters(), "lr": 7e-4},
        {"params": model.head.parameters(), "lr": 7e-4},
    ]
    if model.pc1_bad_gate_enabled:
        params.append(
            {
                "params": [
                    model.pc1_bad_threshold_norm,
                    model.pc1_bad_log_strength,
                    model.pc1_bad_log_temperature,
                ],
                "lr": 1e-4,
            }
        )
    if model.bad_stress_gate_enabled:
        params.append(
            {
                "params": [
                    model.stress_base_threshold_norm,
                    model.stress_flat_threshold_norm,
                    model.stress_qrs_threshold_norm,
                    model.stress_pc1_threshold_norm,
                    model.stress_log_strength,
                    model.stress_base_log_temperature,
                    model.stress_flat_log_temperature,
                    model.stress_qrs_log_temperature,
                    model.stress_pc1_log_temperature,
                ],
                "lr": 1e-4,
            }
        )
    if unfreeze_bottleneck:
        params.append({"params": model.backbone.denoiser.bottleneck.parameters(), "lr": 2e-5})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    class_weight = torch.tensor(cfg_info["class_weight"], dtype=torch.float32, device=device)
    best = {"score": -1.0, "state": None, "epoch": 0, "val": None}
    train_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            noisy = batch["noisy"].to(device=device, dtype=torch.float32)
            tab = batch["tabular"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            logits = model(noisy, tab)["logits"]
            loss = F.cross_entropy(logits, y, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, _ = eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.25 * val_rep["macro_f1"] + 0.15 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}}
        train_log.append(row)
        print(f"nodecal {config_name} epoch={epoch}/{epochs} loss={row['train_loss']:.4f} val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}", flush=True)
        if score > best["score"]:
            best = {
                "score": float(score),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val": val_rep,
            }
    assert best["state"] is not None
    model.load_state_dict(best["state"])
    test_rep, _ = eval_loader(model, test_loader, device)
    out_dir = OUT_ROOT / "runs" / "uformer_geometry_nodecal" / NODE_ID / f"{variant_id}_{config_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt_best.pt"
    trained_gate = trained_pc1_gate(model, mean, std)
    trained_stress_gate = trained_bad_stress_gate(model, mean, std)
    torch.save(
        {
            "model_state": model.state_dict(),
            "denoiser_state": model.backbone.denoiser.state_dict(),
            "config": cfg.__dict__,
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": mean,
            "feature_train_std": std,
            "feature_source": "manifest_aligned_target_geometry",
            "feature_alignment_summary": {"source": "node_train_split_direct", "train_rows": int(len(train_ds)), "val_rows": int(len(val_ds)), "test_rows": int(len(test_ds))},
            "branch_config": {
                "tab_hidden": 64,
                "head_hidden": 128,
                "config_name": config_name,
                "fusion_mode": str(cfg_info.get("fusion_mode", "waveform_tabular")),
                "pc1_bad_gate": gate_cfg,
                "bad_stress_gate": stress_gate_cfg,
                "calibration": "node_train_val",
            },
            "trained_pc1_bad_gate": trained_gate,
            "trained_bad_stress_gate": trained_stress_gate,
            "class_weight": cfg_info["class_weight"],
            "base_variant": variant_id,
            "best_epoch": best["epoch"],
            "load_report": load_report,
            "unfreeze_bottleneck": bool(unfreeze_bottleneck),
            "original_rows_used_for_training": False,
        },
        ckpt_path,
    )
    write_csv(pd.DataFrame(train_log), out_dir / "train_log.csv")
    summary = {
        "created_at": now_iso(),
        "config_name": f"nodecal_{config_name}",
        "variant_id": f"{variant_id}_uformergeom_{NODE_ID}_nodecal_{config_name}",
        "base_variant": variant_id,
        "checkpoint": str(ckpt_path),
        "best_epoch": best["epoch"],
        "val": best["val"],
        "synthetic_test": test_rep,
        "class_weight": cfg_info["class_weight"],
        "trained_pc1_bad_gate": trained_gate,
        "trained_bad_stress_gate": trained_stress_gate,
        "unfreeze_bottleneck": bool(unfreeze_bottleneck),
        "training_source": "N17043 node train split only",
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def load_trained_model(summary: dict[str, Any], device: torch.device) -> tuple[UformerGeometryBranch, dict[str, Any]]:
    ckpt = torch.load(summary["checkpoint"], map_location=device)
    cfg = Uformer1DConfig(**ckpt["config"])
    branch_cfg = ckpt.get("branch_config", {})
    model = UformerGeometryBranch(
        cfg,
        fusion_mode=str(branch_cfg.get("fusion_mode", "waveform_tabular")),
        pc1_bad_gate=branch_cfg.get("pc1_bad_gate"),
        bad_stress_gate=branch_cfg.get("bad_stress_gate"),
    ).to(device)
    model.attach_head(device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def compute_eval_features(signals: np.ndarray, state: dict[str, Any], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    primitives = compute_primitives(signals)
    features = assemble_features(primitives, state).to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return ((features - mean) / std).astype(np.float32)


@torch.no_grad()
def predict_signals(model: UformerGeometryBranch, signals: np.ndarray, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    probs = []
    for start in range(0, len(signals), batch_size):
        end = min(len(signals), start + batch_size)
        x = torch.from_numpy(signals[start:end].astype(np.float32)).to(device)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        tab = torch.from_numpy(features[start:end].astype(np.float32)).to(device)
        logits = model(x, tab)["logits"]
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def diagnostic_for_summary(summary: dict[str, Any], batch_size: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_trained_model(summary, device)
    variant_id = summary["base_variant"]
    data_dir = OUT_ROOT / "synthetic_variants" / variant_id / "datasets"
    mean = np.asarray(ckpt["feature_train_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_train_std"], dtype=np.float32)
    atlas = pd.read_csv(OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv")
    signals = np.load(SIGNALS_PATH)["X"].astype(np.float32)
    if str(ckpt.get("feature_source", "")) == "manifest_aligned_target_geometry":
        for col in FEATURE_COLUMNS:
            if col not in atlas.columns:
                atlas[col] = 0.0
        raw_features = atlas[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        eval_features = ((raw_features - mean) / std).astype(np.float32)
    else:
        with (data_dir / "tabular_feature_state.pkl").open("rb") as f:
            state = pickle.load(f)
        eval_features = compute_eval_features(signals[:, 0, :], state, mean, std)
    probs = predict_signals(model, signals, eval_features, batch_size, device)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    y_all = atlas["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    node_manifest = pd.read_csv(OUT_ROOT / "nodes" / NODE_ID / "node_boundary_manifest.csv")
    node_idx = node_manifest["idx"].astype(int).to_numpy()
    node_y = node_manifest["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    split = node_manifest["split"].astype(str).to_numpy()
    for col in FEATURE_COLUMNS:
        if col not in node_manifest.columns:
            node_manifest[col] = 0.0
    node_raw_features = node_manifest[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    node_features = ((node_raw_features - mean) / std).astype(np.float32)
    node_signals = signals[node_idx]
    node_probs = predict_signals(model, node_signals, node_features, batch_size, device)
    node_pred = np.argmax(node_probs, axis=1).astype(np.int64)
    node_all = metric_report(node_y, node_pred, node_probs)
    node_test_mask = split == "test"
    node_test = metric_report(node_y[node_test_mask], node_pred[node_test_mask], node_probs[node_test_mask])
    atlas_split = atlas["split"].astype(str).to_numpy() if "split" in atlas.columns else np.full(len(atlas), "all", dtype=object)
    atlas_region = atlas.get("original_region", pd.Series("", index=atlas.index)).astype(str).to_numpy()
    buckets = {
        "original_all_10s+": np.ones(len(atlas), dtype=bool),
        "original_test_all_10s+": atlas_split == "test",
        "original_test_main_without_bad_stress": (atlas_split == "test") & ~((y_all == 2) & (atlas_region == "outlier_low_confidence")),
        "original_bad_core_nearboundary": (atlas_split == "test") & (y_all == 2) & (atlas_region != "outlier_low_confidence"),
        "original_bad_outlier_stress": (atlas_split == "test") & (y_all == 2) & (atlas_region == "outlier_low_confidence"),
    }
    bucket_reports = {name: metric_report(y_all[mask], pred[mask], probs[mask]) for name, mask in buckets.items()}
    pred_id = summary["variant_id"]
    pred_path = OUT_ROOT / "predictions" / f"{pred_id}_probs.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, probs=probs)
    diag = {
        "created_at": now_iso(),
        "node_id": NODE_ID,
        "variant_id": pred_id,
        "checkpoint": summary["checkpoint"],
        "model_kind": "uformer_geometry_branch",
        "prediction_mode": "raw",
        "feature_schema": "47_nodefull_waveform_geometry_v1",
        "node_all": node_all,
        "node_test": node_test,
        "original_buckets": bucket_reports,
        "probs_path": str(pred_path),
        "eligible_useful_094": bool(node_test["acc"] >= 0.94 and node_test["good_recall"] >= 0.92 and node_test["medium_recall"] >= 0.90 and node_test["bad_recall"] >= 0.90),
        "eligible_promotion_095": bool(node_test["acc"] >= 0.95 and node_test["good_recall"] >= 0.92 and node_test["medium_recall"] >= 0.90 and node_test["bad_recall"] >= 0.90),
    }
    return diag


def build_features_stage(variant_id: str) -> dict[str, Any]:
    schema = build_sidecar(variant_id)
    write_json(ANALYSIS_OUT / "uformer_geometry_build_features_summary.json", schema)
    write_json(ANALYSIS_REPORT / "uformer_geometry_build_features_summary.json", schema)
    return schema


def selected_candidate_names(configs: str | None = None) -> list[str]:
    if not configs:
        return ["geom_balanced", "geom_medium_guard", "geom_bad_guard"]
    names = [name.strip() for name in configs.split(",") if name.strip()]
    missing = [name for name in names if name not in CANDIDATES]
    if missing:
        raise ValueError(f"Unknown config(s): {missing}")
    return names


def pc1_gate_config(cfg_info: dict[str, Any], mean: np.ndarray, std: np.ndarray) -> dict[str, float] | None:
    if not cfg_info.get("pc1_bad_gate"):
        return None
    pc1_i = FEATURE_COLUMNS.index("pc1")
    threshold_raw = float(cfg_info.get("pc1_bad_threshold", 7.82))
    strength = float(cfg_info.get("pc1_bad_strength", 6.0))
    temperature_raw = float(cfg_info.get("pc1_bad_temperature", 0.40))
    std_pc1 = float(std[pc1_i]) if float(std[pc1_i]) > 1e-6 else 1.0
    return {
        "threshold_raw": threshold_raw,
        "threshold_norm": float((threshold_raw - float(mean[pc1_i])) / std_pc1),
        "temperature_raw": temperature_raw,
        "temperature_norm": float(temperature_raw / std_pc1),
        "strength": strength,
    }


def trained_pc1_gate(model: UformerGeometryBranch, mean: np.ndarray, std: np.ndarray) -> dict[str, float] | None:
    if not model.pc1_bad_gate_enabled:
        return None
    assert model.pc1_bad_threshold_norm is not None
    assert model.pc1_bad_log_strength is not None
    assert model.pc1_bad_log_temperature is not None
    pc1_i = FEATURE_COLUMNS.index("pc1")
    std_pc1 = float(std[pc1_i]) if float(std[pc1_i]) > 1e-6 else 1.0
    threshold_norm = float(model.pc1_bad_threshold_norm.detach().cpu())
    temperature_norm = float((F.softplus(model.pc1_bad_log_temperature.detach().cpu()) + 1e-4).item())
    strength = float((F.softplus(model.pc1_bad_log_strength.detach().cpu()) + 1e-4).item())
    return {
        "threshold_norm": threshold_norm,
        "threshold_raw": float(threshold_norm * std_pc1 + float(mean[pc1_i])),
        "temperature_norm": temperature_norm,
        "temperature_raw": float(temperature_norm * std_pc1),
        "strength": strength,
    }


def norm_threshold(feature: str, raw_value: float, mean: np.ndarray, std: np.ndarray) -> float:
    i = FEATURE_COLUMNS.index(feature)
    scale = float(std[i]) if float(std[i]) > 1e-6 else 1.0
    return float((float(raw_value) - float(mean[i])) / scale)


def norm_temperature(feature: str, raw_value: float, std: np.ndarray) -> float:
    i = FEATURE_COLUMNS.index(feature)
    scale = float(std[i]) if float(std[i]) > 1e-6 else 1.0
    return float(max(float(raw_value), 1e-4) / scale)


def bad_stress_gate_config(cfg_info: dict[str, Any], mean: np.ndarray, std: np.ndarray) -> dict[str, float] | None:
    if not cfg_info.get("bad_stress_gate"):
        return None
    shared_temp = float(cfg_info.get("bad_stress_temperature", 0.18))
    pc1_temp = float(cfg_info.get("bad_stress_pc1_temperature", 1.25))
    baseline_raw = float(cfg_info.get("bad_stress_baseline_threshold", 0.95))
    flatline_raw = float(cfg_info.get("bad_stress_flatline_threshold", 0.30))
    qrs_raw = float(cfg_info.get("bad_stress_qrs_threshold", 0.10))
    pc1_raw = float(cfg_info.get("bad_stress_pc1_threshold", 1.50))
    return {
        "baseline_threshold_raw": baseline_raw,
        "baseline_threshold_norm": norm_threshold("baseline_step", baseline_raw, mean, std),
        "baseline_temperature_raw": shared_temp,
        "baseline_temperature_norm": norm_temperature("baseline_step", shared_temp, std),
        "flatline_threshold_raw": flatline_raw,
        "flatline_threshold_norm": norm_threshold("flatline_ratio", flatline_raw, mean, std),
        "flatline_temperature_raw": shared_temp,
        "flatline_temperature_norm": norm_temperature("flatline_ratio", shared_temp, std),
        "qrs_threshold_raw": qrs_raw,
        "qrs_threshold_norm": norm_threshold("qrs_visibility", qrs_raw, mean, std),
        "qrs_temperature_raw": shared_temp,
        "qrs_temperature_norm": norm_temperature("qrs_visibility", shared_temp, std),
        "pc1_threshold_raw": pc1_raw,
        "pc1_threshold_norm": norm_threshold("pc1", pc1_raw, mean, std),
        "pc1_temperature_raw": pc1_temp,
        "pc1_temperature_norm": norm_temperature("pc1", pc1_temp, std),
        "strength": float(cfg_info.get("bad_stress_strength", 7.0)),
    }


def trained_bad_stress_gate(model: UformerGeometryBranch, mean: np.ndarray, std: np.ndarray) -> dict[str, float] | None:
    if not model.bad_stress_gate_enabled:
        return None
    params = [
        model.stress_base_threshold_norm,
        model.stress_flat_threshold_norm,
        model.stress_qrs_threshold_norm,
        model.stress_pc1_threshold_norm,
        model.stress_log_strength,
        model.stress_base_log_temperature,
        model.stress_flat_log_temperature,
        model.stress_qrs_log_temperature,
        model.stress_pc1_log_temperature,
    ]
    if any(p is None for p in params):
        return None

    def raw_threshold(feature: str, value_norm: float) -> float:
        i = FEATURE_COLUMNS.index(feature)
        scale = float(std[i]) if float(std[i]) > 1e-6 else 1.0
        return float(value_norm * scale + float(mean[i]))

    def raw_temp(feature: str, value_norm: float) -> float:
        i = FEATURE_COLUMNS.index(feature)
        scale = float(std[i]) if float(std[i]) > 1e-6 else 1.0
        return float(value_norm * scale)

    base_norm = float(model.stress_base_threshold_norm.detach().cpu())
    flat_norm = float(model.stress_flat_threshold_norm.detach().cpu())
    qrs_norm = float(model.stress_qrs_threshold_norm.detach().cpu())
    pc1_norm = float(model.stress_pc1_threshold_norm.detach().cpu())
    base_temp = float((F.softplus(model.stress_base_log_temperature.detach().cpu()) + 1e-4).item())
    flat_temp = float((F.softplus(model.stress_flat_log_temperature.detach().cpu()) + 1e-4).item())
    qrs_temp = float((F.softplus(model.stress_qrs_log_temperature.detach().cpu()) + 1e-4).item())
    pc1_temp = float((F.softplus(model.stress_pc1_log_temperature.detach().cpu()) + 1e-4).item())
    return {
        "baseline_threshold_norm": base_norm,
        "baseline_threshold_raw": raw_threshold("baseline_step", base_norm),
        "baseline_temperature_norm": base_temp,
        "baseline_temperature_raw": raw_temp("baseline_step", base_temp),
        "flatline_threshold_norm": flat_norm,
        "flatline_threshold_raw": raw_threshold("flatline_ratio", flat_norm),
        "flatline_temperature_norm": flat_temp,
        "flatline_temperature_raw": raw_temp("flatline_ratio", flat_temp),
        "qrs_threshold_norm": qrs_norm,
        "qrs_threshold_raw": raw_threshold("qrs_visibility", qrs_norm),
        "qrs_temperature_norm": qrs_temp,
        "qrs_temperature_raw": raw_temp("qrs_visibility", qrs_temp),
        "pc1_threshold_norm": pc1_norm,
        "pc1_threshold_raw": raw_threshold("pc1", pc1_norm),
        "pc1_temperature_norm": pc1_temp,
        "pc1_temperature_raw": raw_temp("pc1", pc1_temp),
        "strength": float((F.softplus(model.stress_log_strength.detach().cpu()) + 1e-4).item()),
    }


def quick_stage(variant_id: str, epochs: int, batch_size: int, configs: str | None = None) -> dict[str, Any]:
    if not (OUT_ROOT / "synthetic_variants" / variant_id / "datasets" / "tabular_features.npz").exists():
        build_sidecar(variant_id)
    rows = []
    for name in selected_candidate_names(configs):
        rows.append(train_one(name, variant_id, epochs, batch_size, unfreeze_bottleneck=False))
    payload = {"created_at": now_iso(), "variant_id": variant_id, "rows": rows}
    write_json(ANALYSIS_OUT / "uformer_geometry_quick_summary.json", payload)
    write_json(ANALYSIS_REPORT / "uformer_geometry_quick_summary.json", payload)
    return payload


def nodecal_quick_stage(variant_id: str, epochs: int, batch_size: int, configs: str | None = None) -> dict[str, Any]:
    rows = []
    for name in selected_candidate_names(configs):
        rows.append(train_one_nodecal(name, variant_id, epochs, batch_size, unfreeze_bottleneck=False))
    payload = {"created_at": now_iso(), "node_id": NODE_ID, "variant_id": variant_id, "rows": rows, "training_source": f"{NODE_ID} node train split only"}
    write_json(ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_quick_summary.json", payload)
    write_json(ANALYSIS_REPORT / f"uformer_geometry_nodecal_{NODE_ID}_quick_summary.json", payload)
    return payload


def diagnostic_stage(variant_id: str, batch_size: int) -> dict[str, Any]:
    quick_path = ANALYSIS_OUT / "uformer_geometry_quick_summary.json"
    if not quick_path.exists():
        raise FileNotFoundError(quick_path)
    quick = json.loads(quick_path.read_text(encoding="utf-8"))
    diagnostics = []
    for row in quick["rows"]:
        diagnostics.append(diagnostic_for_summary(row, batch_size))
    flat_rows = []
    for d in diagnostics:
        nt = d["node_test"]
        na = d["node_all"]
        orig = d["original_buckets"]["original_test_all_10s+"]
        main = d["original_buckets"]["original_test_main_without_bad_stress"]
        flat_rows.append(
            {
                "created_at": d["created_at"],
                "node_id": d["node_id"],
                "variant_id": d["variant_id"],
                "model_kind": d["model_kind"],
                "prediction_mode": d["prediction_mode"],
                "node_test_acc": nt["acc"],
                "node_test_macro_f1": nt["macro_f1"],
                "node_test_good_recall": nt["good_recall"],
                "node_test_medium_recall": nt["medium_recall"],
                "node_test_bad_recall": nt["bad_recall"],
                "node_all_acc": na["acc"],
                "node_all_good_recall": na["good_recall"],
                "node_all_medium_recall": na["medium_recall"],
                "node_all_bad_recall": na["bad_recall"],
                "original_test_all_10s_plus_acc": orig["acc"],
                "original_test_all_10s_plus_good_recall": orig["good_recall"],
                "original_test_all_10s_plus_medium_recall": orig["medium_recall"],
                "original_test_all_10s_plus_bad_recall": orig["bad_recall"],
                "original_test_main_without_bad_stress_acc": main["acc"],
                "original_test_main_without_bad_stress_good_recall": main["good_recall"],
                "original_test_main_without_bad_stress_medium_recall": main["medium_recall"],
                "original_test_main_without_bad_stress_bad_recall": main["bad_recall"],
                "eligible_useful_094": d["eligible_useful_094"],
                "eligible_promotion_095": d["eligible_promotion_095"],
                "probs_path": d["probs_path"],
            }
        )
    df = pd.DataFrame(flat_rows).sort_values("node_test_acc", ascending=False)
    out_csv = ANALYSIS_OUT / "uformer_geometry_branch_diagnostic_metrics.csv"
    rep_csv = ANALYSIS_REPORT / out_csv.name
    write_csv(df, out_csv)
    write_csv(df, rep_csv)
    payload = {"created_at": now_iso(), "diagnostics": diagnostics, "metrics_csv": str(out_csv)}
    write_json(ANALYSIS_OUT / "uformer_geometry_diagnostic_promote_summary.json", payload)
    write_json(ANALYSIS_REPORT / "uformer_geometry_diagnostic_promote_summary.json", payload)
    write_report(df, diagnostics)
    return payload


def nodecal_diagnostic_stage(variant_id: str, batch_size: int) -> dict[str, Any]:
    quick_path = ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_quick_summary.json"
    if not quick_path.exists():
        raise FileNotFoundError(quick_path)
    quick = json.loads(quick_path.read_text(encoding="utf-8"))
    diagnostics = []
    for row in quick["rows"]:
        diagnostics.append(diagnostic_for_summary(row, batch_size))
    flat_rows = []
    for d in diagnostics:
        nt = d["node_test"]
        na = d["node_all"]
        orig = d["original_buckets"]["original_test_all_10s+"]
        main = d["original_buckets"]["original_test_main_without_bad_stress"]
        flat_rows.append(
            {
                "created_at": d["created_at"],
                "node_id": d["node_id"],
                "variant_id": d["variant_id"],
                "model_kind": d["model_kind"],
                "prediction_mode": d["prediction_mode"],
                "node_test_acc": nt["acc"],
                "node_test_macro_f1": nt["macro_f1"],
                "node_test_good_recall": nt["good_recall"],
                "node_test_medium_recall": nt["medium_recall"],
                "node_test_bad_recall": nt["bad_recall"],
                "node_all_acc": na["acc"],
                "node_all_good_recall": na["good_recall"],
                "node_all_medium_recall": na["medium_recall"],
                "node_all_bad_recall": na["bad_recall"],
                "original_test_all_10s_plus_acc": orig["acc"],
                "original_test_all_10s_plus_good_recall": orig["good_recall"],
                "original_test_all_10s_plus_medium_recall": orig["medium_recall"],
                "original_test_all_10s_plus_bad_recall": orig["bad_recall"],
                "original_test_main_without_bad_stress_acc": main["acc"],
                "original_test_main_without_bad_stress_good_recall": main["good_recall"],
                "original_test_main_without_bad_stress_medium_recall": main["medium_recall"],
                "original_test_main_without_bad_stress_bad_recall": main["bad_recall"],
                "eligible_useful_094": d["eligible_useful_094"],
                "eligible_promotion_095": d["eligible_promotion_095"],
                "probs_path": d["probs_path"],
            }
        )
    df = pd.DataFrame(flat_rows).sort_values("node_test_acc", ascending=False)
    out_csv = ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_diagnostic_metrics.csv"
    rep_csv = ANALYSIS_REPORT / out_csv.name
    write_csv(df, out_csv)
    write_csv(df, rep_csv)
    payload = {"created_at": now_iso(), "diagnostics": diagnostics, "metrics_csv": str(out_csv), "training_source": quick.get("training_source")}
    write_json(ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_diagnostic_summary.json", payload)
    write_json(ANALYSIS_REPORT / f"uformer_geometry_nodecal_{NODE_ID}_diagnostic_summary.json", payload)
    write_report(df, diagnostics)
    return payload


def write_report(df: pd.DataFrame, diagnostics: list[dict[str, Any]]) -> None:
    best = df.iloc[0].to_dict() if not df.empty else {}
    lines = [
        "# UFormer + SQI/Geometry Branch Experiment",
        "",
        "Experiment-only normal checkpoint path. Mainline UFormer code/checkpoints are not overwritten.",
        "",
        "## Best Node-Test Result",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Variant: `{best['variant_id']}`",
                f"- Node-test acc: `{float(best['node_test_acc']):.6f}`",
                f"- Node-test good/medium/bad recall: `{float(best['node_test_good_recall']):.6f}` / `{float(best['node_test_medium_recall']):.6f}` / `{float(best['node_test_bad_recall']):.6f}`",
                f"- Useful >=0.94 gate: `{bool(best['eligible_useful_094'])}`",
                f"- Promotion >=0.95 gate: `{bool(best['eligible_promotion_095'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Metrics",
            "",
            markdown_table(df),
            "",
            "## Notes",
            "",
            "- Feature normalization is fit on the declared train split only.",
            "- Original BUT is bucketed report-only and never used for selection.",
            "- The branch uses 47 SQI/geometry columns and saves schema/stats inside the checkpoint.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (ANALYSIS_REPORT / "uformer_geometry_branch_report.md").write_text(text, encoding="utf-8")
    (ANALYSIS_OUT / "uformer_geometry_branch_report.md").write_text(text, encoding="utf-8")


def original_report_stage() -> dict[str, Any]:
    path = ANALYSIS_OUT / "uformer_geometry_diagnostic_promote_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for d in payload["diagnostics"]:
        for bucket, rep in d["original_buckets"].items():
            rows.append(
                {
                    "variant_id": d["variant_id"],
                    "bucket": bucket,
                    "n": rep["n"],
                    "acc": rep["acc"],
                    "macro_f1": rep["macro_f1"],
                    "good_recall": rep["good_recall"],
                    "medium_recall": rep["medium_recall"],
                    "bad_recall": rep["bad_recall"],
                }
            )
    df = pd.DataFrame(rows)
    out = ANALYSIS_OUT / "uformer_geometry_original_bucketed_metrics.csv"
    rep = ANALYSIS_REPORT / out.name
    write_csv(df, out)
    write_csv(df, rep)
    metrics_path = ANALYSIS_OUT / "uformer_geometry_branch_diagnostic_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).sort_values("node_test_acc", ascending=False)
        write_report(metrics, payload["diagnostics"])
    return {"created_at": now_iso(), "metrics_csv": str(out), "rows": rows}


def bad_threshold_calibration_stage(configs: str | None = None) -> dict[str, Any]:
    diag_path = ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_diagnostic_summary.json"
    if not diag_path.exists():
        raise FileNotFoundError(diag_path)
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    requested = set(selected_candidate_names(configs)) if configs else None
    atlas = pd.read_csv(OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv")
    y = atlas["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    split = atlas["split"].astype(str).to_numpy()
    region = atlas.get("original_region", pd.Series("", index=atlas.index)).astype(str).to_numpy()
    trainval = (split == "train") | (split == "val")
    buckets = {
        "trainval_selection": trainval,
        "node_test_original_test_all_10s+": split == "test",
        "original_test_main_without_bad_stress": (split == "test") & ~((y == CLASS_TO_INT["bad"]) & (region == "outlier_low_confidence")),
        "bad_core_nearboundary": (split == "test") & (y == CLASS_TO_INT["bad"]) & (region != "outlier_low_confidence"),
        "bad_outlier_stress": (split == "test") & (y == CLASS_TO_INT["bad"]) & (region == "outlier_low_confidence"),
        "node_all_original_all_10s+": np.ones(len(y), dtype=bool),
    }

    def report(probs: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, Any]:
        pred = np.argmax(probs, axis=1).astype(np.int64)
        pred[probs[:, CLASS_TO_INT["bad"]] >= threshold] = CLASS_TO_INT["bad"]
        return metric_report(y[mask], pred[mask], probs[mask])

    summaries: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    for diag in payload["diagnostics"]:
        config_name = str(diag["variant_id"]).split("_nodecal_")[-1]
        if requested is not None and config_name not in requested:
            continue
        probs = np.load(diag["probs_path"])["probs"].astype(np.float32)
        local_sweep: list[dict[str, Any]] = []
        for threshold in np.round(np.linspace(0.01, 0.95, 95), 2):
            rep = report(probs, trainval, float(threshold))
            row = {
                "variant_id": diag["variant_id"],
                "config_name": config_name,
                "threshold": float(threshold),
                "acc": rep["acc"],
                "macro_f1": rep["macro_f1"],
                "good_recall": rep["good_recall"],
                "medium_recall": rep["medium_recall"],
                "bad_recall": rep["bad_recall"],
            }
            local_sweep.append(row)
            sweep_rows.append(row)
        sweep_df = pd.DataFrame(local_sweep)
        valid = sweep_df[
            (sweep_df["acc"] >= 0.95)
            & (sweep_df["good_recall"] >= 0.92)
            & (sweep_df["medium_recall"] >= 0.90)
            & (sweep_df["bad_recall"] >= 0.90)
        ]
        if valid.empty:
            chosen_row = sweep_df.sort_values(["acc", "bad_recall"], ascending=[False, False]).iloc[0]
        else:
            max_acc = valid["acc"].max()
            chosen_row = valid[valid["acc"] == max_acc].sort_values("threshold").iloc[0]
        threshold = float(chosen_row["threshold"])
        for bucket, mask in buckets.items():
            rep = report(probs, mask, threshold)
            summaries.append(
                {
                    "variant_id": diag["variant_id"],
                    "config_name": config_name,
                    "bucket": bucket,
                    "threshold": threshold,
                    "n": rep["n"],
                    "acc": rep["acc"],
                    "macro_f1": rep["macro_f1"],
                    "good_recall": rep["good_recall"],
                    "medium_recall": rep["medium_recall"],
                    "bad_recall": rep["bad_recall"],
                    "good_precision": rep["good_precision"],
                    "medium_precision": rep["medium_precision"],
                    "bad_precision": rep["bad_precision"],
                    "confusion_3x3": rep["confusion_3x3"],
                }
            )
    summary_df = pd.DataFrame([{k: v for k, v in row.items() if k != "confusion_3x3"} for row in summaries])
    sweep_df = pd.DataFrame(sweep_rows)
    metrics_csv = ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_bad_threshold_calibration_metrics.csv"
    sweep_csv = ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_bad_threshold_trainval_sweep.csv"
    write_csv(summary_df, metrics_csv)
    write_csv(summary_df, ANALYSIS_REPORT / metrics_csv.name)
    write_csv(sweep_df, sweep_csv)
    write_csv(sweep_df, ANALYSIS_REPORT / sweep_csv.name)
    lines = [
        "# UFormer Geometry Bad-Threshold Calibration",
        "",
        "Selection uses node train+val only. Test/original buckets are report-only diagnostics.",
        "",
        "## Metrics",
        "",
        markdown_table(summary_df),
        "",
        "## Notes",
        "",
        "- A single bad posterior threshold is selected from train+val probabilities.",
        "- No original test bucket is used to choose the threshold.",
        "- The underlying checkpoint remains a normal UFormer geometry-branch checkpoint.",
    ]
    report_path = ANALYSIS_REPORT / f"uformer_geometry_nodecal_{NODE_ID}_bad_threshold_calibration_report.md"
    out_report_path = ANALYSIS_OUT / report_path.name
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "created_at": now_iso(),
        "node_id": NODE_ID,
        "metrics_csv": str(metrics_csv),
        "sweep_csv": str(sweep_csv),
        "report": str(report_path),
        "rows": summaries,
    }
    write_json(ANALYSIS_OUT / f"uformer_geometry_nodecal_{NODE_ID}_bad_threshold_calibration_summary.json", result)
    write_json(ANALYSIS_REPORT / f"uformer_geometry_nodecal_{NODE_ID}_bad_threshold_calibration_summary.json", result)
    return result


def main() -> None:
    global NODE_ID
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "uformer_geometry_build_features",
            "uformer_geometry_quick",
            "uformer_geometry_diagnostic_promote",
            "uformer_geometry_original_report",
            "uformer_geometry_all",
            "uformer_geometry_nodecal_quick",
            "uformer_geometry_nodecal_diagnostic",
            "uformer_geometry_bad_threshold_calibration",
        ),
        required=True,
    )
    parser.add_argument("--variant-id", default=BASE_VARIANT)
    parser.add_argument("--node-id", default=NODE_ID)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    NODE_ID = str(args.node_id)
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORT.mkdir(parents=True, exist_ok=True)
    if args.stage == "uformer_geometry_build_features":
        payload = build_features_stage(args.variant_id)
    elif args.stage == "uformer_geometry_quick":
        payload = quick_stage(args.variant_id, args.epochs, args.batch_size, configs=args.configs)
    elif args.stage == "uformer_geometry_diagnostic_promote":
        payload = diagnostic_stage(args.variant_id, args.batch_size)
    elif args.stage == "uformer_geometry_original_report":
        payload = original_report_stage()
    elif args.stage == "uformer_geometry_nodecal_quick":
        payload = nodecal_quick_stage(args.variant_id, args.epochs, args.batch_size, configs=args.configs)
    elif args.stage == "uformer_geometry_nodecal_diagnostic":
        payload = nodecal_diagnostic_stage(args.variant_id, args.batch_size)
    elif args.stage == "uformer_geometry_bad_threshold_calibration":
        payload = bad_threshold_calibration_stage(configs=args.configs)
    elif args.stage == "uformer_geometry_all":
        build = build_features_stage(args.variant_id)
        quick = quick_stage(args.variant_id, args.epochs, args.batch_size)
        diagnostic = diagnostic_stage(args.variant_id, args.batch_size)
        original = original_report_stage()
        payload = {"build": build, "quick": quick, "diagnostic": diagnostic, "original": original}
    else:
        raise ValueError(args.stage)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
