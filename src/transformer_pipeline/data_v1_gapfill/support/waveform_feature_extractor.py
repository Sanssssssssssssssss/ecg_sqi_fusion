from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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


def moving_average(z: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, z)


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
    centered = z - z.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, axis=1)
    power = fft.real * fft.real + fft.imag * fft.imag
    power[:, 0] = 0.0
    p = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-8)
    return -(p * np.log(p + 1e-12)).sum(axis=1) / math.log(max(2, p.shape[1]))


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
    wave = np.stack([band_0_1, band_1_5, band_5_15, band_15_30, band_30_45], axis=1)
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


def metric_report(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, Any]:
    labels = [0, 1, 2]
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    out = {
        "acc": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "good_recall": float(recalls[0]),
        "medium_recall": float(recalls[1]),
        "bad_recall": float(recalls[2]),
        "good_to_medium": float(cm[0, 1] / max(1, cm[0].sum())),
        "medium_to_good": float(cm[1, 0] / max(1, cm[1].sum())),
        "bad_to_medium": float(cm[2, 1] / max(1, cm[2].sum())),
        "confusion_3x3": cm.astype(int).tolist(),
    }
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float32)
        out["bad_fpr_nonbad"] = float(np.mean(probs[y_true != 2, 2] >= 0.5)) if np.any(y_true != 2) else 0.0
    else:
        out["bad_fpr_nonbad"] = 0.0
    return out
