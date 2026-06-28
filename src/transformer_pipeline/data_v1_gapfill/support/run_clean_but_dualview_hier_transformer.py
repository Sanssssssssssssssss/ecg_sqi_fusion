"""Fixed-10s clean BUT dual-view hierarchical Transformer diagnostic.

This experiment intentionally does not use variable-length inputs.  It trains
on materialized clean fixed-10s BUT protocol bundles.  The default report is the
curated clean protocol only; legacy full BUT stress buckets are emitted only
when explicitly requested.  The model input is waveform-derived channels only;
interpretable SQI/QRS/baseline/detail features are auxiliary targets and are not
passed to inference.
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_ROOT = ANALYSIS_DIR / "clean_but_protocols"
FULL_SIGNALS_PATH = ROOT / "outputs" / "external_benchmarks" / "e311_but_protocol_adaptation_2026_06_03" / "protocols" / "p1_current_10s_center" / "signals.npz"
FULL_ATLAS_PATH = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
TX_SCRIPT = SCRIPT_DIR / "run_waveform_transformer_search.py"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

EXCLUDED_FORMAL_AUX_PREFIXES = ("pc",)
EXCLUDED_FORMAL_AUX = {
    "pca_margin",
    "boundary_confidence",
    "region_confidence",
    "knn_label_purity",
}
KEY_RECOVERY_FEATURES = [
    "sqi_basSQI",
    "qrs_band_ratio",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "non_qrs_diff_p95",
    "amplitude_entropy",
]

CANDIDATES: dict[str, dict[str, Any]] = {
    "dualview_convtx_hier": {
        "arch": "convtx",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "lr": 3e-4,
        "class_weight": [1.05, 1.30, 2.20],
        "aux_weight": 0.35,
        "core_aux_weight": 0.30,
        "gm_weight": 1.50,
        "bad_weight": 1.80,
        "ordinal_weight": 0.55,
        "false_bad_penalty": 0.40,
        "seed": 20260691,
    },
    "dualview_conformer_hier": {
        "arch": "conformer",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "lr": 2.5e-4,
        "class_weight": [1.05, 1.30, 2.30],
        "aux_weight": 0.35,
        "core_aux_weight": 0.30,
        "gm_weight": 1.50,
        "bad_weight": 1.90,
        "ordinal_weight": 0.55,
        "false_bad_penalty": 0.45,
        "seed": 20260692,
    },
    "dualview_multipatch_hier": {
        "arch": "multipatch",
        "width": 88,
        "layers": 4,
        "heads": 4,
        "lr": 3e-4,
        "class_weight": [1.05, 1.30, 2.20],
        "aux_weight": 0.35,
        "core_aux_weight": 0.30,
        "gm_weight": 1.60,
        "bad_weight": 1.75,
        "ordinal_weight": 0.55,
        "false_bad_penalty": 0.40,
        "seed": 20260693,
    },
}


def load_tx_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_transformer_search_clean_dualview", TX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TX_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TX = load_tx_module()
GEOM = TX.GEOM
ALL_FEATURE_COLUMNS = list(TX.FEATURE_COLUMNS)
FORMAL_FEATURE_COLUMNS = [
    c
    for c in ALL_FEATURE_COLUMNS
    if c not in EXCLUDED_FORMAL_AUX and not any(c == p or c.startswith(p) and c[2:3].isdigit() for p in EXCLUDED_FORMAL_AUX_PREFIXES)
]
FORMAL_FEATURE_IDX = [ALL_FEATURE_COLUMNS.index(c) for c in FORMAL_FEATURE_COLUMNS]
CORE_AUX_COLUMNS = [
    c
    for c in [
        "qrs_visibility",
        "qrs_band_ratio",
        "qrs_prom_p90",
        "template_corr",
        "detector_agreement",
        "baseline_step",
        "flatline_ratio",
        "contact_loss_win_ratio",
        "non_qrs_rms_ratio",
        "non_qrs_diff_p95",
        "diff_abs_p95",
        "band_15_30",
        "band_30_45",
        "amplitude_entropy",
        "low_amp_ratio",
        "sqi_iSQI",
        "sqi_bSQI",
        "sqi_pSQI",
        "sqi_sSQI",
        "sqi_kSQI",
        "sqi_fSQI",
        "sqi_basSQI",
    ]
    if c in FORMAL_FEATURE_COLUMNS
]
CORE_AUX_IDX = [FORMAL_FEATURE_COLUMNS.index(c) for c in CORE_AUX_COLUMNS]


@dataclass
class FeatureNorm:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ChannelStats:
    global_mean: float
    global_std: float


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349
    std = np.std(x, axis=1, keepdims=True)
    scale = np.where(scale > 1e-5, scale, std)
    scale = np.where(scale > 1e-5, scale, 1.0)
    return ((x - med) / scale).astype(np.float32)


def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    kernel = np.ones(int(win), dtype=np.float32) / float(win)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, x).astype(np.float32)


def make_dualview_channels(x: np.ndarray, stats: ChannelStats) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    physical = ((x - float(stats.global_mean)) / max(float(stats.global_std), 1e-5)).astype(np.float32)
    physical = np.clip(physical, -8.0, 8.0)
    z = robust_z(x)
    dz = np.diff(z, axis=1, prepend=z[:, :1]).astype(np.float32)
    baseline_long = rolling_mean(z, 251)
    highpass = (z - baseline_long).astype(np.float32)
    detail_fast = rolling_mean(np.abs(highpass), 31)
    detail_slow = rolling_mean(np.abs(highpass), 125)
    physical_trend = rolling_mean(physical, 251)
    return np.stack(
        [
            physical,
            z,
            dz,
            baseline_long,
            highpass,
            detail_fast,
            detail_slow,
            physical_trend,
        ],
        axis=1,
    ).astype(np.float32)


def compute_channel_stats(signals: np.ndarray) -> ChannelStats:
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    return ChannelStats(global_mean=float(np.mean(x)), global_std=float(np.std(x) if np.std(x) > 1e-5 else 1.0))


class CleanButDataset(Dataset):
    def __init__(
        self,
        protocol_dir: Path,
        split: str,
        channel_stats: ChannelStats | None = None,
        feature_norm: FeatureNorm | None = None,
    ) -> None:
        atlas = pd.read_csv(protocol_dir / "original_region_atlas.csv")
        signals = np.load(protocol_dir / "signals.npz")["X"].astype(np.float32)
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        for col in FORMAL_FEATURE_COLUMNS:
            if col not in atlas.columns:
                atlas[col] = 0.0
        mask = atlas["split"].astype(str).eq(split).to_numpy()
        self.frame = atlas.loc[mask].reset_index(drop=True)
        rows = self.frame["idx"].astype(int).to_numpy()
        if channel_stats is None:
            train_rows = atlas.loc[atlas["split"].astype(str).eq("train"), "idx"].astype(int).to_numpy()
            channel_stats = compute_channel_stats(signals[train_rows])
        if feature_norm is None:
            train_mask = atlas["split"].astype(str).eq("train")
            train_raw = (
                atlas.loc[train_mask, FORMAL_FEATURE_COLUMNS]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            mean = train_raw.mean(axis=0).astype(np.float32)
            std = train_raw.std(axis=0).astype(np.float32)
            std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
            feature_norm = FeatureNorm(mean=mean, std=std)
        self.channel_stats = channel_stats
        self.feature_norm = feature_norm
        self.raw_signals = signals[rows].copy().astype(np.float32)
        self.x = make_dualview_channels(signals[rows], channel_stats)
        raw = (
            self.frame[FORMAL_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        self.aux = ((raw - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.aux_raw = raw.astype(np.float32)
        self.y = self.frame["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.split = self.frame["split"].astype(str).to_numpy()
        self.region = self.frame.get("original_region", pd.Series("", index=self.frame.index)).astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


def augment_clean_bad_raw(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Source-only bad augmentation for clean BUT bad windows.

    This is intentionally physiology/SQI-shaped rather than arbitrary noise:
    baseline steps, contact/dropout shelves, low-amplitude stretches, and sparse
    reset edges.  It is used only on clean BUT train bad rows to test whether the
    BUT->PTB reverse gap is missing severe bad morphology coverage.
    """
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    s = float(strength)

    if rng.random() < min(0.95, 0.45 + 0.18 * s):
        drift = float(rng.uniform(0.20, 0.85) * s)
        y += drift * (0.55 * t + 0.45 * np.sin(np.pi * t + float(rng.uniform(-1.5, 1.5))))

    # PTB synthetic bad has much lower raw zero-crossing/diff statistics than
    # clean-BUT bad while keeping a strong 15-30Hz band shell.  Smooth first,
    # then add a narrow-band artifact so the morphology is not just flat drift.
    if rng.random() < min(0.95, 0.50 + 0.16 * s):
        k = int(rng.integers(5, 19))
        kernel = np.ones(k, dtype=np.float32) / float(k)
        smooth = np.convolve(y, kernel, mode="same").astype(np.float32)
        y = (0.25 * y + 0.75 * smooth).astype(np.float32)

    if rng.random() < min(0.90, 0.40 + 0.16 * s):
        fs = 125.0
        tt = np.arange(n, dtype=np.float32) / fs
        freq = float(rng.uniform(17.0, 28.0))
        amp = float(rng.uniform(0.025, 0.10) * s)
        phase = float(rng.uniform(-np.pi, np.pi))
        env = 0.45 + 0.55 * np.sin(np.pi * (tt / max(tt[-1], 1e-6))) ** 2
        y += (amp * env * np.sin(2.0 * np.pi * freq * tt + phase)).astype(np.float32)

    if rng.random() < min(0.92, 0.38 + 0.18 * s):
        start = int(rng.integers(0, max(1, n - 160)))
        width = int(rng.integers(80, min(n - start, 430)))
        shelf = float(rng.uniform(-0.75, 0.75) * s)
        y[start : start + width] = 0.22 * y[start : start + width] + shelf

    if rng.random() < min(0.80, 0.30 + 0.14 * s):
        start = int(rng.integers(0, max(1, n - 120)))
        width = int(rng.integers(50, min(n - start, 300)))
        y[start : start + width] *= float(rng.uniform(0.02, 0.22))

    if rng.random() < min(0.75, 0.22 + 0.16 * s):
        for _ in range(int(rng.integers(1, 4))):
            pos = int(rng.integers(4, n - 4))
            edge = float(rng.uniform(-1.0, 1.0) * s)
            y[pos:] += edge

    y += rng.normal(0.0, 0.015 * s, size=n).astype(np.float32)
    return y.astype(np.float32)


def augment_clean_good_highamp_flat_raw(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Good-only source augmentation for high-amplitude/flat PTB-like good.

    Pure-cross error analysis showed the clean-BUT->PTB good misses are often
    PTB good rows with high raw RMS/ptp and elevated flatline ratio, while QRS
    remains visible and non-QRS detail is low.  This keeps the row label good:
    it changes lead scale/polarity and smooth low-derivative shelves without
    adding medium-like broadband or local derivative noise.
    """
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    s = float(strength)
    if rng.random() < 0.35:
        y = -y
    y *= float(rng.uniform(1.08, 1.34) * np.clip(0.85 + 0.18 * s, 0.90, 1.25))

    k = int(rng.integers(41, 101))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(y, kernel, mode="same").astype(np.float32)
    if rng.random() < min(0.90, 0.40 + 0.22 * s):
        mix = float(rng.uniform(0.08, 0.22) * np.clip(s, 0.6, 1.6))
        y = (1.0 - mix) * y + mix * smooth
    if rng.random() < min(0.80, 0.35 + 0.15 * s):
        width = int(rng.integers(max(40, n // 90), max(90, n // 18)))
        start = int(rng.integers(0, max(1, n - width)))
        shelf = np.linspace(float(y[start]), float(y[min(n - 1, start + width - 1)]), width, dtype=np.float32)
        y[start : start + width] = 0.82 * y[start : start + width] + 0.18 * shelf
    if rng.random() < 0.45:
        # Very mild baseline wander; no extra high-frequency noise for good.
        tt = np.arange(n, dtype=np.float32) / 125.0
        y += (float(rng.uniform(0.006, 0.020) * s) * np.sin(2.0 * np.pi * float(rng.uniform(0.08, 0.22)) * tt + float(rng.uniform(-np.pi, np.pi)))).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_clean_gm_ptb_like_raw(x: np.ndarray, rng: np.random.Generator, strength: float, label: str, mode: str = "attenuate") -> np.ndarray:
    """Source-only good/medium augmentation toward PTB-like raw morphology.

    Raw-feature audits showed clean BUT good/medium carry stronger DC/very-low
    frequency and amplitude shells than PTB synthetic good/medium.  This
    augmentation narrows that raw-domain gap without injecting target rows: it
    attenuates slow baseline and amplitude, then adds label-appropriate mild
    detail so good remains readable and medium remains degraded.
    """
    if label == "good" and str(mode) == "highamp_flat_qrsvisible":
        return augment_clean_good_highamp_flat_raw(x, rng, strength)

    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    s = float(strength)
    k = int(rng.integers(61, 151))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    slow = np.convolve(y, kernel, mode="same").astype(np.float32)

    slow_keep = float(np.clip(0.55 - 0.16 * s, 0.20, 0.62))
    amp_keep = float(np.clip(0.92 - 0.10 * s, 0.66, 0.95))
    y = amp_keep * (y - slow) + slow_keep * slow
    y -= float(np.mean(y)) * float(np.clip(0.70 + 0.10 * s, 0.60, 0.92))

    if label == "good":
        noise_sigma = 0.004 + 0.004 * s
        detail_amp = 0.003 + 0.003 * s
    else:
        noise_sigma = 0.010 + 0.009 * s
        detail_amp = 0.010 + 0.008 * s
    fs = 125.0
    tt = np.arange(n, dtype=np.float32) / fs
    if rng.random() < min(0.90, 0.45 + 0.18 * s):
        freq = float(rng.uniform(6.0, 14.0) if label == "good" else rng.uniform(8.0, 22.0))
        phase = float(rng.uniform(-np.pi, np.pi))
        env = 0.35 + 0.65 * np.sin(np.pi * (tt / max(tt[-1], 1e-6))) ** 2
        y += (detail_amp * env * np.sin(2.0 * np.pi * freq * tt + phase)).astype(np.float32)
    y += rng.normal(0.0, noise_sigma, size=n).astype(np.float32)
    return y.astype(np.float32)


class AugmentedCleanButDataset(Dataset):
    def __init__(
        self,
        base: CleanButDataset,
        bad_copies: int,
        bad_strength: float,
        seed: int,
        good_copies: int = 0,
        good_strength: float = 1.0,
        good_mode: str = "attenuate",
        medium_copies: int = 0,
        medium_strength: float = 1.0,
    ) -> None:
        self.channel_stats = base.channel_stats
        self.feature_norm = base.feature_norm
        self.frame = base.frame
        self.split = base.split
        self.region = base.region
        self.aux_raw = base.aux_raw
        x_parts = [base.x]
        aux_parts = [base.aux]
        y_parts = [base.y]
        added_by_class = {"good": 0, "medium": 0, "bad": 0}
        rng = np.random.default_rng(int(seed))
        for label, copies, strength in [
            ("good", int(good_copies), float(good_strength)),
            ("medium", int(medium_copies), float(medium_strength)),
        ]:
            idx = np.flatnonzero(base.y == CLASS_TO_INT[label])
            if copies <= 0 or idx.size == 0:
                continue
            aug_raw: list[np.ndarray] = []
            aug_aux: list[np.ndarray] = []
            aug_y: list[int] = []
            for i in idx:
                for _ in range(copies):
                    local_strength = float(strength) * float(rng.uniform(0.75, 1.25))
                    aug_raw.append(augment_clean_gm_ptb_like_raw(base.raw_signals[int(i)], rng, local_strength, label, good_mode if label == "good" else "attenuate"))
                    aug_aux.append(base.aux[int(i)])
                    aug_y.append(CLASS_TO_INT[label])
            if aug_raw:
                x_parts.append(make_dualview_channels(np.stack(aug_raw).astype(np.float32), base.channel_stats))
                aux_parts.append(np.stack(aug_aux).astype(np.float32))
                y_parts.append(np.asarray(aug_y, dtype=np.int64))
                added_by_class[label] = len(aug_y)

        bad_idx = np.flatnonzero(base.y == CLASS_TO_INT["bad"])
        if int(bad_copies) > 0 and bad_idx.size:
            aug_raw: list[np.ndarray] = []
            aug_aux: list[np.ndarray] = []
            aug_y: list[int] = []
            for i in bad_idx:
                for _ in range(int(bad_copies)):
                    strength = float(bad_strength) * float(rng.uniform(0.75, 1.30))
                    aug_raw.append(augment_clean_bad_raw(base.raw_signals[int(i)], rng, strength))
                    aug_aux.append(base.aux[int(i)])
                    aug_y.append(CLASS_TO_INT["bad"])
            if aug_raw:
                x_parts.append(make_dualview_channels(np.stack(aug_raw).astype(np.float32), base.channel_stats))
                aux_parts.append(np.stack(aug_aux).astype(np.float32))
                y_parts.append(np.asarray(aug_y, dtype=np.int64))
                added_by_class["bad"] = len(aug_y)
        self.x = np.concatenate(x_parts, axis=0).astype(np.float32)
        self.aux = np.concatenate(aux_parts, axis=0).astype(np.float32)
        self.y = np.concatenate(y_parts, axis=0).astype(np.int64)
        self.augmentation_summary = {
            "clean_good_aug_copies": int(good_copies),
            "clean_good_aug_strength": float(good_strength),
            "clean_good_aug_mode": str(good_mode),
            "clean_medium_aug_copies": int(medium_copies),
            "clean_medium_aug_strength": float(medium_strength),
            "clean_bad_aug_copies": int(bad_copies),
            "clean_bad_aug_strength": float(bad_strength),
            "good_augmented_added": int(added_by_class["good"]),
            "medium_augmented_added": int(added_by_class["medium"]),
            "bad_source_rows": int(bad_idx.size),
            "bad_augmented_added": int(added_by_class["bad"]),
        }

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class PTBReplayDataset(Dataset):
    def __init__(
        self,
        split: str,
        channel_stats: ChannelStats,
        feature_norm: FeatureNorm,
        max_rows: int = 0,
        seed: int = 20260619,
    ) -> None:
        labels = pd.read_csv(TX.ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
        signals_all = np.load(TX.ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        feat_npz = np.load(TX.ARCH.DATA_DIR / "tabular_features.npz")
        features_all = feat_npz["features"].astype(np.float32)[:, FORMAL_FEATURE_IDX]
        mask = labels["split"].astype(str).eq(split).to_numpy()
        rows = labels.loc[mask, "idx"].to_numpy(dtype=np.int64)
        frame = labels.loc[mask].reset_index(drop=True)
        if int(max_rows) > 0 and len(rows) > int(max_rows):
            rng = np.random.default_rng(int(seed))
            chosen: list[int] = []
            y_names = frame["y_class"].astype(str).to_numpy()
            per_class = max(1, int(max_rows) // 3)
            for cls in ("good", "medium", "bad"):
                idx = np.flatnonzero(y_names == cls)
                if idx.size:
                    take = min(per_class, idx.size)
                    chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
            remaining = int(max_rows) - len(chosen)
            if remaining > 0:
                pool = np.setdiff1d(np.arange(len(rows)), np.asarray(chosen, dtype=int), assume_unique=False)
                if pool.size:
                    chosen.extend(rng.choice(pool, size=min(remaining, pool.size), replace=False).tolist())
            chosen = sorted(set(chosen))
            rows = rows[chosen]
            frame = frame.iloc[chosen].reset_index(drop=True)
        signals = signals_all[rows].copy()
        features = features_all[rows].copy()
        self.frame = frame
        self.x = make_dualview_channels(signals, channel_stats)
        self.aux = ((features - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.y = frame["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class FullButDataset(Dataset):
    def __init__(self, channel_stats: ChannelStats, feature_norm: FeatureNorm) -> None:
        atlas = pd.read_csv(FULL_ATLAS_PATH)
        signals = np.load(FULL_SIGNALS_PATH)["X"].astype(np.float32)
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        for col in FORMAL_FEATURE_COLUMNS:
            if col not in atlas.columns:
                atlas[col] = 0.0
        rows = atlas["idx"].astype(int).to_numpy()
        self.frame = atlas.reset_index(drop=True)
        self.x = make_dualview_channels(signals[rows], channel_stats)
        raw = (
            atlas[FORMAL_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        self.aux = ((raw - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.aux_raw = raw.astype(np.float32)
        self.y = atlas["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.split = atlas["split"].astype(str).to_numpy()
        self.region = atlas.get("original_region", pd.Series("", index=atlas.index)).astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class HierTransformer(nn.Module):
    def __init__(self, cfg: dict[str, Any], in_ch: int, aux_dim: int):
        super().__init__()
        arch = str(cfg["arch"])
        width = int(cfg["width"])
        layers = int(cfg["layers"])
        heads = int(cfg["heads"])
        if arch == "convtx":
            self.encoder = TX.ConvSubsampleTransformer(in_ch, width, layers, heads)
        elif arch == "conformer":
            self.encoder = TX.ConformerLite(in_ch, width, layers, heads)
        elif arch == "multipatch":
            self.encoder = TX.MultiPatchTransformer(in_ch, width, layers, heads)
        else:
            raise ValueError(f"unknown arch {arch}")
        dim = int(self.encoder.out_dim)
        self.shared = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 192), nn.GELU(), nn.Dropout(0.10))
        self.base_logits = nn.Linear(192, 3)
        self.gm_head = nn.Linear(192, 1)
        self.bad_head = nn.Linear(192, 1)
        self.ord_head = nn.Linear(192, 2)
        self.logit_fusion = nn.Sequential(
            nn.LayerNorm(3 + 1 + 1 + 2),
            nn.Linear(7, 3),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, aux_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)
        h = self.shared(feat)
        base = self.base_logits(h)
        gm = self.gm_head(h)
        bad = self.bad_head(h)
        ordinal = self.ord_head(h)
        logits = base + self.logit_fusion(torch.cat([base, gm, bad, ordinal], dim=1))
        return {
            "logits": logits,
            "base_logits": base,
            "gm_logit": gm.squeeze(1),
            "bad_logit": bad.squeeze(1),
            "ordinal_logits": ordinal,
            "aux_pred": self.aux_head(feat),
            "features": feat,
        }


def classification_losses(out: dict[str, torch.Tensor], y: torch.Tensor, aux: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    class_weight = torch.tensor(cfg["class_weight"], dtype=torch.float32, device=device)
    ce = F.cross_entropy(out["logits"], y, weight=class_weight)
    gm_mask = y <= 1
    if torch.any(gm_mask):
        gm_target = (y[gm_mask] == CLASS_TO_INT["medium"]).float()
        gm_loss = F.binary_cross_entropy_with_logits(out["gm_logit"][gm_mask], gm_target)
    else:
        gm_loss = torch.zeros((), device=device)
    bad_target = (y == CLASS_TO_INT["bad"]).float()
    pos = bad_target.sum().clamp_min(1.0)
    neg = (1.0 - bad_target).sum().clamp_min(1.0)
    bad_pos_weight = (neg / pos).clamp(1.0, 8.0)
    bad_loss = F.binary_cross_entropy_with_logits(out["bad_logit"], bad_target, pos_weight=bad_pos_weight)
    nonbad_prob = torch.sigmoid(out["bad_logit"][bad_target < 0.5])
    false_bad = nonbad_prob.mean() if nonbad_prob.numel() else torch.zeros((), device=device)
    ord_target = torch.stack([(y >= 1).float(), (y >= 2).float()], dim=1)
    ord_loss = F.binary_cross_entropy_with_logits(out["ordinal_logits"], ord_target)
    aux_loss = F.smooth_l1_loss(out["aux_pred"], aux)
    core_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_AUX_IDX], aux[:, CORE_AUX_IDX]) if CORE_AUX_IDX else torch.zeros((), device=device)
    return (
        ce
        + float(cfg["gm_weight"]) * gm_loss
        + float(cfg["bad_weight"]) * bad_loss
        + float(cfg["ordinal_weight"]) * ord_loss
        + float(cfg["false_bad_penalty"]) * false_bad
        + float(cfg["aux_weight"]) * aux_loss
        + float(cfg["core_aux_weight"]) * core_loss
    )


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    aux_pred: list[np.ndarray] = []
    aux_true: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x)
        probs.append(F.softmax(out["logits"], dim=1).detach().cpu().numpy())
        aux_pred.append(out["aux_pred"].detach().cpu().numpy())
        aux_true.append(batch["aux"].numpy())
        ys.append(batch["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    ap = np.concatenate(aux_pred)
    at = np.concatenate(aux_true)
    rep = GEOM.metric_report(y, p.argmax(axis=1), p)
    rep["aux_mae"] = float(np.mean(np.abs(ap - at)))
    rep["core_aux_mae"] = float(np.mean(np.abs(ap[:, CORE_AUX_IDX] - at[:, CORE_AUX_IDX]))) if CORE_AUX_IDX else rep["aux_mae"]
    return rep, p, ap, at


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def recovery_rows(candidate: str, bucket: str, pred_norm: np.ndarray, true_norm: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(FORMAL_FEATURE_COLUMNS):
        rows.append(
            {
                "candidate": candidate,
                "bucket": bucket,
                "feature": name,
                "corr_norm": pearson_corr(pred_norm[:, i], true_norm[:, i]),
                "mae_norm": float(np.mean(np.abs(pred_norm[:, i] - true_norm[:, i]))),
                "is_key": name in KEY_RECOVERY_FEATURES,
            }
        )
    return rows


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def bucket_mask(ds: FullButDataset, bucket: str) -> np.ndarray:
    if bucket == "original_all_10s+":
        return np.ones(len(ds.y), dtype=bool)
    if bucket == "original_test_all_10s+":
        return ds.split == "test"
    if bucket == "original_test_main_without_bad_stress":
        return (ds.split == "test") & ~((ds.y == CLASS_TO_INT["bad"]) & (ds.region == "outlier_low_confidence"))
    if bucket == "bad_core_nearboundary":
        return (ds.split == "test") & (ds.y == CLASS_TO_INT["bad"]) & (ds.region != "outlier_low_confidence")
    if bucket == "bad_outlier_stress":
        return (ds.split == "test") & (ds.y == CLASS_TO_INT["bad"]) & (ds.region == "outlier_low_confidence")
    raise ValueError(f"unknown bucket {bucket}")


def bucket_report(ds: FullButDataset, probs: np.ndarray, bucket: str) -> dict[str, Any]:
    mask = bucket_mask(ds, bucket)
    return GEOM.metric_report(ds.y[mask], probs.argmax(axis=1)[mask], probs[mask])


def train_candidate(policy: str, candidate: str, cfg: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"training policy={policy} candidate={candidate}: {cfg}", flush=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol_dir = PROTOCOL_ROOT / policy
    train_ds = CleanButDataset(protocol_dir, "train")
    clean_bad_aug_summary = {
        "clean_good_aug_copies": 0,
        "clean_good_aug_strength": 0.0,
        "clean_good_aug_mode": "attenuate",
        "clean_medium_aug_copies": 0,
        "clean_medium_aug_strength": 0.0,
        "clean_bad_aug_copies": 0,
        "clean_bad_aug_strength": 0.0,
        "good_augmented_added": 0,
        "medium_augmented_added": 0,
        "bad_source_rows": int(np.sum(train_ds.y == CLASS_TO_INT["bad"])),
        "bad_augmented_added": 0,
    }
    if int(args.clean_bad_aug_copies) > 0 or int(args.clean_good_aug_copies) > 0 or int(args.clean_medium_aug_copies) > 0:
        train_ds = AugmentedCleanButDataset(
            train_ds,
            int(args.clean_bad_aug_copies),
            float(args.clean_bad_aug_strength),
            int(cfg["seed"]) + 313,
            int(args.clean_good_aug_copies),
            float(args.clean_good_aug_strength),
            str(args.clean_good_aug_mode),
            int(args.clean_medium_aug_copies),
            float(args.clean_medium_aug_strength),
        )
        clean_bad_aug_summary = dict(train_ds.augmentation_summary)
        print(f"enabled clean-BUT source-only augmentation: {clean_bad_aug_summary}", flush=True)
    val_ds = CleanButDataset(protocol_dir, "val", train_ds.channel_stats, train_ds.feature_norm)
    test_ds = CleanButDataset(protocol_dir, "test", train_ds.channel_stats, train_ds.feature_norm)
    full_ds = FullButDataset(train_ds.channel_stats, train_ds.feature_norm) if bool(args.include_full_diagnostic) else None
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    replay_loader = None
    replay_iter = None
    if float(args.ptb_replay_weight) > 0:
        replay_ds = PTBReplayDataset(
            "train",
            train_ds.channel_stats,
            train_ds.feature_norm,
            int(args.ptb_replay_max_rows),
            int(cfg["seed"]) + 17,
        )
        replay_loader = DataLoader(
            replay_ds,
            batch_size=max(16, int(args.ptb_replay_batch_size)),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        replay_iter = iter(replay_loader)
        print(
            f"enabled PTB replay: rows={len(replay_ds)} batch={args.ptb_replay_batch_size} weight={args.ptb_replay_weight}",
            flush=True,
        )
    full_loader = (
        DataLoader(full_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        if full_ds is not None
        else None
    )
    model = HierTransformer(cfg, int(train_ds.x.shape[1]), len(FORMAL_FEATURE_COLUMNS)).to(device)
    init_checkpoint = str(getattr(args, "init_checkpoint", "") or "").strip()
    if init_checkpoint:
        init = torch.load(init_checkpoint, map_location="cpu")
        state = init.get("model_state", init)
        if any(str(k).startswith("base.") for k in state):
            raise ValueError(f"init checkpoint appears to be stress-aware/incompatible for clean runner: {init_checkpoint}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"init checkpoint incompatible: missing={missing}, unexpected={unexpected}")
        print(f"initialized {candidate} from {init_checkpoint}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -1e9
    logs: list[dict[str, Any]] = []
    epoch_states: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(x)
            loss = classification_losses(out, y, aux, cfg, device)
            if replay_loader is not None and replay_iter is not None:
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_batch = next(replay_iter)
                rx = replay_batch["x"].to(device=device, dtype=torch.float32)
                ry = replay_batch["y"].to(device=device)
                raux = replay_batch["aux"].to(device=device, dtype=torch.float32)
                replay_loss = classification_losses(model(rx), ry, raux, cfg, device)
                loss = loss + float(args.ptb_replay_weight) * replay_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, _, _, _ = eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.15 * val_rep["macro_f1"] + 0.10 * min(val_rep["good_recall"], val_rep["medium_recall"]) - 0.02 * val_rep["aux_mae"]
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"{candidate} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} aux={val_rep['aux_mae']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if bool(getattr(args, "save_epoch_checkpoints", False)):
            epoch_states.append(
                {
                    "epoch": int(epoch),
                    "score": float(score),
                    "val_report": {k: v for k, v in val_rep.items() if k != "confusion_3x3"},
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
            )
    assert best_state is not None
    model.load_state_dict(best_state)
    val_rep, val_probs, val_aux_pred, val_aux_true = eval_loader(model, val_loader, device)
    test_rep, test_probs, test_aux_pred, test_aux_true = eval_loader(model, test_loader, device)
    full_probs = None
    full_aux_pred = None
    full_aux_true = None
    if full_loader is not None:
        _, full_probs, full_aux_pred, full_aux_true = eval_loader(model, full_loader, device)
    suffix = str(getattr(args, "run_suffix", "") or "").strip()
    suffix = ("_" + suffix) if suffix else ""
    run_name = f"{policy}_{candidate}{suffix}"
    metric_rows = [
        metric_row(run_name, "clean_val", val_rep),
        metric_row(run_name, "clean_test", test_rep),
    ]
    if full_ds is not None and full_probs is not None:
        metric_rows.extend(
            [
                metric_row(run_name, "legacy_original_test_all_10s+", bucket_report(full_ds, full_probs, "original_test_all_10s+")),
                metric_row(run_name, "legacy_original_all_10s+", bucket_report(full_ds, full_probs, "original_all_10s+")),
                metric_row(run_name, "legacy_original_test_main_without_bad_stress", bucket_report(full_ds, full_probs, "original_test_main_without_bad_stress")),
                metric_row(run_name, "legacy_bad_core_nearboundary", bucket_report(full_ds, full_probs, "bad_core_nearboundary")),
                metric_row(run_name, "legacy_bad_outlier_stress", bucket_report(full_ds, full_probs, "bad_outlier_stress")),
            ]
        )
    recovery = []
    recovery.extend(recovery_rows(run_name, "clean_val", val_aux_pred, val_aux_true))
    recovery.extend(recovery_rows(run_name, "clean_test", test_aux_pred, test_aux_true))
    if full_aux_pred is not None and full_aux_true is not None:
        recovery.extend(recovery_rows(run_name, "legacy_original_all_10s+", full_aux_pred, full_aux_true))
    run_dir = OUT_ROOT / "runs" / "clean_but_dualview_hier_transformer" / policy / f"{candidate}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "fixed_10s_dualview_hier_transformer",
            "input_contract": "waveform-derived fixed-10s channels only; no tabular feature input at inference",
            "formal_feature_columns": FORMAL_FEATURE_COLUMNS,
            "excluded_formal_aux": sorted(EXCLUDED_FORMAL_AUX),
            "feature_train_mean": train_ds.feature_norm.mean,
            "feature_train_std": train_ds.feature_norm.std,
            "channel_stats": train_ds.channel_stats.__dict__,
            "best_epoch": best_epoch,
            "policy": policy,
            "init_checkpoint": init_checkpoint,
            "run_suffix": suffix,
            "clean_bad_augmentation": clean_bad_aug_summary,
        },
        ckpt_path,
    )
    if epoch_states:
        for item in epoch_states:
            epoch_path = run_dir / f"ckpt_epoch{int(item['epoch']):02d}.pt"
            torch.save(
                {
                    "model_state": item["state"],
                    "candidate_config": cfg,
                    "model_kind": "fixed_10s_dualview_hier_transformer",
                    "input_contract": "waveform-derived fixed-10s channels only; no tabular feature input at inference",
                    "formal_feature_columns": FORMAL_FEATURE_COLUMNS,
                    "excluded_formal_aux": sorted(EXCLUDED_FORMAL_AUX),
                    "feature_train_mean": train_ds.feature_norm.mean,
                    "feature_train_std": train_ds.feature_norm.std,
                    "channel_stats": train_ds.channel_stats.__dict__,
                    "best_epoch": int(item["epoch"]),
                    "epoch_checkpoint": True,
                    "epoch_score": float(item["score"]),
                    "epoch_val_report": item["val_report"],
                    "policy": policy,
                    "init_checkpoint": init_checkpoint,
                    "run_suffix": suffix,
                    "clean_bad_augmentation": clean_bad_aug_summary,
                },
                epoch_path,
            )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "clean_test_probs.npz", probs=test_probs)
    if full_probs is not None:
        np.savez_compressed(run_dir / "legacy_full_but_probs.npz", probs=full_probs)
    summary = {
        "candidate": run_name,
        "policy": policy,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "clean_test": test_rep,
        "include_full_diagnostic": bool(args.include_full_diagnostic),
        "init_checkpoint": init_checkpoint,
        "run_suffix": suffix,
        "clean_bad_augmentation": clean_bad_aug_summary,
        "ptb_replay_weight": float(args.ptb_replay_weight),
        "ptb_replay_max_rows": int(args.ptb_replay_max_rows),
        "feature_columns": FORMAL_FEATURE_COLUMNS,
        "key_recovery_legacy_original_all_10s+": {
            row["feature"]: row["corr_norm"]
            for row in recovery
            if row["bucket"] == "legacy_original_all_10s+" and row["feature"] in KEY_RECOVERY_FEATURES
        },
    }
    if full_ds is not None and full_probs is not None:
        summary["legacy_original_test_all_10s+"] = bucket_report(full_ds, full_probs, "original_test_all_10s+")
        summary["legacy_bad_outlier_stress"] = bucket_report(full_ds, full_probs, "bad_outlier_stress")
    return summary, metric_rows, recovery


def render_report(metrics: pd.DataFrame, recovery: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Clean BUT Dual-View Hierarchical Transformer",
        "",
        "No variable-length modeling. Input is fixed 10s waveform-derived channels only. Interpretable SQI/QRS/baseline/detail features supervise auxiliary heads but are not inference inputs.",
        "",
        "Formal auxiliary targets exclude PCA/atlas/KNN geometry proxies (`pc*`, `pca_margin`, `boundary_confidence`, `region_confidence`, `knn_label_purity`).",
        "",
        "## Metrics",
        "",
        "| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    show_buckets = ["clean_val", "clean_test"]
    if bool(payload.get("include_full_diagnostic", False)):
        show_buckets.extend(
            [
                "legacy_original_test_all_10s+",
                "legacy_original_test_main_without_bad_stress",
                "legacy_bad_core_nearboundary",
                "legacy_bad_outlier_stress",
            ]
        )
    view = metrics[metrics["bucket"].isin(show_buckets)].copy()
    for _, row in view.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} | {float(row.get('aux_mae', 0.0)):.4f} |"
        )
    lines.extend(["", "## Key Feature Recovery", ""])
    key_bucket = "legacy_original_all_10s+" if bool(payload.get("include_full_diagnostic", False)) else "clean_test"
    key = recovery[(recovery["bucket"] == key_bucket) & (recovery["feature"].isin(KEY_RECOVERY_FEATURES))].copy()
    if not key.empty:
        lines.extend(["| Candidate | Feature | Corr | MAE |", "| --- | --- | ---: | ---: |"])
        for _, row in key.sort_values(["candidate", "feature"]).iterrows():
            lines.append(f"| {row['candidate']} | {row['feature']} | {float(row['corr_norm']):.4f} | {float(row['mae_norm']):.4f} |")
    lines.extend(["", "## Checkpoints", ""])
    for item in payload["summaries"]:
        lines.append(f"- `{item['candidate']}`: best_epoch={item['best_epoch']}, checkpoint=`{item['checkpoint']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `clean_test` answers whether fixed-10s cleaned labels are learnable.",
            "- Legacy full/original buckets are emitted only with `--include-full-diagnostic` and are not selection targets.",
            "- CV policies created from `margin_ge_5s_drop_outlier` are clean fixed-10s learnability checks, not record-heldout external tests.",
            "",
            f"Metrics CSV: `{payload['metrics_csv']}`",
            f"Feature recovery CSV: `{payload['feature_recovery_csv']}`",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=str, default="margin_ge_5s_drop_outlier")
    parser.add_argument("--candidates", type=str, default="dualview_convtx_hier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-full-diagnostic", action="store_true")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--ptb-replay-weight", type=float, default=0.0)
    parser.add_argument("--ptb-replay-batch-size", type=int, default=64)
    parser.add_argument("--ptb-replay-max-rows", type=int, default=0)
    parser.add_argument("--clean-good-aug-copies", type=int, default=0)
    parser.add_argument("--clean-good-aug-strength", type=float, default=1.0)
    parser.add_argument("--clean-good-aug-mode", type=str, default="attenuate", choices=["attenuate", "highamp_flat_qrsvisible"])
    parser.add_argument("--clean-medium-aug-copies", type=int, default=0)
    parser.add_argument("--clean-medium-aug-strength", type=float, default=1.0)
    parser.add_argument("--clean-bad-aug-copies", type=int, default=0)
    parser.add_argument("--clean-bad-aug-strength", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="")
    parser.add_argument("--gm-weight", type=float, default=-1.0)
    parser.add_argument("--bad-weight", type=float, default=-1.0)
    parser.add_argument("--false-bad-penalty", type=float, default=-1.0)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    args = parser.parse_args()
    OUT_DIR = ANALYSIS_DIR / "clean_but_dualview_hier_transformer"
    REPORT_OUT_DIR = REPORT_DIR / "clean_but_dualview_hier_transformer"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    for c in candidates:
        if c not in CANDIDATES:
            raise ValueError(f"unknown candidate {c}; expected {sorted(CANDIDATES)}")
    all_metrics: list[dict[str, Any]] = []
    all_recovery: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for policy in policies:
        if not (PROTOCOL_ROOT / policy).exists():
            raise FileNotFoundError(f"missing clean protocol {PROTOCOL_ROOT / policy}")
        for candidate in candidates:
            cfg = dict(CANDIDATES[candidate])
            if str(args.class_weight).strip():
                weights = [float(x) for x in str(args.class_weight).split(",")]
                if len(weights) != 3:
                    raise ValueError("--class-weight must contain three comma-separated values")
                cfg["class_weight"] = weights
            if float(args.gm_weight) >= 0:
                cfg["gm_weight"] = float(args.gm_weight)
            if float(args.bad_weight) >= 0:
                cfg["bad_weight"] = float(args.bad_weight)
            if float(args.false_bad_penalty) >= 0:
                cfg["false_bad_penalty"] = float(args.false_bad_penalty)
            summary, metrics, recovery = train_candidate(policy, candidate, cfg, args)
            summaries.append(summary)
            all_metrics.extend(metrics)
            all_recovery.extend(recovery)
    metrics_df = pd.DataFrame(all_metrics)
    recovery_df = pd.DataFrame(all_recovery)
    metrics_path = OUT_DIR / "clean_but_dualview_hier_transformer_metrics.csv"
    recovery_path = OUT_DIR / "clean_but_dualview_hier_transformer_feature_recovery.csv"
    summary_path = OUT_DIR / "clean_but_dualview_hier_transformer_summary.json"
    report_path = OUT_DIR / "clean_but_dualview_hier_transformer_report.md"
    report_copy = REPORT_OUT_DIR / report_path.name
    metrics_df.to_csv(metrics_path, index=False)
    recovery_df.to_csv(recovery_path, index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_contract": "fixed 10s waveform-derived channels only; no tabular feature input at inference",
        "variable_length_used": False,
        "include_full_diagnostic": bool(args.include_full_diagnostic),
        "formal_feature_columns": FORMAL_FEATURE_COLUMNS,
        "excluded_formal_aux": sorted(EXCLUDED_FORMAL_AUX),
        "metrics_csv": str(metrics_path),
        "feature_recovery_csv": str(recovery_path),
        "report": str(report_copy),
        "summaries": summaries,
    }
    write_json(summary_path, payload)
    report = render_report(metrics_df, recovery_df, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
