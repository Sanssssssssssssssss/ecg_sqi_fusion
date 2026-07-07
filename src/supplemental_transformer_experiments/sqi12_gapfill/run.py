from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from scipy.signal import decimate, resample_poly
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.sqi_pipeline.noise.dower import dower_3_to_12
from src.sqi_pipeline.noise.pca import make_3_orthogonal_from_2_paper
from src.utils.data_downloads import ensure_ptbxl_raw_data, ensure_sqi_raw_data
from src.utils.paths import project_root

ROOT = project_root()
OUT_DEFAULT = ROOT / "outputs" / "transformer" / "supplemental" / "sqi12_gapfill"
SETA_DIR = ROOT / "data" / "physionet" / "challenge-2011" / "set-a"
PTBXL_DIR = ROOT / "data" / "ptb-xl"
NSTDB_DIR = ROOT / "data" / "physionet" / "nstdb"
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABEL_TO_Y = {"unacceptable": 0, "acceptable": 1}
Y_TO_LABEL = {0: "unacceptable", 1: "acceptable"}
NOISE_NOTE_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
NSTDB_NOISE_TYPES = ("em", "ma")
_NSTDB_CACHE: dict[str, np.ndarray] | None = None
SUPPORT_BINS = ("subtle", "mid", "severe")


@dataclass(frozen=True)
class Paths:
    out: Path

    @property
    def data(self) -> Path:
        return self.out / "data"

    @property
    def figures(self) -> Path:
        return self.out / "figures"

    @property
    def reports(self) -> Path:
        return self.out / "reports"

    @property
    def models(self) -> Path:
        return self.out / "models" / "e31_leadwise_shared"

    @property
    def manifest_csv(self) -> Path:
        return self.data / "manifest_seta_labeled.csv"

    @property
    def split_csv(self) -> Path:
        return self.data / "split_seed0_original.csv"

    @property
    def protocol_csv(self) -> Path:
        return self.data / "protocol_gapfill.csv"

    @property
    def signals_npz(self) -> Path:
        return self.data / "signals.npz"

    @property
    def candidate_pool_csv(self) -> Path:
        return self.data / "candidate_pool.csv"

    @property
    def candidate_pool_signals_npz(self) -> Path:
        return self.data / "candidate_pool_signals.npz"

    @property
    def feature_csv(self) -> Path:
        return self.data / "features.csv"

    @property
    def selection_trace_csv(self) -> Path:
        return self.data / "selection_trace.csv"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 45
    patience: int = 45
    batch_size: int = 24
    lr: float = 1.5e-4
    weight_decay: float = 1.0e-4
    width: int = 96
    layers: int = 3
    heads: int = 4
    factor_weight: float = 0.06
    local_weight: float = 0.03
    bad_class_weight: float = 1.25
    grad_accum_steps: int = 1
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.data, paths.figures, paths.reports, paths.models]:
        p.mkdir(parents=True, exist_ok=True)


def dry(step: str, paths: Paths) -> None:
    print(
        json.dumps(
            {
                "step": step,
                "out": str(paths.out),
                "manifest": str(paths.manifest_csv),
                "split": str(paths.split_csv),
                "protocol": str(paths.protocol_csv),
                "signals": str(paths.signals_npz),
            },
            indent=2,
        )
    )


def rid_text(x: Any) -> str:
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def read_id_list(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        rid = Path(s.replace("\\", "/").split("/")[-1]).stem
        if rid and rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def normalize_leads(names: list[str]) -> list[str]:
    out = []
    for name in names:
        u = str(name).strip()
        table = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
        out.append(table.get(u.upper(), u))
    return out


def reorder_12(sig: np.ndarray, names: list[str]) -> np.ndarray:
    names_n = normalize_leads(names)
    missing = [lead for lead in LEADS_12 if lead not in names_n]
    if missing:
        raise ValueError(f"missing required leads {missing}; got {names}")
    idx = [names_n.index(lead) for lead in LEADS_12]
    return np.asarray(sig[:, idx], dtype=np.float32)


def load_seta_125(record_id: str) -> np.ndarray:
    sig, fields = wfdb.rdsamp(str(SETA_DIR / record_id))
    x = reorder_12(sig, list(fields["sig_name"]))
    fs = int(fields["fs"])
    if fs != 500:
        raise ValueError(f"{record_id}: expected 500 Hz, got {fs}")
    if x.shape[0] < 5000:
        raise ValueError(f"{record_id}: expected >=5000 samples, got {x.shape[0]}")
    y = decimate(x[:5000], q=4, axis=0, ftype="fir", zero_phase=True)
    return y[:1250].astype(np.float32)


def clean_noise_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in NOISE_NOTE_COLS:
        if col in df.columns:
            s = df[col]
            mask &= s.isna() | s.astype(str).str.strip().isin(["", "nan", "None"])
    return mask


def load_ptb_125(filename_lr: str) -> np.ndarray:
    sig, fields = wfdb.rdsamp(str(PTBXL_DIR / filename_lr))
    x = reorder_12(sig, list(fields["sig_name"]))
    fs = int(fields["fs"])
    if fs != 100:
        raise ValueError(f"{filename_lr}: expected 100 Hz records100, got {fs}")
    y = resample_poly(x[:1000], up=5, down=4, axis=0)
    if y.shape[0] < 1250:
        y = np.pad(y, ((0, 1250 - y.shape[0]), (0, 0)), mode="edge")
    return y[:1250].astype(np.float32)


def cmd_manifest(paths: Paths, *, run: bool, force: bool) -> None:
    if not run:
        dry("manifest", paths)
        return
    ensure_dirs(paths)
    if paths.manifest_csv.exists() and not force:
        print(f"[manifest] exists: {paths.manifest_csv}")
        return
    records = read_id_list(SETA_DIR / "RECORDS")
    acceptable = set(read_id_list(SETA_DIR / "RECORDS-acceptable"))
    unacceptable = set(read_id_list(SETA_DIR / "RECORDS-unacceptable"))
    rows: list[dict[str, Any]] = []
    for rid in records:
        if rid in acceptable:
            label = "acceptable"
        elif rid in unacceptable:
            label = "unacceptable"
        else:
            continue
        header = wfdb.rdheader(str(SETA_DIR / rid))
        rows.append(
            {
                "record_id": rid,
                "quality_record": label,
                "y": LABEL_TO_Y[label],
                "fs": float(header.fs),
                "sig_len": int(header.sig_len),
                "n_sig": int(header.n_sig),
                "duration_s": float(header.sig_len / header.fs),
                "sig_name": ",".join(normalize_leads(list(header.sig_name))),
                "base_path": str(SETA_DIR),
            }
        )
    df = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    counts = df["quality_record"].value_counts().to_dict()
    if len(df) != 998 or counts.get("acceptable") != 773 or counts.get("unacceptable") != 225:
        raise SystemExit(f"Unexpected Set-A labeled counts: rows={len(df)} counts={counts}")
    df.to_csv(paths.manifest_csv, index=False)
    print(f"[manifest] wrote {paths.manifest_csv} rows={len(df)} counts={counts}")


def stratified_split(y: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for cls in sorted(np.unique(y)):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        out["train"].extend(idx[:n_train].tolist())
        out["val"].extend(idx[n_train : n_train + n_val].tolist())
        out["test"].extend(idx[n_train + n_val :].tolist())
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


def cmd_split(paths: Paths, *, run: bool, force: bool) -> None:
    if not run:
        dry("split", paths)
        return
    ensure_dirs(paths)
    if paths.split_csv.exists() and not force:
        print(f"[split] exists: {paths.split_csv}")
        return
    df = pd.read_csv(paths.manifest_csv)
    df["record_id"] = df["record_id"].apply(rid_text)
    y = df["y"].astype(int).to_numpy()
    split_idx = stratified_split(y, seed=0)
    df["split"] = ""
    for split, idx in split_idx.items():
        df.iloc[idx, df.columns.get_loc("split")] = split
    table = df.groupby(["split", "quality_record"]).size().unstack(fill_value=0)
    expected = {
        ("train", "acceptable"): 541,
        ("train", "unacceptable"): 158,
        ("val", "acceptable"): 116,
        ("val", "unacceptable"): 34,
        ("test", "acceptable"): 116,
        ("test", "unacceptable"): 33,
    }
    for key, n in expected.items():
        if int(table.loc[key[0], key[1]]) != n:
            raise SystemExit(f"Unexpected split count for {key}: {int(table.loc[key[0], key[1]])} != {n}")
    df.to_csv(paths.split_csv, index=False)
    print(f"[split] wrote {paths.split_csv}")
    print(table.to_string())


def rolling_mean_1d(x: np.ndarray, win: int) -> np.ndarray:
    kernel = np.ones(int(win), dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def feature_row(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    out: dict[str, float] = {}
    rms = np.sqrt(np.mean(x * x, axis=0))
    ptp = np.percentile(x, 99, axis=0) - np.percentile(x, 1, axis=0)
    diff = np.percentile(np.abs(np.diff(x, axis=0)), 95, axis=0)
    q10 = np.percentile(x, 10, axis=0)
    q50 = np.percentile(x, 50, axis=0)
    q90 = np.percentile(x, 90, axis=0)
    baseline = np.stack([rolling_mean_1d(x[:, j], 251) for j in range(12)], axis=1)
    base_std = np.std(baseline, axis=0)
    high = x - baseline
    high_rms = np.sqrt(np.mean(high * high, axis=0))
    for j, lead in enumerate(LEADS_12):
        key = lead.replace("a", "A")
        out[f"{key}_rms"] = float(rms[j])
        out[f"{key}_ptp"] = float(ptp[j])
        out[f"{key}_diff95"] = float(diff[j])
        out[f"{key}_q10"] = float(q10[j])
        out[f"{key}_q50"] = float(q50[j])
        out[f"{key}_q90"] = float(q90[j])
        out[f"{key}_base_std"] = float(base_std[j])
        out[f"{key}_high_rms"] = float(high_rms[j])
    out["median_rms"] = float(np.median(rms))
    out["median_ptp"] = float(np.median(ptp))
    out["median_diff95"] = float(np.median(diff))
    out["median_base_std"] = float(np.median(base_std))
    out["median_high_rms"] = float(np.median(high_rms))
    centered = x - np.mean(x, axis=0, keepdims=True)
    sd = np.std(centered, axis=0)
    valid = sd > 1e-6
    if int(valid.sum()) >= 2:
        zn = centered[:, valid] / sd[None, valid]
        c = (zn.T @ zn) / max(zn.shape[0] - 1, 1)
        corr_vals = c[np.triu_indices(c.shape[0], 1)]
        out["lead_corr_mean"] = float(np.nanmean(corr_vals)) if np.isfinite(corr_vals).any() else 0.0
    else:
        out["lead_corr_mean"] = 0.0
    hist, _ = np.histogram(x.ravel(), bins=48, density=True)
    p = hist / max(float(hist.sum()), 1e-12)
    out["amp_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return out


def feature_frame(signals: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([feature_row(signals[i]) for i in range(signals.shape[0])])


def feature_cols(df: pd.DataFrame) -> list[str]:
    meta = {
        "row_idx",
        "record_id",
        "source_record_id",
        "candidate_type",
        "target_support_bin",
        "lead_support_bin",
        "lead_localized",
        "split",
        "quality_record",
        "label",
        "y",
        "generated",
        "donor_split",
        "ptb_ecg_id",
        "ptb_patient_id",
        "style_anchor_record_id",
    }
    return [c for c in df.columns if c not in meta]


def lead_quality_matrix(signals: np.ndarray) -> np.ndarray:
    x = np.asarray(signals, dtype=np.float32)
    amp = np.ptp(x, axis=1)
    centered = x - np.mean(x, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    diff95 = np.percentile(np.abs(np.diff(x, axis=1)), 95, axis=1)
    return np.stack([amp, rms, diff95], axis=2).astype(np.float32)


def support_bin(count: int) -> str:
    if int(count) <= 1:
        return "subtle"
    if int(count) <= 3:
        return "mid"
    return "severe"


def lead_support(signals: np.ndarray, reference_signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = lead_quality_matrix(reference_signals)
    x = lead_quality_matrix(signals)
    center = ref.mean(axis=0, keepdims=True)
    scale = np.maximum(ref.std(axis=0, keepdims=True), 1e-6)
    z = np.max(np.abs((x - center) / scale), axis=2)
    counts = (z > 3.0).sum(axis=1).astype(np.int16)
    bins = np.asarray([support_bin(int(v)) for v in counts], dtype=object)
    return counts, bins, z.astype(np.float32)


def support_leads(z_row: np.ndarray, bin_name: str, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(-np.asarray(z_row, dtype=np.float32))
    if bin_name == "subtle":
        n = 1
    elif bin_name == "mid":
        n = int(rng.choice([2, 3]))
    else:
        n = int(rng.integers(6, 13))
    return np.asarray(order[:n], dtype=np.int64)


def localized_lead_artifact(carrier: np.ndarray, artifact: np.ndarray, leads: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = carrier.astype(np.float32).copy()
    a = artifact.astype(np.float32)
    leads = np.asarray(leads, dtype=np.int64)
    blend = float(rng.uniform(0.82, 1.0))
    if rng.random() < 0.70:
        y[:, leads] = (1.0 - blend) * y[:, leads] + blend * a[:, leads]
    else:
        n = y.shape[0]
        width = int(rng.integers(max(180, n // 3), n + 1))
        start = int(rng.integers(0, max(1, n - width + 1)))
        end = min(n, start + width)
        y[start:end, leads] = (1.0 - blend) * y[start:end, leads] + blend * a[start:end, leads]
    y += rng.normal(0.0, max(float(np.std(y)) * 0.0025, 1e-6), size=y.shape).astype(np.float32)
    return y.astype(np.float32)


def native_morph(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float32).copy()
    n = y.shape[0]
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    gain = rng.normal(1.0, 0.035, size=(1, 12)).astype(np.float32)
    y *= gain
    for j in range(12):
        amp = float(np.std(y[:, j]) * rng.uniform(0.010, 0.035))
        phase = float(rng.uniform(0, 2 * math.pi))
        freq = float(rng.uniform(0.25, 0.85))
        y[:, j] += amp * np.sin(2 * math.pi * freq * t + phase)
    shift = int(rng.integers(-4, 5))
    if shift:
        y = np.roll(y, shift, axis=0)
    y += rng.normal(0.0, max(float(np.std(y)) * 0.004, 1e-5), size=y.shape).astype(np.float32)
    return y.astype(np.float32)


def _smooth_time(x: np.ndarray, width: int = 7) -> np.ndarray:
    pad = width // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.stack([np.convolve(xp[:, j], kernel, mode="valid") for j in range(x.shape[1])], axis=1).astype(np.float32)


def match_detail_to_target(y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """ponytail: cheap guard; replace with SQI-in-pool selection if this becomes mainline."""
    out = y.astype(np.float32).copy()
    t_diff = np.percentile(np.abs(np.diff(target, axis=0)), 95, axis=0, keepdims=True)
    cap = np.maximum(1.35 * t_diff, 0.015 * np.std(target, axis=0, keepdims=True))
    for _ in range(3):
        y_diff = np.percentile(np.abs(np.diff(out, axis=0)), 95, axis=0, keepdims=True)
        scale = np.minimum(1.0, cap / np.maximum(y_diff, 1e-6)).astype(np.float32)
        if float(np.min(scale)) > 0.98:
            break
        smooth = _smooth_time(out, width=7)
        out = smooth + (out - smooth) * scale
    return out.astype(np.float32)


def nstdb_cache() -> dict[str, np.ndarray] | None:
    global _NSTDB_CACHE
    if _NSTDB_CACHE is not None:
        return _NSTDB_CACHE
    if not all((NSTDB_DIR / f"{name}.hea").exists() for name in NSTDB_NOISE_TYPES):
        return None
    out: dict[str, np.ndarray] = {}
    for name in NSTDB_NOISE_TYPES:
        rec = wfdb.rdrecord(str(NSTDB_DIR / name), physical=True)
        sig = np.asarray(rec.p_signal, dtype=np.float64)
        if int(round(float(rec.fs))) != 360 or sig.ndim != 2 or sig.shape[1] != 2:
            return None
        out[name] = sig
    _NSTDB_CACHE = out
    return out


def draw_nstdb_noise12_125(n: int, rng: np.random.Generator) -> np.ndarray | None:
    cache = nstdb_cache()
    if cache is None:
        return None
    name = str(rng.choice(NSTDB_NOISE_TYPES))
    sig360 = cache[name]
    seg_len = int(math.ceil(n * 360 / 125)) + 4
    if sig360.shape[0] <= seg_len:
        return None
    start = int(rng.integers(0, sig360.shape[0] - seg_len))
    seg125 = resample_poly(sig360[start : start + seg_len], up=25, down=72, axis=0)[:n]
    if seg125.shape[0] < n:
        seg125 = np.pad(seg125, ((0, n - seg125.shape[0]), (0, 0)), mode="edge")
    return dower_3_to_12(make_3_orthogonal_from_2_paper(seg125, rng=rng)).astype(np.float32)


def scale_noise_fraction(clean: np.ndarray, noise: np.ndarray, fraction: float) -> np.ndarray:
    clean_sd = np.maximum(np.std(clean, axis=0, keepdims=True), 1e-5)
    noise0 = noise - np.mean(noise, axis=0, keepdims=True)
    noise_sd = np.maximum(np.std(noise0, axis=0, keepdims=True), 1e-6)
    return (noise0 / noise_sd * clean_sd * float(fraction)).astype(np.float32)


def align_like_target(src: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s_med = np.median(src, axis=0, keepdims=True)
    s_std = np.std(src, axis=0, keepdims=True)
    t_med = np.median(target, axis=0, keepdims=True)
    t_std = np.std(target, axis=0, keepdims=True)
    y = (src - s_med) / np.maximum(s_std, 1e-6)
    y = y * np.maximum(t_std, 1e-6) * rng.uniform(0.94, 1.06, size=(1, 12)) + t_med
    y = 0.30 * y + 0.70 * target
    y = native_morph(y.astype(np.float32), rng)
    return match_detail_to_target(y, target)


def noise_style(x: np.ndarray, rng: np.random.Generator, leads: np.ndarray | None = None) -> np.ndarray:
    y = x.astype(np.float32).copy()
    n = y.shape[0]
    lead_idx = np.arange(12, dtype=np.int64) if leads is None else np.asarray(leads, dtype=np.int64)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    scale = max(float(np.std(y[:, lead_idx])), 1e-5)
    wander = np.sin(2 * math.pi * rng.uniform(0.12, 0.45) * t[:, None] + rng.uniform(0, 2 * math.pi, size=(1, len(lead_idx))))
    y[:, lead_idx] += (0.08 * scale * wander).astype(np.float32)
    noise12 = draw_nstdb_noise12_125(n, rng)
    if noise12 is None:
        y[:, lead_idx] += rng.normal(0.0, 0.035 * scale, size=(n, len(lead_idx))).astype(np.float32)
    else:
        y[:, lead_idx] += scale_noise_fraction(y[:, lead_idx], noise12[:, lead_idx], float(rng.uniform(0.018, 0.045)))
    if rng.random() < 0.45:
        start = int(rng.integers(0, max(1, n - 180)))
        width = int(rng.integers(80, 220))
        end = min(n, start + width)
        if noise12 is None:
            y[start:end, lead_idx] += rng.normal(0.0, 0.09 * scale, size=(end - start, len(lead_idx))).astype(np.float32)
        else:
            burst = scale_noise_fraction(y[start:end, lead_idx], noise12[start:end, lead_idx], float(rng.uniform(0.05, 0.11)))
            y[start:end, lead_idx] += (burst * np.hanning(end - start)[:, None]).astype(np.float32)
    return y.astype(np.float32)


def saturation_feature_mask(df: pd.DataFrame) -> pd.Series:
    return df["median_rms"].gt(40.0) & df["median_ptp"].lt(1.0) & df["median_diff95"].lt(0.05)


def saturation_morph(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float32).copy()
    n = y.shape[0]
    t = np.linspace(-0.5, 0.5, n, dtype=np.float32)[:, None]
    gain = rng.normal(1.0, 0.0012, size=(1, 12)).astype(np.float32)
    offset = rng.normal(0.0, 0.025, size=(1, 12)).astype(np.float32)
    drift = rng.normal(0.0, 0.006, size=(1, 12)).astype(np.float32) * t
    return (y * gain + offset + drift).astype(np.float32)


def robust_z_against(target: pd.DataFrame, pool: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    t = np.nan_to_num(target[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(pool[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    center = np.nanmedian(t, axis=0)
    q25 = np.nanpercentile(t, 25, axis=0)
    q75 = np.nanpercentile(t, 75, axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, np.nanstd(t, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return (t - center) / scale, (p - center) / scale


def smc_score(z: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray, target_q: np.ndarray) -> float:
    q = np.nanpercentile(z, [10, 50, 90], axis=0)
    return float(
        0.40 * np.nanmean(np.abs(np.nanmean(z, axis=0) - target_mean))
        + 0.25 * np.nanmean(np.abs(np.nanstd(z, axis=0) - target_std))
        + 0.35 * np.nanmean(np.abs(q - target_q))
    )


def enforce_noise_cap(idx: np.ndarray, pool: pd.DataFrame, pool_z: np.ndarray, target_mean: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0:
        return idx
    selected = list(map(int, idx))
    selected_set = set(selected)
    noise_selected = [i for i in selected if str(pool.iloc[i]["candidate_type"]) == "noise_style"]
    if len(noise_selected) <= cap:
        return np.asarray(selected, dtype=np.int64)
    extras = noise_selected[cap:]
    selected = [i for i in selected if i not in set(extras)]
    candidate_pool = [
        i
        for i in range(len(pool))
        if i not in selected_set and str(pool.iloc[i]["candidate_type"]) != "noise_style"
    ]
    candidate_pool.sort(key=lambda i: float(np.nanmean(np.abs(pool_z[i] - target_mean))))
    selected.extend(candidate_pool[: len(extras)])
    return np.asarray(selected, dtype=np.int64)


def strict_smc_select(
    target: pd.DataFrame,
    pool: pd.DataFrame,
    cols: list[str],
    n: int,
    seed: int,
    noise_cap: int,
    particles: int = 64,
    rounds: int = 7,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    target_z, pool_z = robust_z_against(target, pool, cols)
    target_mean = np.nanmean(target_z, axis=0)
    target_std = np.nanstd(target_z, axis=0)
    target_q = np.nanpercentile(target_z, [10, 50, 90], axis=0)
    dist = np.nanmean(np.abs(pool_z - target_mean[None, :]), axis=1)
    order = np.argsort(dist)
    particles_idx: list[np.ndarray] = []
    particles_idx.append(order[:n])
    prob = 1.0 / np.maximum(dist, 1e-4)
    prob = prob / prob.sum()
    for _ in range(particles - 1):
        particles_idx.append(rng.choice(len(pool), size=n, replace=False, p=prob))
    trace_rows: list[dict[str, Any]] = []
    best_idx = particles_idx[0]
    best_score = float("inf")
    for r in range(rounds):
        scored: list[tuple[float, np.ndarray]] = []
        for pid, idx in enumerate(particles_idx):
            idx = enforce_noise_cap(np.asarray(idx, dtype=np.int64), pool, pool_z, target_mean, noise_cap)
            s = smc_score(pool_z[idx], target_mean, target_std, target_q)
            scored.append((s, idx))
            trace_rows.append({"round": r, "particle": pid, "smc_score": s, "selected_n": len(idx)})
            if s < best_score:
                best_score = s
                best_idx = idx.copy()
        scored.sort(key=lambda x: x[0])
        elites = [idx for _, idx in scored[: max(4, particles // 4)]]
        particles_idx = [best_idx.copy()]
        while len(particles_idx) < particles:
            base = elites[int(rng.integers(0, len(elites)))].copy()
            used = set(map(int, base))
            swaps = max(4, int(0.08 * n))
            remove = rng.choice(len(base), size=min(swaps, len(base)), replace=False)
            available = np.asarray([i for i in range(len(pool)) if i not in used], dtype=np.int64)
            if len(available) < len(remove):
                particles_idx.append(base)
                continue
            add = rng.choice(available, size=len(remove), replace=False, p=None)
            base[remove] = add
            particles_idx.append(base)
    out = enforce_noise_cap(best_idx, pool, pool_z, target_mean, noise_cap)
    return out, pd.DataFrame(trace_rows)


def allocate_by_support(target: pd.DataFrame, n: int) -> dict[str, int]:
    counts = target["lead_support_bin"].value_counts().reindex(SUPPORT_BINS, fill_value=0).astype(float)
    if counts.sum() <= 0:
        counts[:] = 1.0
    raw = counts / counts.sum() * int(n)
    out = {k: int(math.floor(float(v))) for k, v in raw.items()}
    rem = int(n) - sum(out.values())
    frac_order = sorted(SUPPORT_BINS, key=lambda k: float(raw[k] - out[k]), reverse=True)
    for k in frac_order[:rem]:
        out[k] += 1
    return out


def support_stratified_smc_select(
    target: pd.DataFrame,
    pool: pd.DataFrame,
    base_mask: pd.Series,
    cols: list[str],
    n: int,
    seed: int,
    component: str,
    noise_cap: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    selected: list[np.ndarray] = []
    trace_parts: list[pd.DataFrame] = []
    quotas = allocate_by_support(target, int(n))
    selected_pool_index: set[int] = set()
    deficit = 0
    for offset, bin_name in enumerate(SUPPORT_BINS):
        want = int(quotas.get(bin_name, 0))
        if want <= 0:
            continue
        mask = base_mask & pool["lead_support_bin"].astype(str).eq(bin_name) & ~pool["pool_index"].isin(selected_pool_index)
        sub = pool.loc[mask].reset_index(drop=True)
        take = min(want, len(sub))
        deficit += want - take
        if take <= 0:
            continue
        target_bin = target.loc[target["lead_support_bin"].astype(str).eq(bin_name)].reset_index(drop=True)
        if target_bin.empty:
            target_bin = target
        rel_idx, tr = strict_smc_select(
            target_bin,
            sub,
            cols,
            take,
            seed + offset * 97,
            noise_cap=take if noise_cap else 0,
        )
        pool_idx = sub.iloc[rel_idx]["pool_index"].to_numpy(dtype=np.int64)
        selected.append(pool_idx)
        selected_pool_index.update(map(int, pool_idx))
        tr["component"] = f"{component}_{bin_name}"
        tr["component_n"] = int(take)
        trace_parts.append(tr)
    if deficit > 0:
        mask = base_mask & ~pool["pool_index"].isin(selected_pool_index)
        sub = pool.loc[mask].reset_index(drop=True)
        if len(sub) < deficit:
            raise SystemExit(f"candidate pool too small for {component}: need deficit {deficit}, have {len(sub)}")
        rel_idx, tr = strict_smc_select(
            target,
            sub,
            cols,
            int(deficit),
            seed + 4049,
            noise_cap=int(deficit) if noise_cap else 0,
        )
        pool_idx = sub.iloc[rel_idx]["pool_index"].to_numpy(dtype=np.int64)
        selected.append(pool_idx)
        tr["component"] = f"{component}_spill"
        tr["component_n"] = int(deficit)
        trace_parts.append(tr)
    if not selected:
        return np.asarray([], dtype=np.int64), pd.DataFrame()
    return np.concatenate(selected).astype(np.int64), pd.concat(trace_parts, ignore_index=True)


def load_original_signals(split: pd.DataFrame) -> np.ndarray:
    xs = []
    for rid in split["record_id"].astype(str):
        xs.append(load_seta_125(rid))
    return np.stack(xs, axis=0).astype(np.float32)


def ptb_rows(seed: int, max_records: int) -> pd.DataFrame:
    db = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    db = db.loc[clean_noise_mask(db)].dropna(subset=["filename_lr"]).copy()
    db = db.sample(frac=1.0, random_state=seed).head(max_records).reset_index(drop=True)
    return db


def cmd_build(paths: Paths, *, run: bool, force: bool, seed: int, max_ptb: int) -> None:
    if not run:
        dry("build", paths)
        return
    ensure_dirs(paths)
    if paths.protocol_csv.exists() and paths.signals_npz.exists() and not force:
        print(f"[build] exists: {paths.protocol_csv}")
        return
    rng = np.random.default_rng(seed)
    split = pd.read_csv(paths.split_csv)
    split["record_id"] = split["record_id"].apply(rid_text)
    original_x = load_original_signals(split)
    train = split[split["split"].eq("train")].reset_index(drop=True)
    train_acc = int((train["y"] == 1).sum())
    train_bad = int((train["y"] == 0).sum())
    gap = train_acc - train_bad
    if gap != 383:
        raise SystemExit(f"unexpected train gap: {gap}")

    split_features = feature_frame(original_x)
    feat_meta = split[["record_id", "split", "quality_record", "y"]].copy()
    feat_meta["candidate_type"] = "original_seta"
    feat_all = pd.concat([feat_meta.reset_index(drop=True), split_features], axis=1)

    train_bad_global = split.index[(split["split"].eq("train")) & (split["y"].eq(0))].to_numpy()
    train_acc_global = split.index[(split["split"].eq("train")) & (split["y"].eq(1))].to_numpy()
    original_support_count, original_support_bin, original_support_z = lead_support(original_x, original_x[train_acc_global])
    feat_all["lead_support_count"] = original_support_count
    feat_all["lead_support_bin"] = original_support_bin
    target_features = feat_all.iloc[train_bad_global].reset_index(drop=True)
    target_signals = original_x[train_bad_global]

    pool_rows: list[dict[str, Any]] = []
    pool_x: list[np.ndarray] = []
    for local_i, global_i in enumerate(train_bad_global):
        row = split.iloc[int(global_i)]
        for copy_i in range(4):
            src = original_x[int(global_i)]
            src_feat = feature_row(src)
            target_bin = str(original_support_bin[int(global_i)])
            artifact = saturation_morph(src, rng) if src_feat["median_rms"] > 40.0 and src_feat["median_ptp"] < 1.0 and src_feat["median_diff95"] < 0.05 else native_morph(src, rng)
            lead_localized = 0
            if target_bin in {"subtle", "mid"}:
                carrier_i = int(rng.choice(train_acc_global))
                leads = support_leads(original_support_z[int(global_i)], target_bin, rng)
                artifact = localized_lead_artifact(original_x[carrier_i], artifact, leads, rng)
                lead_localized = 1
            pool_x.append(artifact)
            pool_rows.append(
                {
                    "record_id": f"{row.record_id}__seta_native_morph__{copy_i}",
                    "source_record_id": str(row.record_id),
                    "candidate_type": "seta_native_morph",
                    "target_support_bin": target_bin,
                    "lead_localized": lead_localized,
                    "ptb_ecg_id": "",
                    "ptb_patient_id": "",
                    "style_anchor_record_id": str(row.record_id),
                }
            )

    ptb = ptb_rows(seed + 17, max_ptb)
    attempts = 0
    target_need_pool = max(gap * 3, 900)
    for _, row in ptb.iterrows():
        if len(pool_x) >= target_need_pool:
            break
        attempts += 1
        try:
            src = load_ptb_125(str(row["filename_lr"]))
        except Exception:
            continue
        anchor_i = int(rng.integers(0, len(target_signals)))
        anchor = target_signals[anchor_i]
        anchor_global = int(train_bad_global[anchor_i])
        anchor_record = str(split.iloc[anchor_global]["record_id"])
        target_bin = str(original_support_bin[anchor_global])
        artifact = align_like_target(src, anchor, rng)
        lead_localized = 0
        if target_bin in {"subtle", "mid"}:
            carrier_i = int(rng.choice(train_acc_global))
            leads = support_leads(original_support_z[anchor_global], target_bin, rng)
            artifact = localized_lead_artifact(original_x[carrier_i], artifact, leads, rng)
            lead_localized = 1
        pool_x.append(artifact)
        pool_rows.append(
            {
                "record_id": f"ptbxl_{int(row.ecg_id)}__ptb12_morph",
                "source_record_id": f"ptbxl_{int(row.ecg_id)}",
                "candidate_type": "ptb12_morph",
                "target_support_bin": target_bin,
                "lead_localized": lead_localized,
                "ptb_ecg_id": int(row.ecg_id),
                "ptb_patient_id": int(row.patient_id) if pd.notna(row.patient_id) else -1,
                "style_anchor_record_id": anchor_record,
            }
        )
    if len(pool_x) < gap:
        raise SystemExit(f"generated pool too small after {attempts} PTB attempts: {len(pool_x)} < {gap}")

    noise_candidates = min(120, len(train_acc_global))
    for global_i in rng.choice(train_acc_global, size=noise_candidates, replace=False):
        row = split.iloc[int(global_i)]
        anchor_global = int(rng.choice(train_bad_global))
        target_bin = str(original_support_bin[anchor_global])
        leads = support_leads(original_support_z[anchor_global], target_bin, rng)
        pool_x.append(noise_style(original_x[int(global_i)], rng, leads=None if target_bin == "severe" else leads))
        pool_rows.append(
            {
                "record_id": f"{row.record_id}__noise_style__0",
                "source_record_id": str(row.record_id),
                "candidate_type": "noise_style",
                "target_support_bin": target_bin,
                "lead_localized": int(target_bin != "severe"),
                "ptb_ecg_id": "",
                "ptb_patient_id": "",
                "style_anchor_record_id": str(row.record_id),
            }
        )

    pool_signals = np.stack(pool_x, axis=0).astype(np.float32)
    pool = pd.DataFrame(pool_rows)
    pool_feat = feature_frame(pool_signals)
    pool = pd.concat([pool.reset_index(drop=True), pool_feat], axis=1)
    pool_support_count, pool_support_bin, _ = lead_support(pool_signals, original_x[train_acc_global])
    pool["lead_support_count"] = pool_support_count
    pool["lead_support_bin"] = pool_support_bin
    cols = feature_cols(pool)
    pool["pool_index"] = np.arange(len(pool), dtype=int)
    quotas = {
        "seta_native_morph": min(int(round(gap * 0.40)), int(pool["candidate_type"].eq("seta_native_morph").sum())),
        "noise_style": min(19, int(round(gap * 0.05)), int(pool["candidate_type"].eq("noise_style").sum())),
    }
    quotas["ptb12_morph"] = gap - int(quotas["seta_native_morph"]) - int(quotas["noise_style"])
    selected_parts: list[np.ndarray] = []
    trace_parts: list[pd.DataFrame] = []
    target_sat = saturation_feature_mask(target_features)
    pool_sat = saturation_feature_mask(pool)
    target_regular = target_features.loc[~target_sat].reset_index(drop=True)
    sat_quota = min(
        int(quotas["seta_native_morph"]),
        int(round(gap * float(target_sat.mean()))),
        int((pool["candidate_type"].eq("seta_native_morph") & pool_sat).sum()),
    )
    if sat_quota > 0:
        sub = pool.loc[pool["candidate_type"].eq("seta_native_morph") & pool_sat].reset_index(drop=True)
        rel_idx, tr = strict_smc_select(
            target_features.loc[target_sat].reset_index(drop=True),
            sub,
            cols,
            int(sat_quota),
            seed + 19,
            noise_cap=0,
        )
        selected_parts.append(sub.iloc[rel_idx]["pool_index"].to_numpy(dtype=np.int64))
        tr["component"] = "seta_native_saturation"
        tr["component_n"] = int(sat_quota)
        trace_parts.append(tr)
        quotas["seta_native_morph"] = int(quotas["seta_native_morph"]) - int(sat_quota)
    for offset, (label, n_select) in enumerate(quotas.items()):
        mask = pool["candidate_type"].eq(label)
        target_for_label = target_regular if len(target_regular) else target_features
        if label == "seta_native_morph":
            mask &= ~pool_sat
        if label == "ptb12_morph":
            mask &= pool["median_diff95"].le(float(target_features["median_diff95"].quantile(0.99)))
        if int(n_select) <= 0:
            continue
        if int(mask.sum()) < int(n_select):
            raise SystemExit(f"candidate pool too small for {label}: {int(mask.sum())} < {n_select}")
        pool_idx, tr = support_stratified_smc_select(
            target_for_label,
            pool,
            mask,
            cols,
            int(n_select),
            seed + 31 + offset * 101,
            component=label,
            noise_cap=int(n_select) if label == "noise_style" else 0,
        )
        selected_parts.append(pool_idx)
        trace_parts.append(tr)
    selected_idx = np.concatenate(selected_parts)
    trace = pd.concat(trace_parts, ignore_index=True)
    selected_pool = pool.iloc[selected_idx].reset_index(drop=True)
    selected_x = pool_signals[selected_idx]

    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    for _, row in split.iterrows():
        i = len(signals)
        signals.append(original_x[int(row.name)])
        rows.append(
            {
                "row_idx": i,
                "record_id": str(row["record_id"]),
                "source_record_id": str(row["record_id"]),
                "split": str(row["split"]),
                "quality_record": str(row["quality_record"]),
                "label": str(row["quality_record"]),
                "y": int(row["y"]),
                "generated": 0,
                "candidate_type": "original_seta",
                "target_support_bin": "",
                "lead_localized": 0,
                "donor_split": str(row["split"]),
                "ptb_ecg_id": "",
                "ptb_patient_id": "",
                "style_anchor_record_id": "",
            }
        )
    for j, row in selected_pool.iterrows():
        i = len(signals)
        signals.append(selected_x[int(j)])
        source_id = str(row.source_record_id)
        donor_split = "ptb" if str(row.candidate_type) == "ptb12_morph" else "train"
        rows.append(
            {
                "row_idx": i,
                "record_id": str(row.record_id),
                "source_record_id": source_id,
                "split": "train",
                "quality_record": "generated_unacceptable",
                "label": "unacceptable",
                "y": 0,
                "generated": 1,
                "candidate_type": str(row.candidate_type),
                "target_support_bin": str(row.target_support_bin),
                "lead_localized": int(row.lead_localized),
                "donor_split": donor_split,
                "ptb_ecg_id": row.ptb_ecg_id,
                "ptb_patient_id": row.ptb_patient_id,
                "style_anchor_record_id": row.style_anchor_record_id,
            }
        )

    protocol = pd.DataFrame(rows)
    X = np.stack(signals, axis=0).astype(np.float32)
    final_support_count, final_support_bin, _ = lead_support(X, original_x[train_acc_global])
    protocol["lead_support_count"] = final_support_count
    protocol["lead_support_bin"] = final_support_bin
    all_feat = feature_frame(X)
    features_out = pd.concat(
        [
            protocol[
                [
                    "row_idx",
                    "record_id",
                    "split",
                    "label",
                    "y",
                    "generated",
                    "candidate_type",
                    "target_support_bin",
                    "lead_localized",
                    "lead_support_count",
                    "lead_support_bin",
                ]
            ],
            all_feat,
        ],
        axis=1,
    )
    protocol.to_csv(paths.protocol_csv, index=False)
    pool.to_csv(paths.candidate_pool_csv, index=False)
    features_out.to_csv(paths.feature_csv, index=False)
    trace.to_csv(paths.selection_trace_csv, index=False)
    np.savez_compressed(paths.signals_npz, X=X, leads=np.asarray(LEADS_12, dtype=object), fs=np.int32(125))
    np.savez_compressed(paths.candidate_pool_signals_npz, X=pool_signals, leads=np.asarray(LEADS_12, dtype=object), fs=np.int32(125))
    print(f"[build] wrote protocol rows={len(protocol)} signals={X.shape}")
    print(protocol.groupby(["split", "y"]).size().unstack(fill_value=0).to_string())
    print(protocol["candidate_type"].value_counts().to_string())
    print(pd.crosstab(protocol.loc[protocol["generated"].astype(int).eq(1), "candidate_type"], protocol.loc[protocol["generated"].astype(int).eq(1), "lead_support_bin"]).to_string())


def audit_dict(paths: Paths) -> dict[str, Any]:
    manifest = pd.read_csv(paths.manifest_csv)
    split = pd.read_csv(paths.split_csv)
    protocol = pd.read_csv(paths.protocol_csv)
    signals = np.load(paths.signals_npz, allow_pickle=True)["X"]
    allowed = {"original_seta", "seta_native_morph", "ptb12_morph", "noise_style"}
    train = protocol[protocol["split"].eq("train")]
    valtest = protocol[protocol["split"].isin(["val", "test"])]
    split_table = split.groupby(["split", "quality_record"]).size().unstack(fill_value=0)
    out = {
        "manifest_rows": int(len(manifest)),
        "manifest_counts": {str(k): int(v) for k, v in manifest["quality_record"].value_counts().items()},
        "split_counts": {
            f"{a}_{b}": int(v) for (a, b), v in split.groupby(["split", "quality_record"]).size().items()
        },
        "protocol_rows": int(len(protocol)),
        "signals_shape": list(signals.shape),
        "train_counts": {str(k): int(v) for k, v in train["y"].value_counts().sort_index().items()},
        "val_test_generated_rows": int(valtest["generated"].astype(int).sum()),
        "train_generated_donor_split_problems": int(
            ((train["generated"].astype(int).eq(1)) & (~train["donor_split"].astype(str).isin(["train", "ptb"]))).sum()
        ),
        "candidate_types": sorted(protocol["candidate_type"].astype(str).unique().tolist()),
        "candidate_type_counts": {str(k): int(v) for k, v in protocol["candidate_type"].value_counts().items()},
        "ptb_generated_rows": int(protocol["candidate_type"].eq("ptb12_morph").sum()),
    }
    if "lead_support_bin" in protocol.columns:
        gen = train.loc[train["generated"].astype(int).eq(1)]
        out["train_generated_support_counts"] = {
            str(k): int(v) for k, v in gen["lead_support_bin"].astype(str).value_counts().reindex(SUPPORT_BINS, fill_value=0).items()
        }
        out["train_original_unacceptable_support_counts"] = {
            str(k): int(v)
            for k, v in train.loc[(train["generated"].astype(int).eq(0)) & (train["y"].astype(int).eq(0)), "lead_support_bin"]
            .astype(str)
            .value_counts()
            .reindex(SUPPORT_BINS, fill_value=0)
            .items()
        }
    if out["manifest_rows"] != 998 or out["manifest_counts"].get("acceptable") != 773 or out["manifest_counts"].get("unacceptable") != 225:
        raise SystemExit(f"manifest audit failed: {out}")
    expected_split = {
        ("train", "acceptable"): 541,
        ("train", "unacceptable"): 158,
        ("val", "acceptable"): 116,
        ("val", "unacceptable"): 34,
        ("test", "acceptable"): 116,
        ("test", "unacceptable"): 33,
    }
    for key, expected in expected_split.items():
        if int(split_table.loc[key[0], key[1]]) != expected:
            raise SystemExit(f"split audit failed: {key}")
    if out["train_counts"] != {"0": 541, "1": 541}:
        raise SystemExit(f"train balance audit failed: {out}")
    if out["val_test_generated_rows"] != 0 or out["train_generated_donor_split_problems"] != 0:
        raise SystemExit(f"leakage audit failed: {out}")
    if not set(out["candidate_types"]).issubset(allowed):
        raise SystemExit(f"candidate type audit failed: {out}")
    if signals.shape != (1381, 1250, 12):
        raise SystemExit(f"signals shape audit failed: {signals.shape}")
    return out


def cmd_audit(paths: Paths) -> None:
    out = audit_dict(paths)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


def mmd_rbf(a: np.ndarray, b: np.ndarray) -> float:
    x = np.vstack([a, b])
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(med, 1e-6)
    kxx = np.exp(-gamma * ((a[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)).mean()
    kyy = np.exp(-gamma * ((b[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).mean()
    kxy = np.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def swd(a: np.ndarray, b: np.ndarray, seed: int = 0, n_proj: int = 96) -> float:
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    vals = []
    n = min(len(a), len(b))
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        pa = np.sort(a @ v)[:n]
        pb = np.sort(b @ v)[:n]
        vals.append(float(np.mean(np.abs(pa - pb))))
    return float(np.mean(vals))


def cdf_gap(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.percentile(a, [10, 25, 50, 75, 90], axis=0)
    qb = np.percentile(b, [10, 25, 50, 75, 90], axis=0)
    return float(np.mean(np.abs(qa - qb)))


def dual_auc(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    n = min(len(a), len(b))
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(a), size=n, replace=False)
    ib = rng.choice(len(b), size=n, replace=False)
    x = np.vstack([a[ia], b[ib]])
    y = np.r_[np.zeros(n), np.ones(n)]
    if len(np.unique(y)) < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=min(5, n), shuffle=True, random_state=seed)
    prob = np.zeros(len(y), dtype=np.float64)
    for tr, te in cv.split(x, y):
        scaler = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(x[tr]), y[tr])
        prob[te] = clf.predict_proba(scaler.transform(x[te]))[:, 1]
    return float(roc_auc_score(y, prob))


def fit_report(paths: Paths) -> dict[str, Any]:
    protocol = pd.read_csv(paths.protocol_csv)
    feat = pd.read_csv(paths.feature_csv)
    target = feat[(protocol["split"].eq("train")) & (protocol["generated"].eq(0)) & (protocol["y"].eq(0))]
    gen = feat[(protocol["split"].eq("train")) & (protocol["generated"].eq(1)) & (protocol["y"].eq(0))]
    cols = feature_cols(feat)
    target_x = np.nan_to_num(target[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    gen_x = np.nan_to_num(gen[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(target_x)
    a = scaler.transform(target_x)
    b = scaler.transform(gen_x)
    out = {
        "target_original_unacceptable_train_n": int(len(target)),
        "generated_unacceptable_train_n": int(len(gen)),
        "mmd_rbf": mmd_rbf(a, b),
        "swd": swd(a, b),
        "cdf_gap": cdf_gap(a, b),
        "dual_auc": dual_auc(a, b),
        "generated_candidate_counts": gen["candidate_type"].value_counts().to_dict(),
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "fit_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def cmd_plot(paths: Paths) -> None:
    ensure_dirs(paths)
    protocol = pd.read_csv(paths.protocol_csv)
    signals = np.load(paths.signals_npz, allow_pickle=True)["X"]
    feat = pd.read_csv(paths.feature_csv)
    report = fit_report(paths)

    rng = np.random.default_rng(2)
    groups = [
        ("acceptable", protocol[(protocol["y"].eq(1)) & (protocol["generated"].eq(0))]),
        ("unacceptable", protocol[(protocol["y"].eq(0)) & (protocol["generated"].eq(0)) & (protocol["split"].eq("train"))]),
        ("seta_native_morph", protocol[protocol["candidate_type"].eq("seta_native_morph")]),
        ("ptb12_morph", protocol[protocol["candidate_type"].eq("ptb12_morph")]),
        ("noise_style", protocol[protocol["candidate_type"].eq("noise_style")]),
    ]
    fig, axes = plt.subplots(len(groups), 5, figsize=(11, 7), sharex=True, sharey=False)
    t = np.arange(1250) / 125.0
    lead_idx = LEADS_12.index("II")
    for r, (name, df) in enumerate(groups):
        ids = df["row_idx"].to_numpy(dtype=int)
        if len(ids) == 0:
            continue
        pick = rng.choice(ids, size=min(5, len(ids)), replace=False)
        for c in range(5):
            ax = axes[r, c]
            if c < len(pick):
                y = signals[int(pick[c]), :, lead_idx]
                ax.plot(t, y, lw=0.55, color="#2b2b2b")
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            style_axes(ax)
    fig.tight_layout()
    fig.savefig(paths.figures / "raw_ecg_gallery.png", dpi=300)
    plt.close(fig)

    counts = protocol[protocol["generated"].eq(1)]["candidate_type"].value_counts().reindex(["seta_native_morph", "ptb12_morph", "noise_style"]).fillna(0)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#59A14F", "#E15759"])
    ax.set_ylabel("selected rows")
    ax.tick_params(axis="x", rotation=25)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(paths.figures / "candidate_composition.png", dpi=300)
    plt.close(fig)

    target = feat[(protocol["split"].eq("train")) & (protocol["generated"].eq(0)) & (protocol["y"].eq(0))]
    gen = feat[(protocol["split"].eq("train")) & (protocol["generated"].eq(1)) & (protocol["y"].eq(0))]
    key_cols = ["median_rms", "median_ptp", "median_diff95", "median_base_std"]
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.6))
    for ax, col in zip(axes.ravel(), key_cols):
        for df, label, color in [(target, "original", "#2b2b2b"), (gen, "generated", "#4C78A8")]:
            vals = df[col].to_numpy(dtype=float)
            vals = np.sort(vals[np.isfinite(vals)])
            yy = np.linspace(0, 1, len(vals), endpoint=False)
            ax.plot(vals, yy, lw=1.2, label=label, color=color)
        ax.set_xlabel(col)
        ax.set_ylabel("ECDF")
        style_axes(ax)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "feature_ecdf.png", dpi=300)
    plt.close(fig)

    cols = feature_cols(feat)
    target_x = np.nan_to_num(target[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    both_x = np.nan_to_num(pd.concat([target, gen])[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(target_x)
    x = scaler.transform(both_x)
    pca = PCA(n_components=2, random_state=0).fit_transform(x)
    n0 = len(target)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(pca[:n0, 0], pca[:n0, 1], s=14, alpha=0.55, color="#2b2b2b", label="original")
    ax.scatter(pca[n0:, 0], pca[n0:, 1], s=14, alpha=0.55, color="#4C78A8", label="generated")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(paths.figures / "pca_feature_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    vals = [report["mmd_rbf"], report["swd"], report["cdf_gap"], report["dual_auc"]]
    ax.bar(["MMD", "SWD", "CDF", "AUC"], vals, color="#4C78A8")
    ax.set_ylabel("value")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(paths.figures / "fit_metrics.png", dpi=300)
    plt.close(fig)
    print(json.dumps({"figures": str(paths.figures), "fit": report}, indent=2))


class ECGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, factors: np.ndarray, rows: pd.DataFrame) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.factors = torch.from_numpy(factors.astype(np.float32))
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx], "factor": self.factors[idx]}


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class ConformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.08) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width), nn.Dropout(dropout))
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(width)
        self.dwconv = nn.Sequential(nn.Conv1d(width, width, kernel_size=17, padding=8, groups=width), nn.GELU(), nn.Conv1d(width, width, kernel_size=1), nn.Dropout(dropout))
        self.ff2 = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        h = self.conv_norm(x).transpose(1, 2)
        x = x + self.dwconv(h).transpose(1, 2)
        return x + 0.5 * self.ff2(x)


class LeadWiseSharedConformer(nn.Module):
    query_names = ["QRS", "RR_TEMPLATE", "BASELINE", "CONTACT_RESET", "DETAIL_NOISE", "GLOBAL_MORPH", "BORDERLINE", "ARTIFACT"]

    def __init__(self, width: int, layers: int, heads: int, factor_dim: int) -> None:
        super().__init__()
        self.hi_stem = nn.Sequential(
            nn.Conv1d(8, width, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=7, padding=3),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
        )
        self.ctx_down = nn.Sequential(nn.Conv1d(width, width, kernel_size=9, stride=4, padding=4), nn.GroupNorm(max(1, min(8, width // 8)), width), nn.GELU())
        self.pos = PositionalEncoding(width)
        self.query = nn.Parameter(torch.randn(len(self.query_names), width) * 0.02)
        self.blocks = nn.Sequential(*[ConformerBlock(width, heads) for _ in range(layers)])
        self.hi_cross_attn = nn.MultiheadAttention(width, heads, dropout=0.05, batch_first=True)
        self.local_heads = nn.Conv1d(width, 4, kernel_size=1)
        self.factor_head = nn.Sequential(nn.LayerNorm(width * 6), nn.Linear(width * 6, 160), nn.GELU(), nn.Dropout(0.08), nn.Linear(160, factor_dim))
        self.head = nn.Sequential(nn.LayerNorm(width * 4), nn.Linear(width * 4, 96), nn.GELU(), nn.Dropout(0.08), nn.Linear(96, 2))

    @staticmethod
    def lead_views(x: torch.Tensor) -> torch.Tensor:
        b, leads, t = x.shape
        flat = x.reshape(b * leads, 1, t)
        diff = F.pad(flat[:, :, 1:] - flat[:, :, :-1], (1, 0))
        base = F.avg_pool1d(flat, kernel_size=125, stride=1, padding=62)
        smooth = F.avg_pool1d(flat, kernel_size=31, stride=1, padding=15)
        detail = flat - smooth
        views = torch.cat(
            [flat, diff, diff.abs(), base, detail, flat - base, flat.abs(), torch.tanh(flat * flat)],
            dim=1,
        )
        return views.reshape(b, leads, 8, t)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, leads, _ = x.shape
        v = self.lead_views(x).reshape(b * leads, 8, x.shape[-1])
        hi = self.hi_stem(v)
        hi_tokens = hi.transpose(1, 2)
        ctx = self.ctx_down(hi).transpose(1, 2)
        q = self.query[None, :, :].expand(b * leads, -1, -1)
        seq = self.blocks(self.pos(torch.cat([q, ctx], dim=1)))
        query = seq[:, : len(self.query_names), :]
        q_delta, _ = self.hi_cross_attn(query, hi_tokens, hi_tokens, need_weights=False)
        query = query + q_delta
        query = query.reshape(b, leads, len(self.query_names), -1)
        factor_query = query[:, :, :6, :].mean(dim=1)
        factor = self.factor_head(factor_query.reshape(b, -1))
        lead_repr = torch.cat([query[:, :, 6, :], query[:, :, 7, :]], dim=2)
        pooled = torch.cat([lead_repr.mean(dim=1), lead_repr.max(dim=1).values], dim=1)
        local_logits = self.local_heads(hi).reshape(b, leads, 4, -1)
        return {"logits": self.head(pooled), "local_logits": local_logits, "factor": factor}


def pseudo_local_targets(x: torch.Tensor, out_len: int) -> torch.Tensor:
    b, leads, t = x.shape
    lead = x.reshape(b * leads, t)
    diff = torch.abs(lead[:, 1:] - lead[:, :-1])
    diff = F.pad(diff, (1, 0))
    qrs = (diff - diff.mean(dim=1, keepdim=True)) / diff.std(dim=1, keepdim=True).clamp_min(1e-5)
    qrs = torch.sigmoid(qrs - 1.0)
    baseline = torch.abs(F.avg_pool1d(lead[:, None, :], kernel_size=125, stride=1, padding=62).squeeze(1))
    detail = torch.abs(lead - F.avg_pool1d(lead[:, None, :], kernel_size=31, stride=1, padding=15).squeeze(1))
    flat = (torch.abs(lead) < 0.03).float()
    maps = torch.stack([qrs, baseline, detail, flat], dim=1)
    return F.interpolate(maps, size=out_len, mode="linear", align_corners=False).reshape(b, leads, 4, out_len)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics_binary(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for t in np.linspace(0.0, 1.0, 1001):
        m = metrics_binary(y, prob, float(t))
        score = (m["macro_f1"], m["acc"])
        if best is None or score > (best["macro_f1"], best["acc"]):
            best = m
    if best is None:
        raise RuntimeError("no threshold candidates")
    return best


def factor_targets(feat: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cols = ["median_rms", "median_ptp", "median_diff95", "median_base_std", "median_high_rms", "lead_corr_mean", "amp_entropy"]
    x = np.nan_to_num(feat[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mu = x[train_mask].mean(axis=0, keepdims=True)
    sd = np.maximum(x[train_mask].std(axis=0, keepdims=True), 1e-6)
    return ((x - mu) / sd).astype(np.float32), {"columns": cols, "mean": mu.ravel().tolist(), "std": sd.ravel().tolist()}


def load_train_arrays(paths: Paths, normalization: dict[str, Any] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    protocol = pd.read_csv(paths.protocol_csv)
    feat = pd.read_csv(paths.feature_csv)
    X = np.load(paths.signals_npz, allow_pickle=True)["X"].astype(np.float32)
    train_mask = protocol["split"].eq("train").to_numpy()
    if normalization is None:
        mu = X[train_mask].mean(axis=(0, 1), keepdims=True)
        sd = np.maximum(X[train_mask].std(axis=(0, 1), keepdims=True), 1e-6)
    else:
        mu = np.asarray(normalization["mean_per_lead"], dtype=np.float32).reshape(1, 1, -1)
        sd = np.asarray(normalization["std_per_lead"], dtype=np.float32).reshape(1, 1, -1)
    Xn = ((X - mu) / sd).astype(np.float32).transpose(0, 2, 1)
    if normalization is None:
        factors, factor_stats = factor_targets(feat, train_mask)
    else:
        factor_stats = dict(normalization["factors"])
        cols = list(factor_stats["columns"])
        raw = np.nan_to_num(feat[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        f_mu = np.asarray(factor_stats["mean"], dtype=np.float32).reshape(1, -1)
        f_sd = np.maximum(np.asarray(factor_stats["std"], dtype=np.float32).reshape(1, -1), 1e-6)
        factors = ((raw - f_mu) / f_sd).astype(np.float32)
    norm = {"lead_order": LEADS_12, "mean_per_lead": mu.reshape(-1).tolist(), "std_per_lead": sd.reshape(-1).tolist(), "factors": factor_stats}
    return protocol, Xn, protocol["y"].astype(int).to_numpy(), factors, norm


def make_loaders(protocol: pd.DataFrame, X: np.ndarray, y: np.ndarray, factors: np.ndarray, cfg: TrainConfig) -> tuple[dict[str, DataLoader], dict[str, ECGDataset]]:
    loaders: dict[str, DataLoader] = {}
    datasets: dict[str, ECGDataset] = {}
    pin = torch.cuda.is_available()
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        idx = np.flatnonzero(protocol["split"].astype(str).eq(split).to_numpy())
        ds = ECGDataset(X[idx], y[idx], factors[idx], protocol.iloc[idx])
        datasets[split] = ds
        loaders[split] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=pin)
    return loaders, datasets


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)
        prob = torch.softmax(out["logits"], dim=1)[:, 1]
        ys.append(batch["y"].cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _load_torch(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _train_config_from_checkpoint(values: dict[str, Any], *, device: str) -> TrainConfig:
    cfg = asdict(TrainConfig())
    cfg.update({k: values[k] for k in cfg if k in values})
    cfg["device"] = device
    cfg["num_workers"] = 0
    return TrainConfig(**cfg)


def _write_pretrained_eval_outputs(
    paths: Paths,
    *,
    model: nn.Module,
    cfg: TrainConfig,
    protocol: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    factors: np.ndarray,
    norm: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
) -> None:
    loaders, datasets = make_loaders(protocol, X, y, factors, cfg)
    split_metrics: dict[str, Any] = {}
    split_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pred_by_split: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        eval_loader = DataLoader(
            datasets[split],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        yy, pp = predict(model, eval_loader, device)
        split_probs[split] = (yy, pp)
        split_metrics[split] = metrics_binary(yy, pp)
        sub = datasets[split].rows.copy()
        sub["prob_acceptable"] = pp
        sub["pred_fixed"] = (pp >= 0.5).astype(int)
        pred_by_split[split] = sub
    val_threshold = best_threshold(*split_probs["val"])
    test_at_val_threshold = metrics_binary(split_probs["test"][0], split_probs["test"][1], val_threshold["threshold"])
    pred_rows: list[pd.DataFrame] = []
    for split, sub in pred_by_split.items():
        sub = sub.copy()
        sub["pred_val_threshold"] = (sub["prob_acceptable"].to_numpy() >= val_threshold["threshold"]).astype(int)
        pred_rows.append(sub)
    metrics = {
        "task": "binary acceptable(1) vs unacceptable(0)",
        "config": asdict(cfg),
        "device": str(device),
        "n": {k: int(len(v)) for k, v in datasets.items()},
        "normalization": norm,
        "fixed_threshold_metrics": split_metrics,
        "val_selected_threshold": val_threshold,
        "test_at_val_selected_threshold": test_at_val_threshold,
        "checkpoint_source": str(checkpoint_dir),
        "pretrained_inference": True,
    }
    paths.models.mkdir(parents=True, exist_ok=True)
    history_src = checkpoint_dir / "history.csv"
    if history_src.exists():
        shutil.copy2(history_src, paths.models / "history.csv")
    else:
        pd.DataFrame().to_csv(paths.models / "history.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(paths.models / "predictions.csv", index=False)
    shutil.copy2(checkpoint_dir / "best_model.pt", paths.models / "best_model.pt")
    (paths.models / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "# SQI12 Gap-Fill E31 Lead-Wise Shared",
        "",
        f"- pretrained checkpoint: {checkpoint_dir}",
        f"- train/val/test: {metrics['n']['train']} / {metrics['n']['val']} / {metrics['n']['test']}",
        f"- test acc: {split_metrics['test']['acc']:.4f}",
        f"- test macro F1: {split_metrics['test']['macro_f1']:.4f}",
        f"- test AUC: {split_metrics['test']['auc']:.4f}",
        f"- sensitivity: {split_metrics['test']['sensitivity']:.4f}",
        f"- specificity: {split_metrics['test']['specificity']:.4f}",
        f"- val-selected threshold: {val_threshold['threshold']:.3f}",
        f"- test acc at val threshold: {test_at_val_threshold['acc']:.4f}",
        f"- test macro F1 at val threshold: {test_at_val_threshold['macro_f1']:.4f}",
    ]
    (paths.models / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(paths.models), "pretrained": str(checkpoint_dir), "test": split_metrics["test"]}, indent=2))


def cmd_predict_from_checkpoint(paths: Paths, *, run: bool, checkpoint_dir: Path, device: str) -> None:
    if not run:
        print(
            json.dumps(
                {
                    "step": "predict-from-checkpoint",
                    "checkpoint_dir": str(checkpoint_dir),
                    "out_dir": str(paths.models),
                },
                indent=2,
            )
        )
        return
    checkpoint_dir = checkpoint_dir.resolve()
    ckpt_path = checkpoint_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing pretrained E31 checkpoint: {ckpt_path}")
    ensure_dirs(paths)
    torch_device = torch.device("cuda" if str(device).lower() == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt = _load_torch(ckpt_path, torch_device)
    cfg = _train_config_from_checkpoint(dict(ckpt.get("config", {})), device=str(torch_device))
    protocol, X, y, factors, norm = load_train_arrays(paths, ckpt.get("normalization"))
    model = LeadWiseSharedConformer(cfg.width, cfg.layers, cfg.heads, factors.shape[1]).to(torch_device)
    state = ckpt.get("model_state_dict", ckpt.get("model_state"))
    if state is None:
        raise KeyError(f"checkpoint has no model_state_dict/model_state: {ckpt_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    _write_pretrained_eval_outputs(
        paths,
        model=model,
        cfg=cfg,
        protocol=protocol,
        X=X,
        y=y,
        factors=factors,
        norm=norm,
        device=torch_device,
        checkpoint_dir=checkpoint_dir,
    )


def cmd_train(paths: Paths, *, run: bool, cfg: TrainConfig) -> None:
    if not run:
        print(subprocess.list2cmdline([sys.executable, "-m", "src.supplemental_transformer_experiments.sqi12_gapfill.run", "train", "--run"]))
        dry("train", paths)
        return
    ensure_dirs(paths)
    seed_everything(cfg.seed)
    protocol, X, y, factors, norm = load_train_arrays(paths)
    loaders, datasets = make_loaders(protocol, X, y, factors, cfg)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device if cfg.device != "auto" else "cpu")
    model = LeadWiseSharedConformer(cfg.width, cfg.layers, cfg.heads, factors.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best = {"epoch": 0, "val_macro_f1": -1.0, "val_acc": -1.0, "val_auc": -1.0}
    stale = 0
    history: list[dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        accum = max(1, int(cfg.grad_accum_steps))
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(loaders["train"], start=1):
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            fb = batch["factor"].to(device)
            out = model(xb)
            class_weight = torch.as_tensor([cfg.bad_class_weight, 1.0], device=device, dtype=out["logits"].dtype)
            ce = F.cross_entropy(out["logits"], yb, weight=class_weight)
            fl = F.smooth_l1_loss(out["factor"], fb, beta=0.35)
            local_target = pseudo_local_targets(xb, out["local_logits"].shape[-1])
            ll = F.smooth_l1_loss(torch.tanh(out["local_logits"]), local_target)
            loss = ce + cfg.factor_weight * fl + cfg.local_weight * ll
            (loss / accum).backward()
            if bi % accum == 0 or bi == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        yv, pv = predict(model, loaders["val"], device)
        vm = metrics_binary(yv, pv)
        vm_selected = best_threshold(yv, pv)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in vm.items() if k != "confusion_matrix"},
            **{f"val_selected_{k}": v for k, v in vm_selected.items() if k != "confusion_matrix"},
        }
        history.append(row)
        fixed_ok = vm["acc"] >= 0.75
        score = (vm["macro_f1"], vm["acc"], vm["auc"]) if fixed_ok else (-1.0, -1.0, -1.0)
        improved = score > (best["val_macro_f1"], best["val_acc"], best["val_auc"])
        if improved:
            best = {"epoch": epoch, "val_macro_f1": vm["macro_f1"], "val_acc": vm["acc"], "val_auc": vm["auc"]}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_acc={vm['acc']:.4f} val_auc={vm['auc']:.4f} "
            f"val_sel_f1={vm_selected['macro_f1']:.4f}",
            flush=True,
        )
        if stale >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    split_metrics: dict[str, Any] = {}
    split_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pred_by_split: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        eval_loader = DataLoader(
            datasets[split],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        yy, pp = predict(model, eval_loader, device)
        split_probs[split] = (yy, pp)
        split_metrics[split] = metrics_binary(yy, pp)
        sub = datasets[split].rows.copy()
        sub["prob_acceptable"] = pp
        sub["pred_fixed"] = (pp >= 0.5).astype(int)
        pred_by_split[split] = sub
    val_threshold = best_threshold(*split_probs["val"])
    test_at_val_threshold = metrics_binary(split_probs["test"][0], split_probs["test"][1], val_threshold["threshold"])
    pred_rows: list[pd.DataFrame] = []
    for split, sub in pred_by_split.items():
        sub = sub.copy()
        sub["pred_val_threshold"] = (sub["prob_acceptable"].to_numpy() >= val_threshold["threshold"]).astype(int)
        pred_rows.append(sub)
    metrics = {
        "task": "binary acceptable(1) vs unacceptable(0)",
        "config": asdict(cfg),
        "device": str(device),
        "n": {k: int(len(v)) for k, v in datasets.items()},
        "best_epoch": int(best["epoch"]),
        "train_time_s": float(time.time() - t0),
        "normalization": norm,
        "fixed_threshold_metrics": split_metrics,
        "val_selected_threshold": val_threshold,
        "test_at_val_selected_threshold": test_at_val_threshold,
    }
    paths.models.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(paths.models / "history.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(paths.models / "predictions.csv", index=False)
    torch.save({"model_state_dict": best_state, "config": asdict(cfg), "normalization": norm}, paths.models / "best_model.pt")
    (paths.models / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "# SQI12 Gap-Fill E31 Lead-Wise Shared",
        "",
        f"- train/val/test: {metrics['n']['train']} / {metrics['n']['val']} / {metrics['n']['test']}",
        f"- best epoch: {metrics['best_epoch']}",
        f"- test acc: {split_metrics['test']['acc']:.4f}",
        f"- test macro F1: {split_metrics['test']['macro_f1']:.4f}",
        f"- test AUC: {split_metrics['test']['auc']:.4f}",
        f"- sensitivity: {split_metrics['test']['sensitivity']:.4f}",
        f"- specificity: {split_metrics['test']['specificity']:.4f}",
        f"- val-selected threshold: {val_threshold['threshold']:.3f}",
        f"- test acc at val threshold: {test_at_val_threshold['acc']:.4f}",
        f"- test macro F1 at val threshold: {test_at_val_threshold['macro_f1']:.4f}",
    ]
    (paths.models / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(paths.models), "test": split_metrics["test"]}, indent=2))


def macro_f1_from_cm(tn: int, fp: int, fn: int, tp: int) -> float:
    f1_bad = 2 * tn / max(2 * tn + fp + fn, 1)
    f1_good = 2 * tp / max(2 * tp + fp + fn, 1)
    return float((f1_bad + f1_good) / 2)


def cmd_sqi_baselines(paths: Paths, *, run: bool, force: bool, device: str) -> None:
    features = paths.out / "ours_sqi_features" / "features" / "record84_norm.parquet"
    split_csv = paths.out / "ours_sqi_features" / "splits" / "split_ours_gapfill_for_sqi.csv"
    out_dir = paths.out / "baselines"
    if not run:
        print(
            json.dumps(
                {
                    "step": "sqi-baselines",
                    "features": str(features),
                    "split": str(split_csv),
                    "out_dir": str(out_dir),
                },
                indent=2,
            )
        )
        return
    if not features.exists():
        raise SystemExit(f"Missing gap-fill SQI feature table: {features}")
    if not split_csv.exists():
        raise SystemExit(f"Missing gap-fill SQI split: {split_csv}")
    ensure_dirs(paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.sqi_pipeline.models import lm_mlp_search, svm_tables

    svm_out = out_dir / "svm_mainline_strict"
    mlp_out = out_dir / "lm_mlp_mainline_strict"
    svm_tables.run(
        {
            "verbose": False,
            "force": force,
            "seed": 0,
            "features_parquet": str(features),
            "split_csv": str(split_csv),
            "out_dir": str(svm_out),
            "paper_mode": True,
            "threshold_mode": "val_maxacc",
            "final_trainval": False,
            "select_metric": "val_auc",
            "use_standard_scaler": False,
            "log_each": False,
        }
    )
    lm_mlp_search.run(
        {
            "verbose": False,
            "force": force,
            "seed": 0,
            "device": device,
            "dtype": "float64",
            "features_parquet": str(features),
            "split_csv": str(split_csv),
            "out_dir": str(mlp_out),
            "tables": False,
            "threshold_mode": "val_maxacc",
            "final_trainval": False,
            "model_select_metric": "val_acc",
            "final_patience": 15,
        }
    )

    split = pd.read_csv(split_csv)
    split_counts = split.groupby(["split", "quality_record"]).size().unstack(fill_value=0).to_dict()
    table6 = pd.read_csv(svm_out / "table6_12lead_combo_sqi_seed0.csv")
    table7 = pd.read_csv(svm_out / "table7_svm_selected5_seed0.csv")
    svm_all = table6.loc[table6["Group"].eq("All SQI")].iloc[0].to_dict()
    svm_selected5 = table7.iloc[0].to_dict()
    mlp = json.loads((mlp_out / "lm_mlp_test_metrics_seed0.json").read_text(encoding="utf-8"))
    cm = mlp["test_metrics_fixed"]["confusion_matrix"]
    mlp["test_metrics_fixed"]["macro_f1"] = macro_f1_from_cm(cm["tn"], cm["fp"], cm["fn"], cm["tp"])

    summary = {
        "task": "SQI scalar baselines on the SQI12 gap-fill train split",
        "note": "These are SQI-feature comparators. The E31-style model still uses waveform only.",
        "features": str(features),
        "split": str(split_csv),
        "split_counts": split_counts,
        "svm_all_sqi": {
            "acc": float(svm_all["Ac_test"]),
            "auc": float(svm_all["AUC_test"]),
            "threshold": float(svm_all["threshold"]),
            "test_oracle_max_acc": float(svm_all["maxAcc_test"]),
        },
        "svm_selected5": {
            "acc": float(svm_selected5["Ac_test"]),
            "auc": float(svm_selected5["AUC_test"]),
            "threshold": float(svm_selected5["threshold"]),
            "test_oracle_max_acc": float(svm_selected5["maxAcc_test"]),
        },
        "lm_mlp_84_j_1": {
            "J": int(mlp["J"]),
            "acc": float(mlp["test_metrics_fixed"]["acc"]),
            "macro_f1": float(mlp["test_metrics_fixed"]["macro_f1"]),
            "auc": float(mlp["test_metrics_fixed"]["auc"]),
            "threshold": float(mlp["threshold"]),
            "confusion_matrix": mlp["test_metrics_fixed"]["confusion_matrix"],
        },
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "sqi_baseline_strict_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# SQI12 Strict SQI Baselines",
        "",
        "Comparators use the SQI 84-feature table and this experiment's original-only split.",
        "",
        f"- SVM All SQI: acc {summary['svm_all_sqi']['acc']:.4f}, AUC {summary['svm_all_sqi']['auc']:.4f}",
        f"- SVM selected 5: acc {summary['svm_selected5']['acc']:.4f}, AUC {summary['svm_selected5']['auc']:.4f}",
        f"- LM-MLP 84-{summary['lm_mlp_84_j_1']['J']}-1: acc {summary['lm_mlp_84_j_1']['acc']:.4f}, macro F1 {summary['lm_mlp_84_j_1']['macro_f1']:.4f}, AUC {summary['lm_mlp_84_j_1']['auc']:.4f}",
    ]
    (paths.reports / "sqi_baseline_strict_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_pipeline(paths: Paths, args: argparse.Namespace) -> None:
    if not args.run:
        dry("pipeline", paths)
        print(subprocess.list2cmdline([sys.executable, "-m", "src.supplemental_transformer_experiments.sqi12_gapfill.run", "pipeline", "--run", "--train"]))
        return
    cmd_manifest(paths, run=True, force=args.force)
    cmd_split(paths, run=True, force=args.force)
    cmd_build(paths, run=True, force=args.force, seed=args.seed, max_ptb=args.max_ptb)
    cmd_audit(paths)
    cmd_plot(paths)
    if args.train:
        cmd_train(paths, run=True, cfg=TrainConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, seed=args.seed, device=args.device))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolated SQI Set-A 12-lead gap-fill Transformer experiment.")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["manifest", "split", "build", "train"]:
        sp = sub.add_parser(name)
        sp.add_argument("--run", action="store_true")
    sub.add_parser("audit")
    sub.add_parser("plot")
    baselines = sub.add_parser("sqi-baselines")
    baselines.add_argument("--run", action="store_true")
    baselines.add_argument("--device", default="cuda")
    pipe = sub.add_parser("pipeline")
    pipe.add_argument("--run", action="store_true")
    pipe.add_argument("--train", action="store_true")
    for sp_name in ["build", "pipeline"]:
        sp = sub.choices[sp_name]
        sp.add_argument("--max-ptb", type=int, default=2400)
    for sp_name in ["train", "pipeline"]:
        sp = sub.choices[sp_name]
        sp.add_argument("--epochs", type=int, default=45)
        sp.add_argument("--patience", type=int, default=45)
        sp.add_argument("--batch-size", type=int, default=24)
        sp.add_argument("--grad-accum-steps", type=int, default=1)
        sp.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out)
    if getattr(args, "run", False):
        ensure_sqi_raw_data(ROOT)
        ensure_ptbxl_raw_data(ROOT)
    if args.cmd == "manifest":
        cmd_manifest(paths, run=args.run, force=args.force)
    elif args.cmd == "split":
        cmd_split(paths, run=args.run, force=args.force)
    elif args.cmd == "build":
        cmd_build(paths, run=args.run, force=args.force, seed=args.seed, max_ptb=args.max_ptb)
    elif args.cmd == "audit":
        cmd_audit(paths)
    elif args.cmd == "plot":
        cmd_plot(paths)
    elif args.cmd == "train":
        cmd_train(paths, run=args.run, cfg=TrainConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, seed=args.seed, device=args.device))
    elif args.cmd == "sqi-baselines":
        cmd_sqi_baselines(paths, run=args.run, force=args.force, device=args.device)
    elif args.cmd == "pipeline":
        cmd_pipeline(paths, args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
