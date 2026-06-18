"""Run external BUT QDB and CinC2017 checks for the E3.11f Uformer model.

This module is intentionally standalone.  It keeps real-data benchmarks out of
the thesis/mainline trainer and leaves ``src/sqi_pipeline`` untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import time
import urllib.request
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from scipy.io import loadmat
from scipy.signal import resample_poly, welch
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

from src.transformer_pipeline.models.uformer1d import (
    Uformer1DConfig,
    UformerDenoiseSQIModel,
    feature_vector,
)


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
RUN_TAG = "e311_realdata_2026_06_02"
FS_TARGET = 125
WIN_SEC = 10
N_TARGET = FS_TARGET * WIN_SEC
PTB_RMS_TARGET = 0.1616995930671692
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

DEFAULT_DATA_ROOT = ROOT / "data" / "external"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_CKPT = ROOT / "outputs" / "mainline" / "e311_uformer_full_tokens_detach_seed0" / "ckpt_best.pt"
BUT_BASE_URL = "https://physionet.org/files/butqdb/1.0.0"
CINC_BASE_URL = "https://physionet.org/files/challenge-2017/1.0.0"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(path)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def robust_normalize_window(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(x, dtype=np.float64).reshape(-1)
    centered = raw - float(np.median(raw))
    rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
    p05, p95 = np.percentile(centered, [5.0, 95.0])
    robust = float((p95 - p05) / 3.29) if p95 > p05 else rms
    scale = max(robust, rms * 0.5, 1e-6)
    y = centered * (PTB_RMS_TARGET / scale)
    return y.astype(np.float32), {
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "raw_rms_centered": rms,
        "raw_p05": float(p05),
        "raw_p95": float(p95),
        "normalization_scale": float(scale),
    }


def resample_to_target(x: np.ndarray, fs: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if fs == FS_TARGET:
        return arr.astype(np.float32)
    g = math.gcd(int(fs), FS_TARGET)
    return resample_poly(arr, up=FS_TARGET // g, down=int(fs) // g).astype(np.float32)


def pad_or_trim_10s(x: np.ndarray, fs: int) -> tuple[np.ndarray, bool]:
    target = int(fs * WIN_SEC)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    padded = False
    if len(arr) < target:
        pad = target - len(arr)
        mode = "reflect" if len(arr) > 2 else "edge"
        arr = np.pad(arr, (0, pad), mode=mode)
        padded = True
    elif len(arr) > target:
        arr = arr[:target]
    return arr.astype(np.float32), padded


def deterministic_subject_split(subjects: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    values = np.asarray(sorted(set(subjects)), dtype=object)
    rng.shuffle(values)
    n = len(values)
    n_train = max(1, int(round(n * 0.60)))
    n_val = max(1, int(round(n * 0.20))) if n >= 3 else 0
    split: dict[str, str] = {}
    for i, subject in enumerate(values.tolist()):
        if i < n_train:
            split[str(subject)] = "train"
        elif i < n_train + n_val:
            split[str(subject)] = "val"
        else:
            split[str(subject)] = "test"
    return split


def balanced_but_subject_split(meta: pd.DataFrame, seed: int, attempts: int = 20000) -> tuple[dict[str, str], dict[str, Any]]:
    """Choose a subject-grouped split that still covers all classes.

    BUT QDB has very uneven class distribution by subject.  A plain random
    subject split can easily put zero bad windows in test, which makes bad
    recall meaningless.  This keeps the no-leakage subject boundary and searches
    deterministic candidate assignments for better class coverage.
    """

    subjects = sorted(meta["subject_id"].astype(str).unique().tolist())
    counts = (
        meta.groupby(["subject_id", "y_class"]).size().unstack(fill_value=0).reindex(subjects).fillna(0).astype(int)
    )
    for cls in ["good", "medium", "bad"]:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[["good", "medium", "bad"]]
    total = counts.sum(axis=0).to_numpy(dtype=np.float64)
    n = len(subjects)
    n_train = max(1, int(round(0.60 * n)))
    n_val = max(1, int(round(0.20 * n)))
    sizes = {"train": n_train, "val": n_val, "test": n - n_train - n_val}
    targets = {
        "train": 0.60 * total,
        "val": 0.20 * total,
        "test": 0.20 * total,
    }
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, str], dict[str, Any]] | None = None
    for attempt in range(attempts):
        order = subjects.copy()
        rng.shuffle(order)
        assign: dict[str, str] = {}
        for subject in order[: sizes["train"]]:
            assign[str(subject)] = "train"
        for subject in order[sizes["train"] : sizes["train"] + sizes["val"]]:
            assign[str(subject)] = "val"
        for subject in order[sizes["train"] + sizes["val"] :]:
            assign[str(subject)] = "test"
        split_counts_by_class: dict[str, dict[str, int]] = {}
        missing_penalty = 0.0
        balance_penalty = 0.0
        for split_name in ["train", "val", "test"]:
            split_subjects = [s for s in subjects if assign[s] == split_name]
            vals = counts.loc[split_subjects].sum(axis=0)
            split_counts_by_class[split_name] = {str(k): int(v) for k, v in vals.to_dict().items()}
            if (vals <= 0).any():
                missing_penalty += 1000.0 * float((vals <= 0).sum())
            balance_penalty += float(np.sum(np.abs(vals.to_numpy(dtype=np.float64) - targets[split_name]) / (total + 1e-6)))
        subject_penalty = abs(sum(1 for s in subjects if assign[s] == "train") - sizes["train"])
        score = missing_penalty + balance_penalty + 0.01 * subject_penalty
        audit = {
            "attempt": int(attempt),
            "split_subjects": {
                sp: [s for s in subjects if assign[s] == sp]
                for sp in ["train", "val", "test"]
            },
            "split_label_counts": split_counts_by_class,
            "score": float(score),
            "missing_class_penalty": float(missing_penalty),
        }
        if best is None or score < best[0]:
            best = (score, assign, audit)
            if missing_penalty == 0.0 and balance_penalty < 1.0:
                break
    assert best is not None
    return best[1], best[2]


def stratified_record_split(labels: pd.Series, seed: int) -> dict[str, str]:
    record_ids = np.asarray(labels.index.tolist(), dtype=object)
    y = labels.to_numpy()
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        record_ids,
        y,
        test_size=0.40,
        random_state=seed,
        stratify=y,
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=seed + 1,
        stratify=y_temp,
    )
    out = {str(r): "train" for r in train_ids}
    out.update({str(r): "val" for r in val_ids})
    out.update({str(r): "test" for r in test_ids})
    return out


def split_counts(meta: pd.DataFrame, label_col: str) -> dict[str, Any]:
    return {
        "split_counts": {str(k): int(v) for k, v in meta["split"].value_counts().sort_index().to_dict().items()},
        "label_counts": {str(k): int(v) for k, v in meta[label_col].value_counts().sort_index().to_dict().items()},
        "split_label_counts": {
            str(split): {str(k): int(v) for k, v in group[label_col].value_counts().sort_index().to_dict().items()}
            for split, group in meta.groupby("split")
        },
    }


def download_butqdb(data_root: Path, force: bool = False) -> dict[str, Any]:
    out_dir = data_root / "butqdb_1_0_0"
    marker = out_dir / ".download_complete.json"
    if marker.exists() and not force:
        return read_json(marker)
    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("butqdb", str(out_dir), keep_subdirs=True, overwrite=force)
    records = sorted({Path(r).parts[0] for r in wfdb.get_record_list("butqdb")})
    for name in ["RECORDS", "LICENSE.txt", "SHA256SUMS.txt", "ANNOTATORS", "subject-info.csv", "ann_reader.m"]:
        download_file(f"{BUT_BASE_URL}/{name}", out_dir / name)
    for record_id in records:
        download_file(f"{BUT_BASE_URL}/{record_id}/{record_id}_ANN.csv", out_dir / record_id / f"{record_id}_ANN.csv")
    payload = {
        "dataset": "butqdb",
        "physionet_url": f"{BUT_BASE_URL}/",
        "record_count": len(records),
        "download_sec": float(time.perf_counter() - start),
        "out_dir": str(out_dir),
    }
    write_json(marker, payload)
    return payload


def download_cinc2017(data_root: Path, force: bool = False) -> dict[str, Any]:
    out_dir = data_root / "cinc2017_1_0_0"
    marker = out_dir / ".download_complete.json"
    if marker.exists() and not force:
        return read_json(marker)
    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = out_dir / "REFERENCE-v3.csv"
    zip_path = out_dir / "training2017.zip"
    download_file(f"{CINC_BASE_URL}/REFERENCE-v3.csv", ref)
    download_file(f"{CINC_BASE_URL}/training2017.zip", zip_path)
    extract_dir = out_dir / "training2017"
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    payload = {
        "dataset": "cinc2017",
        "physionet_url": f"{CINC_BASE_URL}/",
        "reference_sha256": sha256_file(ref),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "download_sec": float(time.perf_counter() - start),
        "out_dir": str(out_dir),
    }
    write_json(marker, payload)
    return payload


def parse_but_ann(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(path, header=None)
    numeric = ann.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(how="all")
    if numeric.shape[1] < 12:
        raise ValueError(f"BUT annotation should have at least 12 columns: {path}")
    cons = numeric.iloc[:, -3:].copy()
    cons.columns = ["start_sample", "end_sample", "quality_class"]
    cons = cons.dropna()
    cons = cons[(cons["quality_class"] >= 1) & (cons["quality_class"] <= 3)]
    return cons.astype({"start_sample": int, "end_sample": int, "quality_class": int})


def preprocess_butqdb(data_root: Path, out_root: Path, seed: int, force: bool = False) -> dict[str, Any]:
    raw_dir = data_root / "butqdb_1_0_0"
    out_dir = out_root / "processed" / "butqdb"
    signals_path = out_dir / "signals.npz"
    meta_path = out_dir / "metadata.csv"
    audit_path = out_dir / "preprocess_audit.json"
    if signals_path.exists() and meta_path.exists() and audit_path.exists() and not force:
        return read_json(audit_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir() and (p / f"{p.name}_ECG.hea").exists()])
    rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    dropped = {"boundary_or_short": 0, "read_error": 0}
    for record_dir in record_dirs:
        record_id = record_dir.name
        subject_id = record_id[:3]
        ann_path = record_dir / f"{record_id}_ANN.csv"
        if not ann_path.exists():
            continue
        header = wfdb.rdheader(str(record_dir / f"{record_id}_ECG"))
        fs_raw = int(round(float(header.fs)))
        win_raw = fs_raw * WIN_SEC
        ann = parse_but_ann(ann_path)
        for seg_i, seg in ann.iterrows():
            start = int(seg.start_sample)
            end = int(seg.end_sample)
            q = int(seg.quality_class)
            if end - start < win_raw:
                dropped["boundary_or_short"] += 1
                continue
            n_windows = (end - start) // win_raw
            for k in range(n_windows):
                s0 = start + k * win_raw
                s1 = s0 + win_raw
                try:
                    rec = wfdb.rdrecord(str(record_dir / f"{record_id}_ECG"), sampfrom=s0, sampto=s1, physical=True, channels=[0])
                except Exception:
                    dropped["read_error"] += 1
                    continue
                raw = np.asarray(rec.p_signal[:, 0], dtype=np.float32)
                norm, stats = robust_normalize_window(resample_to_target(raw, fs_raw))
                if len(norm) != N_TARGET:
                    norm = pad_or_trim_10s(norm, FS_TARGET)[0]
                rows.append(norm.reshape(1, -1))
                label = {1: "good", 2: "medium", 3: "bad"}[q]
                meta_rows.append(
                    {
                        "window_id": f"butqdb_{record_id}_{s0}_{s1}",
                        "dataset": "butqdb",
                        "record_id": record_id,
                        "subject_id": subject_id,
                        "split": "pending",
                        "start_sec": float(s0 / fs_raw),
                        "end_sec": float(s1 / fs_raw),
                        "fs_original": fs_raw,
                        "label_raw": int(q),
                        "y_class": label,
                        "y": CLASS_TO_INT[label],
                        "source_label_policy": "BUT consensus class 1/2/3 mapped to good/medium/bad",
                        "padded": False,
                        **stats,
                    }
                )
    if not rows:
        raise RuntimeError("No BUT QDB windows were generated; check download and annotation parsing.")
    X = np.stack(rows, axis=0).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    subject_split, split_strategy = balanced_but_subject_split(meta, seed)
    meta["split"] = meta["subject_id"].astype(str).map(subject_split)
    np.savez_compressed(signals_path, X=X)
    meta.to_csv(meta_path, index=False)
    audit = {
        "dataset": "butqdb",
        "signals_shape": list(X.shape),
        "source": "PhysioNet BUT QDB v1.0.0 expert consensus annotations",
        "dropped": dropped,
        "subject_split": subject_split,
        "subject_split_strategy": split_strategy,
        **split_counts(meta, "y_class"),
        "raw_rms_by_class": describe_group(meta, "y_class", "raw_rms_centered"),
    }
    write_json(audit_path, audit)
    export_processed_visuals("butqdb", X, meta, out_dir / "visuals")
    return audit


def preprocess_cinc2017(data_root: Path, out_root: Path, seed: int, force: bool = False) -> dict[str, Any]:
    raw_dir = data_root / "cinc2017_1_0_0"
    out_dir = out_root / "processed" / "cinc2017"
    signals_path = out_dir / "signals.npz"
    meta_path = out_dir / "metadata.csv"
    audit_path = out_dir / "preprocess_audit.json"
    if signals_path.exists() and meta_path.exists() and audit_path.exists() and not force:
        return read_json(audit_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(raw_dir / "REFERENCE-v3.csv", header=None, names=["record_id", "label_raw"])
    record_labels = ref.set_index("record_id")["label_raw"]
    split_map = stratified_record_split(record_labels, seed)
    mat_root = raw_dir / "training2017"
    if not mat_root.exists():
        mat_root = raw_dir
    rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    dropped = {"missing_mat": 0, "read_error": 0}
    fs_raw = 300
    win_raw = fs_raw * WIN_SEC
    for rec in ref.itertuples(index=False):
        record_id = str(rec.record_id)
        label_raw = str(rec.label_raw)
        mat_path = mat_root / f"{record_id}.mat"
        if not mat_path.exists():
            matches = list(raw_dir.rglob(f"{record_id}.mat"))
            mat_path = matches[0] if matches else mat_path
        if not mat_path.exists():
            dropped["missing_mat"] += 1
            continue
        try:
            mat = loadmat(mat_path)
            signal = np.asarray(mat["val"]).reshape(-1).astype(np.float32)
        except Exception:
            dropped["read_error"] += 1
            continue
        if len(signal) < win_raw:
            chunks = [(0, len(signal), pad_or_trim_10s(signal, fs_raw)[0], True)]
        else:
            chunks = []
            n_windows = max(1, len(signal) // win_raw)
            for k in range(n_windows):
                s0 = k * win_raw
                s1 = s0 + win_raw
                chunks.append((s0, s1, signal[s0:s1], False))
        is_noisy = label_raw == "~"
        for k, (s0, s1, raw_chunk, padded) in enumerate(chunks):
            norm, stats = robust_normalize_window(resample_to_target(raw_chunk, fs_raw))
            if len(norm) != N_TARGET:
                norm = pad_or_trim_10s(norm, FS_TARGET)[0]
            rows.append(norm.reshape(1, -1))
            meta_rows.append(
                {
                    "window_id": f"cinc2017_{record_id}_{k}",
                    "dataset": "cinc2017",
                    "record_id": record_id,
                    "subject_id": record_id,
                    "split": split_map[record_id],
                    "start_sec": float(s0 / fs_raw),
                    "end_sec": float(s1 / fs_raw),
                    "fs_original": fs_raw,
                    "label_raw": label_raw,
                    "is_noisy": int(is_noisy),
                    "y_binary": int(is_noisy),
                    "source_label_policy": "CinC2017 '~' mapped to noisy/bad; N/A/O mapped to usable",
                    "padded": bool(padded),
                    **stats,
                }
            )
    if not rows:
        raise RuntimeError("No CinC2017 windows were generated; check download/extraction.")
    X = np.stack(rows, axis=0).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    np.savez_compressed(signals_path, X=X)
    meta.to_csv(meta_path, index=False)
    audit = {
        "dataset": "cinc2017",
        "signals_shape": list(X.shape),
        "source": "PhysioNet/CinC 2017 training labels; '~' is noisy",
        "dropped": dropped,
        "record_label_counts": {str(k): int(v) for k, v in ref["label_raw"].value_counts().sort_index().to_dict().items()},
        **split_counts(meta, "label_raw"),
        "window_noisy_counts": {str(k): int(v) for k, v in meta["is_noisy"].value_counts().sort_index().to_dict().items()},
        "padded_windows": int(meta["padded"].sum()),
    }
    write_json(audit_path, audit)
    export_processed_visuals("cinc2017", X, meta, out_dir / "visuals")
    return audit


def describe_group(meta: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, group in meta.groupby(group_col):
        vals = group[value_col].to_numpy(dtype=np.float64)
        out[str(key)] = {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)) if len(vals) else 0.0,
            "std": float(np.std(vals)) if len(vals) else 0.0,
            "p10": float(np.percentile(vals, 10)) if len(vals) else 0.0,
            "p50": float(np.percentile(vals, 50)) if len(vals) else 0.0,
            "p90": float(np.percentile(vals, 90)) if len(vals) else 0.0,
        }
    return out


def export_processed_visuals(dataset: str, X: np.ndarray, meta: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "butqdb":
        label_col = "y_class"
        groups = ["good", "medium", "bad"]
    else:
        label_col = "label_raw"
        groups = sorted(meta[label_col].astype(str).unique().tolist())
    for group in groups:
        idx = meta.index[meta[label_col].astype(str) == str(group)].to_numpy()
        if len(idx) == 0:
            continue
        chosen = idx[: min(8, len(idx))]
        plot_wave_gallery(X, meta, chosen.tolist(), out_dir / f"processed_{group}_gallery.png", f"{dataset} processed {group}")
    plot_distribution(meta, out_dir / "raw_rms_distribution.png", "raw_rms_centered", label_col)


def plot_distribution(meta: pd.DataFrame, out_png: Path, value_col: str, label_col: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for key, group in meta.groupby(label_col):
        vals = group[value_col].to_numpy(dtype=np.float64)
        if len(vals):
            ax.hist(vals, bins=40, alpha=0.45, label=str(key), density=True)
    ax.set_xlabel(value_col)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_wave_gallery(X: np.ndarray, meta: pd.DataFrame, positions: list[int], out_png: Path, title: str, denoise: np.ndarray | None = None, probs: np.ndarray | None = None) -> None:
    if not positions:
        return
    t = np.arange(N_TARGET, dtype=np.float32) / FS_TARGET
    ncols = 2 if denoise is not None else 1
    fig, axes = plt.subplots(len(positions), ncols, figsize=(6.2 * ncols, max(4.0, 1.15 * len(positions))), sharex=True)
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape(len(positions), ncols)
    for r, pos in enumerate(positions):
        row = meta.iloc[pos]
        arrs = [("input", X[pos, 0])]
        if denoise is not None:
            arrs.append(("denoise", denoise[pos, 0] if denoise.ndim == 3 else denoise[pos]))
        y_min = min(float(np.min(a)) for _, a in arrs)
        y_max = max(float(np.max(a)) for _, a in arrs)
        pad = 0.06 * (y_max - y_min + 1e-8)
        for c, (name, arr) in enumerate(arrs):
            ax = axes_arr[r, c]
            ax.plot(t, arr, linewidth=0.72)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(True, alpha=0.18)
            if r == 0:
                ax.set_title(name, fontsize=10)
        prob_txt = ""
        if probs is not None:
            prob_txt = f"\np={np.round(probs[pos], 3).tolist()}"
        axes_arr[r, 0].set_ylabel(
            f"{row.get('label_raw', row.get('y_class', ''))} {row.get('record_id', '')}\n"
            f"{row.get('split', '')} {row.get('start_sec', 0):.1f}-{row.get('end_sec', 0):.1f}s{prob_txt}",
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
            labelpad=58,
        )
    for ax in axes_arr[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


class ExternalDataset(Dataset):
    def __init__(self, signals_path: Path, metadata_path: Path, split: str):
        self.X = np.load(signals_path)["X"].astype(np.float32)
        self.meta = pd.read_csv(metadata_path)
        keep = self.meta["split"].astype(str).to_numpy() == split
        self.indices = np.where(keep)[0]

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = int(self.indices[i])
        return {
            "x": torch.from_numpy(self.X[idx]),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def load_uformer_model(checkpoint: Path, device: torch.device, feature_set: str = "full_tokens", with_head: bool = True) -> UformerDenoiseSQIModel:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_dict = ckpt.get("config", {})
    cfg = Uformer1DConfig(
        noise_scale=float(cfg_dict.get("noise_scale", 0.9)),
        dropout=float(cfg_dict.get("dropout", 0.05)),
        feature_set=feature_set,
        detach_encoder_features=True,
        head_hidden_dim=int(cfg_dict.get("head_hidden_dim", 128)),
        denoiser_width=float(cfg_dict.get("denoiser_width", 1.0)),
    )
    model = UformerDenoiseSQIModel(cfg).to(device)
    if with_head:
        model.attach_head(model.infer_feature_dim(device=device))
        model.head = model.head.to(device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
    else:
        state = ckpt.get("denoiser_state", ckpt.get("model_state", ckpt))
        model.denoiser.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_model_outputs(model: UformerDenoiseSQIModel, X: np.ndarray, batch_size: int, device: torch.device) -> dict[str, np.ndarray | float]:
    logits_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    denoise_all: list[np.ndarray] = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s : s + batch_size]).to(device=device, dtype=torch.float32)
        out = model(xb)
        logits = out["logits"]
        if logits is None:
            logits = torch.zeros((xb.shape[0], 3), device=device)
        probs = F.softmax(logits, dim=1)
        logits_all.append(logits.cpu().numpy().astype(np.float32))
        probs_all.append(probs.cpu().numpy().astype(np.float32))
        denoise_all.append(out["denoise"].cpu().numpy().astype(np.float32))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - start)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "probs": np.concatenate(probs_all, axis=0),
        "denoise": np.concatenate(denoise_all, axis=0),
        "elapsed_sec": elapsed,
        "peak_cuda_memory_bytes": peak_mem,
    }


@torch.no_grad()
def extract_uformer_features(
    checkpoint: Path,
    X: np.ndarray,
    feature_set: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = load_uformer_model(checkpoint, device, feature_set=feature_set, with_head=False)
    feats: list[np.ndarray] = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s : s + batch_size]).to(device=device, dtype=torch.float32)
        denoiser_out = model.denoiser(xb)
        noise_hat = denoiser_out["noise_hat"]
        denoise = xb - float(model.cfg.noise_scale) * noise_hat
        feat = feature_vector(feature_set, xb, denoise, denoiser_out["features"], denoiser_out["bottleneck"])
        feats.append(feat.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def basic_sqi_features(X: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for sample in X[:, 0]:
        y = sample.astype(np.float64)
        dy = np.diff(y)
        f, pxx = welch(y, fs=FS_TARGET, nperseg=min(256, len(y)))

        def band(lo: float, hi: float) -> float:
            m = (f >= lo) & (f < hi)
            if not np.any(m):
                return 0.0
            df = float(np.median(np.diff(f))) if len(f) > 1 else 1.0
            return float(np.sum(pxx[m]) * df)

        total = band(0.5, 40.0) + 1e-12
        qrs = band(5.0, 15.0)
        wide = band(5.0, 40.0) + 1e-12
        base = band(0.5, 1.0)
        hf = band(20.0, 40.0)
        rows.append(
            [
                float(np.mean(y)),
                float(np.std(y)),
                float(np.max(np.abs(y))),
                float(np.sqrt(np.mean(y * y) + 1e-12)),
                float(np.mean(np.abs(dy))) if len(dy) else 0.0,
                float(np.std(dy)) if len(dy) else 0.0,
                float(qrs / wide),
                float(base / total),
                float(hf / total),
                float(kurtosis(y, fisher=False, bias=False)),
                float(skew(y, bias=False)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def multiclass_report(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_good_medium_bad": [float(v) for v in recall],
        "precision_good_medium_bad": [float(v) for v in precision],
        "f1_good_medium_bad": [float(v) for v in f1],
        "support_good_medium_bad": [int(v) for v in support],
        "confusion_3x3": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def binary_report(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "noisy_recall": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "confusion_2x2": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, score))
        out["auprc"] = float(average_precision_score(y_true, score))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def calibrate_but(probs_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for t_bad in np.linspace(0.05, 0.95, 46):
        for t_good in np.linspace(0.05, 0.95, 46):
            pred = np.full(len(y_val), 1, dtype=np.int64)
            pred[probs_val[:, 2] >= t_bad] = 2
            pred[(pred == 1) & (probs_val[:, 0] >= t_good)] = 0
            rep = multiclass_report(y_val, pred)
            key = (rep["macro_f1"], rep["recall_good_medium_bad"][2], rep["balanced_acc"])
            if best is None or key > best["key"]:
                best = {"t_bad": float(t_bad), "t_good": float(t_good), "key": key, "val_report": rep}
    assert best is not None
    return best


def apply_but_thresholds(probs: np.ndarray, t_good: float, t_bad: float) -> np.ndarray:
    pred = np.full(len(probs), 1, dtype=np.int64)
    pred[probs[:, 2] >= t_bad] = 2
    pred[(pred == 1) & (probs[:, 0] >= t_good)] = 0
    return pred


def calibrate_binary(score_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for thr in np.linspace(0.01, 0.99, 99):
        rep = binary_report(y_val, score_val, float(thr))
        candidates.append({"threshold": float(thr), "val_report": rep})
    strict = [
        c
        for c in candidates
        if c["val_report"]["noisy_recall"] >= 0.80 and c["val_report"]["specificity"] >= 0.50
    ]
    relaxed = [
        c
        for c in candidates
        if c["val_report"]["noisy_recall"] >= 0.60 and c["val_report"]["specificity"] >= 0.50
    ]
    pool = strict or relaxed or candidates
    for c in candidates:
        rep = c["val_report"]
        c["key"] = [
            1 if c in strict else 0,
            1 if c in relaxed else 0,
            rep["f1"],
            rep["noisy_recall"],
            rep["specificity"],
        ]
    return max(pool, key=lambda x: (x["val_report"]["f1"], x["val_report"]["noisy_recall"], x["val_report"]["specificity"]))


def hardware_info(device: torch.device, checkpoint: Path, model: UformerDenoiseSQIModel) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": int(checkpoint.stat().st_size) if checkpoint.exists() else 0,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def eval_zero_shot(dataset: str, out_root: Path, report_root: Path, checkpoint: Path, batch_size: int, force: bool = False) -> dict[str, Any]:
    processed = out_root / "processed" / dataset
    result_dir = out_root / "results" / dataset / "zero_shot_current_uformer"
    report_dir = report_root / dataset / "zero_shot_current_uformer"
    summary_path = result_dir / "test_report.json"
    if summary_path.exists() and not force:
        return read_json(summary_path)
    X = np.load(processed / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(processed / "metadata.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_uformer_model(checkpoint, device, feature_set="full_tokens", with_head=True)
    outputs = run_model_outputs(model, X, batch_size, device)
    probs = outputs["probs"]
    logits = outputs["logits"]
    denoise = outputs["denoise"]
    pred3 = np.argmax(probs, axis=1).astype(np.int64)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_dir / "model_outputs.npz", probs=probs, logits=logits, denoise=denoise, pred3=pred3)
    runtime = {
        **hardware_info(device, checkpoint, model),
        "dataset": dataset,
        "n_windows": int(len(X)),
        "batch_size": int(batch_size),
        "elapsed_sec": float(outputs["elapsed_sec"]),
        "throughput_windows_per_sec": float(len(X) / max(1e-9, outputs["elapsed_sec"])),
        "latency_ms_per_window": float(1000.0 * outputs["elapsed_sec"] / max(1, len(X))),
        "peak_cuda_memory_bytes": int(outputs["peak_cuda_memory_bytes"]),
    }
    write_json(result_dir / "runtime_hardware.json", runtime)
    if dataset == "butqdb":
        y = meta["y"].to_numpy(dtype=np.int64)
        raw_report = multiclass_report(y, pred3, probs)
        val_idx = meta.index[meta["split"].astype(str) == "val"].to_numpy()
        test_idx = meta.index[meta["split"].astype(str) == "test"].to_numpy()
        cal = calibrate_but(probs[val_idx], y[val_idx])
        cal_pred = apply_but_thresholds(probs[test_idx], cal["t_good"], cal["t_bad"])
        cal_report = multiclass_report(y[test_idx], cal_pred, probs[test_idx])
        raw_test_report = multiclass_report(y[test_idx], pred3[test_idx], probs[test_idx])
        payload = {
            "dataset": dataset,
            "task": "three_class_good_medium_bad",
            "raw_all_windows": raw_report,
            "raw_test": raw_test_report,
            "calibrated_test": cal_report,
            "threshold_calibration": cal,
            "runtime_hardware": runtime,
        }
    else:
        y_bin = meta["y_binary"].to_numpy(dtype=np.int64)
        score = probs[:, 2]
        val_idx = meta.index[meta["split"].astype(str) == "val"].to_numpy()
        test_idx = meta.index[meta["split"].astype(str) == "test"].to_numpy()
        cal = calibrate_binary(score[val_idx], y_bin[val_idx])
        raw_report = binary_report(y_bin[test_idx], score[test_idx], 0.5)
        cal_report = binary_report(y_bin[test_idx], score[test_idx], cal["threshold"])
        record_report = record_level_cinc(meta.iloc[test_idx].copy(), score[test_idx], cal["threshold"])
        payload = {
            "dataset": dataset,
            "task": "binary_noisy_rejection",
            "raw_test_pbad_thr0p5": raw_report,
            "calibrated_test": cal_report,
            "record_level_calibrated_test": record_report,
            "threshold_calibration": cal,
            "runtime_hardware": runtime,
        }
    write_json(result_dir / "test_report.json", payload)
    write_json(result_dir / "threshold_calibration.json", payload["threshold_calibration"])
    pred_df = meta.copy()
    pred_df["p_good"] = probs[:, 0]
    pred_df["p_medium"] = probs[:, 1]
    pred_df["p_bad"] = probs[:, 2]
    pred_df["pred3"] = pred3
    pred_df.to_csv(result_dir / "predictions.csv", index=False)
    export_eval_visuals(dataset, X, meta, denoise, probs, pred3, result_dir / "visuals")
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "test_report.json", payload)
    return payload


def record_level_cinc(meta_test: pd.DataFrame, score: np.ndarray, threshold: float) -> dict[str, Any]:
    meta_test = meta_test.copy()
    meta_test["score"] = score
    grouped = meta_test.groupby("record_id").agg({"score": "max", "y_binary": "max"}).reset_index()
    return binary_report(grouped["y_binary"].to_numpy(dtype=np.int64), grouped["score"].to_numpy(dtype=np.float64), threshold)


def export_eval_visuals(dataset: str, X: np.ndarray, meta: pd.DataFrame, denoise: np.ndarray, probs: np.ndarray, pred3: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "butqdb":
        y = meta["y"].to_numpy(dtype=np.int64)
        for cls, name in INT_TO_CLASS.items():
            idx = np.where(y == cls)[0][:8]
            plot_wave_gallery(X, meta, idx.tolist(), out_dir / f"{name}_denoise_gallery.png", f"BUT {name} denoise", denoise=denoise, probs=probs)
        errors = np.where(pred3 != y)[0][:10]
        plot_wave_gallery(X, meta, errors.tolist(), out_dir / "but_errors_gallery.png", "BUT zero-shot errors", denoise=denoise, probs=probs)
    else:
        y = meta["y_binary"].to_numpy(dtype=np.int64)
        noisy = np.where(y == 1)[0][:10]
        usable = np.where(y == 0)[0][:10]
        plot_wave_gallery(X, meta, noisy.tolist(), out_dir / "cinc_noisy_gallery.png", "CinC noisy denoise", denoise=denoise, probs=probs)
        plot_wave_gallery(X, meta, usable.tolist(), out_dir / "cinc_usable_gallery.png", "CinC usable denoise", denoise=denoise, probs=probs)
        missed = np.where((y == 1) & (probs[:, 2] < 0.5))[0][:10]
        plot_wave_gallery(X, meta, missed.tolist(), out_dir / "cinc_noisy_missed_gallery.png", "CinC noisy missed @0.5", denoise=denoise, probs=probs)


def run_probe_suite(dataset: str, out_root: Path, report_root: Path, checkpoint: Path, batch_size: int, force: bool = False) -> dict[str, Any]:
    processed = out_root / "processed" / dataset
    result_dir = out_root / "results" / dataset / "frozen_probe_suite"
    summary_path = result_dir / "probe_summary.json"
    if summary_path.exists() and not force:
        return read_json(summary_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(processed / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(processed / "metadata.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = meta["split"].astype(str).to_numpy()
    if dataset == "butqdb":
        y = meta["y"].to_numpy(dtype=np.int64)
        task = "multiclass"
    else:
        y = meta["y_binary"].to_numpy(dtype=np.int64)
        task = "binary"
    train = split == "train"
    val = split == "val"
    test = split == "test"
    feature_sets = ["full_tokens", "bottleneck_only", "summary_only", "residual_summary_only", "handcrafted_sqi"]
    models = {
        "logreg": lambda: LogisticRegression(max_iter=1200, class_weight="balanced", solver="lbfgs"),
        "linear_svm": lambda: LinearSVC(class_weight="balanced", max_iter=8000),
        "small_mlp": lambda: MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-4, max_iter=500, early_stopping=True, random_state=0),
    }
    summary: dict[str, Any] = {"dataset": dataset, "task": task, "runs": []}
    feature_cache: dict[str, np.ndarray] = {}
    for feature_set in feature_sets:
        if feature_set == "handcrafted_sqi":
            feats = basic_sqi_features(X)
        else:
            feats = extract_uformer_features(checkpoint, X, feature_set, batch_size, device)
        feature_cache[feature_set] = feats
        np.save(result_dir / f"features_{feature_set}.npy", feats.astype(np.float32))
        for model_name, factory in models.items():
            clf = Pipeline([("scaler", StandardScaler()), ("model", factory())])
            clf.fit(feats[train], y[train])
            if task == "multiclass":
                pred = clf.predict(feats[test]).astype(np.int64)
                report = multiclass_report(y[test], pred)
            else:
                score_val, score_test = classifier_score(clf, feats[val]), classifier_score(clf, feats[test])
                cal = calibrate_binary(score_val, y[val])
                report = {
                    "calibration": cal,
                    "test": binary_report(y[test], score_test, cal["threshold"]),
                }
            run = {"feature_set": feature_set, "model": model_name, "report": report}
            summary["runs"].append(run)
            write_json(result_dir / f"{feature_set}_{model_name}_report.json", run)
    write_json(summary_path, summary)
    write_json(report_root / dataset / "frozen_probe_suite" / "probe_summary.json", summary)
    return summary


def classifier_score(clf: Pipeline, X: np.ndarray) -> np.ndarray:
    model = clf.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return clf.predict(X).astype(float)


def make_synthetic_adaptation_specs(out_root: Path, report_root: Path) -> dict[str, Any]:
    synth_dir = out_root / "synthetic_adaptation"
    synth_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "status": "spec_ready_waiting_for_external_results",
        "note": "Generated after real-data preprocessing. These specs must not overwrite the current PTB mainline artifact.",
        "variants": [
            {
                "name": "but_like_3class",
                "goal": "Match BUT QDB good/medium/bad external distributions using relaxed morphology plus stronger real-world drift/HF coverage.",
                "source_artifact_dir": "outputs/transformer",
                "recommended_generator": "synthesize_morph_damage_triplet",
                "recommended_label_version": "e311h_lite_relaxed_morph",
                "output_artifact": str(synth_dir / "but_like_3class"),
            },
            {
                "name": "cinc_like_binary_bad",
                "goal": "Increase unusable/noisy rejection coverage with burst/high-frequency/baseline windows and binary noisy analysis.",
                "source_artifact_dir": "outputs/transformer",
                "recommended_generator": "synthesize_local_counterfactual_balanced",
                "recommended_noise_kinds": "em,ma,bw,mix,burst,hp,ampmod",
                "output_artifact": str(synth_dir / "cinc_like_binary_bad"),
            },
            {
                "name": "mixed_external_like",
                "goal": "Joint external-style synthetic profile for BUT three-class and CinC noisy rejection.",
                "source_artifact_dir": "outputs/transformer",
                "recommended_generator": "two_stage_mix",
                "output_artifact": str(synth_dir / "mixed_external_like"),
            },
        ],
    }
    write_json(synth_dir / "synthetic_adaptation_specs.json", specs)
    write_json(report_root / "synthetic_adaptation_specs.json", specs)
    return specs


def render_readme(out_root: Path, report_root: Path) -> None:
    lines = [
        "# E3.11f External Real-Data Benchmarks",
        "",
        "External validation for the Uformer SQI mainline. Raw data and processed NPZ files stay under ignored local directories.",
        "",
        "## Datasets",
        "",
        "- BUT QDB: expert consensus ECG quality labels, mapped `1/2/3 -> good/medium/bad`.",
        "- CinC2017: `~` label is treated as noisy/unusable; `N/A/O` are usable.",
        "",
        "## Current Artifacts",
        "",
    ]
    for path in sorted((out_root / "results").rglob("*.json")) if (out_root / "results").exists() else []:
        rel = path.relative_to(out_root).as_posix()
        lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Thresholds are calibrated only on validation splits.",
            "- External data has no clean waveform, so denoise visuals are no-reference audits, not supervised denoise scores.",
            "- `src/sqi_pipeline` is intentionally untouched.",
        ]
    )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    checkpoint = Path(args.checkpoint)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "external_benchmark_state.json"
    state: dict[str, Any] = {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": []}
    write_json(state_path, state)
    try:
        if args.stage in {"all", "download"}:
            state["steps"].append({"download_butqdb": download_butqdb(data_root, force=args.force_download)})
            state["steps"].append({"download_cinc2017": download_cinc2017(data_root, force=args.force_download)})
            write_json(state_path, state)
        if args.stage in {"all", "preprocess"}:
            state["steps"].append({"preprocess_butqdb": preprocess_butqdb(data_root, out_root, args.seed, force=args.force_preprocess)})
            state["steps"].append({"preprocess_cinc2017": preprocess_cinc2017(data_root, out_root, args.seed, force=args.force_preprocess)})
            write_json(state_path, state)
        if args.stage in {"all", "eval"}:
            for dataset in ["butqdb", "cinc2017"]:
                state["steps"].append({f"{dataset}_zero_shot": eval_zero_shot(dataset, out_root, report_root, checkpoint, args.batch_size, force=args.force_eval)})
                write_json(state_path, state)
        if args.stage in {"all", "probes"}:
            for dataset in ["butqdb", "cinc2017"]:
                state["steps"].append({f"{dataset}_probes": run_probe_suite(dataset, out_root, report_root, checkpoint, args.batch_size, force=args.force_eval)})
                write_json(state_path, state)
        if args.stage in {"all", "synthetic_specs"}:
            state["steps"].append({"synthetic_adaptation_specs": make_synthetic_adaptation_specs(out_root, report_root)})
        render_readme(out_root, report_root)
        state["status"] = "completed"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_json(state_path, state)
        raise
    write_json(state_path, state)
    return state


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E3.11f external real-data benchmarks.")
    parser.add_argument("--stage", choices=("all", "download", "preprocess", "eval", "probes", "synthetic_specs"), default="all")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    state = run_all(args)
    print(json.dumps({"status": state["status"], "state": str(Path(args.out_root) / "external_benchmark_state.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
