from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, find_peaks
from torch.utils.data import Dataset

from .constants import CLASS_TO_INT, NOISE_TO_INT

FS = 125
WIN_N = 1250


@dataclass(frozen=True)
class DataBundle:
    datasets: dict[str, "E311Dataset"]
    split_info: dict[str, dict[str, Any]]


class E311Dataset(Dataset):
    def __init__(
        self,
        *,
        noisy: np.ndarray,
        clean: np.ndarray,
        level: np.ndarray,
        valid_rr: np.ndarray,
        local_mask: np.ndarray,
        qrs_mask: np.ndarray,
        critical_mask: np.ndarray,
        y: np.ndarray,
        snr_db: np.ndarray,
        group_id: np.ndarray,
        noise_type: np.ndarray,
        idx: np.ndarray,
        input_mode: str = "raw",
    ) -> None:
        self.noisy = noisy.astype(np.float32, copy=False)
        self.clean = clean.astype(np.float32, copy=False)
        self.level = level.astype(np.float32, copy=False)
        self.valid_rr = (valid_rr.astype(np.float32) > 0.5).astype(np.float32)
        self.local_mask = local_mask.astype(np.float32, copy=False)
        self.qrs_mask = qrs_mask.astype(np.float32, copy=False)
        self.critical_mask = critical_mask.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.snr_db = snr_db.astype(np.float32, copy=False)
        self.group_id = group_id.astype(np.int64, copy=False)
        self.noise_type = noise_type.astype(np.int64, copy=False)
        self.idx = idx.astype(np.int64, copy=False)
        self.input_mode = input_mode

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        x_noisy = self.noisy[i]
        return {
            "x_noisy": torch.from_numpy(make_input_channels(x_noisy, self.input_mode)),
            "x_clean": torch.from_numpy(self.clean[i][None, :]),
            "level": torch.from_numpy(self.level[i]),
            "valid_rr": torch.tensor(self.valid_rr[i], dtype=torch.float32),
            "local_mask": torch.from_numpy(self.local_mask[i]),
            "qrs_mask": torch.from_numpy(self.qrs_mask[i]),
            "critical_mask": torch.from_numpy(self.critical_mask[i]),
            "y": torch.tensor(self.y[i], dtype=torch.long),
            "snr_norm": torch.tensor(normalize_snr_db(float(self.snr_db[i])), dtype=torch.float32),
            "snr_db": torch.tensor(float(self.snr_db[i]), dtype=torch.float32),
            "group_id": torch.tensor(self.group_id[i], dtype=torch.long),
            "noise_type": torch.tensor(self.noise_type[i], dtype=torch.long),
            "idx": torch.tensor(self.idx[i], dtype=torch.long),
        }


def load_bundle(
    artifact_dir: Path,
    *,
    out_root: Path,
    level_target_mode: str,
    input_mode: str,
    force_level_cache: bool = False,
) -> DataBundle:
    dataset_dir = artifact_dir / "datasets"
    labels = pd.read_csv(dataset_dir / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    noisy = np.load(dataset_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    clean = np.load(dataset_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    level_npz = np.load(dataset_dir / "synth_10s_125hz_noise_level.npz")
    local_npz = np.load(dataset_dir / "synth_10s_125hz_local_mask.npz")

    if len(labels) != noisy.shape[0] or noisy.shape != clean.shape:
        raise ValueError(f"Dataset mismatch labels={len(labels)} noisy={noisy.shape} clean={clean.shape}")
    if noisy.shape[1] != WIN_N:
        raise ValueError(f"Expected {WIN_N} samples, got {noisy.shape[1]}")

    level, valid_rr = level_targets(
        mode=level_target_mode,
        clean=clean,
        noisy=noisy,
        base_level=level_npz["P"].astype(np.float32),
        base_valid=level_npz["valid_rr"].astype(np.uint8),
        y=np.asarray([CLASS_TO_INT[str(v)] for v in labels["y_class"]], dtype=np.int64),
        out_root=out_root,
        force=force_level_cache,
    )

    y = np.asarray([CLASS_TO_INT[str(v)] for v in labels["y_class"]], dtype=np.int64)
    noise_type = np.asarray([NOISE_TO_INT.get(str(v), 0) for v in labels["noise_kind"]], dtype=np.int64)
    snr = labels["snr_db"].to_numpy(dtype=np.float32)
    group_id = labels.get("counterfactual_group", pd.Series(np.arange(len(labels)))).to_numpy(dtype=np.int64)
    idx = labels["idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(idx, np.arange(len(labels), dtype=np.int64)):
        raise ValueError("labels idx must be sorted 0..n-1")

    datasets: dict[str, E311Dataset] = {}
    info: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        m = labels["split"].astype(str).to_numpy() == split
        split_idx = idx[m]
        datasets[split] = E311Dataset(
            noisy=noisy[split_idx],
            clean=clean[split_idx],
            level=level[split_idx],
            valid_rr=valid_rr[split_idx],
            local_mask=local_npz["M"].astype(np.float32)[split_idx],
            qrs_mask=local_npz["qrs_mask"].astype(np.float32)[split_idx],
            critical_mask=local_npz["critical_mask"].astype(np.float32)[split_idx],
            y=y[split_idx],
            snr_db=snr[split_idx],
            group_id=group_id[split_idx],
            noise_type=noise_type[split_idx],
            idx=split_idx,
            input_mode=input_mode,
        )
        vc = pd.Series(y[split_idx]).value_counts().sort_index()
        info[split] = {
            "n": int(m.sum()),
            "class_counts": {str(k): int(vc.get(v, 0)) for k, v in CLASS_TO_INT.items()},
            "valid_rr_frac": float(valid_rr[split_idx].mean()) if len(split_idx) else 0.0,
        }
    return DataBundle(datasets=datasets, split_info=info)


def level_targets(
    *,
    mode: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    base_level: np.ndarray,
    base_valid: np.ndarray,
    y: np.ndarray,
    out_root: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "noisy_rr":
        return base_level.astype(np.float32), base_valid.astype(np.uint8)
    cache_dir = out_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = "patch_residual_v2" if mode == "patch_residual" else mode
    cache_path = cache_dir / f"level_targets_{cache_key}.npz"
    if cache_path.exists() and not force:
        z = np.load(cache_path)
        return z["P"].astype(np.float32), z["valid_rr"].astype(np.uint8)

    if mode in {"clean_rr", "clean_rr_bad_fallback"}:
        P = np.ones_like(base_level, dtype=np.float32)
        valid = np.zeros_like(base_valid, dtype=np.uint8)
        for i in range(clean.shape[0]):
            P[i], valid[i] = make_clean_rr_target(clean[i], noisy[i])
        if mode == "clean_rr_bad_fallback":
            bad_or_invalid = (y == 2) | (valid == 0)
            P[bad_or_invalid] = 1.0
            valid[bad_or_invalid] = 1
    elif mode == "patch_residual":
        # Use a dataset-level power scale so this target keeps absolute noise
        # severity. Per-record normalization made widespread bad examples look
        # artificially mild and inverted the intended bad-vs-medium ordering.
        residual_power = ((noisy - clean) ** 2).astype(np.float32)
        scale = np.float32(np.percentile(residual_power, 99.0) + 1e-8)
        P = np.clip(residual_power / scale, 0.0, 1.0).astype(np.float32)
        valid = np.ones_like(base_valid, dtype=np.uint8)
    else:
        raise ValueError(f"unknown level_target_mode={mode!r}")

    tmp = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, P=P.astype(np.float32), valid_rr=valid.astype(np.uint8))
    try:
        os.replace(tmp, cache_path)
    except OSError:
        if not cache_path.exists():
            raise
    return P.astype(np.float32), valid.astype(np.uint8)


def make_clean_rr_target(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.uint8]:
    try:
        peaks = detect_r_peaks(clean)
    except Exception:
        return np.ones(WIN_N, dtype=np.float32), np.uint8(0)
    if len(peaks) < 4:
        return np.ones(WIN_N, dtype=np.float32), np.uint8(0)

    p = np.full(WIN_N, np.nan, dtype=np.float32)
    intervals: list[tuple[int, int, float]] = []
    for k in range(1, len(peaks) - 2):
        l, r = int(peaks[k]), int(peaks[k + 1])
        if r <= l + 1:
            continue
        q = rr_probability(clean[l:r], noisy[l:r])
        p[l:r] = q
        intervals.append((l, r, q))
    if not intervals:
        return np.ones(WIN_N, dtype=np.float32), np.uint8(0)
    l0, _, first = intervals[0]
    _, r1, last = intervals[-1]
    p[:l0] = first
    p[r1:] = last
    if np.isnan(p).any():
        good = np.where(~np.isnan(p))[0]
        p = np.interp(np.arange(WIN_N), good, p[good]).astype(np.float32)
    return np.clip(p, 0.0, 1.0).astype(np.float32), np.uint8(1)


def detect_r_peaks(x: np.ndarray) -> np.ndarray:
    b, a = butter(2, [5, 15], btype="bandpass", fs=FS)
    y = filtfilt(b, a, x.astype(np.float64))
    dy2 = np.diff(y, prepend=y[0]) ** 2
    w = max(1, int(0.12 * FS))
    env = np.convolve(dy2, np.ones(w) / w, mode="same")
    thr = float(np.mean(env) + 0.5 * np.std(env))
    cand, _ = find_peaks(env, height=thr, distance=max(1, int(0.25 * FS)))
    refine_w = max(1, int(0.08 * FS))
    out: list[int] = []
    for p in cand:
        l, r = max(0, p - refine_w), min(len(x), p + refine_w + 1)
        q = l + int(np.argmax(np.abs(x[l:r])))
        if not out or q - out[-1] >= int(0.20 * FS):
            out.append(q)
        elif abs(x[q]) > abs(x[out[-1]]):
            out[-1] = q
    return np.asarray(out, dtype=int)


def rr_probability(clean_seg: np.ndarray, noisy_seg: np.ndarray) -> float:
    rho = safe_corr(clean_seg, noisy_seg)
    noise = noisy_seg - clean_seg
    px = float(np.mean(clean_seg**2)) + 1e-12
    pv = float(np.mean(noise**2)) + 1e-12
    snr_rr = 10.0 * float(np.log10(px / pv))
    p_corr = np.clip((0.90 - rho) / (0.90 - 0.50), 0.0, 1.0)
    p_snr = np.clip((16.0 - snr_rr) / (16.0 - (-6.0)), 0.0, 1.0)
    return float(max(p_corr, p_snr))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-10 or sb < 1e-10:
        return 1.0 if np.allclose(a, b, atol=1e-6) else 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))


def make_input_channels(x: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "raw":
        return x.astype(np.float32, copy=False)[None, :]
    robust = robust_normalize(x)
    if input_mode == "robust":
        return robust[None, :]
    if input_mode == "raw_robust":
        return np.stack([x.astype(np.float32, copy=False), robust], axis=0)
    raise ValueError(f"unknown input_mode={input_mode!r}")


def robust_normalize(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    scale = float(q75 - q25)
    if scale < 1e-6:
        scale = float(np.std(x)) + 1e-6
    return np.clip((x.astype(np.float32) - med) / scale, -10.0, 10.0).astype(np.float32)


def normalize_snr_db(snr_db: float) -> float:
    return float((snr_db - 7.0) / 13.0)


def save_split_info(path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
