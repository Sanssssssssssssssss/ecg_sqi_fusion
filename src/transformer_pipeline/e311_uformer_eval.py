"""Legacy E3.11f denoise evaluation helpers kept for archived experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

FS = 125
INT_TO_CLASS = {0: "good", 1: "medium", 2: "bad"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(x: float | np.floating) -> float:
    value = float(x)
    return value if np.isfinite(value) else 0.0


def _moving_average_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=False)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, ((0, 0), (pad_left, pad_right)), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = np.convolve(padded[i], kernel, mode="valid")
    return out


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1)) + 1e-12
    return np.sum(a0 * b0, axis=1) / denom


def _simple_peak_indices(x_abs: np.ndarray, threshold: float, min_distance: int) -> list[int]:
    peaks: list[int] = []
    last = -10**9
    for i in range(1, len(x_abs) - 1):
        if x_abs[i] < threshold or x_abs[i] < x_abs[i - 1] or x_abs[i] < x_abs[i + 1]:
            continue
        if i - last >= min_distance:
            peaks.append(i)
            last = i
        elif peaks and x_abs[i] > x_abs[peaks[-1]]:
            peaks[-1] = i
            last = i
    return peaks


def qrs_like_peak_metrics(clean: np.ndarray, signal: np.ndarray, fs: int = FS) -> dict[str, float]:
    timing_ms: list[float] = []
    amp_err: list[float] = []
    matched = 0
    total = 0
    min_distance = max(1, int(0.25 * fs))
    search = max(1, int(0.08 * fs))
    for c, s in zip(clean, signal):
        c_abs = np.abs(c)
        threshold = max(float(np.percentile(c_abs, 92.0)), float(c_abs.mean() + 0.7 * c_abs.std()))
        peaks = _simple_peak_indices(c_abs, threshold, min_distance)
        total += len(peaks)
        s_abs = np.abs(s)
        for p in peaks:
            lo = max(0, p - search)
            hi = min(len(s_abs), p + search + 1)
            if hi <= lo:
                continue
            local = lo + int(np.argmax(s_abs[lo:hi]))
            matched += 1
            timing_ms.append(abs(local - p) * 1000.0 / float(fs))
            denom = max(float(c_abs[p]), 1e-6)
            amp_err.append(abs(float(s_abs[local]) - float(c_abs[p])) / denom)
    return {
        "qrs_peak_count": int(total),
        "qrs_matched_fraction": _safe_float(float(matched) / float(total)) if total else 0.0,
        "qrs_timing_abs_ms_mean": _safe_float(np.mean(timing_ms)) if timing_ms else 0.0,
        "qrs_amp_rel_abs_err_mean": _safe_float(np.mean(amp_err)) if amp_err else 0.0,
    }


def ssim_1d(clean: np.ndarray, signal: np.ndarray) -> np.ndarray:
    c = clean.astype(np.float64, copy=False)
    s = signal.astype(np.float64, copy=False)
    mux = c.mean(axis=1)
    muy = s.mean(axis=1)
    vx = c.var(axis=1)
    vy = s.var(axis=1)
    cov = ((c - mux[:, None]) * (s - muy[:, None])).mean(axis=1)
    c1 = 0.01**2
    c2 = 0.03**2
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2) + 1e-12)


def patch_corr(clean: np.ndarray, signal: np.ndarray, patch: int = 125) -> np.ndarray:
    vals = []
    for start in range(0, clean.shape[1] - patch + 1, patch):
        vals.append(_row_corr(clean[:, start : start + patch], signal[:, start : start + patch]))
    return np.stack(vals, axis=1).mean(axis=1)


def masked_mae(signal: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denom = np.sum(mask, axis=1) + 1e-8
    return np.sum(np.abs(signal - clean) * mask, axis=1) / denom


def group_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    pred: np.ndarray,
    masks: dict[str, np.ndarray],
    select: np.ndarray,
) -> dict[str, Any]:
    idx = np.where(select)[0]
    if idx.size == 0:
        return {"n": 0}
    c = clean[idx].astype(np.float64, copy=False)
    n = noisy[idx].astype(np.float64, copy=False)
    p = pred[idx].astype(np.float64, copy=False)
    sig = np.mean(c * c, axis=1)
    mse_noisy = np.mean((n - c) ** 2, axis=1)
    mse_pred = np.mean((p - c) ** 2, axis=1)
    snr_noisy = 10.0 * np.log10((sig + 1e-12) / (mse_noisy + 1e-12))
    snr_pred = 10.0 * np.log10((sig + 1e-12) / (mse_pred + 1e-12))
    dc = np.diff(c, axis=1)
    dn = np.diff(n, axis=1)
    dp = np.diff(p, axis=1)
    deriv_corr_noisy = _row_corr(dc, dn)
    deriv_corr_pred = _row_corr(dc, dp)
    hf_noisy = np.mean(np.diff(n - c, axis=1) ** 2, axis=1)
    hf_pred = np.mean(np.diff(p - c, axis=1) ** 2, axis=1)
    drift_noisy = np.mean(_moving_average_same(n - c, FS) ** 2, axis=1)
    drift_pred = np.mean(_moving_average_same(p - c, FS) ** 2, axis=1)
    qrs_noisy = qrs_like_peak_metrics(c, n)
    qrs_pred = qrs_like_peak_metrics(c, p)
    ssim_noisy = ssim_1d(c, n)
    ssim_pred = ssim_1d(c, p)
    pcorr_noisy = patch_corr(c, n)
    pcorr_pred = patch_corr(c, p)
    out: dict[str, Any] = {
        "n": int(idx.size),
        "mse_noisy_vs_clean_mean": _safe_float(np.mean(mse_noisy)),
        "mse_pred_vs_clean_mean": _safe_float(np.mean(mse_pred)),
        "mse_ratio_pred_over_noisy": _safe_float(np.mean(mse_pred) / (np.mean(mse_noisy) + 1e-12)),
        "improvement_fraction": _safe_float(np.mean(mse_pred < mse_noisy)),
        "snr_noisy_db_mean": _safe_float(np.mean(snr_noisy)),
        "snr_pred_db_mean": _safe_float(np.mean(snr_pred)),
        "snr_improve_db_mean": _safe_float(np.mean(snr_pred - snr_noisy)),
        "deriv_corr_noisy_mean": _safe_float(np.mean(deriv_corr_noisy)),
        "deriv_corr_pred_mean": _safe_float(np.mean(deriv_corr_pred)),
        "deriv_corr_delta_mean": _safe_float(np.mean(deriv_corr_pred - deriv_corr_noisy)),
        "hf_error_energy_reduction": _safe_float(1.0 - (np.mean(hf_pred) / (np.mean(hf_noisy) + 1e-12))),
        "drift_error_energy_reduction": _safe_float(1.0 - (np.mean(drift_pred) / (np.mean(drift_noisy) + 1e-12))),
        "qrs_noisy": qrs_noisy,
        "qrs_pred": qrs_pred,
        "qrs_timing_abs_ms_delta": _safe_float(qrs_pred["qrs_timing_abs_ms_mean"] - qrs_noisy["qrs_timing_abs_ms_mean"]),
        "qrs_amp_rel_abs_err_delta": _safe_float(qrs_pred["qrs_amp_rel_abs_err_mean"] - qrs_noisy["qrs_amp_rel_abs_err_mean"]),
        "ssim_noisy_mean": _safe_float(np.mean(ssim_noisy)),
        "ssim_pred_mean": _safe_float(np.mean(ssim_pred)),
        "ssim_delta_mean": _safe_float(np.mean(ssim_pred - ssim_noisy)),
        "patch_corr_noisy_mean": _safe_float(np.mean(pcorr_noisy)),
        "patch_corr_pred_mean": _safe_float(np.mean(pcorr_pred)),
        "patch_corr_delta_mean": _safe_float(np.mean(pcorr_pred - pcorr_noisy)),
    }
    for key, mask in masks.items():
        m = mask[idx].astype(np.float64, copy=False)
        noisy_mae = masked_mae(n, c, m)
        pred_mae = masked_mae(p, c, m)
        out[f"{key}_mae_noisy_mean"] = _safe_float(np.mean(noisy_mae))
        out[f"{key}_mae_pred_mean"] = _safe_float(np.mean(pred_mae))
        out[f"{key}_mae_reduction"] = _safe_float(1.0 - (np.mean(pred_mae) / (np.mean(noisy_mae) + 1e-12)))
    qrs_penalty = max(0.0, out["qrs_amp_rel_abs_err_delta"])
    out["denoise_score"] = _safe_float(
        0.30 * out["snr_improve_db_mean"]
        + 0.25 * (1.0 - out["mse_ratio_pred_over_noisy"])
        + 0.20 * out["critical_mae_reduction"]
        + 0.15 * out["ssim_delta_mean"]
        + 0.10 * out["drift_error_energy_reduction"]
        + 0.05 * out["hf_error_energy_reduction"]
        - 0.15 * qrs_penalty
    )
    return out


def build_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    pred: np.ndarray,
    y: np.ndarray,
    noise_kind: np.ndarray,
    placement: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "shape": list(pred.shape),
        "overall": group_metrics(clean, noisy, pred, masks, np.ones(len(y), dtype=bool)),
        "by_class": {},
        "by_noise_kind": {},
        "by_placement": {},
    }
    for c, name in INT_TO_CLASS.items():
        out["by_class"][name] = group_metrics(clean, noisy, pred, masks, y == c)
    for nk in sorted(set(noise_kind.tolist())):
        out["by_noise_kind"][str(nk)] = group_metrics(clean, noisy, pred, masks, noise_kind == nk)
    for plc in sorted(set(placement.tolist())):
        out["by_placement"][str(plc)] = group_metrics(clean, noisy, pred, masks, placement == plc)
    return out


def cls_report(y: np.ndarray, logits: np.ndarray | None) -> dict[str, Any]:
    if logits is None:
        return {"acc": None, "recall_good_medium_bad": [None, None, None], "confusion_3x3": None}
    pred = np.argmax(logits, axis=1)
    mat = np.zeros((3, 3), dtype=int)
    for a, b in zip(y, pred):
        mat[int(a), int(b)] += 1
    recall = []
    for c in range(3):
        denom = max(1, int(np.sum(y == c)))
        recall.append(float(np.sum((y == c) & (pred == c)) / denom))
    return {"acc": float(np.mean(pred == y)), "recall_good_medium_bad": recall, "confusion_3x3": mat.tolist()}


@torch.no_grad()
def collect_outputs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        k: []
        for k in [
            "noisy",
            "clean",
            "denoise_pred",
            "y",
            "logits",
            "idx",
            "ecg_id",
            "snr_db",
            "original_snr_db",
            "local_mask",
            "critical_mask",
            "qrs_mask",
            "tst_mask",
        ]
    }
    noise_kind_all: list[np.ndarray] = []
    placement_all: list[np.ndarray] = []
    for batch in loader:
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        out = model(noisy)
        denoise = out["denoise"]
        chunks["noisy"].append(noisy.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["clean"].append(batch["clean"].squeeze(1).numpy().astype(np.float32))
        chunks["denoise_pred"].append(denoise.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["y"].append(batch["y"].numpy().astype(np.int64))
        logits = out["logits"]
        if logits is None:
            chunks["logits"].append(np.zeros((int(noisy.shape[0]), 3), dtype=np.float32))
        else:
            chunks["logits"].append(logits.cpu().numpy().astype(np.float32))
        for key in ["idx", "ecg_id"]:
            chunks[key].append(batch[key].numpy().astype(np.int64))
        for key in ["snr_db", "original_snr_db", "local_mask", "critical_mask", "qrs_mask", "tst_mask"]:
            chunks[key].append(batch[key].numpy().astype(np.float32))
        noise_kind_all.append(np.asarray(batch["noise_kind"], dtype=str))
        placement_all.append(np.asarray(batch["placement"], dtype=str))
    arrays = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    arrays["noise_kind"] = np.concatenate(noise_kind_all, axis=0)
    arrays["placement"] = np.concatenate(placement_all, axis=0)
    arrays["y_pred_denoise"] = np.argmax(arrays["logits"], axis=1).astype(np.int64)
    return arrays


def select_positions(arrays: dict[str, np.ndarray], mode: str, max_rows: int = 12) -> list[int]:
    y = arrays["y"].astype(np.int64)
    noisy = arrays["noisy"]
    clean = arrays["clean"]
    denoise = arrays["denoise_pred"]
    noise_mse = np.mean((noisy - clean) ** 2, axis=1)
    pred_mse = np.mean((denoise - clean) ** 2, axis=1)
    local = arrays.get("critical_mask", np.zeros_like(clean)) + arrays.get("qrs_mask", np.zeros_like(clean)) + arrays.get("tst_mask", np.zeros_like(clean))
    local_mass = np.mean(local, axis=1)
    chosen: list[int] = []

    def add(items) -> None:
        for item in list(items):
            pos = int(item)
            if pos not in chosen:
                chosen.append(pos)
            if len(chosen) >= max_rows:
                break

    if mode == "balanced":
        for c in [0, 1, 2]:
            idx = np.where(y == c)[0]
            add(idx[np.argsort(noise_mse[idx])[::-1]][:4])
    elif mode == "hard_bad":
        idx = np.where(y >= 1)[0]
        add(idx[np.argsort(noise_mse[idx])[::-1]][:max_rows])
    elif mode == "good_safety":
        idx = np.where(y == 0)[0]
        add(idx[np.argsort((pred_mse - noise_mse)[idx])[::-1]][:max_rows])
    elif mode == "worst_residual":
        add(np.argsort(pred_mse)[::-1][:max_rows])
    elif mode == "qrs_tst_focus":
        add(np.argsort(local_mass)[::-1][:max_rows])
    return chosen[:max_rows]


def export_fixed_gallery(arrays: dict[str, np.ndarray], positions: list[int], out_png: Path, title: str) -> None:
    if not positions:
        return
    t = np.arange(arrays["clean"].shape[1], dtype=np.float32) / 125.0
    cols = [("clean", arrays["clean"]), ("before/noisy", arrays["noisy"]), ("after/denoise", arrays["denoise_pred"])]
    fig, axes = plt.subplots(len(positions), 3, figsize=(12.2, max(5.8, 1.25 * len(positions))), sharex=True)
    if len(positions) == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, pos in enumerate(positions):
        row_min = min(float(arr[pos].min()) for _, arr in cols)
        row_max = max(float(arr[pos].max()) for _, arr in cols)
        pad = 0.06 * (row_max - row_min + 1e-8)
        y = int(arrays["y"][pos])
        pred = int(arrays["y_pred_denoise"][pos])
        for c, (name, arr) in enumerate(cols):
            ax = axes[r, c]
            ax.plot(t, arr[pos], linewidth=0.68)
            ax.set_ylim(row_min - pad, row_max + pad)
            ax.grid(True, alpha=0.15)
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(
                    f"{INT_TO_CLASS[y]} idx={int(arrays['idx'][pos])}\n"
                    f"ecg={int(arrays['ecg_id'][pos])} snr={float(arrays['snr_db'][pos]):.2f}\n"
                    f"{arrays['noise_kind'][pos]} {arrays['placement'][pos]}\n"
                    f"pred={INT_TO_CLASS.get(pred, pred)}",
                    fontsize=7.2,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=58,
                )
            else:
                ax.set_yticklabels([])
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def evaluate_and_export(model: torch.nn.Module, loader: DataLoader, device: torch.device, out_dir: Path | None) -> dict[str, Any]:
    arrays = collect_outputs(model, loader, device)
    y = arrays["y"].astype(np.int64)
    report = cls_report(y, arrays["logits"])
    masks = {
        "local": arrays["local_mask"],
        "critical": arrays["critical_mask"],
        "qrs": arrays["qrs_mask"],
        "tst": arrays["tst_mask"],
    }
    metrics = build_metrics(arrays["clean"], arrays["noisy"], arrays["denoise_pred"], y, arrays["noise_kind"], arrays["placement"], masks)
    if out_dir is not None:
        (out_dir / "denoise_eval").mkdir(parents=True, exist_ok=True)
        (out_dir / "visuals").mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "test_report.json", report)
        write_json(out_dir / "denoise_eval" / "denoise_metrics.json", metrics)
        np.savez_compressed(
            out_dir / "denoise_eval" / "test_denoise_outputs.npz",
            noisy=arrays["noisy"],
            clean=arrays["clean"],
            denoise_pred=arrays["denoise_pred"],
            y=y,
            y_pred=arrays["y_pred_denoise"],
            idx=arrays["idx"],
            ecg_id=arrays["ecg_id"],
            snr_db=arrays["snr_db"],
            noise_kind=arrays["noise_kind"],
            placement=arrays["placement"],
        )
        for mode in ["balanced", "hard_bad", "good_safety", "worst_residual", "qrs_tst_focus"]:
            export_fixed_gallery(arrays, select_positions(arrays, mode), out_dir / "visuals" / f"{mode}_gallery.png", f"{mode} gallery")
        write_json(
            out_dir / "visuals" / "visual_gallery_manifest.json",
            {
                "balanced": "visuals/balanced_gallery.png",
                "hard_bad": "visuals/hard_bad_gallery.png",
                "good_safety": "visuals/good_safety_gallery.png",
                "worst_residual": "visuals/worst_residual_gallery.png",
                "qrs_tst_focus": "visuals/qrs_tst_focus_gallery.png",
            },
        )
    return {"report": report, "metrics": metrics}
