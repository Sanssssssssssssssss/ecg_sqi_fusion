from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root

logger = logging.getLogger(__name__)

FS = 125
WIN_N = 1250
RHO_THR, RHO_MIN = 0.90, 0.50
SNR_THR, SNR_MIN = 16.0, -6.0


def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    b, a = butter(2, [5, 15], btype="bandpass", fs=fs)
    y = filtfilt(b, a, x.astype(np.float64))
    dy2 = np.diff(y, prepend=y[0]) ** 2
    w = max(1, int(0.12 * fs))
    env = np.convolve(dy2, np.ones(w) / w, mode="same")
    thr = float(np.mean(env) + 0.5 * np.std(env))
    cand, _ = find_peaks(env, height=thr, distance=max(1, int(0.25 * fs)))
    if cand.size == 0:
        return np.array([], dtype=int)

    refine_w = max(1, int(0.08 * fs))
    out = []
    for p in cand:
        l, r = max(0, p - refine_w), min(len(x), p + refine_w + 1)
        q = l + int(np.argmax(np.abs(x[l:r])))
        if not out or q - out[-1] >= int(0.20 * fs):
            out.append(q)
        elif abs(x[q]) > abs(x[out[-1]]):
            out[-1] = q
    return np.asarray(out, dtype=int)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-10 or sb < 1e-10:
        return 1.0 if np.allclose(a, b, atol=1e-6) else 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))


def rr_probability(clean_seg: np.ndarray, noisy_seg: np.ndarray) -> float:
    rho = safe_corr(clean_seg, noisy_seg)
    noise = noisy_seg - clean_seg
    px = float(np.mean(clean_seg**2)) + 1e-12
    pv = float(np.mean(noise**2)) + 1e-12
    snr_rr = 10.0 * float(np.log10(px / pv))

    p_corr = np.clip((RHO_THR - rho) / (RHO_THR - RHO_MIN), 0.0, 1.0)
    p_snr = np.clip((SNR_THR - snr_rr) / (SNR_THR - SNR_MIN), 0.0, 1.0)
    return float(max(p_corr, p_snr))


def make_sample_p(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, int]:
    p = np.full(WIN_N, np.nan, dtype=np.float32)

    try:
        peaks = detect_r_peaks(noisy, FS)
    except Exception:
        return np.ones(WIN_N, dtype=np.float32), 0

    if len(peaks) < 4:
        return np.ones(WIN_N, dtype=np.float32), 0

    rr_intervals = []
    for k in range(1, len(peaks) - 2):
        l, r = int(peaks[k]), int(peaks[k + 1])
        if r <= l + 1:
            continue
        p_rr = rr_probability(clean[l:r], noisy[l:r])
        p[l:r] = p_rr
        rr_intervals.append((l, r, p_rr))

    if len(rr_intervals) == 0:
        return np.ones(WIN_N, dtype=np.float32), 0

    l0, _, p_first = rr_intervals[0]
    _, r1, p_last = rr_intervals[-1]
    if l0 > 0:
        p[:l0] = p_first
    if r1 < WIN_N:
        p[r1:] = p_last

    if np.isnan(p).any():
        idx = np.where(~np.isnan(p))[0]
        if len(idx) == 0:
            return np.ones(WIN_N, dtype=np.float32), 0
        last = p[idx[0]]
        for i in range(WIN_N):
            if not np.isnan(p[i]):
                last = p[i]
            else:
                p[i] = last
        last = p[idx[-1]]
        for i in range(WIN_N - 1, -1, -1):
            if not np.isnan(p[i]):
                last = p[i]
            else:
                p[i] = last

    return np.clip(p, 0.0, 1.0).astype(np.float32), 1


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer")
    seed = int(params.get("seed", 0))
    force = bool(params.get("force", False))
    np.random.seed(seed)

    in_clean = artifact_dir / "datasets" / "synth_10s_125hz_clean.npz"
    in_noisy = artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz"
    in_lbl = artifact_dir / "datasets" / "synth_10s_125hz_labels.csv"
    out_dir = artifact_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "synth_10s_125hz_noise_level.npz"
    out_lbl = out_dir / "synth_10s_125hz_labels_with_level.csv"

    if out_npz.exists() and out_lbl.exists() and not force:
        logger.info("noise_level: outputs exist -> skip (set --force to rerun)")
        return {"step": "noise_level", "skipped": True, "outputs": [str(out_npz), str(out_lbl)]}

    X_clean = np.load(in_clean)["X_clean"].astype(np.float32)
    X_noisy = np.load(in_noisy)["X_noisy"].astype(np.float32)
    labels = pd.read_csv(in_lbl)

    if X_clean.shape != X_noisy.shape or X_clean.shape[1] != WIN_N:
        raise ValueError(f"Unexpected shape clean={X_clean.shape}, noisy={X_noisy.shape}, expect (*,{WIN_N})")
    need = {"idx", "split", "y_class", "snr_db", "noise_kind"}
    if not need.issubset(labels.columns):
        raise ValueError(f"labels missing columns: {sorted(need - set(labels.columns))}")
    if len(labels) != len(X_clean):
        raise ValueError(f"labels rows {len(labels)} != samples {len(X_clean)}")

    labels = labels.sort_values("idx").reset_index(drop=True)
    n = len(labels)
    P = np.ones((n, WIN_N), dtype=np.float32)
    valid_rr = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        if verbose and (i + 1) % 10000 == 0:
            logger.info("noise_level progress: %d/%d", i + 1, n)
        p_i, v_i = make_sample_p(X_clean[i], X_noisy[i])
        P[i] = p_i
        valid_rr[i] = np.uint8(v_i)

    np.savez(out_npz, P=P.astype(np.float32), valid_rr=valid_rr.astype(np.uint8))

    labels2 = labels.copy()
    labels2["valid_rr"] = valid_rr.astype(np.uint8)
    labels2["is_bad"] = (labels2["y_class"].astype(str) == "bad").astype(np.uint8)
    labels2.to_csv(out_lbl, index=False)

    valid_total = int(valid_rr.sum())
    p_mean = float(P.mean())
    p_gt_0_5 = float((P > 0.5).mean())
    logger.info("valid_rr overall: %d/%d", valid_total, n)
    logger.info("P mean overall: %.6f", p_mean)
    logger.info("P>0.5 frac overall: %.6f", p_gt_0_5)
    for sp in ["train", "val", "test"]:
        m = labels2["split"].astype(str).to_numpy() == sp
        if not np.any(m):
            continue
        logger.info("[%s] valid_rr: %d/%d", sp, int(valid_rr[m].sum()), int(m.sum()))
        logger.info("[%s] P mean: %.6f", sp, float(P[m].mean()))
        logger.info("[%s] P>0.5 frac: %.6f", sp, float((P[m] > 0.5).mean()))
    logger.info("Saved: %s", _display(out_npz, root))
    logger.info("Saved: %s", _display(out_lbl, root))
    return {
        "step": "noise_level",
        "skipped": False,
        "outputs": [str(out_npz), str(out_lbl)],
        "rows": int(n),
        "valid_rr": f"{valid_total}/{n}",
        "p_mean": p_mean,
        "p_gt_0_5_frac": p_gt_0_5,
    }


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RR-level pseudo noise labels for transformer training.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
