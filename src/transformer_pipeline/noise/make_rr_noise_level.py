from __future__ import annotations

import sys
from pathlib import Path

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


FS = 125
WIN_N = 1250
SEED = 0
RHO_THR, RHO_MIN = 0.90, 0.50
SNR_THR, SNR_MIN = 16.0, -6.0


def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    b, a = butter(2, [5, 15], btype="bandpass", fs=fs) # type: ignore
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
    px = float(np.mean(clean_seg ** 2)) + 1e-12
    pv = float(np.mean(noise ** 2)) + 1e-12
    snr_rr = 10.0 * float(np.log10(px / pv))

    p_corr = np.clip((RHO_THR - rho) / (RHO_THR - RHO_MIN), 0.0, 1.0)
    p_snr = np.clip((SNR_THR - snr_rr) / (SNR_THR - SNR_MIN), 0.0, 1.0)
    return float(max(p_corr, p_snr))


# def make_sample_p(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, int]:
#     # 默认先设为 0（acceptable），这样不会两边全变橙
#     p = np.zeros(WIN_N, dtype=np.float32)

#     try:
#         peaks = detect_r_peaks(noisy, FS)
#     except Exception:
#         # 检测失败：全 1（unacceptable）
#         return np.ones(WIN_N, dtype=np.float32), 0

#     # 至少需要 4 个峰，才能“丢掉首尾RR”后还剩RR可用
#     if len(peaks) < 4:
#         return np.ones(WIN_N, dtype=np.float32), 0

#     # 只用中间 RR intervals：k = 1..len(peaks)-3
#     for k in range(1, len(peaks) - 2):
#         l, r = int(peaks[k]), int(peaks[k + 1])
#         if r <= l + 1:
#             continue
#         p[l:r] = rr_probability(clean[l:r], noisy[l:r])

#     # 两端（0..peaks[1] 和 peaks[-2]..end）我们保持为 0（acceptable）
#     # 第一个R前、最后一个R后不算（不标坏）
#     return np.clip(p, 0.0, 1.0).astype(np.float32), 1

def make_sample_p(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, int]:
    # default: unknown -> will fill by extrapolation; start with NaN
    p = np.full(WIN_N, np.nan, dtype=np.float32)

    try:
        peaks = detect_r_peaks(noisy, FS)
    except Exception:
        return np.ones(WIN_N, dtype=np.float32), 0

    # Need >=4 peaks so we can drop first/last RR and still have at least one RR interval
    if len(peaks) < 4:
        return np.ones(WIN_N, dtype=np.float32), 0

    # Compute p_rr for inner intervals only: k=1..len(peaks)-3
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

    # Extrapolate boundaries using first/last valid RR probability
    l0, r0, p_first = rr_intervals[0]
    l1, r1, p_last = rr_intervals[-1]

    # left: [0, l0)
    if l0 > 0:
        p[:l0] = p_first
    # right: [r1, end)
    if r1 < WIN_N:
        p[r1:] = p_last

    # any remaining NaNs (shouldn't happen often) -> fill with nearest (simple)
    if np.isnan(p).any():
        # forward fill then backward fill
        idx = np.where(~np.isnan(p))[0]
        if len(idx) == 0:
            return np.ones(WIN_N, dtype=np.float32), 0
        # forward
        last = p[idx[0]]
        for i in range(WIN_N):
            if not np.isnan(p[i]):
                last = p[i]
            else:
                p[i] = last
        # backward
        last = p[idx[-1]]
        for i in range(WIN_N - 1, -1, -1):
            if not np.isnan(p[i]):
                last = p[i]
            else:
                p[i] = last

    return np.clip(p, 0.0, 1.0).astype(np.float32), 1


def main() -> None:
    np.random.seed(SEED)
    root = project_root()
    in_clean = root / "artifact1" / "datasets" / "synth_10s_125hz_clean.npz"
    in_noisy = root / "artifact1" / "datasets" / "synth_10s_125hz_noisy.npz"
    in_lbl = root / "artifact1" / "datasets" / "synth_10s_125hz_labels.csv"

    out_dir = root / "artifact1" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "synth_10s_125hz_noise_level.npz"
    out_lbl = out_dir / "synth_10s_125hz_labels_with_level.csv"

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
        p_i, v_i = make_sample_p(X_clean[i], X_noisy[i])
        P[i] = p_i
        valid_rr[i] = np.uint8(v_i)

    np.savez_compressed(out_npz, P=P.astype(np.float32), valid_rr=valid_rr.astype(np.uint8))

    labels2 = labels.copy()
    labels2["valid_rr"] = valid_rr.astype(np.uint8)
    labels2["is_bad"] = (labels2["y_class"].astype(str) == "bad").astype(np.uint8)
    labels2.to_csv(out_lbl, index=False)

    print(f"valid_rr overall: {int(valid_rr.sum())}/{n}")
    print(f"P mean overall: {float(P.mean()):.6f}")
    print(f"P>0.5 frac overall: {float((P > 0.5).mean()):.6f}")
    for sp in ["train", "val", "test"]:
        m = labels2["split"].astype(str).to_numpy() == sp
        if not np.any(m):
            continue
        print(f"[{sp}] valid_rr: {int(valid_rr[m].sum())}/{int(m.sum())}")
        print(f"[{sp}] P mean: {float(P[m].mean()):.6f}")
        print(f"[{sp}] P>0.5 frac: {float((P[m] > 0.5).mean()):.6f}")
    print(f"Saved: {out_npz}")
    print(f"Saved: {out_lbl}")
    print("NEXT INPUT: artifact1/datasets/synth_10s_125hz_noise_level.npz (+ labels csv)")


if __name__ == "__main__":
    main()
