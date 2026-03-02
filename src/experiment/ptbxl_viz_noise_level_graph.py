from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

try:
    import pywt  # pip install PyWavelets
except Exception:
    pywt = None

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root


# ---------------- Config ----------------
FS = 125
WIN_N = 1250
SEED = 0

Y_CLASSES = ["good", "medium", "bad"]
NOISE_KINDS = ["em", "ma", "mix"]
K_PER_GROUP = 10

# P(t) display
P_MODE = "binary"        # "binary" or "continuous"
P_THR = 0.43             # for binary

# Wavelet weighting display
WAVELET_MODE = "binary"  # "binary" or "continuous" or "off"
WAVELET = "db6"
LEVEL = 4
W_Q = 0.65               # quantile threshold for binary mask
ALPHA = 3.0              # only used for "continuous" (visual scale hint)

# R-peak display
SHOW_RPEAKS = True


# ---------------- Helpers ----------------
def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Lightweight R-peak detector on noisy signal (Pan-Tompkins-ish)."""
    b, a = butter(2, [5, 15], btype="bandpass", fs=fs)  # type: ignore
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


def shade_p(ax: plt.Axes, t: np.ndarray, p: np.ndarray) -> None: # type: ignore
    """Orange shading for noise-level pseudo label P(t)."""
    if P_MODE == "continuous":
        step = 5
        for i in range(0, len(p), step):
            j = min(len(p), i + step)
            pj = float(np.mean(p[i:j]))
            if pj <= 0:
                continue
            ax.axvspan(t[i], t[j - 1], alpha=min(0.35, 0.05 + 0.30 * pj), color="orange")
        return

    # binary
    bad = (p >= P_THR).astype(np.uint8)
    i = 0
    while i < len(bad):
        if bad[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < len(bad) and bad[j] == 1:
            j += 1
        ax.axvspan(t[i], t[j - 1], alpha=0.25, color="orange")
        i = j


def wavelet_weight(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      w_norm in [0,1] length T
      w_mask binary {0,1} length T from quantile(W_Q)
    DWT (wavedec) + interpolate details back to T.
    """
    if pywt is None:
        raise RuntimeError("PyWavelets not installed. pip install PyWavelets")

    x = x.astype(np.float64, copy=False)
    T = len(x)
    wav = pywt.Wavelet(WAVELET)
    maxlev = pywt.dwt_max_level(T, wav.dec_len)
    lev = int(min(LEVEL, maxlev if maxlev >= 1 else 1))

    coeffs = pywt.wavedec(x, wavelet=WAVELET, level=lev, mode="symmetric")
    details = coeffs[1:]  # cD_lev ... cD1

    e = np.zeros(T, dtype=np.float64)
    for d in details:
        d_abs = np.abs(d)
        xp = np.linspace(0, T - 1, num=len(d_abs))
        e += np.interp(np.arange(T), xp, d_abs)

    e -= np.min(e)
    e /= (np.max(e) + 1e-12)
    w = e.astype(np.float32)

    thr = float(np.quantile(w, W_Q))
    m = (w >= thr).astype(np.uint8)
    return w, m


def shade_wavelet(ax: plt.Axes, t: np.ndarray, w: np.ndarray, m: np.ndarray) -> tuple[float, float]: # type: ignore
    """Purple shading for wavelet-important regions."""
    if WAVELET_MODE == "off" or pywt is None:
        return float("nan"), float("nan")

    if WAVELET_MODE == "continuous":
        step = 5
        for i in range(0, len(w), step):
            j = min(len(w), i + step)
            wj = float(np.mean(w[i:j]))
            if wj <= 0:
                continue
            ax.axvspan(t[i], t[j - 1], alpha=min(0.25, 0.03 + 0.22 * wj), color="purple")
        return float(w.mean()), float((w > 0.5).mean())

    # binary
    i = 0
    while i < len(m):
        if m[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < len(m) and m[j] == 1:
            j += 1
        ax.axvspan(t[i], t[j - 1], alpha=0.12, color="purple")
        i = j
    return float(w.mean()), float(m.mean())


def main() -> None:
    root = project_root()
    lbl = pd.read_csv(root / "artifact1" / "datasets" / "synth_10s_125hz_labels_with_level.csv")
    Xc = np.load(root / "artifact1" / "datasets" / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    Xn = np.load(root / "artifact1" / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    P = np.load(root / "artifact1" / "datasets" / "synth_10s_125hz_noise_level.npz")["P"].astype(np.float32)
    valid_rr = np.load(root / "artifact1" / "datasets" / "synth_10s_125hz_noise_level.npz")["valid_rr"].astype(np.uint8)

    assert len(lbl) == Xc.shape[0] == Xn.shape[0] == P.shape[0]
    t = np.arange(WIN_N) / FS
    rng = np.random.default_rng(SEED)

    out_dir = root / "artifact1" / "figs_for_noise_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "noise_level_graph_examples.png"

    # pick samples
    picks = []
    for y in Y_CLASSES:
        for nk in NOISE_KINDS:
            sub = lbl[(lbl["y_class"] == y) & (lbl["noise_kind"] == nk)]
            if len(sub) == 0:
                continue
            take = min(K_PER_GROUP, len(sub))
            picks.append(sub.sample(n=take, random_state=int(rng.integers(0, 1_000_000_000))))
    if not picks:
        print("No samples found.")
        return
    sel = pd.concat(picks, ignore_index=True)

    nrows = len(sel)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(16, 2.4 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, r in zip(axes, sel.itertuples(index=False)):
        idx = int(r.idx)
        xc = Xc[idx]
        xn = Xn[idx]
        p = P[idx]
        vr = int(valid_rr[idx])

        # RR pseudo label shading
        shade_p(ax, t, p)

        # Wavelet shading (based on CLEAN)
        w_mean = float("nan")
        w_frac = float("nan")
        if WAVELET_MODE != "off" and pywt is not None:
            w, m = wavelet_weight(xc)
            w_mean, w_frac = shade_wavelet(ax, t, w, m)

        # R-peaks on noisy (optional)
        n_peaks = -1
        if SHOW_RPEAKS:
            try:
                peaks = detect_r_peaks(xn, FS)
                n_peaks = int(len(peaks))
                for pk in peaks:
                    ax.axvline(pk / FS, linewidth=0.8, alpha=0.35)
            except Exception:
                n_peaks = -1

        ax.plot(t, xn, linewidth=1.2, label="Noisy")
        ax.plot(t, xc, linestyle="--", linewidth=1.0, alpha=0.8, label="Clean")

        title = (
            f"[{r.split}] y={r.y_class} noise={r.noise_kind} SNR={float(r.snr_db):.1f}dB | "
            f"idx={idx} seg_id={r.seg_id} ecg_id={r.ecg_id} | valid_rr={vr} nR={n_peaks} | "
            f"P_mean={float(p.mean()):.3f} P_bin={(p>=P_THR).mean():.3f} | "
            f"w_mean={w_mean:.3f} w_frac={w_frac:.3f} | "
            f"P_MODE={P_MODE} W_MODE={WAVELET_MODE}"
        )
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("ampl.")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "RR pseudo noise-level (orange) + Wavelet-loss weighting proxy (purple) + R-peaks (vertical lines)",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")
    if pywt is None:
        print("NOTE: PyWavelets not installed => wavelet part disabled (pip install PyWavelets).")


if __name__ == "__main__":
    main()