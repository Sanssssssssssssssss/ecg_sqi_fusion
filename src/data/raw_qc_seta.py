# src/data/raw_qc_seta.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import welch

from src.utils.paths import project_root


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# =========================
# Fixed config for Task 1.2
# =========================
SEED = 0
N_SAMPLES = 10
PLOT_SECONDS = 10          # plot first 10 seconds
PSD_FMAX = 40.0            # show 0-40Hz
WELCH_NPERSEG = 1024       # for fs=500, ~2.048s window
WELCH_NOVERLAP = 512
WELCH_DETREND = "constant"
# =========================


def load_record_seta(record_id: str, data_dir: Path) -> tuple[np.ndarray, float, list[str]]:
    """
    Returns:
      sig: (N, n_sig) physical units
      fs: float
      sig_name: list[str]
    """
    rec_path = data_dir / record_id
    rec = wfdb.rdrecord(str(rec_path), physical=True)
    sig = rec.p_signal
    fs = float(rec.fs)
    sig_name = list(rec.sig_name) if rec.sig_name is not None else [f"ch{i}" for i in range(sig.shape[1])]
    return sig, fs, sig_name


def ensure_lead_order(sig: np.ndarray, sig_name: list[str]) -> np.ndarray:
    """
    Reorder columns into the fixed 12-lead order if possible.
    If missing leads, raise error (set-a should be 12 leads).
    """
    name_to_idx = {n: i for i, n in enumerate(sig_name)}
    missing = [l for l in LEADS_12 if l not in name_to_idx]
    if missing:
        raise ValueError(f"Missing leads in record: {missing}. Available={sig_name}")
    idxs = [name_to_idx[l] for l in LEADS_12]
    return sig[:, idxs]


def plot_wave12(record_id: str, sig12: np.ndarray, fs: float, out_png: Path, y: int) -> None:
    n_plot = min(int(round(PLOT_SECONDS * fs)), sig12.shape[0])
    t = np.arange(n_plot) / fs

    fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, sig12[:n_plot, i])
        ax.set_ylabel(LEADS_12[i], rotation=0, labelpad=20, va="center")
        ax.grid(False)

    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle(f"{record_id} | y={y} | 12-lead raw ECG (first {PLOT_SECONDS}s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_std_range(record_id: str, sig12: np.ndarray, out_png: Path, y: int) -> tuple[np.ndarray, np.ndarray]:
    std = sig12.std(axis=0)
    rng = sig12.max(axis=0) - sig12.min(axis=0)

    x = np.arange(len(LEADS_12))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width/2, std, width, label="std")
    ax.bar(x + width/2, rng, width, label="range (max-min)")
    ax.set_xticks(x)
    ax.set_xticklabels(LEADS_12)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"{record_id} | y={y} | per-lead std & range (raw, full length)")
    ax.legend()
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return std, rng


def plot_psd(record_id: str, sig12: np.ndarray, fs: float, out_png: Path, y: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD for each lead; plot mean PSD across 12 leads (and optionally show spread lightly).
    Returns:
      f (Hz), Pxx_mean
    """
    # compute PSD per lead
    psds = []
    for i in range(sig12.shape[1]):
        f, pxx = welch(
            sig12[:, i],
            fs=fs,
            nperseg=min(WELCH_NPERSEG, sig12.shape[0]),
            noverlap=min(WELCH_NOVERLAP, max(0, sig12.shape[0] // 2)),
            detrend=WELCH_DETREND,
        )
        psds.append(pxx)
    psds = np.stack(psds, axis=0)  # (12, len(f))
    pxx_mean = psds.mean(axis=0)

    # limit to 0-40Hz
    m = (f >= 0) & (f <= PSD_FMAX)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(f[m], pxx_mean[m])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz or mV^2/Hz)")
    ax.set_title(f"{record_id} | y={y} | Welch PSD mean over 12 leads (0–{PSD_FMAX}Hz)")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return f, pxx_mean


def band_power_ratio(f: np.ndarray, pxx: np.ndarray, fmax: float) -> float:
    """
    Ratio of power in [0, fmax] to total power [0, Nyquist] for the mean PSD.
    """
    df = float(np.mean(np.diff(f)))
    total = float(np.sum(pxx) * df)
    band = float(np.sum(pxx[(f >= 0) & (f <= fmax)]) * df)
    return band / total if total > 0 else np.nan


def main() -> None:
    root = project_root()

    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    data_dir = root / "data" / "physionet" / "challenge-2011" / "set-a"

    out_examples = root / "artifacts" / "qc" / "raw_examples"
    out_summary = root / "artifacts" / "qc" / "raw_summary.csv"

    print(f"[Task 1.2] Reading split: {split_csv}")
    df_split = pd.read_csv(split_csv)

    # filter train
    df_train = df_split[df_split["split"] == "train"].copy()
    print(f"Train rows: {len(df_train)}")

    # fixed sample with seed
    rng = np.random.default_rng(SEED)
    ids = df_train["record_id"].astype(str).to_numpy()
    if len(ids) == 0:
        raise ValueError("No train records found in split CSV.")
    n = min(N_SAMPLES, len(ids))
    sample_ids = rng.choice(ids, size=n, replace=False).tolist()

    print(f"\nSampled N={n} train records (seed={SEED}):")
    for rid in sample_ids:
        print(" -", rid)

    rows = []
    for rid in sample_ids:
        y = int(df_train.loc[df_train["record_id"].astype(str) == rid, "y"].iloc[0])

        sig, fs, sig_name = load_record_seta(rid, data_dir)
        sig12 = ensure_lead_order(sig, sig_name)

        # basic info
        sig_len = sig12.shape[0]
        n_sig = sig12.shape[1]

        print(f"\n[{rid}] fs={fs}, sig_len={sig_len}, n_sig={n_sig}")

        # plots
        wave_png = out_examples / f"{rid}_wave12.png"
        std_png = out_examples / f"{rid}_std_range.png"
        psd_png = out_examples / f"{rid}_psd.png"

        plot_wave12(rid, sig12, fs, wave_png, y=y)
        std, rngv = plot_std_range(rid, sig12, std_png, y=y)
        f, pxx_mean = plot_psd(rid, sig12, fs, psd_png, y=y)

        # summary metrics
        row = {
            "record_id": rid,
            "y": y,
            "fs": fs,
            "sig_len": sig_len,
            "n_sig": n_sig,
            "global_min": float(sig12.min()),
            "global_max": float(sig12.max()),
            "global_range": float(sig12.max() - sig12.min()),
            "mean_std_12lead": float(std.mean()),
            "mean_range_12lead": float(rngv.mean()),
            "power_ratio_0_40": float(band_power_ratio(f, pxx_mean, fmax=PSD_FMAX)),
        }
        # per-lead std/range
        for i, lead in enumerate(LEADS_12):
            row[f"std_{lead}"] = float(std[i])
            row[f"range_{lead}"] = float(rngv[i])

        rows.append(row)

    df_sum = pd.DataFrame(rows)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(out_summary, index=False)

    print(f"\nSaved raw summary -> {out_summary} (rows={len(df_sum)})")
    print("\nHead of raw_summary:")
    print(df_sum.head(5).to_string(index=False))

    print(f"\nSaved QC images under -> {out_examples}")
    print("Expected files per record:")
    print(" - {record_id}_wave12.png")
    print(" - {record_id}_std_range.png")
    print(" - {record_id}_psd.png")


if __name__ == "__main__":
    main()
