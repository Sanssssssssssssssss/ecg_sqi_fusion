# src/preprocess/qc_resample_125.py
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

SEED = 0
N_QC = 3
FS_IN = 500
FS_OUT = 125
PSD_FMAX = 40.0

# take 2.0 seconds
WAVE_SECONDS = 2.0
LEAD_FOR_QC = "II"  # First, fix it to II.


def ensure_lead_order(sig: np.ndarray, sig_name: list[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(sig_name)}
    idxs = [name_to_idx[l] for l in LEADS_12]
    return sig[:, idxs]


def load_record_500(record_id: str, data_dir: Path) -> tuple[np.ndarray, float, list[str]]:
    rec = wfdb.rdrecord(str(data_dir / record_id), physical=True)
    return rec.p_signal, float(rec.fs), list(rec.sig_name)


def load_npz_125(record_id: str, resampled_dir: Path) -> np.ndarray:
    z = np.load(resampled_dir / f"{record_id}.npz", allow_pickle=True)
    return z["sig_125"]


def plot_wave_compare(record_id: str, x500: np.ndarray, x125: np.ndarray, out_png: Path) -> None:
    # choose middle 2 seconds
    n500 = x500.shape[0]
    mid = n500 // 2
    half = int(round((WAVE_SECONDS / 2.0) * FS_IN))
    s0 = max(0, mid - half)
    s1 = min(n500, mid + half)

    t500 = np.arange(s0, s1) / FS_IN
    seg500 = x500[s0:s1]

    # corresponding indices in 125Hz (divide by 4)
    s0_125 = int(round(s0 / 4))
    s1_125 = int(round(s1 / 4))
    t125 = np.arange(s0_125, s1_125) / FS_OUT
    seg125 = x125[s0_125:s1_125]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t500, seg500, label="500Hz raw")
    ax.plot(t125, seg125, label="125Hz resampled")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"{record_id} | wave compare ({LEAD_FOR_QC}) | {WAVE_SECONDS}s middle segment")
    ax.legend()
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_psd_compare(record_id: str, x500: np.ndarray, x125: np.ndarray, out_png: Path) -> None:
    f500, p500 = welch(x500, fs=FS_IN, nperseg=1024, noverlap=512, detrend="constant")
    f125, p125 = welch(x125, fs=FS_OUT, nperseg=256, noverlap=128, detrend="constant")

    m500 = (f500 >= 0) & (f500 <= PSD_FMAX)
    m125 = (f125 >= 0) & (f125 <= PSD_FMAX)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(f500[m500], p500[m500], label="500Hz PSD")
    ax.plot(f125[m125], p125[m125], label="125Hz PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"{record_id} | PSD compare ({LEAD_FOR_QC}) | 0–{PSD_FMAX}Hz")
    ax.legend()
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    root = project_root()

    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    data_dir = root / "data" / "physionet" / "challenge-2011" / "set-a"
    resampled_dir = root / "artifacts" / "resampled_125"
    out_dir = root / "artifacts" / "qc" / "resample_125"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    train_ids = df.loc[df["split"] == "train", "record_id"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    n = min(N_QC, len(train_ids))
    sample_ids = rng.choice(train_ids, size=n, replace=False).tolist()

    print(f"[QC Task 1.3] sampled {n} train records (seed={SEED}):")
    for rid in sample_ids:
        print(" -", rid)

    for rid in sample_ids:
        sig500, fs, sig_name = load_record_500(rid, data_dir)
        if int(round(fs)) != FS_IN:
            raise ValueError(f"{rid}: expected fs={FS_IN}, got fs={fs}")
        sig12_500 = ensure_lead_order(sig500, sig_name)

        sig12_125 = load_npz_125(rid, resampled_dir)

        if LEAD_FOR_QC not in LEADS_12:
            raise ValueError("LEAD_FOR_QC must be one of the 12 standard leads")
        lead_idx = LEADS_12.index(LEAD_FOR_QC)

        x500 = sig12_500[:, lead_idx]
        x125 = sig12_125[:, lead_idx]

        wave_png = out_dir / f"{rid}_wave_compare_2s.png"
        psd_png = out_dir / f"{rid}_psd_compare.png"

        plot_wave_compare(rid, x500, x125, wave_png)
        plot_psd_compare(rid, x500, x125, psd_png)

        print(f"[saved] {wave_png.name}, {psd_png.name}")

    print(f"\nQC outputs -> {out_dir}")
    print("Expect per record:")
    print(" - {record_id}_wave_compare_2s.png")
    print(" - {record_id}_psd_compare.png")


if __name__ == "__main__":
    main()
