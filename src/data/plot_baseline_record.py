# src/data/plot_baseline_fixed.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# matplotlib: force non-GUI backend (no Tk needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from src.utils.paths import project_root

RECORD_ID = "1007823"   # <- change id here
LEAD_NAME = "I"        # <- e.g. "II", "V2", ...
SECONDS = 10            # <- plot duration (sec)
BASELINE_CUTOFF_HZ = 1.0  # paper figure: 1 Hz low-pass, 2nd order, zero-phase

# paths (fixed)
DATA_DIR_REL = "data/physionet/challenge-2011/set-a"
MANIFEST_REL = "artifacts/manifests/manifest_challenge2011_seta.csv"
OUT_DIR_REL = "artifacts/report_figs"
# =========================


def butter_lowpass_2nd(fc_hz: float, fs_hz: float):
    wn = fc_hz / (fs_hz / 2.0)
    b, a = butter(N=2, Wn=wn, btype="lowpass")
    return b, a


def baseline_lpf_zero_phase(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    b, a = butter_lowpass_2nd(fc_hz=fc, fs_hz=fs)
    return filtfilt(b, a, x)


def read_quality_from_manifest(manifest_csv: Path, record_id: str) -> tuple[str, int]:
    df = pd.read_csv(manifest_csv)
    row = df.loc[df["record_id"].astype(str) == str(record_id)]
    if row.empty:
        return "unknown", 0
    q = str(row.iloc[0]["quality_record"])
    y = +1 if q == "acceptable" else -1 if q == "unacceptable" else 0
    return q, y


def main() -> None:
    root = project_root()

    rec_path = root / DATA_DIR_REL / RECORD_ID  # wfdb uses path without extension
    manifest_csv = root / MANIFEST_REL
    out_dir = root / OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_baseline_fixed] record_id={RECORD_ID}")
    print(f"[plot_baseline_fixed] reading WFDB record: {rec_path}")

    # ---- read WFDB (.hea + .dat) ----
    rec = wfdb.rdrecord(str(rec_path), physical=True)
    fs = float(rec.fs)
    sig = rec.p_signal  # (N, n_sig) in physical units (mV typically)
    sig_len = int(sig.shape[0])
    n_sig = int(sig.shape[1])
    sig_name = list(rec.sig_name) if rec.sig_name is not None else [f"ch{i}" for i in range(n_sig)]

    # ---- print header-like summary ----
    # Similar to: "1007823 12 500 5000"
    print(f"\n[.hea summary]")
    print(f"{RECORD_ID} {n_sig} {int(round(fs))} {sig_len}")

    # Try to print per-channel info comparable to the .hea lines
    # wfdb gives per-channel gain/units in rec.adc_gain, rec.units, etc.
    # Not all fields are always present depending on read mode.
    units = list(getattr(rec, "units", [""] * n_sig))
    gains = list(getattr(rec, "adc_gain", [np.nan] * n_sig))
    baseline = list(getattr(rec, "baseline", [np.nan] * n_sig))

    for i in range(n_sig):
        u = units[i] if i < len(units) else ""
        g = gains[i] if i < len(gains) else np.nan
        b0 = baseline[i] if i < len(baseline) else np.nan
        # Format similar-ish to: "1007823.dat 16 200/mV ... lead"
        # We can't perfectly reconstruct every integer field from the .hea text,
        # but we print the core ones: gain/unit/baseline/name.
        print(f"{RECORD_ID}.dat ch{i:02d} gain={g} unit={u} baseline={b0} name={sig_name[i]}")

    # If comments exist in header, print a few (age/sex usually in comments)
    comments = getattr(rec, "comments", None)
    if comments:
        print("\n[.hea comments]")
        for c in comments[:5]:
            print(c)

    # ---- quality from manifest ----
    q, y = read_quality_from_manifest(manifest_csv, RECORD_ID)
    print(f"\n[manifest] quality_record={q}, y={y}")

    # ---- pick lead ----
    if LEAD_NAME not in sig_name:
        raise ValueError(f"Lead '{LEAD_NAME}' not found. Available leads: {sig_name}")
    lead_idx = sig_name.index(LEAD_NAME)
    print(f"[plot] using lead={LEAD_NAME} (idx={lead_idx})")

    # ---- crop ----
    n_plot = min(int(round(SECONDS * fs)), sig_len)
    t = np.arange(n_plot) / fs
    x = sig[:n_plot, lead_idx]

    # ---- baseline ----
    bl = baseline_lpf_zero_phase(x, fs=fs, fc=BASELINE_CUTOFF_HZ)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x, label="ECG")
    ax.plot(t, bl, "--",linewidth=2, label="baseline (LPF 1Hz, 2nd, zero-phase)")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"record={RECORD_ID} | lead={LEAD_NAME} | quality={q} (y={y})")
    ax.legend()
    fig.tight_layout()

    out_png = out_dir / f"baseline_{RECORD_ID}_{LEAD_NAME}_{SECONDS}s.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"\n[saved] {out_png}")


if __name__ == "__main__":
    main()
