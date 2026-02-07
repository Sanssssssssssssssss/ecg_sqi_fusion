# src/preprocess/qc_resample_alias_check.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import decimate, filtfilt, welch, firwin, kaiserord, butter

from src.utils.paths import project_root


# ---------- fixed config ----------
SEED = 0
N_QC = 3
FS_IN = 500
FS_OUT = 125
Q = FS_IN // FS_OUT  # 4
PSD_FMAX = 40.0
LEAD_FOR_QC = "II"
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# reference "steep" LPF design
PASS_HZ = 40.0
STOP_HZ = 62.5  # Nyquist of 125Hz
TAPS = 801       # large => steep (odd preferred)
W_PASS = 1.0
W_STOP = 80.0    # heavy stopband weight => strong attenuation

# thresholds (very conservative)
TH_PSD_REL_RMSE = 0.05   # 5% relative RMSE in 0-40Hz PSD
TH_TIME_REL_RMS = 0.05   # 5% relative RMS error after lowpass region
# ---------------------------------


def ensure_lead_order(sig: np.ndarray, sig_name: list[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(sig_name)}
    idxs = [name_to_idx[l] for l in LEADS_12]
    return sig[:, idxs]


def load_record_500(record_id: str, data_dir: Path) -> np.ndarray:
    rec = wfdb.rdrecord(str(data_dir / record_id), physical=True)
    if int(round(rec.fs)) != FS_IN:
        raise ValueError(f"{record_id}: expected fs={FS_IN}, got fs={rec.fs}")
    sig12 = ensure_lead_order(rec.p_signal, list(rec.sig_name))
    return sig12


def design_steep_lpf_fir(fs: float) -> np.ndarray:
    """
    Stable reference FIR using Kaiser-window design.

    Spec:
      passband <= PASS_HZ
      stopband starts at STOP_HZ (Nyquist of 125Hz)
      transition width = STOP_HZ - PASS_HZ

    We choose a strong attenuation (e.g., 90 dB) to make it "conservative".
    """
    nyq = fs / 2.0
    if STOP_HZ >= nyq:
        raise ValueError("STOP_HZ must be < Nyquist for FIR design.")

    width = (STOP_HZ - PASS_HZ) / nyq  # normalized transition width (0..1)
    if width <= 0:
        raise ValueError("Need STOP_HZ > PASS_HZ for a transition band.")

    # "conservative" attenuation: 90 dB (very strong stopband)
    ripple_db = 90.0

    # kaiserord returns (numtaps, beta)
    numtaps, beta = kaiserord(ripple_db, width)

    # ensure odd taps for nicer symmetry / filtfilt behavior
    if numtaps % 2 == 0:
        numtaps += 1

    # also cap to something reasonable (optional)
    numtaps = int(min(max(numtaps, 401), 5001))

    taps = firwin(
        numtaps=numtaps,
        cutoff=PASS_HZ,      # cutoff at passband edge
        window=("kaiser", beta),
        fs=fs,
        pass_zero="lowpass",
    )
    return taps.astype(np.float64)



def ref_resample(sig500: np.ndarray, fir_taps: np.ndarray) -> np.ndarray:
    """
    Reference pipeline:
      1) steep LPF with zero-phase
      2) decimate by picking every Q-th sample
    """
    y = filtfilt(fir_taps, [1.0], sig500)
    return y[::Q]


def decimate_resample(sig500: np.ndarray) -> np.ndarray:
    """
    Current pipeline:
      decimate(q=4, ftype='fir', zero_phase=True)
    """
    return decimate(sig500, q=Q, ftype="fir", zero_phase=True)


def psd_0_40(x: np.ndarray, fs: float):
    f, p = welch(x, fs=fs, nperseg=min(1024, len(x)), noverlap=min(512, max(0, len(x)//2)), detrend="constant")
    m = (f >= 0) & (f <= PSD_FMAX)
    return f[m], p[m]


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sqrt(np.mean((a - b) ** 2))
    den = np.sqrt(np.mean(a ** 2)) + 1e-12
    return float(num / den)

def lowpass_40hz_125(x: np.ndarray, fs: float = FS_OUT, fc: float = 40.0) -> np.ndarray:
    """
    40Hz low-pass filter at 125Hz (zero phase) for time-domain interpolation
    where bandwidth is the primary concern, as per the paper.
    """
    wn = fc / (fs / 2.0)
    b, a = butter(N=4, Wn=wn, btype="lowpass")
    return filtfilt(b, a, x)


def main() -> None:
    root = project_root()
    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    data_dir = root / "data" / "physionet" / "challenge-2011" / "set-a"
    out_dir = root / "artifacts" / "qc" / "resample_125_alias_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    train_ids = df.loc[df["split"] == "train", "record_id"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    n = min(N_QC, len(train_ids))
    sample_ids = rng.choice(train_ids, size=n, replace=False).tolist()

    print(f"[Alias QC] sampled {n} train records (seed={SEED}):")
    for rid in sample_ids:
        print(" -", rid)

    # design FIR once
    fir_taps = design_steep_lpf_fir(FS_IN)
    print(f"\n[Alias QC] reference FIR taps={len(fir_taps)} | pass<= {PASS_HZ}Hz | stop>= {STOP_HZ}Hz | weights pass={W_PASS}, stop={W_STOP}")

    rows = []

    lead_idx = LEADS_12.index(LEAD_FOR_QC)

    for rid in sample_ids:
        sig12 = load_record_500(rid, data_dir)
        x500 = sig12[:, lead_idx]

        x_dec = decimate_resample(x500)            # fs=125
        x_ref = ref_resample(x500, fir_taps)       # fs=125

        # align lengths (can differ by 1 due to filter edges)
        L = min(len(x_dec), len(x_ref))
        x_dec = x_dec[:L]
        x_ref = x_ref[:L]

        # PSD compare in 0-40Hz
        f, p_dec = psd_0_40(x_dec, FS_OUT)
        _, p_ref = psd_0_40(x_ref, FS_OUT)

        # time compare: focus on low-frequency component
        # (already lowpassed in ref; decimate is also lowpassed; compare directly)
        time_rel = rel_rmse(x_ref, x_dec)

        # PSD relative RMSE
        psd_rel = rel_rmse(p_ref, p_dec)

        print(f"\n[{rid}] lead={LEAD_FOR_QC} len125={L}")
        print(f"  time_rel_RMS={time_rel:.4f}  |  psd_rel_RMSE_0_40={psd_rel:.4f}")

        verdict = "PASS" if (psd_rel <= TH_PSD_REL_RMSE) else "WARN"
        print(f"  verdict={verdict} (thresholds: psd<={TH_PSD_REL_RMSE})")

        # plots: PSD overlay + time segment overlay
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f, p_dec, label="decimate PSD")
        ax.plot(f, p_ref, label="ref steep-LPF+downsample PSD")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"{rid} | PSD 0–40Hz | lead={LEAD_FOR_QC} | psd_rel={psd_rel:.3f} | {verdict}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{rid}_psd_overlay.png", dpi=200)
        plt.close(fig)

        # time overlay: middle 2 seconds
        mid = L // 2
        half = int(round(1.0 * FS_OUT))  # 1s each side => 2s
        s0 = max(0, mid - half)
        s1 = min(L, mid + half)
        t = np.arange(s0, s1) / FS_OUT

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, x_dec[s0:s1], label="decimate")
        ax.plot(t, x_ref[s0:s1], label="ref steep-LPF+downsample")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(f"{rid} | time overlay (2s mid) | lead={LEAD_FOR_QC} | time_rel={time_rel:.3f} | {verdict}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{rid}_time_overlay.png", dpi=200)
        plt.close(fig)

        # ---------- extra diagnostics: diff + relative error ----------
        eps = 1e-12
        p_diff = p_dec - p_ref
        p_relerr = np.abs(p_diff) / (np.abs(p_ref) + eps)

        # lowpass both to 40Hz then compare in time domain (more meaningful)
        x_dec_low = lowpass_40hz_125(x_dec, fs=FS_OUT, fc=40.0)
        x_ref_low = lowpass_40hz_125(x_ref, fs=FS_OUT, fc=40.0)
        time_rel_low40 = rel_rmse(x_ref_low, x_dec_low)

        # 1) PSD diff
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f, p_diff)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD diff (dec - ref)")
        ax.set_title(f"{rid} | PSD diff 0–40Hz | lead={LEAD_FOR_QC}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{rid}_psd_diff.png", dpi=200)
        plt.close(fig)

        # 2) PSD relative error
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f, p_relerr)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|Pdec-Pref| / |Pref|")
        ax.set_title(f"{rid} | PSD rel. error 0–40Hz | lead={LEAD_FOR_QC} | psd_relRMSE={psd_rel:.3f}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{rid}_psd_relerr.png", dpi=200)
        plt.close(fig)

        # time diff arrays (use the same s0:s1 window you already computed)
        diff = x_dec - x_ref
        diff_low = x_dec_low - x_ref_low

        # 3) Time overlay with diff (raw 125 outputs)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, x_dec[s0:s1], label="decimate")
        ax.plot(t, x_ref[s0:s1], label="ref")
        ax.plot(t, diff[s0:s1], label="diff (dec-ref)")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(f"{rid} | time diff (2s mid) | lead={LEAD_FOR_QC}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{rid}_time_diff_2s.png", dpi=200)
        plt.close(fig)

        # 4) Time diff after lowpass 40Hz (paper bandwidth)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, x_dec_low[s0:s1], label="dec low40")
        ax.plot(t, x_ref_low[s0:s1], label="ref low40")
        ax.plot(t, diff_low[s0:s1], label="diff low40")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(
            f"{rid} | time diff low40 (2s mid) | lead={LEAD_FOR_QC} | time_rel_low40={time_rel_low40:.3f}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{rid}_time_low40_diff_2s.png", dpi=200)
        plt.close(fig)

        print(f"  extra: time_rel_low40={time_rel_low40:.4f} | saved psd_diff/psd_relerr/time_diff/time_low40_diff")

        rows.append({
            "record_id": rid,
            "lead": LEAD_FOR_QC,
            "len_125": L,
            "time_rel_rms": time_rel,
            "time_rel_rms_low40": time_rel_low40,
            "psd_rel_rmse_0_40": psd_rel,
            "verdict": verdict,
            "fir_taps": len(fir_taps),
            "pass_hz": PASS_HZ,
            "stop_hz": STOP_HZ,
            "w_stop": W_STOP,
        })

    df_out = pd.DataFrame(rows)
    out_csv = out_dir / "alias_check_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[Alias QC] saved summary -> {out_csv}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
