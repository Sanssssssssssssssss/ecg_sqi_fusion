# src/noise/qc_noisy_cases.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import project_root


# ================== CONFIG ==================
SEED = 0

# how many per noise type (em / ma)
N_PER_TYPE = 3

FS = 500
CASE_SEC = 10.0
N_SAMPLES = int(FS * CASE_SEC)  # 5000

BALANCED_SPLIT_CSV = "artifacts/splits/split_seta_seed0_balanced.csv"

# unified 500Hz cases folder (contains both clean + noisy)
CASES_500_DIR = "artifacts/cases_500"

OUT_DIR = "artifacts/qc/noise_aug_500"

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# numeric eps for power / snr
POWER_EPS = 1e-12
# ===========================================


def _rid_str(x) -> str:
    """
    Robustly convert record_id/source_record_id to filename-friendly string.
    Handles cases like 1746739.0 -> '1746739'.
    """
    if x is None:
        return ""
    if isinstance(x, (np.integer,)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        if np.isfinite(x):
            return str(int(x))
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        return s[:-2]
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def load_case_500(record_id: str, cases_dir: Path) -> np.ndarray:
    """
    Load a 500Hz case from cases_500/{record_id}.npz with key 'sig_500'.
    """
    p = cases_dir / f"{record_id}.npz"
    z = np.load(p, allow_pickle=True)
    if "sig_500" not in z:
        raise ValueError(f"{p}: missing key 'sig_500' (this QC expects 500Hz cache)")
    sig = z["sig_500"].astype(np.float64)
    fs = int(z["fs"])
    leads = list(z["leads"].tolist())

    if fs != FS:
        raise ValueError(f"{record_id}: fs mismatch {fs}, expected {FS}")
    if sig.shape != (N_SAMPLES, 12):
        raise ValueError(f"{record_id}: shape mismatch {sig.shape}, expected {(N_SAMPLES, 12)}")
    if leads != LEADS_12:
        raise ValueError(f"{record_id}: lead order mismatch")

    return sig


def snr_db(clean: np.ndarray, added: np.ndarray) -> float:
    """
    SNR(dB) = 10 log10( mean(clean^2) / mean(added^2) )
    """
    px = float(np.mean(clean**2)) + POWER_EPS
    pv = float(np.mean(added**2)) + POWER_EPS
    return 10.0 * float(np.log10(px / pv))


def compute_snr_stats(clean12: np.ndarray, noisy12: np.ndarray) -> dict:
    """
    Return per-lead and aggregate SNR diagnostics.
    added = noisy - clean
    """
    added12 = noisy12 - clean12

    snr_lead = []
    px_lead = []
    pv_lead = []
    rms_clean = []
    rms_added = []

    for li in range(12):
        c = clean12[:, li]
        a = added12[:, li]

        px = float(np.mean(c**2)) + POWER_EPS
        pv = float(np.mean(a**2)) + POWER_EPS

        px_lead.append(px)
        pv_lead.append(pv)
        rms_clean.append(float(np.sqrt(px)))
        rms_added.append(float(np.sqrt(pv)))
        snr_lead.append(10.0 * float(np.log10(px / pv)))

    snr_lead = np.asarray(snr_lead, dtype=np.float64)

    # aggregate: pool all leads as one long vector
    snr_all = snr_db(clean12.reshape(-1), added12.reshape(-1))

    return {
        "snr_db_per_lead": snr_lead,
        "snr_db_mean": float(np.mean(snr_lead)),
        "snr_db_min": float(np.min(snr_lead)),
        "snr_db_max": float(np.max(snr_lead)),
        "snr_db_all_leads_pooled": float(snr_all),
        "px_mean_per_lead": np.asarray(px_lead, dtype=np.float64),
        "pv_mean_per_lead": np.asarray(pv_lead, dtype=np.float64),
        "rms_clean_per_lead": np.asarray(rms_clean, dtype=np.float64),
        "rms_added_per_lead": np.asarray(rms_added, dtype=np.float64),
    }


def plot_wave12_clean_vs_noisy(
    clean12: np.ndarray,
    noisy12: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    t = np.arange(N_SAMPLES) / FS
    fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(t, clean12[:, i], label="clean", linewidth=1.0)
        ax.plot(t, noisy12[:, i], label="noisy", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(LEADS_12[i])
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_snr_bar(snr_per_lead: np.ndarray, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(12), snr_per_lead)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(LEADS_12, rotation=0)
    ax.set_ylabel("SNR (dB)")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def pick_samples_by_type(df_aug: pd.DataFrame, noise_type: str, n: int, seed: int) -> pd.DataFrame:
    d = df_aug[df_aug["noise_type"].astype(str) == noise_type].copy()
    if len(d) == 0:
        return d
    n2 = min(n, len(d))
    return d.sample(n=n2, random_state=seed)


def main() -> None:
    root = project_root()
    split_csv = root / BALANCED_SPLIT_CSV
    cases_dir = root / CASES_500_DIR
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[QC500] split_csv: {split_csv}")
    print(f"[QC500] cases_500: {cases_dir}")
    print(f"[QC500] out_dir: {out_dir}")
    print(f"[QC500] fs={FS}, case={CASE_SEC}s, N={N_SAMPLES}")

    df = pd.read_csv(split_csv)

    need_cols = {"record_id", "is_augmented", "source_record_id", "noise_type", "snr_db"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"balanced split missing columns: {sorted(miss)}")

    df["record_id"] = df["record_id"].apply(_rid_str)
    df["source_record_id"] = df["source_record_id"].apply(_rid_str)

    df_aug = df[df["is_augmented"] == 1].copy()
    if len(df_aug) == 0:
        raise ValueError("No augmented rows found. Run make_balanced_noisy_cases first.")

    # show counts
    print("\n[QC500] augmented counts by noise_type:")
    print(df_aug["noise_type"].value_counts(dropna=False).to_string())

    # choose samples: em and ma each N_PER_TYPE
    samples = []
    for nt in ["em", "ma"]:
        samp = pick_samples_by_type(df_aug, nt, N_PER_TYPE, SEED)
        if len(samp) == 0:
            print(f"[QC500] WARN: no augmented rows for noise_type={nt}")
        else:
            samples.append(samp)

    df_s = (
        pd.concat(samples, ignore_index=True)
        if samples
        else df_aug.sample(n=min(5, len(df_aug)), random_state=SEED)
    )

    print(f"\n[QC500] sampled {len(df_s)} augmented records (seed={SEED}, per_type={N_PER_TYPE}):")
    for rid in df_s["record_id"].tolist():
        print(" -", rid)

    rows_debug = []

    # plot each + compute SNR diagnostics
    for _, row in df_s.iterrows():
        new_id = _rid_str(row["record_id"])
        src_id = _rid_str(row["source_record_id"])
        noise_type = str(row.get("noise_type", ""))
        snr_target = float(row.get("snr_db", np.nan))

        if not src_id:
            raise ValueError(f"{new_id}: empty source_record_id")

        clean_path = cases_dir / f"{src_id}.npz"
        noisy_path = cases_dir / f"{new_id}.npz"
        if not clean_path.exists():
            raise FileNotFoundError(f"missing clean npz in cases_500: {clean_path}")
        if not noisy_path.exists():
            raise FileNotFoundError(f"missing noisy npz in cases_500: {noisy_path}")

        clean12 = load_case_500(src_id, cases_dir)
        noisy12 = load_case_500(new_id, cases_dir)

        stats = compute_snr_stats(clean12, noisy12)

        # print a concise line to terminal
        print(
            f"[SNR] {new_id} ({noise_type}) target={snr_target:.2f} dB | "
            f"mean={stats['snr_db_mean']:.2f} min={stats['snr_db_min']:.2f} max={stats['snr_db_max']:.2f} "
            f"pooled={stats['snr_db_all_leads_pooled']:.2f}"
        )

        # save wave plot
        title = (
            f"{new_id} | source={src_id} | noise={noise_type} | "
            f"snr_target={snr_target:.2f}dB | "
            f"snr_mean={stats['snr_db_mean']:.2f} (min {stats['snr_db_min']:.2f}, max {stats['snr_db_max']:.2f}) | "
            f"pooled={stats['snr_db_all_leads_pooled']:.2f} | fs={FS}"
        )
        out_png = out_dir / f"{new_id}_wave12_clean_vs_noisy.png"
        plot_wave12_clean_vs_noisy(clean12, noisy12, title, out_png)

        # save snr bar plot
        out_snr_png = out_dir / f"{new_id}_snr_bar.png"
        plot_snr_bar(
            stats["snr_db_per_lead"],
            title=f"{new_id} | snr_target={snr_target:.2f}dB | mean={stats['snr_db_mean']:.2f}dB",
            out_png=out_snr_png,
        )

        # build debug row (wide columns for quick filtering)
        debug = {
            "record_id": new_id,
            "source_record_id": src_id,
            "noise_type": noise_type,
            "snr_target_db": snr_target,
            "snr_mean_db": stats["snr_db_mean"],
            "snr_min_db": stats["snr_db_min"],
            "snr_max_db": stats["snr_db_max"],
            "snr_pooled_db": stats["snr_db_all_leads_pooled"],
            "wave_png": str(out_png),
            "snr_png": str(out_snr_png),
        }
        for li, lead in enumerate(LEADS_12):
            debug[f"{lead}__snr_db"] = float(stats["snr_db_per_lead"][li])
            debug[f"{lead}__px"] = float(stats["px_mean_per_lead"][li])
            debug[f"{lead}__pv_added"] = float(stats["pv_mean_per_lead"][li])
            debug[f"{lead}__rms_clean"] = float(stats["rms_clean_per_lead"][li])
            debug[f"{lead}__rms_added"] = float(stats["rms_added_per_lead"][li])

        rows_debug.append(debug)

        print(f"[saved] {out_png.name}")
        print(f"[saved] {out_snr_png.name}")

    # save debug table
    df_dbg = pd.DataFrame(rows_debug)
    out_csv = out_dir / "qc_noisy_cases_debug.csv"
    df_dbg.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv} rows={len(df_dbg)}")

    print(f"\nQC outputs -> {out_dir}")


if __name__ == "__main__":
    main()
