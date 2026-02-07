# src/qrs/qc_qrs_cache.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import project_root


SEED = 0
FS = 125
N_QC = 5
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def compute_hr(rpeaks: np.ndarray, fs: int) -> float:
    if rpeaks is None or len(rpeaks) < 2:
        return np.nan
    rr = np.diff(rpeaks) / fs
    med = np.median(rr)
    if med <= 0:
        return np.nan
    return float(60.0 / med)


def load_sig125(rid: str, resampled_dir: Path) -> np.ndarray:
    z = np.load(resampled_dir / f"{rid}.npz", allow_pickle=True)
    return z["sig_125"].astype(np.float32)


def load_qrs(rid: str, qrs_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    z = np.load(qrs_dir / f"{rid}.npz", allow_pickle=True)
    r1 = [np.asarray(x, dtype=int) for x in z["rpeaks_1"].tolist()]
    r2 = [np.asarray(x, dtype=int) for x in z["rpeaks_2"].tolist()]
    tol_ms = int(z["beat_match_tol_ms"])
    return r1, r2, tol_ms


def plot_record(rid: str, sig12: np.ndarray, r1: list[np.ndarray], r2: list[np.ndarray], tol_ms: int, out_png: Path) -> None:
    n = sig12.shape[0]
    t = np.arange(n) / FS

    fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, sig12[:, i], linewidth=0.8)
        if len(r1[i]) > 0:
            ax.scatter(r1[i] / FS, sig12[r1[i], i], s=8, marker="o", label="xqrs" if i == 0 else None)
        if len(r2[i]) > 0:
            ax.scatter(r2[i] / FS, sig12[r2[i], i], s=8, marker="x", label="gqrs" if i == 0 else None)

        ax.set_ylabel(LEADS_12[i])
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle(f"{rid} | ECG+Rpeaks | fs={FS}Hz | tol={tol_ms}ms", y=0.995)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    root = project_root()
    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    resampled_dir = root / "artifacts" / "resampled_125"
    qrs_dir = root / "artifacts" / "qrs"
    out_dir = root / "artifacts" / "qc" / "qrs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    train_ids = df.loc[df["split"] == "train", "record_id"].astype(str).to_numpy()

    rng = np.random.default_rng(SEED)
    sample_ids = rng.choice(train_ids, size=min(N_QC, len(train_ids)), replace=False).tolist()

    print(f"[Task 1.4 QC] sampled {len(sample_ids)} train records (seed={SEED}):")
    for rid in sample_ids:
        print(" -", rid)

    # 1) sample plots
    for rid in sample_ids:
        sig12 = load_sig125(rid, resampled_dir)
        r1, r2, tol_ms = load_qrs(rid, qrs_dir)
        out_png = out_dir / f"{rid}_rpeaks_12lead.png"
        plot_record(rid, sig12, r1, r2, tol_ms, out_png)
        print(f"[saved] {out_png.name}")

    # 2) global stats
    all_ids = df["record_id"].astype(str).tolist()
    rows = []
    for rid in all_ids:
        try:
            r1, r2, _ = load_qrs(rid, qrs_dir)
            for li, lead in enumerate(LEADS_12):
                rows.append({"record_id": rid, "lead": lead, "detector": "xqrs", "n_beats": len(r1[li]), "hr_bpm": compute_hr(r1[li], FS)})
                rows.append({"record_id": rid, "lead": lead, "detector": "gqrs", "n_beats": len(r2[li]), "hr_bpm": compute_hr(r2[li], FS)})
        except Exception as e:
            print(f"[WARN] skip {rid}: {type(e).__name__}: {e}")

    stat_df = pd.DataFrame(rows)
    out_csv = out_dir / "qrs_summary.csv"
    stat_df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv} rows={len(stat_df)}")

    for det in ["xqrs", "gqrs"]:
        sub = stat_df[stat_df["detector"] == det].copy()

        # beats count
        fig, ax = plt.subplots(figsize=(14, 4))
        data = [sub[sub["lead"] == lead]["n_beats"].values for lead in LEADS_12]
        ax.boxplot(data, tick_labels=LEADS_12, showfliers=False)
        ax.set_title(f"Beats count distribution by lead ({det})")
        ax.set_ylabel("n_beats per 10s")
        fig.tight_layout()
        p1 = out_dir / f"beats_count_by_lead_{det}.png"
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        print(f"[saved] {p1.name}")

        # HR
        fig, ax = plt.subplots(figsize=(14, 4))
        data = [sub[sub["lead"] == lead]["hr_bpm"].dropna().values for lead in LEADS_12]
        ax.boxplot(data, tick_labels=LEADS_12, showfliers=False)
        ax.set_title(f"HR distribution by lead ({det})")
        ax.set_ylabel("HR (bpm)")
        fig.tight_layout()
        p2 = out_dir / f"hr_by_lead_{det}.png"
        fig.savefig(p2, dpi=200)
        plt.close(fig)
        print(f"[saved] {p2.name}")

    print(f"\nQC outputs -> {out_dir}")


if __name__ == "__main__":
    main()
