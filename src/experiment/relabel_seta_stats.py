from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import project_root

# reuse your existing modules
from src.preprocess.resample_125 import run as run_resample
from src.qrs.run_qrs_cache import run as run_qrs
from src.features.make_record84 import run as run_record84


# ============================================================
# Config
# ============================================================

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"]

SQI_LIST = ["iSQI", "bSQI", "pSQI",
            "sSQI", "kSQI", "fSQI", "basSQI"]

SEED = 0


# ============================================================
# Plotting
# ============================================================


def plot_record_metric_jitter(sdf: pd.DataFrame, col: str, out_png: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    good = sdf.loc[sdf["y"] == 1, col].to_numpy(dtype=float)
    bad  = sdf.loc[sdf["y"] == -1, col].to_numpy(dtype=float)

    xg = 0.0 + 0.15 * rng.standard_normal(len(good))
    xb = 1.0 + 0.15 * rng.standard_normal(len(bad))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xg, good, s=8, alpha=0.6, label="acceptable")
    ax.scatter(xb, bad,  s=8, alpha=0.6, label="unacceptable")
    ax.set_xticks([0, 1], labels=["acc", "unacc"])
    ax.set_ylabel(col)
    ax.set_title(f"Record-level: {col}")
    ax.legend(loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_sqi_12lead(df: pd.DataFrame,
                    sqi: str,
                    out_png: Path) -> None:
    """
    One SQI → 12 subplots (12 leads)
    Each subplot: scatter all samples (good vs bad)
    """

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, lead in enumerate(LEADS_12):
        ax = axes[i]

        col = f"{lead}__{sqi}"

        good = df[df["y"] == 1][col]
        bad = df[df["y"] == -1][col]

        ax.scatter(np.arange(len(good)), good,
                   s=8, alpha=0.6, label="acceptable")

        ax.scatter(np.arange(len(bad)), bad,
                   s=8, alpha=0.6, label="unacceptable")

        ax.set_title(lead)
        ax.set_xlabel("sample index")
        ax.set_ylabel(sqi)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{sqi} distribution across 12 leads")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_sqi_12lead_jitter(df: pd.DataFrame, sqi: str, out_png: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, lead in enumerate(LEADS_12):
        ax = axes[i]
        col = f"{lead}__{sqi}"

        y_good = df.loc[df["y"] == 1, col].to_numpy()
        y_bad  = df.loc[df["y"] == -1, col].to_numpy()

        # jitter around x=0 and x=1
        x_good = 0.0 + 0.15 * rng.standard_normal(len(y_good))
        x_bad  = 1.0 + 0.15 * rng.standard_normal(len(y_bad))

        ax.scatter(x_good, y_good, s=8, alpha=0.6, label="acceptable")
        ax.scatter(x_bad,  y_bad,  s=8, alpha=0.6, label="unacceptable")

        ax.set_title(lead)
        ax.set_xticks([0, 1], labels=["acc", "unacc"])
        ax.set_ylabel(sqi)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{sqi} distribution across 12 leads (jittered)")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_lead_7sqi_jitter(
    df: pd.DataFrame,
    lead: str,
    sqi_list: list[str],
    out_png: Path,
    seed: int = 0,
) -> None:
    """
    One lead → 7 subplots (one per SQI)
    Each subplot: jittered scatter for acc vs unacc
    """
    rng = np.random.default_rng(seed)

    # 7 plots -> 2x4 grid (leave last empty)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, sqi in enumerate(sqi_list):
        ax = axes[i]
        col = f"{lead}__{sqi}"
        if col not in df.columns:
            ax.set_title(f"{sqi} (missing)")
            ax.axis("off")
            continue

        y_good = df.loc[df["y"] == 1, col].to_numpy()
        y_bad  = df.loc[df["y"] == -1, col].to_numpy()

        x_good = 0.0 + 0.15 * rng.standard_normal(len(y_good))
        x_bad  = 1.0 + 0.15 * rng.standard_normal(len(y_bad))

        ax.scatter(x_good, y_good, s=8, alpha=0.6, label="acceptable")
        ax.scatter(x_bad,  y_bad,  s=8, alpha=0.6, label="unacceptable")

        ax.set_title(sqi)
        ax.set_xticks([0, 1], labels=["acc", "unacc"])
        ax.set_ylabel("value")

    # hide unused last subplot (axes[7])
    for j in range(len(sqi_list), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{lead}: 7 SQIs (jittered)")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def _fmt(x: float | int | None, w: int = 7, nd: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return " " * (w - 1) + "-"
        return f"{float(x):{w}.{nd}f}"
    except Exception:
        return " " * (w - 1) + "?"

def print_sample_block(i: int, rid: str, y: int, row: dict, leads: list[str]) -> None:
    """
    Print 12-lead SQIs for one record in a readable aligned block.
    row: dict-like, keys like 'II__bSQI'
    """
    header = f"sample {i:03d} {rid}: y={y}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    print(
        f"{'lead':>4} | "
        f"{'iSQI':>7} {'bSQI':>7} {'pSQI':>7} {'sSQI':>7} {'kSQI':>7} {'fSQI':>7} {'bas':>7}"
    )
    print("-" * 4 + "-+-" + "-" * 7 * 7 + "-" * 6)  # nice divider

    for ld in leads:
        iSQI = row.get(f"{ld}__iSQI", np.nan)
        bSQI = row.get(f"{ld}__bSQI", np.nan)
        pSQI = row.get(f"{ld}__pSQI", np.nan)
        sSQI = row.get(f"{ld}__sSQI", np.nan)
        kSQI = row.get(f"{ld}__kSQI", np.nan)
        fSQI = row.get(f"{ld}__fSQI", np.nan)
        bas  = row.get(f"{ld}__basSQI", np.nan)

        print(
            f"{ld:>4} | "
            f"{_fmt(iSQI)} {_fmt(bSQI)} {_fmt(pSQI)} {_fmt(sSQI)} {_fmt(kSQI)} {_fmt(fSQI)} {_fmt(bas)}"
        )

def print_n_sample_blocks(
    df: pd.DataFrame,
    n: int = 5,
    seed: int = 0,
    mode: str = "random",      # "head" | "random"
    y_filter: int | None = None,  # 1 or -1
    split_filter: str | None = None,  # "train" | "val" | "test" (if df has 'split')
    leads: list[str] = LEADS_12,
) -> None:
    d = df.copy()

    if split_filter is not None and "split" in d.columns:
        d = d[d["split"].astype(str) == str(split_filter)]

    if y_filter is not None:
        d = d[d["y"].astype(int) == int(y_filter)]

    if len(d) == 0:
        print("\n[print] No rows after filtering.")
        return

    if mode == "random":
        d = d.sample(n=min(n, len(d)), random_state=seed)
    else:
        d = d.head(n)

    # print blocks
    for i, (_, r) in enumerate(d.reset_index(drop=True).iterrows(), start=1):
        rid = str(r.get("record_id", "NA"))
        y = int(r.get("y", 0))
        print_sample_block(i=i, rid=rid, y=y, row=r.to_dict(), leads=leads)

# ============================================================
# Main Pipeline
# ============================================================

def main():

    root = project_root()

    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"

    out_root = root / "artifacts" / "relabel_stats"
    resampled_dir = out_root / "resampled_125"
    qrs_dir = out_root / "qrs"
    feature_dir = out_root / "features"
    fig_dir = out_root / "figures"
    csv_dir = out_root / "csv"

    for d in [resampled_dir, qrs_dir, feature_dir, fig_dir, csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=== RELABEL SET-A STATS PIPELINE ===")

    # # --------------------------------------------------------
    # # 1) Resample to 125 Hz
    # # --------------------------------------------------------
    # run_resample({
    #     "force": True,
    #     "split_csv": split_csv,
    #     "out_dir": resampled_dir,
    # })

    # # --------------------------------------------------------
    # # 2) QRS detection
    # # --------------------------------------------------------
    # run_qrs({
    #     "force": True,
    #     "split_csv": split_csv,
    #     "resampled_dir": resampled_dir,
    #     "out_dir": qrs_dir,
    # })

    # # --------------------------------------------------------
    # # 3) Compute 84 features
    # # --------------------------------------------------------
    # run_record84({
    #     "force": True,
    #     "split_csv": split_csv,
    #     "resampled_dir": resampled_dir,
    #     "qrs_dir": qrs_dir,
    #     "out_dir": feature_dir,
    #     "print_n": 0,
    #     "print_all_leads": False,
    # })

    # --------------------------------------------------------
    # 4) Load feature parquet
    # --------------------------------------------------------
    df = pd.read_parquet(feature_dir / "record84.parquet")

    print("Loaded record84:", df.shape)

    # save raw copy
    df.to_csv(csv_dir / "record84_full.csv", index=False)

    # --------------------------------------------------------
    # 5) Plot 7 SQI big figures
    # --------------------------------------------------------
    for sqi in SQI_LIST:
        out_png = fig_dir / f"{sqi}_12lead_scatter.png"
        print("Plotting:", sqi)
        # plot_sqi_12lead(df, sqi, out_png)
        plot_sqi_12lead_jitter(df, sqi, out_png)

    print("\nDone.")
    print("Outputs ->", out_root)

    # --------------------------------------------------------
    # 6) Plot 12 lead figures: each lead has 7 SQIs
    # --------------------------------------------------------
    lead_fig_dir = fig_dir / "by_lead"
    lead_fig_dir.mkdir(parents=True, exist_ok=True)

    for lead in LEADS_12:
        out_png = lead_fig_dir / f"{lead}_7sqi_jitter.png"
        print("Plotting lead summary:", lead)
        plot_lead_7sqi_jitter(df, lead, SQI_LIST, out_png, seed=SEED)

    # --------------------------------------------------------
    # 6) Print some readable blocks
    # --------------------------------------------------------
    print("\n=== PRINT SAMPLE BLOCKS (mixed) ===")
    print_n_sample_blocks(df, n=3, seed=SEED, mode="random")

    print("\n=== PRINT SAMPLE BLOCKS (acceptable only) ===")
    print_n_sample_blocks(df, n=10, seed=SEED, mode="random", y_filter=1)

    print("\n=== PRINT SAMPLE BLOCKS (unacceptable only) ===")
    print_n_sample_blocks(df, n=10, seed=SEED, mode="random", y_filter=-1)


if __name__ == "__main__":
    main()
