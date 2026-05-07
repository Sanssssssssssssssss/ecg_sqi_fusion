from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import project_root

# =======================
# Fixed config
# =======================
SEED = 0
BINS = 60
LEAD_FOR_QC = "II"   # 你也可以改成别的 lead
# =======================


def plot_two_hists(a: np.ndarray, b: np.ndarray, title: str, xlabel: str, out_png: Path) -> None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(a, bins=BINS, alpha=0.6, label="train")
    ax.hist(b, bins=BINS, alpha=0.6, label="test")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    root = project_root()

    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    in_raw = root / "artifacts" / "features" / "record84.parquet"
    in_norm = root / "artifacts" / "features" / "record84_norm.parquet"

    out_dir = root / "artifacts" / "qc" / "features_norm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[QC 1.6] split_csv: {split_csv}")
    print(f"[QC 1.6] raw:  {in_raw}")
    print(f"[QC 1.6] norm: {in_norm}")
    print(f"[QC 1.6] out_dir: {out_dir}")

    df_split = pd.read_csv(split_csv)[["record_id", "split"]]

    # force same dtype for merge key
    df_split["record_id"] = df_split["record_id"].astype(str)

    raw = pd.read_parquet(in_raw).merge(df_split, on="record_id", how="left")
    norm = pd.read_parquet(in_norm).merge(df_split, on="record_id", how="left")

    # choose columns (lead II by default)
    c_s = f"{LEAD_FOR_QC}__sSQI"
    c_k = f"{LEAD_FOR_QC}__kSQI"
    for c in (c_s, c_k):
        if c not in raw.columns or c not in norm.columns:
            raise ValueError(f"Missing column {c} in raw/norm parquet")

    # raw
    raw_train_s = raw.loc[raw["split"] == "train", c_s].to_numpy()
    raw_test_s  = raw.loc[raw["split"] == "test",  c_s].to_numpy()
    raw_train_k = raw.loc[raw["split"] == "train", c_k].to_numpy()
    raw_test_k  = raw.loc[raw["split"] == "test",  c_k].to_numpy()

    # norm
    norm_train_s = norm.loc[norm["split"] == "train", c_s].to_numpy()
    norm_test_s  = norm.loc[norm["split"] == "test",  c_s].to_numpy()
    norm_train_k = norm.loc[norm["split"] == "train", c_k].to_numpy()
    norm_test_k  = norm.loc[norm["split"] == "test",  c_k].to_numpy()

    # plots
    plot_two_hists(
        raw_train_s, raw_test_s,
        title=f"RAW {c_s}: train vs test",
        xlabel=c_s,
        out_png=out_dir / f"raw_{c_s}_train_vs_test.png"
    )
    plot_two_hists(
        norm_train_s, norm_test_s,
        title=f"NORM {c_s}: train vs test (train-median/std)",
        xlabel=c_s,
        out_png=out_dir / f"norm_{c_s}_train_vs_test.png"
    )
    plot_two_hists(
        raw_train_k, raw_test_k,
        title=f"RAW {c_k}: train vs test",
        xlabel=c_k,
        out_png=out_dir / f"raw_{c_k}_train_vs_test.png"
    )
    plot_two_hists(
        norm_train_k, norm_test_k,
        title=f"NORM {c_k}: train vs test (train-median/std)",
        xlabel=c_k,
        out_png=out_dir / f"norm_{c_k}_train_vs_test.png"
    )

    print("[QC 1.6] saved:")
    print(" -", out_dir / f"raw_{c_s}_train_vs_test.png")
    print(" -", out_dir / f"norm_{c_s}_train_vs_test.png")
    print(" -", out_dir / f"raw_{c_k}_train_vs_test.png")
    print(" -", out_dir / f"norm_{c_k}_train_vs_test.png")

    # extra numeric QC: train after norm should have median ~ 0
    med_s = float(np.nanmedian(norm_train_s[np.isfinite(norm_train_s)]))
    med_k = float(np.nanmedian(norm_train_k[np.isfinite(norm_train_k)]))
    print(f"[QC 1.6] train median after norm: {c_s}={med_s:.6f}, {c_k}={med_k:.6f}")


if __name__ == "__main__":
    main()
