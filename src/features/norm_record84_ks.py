from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.paths import project_root

# =======================
# Fixed config
# =======================
SEED = 0
STD_EPS = 1e-8   # avoid std=0
# =======================


def main() -> None:
    root = project_root()

    split_csv = root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv"
    in_parquet = root / "artifacts" / "features" / "record84.parquet"
    out_dir = root / "artifacts" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stats = out_dir / "norm_stats_seed0.json"
    out_parquet = out_dir / "record84_norm.parquet"

    print(f"split_csv: {split_csv}")
    print(f"in_features: {in_parquet}")
    print(f"out_stats: {out_stats}")
    print(f"out_features: {out_parquet}")
    print(f"rule: (x - median_train) / std_train, only for *__(sSQI|kSQI) columns")
    print(f"STD_EPS={STD_EPS}")

    df_split = pd.read_csv(split_csv)
    df_feat = pd.read_parquet(in_parquet)

    # force same dtype for merge key
    df_split["record_id"] = df_split["record_id"].astype(str)
    df_feat["record_id"] = df_feat["record_id"].astype(str)

    # merge split onto feature table
    df = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="left")
    if df["split"].isna().any():
        miss = df[df["split"].isna()]["record_id"].head(10).tolist()
        raise ValueError(f"Missing split info for some record_id, examples: {miss}")

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_test = int((df["split"] == "test").sum())
    print(f"rows: train={n_train}, val={n_val}, test={n_test}, total={len(df)}")

    # select columns: only kSQI and sSQI
    ks_cols = [c for c in df.columns if c.endswith("__kSQI")]
    ss_cols = [c for c in df.columns if c.endswith("__sSQI")]
    target_cols = ks_cols + ss_cols
    if not target_cols:
        raise ValueError("No __kSQI/__sSQI columns found in record84.parquet")

    # compute stats on train only
    df_train = df.loc[df["split"] == "train", target_cols]
    med = df_train.median(axis=0, skipna=True)
    std = df_train.std(axis=0, ddof=1, skipna=True)

    # protect tiny std
    std_safe = std.copy()
    tiny = std_safe.abs() < STD_EPS
    n_tiny = int(tiny.sum())
    if n_tiny > 0:
        print(f"[WARN] tiny std columns: {n_tiny} -> clamp to 1.0 (examples below)")
        ex = std_safe[tiny].head(10)
        for k, v in ex.items():
            print(f"  {k}: std={v}")
        std_safe[tiny] = 1.0

    # apply normalization to ALL splits (train/val/test) using TRAIN stats
    df_norm = df.copy()
    df_norm[target_cols] = (df_norm[target_cols] - med) / std_safe

    # sanity checks (train median should be ~0 for each column)
    train_norm = df_norm.loc[df_norm["split"] == "train", target_cols]
    med_after = train_norm.median(axis=0, skipna=True)
    max_abs_med = float(np.max(np.abs(med_after.to_numpy())))
    print(f"sanity: max |median_train_after| over normalized cols = {max_abs_med:.6f}")
    if max_abs_med > 1e-6:
        print("[WARN] train median after norm not near 0 (unexpected but may happen with NaNs).")

    # save stats
    stats = {
        "seed": SEED,
        "method": "(x - median_train) / std_train",
        "std_eps": STD_EPS,
        "columns": target_cols,
        "median_train": {k: float(v) for k, v in med.items()},
        "std_train": {k: float(v) for k, v in std_safe.items()},
    }
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote stats -> {out_stats}")

    # write normalized parquet (drop split column to keep same schema style; or keep? drop)
    df_out = df_norm.drop(columns=["split"])
    df_out.to_parquet(out_parquet, index=False)
    print(f"wrote normalized features -> {out_parquet} rows={len(df_out)} cols={df_out.shape[1]}")

    # quick preview
    sample_cols = ["II__sSQI", "II__kSQI"]
    sample_cols = [c for c in sample_cols if c in df_out.columns]
    if sample_cols:
        print("preview (first 40 rows):")
        print(df_out[["record_id", "y"] + sample_cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
