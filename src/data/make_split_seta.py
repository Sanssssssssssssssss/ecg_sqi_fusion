# src/data/make_split_seta.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import project_root


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    split_ratio: tuple[float, float, float] = (0.70, 0.15, 0.15)


LABEL_MAP = {
    "acceptable": +1,
    "unacceptable": -1,
}


def stratified_split_indices(y: np.ndarray, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """
    Record-level stratified split (train/val/test) by y.
    Returns indices (positions) for each split.
    """
    rng = np.random.default_rng(cfg.seed)
    train_r, val_r, test_r = cfg.split_ratio
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"split_ratio must sum to 1.0, got {cfg.split_ratio}")

    idx_train, idx_val, idx_test = [], [], []

    for cls in sorted(np.unique(y)):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(train_r * n))
        n_val = int(round(val_r * n))
        # remainder -> test
        idx_train.extend(cls_idx[:n_train])
        idx_val.extend(cls_idx[n_train:n_train + n_val])
        idx_test.extend(cls_idx[n_train + n_val:])

    return {
        "train": np.array(idx_train, dtype=int),
        "val": np.array(idx_val, dtype=int),
        "test": np.array(idx_test, dtype=int),
    }


def plot_label_counts(df_split: pd.DataFrame, out_png: Path) -> None:
    table = (
        df_split.groupby(["split", "y"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    ax = table.plot(kind="bar")
    ax.set_title("Label counts by split (y=+1 acceptable, y=-1 unacceptable)")
    ax.set_xlabel("split")
    ax.set_ylabel("count")
    ax.legend(title="y")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    cfg = SplitConfig(seed=0, split_ratio=(0.70, 0.15, 0.15))

    root = project_root()
    manifest_path = root / "artifacts" / "manifests" / "manifest_challenge2011_seta.csv"
    out_path = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    qc_png = root / "artifacts" / "report_figs" / "split_seta_seed0_label_counts.png"

    print(f"[Task 1.1] Reading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)

    # --- sanity: required columns from your manifest script ---
    required = ["record_id", "quality_record"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"manifest missing column '{c}', got columns={list(df.columns)}")

    print(f"Manifest rows: {len(df)}")
    print("Quality counts (manifest):")
    print(df["quality_record"].value_counts(dropna=False))

    # --- filter unknown and map labels ---
    df_f = df[df["quality_record"].isin(["acceptable", "unacceptable"])].copy()
    dropped = len(df) - len(df_f)
    print(f"\nFiltered to acceptable/unacceptable: {len(df_f)} rows (dropped {dropped} unknown/other)")

    df_f["y"] = df_f["quality_record"].map(LABEL_MAP).astype(int)

    # --- stratified split ---
    y = df_f["y"].to_numpy()
    split_idx = stratified_split_indices(y, cfg)

    df_f["split"] = ""
    for name, idx in split_idx.items():
        df_f.iloc[idx, df_f.columns.get_loc("split")] = name

    # --- build output ---
    out_df = pd.DataFrame(
        {
            "record_id": df_f["record_id"].astype(str),
            "y": df_f["y"].astype(int),
            "split": df_f["split"].astype(str),
            "seed": cfg.seed,
        }
    )

    # --- QC: label counts ---
    print("\n=== QC: label counts by split ===")
    qc_table = out_df.groupby(["split", "y"]).size().unstack(fill_value=0).sort_index()
    print(qc_table)

    # --- QC: record_id disjoint ---
    r_train = set(out_df.loc[out_df["split"] == "train", "record_id"])
    r_val = set(out_df.loc[out_df["split"] == "val", "record_id"])
    r_test = set(out_df.loc[out_df["split"] == "test", "record_id"])
    inter_tv = r_train & r_val
    inter_tt = r_train & r_test
    inter_vt = r_val & r_test
    print("\n=== QC: record_id intersection sizes ===")
    print(f"train∩val={len(inter_tv)}, train∩test={len(inter_tt)}, val∩test={len(inter_vt)}")
    if inter_tv or inter_tt or inter_vt:
        raise AssertionError("Record leakage detected across splits!")

    # --- save outputs ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved split CSV -> {out_path} (rows={len(out_df)})")

    plot_label_counts(out_df, qc_png)
    print(f"Saved QC plot -> {qc_png}")

    # quick assert ratios (roughly)
    print("\n=== QC: split sizes ===")
    print(out_df["split"].value_counts())


if __name__ == "__main__":
    main()
