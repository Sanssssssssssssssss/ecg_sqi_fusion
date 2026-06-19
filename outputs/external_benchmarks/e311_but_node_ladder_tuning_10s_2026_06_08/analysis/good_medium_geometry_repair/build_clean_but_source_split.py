"""Build a stratified clean-BUT source split for cross-dataset training.

This is experiment-only code.  It keeps the curated fixed-10s clean rows from
``margin_ge_5s_drop_outlier`` and rewrites only split labels into an 80/10/10
source train/val/test split stratified by record, class, and original_region.
The external cross-dataset target remains PTB synthetic; this source-test split
is only a clean-BUT sanity check.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
PROTOCOL_ROOT = OUT_ROOT / "analysis" / "good_medium_geometry_repair" / "clean_but_protocols"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "clean_but_protocols"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def link_or_copy(src: Path, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def assign_source_split(atlas: pd.DataFrame, seed: int, train_frac: float, val_frac: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    split = np.full(len(atlas), "train", dtype=object)
    strata = atlas[["record_id", "class_name", "original_region"]].astype(str).agg("|".join, axis=1)
    for _, idx_series in pd.Series(np.arange(len(atlas))).groupby(strata):
        idx = np.asarray(idx_series.to_numpy(), dtype=np.int64).copy()
        rng.shuffle(idx)
        n = int(idx.size)
        n_val = max(1, int(round(n * float(val_frac)))) if n >= 3 else 0
        n_test = max(1, int(round(n * (1.0 - float(train_frac) - float(val_frac))))) if n >= 5 else 0
        if n_val + n_test >= n:
            n_val = 1 if n >= 3 else 0
            n_test = 1 if n >= 5 else 0
        split[idx[:n_val]] = "val"
        split[idx[n_val : n_val + n_test]] = "test"
    return split


def counts_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, sub in frame.groupby("split", dropna=False):
        row = {"split": split, "n": int(len(sub))}
        for cls in ("good", "medium", "bad"):
            row[cls] = int((sub["class_name"].astype(str) == cls).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split")


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-policy", type=str, default="margin_ge_5s_drop_outlier")
    parser.add_argument("--out-policy", type=str, default="margin_ge_5s_drop_outlier_source80_seed20260619")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = PROTOCOL_ROOT / args.source_policy
    out = PROTOCOL_ROOT / args.out_policy
    if not src.exists():
        raise FileNotFoundError(src)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(src / "metadata.csv")
    atlas = pd.read_csv(src / "original_region_atlas.csv")
    if len(meta) != len(atlas):
        raise ValueError(f"metadata/atlas row mismatch: {len(meta)} vs {len(atlas)}")

    split = assign_source_split(atlas, int(args.seed), float(args.train_frac), float(args.val_frac))
    meta = meta.copy()
    atlas = atlas.copy()
    meta["split"] = split
    atlas["split"] = split
    meta["source_split_policy"] = args.out_policy
    atlas["source_split_policy"] = args.out_policy
    meta["source_split_seed"] = int(args.seed)
    atlas["source_split_seed"] = int(args.seed)

    link_or_copy(src / "signals.npz", out / "signals.npz", overwrite=bool(args.overwrite))
    shutil.copy2(src / "window_segment_margins.csv", out / "window_segment_margins.csv")
    meta.to_csv(out / "metadata.csv", index=False)
    atlas.to_csv(out / "original_region_atlas.csv", index=False)
    counts = counts_table(atlas)
    counts.to_csv(out / "split_counts.csv", index=False)

    audit = {
        "policy": args.out_policy,
        "source_policy": args.source_policy,
        "split_strategy": "window_level_stratified_by_record_class_region_source_80_10_10",
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "n": int(len(atlas)),
        "counts": counts.to_dict(orient="records"),
    }
    write_json(out / "audit.json", audit)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / f"{args.out_policy}_audit.json", audit)
    report = ["# Clean BUT Source Split", "", f"Policy: `{args.out_policy}`", "", markdown_table(counts)]
    (REPORT_DIR / f"{args.out_policy}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
