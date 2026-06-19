"""Build stratified clean-BUT CV folds from a materialized clean protocol.

This is experiment-only code.  It keeps the same cleaned fixed-10s rows and
only rewrites the train/val/test split labels.  The folds are window-level,
stratified by record, class, and original_region; they are not record-heldout
external tests.  The purpose is to make the clean-BUT validation/test slices
balanced enough to judge model stability after low-confidence/outlier windows
have been removed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
DEFAULT_PROTOCOL_ROOT = ANALYSIS_DIR / "clean_but_protocols"
DEFAULT_REPORT_DIR = REPORT_DIR / "clean_but_protocols"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def counts_table(atlas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, sub in atlas.groupby("split", sort=True):
        row: dict[str, Any] = {
            "split": split,
            "n": int(len(sub)),
            "records": int(sub["record_id"].nunique()),
        }
        for cls in ("good", "medium", "bad"):
            row[cls] = int((sub["class_name"].astype(str) == cls).sum())
        for region, n in sub["original_region"].astype(str).value_counts().items():
            row[f"region_{region}"] = int(n)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    frame = df.copy()
    cols = [str(c) for c in frame.columns]
    frame.columns = cols
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def assign_folds(atlas: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_id = np.full(len(atlas), -1, dtype=np.int64)
    strata_cols = ["record_id", "class_name", "original_region"]
    for _, idx in atlas.groupby(strata_cols, dropna=False).groups.items():
        rows = np.asarray(list(idx), dtype=np.int64)
        rng.shuffle(rows)
        for offset, row in enumerate(rows):
            fold_id[row] = int(offset % folds)
    if (fold_id < 0).any():
        raise RuntimeError("some rows were not assigned to a fold")
    return fold_id


def materialize_fold(
    source_dir: Path,
    out_dir: Path,
    meta: pd.DataFrame,
    atlas: pd.DataFrame,
    fold_id: np.ndarray,
    fold: int,
    folds: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    test_fold = int(fold)
    val_fold = int((fold + 1) % folds)
    split = np.where(fold_id == test_fold, "test", np.where(fold_id == val_fold, "val", "train"))

    fold_meta = meta.copy()
    fold_atlas = atlas.copy()
    fold_meta["split"] = split
    fold_atlas["split"] = split
    fold_meta["cv_fold_id"] = fold_id
    fold_atlas["cv_fold_id"] = fold_id
    fold_meta["cv_seed"] = int(seed)
    fold_atlas["cv_seed"] = int(seed)
    fold_meta["cv_policy"] = out_dir.name
    fold_atlas["cv_policy"] = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(source_dir / "signals.npz", out_dir / "signals.npz", overwrite=overwrite)
    fold_meta.to_csv(out_dir / "metadata.csv", index=False)
    fold_atlas.to_csv(out_dir / "original_region_atlas.csv", index=False)
    shutil.copy2(source_dir / "window_segment_margins.csv", out_dir / "window_segment_margins.csv")

    counts = counts_table(fold_atlas)
    counts.to_csv(out_dir / "split_counts.csv", index=False)
    audit = {
        "policy": out_dir.name,
        "source_policy": source_dir.name,
        "split_strategy": "window_level_stratified_by_record_class_region",
        "record_heldout": False,
        "seed": int(seed),
        "fold": int(fold),
        "folds": int(folds),
        "test_fold_id": int(test_fold),
        "val_fold_id": int(val_fold),
        "n": int(len(fold_atlas)),
        "counts": counts.to_dict(orient="records"),
        "signals": str(out_dir / "signals.npz"),
        "metadata": str(out_dir / "metadata.csv"),
        "atlas": str(out_dir / "original_region_atlas.csv"),
    }
    write_json(out_dir / "audit.json", audit)
    return audit


def write_report(report_path: Path, audits: list[dict[str, Any]], counts_all: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Clean BUT Stratified CV Folds",
        "",
        "These folds reuse the already-cleaned fixed-10s windows and rewrite only split labels.",
        "They are window-level folds stratified by record, class, and original_region, not record-heldout external tests.",
        "The goal is stable clean-BUT model selection after dropping low-confidence/outlier windows.",
        "",
        "## Global Fold Assignment",
        "",
    ]
    lines.append(markdown_table(counts_all))
    lines.extend(["", "## Policies", ""])
    for audit in audits:
        lines.append(f"### {audit['policy']}")
        lines.append("")
        lines.append(markdown_table(pd.DataFrame(audit["counts"])))
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-policy", type=str, default="margin_ge_5s_drop_outlier")
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = args.protocol_root / args.source_policy
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if int(args.folds) < 3:
        raise ValueError("--folds must be at least 3")

    meta = pd.read_csv(source_dir / "metadata.csv")
    atlas = pd.read_csv(source_dir / "original_region_atlas.csv")
    if len(meta) != len(atlas):
        raise ValueError(f"metadata/atlas row mismatch: {len(meta)} vs {len(atlas)}")
    for col in ("record_id", "class_name", "original_region"):
        if col not in atlas.columns:
            raise ValueError(f"atlas missing required column {col}")

    fold_id = assign_folds(atlas, int(args.folds), int(args.seed))
    fold_frame = atlas[["record_id", "class_name", "original_region"]].copy()
    fold_frame["fold_id"] = fold_id
    counts_all = (
        fold_frame.groupby(["fold_id", "class_name", "original_region"], dropna=False)
        .size()
        .reset_index(name="n")
        .pivot_table(index="fold_id", columns=["class_name", "original_region"], values="n", fill_value=0, aggfunc="sum")
    )
    counts_all.columns = [f"{cls}:{region}" for cls, region in counts_all.columns]
    counts_all = counts_all.reset_index()
    counts_all.to_csv(args.protocol_root / f"{args.source_policy}_cv_seed{args.seed}_fold_assignment_counts.csv", index=False)

    audits: list[dict[str, Any]] = []
    for fold in range(int(args.folds)):
        out_name = f"{args.source_policy}_cv_seed{args.seed}_fold{fold}"
        audits.append(
            materialize_fold(
                source_dir,
                args.protocol_root / out_name,
                meta,
                atlas,
                fold_id,
                fold,
                int(args.folds),
                int(args.seed),
                bool(args.overwrite),
            )
        )

    summary = {
        "source_policy": args.source_policy,
        "seed": int(args.seed),
        "folds": int(args.folds),
        "split_strategy": "window_level_stratified_by_record_class_region",
        "record_heldout": False,
        "policies": audits,
    }
    summary_path = args.protocol_root / f"{args.source_policy}_cv_seed{args.seed}_summary.json"
    write_json(summary_path, summary)
    write_json(args.report_dir / summary_path.name, summary)
    report_path = args.report_dir / f"{args.source_policy}_cv_seed{args.seed}_report.md"
    write_report(report_path, audits, counts_all)
    print(f"Wrote {report_path}")
    for audit in audits:
        test = next(row for row in audit["counts"] if row["split"] == "test")
        val = next(row for row in audit["counts"] if row["split"] == "val")
        print(
            f"{audit['policy']}: test n={test['n']} g/m/b={test.get('good',0)}/{test.get('medium',0)}/{test.get('bad',0)}; "
            f"val n={val['n']} g/m/b={val.get('good',0)}/{val.get('medium',0)}/{val.get('bad',0)}"
        )


if __name__ == "__main__":
    main()
