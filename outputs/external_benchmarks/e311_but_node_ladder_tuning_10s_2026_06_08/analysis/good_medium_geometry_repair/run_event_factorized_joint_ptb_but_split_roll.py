"""BUT split-roll diagnostic for the joint PTB+BUT waveform model.

This script keeps the formal model unchanged:

* inference input is waveform-derived channels only;
* SQI/factor columns are teacher targets and diagnostics only;
* no route/rule/MLP/tree classifier is used.

The purpose is narrower: audit whether the current cleaned BUT test split is
pathologically record/region skewed, then train the same joint PTB+BUT model on
new diagnostic BUT splits.  The `window_stratified` mode is intentionally not a
strict record-heldout external test; it is a diagnostic capacity/split-bias
check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"

SOURCE_BUT_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier"
JOINT_SCRIPT = ANALYSIS_DIR / "run_event_factorized_joint_ptb_but.py"

ROLL_OUT = ANALYSIS_DIR / "event_joint_ptb_but_split_roll"
ROLL_REPORT_DIR = REPORT_DIR / "event_factorized_sqi_conformer"
ROLL_REPORT = ROLL_REPORT_DIR / "event_factorized_joint_ptb_but_split_roll_report.md"
ROLL_RUN_ROOT = OUT_ROOT / "runs" / "event_joint_ptb_but_split_roll"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JOINT = load_module(JOINT_SCRIPT, "joint_ptb_but_for_split_roll")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    ROLL_OUT.mkdir(parents=True, exist_ok=True)
    ROLL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ROLL_RUN_ROOT.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_protocol(protocol: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame | None]:
    z = np.load(protocol / "signals.npz")
    key = "X" if "X" in z.files else z.files[0]
    x = np.asarray(z[key], dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    atlas = pd.read_csv(protocol / "original_region_atlas.csv")
    metadata_path = protocol / "metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else None
    return x, atlas, metadata


def split_counts(atlas: pd.DataFrame) -> pd.DataFrame:
    cols = ["split", "class_name"]
    rows = atlas.groupby(cols, dropna=False).size().reset_index(name="n")
    return rows.sort_values(cols).reset_index(drop=True)


def record_split_counts(atlas: pd.DataFrame) -> pd.DataFrame:
    rows = atlas.groupby(["split", "record_id", "class_name"], dropna=False).size().reset_index(name="n")
    return rows.sort_values(["split", "n"], ascending=[True, False]).reset_index(drop=True)


def region_split_counts(atlas: pd.DataFrame) -> pd.DataFrame:
    cols = ["split", "class_name", "original_region"]
    rows = atlas.groupby(cols, dropna=False).size().reset_index(name="n")
    return rows.sort_values(["split", "class_name", "n"], ascending=[True, True, False]).reset_index(drop=True)


def assign_window_stratified(atlas: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> pd.Series:
    """Row-level diagnostic split balanced by class and original region."""
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=atlas.index, dtype=object)
    region = atlas.get("original_region", pd.Series(["unknown"] * len(atlas), index=atlas.index)).fillna("unknown").astype(str)
    stratum = atlas["class_name"].fillna("unknown").astype(str) + "||" + region
    for _, idx in atlas.groupby(stratum, dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        rng.shuffle(idx_arr)
        n = len(idx_arr)
        if n <= 2:
            continue
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 6:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))
        out.iloc[idx_arr[:n_test]] = "test"
        out.iloc[idx_arr[n_test : n_test + n_val]] = "val"
    return out


def assign_record_greedy(atlas: pd.DataFrame, seed: int) -> pd.Series:
    """Strict record-level roll for audit.

    The BUT clean set has too few bad-containing records for a fair split, so
    this mode is reported as a stress split rather than the main diagnostic.
    """
    rng = np.random.default_rng(seed)
    recs = atlas["record_id"].astype(str).unique().tolist()
    rng.shuffle(recs)
    total = atlas["class_name"].value_counts().reindex(["good", "medium", "bad"]).fillna(0).to_numpy(dtype=float)
    target_test = total * 0.15
    target_val = total * 0.15
    rec_counts = (
        atlas.groupby(["record_id", "class_name"]).size().unstack(fill_value=0).reindex(columns=["good", "medium", "bad"], fill_value=0)
    )

    def greedy_pick(target: np.ndarray, available: list[str]) -> list[str]:
        picked: list[str] = []
        cur = np.zeros(3, dtype=float)
        remaining = available[:]
        while remaining and (cur < target).any():
            scores = []
            for rec in remaining:
                cand = cur + rec_counts.loc[rec].to_numpy(dtype=float)
                # Penalize overshoot, but value covering missing classes.
                err = np.abs(cand - target).sum() + 0.25 * np.maximum(cand - target, 0).sum()
                scores.append((err, rec))
            scores.sort(key=lambda x: x[0])
            rec = scores[0][1]
            picked.append(rec)
            remaining.remove(rec)
            cur += rec_counts.loc[rec].to_numpy(dtype=float)
        return picked

    available = recs[:]
    test_recs = greedy_pick(target_test, available)
    available = [r for r in available if r not in test_recs]
    val_recs = greedy_pick(target_val, available)
    out = pd.Series("train", index=atlas.index, dtype=object)
    out[atlas["record_id"].astype(str).isin(test_recs)] = "test"
    out[atlas["record_id"].astype(str).isin(val_recs)] = "val"
    return out


def build_resplit_protocol(mode: str, seed: int, val_frac: float, test_frac: float, force: bool = False) -> Path:
    ensure_dirs()
    out = ROLL_OUT / f"but_protocol_{mode}_s{seed}"
    atlas_path = out / "original_region_atlas.csv"
    if atlas_path.exists() and (out / "signals.npz").exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)
    x, atlas, metadata = load_protocol(SOURCE_BUT_PROTOCOL)
    atlas = atlas.copy()
    old_split = atlas["split"].astype(str).copy()
    if mode == "window_stratified":
        new_split = assign_window_stratified(atlas, seed, val_frac, test_frac)
        split_kind = "diagnostic row-level stratified by class+original_region; record leakage allowed"
    elif mode == "record_greedy":
        new_split = assign_record_greedy(atlas, seed)
        split_kind = "strict record-level greedy stress split; limited by sparse bad records"
    else:
        raise ValueError(f"unknown mode: {mode}")

    atlas["previous_split"] = old_split
    atlas["split"] = new_split.to_numpy()
    atlas["split_roll_mode"] = mode
    atlas["split_roll_seed"] = int(seed)
    atlas["split_roll_kind"] = split_kind
    atlas["idx"] = np.arange(len(atlas), dtype=np.int64)
    atlas["source_idx"] = np.arange(len(atlas), dtype=np.int64)

    np.savez_compressed(out / "signals.npz", X=x.astype(np.float32))
    atlas.to_csv(atlas_path, index=False)
    if metadata is not None:
        meta = metadata.copy()
        if len(meta) == len(atlas):
            meta["previous_split"] = meta["split"].astype(str)
            meta["split"] = atlas["split"].to_numpy()
            meta["split_roll_mode"] = mode
            meta["split_roll_seed"] = int(seed)
        meta.to_csv(out / "metadata.csv", index=False)

    counts = split_counts(atlas)
    records = record_split_counts(atlas)
    regions = region_split_counts(atlas)
    counts.to_csv(out / "split_counts.csv", index=False)
    records.to_csv(out / "record_split_counts.csv", index=False)
    regions.to_csv(out / "region_split_counts.csv", index=False)
    summary = {
        "created_at": now(),
        "source_protocol": str(SOURCE_BUT_PROTOCOL),
        "mode": mode,
        "seed": int(seed),
        "split_kind": split_kind,
        "n": int(len(atlas)),
        "split_counts": counts.to_dict(orient="records"),
        "record_split_counts_top": records.head(40).to_dict(orient="records"),
        "region_split_counts_top": regions.head(60).to_dict(orient="records"),
        "warning": "window_stratified is diagnostic only, not a strict record-heldout external split",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def configure_joint_for_roll(mode: str, split_seed: int, but_protocol: Path) -> None:
    run_tag = f"{mode}_s{split_seed}"
    JOINT.DEFAULT_BUT_PROTOCOL = but_protocol
    JOINT.JOINT_OUT = ROLL_OUT / f"joint_{run_tag}"
    JOINT.JOINT_REPORT = ROLL_REPORT_DIR / f"event_factorized_joint_ptb_but_split_roll_{run_tag}.md"
    JOINT.JOINT_RUN_DIR = ROLL_RUN_ROOT / run_tag


def run_one(mode: str, split_seed: int, args: argparse.Namespace) -> pd.DataFrame:
    but_protocol = build_resplit_protocol(mode, split_seed, args.val_frac, args.test_frac, force=bool(args.force_protocol))
    configure_joint_for_roll(mode, split_seed, but_protocol)
    joint_args = argparse.Namespace(
        force_protocol=bool(args.force_protocol),
        seed=int(args.train_seed_base + split_seed),
        seeds=1,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        record_balanced_sampler=bool(args.record_balanced_sampler),
        bad_extra_per_source=int(args.bad_extra_per_source),
        max_train_rows=int(args.max_train_rows),
        max_val_rows=int(args.max_val_rows),
        max_test_rows=int(args.max_test_rows),
        max_eval_rows=int(args.max_eval_rows),
        max_eval_all_rows=int(args.max_eval_all_rows),
        candidates=str(args.candidates),
    )
    JOINT.run(joint_args)
    summary_path = JOINT.JOINT_OUT / "joint_ptb_but_summary.csv"
    summary = pd.read_csv(summary_path)
    summary["split_mode"] = mode
    summary["split_seed"] = int(split_seed)
    summary["but_protocol"] = str(but_protocol)
    summary["joint_out"] = str(JOINT.JOINT_OUT)
    summary["joint_report"] = str(JOINT.JOINT_REPORT)
    return summary


def audit_source() -> None:
    ensure_dirs()
    _, atlas, _ = load_protocol(SOURCE_BUT_PROTOCOL)
    split_counts(atlas).to_csv(ROLL_OUT / "source_but_split_counts.csv", index=False)
    record_split_counts(atlas).to_csv(ROLL_OUT / "source_but_record_split_counts.csv", index=False)
    region_split_counts(atlas).to_csv(ROLL_OUT / "source_but_region_split_counts.csv", index=False)


def write_master_report(aggregate: pd.DataFrame) -> None:
    audit_counts = pd.read_csv(ROLL_OUT / "source_but_split_counts.csv")
    audit_records = pd.read_csv(ROLL_OUT / "source_but_record_split_counts.csv")
    key = aggregate[aggregate["bucket"].isin(["but_test", "but_all", "ptb_test", "joint_test"])].copy()
    cols = [
        "split_mode",
        "split_seed",
        "candidate",
        "bucket",
        "acc",
        "macro_f1",
        "good_recall",
        "medium_recall",
        "bad_recall",
        "record_macro_supported_f1",
        "artifact_positive_nonbad_bad_fpr",
    ]
    cols = [c for c in cols if c in key.columns]
    lines = [
        "# Joint PTB+BUT Split-Roll Diagnostic",
        "",
        f"- Generated: {now()}",
        "- Model: EventFactorizedSQIConformer, waveform-derived channels only at inference.",
        "- Teacher signals: interpretable SQI/factor targets used only during training/diagnostics.",
        "- Purpose: test whether the current cleaned BUT test score is dominated by split skew.",
        "",
        "## Current Clean BUT Split",
        "",
        "Class counts:",
        "",
        md_table(audit_counts),
        "",
        "Top record/class split counts:",
        "",
        md_table(audit_records.head(30)),
        "",
        "Interpretation: the current test split is mostly record `111001` good/medium overlap, while validation has almost no bad. This makes it a hard record/region-shift test, not a balanced clean-BUT test.",
        "",
        "## Split-Roll Results",
        "",
        md_table(key[cols].sort_values(["split_mode", "split_seed", "bucket", "acc"], ascending=[True, True, True, False])),
        "",
        "## Caveat",
        "",
        "`window_stratified` is diagnostic only because it allows record overlap across train/val/test. If it improves sharply, the low current BUT test score is at least partly a split/record-shift issue. It is not a replacement for strict external validation.",
        "",
        "## Output Files",
        "",
        f"- Aggregate summary: `{ROLL_OUT / 'split_roll_aggregate_summary.csv'}`",
        f"- Source split audit: `{ROLL_OUT}`",
        f"- Per-run reports: `{ROLL_REPORT_DIR / 'event_factorized_joint_ptb_but_split_roll_<mode>_<seed>.md'}`",
        f"- Checkpoints/logs: `{ROLL_RUN_ROOT}`",
    ]
    ROLL_REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    audit_source()
    summaries: list[pd.DataFrame] = []
    for mode in [m.strip() for m in str(args.modes).split(",") if m.strip()]:
        for split_seed in parse_seeds(args.split_seeds):
            print(f"[{now()}] split-roll mode={mode} seed={split_seed}", flush=True)
            summaries.append(run_one(mode, split_seed, args))
            aggregate = pd.concat(summaries, ignore_index=True)
            aggregate.to_csv(ROLL_OUT / "split_roll_aggregate_summary.csv", index=False)
            write_master_report(aggregate)
    print(ROLL_REPORT, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build_splits", "run"], default="run")
    parser.add_argument("--modes", type=str, default="window_stratified")
    parser.add_argument("--split-seeds", type=str, default="4101,4102,4103")
    parser.add_argument("--force-protocol", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--train-seed-base", type=int, default=20260000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--record-balanced-sampler", action="store_true", default=True)
    parser.add_argument("--bad-extra-per-source", type=int, default=1200)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--max-eval-all-rows", type=int, default=0)
    parser.add_argument("--candidates", type=str, default="E1_query_only")
    args = parser.parse_args()

    ensure_dirs()
    audit_source()
    if args.stage == "build_splits":
        for mode in [m.strip() for m in str(args.modes).split(",") if m.strip()]:
            for split_seed in parse_seeds(args.split_seeds):
                print(build_resplit_protocol(mode, split_seed, args.val_frac, args.test_frac, force=bool(args.force_protocol)))
    else:
        run(args)


if __name__ == "__main__":
    main()
