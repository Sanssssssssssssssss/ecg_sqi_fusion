"""Materialize strict-clean BUT 10s protocols from the current p1 windows.

This is experiment-only code. It does not touch src/sqi_pipeline, mainline
checkpoints, or the original p1 protocol. It builds filtered 10s datasets with
self-contained signals/metadata/atlas files so waveform-only Transformer
diagnostics can train on cleaner learnable bodies while still reporting the
full/stress buckets separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[6]
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
DEFAULT_SOURCE_PROTOCOL = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_protocol_adaptation_2026_06_03"
    / "protocols"
    / "p1_current_10s_center"
)
DEFAULT_SOURCE_ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
DEFAULT_MARGIN_TABLE = ANALYSIS_DIR / "clean_but_window_policy" / "current_10s_window_segment_margins.csv"
DEFAULT_OUT_DIR = ANALYSIS_DIR / "clean_but_protocols"
DEFAULT_REPORT_DIR = REPORT_DIR / "clean_but_protocols"


@dataclass(frozen=True)
class CleanPolicy:
    name: str
    min_margin_sec: float = 0.0
    drop_outlier_low_confidence: bool = False
    keep_regions: tuple[str, ...] | None = None
    keep_bad_core: bool = True
    min_segment_duration_sec: float | None = None
    max_segment_duration_sec: float | None = None


POLICIES = [
    CleanPolicy("margin_ge_2s_keep_outlier", min_margin_sec=2.0, drop_outlier_low_confidence=False),
    CleanPolicy("margin_ge_5s_keep_outlier", min_margin_sec=5.0, drop_outlier_low_confidence=False),
    CleanPolicy("margin_ge_2s_drop_outlier", min_margin_sec=2.0, drop_outlier_low_confidence=True),
    CleanPolicy("margin_ge_5s_drop_outlier", min_margin_sec=5.0, drop_outlier_low_confidence=True),
    CleanPolicy("margin_ge_8s_drop_outlier", min_margin_sec=8.0, drop_outlier_low_confidence=True),
    CleanPolicy("margin_ge_10s_drop_outlier", min_margin_sec=10.0, drop_outlier_low_confidence=True),
    CleanPolicy("clean_core_plus_overlap_margin2", min_margin_sec=2.0, keep_regions=("clean_core", "good_medium_overlap"), keep_bad_core=False),
    CleanPolicy("clean_core_only_margin2", min_margin_sec=2.0, keep_regions=("clean_core",), keep_bad_core=False),
    CleanPolicy("segment_10_20s_keep_outlier", min_segment_duration_sec=10.0, max_segment_duration_sec=20.0),
    CleanPolicy("segment_10_30s_keep_outlier", min_segment_duration_sec=10.0, max_segment_duration_sec=30.0),
    CleanPolicy("segment_10_60s_keep_outlier", min_segment_duration_sec=10.0, max_segment_duration_sec=60.0),
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    if len(df) == 0:
        return {}
    return {str(k): int(v) for k, v in df["y_class"].value_counts().sort_index().items()}


def split_class_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if len(df) == 0:
        return {}
    return {
        str(split): {str(k): int(v) for k, v in sub["y_class"].value_counts().sort_index().items()}
        for split, sub in df.groupby("split")
    }


def policy_mask(margin: pd.DataFrame, policy: CleanPolicy) -> np.ndarray:
    mask = margin["matched_segment"].fillna(False).astype(bool).to_numpy().copy()
    mask &= margin["min_margin_sec"].astype(float).to_numpy() >= float(policy.min_margin_sec)
    if policy.min_segment_duration_sec is not None:
        mask &= margin["segment_duration_sec"].astype(float).to_numpy() >= float(policy.min_segment_duration_sec)
    if policy.max_segment_duration_sec is not None:
        mask &= margin["segment_duration_sec"].astype(float).to_numpy() <= float(policy.max_segment_duration_sec)
    region = margin["original_region"].fillna("unknown").astype(str)
    y_class = margin["y_class"].fillna("").astype(str)
    if policy.drop_outlier_low_confidence:
        mask &= region.ne("outlier_low_confidence").to_numpy()
    if policy.keep_regions is not None:
        region_mask = region.isin(policy.keep_regions).to_numpy()
        if policy.keep_bad_core:
            region_mask |= (y_class.eq("bad") & region.ne("outlier_low_confidence")).to_numpy()
        mask &= region_mask
    return mask


def materialize_policy(
    policy: CleanPolicy,
    source_x: np.ndarray,
    source_meta: pd.DataFrame,
    source_atlas: pd.DataFrame,
    margin: pd.DataFrame,
    out_root: Path,
) -> dict[str, Any]:
    mask = policy_mask(margin, policy)
    source_idx = np.flatnonzero(mask).astype(int)
    out_dir = out_root / policy.name
    out_dir.mkdir(parents=True, exist_ok=True)

    x_out = source_x[source_idx].astype(np.float32)
    meta_out = source_meta.iloc[source_idx].copy().reset_index(drop=True)
    meta_out.insert(0, "source_idx", source_idx)
    meta_out["idx"] = np.arange(len(meta_out), dtype=int)
    meta_out["clean_policy"] = policy.name
    meta_out["clean_min_margin_sec"] = float(policy.min_margin_sec)
    meta_out["source_protocol"] = "p1_current_10s_center"

    atlas_source = source_atlas.copy()
    atlas_source["idx"] = atlas_source["idx"].astype(int)
    atlas_out = atlas_source.set_index("idx").loc[source_idx].reset_index(drop=False).rename(columns={"idx": "source_idx"})
    atlas_out.insert(0, "idx", np.arange(len(atlas_out), dtype=int))
    atlas_out["clean_policy"] = policy.name
    atlas_out["clean_min_margin_sec"] = float(policy.min_margin_sec)

    margin_cols = [
        "source_idx",
        "matched_segment",
        "segment_idx",
        "segment_start_sec",
        "segment_end_sec",
        "segment_duration_sec",
        "left_margin_sec",
        "right_margin_sec",
        "min_margin_sec",
        "window_center_fraction",
        "original_region",
        "ambiguous_type",
    ]
    margin_out = margin.iloc[source_idx].copy().reset_index(drop=True)
    margin_out.insert(0, "source_idx", source_idx)
    margin_out = margin_out[[c for c in margin_cols if c in margin_out.columns]]

    np.savez_compressed(out_dir / "signals.npz", X=x_out)
    meta_out.to_csv(out_dir / "metadata.csv", index=False)
    atlas_out.to_csv(out_dir / "original_region_atlas.csv", index=False)
    margin_out.to_csv(out_dir / "window_segment_margins.csv", index=False)

    audit = {
        "policy": policy.name,
        "n": int(len(meta_out)),
        "retention_rate": float(len(meta_out) / max(len(source_meta), 1)),
        "class_counts": class_counts(meta_out),
        "split_class_counts": split_class_counts(meta_out),
        "record_count": int(meta_out["record_id"].nunique()) if len(meta_out) else 0,
        "drop_outlier_low_confidence": bool(policy.drop_outlier_low_confidence),
        "keep_regions": list(policy.keep_regions) if policy.keep_regions is not None else None,
        "min_segment_duration_sec": policy.min_segment_duration_sec,
        "max_segment_duration_sec": policy.max_segment_duration_sec,
        "signals": str(out_dir / "signals.npz"),
        "metadata": str(out_dir / "metadata.csv"),
        "atlas": str(out_dir / "original_region_atlas.csv"),
        "margins": str(out_dir / "window_segment_margins.csv"),
    }
    write_json(out_dir / "audit.json", audit)
    return audit


def write_report(report_path: Path, audits: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Clean BUT Protocol Materialization\n\n")
        f.write("These are fixed-length 10s protocol variants. No variable-length model is used here.\n\n")
        f.write("| policy | n | good | medium | bad | record_count | path |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for audit in audits:
            counts = audit["class_counts"]
            f.write(
                f"| {audit['policy']} | {audit['n']} | {counts.get('good', 0)} | {counts.get('medium', 0)} | "
                f"{counts.get('bad', 0)} | {audit['record_count']} | `{Path(audit['signals']).parent}` |\n"
            )
        f.write("\n## How To Use\n\n")
        f.write(
            "Each policy directory contains `signals.npz`, `metadata.csv`, `original_region_atlas.csv`, "
            "`window_segment_margins.csv`, and `audit.json`. The atlas `idx` is reset to the new signal row index, "
            "while `source_idx` preserves the row in the original p1 protocol.\n\n"
        )
        f.write("Recommended first training diagnostics:\n\n")
        f.write("1. `margin_ge_5s_keep_outlier`: strict fixed-10s margin cleanup while preserving stress/outlier labels.\n")
        f.write("2. `margin_ge_5s_drop_outlier`: strict fixed-10s clean body, bad core only.\n")
        f.write("3. `clean_core_plus_overlap_margin2`: good/medium learnable-body sanity, no bad.\n\n")
        f.write("Full BUT and bad outlier stress must remain separate report-only evaluations.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol", type=Path, default=DEFAULT_SOURCE_PROTOCOL)
    parser.add_argument("--source-atlas", type=Path, default=DEFAULT_SOURCE_ATLAS)
    parser.add_argument("--margin-table", type=Path, default=DEFAULT_MARGIN_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    source_x = np.load(args.source_protocol / "signals.npz")["X"].astype(np.float32)
    source_meta = pd.read_csv(args.source_protocol / "metadata.csv")
    source_atlas = pd.read_csv(args.source_atlas)
    margin = pd.read_csv(args.margin_table)
    if len(source_x) != len(source_meta) or len(source_meta) != len(margin):
        raise ValueError(f"row mismatch: X={len(source_x)} metadata={len(source_meta)} margin={len(margin)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    audits = [materialize_policy(policy, source_x, source_meta, source_atlas, margin, args.out_dir) for policy in POLICIES]
    summary = {
        "source_protocol": str(args.source_protocol),
        "source_atlas": str(args.source_atlas),
        "margin_table": str(args.margin_table),
        "policies": audits,
    }
    write_json(args.out_dir / "clean_but_protocols_summary.json", summary)
    write_json(args.report_dir / "clean_but_protocols_summary.json", summary)
    write_report(args.report_dir / "clean_but_protocols_report.md", audits)
    print(f"Wrote {args.report_dir / 'clean_but_protocols_report.md'}")
    for audit in audits:
        print(f"{audit['policy']}: n={audit['n']} counts={audit['class_counts']}")


if __name__ == "__main__":
    main()
