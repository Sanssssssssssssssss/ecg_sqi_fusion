"""Design clean BUT window policies for follow-up waveform experiments.

This is experiment-only analysis code. It does not modify source pipelines,
mainline checkpoints, or existing protocol files. It audits whether the current
fixed 10s BUT windows are too close to consensus-label segment boundaries and
estimates cleaner alternatives:

* strict 10s windows with minimum margin from label boundaries;
* centered 10s windows regenerated from long consensus segments;
* variable-length segment candidates for future masked Transformer input.
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
RUN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
DEFAULT_METADATA = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_protocol_adaptation_2026_06_03"
    / "protocols"
    / "p1_current_10s_center"
    / "metadata.csv"
)
DEFAULT_ATLAS = RUN_ROOT / "original_region_boundary" / "original_region_atlas.csv"
DEFAULT_ANN_ROOT = ROOT / "data" / "external" / "butqdb_1_0_0"
OUT_DIR = RUN_ROOT / "analysis" / "good_medium_geometry_repair" / "clean_but_window_policy"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "clean_but_window_policy"
CLASS_MAP = {"1": "good", "2": "medium", "3": "bad", 1: "good", 2: "medium", 3: "bad"}


@dataclass(frozen=True)
class Policy:
    name: str
    min_margin_sec: float = 0.0
    min_segment_sec: float = 10.0
    drop_outlier_low_confidence: bool = False
    keep_regions: tuple[str, ...] | None = None


WINDOW_POLICIES = [
    Policy("current_all_10s"),
    Policy("margin_ge_1s", min_margin_sec=1.0),
    Policy("margin_ge_2s", min_margin_sec=2.0),
    Policy("margin_ge_5s", min_margin_sec=5.0),
    Policy("margin_ge_10s", min_margin_sec=10.0),
    Policy("margin_ge_2s_drop_outlier", min_margin_sec=2.0, drop_outlier_low_confidence=True),
    Policy("margin_ge_5s_drop_outlier", min_margin_sec=5.0, drop_outlier_low_confidence=True),
    Policy("clean_core_plus_overlap_margin2", min_margin_sec=2.0, keep_regions=("clean_core", "good_medium_overlap")),
    Policy("clean_core_only_margin2", min_margin_sec=2.0, keep_regions=("clean_core",)),
]


REGEN_SPECS = [
    ("center_one_per_segment_margin0", 0.0, "one_center"),
    ("center_one_per_segment_margin2", 2.0, "one_center"),
    ("center_one_per_segment_margin5", 5.0, "one_center"),
    ("center_stride10_margin2", 2.0, "stride10"),
    ("center_stride10_margin5", 5.0, "stride10"),
    ("center_stride5_margin5", 5.0, "stride5"),
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_consensus_segments(ann_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ann_path in sorted(ann_root.glob("*/*_ANN.csv")):
        record_id = ann_path.parent.name
        try:
            arr = pd.read_csv(ann_path, header=None).to_numpy()
        except pd.errors.EmptyDataError:
            continue
        if arr.shape[1] < 12:
            continue
        # BUT ANN files store the consensus interval in the last 3 columns.
        for seg_idx, row in enumerate(arr):
            if pd.isna(row[9]) or pd.isna(row[10]) or pd.isna(row[11]):
                continue
            start_ms = float(row[9])
            end_ms = float(row[10])
            label_raw = str(int(row[11]))
            if end_ms <= start_ms:
                continue
            rows.append(
                {
                    "record_id": str(record_id),
                    "segment_idx": int(seg_idx),
                    "segment_start_sec": start_ms / 1000.0,
                    "segment_end_sec": end_ms / 1000.0,
                    "segment_duration_sec": (end_ms - start_ms) / 1000.0,
                    "label_raw": label_raw,
                    "y_class": CLASS_MAP.get(label_raw, "unknown"),
                }
            )
    return pd.DataFrame(rows)


def attach_segments(meta: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["record_id"] = meta["record_id"].astype(str)
    meta["start_sec"] = meta["start_sec"].astype(float)
    meta["end_sec"] = meta["end_sec"].astype(float)
    meta["y_class"] = meta["y_class"].astype(str)
    seg_by_record = {str(k): g.reset_index(drop=True) for k, g in segments.groupby("record_id")}
    attached: list[dict[str, Any]] = []
    for row in meta.itertuples(index=True):
        record_segments = seg_by_record.get(str(row.record_id))
        match = None
        if record_segments is not None:
            candidates = record_segments[
                (record_segments["y_class"] == str(row.y_class))
                & (record_segments["segment_start_sec"] <= float(row.start_sec) + 1e-6)
                & (record_segments["segment_end_sec"] >= float(row.end_sec) - 1e-6)
            ]
            if len(candidates):
                # If nested intervals exist, use the tightest enclosing segment.
                match = candidates.sort_values("segment_duration_sec").iloc[0]
        if match is None:
            attached.append(
                {
                    "metadata_idx": int(row.Index),
                    "matched_segment": False,
                    "segment_idx": np.nan,
                    "segment_start_sec": np.nan,
                    "segment_end_sec": np.nan,
                    "segment_duration_sec": np.nan,
                    "left_margin_sec": np.nan,
                    "right_margin_sec": np.nan,
                    "min_margin_sec": np.nan,
                    "window_center_offset_from_segment_center_sec": np.nan,
                    "window_center_fraction": np.nan,
                }
            )
            continue
        seg_start = float(match["segment_start_sec"])
        seg_end = float(match["segment_end_sec"])
        seg_dur = float(match["segment_duration_sec"])
        win_center = 0.5 * (float(row.start_sec) + float(row.end_sec))
        seg_center = 0.5 * (seg_start + seg_end)
        denom = max(seg_dur, 1e-9)
        attached.append(
            {
                "metadata_idx": int(row.Index),
                "matched_segment": True,
                "segment_idx": int(match["segment_idx"]),
                "segment_start_sec": seg_start,
                "segment_end_sec": seg_end,
                "segment_duration_sec": seg_dur,
                "left_margin_sec": float(row.start_sec) - seg_start,
                "right_margin_sec": seg_end - float(row.end_sec),
                "min_margin_sec": min(float(row.start_sec) - seg_start, seg_end - float(row.end_sec)),
                "window_center_offset_from_segment_center_sec": win_center - seg_center,
                "window_center_fraction": (win_center - seg_start) / denom,
            }
        )
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(attached).drop(columns=["metadata_idx"])], axis=1)


def join_atlas(meta: pd.DataFrame, atlas_path: Path) -> pd.DataFrame:
    if not atlas_path.exists():
        meta["original_region"] = "unknown"
        meta["ambiguous_type"] = "unknown"
        return meta
    atlas = pd.read_csv(atlas_path)
    keep = [c for c in ["idx", "original_region", "ambiguous_type", "is_clean_strict", "region_confidence"] if c in atlas.columns]
    atlas = atlas[keep].copy()
    atlas["idx"] = atlas["idx"].astype(int)
    meta = meta.reset_index(drop=True).copy()
    meta["idx"] = np.arange(len(meta), dtype=int)
    return meta.merge(atlas, on="idx", how="left")


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["y_class"].value_counts().sort_index().items()}


def split_class_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        str(split): {str(k): int(v) for k, v in sub["y_class"].value_counts().sort_index().items()}
        for split, sub in df.groupby("split")
    }


def summarize_windows(meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in WINDOW_POLICIES:
        mask = meta["matched_segment"].fillna(False)
        mask &= meta["segment_duration_sec"] >= policy.min_segment_sec
        mask &= meta["min_margin_sec"] >= policy.min_margin_sec
        if policy.drop_outlier_low_confidence:
            mask &= meta["original_region"].fillna("unknown") != "outlier_low_confidence"
        if policy.keep_regions is not None:
            mask &= meta["original_region"].fillna("unknown").isin(policy.keep_regions)
        sub = meta[mask].copy()
        row: dict[str, Any] = {
            "policy": policy.name,
            "n": int(len(sub)),
            "retention_rate": float(len(sub) / max(len(meta), 1)),
            "class_counts": json.dumps(class_counts(sub), ensure_ascii=False),
            "split_class_counts": json.dumps(split_class_counts(sub), ensure_ascii=False),
            "record_count": int(sub["record_id"].nunique()) if len(sub) else 0,
            "bad_outlier_low_confidence": int(((sub["y_class"] == "bad") & (sub["original_region"] == "outlier_low_confidence")).sum())
            if "original_region" in sub
            else 0,
            "good_medium_overlap": int((sub["original_region"] == "good_medium_overlap").sum())
            if "original_region" in sub
            else 0,
            "clean_core": int((sub["original_region"] == "clean_core").sum()) if "original_region" in sub else 0,
        }
        for cls in ["good", "medium", "bad"]:
            row[f"{cls}_n"] = int((sub["y_class"] == cls).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def possible_windows_for_segment(duration: float, margin: float, mode: str) -> int:
    interior = duration - 2.0 * margin
    if interior < 10.0:
        return 0
    if mode == "one_center":
        return 1
    stride = 10.0 if mode == "stride10" else 5.0
    return int(np.floor((interior - 10.0) / stride) + 1)


def summarize_regeneration(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable = segments[segments["y_class"].isin(["good", "medium", "bad"])].copy()
    for name, margin, mode in REGEN_SPECS:
        tmp = usable.copy()
        tmp["candidate_windows"] = tmp["segment_duration_sec"].apply(lambda x: possible_windows_for_segment(float(x), margin, mode))
        sub = tmp[tmp["candidate_windows"] > 0].copy()
        row: dict[str, Any] = {
            "regen_policy": name,
            "min_margin_sec": margin,
            "mode": mode,
            "segment_count": int(len(sub)),
            "candidate_window_count": int(sub["candidate_windows"].sum()),
            "record_count": int(sub["record_id"].nunique()) if len(sub) else 0,
        }
        for cls in ["good", "medium", "bad"]:
            cls_sub = sub[sub["y_class"] == cls]
            row[f"{cls}_segments"] = int(len(cls_sub))
            row[f"{cls}_candidate_windows"] = int(cls_sub["candidate_windows"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_margin_by_record(meta: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["split", "record_id", "y_class", "original_region"]
    rows = []
    for keys, sub in meta.groupby(group_cols, dropna=False):
        rows.append(
            {
                "split": keys[0],
                "record_id": keys[1],
                "y_class": keys[2],
                "original_region": keys[3],
                "n": int(len(sub)),
                "segment_duration_median": float(sub["segment_duration_sec"].median()),
                "min_margin_median": float(sub["min_margin_sec"].median()),
                "min_margin_p10": float(sub["min_margin_sec"].quantile(0.10)),
                "min_margin_p25": float(sub["min_margin_sec"].quantile(0.25)),
                "min_margin_p75": float(sub["min_margin_sec"].quantile(0.75)),
                "margin_ge_2s_rate": float((sub["min_margin_sec"] >= 2.0).mean()),
                "margin_ge_5s_rate": float((sub["min_margin_sec"] >= 5.0).mean()),
                "margin_ge_10s_rate": float((sub["min_margin_sec"] >= 10.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "record_id", "y_class", "original_region"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        show[col] = show[col].map(lambda x: f"{x:.6g}" if isinstance(x, float) else str(x))
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("\n", " ") for v in row) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    metadata_path: Path,
    atlas_path: Path,
    ann_root: Path,
    meta: pd.DataFrame,
    policy_summary: pd.DataFrame,
    regen_summary: pd.DataFrame,
    margin_summary: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(meta)
    matched = int(meta["matched_segment"].sum())
    unmatched = total - matched
    top_edge = margin_summary.sort_values("min_margin_median").head(12)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Clean BUT Window Policy Experiment Design\n\n")
        f.write("This report is a data/protocol design artifact. It does not change model selection, training data, or checkpoints.\n\n")
        f.write("## Sources\n\n")
        f.write(f"- metadata: `{metadata_path}`\n")
        f.write(f"- atlas: `{atlas_path}`\n")
        f.write(f"- ANN root: `{ann_root}`\n\n")
        f.write("## Audit Result\n\n")
        f.write(f"- Current protocol windows: `{total}`\n")
        f.write(f"- Matched to consensus segments: `{matched}`\n")
        f.write(f"- Unmatched windows: `{unmatched}`\n\n")
        f.write(
            "The previous 10s-duration audit showed that short consensus segments were already dropped. "
            "This stricter audit asks a different question: how far each fixed 10s window is from the "
            "start/end of its consensus-label segment, and how much data remains if we require a clean "
            "interior margin.\n\n"
        )
        f.write("## Fixed 10s Clean-Window Policies\n\n")
        f.write(markdown_table(policy_summary))
        f.write("\n\n")
        f.write("## Regenerated-Center Window Capacity\n\n")
        f.write(
            "These rows estimate how many windows could be generated from the original consensus segments "
            "if we stop anchoring windows at segment starts and instead sample only from the clean interior.\n\n"
        )
        f.write(markdown_table(regen_summary))
        f.write("\n\n")
        f.write("## Edge-Risk Buckets With Smallest Margins\n\n")
        f.write(markdown_table(top_edge))
        f.write("\n\n")
        f.write("## Recommended Experiment Sequence\n\n")
        f.write("1. **P0 protocol sanity:** keep current test as legacy stress, but introduce grouped CV and record-balanced validation. Do not calibrate on a validation set that has been merged into training.\n")
        f.write("2. **P1 strict-clean 10s:** train/evaluate on `margin_ge_2s_drop_outlier` and `clean_core_plus_overlap_margin2` as learnable-body diagnostics. Keep outliers as stress buckets, not deleted final evidence.\n")
        f.write("3. **P2 regenerated center windows:** create a new protocol from raw consensus segments using `center_one_per_segment_margin5` and `center_stride10_margin5`; this avoids first-10s-at-boundary artifacts.\n")
        f.write("4. **P3 variable-length / MIL:** for long segments, feed multiple clean 10s crops or variable-length chunks with masks; aggregate segment-level decisions. This preserves event context better than a single arbitrary 10s crop.\n")
        f.write("5. **P4 model input repair:** dual-view input: physical/global-normalized waveform, robust waveform, derivative recomputed after augmentation, long baseline, and local envelope/log-RMS. Augment raw waveform first, then recompute derived channels.\n")
        f.write("6. **P5 model head repair:** intrinsic-only SQI/event targets, explicit SQI late fusion, QRS/RR query, contact/baseline query, detail query, bad-stress query, and hierarchical bad-vs-nonbad then good-vs-medium classification.\n\n")
        f.write("## Design Decision\n\n")
        f.write(
            "The clean-data path should not simply remove everything difficult. The clean interior subset is for "
            "learning and ablation. The full BUT and bad/outlier stress buckets must remain report-only stress "
            "evaluations so we can see whether the model is genuinely improving or only becoming cleaner by filtering.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--ann-root", type=Path, default=DEFAULT_ANN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    segments = read_consensus_segments(args.ann_root)
    meta = join_atlas(meta, args.atlas)
    annotated = attach_segments(meta, segments)

    policy_summary = summarize_windows(annotated)
    regen_summary = summarize_regeneration(segments)
    margin_summary = summarize_margin_by_record(annotated)

    annotated_path = args.out_dir / "current_10s_window_segment_margins.csv"
    policy_path = args.out_dir / "clean_window_policy_summary.csv"
    regen_path = args.out_dir / "regenerated_center_window_capacity.csv"
    margin_path = args.out_dir / "margin_by_split_record_region_class.csv"
    summary_path = args.out_dir / "clean_window_policy_summary.json"
    report_path = args.report_dir / "clean_but_window_policy_experiment_design.md"

    annotated.to_csv(annotated_path, index=False)
    policy_summary.to_csv(policy_path, index=False)
    regen_summary.to_csv(regen_path, index=False)
    margin_summary.to_csv(margin_path, index=False)
    write_json(
        summary_path,
        {
            "metadata": str(args.metadata),
            "atlas": str(args.atlas),
            "ann_root": str(args.ann_root),
            "n_windows": int(len(annotated)),
            "matched_windows": int(annotated["matched_segment"].sum()),
            "unmatched_windows": int((~annotated["matched_segment"].fillna(False)).sum()),
            "artifacts": {
                "annotated_windows": str(annotated_path),
                "policy_summary": str(policy_path),
                "regenerated_capacity": str(regen_path),
                "margin_by_record": str(margin_path),
                "report": str(report_path),
            },
        },
    )
    write_report(report_path, args.metadata, args.atlas, args.ann_root, annotated, policy_summary, regen_summary, margin_summary)
    print(f"Wrote {report_path}")
    print(policy_summary.to_string(index=False))
    print(regen_summary.to_string(index=False))


if __name__ == "__main__":
    main()
