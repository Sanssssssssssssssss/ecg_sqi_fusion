#!/usr/bin/env python
"""Audit PTB synthetic vs BUT with the same distribution metric contract as v110.

This script is intentionally small and external-only.  It reuses the v81/v110
feature extraction, robust normalization, RBF-MMD, sliced Wasserstein, CDF loss,
domain AUC, PCA-density overlap, and plotting functions so later protocol
versions can be compared with the exact same mouthfeel as the v110 report.

The main table remains subtype-level, matching v110.  A second table adds
class/global rows for the user's "whole PTB generated distribution vs BUT" MMD.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).parent
REPO_ROOT = ANALYSIS_DIR.parents[4]
REPORT_ROOT = (
    REPO_ROOT
    / "reports"
    / "external_benchmarks"
    / "e311_but_node_ladder_tuning_10s_2026_06_08"
    / "analysis"
    / "good_medium_geometry_repair"
)

sys.path.insert(0, str(ANALYSIS_DIR))

import run_v81_distribution_transport as V81  # noqa: E402


DEFAULT_BUT_PROTOCOL = (
    ANALYSIS_DIR
    / "clean_but_protocols"
    / "margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed"
)
DEFAULT_PTB_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "ptb_v112_gm_buffered_large_hybrid_s20260741"
DEFAULT_V110_METRICS = REPORT_ROOT / "v110_hybrid_v108_v109" / "v110_hybrid_distribution_metrics.csv"


def all055_filter(frame: pd.DataFrame) -> pd.Series:
    """Replicate the current all055 medium bypass filter used in q-ladder audits."""
    class_name = frame["class_name"].astype(str)
    subtype = frame.get("display_subtype", frame.get("transport_subtype", pd.Series("", index=frame.index))).astype(str)
    pmed = pd.to_numeric(frame.get("gm_buffer_p_medium", pd.Series(np.nan, index=frame.index)), errors="coerce")
    drop = (
        class_name.eq("medium")
        & subtype.isin(["medium_outlier_or_bad_boundary", "medium_overlap_boundary"])
        & pmed.notna()
        & (pmed < 0.55)
    )
    return ~drop


def load_and_recompute(path: Path, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame, signals = V81.load_protocol(path)
    frame = V81.recompute_protocol_features(frame, signals, label)
    frame = V81.ensure_subtype(frame)
    return frame, signals


def standard_metric_row(
    *,
    tag: str,
    scope: str,
    target: pd.DataFrame,
    synth: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    center, scale = V81.robust_fit(V81.feature_matrix(target, features))
    bz = V81.robust_z(V81.feature_matrix(target, features), center, scale)
    pz = V81.robust_z(V81.feature_matrix(synth, features), center, scale)
    return {
        "tag": tag,
        "scope": scope,
        "but_n": int(len(target)),
        "ptb_n": int(len(synth)),
        "rbf_mmd": V81.rbf_mmd(bz, pz, rng),
        "sliced_wasserstein": V81.sliced_wasserstein(bz, pz, rng),
        "quantile_loss": V81.quantile_loss(bz, pz),
        "domain_auc": V81.domain_auc(bz, pz, rng),
        "pca_density_overlap": V81.pca_density_overlap(bz, pz),
    }


def compute_global_metrics(but: pd.DataFrame, ptb: pd.DataFrame, tag: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    target_all = but.loc[but["split"].astype(str).isin(["train", "val"])].copy()
    features = [f for f in V81.MATCH_FEATURES if f in target_all.columns or f in ptb.columns]
    rows: list[dict[str, Any]] = []
    rows.append(
        standard_metric_row(
            tag=tag,
            scope="all_labels",
            target=target_all,
            synth=ptb,
            features=features,
            rng=rng,
        )
    )
    for cls in V81.CLASS_ORDER:
        b = target_all.loc[target_all["class_name"].astype(str).eq(cls)]
        p = ptb.loc[ptb["class_name"].astype(str).eq(cls)]
        if len(b) >= 5 and len(p) >= 5:
            rows.append(
                standard_metric_row(
                    tag=tag,
                    scope=f"class_{cls}",
                    target=b,
                    synth=p,
                    features=features,
                    rng=rng,
                )
            )
    return pd.DataFrame(rows)


def summarize_subtype_metrics(metric: pd.DataFrame, tag: str) -> pd.DataFrame:
    if metric.empty:
        return pd.DataFrame()
    return (
        metric.groupby("class_name", as_index=False)
        .agg(
            subtype_rows=("subtype", "count"),
            but_n=("but_n", "sum"),
            ptb_n=("ptb_n", "sum"),
            rbf_mmd_median=("rbf_mmd", "median"),
            rbf_mmd_mean=("rbf_mmd", "mean"),
            sliced_wasserstein_median=("sliced_wasserstein", "median"),
            quantile_loss_median=("quantile_loss", "median"),
            domain_auc_median=("domain_auc", "median"),
            pca_density_overlap_median=("pca_density_overlap", "median"),
            pca_density_overlap_mean=("pca_density_overlap", "mean"),
        )
        .assign(tag=tag)
    )


def write_report(
    report_dir: Path,
    tag: str,
    metric: pd.DataFrame,
    global_metric: pd.DataFrame,
    v110_summary: pd.DataFrame | None,
    counts: dict[str, Any],
) -> Path:
    summary = summarize_subtype_metrics(metric, tag)
    lines: list[str] = []
    lines.append(f"# {tag} V110-Metric Distribution Audit")
    lines.append("")
    lines.append("This report intentionally uses the same metric contract as v110:")
    lines.append("")
    lines.append("- v81/v110 47-feature extractor and `MATCH_FEATURES`;")
    lines.append("- BUT train+val robust median/IQR normalization per subtype/scope;")
    lines.append("- RBF-MMD, sliced-Wasserstein, quantile CDF loss, domain AUC, PCA density overlap;")
    lines.append("- subtype-level table as the primary comparison口径.")
    lines.append("")
    lines.append("## Counts")
    lines.append("```json")
    lines.append(json.dumps(counts, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Class Summary: Subtype-Median V110口径")
    lines.append("```")
    lines.append(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("```")
    lines.append("")
    lines.append("## Whole/Class Global Metric")
    lines.append("This is the new supplement requested by the user. It is not a replacement for the v110 subtype-median口径.")
    lines.append("```")
    lines.append(global_metric.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("```")
    if v110_summary is not None and not v110_summary.empty:
        lines.append("")
        lines.append("## v110 Reference: Subtype-Median口径")
        lines.append("```")
        lines.append(v110_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("```")
    lines.append("")
    lines.append("## Files")
    for name in [
        f"{tag}_distribution_metrics.csv",
        f"{tag}_class_subtype_median_summary.csv",
        f"{tag}_global_distribution_metrics.csv",
        f"{tag}_shared_pca.png",
        f"{tag}_key_feature_cdf_overlay.png",
        f"{tag}_rbf_mmd_heatmap.png",
        f"{tag}_feature_median_gap.csv",
    ]:
        lines.append(f"- `{name}`")
    path = report_dir / f"{tag}_v110_metric_audit_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--but-protocol", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--ptb-protocol", default=str(DEFAULT_PTB_PROTOCOL))
    parser.add_argument("--tag", default="ptb_v112_all055_v110metric")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--seed", type=int, default=20260742)
    parser.add_argument("--filter", choices=["none", "all055"], default="all055")
    args = parser.parse_args()

    report_dir = Path(args.report_dir) if args.report_dir else REPORT_ROOT / args.tag
    report_dir.mkdir(parents=True, exist_ok=True)

    but, _ = load_and_recompute(Path(args.but_protocol), "BUT reference")
    ptb, _ = load_and_recompute(Path(args.ptb_protocol), "PTB synthetic")
    before_n = len(ptb)
    if args.filter == "all055":
        ptb = ptb.loc[all055_filter(ptb)].reset_index(drop=True)
    ptb = V81.ensure_subtype(ptb)

    metric, _ = V81.audit_pair(but, ptb, args.tag, report_dir, rng_seed=int(args.seed))
    V81.plot_shared_pca(but, ptb, report_dir, args.tag)
    V81.plot_metric_heatmap(metric, report_dir, args.tag, "rbf_mmd")
    V81.plot_cdf_overlay(but, ptb, report_dir, args.tag)
    global_metric = compute_global_metrics(but, ptb, args.tag, int(args.seed) + 101)
    global_metric.to_csv(V81.long_path(report_dir / f"{args.tag}_global_distribution_metrics.csv"), index=False)
    summary = summarize_subtype_metrics(metric, args.tag)
    summary.to_csv(V81.long_path(report_dir / f"{args.tag}_class_subtype_median_summary.csv"), index=False)

    v110_summary = None
    if DEFAULT_V110_METRICS.exists():
        v110_metric = pd.read_csv(V81.long_path(DEFAULT_V110_METRICS))
        v110_summary = summarize_subtype_metrics(v110_metric, "v110_hybrid")
    counts = {
        "but_protocol": str(Path(args.but_protocol)),
        "ptb_protocol": str(Path(args.ptb_protocol)),
        "filter": args.filter,
        "ptb_rows_before_filter": int(before_n),
        "ptb_rows_after_filter": int(len(ptb)),
        "but_rows": int(len(but)),
        "but_class_counts": but["class_name"].value_counts().to_dict(),
        "ptb_class_counts": ptb["class_name"].value_counts().to_dict(),
    }
    report_path = write_report(report_dir, args.tag, metric, global_metric, v110_summary, counts)
    print(f"[done] report={report_path}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(global_metric.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
