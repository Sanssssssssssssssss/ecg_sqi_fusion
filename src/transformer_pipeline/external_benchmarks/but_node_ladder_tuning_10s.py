"""Original-aware SemiCleanBUT node-ladder tuning.

This runner turns the SemiCleanBUT boundary widening work into a closed loop:
freeze a boundary node, fit PTB synthetic data to that node in 64D/PCA space,
train only the best fits, diagnose the node, then promote before widening.

SemiClean / Original-aware nodes are diagnostic generator targets only.  They
do not replace the formal BUT 10s P1 original benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_clean_core_targeted_generator_grid_10s as targeted
from src.transformer_pipeline.external_benchmarks import but_cleanbut_boundary_relaxation_10s as relax
from src.transformer_pipeline.external_benchmarks import but_original_aware_semiclean_boundary_10s as original_aware
from src.transformer_pipeline.external_benchmarks import but_semiclean_overlap_target_grid_10s as semiclean
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import ensure_but_10s, now_iso, read_json, update_state
from src.transformer_pipeline.external_benchmarks.run import apply_but_thresholds, multiclass_report


ROOT = targeted.ROOT
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_CLEAN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_clean_label_core_10s_2026_06_06"
DEFAULT_SEMICLEAN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_semiclean_overlap_target_grid_10s_2026_06_07"
DEFAULT_SOURCE_BOUNDARY_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_original_aware_semiclean_boundary_10s_2026_06_07"

STATE_NAME = "node_ladder_state.json"
NODE_REGISTRY_NAME = "node_registry.json"
NODE_REGISTRY_CSV = "node_registry.csv"
AGG_LEADERBOARD_NAME = "node64_distance_leaderboard.csv"
SUMMARY_NAME = "node_ladder_training_summary.jsonl"
DIAGNOSTIC_NAME = "node_ladder_diagnostic_metrics.csv"
PROMOTION_NAME = "node_promotion_decisions.csv"

CLASS_NAMES = targeted.CLASS_NAMES
COLORS = targeted.COLORS
REGION_COLORS = original_aware.REGION_COLORS


NODE_DEFS: tuple[dict[str, Any], ...] = (
    {
        "node_id": "N3600_anchor",
        "level": 3600,
        "role": "frozen_anchor",
        "target_acc": 0.95,
        "target_good": 0.94,
        "target_medium": 0.91,
        "target_bad": 0.98,
        "train_default": False,
        "reason": "Current feasible 0.95 node; freeze as sanity baseline.",
    },
    {
        "node_id": "N4200_bridge",
        "level": 4200,
        "role": "active_bridge",
        "target_acc": 0.95,
        "target_good": 0.94,
        "target_medium": 0.91,
        "target_bad": 0.98,
        "train_default": True,
        "reason": "First failing widening node; add good/medium overlap and controlled near-bad shell.",
    },
    {
        "node_id": "N4800_wide",
        "level": 4800,
        "role": "wide_after_bridge",
        "target_acc": 0.95,
        "target_good": 0.94,
        "target_medium": 0.90,
        "target_bad": 0.98,
        "train_default": False,
        "reason": "Wider target after N4200 is stable; increases medium overlap and bad near-boundary coverage.",
    },
    {
        "node_id": "N5200_stress",
        "level": 5200,
        "role": "stress_diagnostic_only",
        "target_acc": 0.95,
        "target_good": 0.92,
        "target_medium": 0.88,
        "target_bad": 0.95,
        "train_default": False,
        "reason": "Stress node with outlier shell; do not promote until N4800 succeeds.",
    },
    {
        "node_id": "N5600_probe",
        "level": 5600,
        "role": "lower_confidence_probe",
        "target_acc": 0.95,
        "target_good": 0.90,
        "target_medium": 0.84,
        "target_bad": 0.93,
        "train_default": False,
        "reason": "Boundary probe after N5200: more good/medium overlap plus controlled low-confidence shell; fit/visual before training.",
    },
    {
        "node_id": "N6000_probe",
        "level": 6000,
        "role": "extreme_boundary_probe",
        "target_acc": 0.95,
        "target_good": 0.88,
        "target_medium": 0.80,
        "target_bad": 0.90,
        "train_default": False,
        "reason": "Extreme stress probe for finding the widest still-learnable 0.95-ish boundary; fit/visual only by default.",
    },
    {
        "node_id": "N6000_goodcover_retry",
        "level": 6000,
        "role": "good_outer_shell_retry",
        "target_acc": 0.95,
        "target_good": 0.92,
        "target_medium": 0.86,
        "target_bad": 0.90,
        "train_default": False,
        "reason": "Same N6000 boundary, but generator/score focus on good 64D outer-shell coverage before widening further.",
    },
    {
        "node_id": "N6400_gm_probe",
        "level": 6400,
        "role": "gm_low_confidence_probe",
        "target_acc": 0.95,
        "target_good": 0.86,
        "target_medium": 0.78,
        "target_bad": 0.90,
        "train_default": False,
        "reason": "Good/medium low-confidence expansion after bad-side saturation; bad remains all available island/near-boundary rows.",
    },
    {
        "node_id": "N6800_gm_probe",
        "level": 6800,
        "role": "gm_outer_shell_probe",
        "target_acc": 0.95,
        "target_good": 0.84,
        "target_medium": 0.76,
        "target_bad": 0.90,
        "train_default": False,
        "reason": "Outer good/medium shell stress probe; use only after N6400 coverage looks sane.",
    },
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def md_table(df: pd.DataFrame, max_rows: int = 18) -> str:
    return targeted.md_table(df, max_rows=max_rows)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _node_defs_by_id() -> dict[str, dict[str, Any]]:
    return {d["node_id"]: dict(d) for d in NODE_DEFS}


def selected_node_defs(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = [s.strip() for s in str(args.nodes).split(",") if s.strip()]
    defs = _node_defs_by_id()
    if not wanted or wanted == ["all"]:
        return [dict(d) for d in NODE_DEFS]
    missing = [n for n in wanted if n not in defs]
    if missing:
        raise ValueError(f"Unknown node ids: {missing}. Known: {list(defs)}")
    return [defs[n] for n in wanted]


def node_dir(args: argparse.Namespace, node_id: str) -> Path:
    return Path(args.out_root) / "nodes" / node_id


def node_report_dir(args: argparse.Namespace, node_id: str) -> Path:
    return Path(args.report_root) / "nodes" / node_id


def _original_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        stage="all",
        out_root=str(Path(args.out_root) / "original_region_boundary"),
        report_root=str(Path(args.report_root) / "original_region_boundary"),
        clean_root=str(args.clean_root),
        semiclean_root=str(args.semiclean_root),
        protocol_out_root=str(args.protocol_out_root),
        protocol_report_root=str(args.protocol_report_root),
        levels=",".join(str(d["level"]) for d in NODE_DEFS),
        target_acc=float(args.target_acc),
        target_acc_band=float(args.target_acc_band),
        batch_size_eval=int(args.batch_size_eval),
        min_region_eval_n=int(args.min_region_eval_n),
        max_plot_rows=int(args.max_plot_rows),
        cpu=bool(args.cpu),
    )


def _load_features(clean_root: Path, target: pd.DataFrame) -> list[str]:
    thresholds = read_json(clean_root / "clean_rule_thresholds.json")
    features = [f for f in list(thresholds.get("features", [])) if f in target.columns]
    if len(features) < 16:
        raise RuntimeError(f"Node target has too few 64D features: {len(features)}")
    return features


def _source_baseline_rows(args: argparse.Namespace) -> pd.DataFrame:
    source = Path(args.source_boundary_root) / original_aware.LEVEL_METRICS_NAME
    if source.exists():
        return pd.read_csv(source)
    fallback = Path(args.out_root) / "original_region_boundary" / original_aware.LEVEL_METRICS_NAME
    return pd.read_csv(fallback) if fallback.exists() else pd.DataFrame()


def _best_existing_for_level(args: argparse.Namespace, level: int) -> dict[str, Any]:
    metrics = _source_baseline_rows(args)
    if metrics.empty or "level_n_per_class" not in metrics:
        return {}
    raw = metrics.loc[
        metrics["prediction_mode"].astype(str).eq("raw")
        & (pd.to_numeric(metrics["level_n_per_class"], errors="coerce") == int(level))
    ].copy()
    if raw.empty:
        return {}
    raw = raw.sort_values(["acc", "macro_f1"], ascending=[False, False])
    return raw.head(1).to_dict("records")[0]


def build_nodes(args: argparse.Namespace) -> pd.DataFrame:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    oargs = _original_args(args)
    Path(oargs.out_root).mkdir(parents=True, exist_ok=True)
    Path(oargs.report_root).mkdir(parents=True, exist_ok=True)

    update_state(out_root / STATE_NAME, status="building_nodes", stage="build_nodes", updated_at=now_iso())
    atlas_path = Path(oargs.out_root) / original_aware.REGION_ATLAS_NAME
    if atlas_path.exists() and not bool(args.force_nodes):
        atlas = pd.read_csv(atlas_path)
    else:
        atlas = original_aware.build_region_atlas(oargs)
        original_aware.plot_region_atlas(oargs, atlas)

    manifest_path = Path(oargs.out_root) / original_aware.REGION_QUOTA_NAME
    required_selection_cols = {f"selected_l{int(d['level'])}" for d in NODE_DEFS}
    if manifest_path.exists() and not bool(args.force_nodes):
        manifest = pd.read_csv(manifest_path)
        if missing_cols := sorted(required_selection_cols - set(manifest.columns)):
            print(f"Rebuilding region-quota manifest with missing node columns: {missing_cols}", flush=True)
            manifest = original_aware.build_region_quota_manifest(oargs)
            original_aware.plot_boundary_levels(oargs, manifest)
            original_aware.make_region_galleries(oargs, manifest)
    else:
        manifest = original_aware.build_region_quota_manifest(oargs)
        original_aware.plot_boundary_levels(oargs, manifest)
        original_aware.make_region_galleries(oargs, manifest)

    features = _load_features(Path(args.clean_root), manifest)
    registry_rows: list[dict[str, Any]] = []
    for node in NODE_DEFS:
        node_id = node["node_id"]
        level = int(node["level"])
        n_dir = node_dir(args, node_id)
        r_dir = node_report_dir(args, node_id)
        n_dir.mkdir(parents=True, exist_ok=True)
        r_dir.mkdir(parents=True, exist_ok=True)
        col = f"selected_l{level}"
        if col not in manifest.columns:
            raise KeyError(f"Missing node selection column {col}")
        target = manifest.loc[_safe_bool(manifest[col])].copy()
        target.to_csv(n_dir / "node_boundary_manifest.csv", index=False)
        target.to_csv(r_dir / "node_boundary_manifest.csv", index=False)

        comp = (
            target.groupby(["class_name", "original_region"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["class_name", "original_region"])
        )
        comp.to_csv(n_dir / "node_region_composition.csv", index=False)
        comp.to_csv(r_dir / "node_region_composition.csv", index=False)

        quant_rows: list[dict[str, Any]] = []
        dist: dict[str, Any] = {
            "node_id": node_id,
            "level_n_per_class": level,
            "feature_count": int(len(features)),
            "selected_counts": {cls: int(target["class_name"].astype(str).eq(cls).sum()) for cls in CLASS_NAMES},
            "region_counts": comp.to_dict("records"),
            "by_class": {},
        }
        for cls in CLASS_NAMES:
            sub = target.loc[target["class_name"].astype(str).eq(cls)]
            dist["by_class"][cls] = {
                "n": int(len(sub)),
                "ambiguous_fraction": float(sub["clean_tier"].astype(str).eq("ambiguous_boundary").mean()) if len(sub) else math.nan,
                "outlier_fraction": float(sub["original_region"].astype(str).eq("outlier_low_confidence").mean()) if len(sub) else math.nan,
                "pc1_median": float(pd.to_numeric(sub["pc1"], errors="coerce").median()) if len(sub) else math.nan,
                "pc2_median": float(pd.to_numeric(sub["pc2"], errors="coerce").median()) if len(sub) else math.nan,
            }
            for feat in features:
                vals = pd.to_numeric(sub[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if vals.empty:
                    continue
                quant_rows.append(
                    {
                        "node_id": node_id,
                        "class_name": cls,
                        "feature": feat,
                        "q05": float(vals.quantile(0.05)),
                        "q25": float(vals.quantile(0.25)),
                        "median": float(vals.quantile(0.50)),
                        "q75": float(vals.quantile(0.75)),
                        "q95": float(vals.quantile(0.95)),
                    }
                )
        write_json(n_dir / "node_target_distributions.json", dist)
        write_json(r_dir / "node_target_distributions.json", dist)
        pd.DataFrame(quant_rows).to_csv(n_dir / "node_target_quantiles.csv", index=False)
        pd.DataFrame(quant_rows).to_csv(r_dir / "node_target_quantiles.csv", index=False)

        pca_bundle = targeted.fit_clean_pca(target, features, int(args.seed))
        pca_target = pca_bundle["clean_projected"][
            ["idx", "class_name", "original_region", "split", "pc1_common", "pc2_common"]
        ].copy()
        pca_target.to_csv(n_dir / "node_pca_target_points.csv", index=False)
        pca_target.to_csv(r_dir / "node_pca_target_points.csv", index=False)

        baseline = _best_existing_for_level(args, level)
        status = "frozen_anchor" if node_id == "N3600_anchor" else "pending"
        if baseline:
            acc = float(baseline.get("acc", math.nan))
            med = float(baseline.get("medium_recall", math.nan))
            bad = float(baseline.get("bad_recall", math.nan))
            if acc >= float(node["target_acc"]) and med >= float(node["target_medium"]) and bad >= float(node["target_bad"]):
                status = "promoted_from_existing" if node_id != "N3600_anchor" else "frozen_anchor"
        registry_rows.append(
            {
                **node,
                "status": status,
                "node_dir": str(n_dir),
                "report_dir": str(r_dir),
                "existing_best_variant": baseline.get("variant_id"),
                "existing_acc": baseline.get("acc"),
                "existing_macro_f1": baseline.get("macro_f1"),
                "existing_good_recall": baseline.get("good_recall"),
                "existing_medium_recall": baseline.get("medium_recall"),
                "existing_bad_recall": baseline.get("bad_recall"),
                "existing_original_but_macro_f1": baseline.get("original_but_macro_f1"),
                "updated_at": now_iso(),
            }
        )

    reg = pd.DataFrame(registry_rows)
    reg.to_csv(out_root / NODE_REGISTRY_CSV, index=False)
    reg.to_csv(report_root / NODE_REGISTRY_CSV, index=False)
    write_json(
        out_root / NODE_REGISTRY_NAME,
        {
            "run_tag": RUN_TAG,
            "created_at": now_iso(),
            "source_boundary_root": str(args.source_boundary_root),
            "nodes": registry_rows,
            "note": "Prediction-free node targets. Original BUT test is report-only and is not used for node or generator selection.",
        },
    )
    write_json(report_root / NODE_REGISTRY_NAME, read_json(out_root / NODE_REGISTRY_NAME))
    update_state(out_root / STATE_NAME, status="nodes_built", stage="build_nodes", completed=len(NODE_DEFS), total=len(NODE_DEFS), updated_at=now_iso())
    write_report(args)
    return reg


def load_node_target(args: argparse.Namespace, node: dict[str, Any]) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    n_dir = node_dir(args, node["node_id"])
    target_path = n_dir / "node_boundary_manifest.csv"
    if not target_path.exists():
        build_nodes(args)
    target = pd.read_csv(target_path)
    features = _load_features(Path(args.clean_root), target)
    return target, features, node


def _node_mutation(node: dict[str, Any], spec: dict[str, Any], idx: int, stage_name: str) -> dict[str, Any]:
    s = copy.deepcopy(spec)
    level = int(node["level"])
    # The widening node changes the generator, not only the filter.  The knobs
    # are intentionally small for N4200 and larger for N4800/N5200.
    if level >= 4200:
        s["good_residual_mix"] = float(s.get("good_residual_mix", 0.010)) + 0.004
        s["good_baseline_mix"] = float(s.get("good_baseline_mix", 0.040)) + 0.010
        s["medium_event_prob"] = min(0.92, float(s.get("medium_event_prob", 0.78)) + 0.04)
        s["medium_inverse_peak_rate"] = min(0.32, float(s.get("medium_inverse_peak_rate", 0.18)) + 0.02)
        s["medium_step_rate"] = min(0.34, float(s.get("medium_step_rate", 0.20)) + 0.03)
        s["medium_ramp_rate"] = min(0.34, float(s.get("medium_ramp_rate", 0.18)) + 0.03)
        s["bad_alt_cluster_prob"] = max(float(s.get("bad_alt_cluster_prob", 0.0)), 0.16 + (idx % 3) * 0.04)
        if str(s.get("bad_alt_family", "none")) == "none":
            s["bad_alt_family"] = ["bad_1530_spike_core", "bad_narrow_oscillatory_core", "bad_bandlimited_disagree_core"][idx % 3]
    if level >= 4800:
        s["good_residual_mix"] = float(s.get("good_residual_mix", 0.014)) + 0.004
        s["good_baseline_mix"] = float(s.get("good_baseline_mix", 0.050)) + 0.008
        s["medium_event_prob"] = min(0.96, float(s.get("medium_event_prob", 0.84)) + 0.04)
        s["medium_step_rate"] = min(0.38, float(s.get("medium_step_rate", 0.26)) + 0.03)
        s["medium_ramp_rate"] = min(0.38, float(s.get("medium_ramp_rate", 0.24)) + 0.03)
        s["medium_contact_rate"] = min(0.02, float(s.get("medium_contact_rate", 0.01)) + 0.004)
        s["bad_alt_cluster_prob"] = max(float(s.get("bad_alt_cluster_prob", 0.0)), 0.24 + (idx % 3) * 0.04)
        s["bad_baseline_step_rate"] = max(float(s.get("bad_baseline_step_rate", 0.0)), 0.06 + (idx % 2) * 0.04)
    if level >= 5200:
        s["good_residual_mix"] = float(s.get("good_residual_mix", 0.018)) + 0.006
        s["medium_event_prob"] = min(0.98, float(s.get("medium_event_prob", 0.88)) + 0.03)
        s["medium_contact_rate"] = min(0.03, float(s.get("medium_contact_rate", 0.015)) + 0.006)
        s["bad_alt_cluster_prob"] = max(float(s.get("bad_alt_cluster_prob", 0.0)), 0.32)
        s["bad_baseline_step_rate"] = max(float(s.get("bad_baseline_step_rate", 0.0)), 0.14)
    if level >= 5600:
        s["good_residual_mix"] = float(s.get("good_residual_mix", 0.024)) + 0.006
        s["good_baseline_mix"] = float(s.get("good_baseline_mix", 0.058)) + 0.008
        s["good_std_jitter"] = min(0.10, float(s.get("good_std_jitter", 0.055)) + 0.018)
        s["good_qrs_boost"] = max(0.30, float(s.get("good_qrs_boost", 0.48)) - 0.04)
        s["medium_event_prob"] = min(0.995, float(s.get("medium_event_prob", 0.91)) + 0.02)
        s["medium_inverse_peak_rate"] = min(0.36, float(s.get("medium_inverse_peak_rate", 0.22)) + 0.03)
        s["medium_step_rate"] = min(0.42, float(s.get("medium_step_rate", 0.30)) + 0.025)
        s["medium_ramp_rate"] = min(0.42, float(s.get("medium_ramp_rate", 0.30)) + 0.025)
        s["medium_contact_rate"] = min(0.04, float(s.get("medium_contact_rate", 0.02)) + 0.006)
        s["bad_alt_cluster_prob"] = max(float(s.get("bad_alt_cluster_prob", 0.0)), 0.40)
        s["bad_baseline_step_rate"] = max(float(s.get("bad_baseline_step_rate", 0.0)), 0.18)
        s["bad_pseudo_peak_rate"] = max(float(s.get("bad_pseudo_peak_rate", 0.0)), 0.18)
    if level >= 6000:
        s["good_residual_mix"] = float(s.get("good_residual_mix", 0.030)) + 0.004
        s["medium_contact_rate"] = min(0.05, float(s.get("medium_contact_rate", 0.026)) + 0.004)
        s["bad_alt_cluster_prob"] = max(float(s.get("bad_alt_cluster_prob", 0.0)), 0.46)
        s["bad_baseline_step_rate"] = max(float(s.get("bad_baseline_step_rate", 0.0)), 0.22)
    goodcover_node = "goodcover" in str(node.get("node_id", "")).lower()
    if goodcover_node or level >= 6400:
        # N6000 diagnostics showed PCA-region coverage can look good while
        # good-class 64D nearest-neighbor distance remains too large.  These
        # modes keep QRS reliable but soften QRS sharpness and restore low
        # amplitude non-QRS texture, targeting the good/medium shell.
        good_shell_modes = (
            {
                "good_mode": "g_but_texture",
                "good_target_std": 0.286,
                "good_std_jitter": 0.080,
                "good_qrs_boost": 0.42,
                "good_qrs_sharp": 0.085,
                "good_nonqrs_smooth": 17,
                "good_nonqrs_blend": 0.78,
                "good_residual_mix": 0.020,
                "good_baseline_mix": 0.058,
                "good_qrsband_noise": 0.0015,
            },
            {
                "good_mode": "g_peak_locked",
                "good_target_std": 0.292,
                "good_std_jitter": 0.086,
                "good_qrs_boost": 0.46,
                "good_qrs_sharp": 0.095,
                "good_nonqrs_smooth": 19,
                "good_nonqrs_blend": 0.82,
                "good_residual_mix": 0.024,
                "good_baseline_mix": 0.064,
                "good_qrsband_noise": 0.0010,
            },
            {
                "good_mode": "g_but_texture",
                "good_target_std": 0.276,
                "good_std_jitter": 0.092,
                "good_qrs_boost": 0.38,
                "good_qrs_sharp": 0.075,
                "good_nonqrs_smooth": 15,
                "good_nonqrs_blend": 0.74,
                "good_residual_mix": 0.018,
                "good_baseline_mix": 0.052,
                "good_qrsband_noise": 0.0025,
            },
            {
                "good_mode": "g_peak_locked",
                "good_target_std": 0.282,
                "good_std_jitter": 0.078,
                "good_qrs_boost": 0.50,
                "good_qrs_sharp": 0.080,
                "good_nonqrs_smooth": 16,
                "good_nonqrs_blend": 0.76,
                "good_residual_mix": 0.016,
                "good_baseline_mix": 0.070,
                "good_qrsband_noise": 0.0015,
            },
        )
        mode = dict(good_shell_modes[(idx - 1) % len(good_shell_modes)])
        if level >= 6400:
            mode["good_nonqrs_blend"] = max(0.70, float(mode["good_nonqrs_blend"]) - 0.04)
            mode["good_nonqrs_smooth"] = max(13, int(mode["good_nonqrs_smooth"]) - 1)
            mode["good_qrsband_noise"] = float(mode["good_qrsband_noise"]) + 0.001
        if level >= 6800:
            mode["good_target_std"] = max(0.260, float(mode["good_target_std"]) - 0.010)
            mode["good_qrs_boost"] = max(0.34, float(mode["good_qrs_boost"]) - 0.04)
            mode["good_qrsband_noise"] = float(mode["good_qrsband_noise"]) + 0.001
        s.update(mode)

    s["node_id"] = node["node_id"]
    s["node_level"] = level
    s["node_role"] = node["role"]
    s["selection_basis"] = (
        f"Original-aware SemiCleanBUT node {node['node_id']} region-aware 64D/PCA target; "
        "no original BUT test metric is used for generator selection."
    )
    base_id = str(s.get("id", f"spec_{idx}"))
    s["base_spec_id"] = base_id
    s["id"] = targeted.stable_id(f"nl_{node['node_id']}_{stage_name}_{idx:03d}_{base_id}")
    return s


def make_node_specs(args: argparse.Namespace, node: dict[str, Any], stage_name: str, prior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if stage_name == "refine" and prior_rows:
        specs = semiclean.make_semiclean_specs(args, "refine", prior_rows)
    else:
        specs = semiclean.make_semiclean_specs(args, "scan", [])
    out = [_node_mutation(node, spec, i, stage_name) for i, spec in enumerate(specs, start=1)]
    return out[: int(args.max_specs if stage_name == "scan" else args.max_refine_specs)]


def _nearest_distance_summary(a: np.ndarray, b: np.ndarray, radius: float) -> dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {"coverage": math.nan, "miss_rate": 1.0, "mean_nearest": math.nan, "norm_mean_nearest": 1.5}
    # Chunked nearest-neighbor distance; avoids a full huge matrix for wide nodes.
    mins = np.full(len(a), np.inf, dtype=float)
    chunk = 512
    for start in range(0, len(a), chunk):
        pts = a[start : start + chunk]
        d = np.sqrt(((pts[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
        mins[start : start + chunk] = np.minimum(mins[start : start + chunk], d.min(axis=1))
    return {
        "coverage": float(np.mean(mins <= float(radius))),
        "miss_rate": float(np.mean(mins > float(radius))),
        "mean_nearest": float(np.mean(mins)),
        "norm_mean_nearest": float(np.clip(np.mean(mins) / max(float(radius), 1e-6), 0.0, 1.5)),
    }


def node_region_metrics(
    row: dict[str, Any],
    node: dict[str, Any],
    pca_bundle: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    vdir = Path(str(row.get("variant_dir", "")))
    proj_path = vdir / targeted.PROJECTION_NAME
    if not proj_path.exists():
        return {"node_region_score": 1.5}
    synth = pd.read_csv(proj_path)
    target = pca_bundle["clean_projected"].copy()
    rng = np.random.default_rng(int(seed))
    checks = [
        ("good", "clean_core", "good_clean_core", 0.85, 0.10),
        ("good", "good_medium_overlap", "good_gm_overlap", 0.95, 0.15),
        ("medium", "clean_core", "medium_clean_core", 0.85, 0.10),
        ("medium", "good_medium_overlap", "medium_gm_overlap", 1.05, 0.22),
        ("medium", "medium_bad_overlap", "medium_mb_overlap", 1.10, 0.08),
        ("bad", "right_bad_island", "bad_right_island", 0.65, 0.24),
        ("bad", "near_bad_boundary", "bad_near_boundary", 0.95, 0.11),
    ]
    out: dict[str, float] = {}
    score_parts: list[float] = []
    weights: list[float] = []
    for cls, region, prefix, radius, weight in checks:
        tsub = target.loc[
            target["class_name"].astype(str).eq(cls)
            & target.get("original_region", pd.Series("", index=target.index)).astype(str).eq(region),
            ["pc1_common", "pc2_common"],
        ].dropna()
        if len(tsub) > 900:
            take = rng.choice(np.arange(len(tsub)), size=900, replace=False)
            tpts = tsub.iloc[take].to_numpy(dtype=float)
        else:
            tpts = tsub.to_numpy(dtype=float)
        spts = synth.loc[synth["class_name"].astype(str).eq(cls), ["pc1_common", "pc2_common"]].dropna().to_numpy(dtype=float)
        summary = _nearest_distance_summary(tpts, spts, radius)
        for k, v in summary.items():
            out[f"{prefix}_{k}"] = v
        if len(tpts) >= 20 and math.isfinite(summary["norm_mean_nearest"]):
            penalty = 0.55 * summary["miss_rate"] + 0.45 * summary["norm_mean_nearest"]
            score_parts.append(float(penalty))
            weights.append(float(weight))
    if weights:
        out["node_region_score"] = float(np.average(score_parts, weights=weights))
    else:
        out["node_region_score"] = 1.5
    return out


def node_score(row: dict[str, Any], node_metrics: dict[str, float]) -> float:
    good = float(row.get("good_64d_KS", 1.0))
    medium = float(row.get("medium_64d_KS", 1.0))
    bad = float(row.get("bad_64d_KS", 1.0))
    node_id = str(row.get("node_id", "")).lower()
    if "goodcover" in node_id or str(row.get("node_level", "")) in {"6400", "6800"}:
        return float(
            0.22 * good
            + 0.13 * medium
            + 0.12 * bad
            + 0.10 * float(row.get("medium_PCA_range_gap", 1.0))
            + 0.16 * float(row.get("good_PCA_range_gap", 1.0))
            + 0.07 * float(row.get("bad_PCA_covariance_gap", 1.0))
            + 0.20 * float(node_metrics.get("node_region_score", 1.5))
        )
    return float(
        0.12 * good
        + 0.17 * medium
        + 0.16 * bad
        + 0.12 * float(row.get("medium_PCA_range_gap", 1.0))
        + 0.08 * float(row.get("good_PCA_range_gap", 1.0))
        + 0.10 * float(row.get("bad_PCA_covariance_gap", 1.0))
        + 0.25 * float(node_metrics.get("node_region_score", 1.5))
    )


def node_leaderboard_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    keys = [
        "node_id",
        "node_level",
        "variant_id",
        "family",
        "node_score",
        "node_region_score",
        "good_64d_KS",
        "medium_64d_KS",
        "bad_64d_KS",
        "class_worst_64d_KS",
        "good_gm_overlap_coverage",
        "medium_gm_overlap_coverage",
        "medium_mb_overlap_coverage",
        "bad_right_island_coverage",
        "bad_near_boundary_coverage",
        "good_PCA_range_gap",
        "medium_PCA_range_gap",
        "bad_PCA_covariance_gap",
        "domain_separability",
        "variant_dir",
    ]
    df = pd.DataFrame([{k: r.get(k) for k in keys} for r in rows])
    if df.empty:
        return df
    df = df.sort_values(["node_score", "node_region_score", "medium_64d_KS", "bad_64d_KS"], na_position="last").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    return df


def score_node(args: argparse.Namespace, node: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    n_dir = node_dir(args, node["node_id"])
    r_dir = node_report_dir(args, node["node_id"])
    n_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)
    target, features, _ = load_node_target(args, node)
    pca_bundle = targeted.fit_clean_pca(target, features, int(args.seed))
    json_path = n_dir / "node64_distance_leaderboard.json"
    prior_rows = read_json(json_path).get("rows", []) if json_path.exists() else []
    existing = prior_rows if stage_name == "refine" else []
    specs = make_node_specs(args, node, stage_name, prior_rows)
    sample_per_class = int(args.sample_per_class_refine if stage_name == "refine" else args.sample_per_class_scan)
    write_json(n_dir / "node_target_grid_specs.json", {"node": node, "stage": stage_name, "rows": specs})
    write_json(r_dir / "node_target_grid_specs.json", read_json(n_dir / "node_target_grid_specs.json"))

    rows: list[dict[str, Any]] = []
    update_state(out_root / STATE_NAME, status=f"{stage_name}_node_fit", stage="fit_generator", node_id=node["node_id"], total=len(specs), updated_at=now_iso())
    for i, spec in enumerate(specs, start=1):
        update_state(
            out_root / STATE_NAME,
            status=f"{stage_name}_node_fit",
            stage="fit_generator",
            node_id=node["node_id"],
            current=spec["id"],
            completed=i - 1,
            total=len(specs),
            updated_at=now_iso(),
        )
        try:
            row = targeted.score_one(args, spec, target, features, pca_bundle, sample_per_class)
            row["node_id"] = node["node_id"]
            row["node_level"] = int(node["level"])
            node_metrics = node_region_metrics(row, node, pca_bundle, int(args.seed))
            row.update(node_metrics)
            row["node_score"] = node_score(row, node_metrics)
            row["score"] = row["node_score"]
            rows.append(row)
        except Exception as exc:
            fail = {
                "node_id": node["node_id"],
                "node_level": int(node["level"]),
                "variant_id": str(spec.get("id")),
                "family": spec.get("family"),
                "spec": spec,
                "status": "failed",
                "error": str(exc),
                "node_score": 999.0,
                "score": 999.0,
            }
            rows.append(fail)
            append_jsonl(n_dir / "node64_scan_failures.jsonl", fail)
        if i == 1 or i % 6 == 0:
            live = node_leaderboard_frame(existing + rows)
            live.to_csv(n_dir / "node64_distance_leaderboard.csv", index=False)
            live.to_csv(r_dir / "node64_distance_leaderboard.csv", index=False)

    combined = sorted(existing + rows, key=lambda r: float(r.get("node_score", r.get("score", 999.0))))
    write_json(json_path, {"node": node, "stage": stage_name, "rows": combined})
    write_json(r_dir / "node64_distance_leaderboard.json", read_json(json_path))
    lb = node_leaderboard_frame(combined)
    lb.to_csv(n_dir / "node64_distance_leaderboard.csv", index=False)
    lb.to_csv(r_dir / "node64_distance_leaderboard.csv", index=False)
    write_node_visuals(args, node, target, features, pca_bundle, combined)
    write_report(args)
    update_state(
        out_root / STATE_NAME,
        status=f"{stage_name}_node_fit_complete",
        stage="fit_generator",
        node_id=node["node_id"],
        completed=len(rows),
        total=len(specs),
        best_variant=combined[0].get("variant_id") if combined else None,
        best_node_score=combined[0].get("node_score") if combined else None,
        updated_at=now_iso(),
    )
    return combined


def write_node_visuals(
    args: argparse.Namespace,
    node: dict[str, Any],
    target: pd.DataFrame,
    features: list[str],
    pca_bundle: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    n_dir = node_dir(args, node["node_id"])
    r_dir = node_report_dir(args, node["node_id"])
    fig_dir = r_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda r: float(r.get("node_score", r.get("score", 999.0)))):
        vid = str(row.get("variant_id", ""))
        if not vid or vid in seen or float(row.get("node_score", 999.0)) >= 999.0:
            continue
        if not Path(str(row.get("variant_dir", ""))).exists():
            continue
        top_rows.append(row)
        seen.add(vid)
        if len(top_rows) >= int(args.visual_top_n):
            break
    if not top_rows:
        return

    target_plot_parts: list[pd.DataFrame] = []
    target_proj = pca_bundle["clean_projected"].copy()
    for i, cls in enumerate(CLASS_NAMES):
        sub = target_proj.loc[target_proj["class_name"].astype(str).eq(cls)].copy()
        if len(sub) > int(args.clean_plot_rows_per_class):
            sub = sub.sample(int(args.clean_plot_rows_per_class), random_state=int(args.seed) + i)
        target_plot_parts.append(sub)
    target_plot = pd.concat(target_plot_parts, ignore_index=True)
    target_out = target_plot[["class_name", "original_region", "pc1_common", "pc2_common"]].copy()
    target_out["source"] = node["node_id"]
    target_out["variant_id"] = node["node_id"]
    projected_rows = [target_out]
    for row in top_rows:
        proj_path = Path(str(row["variant_dir"])) / targeted.PROJECTION_NAME
        if proj_path.exists():
            proj = pd.read_csv(proj_path)
        else:
            full = pd.read_csv(Path(str(row["variant_dir"])) / "synthetic_atlas64_features.csv")
            xy = targeted.project_features(full, features, pca_bundle)
            proj = pd.DataFrame(
                {
                    "variant_id": row["variant_id"],
                    "source": "PTB synthetic",
                    "class_name": full["class_name"].astype(str),
                    "pc1_common": xy[:, 0],
                    "pc2_common": xy[:, 1],
                }
            )
        proj["node_score"] = row.get("node_score", math.nan)
        projected_rows.append(proj)
    projected = pd.concat(projected_rows, ignore_index=True)
    projected.to_csv(n_dir / "node64_pca_projection.csv", index=False)
    projected.to_csv(r_dir / "node64_pca_projection.csv", index=False)

    _plot_node_overlay(projected, top_rows, node, fig_dir / "top_rules_node_overlay.png")
    _plot_best_node_overlay(projected, top_rows[0], node, fig_dir / "best_rule_node_overlay.png")
    _plot_node_distance_bars(node_leaderboard_frame(rows), node, fig_dir / "node_classwise_distance_bars.png")
    _plot_region_composition(target, node, fig_dir / "node_region_composition.png")


def _plot_node_overlay(projected: pd.DataFrame, top_rows: list[dict[str, Any]], node: dict[str, Any], out_path: Path) -> None:
    target = projected.loc[projected["variant_id"].astype(str).eq(node["node_id"])]
    variants = [str(r["variant_id"]) for r in top_rows]
    ncols = 3
    nrows = int(math.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.5 * nrows), squeeze=False)
    for ax, vid in zip(axes.ravel(), variants):
        for region, color in REGION_COLORS.items():
            sub = target.loc[target.get("original_region", pd.Series("", index=target.index)).astype(str).eq(region)]
            ax.scatter(sub["pc1_common"], sub["pc2_common"], s=5, c=color, alpha=0.13, linewidths=0)
        vf = projected.loc[projected["variant_id"].astype(str).eq(vid)]
        for cls in CLASS_NAMES:
            sdf = vf.loc[vf["class_name"].astype(str).eq(cls)]
            ax.scatter(sdf["pc1_common"], sdf["pc2_common"], s=22, c=COLORS[cls], alpha=0.78, marker="x", linewidths=1.0)
        row = next(r for r in top_rows if str(r["variant_id"]) == vid)
        ax.set_title(
            f"{vid}\nnode={float(row.get('node_score', math.nan)):.3f} "
            f"med/bad={float(row.get('medium_64d_KS', math.nan)):.3f}/{float(row.get('bad_64d_KS', math.nan)):.3f}",
            fontsize=8,
        )
        ax.set_xlabel("node 64D-PC1")
        ax.set_ylabel("node 64D-PC2")
        ax.grid(alpha=0.15)
    for ax in axes.ravel()[len(variants) :]:
        ax.axis("off")
    fig.suptitle(f"{node['node_id']} synthetic-vs-node 64D overlays", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_node_overlay(projected: pd.DataFrame, row: dict[str, Any], node: dict[str, Any], out_path: Path) -> None:
    target = projected.loc[projected["variant_id"].astype(str).eq(node["node_id"])]
    vf = projected.loc[projected["variant_id"].astype(str).eq(row["variant_id"])]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for region, color in REGION_COLORS.items():
        sub = target.loc[target.get("original_region", pd.Series("", index=target.index)).astype(str).eq(region)]
        ax.scatter(sub["pc1_common"], sub["pc2_common"], s=9, c=color, alpha=0.14, linewidths=0, label=f"target {region}")
    for cls in CLASS_NAMES:
        sub = vf.loc[vf["class_name"].astype(str).eq(cls)]
        ax.scatter(sub["pc1_common"], sub["pc2_common"], s=36, c=COLORS[cls], marker="x", alpha=0.82, linewidths=1.1, label=f"synth {cls}")
    ax.set_title(f"Best {node['node_id']} fit\n{row['variant_id']} | node score={float(row.get('node_score', math.nan)):.3f}")
    ax.set_xlabel("node 64D-PC1")
    ax.set_ylabel("node 64D-PC2")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_node_distance_bars(df: pd.DataFrame, node: dict[str, Any], out_path: Path) -> None:
    if df.empty:
        return
    view = df.head(min(24, len(df))).copy()
    labels = [str(v)[:42] for v in view["variant_id"]]
    x = np.arange(len(view))
    width = 0.20
    fig, ax = plt.subplots(figsize=(max(12, len(view) * 0.62), 5.6))
    ax.bar(x - width, view["good_64d_KS"], width, label="good 64D", color=COLORS["good"], alpha=0.75)
    ax.bar(x, view["medium_64d_KS"], width, label="medium 64D", color=COLORS["medium"], alpha=0.75)
    ax.bar(x + width, view["bad_64d_KS"], width, label="bad 64D", color=COLORS["bad"], alpha=0.75)
    ax.plot(x, view["node_region_score"], color="#334155", marker="o", lw=1.0, label="region coverage penalty")
    ax.plot(x, view["node_score"], color="#111827", marker="x", lw=1.0, label="node score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("distance / penalty; lower is better")
    ax.set_title(f"{node['node_id']} 64D and region-aware distance")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_region_composition(target: pd.DataFrame, node: dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    bottom = np.zeros(len(CLASS_NAMES), dtype=float)
    for region, color in REGION_COLORS.items():
        vals = [float((target.loc[target["class_name"].astype(str).eq(cls), "original_region"].astype(str) == region).sum()) for cls in CLASS_NAMES]
        ax.bar(CLASS_NAMES, vals, bottom=bottom, label=region, color=color, alpha=0.82)
        bottom += np.asarray(vals, dtype=float)
    ax.set_ylabel("selected windows")
    ax.set_title(f"{node['node_id']} target region composition")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fit_generator(args: argparse.Namespace) -> None:
    if not (Path(args.out_root) / NODE_REGISTRY_NAME).exists():
        build_nodes(args)
    all_rows: list[dict[str, Any]] = []
    for node in selected_node_defs(args):
        if node["node_id"] == "N3600_anchor" and not bool(args.include_anchor_fit):
            continue
        rows = score_node(args, node, "scan")
        if bool(args.auto_refine):
            best = min(rows, key=lambda r: float(r.get("node_score", 999.0))) if rows else {}
            if float(best.get("node_region_score", 1.5)) > float(args.refine_region_threshold):
                rows = score_node(args, node, "refine")
        all_rows.extend(rows)
    write_aggregate_leaderboard(args)


def write_aggregate_leaderboard(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node in NODE_DEFS:
        path = node_dir(args, node["node_id"]) / "node64_distance_leaderboard.json"
        if path.exists():
            rows.extend(read_json(path).get("rows", []))
    df = node_leaderboard_frame(rows)
    df.to_csv(Path(args.out_root) / AGG_LEADERBOARD_NAME, index=False)
    report_path = Path(args.report_root) / AGG_LEADERBOARD_NAME
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)
    except OSError as exc:
        print(f"Warning: could not mirror aggregate leaderboard to report root: {report_path}: {exc}", flush=True)
    return df


def _training_rows_for_node(args: argparse.Namespace, node: dict[str, Any]) -> list[dict[str, Any]]:
    path = node_dir(args, node["node_id"]) / "node64_distance_leaderboard.json"
    if not path.exists():
        score_node(args, node, "scan")
    rows = read_json(path).get("rows", [])
    rows = [r for r in rows if Path(str(r.get("variant_dir", ""))).exists() and float(r.get("node_score", 999.0)) < 999.0]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        vid = str(row.get("variant_id", ""))
        if vid and vid not in seen and len(selected) < int(args.top_quick):
            selected.append(row)
            seen.add(vid)

    for row in sorted(rows, key=lambda r: float(r.get("node_score", 999.0))):
        add(row)
        if len(selected) >= max(1, int(math.ceil(float(args.top_quick) * 0.65))):
            break
    for key in ("medium_64d_KS", "bad_64d_KS", "medium_gm_overlap_miss_rate", "bad_near_boundary_miss_rate"):
        for row in sorted(rows, key=lambda r, k=key: float(r.get(k, 999.0))):
            add(row)
            if len(selected) >= int(args.top_quick):
                break
        if len(selected) >= int(args.top_quick):
            break
    return selected


def train_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not bool(args.run_training):
        raise RuntimeError("Node quick training requires --run_training.")
    out_rows: list[dict[str, Any]] = []
    losses = [s.strip() for s in str(args.loss_variants).split(",") if s.strip()]
    existing_pairs = {
        (str(r.get("node_id")), str(r.get("base_variant_id") or r.get("spec", {}).get("base_variant_id") or ""), str(r.get("cls_loss_variant") or r.get("spec", {}).get("cls_loss_variant") or ""))
        for r in load_jsonl(Path(args.out_root) / SUMMARY_NAME)
        if int(r.get("returncode", 1) or 1) == 0
    }
    explicit_nodes = [s.strip() for s in str(args.nodes).split(",") if s.strip()]
    node_was_explicit = bool(explicit_nodes and explicit_nodes != ["all"])
    for node in selected_node_defs(args):
        if not node_was_explicit and not bool(node.get("train_default", False)) and not bool(args.train_all_nodes):
            continue
        selected = _training_rows_for_node(args, node)
        total = len(selected) * max(1, len(losses))
        update_state(Path(args.out_root) / STATE_NAME, status="node_quick_training", stage="quick", node_id=node["node_id"], total=total, updated_at=now_iso())
        done = 0
        for row in selected:
            for loss in losses:
                base_variant_id = str(row.get("variant_id"))
                if (str(node["node_id"]), base_variant_id, loss) in existing_pairs:
                    continue
                train_row = copy.deepcopy(row)
                train_row["spec"] = copy.deepcopy(row["spec"])
                train_row["spec"]["base_variant_id"] = base_variant_id
                train_row["spec"]["node_id"] = node["node_id"]
                train_row["spec"]["cls_loss_variant"] = loss
                train_row["spec"]["id"] = targeted.stable_id(f"{row['variant_id']}_{loss}")
                old_loss = args.cls_loss_variant
                args.cls_loss_variant = loss
                update_state(
                    Path(args.out_root) / STATE_NAME,
                    status="node_quick_training",
                    stage="quick",
                    node_id=node["node_id"],
                    current=train_row["spec"]["id"],
                    completed=done,
                    total=total,
                    updated_at=now_iso(),
                )
                result = targeted.train_one(args, train_row, "quick")
                args.cls_loss_variant = old_loss
                result["node_id"] = node["node_id"]
                result["node_level"] = int(node["level"])
                result["base_variant_id"] = row.get("variant_id")
                result["cls_loss_variant"] = loss
                append_jsonl(Path(args.out_root) / SUMMARY_NAME, result)
                append_jsonl(node_dir(args, node["node_id"]) / SUMMARY_NAME, result)
                out_rows.append(result)
                if int(result.get("returncode", 1) or 1) == 0:
                    existing_pairs.add((str(node["node_id"]), base_variant_id, loss))
                done += 1
    update_state(Path(args.out_root) / STATE_NAME, status="node_quick_complete", stage="quick", completed=len(out_rows), total=len(out_rows), updated_at=now_iso())
    write_report(args)
    return out_rows


def train_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not bool(args.run_training):
        raise RuntimeError("Node full training requires --run_training.")
    diag = pd.read_csv(Path(args.out_root) / DIAGNOSTIC_NAME) if (Path(args.out_root) / DIAGNOSTIC_NAME).exists() else evaluate_node_diagnostics(args)
    quick_rows = _node_training_rows(args)
    rows_by_spec = {str(r.get("spec", {}).get("id", "")): r for r in quick_rows}
    out_rows: list[dict[str, Any]] = []
    explicit_nodes = [s.strip() for s in str(args.nodes).split(",") if s.strip()]
    node_was_explicit = bool(explicit_nodes and explicit_nodes != ["all"])
    for node in selected_node_defs(args):
        if not node_was_explicit and not bool(node.get("train_default", False)) and not bool(args.train_all_nodes):
            continue
        sub = diag.loc[
            diag["node_id"].astype(str).eq(node["node_id"])
            & diag["prediction_mode"].astype(str).eq("raw")
        ].copy() if not diag.empty else pd.DataFrame()
        if sub.empty:
            continue
        selected_ids = (
            sub.sort_values(["acc", "macro_f1", "bad_recall", "medium_recall"], ascending=[False, False, False, False])
            .head(int(args.top_full))["variant_id"]
            .astype(str)
            .tolist()
        )
        update_state(Path(args.out_root) / STATE_NAME, status="node_full_training", stage="full", node_id=node["node_id"], total=len(selected_ids), updated_at=now_iso())
        for i, spec_id in enumerate(selected_ids, start=1):
            row = copy.deepcopy(rows_by_spec.get(spec_id))
            if not row:
                continue
            loss = str(row.get("cls_loss_variant") or row.get("spec", {}).get("cls_loss_variant") or args.cls_loss_variant)
            old_loss = args.cls_loss_variant
            args.cls_loss_variant = loss
            update_state(
                Path(args.out_root) / STATE_NAME,
                status="node_full_training",
                stage="full",
                node_id=node["node_id"],
                current=spec_id,
                completed=i - 1,
                total=len(selected_ids),
                updated_at=now_iso(),
            )
            result = targeted.train_one(args, row, "full")
            args.cls_loss_variant = old_loss
            result["node_id"] = node["node_id"]
            result["node_level"] = int(node["level"])
            result["base_variant_id"] = row.get("base_variant_id")
            result["cls_loss_variant"] = loss
            append_jsonl(Path(args.out_root) / SUMMARY_NAME, result)
            append_jsonl(node_dir(args, node["node_id"]) / SUMMARY_NAME, result)
            out_rows.append(result)
    update_state(Path(args.out_root) / STATE_NAME, status="node_full_complete", stage="full", completed=len(out_rows), total=len(out_rows), updated_at=now_iso())
    write_report(args)
    return out_rows


def _node_training_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(args.out_root) / SUMMARY_NAME)
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        spec_id = str(row.get("spec", {}).get("id", ""))
        if not spec_id or int(row.get("returncode", 1)) != 0:
            continue
        ckpt = row.get("but_10s_eval", {}).get("checkpoint")
        if ckpt and Path(str(ckpt)).exists():
            by_key[spec_id] = row
    return list(by_key.values())


def evaluate_node_diagnostics(args: argparse.Namespace) -> pd.DataFrame:
    if not (Path(args.out_root) / NODE_REGISTRY_NAME).exists():
        build_nodes(args)
    rows = _node_training_rows(args)
    if not rows:
        return pd.DataFrame()
    X, meta = ensure_but_10s(args)
    y = meta["y"].to_numpy(dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(args.cpu) else "cpu")
    metric_rows: list[dict[str, Any]] = []
    for node in NODE_DEFS:
        n_id = node["node_id"]
        manifest_path = node_dir(args, n_id) / "node_boundary_manifest.csv"
        if not manifest_path.exists():
            continue
        manifest = pd.read_csv(manifest_path)
        idx = manifest["idx"].astype(int).to_numpy()
        node_rows = [r for r in rows if str(r.get("node_id", "")) == n_id]
        for i, row in enumerate(node_rows, start=1):
            spec_id = str(row.get("spec", {}).get("id", ""))
            pred_cache = Path(args.out_root) / "predictions" / f"{spec_id}_probs.npz"
            pred_cache.parent.mkdir(parents=True, exist_ok=True)
            update_state(Path(args.out_root) / STATE_NAME, status="node_diagnostic", stage="diagnostic", node_id=n_id, current=spec_id, completed=i - 1, total=len(node_rows), updated_at=now_iso())
            if pred_cache.exists():
                probs = np.load(pred_cache)["probs"].astype(np.float32)
            else:
                probs = relax.predict_probs(Path(row["but_10s_eval"]["checkpoint"]), X, int(args.batch_size_eval), device)
                np.savez_compressed(pred_cache, probs=probs.astype(np.float32))
            raw_pred = np.argmax(probs, axis=1).astype(np.int64)
            cal = row.get("but_10s_eval", {}).get("calibration", {})
            cal_pred = apply_but_thresholds(probs, float(cal.get("t_good", 0.5)), float(cal.get("t_bad", 0.5)))
            original = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
            for mode, pred in (("raw", raw_pred), ("calibrated", cal_pred)):
                rep = multiclass_report(y[idx], pred[idx], probs[idx])
                rec = rep.get("recall_good_medium_bad", [math.nan, math.nan, math.nan])
                metric_rows.append(
                    {
                        "node_id": n_id,
                        "node_level": int(node["level"]),
                        "variant_id": spec_id,
                        "base_variant_id": row.get("base_variant_id"),
                        "cls_loss_variant": row.get("cls_loss_variant"),
                        "prediction_mode": mode,
                        "n_total": int(len(idx)),
                        "acc": rep.get("acc"),
                        "balanced_acc": rep.get("balanced_acc"),
                        "macro_f1": rep.get("macro_f1"),
                        "good_recall": rec[0],
                        "medium_recall": rec[1],
                        "bad_recall": rec[2],
                        "confusion_3x3": json.dumps(rep.get("confusion_3x3")),
                        "original_but_acc": original.get("acc"),
                        "original_but_macro_f1": original.get("macro_f1"),
                        **row.get("clean64_distance", {}),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(Path(args.out_root) / DIAGNOSTIC_NAME, index=False)
    metrics.to_csv(Path(args.report_root) / DIAGNOSTIC_NAME, index=False)
    write_report(args)
    update_state(Path(args.out_root) / STATE_NAME, status="node_diagnostic_complete", stage="diagnostic", completed=len(rows), total=len(rows), updated_at=now_iso())
    return metrics


def promote_nodes(args: argparse.Namespace) -> pd.DataFrame:
    metrics_path = Path(args.out_root) / DIAGNOSTIC_NAME
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else evaluate_node_diagnostics(args)
    decisions: list[dict[str, Any]] = []
    reg = read_json(Path(args.out_root) / NODE_REGISTRY_NAME) if (Path(args.out_root) / NODE_REGISTRY_NAME).exists() else {"nodes": []}
    reg_by_id = {str(n["node_id"]): dict(n) for n in reg.get("nodes", [])}
    for node in NODE_DEFS:
        n_id = node["node_id"]
        sub = metrics.loc[metrics["node_id"].astype(str).eq(n_id)].copy() if not metrics.empty else pd.DataFrame()
        existing = _best_existing_for_level(args, int(node["level"]))
        source = "trained"
        if sub.empty and existing:
            sub = pd.DataFrame(
                [
                    {
                        "node_id": n_id,
                        "variant_id": existing.get("variant_id"),
                        "prediction_mode": "raw",
                        "acc": existing.get("acc"),
                        "macro_f1": existing.get("macro_f1"),
                        "good_recall": existing.get("good_recall"),
                        "medium_recall": existing.get("medium_recall"),
                        "bad_recall": existing.get("bad_recall"),
                        "original_but_macro_f1": existing.get("original_but_macro_f1"),
                    }
                ]
            )
            source = "existing"
        if sub.empty:
            decision = {
                "node_id": n_id,
                "status": "no_metrics",
                "promoted": False,
                "reason": "No trained or existing diagnostic metrics available.",
            }
        else:
            numeric_cols = ["acc", "macro_f1", "good_recall", "medium_recall", "bad_recall"]
            for col in numeric_cols:
                if col in sub.columns:
                    sub[col] = pd.to_numeric(sub[col], errors="coerce")
            eligible = sub.loc[
                sub["acc"].ge(float(node["target_acc"]))
                & sub["good_recall"].ge(float(node["target_good"]))
                & sub["medium_recall"].ge(float(node["target_medium"]))
                & sub["bad_recall"].ge(float(node["target_bad"]))
            ].copy()
            ranked = eligible if not eligible.empty else sub
            best = ranked.sort_values(["acc", "macro_f1"], ascending=[False, False]).head(1).to_dict("records")[0]
            acc = float(best.get("acc", 0.0))
            good = float(best.get("good_recall", 0.0))
            med = float(best.get("medium_recall", 0.0))
            bad = float(best.get("bad_recall", 0.0))
            promoted = not eligible.empty
            decision = {
                "node_id": n_id,
                "node_level": int(node["level"]),
                "best_variant": best.get("variant_id"),
                "metric_source": source,
                "prediction_mode": best.get("prediction_mode"),
                "acc": acc,
                "macro_f1": best.get("macro_f1"),
                "good_recall": good,
                "medium_recall": med,
                "bad_recall": bad,
                "original_but_macro_f1": best.get("original_but_macro_f1"),
                "target_acc": node["target_acc"],
                "target_good": node["target_good"],
                "target_medium": node["target_medium"],
                "target_bad": node["target_bad"],
                "promoted": bool(promoted),
                "status": "promoted" if promoted else "needs_generator_refine",
                "reason": "Node meets promotion gates." if promoted else "Do not widen yet; refine this node generator/loss first.",
            }
        decisions.append(decision)
        if n_id in reg_by_id:
            reg_by_id[n_id].update(
                {
                    "status": decision["status"],
                    "promoted": decision.get("promoted", False),
                    "promotion_reason": decision["reason"],
                    "promotion_updated_at": now_iso(),
                }
            )
    out = pd.DataFrame(decisions)
    out.to_csv(Path(args.out_root) / PROMOTION_NAME, index=False)
    out.to_csv(Path(args.report_root) / PROMOTION_NAME, index=False)
    reg["nodes"] = [reg_by_id.get(n["node_id"], n) for n in NODE_DEFS]
    write_json(Path(args.out_root) / NODE_REGISTRY_NAME, reg)
    write_json(Path(args.report_root) / NODE_REGISTRY_NAME, reg)
    write_report(args)
    return out


def write_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    reg = read_csv_safe(out_root / NODE_REGISTRY_CSV)
    agg = read_csv_safe(out_root / AGG_LEADERBOARD_NAME)
    if agg.empty:
        agg = write_aggregate_leaderboard(args)
    diag = read_csv_safe(out_root / DIAGNOSTIC_NAME)
    promo = read_csv_safe(out_root / PROMOTION_NAME)
    train_rows = []
    for r in load_jsonl(out_root / SUMMARY_NAME):
        rep = r.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [math.nan, math.nan, math.nan])
        train_rows.append(
            {
                "node_id": r.get("node_id"),
                "mode": r.get("mode"),
                "variant_id": r.get("spec", {}).get("id"),
                "base_variant_id": r.get("base_variant_id"),
                "loss": r.get("cls_loss_variant"),
                "returncode": r.get("returncode"),
                "original_acc": rep.get("acc"),
                "original_macro_f1": rep.get("macro_f1"),
                "original_good": rec[0],
                "original_medium": rec[1],
                "original_bad": rec[2],
            }
        )
    train_df = pd.DataFrame(train_rows)
    if not train_df.empty:
        train_df.to_csv(out_root / "node_ladder_metric_join.csv", index=False)
        train_df.to_csv(report_root / "node_ladder_metric_join.csv", index=False)

    lines = [
        "# Original-Aware SemiCleanBUT Node Ladder",
        "",
        "This package freezes each SemiClean boundary as a node, fits PTB synthetic data to that node in 64D/PCA space, then trains and promotes only after the node passes. It is a diagnostic/generator workflow, not a replacement for formal BUT original test.",
        "",
        "## Node Registry",
        "",
        md_table(reg[["node_id", "level", "role", "status", "existing_acc", "existing_medium_recall", "existing_bad_recall", "reason"]] if not reg.empty else reg, max_rows=8),
        "",
        "## Best Node 64D Fits",
        "",
        md_table(agg[["node_id", "rank", "variant_id", "node_score", "node_region_score", "good_64d_KS", "medium_64d_KS", "bad_64d_KS", "good_gm_overlap_coverage", "medium_gm_overlap_coverage", "bad_near_boundary_coverage"]] if not agg.empty else agg, max_rows=18),
        "",
        "## Node Diagnostics",
        "",
        md_table(diag[["node_id", "variant_id", "cls_loss_variant", "prediction_mode", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall", "original_but_macro_f1"]] if not diag.empty else diag, max_rows=18) if not diag.empty else "_No node training diagnostics yet._",
        "",
        "## Promotion Decisions",
        "",
        md_table(promo, max_rows=8) if not promo.empty else "_No promotion decisions yet._",
        "",
        "## Files",
        "",
        f"- `{NODE_REGISTRY_NAME}` / `{NODE_REGISTRY_CSV}`: immutable node definitions plus current status.",
        "- `nodes/<node_id>/node_boundary_manifest.csv`: selected BUT windows for that node.",
        "- `nodes/<node_id>/node_target_distributions.json`: 64D target distribution and region mix.",
        "- `nodes/<node_id>/node64_distance_leaderboard.csv`: no-training synthetic fit ranking.",
        "- `nodes/<node_id>/figures/best_rule_node_overlay.png`: node target vs best PTB synthetic.",
        f"- `{SUMMARY_NAME}` and `{DIAGNOSTIC_NAME}`: training and node-filtered diagnostics.",
        "",
        "## Current Recommendation",
        "",
        "Treat `N3600_anchor` as frozen. Work on `N4200_bridge` until it reaches the promotion gates; only then move to `N4800_wide`. If `N4200` misses, inspect medium good/medium-overlap coverage and bad near-boundary coverage before changing network size.",
    ]
    (report_root / "node_ladder_tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    update_state(Path(args.out_root) / STATE_NAME, status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"build_nodes", "all"}:
        build_nodes(args)
    if args.stage in {"fit_generator", "all"}:
        fit_generator(args)
    if args.stage in {"quick", "all"} and bool(args.run_training):
        train_quick(args)
    if args.stage == "full":
        train_full(args)
    if args.stage in {"diagnostic", "all"}:
        evaluate_node_diagnostics(args)
    if args.stage in {"promote", "all"}:
        promote_nodes(args)
    if args.stage in {"report", "all"}:
        write_report(args)
    update_state(Path(args.out_root) / STATE_NAME, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    base = semiclean.build_arg_parser()
    base.description = "Run Original-aware SemiCleanBUT node-ladder generator fitting and training."
    base.set_defaults(
        stage="all",
        out_root=str(DEFAULT_OUT_ROOT),
        report_root=str(DEFAULT_REPORT_ROOT),
        clean_root=str(DEFAULT_CLEAN_ROOT),
        semiclean_root=str(DEFAULT_SEMICLEAN_ROOT),
        source_boundary_root=str(DEFAULT_SOURCE_BOUNDARY_ROOT),
        seed=20260608,
        nodes="N4200_bridge",
        max_specs=120,
        max_refine_specs=72,
        sample_per_class_scan=256,
        sample_per_class_refine=800,
        top_quick=8,
        top_full=4,
        visual_top_n=9,
        clean_plot_rows_per_class=1600,
        core_select_pool_multiplier=3,
        run_training=False,
        auto_refine=True,
    )
    for action in base._actions:
        if action.dest == "stage":
            action.choices = ("build_nodes", "fit_generator", "quick", "full", "diagnostic", "promote", "report", "all")
            action.default = "all"
            break
    # The imported parser already has many generator/training options; add only
    # node-ladder specific switches.
    base.add_argument("--nodes", default="N4200_bridge", help="Comma-separated node ids or 'all'.")
    base.add_argument("--source_boundary_root", default=str(DEFAULT_SOURCE_BOUNDARY_ROOT))
    base.add_argument("--force_nodes", action="store_true")
    base.add_argument("--include_anchor_fit", action="store_true")
    base.add_argument("--train_all_nodes", action="store_true")
    base.add_argument("--loss_variants", default="hard_ce,sample_weighted_ce,soft_sample_weighted_ce")
    base.add_argument("--refine_region_threshold", type=float, default=0.42)
    base.add_argument("--min_region_eval_n", type=int, default=80)
    return base


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
