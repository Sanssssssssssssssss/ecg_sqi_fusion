"""SemiCleanBUT overlap-target fitting and diagnostic training.

This runner defines a prediction-free SemiCleanBUT target: good/medium keep a
controlled amount of overlap, while bad is restricted to the right-side
separated island in the CleanBUT atlas.  The target is a generator target and a
filtered diagnostic only; it is not the formal BUT benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import ensure_but_10s, now_iso, read_json, update_state
from src.transformer_pipeline.external_benchmarks.run import apply_but_thresholds, multiclass_report, plot_wave_gallery


ROOT = targeted.ROOT
RUN_TAG = "e311_but_semiclean_overlap_target_grid_10s_2026_06_07"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_CLEAN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_clean_label_core_10s_2026_06_06"

STATE_NAME = "semiclean_overlap_state.json"
MANIFEST_NAME = "semiclean_overlap_manifest.csv"
TARGET_DISTRIBUTIONS_NAME = "semiclean_overlap_target_distributions.json"
LEADERBOARD_NAME = "semiclean64_distance_leaderboard.csv"
LEADERBOARD_JSON = "semiclean64_distance_leaderboard.json"
PROJECTION_NAME = "semiclean64_pca_projection.csv"
SUMMARY_NAME = "semiclean_overlap_training_summary.jsonl"

CLASS_NAMES = targeted.CLASS_NAMES
COLORS = targeted.COLORS


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


def _num(df: pd.DataFrame, col: str, default: float = math.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_semiclean_manifest(args: argparse.Namespace) -> pd.DataFrame:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    source = Path(args.clean_root) / "clean_subset_manifest.csv"
    if not source.exists():
        raise FileNotFoundError(source)

    manifest = relax.add_confidence_score(pd.read_csv(source))
    for col in ["knn_label_purity", "pca_margin", "class_margin_percentile", "pc1", "pc2"]:
        manifest[col] = _num(manifest, col)
    ambiguous = manifest.get("ambiguous_type", pd.Series("", index=manifest.index)).astype(str)
    cls = manifest["class_name"].astype(str)

    good_eligible = (
        cls.eq("good")
        & (_num(manifest, "knn_label_purity") >= float(args.good_medium_min_purity))
        & (_num(manifest, "pca_margin") >= float(args.good_medium_min_margin))
        & (_num(manifest, "class_margin_percentile") >= float(args.good_medium_min_margin_percentile))
        & ~ambiguous.eq("isolated_good")
    )
    medium_eligible = (
        cls.eq("medium")
        & (_num(manifest, "knn_label_purity") >= float(args.good_medium_min_purity))
        & (_num(manifest, "pca_margin") >= float(args.good_medium_min_margin))
        & (_num(manifest, "class_margin_percentile") >= float(args.good_medium_min_margin_percentile))
        & ~ambiguous.isin(["isolated_medium", "medium_bad_boundary"])
    )
    bad_eligible = (
        cls.eq("bad")
        & (_num(manifest, "pc1") >= float(args.bad_min_pc1))
        & (_num(manifest, "knn_label_purity") >= float(args.bad_min_purity))
        & (_num(manifest, "pca_margin") >= float(args.bad_min_margin))
    )
    manifest["semiclean_eligible"] = good_eligible | medium_eligible | bad_eligible
    manifest["selection_reason"] = "excluded_low_confidence_or_wrong_island"
    manifest.loc[good_eligible, "selection_reason"] = "good_overlap_core_plus_boundary"
    manifest.loc[medium_eligible, "selection_reason"] = "medium_overlap_core_plus_boundary"
    manifest.loc[bad_eligible, "selection_reason"] = "bad_right_island_pc1_ge9_margin_ge8"
    manifest["semiclean_selected"] = False
    manifest["selected_level"] = ""
    manifest["semiclean_rank_in_class"] = np.nan

    target_per_class = int(args.target_per_class)
    selection_summary: dict[str, Any] = {}
    for y, name in enumerate(CLASS_NAMES):
        sub = manifest.loc[manifest["semiclean_eligible"] & manifest["class_name"].eq(name)].copy()
        sub = sub.sort_values(["boundary_confidence", "pca_margin"], ascending=[False, False])
        keep = sub.head(min(target_per_class, len(sub))).copy()
        manifest.loc[keep.index, "semiclean_selected"] = True
        manifest.loc[keep.index, "selected_level"] = f"semiclean_overlap_{len(keep)}"
        manifest.loc[keep.index, "semiclean_rank_in_class"] = np.arange(1, len(keep) + 1, dtype=int)
        selection_summary[name] = {
            "eligible": int(len(sub)),
            "selected": int(len(keep)),
            "min_confidence": float(keep["boundary_confidence"].min()) if len(keep) else math.nan,
            "median_confidence": float(keep["boundary_confidence"].median()) if len(keep) else math.nan,
            "ambiguous_fraction": float(keep["clean_tier"].astype(str).eq("ambiguous_boundary").mean()) if len(keep) else math.nan,
        }

    manifest["confidence"] = manifest["boundary_confidence"]
    preferred = [
        "idx",
        "class_name",
        "y",
        "split",
        "pc1",
        "pc2",
        "confidence",
        "boundary_confidence",
        "knn_label_purity",
        "pca_margin",
        "class_margin_percentile",
        "clean_tier",
        "ambiguous_type",
        "semiclean_eligible",
        "semiclean_selected",
        "selected_level",
        "semiclean_rank_in_class",
        "selection_reason",
    ]
    cols = [c for c in preferred if c in manifest.columns] + [c for c in manifest.columns if c not in preferred]
    manifest = manifest[cols].copy()
    manifest.to_csv(out_root / MANIFEST_NAME, index=False)
    manifest.to_csv(report_root / MANIFEST_NAME, index=False)

    write_json(
        out_root / "semiclean_overlap_rule_thresholds.json",
        {
            "source_manifest": str(source),
            "target_per_class": target_per_class,
            "selection_summary": selection_summary,
            "rules": {
                "bad": f"class=bad AND pc1>={float(args.bad_min_pc1)} AND knn_label_purity>={float(args.bad_min_purity)} AND pca_margin>={float(args.bad_min_margin)}",
                "good": f"class=good AND purity>={float(args.good_medium_min_purity)} AND pca_margin>={float(args.good_medium_min_margin)} AND class_margin_percentile>={float(args.good_medium_min_margin_percentile)} AND ambiguous_type!=isolated_good",
                "medium": f"class=medium AND purity>={float(args.good_medium_min_purity)} AND pca_margin>={float(args.good_medium_min_margin)} AND class_margin_percentile>={float(args.good_medium_min_margin_percentile)} AND ambiguous_type not in isolated_medium/medium_bad_boundary",
            },
            "note": "Prediction-free SemiCleanBUT diagnostic/generator target. It does not replace formal BUT original test.",
        },
    )
    return manifest


def load_semiclean_target(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out_root = Path(args.out_root)
    manifest_path = out_root / MANIFEST_NAME
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else build_semiclean_manifest(args)
    thresholds = read_json(Path(args.clean_root) / "clean_rule_thresholds.json")
    target = manifest.loc[manifest["semiclean_selected"].astype(bool)].copy()
    features = [f for f in list(thresholds.get("features", [])) if f in target.columns]
    if len(features) < 16:
        raise RuntimeError(f"SemiCleanBUT target has too few 64D feature columns: {len(features)}")
    return target, features, manifest


def write_target_distributions(args: argparse.Namespace, target: pd.DataFrame, features: list[str], manifest: pd.DataFrame) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    dists: dict[str, Any] = {
        "target_name": "SemiCleanBUT-Overlap",
        "n_selected": int(len(target)),
        "selected_counts": {c: int(target["class_name"].eq(c).sum()) for c in CLASS_NAMES},
        "feature_count": int(len(features)),
        "note": "Good/medium intentionally preserve moderate overlap; bad is restricted to the separated right island.",
        "by_class": {},
    }
    quantile_rows: list[dict[str, Any]] = []
    for name in CLASS_NAMES:
        sub = target.loc[target["class_name"].eq(name)]
        dists["by_class"][name] = {
            "n": int(len(sub)),
            "ambiguous_fraction": float(sub["clean_tier"].astype(str).eq("ambiguous_boundary").mean()) if len(sub) else math.nan,
            "pc1_median": float(pd.to_numeric(sub["pc1"], errors="coerce").median()) if len(sub) else math.nan,
            "pc2_median": float(pd.to_numeric(sub["pc2"], errors="coerce").median()) if len(sub) else math.nan,
        }
        for feat in features:
            vals = pd.to_numeric(sub[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            quantile_rows.append(
                {
                    "class_name": name,
                    "feature": feat,
                    "q05": float(vals.quantile(0.05)),
                    "q25": float(vals.quantile(0.25)),
                    "median": float(vals.quantile(0.50)),
                    "q75": float(vals.quantile(0.75)),
                    "q95": float(vals.quantile(0.95)),
                }
            )
    write_json(out_root / TARGET_DISTRIBUTIONS_NAME, dists)
    write_json(report_root / TARGET_DISTRIBUTIONS_NAME, dists)
    pd.DataFrame(quantile_rows).to_csv(out_root / "semiclean_overlap_target_quantiles.csv", index=False)
    pd.DataFrame(quantile_rows).to_csv(report_root / "semiclean_overlap_target_quantiles.csv", index=False)
    manifest.loc[manifest["semiclean_eligible"].astype(bool)].groupby(["class_name", "selection_reason", "clean_tier"], dropna=False).size().reset_index(name="n").to_csv(
        out_root / "semiclean_overlap_selection_composition.csv", index=False
    )


def _mutate_for_semiclean(specs: list[dict[str, Any]], *, max_specs: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    good_modes = [
        {"good_target_std": 0.32, "good_residual_mix": 0.006, "good_baseline_mix": 0.045, "good_qrs_boost": 0.48},
        {"good_target_std": 0.36, "good_residual_mix": 0.010, "good_baseline_mix": 0.055, "good_qrs_boost": 0.44},
        {"good_target_std": 0.30, "good_residual_mix": 0.004, "good_baseline_mix": 0.038, "good_qrs_boost": 0.54},
    ]
    medium_modes = [
        {"medium_event_prob": 0.70, "medium_inverse_peak_rate": 0.15, "medium_step_rate": 0.16, "medium_ramp_rate": 0.16},
        {"medium_event_prob": 0.78, "medium_inverse_peak_rate": 0.18, "medium_step_rate": 0.20, "medium_ramp_rate": 0.22},
        {"medium_event_prob": 0.86, "medium_inverse_peak_rate": 0.22, "medium_step_rate": 0.24, "medium_ramp_rate": 0.26},
    ]
    bad_modes = [
        {"bad_alt_cluster_prob": 0.20, "bad_osc_amp": 0.58, "bad_neg_spike_rate": 0.06},
        {"bad_alt_cluster_prob": 0.35, "bad_osc_amp": 0.62, "bad_neg_spike_rate": 0.10},
        {"bad_alt_cluster_prob": 0.15, "bad_osc_amp": 0.54, "bad_neg_spike_rate": 0.04},
    ]
    for i, spec in enumerate(specs):
        s = json.loads(json.dumps(spec))
        s.update(good_modes[i % len(good_modes)])
        s.update(medium_modes[(i // len(good_modes)) % len(medium_modes)])
        s.update(bad_modes[(i // (len(good_modes) * len(medium_modes))) % len(bad_modes)])
        s["selection_basis"] = "SemiCleanBUT-Overlap 64D target: good/medium overlap coverage plus separated right-island bad."
        s["note"] = "SemiCleanBUT overlap target; good/medium are intentionally less pristine and keep boundary overlap, bad targets the separated right island."
        sid = hashlib.sha1((json.dumps(s, sort_keys=True) + str(seed)).encode("utf-8")).hexdigest()[:12]
        family = str(s.get("family", "spec")).replace("bad_", "")
        safe_family = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in family)[:32]
        s["id"] = f"sc_overlap_{safe_family}_{i:03d}_{sid}"
        out.append(s)
        if len(out) >= int(max_specs):
            break
    return out


def make_semiclean_specs(args: argparse.Namespace, stage_name: str, prior_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if stage_name == "refine" and prior_rows:
        parents = [r["spec"] for r in sorted(prior_rows, key=lambda r: float(r.get("semiclean_score", r.get("score", 999.0))))[: int(args.refine_parent_count)]]
        specs = targeted.make_specs(min(int(args.max_specs), int(args.max_refine_specs)), refined=True, parents=parents)
    else:
        specs = targeted.make_specs(int(args.max_specs))
    return _mutate_for_semiclean(specs, max_specs=int(args.max_specs if stage_name == "scan" else args.max_refine_specs), seed=int(args.seed))


def _semiclean_score(row: dict[str, Any]) -> float:
    good = float(row.get("good_64d_KS", 1.0))
    medium = float(row.get("medium_64d_KS", 1.0))
    bad = float(row.get("bad_64d_KS", 1.0))
    worst = float(row.get("class_worst_64d_KS", max(good, medium, bad)))
    score = (
        0.16 * good
        + 0.19 * medium
        + 0.18 * bad
        + 0.10 * worst
        + 0.08 * float(row.get("good_PCA_range_gap", 1.0))
        + 0.08 * float(row.get("medium_PCA_range_gap", 1.0))
        + 0.07 * float(row.get("medium_PCA_covariance_gap", 1.0))
        + 0.08 * float(row.get("bad_PCA_centroid_distance", 1.0))
        + 0.04 * float(row.get("bad_PCA_covariance_gap", 1.0))
        + 0.02 * float(row.get("domain_separability", 1.0))
    )
    return float(score)


def leaderboard_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    for r in rows:
        r["semiclean_score"] = _semiclean_score(r)
        r["score"] = r["semiclean_score"]
        r["target_name"] = "SemiCleanBUT-Overlap"
    df = targeted.leaderboard_frame(rows)
    if "semiclean_score" not in df.columns:
        score_map = {str(r.get("variant_id")): float(r.get("semiclean_score", _semiclean_score(r))) for r in rows}
        df.insert(2, "semiclean_score", df["variant_id"].astype(str).map(score_map))
    return df.sort_values(["semiclean_score", "class_worst_64d_KS", "medium_64d_KS", "bad_64d_KS"], na_position="last").reset_index(drop=True)


def score_specs(args: argparse.Namespace, stage_name: str = "scan") -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    target, features, manifest = load_semiclean_target(args)
    write_target_distributions(args, target, features, manifest)
    pca_bundle = targeted.fit_clean_pca(target, features, int(args.seed))
    prior_rows = read_json(out_root / LEADERBOARD_JSON).get("rows", []) if (out_root / LEADERBOARD_JSON).exists() else []
    specs = make_semiclean_specs(args, stage_name, prior_rows)
    sample_per_class = int(args.sample_per_class_refine if stage_name == "refine" else args.sample_per_class_scan)
    write_json(out_root / "semiclean64_target_grid_specs.json", {"stage": stage_name, "rows": specs})
    write_json(report_root / "semiclean64_target_grid_specs.json", {"stage": stage_name, "rows": specs})

    existing = prior_rows if stage_name == "refine" else []
    rows: list[dict[str, Any]] = []
    update_state(out_root / STATE_NAME, status=f"{stage_name}_running", stage=stage_name, completed=0, total=len(specs), updated_at=now_iso())
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status=f"{stage_name}_running", stage=stage_name, current=spec["id"], completed=i - 1, total=len(specs), updated_at=now_iso())
        try:
            row = targeted.score_one(args, spec, target, features, pca_bundle, sample_per_class)
            row["target_name"] = "SemiCleanBUT-Overlap"
            row["semiclean_score"] = _semiclean_score(row)
            row["score"] = row["semiclean_score"]
            rows.append(row)
        except Exception as exc:
            rows.append({"variant_id": str(spec.get("id")), "family": spec.get("family"), "spec": spec, "status": "failed", "error": str(exc), "score": 999.0, "semiclean_score": 999.0})
            append_jsonl(out_root / "semiclean64_scan_failures.jsonl", rows[-1])
        if i == 1 or i % 6 == 0:
            live = leaderboard_frame(existing + rows)
            live.to_csv(out_root / LEADERBOARD_NAME, index=False)
            live.to_csv(report_root / LEADERBOARD_NAME, index=False)

    combined = sorted(existing + rows, key=lambda r: float(r.get("semiclean_score", r.get("score", 999.0))))
    write_json(out_root / LEADERBOARD_JSON, {"stage": stage_name, "rows": combined, "target": "SemiCleanBUT-Overlap"})
    # Compatibility for targeted.select_training_rows.
    write_json(out_root / "clean64_distance_leaderboard.json", {"stage": stage_name, "rows": combined, "target": "SemiCleanBUT-Overlap"})
    lb = leaderboard_frame(combined)
    lb.to_csv(out_root / LEADERBOARD_NAME, index=False)
    lb.to_csv(report_root / LEADERBOARD_NAME, index=False)
    lb.to_csv(out_root / targeted.LEADERBOARD_NAME, index=False)

    build_visuals(args, target, features, pca_bundle, combined)
    write_report(args)
    best = combined[0] if combined else {}
    update_state(
        out_root / STATE_NAME,
        status=f"{stage_name}_complete",
        stage=stage_name,
        completed=len(rows),
        total=len(specs),
        best_variant=best.get("variant_id"),
        best_semiclean_score=best.get("semiclean_score", best.get("score")),
        best_good_64d_KS=best.get("good_64d_KS"),
        best_medium_64d_KS=best.get("medium_64d_KS"),
        best_bad_64d_KS=best.get("bad_64d_KS"),
        updated_at=now_iso(),
    )
    return combined


def build_visuals(args: argparse.Namespace, target: pd.DataFrame, features: list[str], pca_bundle: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(out_root / MANIFEST_NAME)
    plot_semiclean_target_pca(manifest, fig_dir / "semiclean_target_pca.png", int(args.max_plot_rows))
    make_waveform_galleries(args, manifest)

    top_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda r: float(r.get("semiclean_score", r.get("score", 999.0)))):
        vid = str(row.get("variant_id", ""))
        if (
            not vid
            or vid in seen
            or "variant_dir" not in row
            or not Path(str(row.get("variant_dir", ""))).exists()
            or float(row.get("semiclean_score", row.get("score", 999.0))) >= 999.0
        ):
            continue
        top_rows.append(row)
        seen.add(vid)
        if len(top_rows) >= int(args.visual_top_n):
            break
    if not top_rows:
        return

    target_sample = []
    for i, cls in enumerate(CLASS_NAMES):
        sub = pca_bundle["clean_projected"].loc[pca_bundle["clean_projected"]["class_name"].astype(str).eq(cls)]
        if len(sub) > int(args.clean_plot_rows_per_class):
            sub = sub.sample(int(args.clean_plot_rows_per_class), random_state=int(args.seed) + i)
        target_sample.append(sub)
    target_plot = pd.concat(target_sample, ignore_index=True)
    target_out = target_plot[["class_name", "pc1_common", "pc2_common"]].copy()
    target_out["source"] = "SemiCleanBUT-Overlap"
    target_out["variant_id"] = "SemiCleanBUT-Overlap"
    projected_rows = [target_out]
    for row in top_rows:
        vdir = Path(str(row["variant_dir"]))
        proj_path = vdir / targeted.PROJECTION_NAME
        if proj_path.exists():
            proj = pd.read_csv(proj_path)
        else:
            full = pd.read_csv(vdir / "synthetic_atlas64_features.csv")
            xy = targeted.project_features(full, features, pca_bundle)
            proj = pd.DataFrame({"variant_id": row["variant_id"], "source": "PTB synthetic", "class_name": full["class_name"].astype(str), "pc1_common": xy[:, 0], "pc2_common": xy[:, 1]})
        for key in ["semiclean_score", "good_64d_KS", "medium_64d_KS", "bad_64d_KS"]:
            proj[key] = row.get(key, math.nan)
        projected_rows.append(proj)
    projected = pd.concat(projected_rows, ignore_index=True)
    projected.to_csv(out_root / PROJECTION_NAME, index=False)
    projected.to_csv(report_root / PROJECTION_NAME, index=False)
    plot_overlay(projected, top_rows, fig_dir / "top_rules_semiclean_overlay.png")
    plot_best_overlay(projected, top_rows[0], fig_dir / "best_semiclean_overlay.png")
    plot_distance_bars(leaderboard_frame(rows), fig_dir / "semiclean_classwise_distance_bars.png")


def plot_semiclean_target_pca(manifest: pd.DataFrame, out_path: Path, max_rows: int) -> None:
    plot_df = manifest.copy()
    selected = plot_df.loc[plot_df["semiclean_selected"].astype(bool)].copy()
    bg = plot_df.sample(min(len(plot_df), int(max_rows)), random_state=20260607)
    shell = selected.sort_values("boundary_confidence").groupby("class_name", group_keys=False).head(180)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(bg["pc1"], bg["pc2"], s=3, c="#d1d5db", alpha=0.14, linewidths=0)
        ax.grid(alpha=0.18)
        ax.set_xlabel("atlas PC1")
        ax.set_ylabel("atlas PC2")
    for cls in CLASS_NAMES:
        sub = selected.loc[selected["class_name"].eq(cls)]
        axes[0].scatter(sub["pc1"], sub["pc2"], s=7, c=COLORS[cls], alpha=0.68, linewidths=0, label=cls)
    axes[0].set_title("SemiClean selected classes")
    axes[0].legend(fontsize=8)
    for tier, color in [("clean_core_strict", "#2563eb"), ("clean_core_train_target", "#22c55e"), ("ambiguous_boundary", "#ef4444")]:
        sub = selected.loc[selected["clean_tier"].eq(tier)]
        axes[1].scatter(sub["pc1"], sub["pc2"], s=7, c=color, alpha=0.62, linewidths=0, label=tier)
    axes[1].set_title("Selected clean / ambiguous tiers")
    axes[1].legend(fontsize=7)
    axes[2].scatter(selected["pc1"], selected["pc2"], s=4, c="#9ca3af", alpha=0.22, linewidths=0)
    for cls in CLASS_NAMES:
        sub = shell.loc[shell["class_name"].eq(cls)]
        axes[2].scatter(sub["pc1"], sub["pc2"], s=12, c=COLORS[cls], alpha=0.9, linewidths=0, label=f"{cls} shell")
    axes[2].set_title("Lowest-confidence included shell")
    axes[2].legend(fontsize=8)
    fig.suptitle("SemiCleanBUT-Overlap target: good/medium overlap, bad separated right island", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_overlay(projected: pd.DataFrame, top_rows: list[dict[str, Any]], out_path: Path) -> None:
    target = projected.loc[projected["source"].eq("SemiCleanBUT-Overlap")]
    variants = [str(r["variant_id"]) for r in top_rows]
    ncols = 3
    nrows = int(math.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.4 * nrows), squeeze=False)
    for ax, vid in zip(axes.ravel(), variants):
        for cls in CLASS_NAMES:
            cdf = target.loc[target["class_name"].eq(cls)]
            ax.scatter(cdf["pc1_common"], cdf["pc2_common"], s=6, c=COLORS[cls], alpha=0.13, linewidths=0)
        vf = projected.loc[projected["variant_id"].eq(vid)]
        for cls in CLASS_NAMES:
            sdf = vf.loc[vf["class_name"].eq(cls)]
            ax.scatter(sdf["pc1_common"], sdf["pc2_common"], s=24, c=COLORS[cls], alpha=0.76, marker="x", linewidths=1.0)
        row = next(r for r in top_rows if str(r["variant_id"]) == vid)
        ax.set_title(f"{vid}\nscore={float(row.get('semiclean_score', row.get('score', math.nan))):.3f} gm/b={float(row.get('medium_64d_KS', math.nan)):.3f}/{float(row.get('bad_64d_KS', math.nan)):.3f}", fontsize=8)
        ax.set_xlabel("SemiClean 64D-PC1")
        ax.set_ylabel("SemiClean 64D-PC2")
        ax.grid(alpha=0.15)
    for ax in axes.ravel()[len(variants) :]:
        ax.axis("off")
    fig.suptitle("SemiCleanBUT-Overlap 64D target scan: top PTB synthetic overlays", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_overlay(projected: pd.DataFrame, row: dict[str, Any], out_path: Path) -> None:
    target = projected.loc[projected["source"].eq("SemiCleanBUT-Overlap")]
    vf = projected.loc[projected["variant_id"].eq(row["variant_id"])]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for cls in CLASS_NAMES:
        sub = target.loc[target["class_name"].eq(cls)]
        ax.scatter(sub["pc1_common"], sub["pc2_common"], s=10, c=COLORS[cls], alpha=0.17, linewidths=0, label=f"SemiClean {cls}")
    for cls in CLASS_NAMES:
        sub = vf.loc[vf["class_name"].eq(cls)]
        ax.scatter(sub["pc1_common"], sub["pc2_common"], s=38, c=COLORS[cls], marker="x", alpha=0.82, linewidths=1.1, label=f"Synth {cls}")
    ax.set_title(f"Best SemiCleanBUT overlap fit\n{row['variant_id']} | score={float(row.get('semiclean_score', row.get('score', math.nan))):.3f}")
    ax.set_xlabel("SemiClean 64D-PC1")
    ax.set_ylabel("SemiClean 64D-PC2")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distance_bars(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    view = df.head(min(24, len(df))).copy()
    labels = [str(v)[:42] for v in view["variant_id"]]
    x = np.arange(len(view))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(12, len(view) * 0.62), 5.4))
    ax.bar(x - width, view["good_64d_KS"], width, label="good", color=COLORS["good"], alpha=0.75)
    ax.bar(x, view["medium_64d_KS"], width, label="medium", color=COLORS["medium"], alpha=0.75)
    ax.bar(x + width, view["bad_64d_KS"], width, label="bad", color=COLORS["bad"], alpha=0.75)
    ax.plot(x, view["semiclean_score"], color="#111827", marker="o", lw=1.0, label="SemiClean score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("64D distance; lower is closer")
    ax.set_title("SemiCleanBUT 64D distance leaderboard")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_waveform_galleries(args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    report_root = Path(args.report_root)
    gallery = report_root / "semiclean_boundary_waveform_galleries"
    gallery.mkdir(parents=True, exist_ok=True)
    X, meta = ensure_but_10s(args)
    selected = manifest.loc[manifest["semiclean_selected"].astype(bool)].copy()
    for cls in CLASS_NAMES:
        sub = selected.loc[selected["class_name"].eq(cls)].sort_values("boundary_confidence").head(12)
        idx = sub["idx"].astype(int).tolist()
        if idx:
            plot_wave_gallery(X, meta, idx, gallery / f"{cls}_lowest_confidence_included.png", f"SemiClean {cls} lowest-confidence included")


def _semiclean_n_grid(manifest: pd.DataFrame, target_per_class: int) -> list[int]:
    eligible = manifest.loc[manifest["semiclean_eligible"].astype(bool)]
    max_n = int(min((eligible["class_name"].astype(str).eq(c)).sum() for c in CLASS_NAMES))
    base = [600, 1200, 2400, 3600, 4200, int(target_per_class), max_n]
    return sorted({int(v) for v in base if 50 <= int(v) <= max_n})


def select_semiclean_n(manifest: pd.DataFrame, n_per_class: int) -> np.ndarray:
    selected: list[int] = []
    eligible = manifest.loc[manifest["semiclean_eligible"].astype(bool)].copy()
    for cls in CLASS_NAMES:
        sub = eligible.loc[eligible["class_name"].astype(str).eq(cls)].sort_values(["boundary_confidence", "pca_margin"], ascending=[False, False])
        selected.extend(int(v) for v in sub.head(int(n_per_class))["idx"].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def completed_training_rows(out_root: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(out_root / targeted.SUMMARY_NAME)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        vid = str(row.get("spec", {}).get("id", ""))
        ckpt = row.get("but_10s_eval", {}).get("checkpoint")
        if vid and int(row.get("returncode", 1)) == 0 and ckpt and Path(ckpt).exists():
            by_id[vid] = row
    for path in (out_root / "runs" / "quick").glob("*/clean_core_targeted_run_summary.json"):
        row = read_json(path)
        vid = str(row.get("spec", {}).get("id", ""))
        ckpt = row.get("but_10s_eval", {}).get("checkpoint")
        if vid and int(row.get("returncode", 1)) == 0 and ckpt and Path(ckpt).exists():
            by_id[vid] = row
    return list(by_id.values())


def evaluate_semiclean_filtered(args: argparse.Namespace) -> pd.DataFrame:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    manifest = pd.read_csv(out_root / MANIFEST_NAME) if (out_root / MANIFEST_NAME).exists() else build_semiclean_manifest(args)
    X, meta = ensure_but_10s(args)
    y = meta["y"].to_numpy(dtype=np.int64)
    rows = completed_training_rows(out_root)
    if not rows:
        return pd.DataFrame()
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(args.cpu) else "cpu")
    grid = _semiclean_n_grid(manifest, int(args.target_per_class))
    selected_by_n = {n: select_semiclean_n(manifest, n) for n in grid}
    metric_rows: list[dict[str, Any]] = []
    pred_dir = out_root / "semiclean_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, start=1):
        vid = str(row.get("spec", {}).get("id"))
        ckpt = Path(row["but_10s_eval"]["checkpoint"])
        update_state(out_root / STATE_NAME, status="semiclean_evaluating", stage="diagnostic", current=vid, completed=i - 1, total=len(rows), updated_at=now_iso())
        pred_cache = pred_dir / f"{vid}_probs.npz"
        if pred_cache.exists():
            probs = np.load(pred_cache)["probs"].astype(np.float32)
        else:
            probs = relax.predict_probs(ckpt, X, int(args.batch_size_eval), device)
            np.savez_compressed(pred_cache, probs=probs.astype(np.float32))
        raw_pred = np.argmax(probs, axis=1).astype(np.int64)
        cal = row.get("but_10s_eval", {}).get("calibration", {})
        cal_pred = apply_but_thresholds(probs, float(cal.get("t_good", 0.5)), float(cal.get("t_bad", 0.5)))
        original = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        for n, idx in selected_by_n.items():
            subset = manifest.loc[manifest["idx"].isin(idx)]
            comp = {
                "ambiguous_fraction": float(subset["clean_tier"].astype(str).eq("ambiguous_boundary").mean()) if len(subset) else math.nan,
                "good_medium_overlap_fraction": float(subset["selection_reason"].astype(str).isin(["good_overlap_core_plus_boundary", "medium_overlap_core_plus_boundary"]).mean()) if len(subset) else math.nan,
                "bad_pc1_lt9_fraction": float((pd.to_numeric(subset.loc[subset["class_name"].eq("bad"), "pc1"], errors="coerce") < float(args.bad_min_pc1)).mean()) if len(subset.loc[subset["class_name"].eq("bad")]) else math.nan,
            }
            for mode, pred in (("raw", raw_pred), ("calibrated", cal_pred)):
                rep = multiclass_report(y[idx], pred[idx], probs[idx])
                rec = rep.get("recall_good_medium_bad", [math.nan, math.nan, math.nan])
                metric_rows.append(
                    {
                        "variant_id": vid,
                        "prediction_mode": mode,
                        "n_per_class": int(n),
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
                        **comp,
                        **row.get("clean64_distance", {}),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_root / "semiclean_filtered_metrics.csv", index=False)
    metrics.to_csv(report_root / "semiclean_filtered_metrics.csv", index=False)
    best_rows = []
    for (vid, mode), sub in metrics.groupby(["variant_id", "prediction_mode"]):
        target = float(args.target_acc)
        in_band = sub.loc[(sub["acc"] >= target - float(args.target_acc_band)) & (sub["acc"] <= target + float(args.target_acc_band))].sort_values(["n_per_class", "acc"], ascending=[False, False])
        if in_band.empty:
            above = sub.loc[sub["acc"] >= target].sort_values(["n_per_class", "acc"], ascending=[False, True])
            chosen = above.iloc[0] if not above.empty else sub.iloc[(sub["acc"] - target).abs().argsort()[:1]].iloc[0]
            status = "above_target_or_closest"
        else:
            chosen = in_band.iloc[0]
            status = "in_0p95_band_max_coverage"
        d = chosen.to_dict()
        d["status"] = status
        best_rows.append(d)
    best = pd.DataFrame(best_rows).sort_values(["status", "n_per_class", "acc"], ascending=[True, False, False])
    best.to_csv(out_root / "semiclean_best_at_0p95.csv", index=False)
    best.to_csv(report_root / "semiclean_best_at_0p95.csv", index=False)
    plot_semiclean_relaxation(args, metrics, best)
    write_report(args)
    update_state(out_root / STATE_NAME, status="semiclean_diagnostic_complete", stage="diagnostic", completed=len(rows), total=len(rows), updated_at=now_iso())
    return metrics


def plot_semiclean_relaxation(args: argparse.Namespace, metrics: pd.DataFrame, best: pd.DataFrame) -> None:
    if metrics.empty:
        return
    fig_dir = Path(args.report_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = metrics.loc[metrics["prediction_mode"].eq("raw")]
    top = best.loc[best["prediction_mode"].eq("raw")].sort_values(["n_per_class", "acc"], ascending=[False, False]).head(8)["variant_id"].tolist()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for vid in top:
        sub = raw.loc[raw["variant_id"].eq(vid)].sort_values("n_per_class")
        ax.plot(sub["n_per_class"], sub["acc"], marker="o", ms=3, lw=1.5, label=str(vid)[:52])
    ax.axhline(float(args.target_acc), color="#111827", ls="--", lw=1.0, label=f"target={float(args.target_acc):.2f}")
    ax.fill_between([0, raw["n_per_class"].max()], float(args.target_acc) - float(args.target_acc_band), float(args.target_acc) + float(args.target_acc_band), color="#9ca3af", alpha=0.14, label="target band")
    ax.set_xlabel("selected windows per class")
    ax.set_ylabel("SemiClean filtered diagnostic accuracy")
    ax.set_ylim(0.25, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("SemiCleanBUT boundary relaxation curve")
    fig.tight_layout()
    fig.savefig(fig_dir / "semiclean_relaxation_curve.png", dpi=180)
    plt.close(fig)


def run_quick(args: argparse.Namespace) -> None:
    if not bool(args.run_training):
        raise RuntimeError("Quick training requires --run_training.")
    rows = targeted.run_quick(args)
    for row in rows:
        append_jsonl(Path(args.out_root) / SUMMARY_NAME, row)
    evaluate_semiclean_filtered(args)


def write_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    lb = pd.read_csv(out_root / LEADERBOARD_NAME) if (out_root / LEADERBOARD_NAME).exists() else pd.DataFrame()
    metrics = pd.read_csv(out_root / "semiclean_filtered_metrics.csv") if (out_root / "semiclean_filtered_metrics.csv").exists() else pd.DataFrame()
    best = pd.read_csv(out_root / "semiclean_best_at_0p95.csv") if (out_root / "semiclean_best_at_0p95.csv").exists() else pd.DataFrame()
    manifest = pd.read_csv(out_root / MANIFEST_NAME) if (out_root / MANIFEST_NAME).exists() else pd.DataFrame()
    sel = manifest.loc[manifest["semiclean_selected"].astype(bool)] if not manifest.empty else pd.DataFrame()
    counts = sel["class_name"].value_counts().to_dict() if not sel.empty else {}
    amb_frac = float(sel["clean_tier"].astype(str).eq("ambiguous_boundary").mean()) if not sel.empty else math.nan
    bad_left = float((pd.to_numeric(sel.loc[sel["class_name"].eq("bad"), "pc1"], errors="coerce") < float(args.bad_min_pc1)).mean()) if not sel.empty and len(sel.loc[sel["class_name"].eq("bad")]) else math.nan
    best_cpu = lb.head(1).to_dict("records")[0] if not lb.empty else {}
    lines = [
        "# SemiCleanBUT-Overlap Target Fit",
        "",
        "SemiCleanBUT is a filtered diagnostic/generator target. It keeps good/medium overlap while forcing bad onto the separated right island; formal original BUT remains only a reference metric.",
        "",
        "## Target Definition",
        "",
        f"- Selected counts: `{counts}`",
        f"- Ambiguous fraction: `{amb_frac:.3f}`",
        f"- Bad selected with pc1 < {float(args.bad_min_pc1):.1f}: `{bad_left:.4f}`",
        "- Bad rule: `bad & pc1 >= 9.0 & knn_label_purity >= 0.95 & pca_margin >= 8.0` by default.",
        "- Good/medium rule: relaxed label-purity/margin with isolated outliers removed, then top confidence per class.",
        "",
        "## Best CPU 64D Fit",
        "",
        f"- Best variant: `{best_cpu.get('variant_id', 'n/a')}`",
        f"- SemiClean score: `{float(best_cpu.get('semiclean_score', best_cpu.get('score', math.nan))):.4f}`",
        f"- 64D KS good/medium/bad: `{float(best_cpu.get('good_64d_KS', math.nan)):.3f}/{float(best_cpu.get('medium_64d_KS', math.nan)):.3f}/{float(best_cpu.get('bad_64d_KS', math.nan)):.3f}`",
        "",
        "## Top CPU Candidates",
        "",
        targeted.md_table(lb[["rank", "variant_id", "family", "semiclean_score", "good_64d_KS", "medium_64d_KS", "bad_64d_KS", "good_PCA_range_gap", "medium_PCA_range_gap"]] if not lb.empty else lb, max_rows=16),
        "",
        "## SemiClean Training Diagnostics",
        "",
        targeted.md_table(best[["status", "variant_id", "prediction_mode", "n_per_class", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall", "ambiguous_fraction", "original_but_macro_f1"]] if not best.empty else best, max_rows=16) if not best.empty else "_No SemiClean quick training metrics yet._",
        "",
        "## Figures",
        "",
        "- `figures/semiclean_target_pca.png`: selected SemiClean target and lowest-confidence shell.",
        "- `figures/top_rules_semiclean_overlay.png`: top synthetic rules against SemiClean target.",
        "- `figures/best_semiclean_overlay.png`: best current synthetic fit.",
        "- `figures/semiclean_relaxation_curve.png`: filtered diagnostic accuracy as boundary is relaxed.",
        "- `semiclean_boundary_waveform_galleries/`: lowest-confidence included examples.",
        "",
        "## Caveat",
        "",
        "This filtered diagnostic deliberately uses all splits for a class-balanced target because the formal CleanBUT clean-core test split is not class-complete for bad. It is not a replacement benchmark.",
    ]
    (report_root / "semiclean_overlap_target_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / STATE_NAME
    if state_path.exists():
        try:
            state = read_json(state_path)
            state.pop("error", None)
            write_json(state_path, state)
        except Exception:
            pass
    update_state(out_root / STATE_NAME, status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"target", "all"}:
        manifest = build_semiclean_manifest(args)
        target, features, _ = load_semiclean_target(args)
        write_target_distributions(args, target, features, manifest)
        build_visuals(args, target, features, targeted.fit_clean_pca(target, features, int(args.seed)), [])
        write_report(args)
    if args.stage in {"scan", "all"}:
        rows = score_specs(args, "scan")
        if args.stage == "all" and bool(args.auto_refine):
            best = min(rows, key=lambda r: float(r.get("semiclean_score", r.get("score", 999.0)))) if rows else {}
            if float(best.get("class_worst_64d_KS", 999.0)) > float(args.refine_worst_threshold):
                score_specs(args, "refine")
    if args.stage == "refine":
        score_specs(args, "refine")
    if args.stage == "quick":
        run_quick(args)
    if args.stage == "diagnostic":
        evaluate_semiclean_filtered(args)
    if args.stage == "report":
        write_report(args)
    complete_payload: dict[str, Any] = {"status": "complete", "stage": args.stage, "updated_at": now_iso()}
    if args.stage in {"quick", "diagnostic"}:
        completed = len(completed_training_rows(out_root))
        complete_payload.update({"completed": completed, "total": completed})
    update_state(out_root / STATE_NAME, **complete_payload)


def build_arg_parser() -> argparse.ArgumentParser:
    base = targeted.build_arg_parser()
    base.description = "Run SemiCleanBUT overlap target fitting and filtered diagnostics."
    base.set_defaults(
        stage="all",
        out_root=str(DEFAULT_OUT_ROOT),
        report_root=str(DEFAULT_REPORT_ROOT),
        clean_root=str(DEFAULT_CLEAN_ROOT),
        seed=20260607,
        max_specs=96,
        max_refine_specs=48,
        sample_per_class_scan=256,
        sample_per_class_refine=800,
        top_quick=8,
        top_full=4,
        visual_top_n=9,
        clean_plot_rows_per_class=1800,
        core_select_pool_multiplier=3,
        run_training=False,
    )
    # Replace stage choices from the imported parser.
    for action in base._actions:
        if action.dest == "stage":
            action.choices = ("target", "scan", "refine", "quick", "diagnostic", "report", "all")
            action.default = "all"
            break
    base.add_argument("--target_per_class", type=int, default=4800)
    base.add_argument("--bad_min_pc1", type=float, default=9.0)
    base.add_argument("--bad_min_purity", type=float, default=0.95)
    base.add_argument("--bad_min_margin", type=float, default=8.0)
    base.add_argument("--good_medium_min_purity", type=float, default=0.75)
    base.add_argument("--good_medium_min_margin", type=float, default=0.60)
    base.add_argument("--good_medium_min_margin_percentile", type=float, default=0.30)
    base.add_argument("--target_acc", type=float, default=0.95)
    base.add_argument("--target_acc_band", type=float, default=0.01)
    base.add_argument("--max_plot_rows", type=int, default=20000)
    base.add_argument("--auto_refine", action="store_true")
    base.add_argument("--refine_worst_threshold", type=float, default=0.55)
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
