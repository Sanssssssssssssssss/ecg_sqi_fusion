"""Original-aware SemiCleanBUT boundary widening diagnostics.

This experiment keeps SemiCleanBUT as a diagnostic/generator target, not the
formal BUT benchmark.  It adds prediction-free original-region labels to the
CleanBUT atlas and selects class-balanced boundary levels by region quotas
instead of a single confidence rank.
"""

from __future__ import annotations

import argparse
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
from src.transformer_pipeline.external_benchmarks import but_cleanbut_boundary_relaxation_10s as relax
from src.transformer_pipeline.external_benchmarks import but_semiclean_overlap_target_grid_10s as semiclean
from src.transformer_pipeline.external_benchmarks import but_clean_core_targeted_generator_grid_10s as targeted
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import ensure_but_10s, now_iso, read_json, update_state
from src.transformer_pipeline.external_benchmarks.run import apply_but_thresholds, multiclass_report, plot_wave_gallery


ROOT = targeted.ROOT
RUN_TAG = "e311_but_original_aware_semiclean_boundary_10s_2026_06_07"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_CLEAN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_clean_label_core_10s_2026_06_06"
DEFAULT_SEMICLEAN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_semiclean_overlap_target_grid_10s_2026_06_07"

STATE_NAME = "original_aware_semiclean_state.json"
REGION_ATLAS_NAME = "original_region_atlas.csv"
REGION_QUOTA_NAME = "region_quota_boundary_manifest.csv"
LEVEL_METRICS_NAME = "region_quota_boundary_metrics.csv"
REGION_METRICS_NAME = "original_region_model_metrics.csv"

CLASS_NAMES = ("good", "medium", "bad")
COLORS = {"good": "#1b9e77", "medium": "#fdae61", "bad": "#d73027"}
REGION_COLORS = {
    "clean_core": "#2563eb",
    "good_medium_overlap": "#f59e0b",
    "medium_bad_overlap": "#dc2626",
    "right_bad_island": "#7c2d12",
    "near_bad_boundary": "#fb7185",
    "outlier_low_confidence": "#6b7280",
}


def _num(df: pd.DataFrame, col: str, default: float = math.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _level_quota(level: int) -> dict[str, dict[str, float]]:
    """Region quota policy for each class at a boundary level."""

    # As levels widen, good/medium intentionally include more overlap shell.
    if level <= 3600:
        gm = {"clean_core": 0.66, "good_medium_overlap": 0.34, "outlier_low_confidence": 0.00}
        med = {"clean_core": 0.56, "good_medium_overlap": 0.39, "medium_bad_overlap": 0.05}
        bad = {"right_bad_island": 1.00, "near_bad_boundary": 0.00, "medium_bad_overlap": 0.00}
    elif level <= 4200:
        gm = {"clean_core": 0.56, "good_medium_overlap": 0.44, "outlier_low_confidence": 0.00}
        med = {"clean_core": 0.45, "good_medium_overlap": 0.48, "medium_bad_overlap": 0.07}
        bad = {"right_bad_island": 0.92, "near_bad_boundary": 0.08, "medium_bad_overlap": 0.00}
    elif level <= 4800:
        gm = {"clean_core": 0.45, "good_medium_overlap": 0.55, "outlier_low_confidence": 0.00}
        med = {"clean_core": 0.34, "good_medium_overlap": 0.56, "medium_bad_overlap": 0.10}
        bad = {"right_bad_island": 0.85, "near_bad_boundary": 0.12, "medium_bad_overlap": 0.03}
    elif level <= 5200:
        gm = {"clean_core": 0.38, "good_medium_overlap": 0.60, "outlier_low_confidence": 0.02}
        med = {"clean_core": 0.28, "good_medium_overlap": 0.58, "medium_bad_overlap": 0.14}
        bad = {"right_bad_island": 0.80, "near_bad_boundary": 0.15, "medium_bad_overlap": 0.05}
    elif level <= 5600:
        gm = {"clean_core": 0.32, "good_medium_overlap": 0.64, "outlier_low_confidence": 0.04}
        med = {
            "clean_core": 0.22,
            "good_medium_overlap": 0.57,
            "medium_bad_overlap": 0.17,
            "outlier_low_confidence": 0.04,
        }
        bad = {
            "right_bad_island": 0.70,
            "near_bad_boundary": 0.20,
            "medium_bad_overlap": 0.07,
            "outlier_low_confidence": 0.03,
        }
    elif level <= 6000:
        gm = {"clean_core": 0.26, "good_medium_overlap": 0.68, "outlier_low_confidence": 0.06}
        med = {
            "clean_core": 0.16,
            "good_medium_overlap": 0.57,
            "medium_bad_overlap": 0.20,
            "outlier_low_confidence": 0.07,
        }
        bad = {
            "right_bad_island": 0.62,
            "near_bad_boundary": 0.23,
            "medium_bad_overlap": 0.10,
            "outlier_low_confidence": 0.05,
        }
    else:
        # Beyond 6000/class the source bad class is saturated in the current
        # CleanBUT subset, so widening mainly probes the good/medium lower-
        # confidence shells while holding bad at every usable island/boundary row.
        gm = {"clean_core": 0.20, "good_medium_overlap": 0.70, "outlier_low_confidence": 0.10}
        med = {
            "clean_core": 0.12,
            "good_medium_overlap": 0.56,
            "medium_bad_overlap": 0.22,
            "outlier_low_confidence": 0.10,
        }
        bad = {
            "right_bad_island": 0.58,
            "near_bad_boundary": 0.25,
            "medium_bad_overlap": 0.10,
            "outlier_low_confidence": 0.07,
        }
    return {"good": gm, "medium": med, "bad": bad}


def _ordered_fill_regions(class_name: str) -> list[str]:
    if class_name == "bad":
        return ["right_bad_island", "near_bad_boundary", "medium_bad_overlap", "outlier_low_confidence"]
    if class_name == "medium":
        return ["clean_core", "good_medium_overlap", "medium_bad_overlap", "near_bad_boundary", "outlier_low_confidence"]
    return ["clean_core", "good_medium_overlap", "outlier_low_confidence"]


def _mark_regions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cls = out["class_name"].astype(str)
    amb = out.get("ambiguous_type", pd.Series("", index=out.index)).astype(str)
    tier = out.get("clean_tier", pd.Series("", index=out.index)).astype(str)
    nearest = out.get("nearest_other_class", pd.Series("", index=out.index)).astype(str)
    pc1 = _num(out, "pc1")
    purity = _num(out, "knn_label_purity", 0.0)
    margin = _num(out, "pca_margin", -999.0)
    centrality = _num(out, "class_centrality_percentile", 0.0)

    clean_core = tier.isin(["clean_core_strict", "clean_core_train_target"]) & (purity >= 0.85) & (margin >= 0.0)
    gm_overlap = (
        cls.isin(["good", "medium"])
        & (
            amb.eq("good_medium_boundary")
            | ((cls.eq("good") & nearest.eq("medium")) | (cls.eq("medium") & nearest.eq("good")))
            | (tier.eq("ambiguous_boundary") & (pc1 < 5.0))
        )
        & (purity >= 0.60)
        & (margin >= -0.75)
    )
    mb_overlap = (
        cls.isin(["medium", "bad"])
        & (
            amb.eq("medium_bad_boundary")
            | ((cls.eq("medium") & nearest.eq("bad")) | (cls.eq("bad") & nearest.eq("medium")))
            | (tier.eq("ambiguous_boundary") & (pc1 >= 4.0))
        )
        & (purity >= 0.55)
        & (margin >= -1.25)
    )
    right_bad = cls.eq("bad") & (pc1 >= 9.0) & (purity >= 0.95) & (margin >= 8.0)
    near_bad = cls.eq("bad") & ~right_bad & (pc1 >= 8.0)
    outlier = (
        (purity < 0.55)
        | (margin < -1.25)
        | amb.isin(["isolated_good", "isolated_medium", "bad_outlier"])
        | (centrality < 0.01)
    )

    region = np.full(len(out), "good_medium_overlap", dtype=object)
    # Clean tiers are the central anchors; overlap shells are added around them.
    region[clean_core.to_numpy()] = "clean_core"
    region[(gm_overlap & ~clean_core).to_numpy()] = "good_medium_overlap"
    region[(mb_overlap & ~clean_core).to_numpy()] = "medium_bad_overlap"
    region[(near_bad & ~clean_core).to_numpy()] = "near_bad_boundary"
    region[right_bad.to_numpy()] = "right_bad_island"
    region[outlier.to_numpy()] = "outlier_low_confidence"
    # Bad's separated island is the stable target even when the source row is
    # in an ambiguous clean tier.
    region[(right_bad & ~outlier).to_numpy()] = "right_bad_island"
    # Bad outliers that still sit next to the right-side bad island are treated
    # as a controlled near-boundary shell, not as arbitrary low-confidence noise.
    region[(near_bad & outlier).to_numpy()] = "near_bad_boundary"
    out["original_region"] = region
    out["region_confidence"] = out["boundary_confidence"].astype(float)
    out.loc[out["original_region"].eq("good_medium_overlap"), "region_confidence"] *= 0.88
    out.loc[out["original_region"].eq("medium_bad_overlap"), "region_confidence"] *= 0.80
    out.loc[out["original_region"].eq("near_bad_boundary"), "region_confidence"] *= 0.78
    out.loc[out["original_region"].eq("outlier_low_confidence"), "region_confidence"] *= 0.25
    return out


def build_region_atlas(args: argparse.Namespace) -> pd.DataFrame:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    source = Path(args.clean_root) / "clean_subset_manifest.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    atlas = relax.add_confidence_score(pd.read_csv(source))
    atlas = _mark_regions(atlas)
    atlas.to_csv(out_root / REGION_ATLAS_NAME, index=False)
    atlas.to_csv(report_root / REGION_ATLAS_NAME, index=False)

    rows: list[dict[str, Any]] = []
    for split in ["all", "train", "val", "test"]:
        sdf = atlas if split == "all" else atlas.loc[atlas["split"].astype(str).eq(split)]
        for cls in CLASS_NAMES:
            cdf = sdf.loc[sdf["class_name"].eq(cls)]
            for region, sub in cdf.groupby("original_region", dropna=False):
                rows.append(
                    {
                        "split": split,
                        "class_name": cls,
                        "original_region": region,
                        "n": int(len(sub)),
                        "fraction_within_class": float(len(sub) / max(1, len(cdf))),
                        "median_confidence": float(sub["boundary_confidence"].median()) if len(sub) else math.nan,
                        "median_pc1": float(pd.to_numeric(sub["pc1"], errors="coerce").median()) if len(sub) else math.nan,
                        "median_pc2": float(pd.to_numeric(sub["pc2"], errors="coerce").median()) if len(sub) else math.nan,
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "original_region_summary.csv", index=False)
    summary.to_csv(report_root / "original_region_summary.csv", index=False)
    write_json(
        out_root / "original_region_rules.json",
        {
            "source": str(source),
            "note": "Prediction-free original-region labels derived from CleanBUT 64D PCA/kNN/margin features. Model predictions are not used.",
            "regions": {
                "clean_core": "clean_tier in clean_core_* with purity>=0.85 and positive PCA margin unless another boundary rule applies",
                "good_medium_overlap": "good/medium windows near the good-medium PCA/kNN boundary with purity>=0.60",
                "medium_bad_overlap": "medium/bad windows near the medium-bad PCA/kNN boundary with purity>=0.55",
                "right_bad_island": "bad windows on the separated bad island: pc1>=9, purity>=0.95, margin>=8",
                "near_bad_boundary": "bad windows near the bad island but not tight core: pc1>=5, purity>=0.60",
                "outlier_low_confidence": "negative/low margin, low purity, isolated, or very low centrality windows",
            },
        },
    )
    return atlas


def _take_region(sub: pd.DataFrame, region: str, n: int, already: set[int]) -> list[int]:
    if n <= 0:
        return []
    cand = sub.loc[sub["original_region"].eq(region) & ~sub["idx"].astype(int).isin(already)].copy()
    cand = cand.sort_values(["region_confidence", "pca_margin"], ascending=[False, False])
    return [int(v) for v in cand.head(int(n))["idx"].tolist()]


def _select_class_level(atlas: pd.DataFrame, class_name: str, level: int) -> list[int]:
    sub = atlas.loc[atlas["class_name"].astype(str).eq(class_name)].copy()
    quotas = _level_quota(level)[class_name]
    selected: list[int] = []
    seen: set[int] = set()
    for region, frac in quotas.items():
        n = int(round(float(level) * float(frac)))
        take = _take_region(sub, region, n, seen)
        selected.extend(take)
        seen.update(take)
    if len(selected) < int(level):
        for region in _ordered_fill_regions(class_name):
            take = _take_region(sub, region, int(level) - len(selected), seen)
            selected.extend(take)
            seen.update(take)
            if len(selected) >= int(level):
                break
    # Last-resort fill by confidence, but keep bad away from the far-left bad outlier
    # until every structured region is exhausted.
    if len(selected) < int(level):
        cand = sub.loc[~sub["idx"].astype(int).isin(seen)].copy()
        if class_name == "bad":
            cand = cand.loc[pd.to_numeric(cand["pc1"], errors="coerce") >= 3.0]
        cand = cand.sort_values(["region_confidence", "pca_margin"], ascending=[False, False])
        take = [int(v) for v in cand.head(int(level) - len(selected))["idx"].tolist()]
        selected.extend(take)
    return selected[: int(level)]


def build_region_quota_manifest(args: argparse.Namespace) -> pd.DataFrame:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    atlas_path = out_root / REGION_ATLAS_NAME
    atlas = pd.read_csv(atlas_path) if atlas_path.exists() else build_region_atlas(args)
    levels = [int(v) for v in str(args.levels).split(",") if str(v).strip()]
    manifest = atlas.copy()
    for level in levels:
        manifest[f"selected_l{level}"] = False
        manifest[f"selected_l{level}_rank"] = np.nan
    composition_rows: list[dict[str, Any]] = []
    for level in levels:
        for cls in CLASS_NAMES:
            selected = _select_class_level(atlas, cls, level)
            idx_to_rank = {idx: rank for rank, idx in enumerate(selected, start=1)}
            mask = manifest["idx"].astype(int).isin(idx_to_rank)
            manifest.loc[mask, f"selected_l{level}"] = True
            manifest.loc[mask, f"selected_l{level}_rank"] = manifest.loc[mask, "idx"].astype(int).map(idx_to_rank)
        chosen = manifest.loc[_safe_bool(manifest[f"selected_l{level}"])].copy()
        for cls in CLASS_NAMES:
            cdf = chosen.loc[chosen["class_name"].eq(cls)]
            row: dict[str, Any] = {
                "level_n_per_class": int(level),
                "class_name": cls,
                "selected_n": int(len(cdf)),
                "ambiguous_fraction": float(cdf["clean_tier"].astype(str).eq("ambiguous_boundary").mean()) if len(cdf) else math.nan,
                "outlier_fraction": float(cdf["original_region"].eq("outlier_low_confidence").mean()) if len(cdf) else math.nan,
                "median_confidence": float(cdf["boundary_confidence"].median()) if len(cdf) else math.nan,
            }
            for region in REGION_COLORS:
                row[f"region_{region}"] = int(cdf["original_region"].eq(region).sum())
            composition_rows.append(row)
    manifest["soft_good"] = 0.0
    manifest["soft_medium"] = 0.0
    manifest["soft_bad"] = 0.0
    manifest.loc[manifest["class_name"].eq("good"), "soft_good"] = 1.0
    manifest.loc[manifest["class_name"].eq("medium"), "soft_medium"] = 1.0
    manifest.loc[manifest["class_name"].eq("bad"), "soft_bad"] = 1.0
    gm = manifest["original_region"].eq("good_medium_overlap")
    manifest.loc[gm & manifest["class_name"].eq("good"), ["soft_good", "soft_medium"]] = [0.72, 0.28]
    manifest.loc[gm & manifest["class_name"].eq("medium"), ["soft_good", "soft_medium"]] = [0.32, 0.68]
    mb = manifest["original_region"].eq("medium_bad_overlap")
    manifest.loc[mb & manifest["class_name"].eq("medium"), ["soft_medium", "soft_bad"]] = [0.72, 0.28]
    manifest.loc[mb & manifest["class_name"].eq("bad"), ["soft_medium", "soft_bad"]] = [0.22, 0.78]
    manifest["sample_weight"] = 1.0
    manifest.loc[gm, "sample_weight"] = 0.75
    manifest.loc[mb, "sample_weight"] = 0.65
    manifest.loc[manifest["original_region"].eq("outlier_low_confidence"), "sample_weight"] = 0.35
    manifest.loc[manifest["original_region"].eq("right_bad_island"), "sample_weight"] = 1.0

    manifest.to_csv(out_root / REGION_QUOTA_NAME, index=False)
    manifest.to_csv(report_root / REGION_QUOTA_NAME, index=False)
    comp = pd.DataFrame(composition_rows)
    comp.to_csv(out_root / "region_quota_level_composition.csv", index=False)
    comp.to_csv(report_root / "region_quota_level_composition.csv", index=False)
    write_json(
        out_root / "region_quota_rules.json",
        {
            "levels": levels,
            "quota_policy": {str(level): _level_quota(level) for level in levels},
            "soft_label_policy": {
                "good_medium_overlap_good": [0.72, 0.28, 0.0],
                "good_medium_overlap_medium": [0.32, 0.68, 0.0],
                "medium_bad_overlap_medium": [0.0, 0.72, 0.28],
                "medium_bad_overlap_bad": [0.0, 0.22, 0.78],
            },
            "note": "Region quotas and soft labels are prediction-free; they are intended for future boundary-aware training.",
        },
    )
    return manifest


def _completed_rows(train_root: Path) -> list[dict[str, Any]]:
    rows = semiclean.completed_training_rows(train_root)
    by_id = {str(r.get("spec", {}).get("id", "")): r for r in rows if r.get("spec", {}).get("id")}
    return list(by_id.values())


def _load_or_predict(args: argparse.Namespace, row: dict[str, Any], X: np.ndarray, device: torch.device) -> np.ndarray:
    train_root = Path(args.semiclean_root)
    vid = str(row.get("spec", {}).get("id", ""))
    cache_candidates = [
        train_root / "semiclean_predictions" / f"{vid}_probs.npz",
        Path(args.out_root) / "predictions" / f"{vid}_probs.npz",
    ]
    for path in cache_candidates:
        if path.exists():
            return np.load(path)["probs"].astype(np.float32)
    ckpt = Path(row["but_10s_eval"]["checkpoint"])
    probs = relax.predict_probs(ckpt, X, int(args.batch_size_eval), device)
    pred_dir = Path(args.out_root) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"{vid}_probs.npz", probs=probs.astype(np.float32))
    return probs.astype(np.float32)


def _selected_indices(manifest: pd.DataFrame, level: int) -> np.ndarray:
    col = f"selected_l{int(level)}"
    return manifest.loc[_safe_bool(manifest[col]), "idx"].astype(int).to_numpy()


def _composition_for_indices(manifest: pd.DataFrame, idx: np.ndarray) -> dict[str, Any]:
    sub = manifest.loc[manifest["idx"].astype(int).isin(idx)].copy()
    out: dict[str, Any] = {
        "ambiguous_fraction": float(sub["clean_tier"].astype(str).eq("ambiguous_boundary").mean()) if len(sub) else math.nan,
        "outlier_fraction": float(sub["original_region"].eq("outlier_low_confidence").mean()) if len(sub) else math.nan,
    }
    for region in REGION_COLORS:
        out[f"region_{region}_fraction"] = float(sub["original_region"].eq(region).mean()) if len(sub) else math.nan
    return out


def evaluate_region_quota(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    manifest_path = out_root / REGION_QUOTA_NAME
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else build_region_quota_manifest(args)
    X, meta = ensure_but_10s(args)
    y = meta["y"].to_numpy(dtype=np.int64)
    rows = _completed_rows(Path(args.semiclean_root))
    if not rows:
        raise FileNotFoundError(f"No completed SemiClean checkpoints found under {args.semiclean_root}")
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(args.cpu) else "cpu")
    levels = [int(v) for v in str(args.levels).split(",") if str(v).strip()]
    metric_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        vid = str(row.get("spec", {}).get("id", ""))
        update_state(out_root / STATE_NAME, status="evaluating", stage="diagnostic", current=vid, completed=i - 1, total=len(rows), updated_at=now_iso())
        probs = _load_or_predict(args, row, X, device)
        raw_pred = np.argmax(probs, axis=1).astype(np.int64)
        cal = row.get("but_10s_eval", {}).get("calibration", {})
        cal_pred = apply_but_thresholds(probs, float(cal.get("t_good", 0.5)), float(cal.get("t_bad", 0.5)))
        original = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        for level in levels:
            idx = _selected_indices(manifest, level)
            comp = _composition_for_indices(manifest, idx)
            for mode, pred in (("raw", raw_pred), ("calibrated", cal_pred)):
                rep = multiclass_report(y[idx], pred[idx], probs[idx])
                rec = rep.get("recall_good_medium_bad", [math.nan, math.nan, math.nan])
                metric_rows.append(
                    {
                        "variant_id": vid,
                        "prediction_mode": mode,
                        "level_n_per_class": int(level),
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
        full = manifest[["idx", "class_name", "split", "original_region"]].copy()
        full["idx"] = full["idx"].astype(int)
        for split in ["all", "train", "val", "test"]:
            split_mask = np.ones(len(full), dtype=bool) if split == "all" else full["split"].astype(str).eq(split).to_numpy()
            for region in REGION_COLORS:
                pos = full.index[split_mask & full["original_region"].astype(str).eq(region)].to_numpy()
                if len(pos) < int(args.min_region_eval_n):
                    continue
                idx = full.iloc[pos]["idx"].astype(int).to_numpy()
                for mode, pred in (("raw", raw_pred), ("calibrated", cal_pred)):
                    rep = multiclass_report(y[idx], pred[idx], probs[idx])
                    rec = rep.get("recall_good_medium_bad", [math.nan, math.nan, math.nan])
                    region_rows.append(
                        {
                            "variant_id": vid,
                            "prediction_mode": mode,
                            "split": split,
                            "original_region": region,
                            "n": int(len(idx)),
                            "acc": rep.get("acc"),
                            "balanced_acc": rep.get("balanced_acc"),
                            "macro_f1": rep.get("macro_f1"),
                            "good_recall": rec[0],
                            "medium_recall": rec[1],
                            "bad_recall": rec[2],
                            "class_counts": json.dumps({c: int((full.iloc[pos]["class_name"].astype(str) == c).sum()) for c in CLASS_NAMES}),
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    regions = pd.DataFrame(region_rows)
    metrics.to_csv(out_root / LEVEL_METRICS_NAME, index=False)
    regions.to_csv(out_root / REGION_METRICS_NAME, index=False)
    metrics.to_csv(report_root / LEVEL_METRICS_NAME, index=False)
    regions.to_csv(report_root / REGION_METRICS_NAME, index=False)

    best_rows: list[dict[str, Any]] = []
    for (vid, mode), sub in metrics.groupby(["variant_id", "prediction_mode"]):
        target = float(args.target_acc)
        band = float(args.target_acc_band)
        in_band = sub.loc[(sub["acc"] >= target - band) & (sub["acc"] <= target + band)].sort_values(["level_n_per_class", "acc"], ascending=[False, False])
        if in_band.empty:
            above = sub.loc[sub["acc"] >= target].sort_values(["level_n_per_class", "acc"], ascending=[False, True])
            chosen = above.iloc[0] if not above.empty else sub.iloc[(sub["acc"] - target).abs().argsort()[:1]].iloc[0]
            status = "above_target_or_closest"
        else:
            chosen = in_band.iloc[0]
            status = "in_0p95_band_max_coverage"
        d = chosen.to_dict()
        d["status"] = status
        best_rows.append(d)
    best = pd.DataFrame(best_rows).sort_values(["status", "level_n_per_class", "acc"], ascending=[True, False, False])
    best.to_csv(out_root / "region_quota_best_at_0p95.csv", index=False)
    best.to_csv(report_root / "region_quota_best_at_0p95.csv", index=False)
    return metrics, regions


def _sample_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= int(max_rows):
        return df
    return df.sample(int(max_rows), random_state=int(seed))


def plot_region_atlas(args: argparse.Namespace, atlas: pd.DataFrame) -> None:
    fig_dir = Path(args.report_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_df = _sample_df(atlas, int(args.max_plot_rows), 20260607)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), sharex=True, sharey=True)
    for cls in CLASS_NAMES:
        sub = plot_df.loc[plot_df["class_name"].eq(cls)]
        axes[0].scatter(sub["pc1"], sub["pc2"], s=5, c=COLORS[cls], alpha=0.42, linewidths=0, label=cls)
    axes[0].set_title("Original BUT atlas by label")
    for region, color in REGION_COLORS.items():
        sub = plot_df.loc[plot_df["original_region"].eq(region)]
        axes[1].scatter(sub["pc1"], sub["pc2"], s=5, c=color, alpha=0.45, linewidths=0, label=region)
    axes[1].set_title("Prediction-free original regions")
    for ax in axes:
        ax.set_xlabel("CleanBUT 64D-PC1")
        ax.set_ylabel("CleanBUT 64D-PC2")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7, markerscale=2)
    fig.suptitle("Original-aware region atlas")
    fig.tight_layout()
    fig.savefig(fig_dir / "original_region_atlas_pca.png", dpi=190)
    plt.close(fig)


def plot_boundary_levels(args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    fig_dir = Path(args.report_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(v) for v in str(args.levels).split(",") if str(v).strip()]
    bg = _sample_df(manifest, int(args.max_plot_rows), 20260608)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    for ax, level in zip(axes.ravel(), levels[:4]):
        ax.scatter(bg["pc1"], bg["pc2"], s=3, c="#d1d5db", alpha=0.14, linewidths=0)
        sel = manifest.loc[_safe_bool(manifest[f"selected_l{level}"])].copy()
        if len(sel) > int(args.max_plot_rows):
            sel = sel.sample(int(args.max_plot_rows), random_state=level)
        for cls in CLASS_NAMES:
            sub = sel.loc[sel["class_name"].eq(cls)]
            ax.scatter(sub["pc1"], sub["pc2"], s=6, c=COLORS[cls], alpha=0.68, linewidths=0, label=cls)
        ax.set_title(f"Region-quota SemiClean level {level}/class")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.16)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "region_quota_boundary_levels_pca.png", dpi=190)
    plt.close(fig)

    comp = pd.read_csv(Path(args.out_root) / "region_quota_level_composition.csv")
    for level in levels:
        view = comp.loc[comp["level_n_per_class"].eq(level)].copy()
        if view.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.2, 5.1))
        bottom = np.zeros(len(CLASS_NAMES), dtype=float)
        for region, color in REGION_COLORS.items():
            vals = []
            for cls in CLASS_NAMES:
                vals.append(float(view.loc[view["class_name"].eq(cls), f"region_{region}"].sum()))
            ax.bar(CLASS_NAMES, vals, bottom=bottom, label=region, color=color, alpha=0.82)
            bottom += np.asarray(vals, dtype=float)
        ax.set_ylabel("selected windows")
        ax.set_title(f"Region composition at level {level}/class")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"region_composition_l{level}.png", dpi=180)
        plt.close(fig)


def plot_metric_curves(args: argparse.Namespace, metrics: pd.DataFrame, region_metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    fig_dir = Path(args.report_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = metrics.loc[metrics["prediction_mode"].eq("raw")].copy()
    top = (
        raw.sort_values(["level_n_per_class", "acc"], ascending=[False, False])
        .drop_duplicates("variant_id")
        .head(8)["variant_id"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for vid in top:
        sub = raw.loc[raw["variant_id"].eq(vid)].sort_values("level_n_per_class")
        ax.plot(sub["level_n_per_class"], sub["acc"], marker="o", ms=4, lw=1.5, label=str(vid)[:48])
    ax.axhline(float(args.target_acc), color="#111827", ls="--", lw=1.0, label=f"target={float(args.target_acc):.2f}")
    ax.fill_between(
        [raw["level_n_per_class"].min(), raw["level_n_per_class"].max()],
        float(args.target_acc) - float(args.target_acc_band),
        float(args.target_acc) + float(args.target_acc_band),
        color="#9ca3af",
        alpha=0.12,
    )
    ax.set_xlabel("region-quota selected windows per class")
    ax.set_ylabel("filtered diagnostic accuracy")
    ax.set_ylim(0.25, 1.02)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Original-aware SemiClean boundary relaxation")
    fig.tight_layout()
    fig.savefig(fig_dir / "region_quota_relaxation_curve.png", dpi=180)
    plt.close(fig)

    if not region_metrics.empty:
        val = region_metrics.loc[region_metrics["prediction_mode"].eq("raw") & region_metrics["split"].isin(["val", "test"])].copy()
        best_vid = raw.sort_values(["original_but_macro_f1", "acc"], ascending=[False, False]).head(1)["variant_id"].iloc[0]
        val = val.loc[val["variant_id"].eq(best_vid)]
        pivot = val.pivot_table(index="original_region", columns="split", values="acc", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        im = ax.imshow(pivot.fillna(np.nan).to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{float(v):.2f}", ha="center", va="center", color="white" if float(v) < 0.55 else "black", fontsize=8)
        ax.set_title(f"Original region accuracy by split\n{best_vid}")
        fig.colorbar(im, ax=ax, label="accuracy")
        fig.tight_layout()
        fig.savefig(fig_dir / "original_region_accuracy_heatmap.png", dpi=180)
        plt.close(fig)


def make_region_galleries(args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    report_root = Path(args.report_root)
    gallery_dir = report_root / "original_region_waveform_galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    X, meta = ensure_but_10s(args)
    for region in ["good_medium_overlap", "medium_bad_overlap", "right_bad_island", "near_bad_boundary", "outlier_low_confidence"]:
        sub = manifest.loc[manifest["original_region"].eq(region)].sort_values("boundary_confidence", ascending=True).head(12)
        idx = sub["idx"].astype(int).tolist()
        if idx:
            plot_wave_gallery(X, meta, idx, gallery_dir / f"{region}.png", f"Original region: {region}")
    for level in [int(v) for v in str(args.levels).split(",") if str(v).strip()]:
        col = f"selected_l{level}"
        if col not in manifest:
            continue
        sel = manifest.loc[_safe_bool(manifest[col])]
        for cls in CLASS_NAMES:
            sub = sel.loc[sel["class_name"].eq(cls)].sort_values("region_confidence", ascending=True).head(12)
            idx = sub["idx"].astype(int).tolist()
            if idx:
                plot_wave_gallery(X, meta, idx, gallery_dir / f"level_{level}_{cls}_lowest_confidence.png", f"Level {level} {cls} boundary shell")


def write_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(out_root / "original_region_summary.csv") if (out_root / "original_region_summary.csv").exists() else pd.DataFrame()
    comp = pd.read_csv(out_root / "region_quota_level_composition.csv") if (out_root / "region_quota_level_composition.csv").exists() else pd.DataFrame()
    metrics = pd.read_csv(out_root / LEVEL_METRICS_NAME) if (out_root / LEVEL_METRICS_NAME).exists() else pd.DataFrame()
    best = pd.read_csv(out_root / "region_quota_best_at_0p95.csv") if (out_root / "region_quota_best_at_0p95.csv").exists() else pd.DataFrame()
    regions = pd.read_csv(out_root / REGION_METRICS_NAME) if (out_root / REGION_METRICS_NAME).exists() else pd.DataFrame()
    best_raw = best.loc[best["prediction_mode"].eq("raw")].sort_values(["status", "level_n_per_class", "acc"], ascending=[True, False, False]) if not best.empty else pd.DataFrame()
    lines = [
        "# Original-Aware SemiCleanBUT Boundary",
        "",
        "This package widens the SemiClean diagnostic boundary with prediction-free original-region quotas. It does not replace formal original BUT 10s P1; original test remains report-only.",
        "",
        "## Region Atlas",
        "",
        targeted.md_table(summary.loc[summary["split"].eq("all"), ["class_name", "original_region", "n", "fraction_within_class", "median_confidence", "median_pc1", "median_pc2"]] if not summary.empty else summary, max_rows=24),
        "",
        "## Region-Quota Boundary Levels",
        "",
        targeted.md_table(comp[["level_n_per_class", "class_name", "selected_n", "ambiguous_fraction", "outlier_fraction", "region_clean_core", "region_good_medium_overlap", "region_medium_bad_overlap", "region_right_bad_island", "region_near_bad_boundary"]] if not comp.empty else comp, max_rows=24),
        "",
        "## Best Filtered Diagnostics",
        "",
        targeted.md_table(best_raw[["status", "variant_id", "prediction_mode", "level_n_per_class", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall", "ambiguous_fraction", "original_but_macro_f1"]] if not best_raw.empty else best_raw, max_rows=12),
        "",
        "## Original Region Accuracy",
        "",
        targeted.md_table(regions.loc[(regions["prediction_mode"].eq("raw")) & (regions["split"].isin(["val", "test"])), ["variant_id", "split", "original_region", "n", "acc", "macro_f1", "class_counts"]] if not regions.empty else regions, max_rows=24),
        "",
        "## Files",
        "",
        f"- `{REGION_ATLAS_NAME}`: original-region labels for every BUT window.",
        f"- `{REGION_QUOTA_NAME}`: level selections plus sample weights and soft labels for boundary-aware training.",
        f"- `{LEVEL_METRICS_NAME}`: region-quota filtered diagnostic metrics.",
        f"- `{REGION_METRICS_NAME}`: original BUT model behavior by region and split.",
        "- `figures/original_region_atlas_pca.png`: original labels and region atlas.",
        "- `figures/region_quota_boundary_levels_pca.png`: boundary-level selections.",
        "- `figures/region_quota_relaxation_curve.png`: acc as boundary widens.",
        "- `figures/original_region_accuracy_heatmap.png`: val/test region accuracy for the best original macro candidate.",
        "- `original_region_waveform_galleries/`: overlap, bad island, near-boundary, and boundary-shell examples.",
        "",
        "## Interpretation",
        "",
        "If the 4800/class level remains below 0.95, inspect medium recall and the good-medium overlap rows first. Bad is intentionally mostly locked to the right island, with only a controlled near-boundary quota as the level widens.",
    ]
    (report_root / "original_aware_semiclean_boundary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / STATE_NAME
    if state_path.exists():
        state = read_json(state_path)
        state.pop("error", None)
        write_json(state_path, state)
    update_state(out_root / STATE_NAME, status="running", stage=args.stage, updated_at=now_iso())
    atlas = pd.DataFrame()
    manifest = pd.DataFrame()
    metrics = pd.DataFrame()
    region_metrics = pd.DataFrame()
    if args.stage in {"atlas", "all"}:
        atlas = build_region_atlas(args)
        plot_region_atlas(args, atlas)
    if args.stage in {"manifest", "all"}:
        manifest = build_region_quota_manifest(args)
        plot_boundary_levels(args, manifest)
        make_region_galleries(args, manifest)
    if args.stage in {"diagnostic", "all"}:
        metrics, region_metrics = evaluate_region_quota(args)
        plot_metric_curves(args, metrics, region_metrics)
    if args.stage in {"report", "all"}:
        if atlas.empty and (out_root / REGION_ATLAS_NAME).exists():
            atlas = pd.read_csv(out_root / REGION_ATLAS_NAME)
            plot_region_atlas(args, atlas)
        if manifest.empty and (out_root / REGION_QUOTA_NAME).exists():
            manifest = pd.read_csv(out_root / REGION_QUOTA_NAME)
            plot_boundary_levels(args, manifest)
        if (out_root / LEVEL_METRICS_NAME).exists() and (out_root / REGION_METRICS_NAME).exists():
            metrics = pd.read_csv(out_root / LEVEL_METRICS_NAME)
            region_metrics = pd.read_csv(out_root / REGION_METRICS_NAME)
            plot_metric_curves(args, metrics, region_metrics)
        write_report(args)
    state = read_json(out_root / STATE_NAME)
    total = int(state.get("total", 0) or 0)
    complete_payload: dict[str, Any] = {"status": "complete", "stage": args.stage, "updated_at": now_iso()}
    if total > 0:
        complete_payload.update({"completed": total, "total": total})
    update_state(out_root / STATE_NAME, **complete_payload)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build original-aware SemiCleanBUT region atlas and boundary diagnostics.")
    p.add_argument("--stage", choices=("atlas", "manifest", "diagnostic", "report", "all"), default="all")
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    p.add_argument("--clean_root", default=str(DEFAULT_CLEAN_ROOT))
    p.add_argument("--semiclean_root", default=str(DEFAULT_SEMICLEAN_ROOT))
    p.add_argument("--protocol_out_root", default=str(targeted.PROTOCOL_OUT_ROOT))
    p.add_argument("--protocol_report_root", default=str(targeted.PROTOCOL_REPORT_ROOT))
    p.add_argument("--levels", default="3600,4200,4800,5200")
    p.add_argument("--target_acc", type=float, default=0.95)
    p.add_argument("--target_acc_band", type=float, default=0.01)
    p.add_argument("--batch_size_eval", type=int, default=256)
    p.add_argument("--min_region_eval_n", type=int, default=80)
    p.add_argument("--max_plot_rows", type=int, default=18000)
    p.add_argument("--cpu", action="store_true")
    return p


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
