"""BUT split roll and interpretable waveform-SQI audit.

This is external-benchmark experiment code only.  It does not modify
``src/sqi_pipeline`` or mainline checkpoints.

Stages:
  split_audit   - generate strict subject-level BUT split candidates and a
                  diagnostic window-random upper-bound split.
  feature_audit - compare waveform-computable SQI/morphology/RR features
                  across BUT splits and PTB synthetic target geometry.
  capacity      - optional BUT-only waveform Transformer capacity diagnostic
                  on one generated split scheme.

Official PTB->BUT model selection must not use BUT rows.  The capacity stage is
diagnostic only and writes that contract into every output artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
AUDIT_DIR = ANALYSIS_DIR / "but_split_interpretable_audit"
AUDIT_REPORT_DIR = REPORT_DIR / "but_split_interpretable_audit"
ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
METADATA = ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02" / "processed" / "butqdb" / "metadata.csv"
TAXONOMY_PATH = ANALYSIS_DIR / "waveform_sqi_module_probe_feature_taxonomy.csv"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"
RUN_ROOT = OUT_ROOT / "runs" / "waveform_geometry_student" / "N17043_gm_probe" / "search"
DEFAULT_SYNTH_AUX = (
    OUT_ROOT
    / "synthetic_variants"
    / "nl_n17043_gm_trim_bad_boundaryblocks_large_badcore_balanc_678e0e2e2775"
    / "geometry_selected_aux_rows.csv"
)

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
SPLITS = ("train", "val", "test")
TARGET_PROPS = {"train": 0.50, "val": 0.25, "test": 0.25}


STABLE_WAVEFORM_FEATURES = [
    "qrs_visibility",
    "qrs_band_ratio",
    "qrs_prom_p90",
    "template_corr",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "diff_abs_p95",
    "band_0p3_1",
    "band_1_5",
    "band_5_15",
    "band_15_30",
    "band_30_45",
    "rms",
    "std",
    "mean_abs",
    "ptp_p99_p01",
    "amplitude_entropy",
    "low_amp_ratio",
    "sqi_iSQI",
    "sqi_bSQI",
    "sqi_pSQI",
    "sqi_sSQI",
    "sqi_kSQI",
    "sqi_fSQI",
    "sqi_basSQI",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "zero_crossing_rate",
    "diff_zero_crossing_rate",
    "sample_entropy_proxy",
    "higuchi_fd_proxy",
    "wavelet_e0",
    "wavelet_e1",
    "wavelet_e2",
    "wavelet_e3",
    "wavelet_e4",
    "fatal_or_score",
]
WEAK_GEOMETRY_PROXIES = ["pc1", "pc2", "pc3", "pc4", "pca_margin"]
ATLAS_ONLY_GEOMETRY = [
    "boundary_confidence",
    "region_confidence",
    "knn_label_purity",
    "class_margin_percentile",
    "class_centrality_percentile",
]


def ensure_dirs() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_atlas() -> pd.DataFrame:
    df = pd.read_csv(ATLAS)
    df = df.reset_index(drop=True)
    df.insert(0, "row_pos", np.arange(len(df), dtype=np.int64))
    df["subject_id"] = df["subject_id"].astype(str)
    df["record_id"] = df["record_id"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    df["original_region"] = df["original_region"].astype(str)
    return df


def load_taxonomy(atlas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if TAXONOMY_PATH.exists():
        raw = pd.read_csv(TAXONOMY_PATH)
        for _, row in raw.iterrows():
            feat = str(row["feature"])
            if feat in atlas.columns:
                rows.append({"feature": feat, "taxonomy": str(row["taxonomy"])})
    seen = {r["feature"] for r in rows}
    for feat in STABLE_WAVEFORM_FEATURES:
        if feat in atlas.columns and feat not in seen:
            rows.append({"feature": feat, "taxonomy": "stable_waveform_sqi_morph_rr"})
            seen.add(feat)
    for feat in WEAK_GEOMETRY_PROXIES:
        if feat in atlas.columns and feat not in seen:
            rows.append({"feature": feat, "taxonomy": "weak_target_distribution_proxy"})
            seen.add(feat)
    for feat in ATLAS_ONLY_GEOMETRY:
        if feat in atlas.columns and feat not in seen:
            rows.append({"feature": feat, "taxonomy": "atlas_label_geometry_not_waveform_fact"})
            seen.add(feat)
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            view[col] = view[col].astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def split_vector_from_subjects(df: pd.DataFrame, subject_to_split: dict[str, str]) -> np.ndarray:
    return df["subject_id"].map(subject_to_split).fillna("train").to_numpy(dtype=object)


def summarize_split(df: pd.DataFrame, split_col: str) -> dict[str, Any]:
    total = max(len(df), 1)
    out: dict[str, Any] = {}
    for split in SPLITS:
        s = df[df[split_col].eq(split)]
        n = len(s)
        out[f"{split}_n"] = int(n)
        out[f"{split}_prop"] = float(n / total)
        out[f"{split}_subjects"] = int(s["subject_id"].nunique())
        out[f"{split}_records"] = int(s["record_id"].nunique())
        out[f"{split}_max_record_share"] = float(s.groupby("record_id").size().max() / max(n, 1)) if n else 1.0
        for cls in CLASS_TO_INT:
            out[f"{split}_{cls}"] = int((s["class_name"] == cls).sum())
        for region in ["clean_core", "good_medium_overlap", "medium_bad_overlap", "near_bad_boundary", "outlier_low_confidence"]:
            out[f"{split}_{region}"] = int((s["original_region"] == region).sum())
    return out


def distribution_distance(counts: pd.Series, target_dist: pd.Series) -> float:
    denom = float(counts.sum())
    if denom <= 0:
        return 10.0
    dist = counts / denom
    dist = dist.reindex(target_dist.index).fillna(0.0)
    return float(np.abs(dist - target_dist).sum())


def score_split(df: pd.DataFrame, split_col: str) -> float:
    total = len(df)
    class_target = df["class_name"].value_counts(normalize=True).sort_index()
    region_target = df["original_region"].value_counts(normalize=True).sort_index()
    cr_target = df.groupby(["class_name", "original_region"]).size()
    cr_target = cr_target / cr_target.sum()
    score = 0.0
    for split in SPLITS:
        s = df[df[split_col].eq(split)]
        n = len(s)
        prop = n / max(total, 1)
        score += 1.25 * abs(prop - TARGET_PROPS[split])
        if n == 0:
            score += 25.0
            continue
        missing_classes = 3 - s["class_name"].nunique()
        score += 5.0 * missing_classes
        score += 2.5 * distribution_distance(s["class_name"].value_counts().sort_index(), class_target)
        score += 1.25 * distribution_distance(s["original_region"].value_counts().sort_index(), region_target)
        cr_counts = s.groupby(["class_name", "original_region"]).size()
        score += 1.75 * distribution_distance(cr_counts, cr_target)
        max_record_share = s.groupby("record_id").size().max() / max(n, 1)
        score += 1.25 * max(0.0, float(max_record_share) - 0.75)
    return float(score)


def random_subject_assignment(subjects: list[str], rng: np.random.Generator) -> dict[str, str]:
    # The 3 largest subjects make a 60/20/20 strict split infeasible.  Encourage
    # one large subject per split, then distribute the small subjects randomly.
    subject_sizes = {s: 0 for s in subjects}
    return {s: str(rng.choice(SPLITS, p=[0.50, 0.25, 0.25])) for s in subject_sizes}


def build_window_random_split(df: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assigned = np.empty(len(df), dtype=object)
    for _, sub in df.groupby(["class_name", "original_region"], dropna=False):
        idx = sub.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(0.60 * n))
        n_val = int(round(0.20 * n))
        assigned[idx[:n_train]] = "train"
        assigned[idx[n_train : n_train + n_val]] = "val"
        assigned[idx[n_train + n_val :]] = "test"
    return assigned


def split_array_from_candidate_seed(assignments: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve a strict subject-level split from the saved split-roll table."""
    candidates_path = AUDIT_DIR / "but_subject_split_roll_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Run split_audit first: {candidates_path}")
    candidates = pd.read_csv(candidates_path)
    match = candidates[candidates["candidate_seed"].astype(int).eq(int(seed))]
    if match.empty:
        raise ValueError(f"candidate split seed {seed} not found in {candidates_path}")
    row = match.iloc[0].to_dict()
    subject_to_split: dict[str, str] = {}
    for split in SPLITS:
        raw = str(row.get(f"{split}_subjects", "") or "")
        for subject in [part.strip() for part in raw.split(",") if part.strip()]:
            subject_to_split[str(subject)] = split
    if not subject_to_split:
        raise ValueError(f"candidate split seed {seed} did not define subjects")
    split_array = assignments["subject_id"].astype(str).map(subject_to_split).fillna("train").to_numpy(dtype=object)
    return split_array, row


def roll_strict_splits(df: pd.DataFrame, seeds: int) -> pd.DataFrame:
    subjects = sorted(df["subject_id"].unique().tolist())
    class_labels = sorted(CLASS_TO_INT.keys())
    region_labels = sorted(df["original_region"].unique().tolist())
    cr_labels = [(c, r) for c in class_labels for r in region_labels]
    subject_rows = []
    for subject in subjects:
        s = df[df["subject_id"].eq(subject)]
        row: dict[str, Any] = {"subject_id": subject, "n": int(len(s))}
        for cls in class_labels:
            row[f"class:{cls}"] = int(s["class_name"].eq(cls).sum())
        for region in region_labels:
            row[f"region:{region}"] = int(s["original_region"].eq(region).sum())
        for cls, region in cr_labels:
            row[f"cr:{cls}:{region}"] = int((s["class_name"].eq(cls) & s["original_region"].eq(region)).sum())
        # Large subjects may contain more than one record.  Track record
        # dominance with the largest record inside each subject.
        row["max_record_n"] = int(s.groupby("record_id").size().max())
        subject_rows.append(row)
    subj = pd.DataFrame(subject_rows)
    n_arr = subj["n"].to_numpy(dtype=np.float64)
    max_record_arr = subj["max_record_n"].to_numpy(dtype=np.float64)
    class_mat = subj[[f"class:{c}" for c in class_labels]].to_numpy(dtype=np.float64)
    region_mat = subj[[f"region:{r}" for r in region_labels]].to_numpy(dtype=np.float64)
    cr_mat = subj[[f"cr:{c}:{r}" for c, r in cr_labels]].to_numpy(dtype=np.float64)
    total_n = float(n_arr.sum())
    class_target = class_mat.sum(axis=0) / max(class_mat.sum(), 1.0)
    region_target = region_mat.sum(axis=0) / max(region_mat.sum(), 1.0)
    cr_target = cr_mat.sum(axis=0) / max(cr_mat.sum(), 1.0)
    split_to_int = {s: i for i, s in enumerate(SPLITS)}

    def score_assignment(assign: np.ndarray) -> tuple[float, dict[str, Any]]:
        score = 0.0
        summary: dict[str, Any] = {}
        for split, split_i in split_to_int.items():
            mask = assign == split_i
            sn = float(n_arr[mask].sum())
            summary[f"{split}_n"] = int(sn)
            summary[f"{split}_prop"] = float(sn / max(total_n, 1.0))
            summary[f"{split}_subject_count"] = int(mask.sum())
            summary[f"{split}_record_count"] = int(mask.sum())
            summary[f"{split}_max_record_share"] = float(max_record_arr[mask].max() / max(sn, 1.0)) if mask.any() else 1.0
            score += 1.25 * abs((sn / max(total_n, 1.0)) - TARGET_PROPS[split])
            if sn <= 0:
                score += 25.0
                for cls in class_labels:
                    summary[f"{split}_{cls}"] = 0
                for region in ["clean_core", "good_medium_overlap", "medium_bad_overlap", "near_bad_boundary", "outlier_low_confidence"]:
                    summary[f"{split}_{region}"] = 0
                continue
            cls_counts = class_mat[mask].sum(axis=0)
            region_counts = region_mat[mask].sum(axis=0)
            cr_counts = cr_mat[mask].sum(axis=0)
            score += 5.0 * int((cls_counts > 0).sum() < 3)
            score += 2.5 * float(np.abs((cls_counts / max(cls_counts.sum(), 1.0)) - class_target).sum())
            score += 1.25 * float(np.abs((region_counts / max(region_counts.sum(), 1.0)) - region_target).sum())
            score += 1.75 * float(np.abs((cr_counts / max(cr_counts.sum(), 1.0)) - cr_target).sum())
            score += 1.25 * max(0.0, summary[f"{split}_max_record_share"] - 0.75)
            for j, cls in enumerate(class_labels):
                summary[f"{split}_{cls}"] = int(cls_counts[j])
            region_lookup = {r: int(region_counts[j]) for j, r in enumerate(region_labels)}
            for region in ["clean_core", "good_medium_overlap", "medium_bad_overlap", "near_bad_boundary", "outlier_low_confidence"]:
                summary[f"{split}_{region}"] = region_lookup.get(region, 0)
        return float(score), summary

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260618)
    for i in range(int(seeds)):
        assign = rng.choice(np.array([0, 1, 2], dtype=np.int64), size=len(subjects), p=[0.50, 0.25, 0.25])
        score, summary = score_assignment(assign)
        test_mask = assign == split_to_int["test"]
        test_n = max(float(n_arr[test_mask].sum()), 1.0)
        region_counts = region_mat[test_mask].sum(axis=0) if test_mask.any() else np.zeros(len(region_labels))
        class_counts = class_mat[test_mask].sum(axis=0) if test_mask.any() else np.zeros(len(class_labels))
        region_lookup = {r: float(region_counts[j]) for j, r in enumerate(region_labels)}
        class_lookup = {c: float(class_counts[j]) for j, c in enumerate(class_labels)}
        hard_score = float(
            region_lookup.get("outlier_low_confidence", 0.0) / test_n
            + 0.35 * region_lookup.get("good_medium_overlap", 0.0) / test_n
            + 0.25 * class_lookup.get("bad", 0.0) / test_n
        )
        assignment = {subjects[j]: SPLITS[int(assign[j])] for j in range(len(subjects))}
        rows.append(
            {
                "candidate_seed": i,
                "score": score,
                "hard_score": hard_score,
                "train_subjects": ",".join(sorted(k for k, v in assignment.items() if v == "train")),
                "val_subjects": ",".join(sorted(k for k, v in assignment.items() if v == "val")),
                "test_subjects": ",".join(sorted(k for k, v in assignment.items() if v == "test")),
                **summary,
            }
        )
    cand = pd.DataFrame(rows).sort_values(["score", "hard_score"], ascending=[True, False]).reset_index(drop=True)
    return cand


def install_split_columns(df: pd.DataFrame, candidates: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df[["row_pos", "idx", "subject_id", "record_id", "class_name", "original_region", "split"]].copy()
    out = out.rename(columns={"split": "current_split"})

    usable = candidates[
        candidates["train_n"].ge(12000)
        & candidates["val_n"].between(1500, 11000)
        & candidates["test_n"].between(6500, 11000)
        & candidates["train_good"].gt(0)
        & candidates["train_medium"].gt(0)
        & candidates["train_bad"].gt(0)
        & candidates["val_good"].gt(0)
        & candidates["val_medium"].gt(0)
        & candidates["val_bad"].gt(0)
        & candidates["test_good"].gt(0)
        & candidates["test_medium"].gt(0)
        & candidates["test_bad"].ge(100)
        & candidates["val_bad"].ge(50)
    ].copy()
    best_pool = usable if not usable.empty else candidates
    best = best_pool.sort_values(["score", "hard_score"], ascending=[True, False]).iloc[0].to_dict()
    hard_pool = usable if not usable.empty else candidates
    hard = hard_pool.sort_values(["hard_score", "score"], ascending=[False, True]).iloc[0].to_dict()

    def add_scheme(name: str, row: dict[str, Any]) -> None:
        mapping: dict[str, str] = {}
        for split in SPLITS:
            for subject in str(row[f"{split}_subjects"]).split(","):
                if subject:
                    mapping[subject] = split
        out[f"{name}_split"] = out["subject_id"].map(mapping).fillna("train")

    add_scheme("balanced_best", best)
    add_scheme("hard_test", hard)
    out["window_random_diagnostic_split"] = build_window_random_split(df, seed)
    meta = {
        "balanced_best_candidate_seed": int(best["candidate_seed"]),
        "hard_test_candidate_seed": int(hard["candidate_seed"]),
        "balanced_best_score": float(best["score"]),
        "hard_test_score": float(hard["score"]),
        "hard_test_hard_score": float(hard["hard_score"]),
        "window_random_seed": int(seed),
        "strict_split_unit": "subject_id",
        "diagnostic_window_random_is_not_external_claim": True,
    }
    return out, meta


def split_count_frame(assignments: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col in ["current_split", "balanced_best_split", "hard_test_split", "window_random_diagnostic_split"]:
        tmp = assignments.groupby([col, "class_name", "original_region", "record_id"], dropna=False).size().reset_index(name="n")
        tmp = tmp.rename(columns={col: "split"})
        tmp.insert(0, "scheme", col.replace("_split", ""))
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def feature_cols_for_taxonomy(taxonomy: pd.DataFrame, atlas: pd.DataFrame) -> list[str]:
    return [str(f) for f in taxonomy["feature"].tolist() if str(f) in atlas.columns]


def safe_array(frame: pd.DataFrame, feature: str) -> np.ndarray:
    return (
        pd.to_numeric(frame[feature], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float64)
    )


def feature_ks_by_scheme(atlas: pd.DataFrame, assignments: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    cols = feature_cols_for_taxonomy(taxonomy, atlas)
    merged = atlas.merge(assignments[["row_pos", "balanced_best_split", "hard_test_split", "window_random_diagnostic_split"]], on="row_pos", how="left")
    merged["current_split"] = merged["split"]
    tax_map = dict(zip(taxonomy["feature"], taxonomy["taxonomy"]))
    rows: list[dict[str, Any]] = []
    for scheme_col in ["current_split", "balanced_best_split", "hard_test_split", "window_random_diagnostic_split"]:
        scheme = scheme_col.replace("_split", "")
        for cls in ["all", "good", "medium", "bad"]:
            base = merged if cls == "all" else merged[merged["class_name"].eq(cls)]
            train = base[base[scheme_col].eq("train")]
            test = base[base[scheme_col].eq("test")]
            for feat in cols:
                a = safe_array(train, feat)
                b = safe_array(test, feat)
                if len(a) < 20 or len(b) < 20:
                    continue
                rows.append(
                    {
                        "scheme": scheme,
                        "class_name": cls,
                        "feature": feat,
                        "taxonomy": tax_map.get(feat, "unknown"),
                        "ks_train_vs_test": float(ks_2samp(a, b).statistic),
                        "train_median": float(np.median(a)),
                        "test_median": float(np.median(b)),
                        "delta_median_test_minus_train": float(np.median(b) - np.median(a)),
                        "train_p10": float(np.quantile(a, 0.10)),
                        "train_p90": float(np.quantile(a, 0.90)),
                        "test_p10": float(np.quantile(b, 0.10)),
                        "test_p90": float(np.quantile(b, 0.90)),
                        "n_train": int(len(a)),
                        "n_test": int(len(b)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["scheme", "class_name", "ks_train_vs_test"], ascending=[True, True, False])


def load_synth_aux(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    synth = pd.read_csv(path)
    if "selected_class" in synth.columns:
        synth["class_name"] = synth["selected_class"].astype(str)
    elif "y_class" in synth.columns:
        synth["class_name"] = synth["y_class"].astype(str)
    elif "y" in synth.columns:
        synth["class_name"] = synth["y"].map(INT_TO_CLASS).fillna(synth["y"].astype(str))
    else:
        return None
    return synth


def ptb_but_feature_gap(atlas: pd.DataFrame, synth: pd.DataFrame | None, taxonomy: pd.DataFrame) -> pd.DataFrame:
    if synth is None:
        return pd.DataFrame()
    stable = taxonomy[taxonomy["taxonomy"].eq("stable_waveform_sqi_morph_rr")]["feature"].tolist()
    cols = [f for f in stable if f in atlas.columns and f in synth.columns]
    rows: list[dict[str, Any]] = []
    for cls in ["all", "good", "medium", "bad"]:
        s = synth if cls == "all" else synth[synth["class_name"].eq(cls)]
        for but_bucket, but in [
            ("but_current_train", atlas[atlas["split"].eq("train")]),
            ("but_current_test", atlas[atlas["split"].eq("test")]),
            ("but_all", atlas),
        ]:
            b = but if cls == "all" else but[but["class_name"].eq(cls)]
            for feat in cols:
                a = safe_array(s, feat)
                c = safe_array(b, feat)
                if len(a) < 20 or len(c) < 20:
                    continue
                rows.append(
                    {
                        "class_name": cls,
                        "but_bucket": but_bucket,
                        "feature": feat,
                        "ks_ptb_synth_vs_but": float(ks_2samp(a, c).statistic),
                        "ptb_median": float(np.median(a)),
                        "but_median": float(np.median(c)),
                        "delta_but_minus_ptb": float(np.median(c) - np.median(a)),
                        "ptb_nonmissing": int(len(a)),
                        "but_nonmissing": int(len(c)),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["class_name", "but_bucket", "ks_ptb_synth_vs_but"], ascending=[True, True, False])


def pc_axis_correlations(atlas: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    stable = taxonomy[taxonomy["taxonomy"].eq("stable_waveform_sqi_morph_rr")]["feature"].tolist()
    axes = [c for c in ["pc1", "pc2", "pc3", "pc4", "pca_margin"] if c in atlas.columns]
    rows: list[dict[str, Any]] = []
    for axis in axes:
        ax = pd.to_numeric(atlas[axis], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for feat in stable:
            if feat not in atlas.columns:
                continue
            fv = pd.to_numeric(atlas[feat], errors="coerce").replace([np.inf, -np.inf], np.nan)
            ok = ax.notna() & fv.notna()
            if ok.sum() < 50:
                continue
            corr = float(np.corrcoef(ax[ok].to_numpy(dtype=np.float64), fv[ok].to_numpy(dtype=np.float64))[0, 1])
            rows.append({"axis": axis, "feature": feat, "pearson_corr": corr, "abs_corr": abs(corr), "n": int(ok.sum())})
    return pd.DataFrame(rows).sort_values(["axis", "abs_corr"], ascending=[True, False])


def run_split_audit(args: argparse.Namespace) -> None:
    ensure_dirs()
    atlas = load_atlas()
    taxonomy = load_taxonomy(atlas)
    candidates = roll_strict_splits(atlas.copy(), int(args.seeds))
    assignments, meta = install_split_columns(atlas, candidates, int(args.window_seed))
    counts = split_count_frame(assignments)
    taxonomy.to_csv(AUDIT_DIR / "feature_taxonomy_interpretable.csv", index=False)
    candidates.to_csv(AUDIT_DIR / "but_subject_split_roll_candidates.csv", index=False)
    assignments.to_csv(AUDIT_DIR / "but_split_roll_row_assignments.csv", index=False)
    counts.to_csv(AUDIT_DIR / "but_split_roll_counts.csv", index=False)
    (AUDIT_DIR / "but_split_roll_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = [
        "# BUT Split Roll Audit",
        "",
        "External-only diagnostic. Strict candidates split by `subject_id`; the window-random split is an explicitly leaky capacity upper bound.",
        "",
        "## Selected Strict Splits",
        "",
        md_table(pd.DataFrame([meta])),
        "",
        "## Best Candidate Rows",
        "",
        md_table(candidates.head(12), max_rows=12),
        "",
        "## Counts Snapshot",
        "",
        md_table(counts.groupby(["scheme", "split", "class_name"], dropna=False)["n"].sum().reset_index(), max_rows=40),
        "",
        "## Contract",
        "",
        "- `balanced_best_split` and `hard_test_split` are subject-level and suitable for BUT-only capacity diagnostics.",
        "- `window_random_diagnostic_split` is only an upper-bound sanity check and must not be used as an external-test claim.",
        "- PTB->BUT selection still cannot use any BUT split.",
        "",
    ]
    path = AUDIT_REPORT_DIR / "but_split_roll_report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"wrote {path}")


def run_feature_audit(args: argparse.Namespace) -> None:
    ensure_dirs()
    atlas = load_atlas()
    assignments_path = AUDIT_DIR / "but_split_roll_row_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Run split_audit first: {assignments_path}")
    assignments = pd.read_csv(assignments_path)
    taxonomy = load_taxonomy(atlas)
    taxonomy.to_csv(AUDIT_DIR / "feature_taxonomy_interpretable.csv", index=False)
    ks = feature_ks_by_scheme(atlas, assignments, taxonomy)
    ks.to_csv(AUDIT_DIR / "but_split_feature_ks.csv", index=False)
    synth = load_synth_aux(Path(args.synth_aux))
    ptb_gap = ptb_but_feature_gap(atlas, synth, taxonomy)
    ptb_gap.to_csv(AUDIT_DIR / "ptb_synthetic_vs_but_feature_gap.csv", index=False)
    pc_corr = pc_axis_correlations(atlas, taxonomy)
    pc_corr.to_csv(AUDIT_DIR / "pc_axis_waveform_feature_correlations.csv", index=False)
    stable = taxonomy[taxonomy["taxonomy"].eq("stable_waveform_sqi_morph_rr")]
    weak = taxonomy[taxonomy["taxonomy"].eq("weak_target_distribution_proxy")]
    atlas_only = taxonomy[taxonomy["taxonomy"].eq("atlas_label_geometry_not_waveform_fact")]
    top_current = ks[(ks["scheme"].eq("current")) & (ks["taxonomy"].eq("stable_waveform_sqi_morph_rr"))].head(16)
    top_ptb = ptb_gap[ptb_gap["but_bucket"].eq("but_current_test")].head(16) if not ptb_gap.empty else pd.DataFrame()
    report = [
        "# Interpretable Feature Distribution Audit",
        "",
        "This separates waveform-computable targets from weak geometry proxies and atlas-only label geometry.",
        "",
        "## Feature Taxonomy Counts",
        "",
        md_table(taxonomy.groupby("taxonomy").size().reset_index(name="n")),
        "",
        "## Stable Waveform-Computable Targets",
        "",
        md_table(stable, max_rows=80),
        "",
        "## Diagnostic-Only Geometry Proxies",
        "",
        md_table(weak, max_rows=40),
        "",
        "## Not Final Inference/Claim Targets",
        "",
        md_table(atlas_only, max_rows=40),
        "",
        "## Current Split: Largest Stable Feature Train-vs-Test Gaps",
        "",
        md_table(top_current, max_rows=16),
        "",
        "## PTB Synthetic vs BUT Current Test: Largest Stable Feature Gaps",
        "",
        md_table(top_ptb, max_rows=16),
        "",
        "## PC Axis Interpretation",
        "",
        "The PC columns are PCA coordinates in target-geometry/SQI space. The table below reports correlations with waveform-computable features, not physiological facts.",
        "",
        md_table(pc_corr.groupby("axis").head(8), max_rows=40),
        "",
        f"Feature KS CSV: `{AUDIT_DIR / 'but_split_feature_ks.csv'}`",
        f"PTB/BUT gap CSV: `{AUDIT_DIR / 'ptb_synthetic_vs_but_feature_gap.csv'}`",
        f"PC correlation CSV: `{AUDIT_DIR / 'pc_axis_waveform_feature_correlations.csv'}`",
        "",
    ]
    path = AUDIT_REPORT_DIR / "interpretable_feature_distribution_report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"wrote {path}")


class CapacitySplitDataset(Dataset):
    def __init__(self, base: Any, split_assignments: np.ndarray, split: str):
        self.base = base
        self.indices = np.flatnonzero(split_assignments == split).astype(np.int64)
        self.y = base.y[self.indices]
        self.split = split_assignments[self.indices]
        self.region = base.region[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[int(self.indices[i])]


def load_student_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predict_probs(model: torch.nn.Module, ds: Dataset, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x, mask_ratio=0.0)
        chunks.append(torch.softmax(out["logits"], dim=1).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def metric_report(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    conf = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y.astype(np.int64), pred.astype(np.int64)):
        conf[t, p] += 1
    recall = np.diag(conf) / np.maximum(conf.sum(axis=1), 1)
    precision = np.diag(conf) / np.maximum(conf.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "acc": float((pred == y).mean()),
        "macro_f1": float(np.nanmean(f1)),
        "good_recall": float(recall[0]),
        "medium_recall": float(recall[1]),
        "bad_recall": float(recall[2]),
        "confusion_3x3": conf.tolist(),
    }


def split_prediction_frame(atlas: pd.DataFrame, name: str, ds: CapacitySplitDataset, probs: np.ndarray) -> pd.DataFrame:
    pred = probs.argmax(axis=1).astype(np.int64)
    frame = atlas.iloc[ds.indices].copy().reset_index(drop=True)
    frame.insert(0, "capacity_bucket", name)
    frame["true_int"] = ds.y.astype(np.int64)
    frame["pred_int"] = pred
    frame["true_label"] = [INT_TO_CLASS[int(v)] for v in ds.y]
    frame["pred_label"] = [INT_TO_CLASS[int(v)] for v in pred]
    frame["correct"] = pred == ds.y
    frame["prob_good"] = probs[:, 0]
    frame["prob_medium"] = probs[:, 1]
    frame["prob_bad"] = probs[:, 2]
    return frame


def run_capacity(args: argparse.Namespace) -> None:
    ensure_dirs()
    assignments_path = AUDIT_DIR / "but_split_roll_row_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Run split_audit first: {assignments_path}")
    assignments = pd.read_csv(assignments_path)
    split_col = f"{args.scheme}_split"
    candidate_split_meta: dict[str, Any] | None = None
    if int(args.candidate_split_seed) >= 0:
        split_array, candidate_split_meta = split_array_from_candidate_seed(assignments, int(args.candidate_split_seed))
        scheme_label = f"candidate_seed_{int(args.candidate_split_seed)}"
    else:
        if split_col not in assignments.columns:
            raise ValueError(f"Unknown split scheme {args.scheme!r}; expected column {split_col}")
        split_array = assignments[split_col].to_numpy(dtype=object)
        scheme_label = str(args.scheme)
    mod = load_student_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(args.batch_size or (256 if device.type == "cuda" else 96))
    ckpt = torch.load(RUN_ROOT / args.candidate / "ckpt_best.pt", map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    cfg.update(
        {
            "aux_weight": 0.0,
            "core_aux_weight": 0.0,
            "hard_aux_weight": 0.0,
            "top14_aux_weight": 0.0,
            "atlas_distill_weight": 0.0,
            "neighbor_weight": 0.0,
            "rank_weight": 0.0,
            "supcon_weight": 0.0,
            "bad_aux_weight": 0.0,
            "bad_specificity_weight": 0.0,
            "stress_aux_weight": 0.0,
            "waveform_proto_weight": 0.0,
            "embedding_proto_weight": 0.0,
            "lr": float(args.lr),
            "seed": int(args.model_seed),
        }
    )
    torch.manual_seed(int(cfg["seed"]))
    synth_train = mod.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    norm = mod.ARCH.FeatureNorm(mean=synth_train.mean, std=synth_train.std)
    original = mod.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    if len(split_array) != len(original):
        raise RuntimeError(f"Split assignment rows {len(split_array)} != OriginalWaveDataset rows {len(original)}")
    train_ds = CapacitySplitDataset(original, split_array, "train")
    val_ds = CapacitySplitDataset(original, split_array, "val")
    test_ds = CapacitySplitDataset(original, split_array, "test")
    model = mod.GeometryStudent(cfg, int(synth_train.x.shape[1])).to(device)
    if args.init:
        model.load_state_dict(ckpt["model_state"], strict=False)
    counts = np.bincount(train_ds.y, minlength=3).astype(np.float32)
    class_weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    class_weight = class_weight / class_weight.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    logs: list[dict[str, Any]] = []
    best_state = None
    best_score = -1e9
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            out = model(x, mask_ratio=0.0)
            ce = F.cross_entropy(out["logits"], y, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(ce.detach().cpu()))
        val_probs = predict_probs(model, val_ds, device, batch_size * 2)
        val_rep = metric_report(val_ds.y, val_probs)
        score = val_rep["acc"] + 0.20 * val_rep["macro_f1"] + 0.10 * min(
            val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"]
        )
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"scheme={args.scheme} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_acc={val_rep['acc']:.4f} rec={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    tag = args.tag or f"{scheme_label}_{'init' if args.init else 'scratch'}_{args.epochs}ep"
    ckpt_out = AUDIT_DIR / f"but_capacity_{tag}_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "source_candidate": args.candidate,
            "initialized_from_source": bool(args.init),
            "epochs": int(args.epochs),
            "split_scheme": scheme_label,
            "candidate_split_seed": int(args.candidate_split_seed),
            "candidate_split_meta": candidate_split_meta,
            "input_contract": "BUT-only capacity diagnostic; waveform input only; not a formal PTB->BUT model claim",
        },
        ckpt_out,
    )
    atlas = load_atlas()
    rows = []
    pred_frames = []
    for name, ds in [("but_train", train_ds), ("but_val", val_ds), ("but_test", test_ds)]:
        probs = predict_probs(model, ds, device, batch_size * 2)
        pred_frames.append(split_prediction_frame(atlas, name, ds, probs))
        rep = metric_report(ds.y, probs)
        rep["bucket"] = name
        rep["confusion_3x3"] = json.dumps(rep["confusion_3x3"])
        rows.append(rep)
    metrics = pd.DataFrame(rows)
    metrics_path = AUDIT_DIR / f"but_capacity_{tag}_metrics.csv"
    pred_path = AUDIT_DIR / f"but_capacity_{tag}_predictions.csv"
    log_path = AUDIT_DIR / f"but_capacity_{tag}_train_log.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(logs).to_csv(log_path, index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(pred_path, index=False)
    report = [
        f"# BUT-Only Capacity Diagnostic: {tag}",
        "",
        "Diagnostic only. This uses BUT train rows and must not be mixed into PTB->BUT claims.",
        f"Split scheme: `{scheme_label}`.",
        "",
        md_table(metrics, max_rows=10),
        "",
        f"Metrics CSV: `{metrics_path}`",
        f"Predictions CSV: `{pred_path}`",
        f"Checkpoint: `{ckpt_out}`",
        "",
    ]
    report_path = AUDIT_REPORT_DIR / f"but_capacity_{tag}_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["split_audit", "feature_audit", "capacity", "all_audit"], default="all_audit")
    parser.add_argument("--seeds", type=int, default=20000)
    parser.add_argument("--window-seed", type=int, default=20260618)
    parser.add_argument("--synth-aux", type=str, default=str(DEFAULT_SYNTH_AUX))
    parser.add_argument("--scheme", type=str, default="balanced_best")
    parser.add_argument("--candidate-split-seed", type=int, default=-1)
    parser.add_argument("--candidate", type=str, default="featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--model-seed", type=int, default=20261010)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    if args.stage in {"split_audit", "all_audit"}:
        run_split_audit(args)
    if args.stage in {"feature_audit", "all_audit"}:
        run_feature_audit(args)
    if args.stage == "capacity":
        run_capacity(args)


if __name__ == "__main__":
    main()
