from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from src.transformer_pipeline.config import TransformerPipelineConfig


OLD_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OLD_P1_TAG = "e311_but_protocol_adaptation_2026_06_03"
SUPPORT_POLICY = "margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed"
PTB_CARRIER_POLICY = "raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680"
FS_TARGET = 125
WIN_SEC = 10
N_TARGET = FS_TARGET * WIN_SEC
PTB_RMS_TARGET = 0.1616995930671692
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CLASS_MAP = {"1": "good", "2": "medium", "3": "bad", 1: "good", 2: "medium", 3: "bad"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_digest(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": h.hexdigest()}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def archive_root(root: Path) -> Path:
    if os.environ.get("ECG_DISABLE_LOCAL_ARCHIVE") == "1":
        return root / "__local_archive_disabled__"
    # CleanBUT PCA/kNN support artifacts predate the current cleaned source tree.
    # Keep them explicit: raw BUT is rebuilt here, support assets are restored
    # only to preserve the historical v116 data distribution exactly.
    return root.parent / "ecg_keep_20260528_172844" / "outputs_archive" / "external_benchmarks"


def long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def robust_normalize_window(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(x, dtype=np.float64).reshape(-1)
    centered = raw - float(np.median(raw))
    rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
    p05, p95 = np.percentile(centered, [5.0, 95.0])
    robust = float((p95 - p05) / 3.29) if p95 > p05 else rms
    scale = max(robust, rms * 0.5, 1e-6)
    y = centered * (PTB_RMS_TARGET / scale)
    return y.astype(np.float32), {
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "raw_rms_centered": rms,
        "raw_p05": float(p05),
        "raw_p95": float(p95),
        "normalization_scale": float(scale),
    }


def resample_to_target(x: np.ndarray, fs: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if fs == FS_TARGET:
        return arr.astype(np.float32)
    g = math.gcd(int(fs), FS_TARGET)
    return resample_poly(arr, up=FS_TARGET // g, down=int(fs) // g).astype(np.float32)


def pad_or_trim_10s(x: np.ndarray, fs: int) -> tuple[np.ndarray, bool]:
    target = int(fs * WIN_SEC)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    padded = False
    if len(arr) < target:
        arr = np.pad(arr, (0, target - len(arr)), mode="reflect" if len(arr) > 2 else "edge")
        padded = True
    elif len(arr) > target:
        arr = arr[:target]
    return arr.astype(np.float32), padded


def parse_but_ann(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(path, header=None)
    numeric = ann.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric.shape[1] < 12:
        raise ValueError(f"BUT annotation should have at least 12 columns: {path}")
    cons = numeric.iloc[:, -3:].copy()
    cons.columns = ["start_sample", "end_sample", "quality_class"]
    cons = cons.dropna()
    cons = cons[(cons["quality_class"] >= 1) & (cons["quality_class"] <= 3)]
    return cons.astype({"start_sample": int, "end_sample": int, "quality_class": int})


def split_counts(meta: pd.DataFrame, label_col: str = "y_class") -> dict[str, Any]:
    return {
        "split_counts": {str(k): int(v) for k, v in meta["split"].value_counts().sort_index().items()},
        "label_counts": {str(k): int(v) for k, v in meta[label_col].value_counts().sort_index().items()},
        "split_label_counts": {
            str(split): {str(k): int(v) for k, v in group[label_col].value_counts().sort_index().items()}
            for split, group in meta.groupby("split")
        },
    }


def balanced_but_subject_split(meta: pd.DataFrame, seed: int, attempts: int = 20000) -> tuple[dict[str, str], dict[str, Any]]:
    subjects = sorted(meta["subject_id"].astype(str).unique().tolist())
    counts = meta.groupby(["subject_id", "y_class"]).size().unstack(fill_value=0).reindex(subjects).fillna(0).astype(int)
    for cls in ["good", "medium", "bad"]:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[["good", "medium", "bad"]]
    total = counts.sum(axis=0).to_numpy(dtype=np.float64)
    n_train = max(1, int(round(0.60 * len(subjects))))
    n_val = max(1, int(round(0.20 * len(subjects))))
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, str], dict[str, Any]] | None = None
    for attempt in range(int(attempts)):
        order = subjects.copy()
        rng.shuffle(order)
        assign = {s: "train" for s in order[:n_train]}
        assign.update({s: "val" for s in order[n_train : n_train + n_val]})
        assign.update({s: "test" for s in order[n_train + n_val :]})
        split_counts_by_class: dict[str, dict[str, int]] = {}
        missing_penalty = 0.0
        balance_penalty = 0.0
        for split_name, frac in [("train", 0.60), ("val", 0.20), ("test", 0.20)]:
            vals = counts.loc[[s for s in subjects if assign[s] == split_name]].sum(axis=0)
            split_counts_by_class[split_name] = {str(k): int(v) for k, v in vals.to_dict().items()}
            missing_penalty += 1000.0 * float((vals <= 0).sum())
            balance_penalty += float(np.sum(np.abs(vals.to_numpy(dtype=np.float64) - frac * total) / (total + 1e-6)))
        score = missing_penalty + balance_penalty
        audit = {"attempt": int(attempt), "split_label_counts": split_counts_by_class, "score": float(score)}
        if best is None or score < best[0]:
            best = (score, assign, audit)
            if missing_penalty == 0.0 and balance_penalty < 1.0:
                break
    assert best is not None
    return best[1], best[2]


def preprocess_butqdb(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    raw_dir = cfg.butqdb_root
    out_dir = cfg.artifacts_dir / "source" / "processed_butqdb"
    signals_path = out_dir / "signals.npz"
    meta_path = out_dir / "metadata.csv"
    audit_path = out_dir / "preprocess_audit.json"
    if signals_path.exists() and meta_path.exists() and audit_path.exists() and not cfg.force:
        return read_json(audit_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    dropped = {"boundary_or_short": 0, "read_error": 0}
    record_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir() and (p / f"{p.name}_ECG.hea").exists()])
    for record_dir in record_dirs:
        record_id = record_dir.name
        subject_id = record_id[:3]
        ann_path = record_dir / f"{record_id}_ANN.csv"
        if not ann_path.exists():
            continue
        header = wfdb.rdheader(str(record_dir / f"{record_id}_ECG"))
        fs_raw = int(round(float(header.fs)))
        win_raw = fs_raw * WIN_SEC
        for seg in parse_but_ann(ann_path).itertuples(index=False):
            start, end, q = int(seg.start_sample), int(seg.end_sample), int(seg.quality_class)
            if end - start < win_raw:
                dropped["boundary_or_short"] += 1
                continue
            for k in range((end - start) // win_raw):
                s0 = start + k * win_raw
                s1 = s0 + win_raw
                try:
                    rec = wfdb.rdrecord(str(record_dir / f"{record_id}_ECG"), sampfrom=s0, sampto=s1, physical=True, channels=[0])
                except Exception:
                    dropped["read_error"] += 1
                    continue
                norm, stats = robust_normalize_window(resample_to_target(np.asarray(rec.p_signal[:, 0], dtype=np.float32), fs_raw))
                if len(norm) != N_TARGET:
                    norm = pad_or_trim_10s(norm, FS_TARGET)[0]
                label = CLASS_MAP[q]
                rows.append(norm.reshape(1, -1))
                meta_rows.append(
                    {
                        "window_id": f"butqdb_{record_id}_{s0}_{s1}",
                        "dataset": "butqdb",
                        "record_id": record_id,
                        "subject_id": subject_id,
                        "split": "pending",
                        "start_sec": float(s0 / fs_raw),
                        "end_sec": float(s1 / fs_raw),
                        "fs_original": fs_raw,
                        "label_raw": q,
                        "y_class": label,
                        "y": CLASS_TO_INT[label],
                        "source_label_policy": "BUT consensus class 1/2/3 mapped to good/medium/bad",
                        "padded": False,
                        **stats,
                    }
                )
    if not rows:
        raise RuntimeError(f"No BUT QDB windows were generated from {raw_dir}")
    x = np.stack(rows, axis=0).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    subject_split, split_strategy = balanced_but_subject_split(meta, cfg.seed)
    meta["split"] = meta["subject_id"].astype(str).map(subject_split)
    np.savez_compressed(signals_path, X=x)
    meta.to_csv(meta_path, index=False)
    audit = {
        "dataset": "butqdb",
        "signals_shape": list(x.shape),
        "dropped": dropped,
        "subject_split_strategy": split_strategy,
        **split_counts(meta),
    }
    write_json(audit_path, audit)
    return audit


def materialize_p1(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    processed = cfg.artifacts_dir / "source" / "processed_butqdb"
    out_dir = cfg.artifacts_dir / "source" / "p1_current_10s_center"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(processed / "signals.npz", out_dir / "signals.npz")
    meta = pd.read_csv(processed / "metadata.csv")
    meta["protocol"] = "p1_current_10s_center"
    meta["protocol_description"] = "Current 10s consensus-window protocol."
    meta["label_purity"] = 1.0
    meta.to_csv(out_dir / "metadata.csv", index=False)
    audit = {"protocol": "p1_current_10s_center", "kind": "current_10s", **split_counts(meta)}
    write_json(out_dir / "protocol_audit.json", audit)
    return audit


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
        for seg_idx, row in enumerate(arr):
            if pd.isna(row[9]) or pd.isna(row[10]) or pd.isna(row[11]):
                continue
            start_ms, end_ms = float(row[9]), float(row[10])
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
                match = candidates.sort_values("segment_duration_sec").iloc[0]
        if match is None:
            attached.append({"matched_segment": False})
            continue
        seg_start, seg_end = float(match["segment_start_sec"]), float(match["segment_end_sec"])
        seg_dur = float(match["segment_duration_sec"])
        win_center = 0.5 * (float(row.start_sec) + float(row.end_sec))
        attached.append(
            {
                "matched_segment": True,
                "segment_idx": int(match["segment_idx"]),
                "segment_start_sec": seg_start,
                "segment_end_sec": seg_end,
                "segment_duration_sec": seg_dur,
                "left_margin_sec": float(row.start_sec) - seg_start,
                "right_margin_sec": seg_end - float(row.end_sec),
                "min_margin_sec": min(float(row.start_sec) - seg_start, seg_end - float(row.end_sec)),
                "window_center_fraction": (win_center - seg_start) / max(seg_dur, 1e-9),
            }
        )
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(attached)], axis=1)


def ensure_original_region_atlas(cfg: TransformerPipelineConfig) -> Path:
    out = cfg.analysis_dir / "original_region_boundary" / "original_region_atlas.csv"
    if out.exists() and not cfg.force:
        return out
    archived = archive_root(cfg.root) / OLD_TAG / "original_region_boundary" / "original_region_atlas.csv"
    if archived.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archived, out)
        write_json(out.with_suffix(".recovery.json"), {"source": str(archived), "method": "restored_from_local_archive"})
        return out
    meta_path = cfg.artifacts_dir / "source" / "p1_current_10s_center" / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing original region atlas and metadata fallback: {archived}")
    meta = pd.read_csv(meta_path)
    atlas = meta[["y", "y_class", "record_id", "subject_id"]].copy()
    atlas.insert(0, "idx", np.arange(len(atlas), dtype=int))
    atlas["class_name"] = atlas["y_class"].astype(str)
    atlas["original_region"] = "public_rebuild_unknown_region"
    atlas["ambiguous_type"] = "public_rebuild_unknown_region"
    atlas["is_clean_strict"] = False
    atlas["region_confidence"] = 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(out, index=False)
    write_json(
        out.with_suffix(".recovery.json"),
        {
            "source": str(meta_path),
            "method": "public_rebuild_minimal_region_atlas",
            "warning": "historical region labels unavailable; distribution-sensitive v116 numbers require frozen support audit",
        },
    )
    return out


def build_margin_table(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    p1 = cfg.artifacts_dir / "source" / "p1_current_10s_center"
    out_dir = cfg.analysis_dir / "clean_but_window_policy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "current_10s_window_segment_margins.csv"
    meta = pd.read_csv(p1 / "metadata.csv")
    segments = read_consensus_segments(cfg.butqdb_root)
    margin = attach_segments(meta, segments)
    atlas = pd.read_csv(ensure_original_region_atlas(cfg))
    keep = [c for c in ["idx", "original_region", "ambiguous_type", "is_clean_strict", "region_confidence"] if c in atlas.columns]
    atlas = atlas[keep].copy()
    atlas["idx"] = atlas["idx"].astype(int)
    margin["idx"] = np.arange(len(margin), dtype=int)
    margin = margin.merge(atlas, on="idx", how="left")
    margin.to_csv(out, index=False)
    summary = {
        "rows": int(len(margin)),
        "matched_rows": int(margin["matched_segment"].fillna(False).sum()),
        "path": str(out),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


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
    CleanPolicy("margin_ge_2s_keep_outlier", min_margin_sec=2.0),
    CleanPolicy("margin_ge_5s_keep_outlier", min_margin_sec=5.0),
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


def materialize_policy(cfg: TransformerPipelineConfig, policy: CleanPolicy) -> dict[str, Any]:
    p1 = cfg.artifacts_dir / "source" / "p1_current_10s_center"
    source_x = np.load(p1 / "signals.npz")["X"].astype(np.float32)
    source_meta = pd.read_csv(p1 / "metadata.csv")
    source_atlas = pd.read_csv(ensure_original_region_atlas(cfg))
    margin = pd.read_csv(cfg.analysis_dir / "clean_but_window_policy" / "current_10s_window_segment_margins.csv")
    idx = np.flatnonzero(policy_mask(margin, policy)).astype(int)
    out_dir = cfg.analysis_dir / "clean_but_protocols" / policy.name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = source_meta.iloc[idx].copy().reset_index(drop=True)
    meta.insert(0, "source_idx", idx)
    meta["idx"] = np.arange(len(meta), dtype=int)
    meta["clean_policy"] = policy.name
    meta["clean_min_margin_sec"] = float(policy.min_margin_sec)
    meta["source_protocol"] = "p1_current_10s_center"
    atlas = source_atlas.set_index("idx").loc[idx].reset_index(drop=False).rename(columns={"idx": "source_idx"})
    atlas.insert(0, "idx", np.arange(len(atlas), dtype=int))
    atlas["clean_policy"] = policy.name
    atlas["clean_min_margin_sec"] = float(policy.min_margin_sec)
    margin_out = margin.iloc[idx].copy().reset_index(drop=True)
    margin_out.insert(0, "source_idx", idx)
    np.savez_compressed(out_dir / "signals.npz", X=source_x[idx].astype(np.float32))
    meta.to_csv(out_dir / "metadata.csv", index=False)
    atlas.to_csv(out_dir / "original_region_atlas.csv", index=False)
    margin_out.to_csv(out_dir / "window_segment_margins.csv", index=False)
    audit = {
        "policy": policy.name,
        "n": int(len(meta)),
        "class_counts": {str(k): int(v) for k, v in meta["y_class"].value_counts().sort_index().items()},
        "record_count": int(meta["record_id"].nunique()) if len(meta) else 0,
    }
    write_json(out_dir / "audit.json", audit)
    return audit


def build_clean_protocols(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    audits = [materialize_policy(cfg, policy) for policy in POLICIES]
    summary = {"policies": audits}
    out = cfg.analysis_dir / "clean_but_protocols" / "clean_but_protocols_summary.json"
    write_json(out, summary)
    return summary


def copy_tree(src: Path, dst: Path, *, force: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and force:
        shutil.rmtree(long_path(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(long_path(src), long_path(dst), dirs_exist_ok=True)
    return True


def ensure_cleanbut_support_assets(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    """Restore historical support pools required by the v116 generator.

    The v116 line was tuned against a CleanBUT PCA/kNN support pool and a raw
    PTB carrier pool. Recomputing those old support pools with simplified code
    changes candidate ordering, so the final reproducible line restores the
    exact support assets and then audits the final protocol independently.
    """
    archive = archive_root(cfg.root) / OLD_TAG / "analysis" / "good_medium_geometry_repair"
    pairs = [
        (
            archive / "clean_but_protocols" / SUPPORT_POLICY,
            cfg.analysis_dir / "clean_but_protocols" / SUPPORT_POLICY,
        ),
        (
            archive / "raw_ptbxl_carrier_protocols" / PTB_CARRIER_POLICY,
            cfg.analysis_dir / "raw_ptbxl_carrier_protocols" / PTB_CARRIER_POLICY,
        ),
    ]
    copied = []
    missing = []
    for src, dst in pairs:
        if dst.exists() and not cfg.force:
            copied.append({"src": str(dst), "dst": str(dst), "copied": False, "method": "existing_local"})
            continue
        ok = copy_tree(src, dst, force=cfg.force)
        if ok:
            copied.append({"src": str(src), "dst": str(dst), "copied": ok, "method": "restored_from_local_archive"})
            continue
        if dst.name == SUPPORT_POLICY:
            fallback = cfg.analysis_dir / "clean_but_protocols" / "margin_ge_5s_keep_outlier"
            ok = copy_tree(fallback, dst, force=cfg.force)
            if ok:
                copied.append({"src": str(fallback), "dst": str(dst), "copied": ok, "method": "public_but_fallback_not_historical"})
                continue
        if dst.name == PTB_CARRIER_POLICY:
            default_built = cfg.root / "outputs" / "transformer" / "v116_e31" / "analysis" / "good_medium_geometry_repair" / "raw_ptbxl_carrier_protocols" / PTB_CARRIER_POLICY
            if default_built.exists():
                built = default_built
            else:
                from src.transformer_pipeline.data_v1_gapfill.support import build_raw_ptbxl_carrier_protocol as ptb_builder

                built = ptb_builder.build(
                    argparse.Namespace(
                        name=PTB_CARRIER_POLICY,
                        lead="I",
                        max_records=9000,
                        seed=20260679,
                        clean_noise_notes=True,
                    )
                )
            ok = copy_tree(built, dst, force=cfg.force) if built.resolve() != dst.resolve() else dst.exists()
            if ok:
                copied.append({"src": str(built), "dst": str(dst), "copied": built.resolve() != dst.resolve(), "method": "public_ptbxl_builder"})
                continue
        missing.append({"expected_archive": str(src), "dst": str(dst)})
    payload = {"assets": copied, "missing": missing}
    write_json(cfg.artifacts_dir / "source" / "cleanbut_support_assets.json", payload)
    if missing:
        raise FileNotFoundError(f"Missing CleanBUT support assets after public-data fallback: {missing}")
    return payload


def audit_source(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    protocol = cfg.analysis_dir / "clean_but_protocols" / "margin_ge_5s_drop_outlier" / "original_region_atlas.csv"
    support = (
        cfg.analysis_dir
        / "clean_but_protocols"
        / SUPPORT_POLICY
        / "original_region_atlas.csv"
    )
    atlas = pd.read_csv(protocol)
    support_atlas = pd.read_csv(support)
    out = {
        "candidate_gap5_rows": int(len(atlas)),
        "candidate_gap5_class_counts": atlas["class_name"].value_counts().sort_index().astype(int).to_dict(),
        "support_pool_rows": int(len(support_atlas)),
        "support_pool_class_counts": support_atlas["class_name"].value_counts().sort_index().astype(int).to_dict(),
        "path": str(protocol),
        "support_path": str(support),
    }
    expected_support = {"bad": 2391, "good": 15042, "medium": 9212}
    out["historical_support_exact"] = out["support_pool_class_counts"] == expected_support and out["support_pool_rows"] == 26645
    if not out["historical_support_exact"]:
        out["historical_expected_support"] = expected_support
        out["support_warning"] = "public-data fallback support differs from historical v116 support"
    if not out["historical_support_exact"] and not (cfg.artifacts_dir / "source" / "cleanbut_support_assets.json").exists():
        raise SystemExit(f"CleanBUT support audit failed: {out}")
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


def clean_smoke(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    processed = preprocess_butqdb(cfg)
    p1 = materialize_p1(cfg)
    margins = build_margin_table(cfg)
    clean = build_clean_protocols(cfg)
    support = ensure_cleanbut_support_assets(cfg)
    source = audit_source(cfg)
    source_dir = cfg.artifacts_dir / "source"
    artifacts = {
        "processed_metadata": file_digest(source_dir / "processed_butqdb" / "metadata.csv"),
        "processed_signals": file_digest(source_dir / "processed_butqdb" / "signals.npz"),
        "p1_metadata": file_digest(source_dir / "p1_current_10s_center" / "metadata.csv"),
        "p1_signals": file_digest(source_dir / "p1_current_10s_center" / "signals.npz"),
        "candidate_atlas": file_digest(Path(source["path"])),
        "support_atlas": file_digest(Path(source["support_path"])),
    }
    out = {
        "reproduction_scope": "public-data smoke; exact frozen replay only when historical_support_exact=true",
        "processed": processed,
        "p1": p1,
        "margins": margins,
        "clean": clean,
        "support": support,
        "source": source,
        "artifacts": artifacts,
    }
    write_json(cfg.artifacts_dir / "source" / "clean_smoke_summary.json", out)
    return out


def run(cfg: TransformerPipelineConfig) -> dict[str, Any]:
    print(f"{now()} preprocess BUT QDB")
    processed = preprocess_butqdb(cfg)
    print(f"{now()} materialize p1")
    p1 = materialize_p1(cfg)
    print(f"{now()} build margin table")
    margins = build_margin_table(cfg)
    print(f"{now()} build clean protocols")
    clean = build_clean_protocols(cfg)
    print(f"{now()} ensure CleanBUT support assets")
    support = ensure_cleanbut_support_assets(cfg)
    source = audit_source(cfg)
    return {"processed": processed, "p1": p1, "margins": margins, "clean": clean, "support": support, "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BUT source assets for transformer v116.")
    parser.add_argument("--artifacts-dir", default="outputs/transformer/v116_e31")
    parser.add_argument("--seed", type=int, default=20260876)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = TransformerPipelineConfig.build(artifacts_dir=args.artifacts_dir, seed=args.seed, force=args.force)
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
