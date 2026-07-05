from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.svm import LinearSVC

from src.sqi_pipeline.features.norm_record84_ks import run as run_norm_record84
from src.sqi_pipeline.features.sqi import (
    sqi_bSQI_li2008_global,
    sqi_basSQI,
    sqi_fSQI,
    sqi_kSQI,
    sqi_pSQI,
    sqi_sSQI,
)
from src.sqi_pipeline.qrs.detectors import run_gqrs, run_xqrs
from src.transformer_pipeline.data_v1_gapfill.common import protocol_dir, split_dir
from src.utils.paths import project_root


ROOT = project_root()
OUT_DEFAULT = ROOT / "outputs" / "transformer" / "supplemental" / "but_sqi_baseline"
FS = 125
BEAT_MATCH_TOL_MS = 150
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SINGLE_LEAD = "single"
SQI_ORDER = ["iSQI", "bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI"]
WELCH_KW = dict(fmax=40.0, window="hann", nperseg=256, noverlap=128, detrend="constant")
FLATLINE_EPS = 1e-4


@dataclass(frozen=True)
class Paths:
    out: Path

    @property
    def data(self) -> Path:
        return self.out / "data"

    @property
    def qrs(self) -> Path:
        return self.out / "qrs"

    @property
    def features(self) -> Path:
        return self.out / "features"

    @property
    def models(self) -> Path:
        return self.out / "models"

    @property
    def reports(self) -> Path:
        return self.out / "reports"

    @property
    def split_csv(self) -> Path:
        return self.data / "split_but_v116_good_vs_rest.csv"

    @property
    def manifest_csv(self) -> Path:
        return self.data / "manifest_but_v116_good_vs_rest.csv"

    @property
    def qrs_summary_csv(self) -> Path:
        return self.qrs / "qrs_summary_seed0.csv"

    @property
    def record84_parquet(self) -> Path:
        return self.features / "record84.parquet"

    @property
    def record7_parquet(self) -> Path:
        return self.features / "record7.parquet"

    @property
    def lead7_parquet(self) -> Path:
        return self.features / "lead7.parquet"

    @property
    def record84_norm_parquet(self) -> Path:
        return self.features / "record84_norm.parquet"

    @property
    def record7_norm_parquet(self) -> Path:
        return self.features / "record7_norm.parquet"

    @property
    def norm_stats_json(self) -> Path:
        return self.features / "norm_stats_seed0.json"

    @property
    def norm_stats_record7_json(self) -> Path:
        return self.features / "norm_stats_record7_seed0.json"

    @property
    def summary_json(self) -> Path:
        return self.reports / "but_sqi_baseline_summary.json"

    @property
    def summary_md(self) -> Path:
        return self.reports / "but_sqi_baseline_summary.md"


def ensure_dirs(paths: Paths) -> None:
    for path in [paths.data, paths.qrs, paths.features, paths.models, paths.reports]:
        path.mkdir(parents=True, exist_ok=True)


def dry(step: str, paths: Paths) -> None:
    print(
        json.dumps(
            {
                "step": step,
                "out": str(paths.out),
                "manifest": str(paths.manifest_csv),
                "split": str(paths.split_csv),
                "features": str(paths.record7_norm_parquet),
                "models": str(paths.models),
            },
            indent=2,
        )
    )


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "__" in c]


def _features_ready(paths: Paths) -> bool:
    required = [
        paths.split_csv,
        paths.qrs_summary_csv,
        paths.record84_parquet,
        paths.record84_norm_parquet,
        paths.record7_parquet,
        paths.record7_norm_parquet,
        paths.norm_stats_json,
        paths.norm_stats_record7_json,
    ]
    if any(not p.exists() for p in required):
        return False
    try:
        split_n = len(pd.read_csv(paths.split_csv, usecols=["record_id"]))
        qrs = pd.read_csv(paths.qrs_summary_csv, usecols=["record_id", "detector_profile"])
        rec84 = pd.read_parquet(paths.record84_norm_parquet)
        rec7 = pd.read_parquet(paths.record7_norm_parquet)
    except Exception:
        return False
    return (
        len(qrs) == split_n
        and len(rec84) == split_n
        and len(rec7) == split_n
        and len(_feature_cols(rec84)) == 84
        and len(_feature_cols(rec7)) == 7
        and set(qrs["detector_profile"].astype(str)) == {"wfdb_singlelead_li2008_proxy"}
    )


def cmd_prepare(paths: Paths, *, run: bool, force: bool) -> None:
    if not run:
        dry("prepare", paths)
        return
    ensure_dirs(paths)
    if paths.split_csv.exists() and paths.manifest_csv.exists() and not force:
        print(f"[prepare] exists: {paths.split_csv}")
        return

    split = pd.read_csv(split_dir() / "original_region_atlas.csv", low_memory=False)
    active = split.loc[split["split"].isin(["train", "val", "test"])].copy()
    active["idx"] = pd.to_numeric(active["idx"], errors="raise").astype(int)
    active = active.sort_values("idx").reset_index(drop=True)
    active["but_record_id"] = active["record_id"].astype(str)
    active["record_id"] = active["idx"].map(lambda x: f"butv116_{int(x):05d}")
    active["binary_task"] = "good_vs_medium_bad"
    active["quality_record"] = np.where(active["class_name"].eq("good"), "acceptable", "unacceptable")
    active["y"] = np.where(active["class_name"].eq("good"), 1, -1).astype(int)

    keep = [
        "record_id",
        "idx",
        "source_idx",
        "but_record_id",
        "split",
        "y",
        "quality_record",
        "class_name",
        "binary_task",
        "v116_candidate_type",
        "v116_generated",
        "record_id",
    ]
    keep = list(dict.fromkeys([c for c in keep if c in active.columns]))
    out = active[keep].copy()
    out.to_csv(paths.split_csv, index=False)
    out.to_csv(paths.manifest_csv, index=False)

    counts = {
        "rows": int(len(out)),
        "split_class_counts": out.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(),
        "binary_counts": out.groupby(["split", "quality_record"]).size().unstack(fill_value=0).to_dict(),
        "val_test_generated_rows": int(
            out.loc[out["split"].isin(["val", "test"]) & ~out["v116_candidate_type"].astype(str).eq("original_but")].shape[0]
        )
        if "v116_candidate_type" in out.columns
        else None,
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "prepare_audit.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


def _feature_names(leads: list[str] | None = None) -> list[str]:
    names: list[str] = []
    for lead in leads or LEADS_12:
        names.extend([f"{lead}__{sqi}" for sqi in SQI_ORDER])
    return names


def _compute_one(item: tuple[str, int, int, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    record_id, y, idx, x = item
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    tol_samp = int(round(BEAT_MATCH_TOL_MS * FS / 1000.0))
    status = "ok"
    err = ""
    try:
        r1 = run_xqrs(x, fs=FS)
    except Exception as exc:
        r1 = np.asarray([], dtype=int)
        status = "xqrs_failed"
        err = repr(exc)
    try:
        r2 = run_gqrs(x, fs=FS)
    except Exception as exc:
        r2 = np.asarray([], dtype=int)
        status = status + "+gqrs_failed" if status != "ok" else "gqrs_failed"
        err = (err + "; " if err else "") + repr(exc)

    b = float(sqi_bSQI_li2008_global(r1, r2, tol_samp))
    i = b
    p = float(sqi_pSQI(x, fs=FS, welch_kwargs=WELCH_KW))
    s = float(sqi_sSQI(x))
    k = float(sqi_kSQI(x))
    f = float(sqi_fSQI(x, flatline_eps=FLATLINE_EPS))
    bas = float(sqi_basSQI(x, fs=FS, welch_kwargs=WELCH_KW))

    vals: list[float] = []
    lead_rows: list[dict[str, Any]] = []
    for li, lead in enumerate(LEADS_12):
        row = {
            "record_id": record_id,
            "y": int(y),
            "lead": lead,
            "iSQI": i,
            "bSQI": b,
            "pSQI": p,
            "sSQI": s,
            "kSQI": k,
            "fSQI": f,
            "basSQI": bas,
        }
        lead_rows.append(row)
        vals.extend([row["iSQI"], b, p, s, k, f, bas])

    feat = {"record_id": record_id, "y": int(y)}
    feat.update({name: float(value) if math.isfinite(float(value)) else np.nan for name, value in zip(_feature_names(), vals)})
    single = {"record_id": record_id, "y": int(y)}
    single.update(
        {
            name: float(value) if math.isfinite(float(value)) else np.nan
            for name, value in zip(_feature_names([SINGLE_LEAD]), [i, b, p, s, k, f, bas])
        }
    )
    qrs = {
        "record_id": record_id,
        "idx": int(idx),
        "n_xqrs": int(len(r1)),
        "n_gqrs": int(len(r2)),
        "detector_profile": "wfdb_singlelead_li2008_proxy",
        "beat_match_tol_ms": int(BEAT_MATCH_TOL_MS),
        "status": status,
        "error": err,
    }
    return feat, single, lead_rows, qrs


def cmd_features(paths: Paths, *, run: bool, force: bool, jobs: int) -> None:
    if not run:
        dry("features", paths)
        return
    ensure_dirs(paths)
    if _features_ready(paths) and not force:
        print(f"[features] exists: {paths.record7_norm_parquet}")
        return
    if not paths.split_csv.exists():
        raise SystemExit(f"missing split: {paths.split_csv}")

    split = pd.read_csv(paths.split_csv)
    protocol_npz = np.load(protocol_dir() / "signals.npz", allow_pickle=True)
    X = np.asarray(protocol_npz["X"], dtype=np.float32)
    items = [
        (str(row.record_id), int(row.y), int(row.idx), X[int(row.idx)])
        for row in split[["record_id", "y", "idx"]].itertuples(index=False)
    ]

    t0 = time.time()
    feat_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    qrs_rows: list[dict[str, Any]] = []
    workers = max(1, int(jobs))
    print(f"[features] records={len(items)} jobs={workers}")
    if workers == 1:
        iterator = map(_compute_one, items)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_compute_one, items, chunksize=32)
    try:
        for i, (feat, single, leads, qrs) in enumerate(iterator, start=1):
            feat_rows.append(feat)
            single_rows.append(single)
            lead_rows.extend(leads)
            qrs_rows.append(qrs)
            if i <= 5 or i % 500 == 0:
                print(f"[features] {i}/{len(items)}")
    finally:
        if workers != 1:
            pool.shutdown(wait=True, cancel_futures=False)

    pd.DataFrame(feat_rows).to_parquet(paths.record84_parquet, index=False)
    pd.DataFrame(single_rows).to_parquet(paths.record7_parquet, index=False)
    pd.DataFrame(lead_rows).to_parquet(paths.lead7_parquet, index=False)
    pd.DataFrame(qrs_rows).to_csv(paths.qrs_summary_csv, index=False)
    print(f"[features] wrote {paths.record84_parquet} rows={len(feat_rows)} time_s={time.time() - t0:.1f}")

    run_norm_record84(
        {
            "force": True,
            "seed": 0,
            "split_csv": str(paths.split_csv),
            "in_parquet": str(paths.record84_parquet),
            "out_dir": str(paths.features),
            "out_stats": str(paths.norm_stats_json),
            "out_parquet": str(paths.record84_norm_parquet),
        }
    )
    run_norm_record84(
        {
            "force": True,
            "seed": 0,
            "split_csv": str(paths.split_csv),
            "in_parquet": str(paths.record7_parquet),
            "out_dir": str(paths.features),
            "out_stats": str(paths.norm_stats_record7_json),
            "out_parquet": str(paths.record7_norm_parquet),
        }
    )


def macro_f1_from_cm(tn: int, fp: int, fn: int, tp: int) -> float:
    f1_bad = 0.0 if (2 * tn + fn + fp) == 0 else 2 * tn / (2 * tn + fn + fp)
    f1_good = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    return float((f1_bad + f1_good) / 2)


def _metrics_from_scores(y01: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y01 = np.asarray(y01, dtype=int).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    pred = (score > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y01, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    try:
        auc = float(roc_auc_score(y01, score))
    except Exception:
        auc = float("nan")
    acceptable_recall = float(tp / max(1, tp + fn))
    unacceptable_recall = float(tn / max(1, tn + fp))
    return {
        "acc": float(acc),
        "macro_f1": macro_f1_from_cm(int(tn), int(fp), int(fn), int(tp)),
        "sensitivity": unacceptable_recall,
        "specificity": acceptable_recall,
        "acceptable_recall": acceptable_recall,
        "unacceptable_recall": unacceptable_recall,
        "auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _best_threshold(y01: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    y01 = np.asarray(y01, dtype=int).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    lo, hi = np.quantile(score, [0.001, 0.999])
    grid = np.linspace(float(lo), float(hi), 2001)
    best: dict[str, Any] | None = None
    for t in grid:
        met = _metrics_from_scores(y01, score, float(t))
        row = {"threshold": float(t), **met}
        if best is None or row["acc"] > best["acc"] or (row["acc"] == best["acc"] and row["macro_f1"] > best["macro_f1"]):
            best = row
    if best is None:
        raise RuntimeError("empty validation threshold grid")
    return best


def _run_linear_svm(paths: Paths, *, force: bool) -> None:
    out_dir = paths.models / "svm_single7_linear"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "svm_single7_linear_metrics_seed0.json"
    pred_path = out_dir / "svm_single7_linear_predictions_seed0.csv"
    if metrics_path.exists() and pred_path.exists() and not force:
        print(f"[train] exists: {metrics_path}")
        return

    df = pd.read_parquet(paths.record7_norm_parquet)
    split = pd.read_csv(paths.split_csv)[["record_id", "split"]]
    df = df.merge(split, on="record_id", how="inner")
    feature_cols = [c for c in df.columns if "__" in c]
    if len(feature_cols) != 7:
        raise ValueError(f"expected 7 single-lead SQI features, found {len(feature_cols)}")
    df["y01"] = (df["y"].astype(int) == 1).astype(int)

    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("val").to_numpy()
    te = df["split"].eq("test").to_numpy()
    Xtr = df.loc[tr, feature_cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, feature_cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, feature_cols].to_numpy(dtype=np.float64)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)

    best: dict[str, Any] | None = None
    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        model = LinearSVC(C=float(c), dual=False, max_iter=20000, random_state=0)
        t0 = time.time()
        model.fit(Xtr, ytr)
        train_s = float(time.time() - t0)
        val_score = model.decision_function(Xva).astype(np.float64)
        thr = _best_threshold(yva, val_score)
        row = {
            "C": float(c),
            "train_time_s": train_s,
            "model": model,
            "threshold": float(thr["threshold"]),
            "val": thr,
        }
        if best is None or row["val"]["acc"] > best["val"]["acc"] or (
            row["val"]["acc"] == best["val"]["acc"] and row["val"]["auc"] > best["val"]["auc"]
        ):
            best = row
        print(f"[svm] C={c:g} val_acc={thr['acc']:.4f} val_auc={thr['auc']:.4f} thr={thr['threshold']:.4f}")
    if best is None:
        raise RuntimeError("SVM search produced no model")

    model = best["model"]
    threshold = float(best["threshold"])
    tr_score = model.decision_function(Xtr).astype(np.float64)
    va_score = model.decision_function(Xva).astype(np.float64)
    te_score = model.decision_function(Xte).astype(np.float64)
    metrics = {
        "model": "LinearSVC_single_lead_7_SQI",
        "note": "BUT v116 single-lead Li2008-compatible SQI baseline. iSQI is a detector-agreement proxy; no pseudo-12 lead duplication is used.",
        "feature_path": str(paths.record7_norm_parquet),
        "split_path": str(paths.split_csv),
        "n_features": int(len(feature_cols)),
        "best_C": float(best["C"]),
        "threshold": threshold,
        "train": _metrics_from_scores(ytr, tr_score, threshold),
        "val": _metrics_from_scores(yva, va_score, threshold),
        "test": _metrics_from_scores(yte, te_score, threshold),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pred = pd.DataFrame(
        {
            "record_id": df.loc[te, "record_id"].astype(str).to_numpy(),
            "y": yte,
            "score": te_score,
            "pred": (te_score > threshold).astype(int),
        }
    )
    pred.to_csv(pred_path, index=False)
    print(json.dumps(metrics["test"], indent=2))


def cmd_train(paths: Paths, *, run: bool, force: bool, device: str) -> None:
    if not run:
        dry("train", paths)
        return
    ensure_dirs(paths)
    if not paths.record7_norm_parquet.exists():
        raise SystemExit(f"missing normalized features: {paths.record7_norm_parquet}")

    _run_linear_svm(paths, force=force)


def cmd_summary(paths: Paths) -> None:
    split = pd.read_csv(paths.split_csv)
    svm_out = paths.models / "svm_single7_linear"
    svm = json.loads((svm_out / "svm_single7_linear_metrics_seed0.json").read_text(encoding="utf-8"))

    qrs = pd.read_csv(paths.qrs_summary_csv)
    summary = {
        "task": "BUT v116 good-vs-medium/bad classical SQI baselines",
        "note": "BUT is single-lead; the active baseline uses seven single-lead SQIs. The legacy pseudo-12 record84 artifact is kept only for historical comparison.",
        "source_protocol": str(protocol_dir()),
        "source_split": str(split_dir()),
        "features": str(paths.record7_norm_parquet),
        "split_counts_class": split.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(),
        "split_counts_binary": split.groupby(["split", "quality_record"]).size().unstack(fill_value=0).to_dict(),
        "qrs_status_counts": qrs["status"].value_counts(dropna=False).astype(int).to_dict(),
        "svm_single7_linear": {
            "acc": float(svm["test"]["acc"]),
            "macro_f1": float(svm["test"]["macro_f1"]),
            "auc": float(svm["test"]["auc"]),
            "threshold": float(svm["threshold"]),
            "best_C": float(svm["best_C"]),
        },
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# BUT v116 SQI Baseline Comparator",
        "",
        f"- Linear SVM single-lead 7-SQI: acc {summary['svm_single7_linear']['acc']:.4f}, macro F1 {summary['svm_single7_linear']['macro_f1']:.4f}, AUC {summary['svm_single7_linear']['auc']:.4f}",
        "",
        "Binary mapping: good=1, medium/bad=-1. Val/test are original BUT only from the official v116 split.",
    ]
    paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_pipeline(paths: Paths, args: argparse.Namespace) -> None:
    if not args.run:
        dry("pipeline", paths)
        print(
            subprocess.list2cmdline(
                [sys.executable, "-m", "src.supplemental_transformer_experiments.but_sqi_baseline.run", "pipeline", "--run"]
            )
        )
        return
    cmd_prepare(paths, run=True, force=args.force)
    cmd_features(paths, run=True, force=args.force, jobs=args.jobs)
    cmd_train(paths, run=True, force=args.force, device=args.device)
    cmd_summary(paths)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BUT v116 classical SQI baseline control.")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    p.add_argument("--device", default="cuda")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "features", "train"]:
        sp = sub.add_parser(name)
        sp.add_argument("--run", action="store_true")
    sub.add_parser("summary")
    pipe = sub.add_parser("pipeline")
    pipe.add_argument("--run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    paths = Paths(out if out.is_absolute() else ROOT / out)
    if args.cmd == "prepare":
        cmd_prepare(paths, run=args.run, force=args.force)
    elif args.cmd == "features":
        cmd_features(paths, run=args.run, force=args.force, jobs=args.jobs)
    elif args.cmd == "train":
        cmd_train(paths, run=args.run, force=args.force, device=args.device)
    elif args.cmd == "summary":
        cmd_summary(paths)
    elif args.cmd == "pipeline":
        cmd_pipeline(paths, args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
