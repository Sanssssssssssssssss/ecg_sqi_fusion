"""Report-only complement audit for waveform Transformer endpoints.

This script evaluates existing waveform-only checkpoints on the held-out
Original BUT atlas and compares which rows each endpoint fixes or regresses.
It never trains, calibrates, or selects on Original BUT.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
RUNNER = ANALYSIS_DIR / "run_waveform_geometry_student.py"
NODE_ID = "N17043_gm_probe"
SEARCH_DIR = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search"
OUT_DIR = ANALYSIS_DIR / "waveform_endpoint_complement"

DEFAULT_CANDIDATES = [
    "featurefirst_top20_qrsbase_artbad_dualcoreout_strict_a050",
    "featurefirst_top20_qrsbase_artbad_dualcoreout_balanced_a050",
    "featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050",
    "featurefirst_top20_dualcoreout_mediumshell_headlight_a050",
    "featurefirst_top20_dualcoreout_mediumshell_balanced_a050",
    "featurefirst_top20_dualcoreout_mediumshell_goodguard_a050",
    "featurefirst_wavecomp_dualcoreout_encoderlite_baselinegm_a050",
    "featurefirst_wavecomp_dualcoreout_encoderlite_mediumguard_a050",
    "featurefirst_wavecomp_dualcoreout_encoderlite_goodguard_a050",
]


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metric_rows(run: Any, ds: Any, probs: np.ndarray, pred: np.ndarray, name: str, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
        mask = run.ARCH.bucket_mask(ds, bucket)
        rep = run.GEOM.metric_report(ds.y[mask], pred[mask], probs[mask])
        row = {"candidate": name, "mode": mode, "bucket": bucket}
        row.update({k: v for k, v in rep.items() if k != "confusion_3x3"})
        row["confusion_3x3"] = json.dumps(rep["confusion_3x3"])
        rows.append(row)
    return rows


def evaluate_candidate(run: Any, name: str, device: torch.device, batch_size: int) -> tuple[Any, dict[str, np.ndarray], pd.DataFrame]:
    ckpt_path = SEARCH_DIR / name / "ckpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt.get("candidate_config", run.CANDIDATES[name]))
    norm = run.ARCH.FeatureNorm(
        mean=np.asarray(ckpt["feature_train_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_train_std"], dtype=np.float32),
    )
    ds = run.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = run.GeometryStudent(cfg, int(ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"])
    payload = run.eval_loader(model, loader, device)
    raw_pred = payload.probs.argmax(axis=1).astype(np.int64)
    threshold = float(ckpt.get("bad_threshold_trainval", np.nan))
    if np.isfinite(threshold):
        badcal_pred = run.ARCH.apply_bad_threshold(payload.probs, threshold)
    else:
        badcal_pred = raw_pred
    preds = {"raw": raw_pred, "badcal": badcal_pred, "probs": payload.probs}
    rows = []
    rows.extend(metric_rows(run, ds, payload.probs, raw_pred, name, "raw"))
    rows.extend(metric_rows(run, ds, payload.probs, badcal_pred, name, "badcal"))
    return ds, preds, pd.DataFrame(rows)


def pair_complement_rows(
    y: np.ndarray,
    split: np.ndarray,
    region: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    buckets = {
        "original_test_all_10s+": split == "test",
        "original_all_10s+": np.ones_like(y, dtype=bool),
        "bad_core_nearboundary": (split == "test") & (y == 2) & (region != "outlier_low_confidence"),
        "bad_outlier_stress": (split == "test") & (y == 2) & (region == "outlier_low_confidence"),
    }
    for bucket, mask in buckets.items():
        a_ok = pred_a[mask] == y[mask]
        b_ok = pred_b[mask] == y[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        row = {
            "base": name_a,
            "challenger": name_b,
            "mode": mode,
            "bucket": bucket,
            "n": n,
            "base_acc": float(a_ok.mean()),
            "challenger_acc": float(b_ok.mean()),
            "oracle_acc": float((a_ok | b_ok).mean()),
            "challenger_fixes_base": int((~a_ok & b_ok).sum()),
            "challenger_regresses_base": int((a_ok & ~b_ok).sum()),
            "both_wrong": int((~a_ok & ~b_ok).sum()),
            "both_correct": int((a_ok & b_ok).sum()),
        }
        for cls_name, cls_id in [("good", 0), ("medium", 1), ("bad", 2)]:
            cls_mask = mask & (y == cls_id)
            if cls_mask.any():
                a_cls = pred_a[cls_mask] == y[cls_mask]
                b_cls = pred_b[cls_mask] == y[cls_mask]
                row[f"{cls_name}_oracle_recall"] = float((a_cls | b_cls).mean())
                row[f"{cls_name}_fixes"] = int((~a_cls & b_cls).sum())
                row[f"{cls_name}_regresses"] = int((a_cls & ~b_cls).sum())
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--base", type=str, default="featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    run = load_runner()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    names = [x for x in names if (SEARCH_DIR / x / "ckpt_best.pt").exists()]
    if args.base not in names:
        names.insert(0, args.base)

    metrics: list[pd.DataFrame] = []
    pred_map: dict[tuple[str, str], np.ndarray] = {}
    prob_map: dict[str, np.ndarray] = {}
    ds_ref = None
    for name in names:
        ds, preds, rows = evaluate_candidate(run, name, device, int(args.batch_size))
        ds_ref = ds
        metrics.append(rows)
        pred_map[(name, "raw")] = preds["raw"]
        pred_map[(name, "badcal")] = preds["badcal"]
        prob_map[name] = preds["probs"]
        print(f"evaluated {name}", flush=True)

    assert ds_ref is not None
    atlas = pd.read_csv(run.ARCH.ORIGINAL_ATLAS).reset_index(drop=True)
    base_cols = [
        "idx",
        "split",
        "class_name",
        "original_region",
    ]
    base_cols = [c for c in base_cols if c in atlas.columns]
    wide = atlas[base_cols].copy()
    y = np.asarray(ds_ref.y, dtype=np.int64)
    wide["y"] = y
    for name in names:
        for mode in ["raw", "badcal"]:
            pred = pred_map[(name, mode)]
            wide[f"{name}__{mode}_pred"] = pred
            wide[f"{name}__{mode}_correct"] = pred == y
        probs = prob_map[name]
        wide[f"{name}__p_good"] = probs[:, 0]
        wide[f"{name}__p_medium"] = probs[:, 1]
        wide[f"{name}__p_bad"] = probs[:, 2]
    wide.to_csv(OUT_DIR / "endpoint_original_predictions_wide.csv", index=False)

    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "endpoint_bucket_metrics.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    split = np.asarray(ds_ref.split)
    region = np.asarray(ds_ref.region)
    for mode in ["raw", "badcal"]:
        base_pred = pred_map[(args.base, mode)]
        for name in names:
            if name == args.base:
                continue
            pair_rows.extend(pair_complement_rows(y, split, region, base_pred, pred_map[(name, mode)], args.base, name, mode))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT_DIR / "endpoint_pair_complement.csv", index=False)

    summary = {
        "input_contract": "report-only endpoint audit; Original BUT is not used for training, calibration, or selection",
        "device": str(device),
        "candidates": names,
        "base": args.base,
        "metrics_csv": str(OUT_DIR / "endpoint_bucket_metrics.csv"),
        "pair_complement_csv": str(OUT_DIR / "endpoint_pair_complement.csv"),
        "predictions_csv": str(OUT_DIR / "endpoint_original_predictions_wide.csv"),
    }
    (OUT_DIR / "endpoint_complement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics_df[metrics_df["bucket"].eq("original_test_all_10s+")][["candidate", "mode", "acc", "good_recall", "medium_recall", "bad_recall"]])
    print(pairs[pairs["bucket"].eq("original_test_all_10s+")].sort_values("oracle_acc", ascending=False).head(12))


if __name__ == "__main__":
    main()
