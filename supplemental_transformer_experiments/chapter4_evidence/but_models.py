from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi

from .common import Paths, binary_metrics, dry, ensure_dirs, multiclass_summary, read_json, write_json


def _ensure_but_sqi(paths: Paths, *, force: bool, device: str) -> dict[str, Any]:
    bp = but_sqi.Paths(paths.but / "sqi_baseline")
    if force or not bp.summary_json.exists():
        but_sqi.cmd_prepare(bp, run=True, force=force)
        but_sqi.cmd_features(bp, run=True, force=force, jobs=4)
        but_sqi.cmd_train(bp, run=True, force=force, device=device)
        but_sqi.cmd_summary(bp)
    return json.loads(bp.summary_json.read_text(encoding="utf-8"))


def _find_e31_predictions() -> Path:
    roots = list((Path("outputs") / "transformer" / "v116_e31" / "runs").glob("**/E31_wave_mechanism_conformer_fold0_seed0/test_predictions.npz"))
    if not roots:
        raise FileNotFoundError("missing E31 v116 test_predictions.npz")
    return roots[0].resolve()


def _e31_summary() -> dict[str, Any]:
    pred_path = _find_e31_predictions()
    z = np.load(pred_path, allow_pickle=True)
    y = z["y"].astype(int)
    probs = z["probs"].astype(np.float64)
    pred = probs.argmax(axis=1)
    multi = multiclass_summary(y, probs, ["good", "intermediate", "poor"])
    poor_fp = int(((y != 2) & (pred == 2)).sum())
    nonpoor = int((y != 2).sum())
    multi["poor_fpr_nonpoor"] = float(poor_fp / max(1, nonpoor))
    ybin = (y == 0).astype(int)
    predbin = (pred == 0).astype(int)
    collapsed = binary_metrics(ybin, score=probs[:, 0], pred=predbin)
    collapsed["auc"] = float(roc_auc_score(ybin, probs[:, 0]))
    return {"prediction_path": str(pred_path), "three_class": multi, "good_vs_rest": collapsed}


def _but_test_rows(bp: but_sqi.Paths) -> pd.DataFrame:
    feat = pd.read_parquet(bp.record84_norm_parquet)
    split = pd.read_csv(bp.split_csv)
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    merged = feat[["record_id", "y"]].merge(
        split[["record_id", "split", "y", "class_name"]],
        on="record_id",
        how="inner",
        suffixes=("_feature", ""),
    )
    test = merged.loc[merged["split"].eq("test")].reset_index(drop=True)
    if test.empty:
        raise RuntimeError("BUT SQI baseline test split is empty")
    return test


def _class_recalls_from_binary(test: pd.DataFrame, pred_good: np.ndarray) -> dict[str, float]:
    pred_good = np.asarray(pred_good, dtype=int).ravel()
    if len(test) != len(pred_good):
        raise ValueError(f"prediction length mismatch: {len(pred_good)} vs {len(test)}")
    out: dict[str, float] = {}
    for cls, key in [("good", "good_recall"), ("medium", "intermediate_recall"), ("bad", "poor_recall")]:
        mask = test["class_name"].eq(cls).to_numpy()
        if not mask.any():
            out[key] = float("nan")
        elif cls == "good":
            out[key] = float((pred_good[mask] == 1).mean())
        else:
            out[key] = float((pred_good[mask] == 0).mean())
    return out


def _svm_binary_details(bp: but_sqi.Paths, test: pd.DataFrame) -> dict[str, Any]:
    metrics = read_json(bp.models / "svm_84sqi_linear" / "svm_84sqi_linear_metrics_seed0.json")
    pred = pd.read_csv(bp.models / "svm_84sqi_linear" / "svm_84sqi_linear_predictions_seed0.csv")
    pred["record_id"] = pred["record_id"].astype(str)
    aligned = pred.set_index("record_id").loc[test["record_id"], "pred"].to_numpy(dtype=int)
    if not np.array_equal((test["y"].to_numpy() == 1).astype(int), pred.set_index("record_id").loc[test["record_id"], "y"].to_numpy(dtype=int)):
        raise RuntimeError("SVM prediction labels do not align with BUT test rows")
    return {
        **_class_recalls_from_binary(test, aligned),
        "confusion": metrics["test"]["confusion_matrix"],
    }


def _mlp_binary_details(bp: but_sqi.Paths, test: pd.DataFrame) -> dict[str, Any]:
    metrics = read_json(bp.models / "lm_mlp_mainline_strict" / "lm_mlp_test_metrics_seed0.json")
    probs = np.load(bp.models / "lm_mlp_mainline_strict" / "probs" / "lm_mlp_test_probs_seed0.npz", allow_pickle=True)
    y = probs["y01_test"].astype(int)
    score = probs["p_test"].astype(np.float64)
    threshold = float(probs["threshold_fixed"])
    expected_y = (test["y"].to_numpy() == 1).astype(int)
    if len(y) != len(test) or not np.array_equal(expected_y, y):
        raise RuntimeError("MLP probability file does not align with BUT test rows")
    pred = (score > threshold).astype(int)
    return {
        **_class_recalls_from_binary(test, pred),
        "confusion": metrics["test_metrics_fixed"]["confusion_matrix"],
    }


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-models", paths)
        return {"step": "but-models", "skipped": True}
    ensure_dirs(paths)
    sqi = _ensure_but_sqi(paths, force=force, device=device)
    bp = but_sqi.Paths(paths.but / "sqi_baseline")
    but_test = _but_test_rows(bp)
    svm_details = _svm_binary_details(bp, but_test)
    mlp_details = _mlp_binary_details(bp, but_test)
    e31 = _e31_summary()
    rows = [
        {
            "run_id": "but_linear_svm_84sqi_good_vs_rest",
            "model": "Linear SVM 84-SQI",
            "input": "single-lead SQI copied to pseudo-12-lead 84-SQI",
            "task": "good vs medium/bad",
            "test_acc": sqi["svm_84sqi_linear"]["acc"],
            "test_macro_f1": sqi["svm_84sqi_linear"]["macro_f1"],
            "test_auc": sqi["svm_84sqi_linear"]["auc"],
            "good_recall": svm_details["good_recall"],
            "intermediate_recall": svm_details["intermediate_recall"],
            "poor_recall": svm_details["poor_recall"],
            "poor_fpr_nonpoor": "",
            "collapsed_good_vs_rest_acc": sqi["svm_84sqi_linear"]["acc"],
            "collapsed_good_vs_rest_auc": sqi["svm_84sqi_linear"]["auc"],
            "collapsed_good_vs_rest_confusion": json.dumps(svm_details["confusion"]),
            "confusion": json.dumps(svm_details["confusion"]),
        },
        {
            "run_id": "but_lm_mlp_84sqi_good_vs_rest",
            "model": "LM-MLP 84-J-1",
            "input": "single-lead SQI copied to pseudo-12-lead 84-SQI",
            "task": "good vs medium/bad",
            "test_acc": sqi["lm_mlp_84_j_1"]["acc"],
            "test_macro_f1": sqi["lm_mlp_84_j_1"]["macro_f1"],
            "test_auc": sqi["lm_mlp_84_j_1"]["auc"],
            "good_recall": mlp_details["good_recall"],
            "intermediate_recall": mlp_details["intermediate_recall"],
            "poor_recall": mlp_details["poor_recall"],
            "poor_fpr_nonpoor": "",
            "collapsed_good_vs_rest_acc": sqi["lm_mlp_84_j_1"]["acc"],
            "collapsed_good_vs_rest_auc": sqi["lm_mlp_84_j_1"]["auc"],
            "collapsed_good_vs_rest_confusion": json.dumps(mlp_details["confusion"]),
            "confusion": json.dumps(mlp_details["confusion"]),
        },
        {
            "run_id": "but_e31_wave_mechanism_conformer",
            "model": "E31 wave-mechanism Conformer",
            "input": "8-channel waveform-derived time series",
            "task": "good/medium/bad",
            "test_acc": e31["three_class"]["acc"],
            "test_macro_f1": e31["three_class"]["macro_f1"],
            "test_auc": "",
            "good_recall": e31["three_class"]["good_recall"],
            "intermediate_recall": e31["three_class"]["intermediate_recall"],
            "poor_recall": e31["three_class"]["poor_recall"],
            "poor_fpr_nonpoor": e31["three_class"]["poor_fpr_nonpoor"],
            "collapsed_good_vs_rest_acc": e31["good_vs_rest"]["acc"],
            "collapsed_good_vs_rest_auc": e31["good_vs_rest"]["auc"],
            "collapsed_good_vs_rest_confusion": json.dumps(e31["good_vs_rest"]["confusion"]),
            "confusion": json.dumps(e31["three_class"]["confusion"]),
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(paths.tables / "but_model_comparison.csv", index=False)
    out = {"sqi_baseline": sqi, "e31": e31, "table": rows}
    write_json(paths.but_models_json, out)
    print(table.to_string(index=False))
    return {"step": "but-models", "skipped": False, "outputs": out}
