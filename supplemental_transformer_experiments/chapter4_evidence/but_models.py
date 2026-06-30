from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi

from .common import Paths, binary_metrics, dry, ensure_dirs, multiclass_summary, write_json


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


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-models", paths)
        return {"step": "but-models", "skipped": True}
    ensure_dirs(paths)
    sqi = _ensure_but_sqi(paths, force=force, device=device)
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
            "good_recall": "",
            "intermediate_recall": "",
            "poor_recall": "",
            "poor_fpr_nonpoor": "",
            "collapsed_good_vs_rest_acc": "",
            "collapsed_good_vs_rest_auc": "",
            "collapsed_good_vs_rest_confusion": "",
            "confusion": "",
        },
        {
            "run_id": "but_lm_mlp_84sqi_good_vs_rest",
            "model": "LM-MLP 84-J-1",
            "input": "single-lead SQI copied to pseudo-12-lead 84-SQI",
            "task": "good vs medium/bad",
            "test_acc": sqi["lm_mlp_84_j_1"]["acc"],
            "test_macro_f1": sqi["lm_mlp_84_j_1"]["macro_f1"],
            "test_auc": sqi["lm_mlp_84_j_1"]["auc"],
            "good_recall": "",
            "intermediate_recall": "",
            "poor_recall": "",
            "poor_fpr_nonpoor": "",
            "collapsed_good_vs_rest_acc": "",
            "collapsed_good_vs_rest_auc": "",
            "collapsed_good_vs_rest_confusion": "",
            "confusion": "",
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
