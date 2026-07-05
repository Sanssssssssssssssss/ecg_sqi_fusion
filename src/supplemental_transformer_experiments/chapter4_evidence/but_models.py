from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import torch

from src.supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi

from src.sqi_pipeline.models.lm_mlp import LMConfig, LMMLP
from .common import Paths, binary_metrics, dry, ensure_dirs, multiclass_summary, read_json, write_json

SELECTED5_SQI = {"bSQI", "basSQI", "kSQI", "sSQI", "fSQI"}
SVM_RBF_C = 1.0
SVM_RBF_GAMMA = 0.14
LM_MLP_J = 8


def _ensure_but_sqi(paths: Paths, *, force: bool, device: str) -> dict[str, Any]:
    bp = but_sqi.Paths(paths.but / "sqi_baseline")
    if force or not bp.summary_json.exists():
        but_sqi.cmd_prepare(bp, run=True, force=force)
        but_sqi.cmd_features(bp, run=True, force=force, jobs=4)
        but_sqi.cmd_train(bp, run=True, force=force, device=device)
        but_sqi.cmd_summary(bp)
    return json.loads(bp.summary_json.read_text(encoding="utf-8"))


def _find_e31_predictions() -> Path:
    run_root = Path("outputs") / "transformer" / "v116_e31" / "runs"
    roots = list(run_root.glob("**/E31_query_mean_fused_conformer_fold0_seed0/test_predictions.npz"))
    roots += list(run_root.glob("**/E31_wave_mechanism_conformer_fold0_seed0/test_predictions.npz"))
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
    multi["poor_vs_rest_auc"] = float(roc_auc_score((y == 2).astype(int), probs[:, 2]))
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


def _but_multiclass_data(bp: but_sqi.Paths, *, selected5: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat = pd.read_parquet(bp.record84_norm_parquet)
    split = pd.read_csv(bp.split_csv)
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    df = split[["record_id", "split", "class_name"]].merge(feat, on="record_id", how="inner")
    label_map = {"good": 0, "medium": 1, "bad": 2}
    df["y3"] = df["class_name"].map(label_map).astype(int)
    cols = [c for c in feat.columns if "__" in c]
    if len(cols) != 84:
        raise RuntimeError(f"expected 84 SQI feature columns, got {len(cols)}")
    if selected5:
        cols = [c for c in cols if c.split("__", 1)[1] in SELECTED5_SQI]
        if len(cols) != 60:
            raise RuntimeError(f"expected 60 selected-five SQI feature columns, got {len(cols)}")

    def part(name: str) -> tuple[np.ndarray, np.ndarray]:
        rows = df.loc[df["split"].eq(name)]
        return rows[cols].to_numpy(np.float64), rows["y3"].to_numpy(int)

    xtr, ytr = part("train")
    xva, yva = part("val")
    xte, yte = part("test")
    return xtr, ytr, xva, yva, xte, yte


def _collapse_good_vs_rest(y: np.ndarray, pred: np.ndarray, score_good: np.ndarray) -> dict[str, Any]:
    return binary_metrics((np.asarray(y) == 0).astype(int), score=score_good, pred=(np.asarray(pred) == 0).astype(int))


def _finish_multiclass(y: np.ndarray, score: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    pred = np.asarray(score).argmax(axis=1)
    multi = multiclass_summary(y, score, ["good", "intermediate", "poor"])
    poor_fp = int(((y != 2) & (pred == 2)).sum())
    multi["poor_fpr_nonpoor"] = float(poor_fp / max(1, int((y != 2).sum())))
    multi["poor_vs_rest_auc"] = float(roc_auc_score((y == 2).astype(int), score[:, 2]))
    collapsed = _collapse_good_vs_rest(y, pred, score[:, 0])
    collapsed["auc"] = float(roc_auc_score((y == 0).astype(int), score[:, 0]))
    return multi, collapsed


def _svm_rbf_multiclass_details(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xte: np.ndarray, yte: np.ndarray) -> dict[str, Any]:
    model = SVC(kernel="rbf", C=SVM_RBF_C, gamma=SVM_RBF_GAMMA, probability=True, random_state=0)
    model.fit(xtr, ytr)
    score = model.predict_proba(xte).astype(np.float64)
    multi, collapsed = _finish_multiclass(yte, score)
    multi["val_acc"] = float((model.predict(xva) == yva).mean())
    multi["svm_C"] = SVM_RBF_C
    multi["svm_gamma"] = SVM_RBF_GAMMA
    return {"three_class": multi, "good_vs_rest": collapsed}


def _lm_mlp_ovr_scores(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xte: np.ndarray, *, device: str) -> tuple[np.ndarray, np.ndarray]:
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cfg = LMConfig()
    val_scores: list[np.ndarray] = []
    test_scores: list[np.ndarray] = []
    xtr_t = torch.tensor(xtr, device=dev, dtype=dtype)
    xva_t = torch.tensor(xva, device=dev, dtype=dtype)
    xte_t = torch.tensor(xte, device=dev, dtype=dtype)
    for cls in range(3):
        model = LMMLP(J=LM_MLP_J, D=xtr.shape[1], device=dev, dtype=dtype, seed=cls)
        model.fit_lm(
            X_train=xtr_t,
            y_train=torch.tensor((ytr == cls).astype(np.float64), device=dev, dtype=dtype),
            cfg=cfg,
            X_val=xva_t,
            y_val=torch.tensor((yva == cls).astype(np.float64), device=dev, dtype=dtype),
            model_select_metric="val_acc",
            patience=15,
            threshold=0.5,
        )
        val_scores.append(model.predict_proba(xva_t).astype(np.float64))
        test_scores.append(model.predict_proba(xte_t).astype(np.float64))
    return np.stack(val_scores, axis=1), np.stack(test_scores, axis=1)


def _lm_mlp_multiclass_details(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xte: np.ndarray, yte: np.ndarray, *, device: str) -> dict[str, Any]:
    val_score, score = _lm_mlp_ovr_scores(xtr, ytr, xva, yva, xte, device=device)
    multi, collapsed = _finish_multiclass(yte, score)
    multi["val_acc"] = float((val_score.argmax(axis=1) == yva).mean())
    multi["J"] = LM_MLP_J
    return {"three_class": multi, "good_vs_rest": collapsed}


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-models", paths)
        return {"step": "but-models", "skipped": True}
    ensure_dirs(paths)
    sqi = _ensure_but_sqi(paths, force=force, device=device)
    bp = but_sqi.Paths(paths.but / "sqi_baseline")
    xtr, ytr, xva, yva, xte, yte = _but_multiclass_data(bp)
    xtr5, ytr5, xva5, yva5, xte5, yte5 = _but_multiclass_data(bp, selected5=True)
    svm_details = _svm_rbf_multiclass_details(xtr, ytr, xva, yva, xte, yte)
    svm5_details = _svm_rbf_multiclass_details(xtr5, ytr5, xva5, yva5, xte5, yte5)
    mlp_details = _lm_mlp_multiclass_details(xtr, ytr, xva, yva, xte, yte, device=device)
    e31 = _e31_summary()
    rows = [
        {
            "run_id": "but_svm_rbf_84sqi_multiclass",
            "model": "SQI SVM-RBF all84",
            "input": "single-lead SQI copied to pseudo-12-lead 84-SQI",
            "task": "good/medium/bad",
            "test_acc": svm_details["three_class"]["acc"],
            "test_macro_f1": svm_details["three_class"]["macro_f1"],
            "test_auc": "",
            "good_recall": svm_details["three_class"]["good_recall"],
            "intermediate_recall": svm_details["three_class"]["intermediate_recall"],
            "poor_recall": svm_details["three_class"]["poor_recall"],
            "poor_fpr_nonpoor": svm_details["three_class"]["poor_fpr_nonpoor"],
            "poor_vs_rest_auc": svm_details["three_class"]["poor_vs_rest_auc"],
            "collapsed_good_vs_rest_acc": svm_details["good_vs_rest"]["acc"],
            "collapsed_good_vs_rest_auc": svm_details["good_vs_rest"]["auc"],
            "collapsed_good_vs_rest_confusion": json.dumps(svm_details["good_vs_rest"]["confusion"]),
            "confusion": json.dumps(svm_details["three_class"]["confusion"]),
        },
        {
            "run_id": "but_svm_rbf_selected5_sqi_multiclass",
            "model": "SQI SVM-RBF selected5",
            "input": "selected-five SQI copied to pseudo-12-lead",
            "task": "good/medium/bad",
            "test_acc": svm5_details["three_class"]["acc"],
            "test_macro_f1": svm5_details["three_class"]["macro_f1"],
            "test_auc": "",
            "good_recall": svm5_details["three_class"]["good_recall"],
            "intermediate_recall": svm5_details["three_class"]["intermediate_recall"],
            "poor_recall": svm5_details["three_class"]["poor_recall"],
            "poor_fpr_nonpoor": svm5_details["three_class"]["poor_fpr_nonpoor"],
            "poor_vs_rest_auc": svm5_details["three_class"]["poor_vs_rest_auc"],
            "collapsed_good_vs_rest_acc": svm5_details["good_vs_rest"]["acc"],
            "collapsed_good_vs_rest_auc": svm5_details["good_vs_rest"]["auc"],
            "collapsed_good_vs_rest_confusion": json.dumps(svm5_details["good_vs_rest"]["confusion"]),
            "confusion": json.dumps(svm5_details["three_class"]["confusion"]),
        },
        {
            "run_id": "but_lm_mlp_84sqi_ovr_multiclass",
            "model": f"SQI LM-MLP 84-{LM_MLP_J}-1 OvR",
            "input": "single-lead SQI copied to pseudo-12-lead 84-SQI",
            "task": "good/medium/bad",
            "test_acc": mlp_details["three_class"]["acc"],
            "test_macro_f1": mlp_details["three_class"]["macro_f1"],
            "test_auc": "",
            "good_recall": mlp_details["three_class"]["good_recall"],
            "intermediate_recall": mlp_details["three_class"]["intermediate_recall"],
            "poor_recall": mlp_details["three_class"]["poor_recall"],
            "poor_fpr_nonpoor": mlp_details["three_class"]["poor_fpr_nonpoor"],
            "poor_vs_rest_auc": mlp_details["three_class"]["poor_vs_rest_auc"],
            "collapsed_good_vs_rest_acc": mlp_details["good_vs_rest"]["acc"],
            "collapsed_good_vs_rest_auc": mlp_details["good_vs_rest"]["auc"],
            "collapsed_good_vs_rest_confusion": json.dumps(mlp_details["good_vs_rest"]["confusion"]),
            "confusion": json.dumps(mlp_details["three_class"]["confusion"]),
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
            "poor_vs_rest_auc": e31["three_class"]["poor_vs_rest_auc"],
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
