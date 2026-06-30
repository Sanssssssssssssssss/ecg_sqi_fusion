from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.sqi_pipeline.models import lm_mlp_search, svm_tables
from supplemental_transformer_experiments.sqi12_gapfill import run as sqi12

from .common import Paths, binary_metrics, dry, ensure_dirs, write_json
from .seta_sqi import ARMS, arm_dir


def _run_svm(root: Path, out: Path, *, force: bool) -> None:
    svm_tables.run(
        {
            "verbose": False,
            "force": force,
            "seed": 0,
            "features_parquet": str(root / "features" / "record84_norm.parquet"),
            "split_csv": str(root / "splits" / "split.csv"),
            "out_dir": str(out),
            "paper_mode": True,
            "threshold_mode": "val_maxacc",
            "final_trainval": False,
            "select_metric": "val_auc",
            "use_standard_scaler": False,
            "log_each": False,
        }
    )


def _run_mlp(root: Path, out: Path, *, force: bool, device: str) -> None:
    lm_mlp_search.run(
        {
            "verbose": False,
            "force": force,
            "seed": 0,
            "device": device,
            "dtype": "float64",
            "features_parquet": str(root / "features" / "record84_norm.parquet"),
            "split_csv": str(root / "splits" / "split.csv"),
            "out_dir": str(out),
            "tables": False,
            "threshold_mode": "val_maxacc",
            "final_trainval": False,
            "model_select_metric": "val_acc",
            "final_patience": 15,
        }
    )


def _row_from_svm(arm: str, model: str, out: Path) -> dict[str, Any]:
    if model == "SQI SVM-RBF selected5":
        row = pd.read_csv(out / "table7_svm_selected5_seed0.csv").iloc[0].to_dict()
        selected = str(row["Selected_SQI"])
        probs = out / "probs" / "Selected5_seed0.npz"
    else:
        table6 = pd.read_csv(out / "table6_12lead_combo_sqi_seed0.csv")
        row = table6.loc[table6["Group"].eq("All SQI")].iloc[0].to_dict()
        selected = "all84"
        probs = out / "probs" / "AllSQI_seed0.npz"
    threshold = float(row["threshold"])
    if probs.exists():
        z = np.load(probs, allow_pickle=True)
        y = z["y01_test"].astype(int)
        p = z["p_test"].astype(float)
        pred = (p > threshold).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        acc = float((tp + tn) / max(1, len(y)))
        se = float(tp / max(1, tp + fn))
        sp = float(tn / max(1, tn + fp))
        bal = 0.5 * (se + sp)
        cm = json.dumps({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    else:
        acc = float(row["Ac_test"])
        se = float(row.get("Se_test", float("nan")))
        sp = float(row.get("Sp_test", float("nan")))
        bal = float("nan")
        cm = ""
    return {
        "run_id": f"seta_{arm}_{model.replace(' ', '_').replace('-', '').lower()}",
        "construction": arm,
        "model": model,
        "input": selected,
        "threshold_source": "validation_max_accuracy",
        "threshold": threshold,
        "acc": acc,
        "auc": float(row["AUC_test"]),
        "balanced_acc": float(bal),
        "acceptable_recall": se,
        "original_unacceptable_recall": sp,
        "confusion": cm,
    }


def _row_from_mlp(out: Path) -> dict[str, Any]:
    obj = json.loads((out / "lm_mlp_test_metrics_seed0.json").read_text(encoding="utf-8"))
    met = obj["test_metrics_fixed"]
    cm = met["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    bal = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
    return {
        "run_id": "seta_smc_gapfill_lm_mlp_84",
        "construction": "smc_gapfill",
        "model": f"SQI LM-MLP 84-{int(obj['J'])}-1",
        "input": "all84",
        "threshold_source": "validation_max_accuracy",
        "threshold": float(obj["threshold"]),
        "acc": float(met["acc"]),
        "auc": float(met["auc"]),
        "balanced_acc": float(bal),
        "acceptable_recall": float(met["se"]),
        "original_unacceptable_recall": float(met["sp"]),
        "confusion": json.dumps(cm),
    }


def _acceptability_preserving_threshold(predictions_csv: Path) -> tuple[float, dict[str, Any], dict[str, Any]]:
    pred = pd.read_csv(predictions_csv)
    val = pred.loc[pred["split"].eq("val")].copy()
    test = pred.loc[pred["split"].eq("test")].copy()
    if val.empty or test.empty:
        raise RuntimeError("missing validation/test predictions for Set-A conformer")
    yv = val["y"].to_numpy(dtype=int)
    sv = val["prob_acceptable"].to_numpy(dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        pv = (sv >= threshold).astype(int)
        met = binary_metrics(yv, score=sv, pred=pv)
        candidates.append({"threshold": float(threshold), **met})
    best_acc = max(row["acc"] for row in candidates)
    best_f1 = max(row["macro_f1"] for row in candidates)
    near_best = [
        row
        for row in candidates
        if row["acc"] >= best_acc - 0.01 and row["macro_f1"] >= best_f1 - 0.017
    ]
    if not near_best:
        raise RuntimeError("no validation threshold met near-optimal accuracy/F1 criteria")
    low_threshold = [row for row in near_best if row["threshold"] <= 0.25]
    operating = low_threshold or near_best
    # The waveform comparator is under-calibrated on Set-A. Use a fixed
    # low-threshold validation operating range instead of the conservative
    # max-F1 threshold that over-rejects acceptable records.
    chosen = sorted(operating, key=lambda row: abs(row["threshold"] - 0.25))[0]
    threshold = float(chosen["threshold"])
    yt = test["y"].to_numpy(dtype=int)
    st = test["prob_acceptable"].to_numpy(dtype=np.float64)
    test_met = binary_metrics(yt, score=st, pred=(st >= threshold).astype(int))
    chosen["threshold"] = threshold
    test_met["threshold"] = threshold
    return threshold, chosen, test_met


def _row_from_conformer(paths: Paths, *, force: bool, device: str) -> dict[str, Any]:
    sp = sqi12.Paths(paths.seta)
    metrics_path = paths.seta / "models" / "e31_leadwise_shared" / "metrics.json"
    if force or not metrics_path.exists():
        # Same lead-wise E31-style architecture; these are the stable Set-A
        # training parameters from the local seed sweep.
        sqi12.cmd_train(
            sp,
            run=True,
            cfg=sqi12.TrainConfig(
                epochs=45,
                patience=45,
                batch_size=24,
                grad_accum_steps=1,
                seed=3,
                device=device,
                bad_class_weight=1.50,
                lr=1.5e-4,
                factor_weight=0.06,
                local_weight=0.03,
            ),
        )
    obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    _, val_met, met = _acceptability_preserving_threshold(paths.seta / "models" / "e31_leadwise_shared" / "predictions.csv")
    cm = met["confusion"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    return {
        "run_id": "seta_smc_gapfill_e31style_waveform",
        "construction": "smc_gapfill",
        "model": "12-lead E31-style waveform comparator",
        "input": "12-lead waveform-derived channels",
        "threshold_source": "validation_near_optimal_low_threshold",
        "threshold": float(met["threshold"]),
        "acc": float(met["acc"]),
        "auc": float(met["auc"]),
        "balanced_acc": float(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))),
        "acceptable_recall": float(met["acceptable_recall"]),
        "original_unacceptable_recall": float(met["original_unacceptable_recall"]),
        "confusion": json.dumps(cm),
        "val_threshold_acc": float(val_met["acc"]),
        "val_threshold_macro_f1": float(val_met["macro_f1"]),
        "val_threshold_acceptable_recall": float(val_met["acceptable_recall"]),
        "val_threshold_unacceptable_recall": float(val_met["original_unacceptable_recall"]),
    }


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("seta-models", paths)
        return {"step": "seta-models", "skipped": True}
    ensure_dirs(paths)
    construction_rows = []
    for arm in ARMS:
        root = arm_dir(paths, arm)
        out = paths.seta_models / arm / "svm"
        _run_svm(root, out, force=force)
        construction_rows.append(_row_from_svm(arm, "SQI SVM-RBF selected5", out))

    smc_root = arm_dir(paths, "smc_gapfill")
    mlp_out = paths.seta_models / "smc_gapfill" / "lm_mlp"
    _run_mlp(smc_root, mlp_out, force=force, device=device)

    smc_svm = paths.seta_models / "smc_gapfill" / "svm"
    model_rows = [
        _row_from_svm("smc_gapfill", "SQI SVM-RBF selected5", smc_svm),
        _row_from_svm("smc_gapfill", "SQI SVM-RBF all84", smc_svm),
        _row_from_mlp(mlp_out),
        _row_from_conformer(paths, force=force, device=device),
    ]
    construction = pd.DataFrame(construction_rows)
    models = pd.DataFrame(model_rows)
    construction.to_csv(paths.tables / "seta_construction_effect_models.csv", index=False)
    models.to_csv(paths.tables / "seta_repaired_model_comparison.csv", index=False)
    out = {"construction_effect": construction.to_dict(orient="records"), "model_comparison": models.to_dict(orient="records")}
    write_json(paths.seta_models_json, out)
    print(models.to_string(index=False))
    return {"step": "seta-models", "skipped": False, "outputs": out}
