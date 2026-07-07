from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.sqi_pipeline.models import lm_mlp_search, svm_tables
from src.sqi_pipeline.models.svm_rbf import SVMConfig, SVMRBF
from src.supplemental_transformer_experiments.sqi12_gapfill import run as sqi12

from .common import Paths, binary_metrics, dry, ensure_dirs, write_json
from .pretrained import SETA_E31_DIR, require_files, use_pretrained_on_cpu
from .seta_sqi import ARMS, arm_dir


SELECTED5 = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]


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
        table6 = pd.read_csv(out / "table6_12lead_combo_sqi_seed0.csv")
        row = table6.loc[table6["Group"].eq("Quintuplets")].iloc[0].to_dict()
        selected = str(row["Selected_SQI"])
        probs = out / "probs" / "Quintuplets_seed0.npz"
        parameter_source = "paper_table6_fixed_C1_gamma0.14"
    else:
        table6 = pd.read_csv(out / "table6_12lead_combo_sqi_seed0.csv")
        row = table6.loc[table6["Group"].eq("All SQI")].iloc[0].to_dict()
        selected = "all84"
        probs = out / "probs" / "AllSQI_seed0.npz"
        parameter_source = "paper_table6_fixed_C1_gamma0.14"
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
        se = float(row.get("Sp_test", float("nan")))
        sp = float(row.get("Se_test", float("nan")))
        bal = float("nan")
        cm = ""
    return {
        "run_id": f"seta_{arm}_{model.replace(' ', '_').replace('-', '').lower()}",
        "construction": arm,
        "model": model,
        "input": selected,
        "parameter_source": parameter_source,
        "threshold_source": "validation_max_accuracy",
        "threshold": threshold,
        "acc": acc,
        "auc": float(row["AUC_test"]),
        "balanced_acc": float(bal),
        "acceptable_recall": se,
        "original_unacceptable_recall": sp,
        "confusion": cm,
    }


def _selected5_cols() -> list[str]:
    return [f"{lead}__{sqi}" for lead in svm_tables.LEADS_12 for sqi in SELECTED5]


def _best_acc_threshold(y01: np.ndarray, p: np.ndarray) -> float:
    best_t, best_acc = 0.5, -1.0
    for threshold in np.linspace(0.0, 1.0, 2001):
        acc = float(((p > threshold).astype(int) == y01).mean())
        if acc > best_acc:
            best_t, best_acc = float(threshold), acc
    return best_t


def _row_from_source_only_svm(arm: str, root: Path) -> dict[str, Any]:
    split = pd.read_csv(root / "splits" / "split.csv")
    feat = pd.read_parquet(root / "features" / "record84_norm.parquet").drop(columns=["y"], errors="ignore")
    split["record_id"] = split["record_id"].astype(str)
    feat["record_id"] = feat["record_id"].astype(str)
    df = feat.merge(split[["record_id", "split", "y", "is_augmented", "candidate_type"]], on="record_id", how="inner")
    is_aug = pd.to_numeric(df["is_augmented"], errors="coerce").fillna(0).astype(int)
    y_pm = pd.to_numeric(df["y"], errors="coerce").astype(int)
    y01 = y_pm.eq(1).astype(int).to_numpy()
    train_acc = df["split"].eq("train") & y_pm.eq(1) & is_aug.eq(0)
    train_poor = df["split"].eq("train") & y_pm.eq(-1)
    source_contract = "original train unacceptable only"
    if arm != "native_imbalanced":
        train_poor &= is_aug.eq(1)
        source_contract = "generated train unacceptable only; original train unacceptable excluded"
    val = df["split"].eq("val") & is_aug.eq(0)
    test = df["split"].eq("test") & is_aug.eq(0)
    cols = _selected5_cols()
    model = SVMRBF(SVMConfig(seed=0, use_standard_scaler=False))
    model.fit_fixed(df.loc[train_acc | train_poor, cols].to_numpy(float), y01[(train_acc | train_poor).to_numpy()], C=1.0, gamma=0.14)
    p_val = model.predict_proba(df.loc[val, cols].to_numpy(float))
    threshold = _best_acc_threshold(y01[val.to_numpy()], p_val)
    p_test = model.predict_proba(df.loc[test, cols].to_numpy(float))
    pred = (p_test > threshold).astype(int)
    met = binary_metrics(y01[test.to_numpy()], score=p_test, pred=pred)
    return {
        "run_id": f"seta_{arm}_source_only_sqi_svmrbf_selected5",
        "construction": arm,
        "model": "SQI SVM-RBF selected5",
        "input": ",".join(SELECTED5),
        "parameter_source": "paper_table6_fixed_C1_gamma0.14",
        "threshold_source": "validation_max_accuracy",
        "threshold": threshold,
        "train_acceptable_original_n": int(train_acc.sum()),
        "train_poor_source_n": int(train_poor.sum()),
        "source_only_train_poor_contract": source_contract,
        "acc": float(met["acc"]),
        "auc": float(met["auc"]),
        "balanced_acc": float(met["balanced_acc"]),
        "acceptable_recall": float(met["acceptable_recall"]),
        "original_unacceptable_recall": float(met["original_unacceptable_recall"]),
        "confusion": json.dumps(met["confusion"]),
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
        "acceptable_recall": float(met.get("acceptable_recall", met["sp"])),
        "original_unacceptable_recall": float(met.get("unacceptable_recall", met["se"])),
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
        if row["acc"] >= best_acc - 0.02 and row["macro_f1"] >= best_f1 - 0.03
    ]
    if not near_best:
        raise RuntimeError("no validation threshold met near-optimal accuracy/F1 criteria")
    floor = [
        row
        for row in near_best
        if row["acceptable_recall"] >= 0.90 and row["original_unacceptable_recall"] >= 0.80
    ]
    operating = floor or near_best
    chosen = sorted(
        operating,
        key=lambda row: (
            abs(row["acceptable_recall"] - row["original_unacceptable_recall"]),
            -row["balanced_acc"],
            -row["macro_f1"],
        ),
    )[0]
    chosen_cm = chosen["confusion"]
    plateau = [
        row
        for row in candidates
        if row["confusion"] == chosen_cm
        and row["acc"] == chosen["acc"]
        and row["macro_f1"] == chosen["macro_f1"]
    ]
    threshold = float(np.median([row["threshold"] for row in plateau])) if plateau else float(chosen["threshold"])
    yt = test["y"].to_numpy(dtype=int)
    st = test["prob_acceptable"].to_numpy(dtype=np.float64)
    test_met = binary_metrics(yt, score=st, pred=(st >= threshold).astype(int))
    chosen["threshold"] = threshold
    test_met["threshold"] = threshold
    return threshold, chosen, test_met


def _fixed_original_bad_recall(predictions_csv: Path) -> dict[str, float]:
    pred = pd.read_csv(predictions_csv)
    out: dict[str, float] = {}
    for split in ["train", "val", "test"]:
        rows = pred.loc[
            pred["split"].eq(split)
            & pred["y"].astype(int).eq(0)
            & pred["generated"].astype(int).eq(0)
        ]
        value = float(rows["pred_fixed"].astype(int).eq(0).mean()) if len(rows) else float("nan")
        key = "val_bad_recall_fixed05" if split == "val" else f"{split}_original_bad_recall_fixed05"
        out[key] = value
    return out


def _row_from_conformer(paths: Paths, *, force: bool, device: str) -> dict[str, Any]:
    sp = sqi12.Paths(paths.seta)
    metrics_path = paths.seta / "models" / "e31_leadwise_shared" / "metrics.json"
    predictions_path = paths.seta / "models" / "e31_leadwise_shared" / "predictions.csv"
    if force or not metrics_path.exists():
        if use_pretrained_on_cpu(device):
            require_files(SETA_E31_DIR, ("best_model.pt", "history.csv"))
            sqi12.cmd_predict_from_checkpoint(sp, run=True, checkpoint_dir=SETA_E31_DIR, device=device)
        else:
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
    _, val_met, met = _acceptability_preserving_threshold(predictions_path)
    cm = met["confusion"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    return {
        "run_id": "seta_smc_gapfill_e31style_waveform",
        "construction": "smc_gapfill",
        "model": "12-lead E31-style waveform comparator",
        "input": "12-lead waveform-derived channels",
        "threshold_source": "validation_recall_balanced",
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
        **_fixed_original_bad_recall(predictions_path),
    }


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("seta-models", paths)
        return {"step": "seta-models", "skipped": True}
    ensure_dirs(paths)
    source_only_rows = []
    for arm in ARMS:
        root = arm_dir(paths, arm)
        source_only_rows.append(_row_from_source_only_svm(arm, root))

    smc_root = arm_dir(paths, "smc_gapfill")
    smc_svm = paths.seta_models / "smc_gapfill" / "svm"
    _run_svm(smc_root, smc_svm, force=force)

    mlp_out = paths.seta_models / "smc_gapfill" / "lm_mlp"
    _run_mlp(smc_root, mlp_out, force=force, device=device)

    model_rows = [
        _row_from_svm("smc_gapfill", "SQI SVM-RBF selected5", smc_svm),
        _row_from_svm("smc_gapfill", "SQI SVM-RBF all84", smc_svm),
        _row_from_mlp(mlp_out),
        _row_from_conformer(paths, force=force, device=device),
    ]
    source_only = pd.DataFrame(source_only_rows)
    models = pd.DataFrame(model_rows)
    source_only.to_csv(paths.tables / "seta_construction_source_only_models.csv", index=False)
    models.to_csv(paths.tables / "seta_repaired_model_comparison.csv", index=False)
    from .report import _seta_construction_source_audit

    _seta_construction_source_audit(paths)
    out = {
        "construction_source_only": source_only.to_dict(orient="records"),
        "model_comparison": models.to_dict(orient="records"),
    }
    write_json(paths.seta_models_json, out)
    print(models.to_string(index=False))
    return {"step": "seta-models", "skipped": False, "outputs": out}
