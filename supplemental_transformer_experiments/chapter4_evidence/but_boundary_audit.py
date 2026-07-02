from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi

from .but_models import LM_MLP_J, SVM_RBF_C, SVM_RBF_GAMMA, _ensure_but_sqi, _find_e31_predictions, _lm_mlp_ovr_scores
from .common import Paths, dry, ensure_dirs, write_json


LABELS = ["good", "medium", "bad"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def _score_columns(feat: pd.DataFrame) -> list[str]:
    cols = [c for c in feat.columns if "__" in c]
    if len(cols) != 84:
        raise RuntimeError(f"expected exactly 84 SQI feature columns, got {len(cols)}")
    banned = {"idx", "source_idx", "split", "v116_generated", "v116_candidate_type"}
    leak = sorted(set(cols) & banned)
    if leak:
        raise RuntimeError(f"metadata leaked into SQI feature columns: {leak}")
    return cols


def _load_frame(bp: but_sqi.Paths) -> tuple[pd.DataFrame, list[str]]:
    feat = pd.read_parquet(bp.record84_norm_parquet)
    split = pd.read_csv(bp.split_csv)
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    cols = _score_columns(feat)
    keep = ["record_id", "idx", "source_idx", "split", "class_name", "v116_candidate_type", "v116_generated"]
    df = split[keep].merge(feat[["record_id", *cols]], on="record_id", how="inner")
    df["y3"] = df["class_name"].map(LABEL_TO_ID).astype(int)
    test = df.loc[df["split"].eq("test")]
    counts = test["class_name"].value_counts().to_dict()
    if len(test) != 1849 or counts != {"good": 1053, "medium": 632, "bad": 164}:
        raise RuntimeError(f"unexpected BUT test split: n={len(test)}, counts={counts}")
    generated = pd.to_numeric(test["v116_generated"], errors="coerce").fillna(0).astype(int)
    if int(generated.sum()) != 0:
        raise RuntimeError(f"test generated rows must be 0, got {int(generated.sum())}")
    return df, cols


def _split_xy(df: pd.DataFrame, cols: list[str], split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = df.loc[df["split"].eq(split)].reset_index(drop=True)
    return rows, rows[cols].to_numpy(np.float64), rows["y3"].to_numpy(int)


def _ordered_scores(model: Any, raw: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    order = [classes.index(i) for i in range(3)]
    return np.asarray(raw, dtype=np.float64)[:, order]


def _fit_svm(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = SVC(kernel="rbf", C=SVM_RBF_C, gamma=SVM_RBF_GAMMA, probability=True, random_state=0)
    model.fit(xtr, ytr)
    score = _ordered_scores(model, model.predict_proba(xte))
    return score.argmax(axis=1).astype(int), score


def _fit_mlp(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xte: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    _, score = _lm_mlp_ovr_scores(xtr, ytr, xva, yva, xte, device=device)
    return score.argmax(axis=1).astype(int), score


def _rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=np.float64), np.ones(k, dtype=np.float64) / k, mode="same")


def _local_detail(x: np.ndarray) -> tuple[float, float]:
    z = np.asarray(x, dtype=np.float64).reshape(-1)
    z = (z - np.median(z)) / max(float(np.std(z)), 1e-6)
    base = _rolling_mean(z, 251)
    high = z - base
    detail = _rolling_mean(np.abs(high), 125)
    return float(np.nanmax(detail)), float(np.nanpercentile(base, 95) - np.nanpercentile(base, 5))


def _boundary_row(model: str, y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    gm = int(cm[0, 1])
    gb = int(cm[0, 2])
    mg = int(cm[1, 0])
    bg = int(cm[2, 0])
    boundary = gm + mg
    denom = int((y == 0).sum() + (y == 1).sum())
    return {
        "model": model,
        "good_to_medium": gm,
        "good_to_bad": gb,
        "medium_to_good": mg,
        "bad_to_good": bg,
        "boundary_exchange_errors": boundary,
        "good_medium_test_n": denom,
        "boundary_exchange_rate": boundary / denom,
        "confusion": json.dumps(cm.astype(int).tolist()),
    }


def _nearest_good(records: pd.DataFrame, xte: np.ndarray, xtr: np.ndarray) -> pd.DataFrame:
    scaler = StandardScaler().fit(xtr)
    z = scaler.transform(xte)
    good_idx = np.flatnonzero(records["true_y"].to_numpy() == 0)
    med_idx = np.flatnonzero(records["true_y"].to_numpy() == 1)
    nearest_id = np.full(len(records), "", dtype=object)
    nearest_dist = np.full(len(records), np.nan, dtype=float)
    if len(good_idx) == 0:
        return records
    zg = z[good_idx]
    for i in med_idx:
        d = np.linalg.norm(zg - z[i], axis=1)
        j = int(np.argmin(d))
        nearest_id[i] = str(records.iloc[good_idx[j]]["record_id"])
        nearest_dist[i] = float(d[j])
    records["sqi_nearest_good_id"] = nearest_id
    records["sqi_distance"] = nearest_dist
    return records


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-boundary-audit", paths)
        return {"step": "but-boundary-audit", "skipped": True}
    ensure_dirs(paths)
    _ensure_but_sqi(paths, force=force, device=device)
    bp = but_sqi.Paths(paths.but / "sqi_baseline")
    df, cols = _load_frame(bp)
    train, xtr, ytr = _split_xy(df, cols, "train")
    _, xva, yva = _split_xy(df, cols, "val")
    test, xte, yte = _split_xy(df, cols, "test")

    svm_pred, svm_score = _fit_svm(xtr, ytr, xva, yva, xte)
    mlp_pred, mlp_prob = _fit_mlp(xtr, ytr, xva, yva, xte, device)
    e31_path = _find_e31_predictions()
    e31 = np.load(e31_path, allow_pickle=True)
    if not np.array_equal(e31["y"].astype(int), yte):
        raise RuntimeError("E31 y does not align with official BUT test split order")
    e31_prob = e31["probs"].astype(np.float64)
    e31_pred = e31_prob.argmax(axis=1).astype(int)

    boundary = pd.DataFrame(
        [
            _boundary_row("SQI SVM-RBF all84", yte, svm_pred),
            _boundary_row(f"SQI LM-MLP 84-{LM_MLP_J}-1 OvR", yte, mlp_pred),
            _boundary_row("Conformer", yte, e31_pred),
        ]
    )
    boundary.to_csv(paths.tables / "but_good_medium_boundary_audit.csv", index=False)

    records = test[["record_id", "idx", "source_idx", "split", "class_name", "v116_candidate_type", "v116_generated", "y3"]].copy()
    records = records.rename(columns={"y3": "true_y", "class_name": "true_class"})
    records["svm_pred"] = [LABELS[i] for i in svm_pred]
    records["mlp_pred"] = [LABELS[i] for i in mlp_pred]
    records["conformer_pred"] = [LABELS[i] for i in e31_pred]
    records["svm_good_score"] = svm_score[:, 0]
    records["svm_medium_score"] = svm_score[:, 1]
    records["svm_bad_score"] = svm_score[:, 2]
    records["mlp_good_prob"] = mlp_prob[:, 0]
    records["mlp_medium_prob"] = mlp_prob[:, 1]
    records["mlp_bad_prob"] = mlp_prob[:, 2]
    records["conformer_good_prob"] = e31_prob[:, 0]
    records["conformer_medium_prob"] = e31_prob[:, 1]
    records["conformer_bad_prob"] = e31_prob[:, 2]
    records = _nearest_good(records, xte, xtr)

    signals = np.load(Path("outputs") / "transformer" / "v116_e31" / "source" / "processed_butqdb" / "signals.npz", allow_pickle=True)["X"]
    details = []
    bases = []
    for source_idx in pd.to_numeric(records["source_idx"], errors="raise").astype(int):
        detail, base = _local_detail(signals[int(source_idx), 0])
        details.append(detail)
        bases.append(base)
    records["l_detail"] = details
    records["l_base"] = bases
    records.to_csv(paths.tables / "but_good_medium_boundary_records.csv", index=False)

    out = {
        "boundary_table": str(paths.tables / "but_good_medium_boundary_audit.csv"),
        "records_table": str(paths.tables / "but_good_medium_boundary_records.csv"),
        "e31_prediction_path": str(e31_path),
        "feature_columns": len(cols),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    write_json(paths.reports / "but_good_medium_boundary_audit.json", out)
    print(boundary.to_string(index=False))
    return {"step": "but-boundary-audit", "skipped": False, "outputs": out}
