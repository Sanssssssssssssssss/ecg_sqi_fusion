from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.sqi_pipeline.models.lm_mlp import LMConfig, LMMLP
from src.sqi_pipeline.models.lm_mlp_search import compute_metrics

from .common import (
    SELECTED_FIVE,
    SQIS,
    binary_metrics,
    feature_cols_for_sqis,
    fit_fixed_svm,
    load_split_frame,
    predict_score,
    validate_integrity,
    write_json,
    write_table,
)


TABLE6_COMBOS: list[tuple[str, list[str]]] = [
    ("Pairs", ["iSQI", "basSQI"]),
    ("Triplets", ["bSQI", "basSQI", "pSQI"]),
    ("Quadruplets", ["bSQI", "basSQI", "kSQI", "sSQI"]),
    ("Quintuplets", ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]),
    ("Sextuplets", ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI"]),
    ("All SQI", SQIS),
]

DEFAULT_C_GRID = (1.0, 2.0, 8.0, 25.0, 32.0, 128.0, 512.0)
DEFAULT_GAMMA_GRID = (2.0**-11, 2.0**-9, 2.0**-7, 2.0**-5, 2.0**-3, 0.14, 0.5, 0.7, 1.0, 1.5, 2.0)
J_RANGE = range(2, 18)


def _split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = df["split"].astype(str).to_numpy()
    return split == "train", split == "val", split == "test"


def _max_acc_threshold_first(y01: np.ndarray, score: np.ndarray, *, n_grid: int = 2001) -> dict[str, Any]:
    y = np.asarray(y01, dtype=int).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    best: dict[str, Any] = {"threshold": 0.5, "Ac": -1.0, "Se": np.nan, "Sp": np.nan}
    for threshold in np.linspace(0.0, 1.0, int(n_grid)):
        met = binary_metrics(y, s, float(threshold))
        if float(met["Ac"]) > float(best["Ac"]):
            best = dict(met)
    return best


def _roc_auc_safe(y01: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y01, dtype=int).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64).ravel()))


def _row_from_metrics(
    *,
    model_seed: int,
    model: str,
    feature_set: str,
    sqis: Sequence[str],
    n_features: int,
    C: float | None,
    gamma: float | None,
    J: int | None,
    threshold: float,
    threshold_mode: str,
    val_score: np.ndarray,
    val_y: np.ndarray,
    test_score: np.ndarray,
    test_y: np.ndarray,
    val_threshold_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    val_met = binary_metrics(val_y, val_score, threshold)
    if val_threshold_metrics is not None:
        val_met = {**val_met, **{k: val_threshold_metrics[k] for k in ["Ac", "Se", "Sp"] if k in val_threshold_metrics}}
    test_met = binary_metrics(test_y, test_score, threshold)
    return {
        "model_seed": int(model_seed),
        "model": model,
        "feature_set": feature_set,
        "sqis": ",".join(sqis),
        "n_features": int(n_features),
        "C": np.nan if C is None else float(C),
        "gamma": np.nan if gamma is None else float(gamma),
        "J": np.nan if J is None else int(J),
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
        "val_Ac": float(val_met["Ac"]),
        "val_Se": float(val_met["Se"]),
        "val_Sp": float(val_met["Sp"]),
        "val_AUC": float(_roc_auc_safe(val_y, val_score)),
        "test_Ac": float(test_met["Ac"]),
        "test_Se": float(test_met["Se"]),
        "test_Sp": float(test_met["Sp"]),
        "test_AUC": float(test_met["AUC"]),
        "test_tn": int(test_met["tn"]),
        "test_fp": int(test_met["fp"]),
        "test_fn": int(test_met["fn"]),
        "test_tp": int(test_met["tp"]),
    }


def _fit_svm_protocol(
    df: pd.DataFrame,
    sqis: Sequence[str],
    *,
    model_seed: int,
    feature_set: str,
    model_name: str,
    C: float,
    gamma: float,
) -> dict[str, Any]:
    cols = feature_cols_for_sqis(sqis)
    tr, va, te = _split_masks(df)
    Xtr = df.loc[tr, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, cols].to_numpy(dtype=np.float64)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)

    selector = fit_fixed_svm(Xtr, ytr, C=float(C), gamma=float(gamma), seed=model_seed)
    p_val = predict_score(selector, Xva)
    thr = _max_acc_threshold_first(yva, p_val)

    trv = tr | va
    final = fit_fixed_svm(
        df.loc[trv, cols].to_numpy(dtype=np.float64),
        df.loc[trv, "y01"].to_numpy(dtype=int),
        C=float(C),
        gamma=float(gamma),
        seed=model_seed,
    )
    p_test = predict_score(final, Xte)
    return _row_from_metrics(
        model_seed=model_seed,
        model=model_name,
        feature_set=feature_set,
        sqis=sqis,
        n_features=len(cols),
        C=float(C),
        gamma=float(gamma),
        J=None,
        threshold=float(thr["threshold"]),
        threshold_mode="val_maxacc",
        val_score=p_val,
        val_y=yva,
        test_score=p_test,
        test_y=yte,
        val_threshold_metrics=thr,
    )


def _select_tuned_svm_params(
    df: pd.DataFrame,
    sqis: Sequence[str],
    *,
    model_seed: int,
    C_grid: Sequence[float],
    gamma_grid: Sequence[float],
) -> dict[str, Any]:
    cols = feature_cols_for_sqis(sqis)
    tr, va, _te = _split_masks(df)
    Xtr = df.loc[tr, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    best: dict[str, Any] | None = None
    for C, gamma in itertools.product(C_grid, gamma_grid):
        model = fit_fixed_svm(Xtr, ytr, C=float(C), gamma=float(gamma), seed=model_seed)
        p_val = predict_score(model, Xva)
        val_auc = _roc_auc_safe(yva, p_val)
        val_met = binary_metrics(yva, p_val, 0.5)
        row = {
            "C": float(C),
            "gamma": float(gamma),
            "val_AUC_select": float(val_auc),
            "val_Ac_at_0p5": float(val_met["Ac"]),
        }
        if best is None or (
            row["val_AUC_select"],
            row["val_Ac_at_0p5"],
            -row["C"],
            -row["gamma"],
        ) > (
            best["val_AUC_select"],
            best["val_Ac_at_0p5"],
            -best["C"],
            -best["gamma"],
        ):
            best = row
    assert best is not None
    return best


def _fit_tuned_svm_table7(df: pd.DataFrame, *, model_seed: int) -> dict[str, Any]:
    best = _select_tuned_svm_params(
        df,
        SELECTED_FIVE,
        model_seed=model_seed,
        C_grid=DEFAULT_C_GRID,
        gamma_grid=DEFAULT_GAMMA_GRID,
    )
    row = _fit_svm_protocol(
        df,
        SELECTED_FIVE,
        model_seed=model_seed,
        feature_set="Selected5 tuned",
        model_name="Tuned five-SQI SVM",
        C=float(best["C"]),
        gamma=float(best["gamma"]),
    )
    row["val_AUC_select"] = float(best["val_AUC_select"])
    row["val_Ac_at_0p5"] = float(best["val_Ac_at_0p5"])
    return row


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(x: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float64), device=device, dtype=dtype)


def _fit_mlp_once(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    J: int,
    seed: int,
    cfg: LMConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LMMLP, dict[str, Any]]:
    model = LMMLP(J=int(J), D=Xtr.shape[1], device=device, dtype=dtype, seed=seed)
    summary = model.fit_lm(
        _to_tensor(Xtr, device=device, dtype=dtype),
        _to_tensor(ytr, device=device, dtype=dtype),
        cfg,
        X_val=_to_tensor(Xva, device=device, dtype=dtype),
        y_val=_to_tensor(yva, device=device, dtype=dtype),
        model_select_metric="val_acc",
        patience=101,
        threshold=0.7,
    )
    return model, summary


def _fit_mlp_table7(df: pd.DataFrame, *, model_seed: int) -> dict[str, Any]:
    cols = feature_cols_for_sqis(SELECTED_FIVE, order="sqi_lead")
    tr, va, te = _split_masks(df)
    Xtr = df.loc[tr, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, cols].to_numpy(dtype=np.float64)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cfg = LMConfig(epochs_max=100, stop_error=1e-5, stop_grad=1e-5)

    search_rows: list[dict[str, Any]] = []
    for J in J_RANGE:
        _seed_all(model_seed)
        model, summary = _fit_mlp_once(Xtr, ytr, Xva, yva, J=int(J), seed=model_seed, cfg=cfg, device=device, dtype=dtype)
        p_val = model.predict_proba(_to_tensor(Xva, device=device, dtype=dtype))
        met = compute_metrics(yva, p_val, threshold=0.7)
        search_rows.append(
            {
                "J": int(J),
                "val_acc": float(met["acc"]),
                "val_auc": float(met["auc"]),
                "epochs_used": int(summary["epochs_used"]),
                "stop_reason": str(summary["stop_reason"]),
            }
        )
    search = pd.DataFrame(search_rows)
    best_J = int(search.sort_values(["val_acc", "J"], ascending=[False, True]).iloc[0]["J"])

    _seed_all(model_seed)
    probe, _summary = _fit_mlp_once(Xtr, ytr, Xva, yva, J=best_J, seed=model_seed, cfg=cfg, device=device, dtype=dtype)
    p_val = probe.predict_proba(_to_tensor(Xva, device=device, dtype=dtype))
    thr = _max_acc_threshold_first(yva, p_val)

    Xtrv = np.concatenate([Xtr, Xva], axis=0)
    ytrv = np.concatenate([ytr, yva], axis=0)
    _seed_all(model_seed)
    final = LMMLP(J=best_J, D=Xtrv.shape[1], device=device, dtype=dtype, seed=model_seed)
    final.fit_lm(
        _to_tensor(Xtrv, device=device, dtype=dtype),
        _to_tensor(ytrv, device=device, dtype=dtype),
        cfg,
        X_val=None,
        y_val=None,
        model_select_metric="val_acc",
        patience=101,
        threshold=float(thr["threshold"]),
    )
    p_test = final.predict_proba(_to_tensor(Xte, device=device, dtype=dtype))
    row = _row_from_metrics(
        model_seed=model_seed,
        model="Five-SQI MLP",
        feature_set="Selected5 MLP",
        sqis=SELECTED_FIVE,
        n_features=len(cols),
        C=None,
        gamma=None,
        J=best_J,
        threshold=float(thr["threshold"]),
        threshold_mode="train_internal_0.7_then_val_maxacc",
        val_score=p_val,
        val_y=yva,
        test_score=p_test,
        test_y=yte,
        val_threshold_metrics=thr,
    )
    row["device"] = str(device)
    row["dtype"] = str(dtype)
    row["search_best_val_acc"] = float(search["val_acc"].max())
    row["search_best_val_auc"] = float(search.loc[search["J"].eq(best_J), "val_auc"].iloc[0])
    return row


def _table56_rows_for_seed(df: pd.DataFrame, *, model_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sqi in SQIS:
        rows.append(
            _fit_svm_protocol(
                df,
                [sqi],
                model_seed=model_seed,
                feature_set=sqi,
                model_name="Fixed SVM Table5",
                C=1.0,
                gamma=0.14,
            )
        )
    for name, sqis in TABLE6_COMBOS:
        rows.append(
            _fit_svm_protocol(
                df,
                sqis,
                model_seed=model_seed,
                feature_set=name,
                model_name="Fixed SVM Table6",
                C=1.0,
                gamma=0.14,
            )
        )
    return rows


def _table7_rows_for_seed(df: pd.DataFrame, *, model_seed: int) -> list[dict[str, Any]]:
    fixed = _fit_svm_protocol(
        df,
        SELECTED_FIVE,
        model_seed=model_seed,
        feature_set="Selected5 fixed",
        model_name="Fixed five-SQI SVM",
        C=1.0,
        gamma=0.14,
    )
    tuned = _fit_tuned_svm_table7(df, model_seed=model_seed)
    mlp = _fit_mlp_table7(df, model_seed=model_seed)
    return [fixed, tuned, mlp]


def _summarise_metric_rows(raw: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = ["test_Ac", "test_Se", "test_Sp", "test_AUC", "threshold"]
    rows: list[dict[str, Any]] = []
    for key, g in raw.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: val for col, val in zip(group_cols, key)}
        row: dict[str, Any] = {**base, "n_seeds": int(g["model_seed"].nunique())}
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{metric}_max"] = float(vals.max()) if len(vals) else np.nan
        if "C" in g and "gamma" in g and g[["C", "gamma"]].notna().all(axis=1).any():
            freq = g.dropna(subset=["C", "gamma"]).groupby(["C", "gamma"]).size().sort_values(ascending=False)
            row["C_gamma_frequency"] = "; ".join(f"{c:g},{gm:g}:{int(n)}" for (c, gm), n in freq.items())
        if "J" in g and g["J"].notna().any():
            freq_j = g.dropna(subset=["J"]).groupby("J").size().sort_values(ascending=False)
            row["J_frequency"] = "; ".join(f"{int(j)}:{int(n)}" for j, n in freq_j.items())
        rows.append(row)
    return pd.DataFrame(rows)


def run_model_seed_stability(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    model_seeds: Sequence[int],
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_split_frame(artifacts_dir, normalized=True)
    integrity = validate_integrity(df)
    if integrity["source_group_leakage_n"] != 0:
        raise RuntimeError(f"Fixed split has source leakage: {integrity['source_group_leakage_examples'][:5]}")
    write_json(
        out / "model_seed_stability_manifest.json",
        {
            "artifacts_dir": str(artifacts_dir),
            "model_seeds": [int(s) for s in model_seeds],
            "protocol": "fixed seed-0 corpus, fixed normalized features, fixed existing train/val/test split; only model_seed changes",
            "integrity": integrity,
        },
    )

    rows56: list[dict[str, Any]] = []
    rows7: list[dict[str, Any]] = []
    raw56_csv = out / "table56_model_seed_raw.csv"
    raw7_csv = out / "table7_model_seed_raw.csv"
    for seed in model_seeds:
        rows56.extend(_table56_rows_for_seed(df, model_seed=int(seed)))
        write_table(pd.DataFrame(rows56), raw56_csv)
        rows7.extend(_table7_rows_for_seed(df, model_seed=int(seed)))
        write_table(pd.DataFrame(rows7), raw7_csv)

    raw56 = pd.DataFrame(rows56)
    raw7 = pd.DataFrame(rows7)
    summary56 = _summarise_metric_rows(raw56, group_cols=["model", "feature_set", "sqis"])
    summary7 = _summarise_metric_rows(raw7, group_cols=["model", "feature_set", "sqis"])
    summary56_csv = out / "table56_model_seed_summary.csv"
    summary7_csv = out / "table7_model_seed_summary.csv"
    write_table(summary56, summary56_csv)
    write_table(summary7, summary7_csv)
    return {
        "outputs": [str(raw56_csv), str(summary56_csv), str(raw7_csv), str(summary7_csv)],
        "integrity": integrity,
        "n_table56_rows": int(len(raw56)),
        "n_table7_rows": int(len(raw7)),
    }
