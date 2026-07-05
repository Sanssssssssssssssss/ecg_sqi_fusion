from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.sqi_pipeline.models.lm_mlp import LMConfig, LMMLP
from src.sqi_pipeline.models.lm_mlp_search import compute_metrics, find_maxacc_threshold

from .common import (
    SELECTED_FIVE,
    binary_metrics,
    feature_cols_for_sqis,
    fit_fixed_svm,
    load_split_frame,
    max_accuracy_threshold,
    predict_score,
    write_table,
)


DEFAULT_C_GRID = (1.0, 2.0, 8.0, 25.0, 32.0, 128.0, 512.0)
DEFAULT_GAMMA_GRID = (2.0**-11, 2.0**-9, 2.0**-7, 2.0**-5, 2.0**-3, 0.14, 0.5, 0.7, 1.0, 1.5, 2.0)


def make_source_grouped_split(df: pd.DataFrame, seed: int) -> np.ndarray:
    groups = []
    for source, g in df.groupby("source_record_id", sort=True):
        n_aug = int(g["is_augmented"].astype(int).sum()) if "is_augmented" in g else 0
        n_good = int((g["y"].astype(int) == 1).sum())
        if n_aug > 0:
            group_type = "acceptable_with_noise"
        elif n_good > 0:
            group_type = "acceptable_clean_only"
        else:
            group_type = "unacceptable_original"
        groups.append({"source_record_id": str(source), "group_type": group_type})
    group_df = pd.DataFrame(groups)
    rng = np.random.default_rng(seed)
    split_map: dict[str, str] = {}
    for group_type, g in group_df.groupby("group_type", sort=True):
        ids = g["source_record_id"].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        for source in ids[:n_train]:
            split_map[str(source)] = "train"
        for source in ids[n_train : n_train + n_val]:
            split_map[str(source)] = "val"
        for source in ids[n_train + n_val :]:
            split_map[str(source)] = "test"
    return df["source_record_id"].astype(str).map(split_map).astype(str).to_numpy()


def _split_masks(split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split == "train", split == "val", split == "test"


def run_svm_one_split(
    df: pd.DataFrame,
    split: np.ndarray,
    *,
    split_seed: int,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    gamma_grid: tuple[float, ...] = DEFAULT_GAMMA_GRID,
) -> dict[str, Any]:
    cols = feature_cols_for_sqis(SELECTED_FIVE)
    tr, va, te = _split_masks(split)
    Xtr = df.loc[tr, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    best: dict[str, Any] | None = None
    for C, gamma in itertools.product(C_grid, gamma_grid):
        model = fit_fixed_svm(Xtr, ytr, C=float(C), gamma=float(gamma), seed=split_seed)
        p_val = predict_score(model, Xva)
        thr = max_accuracy_threshold(yva, p_val)
        score = binary_metrics(yva, p_val, thr["threshold"])
        row = {
            "best_C": float(C),
            "best_gamma": float(gamma),
            "threshold": float(thr["threshold"]),
            "val_Ac": float(score["Ac"]),
            "val_AUC": float(score["AUC"]),
        }
        if best is None or (row["val_AUC"], row["val_Ac"], -row["best_C"], -row["best_gamma"]) > (
            best["val_AUC"],
            best["val_Ac"],
            -best["best_C"],
            -best["best_gamma"],
        ):
            best = row
    assert best is not None
    trv = tr | va
    model = fit_fixed_svm(
        df.loc[trv, cols].to_numpy(dtype=np.float64),
        df.loc[trv, "y01"].to_numpy(dtype=int),
        C=float(best["best_C"]),
        gamma=float(best["best_gamma"]),
        seed=split_seed,
    )
    p_test = predict_score(model, df.loc[te, cols].to_numpy(dtype=np.float64))
    met = binary_metrics(df.loc[te, "y01"].to_numpy(dtype=int), p_test, float(best["threshold"]))
    return {
        "model": "SVM",
        "split_seed": int(split_seed),
        "mlp_init_seed": np.nan,
        "best_J": np.nan,
        **best,
        "test_Ac": float(met["Ac"]),
        "test_Se": float(met["Se"]),
        "test_Sp": float(met["Sp"]),
        "test_AUC": float(met["AUC"]),
        "test_n": int(met["n"]),
    }


def run_mlp_one_split(
    df: pd.DataFrame,
    split: np.ndarray,
    *,
    split_seed: int,
    init_seed: int,
    j_range: range = range(2, 18),
) -> dict[str, Any]:
    cols = feature_cols_for_sqis(SELECTED_FIVE)
    tr, va, te = _split_masks(split)
    Xtr = df.loc[tr, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, cols].to_numpy(dtype=np.float64)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)
    device = torch.device("cpu")
    dtype = torch.float64
    cfg = LMConfig(epochs_max=100, stop_error=1e-5, stop_grad=1e-5)
    Xt = torch.tensor(Xtr, device=device, dtype=dtype)
    yt = torch.tensor(ytr, device=device, dtype=dtype)
    Xv = torch.tensor(Xva, device=device, dtype=dtype)
    yv = torch.tensor(yva, device=device, dtype=dtype)
    rows = []
    model_seed = int(split_seed * 1000 + init_seed)
    for J in j_range:
        m = LMMLP(J=int(J), D=len(cols), device=device, dtype=dtype, seed=model_seed)
        m.fit_lm(Xt, yt, cfg, X_val=Xv, y_val=yv, model_select_metric="val_acc", patience=101, threshold=0.7)
        p_val = m.predict_proba(Xv)
        met = compute_metrics(yva, p_val, threshold=0.7)
        rows.append({"J": int(J), "val_Ac": float(met["acc"]), "val_AUC": float(met["auc"])})
    ranked = sorted(rows, key=lambda r: (r["val_Ac"], r["val_AUC"], -r["J"]), reverse=True)
    best_J = int(ranked[0]["J"])
    probe = LMMLP(J=best_J, D=len(cols), device=device, dtype=dtype, seed=model_seed)
    probe.fit_lm(Xt, yt, cfg, X_val=Xv, y_val=yv, model_select_metric="val_acc", patience=101, threshold=0.7)
    p_val = probe.predict_proba(Xv)
    thr = find_maxacc_threshold(yva, p_val)
    Xtrv = np.concatenate([Xtr, Xva], axis=0)
    ytrv = np.concatenate([ytr, yva], axis=0)
    final = LMMLP(J=best_J, D=len(cols), device=device, dtype=dtype, seed=model_seed)
    final.fit_lm(
        torch.tensor(Xtrv, device=device, dtype=dtype),
        torch.tensor(ytrv, device=device, dtype=dtype),
        cfg,
        X_val=None,
        y_val=None,
        model_select_metric="val_acc",
        patience=101,
        threshold=float(thr["threshold"]),
    )
    p_test = final.predict_proba(torch.tensor(Xte, device=device, dtype=dtype))
    met = compute_metrics(yte, p_test, threshold=float(thr["threshold"]))
    return {
        "model": "MLP",
        "split_seed": int(split_seed),
        "mlp_init_seed": int(init_seed),
        "best_J": int(best_J),
        "best_C": np.nan,
        "best_gamma": np.nan,
        "threshold": float(thr["threshold"]),
        "val_Ac": float(thr["acc"]),
        "val_AUC": float(roc_auc_safe(yva, p_val)),
        "test_Ac": float(met["acc"]),
        "test_Se": float(met["se"]),
        "test_Sp": float(met["sp"]),
        "test_AUC": float(met["auc"]),
        "test_n": int(len(yte)),
    }


def roc_auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def summarise_stability(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = ["test_Ac", "test_Se", "test_Sp", "test_AUC", "threshold", "best_J"]
    out = []
    for model, g in rows.groupby("model", sort=True):
        for metric in metrics:
            if metric not in g:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            out.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "n": int(len(vals)),
                }
            )
    return pd.DataFrame(out)


def run_stability(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    split_seeds: list[int],
    mlp_init_seeds: list[int],
    include_mlp: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    df = load_split_frame(artifacts_dir, normalized=True)
    rows = []
    for split_seed in split_seeds:
        split = make_source_grouped_split(df, split_seed)
        rows.append(run_svm_one_split(df, split, split_seed=split_seed))
        if include_mlp:
            for init_seed in mlp_init_seeds:
                rows.append(run_mlp_one_split(df, split, split_seed=split_seed, init_seed=init_seed))
        partial = pd.DataFrame(rows)
        partial.to_csv(out / "multi_seed_stability_raw.csv", index=False)
    raw = pd.DataFrame(rows)
    raw_csv = out / "multi_seed_stability_raw.csv"
    write_table(raw, raw_csv, md_path=rep / "multi_seed_stability_raw.md")
    summary = summarise_stability(raw)
    summary_csv = out / "multi_seed_stability_summary.csv"
    write_table(summary, summary_csv, md_path=rep / "multi_seed_stability_summary.md")
    return {"outputs": [str(raw_csv), str(summary_csv)]}
