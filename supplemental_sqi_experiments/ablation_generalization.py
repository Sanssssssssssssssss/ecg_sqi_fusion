from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _eval_svm_protocol(df: pd.DataFrame, sqis: list[str], train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray, *, C: float, gamma: float, seed: int) -> dict[str, Any]:
    cols = feature_cols_for_sqis(sqis)
    Xtr = df.loc[train_mask, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[train_mask, "y01"].to_numpy(dtype=int)
    Xva = df.loc[val_mask, cols].to_numpy(dtype=np.float64)
    yva = df.loc[val_mask, "y01"].to_numpy(dtype=int)
    Xte = df.loc[test_mask, cols].to_numpy(dtype=np.float64)
    yte = df.loc[test_mask, "y01"].to_numpy(dtype=int)
    select_model = fit_fixed_svm(Xtr, ytr, C=C, gamma=gamma, seed=seed)
    p_val = predict_score(select_model, Xva)
    thr = max_accuracy_threshold(yva, p_val)
    fit_mask = train_mask | val_mask
    final = fit_fixed_svm(df.loc[fit_mask, cols].to_numpy(dtype=np.float64), df.loc[fit_mask, "y01"].to_numpy(dtype=int), C=C, gamma=gamma, seed=seed)
    p_test = predict_score(final, Xte)
    met = binary_metrics(yte, p_test, float(thr["threshold"]))
    return {"threshold": float(thr["threshold"]), "val_Ac": float(thr["Ac"]), **{f"test_{k}": v for k, v in met.items() if k in {"Ac", "Se", "Sp", "AUC", "tn", "fp", "fn", "tp", "n"}}}


def run_leave_one_out(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    C: float = 1.0,
    gamma: float = 0.14,
    seed: int = 0,
) -> dict[str, Any]:
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    df = load_split_frame(artifacts_dir, normalized=True)
    tr = df["split"].astype(str).eq("train").to_numpy()
    va = df["split"].astype(str).eq("val").to_numpy()
    te = df["split"].astype(str).eq("test").to_numpy()
    rows = []
    base = _eval_svm_protocol(df, SELECTED_FIVE, tr, va, te, C=C, gamma=gamma, seed=seed)
    rows.append({"model": "SVM", "setting": "selected-five", "removed_sqi": "", "sqis": ",".join(SELECTED_FIVE), **base})
    for removed in SELECTED_FIVE:
        sqis = [s for s in SELECTED_FIVE if s != removed]
        res = _eval_svm_protocol(df, sqis, tr, va, te, C=C, gamma=gamma, seed=seed)
        rows.append({"model": "SVM", "setting": "leave-one-out", "removed_sqi": removed, "sqis": ",".join(sqis), **res})
    tab = pd.DataFrame(rows)
    base_acc = float(tab.loc[tab["setting"].eq("selected-five"), "test_Ac"].iloc[0])
    tab["delta_test_Ac_vs_selected5"] = tab["test_Ac"] - base_acc
    csv = out / "leave_one_sqi_out_svm.csv"
    write_table(tab, csv, md_path=rep / "leave_one_sqi_out_svm.md")
    return {"outputs": [str(csv)]}


def run_cross_noise_generalization(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    C: float = 1.0,
    gamma: float = 0.14,
    seed: int = 0,
) -> dict[str, Any]:
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    df = load_split_frame(artifacts_dir, normalized=True)
    split = df["split"].astype(str)
    acceptable = df["sample_group"].eq("original acceptable")
    scenarios = [
        ("train_em_test_ma", df["sample_group"].isin(["original acceptable", "synthetic em"]), df["sample_group"].isin(["original acceptable", "synthetic em"]), df["sample_group"].isin(["original acceptable", "synthetic ma"])),
        ("train_ma_test_em", df["sample_group"].isin(["original acceptable", "synthetic ma"]), df["sample_group"].isin(["original acceptable", "synthetic ma"]), df["sample_group"].isin(["original acceptable", "synthetic em"])),
        ("synthetic_poor_to_original_poor", df["sample_group"].isin(["original acceptable", "synthetic em", "synthetic ma"]), df["sample_group"].isin(["original acceptable", "synthetic em", "synthetic ma"]), df["sample_group"].isin(["original acceptable", "original unacceptable"])),
        ("original_poor_to_synthetic_poor", df["sample_group"].isin(["original acceptable", "original unacceptable"]), df["sample_group"].isin(["original acceptable", "original unacceptable"]), df["sample_group"].isin(["original acceptable", "synthetic em", "synthetic ma"])),
    ]
    rows = []
    for name, train_pool, val_pool, test_pool in scenarios:
        tr = split.eq("train").to_numpy() & train_pool.to_numpy()
        va = split.eq("val").to_numpy() & val_pool.to_numpy()
        te = split.eq("test").to_numpy() & test_pool.to_numpy()
        if len(np.unique(df.loc[tr, "y01"])) < 2 or len(np.unique(df.loc[va, "y01"])) < 2 or len(np.unique(df.loc[te, "y01"])) < 2:
            rows.append({"scenario": name, "skipped": True, "reason": "one class missing"})
            continue
        res = _eval_svm_protocol(df, SELECTED_FIVE, tr, va, te, C=C, gamma=gamma, seed=seed)
        rows.append(
            {
                "scenario": name,
                "skipped": False,
                "train_n": int(tr.sum()),
                "val_n": int(va.sum()),
                "test_n": int(te.sum()),
                "sqis": ",".join(SELECTED_FIVE),
                **res,
            }
        )
    tab = pd.DataFrame(rows)
    csv = out / "cross_noise_generalization_svm.csv"
    write_table(tab, csv, md_path=rep / "cross_noise_generalization_svm.md")
    return {"outputs": [str(csv)]}
