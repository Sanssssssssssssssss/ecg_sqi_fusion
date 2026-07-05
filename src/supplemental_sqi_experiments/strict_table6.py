from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    SELECTED_FIVE,
    SQIS,
    binary_metrics,
    feature_cols_for_sqis,
    fit_fixed_svm,
    load_split_frame,
    max_accuracy_threshold,
    predict_score,
    source_bootstrap_ci,
    split_arrays,
    validate_integrity,
    write_json,
    write_table,
)
from .plotting import plot_strict_table6


def _subset_id(sqis: tuple[str, ...]) -> str:
    return "+".join(sqis)


def _fit_select_one(
    df: pd.DataFrame,
    sqis: tuple[str, ...],
    *,
    C: float,
    gamma: float,
    seed: int,
) -> dict[str, Any]:
    cols = feature_cols_for_sqis(sqis)
    tr = df["split"].astype(str).eq("train").to_numpy()
    va = df["split"].astype(str).eq("val").to_numpy()
    Xtr = df.loc[tr, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    t0 = time.perf_counter()
    model = fit_fixed_svm(Xtr, ytr, C=C, gamma=gamma, seed=seed)
    p_val = predict_score(model, Xva)
    train_s = time.perf_counter() - t0
    thr = max_accuracy_threshold(yva, p_val)
    met = binary_metrics(yva, p_val, thr["threshold"])
    return {
        "subset_id": _subset_id(sqis),
        "sqis": ",".join(sqis),
        "cardinality": int(len(sqis)),
        "n_features": int(len(cols)),
        "C": float(C),
        "gamma": float(gamma),
        "threshold_val_maxacc": float(thr["threshold"]),
        "val_Ac": float(met["Ac"]),
        "val_Se": float(met["Se"]),
        "val_Sp": float(met["Sp"]),
        "val_AUC": float(met["AUC"]),
        "val_tn": int(met["tn"]),
        "val_fp": int(met["fp"]),
        "val_fn": int(met["fn"]),
        "val_tp": int(met["tp"]),
        "train_time_s": float(train_s),
    }


def _test_selected(
    df: pd.DataFrame,
    row: pd.Series,
    *,
    C: float,
    gamma: float,
    seed: int,
    probs_dir: Path,
) -> dict[str, Any]:
    sqis = tuple(str(row["sqis"]).split(","))
    cols = feature_cols_for_sqis(sqis)
    trv = df["split"].astype(str).isin(["train", "val"]).to_numpy()
    te = df["split"].astype(str).eq("test").to_numpy()
    Xtrv = df.loc[trv, cols].to_numpy(dtype=np.float64)
    ytrv = df.loc[trv, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, cols].to_numpy(dtype=np.float64)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)
    model = fit_fixed_svm(Xtrv, ytrv, C=C, gamma=gamma, seed=seed)
    p_test = predict_score(model, Xte)
    threshold = float(row["threshold_val_maxacc"])
    met = binary_metrics(yte, p_test, threshold)
    key = f"card{int(row['cardinality'])}_{str(row['subset_id']).replace('+', '-')}"
    probs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probs_dir / f"{key}_seed{seed}.npz",
        y01_test=yte.astype(np.int32),
        p_test=p_test.astype(np.float64),
        threshold=np.array(threshold, dtype=np.float64),
        sqis=np.array(sqis, dtype=object),
    )
    test_df = df.loc[te, ["record_id", "source_record_id", "y", "y01", "sample_group"]].copy()
    test_df["score"] = p_test
    ci = source_bootstrap_ci(test_df, "score", threshold=threshold, n_boot=2000, seed=seed)
    ci_wide = {
        f"{r.metric}_ci_low": float(r.ci_low)
        for r in ci.itertuples(index=False)
    } | {
        f"{r.metric}_ci_high": float(r.ci_high)
        for r in ci.itertuples(index=False)
    }
    return {
        "cardinality": int(row["cardinality"]),
        "subset_id": str(row["subset_id"]),
        "Selected_SQI": str(row["sqis"]),
        "n_features": int(row["n_features"]),
        "C": float(C),
        "gamma": float(gamma),
        "threshold": threshold,
        "val_Ac": float(row["val_Ac"]),
        "val_Se": float(row["val_Se"]),
        "val_Sp": float(row["val_Sp"]),
        "val_AUC": float(row["val_AUC"]),
        "test_Ac": float(met["Ac"]),
        "test_Se": float(met["Se"]),
        "test_Sp": float(met["Sp"]),
        "test_AUC": float(met["AUC"]),
        "test_tn": int(met["tn"]),
        "test_fp": int(met["fp"]),
        "test_fn": int(met["fn"]),
        "test_tp": int(met["tp"]),
        **ci_wide,
    }


def _inclusion_frequency(all_val: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for card, g in all_val.groupby("cardinality", sort=True):
        ranked = g.sort_values(["val_Ac", "val_AUC", "subset_id"], ascending=[False, False, True]).head(top_k)
        for sqi in SQIS:
            rows.append(
                {
                    "rank_scope": "top10_by_cardinality",
                    "cardinality": int(card),
                    "SQI": sqi,
                    "inclusion_frequency": float(ranked["sqis"].str.split(",").apply(lambda xs: sqi in xs).mean()),
                    "n_ranked_subsets": int(len(ranked)),
                }
            )
    ranked_all = all_val.sort_values(["val_Ac", "val_AUC", "subset_id"], ascending=[False, False, True]).head(top_k)
    for sqi in SQIS:
        rows.append(
            {
                "rank_scope": "top10_overall",
                "cardinality": -1,
                "SQI": sqi,
                "inclusion_frequency": float(ranked_all["sqis"].str.split(",").apply(lambda xs: sqi in xs).mean()),
                "n_ranked_subsets": int(len(ranked_all)),
            }
        )
    return pd.DataFrame(rows)


def _marginal_effect(all_val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sqi in SQIS:
        present = all_val["sqis"].str.split(",").apply(lambda xs: sqi in xs)
        for card in [-1] + sorted(all_val["cardinality"].unique().tolist()):
            if card == -1:
                g = all_val
                card_label = "all"
            else:
                g = all_val[all_val["cardinality"].eq(card)]
                card_label = str(card)
            p = present.loc[g.index]
            if p.nunique() < 2:
                continue
            rows.append(
                {
                    "SQI": sqi,
                    "cardinality": card_label,
                    "mean_val_Ac_with": float(g.loc[p, "val_Ac"].mean()),
                    "mean_val_Ac_without": float(g.loc[~p, "val_Ac"].mean()),
                    "delta_val_Ac": float(g.loc[p, "val_Ac"].mean() - g.loc[~p, "val_Ac"].mean()),
                    "mean_val_AUC_with": float(g.loc[p, "val_AUC"].mean()),
                    "mean_val_AUC_without": float(g.loc[~p, "val_AUC"].mean()),
                    "delta_val_AUC": float(g.loc[p, "val_AUC"].mean() - g.loc[~p, "val_AUC"].mean()),
                }
            )
    return pd.DataFrame(rows)


def run_strict_table6(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    seed: int = 0,
    C: float = 1.0,
    gamma: float = 0.14,
    force: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    all_csv = out / "all_127_subset_val.csv"
    selected_csv = out / "selected_by_cardinality_test.csv"
    if all_csv.exists() and selected_csv.exists() and not force:
        existing = [
            all_csv,
            selected_csv,
            out / "sqi_inclusion_frequency.csv",
            out / "marginal_inclusion_effect.csv",
            out / "top10_validation_subsets_by_cardinality.csv",
            out / "subset_identity_match.json",
            Path(report_dir) / "fig_supp_01_strict_table6_subset_selection.svg",
            Path(report_dir) / "fig_supp_01_strict_table6_subset_selection.pdf",
            Path(report_dir) / "fig_supp_01_strict_table6_subset_selection.png",
        ]
        return {"skipped": True, "outputs": [str(p) for p in existing if p.exists()]}

    df = load_split_frame(artifacts_dir, normalized=True)
    integrity = validate_integrity(df)
    rows = []
    for r in range(1, len(SQIS) + 1):
        for sqis in itertools.combinations(SQIS, r):
            rows.append(_fit_select_one(df, sqis, C=C, gamma=gamma, seed=seed))
    all_val = pd.DataFrame(rows).sort_values(["cardinality", "val_Ac", "val_AUC", "subset_id"], ascending=[True, False, False, True])
    write_table(all_val, all_csv, md_path=rep / "all_127_subset_val.md")

    selected_rows = []
    probs_dir = out / "selected_subset_probs"
    for card, g in all_val.groupby("cardinality", sort=True):
        selected = g.sort_values(["val_Ac", "val_AUC", "subset_id"], ascending=[False, False, True]).iloc[0]
        selected_rows.append(_test_selected(df, selected, C=C, gamma=gamma, seed=seed, probs_dir=probs_dir))
    selected_test = pd.DataFrame(selected_rows)
    write_table(selected_test, selected_csv, md_path=rep / "selected_by_cardinality_test.md")

    inclusion = _inclusion_frequency(all_val)
    inclusion_csv = out / "sqi_inclusion_frequency.csv"
    write_table(inclusion, inclusion_csv, md_path=rep / "sqi_inclusion_frequency.md")
    marginal = _marginal_effect(all_val)
    marginal_csv = out / "marginal_inclusion_effect.csv"
    write_table(marginal, marginal_csv, md_path=rep / "marginal_inclusion_effect.md")

    topk = (
        all_val.sort_values(["val_Ac", "val_AUC", "subset_id"], ascending=[False, False, True])
        .groupby("cardinality", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )
    topk_csv = out / "top10_validation_subsets_by_cardinality.csv"
    write_table(topk, topk_csv, md_path=rep / "top10_validation_subsets_by_cardinality.md")

    best5 = selected_test[selected_test["cardinality"].eq(5)].iloc[0].to_dict()
    identity = {
        "paper_selected_five": SELECTED_FIVE,
        "validation_recovered_cardinality5": str(best5["Selected_SQI"]).split(","),
        "matches_paper_selected_five_as_set": set(str(best5["Selected_SQI"]).split(",")) == set(SELECTED_FIVE),
        "matches_paper_selected_five_ordered": str(best5["Selected_SQI"]).split(",") == SELECTED_FIVE,
        "svm_C": float(C),
        "svm_gamma": float(gamma),
        "protocol": "All 127 non-empty SQI subsets fit on train; validation selects one subset per cardinality; selected subsets refit on train+val and evaluated once on test.",
        "integrity": integrity,
    }
    identity_json = write_json(out / "subset_identity_match.json", identity)
    plot_paths = plot_strict_table6(selected_test, inclusion, rep / "fig_supp_01_strict_table6_subset_selection")
    return {
        "skipped": False,
        "outputs": [
            str(all_csv),
            str(selected_csv),
            str(inclusion_csv),
            str(marginal_csv),
            str(topk_csv),
            str(identity_json),
            *[str(p) for p in plot_paths],
        ],
        "identity": identity,
    }
