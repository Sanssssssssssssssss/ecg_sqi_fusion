from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .ablation_generalization import _eval_svm_protocol
from .common import (
    GROUP_ORDER,
    LEADS_12,
    SELECTED_FIVE,
    SQIS,
    binary_metrics,
    feature_cols_for_sqis,
    fit_fixed_svm,
    load_split_frame,
    max_accuracy_threshold,
    predict_score,
    write_table,
)
from .plotting import (
    plot_bassqi_domain_shift,
    plot_domain_shift,
    plot_fsqi_mechanism_updated,
    plot_sqi_subgroup_separability,
)


PAPER_SELECTED_FIVE_SET = set(SELECTED_FIVE)
THRESHOLDS_MV = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], dtype=float)
POOR_DOMAIN_GROUPS = {
    "original poor": ["original unacceptable"],
    "em": ["synthetic em"],
    "ma": ["synthetic ma"],
}
SUBSET_MODELS = {
    "iSQI": ["iSQI"],
    "basSQI": ["basSQI"],
    "paper pair": ["iSQI", "basSQI"],
    "paper quintuplet": SELECTED_FIVE,
    "all seven": SQIS,
}


def _subset_set(value: str) -> set[str]:
    return {x.strip() for x in str(value).split(",") if x.strip()}


def summarize_paper_quintuplet(strict_dir: str | Path, out_dir: str | Path, report_dir: str | Path) -> list[str]:
    strict = Path(strict_dir)
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    all_val = pd.read_csv(strict / "all_127_subset_val.csv")
    five = all_val[all_val["cardinality"].eq(5)].copy()
    five = five.sort_values(["val_Ac", "val_AUC", "subset_id"], ascending=[False, False, True]).reset_index(drop=True)
    five["validation_rank"] = np.arange(1, len(five) + 1)
    five["is_paper_quintuplet"] = five["sqis"].map(_subset_set).map(lambda s: s == PAPER_SELECTED_FIVE_SET)
    if not five["is_paper_quintuplet"].any():
        raise RuntimeError("Paper selected-five subset was not found in the strict 127-subset validation table.")

    paper = five[five["is_paper_quintuplet"]].iloc[0]
    best = five.iloc[0]
    n_val = int(paper["val_tn"] + paper["val_fp"] + paper["val_fn"] + paper["val_tp"])
    one_error = 1.0 / max(1, n_val)
    plateau = five[five["val_Ac"] >= float(best["val_Ac"]) - one_error].copy()

    rank = pd.DataFrame(
        [
            {
                "paper_selected_five": ",".join(SELECTED_FIVE),
                "paper_subset_id_in_strict_table": str(paper["subset_id"]),
                "validation_rank_among_21_five_sqi_subsets": int(paper["validation_rank"]),
                "n_five_sqi_subsets": int(len(five)),
                "paper_val_Ac": float(paper["val_Ac"]),
                "paper_val_AUC": float(paper["val_AUC"]),
                "best_five_subset": str(best["sqis"]),
                "best_five_val_Ac": float(best["val_Ac"]),
                "best_five_val_AUC": float(best["val_AUC"]),
                "paper_gap_to_best_val_Ac": float(best["val_Ac"] - paper["val_Ac"]),
                "paper_gap_to_best_val_Ac_pp": float((best["val_Ac"] - paper["val_Ac"]) * 100.0),
                "one_validation_error": float(one_error),
                "one_validation_error_pp": float(one_error * 100.0),
                "within_one_validation_error_of_best": bool(float(best["val_Ac"] - paper["val_Ac"]) <= one_error + 1e-12),
            }
        ]
    )
    plateau_summary = pd.DataFrame(
        [
            {
                "cardinality": 5,
                "n_validation_records": n_val,
                "best_val_Ac": float(best["val_Ac"]),
                "one_validation_error": float(one_error),
                "n_subsets_within_one_error": int(len(plateau)),
                "fraction_subsets_within_one_error": float(len(plateau) / max(1, len(five))),
                "paper_subset_rank": int(paper["validation_rank"]),
                "paper_gap_to_best_val_Ac": float(best["val_Ac"] - paper["val_Ac"]),
                "interpretation": "flat_plateau" if float(best["val_Ac"] - paper["val_Ac"]) <= one_error + 1e-12 else "changed_selection_structure",
            }
        ]
    )

    rank_csv = out / "table6_paper_quintuplet_rank.csv"
    plateau_csv = out / "table6_five_sqi_plateau_summary.csv"
    write_table(rank, rank_csv, md_path=rep / "table6_paper_quintuplet_rank.md")
    write_table(plateau_summary, plateau_csv, md_path=rep / "table6_five_sqi_plateau_summary.md")
    five_out = five[["validation_rank", "subset_id", "sqis", "val_Ac", "val_AUC", "threshold_val_maxacc", "is_paper_quintuplet"]]
    write_table(five_out, out / "table6_all_five_sqi_validation_rank.csv", md_path=rep / "table6_all_five_sqi_validation_rank.md")
    return [str(rank_csv), str(plateau_csv), str(out / "table6_all_five_sqi_validation_rank.csv")]


def _domain_frame(df: pd.DataFrame, sqis: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    cols = feature_cols_for_sqis(sqis or SQIS)
    poor = df[df["sample_group"].isin(["original unacceptable", "synthetic em", "synthetic ma"])].copy()
    poor["domain01"] = poor["sample_group"].isin(["synthetic em", "synthetic ma"]).astype(int)
    return poor, cols


def _grouped_domain_auc(df: pd.DataFrame, cols: list[str], *, seed: int = 0, n_splits: int = 5) -> tuple[float, np.ndarray]:
    y = df["domain01"].to_numpy(dtype=int)
    groups = df["source_record_id"].astype(str).to_numpy()
    X = df[cols].to_numpy(dtype=np.float64)
    n_splits = min(int(n_splits), int(np.bincount(y).min()), len(np.unique(groups)))
    if n_splits < 2:
        return float("nan"), np.full(len(df), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score = np.full(len(df), np.nan, dtype=float)
    for train_idx, test_idx in cv.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=4000, solver="lbfgs", random_state=seed)),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        score[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    ok = np.isfinite(score)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan"), score
    return float(roc_auc_score(y[ok], score[ok])), score


def _source_group_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 0,
    n_boot: int = 2000,
) -> tuple[float, float, int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups, dtype=str)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    groups = groups[ok]
    unique = np.array(sorted(pd.unique(groups)))
    group_to_idx = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(int(n_boot)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], score[idx])))
    if not vals:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def _mmd_rbf_permutation(X0: np.ndarray, X1: np.ndarray, *, seed: int = 0, n_perm: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    X = np.vstack([X0, X1]).astype(np.float64, copy=False)
    n0 = int(len(X0))
    n1 = int(len(X1))
    d2 = squareform(pdist(X, metric="sqeuclidean"))
    upper = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    gamma = 1.0 / max(2.0 * med, 1e-12)
    K = np.exp(-gamma * d2)

    def mmd2(idx0: np.ndarray, idx1: np.ndarray) -> float:
        k00 = K[np.ix_(idx0, idx0)].mean()
        k11 = K[np.ix_(idx1, idx1)].mean()
        k01 = K[np.ix_(idx0, idx1)].mean()
        return float(k00 + k11 - 2.0 * k01)

    idx = np.arange(n0 + n1)
    obs = mmd2(idx[:n0], idx[n0:])
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        perm = rng.permutation(idx)
        null[i] = mmd2(perm[:n0], perm[n0:])
    p = float((1.0 + np.sum(null >= obs)) / (len(null) + 1.0))
    return {
        "comparison": "original poor vs synthetic poor",
        "metric": "RBF-MMD2",
        "estimate": obs,
        "p_value_permutation": p,
        "n_permutations": int(n_perm),
        "kernel_gamma_median_heuristic": gamma,
        "n_original_poor": n0,
        "n_synthetic_poor": n1,
    }


def compute_domain_shift(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    seed: int = 0,
    C: float = 1.0,
    gamma: float = 0.14,
    n_perm: int = 1000,
) -> list[str]:
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    df = load_split_frame(artifacts_dir, normalized=True)
    cols84 = feature_cols_for_sqis(SQIS)
    train = df["split"].astype(str).eq("train").to_numpy()
    test = df["split"].astype(str).eq("test").to_numpy()

    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=seed)
    X_train = scaler.fit_transform(df.loc[train, cols84].to_numpy(dtype=np.float64))
    pca.fit(X_train)
    X_test = scaler.transform(df.loc[test, cols84].to_numpy(dtype=np.float64))
    scores = pca.transform(X_test)
    pca_df = df.loc[test, ["record_id", "source_record_id", "y", "y01", "split", "sample_group"]].copy()
    pca_df["PC1"] = scores[:, 0]
    pca_df["PC2"] = scores[:, 1]
    pca_df["explained_variance_PC1"] = float(pca.explained_variance_ratio_[0])
    pca_df["explained_variance_PC2"] = float(pca.explained_variance_ratio_[1])
    pca_csv = out / "domain_shift_pca_test_projection.csv"
    write_table(pca_df, pca_csv)

    poor, domain_cols = _domain_frame(df, SQIS)
    overall_auc, oof_score = _grouped_domain_auc(poor, domain_cols, seed=seed)
    poor_scores = poor[["record_id", "source_record_id", "sample_group", "domain01"]].copy()
    poor_scores["domain_score_synthetic"] = oof_score
    write_table(poor_scores, out / "domain_classifier_oof_scores.csv")

    X_poor = StandardScaler().fit_transform(poor[domain_cols].to_numpy(dtype=np.float64))
    X0 = X_poor[poor["domain01"].eq(0).to_numpy()]
    X1 = X_poor[poor["domain01"].eq(1).to_numpy()]
    mmd = _mmd_rbf_permutation(X0, X1, seed=seed, n_perm=n_perm)
    metrics = pd.DataFrame(
        [
            {
                "metric": "source_grouped_logistic_domain_auc",
                "estimate": float(overall_auc),
                "comparison": "original poor vs synthetic poor",
                "bootstrap_or_test": "StratifiedGroupKFold by source_record_id",
                "n_original_poor": int((poor["domain01"] == 0).sum()),
                "n_synthetic_poor": int((poor["domain01"] == 1).sum()),
            },
            mmd,
        ]
    )
    metrics_csv = out / "domain_shift_metrics.csv"
    write_table(metrics, metrics_csv, md_path=rep / "domain_shift_metrics.md")

    per_rows = []
    for sqi in SQIS:
        d, cols = _domain_frame(df, [sqi])
        auc, sqi_score = _grouped_domain_auc(d, cols, seed=seed)
        ci_low, ci_high, n_valid = _source_group_auc_ci(
            d["domain01"].to_numpy(dtype=int),
            sqi_score,
            d["source_record_id"].astype(str).to_numpy(),
            seed=seed + 101,
        )
        per_rows.append(
            {
                "SQI": sqi,
                "n_features": len(cols),
                "domain_auc_original_vs_synthetic_poor": float(auc),
                "domain_auc_ci_low": ci_low,
                "domain_auc_ci_high": ci_high,
                "bootstrap_unit": "source_record_id",
                "n_bootstrap_valid": n_valid,
            }
        )
    per_sqi = pd.DataFrame(per_rows)
    per_sqi_csv = out / "per_sqi_domain_auc.csv"
    write_table(per_sqi, per_sqi_csv, md_path=rep / "per_sqi_domain_auc.md")

    cross = _cross_domain_auc_matrix(df, C=C, gamma=gamma, seed=seed)
    cross_csv = out / "cross_domain_auc_matrix.csv"
    write_table(cross, cross_csv, md_path=rep / "cross_domain_auc_matrix.md")

    paths = plot_domain_shift(pca_df, per_sqi, cross, metrics, rep / "fig_13_sqi_domain_shift")
    outputs = [str(pca_csv), str(metrics_csv), str(per_sqi_csv), str(cross_csv), *[str(p) for p in paths]]
    if float(per_sqi.loc[per_sqi["SQI"].eq("basSQI"), "domain_auc_original_vs_synthetic_poor"].iloc[0]) >= 0.80:
        bassqi_outputs = _bassqi_outputs(df, out, rep, C=C, gamma=gamma, seed=seed)
        outputs.extend(bassqi_outputs)
    outputs.extend(compute_sqi_subgroup_separability(df=df, out_dir=out, report_dir=rep, C=C, gamma=gamma, seed=seed))
    return outputs


def _cross_domain_auc_matrix(df: pd.DataFrame, *, C: float, gamma: float, seed: int) -> pd.DataFrame:
    split = df["split"].astype(str)
    train_domains = {
        "original poor": ["original unacceptable"],
        "em": ["synthetic em"],
        "ma": ["synthetic ma"],
        "synthetic poor": ["synthetic em", "synthetic ma"],
    }
    test_domains = {
        "original poor": ["original unacceptable"],
        "em": ["synthetic em"],
        "ma": ["synthetic ma"],
    }
    rows: list[dict[str, Any]] = []
    for train_name, train_groups in train_domains.items():
        train_pool = df["sample_group"].isin(["original acceptable", *train_groups])
        for test_name, test_groups in test_domains.items():
            test_pool = df["sample_group"].isin(["original acceptable", *test_groups])
            tr = split.eq("train").to_numpy() & train_pool.to_numpy()
            va = split.eq("val").to_numpy() & train_pool.to_numpy()
            te = split.eq("test").to_numpy() & test_pool.to_numpy()
            if (
                len(np.unique(df.loc[tr, "y01"])) < 2
                or len(np.unique(df.loc[va, "y01"])) < 2
                or len(np.unique(df.loc[te, "y01"])) < 2
            ):
                rows.append({"train_poor_domain": train_name, "test_poor_domain": test_name, "skipped": True})
                continue
            res = _eval_svm_protocol(df, SELECTED_FIVE, tr, va, te, C=C, gamma=gamma, seed=seed)
            rows.append(
                {
                    "train_poor_domain": train_name,
                    "test_poor_domain": test_name,
                    "skipped": False,
                    "train_n": int(tr.sum()),
                    "val_n": int(va.sum()),
                    "test_n": int(te.sum()),
                    "test_Ac": float(res["test_Ac"]),
                    "test_Se": float(res["test_Se"]),
                    "test_Sp": float(res["test_Sp"]),
                    "test_AUC": float(res["test_AUC"]),
                    "threshold": float(res["threshold"]),
                }
            )
    return pd.DataFrame(rows)


def _fit_full_test_svm(df: pd.DataFrame, sqis: list[str], *, C: float, gamma: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cols = feature_cols_for_sqis(sqis)
    split = df["split"].astype(str)
    tr = split.eq("train").to_numpy()
    va = split.eq("val").to_numpy()
    te = split.eq("test").to_numpy()
    select = fit_fixed_svm(df.loc[tr, cols].to_numpy(dtype=np.float64), df.loc[tr, "y01"].to_numpy(dtype=int), C=C, gamma=gamma, seed=seed)
    p_val = predict_score(select, df.loc[va, cols].to_numpy(dtype=np.float64))
    thr = max_accuracy_threshold(df.loc[va, "y01"].to_numpy(dtype=int), p_val)
    final = fit_fixed_svm(df.loc[tr | va, cols].to_numpy(dtype=np.float64), df.loc[tr | va, "y01"].to_numpy(dtype=int), C=C, gamma=gamma, seed=seed)
    p_test = predict_score(final, df.loc[te, cols].to_numpy(dtype=np.float64))
    pred_accept = p_test > float(thr["threshold"])
    met = binary_metrics(df.loc[te, "y01"].to_numpy(dtype=int), p_test, float(thr["threshold"]))
    return p_test, pred_accept, te, {"threshold": float(thr["threshold"]), **met}


def _true_class_recall_rows(
    df_test: pd.DataFrame,
    pred_accept: np.ndarray,
    score: np.ndarray,
    *,
    model: str,
    threshold: float,
) -> list[dict[str, Any]]:
    tmp = df_test[["sample_group", "y01"]].copy().reset_index(drop=True)
    tmp["pred_accept"] = np.asarray(pred_accept, dtype=bool)
    tmp["score"] = np.asarray(score, dtype=float)
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        g = tmp[tmp["sample_group"].eq(group)]
        if g.empty:
            continue
        if int(g["y01"].iloc[0]) == 1:
            value = float(g["pred_accept"].mean())
            metric = "acceptable_specificity"
        else:
            value = float((~g["pred_accept"]).mean())
            metric = "poor_recall"
        rows.append(
            {
                "model": model,
                "sample_group": group,
                "n": int(len(g)),
                "metric": metric,
                "value": value,
                "threshold": float(threshold),
                "score_median": float(g["score"].median()),
            }
        )
    return rows


def _bassqi_outputs(df: pd.DataFrame, out: Path, rep: Path, *, C: float, gamma: float, seed: int) -> list[str]:
    cols = [f"{lead}__basSQI" for lead in LEADS_12]
    tab = df[["record_id", "source_record_id", "split", "sample_group", "y", "y01"]].copy()
    bas = df[cols].to_numpy(dtype=np.float64)
    tab["mean_1_minus_basSQI"] = 1.0 - np.mean(bas, axis=1)
    tab["median_1_minus_basSQI"] = 1.0 - np.median(bas, axis=1)
    tab["max_1_minus_basSQI"] = 1.0 - np.min(bas, axis=1)
    csv = out / "bassqi_domain_shift.csv"
    write_table(tab, csv, md_path=rep / "bassqi_domain_shift.md")

    clean = tab[tab["sample_group"].eq("original acceptable")].set_index("record_id")
    delta_rows: list[dict[str, Any]] = []
    synth = tab[tab["sample_group"].isin(["synthetic em", "synthetic ma"])].copy()
    for row in synth.itertuples(index=False):
        sid = str(row.source_record_id)
        if sid not in clean.index:
            continue
        base = clean.loc[sid]
        noise_type = "em" if row.sample_group == "synthetic em" else "ma"
        delta_rows.append(
            {
                "record_id": str(row.record_id),
                "source_record_id": sid,
                "noise_type": noise_type,
                "split": str(row.split),
                "clean_mean_1_minus_basSQI": float(base["mean_1_minus_basSQI"]),
                "noisy_mean_1_minus_basSQI": float(row.mean_1_minus_basSQI),
                "delta_mean_1_minus_basSQI": float(row.mean_1_minus_basSQI - base["mean_1_minus_basSQI"]),
            }
        )
    delta = pd.DataFrame(delta_rows)
    delta_csv = out / "bassqi_matched_augmentation_delta.csv"
    write_table(delta, delta_csv, md_path=rep / "bassqi_matched_augmentation_delta.md")

    score, pred, te, met = _fit_full_test_svm(df, ["basSQI"], C=C, gamma=gamma, seed=seed)
    recall = pd.DataFrame(
        _true_class_recall_rows(
            df.loc[te].reset_index(drop=True),
            pred,
            score,
            model="basSQI fixed RBF-SVM",
            threshold=float(met["threshold"]),
        )
    )
    recall_csv = out / "bassqi_subgroup_recall.csv"
    write_table(recall, recall_csv, md_path=rep / "bassqi_subgroup_recall.md")
    caption = rep / "fig_14_bassqi_domain_shift_caption.md"
    caption.write_text(
        "Figure 14 uses `1-basSQI = P_0-1 / P_0-40`, the low-frequency power fraction.\n",
        encoding="utf-8",
    )
    paths = plot_bassqi_domain_shift(tab, delta, recall, rep / "fig_14_bassqi_domain_shift")
    return [str(csv), str(delta_csv), str(recall_csv), str(caption), *[str(p) for p in paths]]


def compute_sqi_subgroup_separability(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    report_dir: Path,
    C: float,
    gamma: float,
    seed: int,
) -> list[str]:
    out = Path(out_dir)
    rep = Path(report_dir)
    split = df["split"].astype(str)
    sep_rows: list[dict[str, Any]] = []
    for sqi in ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI", "pSQI"]:
        for domain, groups in POOR_DOMAIN_GROUPS.items():
            pool = df["sample_group"].isin(["original acceptable", *groups])
            tr = split.eq("train").to_numpy() & pool.to_numpy()
            va = split.eq("val").to_numpy() & pool.to_numpy()
            te = split.eq("test").to_numpy() & pool.to_numpy()
            if (
                len(np.unique(df.loc[tr, "y01"])) < 2
                or len(np.unique(df.loc[va, "y01"])) < 2
                or len(np.unique(df.loc[te, "y01"])) < 2
            ):
                sep_rows.append({"SQI": sqi, "poor_domain": domain, "skipped": True})
                continue
            res = _eval_svm_protocol(df, [sqi], tr, va, te, C=C, gamma=gamma, seed=seed)
            sep_rows.append(
                {
                    "SQI": sqi,
                    "poor_domain": domain,
                    "skipped": False,
                    "train_n": int(tr.sum()),
                    "val_n": int(va.sum()),
                    "test_n": int(te.sum()),
                    "test_Ac": float(res["test_Ac"]),
                    "test_Se": float(res["test_Se"]),
                    "test_Sp": float(res["test_Sp"]),
                    "test_AUC": float(res["test_AUC"]),
                    "threshold": float(res["threshold"]),
                }
            )
    separability = pd.DataFrame(sep_rows)
    sep_csv = out / "sqi_poor_domain_separability.csv"
    write_table(separability, sep_csv, md_path=rep / "sqi_poor_domain_separability.md")

    recall_rows: list[dict[str, Any]] = []
    pred_store: dict[str, dict[str, Any]] = {}
    for model_name, sqis in SUBSET_MODELS.items():
        score, pred, te, met = _fit_full_test_svm(df, sqis, C=C, gamma=gamma, seed=seed)
        test_df = df.loc[te].reset_index(drop=True)
        recall_rows.extend(
            _true_class_recall_rows(
                test_df,
                pred,
                score,
                model=model_name,
                threshold=float(met["threshold"]),
            )
        )
        pred_store[model_name] = {"pred_accept": pred, "score": score, "test_df": test_df, "threshold": float(met["threshold"])}
    recall = pd.DataFrame(recall_rows)
    recall_csv = out / "subset_subgroup_recall.csv"
    write_table(recall, recall_csv, md_path=rep / "subset_subgroup_recall.md")

    pair = pred_store["paper pair"]
    quint = pred_store["paper quintuplet"]
    test_df = pair["test_df"].copy()
    y = test_df["y01"].to_numpy(dtype=bool)
    pair_wrong = pair["pred_accept"] != y
    quint_correct = quint["pred_accept"] == y
    rescue_rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        mask = test_df["sample_group"].eq(group).to_numpy()
        denom = int(np.sum(mask & pair_wrong))
        num = int(np.sum(mask & pair_wrong & quint_correct))
        rescue_rows.append(
            {
                "sample_group": group,
                "pair_wrong_n": denom,
                "pair_wrong_quintuplet_correct_n": num,
                "rescue_rate": float(num / denom) if denom else np.nan,
                "pair_error_rate": float(np.mean(pair_wrong[mask])) if np.any(mask) else np.nan,
                "quintuplet_error_rate": float(np.mean(~quint_correct[mask])) if np.any(mask) else np.nan,
            }
        )
    rescue = pd.DataFrame(rescue_rows)
    rescue_csv = out / "pair_to_quintuplet_error_rescue.csv"
    write_table(rescue, rescue_csv, md_path=rep / "pair_to_quintuplet_error_rescue.md")
    paths = plot_sqi_subgroup_separability(separability, recall, rescue, rep / "fig_15_sqi_subgroup_separability")
    return [str(sep_csv), str(recall_csv), str(rescue_csv), *[str(p) for p in paths]]


def _load_sig125(art: Path, record_id: str) -> np.ndarray:
    z = np.load(art / "resampled_125" / f"{record_id}.npz", allow_pickle=True)
    return z["sig_125"].astype(np.float64)


def _fsqi_features_for_threshold(art: Path, record_ids: list[str], threshold: float) -> np.ndarray:
    X = np.empty((len(record_ids), len(LEADS_12)), dtype=np.float64)
    for i, rid in enumerate(record_ids):
        sig = _load_sig125(art, rid)
        dx = np.abs(np.diff(sig, axis=0))
        X[i, :] = np.mean(dx < float(threshold), axis=0)
    return X


def _subgroup_recall_rows(df: pd.DataFrame, score: np.ndarray, pred_accept: np.ndarray, *, threshold_mv: float, score_threshold: float, model: str) -> list[dict[str, Any]]:
    tmp = df[["sample_group", "y01"]].copy()
    tmp["score"] = score
    tmp["pred_accept"] = pred_accept
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        g = tmp[tmp["sample_group"].eq(group)]
        if g.empty:
            continue
        if int(g["y01"].iloc[0]) == 1:
            recall = float(g["pred_accept"].mean())
            recall_name = "acceptable_recall"
        else:
            recall = float((~g["pred_accept"]).mean())
            recall_name = "poor_recall"
        rows.append(
            {
                "model": model,
                "threshold_mv": float(threshold_mv),
                "score_threshold": float(score_threshold),
                "sample_group": group,
                "n": int(len(g)),
                "recall_type": recall_name,
                "recall": recall,
                "score_median": float(g["score"].median()),
                "score_q10": float(g["score"].quantile(0.10)),
                "score_q90": float(g["score"].quantile(0.90)),
            }
        )
    return rows


def _fit_eval_score_model(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    model_name: str,
    seed: int,
    C: float = 1.0,
    gamma: float = 0.14,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    tr = df["split"].astype(str).eq("train").to_numpy()
    va = df["split"].astype(str).eq("val").to_numpy()
    te = df["split"].astype(str).eq("test").to_numpy()
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)
    if model_name == "RBF-SVM":
        select = fit_fixed_svm(X[tr], ytr, C=C, gamma=gamma, seed=seed)
        p_val = predict_score(select, X[va])
        thr = max_accuracy_threshold(yva, p_val)
        final = fit_fixed_svm(X[tr | va], df.loc[tr | va, "y01"].to_numpy(dtype=int), C=C, gamma=gamma, seed=seed)
        p_test = predict_score(final, X[te])
    elif model_name == "logistic":
        select = Pipeline([("scale", StandardScaler()), ("logreg", LogisticRegression(max_iter=4000, random_state=seed))])
        select.fit(X[tr], ytr)
        p_val = select.predict_proba(X[va])[:, 1]
        thr = max_accuracy_threshold(yva, p_val)
        final = Pipeline([("scale", StandardScaler()), ("logreg", LogisticRegression(max_iter=4000, random_state=seed))])
        final.fit(X[tr | va], df.loc[tr | va, "y01"].to_numpy(dtype=int))
        p_test = final.predict_proba(X[te])[:, 1]
    elif model_name == "linear-SVM":
        select = Pipeline([("scale", StandardScaler()), ("svc", LinearSVC(C=1.0, random_state=seed, max_iter=10000, dual="auto"))])
        select.fit(X[tr], ytr)
        raw_val = select.decision_function(X[va])
        p_val = _minmax_from_reference(raw_val, raw_val)
        thr = max_accuracy_threshold(yva, p_val)
        final = Pipeline([("scale", StandardScaler()), ("svc", LinearSVC(C=1.0, random_state=seed, max_iter=10000, dual="auto"))])
        final.fit(X[tr | va], df.loc[tr | va, "y01"].to_numpy(dtype=int))
        raw_test = final.decision_function(X[te])
        p_test = _minmax_from_reference(raw_test, raw_val)
    else:
        raise ValueError(model_name)
    met = binary_metrics(yte, p_test, float(thr["threshold"]))
    row = {
        "model": model_name,
        "val_Ac": float(thr["Ac"]),
        "score_threshold": float(thr["threshold"]),
        "test_Ac": float(met["Ac"]),
        "test_Se": float(met["Se"]),
        "test_Sp": float(met["Sp"]),
        "test_AUC": float(met["AUC"]),
        "test_tn": int(met["tn"]),
        "test_fp": int(met["fp"]),
        "test_fn": int(met["fn"]),
        "test_tp": int(met["tp"]),
    }
    return row, p_test, te


def _minmax_from_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(reference))
    hi = float(np.nanmax(reference))
    return np.clip((np.asarray(values, dtype=float) - lo) / max(1e-12, hi - lo), 0.0, 1.0)


def compute_fsqi_final(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    seed: int = 0,
    C: float = 1.0,
    gamma: float = 0.14,
) -> list[str]:
    art = Path(artifacts_dir)
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    df = load_split_frame(art, normalized=True)
    record_ids = df["record_id"].astype(str).tolist()

    fcols = [f"{lead}__fSQI" for lead in LEADS_12]
    fsqi = df[fcols].to_numpy(dtype=np.float64)
    lead_summary = df[["record_id", "source_record_id", "split", "sample_group", "y", "y01"]].copy()
    lead_summary["median_fSQI"] = np.median(fsqi, axis=1)
    lead_summary["max_fSQI"] = np.max(fsqi, axis=1)
    lead_summary["n_leads_fSQI_gt_0.01"] = (fsqi > 0.01).sum(axis=1)
    lead_summary["n_leads_fSQI_gt_0.10"] = (fsqi > 0.10).sum(axis=1)
    summary_csv = out / "fsqi_record_lead_summary.csv"
    write_table(lead_summary, summary_csv)

    scan_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    best_for_plot: tuple[float, np.ndarray, np.ndarray, float] | None = None
    best_val = -np.inf
    for thr_mv in THRESHOLDS_MV:
        X = _fsqi_features_for_threshold(art, record_ids, float(thr_mv))
        row, p_test, te = _fit_eval_score_model(df, X, model_name="RBF-SVM", seed=seed, C=C, gamma=gamma)
        row.update({"threshold_mv": float(thr_mv), "n_features": int(X.shape[1]), "C": float(C), "gamma": float(gamma)})
        scan_rows.append(row)
        pred = p_test > float(row["score_threshold"])
        subgroup_rows.extend(
            _subgroup_recall_rows(
                df.loc[te].reset_index(drop=True),
                p_test,
                pred,
                threshold_mv=float(thr_mv),
                score_threshold=float(row["score_threshold"]),
                model="fSQI RBF-SVM",
            )
        )
        if float(row["val_Ac"]) > best_val:
            best_val = float(row["val_Ac"])
            best_for_plot = (float(thr_mv), p_test, te, float(row["score_threshold"]))

    scan = pd.DataFrame(scan_rows)
    scan_csv = out / "fsqi_fixed_rbf_threshold_scan.csv"
    write_table(scan, scan_csv, md_path=rep / "fsqi_fixed_rbf_threshold_scan.md")
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup_csv = out / "fsqi_subgroup_recall.csv"
    write_table(subgroup, subgroup_csv, md_path=rep / "fsqi_subgroup_recall.md")

    default_X = _fsqi_features_for_threshold(art, record_ids, 1e-4)
    compare_rows = []
    for model_name in ["logistic", "linear-SVM", "RBF-SVM"]:
        row, _, _ = _fit_eval_score_model(df, default_X, model_name=model_name, seed=seed, C=C, gamma=gamma)
        row.update({"threshold_mv": 1e-4, "n_features": int(default_X.shape[1])})
        compare_rows.append(row)
    compare = pd.DataFrame(compare_rows)
    compare_csv = out / "fsqi_linear_vs_rbf.csv"
    write_table(compare, compare_csv, md_path=rep / "fsqi_linear_vs_rbf.md")

    if best_for_plot is None:
        raise RuntimeError("No fSQI threshold scan result was produced.")
    best_thr, best_score, best_mask, best_score_thr = best_for_plot
    score_df = df.loc[best_mask, ["record_id", "source_record_id", "sample_group", "y", "y01"]].copy().reset_index(drop=True)
    score_df["fsqi_rbf_score"] = best_score
    score_df["score_threshold"] = best_score_thr
    score_df["threshold_mv"] = best_thr
    write_table(score_df, out / "fsqi_best_threshold_test_scores.csv")
    paths = plot_fsqi_mechanism_updated(lead_summary, scan, subgroup, compare, rep / "fig_12_fsqi_mechanism")
    return [str(summary_csv), str(scan_csv), str(subgroup_csv), str(compare_csv), str(out / "fsqi_best_threshold_test_scores.csv"), *[str(p) for p in paths]]


def copy_figures_to_shared(report_dir: str | Path, shared_images_dir: str | Path) -> list[str]:
    rep = Path(report_dir)
    shared = Path(shared_images_dir)
    shared.mkdir(parents=True, exist_ok=True)
    mapping = {
        "fig_12_fsqi_mechanism": "fig_12_fsqi_mechanism",
        "fig_13_sqi_domain_shift": "fig_13_sqi_domain_shift",
        "fig_14_bassqi_domain_shift": "fig_14_bassqi_domain_shift",
        "fig_15_sqi_subgroup_separability": "fig_15_sqi_subgroup_separability",
    }
    copied: list[str] = []
    for src_stem, dst_stem in mapping.items():
        for suffix in [".png", ".pdf", ".svg"]:
            src = rep / f"{src_stem}{suffix}"
            if not src.exists():
                continue
            dst = shared / f"{dst_stem}{suffix}"
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def run_final_claims(
    *,
    artifacts_dir: str | Path,
    strict_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    shared_images_dir: str | Path,
    seed: int = 0,
    C: float = 1.0,
    gamma: float = 0.14,
    n_perm: int = 1000,
) -> dict[str, Any]:
    out = Path(out_dir)
    rep = Path(report_dir)
    outputs: list[str] = []
    outputs.extend(summarize_paper_quintuplet(strict_dir, out, rep))
    outputs.extend(compute_domain_shift(artifacts_dir=artifacts_dir, out_dir=out, report_dir=rep, seed=seed, C=C, gamma=gamma, n_perm=n_perm))
    outputs.extend(compute_fsqi_final(artifacts_dir=artifacts_dir, out_dir=out, report_dir=rep, seed=seed, C=C, gamma=gamma))
    copied = copy_figures_to_shared(rep, shared_images_dir)
    outputs.extend(copied)
    return {"outputs": outputs, "shared_images": copied}
