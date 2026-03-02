from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.utils.paths import project_root
from src.models.svm_rbf import SVMConfig, SVMRBF, _setup_logging

logger = logging.getLogger(__name__)

SEED = 0

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SQI_LIST = ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]

# === Paper Table 6 combos (12-lead) ===
COMBOS = [
    ("Pairs",       ["iSQI", "basSQI"]),
    ("Triplets",    ["bSQI", "basSQI", "pSQI"]),
    ("Quadruplets", ["bSQI", "basSQI", "kSQI", "sSQI"]),
    ("Quintuplets", ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]),
    ("Sextuplets",  ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI"]),
    ("All SQI",     ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]),
]


def _load_df(features_parquet: Path, split_csv: Path) -> pd.DataFrame:
    df_feat = pd.read_parquet(features_parquet)
    df_split = pd.read_csv(split_csv)

    df_feat["record_id"] = df_feat["record_id"].astype(str)
    df_split["record_id"] = df_split["record_id"].astype(str)

    df = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="inner")
    if df["split"].isna().any():
        raise ValueError("Some records missing split after merge (unexpected)")

    # y: ±1 -> 0/1 (ONLY inside model task)
    y_pm1 = df["y"].astype(int).to_numpy()
    df["y01"] = (y_pm1 == 1).astype(np.int32)
    return df


def _get_feature_cols_12lead(sqis: list[str]) -> list[str]:
    cols: list[str] = []
    for lead in LEADS_12:
        for s in sqis:
            cols.append(f"{lead}__{s}")
    return cols


def _metrics(y01: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y01 = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    pred = (p > float(threshold)).astype(int)

    cm = confusion_matrix(y01, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    se = tp / max(1, (tp + fn))
    sp = tn / max(1, (tn + fp))

    try:
        auc = float(roc_auc_score(y01, p))
    except Exception:
        auc = float("nan")

    return {
        "Ac": float(acc),
        "Se": float(se),
        "Sp": float(sp),
        "AUC": float(auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _max_acc_threshold(y01: np.ndarray, p: np.ndarray, n_grid: int = 2001) -> dict[str, Any]:
    y = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    ts = np.linspace(0.0, 1.0, int(n_grid))

    best = {"threshold": 0.5, "acc": -1.0, "se": np.nan, "sp": np.nan, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for t in ts:
        pred = (p > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        if acc > best["acc"]:
            se = tp / max(1, tp + fn)
            sp = tn / max(1, tn + fp)
            best = {"threshold": float(t), "acc": float(acc), "se": float(se), "sp": float(sp),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return best


def plot_roc_with_maxacc(y01: np.ndarray, p: np.ndarray, out_png: Path) -> dict[str, Any]:
    fpr, tpr, thr = roc_curve(y01.astype(int), p.astype(np.float64))
    best = _max_acc_threshold(y01, p)

    if len(thr) > 0:
        idx = int(np.argmin(np.abs(thr - best["threshold"])))
        x = float(fpr[idx]); y = float(tpr[idx])
    else:
        x, y = 0.0, 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.scatter([x], [y])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"SVM ROC (test) + maxAcc@thr={best['threshold']:.3f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return best


def _save_probs_npz(out_npz: Path, y01: np.ndarray, p: np.ndarray, threshold_fixed: float) -> None:
    np.savez_compressed(
        out_npz,
        y01_test=y01.astype(np.int32),
        p_test=p.astype(np.float64),
        threshold_fixed=np.array(threshold_fixed, dtype=np.float64),
    )


def _fit_fixed_params(model: SVMRBF, X: np.ndarray, y: np.ndarray, *, C: float, gamma: float) -> dict[str, Any]:
    """
    Fit SVMRBF once with fixed (C, gamma), no grid search.
    Keeps using YOUR SVMRBF object; we just build its pipeline and set params.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int).ravel()

    pipe = model._make_pipeline()
    pipe.set_params(svc__C=float(C), svc__gamma=float(gamma))

    t0 = time.time()
    pipe.fit(X, y)
    total_s = float(time.time() - t0)

    model.best_estimator_ = pipe
    model.best_params_ = {"svc__C": float(C), "svc__gamma": float(gamma)}
    model.best_cv_score_ = float("nan")

    return {"train_time_s": total_s}


def _fit_eval_12lead(
    df: pd.DataFrame,
    sqis: list[str],
    svm_cfg: SVMConfig,
    threshold: float = 0.5,
    *,
    # selection control
    use_global_best: bool = True,
    C_star: float | None = None,
    gamma_star: float | None = None,
    # only used if not use_global_best
    select_metric: str = "val_acc",
    log_each: bool = True,
    # bookkeeping
    global_report: dict[str, Any] | None = None,
    save_model: bool = False,
    model_out: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cols = _get_feature_cols_12lead(sqis)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns (examples): {missing[:10]}")

    tr = df["split"].to_numpy() == "train"
    va = df["split"].to_numpy() == "val"
    te = df["split"].to_numpy() == "test"

    Xtr = df.loc[tr, cols].to_numpy(dtype=np.float64)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, cols].to_numpy(dtype=np.float64)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, cols].to_numpy(dtype=np.float64)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)

    model = SVMRBF(svm_cfg)

    # ---- training ----
    if use_global_best:
        if C_star is None or gamma_star is None:
            raise ValueError("use_global_best=True but C_star/gamma_star not provided")
        fit_info = _fit_fixed_params(model, Xtr, ytr, C=float(C_star), gamma=float(gamma_star))
        train_time = float(fit_info["train_time_s"])

        # keep table fields intact (fill with global search report if provided)
        best_params = {"svc__C": float(C_star), "svc__gamma": float(gamma_star)}
        if global_report is None:
            best_score = float("nan")
            best_val_acc = float("nan")
            best_val_auc = float("nan")
            n_grid = 0
        else:
            best_score = float(global_report.get("best_score", np.nan))
            best_val_acc = float(global_report.get("best_val_acc", np.nan))
            best_val_auc = float(global_report.get("best_val_auc", np.nan))
            n_grid = int(global_report.get("n_grid", 0))

    else:
        gs = model.fit_gridsearch(
            Xtr, ytr,
            Xva, yva,
            select_metric=select_metric,
            threshold=threshold,
            log_each=log_each,
        )
        train_time = float(gs["train_time_s"])
        best_score = float(gs["best_score"])
        best_val_acc = float(gs["best_val_acc"])
        best_val_auc = float(gs["best_val_auc"])
        best_params = dict(gs["best_params"])
        n_grid = int(gs.get("n_grid", -1))

    # ---- eval ----
    p_tr = model.predict_proba(Xtr)
    p_te = model.predict_proba(Xte)

    met_tr = _metrics(ytr, p_tr, threshold=threshold)
    met_te = _metrics(yte, p_te, threshold=threshold)

    out = {
        "sqis": ",".join(sqis),
        "n_features": int(len(cols)),
        "best_score": float(best_score),
        "best_val_acc": float(best_val_acc),
        "best_val_auc": float(best_val_auc),
        "best_params": best_params,  # {"svc__C":..., "svc__gamma":...}
        "n_grid": int(n_grid),
        "train_time_s": float(train_time),
        "train": met_tr,
        "test": met_te,
        "p_test": p_te,
        "y01_test": yte,
    }

    if save_model and model_out is not None:
        model.save(model_out, meta=meta or {})
        out["model_path"] = str(model_out)

    return out


def run(params: dict[str, Any]) -> dict[str, Any]:
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    global SEED
    if params.get("seed") is not None:
        SEED = int(params["seed"])

    root = project_root()
    features_parquet = Path(str(params.get("features_parquet") or (root / "artifacts" / "features" / "record84_norm.parquet")))
    split_csv = Path(str(params.get("split_csv") or (root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv")))
    out_dir = Path(str(params.get("out_dir") or (root / "artifacts" / "models" / "svm")))
    out_dir.mkdir(parents=True, exist_ok=True)

    roc_dir = out_dir / "roc"; roc_dir.mkdir(parents=True, exist_ok=True)
    probs_dir = out_dir / "probs"; probs_dir.mkdir(parents=True, exist_ok=True)
    maxacc_dir = out_dir / "maxacc"; maxacc_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(params.get("threshold", 0.5))
    save_models = bool(params.get("save_models", False))

    select_metric = str(params.get("select_metric", "val_acc"))
    log_each = bool(params.get("log_each", True))

    # outputs
    out_table5 = out_dir / f"table5_12lead_single_sqi_seed{SEED}.csv"
    out_table6 = out_dir / f"table6_12lead_combo_sqi_seed{SEED}.csv"

    if (not force) and out_table5.exists() and out_table6.exists():
        logger.info("svm_tables: outputs exist -> skip (set force=True to rerun)")
        return {"step": "svm_tables", "skipped": True, "outputs": [str(out_table5), str(out_table6)]}

    logger.info("features: %s", features_parquet)
    logger.info("split_csv: %s", split_csv)
    logger.info("out_dir: %s", out_dir)
    logger.info("threshold=%.3f | save_models=%s", threshold, save_models)

    df = _load_df(features_parquet, split_csv)
    ntr = int((df["split"] == "train").sum())
    nva = int((df["split"] == "val").sum())
    nte = int((df["split"] == "test").sum())
    logger.info("rows: train=%d val=%d test=%d total=%d", ntr, nva, nte, len(df))

    # SVM config (grid-search + optional scaler)
    svm_cfg = SVMConfig(
        seed=SEED,
        cv_folds=int(params.get("cv_folds", 5)),
        use_standard_scaler=bool(params.get("use_standard_scaler", False)),
    )
    if params.get("C_list") is not None:
        object.__setattr__(svm_cfg, "C_list", tuple(float(x) for x in params["C_list"]))
    if params.get("gamma_list") is not None:
        object.__setattr__(svm_cfg, "gamma_list", tuple(float(x) for x in params["gamma_list"]))

    # ----------------------------------------
    # GLOBAL grid search (run ONCE) to get C*, gamma*
    # ----------------------------------------
    global_search_sqis = list(params.get(
        "global_search_sqis",
        ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"],
    ))
    global_cols = _get_feature_cols_12lead(global_search_sqis)

    tr_mask = df["split"].to_numpy() == "train"
    Xtr_global = df.loc[tr_mask, global_cols].to_numpy(dtype=np.float64)
    ytr_global = df.loc[tr_mask, "y01"].to_numpy(dtype=int)

    va_mask = df["split"].to_numpy() == "val"
    Xva_global = df.loc[va_mask, global_cols].to_numpy(dtype=np.float64)
    yva_global = df.loc[va_mask, "y01"].to_numpy(dtype=int)

    logger.info("GLOBAL search: sqis=%s | n_features=%d", ",".join(global_search_sqis), Xtr_global.shape[1]) # type: ignore

    global_model = SVMRBF(svm_cfg)
    gs_global = global_model.fit_gridsearch(
        Xtr_global, ytr_global,
        Xva_global, yva_global,
        select_metric=select_metric,
        threshold=threshold,
        log_each=log_each,
    )

    C_star = float(gs_global["best_params"]["svc__C"])
    gamma_star = float(gs_global["best_params"]["svc__gamma"])

    global_report = {
        "best_score": float(gs_global["best_score"]),
        "best_val_acc": float(gs_global["best_val_acc"]),
        "best_val_auc": float(gs_global["best_val_auc"]),
        "n_grid": int(gs_global.get("n_grid", 0)),
    }

    logger.info(
        "GLOBAL BEST: C=%.6g gamma=%.6g | val_acc=%.4f val_auc=%.4f",
        C_star, gamma_star, global_report["best_val_acc"], global_report["best_val_auc"]
    )

    # -----------------------
    # Table 5: each SQI alone (12-lead) -- USE GLOBAL BEST (no re-search)
    # -----------------------
    rows5: list[dict[str, Any]] = []
    logger.info("Table 5: 12-lead, each SQI individually (USING GLOBAL BEST params)")

    for sqi in SQI_LIST:
        sqis = [sqi]

        model_path = None
        if save_models:
            (out_dir / "models").mkdir(parents=True, exist_ok=True)
            model_path = out_dir / "models" / f"svm_12lead__{sqi}__seed{SEED}.pkl"

        res = _fit_eval_12lead(
            df=df,
            sqis=sqis,
            svm_cfg=svm_cfg,
            threshold=threshold,
            use_global_best=True,
            C_star=C_star,
            gamma_star=gamma_star,
            global_report=global_report,
            save_model=save_models,
            model_out=model_path,
            meta={"mode": "12lead", "sqis": sqis, "global_best": {"C": C_star, "gamma": gamma_star}},
        )

        key = f"{sqi}"
        out_probs = probs_dir / f"{key}_seed{SEED}.npz"
        _save_probs_npz(out_probs, res["y01_test"], res["p_test"], threshold)

        out_roc = roc_dir / f"{key}_roc_maxacc_seed{SEED}.png"
        best_thr = plot_roc_with_maxacc(res["y01_test"], res["p_test"], out_roc)

        out_maxacc = maxacc_dir / f"{key}_maxacc_seed{SEED}.csv"
        pd.DataFrame([best_thr]).to_csv(out_maxacc, index=False)

        rows5.append({
            "SQI": sqi,
            "n_features": res["n_features"],
            "val_score": res["best_score"],
            "val_acc_best": res["best_val_acc"],
            "val_auc_best": res["best_val_auc"],
            "best_C": res["best_params"]["svc__C"],
            "best_gamma": res["best_params"]["svc__gamma"],
            "n_grid": res["n_grid"],
            "Ac_train": res["train"]["Ac"],
            "Se_train": res["train"]["Se"],
            "Sp_train": res["train"]["Sp"],
            "Ac_test": res["test"]["Ac"],
            "Se_test": res["test"]["Se"],
            "Sp_test": res["test"]["Sp"],
            "AUC_test": res["test"]["AUC"],
            "maxAcc_thr_test": best_thr["threshold"],
            "maxAcc_test": best_thr["acc"],
        })

        logger.info("  %-6s | using GLOBAL (C,gamma)=(%.3g,%.3g) | test Ac=%.4f",
                    sqi, C_star, gamma_star, res["test"]["Ac"])

    pd.DataFrame(rows5).to_csv(out_table5, index=False)
    logger.info("[saved] %s", out_table5)

    # -----------------------
    # Table 6: SQI combos (12-lead) -- USE GLOBAL BEST (no re-search)
    # -----------------------
    rows6: list[dict[str, Any]] = []
    logger.info("Table 6: 12-lead, SQI combinations (USING GLOBAL BEST params)")

    for group, sqis in COMBOS:
        safe_name = group.replace(" ", "")
        model_path = None
        if save_models:
            (out_dir / "models").mkdir(parents=True, exist_ok=True)
            model_path = out_dir / "models" / f"svm_12lead__{safe_name}__seed{SEED}.pkl"

        res = _fit_eval_12lead(
            df=df,
            sqis=sqis,
            svm_cfg=svm_cfg,
            threshold=threshold,
            use_global_best=True,
            C_star=C_star,
            gamma_star=gamma_star,
            global_report=global_report,
            save_model=save_models,
            model_out=model_path,
            meta={"mode": "12lead", "group": group, "sqis": sqis, "global_best": {"C": C_star, "gamma": gamma_star}},
        )

        key = f"{group.replace(' ', '')}"
        out_probs = probs_dir / f"{key}_seed{SEED}.npz"
        _save_probs_npz(out_probs, res["y01_test"], res["p_test"], threshold)

        out_roc = roc_dir / f"{key}_roc_maxacc_seed{SEED}.png"
        best_thr = plot_roc_with_maxacc(res["y01_test"], res["p_test"], out_roc)

        out_maxacc = maxacc_dir / f"{key}_maxacc_seed{SEED}.csv"
        pd.DataFrame([best_thr]).to_csv(out_maxacc, index=False)

        rows6.append({
            "Group": group,
            "Selected_SQI": ",".join(sqis),
            "n_features": res["n_features"],
            "val_score": res["best_score"],
            "val_acc_best": res["best_val_acc"],
            "val_auc_best": res["best_val_auc"],
            "best_C": res["best_params"]["svc__C"],
            "best_gamma": res["best_params"]["svc__gamma"],
            "n_grid": res["n_grid"],
            "Ac_train": res["train"]["Ac"],
            "Ac_test": res["test"]["Ac"],
            "AUC_test": res["test"]["AUC"],
            "maxAcc_thr_test": best_thr["threshold"],
            "maxAcc_test": best_thr["acc"],
        })

        logger.info("  %-11s | using GLOBAL (C,gamma)=(%.3g,%.3g) | test Ac=%.4f",
                    group, C_star, gamma_star, res["test"]["Ac"])

    pd.DataFrame(rows6).to_csv(out_table6, index=False)
    logger.info("[saved] %s", out_table6)

    return {
        "step": "svm_tables",
        "skipped": False,
        "outputs": [
            str(out_table5),
            str(out_table6),
            str(roc_dir),
            str(probs_dir),
            str(maxacc_dir),
        ],
    }


def main() -> None:
    params = {
        "verbose": False,
        "force": True,
        "C_list": (1.0, 2.0**1, 2.0**3, 25, 2.0**5, 2.0**7, 2.0**9),
        "gamma_list": (2.0**-11, 2.0**-9, 2.0**-7, 2.0**-5, 2.0**-3, 0.14, 2.0**-1, 0.7, 1, 1.5, 2),
        "select_metric": "val_acc",    # or "val_auc"
        "log_each": True,              # only affects GLOBAL search printing
        "global_search_sqis": ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"],
    }
    run(params)


if __name__ == "__main__":
    main()