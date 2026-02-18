# src/models/lm_mlp_search.py
from __future__ import annotations

import json
import pickle
import time
import logging
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import torch

from src.utils.paths import project_root
from src.models.lm_mlp import LMMLP, LMConfig


# =======================
# Fixed config
# =======================
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

J_RANGE = list(range(2, 18))  # 2..17
MODEL_SELECT_METRIC = "val_acc"  # paper uses val accuracy; also store AUC

# Data
FEATURES_PARQUET = "record84_norm.parquet"
SPLIT_CSV = "split_seta_seed0_balanced.csv"

# Training config (paper)
EPOCHS_MAX = 100
STOP_ERROR = 1e-5
STOP_GRAD = 1e-5

# Val early stop for final model (your requirement: train with val preventing overfit)
FINAL_PATIENCE = 15
THRESH = 0.7  # label uses 0/1; predict proba >0.5 -> 1

# LM damping behavior (stable defaults)
MU_INIT = 1e-3
MU_DEC = 0.1
MU_INC = 10.0
MAX_DAMP_TRIES = 10
# =======================

# =======================
# Extra: paper-like tables (on YOUR train/val/test split)
# =======================
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SQI_LIST = ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]

# Table 6 combos (use your chosen list; you can edit later)
COMBOS = [
    ("Pairs",       ["iSQI", "basSQI"]),
    ("Triplets",    ["bSQI", "basSQI", "pSQI"]),
    ("Quadruplets", ["bSQI", "basSQI", "kSQI", "sSQI"]),
    ("Quintuplets", ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]),
    ("Sextuplets",  ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI"]),
    ("All SQI",     ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]),
]

# Table 7 "selected five SQIs"
SELECTED_5 = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]

# where to write
TABLE_DIR_NAME = "lm_mlp_tables"
# =======================


logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_Xy_split(features_parquet: Path, split_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feat_path = features_parquet
    split_path = split_csv

    logger.info("features: %s", feat_path)
    logger.info("split_csv: %s", split_path)

    df_feat = pd.read_parquet(feat_path)
    df_split = pd.read_csv(split_path)

    # unify record_id type to avoid merge error
    df_feat["record_id"] = df_feat["record_id"].astype(str)
    df_split["record_id"] = df_split["record_id"].astype(str)

    df = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="inner")

    # label mapping: y in parquet is ±1. we convert to 0/1 here globally for this task.
    y_pm1 = df["y"].astype(int).to_numpy()
    y01 = (y_pm1 == 1).astype(np.int32)

    # feature columns: exclude record_id, y, split
    drop_cols = {"record_id", "y", "split"}
    feat_cols = sorted([c for c in df.columns if c not in drop_cols])

    X = df[feat_cols].to_numpy(dtype=np.float64)
    split = df["split"].astype(str).to_numpy()

    if X.shape[1] != 84:
        raise ValueError(f"Expected 84 features, got {X.shape[1]}")

    logger.info("X shape: %s (#features=%d)", X.shape, X.shape[1])
    logger.info("splits: train=%d val=%d test=%d",
                int((split == "train").sum()), int((split == "val").sum()), int((split == "test").sum()))
    logger.info("labels(train): good=%d bad=%d",
                int(y01[split == "train"].sum()),
                int((split == "train").sum() - y01[split == "train"].sum()))

    return X, y01, split, feat_cols


def metrics_from_probs(y01: np.ndarray, p: np.ndarray) -> dict:
    y01 = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    pred = (p > THRESH).astype(int)

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
        "acc": float(acc),
        "se": float(se),
        "sp": float(sp),
        "auc": float(auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def _cols_for_sqis_12lead(sqis: list[str]) -> list[str]:
    cols: list[str] = []
    for s in sqis:
        for ld in LEADS_12:
            cols.append(f"{ld}__{s}")
    return cols


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

    # find closest ROC point to that threshold
    if len(thr) > 0:
        idx = int(np.argmin(np.abs(thr - best["threshold"])))
        x = float(fpr[idx]); y = float(tpr[idx])
    else:
        x, y = 0.0, 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.scatter([x], [y])  # no explicit color
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (test) + maxAcc@thr={best['threshold']:.3f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return best


def plot_val_curve(df_res: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_res["J"], df_res["val_acc"], marker="o", label="val_acc")
    ax.plot(df_res["J"], df_res["val_auc"], marker="o", label="val_auc")
    ax.set_xlabel("Hidden units J")
    ax.set_ylabel("Metric")
    ax.set_title("LM-MLP architecture search (84–J–1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_roc(y01: np.ndarray, p: np.ndarray, out_png: Path) -> None:
    fpr, tpr, _ = roc_curve(y01.astype(int), p.astype(np.float64))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (test)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_confmat(y01: np.ndarray, p: np.ndarray, out_png: Path) -> None:
    pred = (p > THRESH).astype(int)
    cm = confusion_matrix(y01.astype(int), pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["bad(0)", "good(1)"])
    ax.set_yticks([0, 1], labels=["bad(0)", "good(1)"])
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def _outputs_exist(out_dir: Path) -> bool:
    models_dir = out_dir / "models"
    out_csv = models_dir / "search_J_results_seed0.csv"
    best_json = models_dir / "best_J_seed0.json"
    out_model_any = any(models_dir.glob("model_*-*_seed0.pkl")) or any(models_dir.glob("model_84-*-1_seed0.pkl"))

    out_metrics = out_dir / "lm_mlp_test_metrics_seed0.json"

    if not (out_csv.exists() and out_csv.stat().st_size > 0):
        return False
    if not (best_json.exists() and best_json.stat().st_size > 0):
        return False
    if not out_model_any:
        return False
    if not (out_metrics.exists() and out_metrics.stat().st_size > 0):
        return False
    return True


def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - seed: int
      - device: str ("cuda"|"cpu")  (optional override)
      - dtype: str ("float64"|"float32") (optional override)
      - features_parquet: str (default record84_norm.parquet)
      - split_csv: str (default split_seta_seed0_balanced.csv)
      - model_dir: str (default artifacts/models/lm_mlp)
      - eval_dir: str (default artifacts/eval)

    Returns dict with outputs list and skipped bool.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    run_tables = bool(params.get("tables", False))
    tables_mode = str(params.get("tables_mode", "fixedJ"))  # "fixedJ" or "searchJ"
    J_fixed = int(params.get("J_fixed", 12))
    _setup_logging(verbose)

    # optional overrides (keep defaults identical)
    global SEED, DEVICE, DTYPE, FEATURES_PARQUET, SPLIT_CSV
    if "seed" in params and params["seed"] is not None:
        SEED = int(params["seed"])

    if "device" in params and params["device"] is not None:
        d = str(params["device"]).lower()
        if d == "cuda":
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif d == "cpu":
            DEVICE = torch.device("cpu")

    if "dtype" in params and params["dtype"] is not None:
        dt = str(params["dtype"]).lower()
        if dt in ("float64", "double", "f64"):
            DTYPE = torch.float64
        elif dt in ("float32", "float", "f32"):
            DTYPE = torch.float32

    if "features_parquet" in params and params["features_parquet"]:
        FEATURES_PARQUET = str(params["features_parquet"])
    if "split_csv" in params and params["split_csv"]:
        SPLIT_CSV = str(params["split_csv"])

    seed_all(SEED)
    root = project_root()

    # --- align with SVM-style out_dir layout ---
    features_parquet = Path(
        str(params.get("features_parquet") or (root / "artifacts" / "features" / "record84_norm.parquet")))
    split_csv = Path(str(params.get("split_csv") or (root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv")))

    out_dir = Path(str(params.get("out_dir") or (root / "artifacts" / "models" / "lm_mlp")))
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    roc_dir = out_dir / "roc"
    roc_dir.mkdir(parents=True, exist_ok=True)

    probs_dir = out_dir / "probs"
    probs_dir.mkdir(parents=True, exist_ok=True)

    maxacc_dir = out_dir / "maxacc"
    maxacc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("device: %s", DEVICE)
    logger.info("dtype: %s", DTYPE)

    if (not force) and _outputs_exist(out_dir):
        logger.info("lm_mlp_search: outputs exist -> skip (set force=True to rerun)")
        outs = [
            str(models_dir / "search_J_results_seed0.csv"),
            str(models_dir / "best_J_seed0.json"),
            str(out_dir / "lm_mlp_test_metrics_seed0.json"),
        ]
        return {"step": "lm_mlp_search", "skipped": True, "outputs": outs}

    X, y01, split, feat_cols = load_Xy_split(features_parquet, split_csv)

    Xtr = X[split == "train"]
    ytr = y01[split == "train"]
    Xva = X[split == "val"]
    yva = y01[split == "val"]
    Xte = X[split == "test"]
    yte = y01[split == "test"]

    # torch tensors (full-batch LM)
    Xtr_t = torch.tensor(Xtr, device=DEVICE, dtype=DTYPE)
    ytr_t = torch.tensor(ytr, device=DEVICE, dtype=DTYPE)
    Xva_t = torch.tensor(Xva, device=DEVICE, dtype=DTYPE)
    yva_t = torch.tensor(yva, device=DEVICE, dtype=DTYPE)
    Xte_t = torch.tensor(Xte, device=DEVICE, dtype=DTYPE)
    yte_t = torch.tensor(yte, device=DEVICE, dtype=DTYPE)

    cfg = LMConfig(
        epochs_max=EPOCHS_MAX,
        stop_error=STOP_ERROR,
        stop_grad=STOP_GRAD,
        mu_init=MU_INIT,
        mu_dec=MU_DEC,
        mu_inc=MU_INC,
        max_damping_tries=MAX_DAMP_TRIES,
    )

    # -------------------------
    # 1) search J=2..17 by val
    # -------------------------
    results = []
    stop_counts: dict[str, int] = {}

    logger.info("[Task 1.7] LM-MLP search J=2..17")
    for J in J_RANGE:
        t0 = time.time()
        m = LMMLP(J=J, device=DEVICE, dtype=DTYPE, seed=SEED)

        train_summary = m.fit_lm(
            X_train=Xtr_t,
            y_train=ytr_t,
            cfg=cfg,
            X_val=Xva_t,
            y_val=yva_t,
            model_select_metric=MODEL_SELECT_METRIC,
            patience=FINAL_PATIENCE,
            threshold=THRESH,
        )

        p_val = m.predict_proba(Xva_t)
        met_val = metrics_from_probs(yva, p_val)

        row = {
            "J": int(J),
            "epochs_used": int(train_summary["epochs_used"]),
            "final_error": float(train_summary["final_error"]),
            "final_grad": float(train_summary["final_grad"]),
            "stop_reason": str(train_summary["stop_reason"]),
            "val_acc": float(met_val["acc"]),
            "val_auc": float(met_val["auc"]),
            "train_time_s": float(time.time() - t0),
        }
        results.append(row)
        stop_counts[row["stop_reason"]] = stop_counts.get(row["stop_reason"], 0) + 1

        logger.info(
            "J=%02d | val_acc=%.4f val_auc=%.4f | epochs=%d err=%.3e grad=%.3e stop=%s",
            J, row["val_acc"], row["val_auc"],
            row["epochs_used"], row["final_error"], row["final_grad"], row["stop_reason"]
        )

    df_res = pd.DataFrame(results).sort_values("J").reset_index(drop=True)

    out_csv = models_dir / "search_J_results_seed0.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info("[saved] %s", out_csv)

    out_curve = models_dir / "val_metric_vs_J_seed0.png"
    plot_val_curve(df_res, out_curve)
    logger.info("[saved] %s", out_curve)

    metric_col = "val_acc" if MODEL_SELECT_METRIC == "val_acc" else "val_auc"
    df_rank = df_res.sort_values(by=[metric_col, "val_auc", "J"], ascending=[False, False, True])
    best_row = df_rank.iloc[0].to_dict()
    bestJ = int(best_row["J"])

    best_json = models_dir / "best_J_seed0.json"
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "device": str(DEVICE),
                "dtype": str(DTYPE),
                "J_star": bestJ,
                "model_select_metric": MODEL_SELECT_METRIC,
                "tie_break": "val_metric desc, then val_auc desc, then smaller J",
                "stop_counts": stop_counts,
                "lm_config": {
                    "epochs_max": EPOCHS_MAX,
                    "stop_error": STOP_ERROR,
                    "stop_grad": STOP_GRAD,
                    "mu_init": MU_INIT,
                    "mu_dec": MU_DEC,
                    "mu_inc": MU_INC,
                    "max_damping_tries": MAX_DAMP_TRIES,
                    "final_patience": FINAL_PATIENCE,
                    "threshold": THRESH,
                },
            },
            f,
            indent=2,
        )
    logger.info("[saved] %s", best_json)

    logger.info("Stop-reason counts: %s", stop_counts)
    logger.info("Best J*=%d by %s", bestJ, MODEL_SELECT_METRIC)

    # ----------------------------------------
    # 2) final model: train with val to prevent overfit
    #    test evaluated ONCE
    # ----------------------------------------
    seed_all(SEED)
    t0 = time.time()
    m_final = LMMLP(J=bestJ, device=DEVICE, dtype=DTYPE, seed=SEED)

    train_summary_final = m_final.fit_lm(
        X_train=Xtr_t,
        y_train=ytr_t,
        cfg=cfg,
        X_val=Xva_t,
        y_val=yva_t,
        model_select_metric=MODEL_SELECT_METRIC,
        patience=FINAL_PATIENCE,
        threshold=THRESH,
    )
    train_time = float(time.time() - t0)

    p_test = m_final.predict_proba(Xte_t)
    met_test = metrics_from_probs(yte, p_test)

    out_test_probs = probs_dir / "lm_mlp_test_probs_seed0.npz"
    np.savez_compressed(
        out_test_probs,
        y01_test=yte.astype(np.int32),
        p_test=p_test.astype(np.float64),
        threshold_fixed=np.array(THRESH, dtype=np.float64),
    )
    logger.info("[saved] %s", out_test_probs)

    out_roc_maxacc = roc_dir / "lm_mlp_roc_maxacc_seed0.png"
    best_thr_full = plot_roc_with_maxacc(yte, p_test, out_roc_maxacc)
    logger.info("[saved] %s", out_roc_maxacc)

    out_maxacc_csv = maxacc_dir / "lm_mlp_maxacc_seed0.csv"
    pd.DataFrame([best_thr_full]).to_csv(out_maxacc_csv, index=False)
    logger.info("[saved] %s", out_maxacc_csv)

    out_model = models_dir / f"model_84-{bestJ}-1_seed0.pkl"
    with open(out_model, "wb") as f:
        payload = {
            "seed": SEED,
            "J": bestJ,
            "feature_columns": feat_cols,
            "threshold": THRESH,
            "model": m_final.to_pickle_dict(),
        }
        pickle.dump(payload, f)
    logger.info("[saved] %s", out_model)

    out_metrics = out_dir / "lm_mlp_test_metrics_seed0.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "device": str(DEVICE),
                "J": bestJ,
                "threshold": THRESH,
                "train_time_s": train_time,
                "train_stop": train_summary_final,
                "test_metrics": met_test,
            },
            f,
            indent=2,
        )
    logger.info("[saved] %s", out_metrics)

    out_roc = roc_dir / "lm_mlp_roc_seed0.png"
    plot_roc(yte, p_test, out_roc)
    logger.info("[saved] %s", out_roc)

    out_cm = out_dir / "lm_mlp_confmat_seed0.png"
    plot_confmat(yte, p_test, out_cm)
    logger.info("[saved] %s", out_cm)

    # keep your final console summary (nice for humans)
    logger.info("=== FINAL TEST (once) ===")
    logger.info("J*=%d | acc=%.4f se=%.4f sp=%.4f auc=%.4f",
                bestJ, met_test["acc"], met_test["se"], met_test["sp"], met_test["auc"])
    logger.info("confmat: %s", met_test["confusion_matrix"])

    # =========================================================
    # Extra: paper-like tables (MLP) on your split (no set-b)
    # =========================================================
    extra_outputs: list[str] = []
    if run_tables:
        table_dir = tables_dir
        table_dir.mkdir(parents=True, exist_ok=True)

        # reload merged dataframe so we can slice columns by name safely
        df_feat = pd.read_parquet(features_parquet)
        df_split = pd.read_csv(split_csv)
        df_feat["record_id"] = df_feat["record_id"].astype(str)
        df_split["record_id"] = df_split["record_id"].astype(str)
        dfm = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="inner")

        y_pm1_all = dfm["y"].astype(int).to_numpy()
        y01_all = (y_pm1_all == 1).astype(np.int32)
        split_all = dfm["split"].astype(str).to_numpy()

        # helper runs "search J then final train+test once" for chosen columns
        def _run_setting(name: str, sqis: list[str]) -> dict[str, Any]:
            cols = _cols_for_sqis_12lead(sqis)
            for c in cols:
                if c not in dfm.columns:
                    raise ValueError(f"Missing feature column: {c}")

            Xsub = dfm[cols].to_numpy(dtype=np.float64)
            Xtr_s = Xsub[split_all == "train"]; ytr_s = y01_all[split_all == "train"]
            Xva_s = Xsub[split_all == "val"];   yva_s = y01_all[split_all == "val"]
            Xte_s = Xsub[split_all == "test"];  yte_s = y01_all[split_all == "test"]

            Xtr_t_s = torch.tensor(Xtr_s, device=DEVICE, dtype=DTYPE)
            ytr_t_s = torch.tensor(ytr_s, device=DEVICE, dtype=DTYPE)
            Xva_t_s = torch.tensor(Xva_s, device=DEVICE, dtype=DTYPE)
            yva_t_s = torch.tensor(yva_s, device=DEVICE, dtype=DTYPE)
            Xte_t_s = torch.tensor(Xte_s, device=DEVICE, dtype=DTYPE)
            yte_t_s = torch.tensor(yte_s, device=DEVICE, dtype=DTYPE)

            # search J
            Din = len(cols)

            rows = []
            if tables_mode == "fixedJ":
                bestJ_local = int(J_fixed)
            else:
                bestJ_local = None
                best_metric = -1.0

                for J in J_RANGE:
                    m = LMMLP(J=J, D=Din, device=DEVICE, dtype=DTYPE, seed=SEED)
                    summ = m.fit_lm(
                        X_train=Xtr_t_s, y_train=ytr_t_s,
                        cfg=cfg,
                        X_val=Xva_t_s, y_val=yva_t_s,
                        model_select_metric=MODEL_SELECT_METRIC,
                        patience=FINAL_PATIENCE,
                        threshold=THRESH,
                    )
                    p_val = m.predict_proba(Xva_t_s)
                    met_val = metrics_from_probs(yva_s, p_val)
                    val_metric = float(met_val["acc"]) if MODEL_SELECT_METRIC == "val_acc" else float(met_val["auc"])

                    rows.append({
                        "name": name,
                        "sqis": ",".join(sqis),
                        "n_features": len(cols),
                        "J": int(J),
                        "val_acc": float(met_val["acc"]),
                        "val_auc": float(met_val["auc"]),
                        "epochs_used": int(summ["epochs_used"]),
                        "stop_reason": str(summ["stop_reason"]),
                    })

                    if val_metric > best_metric:
                        best_metric = val_metric
                        bestJ_local = int(J)

                assert bestJ_local is not None

            # final train with val early stop
            seed_all(SEED)
            m_final_local = LMMLP(J=bestJ_local, D=Din, device=DEVICE, dtype=DTYPE, seed=SEED)
            _ = m_final_local.fit_lm(
                X_train=Xtr_t_s, y_train=ytr_t_s,
                cfg=cfg,
                X_val=Xva_t_s, y_val=yva_t_s,
                model_select_metric=MODEL_SELECT_METRIC,
                patience=FINAL_PATIENCE,
                threshold=THRESH,
            )

            p_tr = m_final_local.predict_proba(Xtr_t_s)
            p_te = m_final_local.predict_proba(Xte_t_s)

            met_tr = metrics_from_probs(ytr_s, p_tr)
            met_te = metrics_from_probs(yte_s, p_te)
            best_thr = _max_acc_threshold(yte_s, p_te)

            if tables_mode == "fixedJ":
                rows = [{
                    "name": name,
                    "sqis": ",".join(sqis),
                    "n_features": len(cols),
                    "mode": "fixedJ",
                    "J_fixed": int(J_fixed),
                }]

            # write per-setting artifacts
            key = name.replace(" ", "_")
            out_search = table_dir / f"searchJ_{key}_seed{SEED}.csv"
            pd.DataFrame(rows).to_csv(out_search, index=False)

            out_probs = probs_dir / f"{key}_seed{SEED}.npz"
            np.savez_compressed(out_probs, y01_test=yte_s.astype(np.int32), p_test=p_te.astype(np.float64))

            out_roc = roc_dir / f"{key}_maxacc_seed{SEED}.png"
            _ = plot_roc_with_maxacc(yte_s, p_te, out_roc)

            out_maxacc = maxacc_dir / f"{key}_seed{SEED}.csv"
            pd.DataFrame([best_thr]).to_csv(out_maxacc, index=False)

            return {
                "name": name,
                "sqis": ",".join(sqis),
                "n_features": int(len(cols)),
                "J_star": int(bestJ_local),
                "train": met_tr,
                "test": met_te,
                "test_maxacc": best_thr,
                "search_csv": str(out_search),
                "probs_npz": str(out_probs),
                "roc_png": str(out_roc),
                "maxacc_csv": str(out_maxacc),
            }

        # -------- Table 5 --------
        logger.info("Table5-like (MLP): each SQI individually on 12-lead")
        rows5 = []
        for s in SQI_LIST:
            res = _run_setting(name=s, sqis=[s])
            rows5.append({
                "SQI": s,
                "n_features": res["n_features"],
                "J_star": res["J_star"],
                "Ac_train": res["train"]["acc"],
                "Se_train": res["train"]["se"],
                "Sp_train": res["train"]["sp"],
                "Ac_test":  res["test"]["acc"],
                "Se_test":  res["test"]["se"],
                "Sp_test":  res["test"]["sp"],
                "AUC_test": res["test"]["auc"],
                "maxAcc_thr_test": res["test_maxacc"]["threshold"],
                "maxAcc_test": res["test_maxacc"]["acc"],
            })
            logger.info("  %-6s | J*=%d | test Ac=%.4f", s, res["J_star"], res["test"]["acc"])

        out5 = table_dir / f"table5_mlp_12lead_single_sqi_seed{SEED}.csv"
        pd.DataFrame(rows5).to_csv(out5, index=False)
        logger.info("[saved] %s", out5)

        # -------- Table 6 --------
        logger.info("Table6-like (MLP): SQI combinations on 12-lead")
        rows6 = []
        for grp, sqis in COMBOS:
            res = _run_setting(name=grp, sqis=sqis)
            rows6.append({
                "Group": grp,
                "Selected_SQI": ",".join(sqis),
                "n_features": res["n_features"],
                "J_star": res["J_star"],
                "Ac_train": res["train"]["acc"],
                "Ac_test":  res["test"]["acc"],
                "AUC_test": res["test"]["auc"],
                "maxAcc_thr_test": res["test_maxacc"]["threshold"],
                "maxAcc_test": res["test_maxacc"]["acc"],
            })
            logger.info("  %-11s | J*=%d | test Ac=%.4f", grp, res["J_star"], res["test"]["acc"])

        out6 = table_dir / f"table6_mlp_12lead_combo_sqi_seed{SEED}.csv"
        pd.DataFrame(rows6).to_csv(out6, index=False)
        logger.info("[saved] %s", out6)

        # -------- Table 7 --------
        logger.info("Table7-like (MLP): selected five SQIs on 12-lead (your split proxy)")
        res7 = _run_setting(name="Selected5", sqis=SELECTED_5)
        out7 = table_dir / f"table7_mlp_selected5_seed{SEED}.csv"
        pd.DataFrame([{
            "Setting": "your split: train/train+val earlystop, test once",
            "Selected_SQI": ",".join(SELECTED_5),
            "n_features": res7["n_features"],
            "J_star": res7["J_star"],
            "Ac_train": res7["train"]["acc"],
            "Se_train": res7["train"]["se"],
            "Sp_train": res7["train"]["sp"],
            "Ac_test":  res7["test"]["acc"],
            "Se_test":  res7["test"]["se"],
            "Sp_test":  res7["test"]["sp"],
            "AUC_test": res7["test"]["auc"],
            "maxAcc_thr_test": res7["test_maxacc"]["threshold"],
            "maxAcc_test": res7["test_maxacc"]["acc"],
        }]).to_csv(out7, index=False)
        logger.info("[saved] %s", out7)

        extra_outputs.extend([str(out5), str(out6), str(out7)])

    outs = [
               str(out_csv),
               str(out_curve),
               str(best_json),
               str(out_model),
               str(out_metrics),
               str(out_roc),
               str(out_cm),
               str(out_test_probs),
               str(out_roc_maxacc),
               str(out_maxacc_csv),
           ] + extra_outputs

    return {"step": "lm_mlp_search", "skipped": False, "outputs": outs}


def main() -> None:
    params = {
        "verbose": False,
        "force": False,
        "tables": True
        # 可选 overrides:
        # "device": "cuda",
        # "dtype": "float64",
    }
    run(params)

if __name__ == "__main__":
    main()
