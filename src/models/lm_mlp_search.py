# src/models/lm_mlp_search.py
from __future__ import annotations

import json
import pickle
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import torch

from src.utils.paths import project_root
from src.models.lm_mlp import LMMLP, LMConfig


logger = logging.getLogger(__name__)


# ============================================================
# Config / Data containers
# ============================================================

TablesMode = Literal["fixedJ", "searchJ"]


@dataclass(frozen=True)
class RunConfig:
    # core
    seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float64

    # search
    j_range: tuple[int, ...] = tuple(range(2, 18))  # 2..17
    model_select_metric: Literal["val_acc", "val_auc"] = "val_acc"

    # data
    features_parquet: Path = Path("record84_norm.parquet")
    split_csv: Path = Path("split_seta_seed0_balanced.csv")

    # training config (paper)
    epochs_max: int = 100
    stop_error: float = 1e-5
    stop_grad: float = 1e-5

    # early stop for final model
    final_patience: int = 15
    threshold_fixed: float = 0.7

    # LM damping
    mu_init: float = 1e-3
    mu_dec: float = 0.1
    mu_inc: float = 10.0
    max_damping_tries: int = 10

    # tables
    tables: bool = False
    tables_mode: TablesMode = "fixedJ"
    j_fixed: int = 12
    # when tables=True, choose whether to reuse merged df from main load
    reuse_merged_df: bool = True

    # runtime flags
    verbose: bool = False
    force: bool = False

    @property
    def lm_cfg(self) -> LMConfig:
        return LMConfig(
            epochs_max=self.epochs_max,
            stop_error=self.stop_error,
            stop_grad=self.stop_grad,
            mu_init=self.mu_init,
            mu_dec=self.mu_dec,
            mu_inc=self.mu_inc,
            max_damping_tries=self.max_damping_tries,
        )


@dataclass(frozen=True)
class Paths:
    root: Path
    features_parquet: Path
    split_csv: Path
    out_dir: Path

    models_dir: Path
    tables_dir: Path
    roc_dir: Path
    probs_dir: Path
    maxacc_dir: Path

    def mkdirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.roc_dir.mkdir(parents=True, exist_ok=True)
        self.probs_dir.mkdir(parents=True, exist_ok=True)
        self.maxacc_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DataSplit:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    feat_cols: list[str]
    # optional: merged df for tables slicing by column name
    dfm: pd.DataFrame | None = None
    split_all: np.ndarray | None = None
    y01_all: np.ndarray | None = None


@dataclass(frozen=True)
class SearchResult:
    df_res: pd.DataFrame
    bestJ: int
    stop_counts: dict[str, int]
    best_row: dict[str, Any]


@dataclass(frozen=True)
class FinalResult:
    bestJ: int
    train_summary: dict[str, Any]
    train_time_s: float
    p_test: np.ndarray
    test_metrics_fixed: dict[str, Any]
    test_bestthr: dict[str, Any]


# ============================================================
# Fixed table definitions (keep as module constants)
# ============================================================

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SQI_LIST = ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]

COMBOS = [
    ("Pairs",       ["iSQI", "basSQI"]),
    ("Triplets",    ["bSQI", "basSQI", "pSQI"]),
    ("Quadruplets", ["bSQI", "basSQI", "kSQI", "sSQI"]),
    ("Quintuplets", ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]),
    ("Sextuplets",  ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI"]),
    ("All SQI",     ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]),
]
SELECTED_5 = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]


# ============================================================
# Utilities
# ============================================================

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cfg(params: dict[str, Any]) -> RunConfig:
    """
    Convert params dict into an immutable RunConfig.
    Keeps defaults identical to your current script.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    tables = bool(params.get("tables", False))
    tables_mode = str(params.get("tables_mode", "fixedJ"))
    j_fixed = int(params.get("J_fixed", 12))  # keep your param key

    seed = int(params.get("seed", 0)) if params.get("seed") is not None else 0

    # device override
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.get("device") is not None:
        d = str(params["device"]).lower()
        if d == "cpu":
            device = torch.device("cpu")
        elif d == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dtype override
    dtype = torch.float64
    if params.get("dtype") is not None:
        dt = str(params["dtype"]).lower()
        if dt in ("float32", "float", "f32"):
            dtype = torch.float32
        elif dt in ("float64", "double", "f64"):
            dtype = torch.float64

    # paths can be overridden as strings; Paths() will finalize defaults
    feat = Path(str(params.get("features_parquet"))) if params.get("features_parquet") else Path("record84_norm.parquet")
    spl = Path(str(params.get("split_csv"))) if params.get("split_csv") else Path("split_seta_seed0_balanced.csv")

    return RunConfig(
        seed=seed,
        device=device,
        dtype=dtype,
        features_parquet=feat,
        split_csv=spl,
        tables=tables,
        tables_mode="searchJ" if tables_mode == "searchJ" else "fixedJ",
        j_fixed=j_fixed,
        verbose=verbose,
        force=force,
    )


def prepare_paths(cfg: RunConfig, params: dict[str, Any]) -> Paths:
    root = project_root()

    features_parquet = Path(
        str(params.get("features_parquet") or (root / "artifacts" / "features" / "record84_norm.parquet"))
    )
    split_csv = Path(
        str(params.get("split_csv") or (root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv"))
    )
    out_dir = Path(str(params.get("out_dir") or (root / "artifacts" / "models" / "lm_mlp")))

    return Paths(
        root=root,
        features_parquet=features_parquet,
        split_csv=split_csv,
        out_dir=out_dir,
        models_dir=out_dir / "models",
        tables_dir=out_dir / "tables",
        roc_dir=out_dir / "roc",
        probs_dir=out_dir / "probs",
        maxacc_dir=out_dir / "maxacc",
    )


def outputs_exist(paths: Paths, seed: int) -> bool:
    out_csv = paths.models_dir / f"search_J_results_seed{seed}.csv"
    best_json = paths.models_dir / f"best_J_seed{seed}.json"
    out_metrics = paths.out_dir / f"lm_mlp_test_metrics_seed{seed}.json"
    out_model_any = any(paths.models_dir.glob(f"model_*-*_seed{seed}.pkl")) or any(paths.models_dir.glob(f"model_84-*-1_seed{seed}.pkl"))

    if not (out_csv.exists() and out_csv.stat().st_size > 0):
        return False
    if not (best_json.exists() and best_json.stat().st_size > 0):
        return False
    if not out_model_any:
        return False
    if not (out_metrics.exists() and out_metrics.stat().st_size > 0):
        return False
    return True


def load_split_data(paths: Paths, cfg: RunConfig, keep_merged_df: bool = False) -> DataSplit:
    df_feat = pd.read_parquet(paths.features_parquet)
    df_split = pd.read_csv(paths.split_csv)

    df_feat["record_id"] = df_feat["record_id"].astype(str)
    df_split["record_id"] = df_split["record_id"].astype(str)

    dfm = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="inner")

    y_pm1 = dfm["y"].astype(int).to_numpy()
    y01 = (y_pm1 == 1).astype(np.int32)
    split = dfm["split"].astype(str).to_numpy()

    drop_cols = {"record_id", "y", "split"}
    feat_cols = sorted([c for c in dfm.columns if c not in drop_cols])

    X = dfm[feat_cols].to_numpy(dtype=np.float64)
    if X.shape[1] != 84:
        raise ValueError(f"Expected 84 features, got {X.shape[1]}")

    Xtr = X[split == "train"]; ytr = y01[split == "train"]
    Xva = X[split == "val"];   yva = y01[split == "val"]
    Xte = X[split == "test"];  yte = y01[split == "test"]

    logger.info("features: %s", paths.features_parquet)
    logger.info("split_csv: %s", paths.split_csv)
    logger.info("X shape: %s (#features=%d)", X.shape, X.shape[1])
    logger.info("splits: train=%d val=%d test=%d",
                int((split == "train").sum()), int((split == "val").sum()), int((split == "test").sum()))
    logger.info("labels(train): good=%d bad=%d",
                int(ytr.sum()), int(len(ytr) - int(ytr.sum())))

    return DataSplit(
        Xtr=Xtr, ytr=ytr,
        Xva=Xva, yva=yva,
        Xte=Xte, yte=yte,
        feat_cols=feat_cols,
        dfm=dfm if keep_merged_df else None,
        split_all=split if keep_merged_df else None,
        y01_all=y01 if keep_merged_df else None,
    )


def to_tensors(X: np.ndarray, y: np.ndarray, cfg: RunConfig) -> tuple[torch.Tensor, torch.Tensor]:
    Xt = torch.tensor(X, device=cfg.device, dtype=cfg.dtype)
    yt = torch.tensor(y, device=cfg.device, dtype=cfg.dtype)
    return Xt, yt


def compute_metrics(y01: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y01 = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    pred = (p > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y01, pred, labels=[0, 1]).ravel()
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
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_maxacc_threshold(y01: np.ndarray, p: np.ndarray, n_grid: int = 2001) -> dict[str, Any]:
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


def plot_roc(y01: np.ndarray, p: np.ndarray, out_png: Path, title: str = "ROC (test)") -> None:
    fpr, tpr, _ = roc_curve(y01.astype(int), p.astype(np.float64))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_roc_with_maxacc(y01: np.ndarray, p: np.ndarray, out_png: Path) -> dict[str, Any]:
    fpr, tpr, thr = roc_curve(y01.astype(int), p.astype(np.float64))
    best = find_maxacc_threshold(y01, p)

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
    ax.set_title(f"ROC (test) + maxAcc@thr={best['threshold']:.3f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return best


def plot_confmat(y01: np.ndarray, p: np.ndarray, threshold: float, out_png: Path) -> None:
    pred = (p > threshold).astype(int)
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


# ============================================================
# Training: search J and final fit
# ============================================================

def search_best_J(data: DataSplit, cfg: RunConfig) -> SearchResult:
    Xtr_t, ytr_t = to_tensors(data.Xtr, data.ytr, cfg)
    Xva_t, yva_t = to_tensors(data.Xva, data.yva, cfg)

    results: list[dict[str, Any]] = []
    stop_counts: dict[str, int] = {}

    logger.info("[Task 1.7] LM-MLP search J=%d..%d", min(cfg.j_range), max(cfg.j_range))
    for J in cfg.j_range:
        t0 = time.time()
        m = LMMLP(J=int(J), device=cfg.device, dtype=cfg.dtype, seed=cfg.seed)

        train_summary = m.fit_lm(
            X_train=Xtr_t,
            y_train=ytr_t,
            cfg=cfg.lm_cfg,
            X_val=Xva_t,
            y_val=yva_t,
            model_select_metric=cfg.model_select_metric,
            patience=cfg.final_patience,
            threshold=cfg.threshold_fixed,
        )

        p_val = m.predict_proba(Xva_t)
        met_val = compute_metrics(data.yva, p_val, threshold=cfg.threshold_fixed)

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

    metric_col = "val_acc" if cfg.model_select_metric == "val_acc" else "val_auc"
    df_rank = df_res.sort_values(by=[metric_col, "val_auc", "J"], ascending=[False, False, True])
    best_row = df_rank.iloc[0].to_dict()
    bestJ = int(best_row["J"])

    logger.info("Stop-reason counts: %s", stop_counts)
    logger.info("Best J*=%d by %s", bestJ, cfg.model_select_metric)

    return SearchResult(df_res=df_res, bestJ=bestJ, stop_counts=stop_counts, best_row=best_row)


def train_final(data: DataSplit, cfg: RunConfig, bestJ: int) -> FinalResult:
    seed_all(cfg.seed)

    Xtr_t, ytr_t = to_tensors(data.Xtr, data.ytr, cfg)
    Xva_t, yva_t = to_tensors(data.Xva, data.yva, cfg)
    Xte_t, _ = to_tensors(data.Xte, data.yte, cfg)

    t0 = time.time()
    m_final = LMMLP(J=int(bestJ), device=cfg.device, dtype=cfg.dtype, seed=cfg.seed)
    train_summary = m_final.fit_lm(
        X_train=Xtr_t,
        y_train=ytr_t,
        cfg=cfg.lm_cfg,
        X_val=Xva_t,
        y_val=yva_t,
        model_select_metric=cfg.model_select_metric,
        patience=cfg.final_patience,
        threshold=cfg.threshold_fixed,
    )
    train_time_s = float(time.time() - t0)

    p_test = m_final.predict_proba(Xte_t)
    test_metrics_fixed = compute_metrics(data.yte, p_test, threshold=cfg.threshold_fixed)
    test_bestthr = find_maxacc_threshold(data.yte, p_test)

    # store model payload (pickle dict) in train_summary? no — return only what is needed,
    # model serialization happens in save_main_artifacts for consistency.
    # But we DO need model object to pickle. We'll re-instantiate? expensive.
    # Easiest: return model pickle dict via train_summary bundle.
    # We’ll tuck it under train_summary for now (small and explicit).
    train_summary = dict(train_summary)
    train_summary["_model_pickle_dict"] = m_final.to_pickle_dict()

    return FinalResult(
        bestJ=int(bestJ),
        train_summary=train_summary,
        train_time_s=train_time_s,
        p_test=p_test,
        test_metrics_fixed=test_metrics_fixed,
        test_bestthr=test_bestthr,
    )


# ============================================================
# Saving artifacts (main)
# ============================================================

def save_main_artifacts(paths: Paths, cfg: RunConfig, data: DataSplit, search: SearchResult, final: FinalResult) -> list[str]:
    outs: list[str] = []

    # 1) search CSV
    out_csv = paths.models_dir / f"search_J_results_seed{cfg.seed}.csv"
    search.df_res.to_csv(out_csv, index=False)
    outs.append(str(out_csv))
    logger.info("[saved] %s", out_csv)

    # 2) val curve
    out_curve = paths.models_dir / f"val_metric_vs_J_seed{cfg.seed}.png"
    plot_val_curve(search.df_res, out_curve)
    outs.append(str(out_curve))
    logger.info("[saved] %s", out_curve)

    # 3) best json
    best_json = paths.models_dir / f"best_J_seed{cfg.seed}.json"
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": cfg.seed,
                "device": str(cfg.device),
                "dtype": str(cfg.dtype),
                "J_star": final.bestJ,
                "model_select_metric": cfg.model_select_metric,
                "tie_break": "val_metric desc, then val_auc desc, then smaller J",
                "stop_counts": search.stop_counts,
                "lm_config": {
                    "epochs_max": cfg.epochs_max,
                    "stop_error": cfg.stop_error,
                    "stop_grad": cfg.stop_grad,
                    "mu_init": cfg.mu_init,
                    "mu_dec": cfg.mu_dec,
                    "mu_inc": cfg.mu_inc,
                    "max_damping_tries": cfg.max_damping_tries,
                    "final_patience": cfg.final_patience,
                    "threshold_fixed": cfg.threshold_fixed,
                },
            },
            f,
            indent=2,
        )
    outs.append(str(best_json))
    logger.info("[saved] %s", best_json)

    # 4) model pickle
    out_model = paths.models_dir / f"model_84-{final.bestJ}-1_seed{cfg.seed}.pkl"
    with open(out_model, "wb") as f:
        payload = {
            "seed": cfg.seed,
            "J": final.bestJ,
            "feature_columns": data.feat_cols,
            "threshold": cfg.threshold_fixed,
            "model": final.train_summary["_model_pickle_dict"],
        }
        pickle.dump(payload, f)
    outs.append(str(out_model))
    logger.info("[saved] %s", out_model)

    # 5) metrics json
    out_metrics = paths.out_dir / f"lm_mlp_test_metrics_seed{cfg.seed}.json"
    train_summary_clean = dict(final.train_summary)
    train_summary_clean.pop("_model_pickle_dict", None)

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": cfg.seed,
                "device": str(cfg.device),
                "J": final.bestJ,
                "threshold_fixed": cfg.threshold_fixed,
                "train_time_s": final.train_time_s,
                "train_stop": train_summary_clean,
                "test_metrics_fixed": final.test_metrics_fixed,
                "test_maxacc": final.test_bestthr,
            },
            f,
            indent=2,
        )
    outs.append(str(out_metrics))
    logger.info("[saved] %s", out_metrics)

    # 6) test probs npz
    out_test_probs = paths.probs_dir / f"lm_mlp_test_probs_seed{cfg.seed}.npz"
    np.savez_compressed(
        out_test_probs,
        y01_test=data.yte.astype(np.int32),
        p_test=final.p_test.astype(np.float64),
        threshold_fixed=np.array(cfg.threshold_fixed, dtype=np.float64),
    )
    outs.append(str(out_test_probs))
    logger.info("[saved] %s", out_test_probs)

    # 7) ROC (fixed)
    out_roc = paths.roc_dir / f"lm_mlp_roc_seed{cfg.seed}.png"
    plot_roc(data.yte, final.p_test, out_roc, title="ROC (test)")
    outs.append(str(out_roc))
    logger.info("[saved] %s", out_roc)

    # 8) ROC + maxAcc
    out_roc_maxacc = paths.roc_dir / f"lm_mlp_roc_maxacc_seed{cfg.seed}.png"
    best_thr_full = plot_roc_with_maxacc(data.yte, final.p_test, out_roc_maxacc)
    outs.append(str(out_roc_maxacc))
    logger.info("[saved] %s", out_roc_maxacc)

    # 9) maxacc csv
    out_maxacc_csv = paths.maxacc_dir / f"lm_mlp_maxacc_seed{cfg.seed}.csv"
    pd.DataFrame([best_thr_full]).to_csv(out_maxacc_csv, index=False)
    outs.append(str(out_maxacc_csv))
    logger.info("[saved] %s", out_maxacc_csv)

    # 10) confmat (fixed threshold)
    out_cm = paths.out_dir / f"lm_mlp_confmat_seed{cfg.seed}.png"
    plot_confmat(data.yte, final.p_test, threshold=cfg.threshold_fixed, out_png=out_cm)
    outs.append(str(out_cm))
    logger.info("[saved] %s", out_cm)

    # console summary
    logger.info("=== FINAL TEST (once) ===")
    logger.info("J*=%d | acc=%.4f se=%.4f sp=%.4f auc=%.4f",
                final.bestJ,
                final.test_metrics_fixed["acc"],
                final.test_metrics_fixed["se"],
                final.test_metrics_fixed["sp"],
                final.test_metrics_fixed["auc"])
    logger.info("confmat: %s", final.test_metrics_fixed["confusion_matrix"])

    return outs


# ============================================================
# Tables (optional) — keep logic, but move out of run()
# ============================================================

def _cols_for_sqis_12lead(sqis: list[str]) -> list[str]:
    return [f"{ld}__{s}" for s in sqis for ld in LEADS_12]


@dataclass(frozen=True)
class SettingResult:
    name: str
    sqis: list[str]
    n_features: int
    J_star: int
    train_fixed: dict[str, Any]
    test_fixed: dict[str, Any]
    test_maxacc: dict[str, Any]

    search_csv: str
    probs_npz: str
    roc_png: str
    maxacc_csv: str


def fit_eval_setting_from_dfm(
    *,
    dfm: pd.DataFrame,
    split_all: np.ndarray,
    y01_all: np.ndarray,
    cfg: RunConfig,
    paths: Paths,
    name: str,
    sqis: list[str],
    bestJ_global: int,
) -> SettingResult:
    cols = _cols_for_sqis_12lead(sqis)
    for c in cols:
        if c not in dfm.columns:
            raise ValueError(f"Missing feature column: {c}")

    Xsub = dfm[cols].to_numpy(dtype=np.float64)
    Xtr = Xsub[split_all == "train"]; ytr = y01_all[split_all == "train"]
    Xva = Xsub[split_all == "val"];   yva = y01_all[split_all == "val"]
    Xte = Xsub[split_all == "test"];  yte = y01_all[split_all == "test"]

    Xtr_t, ytr_t = to_tensors(Xtr, ytr, cfg)
    Xva_t, yva_t = to_tensors(Xva, yva, cfg)
    Xte_t, _ = to_tensors(Xte, yte, cfg)

    Din = len(cols)

    # --- choose J ---
    rows: list[dict[str, Any]] = []
    if cfg.tables_mode == "fixedJ":
        bestJ_local = int(bestJ_global if bestJ_global > 0 else cfg.j_fixed)
        rows = [{
            "name": name,
            "sqis": ",".join(sqis),
            "n_features": Din,
            "mode": "fixedJ",
            "J_fixed": int(bestJ_local),
        }]
    else:
        bestJ_local = None
        best_metric = -1.0
        for J in cfg.j_range:
            m = LMMLP(J=int(J), D=Din, device=cfg.device, dtype=cfg.dtype, seed=cfg.seed)
            summ = m.fit_lm(
                X_train=Xtr_t, y_train=ytr_t,
                cfg=cfg.lm_cfg,
                X_val=Xva_t, y_val=yva_t,
                model_select_metric=cfg.model_select_metric,
                patience=cfg.final_patience,
                threshold=cfg.threshold_fixed,
            )
            p_val = m.predict_proba(Xva_t)
            met_val = compute_metrics(yva, p_val, threshold=cfg.threshold_fixed)
            val_metric = float(met_val["acc"]) if cfg.model_select_metric == "val_acc" else float(met_val["auc"])

            rows.append({
                "name": name,
                "sqis": ",".join(sqis),
                "n_features": Din,
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

    # --- final fit ---
    seed_all(cfg.seed)
    m_final = LMMLP(J=int(bestJ_local), D=Din, device=cfg.device, dtype=cfg.dtype, seed=cfg.seed)
    _ = m_final.fit_lm(
        X_train=Xtr_t, y_train=ytr_t,
        cfg=cfg.lm_cfg,
        X_val=Xva_t, y_val=yva_t,
        model_select_metric=cfg.model_select_metric,
        patience=cfg.final_patience,
        threshold=cfg.threshold_fixed,
    )

    p_tr = m_final.predict_proba(Xtr_t)
    p_te = m_final.predict_proba(Xte_t)

    met_tr = compute_metrics(ytr, p_tr, threshold=cfg.threshold_fixed)
    met_te = compute_metrics(yte, p_te, threshold=cfg.threshold_fixed)
    best_thr = find_maxacc_threshold(yte, p_te)

    # --- write per-setting artifacts ---
    key = name.replace(" ", "_")
    out_search = paths.tables_dir / f"searchJ_{key}_seed{cfg.seed}.csv"
    pd.DataFrame(rows).to_csv(out_search, index=False)

    out_probs = paths.probs_dir / f"{key}_seed{cfg.seed}.npz"
    np.savez_compressed(out_probs, y01_test=yte.astype(np.int32), p_test=p_te.astype(np.float64))

    out_roc = paths.roc_dir / f"{key}_maxacc_seed{cfg.seed}.png"
    _ = plot_roc_with_maxacc(yte, p_te, out_roc)

    out_maxacc = paths.maxacc_dir / f"{key}_seed{cfg.seed}.csv"
    pd.DataFrame([best_thr]).to_csv(out_maxacc, index=False)

    return SettingResult(
        name=name,
        sqis=sqis,
        n_features=Din,
        J_star=int(bestJ_local),
        train_fixed=met_tr,
        test_fixed=met_te,
        test_maxacc=best_thr,
        search_csv=str(out_search),
        probs_npz=str(out_probs),
        roc_png=str(out_roc),
        maxacc_csv=str(out_maxacc),
    )


def run_tables(data: DataSplit, cfg: RunConfig, paths: Paths, bestJ_global: int) -> list[str]:
    if data.dfm is None or data.split_all is None or data.y01_all is None:
        raise ValueError("tables=True requires merged dfm/split_all/y01_all. Use load_split_data(..., keep_merged_df=True).")

    outs: list[str] = []
    dfm = data.dfm
    split_all = data.split_all
    y01_all = data.y01_all

    # -------- Table 5 --------
    logger.info("Table5-like (MLP): each SQI individually on 12-lead")
    rows5: list[dict[str, Any]] = []
    for s in SQI_LIST:
        res = fit_eval_setting_from_dfm(
            dfm=dfm, split_all=split_all, y01_all=y01_all,
            cfg=cfg, paths=paths, name=s, sqis=[s],
            bestJ_global=bestJ_global,
        )
        rows5.append({
            "SQI": s,
            "n_features": res.n_features,
            "J_star": res.J_star,
            "Ac_train": res.train_fixed["acc"],
            "Se_train": res.train_fixed["se"],
            "Sp_train": res.train_fixed["sp"],
            "Ac_test":  res.test_fixed["acc"],
            "Se_test":  res.test_fixed["se"],
            "Sp_test":  res.test_fixed["sp"],
            "AUC_test": res.test_fixed["auc"],
            "maxAcc_thr_test": res.test_maxacc["threshold"],
            "maxAcc_test": res.test_maxacc["acc"],
        })
        logger.info("  %-6s | J*=%d | test Ac=%.4f", s, res.J_star, res.test_fixed["acc"])
        outs.extend([res.search_csv, res.probs_npz, res.roc_png, res.maxacc_csv])

    out5 = paths.tables_dir / f"table5_mlp_12lead_single_sqi_seed{cfg.seed}.csv"
    pd.DataFrame(rows5).to_csv(out5, index=False)
    logger.info("[saved] %s", out5)
    outs.append(str(out5))

    # -------- Table 6 --------
    logger.info("Table6-like (MLP): SQI combinations on 12-lead")
    rows6: list[dict[str, Any]] = []
    for grp, sqis in COMBOS:
        res = fit_eval_setting_from_dfm(
            dfm=dfm, split_all=split_all, y01_all=y01_all,
            cfg=cfg, paths=paths, name=grp, sqis=sqis,
            bestJ_global=bestJ_global,
        )
        rows6.append({
            "Group": grp,
            "Selected_SQI": ",".join(sqis),
            "n_features": res.n_features,
            "J_star": res.J_star,
            "Ac_train": res.train_fixed["acc"],
            "Ac_test":  res.test_fixed["acc"],
            "AUC_test": res.test_fixed["auc"],
            "maxAcc_thr_test": res.test_maxacc["threshold"],
            "maxAcc_test": res.test_maxacc["acc"],
        })
        logger.info("  %-11s | J*=%d | test Ac=%.4f", grp, res.J_star, res.test_fixed["acc"])
        outs.extend([res.search_csv, res.probs_npz, res.roc_png, res.maxacc_csv])

    out6 = paths.tables_dir / f"table6_mlp_12lead_combo_sqi_seed{cfg.seed}.csv"
    pd.DataFrame(rows6).to_csv(out6, index=False)
    logger.info("[saved] %s", out6)
    outs.append(str(out6))

    # -------- Table 7 --------
    logger.info("Table7-like (MLP): selected five SQIs on 12-lead (your split proxy)")
    res7 = fit_eval_setting_from_dfm(
        dfm=dfm, split_all=split_all, y01_all=y01_all,
        cfg=cfg, paths=paths, name="Selected5", sqis=SELECTED_5,
        bestJ_global=bestJ_global,
    )
    out7 = paths.tables_dir / f"table7_mlp_selected5_seed{cfg.seed}.csv"
    pd.DataFrame([{
        "Setting": "your split: train/train+val earlystop, test once",
        "Selected_SQI": ",".join(SELECTED_5),
        "n_features": res7.n_features,
        "J_star": res7.J_star,
        "Ac_train": res7.train_fixed["acc"],
        "Se_train": res7.train_fixed["se"],
        "Sp_train": res7.train_fixed["sp"],
        "Ac_test":  res7.test_fixed["acc"],
        "Se_test":  res7.test_fixed["se"],
        "Sp_test":  res7.test_fixed["sp"],
        "AUC_test": res7.test_fixed["auc"],
        "maxAcc_thr_test": res7.test_maxacc["threshold"],
        "maxAcc_test": res7.test_maxacc["acc"],
    }]).to_csv(out7, index=False)
    logger.info("[saved] %s", out7)
    outs.extend([res7.search_csv, res7.probs_npz, res7.roc_png, res7.maxacc_csv, str(out7)])

    return outs


# ============================================================
# Orchestrator run()
# ============================================================

def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - seed: int
      - device: str ("cuda"|"cpu")
      - dtype: str ("float64"|"float32")
      - features_parquet: str
      - split_csv: str
      - out_dir: str
      - tables: bool
      - tables_mode: "fixedJ"|"searchJ"
      - J_fixed: int  (only for tables_mode="fixedJ")
    """
    cfg = build_cfg(params)
    _setup_logging(cfg.verbose)
    seed_all(cfg.seed)

    paths = prepare_paths(cfg, params)
    paths.mkdirs()

    logger.info("device: %s", cfg.device)
    logger.info("dtype: %s", cfg.dtype)

    if (not cfg.force) and outputs_exist(paths, cfg.seed):
        logger.info("lm_mlp_search: outputs exist -> skip (set force=True to rerun)")
        outs = [
            str(paths.models_dir / f"search_J_results_seed{cfg.seed}.csv"),
            str(paths.models_dir / f"best_J_seed{cfg.seed}.json"),
            str(paths.out_dir / f"lm_mlp_test_metrics_seed{cfg.seed}.json"),
        ]
        return {"step": "lm_mlp_search", "skipped": True, "outputs": outs}

    # load once; keep merged df only if tables
    data = load_split_data(paths, cfg, keep_merged_df=cfg.tables and cfg.reuse_merged_df)

    # search + final
    search = search_best_J(data, cfg)
    final = train_final(data, cfg, bestJ=search.bestJ)

    # save main artifacts
    outs = save_main_artifacts(paths, cfg, data, search, final)

    # optional tables
    if cfg.tables:
        outs.extend(run_tables(data, cfg, paths, bestJ_global=final.bestJ))

    return {"step": "lm_mlp_search", "skipped": False, "outputs": outs}


def main() -> None:
    params = {
        "verbose": False,
        "force": True,
        "tables": True,
        # "tables_mode": "fixedJ",  # recommended default
        # "J_fixed": 12,
        # "device": "cuda",
        # "dtype": "float64",
    }
    run(params)


if __name__ == "__main__":
    main()
