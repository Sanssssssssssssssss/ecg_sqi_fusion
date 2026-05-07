# src/models/gnb_baseline.py
from __future__ import annotations

import json
import pickle
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
try:
    # sklearn>=1.8 path
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover
    FrozenEstimator = None  # type: ignore

from src.utils.paths import project_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0

    # data
    features_parquet: Path = Path("record84_norm.parquet")
    split_csv: Path = Path("split_seta_seed0_balanced.csv")

    # feature selection
    feature_set: Literal["all84", "selected5", "single1"] = "all84"
    single_sqi: str = "fSQI"   # used when feature_set == "single1"

    # evaluation
    threshold_fixed: float = 0.5
    maxacc_grid: int = 2001

    # runtime
    verbose: bool = False
    force: bool = False

        # gnb knobs
    var_smoothing: float = 1e-3

    # threshold selection (on val)
    threshold_metric: Literal["acc", "youden"] = "acc"
    use_calibration: bool = False
    calib_method: Literal["sigmoid", "isotonic"] = "sigmoid"


@dataclass(frozen=True)
class Paths:
    root: Path
    features_parquet: Path
    split_csv: Path
    out_dir: Path

    models_dir: Path
    roc_dir: Path
    probs_dir: Path
    maxacc_dir: Path

    def mkdirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.roc_dir.mkdir(parents=True, exist_ok=True)
        self.probs_dir.mkdir(parents=True, exist_ok=True)
        self.maxacc_dir.mkdir(parents=True, exist_ok=True)


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SELECTED_5 = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]
SINGLE_1 = ["fSQI"]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def seed_all(seed: int) -> None:
    np.random.seed(seed)


def _cols_for_sqis_12lead(sqis: list[str]) -> list[str]:
    return [f"{ld}__{s}" for s in sqis for ld in LEADS_12]


def prepare_paths(cfg: RunConfig, params: dict[str, Any]) -> Paths:
    root = project_root()

    features_parquet = Path(str(params.get("features_parquet") or (root / "artifacts" / "features" / "record84_norm.parquet")))
    split_csv = Path(str(params.get("split_csv") or (root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv")))

    out_dir = Path(str(params.get("out_dir") or (root / "artifacts" / "models" / "baselines" / "gnb")))

    return Paths(
        root=root,
        features_parquet=features_parquet,
        split_csv=split_csv,
        out_dir=out_dir,
        models_dir=out_dir / "models",
        roc_dir=out_dir / "roc",
        probs_dir=out_dir / "probs",
        maxacc_dir=out_dir / "maxacc",
    )


def _read_inputs(paths: Paths, feature_set: str, single_sqi: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df_feat = pd.read_parquet(paths.features_parquet)
    df_split = pd.read_csv(paths.split_csv)

    df_feat["record_id"] = df_feat["record_id"].astype(str)
    df_split["record_id"] = df_split["record_id"].astype(str)

    dfm = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="inner")

    if "y" not in dfm.columns:
        raise ValueError("features parquet must contain column 'y' with ±1 labels.")

    y_pm1 = dfm["y"].astype(int).to_numpy()
    y01 = (y_pm1 == 1).astype(np.int32)
    split = dfm["split"].astype(str).to_numpy()

    drop_cols = {"record_id", "y", "split"}

    if feature_set == "selected5":
        feat_cols = _cols_for_sqis_12lead(SELECTED_5)
        missing = [c for c in feat_cols if c not in dfm.columns]
        if missing:
            raise ValueError(f"Missing selected5 columns: {missing[:5]} ... total={len(missing)}")

    elif feature_set == "single1":
        sqi = str(single_sqi)
        feat_cols = _cols_for_sqis_12lead([sqi])
        missing = [c for c in feat_cols if c not in dfm.columns]
        if missing:
            raise ValueError(f"Missing single1 columns for sqi={sqi}: {missing[:5]} ... total={len(missing)}")

    else:
        feat_cols = sorted([c for c in dfm.columns if c not in drop_cols])

    X = dfm[feat_cols].to_numpy(dtype=np.float64)

    return X, y01, split, feat_cols


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

def select_threshold_on_val(y01: np.ndarray, p: np.ndarray, metric: str = "acc", n_grid: int = 2001) -> dict[str, Any]:
    """
    Select threshold using validation set only.
    metric:
      - "acc": maximize accuracy
      - "youden": maximize Youden's J = TPR - FPR
    Tie-break: prefer threshold closer to 0.5 (stability), then smaller threshold.
    """
    y = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    ts = np.linspace(0.0, 1.0, int(n_grid))

    best = None
    for t in ts:
        pred = (p > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        youden = tpr - fpr

        score = acc if metric == "acc" else youden

        cand = {
            "threshold": float(t),
            "score": float(score),
            "metric": str(metric),
            "acc": float(acc),
            "se": float(tpr),
            "sp": float(tn / max(1, tn + fp)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

        if best is None:
            best = cand
        else:
            if cand["score"] > best["score"]:
                best = cand
            elif cand["score"] == best["score"]:
                # tie-break: threshold closer to 0.5 (avoid extreme thresholds)
                if abs(cand["threshold"] - 0.5) < abs(best["threshold"] - 0.5):
                    best = cand
                elif abs(cand["threshold"] - 0.5) == abs(best["threshold"] - 0.5):
                    if cand["threshold"] < best["threshold"]:
                        best = cand

    assert best is not None
    return best


def plot_roc(y01: np.ndarray, p: np.ndarray, out_png: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y01.astype(int), p.astype(np.float64))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_roc_with_maxacc(y01: np.ndarray, p: np.ndarray, out_png: Path, n_grid: int) -> dict[str, Any]:
    fpr, tpr, thr = roc_curve(y01.astype(int), p.astype(np.float64))
    best = find_maxacc_threshold(y01, p, n_grid=n_grid)

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


def outputs_exist(paths: Paths, seed: int) -> bool:
    m = paths.models_dir / f"gnb_seed{seed}.pkl"
    met = paths.out_dir / f"gnb_test_metrics_seed{seed}.json"
    return m.exists() and met.exists() and m.stat().st_size > 0 and met.stat().st_size > 0


def build_cfg(params: dict[str, Any]) -> RunConfig:
    return RunConfig(
        seed=int(params.get("seed", 0)),
        feature_set=("selected5" if str(params.get("feature_set", "all84")).lower() == "selected5"
                     else "single1" if str(params.get("feature_set", "all84")).lower() in {"single1", "single", "single_sqi"}
                     else "all84"),
        single_sqi=str(params.get("single_sqi", "fSQI")),

        threshold_fixed=float(params.get("threshold_fixed", 0.5)),
        maxacc_grid=int(params.get("maxacc_grid", 2001)),

        var_smoothing=float(params.get("var_smoothing", 1e-3)),
        threshold_metric=("youden" if str(params.get("threshold_metric", "acc")).lower() == "youden" else "acc"),
        use_calibration=bool(params.get("use_calibration", False)),
        calib_method=("isotonic" if str(params.get("calib_method", "sigmoid")).lower() == "isotonic" else "sigmoid"),

        verbose=bool(params.get("verbose", False)),
        force=bool(params.get("force", False)),
    )


def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    params (optional):
      - seed: int
      - verbose: bool
      - force: bool
      - features_parquet: str
      - split_csv: str
      - out_dir: str
      - feature_set: "all84" | "selected5"
      - threshold_fixed: float
      - maxacc_grid: int
    """
    cfg = build_cfg(params)
    _setup_logging(cfg.verbose)
    seed_all(cfg.seed)

    paths = prepare_paths(cfg, params)
    paths.mkdirs()

    if (not cfg.force) and outputs_exist(paths, cfg.seed):
        outs = [
            str(paths.models_dir / f"gnb_seed{cfg.seed}.pkl"),
            str(paths.out_dir / f"gnb_test_metrics_seed{cfg.seed}.json"),
        ]
        logger.info("gnb: outputs exist -> skip (force=True to rerun)")
        return {"step": "gnb", "skipped": True, "outputs": outs}
    if cfg.feature_set == "single1":
        logger.info("single1 sqi=%s (12 leads)", cfg.single_sqi)

    X, y01, split, feat_cols = _read_inputs(paths, cfg.feature_set, cfg.single_sqi)

    Xtr = X[split == "train"]; ytr = y01[split == "train"]
    Xva = X[split == "val"];   yva = y01[split == "val"]
    Xte = X[split == "test"];  yte = y01[split == "test"]

    logger.info("features: %s", paths.features_parquet)
    logger.info("split_csv: %s", paths.split_csv)
    logger.info("feature_set=%s | D=%d", cfg.feature_set, X.shape[1])
    logger.info("splits: train=%d val=%d test=%d",
                int((split == "train").sum()), int((split == "val").sum()), int((split == "test").sum()))

        # ---- fit on train only ----
    t0 = time.time()
    base = GaussianNB(var_smoothing=float(cfg.var_smoothing))
    base.fit(Xtr, ytr)
    train_time_s = float(time.time() - t0)

    # ---- optionally calibrate using val (recommended if probs are weird) ----
    if cfg.use_calibration:
        if FrozenEstimator is not None:
            # sklearn>=1.8: prefit mode replaced by FrozenEstimator
            model = CalibratedClassifierCV(
                estimator=FrozenEstimator(base),
                method=cfg.calib_method,
                cv=None,
            )
            model.fit(Xva, yva)  # calibration learns mapping on val
        else:
            # compatibility fallback for older sklearn
            model = CalibratedClassifierCV(base, method=cfg.calib_method, cv="prefit")
            model.fit(Xva, yva)  # calibration learns mapping on val
        model_tag = f"gnb_cal_{cfg.calib_method}"
    else:
        model = base
        model_tag = "gnb"

    # ---- select threshold on VAL ----
    p_val = model.predict_proba(Xva)[:, 1]
    thr_val_best = select_threshold_on_val(
        yva, p_val,
        metric=str(cfg.threshold_metric),
        n_grid=int(cfg.maxacc_grid),
    )
    thr_star = float(thr_val_best["threshold"])

    # ---- TEST once using val-selected threshold ----
    p_test = model.predict_proba(Xte)[:, 1]
    test_metrics_at_thrstar = compute_metrics(yte, p_test, threshold=thr_star)
    test_maxacc = find_maxacc_threshold(yte, p_test, n_grid=cfg.maxacc_grid)

    # optional: keep fixed-0.5 for reference only (not the main result)
    test_metrics_fixed = compute_metrics(yte, p_test, threshold=float(cfg.threshold_fixed))

    outs: list[str] = []

    out_model = paths.models_dir / f"gnb_seed{cfg.seed}.pkl"
    with open(out_model, "wb") as f:
        payload = {
            "seed": cfg.seed,
            "feature_set": cfg.feature_set,
            "feature_columns": feat_cols,
            "threshold_fixed": float(cfg.threshold_fixed),
            "sklearn_model": model,
        }
        pickle.dump(payload, f)
    outs.append(str(out_model))
    logger.info("[saved] %s", out_model)

    out_probs = paths.probs_dir / f"gnb_test_probs_seed{cfg.seed}.npz"
    np.savez_compressed(out_probs, y01_test=yte.astype(np.int32), p_test=p_test.astype(np.float64))
    outs.append(str(out_probs))
    logger.info("[saved] %s", out_probs)

    out_roc = paths.roc_dir / f"gnb_roc_seed{cfg.seed}.png"
    plot_roc(yte, p_test, out_roc, title="GNB ROC (test)")
    outs.append(str(out_roc))
    logger.info("[saved] %s", out_roc)

    out_roc_maxacc = paths.roc_dir / f"gnb_roc_maxacc_seed{cfg.seed}.png"
    _ = plot_roc_with_maxacc(yte, p_test, out_roc_maxacc, n_grid=cfg.maxacc_grid)
    outs.append(str(out_roc_maxacc))
    logger.info("[saved] %s", out_roc_maxacc)

    out_maxacc = paths.maxacc_dir / f"gnb_maxacc_seed{cfg.seed}.csv"
    pd.DataFrame([test_maxacc]).to_csv(out_maxacc, index=False)
    outs.append(str(out_maxacc))
    logger.info("[saved] %s", out_maxacc)

    out_metrics = paths.out_dir / f"gnb_test_metrics_seed{cfg.seed}.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
                json.dump(
            {
                "seed": cfg.seed,
                "feature_set": cfg.feature_set,
                "single_sqi": getattr(cfg, "single_sqi", None),
                "D": int(X.shape[1]),

                "gnb": {
                    "var_smoothing": float(cfg.var_smoothing),
                    "use_calibration": bool(cfg.use_calibration),
                    "calib_method": str(cfg.calib_method) if cfg.use_calibration else None,
                    "threshold_metric": str(cfg.threshold_metric),
                },

                "threshold_fixed": float(cfg.threshold_fixed),
                "train_time_s": train_time_s,

                "val_threshold_selected": thr_val_best,          # contains threshold + val acc/se/sp
                "test_metrics_at_val_threshold": test_metrics_at_thrstar,

                "test_metrics_fixed_ref": test_metrics_fixed,    # reference only
            },
            f,
            indent=2,
        )
    outs.append(str(out_metrics))
    logger.info("[saved] %s", out_metrics)

    logger.info("=== GNB FINAL TEST (val-selected threshold) ===")
    logger.info("thr*=%.4f | acc=%.4f se=%.4f sp=%.4f auc=%.4f",
                thr_star,
                test_metrics_at_thrstar["acc"], test_metrics_at_thrstar["se"],
                test_metrics_at_thrstar["sp"], test_metrics_at_thrstar["auc"])
    logger.info("REF (thr=%.3f): acc=%.4f se=%.4f sp=%.4f auc=%.4f",
                cfg.threshold_fixed,
                test_metrics_fixed["acc"], test_metrics_fixed["se"],
                test_metrics_fixed["sp"], test_metrics_fixed["auc"])
    
    logger.info("p_test stats: min=%.4g p1=%.4g p5=%.4g p50=%.4g p95=%.4g p99=%.4g max=%.4g mean=%.4g",
            float(np.min(p_test)),
            float(np.quantile(p_test, 0.01)),
            float(np.quantile(p_test, 0.05)),
            float(np.quantile(p_test, 0.50)),
            float(np.quantile(p_test, 0.95)),
            float(np.quantile(p_test, 0.99)),
            float(np.max(p_test)),
            float(np.mean(p_test)))
    logger.info("pred@thr=0.5: positive_rate=%.4f", float((p_test > 0.5).mean()))


    return {"step": "gnb", "skipped": False, "outputs": outs}


def main() -> None:
    params = {
        "verbose": False,
        "force": True,

        # choose one:
        # "feature_set": "all84",
        # "feature_set": "selected5",
        "feature_set": "single1",
        "single_sqi": "basSQI",   # try: "bSQI", "basSQI", "kSQI", "sSQI", "fSQI"

        "var_smoothing": 1e-4,
        "threshold_metric": "youden",
        "use_calibration": False,
        "calib_method": "sigmoid",

        "maxacc_grid": 2001,
    }
    run(params)


if __name__ == "__main__":
    main()
