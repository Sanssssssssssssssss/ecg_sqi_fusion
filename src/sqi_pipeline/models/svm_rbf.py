from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass(frozen=True)
class SVMConfig:
    seed: int = 0
    cv_folds: int = 5

    # scaling: even though many SQIs in [0,1], kSQI/sSQI are z-scored -> scale is helpful for RBF
    use_standard_scaler: bool = True

    # grid search ranges (exponentially growing sequences)
    # defaults are reasonable and match common libSVM guidance style
    C_list: tuple[float, ...] = (2.0 ** 1, 2.0 ** 3)
    gamma_list: tuple[float, ...] = (2.0 ** -1, 2.0 ** 1, 2.0 ** 3)

    # scoring metric
    scoring: str = "accuracy"

    # SVC config (rbf)
    probability: bool = True  # to allow ROC/AUC and probabilities
    class_weight: str | None = None  # you already balance train/test; keep None for faithful baseline


class SVMRBF:
    """
    RBF SVM with grid-search over C and gamma (libSVM-style).
    """

    def __init__(self, cfg: SVMConfig):
        self.cfg = cfg
        self.best_estimator_: Pipeline | None = None
        self.best_params_: dict[str, Any] | None = None
        self.best_cv_score_: float | None = None

    def _make_pipeline(self) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if self.cfg.use_standard_scaler:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        steps.append(
            ("svc", SVC(
                kernel="rbf",
                probability=self.cfg.probability,
                class_weight=self.cfg.class_weight,
                random_state=self.cfg.seed,
            ))
        )
        return Pipeline(steps)

    def fit_gridsearch(
            self,
            Xtr: np.ndarray,
            ytr: np.ndarray,
            Xva: np.ndarray,
            yva: np.ndarray,
            *,
            C_list: tuple[float, ...] | None = None,
            gamma_list: tuple[float, ...] | None = None,
            threshold: float = 0.5,
            select_metric: str = "val_acc",   # "val_acc" or "val_auc"
            log_each: bool = True,
        ) -> dict[str, Any]:
            """
            Grid search over (C, gamma) using PROVIDED validation set (Xva, yva).
            No random split; respects split_csv's 'val'.

            Returns dict compatible with your existing callers.
            """
            Xtr = np.asarray(Xtr, dtype=np.float64)
            ytr = np.asarray(ytr, dtype=int).ravel()
            Xva = np.asarray(Xva, dtype=np.float64)
            yva = np.asarray(yva, dtype=int).ravel()

            C_list = tuple(C_list) if C_list is not None else self.cfg.C_list
            gamma_list = tuple(gamma_list) if gamma_list is not None else self.cfg.gamma_list

            if len(C_list) == 0 or len(gamma_list) == 0:
                raise ValueError("C_list and gamma_list must be non-empty")

            best_metric = -1.0
            best_params: dict[str, float] = {"svc__C": float(C_list[0]), "svc__gamma": float(gamma_list[0])}
            best_val_acc = float("nan")
            best_val_auc = float("nan")
            n_grid = 0

            t0 = time.time()

            # local helper: metrics on provided val
            def _val_metrics(p: np.ndarray) -> tuple[float, float]:
                # acc at fixed threshold
                pred = (p > float(threshold)).astype(int)
                acc = float((pred == yva).mean())

                # auc
                try:
                    auc = float(roc_auc_score(yva, p))
                except Exception:
                    auc = float("nan")
                return acc, auc

            for C in C_list:
                for g in gamma_list:
                    n_grid += 1

                    pipe = self._make_pipeline()
                    pipe.set_params(svc__C=float(C), svc__gamma=float(g))
                    pipe.fit(Xtr, ytr)

                    # compute val probs
                    p_val = pipe.predict_proba(Xva)[:, 1].astype(np.float64)

                    val_acc, val_auc = _val_metrics(p_val)

                    metric = val_auc if select_metric == "val_auc" else val_acc

                    if log_each:
                        logger.info("C=%.6g gamma=%.6g | val_acc=%.4f val_auc=%.4f", C, g, val_acc, val_auc)

                    # tie-break: metric desc, then val_auc desc, then smaller C, then smaller gamma
                    if (
                        (metric > best_metric)
                        or (metric == best_metric and val_auc > best_val_auc)
                        or (metric == best_metric and val_auc == best_val_auc and float(C) < float(best_params["svc__C"]))
                        or (
                            metric == best_metric
                            and val_auc == best_val_auc
                            and float(C) == float(best_params["svc__C"])
                            and float(g) < float(best_params["svc__gamma"])
                        )
                    ):
                        best_metric = float(metric)
                        best_val_acc = float(val_acc)
                        best_val_auc = float(val_auc)
                        best_params = {"svc__C": float(C), "svc__gamma": float(g)}
                        self.best_estimator_ = pipe
                        self.best_params_ = dict(best_params)

            # refit best on ALL train (clean)
            best_pipe = self._make_pipeline()
            best_pipe.set_params(**best_params)
            best_pipe.fit(Xtr, ytr)

            self.best_estimator_ = best_pipe
            self.best_params_ = dict(best_params)
            self.best_cv_score_ = float("nan")

            return {
                "train_time_s": float(time.time() - t0),
                "best_score": float(best_metric),
                "best_val_acc": float(best_val_acc),
                "best_val_auc": float(best_val_auc),
                "best_params": dict(best_params),
                "n_grid": int(n_grid),
            }
    
    def fit_fixed(self, X: np.ndarray, y01: np.ndarray, *, C: float, gamma: float) -> dict[str, Any]:
        X = np.asarray(X, dtype=np.float64)
        y01 = np.asarray(y01, dtype=int).ravel()

        # 复用你 fit_gridsearch 里的 make_pipe
        steps: list[tuple[str, Any]] = []
        if self.cfg.use_standard_scaler:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        steps.append(("svc", SVC(
            kernel="rbf",
            C=float(C),
            gamma=float(gamma),
            probability=self.cfg.probability,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.seed,
        )))
        pipe = Pipeline(steps)

        t0 = time.time()
        pipe.fit(X, y01)
        train_s = float(time.time() - t0)

        self.best_estimator_ = pipe
        self.best_params_ = {"svc__C": float(C), "svc__gamma": float(gamma)}
        self.best_cv_score_ = float("nan")

        return {"train_time_s": train_s, "params": dict(self.best_params_)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)
        # proba for class 1
        p = self.best_estimator_.predict_proba(X)[:, 1]
        return p.astype(np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p > float(threshold)).astype(int)

    def save(self, out_pkl: Path, meta: dict[str, Any] | None = None) -> None:
        """
        Save best_estimator_ using joblib (sklearn standard).
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Model not fitted")
        import joblib

        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": asdict(self.cfg),
            "best_params": self.best_params_,
            "best_cv_score": self.best_cv_score_,
            "meta": meta or {},
            "estimator": self.best_estimator_,
        }
        joblib.dump(payload, out_pkl)

    @staticmethod
    def load(pkl_path: Path) -> dict[str, Any]:
        import joblib
        return joblib.load(pkl_path)

    def write_config_json(self, out_json: Path, extra: dict[str, Any] | None = None) -> None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        d = {"svm_config": asdict(self.cfg)}
        if extra:
            d.update(extra)
        out_json.write_text(json.dumps(d, indent=2), encoding="utf-8")
