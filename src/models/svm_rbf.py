from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
    C_list: tuple[float, ...] = (2.0 ** -5, 2.0 ** -3, 2.0 ** -1, 2.0 ** 1, 2.0 ** 3, 2.0 ** 5, 2.0 ** 7, 2.0 ** 9, 2.0 ** 11, 2.0 ** 13, 2.0 ** 15)
    gamma_list: tuple[float, ...] = (2.0 ** -15, 2.0 ** -13, 2.0 ** -11, 2.0 ** -9, 2.0 ** -7, 2.0 ** -5, 2.0 ** -3, 2.0 ** -1, 2.0 ** 1, 2.0 ** 3)

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

    def fit_gridsearch(self, X: np.ndarray, y01: np.ndarray) -> dict[str, Any]:
        X = np.asarray(X, dtype=np.float64)
        y01 = np.asarray(y01, dtype=int).ravel()

        pipe = self._make_pipeline()

        param_grid = {
            "svc__C": list(self.cfg.C_list),
            "svc__gamma": list(self.cfg.gamma_list),
        }

        cv = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.seed)
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=self.cfg.scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X, y01)

        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = dict(gs.best_params_)
        self.best_cv_score_ = float(gs.best_score_)

        return {
            "best_params": self.best_params_,
            "best_cv_score": self.best_cv_score_,
        }

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
