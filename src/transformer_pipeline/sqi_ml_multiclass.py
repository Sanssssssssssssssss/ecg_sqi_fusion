from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.transformer_pipeline.sqi_ml_baseline import (
    build_qrs_cache,
    build_single_lead_features,
    export_single_lead_resampled_cache,
    normalize_single_lead_features,
)
from src.utils.paths import project_root

logger = logging.getLogger(__name__)

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
LABELS = [0, 1, 2]


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()
    seed = int(params.get("seed", 0))
    transformer_dir = _path(params.get("transformer_artifact_dir"), root / "outputs" / "transformer")
    out_dir = _path(params.get("out_dir"), root / "outputs" / "transformer_sqi_ml_three_class")
    paths = _paths(out_dir, seed)

    start = time.perf_counter()
    split = build_three_class_split(
        transformer_dir=transformer_dir,
        out_csv=paths["split"],
        seed=seed,
        force=force,
    )
    export_single_lead_resampled_cache(
        transformer_dir=transformer_dir,
        split=split,
        out_dir=paths["resampled"],
        force=force,
    )
    build_qrs_cache(split=split, resampled_dir=paths["resampled"], out_dir=paths["qrs"], force=force)
    features = build_single_lead_features(
        split=split,
        resampled_dir=paths["resampled"],
        qrs_dir=paths["qrs"],
        out_dir=paths["features"],
        force=force,
    )
    norm = normalize_single_lead_features(
        split_csv=paths["split"],
        in_parquet=paths["record7"],
        out_parquet=paths["record7_norm"],
        out_stats=paths["norm_stats"],
        force=force,
    )

    data = load_xy_multiclass(paths["record7_norm"], paths["split"])
    svm = train_svm_multiclass(data=data, out_dir=paths["svm"], seed=seed, force=force)
    mlp = train_mlp_multiclass(data=data, out_dir=paths["mlp"], seed=seed, force=force)

    summary = {
        "seed": seed,
        "artifact_dir": _rel(out_dir, root),
        "source_transformer_artifact_dir": _rel(transformer_dir, root),
        "task": "three_class_good_medium_bad",
        "duration_sec": float(time.perf_counter() - start),
        "label_map": CLASS_TO_INT,
        "data": {
            "split_shape": list(split.shape),
            "split_counts": _counts(split["split"]),
            "y_class_counts": _counts(split["y_class"]),
        },
        "features": {"record7": features["record"], "lead7": features["lead"], "record7_norm": norm},
        "metrics": {"svm_rbf": svm, "mlp": mlp},
    }
    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary_md"].write_text(render_markdown(summary), encoding="utf-8")
    print(render_console(summary))
    logger.info("wrote summary: %s", _rel(paths["summary_md"], root))
    return {
        "step": "transformer_sqi_ml_multiclass",
        "skipped": False,
        "outputs": [_rel(paths["summary_json"], root), _rel(paths["summary_md"], root)],
        "duration_sec": summary["duration_sec"],
    }


def build_three_class_split(*, transformer_dir: Path, out_csv: Path, seed: int, force: bool) -> pd.DataFrame:
    if out_csv.exists() and not force:
        logger.info("split exists -> skip: %s", out_csv)
        return pd.read_csv(out_csv)

    labels_path = transformer_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    labels = pd.read_csv(labels_path)
    required = {"idx", "seg_id", "ecg_id", "split", "y_class", "snr_db", "noise_kind"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"transformer labels missing columns: {sorted(missing)}")

    df = labels.copy().sort_values("idx").reset_index(drop=True)
    unknown = sorted(set(df["y_class"].astype(str)) - set(CLASS_TO_INT))
    if unknown:
        raise ValueError(f"unknown y_class values: {unknown}")
    df["y"] = df["y_class"].astype(str).map(CLASS_TO_INT).astype(int)
    df["record_id"] = [f"tx3_{int(i):05d}" for i in df["idx"]]
    df["seed"] = int(seed)
    df["source_idx"] = df["idx"].astype(int)

    out = df[
        [
            "record_id",
            "y",
            "split",
            "seed",
            "source_idx",
            "y_class",
            "snr_db",
            "noise_kind",
            "seg_id",
            "ecg_id",
        ]
    ].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info("three-class split: %s rows=%d", out_csv, len(out))
    logger.info("split counts: %s", out["split"].value_counts().sort_index().to_dict())
    logger.info("class counts: %s", out["y_class"].value_counts().sort_index().to_dict())
    return out


def load_xy_multiclass(features_parquet: Path, split_csv: Path) -> dict[str, Any]:
    feat = pd.read_parquet(features_parquet)
    split = pd.read_csv(split_csv)
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    df = feat.merge(split[["record_id", "split", "y_class"]], on="record_id", how="inner")
    drop_cols = {"record_id", "y", "split", "y_class"}
    feat_cols = sorted(c for c in df.columns if c not in drop_cols)
    x = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["y"].astype(int).to_numpy()
    split_arr = df["split"].astype(str).to_numpy()
    return {
        "feat_cols": feat_cols,
        "X_train": x[split_arr == "train"],
        "y_train": y[split_arr == "train"],
        "X_val": x[split_arr == "val"],
        "y_val": y[split_arr == "val"],
        "X_test": x[split_arr == "test"],
        "y_test": y[split_arr == "test"],
    }


def train_svm_multiclass(*, data: dict[str, Any], out_dir: Path, seed: int, force: bool) -> dict[str, Any]:
    metrics_path = out_dir / f"svm_rbf_three_class_metrics_seed{seed}.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    c_list = (2.0, 8.0, 32.0)
    gamma_list = (0.125, 0.5, 2.0)
    rows: list[dict[str, Any]] = []
    best_score: tuple[float, float, float, float] | None = None
    best_model: Pipeline | None = None
    t0 = time.perf_counter()

    for c in c_list:
        for gamma in gamma_list:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            C=float(c),
                            gamma=float(gamma),
                            class_weight=None,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            model.fit(data["X_train"], data["y_train"])
            pred_val = model.predict(data["X_val"])
            val_acc = float(accuracy_score(data["y_val"], pred_val))
            val_macro_f1 = float(f1_score(data["y_val"], pred_val, labels=LABELS, average="macro"))
            rows.append({"C": float(c), "gamma": float(gamma), "val_acc": val_acc, "val_macro_f1": val_macro_f1})
            score = (val_acc, val_macro_f1, -float(c), -float(gamma))
            if best_score is None or score > best_score:
                best_score = score
                best_model = model
            logger.info("svm C=%.4g gamma=%.4g val_acc=%.4f macro_f1=%.4f", c, gamma, val_acc, val_macro_f1)

    assert best_model is not None
    model = best_model
    pred_train = model.predict(data["X_train"])
    pred_val = model.predict(data["X_val"])
    pred_test = model.predict(data["X_test"])
    result = {
        "model": "RBF SVM",
        "feature_set": "all7",
        "n_features": int(data["X_train"].shape[1]),
        "grid": rows,
        "best_params": dict(model.named_steps["svc"].get_params(deep=False)),
        "train_time_s": float(time.perf_counter() - t0),
        "train": multiclass_metrics(data["y_train"], pred_train),
        "val": multiclass_metrics(data["y_val"], pred_val),
        "test": multiclass_metrics(data["y_test"], pred_test),
    }
    result["best_params"] = {
        "C": float(result["best_params"]["C"]),
        "gamma": float(result["best_params"]["gamma"]),
    }
    joblib.dump({"model": model, "feature_columns": data["feat_cols"], "metrics": result}, out_dir / f"svm_rbf_three_class_seed{seed}.joblib")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("svm three-class test_acc=%.4f", result["test"]["acc"])
    return result


def train_mlp_multiclass(*, data: dict[str, Any], out_dir: Path, seed: int, force: bool) -> dict[str, Any]:
    metrics_path = out_dir / f"mlp_three_class_metrics_seed{seed}.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_sizes = (4, 8, 12, 16, 24, 32)
    rows: list[dict[str, Any]] = []
    best_score: tuple[float, float, int] | None = None
    best_model: Pipeline | None = None
    t0 = time.perf_counter()

    for h in hidden_sizes:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(int(h),),
                        activation="logistic",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        batch_size=256,
                        max_iter=600,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(data["X_train"], data["y_train"])
        pred_val = model.predict(data["X_val"])
        val_acc = float(accuracy_score(data["y_val"], pred_val))
        val_macro_f1 = float(f1_score(data["y_val"], pred_val, labels=LABELS, average="macro"))
        n_iter = int(model.named_steps["mlp"].n_iter_)
        rows.append({"hidden": int(h), "val_acc": val_acc, "val_macro_f1": val_macro_f1, "n_iter": n_iter})
        score = (val_acc, val_macro_f1, -int(h))
        if best_score is None or score > best_score:
            best_score = score
            best_model = model
        logger.info("mlp hidden=%d val_acc=%.4f macro_f1=%.4f n_iter=%d", h, val_acc, val_macro_f1, n_iter)

    assert best_model is not None
    model = best_model
    pred_train = model.predict(data["X_train"])
    pred_val = model.predict(data["X_val"])
    pred_test = model.predict(data["X_test"])
    result = {
        "model": "sklearn MLPClassifier",
        "feature_set": "all7",
        "n_features": int(data["X_train"].shape[1]),
        "search": rows,
        "best_hidden": int(model.named_steps["mlp"].hidden_layer_sizes[0]),
        "train_time_s": float(time.perf_counter() - t0),
        "train": multiclass_metrics(data["y_train"], pred_train),
        "val": multiclass_metrics(data["y_val"], pred_val),
        "test": multiclass_metrics(data["y_test"], pred_test),
    }
    joblib.dump({"model": model, "feature_columns": data["feat_cols"], "metrics": result}, out_dir / f"mlp_three_class_seed{seed}.joblib")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("mlp three-class test_acc=%.4f", result["test"]["acc"])
    return result


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=LABELS)
    recalls: dict[str, float] = {}
    for i, label in enumerate(LABELS):
        denom = max(1, int(cm[i].sum()))
        recalls[INT_TO_CLASS[label]] = float(cm[i, i] / denom)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro")),
        "confusion_matrix_3x3": cm.astype(int).tolist(),
        "per_class_recall": recalls,
    }


def render_console(summary: dict[str, Any]) -> str:
    svm = summary["metrics"]["svm_rbf"]["test"]
    mlp = summary["metrics"]["mlp"]["test"]
    return "\n".join(
        [
            "Transformer SQI-ML three-class result",
            f"SVM-RBF test acc={svm['acc']:.4f}, macro_f1={svm['macro_f1']:.4f}, medium_recall={svm['per_class_recall']['medium']:.4f}",
            f"MLP test acc={mlp['acc']:.4f}, macro_f1={mlp['macro_f1']:.4f}, medium_recall={mlp['per_class_recall']['medium']:.4f}",
        ]
    )


def render_markdown(summary: dict[str, Any]) -> str:
    svm = summary["metrics"]["svm_rbf"]
    mlp = summary["metrics"]["mlp"]
    return "\n".join(
        [
            "# Transformer Dataset SQI-ML Three-Class Baseline",
            "",
            f"Artifact dir: `{summary['artifact_dir']}`",
            f"Source transformer artifact: `{summary['source_transformer_artifact_dir']}`",
            "",
            "Task: `good=0`, `medium=1`, `bad=2` using the same single-lead 7 SQI feature adapter.",
            "",
            "## Data",
            "",
            f"- split shape: `{summary['data']['split_shape']}`",
            f"- split counts: `{summary['data']['split_counts']}`",
            f"- y_class counts: `{summary['data']['y_class_counts']}`",
            "",
            "## Results",
            "",
            "| Model | Features | Test Acc | Balanced Acc | Macro F1 | Medium Recall |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            f"| SVM-RBF | 7 SQI | {svm['test']['acc']:.4f} | {svm['test']['balanced_acc']:.4f} | {svm['test']['macro_f1']:.4f} | {svm['test']['per_class_recall']['medium']:.4f} |",
            f"| MLP | 7 SQI | {mlp['test']['acc']:.4f} | {mlp['test']['balanced_acc']:.4f} | {mlp['test']['macro_f1']:.4f} | {mlp['test']['per_class_recall']['medium']:.4f} |",
            "",
            "SVM confusion matrix:",
            "",
            "```text",
            str(svm["test"]["confusion_matrix_3x3"]),
            "```",
            "",
            "MLP confusion matrix:",
            "",
            "```text",
            str(mlp["test"]["confusion_matrix_3x3"]),
            "```",
        ]
    )


def _paths(out_dir: Path, seed: int) -> dict[str, Path]:
    return {
        "split": out_dir / "splits" / f"transformer_three_class_seed{seed}.csv",
        "resampled": out_dir / "resampled_125",
        "qrs": out_dir / "qrs",
        "features": out_dir / "features",
        "record7": out_dir / "features" / "record7.parquet",
        "lead7": out_dir / "features" / "lead7.parquet",
        "record7_norm": out_dir / "features" / "record7_norm.parquet",
        "norm_stats": out_dir / "features" / f"norm_stats_seed{seed}.json",
        "svm": out_dir / "models" / "svm",
        "mlp": out_dir / "models" / "mlp",
        "summary_json": out_dir / "three_class_summary.json",
        "summary_md": out_dir / "three_class_summary.md",
    }


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().to_dict().items()}


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three-class SQI+ML baselines on transformer Lead I data.")
    parser.add_argument("--transformer_artifact_dir", default="outputs/transformer")
    parser.add_argument("--out_dir", default="outputs/transformer_sqi_ml_three_class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(vars(args))


if __name__ == "__main__":
    main()
