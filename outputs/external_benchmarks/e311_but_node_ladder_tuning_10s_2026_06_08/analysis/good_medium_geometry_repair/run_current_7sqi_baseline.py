"""Current 7-SQI baseline for the N17043 boundary-block experiment.

This is a fresh experiment-only baseline. It does not reuse old runner
artifacts. It trains 7-SQI-only classifiers on the current synthetic train
split, selects on the current synthetic validation split, and evaluates the
same held-out original BUT split used by the UFormer+geometry report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_OUT = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
ANALYSIS_REPORT = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
VARIANT_ID = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
VARIANT_DIR = OUT_ROOT / "synthetic_variants" / VARIANT_ID
DATA_DIR = VARIANT_DIR / "datasets"
ATLAS_PATH = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
LABELS = [0, 1, 2]

SQI_COLUMNS = [
    "sqi_bSQI",
    "sqi_iSQI",
    "sqi_kSQI",
    "sqi_sSQI",
    "sqi_pSQI",
    "sqi_fSQI",
    "sqi_basSQI",
]

FINAL_UFORMER_REFERENCE = {
    "model": "UFormer+47 SQI/geometry branch, threshold p_bad>=0.13",
    "original_test_all_10s+": {
        "n": 8477,
        "acc": 0.963548425150407,
        "macro_f1": 0.9306830954417679,
        "good_recall": 0.9563186813186814,
        "medium_recall": 0.9728874830546769,
        "bad_recall": 0.927007299270073,
    },
    "original_all_10s+": {
        "n": 32956,
        "acc": 0.9853744386454667,
        "macro_f1": 0.985233025715266,
        "good_recall": 0.9888517279821628,
        "medium_recall": 0.9753481369966127,
        "bad_recall": 0.9943235572374646,
    },
}


def main() -> None:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORT.mkdir(parents=True, exist_ok=True)

    synthetic = load_synthetic_7sqi()
    original = load_original_7sqi()

    candidates = train_candidates(synthetic)
    best = select_best(candidates)
    rows = evaluate_all(best, synthetic, original)

    metrics_csv = ANALYSIS_OUT / "current_7sqi_baseline_metrics.csv"
    predictions_original_csv = ANALYSIS_OUT / "current_7sqi_baseline_original_predictions.csv"
    predictions_synthetic_csv = ANALYSIS_OUT / "current_7sqi_baseline_synthetic_predictions.csv"
    summary_json = ANALYSIS_OUT / "current_7sqi_baseline_summary.json"
    report_out = ANALYSIS_OUT / "current_7sqi_baseline_report.md"
    report_copy = ANALYSIS_REPORT / "current_7sqi_baseline_report.md"
    model_path = ANALYSIS_OUT / "current_7sqi_baseline_best_model.joblib"

    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    export_predictions(best["model"], synthetic, predictions_synthetic_csv, is_original=False)
    export_predictions(best["model"], original, predictions_original_csv, is_original=True)
    joblib.dump(
        {
            "model": best["model"],
            "model_name": best["name"],
            "params": best["params"],
            "feature_columns": SQI_COLUMNS,
            "train_source": str(DATA_DIR),
            "selection": "synthetic validation split only",
            "original_used_for_selection": False,
        },
        model_path,
    )

    summary = {
        "variant_id": VARIANT_ID,
        "feature_columns": SQI_COLUMNS,
        "selection": "synthetic validation split only",
        "original_used_for_selection": False,
        "synthetic_counts": split_counts(synthetic["meta"]),
        "original_counts": original_counts(original["meta"]),
        "candidate_grid": [
            {
                "name": c["name"],
                "params": c["params"],
                "val_acc": c["val"]["acc"],
                "val_macro_f1": c["val"]["macro_f1"],
                "val_good_recall": c["val"]["good_recall"],
                "val_medium_recall": c["val"]["medium_recall"],
                "val_bad_recall": c["val"]["bad_recall"],
            }
            for c in candidates
        ],
        "best": {
            "name": best["name"],
            "params": best["params"],
            "metrics": rows,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "predictions_original_csv": str(predictions_original_csv),
            "predictions_synthetic_csv": str(predictions_synthetic_csv),
            "model_path": str(model_path),
            "report": str(report_copy),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(summary)
    report_out.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")

    print(report)


def load_synthetic_7sqi() -> dict[str, Any]:
    labels_path = DATA_DIR / "synth_10s_125hz_labels_with_level.csv"
    features_path = DATA_DIR / "tabular_features.npz"
    schema_path = DATA_DIR / "tabular_feature_schema.json"
    labels = pd.read_csv(labels_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_columns = list(schema["feature_columns"])
    missing = sorted(set(SQI_COLUMNS) - set(feature_columns))
    if missing:
        raise ValueError(f"synthetic sidecar missing SQI columns: {missing}")
    feature_index = [feature_columns.index(c) for c in SQI_COLUMNS]
    features_npz = np.load(features_path)
    features = features_npz["features"][:, feature_index].astype(np.float64)
    if len(labels) != features.shape[0]:
        raise ValueError(f"labels/features row mismatch: labels={len(labels)} features={features.shape[0]}")
    y = labels["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    if np.isnan(y.astype(float)).any():
        raise ValueError("synthetic labels contain unknown classes")
    meta = labels[["idx", "split", "y_class", "label_subtype", "sample_source"]].copy()
    meta["y"] = y
    return {"X": features, "y": y, "meta": meta}


def load_original_7sqi() -> dict[str, Any]:
    atlas = pd.read_csv(ATLAS_PATH)
    missing = sorted(set(SQI_COLUMNS) - set(atlas.columns))
    if missing:
        raise ValueError(f"original atlas missing SQI columns: {missing}")
    y = atlas["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    if np.isnan(y.astype(float)).any():
        raise ValueError("original atlas contains unknown classes")
    features = atlas[SQI_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    keep_cols = ["idx", "split", "class_name", "record_id", "subject_id", "original_region", "ambiguous_type"]
    meta = atlas[[c for c in keep_cols if c in atlas.columns]].copy()
    meta["y"] = y
    meta["y_class"] = meta["class_name"].astype(str)
    return {"X": features, "y": y, "meta": meta}


def train_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    split = data["meta"]["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    X_train = data["X"][train_mask]
    y_train = data["y"][train_mask]
    X_val = data["X"][val_mask]
    y_val = data["y"][val_mask]

    candidates: list[dict[str, Any]] = []

    for class_weight in [None, "balanced"]:
        for c in [1.0, 4.0, 16.0]:
            for gamma in ["scale", 0.25, 1.0]:
                name = "7SQI_RBF_SVM_balanced" if class_weight == "balanced" else "7SQI_RBF_SVM"
                params = {"C": c, "gamma": gamma, "class_weight": class_weight}
                model = Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "svc",
                            SVC(
                                kernel="rbf",
                                C=float(c),
                                gamma=gamma,
                                class_weight=class_weight,
                                random_state=20260614,
                            ),
                        ),
                    ]
                )
                model.fit(X_train, y_train)
                val = metric_dict(y_val, model.predict(X_val))
                candidates.append({"name": name, "params": params, "model": model, "val": val})

    for hidden in [(32,), (64,), (64, 32)]:
        for alpha in [1e-4, 1e-3]:
            params = {"hidden_layer_sizes": hidden, "alpha": alpha}
            model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=hidden,
                            alpha=float(alpha),
                            activation="relu",
                            solver="adam",
                            learning_rate_init=1e-3,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=25,
                            max_iter=600,
                            random_state=20260614,
                        ),
                    ),
                ]
            )
            model.fit(X_train, y_train)
            val = metric_dict(y_val, model.predict(X_val))
            candidates.append({"name": "7SQI_MLP", "params": params, "model": model, "val": val})

    return candidates


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def score(c: dict[str, Any]) -> tuple[float, float, float, float]:
        m = c["val"]
        min_recall = min(m["good_recall"], m["medium_recall"], m["bad_recall"])
        return (m["acc"], m["macro_f1"], min_recall, -model_complexity(c))

    return max(candidates, key=score)


def model_complexity(candidate: dict[str, Any]) -> float:
    params = candidate["params"]
    if candidate["name"].startswith("7SQI_RBF"):
        gamma = params.get("gamma")
        gamma_value = 0.0 if gamma == "scale" else float(gamma)
        return float(params.get("C", 0.0)) + gamma_value
    hidden = params.get("hidden_layer_sizes", ())
    return float(sum(hidden)) + float(params.get("alpha", 0.0))


def evaluate_all(best: dict[str, Any], synthetic: dict[str, Any], original: dict[str, Any]) -> list[dict[str, Any]]:
    model = best["model"]
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        mask = synthetic["meta"]["split"].astype(str).to_numpy() == split_name
        rows.append(metric_row(best, f"synthetic_{split_name}", synthetic["y"][mask], model.predict(synthetic["X"][mask])))

    original_meta = original["meta"]
    test_mask = original_meta["split"].astype(str).to_numpy() == "test"
    all_mask = np.ones(len(original_meta), dtype=bool)
    bad_mask = original["y"] == CLASS_TO_INT["bad"]
    outlier_mask = original_meta.get("original_region", pd.Series([""] * len(original_meta))).astype(str).to_numpy() == "outlier_low_confidence"
    rows.append(metric_row(best, "original_test_all_10s+", original["y"][test_mask], model.predict(original["X"][test_mask])))
    rows.append(metric_row(best, "original_all_10s+", original["y"][all_mask], model.predict(original["X"][all_mask])))
    main_mask = test_mask & ~(bad_mask & outlier_mask)
    rows.append(metric_row(best, "original_test_without_bad_outlier_stress", original["y"][main_mask], model.predict(original["X"][main_mask])))
    core_mask = test_mask & bad_mask & ~outlier_mask
    stress_mask = test_mask & bad_mask & outlier_mask
    rows.append(metric_row(best, "bad_core_nearboundary", original["y"][core_mask], model.predict(original["X"][core_mask])))
    rows.append(metric_row(best, "bad_outlier_stress", original["y"][stress_mask], model.predict(original["X"][stress_mask])))
    return rows


def metric_row(best: dict[str, Any], bucket: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    row = {
        "model_name": best["name"],
        "params_json": json.dumps(best["params"], sort_keys=True),
        "bucket": bucket,
    }
    row.update(metric_dict(y_true, y_pred))
    row["confusion_3x3"] = json.dumps(confusion_matrix(y_true, y_pred, labels=LABELS).tolist())
    return row


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "n": 0,
            "acc": 0.0,
            "macro_f1": 0.0,
            "good_recall": 0.0,
            "medium_recall": 0.0,
            "bad_recall": 0.0,
            "good_precision": 0.0,
            "medium_precision": 0.0,
            "bad_precision": 0.0,
        }
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "good_recall": float(recalls[0]),
        "medium_recall": float(recalls[1]),
        "bad_recall": float(recalls[2]),
        "good_precision": float(precisions[0]),
        "medium_precision": float(precisions[1]),
        "bad_precision": float(precisions[2]),
    }


def export_predictions(model: Pipeline, data: dict[str, Any], out_csv: Path, *, is_original: bool) -> None:
    pred = model.predict(data["X"])
    df = data["meta"].copy()
    true_col = "class_name" if is_original and "class_name" in df.columns else "y_class"
    df["y_true"] = data["y"]
    df["y_pred"] = pred
    df["pred_class"] = [INT_TO_CLASS[int(v)] for v in pred]
    df["correct"] = df["y_true"].to_numpy() == df["y_pred"].to_numpy()
    if true_col in df.columns:
        df["true_class"] = df[true_col].astype(str)
    df.to_csv(out_csv, index=False)


def split_counts(meta: pd.DataFrame) -> dict[str, Any]:
    return {
        "split": meta["split"].astype(str).value_counts().sort_index().to_dict(),
        "class": meta["y_class"].astype(str).value_counts().sort_index().to_dict(),
        "split_class": (
            meta.groupby(["split", "y_class"], dropna=False).size().reset_index(name="n").to_dict(orient="records")
        ),
    }


def original_counts(meta: pd.DataFrame) -> dict[str, Any]:
    out = {
        "split": meta["split"].astype(str).value_counts().sort_index().to_dict(),
        "class": meta["y_class"].astype(str).value_counts().sort_index().to_dict(),
        "test_class": meta.loc[meta["split"].astype(str) == "test", "y_class"].astype(str).value_counts().sort_index().to_dict(),
    }
    if "original_region" in meta.columns:
        test_bad = meta[(meta["split"].astype(str) == "test") & (meta["y_class"].astype(str) == "bad")]
        out["test_bad_region"] = test_bad["original_region"].astype(str).value_counts().sort_index().to_dict()
    return out


def render_report(summary: dict[str, Any]) -> str:
    metrics = pd.DataFrame(summary["best"]["metrics"])
    final_test = FINAL_UFORMER_REFERENCE["original_test_all_10s+"]
    final_all = FINAL_UFORMER_REFERENCE["original_all_10s+"]
    original_test = metrics[metrics["bucket"] == "original_test_all_10s+"].iloc[0].to_dict()
    original_all = metrics[metrics["bucket"] == "original_all_10s+"].iloc[0].to_dict()
    syn_test = metrics[metrics["bucket"] == "synthetic_test"].iloc[0].to_dict()
    best_name = summary["best"]["name"]
    best_params = json.dumps(summary["best"]["params"], sort_keys=True)

    lines = [
        "# Current 7SQI Baseline",
        "",
        "Fresh current-experiment baseline: train on the N17043 synthetic train split, select on synthetic val only, evaluate the same original BUT held-out split. Original BUT is not used for training or selection.",
        "",
        "## Setup",
        "",
        f"- Synthetic variant: `{summary['variant_id']}`",
        f"- Features: `{', '.join(SQI_COLUMNS)}`",
        "- Classifier candidates: RBF SVM, balanced RBF SVM, MLP",
        f"- Selected model: `{best_name}`",
        f"- Selected params: `{best_params}`",
        "",
        "## Main Results",
        "",
        "| Model | Bucket | n | Acc | Macro-F1 | Good R | Medium R | Bad R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        fmt_table_row("7SQI baseline", syn_test),
        fmt_table_row("7SQI baseline", original_test),
        fmt_table_row("7SQI baseline", original_all),
        fmt_ref_row(FINAL_UFORMER_REFERENCE["model"], "original_test_all_10s+", final_test),
        fmt_ref_row(FINAL_UFORMER_REFERENCE["model"], "original_all_10s+", final_all),
        "",
        "## Original Buckets",
        "",
        "| Bucket | n | Acc | Macro-F1 | Good R | Medium R | Bad R | Confusion [g,m,b]x[g,m,b] |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in metrics[metrics["bucket"].astype(str).str.startswith(("original", "bad_"))].iterrows():
        lines.append(
            f"| {row['bucket']} | {int(row['n'])} | {row['acc']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['good_recall']:.6f} | {row['medium_recall']:.6f} | {row['bad_recall']:.6f} | `{row['confusion_3x3']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The 7SQI-only baseline is a deliberately narrow feature baseline: it sees SQI summary statistics only, no waveform tokens and no 47-feature geometry branch.",
            "- Selection is clean: the original BUT rows are held out and report-only.",
            "- The gap versus the UFormer+geometry branch quantifies how much the current full model gains from waveform representation plus target-aware geometry/SQI fusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt_table_row(model_name: str, row: dict[str, Any]) -> str:
    return (
        f"| {model_name} | {row['bucket']} | {int(row['n'])} | {row['acc']:.6f} | {row['macro_f1']:.6f} | "
        f"{row['good_recall']:.6f} | {row['medium_recall']:.6f} | {row['bad_recall']:.6f} |"
    )


def fmt_ref_row(model_name: str, bucket: str, row: dict[str, Any]) -> str:
    return (
        f"| {model_name} | {bucket} | {int(row['n'])} | {row['acc']:.6f} | {row['macro_f1']:.6f} | "
        f"{row['good_recall']:.6f} | {row['medium_recall']:.6f} | {row['bad_recall']:.6f} |"
    )


if __name__ == "__main__":
    main()
