"""Aux-prediction upper-bound audit for waveform geometry students.

This external-only diagnostic answers one narrow question:

    Does the Transformer-predicted SQI/geometry representation contain enough
    information to classify the current synthetic/node task and transfer to the
    report-only BUT buckets?

The script never trains on BUT rows.  It loads saved waveform-only checkpoints,
collects their aux_pred outputs on synthetic train/val/test and original BUT,
fits small classifiers on synthetic-train aux_pred only, then reports synthetic
and BUT bucket metrics.  It also fits the same classifiers on true synthetic
aux values as an upper reference for the feature space itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path.cwd()
ANALYSIS_DIR = Path(__file__).parent
RUNNER_PATH = ANALYSIS_DIR / "run_waveform_geometry_student.py"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
NODE_ID = "N17043_gm_probe"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "aux_pred_upper_bound"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class SplitPayload:
    name: str
    y: np.ndarray
    probs: np.ndarray
    pred: np.ndarray
    aux_pred: np.ndarray
    aux_true: np.ndarray
    split: np.ndarray | None = None
    region: np.ndarray | None = None


def candidate_dir(candidate: str, run_label: str) -> Path:
    return OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / run_label / candidate


def feature_sets(runner: Any) -> dict[str, list[str]]:
    columns = list(runner.FEATURE_COLUMNS)
    sets: dict[str, list[str]] = {
        "top20_interpretable": [c for c in runner.TOP20_INTERPRETABLE_FEATURE_COLUMNS if c in columns],
        "top23_hard": [c for c in runner.TOP23_HARD_FEATURE_COLUMNS if c in columns],
        "p20_failure_axis": [c for c in runner.P20_FAILURE_AXIS_FEATURE_COLUMNS if c in columns],
        "full_aux": columns,
        "hard_missing_axes": [
            c
            for c in [
                "qrs_visibility",
                "detector_agreement",
                "baseline_step",
                "sqi_basSQI",
                "flatline_ratio",
                "pc2",
                "pc3",
                "boundary_confidence",
                "knn_label_purity",
                "pca_margin",
                "non_qrs_diff_p95",
            ]
            if c in columns
        ],
    }
    return {name: cols for name, cols in sets.items() if cols}


def indices_for(runner: Any, cols: list[str]) -> list[int]:
    return [list(runner.FEATURE_COLUMNS).index(c) for c in cols]


def make_classifier(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=31,
            l2_regularization=0.025,
            random_state=int(seed),
        )
    if kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=220,
            max_features=0.65,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        )
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", multi_class="auto", random_state=int(seed)),
        )
    raise ValueError(f"unknown classifier kind: {kind}")


def clean_matrix(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_model_payloads(runner: Any, candidate: str, run_label: str, batch_size: int) -> dict[str, SplitPayload]:
    run_dir = candidate_dir(candidate, run_label)
    ckpt_path = run_dir / "ckpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["candidate_config"]

    channels = str(cfg["channels"])
    base_train = runner.ARCH.SyntheticWaveDataset("train", channels)
    norm = runner.ARCH.FeatureNorm(mean=base_train.mean, std=base_train.std)
    datasets = {
        "synthetic_train": runner.ARCH.SyntheticWaveDataset("train", channels),
        "synthetic_val": runner.ARCH.SyntheticWaveDataset("val", channels),
        "synthetic_test": runner.ARCH.SyntheticWaveDataset("test", channels),
        "original_all": runner.ARCH.OriginalWaveDataset(norm, channels),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = runner.GeometryStudent(cfg, int(datasets["synthetic_train"].x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    out: dict[str, SplitPayload] = {}
    for split_name, ds in datasets.items():
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        payload = runner.eval_loader(model, loader, device)
        out[split_name] = SplitPayload(
            name=split_name,
            y=payload.y,
            probs=payload.probs,
            pred=payload.pred,
            aux_pred=payload.aux_pred,
            aux_true=np.asarray(ds.aux, dtype=np.float32),
            split=np.asarray(getattr(ds, "split", []), dtype=object) if hasattr(ds, "split") else None,
            region=np.asarray(getattr(ds, "region", []), dtype=object) if hasattr(ds, "region") else None,
        )
    return out


def bucket_masks(runner: Any, original_payload: SplitPayload) -> dict[str, np.ndarray]:
    y = original_payload.y
    split = np.asarray(original_payload.split, dtype=object)
    region = np.asarray(original_payload.region, dtype=object)
    bad = int(runner.CLASS_TO_INT["bad"])
    return {
        "original_test_all_10s+": split == "test",
        "original_all_10s+": np.ones(len(y), dtype=bool),
        "bad_core_nearboundary": (split == "test") & (y == bad) & (region != "outlier_low_confidence"),
        "bad_outlier_stress": (split == "test") & (y == bad) & (region == "outlier_low_confidence"),
    }


def safe_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x)
    else:
        pred = model.predict(x)
        raw = np.zeros((len(pred), 3), dtype=np.float32)
        raw[np.arange(len(pred)), pred.astype(int)] = 1.0
    out = np.zeros((x.shape[0], 3), dtype=np.float32)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        classes = getattr(final, "classes_", np.arange(raw.shape[1]))
    if classes is None:
        classes = np.arange(raw.shape[1])
    for j, cls in enumerate(classes):
        if 0 <= int(cls) < 3:
            out[:, int(cls)] = raw[:, j]
    out = np.clip(out, 1e-6, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def append_metric_rows(
    rows: list[dict[str, Any]],
    runner: Any,
    candidate: str,
    representation: str,
    feature_set_name: str,
    classifier_name: str,
    split_name: str,
    y: np.ndarray,
    probs: np.ndarray,
) -> None:
    pred = probs.argmax(axis=1).astype(np.int64)
    rep = runner.GEOM.metric_report(y, pred, probs)
    row = {
        "candidate": candidate,
        "representation": representation,
        "feature_set": feature_set_name,
        "classifier": classifier_name,
        "bucket": split_name,
        **rep,
    }
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    rows.append(row)


def feature_recovery_summary(runner: Any, candidate: str, payloads: dict[str, SplitPayload]) -> pd.DataFrame:
    rows = []
    hard_names = set(runner.HARD_FEATURE_COLUMNS) | {
        "qrs_visibility",
        "detector_agreement",
        "baseline_step",
        "sqi_basSQI",
        "flatline_ratio",
        "pc2",
        "pc3",
        "pca_margin",
        "boundary_confidence",
        "knn_label_purity",
    }
    for split_name in ["synthetic_test", "original_all"]:
        payload = payloads[split_name]
        for i, col in enumerate(runner.FEATURE_COLUMNS):
            t = payload.aux_true[:, i].astype(float)
            p = payload.aux_pred[:, i].astype(float)
            corr = float("nan")
            if np.std(t) > 1e-8 and np.std(p) > 1e-8:
                corr = float(np.corrcoef(t, p)[0, 1])
            rows.append(
                {
                    "candidate": candidate,
                    "split": split_name,
                    "feature": col,
                    "corr": corr,
                    "mae_z": float(np.mean(np.abs(p - t))),
                    "is_hard_or_key": col in hard_names,
                }
            )
    return pd.DataFrame(rows)


def run_candidate(runner: Any, candidate: str, run_label: str, batch_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = evaluate_model_payloads(runner, candidate, run_label, batch_size)
    fsets = feature_sets(runner)
    rows: list[dict[str, Any]] = []

    for representation in ["aux_pred", "aux_true_reference"]:
        train_matrix = payloads["synthetic_train"].aux_pred if representation == "aux_pred" else payloads["synthetic_train"].aux_true
        for fset_name, cols in fsets.items():
            idx = indices_for(runner, cols)
            x_train = clean_matrix(train_matrix[:, idx])
            y_train = payloads["synthetic_train"].y.astype(np.int64)
            for clf_name in ["hgb", "extratrees", "logreg"]:
                clf = make_classifier(clf_name, seed)
                clf.fit(x_train, y_train)

                for split_name in ["synthetic_val", "synthetic_test"]:
                    split_matrix = payloads[split_name].aux_pred if representation == "aux_pred" else payloads[split_name].aux_true
                    probs = safe_proba(clf, clean_matrix(split_matrix[:, idx]))
                    append_metric_rows(
                        rows,
                        runner,
                        candidate,
                        representation,
                        fset_name,
                        clf_name,
                        split_name,
                        payloads[split_name].y,
                        probs,
                    )

                original = payloads["original_all"]
                original_matrix = original.aux_pred if representation == "aux_pred" else original.aux_true
                original_probs = safe_proba(clf, clean_matrix(original_matrix[:, idx]))
                for bucket, mask in bucket_masks(runner, original).items():
                    append_metric_rows(
                        rows,
                        runner,
                        candidate,
                        representation,
                        fset_name,
                        clf_name,
                        bucket,
                        original.y[mask],
                        original_probs[mask],
                    )

    return pd.DataFrame(rows), feature_recovery_summary(runner, candidate, payloads)


def render_report(metrics: pd.DataFrame, recovery: pd.DataFrame, candidates: list[str]) -> str:
    lines = [
        "# Aux-Prediction Upper-Bound Audit",
        "",
        "This is an external-only diagnostic.  Classifiers are fit on synthetic-train aux values only; BUT rows are report-only.",
        "",
        "## Best Aux-Pred Classifiers On Original Test",
        "",
        "| candidate | feature set | classifier | acc | macro-F1 | good R | medium R | bad R |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    test = metrics[(metrics["representation"] == "aux_pred") & (metrics["bucket"] == "original_test_all_10s+")].copy()
    if len(test):
        best = test.sort_values(["acc", "macro_f1"], ascending=False).groupby("candidate", as_index=False).head(5)
        for _, row in best.iterrows():
            lines.append(
                f"| {row['candidate']} | {row['feature_set']} | {row['classifier']} | "
                f"{float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
                f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} |"
            )
    lines += [
        "",
        "## Feature-Space Reference On Original Test",
        "",
        "| candidate | feature set | classifier | acc | macro-F1 | good R | medium R | bad R |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    ref = metrics[(metrics["representation"] == "aux_true_reference") & (metrics["bucket"] == "original_test_all_10s+")].copy()
    if len(ref):
        best_ref = ref.sort_values(["acc", "macro_f1"], ascending=False).groupby("candidate", as_index=False).head(3)
        for _, row in best_ref.iterrows():
            lines.append(
                f"| {row['candidate']} | {row['feature_set']} | {row['classifier']} | "
                f"{float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
                f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} |"
            )
    lines += [
        "",
        "## Hard Feature Recovery",
        "",
        "| candidate | split | feature | corr | mae_z |",
        "|---|---|---|---:|---:|",
    ]
    hard = recovery[recovery["is_hard_or_key"]].copy()
    key_order = [
        "qrs_visibility",
        "detector_agreement",
        "baseline_step",
        "sqi_basSQI",
        "flatline_ratio",
        "pc2",
        "pc3",
        "pca_margin",
        "boundary_confidence",
        "knn_label_purity",
        "non_qrs_diff_p95",
    ]
    hard["feature_rank"] = hard["feature"].map({name: i for i, name in enumerate(key_order)}).fillna(999)
    hard = hard.sort_values(["candidate", "split", "feature_rank", "feature"])
    for _, row in hard.iterrows():
        if row["feature"] not in key_order:
            continue
        lines.append(
            f"| {row['candidate']} | {row['split']} | {row['feature']} | "
            f"{float(row['corr']):.6f} | {float(row['mae_z']):.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If aux_pred classifiers approach the true-feature reference, the bottleneck is the final class head/fusion.",
        "- If aux_pred classifiers stay near the waveform model's own accuracy, the bottleneck is feature recovery in the encoder/tokenizer.",
        "- The true-feature rows are an upper reference for the 47-column geometry/SQI space, not a deployable waveform-only result.",
        "",
        "## Files",
        "",
        f"- metrics: `{(REPORT_DIR / 'aux_pred_upper_bound_metrics.csv').as_posix()}`",
        f"- recovery: `{(REPORT_DIR / 'aux_pred_feature_recovery.csv').as_posix()}`",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="featurefirst_top20_hardrec_a050,featurefirst_top20_hardrec_eventqrs_primctx_a050,featurefirst_top20_hardrec_teacher_a050",
    )
    parser.add_argument("--run-label", default="search")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261701)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    metrics_parts: list[pd.DataFrame] = []
    recovery_parts: list[pd.DataFrame] = []
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    for candidate in candidates:
        print(f"[aux-pred] evaluating {candidate}", flush=True)
        metric_df, recovery_df = run_candidate(runner, candidate, args.run_label, args.batch_size, args.seed)
        metrics_parts.append(metric_df)
        recovery_parts.append(recovery_df)

    metrics = pd.concat(metrics_parts, ignore_index=True)
    recovery = pd.concat(recovery_parts, ignore_index=True)
    metrics.to_csv(REPORT_DIR / "aux_pred_upper_bound_metrics.csv", index=False)
    recovery.to_csv(REPORT_DIR / "aux_pred_feature_recovery.csv", index=False)
    summary = {
        "candidates": candidates,
        "best_aux_pred_original_test": metrics[
            (metrics["representation"] == "aux_pred") & (metrics["bucket"] == "original_test_all_10s+")
        ]
        .sort_values(["acc", "macro_f1"], ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }
    (REPORT_DIR / "aux_pred_upper_bound_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "aux_pred_upper_bound_report.md").write_text(render_report(metrics, recovery, candidates), encoding="utf-8")
    print(REPORT_DIR / "aux_pred_upper_bound_report.md")


if __name__ == "__main__":
    main()
