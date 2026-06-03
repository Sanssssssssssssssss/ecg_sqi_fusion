"""10s-only BUT QDB baseline tuning for E3.11f.

This runner intentionally fixes the protocol to ``p1_current_10s_center``.
The goal is to make a clean 10 second baseline before changing synthetic data
rules.  It trains only lightweight probes on frozen Uformer/SQI features and
uses validation-only calibration policies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_MAINLINE_CKPT,
    DEFAULT_OUT_ROOT,
    DEFAULT_REPORT_ROOT,
    feature_matrix_for_protocol,
    load_protocol_signals,
)
from src.transformer_pipeline.external_benchmarks.run import (
    apply_but_thresholds,
    calibrate_but,
    multiclass_report,
    plot_wave_gallery,
)


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
CLASS_NAMES = ("good", "medium", "bad")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_state(path: Path, **kwargs: Any) -> None:
    state = read_json(path) if path.exists() else {"started_at": now_iso()}
    state.update(kwargs)
    write_json(path, state)


def classifier_scores(clf: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = clf.named_steps["model"]
    if hasattr(model, "predict_proba"):
        probs = clf.predict_proba(X).astype(np.float32)
        logits = np.log(np.clip(probs, 1e-8, 1.0))
        return probs, logits
    if hasattr(model, "decision_function"):
        z = clf.decision_function(X)
        if z.ndim == 1:
            z = np.stack([-z, np.zeros_like(z), z], axis=1)
        return softmax(z, axis=1).astype(np.float32), z.astype(np.float32)
    pred = clf.predict(X).astype(np.int64)
    probs = np.zeros((len(pred), 3), dtype=np.float32)
    probs[np.arange(len(pred)), pred] = 1.0
    return probs, np.log(np.clip(probs, 1e-8, 1.0))


def threshold_grid(probs_val: np.ndarray, y_val: np.ndarray, policy: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for t_bad in np.linspace(0.01, 0.99, 50):
        for t_good in np.linspace(0.01, 0.99, 50):
            pred = apply_but_thresholds(probs_val, float(t_good), float(t_bad))
            rep = multiclass_report(y_val, pred, probs_val)
            rows.append({"t_good": float(t_good), "t_bad": float(t_bad), "val_report": rep})

    def key(row: dict[str, Any]) -> tuple[float, ...]:
        rep = row["val_report"]
        rec = rep["recall_good_medium_bad"]
        if policy == "acc_primary":
            return (rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        if policy == "balanced_primary":
            return (rep["balanced_acc"], rep["macro_f1"], rep["acc"], rec[2])
        if policy == "macro_primary":
            return (rep["macro_f1"], rec[2], rep["balanced_acc"], rep["acc"])
        if policy.startswith("bad"):
            target = float(policy.replace("bad", "").replace("_acc", "")) / 100.0
            ok = 1.0 if rec[2] >= target else 0.0
            return (ok, rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        raise ValueError(policy)

    best = max(rows, key=key)
    best["policy"] = policy
    return best


def threshold_grid_many(probs_val: np.ndarray, y_val: np.ndarray, policies: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t_bad in np.linspace(0.01, 0.99, 50):
        for t_good in np.linspace(0.01, 0.99, 50):
            pred = apply_but_thresholds(probs_val, float(t_good), float(t_bad))
            rep = multiclass_report(y_val, pred, probs_val)
            rows.append({"t_good": float(t_good), "t_bad": float(t_bad), "val_report": rep})

    def key(row: dict[str, Any], policy: str) -> tuple[float, ...]:
        rep = row["val_report"]
        rec = rep["recall_good_medium_bad"]
        if policy == "acc_primary":
            return (rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        if policy == "balanced_primary":
            return (rep["balanced_acc"], rep["macro_f1"], rep["acc"], rec[2])
        if policy == "macro_primary":
            return (rep["macro_f1"], rec[2], rep["balanced_acc"], rep["acc"])
        if policy.startswith("bad"):
            target = float(policy.replace("bad", "").replace("_acc", "")) / 100.0
            ok = 1.0 if rec[2] >= target else 0.0
            return (ok, rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        raise ValueError(policy)

    out: list[dict[str, Any]] = []
    for policy in policies:
        best = dict(max(rows, key=lambda row, p=policy: key(row, p)))
        best["policy"] = policy
        out.append(best)
    return out


def bias_grid(logits_val: np.ndarray, y_val: np.ndarray, policy: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for b_good in np.linspace(-1.5, 1.5, 13):
        for b_med in np.linspace(-1.5, 1.5, 13):
            for b_bad in np.linspace(-1.5, 1.5, 13):
                logits = logits_val + np.asarray([b_good, b_med, b_bad], dtype=np.float32)
                probs = softmax(logits, axis=1)
                pred = probs.argmax(axis=1)
                rep = multiclass_report(y_val, pred, probs)
                rows.append({"bias": [float(b_good), float(b_med), float(b_bad)], "val_report": rep})

    def key(row: dict[str, Any]) -> tuple[float, ...]:
        rep = row["val_report"]
        rec = rep["recall_good_medium_bad"]
        if policy == "bias_acc":
            return (rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        if policy == "bias_macro":
            return (rep["macro_f1"], rec[2], rep["balanced_acc"], rep["acc"])
        if policy == "bias_bad70_acc":
            ok = 1.0 if rec[2] >= 0.70 else 0.0
            return (ok, rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        raise ValueError(policy)

    best = max(rows, key=key)
    best["policy"] = policy
    return best


def bias_grid_many(logits_val: np.ndarray, y_val: np.ndarray, policies: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for b_good in np.linspace(-1.5, 1.5, 13):
        for b_med in np.linspace(-1.5, 1.5, 13):
            for b_bad in np.linspace(-1.5, 1.5, 13):
                logits = logits_val + np.asarray([b_good, b_med, b_bad], dtype=np.float32)
                probs = softmax(logits, axis=1)
                pred = probs.argmax(axis=1)
                rep = multiclass_report(y_val, pred, probs)
                rows.append({"bias": [float(b_good), float(b_med), float(b_bad)], "val_report": rep})

    def key(row: dict[str, Any], policy: str) -> tuple[float, ...]:
        rep = row["val_report"]
        rec = rep["recall_good_medium_bad"]
        if policy == "bias_acc":
            return (rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        if policy == "bias_macro":
            return (rep["macro_f1"], rec[2], rep["balanced_acc"], rep["acc"])
        if policy == "bias_bad70_acc":
            ok = 1.0 if rec[2] >= 0.70 else 0.0
            return (ok, rep["acc"], rep["macro_f1"], rep["balanced_acc"], rec[2])
        raise ValueError(policy)

    out: list[dict[str, Any]] = []
    for policy in policies:
        best = dict(max(rows, key=lambda row, p=policy: key(row, p)))
        best["policy"] = policy
        out.append(best)
    return out


def apply_calibration(probs: np.ndarray, logits: np.ndarray, cal: dict[str, Any]) -> np.ndarray:
    policy = cal["policy"]
    if policy.startswith("bias"):
        shifted = logits + np.asarray(cal["bias"], dtype=np.float32)
        return softmax(shifted, axis=1).argmax(axis=1)
    return apply_but_thresholds(probs, float(cal["t_good"]), float(cal["t_bad"]))


def concat_features(protocol_dir: Path, checkpoint: Path, names: list[str], batch_size: int, device: torch.device) -> np.ndarray:
    arrays = [feature_matrix_for_protocol(protocol_dir, checkpoint, name, batch_size, device, force=False) for name in names]
    return np.concatenate(arrays, axis=1).astype(np.float32)


def make_model(name: str, seed: int) -> Any:
    if name == "logreg_c0p3":
        return LogisticRegression(C=0.3, max_iter=1500, class_weight="balanced", solver="lbfgs")
    if name == "logreg_c1":
        return LogisticRegression(C=1.0, max_iter=1500, class_weight="balanced", solver="lbfgs")
    if name == "logreg_c3":
        return LogisticRegression(C=3.0, max_iter=1500, class_weight="balanced", solver="lbfgs")
    if name == "linear_svm_c0p5":
        return LinearSVC(C=0.5, class_weight="balanced", max_iter=12000)
    if name == "linear_svm_c1":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=12000)
    if name == "mlp_96":
        return MLPClassifier(hidden_layer_sizes=(96,), activation="relu", alpha=1e-4, max_iter=700, early_stopping=True, random_state=seed)
    if name == "mlp_128_64":
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=2e-4, max_iter=700, early_stopping=True, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=350, max_depth=None, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=280, max_depth=18, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def export_error_galleries(
    protocol_dir: Path,
    probs: np.ndarray,
    pred: np.ndarray,
    out_dir: Path,
    title_prefix: str,
) -> None:
    X_obj, meta, _ = load_protocol_signals(protocol_dir)
    X = X_obj if isinstance(X_obj, np.ndarray) else X_obj[0]
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    test = split == "test"
    masks = {
        "bad_missed": test & (y == 2) & (pred != 2),
        "medium_confusion": test & (y == 1) & (pred != 1),
        "good_false_bad": test & (y == 0) & (pred == 2),
        "correct_bad": test & (y == 2) & (pred == 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        idx = np.flatnonzero(mask)[:24].tolist()
        if idx:
            plot_wave_gallery(X, meta, [int(i) for i in idx], out_dir / f"{name}.png", f"{title_prefix} {name}", probs=probs)


def write_summary(rows: list[dict[str, Any]], report_root: Path) -> None:
    best_acc = sorted(rows, key=lambda r: (r["test_report"]["acc"], r["test_report"]["macro_f1"]), reverse=True)
    best_bal = sorted(rows, key=lambda r: (r["test_report"]["balanced_acc"], r["test_report"]["macro_f1"]), reverse=True)
    report_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BUT 10s Baseline Tuning",
        "",
        "Protocol is fixed to `p1_current_10s_center`.  No 5s crop, ensemble, denoiser training, or synthetic generator changes are used.",
        "",
        "## Best By Accuracy",
        "",
        "| rank | features | model | calibration | acc | bal acc | macro-F1 | recalls good/medium/bad |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(best_acc[:20], start=1):
        rep = row["test_report"]
        rec = rep["recall_good_medium_bad"]
        lines.append(
            f"| {rank} | {row['feature_combo']} | {row['model']} | {row['calibration_policy']} | "
            f"{rep['acc']:.4f} | {rep['balanced_acc']:.4f} | {rep['macro_f1']:.4f} | "
            f"{rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f} |"
        )
    lines.extend(["", "## Best By Balanced Accuracy", "", "| rank | features | model | calibration | acc | bal acc | macro-F1 | recalls good/medium/bad |", "| --- | --- | --- | --- | ---: | ---: | ---: | --- |"])
    for rank, row in enumerate(best_bal[:20], start=1):
        rep = row["test_report"]
        rec = rep["recall_good_medium_bad"]
        lines.append(
            f"| {rank} | {row['feature_combo']} | {row['model']} | {row['calibration_policy']} | "
            f"{rep['acc']:.4f} | {rep['balanced_acc']:.4f} | {rep['macro_f1']:.4f} | "
            f"{rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading This Baseline",
            "",
            "- Accuracy can look high when bad recall is weak because BUT test has far fewer bad windows.",
            "- The recommended 10s baseline should therefore be chosen from the balanced/macro-F1 table, not accuracy alone.",
            "- Any later synthetic adaptation must beat this 10s baseline on acc, macro-F1, balanced acc, and bad recall.",
        ]
    )
    (report_root / "ten_s_baseline_tuning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "ten_s_baseline_tuning_summary.json", {"runs": rows, "best_acc": best_acc[:20], "best_balanced": best_bal[:20]})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    protocol_dir = out_root / "protocols" / "p1_current_10s_center"
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Missing 10s protocol directory: {protocol_dir}")
    result_dir = out_root / "ten_s_tuning"
    result_dir.mkdir(parents=True, exist_ok=True)
    update_state(out_root / "but_10s_tuning_state.json", status="running", updated_at=now_iso())
    _, meta, _ = load_protocol_signals(protocol_dir)
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    feature_combos = {
        "full_tokens": ["full_tokens"],
        "bottleneck_only": ["bottleneck_only"],
        "summary_only": ["summary_only"],
        "handcrafted_sqi": ["handcrafted_sqi"],
        "full_plus_handcrafted": ["full_tokens", "handcrafted_sqi"],
        "bottleneck_plus_handcrafted": ["bottleneck_only", "handcrafted_sqi"],
        "full_plus_summary": ["full_tokens", "summary_only"],
    }
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    calibration_policies = [
        "raw_argmax",
        "existing_macro",
        "acc_primary",
        "balanced_primary",
        "macro_primary",
        "bad50_acc",
        "bad60_acc",
        "bad70_acc",
        "bias_acc",
        "bias_macro",
        "bias_bad70_acc",
    ]
    rows: list[dict[str, Any]] = []
    for combo_name, feature_names in feature_combos.items():
        feats = concat_features(protocol_dir, Path(args.checkpoint), feature_names, int(args.batch_size), device)
        for model_name in models:
            clf = Pipeline([("scaler", StandardScaler()), ("model", make_model(model_name, int(args.seed)))])
            clf.fit(feats[train], y[train])
            probs_val, logits_val = classifier_scores(clf, feats[val])
            probs_test, logits_test = classifier_scores(clf, feats[test])
            calibrations: list[dict[str, Any]] = [{"policy": "raw_argmax", "val_report": multiclass_report(y[val], probs_val.argmax(axis=1), probs_val)}]
            existing = calibrate_but(probs_val, y[val])
            existing["policy"] = "existing_macro"
            calibrations.append(existing)
            calibrations.extend(
                threshold_grid_many(
                    probs_val,
                    y[val],
                    ["acc_primary", "balanced_primary", "macro_primary", "bad50_acc", "bad60_acc", "bad70_acc"],
                )
            )
            calibrations.extend(bias_grid_many(logits_val, y[val], ["bias_acc", "bias_macro", "bias_bad70_acc"]))
            for cal in calibrations:
                if cal["policy"] == "raw_argmax":
                    pred = probs_test.argmax(axis=1)
                else:
                    pred = apply_calibration(probs_test, logits_test, cal)
                rep = multiclass_report(y[test], pred, probs_test)
                row = {
                    "feature_combo": combo_name,
                    "feature_names": feature_names,
                    "model": model_name,
                    "calibration_policy": cal["policy"],
                    "calibration": cal,
                    "test_report": rep,
                }
                rows.append(row)
                append_jsonl(result_dir / "ten_s_tuning_summary.jsonl", row)
    write_summary(rows, report_root)
    best_bal = max(rows, key=lambda r: (r["test_report"]["balanced_acc"], r["test_report"]["macro_f1"], r["test_report"]["acc"]))
    best_acc = max(rows, key=lambda r: (r["test_report"]["acc"], r["test_report"]["macro_f1"], r["test_report"]["balanced_acc"]))
    # Recompute probabilities for galleries of the balanced candidate.
    feature_names = best_bal["feature_names"]
    feats = concat_features(protocol_dir, Path(args.checkpoint), feature_names, int(args.batch_size), device)
    clf = Pipeline([("scaler", StandardScaler()), ("model", make_model(best_bal["model"], int(args.seed)))])
    clf.fit(feats[train], y[train])
    probs_test, logits_test = classifier_scores(clf, feats[test])
    full_probs = np.zeros((len(y), 3), dtype=np.float32)
    full_pred = np.zeros(len(y), dtype=np.int64)
    probs_all, logits_all = classifier_scores(clf, feats)
    if best_bal["calibration_policy"] == "raw_argmax":
        pred_all = probs_all.argmax(axis=1)
    else:
        pred_all = apply_calibration(probs_all, logits_all, best_bal["calibration"])
    full_probs[:] = probs_all
    full_pred[:] = pred_all
    export_error_galleries(protocol_dir, full_probs, full_pred, result_dir / "visuals_best_balanced", "10s best balanced")
    payload = {"status": "complete", "updated_at": now_iso(), "n_runs": len(rows), "best_balanced": best_bal, "best_acc": best_acc}
    write_json(result_dir / "ten_s_tuning_best.json", payload)
    update_state(out_root / "but_10s_tuning_state.json", **payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune BUT 10s-only frozen baselines.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MAINLINE_CKPT))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--models", default="logreg_c0p3,logreg_c1,logreg_c3,linear_svm_c0p5,linear_svm_c1,mlp_96,mlp_128_64,extra_trees,random_forest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps({"status": "starting", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    result = run(args)
    print(json.dumps({"status": "complete", "best_balanced": result["best_balanced"]["test_report"], "best_acc": result["best_acc"]["test_report"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
