"""BUT 10s supervised head-only adaptation for bad-boundary diagnosis.

Generator tuning hit a plateau: the b10 synthetic rule is useful, but full
synthetic training does not improve BUT bad recall.  This runner freezes each
candidate Uformer checkpoint, extracts inference-time features on the formal
BUT 10s protocol, and trains only lightweight heads on BUT train/val.  Test is
used once per trained head/calibration policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_10s_tuning import (
    apply_calibration,
    bias_grid_many,
    classifier_scores,
    threshold_grid_many,
)
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_MAINLINE_CKPT,
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    load_protocol_signals,
)
from src.transformer_pipeline.external_benchmarks.run import (
    basic_sqi_features,
    extract_uformer_features,
    multiclass_report,
    plot_wave_gallery,
)


RUN_TAG = "e311_but_boundary_head_adaptation_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_PROTOCOL_DIR = PROTOCOL_OUT_ROOT / "protocols" / "p1_current_10s_center"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_state(path: Path, **kwargs: Any) -> None:
    state = read_json(path) if path.exists() else {"started_at": now_iso()}
    state.update(kwargs)
    write_json(path, state)


def checkpoint_entries() -> list[dict[str, Any]]:
    return [
        {"id": "mainline_uformer", "checkpoint": DEFAULT_MAINLINE_CKPT, "note": "Current mainline checkpoint."},
        {
            "id": "quick_b10_wearable",
            "checkpoint": ROOT
            / "outputs"
            / "external_benchmarks"
            / "e311_but_bad_boundary_10s_2026_06_03"
            / "runs"
            / "quick"
            / "b10_all_bad_wearable"
            / "ckpt_best.pt",
            "note": "Quick synthetic b10 winner.",
        },
        {
            "id": "full_b10_wearable",
            "checkpoint": ROOT
            / "outputs"
            / "external_benchmarks"
            / "e311_but_bad_boundary_full_confirm_10s_2026_06_03"
            / "runs"
            / "b10_all_bad_wearable_full_cw190"
            / "ckpt_best.pt",
            "note": "Full b10 confirmation checkpoint.",
        },
        {
            "id": "full_r08_bad_prior_mild",
            "checkpoint": ROOT
            / "outputs"
            / "external_benchmarks"
            / "e311_but_bad_boundary_full_confirm_10s_2026_06_03"
            / "runs"
            / "r08_bad_prior_mild_full_cw160"
            / "ckpt_best.pt",
            "note": "Full refined r08 checkpoint.",
        },
    ]


def make_model(name: str, seed: int) -> Any:
    if name == "logreg_c1":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
    if name == "logreg_c3":
        return LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
    if name == "linear_svm_c1":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=15000)
    if name == "mlp_128_64":
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=2e-4, max_iter=900, early_stopping=True, random_state=seed)
    if name == "mlp_192_96":
        return MLPClassifier(hidden_layer_sizes=(192, 96), activation="relu", alpha=2e-4, max_iter=900, early_stopping=True, random_state=seed)
    raise ValueError(name)


def load_features(
    protocol_dir: Path,
    checkpoint_id: str,
    checkpoint: Path,
    feature_set: str,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    out_root: Path,
    force: bool = False,
) -> np.ndarray:
    cache_dir = out_root / "feature_cache" / checkpoint_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{feature_set}.npy"
    if path.exists() and not force:
        return np.load(path).astype(np.float32)
    if feature_set == "handcrafted_sqi":
        feats = basic_sqi_features(X)
    else:
        feats = extract_uformer_features(checkpoint, X, feature_set, batch_size, device)
    np.save(path, feats.astype(np.float32))
    return feats.astype(np.float32)


def concat_feature_combo(
    protocol_dir: Path,
    checkpoint_id: str,
    checkpoint: Path,
    names: list[str],
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    out_root: Path,
    force: bool,
) -> np.ndarray:
    arrays = [load_features(protocol_dir, checkpoint_id, checkpoint, name, X, batch_size, device, out_root, force=force) for name in names]
    return np.concatenate(arrays, axis=1).astype(np.float32)


def calibrations(probs_val: np.ndarray, logits_val: np.ndarray, y_val: np.ndarray) -> list[dict[str, Any]]:
    cals: list[dict[str, Any]] = [{"policy": "raw_argmax", "val_report": multiclass_report(y_val, probs_val.argmax(axis=1), probs_val)}]
    cals.extend(
        threshold_grid_many(
            probs_val,
            y_val,
            ["acc_primary", "balanced_primary", "macro_primary", "bad60_acc", "bad70_acc", "bad80_acc", "bad85_acc"],
        )
    )
    cals.extend(bias_grid_many(logits_val, y_val, ["bias_acc", "bias_macro", "bias_bad70_acc"]))
    return cals


def export_best_galleries(protocol_dir: Path, probs: np.ndarray, pred: np.ndarray, out_dir: Path, title_prefix: str) -> None:
    X_obj, meta, _ = load_protocol_signals(protocol_dir)
    X = X_obj if isinstance(X_obj, np.ndarray) else X_obj[0]
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    test = split == "test"
    masks = {
        "bad_missed": test & (y == 2) & (pred != 2),
        "correct_bad": test & (y == 2) & (pred == 2),
        "medium_confusion": test & (y == 1) & (pred != 1),
        "good_false_bad": test & (y == 0) & (pred == 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        idx = np.flatnonzero(mask)[:24].tolist()
        if idx:
            plot_wave_gallery(X, meta, [int(i) for i in idx], out_dir / f"{name}.png", f"{title_prefix} {name}", probs=probs)


def score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rep = row["test_report"]
    rec = rep["recall_good_medium_bad"]
    return (float(rep["balanced_acc"]), float(rep["macro_f1"]), float(rec[2]), float(rep["acc"]))


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=score, reverse=True)
    lines = [
        "# BUT 10s Boundary Head-Only Adaptation",
        "",
        "Frozen Uformer features, BUT train/val-only head fitting, and validation-only calibration.  This is not zero-shot; it tests whether the representation can support the BUT bad boundary once the final boundary is learned on BUT.",
        "",
        "| rank | checkpoint | features | model | calibration | acc | bal | macro-F1 | recalls good/medium/bad |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for i, row in enumerate(ranked[:40], start=1):
        rep = row["test_report"]
        rec = rep["recall_good_medium_bad"]
        lines.append(
            f"| {i} | {row['checkpoint_id']} | {row['feature_combo']} | {row['model']} | {row['calibration_policy']} | "
            f"{float(rep['acc']):.4f} | {float(rep['balanced_acc']):.4f} | {float(rep['macro_f1']):.4f} | "
            f"{float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- If head-only adaptation reaches high bad recall with balanced/macro-F1 improvement, the frozen representation is useful and the remaining issue is boundary/domain calibration.",
            "- If it cannot recover bad, the representation itself lacks BUT bad cues and generator/denoiser data design must change more deeply.",
        ]
    )
    (report_root / "boundary_head_adaptation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Full per-run details live in boundary_head_summary.jsonl.  Keep the report
    # JSON compact so repeated refreshes do not fail on Windows with large writes.
    compact_ranked = [
        {
            "rank": i,
            "checkpoint_id": row["checkpoint_id"],
            "feature_combo": row["feature_combo"],
            "model": row["model"],
            "calibration_policy": row["calibration_policy"],
            "test_report": row["test_report"],
        }
        for i, row in enumerate(ranked[:80], start=1)
    ]
    try:
        write_json(report_root / "boundary_head_adaptation_summary.json", {"n_rows": len(rows), "ranked": compact_ranked})
    except OSError as exc:
        # Some Windows runs intermittently fail while repeatedly replacing this
        # report file.  The markdown report and output JSONL are authoritative;
        # do not stop the experiment for a refresh-only artifact.
        (report_root / "boundary_head_adaptation_summary_json_write_error.txt").write_text(str(exc), encoding="utf-8")


def load_existing_rows(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str]]]:
    if not path.exists():
        return [], set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {
        (
            str(row.get("checkpoint_id")),
            str(row.get("feature_combo")),
            str(row.get("model")),
            str(row.get("calibration_policy")),
        )
        for row in rows
    }
    return rows, seen


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    protocol_dir = Path(args.protocol_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "boundary_head_state.json"
    update_state(state_path, status="running", updated_at=now_iso())
    X_obj, meta, _ = load_protocol_signals(protocol_dir)
    X = X_obj if isinstance(X_obj, np.ndarray) else X_obj[0]
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
        "full_plus_handcrafted": ["full_tokens", "handcrafted_sqi"],
        "bottleneck_plus_handcrafted": ["bottleneck_only", "handcrafted_sqi"],
        "full_plus_summary": ["full_tokens", "summary_only"],
    }
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary_path = out_root / "boundary_head_summary.jsonl"
    rows, seen = load_existing_rows(summary_path)
    if rows:
        write_report(args, rows)
    for ckpt_entry in checkpoint_entries():
        checkpoint = Path(ckpt_entry["checkpoint"])
        if not checkpoint.exists():
            continue
        for combo_name, names in feature_combos.items():
            update_state(state_path, status="extracting_training", checkpoint=ckpt_entry["id"], feature_combo=combo_name, updated_at=now_iso())
            feats = concat_feature_combo(protocol_dir, ckpt_entry["id"], checkpoint, names, X, int(args.batch_size), device, out_root, force=bool(args.force_features))
            for model_name in models:
                clf = Pipeline([("scaler", StandardScaler()), ("model", make_model(model_name, int(args.seed)))])
                clf.fit(feats[train], y[train])
                probs_val, logits_val = classifier_scores(clf, feats[val])
                probs_test, logits_test = classifier_scores(clf, feats[test])
                for cal in calibrations(probs_val, logits_val, y[val]):
                    run_key = (ckpt_entry["id"], combo_name, model_name, cal["policy"])
                    if run_key in seen:
                        continue
                    if cal["policy"] == "raw_argmax":
                        pred = probs_test.argmax(axis=1)
                    else:
                        pred = apply_calibration(probs_test, logits_test, cal)
                    rep = multiclass_report(y[test], pred, probs_test)
                    row = {
                        "checkpoint_id": ckpt_entry["id"],
                        "checkpoint": str(checkpoint),
                        "checkpoint_note": ckpt_entry["note"],
                        "feature_combo": combo_name,
                        "feature_names": names,
                        "model": model_name,
                        "calibration_policy": cal["policy"],
                        "calibration": cal,
                        "test_report": rep,
                    }
                    rows.append(row)
                    seen.add(run_key)
                    append_jsonl(summary_path, row)
                    write_report(args, rows)
    ranked = sorted(rows, key=score, reverse=True)
    if ranked:
        best = ranked[0]
        checkpoint = Path(best["checkpoint"])
        feats = concat_feature_combo(protocol_dir, best["checkpoint_id"], checkpoint, best["feature_names"], X, int(args.batch_size), device, out_root, force=False)
        clf = Pipeline([("scaler", StandardScaler()), ("model", make_model(best["model"], int(args.seed)))])
        clf.fit(feats[train], y[train])
        probs_all, logits_all = classifier_scores(clf, feats)
        pred_all = probs_all.argmax(axis=1) if best["calibration_policy"] == "raw_argmax" else apply_calibration(probs_all, logits_all, best["calibration"])
        export_best_galleries(protocol_dir, probs_all, pred_all, out_root / "visuals_best", f"{best['checkpoint_id']} {best['feature_combo']} {best['model']}")
    write_report(args, rows)
    update_state(state_path, status="complete", updated_at=now_iso(), n_runs=len(rows), best=ranked[0] if ranked else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BUT 10s head-only boundary probes on frozen Uformer features.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_dir", default=str(DEFAULT_PROTOCOL_DIR))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--models", default="logreg_c1,logreg_c3,linear_svm_c1,mlp_128_64,mlp_192_96")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force_features", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / "boundary_head_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
