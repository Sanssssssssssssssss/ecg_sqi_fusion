"""Synthetic-only router probe over waveform Transformer probabilities.

This external-only diagnostic tests whether existing waveform-only Transformer
experts contain complementary information that can be routed using synthetic
train/val labels alone.  It never uses original BUT for training, threshold
selection, or model selection; original buckets are report-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
RUNNER_PATH = ANALYSIS_DIR / "run_waveform_geometry_student.py"
NODE_ID = "N17043_gm_probe"

DEFAULT_CANDIDATES = [
    "featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_jointartifact_badrecall_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_router_event_mediumrecall_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_veto_finetune_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_dual_balanced_a050",
    "featurefirst_top20_qrsbase_dualcoreout_auxphysio_badguard_a050",
    "featurefirst_wavecomp_dualcoreout_encoderlite_mediumguard_a050",
]


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_optional_state(runner: Any, model: torch.nn.Module, cfg: dict[str, Any], base_train_ds: Any, ckpt: dict[str, Any]) -> None:
    if getattr(model, "use_teacher_atlas", False):
        if ckpt.get("teacher_prototypes") is not None:
            model.set_teacher_prototypes(np.asarray(ckpt["teacher_prototypes"], dtype=np.float32))
        else:
            model.set_teacher_prototypes(
                runner.build_teacher_prototypes(
                    base_train_ds.aux,
                    base_train_ds.y,
                    int(cfg["teacher_prototypes_per_class"]),
                    str(cfg.get("teacher_prototype_mode", "pc1_quantile")),
                )
            )
    if getattr(model, "use_waveform_atlas", False):
        if ckpt.get("waveform_atlas_prototypes") is not None:
            model.set_waveform_atlas(
                np.asarray(ckpt["waveform_atlas_prototypes"], dtype=np.float32),
                np.asarray(ckpt["waveform_atlas_mean"], dtype=np.float32),
                np.asarray(ckpt["waveform_atlas_std"], dtype=np.float32),
            )
        else:
            wave_proto, wave_mean, wave_std = runner.build_waveform_atlas(
                base_train_ds,
                int(cfg["waveform_atlas_prototypes_per_class"]),
                str(cfg.get("waveform_atlas_mode", "farthest")),
            )
            model.set_waveform_atlas(wave_proto, wave_mean, wave_std)


def compatible_load(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible, strict=False)


def load_model_payloads(runner: Any, name: str, split_loaders: dict[str, DataLoader], device: torch.device) -> dict[str, np.ndarray]:
    ckpt_path = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search" / name / "ckpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    base_train_ds = runner.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    model = runner.GeometryStudent(cfg, int(base_train_ds.x.shape[1])).to(device)
    configure_optional_state(runner, model, cfg, base_train_ds, ckpt)
    compatible_load(model, ckpt["model_state"])
    out: dict[str, np.ndarray] = {}
    for split, loader in split_loaders.items():
        payload = runner.eval_loader(model, loader, device)
        out[f"{split}_probs"] = payload.probs.astype(np.float32)
        out[f"{split}_pred"] = payload.pred.astype(np.int64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def make_router_features(prob_blocks: list[np.ndarray]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for probs in prob_blocks:
        margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
        entropy = -(probs * np.log(np.clip(probs, 1e-7, 1.0))).sum(axis=1)
        gm_delta = probs[:, 1] - probs[:, 0]
        bad_gap = probs[:, 2] - np.maximum(probs[:, 0], probs[:, 1])
        parts.extend([probs, margin[:, None], entropy[:, None], gm_delta[:, None], bad_gap[:, None]])
    return np.concatenate(parts, axis=1).astype(np.float32)


def metric_row(runner: Any, candidate: str, bucket: str, y: np.ndarray, probs: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    rep = runner.GEOM.metric_report(y, pred, probs)
    row = {"candidate": candidate, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-name", default="waveform_probability_router_probe")
    args = parser.parse_args()

    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    runner = load_runner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The router probe intentionally uses a single fixed channel contract.  The
    # current candidates in this probe are all robust3 waveform-only models.
    train_ds = runner.ARCH.SyntheticWaveDataset("train", "robust3")
    val_ds = runner.ARCH.SyntheticWaveDataset("val", "robust3")
    test_ds = runner.ARCH.SyntheticWaveDataset("test", "robust3")
    norm = runner.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = runner.ARCH.OriginalWaveDataset(norm, "robust3")
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "original": DataLoader(original_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    model_probs: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        print(f"collect {name}", flush=True)
        model_probs[name] = load_model_payloads(runner, name, loaders, device)

    x_train = make_router_features([model_probs[name]["train_probs"] for name in names])
    x_val = make_router_features([model_probs[name]["val_probs"] for name in names])
    x_test = make_router_features([model_probs[name]["test_probs"] for name in names])
    x_original = make_router_features([model_probs[name]["original_probs"] for name in names])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rows: list[dict[str, Any]] = []
    best = None
    for c_value in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                max_iter=2000,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=13,
            ),
        )
        clf.fit(x_train, train_ds.y)
        val_probs = clf.predict_proba(x_val).astype(np.float32)
        val_pred = val_probs.argmax(axis=1)
        val_rep = runner.GEOM.metric_report(val_ds.y, val_pred, val_probs)
        score = (
            float(val_rep["acc"])
            + 0.22 * float(val_rep["macro_f1"])
            + 0.08 * min(float(val_rep["good_recall"]), float(val_rep["medium_recall"]), float(val_rep["bad_recall"]))
        )
        rows.append({"candidate": f"router_C{c_value:g}", "bucket": "synthetic_val", **val_rep, "score": score})
        if best is None or score > best[0]:
            best = (score, c_value, clf, val_probs)
    assert best is not None
    _, best_c, best_clf, val_probs = best
    threshold = runner.ARCH.calibrate_bad_threshold(val_ds.y, val_probs)

    split_specs = [
        ("synthetic_test", test_ds.y, x_test, None),
        ("original_test_all_10s+", original_ds.y, x_original, runner.ARCH.bucket_mask(original_ds, "original_test_all_10s+")),
        ("original_all_10s+", original_ds.y, x_original, runner.ARCH.bucket_mask(original_ds, "original_all_10s+")),
        ("bad_core_nearboundary", original_ds.y, x_original, runner.ARCH.bucket_mask(original_ds, "bad_core_nearboundary")),
        ("bad_outlier_stress", original_ds.y, x_original, runner.ARCH.bucket_mask(original_ds, "bad_outlier_stress")),
    ]
    for bucket, y, x, mask in split_specs:
        probs = best_clf.predict_proba(x).astype(np.float32)
        if mask is not None:
            probs_eval = probs[mask]
            y_eval = y[mask]
        else:
            probs_eval = probs
            y_eval = y
        pred = probs_eval.argmax(axis=1)
        rows.append(metric_row(runner, f"prob_router_C{best_c:g}", bucket, y_eval, probs_eval, pred))
        badcal_pred = runner.ARCH.apply_bad_threshold(probs_eval, threshold)
        rows.append(metric_row(runner, f"prob_router_C{best_c:g}_badcal", bucket, y_eval, probs_eval, badcal_pred))

    out_dir = REPORT_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = {
        "contract": "router trained on synthetic waveform-transformer probabilities only; original BUT report-only",
        "candidates": names,
        "best_c": best_c,
        "bad_threshold_from_synthetic_val": float(threshold),
        "metrics_csv": str(out_dir / "metrics.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    show = metrics[metrics["bucket"].isin(["synthetic_test", "original_test_all_10s+"])].copy()
    print(show[["candidate", "bucket", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall"]].to_csv(index=False), flush=True)


if __name__ == "__main__":
    main()
