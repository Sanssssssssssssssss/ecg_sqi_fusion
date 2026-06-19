"""PTB-only router over waveform Transformer endpoints.

The router consumes only endpoint probability outputs produced from ECG
waveforms. It does not consume 47-column SQI/geometry features and it never
uses Original BUT for training, validation, threshold selection, or model
selection. Original BUT is report-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
RUNNER = ANALYSIS_DIR / "run_waveform_geometry_student.py"
NODE_ID = "N17043_gm_probe"
SEARCH_DIR = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search"
OUT_DIR = ANALYSIS_DIR / "waveform_endpoint_router"

DEFAULT_ENDPOINTS = [
    "featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050",
    "waveprimtoken_v5_p20",
    "featurefirst_top20_rawbeat_artifact_auxctx_physioctx_badrecall_a050",
    "featurefirst_top20_qrsbase_physioctx_badrecall_a050",
    "detbase_rrtokens_qfeatbin_badguard",
    "physio_sqi_token_balanced_a050",
    "featurefirst_top20_dualcoreout_mediumshell_balanced_a050",
]


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1.0)
    return -(p * np.log(p)).sum(axis=1, keepdims=True)


def endpoint_features(prob_blocks: list[np.ndarray]) -> np.ndarray:
    feats: list[np.ndarray] = []
    for probs in prob_blocks:
        top2 = np.sort(probs, axis=1)[:, -2:]
        margin = (top2[:, 1] - top2[:, 0])[:, None]
        gm_margin = (probs[:, 0] - probs[:, 1])[:, None]
        bad_margin = (probs[:, 2] - np.maximum(probs[:, 0], probs[:, 1]))[:, None]
        feats.extend([probs, margin, gm_margin, bad_margin, entropy(probs)])
    stack = np.stack([p.argmax(axis=1) for p in prob_blocks], axis=1)
    agree_frac = []
    for cls in [0, 1, 2]:
        agree_frac.append((stack == cls).mean(axis=1, keepdims=True))
    feats.extend(agree_frac)
    return np.concatenate(feats, axis=1).astype(np.float32)


def make_metric_row(run: Any, candidate: str, split: str, y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    rep = run.GEOM.metric_report(y, pred, probs)
    row = {"candidate": candidate, "bucket": split}
    row.update({k: v for k, v in rep.items() if k != "confusion_3x3"})
    row["confusion_3x3"] = json.dumps(rep["confusion_3x3"])
    return row


def original_bucket_rows(run: Any, ds: Any, candidate: str, pred: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
        mask = run.ARCH.bucket_mask(ds, bucket)
        rows.append(make_metric_row(run, candidate, bucket, ds.y[mask], pred[mask], probs[mask]))
    return rows


def load_model(run: Any, name: str, device: torch.device) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    ckpt_path = SEARCH_DIR / name / "ckpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt.get("candidate_config", run.CANDIDATES[name]))
    tmp_ds = run.ARCH.SyntheticWaveDataset("test", str(cfg["channels"]))
    model = run.GeometryStudent(cfg, int(tmp_ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def eval_endpoint_on_dataset(run: Any, model: Any, ds: Any, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    payload = run.eval_loader(model, loader, device)
    return payload.probs.astype(np.float32)


def eval_endpoint(run: Any, model: Any, cfg: dict[str, Any], ckpt: dict[str, Any], split: str, device: torch.device, batch_size: int):
    if split in {"train", "val", "test"}:
        ds = run.ARCH.SyntheticWaveDataset(split, str(cfg["channels"]))
    elif split == "original":
        norm = run.ARCH.FeatureNorm(
            mean=np.asarray(ckpt["feature_train_mean"], dtype=np.float32),
            std=np.asarray(ckpt["feature_train_std"], dtype=np.float32),
        )
        ds = run.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    else:
        raise ValueError(split)
    return ds, eval_endpoint_on_dataset(run, model, ds, device, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", type=str, default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261601)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run = load_runner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endpoints = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    endpoints = [x for x in endpoints if (SEARCH_DIR / x / "ckpt_best.pt").exists()]
    if len(endpoints) < 2:
        raise RuntimeError("need at least two available endpoints")

    models = {}
    cfgs = {}
    ckpts = {}
    for name in endpoints:
        models[name], cfgs[name], ckpts[name] = load_model(run, name, device)
        print(f"loaded {name}", flush=True)

    probs_by_split: dict[str, list[np.ndarray]] = {k: [] for k in ["train", "val", "test", "original"]}
    y_by_split: dict[str, np.ndarray] = {}
    original_ds = None
    for name in endpoints:
        for split in ["train", "val", "test", "original"]:
            ds, probs = eval_endpoint(run, models[name], cfgs[name], ckpts[name], split, device, int(args.batch_size))
            probs_by_split[split].append(probs)
            if split not in y_by_split:
                y_by_split[split] = np.asarray(ds.y, dtype=np.int64)
            if split == "original" and original_ds is None:
                original_ds = ds
        print(f"evaluated {name}", flush=True)

    features = {split: endpoint_features(blocks) for split, blocks in probs_by_split.items()}
    prob_mean = {split: np.mean(np.stack(blocks, axis=0), axis=0) for split, blocks in probs_by_split.items()}
    rows = []
    for split in ["val", "test"]:
        pred = prob_mean[split].argmax(axis=1).astype(np.int64)
        rows.append(make_metric_row(run, "prob_mean", f"synthetic_{split}", y_by_split[split], pred, prob_mean[split]))
    assert original_ds is not None
    rows.extend(original_bucket_rows(run, original_ds, "prob_mean", prob_mean["original"].argmax(axis=1).astype(np.int64), prob_mean["original"]))

    train_x, train_y = features["train"], y_by_split["train"]
    val_x, val_y = features["val"], y_by_split["val"]
    test_x, test_y = features["test"], y_by_split["test"]
    orig_x = features["original"]

    routers = {
        "router_logreg_balanced": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            C=0.75,
            multi_class="auto",
            random_state=int(args.seed),
        ),
        "router_hgb_lite": HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=int(args.seed),
        ),
        "router_extratrees_lite": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=7,
            min_samples_leaf=16,
            class_weight="balanced",
            random_state=int(args.seed),
            n_jobs=-1,
        ),
    }
    router_summaries = []
    for name, clf in routers.items():
        clf.fit(train_x, train_y)
        for split, x, y in [("synthetic_val", val_x, val_y), ("synthetic_test", test_x, test_y)]:
            probs = clf.predict_proba(x).astype(np.float32)
            if probs.shape[1] != 3:
                aligned = np.zeros((len(x), 3), dtype=np.float32)
                for j, cls in enumerate(clf.classes_):
                    aligned[:, int(cls)] = probs[:, j]
                probs = aligned
            pred = probs.argmax(axis=1).astype(np.int64)
            row = make_metric_row(run, name, split, y, pred, probs)
            try:
                row["log_loss"] = float(log_loss(y, probs, labels=[0, 1, 2]))
            except Exception:
                row["log_loss"] = np.nan
            rows.append(row)
        probs = clf.predict_proba(orig_x).astype(np.float32)
        if probs.shape[1] != 3:
            aligned = np.zeros((len(orig_x), 3), dtype=np.float32)
            for j, cls in enumerate(clf.classes_):
                aligned[:, int(cls)] = probs[:, j]
            probs = aligned
        pred = probs.argmax(axis=1).astype(np.int64)
        rows.extend(original_bucket_rows(run, original_ds, name, pred, probs))
        router_summaries.append(
            {
                "name": name,
                "classes": [int(c) for c in clf.classes_],
                "train_input": "endpoint probabilities, margins, entropy, endpoint agreement fractions",
            }
        )

    metrics = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "endpoint_router_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "input_contract": "waveform endpoint probabilities only; no SQI/geometry features as inputs",
        "original_used_for_training": False,
        "selection": "synthetic train/val/test only",
        "device": str(device),
        "endpoints": endpoints,
        "routers": router_summaries,
        "metrics_csv": str(metrics_path),
    }
    (OUT_DIR / "endpoint_router_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_lines = [
        "# Waveform Endpoint Router Probe",
        "",
        "Report-only probe. Router training uses PTB synthetic endpoint probabilities only; Original BUT is held out.",
        "",
        "| Candidate | Bucket | Acc | Macro-F1 | Good | Medium | Bad |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    view = metrics[metrics["bucket"].isin(["synthetic_test", "original_test_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"])]
    for _, r in view.iterrows():
        report_lines.append(
            f"| {r['candidate']} | {r['bucket']} | {float(r['acc']):.6f} | {float(r['macro_f1']):.6f} | "
            f"{float(r['good_recall']):.3f} | {float(r['medium_recall']):.3f} | {float(r['bad_recall']):.3f} |"
        )
    report = "\n".join(report_lines) + "\n"
    (OUT_DIR / "endpoint_router_report.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "endpoint_router_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
