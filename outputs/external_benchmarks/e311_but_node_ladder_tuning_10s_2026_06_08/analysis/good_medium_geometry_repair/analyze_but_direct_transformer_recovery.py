"""Audit direct-BUT waveform Transformer predictions and feature recovery.

External-only diagnostic.  This script loads a waveform-only Transformer
checkpoint trained on BUT train/val labels and evaluates held-out BUT test
predictions plus auxiliary SQI/geometry recovery.  It does not train and it
does not modify mainline code.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ORIG_SCRIPT = ANALYSIS_DIR / "run_waveform_transformer_original_adaptation.py"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
KEY_FEATURES = [
    "sqi_basSQI",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "qrs_band_ratio",
    "non_qrs_diff_p95",
    "diff_abs_p95",
    "band_0p3_1",
    "band_15_30",
    "band_30_45",
    "wavelet_e0",
    "wavelet_e4",
    "hjorth_complexity",
]


def load_orig_module() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_transformer_original_adaptation", ORIG_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ORIG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 5:
        return float("nan")
    aa = a[mask].astype(float)
    bb = b[mask].astype(float)
    if float(np.std(aa)) < 1e-8 or float(np.std(bb)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def metric_report(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(y)), "acc": float(np.mean(pred == y)) if len(y) else float("nan")}
    conf = np.zeros((3, 3), dtype=int)
    for t, p in zip(y.astype(int), pred.astype(int)):
        conf[t, p] += 1
    out["confusion_3x3"] = conf.tolist()
    f1s = []
    for cls_name, cls in CLASS_TO_INT.items():
        tp = conf[cls, cls]
        row_sum = conf[cls].sum()
        col_sum = conf[:, cls].sum()
        rec = tp / row_sum if row_sum else float("nan")
        prec = tp / col_sum if col_sum else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0 else 0.0
        out[f"{cls_name}_recall"] = float(rec)
        out[f"{cls_name}_precision"] = float(prec)
        out[f"{cls_name}_f1"] = float(f1)
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    out["good_to_medium"] = int(conf[CLASS_TO_INT["good"], CLASS_TO_INT["medium"]])
    out["medium_to_good"] = int(conf[CLASS_TO_INT["medium"], CLASS_TO_INT["good"]])
    out["bad_to_good"] = int(conf[CLASS_TO_INT["bad"], CLASS_TO_INT["good"]])
    out["bad_to_medium"] = int(conf[CLASS_TO_INT["bad"], CLASS_TO_INT["medium"]])
    return out


@torch.no_grad()
def evaluate(model: torch.nn.Module, ds: Any, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    aux_pred = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x)
        probs.append(F.softmax(out["logits"], dim=1).detach().cpu().numpy())
        aux_pred.append(out["aux_pred"].detach().cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(aux_pred, axis=0)


def render_table(frame: pd.DataFrame, limit: int = 30) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            view[col] = view[col].astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = ANALYSIS_DIR / "but_direct_transformer_recovery"
    report_dir = REPORT_DIR / "but_direct_transformer_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.checkpoint).parent.name

    ORIG = load_orig_module()
    TX = ORIG.TX
    feature_columns = list(ORIG.FEATURE_COLUMNS)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    mean = np.asarray(ckpt.get("feature_train_mean"), dtype=np.float32)
    std = np.asarray(ckpt.get("feature_train_std"), dtype=np.float32)
    norm = ORIG.FeatureNorm(mean=mean, std=std)
    test_ds = ORIG.OriginalSplitDataset("test", str(cfg["channels"]), norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TX.TransformerAuxClassifier(cfg, int(test_ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    probs, aux_pred_norm = evaluate(model, test_ds, int(args.batch_size), device)
    pred = probs.argmax(axis=1).astype(np.int64)
    if float(args.threshold) >= 0:
        pred = pred.copy()
        pred[probs[:, CLASS_TO_INT["bad"]] >= float(args.threshold)] = CLASS_TO_INT["bad"]

    pred_frame = test_ds.frame.copy()
    pred_frame.insert(0, "capacity_bucket", "but_test")
    pred_frame["true_int"] = test_ds.y
    pred_frame["pred_int"] = pred
    pred_frame["true_label"] = [INT_TO_CLASS[int(x)] for x in test_ds.y]
    pred_frame["pred_label"] = [INT_TO_CLASS[int(x)] for x in pred]
    pred_frame["correct"] = pred == test_ds.y
    pred_frame["prob_good"] = probs[:, 0]
    pred_frame["prob_medium"] = probs[:, 1]
    pred_frame["prob_bad"] = probs[:, 2]
    pred_path = out_dir / f"{tag}_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    target_norm = np.asarray(test_ds.aux, dtype=np.float32)
    recovery_rows = []
    for i, feat in enumerate(feature_columns):
        y_true = target_norm[:, i]
        y_hat = aux_pred_norm[:, i]
        recovery_rows.append(
            {
                "feature": feat,
                "corr_norm": corr(y_true, y_hat),
                "mae_norm": float(np.nanmean(np.abs(y_hat - y_true))),
                "target_median_norm": float(np.nanmedian(y_true)),
                "pred_median_norm": float(np.nanmedian(y_hat)),
                "is_key_feature": feat in KEY_FEATURES,
            }
        )
    recovery = pd.DataFrame(recovery_rows).sort_values(["is_key_feature", "corr_norm"], ascending=[False, True])
    rec_path = out_dir / f"{tag}_feature_recovery.csv"
    recovery.to_csv(rec_path, index=False)

    metrics = pd.DataFrame([metric_report(test_ds.y, pred, probs)])
    metrics_path = out_dir / f"{tag}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    key = recovery[recovery["feature"].isin(KEY_FEATURES)].sort_values("corr_norm")
    report = "\n".join(
        [
            f"# BUT Direct Transformer Recovery: {tag}",
            "",
            "Diagnostic only. Model inference uses waveform only; feature columns are used here only as recovery targets.",
            "",
            "## Test Metrics",
            "",
            render_table(metrics),
            "",
            "## Key Feature Recovery",
            "",
            render_table(key, 30),
            "",
            "## Worst Overall Feature Recovery",
            "",
            render_table(recovery.sort_values("corr_norm"), 30),
            "",
            f"Predictions CSV: `{pred_path}`",
            f"Feature recovery CSV: `{rec_path}`",
            f"Metrics CSV: `{metrics_path}`",
            "",
        ]
    )
    report_path = report_dir / f"{tag}_report.md"
    report_path.write_text(report, encoding="utf-8")
    (out_dir / f"{tag}_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "tag": tag,
                "threshold": float(args.threshold),
                "report": str(report_path),
                "predictions_csv": str(pred_path),
                "feature_recovery_csv": str(rec_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
