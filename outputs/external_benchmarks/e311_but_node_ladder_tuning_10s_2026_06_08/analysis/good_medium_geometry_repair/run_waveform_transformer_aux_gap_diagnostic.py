"""Diagnose why waveform Transformers lag tabular SQI/geometry MLPs.

The key question is whether the best waveform Transformer actually learns the
feature geometry that the tabular MLP uses.  This script compares the
Transformer auxiliary predictions against the true 47-column SQI/geometry
features on original train/val/test splits, then merges that with the reduced
feature ranking.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ORIG_SCRIPT = ANALYSIS_DIR / "run_waveform_transformer_original_adaptation.py"
RANKING_PATH = ANALYSIS_DIR / "current_reduced_feature_ranking.csv"
CHECKPOINT = OUT_ROOT / "runs" / "waveform_transformer_augmented_original" / "N17043_gm_probe" / "aug_convtx_balanced_focal" / "ckpt_best.pt"


def load_orig_module() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_transformer_original_adaptation", ORIG_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ORIG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORIG = load_orig_module()
TX = ORIG.TX
GEOM = ORIG.GEOM
FEATURE_COLUMNS = list(ORIG.FEATURE_COLUMNS)
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    train_ds = ORIG.OriginalSplitDataset("train", str(cfg["channels"]))
    model = TX.TransformerAuxClassifier(cfg, int(train_ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    split_rows = []
    all_feature_rows = []
    per_row_frames = []
    for split in ["train", "val", "test"]:
        ds = ORIG.OriginalSplitDataset(split, str(cfg["channels"]), train_ds.norm)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        out = collect(model, loader, device)
        y = ds.y
        probs = out["probs"]
        pred = probs.argmax(axis=1)
        rep = GEOM.metric_report(y, pred, probs)
        split_rows.append({"split": split, **{k: v for k, v in rep.items() if k != "confusion_3x3"}})
        true_aux = out["aux_true"]
        pred_aux = out["aux_pred"]
        for i, feature in enumerate(FEATURE_COLUMNS):
            t = true_aux[:, i]
            p = pred_aux[:, i]
            row = {
                "split": split,
                "feature": feature,
                "mae_z": float(np.mean(np.abs(p - t))),
                "rmse_z": float(np.sqrt(np.mean((p - t) ** 2))),
                "corr": safe_corr(t, p),
                "r2": safe_r2(t, p),
                "true_gm_auc": binary_auc(t[(y == 0) | (y == 1)], (y[(y == 0) | (y == 1)] == 1).astype(int)),
                "pred_gm_auc": binary_auc(p[(y == 0) | (y == 1)], (y[(y == 0) | (y == 1)] == 1).astype(int)),
                "true_bad_auc": binary_auc(t, (y == 2).astype(int)),
                "pred_bad_auc": binary_auc(p, (y == 2).astype(int)),
            }
            all_feature_rows.append(row)
        row_frame = pd.DataFrame({
            "split": split,
            "class_id": y,
            "pred_id": pred,
            "p_good": probs[:, 0],
            "p_medium": probs[:, 1],
            "p_bad": probs[:, 2],
            "original_region": ds.region,
        })
        for i, feature in enumerate(FEATURE_COLUMNS):
            row_frame[f"true_{feature}"] = true_aux[:, i]
            row_frame[f"pred_{feature}"] = pred_aux[:, i]
        per_row_frames.append(row_frame)

    feature_df = pd.DataFrame(all_feature_rows)
    ranking = pd.read_csv(RANKING_PATH) if RANKING_PATH.exists() else pd.DataFrame({"feature": FEATURE_COLUMNS})
    test_features = feature_df[feature_df["split"].eq("test")].merge(ranking, on="feature", how="left")
    test_features["gm_auc_loss"] = test_features["true_gm_auc"] - test_features["pred_gm_auc"]
    test_features["bad_auc_loss"] = test_features["true_bad_auc"] - test_features["pred_bad_auc"]
    test_features["importance_badness"] = test_features.get("ranking_score", 0).fillna(0) * (1.0 - test_features["corr"].fillna(0).clip(lower=-1, upper=1))
    top_gap = test_features.sort_values(["importance_badness", "mae_z"], ascending=[False, False]).head(25)

    per_row = pd.concat(per_row_frames, ignore_index=True)
    test_rows = per_row[per_row["split"].eq("test")].copy()
    error_summary = summarize_errors(test_rows, top_gap["feature"].head(12).tolist())

    feature_path = ANALYSIS_DIR / "waveform_transformer_aux_feature_gap.csv"
    split_path = ANALYSIS_DIR / "waveform_transformer_aux_split_metrics.csv"
    error_path = ANALYSIS_DIR / "waveform_transformer_aux_error_feature_gap.csv"
    report_path = REPORT_DIR / "waveform_transformer_aux_gap_diagnostic_report.md"
    summary_path = ANALYSIS_DIR / "waveform_transformer_aux_gap_diagnostic_summary.json"
    feature_df.merge(ranking, on="feature", how="left").to_csv(feature_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    error_summary.to_csv(error_path, index=False)
    payload = {
        "created_at": now(),
        "checkpoint": str(CHECKPOINT),
        "feature_gap_csv": str(feature_path),
        "split_metrics_csv": str(split_path),
        "error_feature_gap_csv": str(error_path),
        "top_gap": top_gap.to_dict(orient="records"),
        "split_metrics": split_rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(payload, top_gap, error_summary, pd.DataFrame(split_rows))
    report_path.write_text(report, encoding="utf-8")
    (ANALYSIS_DIR / report_path.name).write_text(report, encoding="utf-8")
    print(report)


@torch.no_grad()
def collect(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    probs = []
    aux_true = []
    aux_pred = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        aux = batch["aux"].to(device=device, dtype=torch.float32)
        out = model(x)
        probs.append(F.softmax(out["logits"], dim=1).detach().cpu().numpy())
        aux_true.append(aux.detach().cpu().numpy())
        aux_pred.append(out["aux_pred"].detach().cpu().numpy())
    return {
        "probs": np.concatenate(probs),
        "aux_true": np.concatenate(aux_true),
        "aux_pred": np.concatenate(aux_pred),
    }


def summarize_errors(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    masks = {
        "correct_good": (frame["class_id"] == 0) & (frame["pred_id"] == 0),
        "good_to_medium": (frame["class_id"] == 0) & (frame["pred_id"] == 1),
        "correct_medium": (frame["class_id"] == 1) & (frame["pred_id"] == 1),
        "medium_to_good": (frame["class_id"] == 1) & (frame["pred_id"] == 0),
        "correct_bad": (frame["class_id"] == 2) & (frame["pred_id"] == 2),
        "bad_to_medium": (frame["class_id"] == 2) & (frame["pred_id"] == 1),
    }
    rows = []
    for name, mask in masks.items():
        sub = frame[mask]
        for feature in features:
            true_col = f"true_{feature}"
            pred_col = f"pred_{feature}"
            if len(sub) == 0 or true_col not in sub:
                continue
            rows.append({
                "group": name,
                "n": int(len(sub)),
                "feature": feature,
                "true_median_z": float(np.nanmedian(sub[true_col])),
                "pred_median_z": float(np.nanmedian(sub[pred_col])),
                "median_abs_error_z": float(np.nanmedian(np.abs(sub[pred_col] - sub[true_col]))),
            })
    return pd.DataFrame(rows)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) < 1e-8 or np.nanstd(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sum((a - np.mean(a)) ** 2))
    if denom <= 1e-8:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / denom)


def binary_auc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    pos = score[label == 1]
    neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average tied ranks.
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_ranks = ranks[label == 1]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(max(auc, 1.0 - auc))


def render_report(payload: dict[str, Any], top_gap: pd.DataFrame, error_summary: pd.DataFrame, split_metrics: pd.DataFrame) -> str:
    lines = [
        "# Waveform Transformer Aux Gap Diagnostic",
        "",
        "Question: does the waveform Transformer learn the SQI/geometry quantities that make the tabular MLP strong?",
        "",
        "## Split Metrics",
        "",
        "| split | acc | macro-F1 | good R | medium R | bad R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in split_metrics.iterrows():
        lines.append(
            f"| {row['split']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} |"
        )
    lines.extend([
        "",
        "## Highest-Impact Missing Auxiliary Features On Test",
        "",
        "| feature | ranking score | MAE z | corr | true GM AUC | pred GM AUC | true bad AUC | pred bad AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in top_gap.head(16).iterrows():
        lines.append(
            f"| {row['feature']} | {float(row.get('ranking_score', 0) if pd.notna(row.get('ranking_score', np.nan)) else 0):.3f} | "
            f"{float(row['mae_z']):.3f} | {float(row['corr']) if pd.notna(row['corr']) else float('nan'):.3f} | "
            f"{float(row['true_gm_auc']):.3f} | {float(row['pred_gm_auc']):.3f} | "
            f"{float(row['true_bad_auc']):.3f} | {float(row['pred_bad_auc']):.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If a feature has high tabular ranking but low aux correlation/high MAE, the waveform encoder is not encoding the quantity the MLP uses.",
        "- If true AUC is strong but predicted-feature AUC is weak, the feature is learnable/useful in tabular form but not recovered by the current Transformer representation.",
        "- This separates architecture/representation failure from simple class-weight or threshold failure.",
        "",
        f"Feature gap CSV: `{payload['feature_gap_csv']}`",
        f"Error feature CSV: `{payload['error_feature_gap_csv']}`",
    ])
    return "\n".join(lines)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    main()
