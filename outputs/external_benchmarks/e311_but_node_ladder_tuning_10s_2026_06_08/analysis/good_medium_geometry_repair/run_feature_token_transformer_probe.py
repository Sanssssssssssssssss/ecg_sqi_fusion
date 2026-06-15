"""Feature-token Transformer probe for current N17043 feature-assisted route.

This is intentionally not a waveform-only model. It tests the user's current
question: if the high-value SQI/geometry features count as information, can a
small Transformer over feature tokens reproduce the 0.90+ held-out BUT result?

Training uses the current BUT/node train split, validation selects checkpoints,
and the BUT test split is held out for reporting. This is a deliberately
feature-assisted split experiment, not the PTB-only waveform student.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
NODE_ID = "N17043_gm_probe"
MANIFEST_PATH = OUT_ROOT / "nodes" / NODE_ID / "node_boundary_manifest.csv"
ORIGINAL_ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
LABELS = [0, 1, 2]

AUTO_TOP20 = [
    "pca_margin",
    "sample_entropy_proxy",
    "flatline_ratio",
    "pc1",
    "higuchi_fd_proxy",
    "non_qrs_diff_p95",
    "diff_zero_crossing_rate",
    "zero_crossing_rate",
    "qrs_prom_p90",
    "sqi_fSQI",
    "amplitude_entropy",
    "sqi_kSQI",
    "sqi_sSQI",
    "low_amp_ratio",
    "non_qrs_rms_ratio",
    "baseline_step",
    "ptp_p99_p01",
    "band_30_45",
    "qrs_visibility",
    "band_15_30",
]
RANKED_TOP22 = AUTO_TOP20 + ["hjorth_mobility", "wavelet_e4"]
TOP14_BALANCED = [
    "pc1",
    "qrs_visibility",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "pca_margin",
    "qrs_band_ratio",
    "template_corr",
    "amplitude_entropy",
    "sqi_bSQI",
    "region_confidence",
    "detector_agreement",
    "mean_abs",
    "sqi_basSQI",
]

CANDIDATES = {
    "featuretx_top14": TOP14_BALANCED,
    "featuretx_top20": AUTO_TOP20,
    "featuretx_top22": RANKED_TOP22,
}


@dataclass
class Norm:
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, split: str, columns: list[str], norm: Norm):
        if split == "all":
            mask = np.ones(len(frame), dtype=bool)
        else:
            mask = frame["split"].astype(str).eq(split).to_numpy()
        self.frame = frame.loc[mask].reset_index(drop=True)
        raw = self.frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, norm.median[None, :]).astype(np.float32)
        self.x = ((raw - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32)
        self.y = self.frame["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": torch.from_numpy(self.x[i]), "y": torch.tensor(int(self.y[i]), dtype=torch.long)}


class FeatureTokenTransformer(nn.Module):
    def __init__(self, n_features: int, width: int = 96, layers: int = 3, heads: int = 4):
        super().__init__()
        self.value_proj = nn.Sequential(nn.Linear(1, width), nn.GELU())
        self.feature_embed = nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(width * 3), nn.Linear(width * 3, 128), nn.GELU(), nn.Dropout(0.08), nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_embed[None, :, :]
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        body = h[:, 1:, :]
        pooled = torch.cat([h[:, 0, :], body.mean(dim=1), body.abs().amax(dim=1)], dim=1)
        return self.head(pooled)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fit_norm(frame: pd.DataFrame, columns: list[str]) -> Norm:
    train = frame[frame["split"].astype(str).eq("train")]
    raw = train[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    median = np.nanmedian(raw, axis=0).astype(np.float32)
    raw = np.where(np.isfinite(raw), raw, median[None, :])
    mean = raw.mean(axis=0).astype(np.float32)
    std = raw.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return Norm(columns=columns, median=median, mean=mean, std=std)


def report_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=LABELS)
    rec = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
    prec = precision_score(y, pred, labels=LABELS, average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "good_recall": float(rec[0]),
        "medium_recall": float(rec[1]),
        "bad_recall": float(rec[2]),
        "good_precision": float(prec[0]),
        "medium_precision": float(prec[1]),
        "bad_precision": float(prec[2]),
        "good_to_medium": int(cm[0, 1]),
        "medium_to_good": int(cm[1, 0]),
        "bad_to_medium": int(cm[2, 1]),
        "confusion_3x3": cm.tolist(),
    }


def metric_row(candidate: str, bucket: str, y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    return {"candidate": candidate, "bucket": bucket, **report_metrics(y, prob)}


def bucket_rows(ds: FeatureDataset, bucket: str) -> np.ndarray:
    frame = ds.frame
    y = ds.y
    if bucket == "original_test_all_10s+":
        return frame["split"].astype(str).eq("test").to_numpy()
    if bucket == "original_all_10s+":
        return np.ones(len(frame), dtype=bool)
    region = frame.get("original_region", pd.Series("", index=frame.index)).astype(str).to_numpy()
    bad = y == CLASS_TO_INT["bad"]
    outlier = np.char.lower(region.astype(str)) == "outlier_low_confidence"
    if bucket == "bad_core_nearboundary":
        return bad & (~outlier)
    if bucket == "bad_outlier_stress":
        return bad & outlier
    raise ValueError(bucket)


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        ys.append(batch["y"].numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_one(name: str, columns: list[str], manifest: pd.DataFrame, original: pd.DataFrame, epochs: int = 24) -> dict[str, Any]:
    torch.manual_seed(20260940 + len(columns))
    np.random.seed(20260940 + len(columns))
    norm = fit_norm(manifest, columns)
    train_ds = FeatureDataset(manifest, "train", columns, norm)
    val_ds = FeatureDataset(manifest, "val", columns, norm)
    test_ds = FeatureDataset(manifest, "test", columns, norm)
    original_ds = FeatureDataset(original, "all", columns, norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)
    original_loader = DataLoader(original_ds, batch_size=512)
    model = FeatureTokenTransformer(len(columns), width=96 if len(columns) <= 20 else 112).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    weight = torch.tensor([1.00, 1.36, 2.45], dtype=torch.float32, device=device)
    best_state = None
    best_score = -1e9
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            loss = F.cross_entropy(model(x), y, weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        vy, vp = eval_model(model, val_loader, device)
        rep = report_metrics(vy, vp)
        score = rep["acc"] + 0.20 * rep["macro_f1"] + 0.15 * min(rep["good_recall"], rep["medium_recall"], rep["bad_recall"])
        logs.append({"epoch": epoch, "loss": float(np.mean(losses)), **rep})
        print(f"{name} epoch={epoch}/{epochs} val_acc={rep['acc']:.4f} recalls={rep['good_recall']:.3f}/{rep['medium_recall']:.3f}/{rep['bad_recall']:.3f}", flush=True)
        if score > best_score:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    ty, tp = eval_model(model, test_loader, device)
    oy, op = eval_model(model, original_loader, device)
    rows = [metric_row(name, "node_test", ty, tp)]
    for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
        mask = bucket_rows(original_ds, bucket)
        rows.append(metric_row(name, bucket, oy[mask], op[mask]))
    run_dir = OUT_ROOT / "runs" / "feature_token_transformer" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_columns": columns,
            "norm_mean": norm.mean,
            "norm_std": norm.std,
            "norm_median": norm.median,
            "input_contract": "feature-assisted: selected SQI/geometry columns are inference inputs",
            "original_rows_used_for_training": True,
            "but_usage": "train/val/test split; test held out",
        },
        run_dir / "ckpt_best.pt",
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "original_probs.npz", probs=op)
    return {"candidate": name, "feature_columns": columns, "metrics": rows, "checkpoint": str(run_dir / "ckpt_best.pt")}


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.6f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    original = pd.read_csv(ORIGINAL_ATLAS)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, cols in CANDIDATES.items():
        missing = sorted(set(cols) - set(manifest.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        summary = train_one(name, cols, manifest, original)
        summaries.append({k: v for k, v in summary.items() if k != "metrics"})
        all_rows.extend(summary["metrics"])
    metrics = pd.DataFrame(all_rows)
    metrics_path = ANALYSIS_DIR / "feature_token_transformer_probe_metrics.csv"
    report_path = ANALYSIS_DIR / "feature_token_transformer_probe_report.md"
    summary_path = ANALYSIS_DIR / "feature_token_transformer_probe_summary.json"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "node_id": NODE_ID,
        "feature_assisted_not_waveform_only": True,
        "original_used_for_training": True,
        "training_rows_source": "current node_boundary_manifest train split; same split/counts as original_region_atlas",
        "metrics_csv": str(metrics_path),
        "candidates": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    focus = metrics[metrics["bucket"].isin(["original_test_all_10s+", "original_all_10s+", "node_test"])]
    report = "\n".join(
        [
            "# Feature-Token Transformer Probe",
            "",
            "This is a feature-assisted Transformer over selected SQI/geometry columns. It is not waveform-only.",
            "Selection uses the current BUT/node train/val split; the BUT test split is held out for reporting.",
            "",
            "## Main Metrics",
            "",
            markdown_table(focus[["candidate", "bucket", "n", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall", "good_to_medium", "medium_to_good", "bad_to_medium"]]),
            "",
            "## All Buckets",
            "",
            markdown_table(metrics[["candidate", "bucket", "n", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall", "good_to_medium", "medium_to_good", "bad_to_medium"]]),
            "",
            f"Metrics CSV: `{metrics_path}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    (REPORT_DIR / report_path.name).write_text(report, encoding="utf-8")
    (REPORT_DIR / summary_path.name).write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    (REPORT_DIR / metrics_path.name).write_text(metrics.to_csv(index=False), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
