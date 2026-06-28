"""Waveform-only architecture search for N17043 boundary learning.

This experiment deliberately avoids feeding SQI/geometry columns into the
classifier.  The model input is ECG waveform channels only.  The 47-column
SQI/geometry table is used only as an auxiliary regression target on synthetic
train rows, so a good result must come from waveform representation learning.

Selection uses synthetic train/val only.  Original BUT buckets are report-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


FEATURE_SCRIPT = SCRIPT_DIR / "waveform_feature_extractor.py"
NODE_ID = "N17043_gm_probe"
VARIANT_ID = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
DATA_DIR = OUT_ROOT / "synthetic_variants" / VARIANT_ID / "datasets"
SIGNALS_PATH = OUT_ROOT / "source" / "p1_current_10s_center" / "signals.npz"
ORIGINAL_ATLAS = ANALYSIS_DIR / "original_region_boundary" / "original_region_atlas.csv"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
LABELS = [0, 1, 2]
REFERENCE_47 = {
    "name": "47-feature tabular current best",
    "original_test_acc": 0.963548425150407,
    "original_test_macro_f1": 0.9306830954417679,
    "original_test_good_recall": 0.9563186813186814,
    "original_test_medium_recall": 0.9728874830546769,
    "original_test_bad_recall": 0.927007299270073,
}

CORE_AUX_COLUMNS = [
    "pc1",
    "pca_margin",
    "qrs_visibility",
    "qrs_prom_p90",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "amplitude_entropy",
    "low_amp_ratio",
    "sqi_bSQI",
    "sqi_sSQI",
    "sqi_kSQI",
]

CANDIDATES = {
    "resnet_robust3_aux": {
        "arch": "resnet",
        "channels": "robust3",
        "width": 64,
        "aux_weight": 0.45,
        "core_aux_weight": 0.35,
        "class_weight": [1.05, 1.25, 1.90],
        "lr": 8e-4,
        "seed": 20260651,
    },
    "tcn_robust3_aux": {
        "arch": "tcn",
        "channels": "robust3",
        "width": 64,
        "aux_weight": 0.45,
        "core_aux_weight": 0.35,
        "class_weight": [1.05, 1.25, 1.90],
        "lr": 8e-4,
        "seed": 20260652,
    },
    "patchtx_robust3_aux": {
        "arch": "patchtx",
        "channels": "robust3",
        "width": 96,
        "aux_weight": 0.45,
        "core_aux_weight": 0.35,
        "class_weight": [1.05, 1.25, 1.90],
        "lr": 5e-4,
        "seed": 20260653,
    },
    "resnet_raw1_aux": {
        "arch": "resnet",
        "channels": "raw1",
        "width": 64,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight": [1.05, 1.25, 1.90],
        "lr": 8e-4,
        "seed": 20260654,
    },
    "resnet_robust3_badstrong_aux": {
        "arch": "resnet",
        "channels": "robust3",
        "width": 64,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight": [1.00, 1.20, 4.00],
        "lr": 7e-4,
        "seed": 20260655,
    },
    "resnet_robust3_focal_bad": {
        "arch": "resnet",
        "channels": "robust3",
        "width": 64,
        "aux_weight": 0.25,
        "core_aux_weight": 0.20,
        "class_weight": [1.00, 1.20, 3.50],
        "focal_gamma": 1.5,
        "lr": 7e-4,
        "seed": 20260656,
    },
    "resnet_robust3_wide96_aux": {
        "arch": "resnet",
        "channels": "robust3",
        "width": 96,
        "aux_weight": 0.45,
        "core_aux_weight": 0.35,
        "class_weight": [1.05, 1.25, 2.20],
        "lr": 5e-4,
        "seed": 20260657,
    },
}


def load_feature_module() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_feature_extractor", FEATURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FEATURE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FEATURES = load_feature_module()
FEATURE_COLUMNS = list(FEATURES.FEATURE_COLUMNS)
CORE_AUX_IDX = [FEATURE_COLUMNS.index(c) for c in CORE_AUX_COLUMNS if c in FEATURE_COLUMNS]


@dataclass
class FeatureNorm:
    mean: np.ndarray
    std: np.ndarray


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349
    std = np.std(x, axis=1, keepdims=True)
    scale = np.where(scale > 1e-5, scale, std)
    scale = np.where(scale > 1e-5, scale, 1.0)
    return ((x - med) / scale).astype(np.float32)


def make_channels(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    if mode == "raw1":
        return x[:, None, :].astype(np.float32)
    z = robust_z(x)
    if mode == "robust1":
        return z[:, None, :]
    dz = np.diff(z, axis=1, prepend=z[:, :1])
    win = 63
    kernel = np.ones(win, dtype=np.float32) / float(win)
    baseline = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, z)
    if mode == "robust3":
        return np.stack([z, dz.astype(np.float32), baseline.astype(np.float32)], axis=1)
    raise ValueError(f"unknown channel mode {mode}")


class SyntheticWaveDataset(Dataset):
    def __init__(self, split: str, channel_mode: str):
        labels = pd.read_csv(DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
        mask = labels["split"].astype(str).eq(split).to_numpy()
        self.frame = labels.loc[mask].reset_index(drop=True)
        rows = self.frame["idx"].to_numpy(dtype=np.int64)
        noisy = np.load(DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)[rows]
        feat_npz = np.load(DATA_DIR / "tabular_features.npz")
        features = feat_npz["features"].astype(np.float32)
        self.mean = feat_npz["train_mean"].astype(np.float32)
        self.std = np.where(feat_npz["train_std"].astype(np.float32) > 1e-6, feat_npz["train_std"].astype(np.float32), 1.0)
        self.x = make_channels(noisy, channel_mode)
        self.aux = ((features[rows] - self.mean) / self.std).astype(np.float32)
        self.y = self.frame["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class OriginalWaveDataset(Dataset):
    def __init__(self, norm: FeatureNorm, channel_mode: str):
        atlas = pd.read_csv(ORIGINAL_ATLAS)
        for col in FEATURE_COLUMNS:
            if col not in atlas.columns:
                atlas[col] = 0.0
        signals = np.load(SIGNALS_PATH)["X"].astype(np.float32)
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        rows = atlas["idx"].astype(int).to_numpy()
        self.x = make_channels(signals[rows], channel_mode)
        raw = atlas[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        self.aux = ((raw - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32)
        self.y = atlas["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.split = atlas["split"].astype(str).to_numpy()
        self.region = atlas.get("original_region", pd.Series("", index=atlas.index)).astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1, dropout: float = 0.08):
        super().__init__()
        pad = dilation * 3
        groups = max(1, min(8, out_ch // 8))
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=pad, dilation=dilation),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=pad, dilation=dilation),
            nn.GroupNorm(groups, out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + self.skip(x))


class WaveResNet(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, kernel_size=11, padding=5), nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.Sequential(
            ResBlock(width, width, 1),
            ResBlock(width, width * 2, 2),
            ResBlock(width * 2, width * 2, 1, dilation=2),
            ResBlock(width * 2, width * 3, 2),
            ResBlock(width * 3, width * 3, 1, dilation=4),
            ResBlock(width * 3, width * 4, 2),
            ResBlock(width * 4, width * 4, 1, dilation=8),
        )
        self.out_dim = width * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        avg = F.adaptive_avg_pool1d(h, 1).flatten(1)
        mx = F.adaptive_max_pool1d(h.abs(), 1).flatten(1)
        return torch.cat([avg, mx], dim=1)


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.08):
        super().__init__()
        pad = dilation * 4
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=9, padding=pad, dilation=dilation),
            nn.GroupNorm(max(1, min(8, ch // 8)), ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel_size=9, padding=pad, dilation=dilation),
            nn.GroupNorm(max(1, min(8, ch // 8)), ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class WaveTCN(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, kernel_size=13, padding=6), nn.GroupNorm(8, width), nn.GELU())
        blocks = []
        for dilation in [1, 2, 4, 8, 16, 32, 1, 2, 4, 8]:
            blocks.append(TCNBlock(width, dilation))
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(width, width * 2, kernel_size=1)
        self.out_dim = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.blocks(self.stem(x)))
        avg = F.adaptive_avg_pool1d(h, 1).flatten(1)
        mx = F.adaptive_max_pool1d(h.abs(), 1).flatten(1)
        return torch.cat([avg, mx], dim=1)


class PatchTransformer(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 3,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.out_dim = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x).transpose(1, 2)
        h = self.encoder(h)
        avg = h.mean(dim=1)
        mx = h.abs().amax(dim=1)
        return torch.cat([avg, mx], dim=1)


class WaveAuxClassifier(nn.Module):
    def __init__(self, arch: str, in_ch: int, width: int, aux_dim: int):
        super().__init__()
        if arch == "resnet":
            self.encoder = WaveResNet(in_ch, width)
        elif arch == "tcn":
            self.encoder = WaveTCN(in_ch, width)
        elif arch == "patchtx":
            self.encoder = PatchTransformer(in_ch, width)
        else:
            raise ValueError(f"unknown arch {arch}")
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Linear(self.encoder.out_dim, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 3),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Linear(self.encoder.out_dim, 160),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(160, aux_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)
        return {"logits": self.class_head(feat), "aux_pred": self.aux_head(feat), "features": feat}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}; expected {sorted(CANDIDATES)}")
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for name in names:
        summary = train_candidate(name, CANDIDATES[name], args)
        summaries.append(summary)
        all_rows.extend(summary["metric_rows"])
    metrics = pd.DataFrame(all_rows)
    metrics_path = ANALYSIS_DIR / "waveform_architecture_search_metrics.csv"
    summary_path = ANALYSIS_DIR / "waveform_architecture_search_summary.json"
    report_path = ANALYSIS_DIR / "waveform_architecture_search_report.md"
    report_copy = REPORT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "variant_id": VARIANT_ID,
        "input_contract": "waveform channels only; 47 SQI/geometry columns are auxiliary training targets, not inputs",
        "original_used_for_training": False,
        "reference_47": REFERENCE_47,
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


def train_candidate(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    print(f"training {name}: {cfg}", flush=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SyntheticWaveDataset("train", str(cfg["channels"]))
    val_ds = SyntheticWaveDataset("val", str(cfg["channels"]))
    test_ds = SyntheticWaveDataset("test", str(cfg["channels"]))
    norm = FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = OriginalWaveDataset(norm, str(cfg["channels"]))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    original_loader = DataLoader(original_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    in_ch = int(train_ds.x.shape[1])
    model = WaveAuxClassifier(str(cfg["arch"]), in_ch, int(cfg["width"]), len(FEATURE_COLUMNS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    class_weight = torch.tensor(cfg["class_weight"], dtype=torch.float32, device=device)
    best_state = None
    best_epoch = 0
    best_score = -1e9
    logs = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(x)
            cls_loss = classification_loss(out["logits"], y, class_weight, float(cfg.get("focal_gamma", 0.0)))
            aux_loss = F.smooth_l1_loss(out["aux_pred"], aux)
            if CORE_AUX_IDX:
                core_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_AUX_IDX], aux[:, CORE_AUX_IDX])
            else:
                core_loss = torch.zeros((), device=device)
            loss = cls_loss + float(cfg["aux_weight"]) * aux_loss + float(cfg["core_aux_weight"]) * core_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, _ = eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.2 * val_rep["macro_f1"] + 0.15 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"]) - 0.02 * val_rep["aux_mae"]
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"{name} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} aux_mae={val_rep['aux_mae']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_rep, val_probs = eval_loader(model, val_loader, device)
    test_rep, test_probs = eval_loader(model, test_loader, device)
    orig_rep, orig_probs = eval_loader(model, original_loader, device)
    threshold = calibrate_bad_threshold(val_ds.y, val_probs)
    metric_rows = [
        metric_row(name, "synthetic_val", val_rep),
        metric_row(name, "synthetic_test", test_rep),
        metric_row(f"{name}_badcal", "synthetic_test", FEATURES.metric_report(test_ds.y, apply_bad_threshold(test_probs, threshold), test_probs)),
        metric_row(name, "original_test_all_10s+", bucket_report(original_ds, orig_probs, "original_test_all_10s+")),
        metric_row(f"{name}_badcal", "original_test_all_10s+", bucket_report(original_ds, orig_probs, "original_test_all_10s+", threshold)),
        metric_row(name, "original_all_10s+", bucket_report(original_ds, orig_probs, "original_all_10s+")),
        metric_row(f"{name}_badcal", "original_all_10s+", bucket_report(original_ds, orig_probs, "original_all_10s+", threshold)),
        metric_row(name, "bad_core_nearboundary", bucket_report(original_ds, orig_probs, "bad_core_nearboundary")),
        metric_row(f"{name}_badcal", "bad_core_nearboundary", bucket_report(original_ds, orig_probs, "bad_core_nearboundary", threshold)),
        metric_row(name, "bad_outlier_stress", bucket_report(original_ds, orig_probs, "bad_outlier_stress")),
        metric_row(f"{name}_badcal", "bad_outlier_stress", bucket_report(original_ds, orig_probs, "bad_outlier_stress", threshold)),
    ]
    run_dir = OUT_ROOT / "runs" / "waveform_architecture_search" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "waveform_only_architecture_search_aux_sqi_geometry",
            "input_contract": "waveform channels only; feature targets are auxiliary labels",
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": norm.mean,
            "feature_train_std": norm.std,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
            "original_rows_used_for_training": False,
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "synthetic_test_probs.npz", probs=test_probs)
    np.savez_compressed(run_dir / "original_probs.npz", probs=orig_probs)
    return {
        "candidate": name,
        "config": cfg,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "bad_threshold_trainval": threshold,
        "synthetic_test": test_rep,
        "original_test_all_10s+": bucket_report(original_ds, orig_probs, "original_test_all_10s+", threshold),
        "metric_rows": metric_rows,
    }


def classification_loss(logits: torch.Tensor, y: torch.Tensor, class_weight: torch.Tensor, focal_gamma: float) -> torch.Tensor:
    if focal_gamma <= 0.0:
        return F.cross_entropy(logits, y, weight=class_weight)
    ce = F.cross_entropy(logits, y, weight=class_weight, reduction="none")
    pt = torch.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1).clamp(1e-4, 1.0)
    return (((1.0 - pt) ** float(focal_gamma)) * ce).mean()


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    ys = []
    probs = []
    aux_abs = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        aux = batch["aux"].to(device=device, dtype=torch.float32)
        out = model(x)
        probs.append(F.softmax(out["logits"], dim=1).detach().cpu().numpy())
        aux_abs.append(torch.mean(torch.abs(out["aux_pred"] - aux), dim=0).detach().cpu().numpy())
        ys.append(batch["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    rep = FEATURES.metric_report(y, p.argmax(axis=1), p)
    rep["aux_mae"] = float(np.mean(np.stack(aux_abs)))
    rep["core_aux_mae"] = float(np.mean(np.stack(aux_abs)[:, CORE_AUX_IDX])) if CORE_AUX_IDX else rep["aux_mae"]
    return rep, p


def calibrate_bad_threshold(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    best: tuple[tuple[float, float, float], float] | None = None
    for threshold in np.round(np.linspace(0.01, 0.95, 95), 2):
        pred = apply_bad_threshold(probs_val, float(threshold))
        rep = FEATURES.metric_report(y_val, pred, probs_val)
        score = (rep["acc"], rep["macro_f1"], min(rep["good_recall"], rep["medium_recall"], rep["bad_recall"]))
        if best is None or score > best[0]:
            best = (score, float(threshold))
    assert best is not None
    return best[1]


def apply_bad_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    pred = probs.argmax(axis=1).astype(np.int64)
    pred[probs[:, CLASS_TO_INT["bad"]] >= float(threshold)] = CLASS_TO_INT["bad"]
    return pred


def bucket_mask(ds: OriginalWaveDataset, bucket: str) -> np.ndarray:
    if bucket == "original_all_10s+":
        return np.ones(len(ds.y), dtype=bool)
    if bucket == "original_test_all_10s+":
        return ds.split == "test"
    if bucket == "original_test_main_without_bad_stress":
        return (ds.split == "test") & ~((ds.y == CLASS_TO_INT["bad"]) & (ds.region == "outlier_low_confidence"))
    if bucket == "bad_core_nearboundary":
        return (ds.split == "test") & (ds.y == CLASS_TO_INT["bad"]) & (ds.region != "outlier_low_confidence")
    if bucket == "bad_outlier_stress":
        return (ds.split == "test") & (ds.y == CLASS_TO_INT["bad"]) & (ds.region == "outlier_low_confidence")
    raise ValueError(f"unknown bucket {bucket}")


def bucket_report(ds: OriginalWaveDataset, probs: np.ndarray, bucket: str, threshold: float | None = None) -> dict[str, Any]:
    mask = bucket_mask(ds, bucket)
    pred = probs.argmax(axis=1).astype(np.int64) if threshold is None else apply_bad_threshold(probs, threshold)
    return FEATURES.metric_report(ds.y[mask], pred[mask], probs[mask])


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only Architecture Search",
        "",
        "Classifier input is waveform channels only. The 47 SQI/geometry columns supervise an auxiliary head during training but are not passed into the classifier.",
        "",
        "## Key Metrics",
        "",
        "| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    view = metrics[metrics["bucket"].isin(["synthetic_test", "original_test_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"])].copy()
    for _, row in view.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} | {float(row.get('aux_mae', 0.0)):.4f} |"
        )
    ref = payload["reference_47"]
    lines.extend(
        [
            "",
            "## Reference",
            "",
            f"- 47-feature tabular current best original test acc: `{ref['original_test_acc']:.6f}`; recalls good/medium/bad `{ref['original_test_good_recall']:.3f}/{ref['original_test_medium_recall']:.3f}/{ref['original_test_bad_recall']:.3f}`.",
            "",
            "## Candidate Configs",
            "",
        ]
    )
    for item in payload["summaries"]:
        lines.append(
            f"- `{item['candidate']}`: best_epoch={item['best_epoch']}, threshold={item['bad_threshold_trainval']:.2f}, checkpoint=`{item['checkpoint']}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Original BUT rows are report-only and are never used for training or model selection.",
            "- This search tests whether architecture + waveform transforms can close the gap without tabular inputs.",
            f"- Metrics CSV: `{payload['metrics_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


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
