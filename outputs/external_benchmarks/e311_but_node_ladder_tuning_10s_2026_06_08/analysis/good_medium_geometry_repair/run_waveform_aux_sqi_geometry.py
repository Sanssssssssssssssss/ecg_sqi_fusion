"""Waveform-only UFormer with auxiliary SQI/geometry prediction.

The 47-column SQI/geometry model is a strong teacher signal, but this
experiment does not feed those columns into the classifier.  The model sees
only the ECG waveform, predicts the class, and is additionally supervised to
predict the normalized 47 SQI/geometry targets from its UFormer feature vector.

Selection uses synthetic/node train/val only.  Original BUT buckets are
report-only diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
from torch.utils.data import DataLoader, Dataset

from src.transformer_pipeline.models.uformer1d import Uformer1DConfig, UformerDenoiseSQIModel


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
GEOM_SCRIPT = ANALYSIS_DIR / "uformer_geometry_branch_experiment.py"
NODE_ID = "N17043_gm_probe"
VARIANT_ID = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
DATA_DIR = OUT_ROOT / "synthetic_variants" / VARIANT_ID / "datasets"
BASE_CKPT = OUT_ROOT / "runs" / "quick" / VARIANT_ID / "stage1_denoiser_pretrain" / "ckpt_best_stage1_denoiser.pt"
SIGNALS_PATH = ROOT / "outputs" / "external_benchmarks" / "e311_but_protocol_adaptation_2026_06_03" / "protocols" / "p1_current_10s_center" / "signals.npz"
ORIGINAL_ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"

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
    "wave_aux_headonly_a025": {
        "class_weight": [1.05, 1.25, 1.85],
        "aux_weight": 0.25,
        "core_aux_weight": 0.25,
        "denoise_weight": 0.0,
        "unfreeze": "none",
        "lr_head": 7e-4,
        "lr_backbone": 0.0,
        "seed": 20260641,
    },
    "wave_aux_headonly_a075": {
        "class_weight": [1.05, 1.25, 1.85],
        "aux_weight": 0.75,
        "core_aux_weight": 0.35,
        "denoise_weight": 0.0,
        "unfreeze": "none",
        "lr_head": 7e-4,
        "lr_backbone": 0.0,
        "seed": 20260642,
    },
    "wave_aux_bottleneck_lowlr_a050": {
        "class_weight": [1.05, 1.25, 1.85],
        "aux_weight": 0.50,
        "core_aux_weight": 0.30,
        "denoise_weight": 0.02,
        "unfreeze": "bottleneck",
        "lr_head": 6e-4,
        "lr_backbone": 2e-5,
        "seed": 20260643,
    },
}


def load_geom_module() -> Any:
    spec = importlib.util.spec_from_file_location("uformer_geometry_branch_experiment", GEOM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GEOM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEOM = load_geom_module()
FEATURE_COLUMNS = list(GEOM.FEATURE_COLUMNS)
CORE_AUX_IDX = [FEATURE_COLUMNS.index(c) for c in CORE_AUX_COLUMNS if c in FEATURE_COLUMNS]


@dataclass
class FeatureNorm:
    mean: np.ndarray
    std: np.ndarray


class SyntheticAuxDataset(Dataset):
    def __init__(self, split: str):
        labels = pd.read_csv(DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
        mask = labels["split"].astype(str).eq(split).to_numpy()
        self.frame = labels.loc[mask].reset_index(drop=True)
        rows = self.frame["idx"].to_numpy(dtype=np.int64)
        noisy = np.load(DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        clean = np.load(DATA_DIR / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
        feat_npz = np.load(DATA_DIR / "tabular_features.npz")
        features = feat_npz["features"].astype(np.float32)
        self.mean = feat_npz["train_mean"].astype(np.float32)
        self.std = np.where(feat_npz["train_std"].astype(np.float32) > 1e-6, feat_npz["train_std"].astype(np.float32), 1.0)
        self.noisy = noisy[rows]
        self.clean = clean[rows]
        self.aux = ((features[rows] - self.mean) / self.std).astype(np.float32)
        self.y = self.frame["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "noisy": torch.from_numpy(self.noisy[i]).unsqueeze(0),
            "clean": torch.from_numpy(self.clean[i]).unsqueeze(0),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class OriginalAuxDataset(Dataset):
    def __init__(self, norm: FeatureNorm):
        atlas = pd.read_csv(ORIGINAL_ATLAS)
        for col in FEATURE_COLUMNS:
            if col not in atlas.columns:
                atlas[col] = 0.0
        signals = np.load(SIGNALS_PATH)["X"].astype(np.float32)
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        signal_rows = atlas["idx"].astype(int).to_numpy()
        self.noisy = signals[signal_rows]
        raw = atlas[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        self.aux = ((raw - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32)
        self.y = atlas["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.split = atlas["split"].astype(str).to_numpy()
        self.region = atlas.get("original_region", pd.Series("", index=atlas.index)).astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "noisy": torch.from_numpy(self.noisy[i]).unsqueeze(0),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


class WaveformAuxUformer(nn.Module):
    def __init__(self, cfg: Uformer1DConfig, aux_dim: int, hidden: int = 256, aux_hidden: int = 128):
        super().__init__()
        self.cfg = cfg
        self.backbone = UformerDenoiseSQIModel(cfg)
        self.class_head: nn.Module | None = None
        self.aux_head: nn.Module | None = None
        self.hidden = int(hidden)
        self.aux_hidden = int(aux_hidden)
        self.aux_dim = int(aux_dim)

    def attach_heads(self, device: torch.device, seq_len: int = 1250) -> None:
        with torch.no_grad():
            sample = torch.zeros(2, 1, seq_len, device=device)
            feature_dim = int(self.backbone(sample)["features"].shape[1])
        self.class_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.hidden, 3),
        ).to(device)
        self.aux_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.aux_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.aux_hidden, self.aux_dim),
        ).to(device)

    def forward(self, noisy: torch.Tensor, frozen_backbone: bool) -> dict[str, torch.Tensor]:
        if frozen_backbone:
            with torch.no_grad():
                out = self.backbone(noisy)
        else:
            out = self.backbone(noisy)
        feat = out["features"]
        assert isinstance(feat, torch.Tensor)
        if self.class_head is None or self.aux_head is None:
            raise RuntimeError("heads are not attached")
        return {
            "logits": self.class_head(feat),
            "aux_pred": self.aux_head(feat),
            "denoise": out["denoise"],
            "features": feat,
        }


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}; expected one of {sorted(CANDIDATES)}")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in names:
        summary = train_candidate(name, CANDIDATES[name], args.epochs, args.batch_size, args.num_workers)
        summaries.append(summary)
        rows.extend(summary["metric_rows"])

    metrics = pd.DataFrame(rows)
    metrics_path = ANALYSIS_DIR / "waveform_aux_sqi_geometry_metrics.csv"
    summary_path = ANALYSIS_DIR / "waveform_aux_sqi_geometry_summary.json"
    report_path = ANALYSIS_DIR / "waveform_aux_sqi_geometry_report.md"
    report_copy = REPORT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "variant_id": VARIANT_ID,
        "model_family": "waveform_only_uformer_aux_sqi_geometry",
        "input_contract": "ECG waveform only; SQI/geometry columns are auxiliary training targets, never classifier inputs",
        "original_used_for_training": False,
        "feature_columns": FEATURE_COLUMNS,
        "core_aux_columns": CORE_AUX_COLUMNS,
        "reference_47": REFERENCE_47,
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "candidates": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


def train_candidate(name: str, cfg_info: dict[str, Any], epochs: int, batch_size: int, num_workers: int) -> dict[str, Any]:
    print(f"training {name} epochs={epochs}", flush=True)
    torch.manual_seed(int(cfg_info["seed"]))
    np.random.seed(int(cfg_info["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SyntheticAuxDataset("train")
    val_ds = SyntheticAuxDataset("val")
    test_ds = SyntheticAuxDataset("test")
    norm = FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = OriginalAuxDataset(norm)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin)
    original_loader = DataLoader(original_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin)

    base_cfg = GEOM.load_base_config(BASE_CKPT)
    unfreeze = str(cfg_info.get("unfreeze", "none"))
    cfg = Uformer1DConfig(
        noise_scale=base_cfg.noise_scale,
        dropout=0.08,
        feature_set=base_cfg.feature_set,
        detach_encoder_features=(unfreeze == "none"),
        head_hidden_dim=base_cfg.head_hidden_dim,
        denoiser_width=base_cfg.denoiser_width,
    )
    model = WaveformAuxUformer(cfg, aux_dim=len(FEATURE_COLUMNS)).to(device)
    load_report = load_denoiser(model, BASE_CKPT, device)
    model.attach_heads(device=device)
    frozen_backbone = configure_trainable(model, unfreeze)
    params = [{"params": list(model.class_head.parameters()) + list(model.aux_head.parameters()), "lr": float(cfg_info["lr_head"])}]
    if not frozen_backbone:
        backbone_params = [p for p in model.backbone.denoiser.parameters() if p.requires_grad]
        params.append({"params": backbone_params, "lr": float(cfg_info["lr_backbone"])})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    class_weight = torch.tensor(cfg_info["class_weight"], dtype=torch.float32, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -1e9
    train_log: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        cls_losses = []
        aux_losses = []
        for batch in train_loader:
            noisy = batch["noisy"].to(device=device, dtype=torch.float32)
            clean = batch["clean"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux_target = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(noisy, frozen_backbone=frozen_backbone)
            cls_loss = F.cross_entropy(out["logits"], y, weight=class_weight)
            aux_loss = F.smooth_l1_loss(out["aux_pred"], aux_target)
            if CORE_AUX_IDX:
                core_aux_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_AUX_IDX], aux_target[:, CORE_AUX_IDX])
            else:
                core_aux_loss = torch.zeros((), device=device)
            denoise_loss = F.mse_loss(out["denoise"], clean)
            loss = (
                cls_loss
                + float(cfg_info["aux_weight"]) * aux_loss
                + float(cfg_info["core_aux_weight"]) * core_aux_loss
                + float(cfg_info["denoise_weight"]) * denoise_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            cls_losses.append(float(cls_loss.detach().cpu()))
            aux_losses.append(float(aux_loss.detach().cpu()))
        val_rep, _ = eval_loader(model, val_loader, device, frozen_backbone)
        score = val_rep["acc"] + 0.20 * val_rep["macro_f1"] + 0.15 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"]) - 0.02 * val_rep["aux_mae"]
        train_log.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_cls_loss": float(np.mean(cls_losses)),
                "train_aux_loss": float(np.mean(aux_losses)),
                **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"},
            }
        )
        print(
            f"{name} epoch={epoch}/{epochs} loss={np.mean(losses):.4f} "
            f"val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} "
            f"aux_mae={val_rep['aux_mae']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    val_rep, val_probs = eval_loader(model, val_loader, device, frozen_backbone)
    test_rep, test_probs = eval_loader(model, test_loader, device, frozen_backbone)
    orig_rep, orig_probs = eval_loader(model, original_loader, device, frozen_backbone)
    threshold = calibrate_bad_threshold(val_ds.y, val_probs)
    test_badcal = apply_bad_threshold(test_probs, threshold)
    orig_badcal = apply_bad_threshold(orig_probs, threshold)

    metric_rows = [
        metric_row(name, "synthetic_val", val_rep),
        metric_row(name, "synthetic_test", test_rep),
        metric_row(f"{name}_badcal", "synthetic_test", GEOM.metric_report(test_ds.y, test_badcal, test_probs)),
        metric_row(name, "original_test_all_10s+", bucket_report(original_ds, orig_probs, "original_test_all_10s+")),
        metric_row(f"{name}_badcal", "original_test_all_10s+", bucket_report(original_ds, orig_probs, "original_test_all_10s+", threshold)),
        metric_row(name, "original_all_10s+", bucket_report(original_ds, orig_probs, "original_all_10s+")),
        metric_row(f"{name}_badcal", "original_all_10s+", bucket_report(original_ds, orig_probs, "original_all_10s+", threshold)),
        metric_row(name, "bad_core_nearboundary", bucket_report(original_ds, orig_probs, "bad_core_nearboundary")),
        metric_row(f"{name}_badcal", "bad_core_nearboundary", bucket_report(original_ds, orig_probs, "bad_core_nearboundary", threshold)),
        metric_row(name, "bad_outlier_stress", bucket_report(original_ds, orig_probs, "bad_outlier_stress")),
        metric_row(f"{name}_badcal", "bad_outlier_stress", bucket_report(original_ds, orig_probs, "bad_outlier_stress", threshold)),
    ]

    run_dir = OUT_ROOT / "runs" / "waveform_aux_sqi_geometry" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "denoiser_state": model.backbone.denoiser.state_dict(),
            "config": cfg.__dict__,
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": norm.mean,
            "feature_train_std": norm.std,
            "aux_target_columns": FEATURE_COLUMNS,
            "class_weight": cfg_info["class_weight"],
            "candidate_config": cfg_info,
            "model_kind": "waveform_only_uformer_aux_sqi_geometry",
            "input_contract": "waveform only; feature columns are training targets, not model inputs",
            "best_epoch": best_epoch,
            "base_checkpoint": str(BASE_CKPT),
            "load_report": load_report,
            "bad_threshold_trainval": threshold,
            "original_rows_used_for_training": False,
        },
        ckpt_path,
    )
    pd.DataFrame(train_log).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "synthetic_test_probs.npz", probs=test_probs)
    np.savez_compressed(run_dir / "original_probs.npz", probs=orig_probs)
    return {
        "candidate": name,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "bad_threshold_trainval": threshold,
        "load_report": load_report,
        "frozen_backbone": frozen_backbone,
        "config": cfg_info,
        "synthetic_val": val_rep,
        "synthetic_test": test_rep,
        "original_test_all_10s+": bucket_report(original_ds, orig_probs, "original_test_all_10s+", threshold),
        "metric_rows": metric_rows,
    }


def load_denoiser(model: WaveformAuxUformer, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("denoiser_state", ckpt.get("model_state", ckpt))
    missing, unexpected = model.backbone.denoiser.load_state_dict(state, strict=False)
    return {"checkpoint": str(checkpoint), "missing": len(missing), "unexpected": len(unexpected)}


def configure_trainable(model: WaveformAuxUformer, unfreeze: str) -> bool:
    for p in model.backbone.denoiser.parameters():
        p.requires_grad = False
    if unfreeze == "none":
        return True
    if unfreeze == "bottleneck":
        for p in model.backbone.denoiser.bottleneck.parameters():
            p.requires_grad = True
        return False
    raise ValueError(f"unknown unfreeze mode: {unfreeze}")


@torch.no_grad()
def eval_loader(model: WaveformAuxUformer, loader: DataLoader, device: torch.device, frozen_backbone: bool) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    ys = []
    probs = []
    aux_abs = []
    for batch in loader:
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        aux_target = batch["aux"].to(device=device, dtype=torch.float32)
        out = model(noisy, frozen_backbone=frozen_backbone)
        prob = F.softmax(out["logits"], dim=1)
        probs.append(prob.cpu().numpy())
        ys.append(batch["y"].numpy())
        aux_abs.append(torch.mean(torch.abs(out["aux_pred"] - aux_target), dim=0).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    rep = GEOM.metric_report(y, p.argmax(axis=1), p)
    rep["aux_mae"] = float(np.mean(np.stack(aux_abs)))
    if CORE_AUX_IDX:
        rep["core_aux_mae"] = float(np.mean(np.stack(aux_abs)[:, CORE_AUX_IDX]))
    else:
        rep["core_aux_mae"] = rep["aux_mae"]
    return rep, p


def calibrate_bad_threshold(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    best: tuple[tuple[float, float, float], float] | None = None
    for threshold in np.round(np.linspace(0.01, 0.95, 95), 2):
        pred = apply_bad_threshold(probs_val, float(threshold))
        rep = GEOM.metric_report(y_val, pred, probs_val)
        score = (rep["acc"], rep["macro_f1"], min(rep["good_recall"], rep["medium_recall"], rep["bad_recall"]))
        if best is None or score > best[0]:
            best = (score, float(threshold))
    assert best is not None
    return best[1]


def apply_bad_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    pred = probs.argmax(axis=1).astype(np.int64)
    pred[probs[:, CLASS_TO_INT["bad"]] >= float(threshold)] = CLASS_TO_INT["bad"]
    return pred


def bucket_mask(ds: OriginalAuxDataset, bucket: str) -> np.ndarray:
    split = ds.split
    region = ds.region
    y = ds.y
    if bucket == "original_all_10s+":
        return np.ones(len(y), dtype=bool)
    if bucket == "original_test_all_10s+":
        return split == "test"
    if bucket == "bad_core_nearboundary":
        return (split == "test") & (y == CLASS_TO_INT["bad"]) & (region != "outlier_low_confidence")
    if bucket == "bad_outlier_stress":
        return (split == "test") & (y == CLASS_TO_INT["bad"]) & (region == "outlier_low_confidence")
    if bucket == "original_test_main_without_bad_stress":
        return (split == "test") & ~((y == CLASS_TO_INT["bad"]) & (region == "outlier_low_confidence"))
    raise ValueError(f"unknown bucket {bucket}")


def bucket_report(ds: OriginalAuxDataset, probs: np.ndarray, bucket: str, threshold: float | None = None) -> dict[str, Any]:
    mask = bucket_mask(ds, bucket)
    pred = probs.argmax(axis=1).astype(np.int64) if threshold is None else apply_bad_threshold(probs, threshold)
    return GEOM.metric_report(ds.y[mask], pred[mask], probs[mask])


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only UFormer + Auxiliary SQI/Geometry Prediction",
        "",
        "The classifier input is ECG waveform only. The 47 SQI/geometry columns supervise an auxiliary regression head during training, but they are not passed to the classifier at inference.",
        "",
        "## Held-Out Original BUT Test",
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
    for item in payload["candidates"]:
        lines.append(
            f"- `{item['candidate']}`: frozen_backbone={item['frozen_backbone']}, best_epoch={item['best_epoch']}, "
            f"bad_threshold_trainval={item['bad_threshold_trainval']:.2f}, checkpoint=`{item['checkpoint']}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Original BUT rows are report-only and are never used for training, feature normalization, threshold selection, or model selection.",
            "- This is a normal neural checkpoint experiment, not a rule artifact and not a tabular-input model.",
            f"- Metrics CSV: `{payload['metrics_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
