"""Transformer-only waveform training with ECG-style augmentation.

Inputs are waveform channels only.  SQI/geometry columns are auxiliary targets,
not classifier inputs.  This diagnostic tests whether simple waveform
augmentations can reduce original BUT train/test record-domain shift.
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ORIG_SCRIPT = ANALYSIS_DIR / "run_waveform_transformer_original_adaptation.py"
NODE_ID = "N17043_gm_probe"


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
ARCH = ORIG.ARCH
GEOM = ORIG.GEOM
FEATURE_COLUMNS = list(ORIG.FEATURE_COLUMNS)
CORE_AUX_IDX = list(ORIG.CORE_AUX_IDX)

CANDIDATES = {
    "aug_convtx_balanced_focal": {
        "arch": "convtx",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.00, 1.18, 1.85],
        "focal_gamma": 1.0,
        "augment_strength": 1.0,
        "lr": 4.5e-4,
        "seed": 20260701,
    },
    "aug_convtx_medium_guard": {
        "arch": "convtx",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.00, 1.45, 1.70],
        "focal_gamma": 0.7,
        "augment_strength": 1.0,
        "lr": 4.0e-4,
        "seed": 20260702,
    },
    "aug_conformer_balanced_focal": {
        "arch": "conformer",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.00, 1.18, 1.85],
        "focal_gamma": 1.0,
        "augment_strength": 1.0,
        "lr": 4.5e-4,
        "seed": 20260703,
    },
    "aug_convtx_balanced_focal_trainval": {
        "arch": "convtx",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.00, 1.18, 1.85],
        "focal_gamma": 1.0,
        "augment_strength": 1.0,
        "lr": 4.5e-4,
        "seed": 20260704,
        "train_scope": "trainval",
    },
    "aug_convtx_badoutlier_style": {
        "arch": "convtx",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.32,
        "core_aux_weight": 0.22,
        "class_weight_mult": [1.00, 1.12, 2.35],
        "focal_gamma": 1.2,
        "augment_strength": 1.0,
        "bad_outlier_aug_strength": 2.2,
        "lr": 4.0e-4,
        "seed": 20260705,
    },
}


class AugmentedDataset(Dataset):
    def __init__(self, base: ORIG.OriginalSplitDataset, strength: float, seed: int, bad_strength: float | None = None):
        self.base = base
        self.strength = float(strength)
        self.bad_strength = None if bad_strength is None else float(bad_strength)
        self.rng = np.random.default_rng(seed)
        self.y = base.y
        self.region = base.region

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        x = np.array(item["x"], dtype=np.float32, copy=True)
        y = int(item["y"])
        x = augment_waveform_channels(x, self.rng, self.strength, y, self.bad_strength)
        return {"x": torch.from_numpy(x), "y": item["y"], "aux": item["aux"]}


class ConcatOriginalDataset(Dataset):
    def __init__(self, parts: list[ORIG.OriginalSplitDataset]):
        self.parts = parts
        self.x = np.concatenate([p.x for p in parts], axis=0)
        self.y = np.concatenate([p.y for p in parts], axis=0)
        self.aux = np.concatenate([p.aux for p in parts], axis=0)
        self.region = np.concatenate([p.region for p in parts], axis=0)
        self.norm = parts[0].norm

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "aux": torch.from_numpy(self.aux[idx]),
        }


def augment_waveform_channels(
    x: np.ndarray,
    rng: np.random.Generator,
    strength: float,
    y: int | None = None,
    bad_strength: float | None = None,
) -> np.ndarray:
    if strength <= 0:
        return x
    c, n = x.shape
    if rng.random() < 0.75:
        shift = int(rng.integers(-48, 49))
        x = np.roll(x, shift, axis=1)
    if rng.random() < 0.85:
        scale = float(rng.uniform(0.82, 1.18))
        x *= scale
    if rng.random() < 0.65:
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        freq = float(rng.uniform(0.35, 1.25))
        amp = float(rng.uniform(0.015, 0.075) * strength)
        x += amp * np.sin(2.0 * np.pi * freq * t + phase)[None, :]
    if rng.random() < 0.75:
        sigma = float(rng.uniform(0.008, 0.045) * strength)
        x += rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    if rng.random() < 0.40:
        width = int(rng.integers(24, 120))
        start = int(rng.integers(0, max(1, n - width)))
        fill = x[:, max(0, start - 1) : max(1, start)].mean(axis=1, keepdims=True)
        x[:, start : start + width] = fill
    if rng.random() < 0.35:
        slope = float(rng.uniform(-0.08, 0.08) * strength)
        x += slope * np.linspace(-0.5, 0.5, n, dtype=np.float32)[None, :]
    if y == 2 and bad_strength is not None and bad_strength > 0:
        x = augment_bad_outlier_style(x, rng, bad_strength)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_bad_outlier_style(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    c, n = x.shape
    if rng.random() < 0.85:
        width = int(rng.integers(120, 420))
        start = int(rng.integers(0, max(1, n - width)))
        if rng.random() < 0.5:
            x[:, start : start + width] *= float(rng.uniform(0.05, 0.35))
        else:
            x[:, start : start + width] += float(rng.uniform(-0.35, 0.35))
    if rng.random() < 0.75:
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x += float(rng.uniform(0.08, 0.22)) * np.sin(2 * np.pi * float(rng.uniform(0.08, 0.35)) * t + float(rng.uniform(0, 2 * np.pi)))[None, :]
    if rng.random() < 0.70:
        hf = rng.normal(0.0, float(rng.uniform(0.06, 0.16)), size=x.shape).astype(np.float32)
        x += hf
    if rng.random() < 0.50:
        clip = float(rng.uniform(0.45, 1.05))
        x = np.clip(x, -clip, clip)
    if rng.random() < 0.40:
        x *= float(rng.uniform(0.35, 1.80))
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    summaries = []
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}; expected {sorted(CANDIDATES)}")
        summary = train_candidate(name, CANDIDATES[name], args)
        summaries.append(summary)
        rows.extend(summary["metric_rows"])
    metrics = pd.DataFrame(rows)
    metrics_path = ANALYSIS_DIR / "waveform_transformer_augmented_original_metrics.csv"
    report_path = ANALYSIS_DIR / "waveform_transformer_augmented_original_report.md"
    report_copy = REPORT_DIR / report_path.name
    summary_path = ANALYSIS_DIR / "waveform_transformer_augmented_original_summary.json"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "input_contract": "waveform only; SQI/geometry features are auxiliary targets, not inputs",
        "training_scope": "original train labels with ECG-style waveform augmentation; val selection; test held out",
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
    base_train_only = ORIG.OriginalSplitDataset("train", str(cfg["channels"]))
    val_ds = ORIG.OriginalSplitDataset("val", str(cfg["channels"]), base_train_only.norm)
    train_scope = str(cfg.get("train_scope", "train"))
    if train_scope == "trainval":
        base_train = ConcatOriginalDataset([base_train_only, val_ds])
    else:
        base_train = base_train_only
    if "bad_outlier_aug_strength" in cfg:
        train_ds = AugmentedDataset(base_train, float(cfg["augment_strength"]), int(cfg["seed"]), float(cfg["bad_outlier_aug_strength"]))
    else:
        train_ds = AugmentedDataset(base_train, float(cfg["augment_strength"]), int(cfg["seed"]))
    test_ds = ORIG.OriginalSplitDataset("test", str(cfg["channels"]), base_train_only.norm)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = TX.TransformerAuxClassifier(cfg, int(base_train.x.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    counts = np.bincount(base_train.y, minlength=3).astype(np.float32)
    inv = counts.sum() / np.maximum(counts, 1.0)
    weights = inv / inv.mean()
    weights *= np.asarray(cfg["class_weight_mult"], dtype=np.float32)
    weights /= weights.mean()
    class_weight = torch.tensor(weights.astype(np.float32), device=device)
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
            ce = F.cross_entropy(out["logits"], y, weight=class_weight, reduction="none")
            if float(cfg["focal_gamma"]) > 0:
                pt = torch.softmax(out["logits"], dim=1).gather(1, y[:, None]).squeeze(1).clamp(1e-5, 1.0)
                cls_loss = torch.mean(((1.0 - pt) ** float(cfg["focal_gamma"])) * ce)
            else:
                cls_loss = torch.mean(ce)
            aux_loss = F.smooth_l1_loss(out["aux_pred"], aux)
            core_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_AUX_IDX], aux[:, CORE_AUX_IDX]) if CORE_AUX_IDX else torch.zeros((), device=device)
            loss = cls_loss + float(cfg["aux_weight"]) * aux_loss + float(cfg["core_aux_weight"]) * core_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, val_probs = ORIG.eval_loader(model, val_loader, device)
        if train_scope == "trainval":
            score = float(epoch)
        else:
            score = val_rep["acc"] + 0.30 * val_rep["macro_f1"] + 0.45 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"{name} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} aux_mae={val_rep['aux_mae']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_rep, val_probs = ORIG.eval_loader(model, val_loader, device)
    test_rep, test_probs = ORIG.eval_loader(model, test_loader, device)
    threshold = ORIG.calibrate_bad_threshold(val_ds.y, val_probs)
    test_badcal = GEOM.metric_report(test_ds.y, ARCH.apply_bad_threshold(test_probs, threshold), test_probs)
    metric_rows = [
        ARCH.metric_row(name, "original_val_in_train" if train_scope == "trainval" else "original_val", val_rep),
        ARCH.metric_row(name, "original_test_all_10s+", test_rep),
        ARCH.metric_row(f"{name}_badcal", "original_test_all_10s+", test_badcal),
        ARCH.metric_row(name, "bad_core_nearboundary", ORIG.bucket_report(test_ds, test_probs, "bad_core_nearboundary")),
        ARCH.metric_row(f"{name}_badcal", "bad_core_nearboundary", ORIG.bucket_report(test_ds, test_probs, "bad_core_nearboundary", threshold)),
        ARCH.metric_row(name, "bad_outlier_stress", ORIG.bucket_report(test_ds, test_probs, "bad_outlier_stress")),
        ARCH.metric_row(f"{name}_badcal", "bad_outlier_stress", ORIG.bucket_report(test_ds, test_probs, "bad_outlier_stress", threshold)),
    ]
    run_dir = OUT_ROOT / "runs" / "waveform_transformer_augmented_original" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "waveform_only_transformer_augmented_original",
            "input_contract": "waveform only; no feature input",
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": base_train_only.norm.mean,
            "feature_train_std": base_train_only.norm.std,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
            "training_scope": "original train+val labels with waveform augmentation; test held out" if train_scope == "trainval" else "original train labels with waveform augmentation; val selection; test held out",
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "original_test_probs.npz", probs=test_probs)
    return {
        "candidate": name,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "bad_threshold_trainval": threshold,
        "metric_rows": metric_rows,
    }


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only Transformer Augmented Original Training",
        "",
        "Original train labels are used with ECG-style waveform augmentation; validation selects epoch and bad threshold; original test is held out.",
        "",
        "| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} |"
        )
    lines.extend(["", "## Candidates", ""])
    for item in payload["summaries"]:
        lines.append(f"- `{item['candidate']}`: best_epoch={item['best_epoch']}, threshold={item['bad_threshold_trainval']:.2f}, checkpoint=`{item['checkpoint']}`")
    lines.extend(["", f"Metrics CSV: `{payload['metrics_csv']}`", ""])
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
