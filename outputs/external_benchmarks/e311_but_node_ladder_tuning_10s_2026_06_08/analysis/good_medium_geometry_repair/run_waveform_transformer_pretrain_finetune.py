"""Transformer-only waveform synthetic-pretrain + original-finetune diagnostic.

The classifier receives waveform channels only.  SQI/geometry columns are used
only as auxiliary prediction targets.  This script asks whether the synthetic
boundary-block training can initialize a Transformer that then adapts to
original BUT train labels and generalizes to the held-out original test split.
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
from torch.utils.data import DataLoader


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

SYNTH_RUN_DIR = OUT_ROOT / "runs" / "waveform_transformer_search" / NODE_ID

CANDIDATES = {
    "synthft_conformer_head_then_all": {
        "init_checkpoint": SYNTH_RUN_DIR / "conformer_robust3_auxheavy" / "ckpt_best.pt",
        "lr": 1.2e-4,
        "encoder_lr_mult": 0.25,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.0, 1.10, 1.45],
        "seed": 20260691,
    },
    "synthft_conformer_badguard": {
        "init_checkpoint": SYNTH_RUN_DIR / "conformer_robust3_auxheavy" / "ckpt_best.pt",
        "lr": 9.0e-5,
        "encoder_lr_mult": 0.20,
        "aux_weight": 0.30,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.0, 1.05, 1.85],
        "seed": 20260692,
    },
    "synthft_convtx_balanced": {
        "init_checkpoint": SYNTH_RUN_DIR / "convtx_robust3_auxheavy" / "ckpt_best.pt",
        "lr": 1.1e-4,
        "encoder_lr_mult": 0.25,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.0, 1.12, 1.55],
        "seed": 20260693,
    },
    "synthft_convtx_badguard": {
        "init_checkpoint": SYNTH_RUN_DIR / "convtx_robust3_auxheavy" / "ckpt_best.pt",
        "lr": 8.0e-5,
        "encoder_lr_mult": 0.20,
        "aux_weight": 0.30,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.0, 1.05, 1.95],
        "seed": 20260694,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
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
    metrics_path = ANALYSIS_DIR / "waveform_transformer_pretrain_finetune_metrics.csv"
    summary_path = ANALYSIS_DIR / "waveform_transformer_pretrain_finetune_summary.json"
    report_path = ANALYSIS_DIR / "waveform_transformer_pretrain_finetune_report.md"
    report_copy = REPORT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "model_family": "Transformer-only waveform",
        "input_contract": "waveform only; SQI/geometry features are auxiliary targets, not inputs",
        "training_scope": "synthetic checkpoint initialization plus original train labels; val selection; test held out",
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
    init_checkpoint = Path(cfg["init_checkpoint"])
    ckpt = torch.load(init_checkpoint, map_location="cpu")
    model_cfg = dict(ckpt["candidate_config"])
    model_cfg["lr"] = float(cfg["lr"])
    model_cfg["aux_weight"] = float(cfg["aux_weight"])
    model_cfg["core_aux_weight"] = float(cfg["core_aux_weight"])
    train_ds = ORIG.OriginalSplitDataset("train", str(model_cfg["channels"]))
    val_ds = ORIG.OriginalSplitDataset("val", str(model_cfg["channels"]), train_ds.norm)
    test_ds = ORIG.OriginalSplitDataset("test", str(model_cfg["channels"]), train_ds.norm)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = TX.TransformerAuxClassifier(model_cfg, int(train_ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    param_groups = [
        {"params": list(model.encoder.parameters()), "lr": float(cfg["lr"]) * float(cfg["encoder_lr_mult"])},
        {"params": list(model.class_head.parameters()) + list(model.aux_head.parameters()), "lr": float(cfg["lr"])},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    counts = np.bincount(train_ds.y, minlength=3).astype(np.float32)
    inv = counts.sum() / np.maximum(counts, 1.0)
    weights = inv / inv.mean()
    weights = weights * np.asarray(cfg["class_weight_mult"], dtype=np.float32)
    weights = weights / weights.mean()
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
            cls_loss = F.cross_entropy(out["logits"], y, weight=class_weight)
            aux_loss = F.smooth_l1_loss(out["aux_pred"], aux)
            core_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_AUX_IDX], aux[:, CORE_AUX_IDX]) if CORE_AUX_IDX else torch.zeros((), device=device)
            loss = cls_loss + float(cfg["aux_weight"]) * aux_loss + float(cfg["core_aux_weight"]) * core_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, val_probs = ORIG.eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.30 * val_rep["macro_f1"] + 0.35 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
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
        ARCH.metric_row(name, "original_val", val_rep),
        ARCH.metric_row(name, "original_test_all_10s+", test_rep),
        ARCH.metric_row(f"{name}_badcal", "original_test_all_10s+", test_badcal),
        ARCH.metric_row(name, "bad_core_nearboundary", ORIG.bucket_report(test_ds, test_probs, "bad_core_nearboundary")),
        ARCH.metric_row(f"{name}_badcal", "bad_core_nearboundary", ORIG.bucket_report(test_ds, test_probs, "bad_core_nearboundary", threshold)),
        ARCH.metric_row(name, "bad_outlier_stress", ORIG.bucket_report(test_ds, test_probs, "bad_outlier_stress")),
        ARCH.metric_row(f"{name}_badcal", "bad_outlier_stress", ORIG.bucket_report(test_ds, test_probs, "bad_outlier_stress", threshold)),
    ]
    run_dir = OUT_ROOT / "runs" / "waveform_transformer_pretrain_finetune" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": model_cfg,
            "finetune_config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            "model_kind": "waveform_only_transformer_synthetic_pretrain_original_finetune",
            "input_contract": "waveform only; no feature input",
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": train_ds.norm.mean,
            "feature_train_std": train_ds.norm.std,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
            "init_checkpoint": str(init_checkpoint),
            "training_scope": "synthetic checkpoint initialization; original train labels only; val selection; test held out",
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "original_test_probs.npz", probs=test_probs)
    return {
        "candidate": name,
        "checkpoint": str(ckpt_path),
        "init_checkpoint": str(init_checkpoint),
        "best_epoch": best_epoch,
        "bad_threshold_trainval": threshold,
        "metric_rows": metric_rows,
    }


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only Transformer Pretrain + Finetune",
        "",
        "Diagnostic only: initialized from synthetic boundary-block Transformer checkpoints, then fine-tuned on original BUT train labels. Validation selects epoch and bad threshold; original test is held out.",
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
        lines.append(
            f"- `{item['candidate']}`: best_epoch={item['best_epoch']}, threshold={item['bad_threshold_trainval']:.2f}, "
            f"init=`{item['init_checkpoint']}`, checkpoint=`{item['checkpoint']}`"
        )
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
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


if __name__ == "__main__":
    main()
