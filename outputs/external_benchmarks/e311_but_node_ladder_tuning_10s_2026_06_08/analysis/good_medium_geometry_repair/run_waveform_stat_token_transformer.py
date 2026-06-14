"""Waveform-only StatToken Transformer diagnostic.

This model still receives waveform channels only.  It adds an internal
waveform-stat token branch computed on-the-fly from the input tensor, then
fuses that with ConvSubsample Transformer features.  The goal is to test
whether explicit global waveform statistics help the Transformer recover the
SQI/geometry quantities that the tabular MLP uses.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
AUG_SCRIPT = ANALYSIS_DIR / "run_waveform_transformer_augmented_original.py"
NODE_ID = "N17043_gm_probe"


def load_aug_module() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_transformer_augmented_original", AUG_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AUG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUG = load_aug_module()
ORIG = AUG.ORIG
TX = AUG.TX
ARCH = AUG.ARCH
GEOM = AUG.GEOM
FEATURE_COLUMNS = list(AUG.FEATURE_COLUMNS)
CORE_AUX_IDX = list(AUG.CORE_AUX_IDX)

CANDIDATES = {
    "stattx_balanced": {
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.00, 1.18, 1.85],
        "focal_gamma": 1.0,
        "augment_strength": 1.0,
        "lr": 4.2e-4,
        "seed": 20260711,
    },
    "stattx_medium_guard": {
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "aux_weight": 0.35,
        "core_aux_weight": 0.25,
        "class_weight_mult": [1.00, 1.35, 1.80],
        "focal_gamma": 0.9,
        "augment_strength": 1.0,
        "lr": 4.0e-4,
        "seed": 20260712,
    },
}


class StatTokenTransformer(nn.Module):
    def __init__(self, cfg: dict[str, Any], in_ch: int):
        super().__init__()
        width = int(cfg["width"])
        self.encoder = TX.ConvSubsampleTransformer(in_ch, width, int(cfg["layers"]), int(cfg["heads"]))
        self.stat_dim = in_ch * 11
        hidden = int(cfg["stat_hidden"])
        self.stat_branch = nn.Sequential(
            nn.LayerNorm(self.stat_dim),
            nn.Linear(self.stat_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        fusion_dim = self.encoder.out_dim + hidden
        self.class_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 3),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, len(FEATURE_COLUMNS)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encoder(x)
        stats = waveform_stats(x)
        stat_feat = self.stat_branch(stats)
        feat = torch.cat([enc, stat_feat], dim=1)
        return {"logits": self.class_head(feat), "aux_pred": self.aux_head(feat), "features": feat, "stats": stats}


def waveform_stats(x: torch.Tensor) -> torch.Tensor:
    # x: B,C,L. These are differentiable or piecewise-differentiable waveform
    # summaries computed inside the model, not precomputed SQI inputs.
    eps = 1e-6
    diff = x[:, :, 1:] - x[:, :, :-1]
    mean = x.mean(dim=2)
    std = x.std(dim=2)
    mean_abs = x.abs().mean(dim=2)
    rms = torch.sqrt((x * x).mean(dim=2) + eps)
    peak_to_peak = x.amax(dim=2) - x.amin(dim=2)
    diff_abs = diff.abs().mean(dim=2)
    diff_rms = torch.sqrt((diff * diff).mean(dim=2) + eps)
    flat = (diff.abs() < 0.015).float().mean(dim=2)
    low_amp = (x.abs() < 0.050).float().mean(dim=2)
    zcr = ((x[:, :, 1:] * x[:, :, :-1]) < 0).float().mean(dim=2)
    # A smooth high-tail proxy: average of largest 10 percent abs diffs.
    k = max(1, int(diff.shape[2] * 0.10))
    diff_tail = torch.topk(diff.abs(), k=k, dim=2).values.mean(dim=2)
    return torch.cat([mean, std, mean_abs, rms, peak_to_peak, diff_abs, diff_rms, flat, low_amp, zcr, diff_tail], dim=1)


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
    metrics_path = ANALYSIS_DIR / "waveform_stat_token_transformer_metrics.csv"
    report_path = ANALYSIS_DIR / "waveform_stat_token_transformer_report.md"
    report_copy = REPORT_DIR / report_path.name
    summary_path = ANALYSIS_DIR / "waveform_stat_token_transformer_summary.json"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "input_contract": "waveform only; internal stat token computed from waveform; no 47-feature input",
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
    base_train = ORIG.OriginalSplitDataset("train", str(cfg["channels"]))
    train_ds = AUG.AugmentedDataset(base_train, float(cfg["augment_strength"]), int(cfg["seed"]))
    val_ds = ORIG.OriginalSplitDataset("val", str(cfg["channels"]), base_train.norm)
    test_ds = ORIG.OriginalSplitDataset("test", str(cfg["channels"]), base_train.norm)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = StatTokenTransformer(cfg, int(base_train.x.shape[1])).to(device)
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
            pt = torch.softmax(out["logits"], dim=1).gather(1, y[:, None]).squeeze(1).clamp(1e-5, 1.0)
            cls_loss = torch.mean(((1.0 - pt) ** float(cfg["focal_gamma"])) * ce)
            aux_loss = F.smooth_l1_loss(out["aux_pred"], aux)
            core_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_AUX_IDX], aux[:, CORE_AUX_IDX]) if CORE_AUX_IDX else torch.zeros((), device=device)
            loss = cls_loss + float(cfg["aux_weight"]) * aux_loss + float(cfg["core_aux_weight"]) * core_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, val_probs = eval_loader(model, val_loader, device)
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
    val_rep, val_probs = eval_loader(model, val_loader, device)
    test_rep, test_probs = eval_loader(model, test_loader, device)
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
    run_dir = OUT_ROOT / "runs" / "waveform_stat_token_transformer" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "waveform_only_stat_token_transformer",
            "input_contract": "waveform only; internal stat token computed from waveform",
            "feature_columns": FEATURE_COLUMNS,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    return {"candidate": name, "checkpoint": str(ckpt_path), "best_epoch": best_epoch, "bad_threshold_trainval": threshold, "metric_rows": metric_rows}


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
    rep = GEOM.metric_report(y, p.argmax(axis=1), p)
    rep["aux_mae"] = float(np.mean(np.stack(aux_abs)))
    return rep, p


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only StatToken Transformer",
        "",
        "The model receives waveform channels only.  Global stat tokens are computed inside the model from waveform tensors and fused with ConvSubsample Transformer features.",
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
