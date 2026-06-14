"""Transformer-only waveform adaptation on original BUT train/val/test split.

This diagnostic asks a narrow question: can a Transformer-family waveform model
learn the BUT morphology when trained on original-domain train labels?  The
classifier still receives waveform channels only.  SQI/geometry columns are
used only as auxiliary prediction targets.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
TX_SCRIPT = ANALYSIS_DIR / "run_waveform_transformer_search.py"
SIGNALS_PATH = ROOT / "outputs" / "external_benchmarks" / "e311_but_protocol_adaptation_2026_06_03" / "protocols" / "p1_current_10s_center" / "signals.npz"
ORIGINAL_ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
NODE_ID = "N17043_gm_probe"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}

CANDIDATES = {
    "orig_conformer_robust3_aux": {
        "arch": "conformer",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.45,
        "core_aux_weight": 0.30,
        "lr": 5e-4,
        "seed": 20260681,
    },
    "orig_multipatch_robust3_aux": {
        "arch": "multipatch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.45,
        "core_aux_weight": 0.30,
        "lr": 5e-4,
        "seed": 20260682,
    },
    "orig_convtx_robust3_aux": {
        "arch": "convtx",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "aux_weight": 0.45,
        "core_aux_weight": 0.30,
        "lr": 5e-4,
        "seed": 20260683,
    },
}


def load_tx_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_transformer_search", TX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TX_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TX = load_tx_module()
ARCH = TX.ARCH
GEOM = TX.GEOM
FEATURE_COLUMNS = list(TX.FEATURE_COLUMNS)
CORE_AUX_IDX = list(TX.CORE_AUX_IDX)


@dataclass
class FeatureNorm:
    mean: np.ndarray
    std: np.ndarray


class OriginalSplitDataset(Dataset):
    def __init__(self, split: str, channel_mode: str, norm: FeatureNorm | None = None):
        atlas = pd.read_csv(ORIGINAL_ATLAS)
        for col in FEATURE_COLUMNS:
            if col not in atlas.columns:
                atlas[col] = 0.0
        raw_all = atlas[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        train_mask = atlas["split"].astype(str).eq("train").to_numpy()
        if norm is None:
            mean = raw_all[train_mask].mean(axis=0).astype(np.float32)
            std = raw_all[train_mask].std(axis=0).astype(np.float32)
            std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
            norm = FeatureNorm(mean=mean, std=std)
        self.norm = norm
        mask = atlas["split"].astype(str).eq(split).to_numpy()
        self.frame = atlas.loc[mask].reset_index(drop=True)
        signals = np.load(SIGNALS_PATH)["X"].astype(np.float32)
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        idx = self.frame["idx"].astype(int).to_numpy()
        self.x = ARCH.make_channels(signals[idx], channel_mode)
        raw = self.frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        self.aux = ((raw - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32)
        self.y = self.frame["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.region = self.frame.get("original_region", pd.Series("", index=self.frame.index)).astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries = []
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}; expected {sorted(CANDIDATES)}")
        summary = train_candidate(name, CANDIDATES[name], args)
        summaries.append(summary)
        rows.extend(summary["metric_rows"])
    metrics = pd.DataFrame(rows)
    metrics_path = ANALYSIS_DIR / "waveform_transformer_original_adaptation_metrics.csv"
    summary_path = ANALYSIS_DIR / "waveform_transformer_original_adaptation_summary.json"
    report_path = ANALYSIS_DIR / "waveform_transformer_original_adaptation_report.md"
    report_copy = REPORT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "training_scope": "original BUT train labels only; val selects epoch and bad threshold; test held out",
        "input_contract": "waveform only; SQI/geometry columns auxiliary targets only",
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
    train_ds = OriginalSplitDataset("train", str(cfg["channels"]))
    val_ds = OriginalSplitDataset("val", str(cfg["channels"]), train_ds.norm)
    test_ds = OriginalSplitDataset("test", str(cfg["channels"]), train_ds.norm)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = TX.TransformerAuxClassifier(cfg, int(train_ds.x.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    counts = np.bincount(train_ds.y, minlength=3).astype(np.float32)
    inv = counts.sum() / np.maximum(counts, 1.0)
    class_weight = torch.tensor((inv / inv.mean()).astype(np.float32), device=device)
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
        val_rep, val_probs = eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.25 * val_rep["macro_f1"] + 0.20 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
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
    threshold = calibrate_bad_threshold(val_ds.y, val_probs)
    test_badcal = GEOM.metric_report(test_ds.y, ARCH.apply_bad_threshold(test_probs, threshold), test_probs)
    metric_rows = [
        ARCH.metric_row(name, "original_val", val_rep),
        ARCH.metric_row(name, "original_test_all_10s+", test_rep),
        ARCH.metric_row(f"{name}_badcal", "original_test_all_10s+", test_badcal),
        ARCH.metric_row(name, "bad_core_nearboundary", bucket_report(test_ds, test_probs, "bad_core_nearboundary")),
        ARCH.metric_row(f"{name}_badcal", "bad_core_nearboundary", bucket_report(test_ds, test_probs, "bad_core_nearboundary", threshold)),
        ARCH.metric_row(name, "bad_outlier_stress", bucket_report(test_ds, test_probs, "bad_outlier_stress")),
        ARCH.metric_row(f"{name}_badcal", "bad_outlier_stress", bucket_report(test_ds, test_probs, "bad_outlier_stress", threshold)),
    ]
    run_dir = OUT_ROOT / "runs" / "waveform_transformer_original_adaptation" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "waveform_only_transformer_original_trainval_adaptation",
            "input_contract": "waveform only; no feature input",
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": train_ds.norm.mean,
            "feature_train_std": train_ds.norm.std,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
            "training_scope": "original train labels only; val selection; test held out",
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
        "test": test_badcal,
        "metric_rows": metric_rows,
    }


@torch.no_grad()
def eval_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    ys = []
    probs = []
    aux_abs = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        aux = batch["aux"].to(device=device, dtype=torch.float32)
        out = model(x)
        probs.append(F.softmax(out["logits"], dim=1).detach().cpu().numpy())
        ys.append(batch["y"].numpy())
        aux_abs.append(torch.mean(torch.abs(out["aux_pred"] - aux), dim=0).detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    rep = GEOM.metric_report(y, p.argmax(axis=1), p)
    rep["aux_mae"] = float(np.mean(np.stack(aux_abs)))
    return rep, p


def calibrate_bad_threshold(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    best: tuple[tuple[float, float, float], float] | None = None
    for threshold in np.round(np.linspace(0.01, 0.95, 95), 2):
        pred = ARCH.apply_bad_threshold(probs_val, float(threshold))
        rep = GEOM.metric_report(y_val, pred, probs_val)
        score = (rep["acc"], rep["macro_f1"], min(rep["good_recall"], rep["medium_recall"], rep["bad_recall"]))
        if best is None or score > best[0]:
            best = (score, float(threshold))
    assert best is not None
    return best[1]


def bucket_report(ds: OriginalSplitDataset, probs: np.ndarray, bucket: str, threshold: float | None = None) -> dict[str, Any]:
    if bucket == "bad_core_nearboundary":
        mask = (ds.y == CLASS_TO_INT["bad"]) & (ds.region != "outlier_low_confidence")
    elif bucket == "bad_outlier_stress":
        mask = (ds.y == CLASS_TO_INT["bad"]) & (ds.region == "outlier_low_confidence")
    else:
        raise ValueError(bucket)
    pred = probs.argmax(axis=1).astype(np.int64) if threshold is None else ARCH.apply_bad_threshold(probs, threshold)
    return GEOM.metric_report(ds.y[mask], pred[mask], probs[mask])


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only Transformer Original Adaptation",
        "",
        "Diagnostic only: original BUT train labels are used for training, val for selection, test held out. Inputs remain waveform only.",
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
