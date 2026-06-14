"""Waveform-only Transformer probability ensemble probe.

This diagnostic combines only waveform-model probabilities.  No SQI/geometry
feature is used as input.  Ensemble weights and the optional bad threshold are
selected on the original BUT validation split and evaluated once on held-out
original test.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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

RUNS = OUT_ROOT / "runs"

MODEL_PATHS = {
    "orig_conformer": RUNS / "waveform_transformer_original_adaptation" / NODE_ID / "orig_conformer_robust3_aux" / "ckpt_best.pt",
    "orig_multipatch": RUNS / "waveform_transformer_original_adaptation" / NODE_ID / "orig_multipatch_robust3_aux" / "ckpt_best.pt",
    "orig_convtx": RUNS / "waveform_transformer_original_adaptation" / NODE_ID / "orig_convtx_robust3_aux" / "ckpt_best.pt",
    "synth_conformer": RUNS / "waveform_transformer_search" / NODE_ID / "conformer_robust3_auxheavy" / "ckpt_best.pt",
    "synth_multipatch": RUNS / "waveform_transformer_search" / NODE_ID / "multipatch_robust3_auxheavy" / "ckpt_best.pt",
    "synth_convtx": RUNS / "waveform_transformer_search" / NODE_ID / "convtx_robust3_auxheavy" / "ckpt_best.pt",
    "ft_convtx_balanced": RUNS / "waveform_transformer_pretrain_finetune" / NODE_ID / "synthft_convtx_balanced" / "ckpt_best.pt",
    "aug_convtx_balanced": RUNS / "waveform_transformer_augmented_original" / NODE_ID / "aug_convtx_balanced_focal" / "ckpt_best.pt",
    "aug_convtx_medium": RUNS / "waveform_transformer_augmented_original" / NODE_ID / "aug_convtx_medium_guard" / "ckpt_best.pt",
    "aug_conformer": RUNS / "waveform_transformer_augmented_original" / NODE_ID / "aug_conformer_balanced_focal" / "ckpt_best.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", type=str, default=",".join(MODEL_PATHS.keys()))
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    probs: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] = {}
    for name in model_names:
        path = MODEL_PATHS[name]
        if not path.exists():
            print(f"skip missing {name}: {path}", flush=True)
            continue
        pred = compute_model_probs(name, path, args)
        probs[name] = {"val": pred["val_probs"], "test": pred["test_probs"]}
        labels["val"] = pred["val_y"]
        labels["test"] = pred["test_y"]
    if not probs:
        raise RuntimeError("no model probabilities available")
    rows = search_ensembles(probs, labels["val"], labels["test"])
    top = rows.head(int(args.top_k)).copy()
    metrics_path = ANALYSIS_DIR / "waveform_transformer_ensemble_probe_metrics.csv"
    report_path = ANALYSIS_DIR / "waveform_transformer_ensemble_probe_report.md"
    report_copy = REPORT_DIR / report_path.name
    summary_path = ANALYSIS_DIR / "waveform_transformer_ensemble_probe_summary.json"
    rows.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "input_contract": "waveform model probabilities only; no SQI/geometry feature input",
        "selection": "ensemble weights and bad threshold selected on original validation split only",
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "available_models": {name: str(MODEL_PATHS[name]) for name in probs},
        "top": top.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(top, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


def compute_model_probs(name: str, ckpt_path: Path, args: argparse.Namespace) -> dict[str, np.ndarray]:
    print(f"scoring {name}: {ckpt_path}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    train_ds = ORIG.OriginalSplitDataset("train", str(cfg["channels"]))
    val_ds = ORIG.OriginalSplitDataset("val", str(cfg["channels"]), train_ds.norm)
    test_ds = ORIG.OriginalSplitDataset("test", str(cfg["channels"]), train_ds.norm)
    pin = device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = TX.TransformerAuxClassifier(cfg, int(train_ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    val_probs = eval_probs(model, val_loader, device)
    test_probs = eval_probs(model, test_loader, device)
    return {"val_probs": val_probs, "test_probs": test_probs, "val_y": val_ds.y, "test_y": test_ds.y}


@torch.no_grad()
def eval_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x)
        probs.append(F.softmax(out["logits"], dim=1).detach().cpu().numpy())
    return np.concatenate(probs)


def search_ensembles(probs: dict[str, dict[str, np.ndarray]], y_val: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    names = list(probs)
    rows: list[dict[str, Any]] = []
    thresholds = [None] + [round(x, 2) for x in np.linspace(0.02, 0.60, 30)]
    for subset_size in range(1, min(4, len(names)) + 1):
        for subset in itertools.combinations(names, subset_size):
            for weights in weight_grid(subset_size):
                val_probs = mix_probs(probs, subset, weights, "val")
                test_probs = mix_probs(probs, subset, weights, "test")
                for threshold in thresholds:
                    val_pred = val_probs.argmax(axis=1) if threshold is None else ARCH.apply_bad_threshold(val_probs, float(threshold))
                    test_pred = test_probs.argmax(axis=1) if threshold is None else ARCH.apply_bad_threshold(test_probs, float(threshold))
                    val_rep = GEOM.metric_report(y_val, val_pred, val_probs)
                    test_rep = GEOM.metric_report(y_test, test_pred, test_probs)
                    score = val_rep["acc"] + 0.30 * val_rep["macro_f1"] + 0.35 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
                    row = {
                        "score": float(score),
                        "models": "+".join(subset),
                        "weights": ",".join(f"{w:.2f}" for w in weights),
                        "bad_threshold": "" if threshold is None else float(threshold),
                    }
                    row.update({f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"})
                    row.update({f"test_{k}": v for k, v in test_rep.items() if k != "confusion_3x3"})
                    rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["score", "val_acc", "val_macro_f1", "test_acc"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def weight_grid(size: int) -> list[tuple[float, ...]]:
    if size == 1:
        return [(1.0,)]
    vals = np.linspace(0.0, 1.0, 11)
    out = []
    for raw in itertools.product(vals, repeat=size):
        s = float(sum(raw))
        if s <= 0:
            continue
        weights = tuple(float(x / s) for x in raw)
        if min(weights) <= 0.0:
            continue
        if weights not in out:
            out.append(weights)
    return out


def mix_probs(probs: dict[str, dict[str, np.ndarray]], subset: tuple[str, ...], weights: tuple[float, ...], split: str) -> np.ndarray:
    acc = None
    for name, weight in zip(subset, weights):
        cur = probs[name][split] * float(weight)
        acc = cur if acc is None else acc + cur
    assert acc is not None
    acc = np.clip(acc, 1e-8, 1.0)
    return acc / acc.sum(axis=1, keepdims=True)


def render_report(top: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform-Only Transformer Ensemble Probe",
        "",
        "Weights and bad threshold are selected on original validation only. Test is held out. Inputs are waveform-model probabilities only.",
        "",
        "| Rank | Models | Weights | Bad thr | Test Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Val Acc |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in top.reset_index(drop=True).iterrows():
        lines.append(
            f"| {i + 1} | {row['models']} | {row['weights']} | {row['bad_threshold']} | "
            f"{float(row['test_acc']):.6f} | {float(row['test_macro_f1']):.6f} | "
            f"{float(row['test_good_recall']):.6f} | {float(row['test_medium_recall']):.6f} | {float(row['test_bad_recall']):.6f} | "
            f"{int(row['test_good_to_medium'])} | {int(row['test_medium_to_good'])} | {int(row['test_bad_to_medium'])} | {float(row['val_acc']):.6f} |"
        )
    lines.extend(["", "## Models", ""])
    for name, path in payload["available_models"].items():
        lines.append(f"- `{name}`: `{path}`")
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
