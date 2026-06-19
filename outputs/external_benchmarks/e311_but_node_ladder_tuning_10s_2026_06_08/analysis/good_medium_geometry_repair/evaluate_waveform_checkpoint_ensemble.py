"""Evaluate an ensemble of waveform-only checkpoints on PTB or clean-BUT.

The ensemble averages class probabilities from independently trained waveform
Transformers.  Each checkpoint uses its own saved channel/feature normalization;
no tabular features are passed as inference input beyond the existing auxiliary
targets required by the model head.  This is read-only evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
PTB_ALIGN_SCRIPT = ANALYSIS_DIR / "run_ptb_bad_alignment_cross_dataset.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ALIGN = load_module("ptb_align_for_ensemble_eval", PTB_ALIGN_SCRIPT)
DUAL = ALIGN.DUAL


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def make_model(ckpt: dict[str, Any], in_ch: int, aux_dim: int) -> torch.nn.Module:
    cfg = dict(ckpt["candidate_config"])
    state = ckpt["model_state"]
    if any(str(k).startswith("base.") for k in state):
        model = ALIGN.StressAwareHierTransformer(cfg, in_ch, aux_dim)
    else:
        model = DUAL.HierTransformer(cfg, in_ch, aux_dim)
    model.load_state_dict(state)
    return model


def load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError(f"not an experiment checkpoint: {path}")
    return ckpt


def ptb_dataset(split: str, ckpt: dict[str, Any]):
    channel_stats = DUAL.ChannelStats(**ckpt["channel_stats"])
    feature_norm = DUAL.FeatureNorm(
        mean=np.asarray(ckpt["feature_train_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_train_std"], dtype=np.float32),
    )
    return ALIGN.BadAlignedSyntheticDualviewDataset(
        split,
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        channel_stats,
        feature_norm,
        False,
        "",
        0.0,
        20260619,
    )


def clean_dataset(policy: str, split: str, ckpt: dict[str, Any]):
    channel_stats = DUAL.ChannelStats(**ckpt["channel_stats"])
    feature_norm = DUAL.FeatureNorm(
        mean=np.asarray(ckpt["feature_train_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_train_std"], dtype=np.float32),
    )
    return DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, split, channel_stats, feature_norm)


@torch.no_grad()
def predict_probs(ckpt: dict[str, Any], ds, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model = make_model(ckpt, int(ds.x.shape[1]), len(DUAL.FORMAL_FEATURE_COLUMNS)).to(device)
    model.eval()
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    probs = []
    labels = []
    for batch in loader:
        out = model(batch["x"].to(device))
        probs.append(torch.softmax(out["logits"], dim=1).detach().cpu().numpy())
        labels.append(batch["y"].numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def metric_row(candidate: str, target: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in rep.items() if k != "confusion_3x3"}
    row.update({"candidate": candidate, "target": target})
    row["confusion_3x3"] = json.dumps(rep.get("confusion_3x3", []), ensure_ascii=False)
    return row


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform Checkpoint Ensemble Evaluation",
        "",
        "Read-only probability averaging of waveform Transformer checkpoints.",
        "",
        "## Checkpoints",
        "",
    ]
    for path in payload["checkpoints"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['target']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} |"
        )
    lines.extend(["", f"Metrics CSV: `{payload['metrics_csv']}`"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="semicolon-separated checkpoint paths")
    parser.add_argument("--mode", type=str, required=True, choices=["ptb", "clean"])
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--policies", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--name", type=str, default="ensemble")
    parser.add_argument("--weights", type=str, default="", help="optional semicolon-separated probability weights matching --checkpoints")
    args = parser.parse_args()

    paths = [Path(p.strip()) for p in args.checkpoints.split(";") if p.strip()]
    if len(paths) < 2:
        raise ValueError("ensemble needs at least two checkpoints")
    ckpts = [load_checkpoint(path) for path in paths]
    if str(args.weights).strip():
        weights = np.asarray([float(x.strip()) for x in str(args.weights).split(";") if x.strip()], dtype=np.float64)
        if weights.shape[0] != len(paths):
            raise ValueError("--weights must have one value per checkpoint")
        if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
            raise ValueError("--weights must be finite and sum positive")
        weights = weights / float(weights.sum())
    else:
        weights = np.ones(len(paths), dtype=np.float64) / float(len(paths))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    if args.mode == "ptb":
        targets = [s.strip() for s in args.splits.split(",") if s.strip()]
        for split in targets:
            summed = None
            y_ref = None
            for weight, ckpt in zip(weights, ckpts):
                ds = ptb_dataset(split, ckpt)
                probs, y = predict_probs(ckpt, ds, int(args.batch_size), device)
                weighted = probs * float(weight)
                summed = weighted if summed is None else summed + weighted
                if y_ref is None:
                    y_ref = y
                elif not np.array_equal(y_ref, y):
                    raise RuntimeError(f"label order mismatch for PTB {split}")
            avg = summed
            rep = DUAL.GEOM.metric_report(y_ref, avg.argmax(axis=1), avg)
            rows.append(metric_row(args.name, f"ptb_synthetic_{split}", rep))
            print(f"{args.name} ptb_{split}: acc={rep['acc']:.6f} recalls={rep['good_recall']:.3f}/{rep['medium_recall']:.3f}/{rep['bad_recall']:.3f}", flush=True)
    else:
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
        if not policies:
            raise ValueError("--policies required for clean mode")
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        for policy in policies:
            for split in splits:
                summed = None
                y_ref = None
                for weight, ckpt in zip(weights, ckpts):
                    ds = clean_dataset(policy, split, ckpt)
                    probs, y = predict_probs(ckpt, ds, int(args.batch_size), device)
                    weighted = probs * float(weight)
                    summed = weighted if summed is None else summed + weighted
                    if y_ref is None:
                        y_ref = y
                    elif not np.array_equal(y_ref, y):
                        raise RuntimeError(f"label order mismatch for clean {policy} {split}")
                avg = summed
                rep = DUAL.GEOM.metric_report(y_ref, avg.argmax(axis=1), avg)
                rows.append(metric_row(args.name, f"{policy}_{split}", rep))
                print(f"{args.name} {policy}_{split}: acc={rep['acc']:.6f} recalls={rep['good_recall']:.3f}/{rep['medium_recall']:.3f}/{rep['bad_recall']:.3f}", flush=True)

    metrics = pd.DataFrame(rows)
    out_dir = ANALYSIS_DIR / "waveform_checkpoint_ensemble_eval"
    report_dir = REPORT_DIR / "waveform_checkpoint_ensemble_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in args.name)[:140]
    metrics_path = out_dir / f"{safe}_metrics.csv"
    report_path = out_dir / f"{safe}_report.md"
    summary_path = out_dir / f"{safe}_summary.json"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": args.name,
        "mode": args.mode,
        "checkpoints": [str(p) for p in paths],
        "weights": [float(w) for w in weights],
        "metrics_csv": str(metrics_path),
        "report": str(report_dir / report_path.name),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    (report_dir / report_path.name).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
