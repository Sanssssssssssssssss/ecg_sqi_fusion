"""Evaluate waveform-only checkpoints on the PTB synthetic val/test splits.

This is read-only evaluation.  It is meant for the reverse cross-domain check:
train on clean BUT, test on PTB synthetic.  It also works for PTB-trained
checkpoints as a self-test sanity check.
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


ALIGN = load_module("ptb_align_for_ptb_eval", PTB_ALIGN_SCRIPT)
DUAL = ALIGN.DUAL
FORMAL_FEATURE_COLUMNS = ALIGN.FORMAL_FEATURE_COLUMNS


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def metric_row(candidate: str, split: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in rep.items() if k != "confusion_3x3"}
    row.update({"candidate": candidate, "bucket": f"ptb_synthetic_{split}"})
    row["confusion_3x3"] = json.dumps(rep.get("confusion_3x3", []), ensure_ascii=False)
    return row


def make_model(ckpt: dict[str, Any], in_ch: int, aux_dim: int) -> torch.nn.Module:
    cfg = dict(ckpt["candidate_config"])
    state = ckpt["model_state"]
    if any(str(k).startswith("base.") for k in state):
        model = ALIGN.StressAwareHierTransformer(cfg, in_ch, aux_dim)
    else:
        model = DUAL.HierTransformer(cfg, in_ch, aux_dim)
    model.load_state_dict(state)
    return model


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint On PTB Synthetic",
        "",
        "Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.",
        "",
        f"Checkpoint: `{payload['checkpoint']}`",
        "",
        "| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} |"
        )
    lines.extend(["", f"Metrics CSV: `{payload['metrics_csv']}`"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    channel_stats = DUAL.ChannelStats(**ckpt["channel_stats"])
    feature_norm = DUAL.FeatureNorm(
        mean=np.asarray(ckpt["feature_train_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_train_std"], dtype=np.float32),
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("no splits supplied")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = args.name.strip() or args.checkpoint.parent.name
    model: torch.nn.Module | None = None
    rows: list[dict[str, Any]] = []
    for split in splits:
        ds = ALIGN.BadAlignedSyntheticDualviewDataset(
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
        if model is None:
            model = make_model(ckpt, int(ds.x.shape[1]), len(FORMAL_FEATURE_COLUMNS)).to(device)
            model.eval()
        loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        rep, _, _, _ = DUAL.eval_loader(model, loader, device)
        rows.append(metric_row(candidate, split, rep))
        print(
            f"{candidate} ptb_{split}: acc={rep['acc']:.6f} "
            f"recalls={rep['good_recall']:.3f}/{rep['medium_recall']:.3f}/{rep['bad_recall']:.3f}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    out_dir = ANALYSIS_DIR / "ptb_synthetic_checkpoint_eval"
    report_dir = REPORT_DIR / "ptb_synthetic_checkpoint_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name.strip() or f"ptb_eval_{args.checkpoint.parent.name}"
    safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:140]
    metrics_path = out_dir / f"{safe_stem}_metrics.csv"
    summary_path = out_dir / f"{safe_stem}_summary.json"
    report_path = out_dir / f"{safe_stem}_report.md"
    report_copy = report_dir / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
