"""Run waveform Transformer sanity checks on materialized clean BUT protocols.

This wrapper reuses the existing waveform-only Transformer diagnostic runner
without changing its source. It points the runner at clean protocol bundles
created by ``build_clean_but_protocols.py`` and writes a separate clean-protocol
summary.

No variable-length modeling is used here.
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


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ORIG_SCRIPT = ANALYSIS_DIR / "run_waveform_transformer_original_adaptation.py"
PROTOCOL_ROOT = ANALYSIS_DIR / "clean_but_protocols"
OUT_DIR = ANALYSIS_DIR / "clean_but_protocol_transformer_sanity"
REPORT_OUT_DIR = REPORT_DIR / "clean_but_protocol_transformer_sanity"


def load_orig_module() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_transformer_original_adaptation_clean_sanity", ORIG_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ORIG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def metric_line(row: pd.Series) -> str:
    def f(name: str) -> float:
        try:
            return float(row[name])
        except Exception:
            return float("nan")

    return (
        f"| {row['policy']} | {row['candidate']} | {row['bucket']} | {f('acc'):.6f} | {f('macro_f1'):.6f} | "
        f"{f('good_recall'):.6f} | {f('medium_recall'):.6f} | {f('bad_recall'):.6f} |"
    )


def render_report(metrics: pd.DataFrame, summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Clean BUT Protocol Transformer Sanity",
        "",
        "Fixed-length 10s only. No variable-length model is used. Inputs remain waveform-only; SQI/geometry columns are auxiliary targets.",
        "",
        "Important caveat: after dropping `outlier_low_confidence`, the legacy validation split can contain very few bad rows, so bad-threshold calibration is diagnostic only.",
        "",
        "| Policy | Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(metric_line(row))
    lines.extend(["", "## Checkpoints", ""])
    for item in summaries:
        lines.append(
            f"- `{item['policy']}` / `{item['candidate']}`: best_epoch={item['best_epoch']}, "
            f"threshold={item['bad_threshold_trainval']:.2f}, checkpoint=`{item['checkpoint']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "- These runs test whether cleaning the fixed 10s protocol makes the waveform Transformer problem easier.",
            "- They do not replace full BUT stress evaluation.",
            "- Full/outlier stress must stay report-only and separate.",
            "- If clean policies improve sharply, the next model work should use clean-body training plus explicit stress-bucket reporting.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=str, default="margin_ge_2s_drop_outlier,margin_ge_5s_drop_outlier")
    parser.add_argument("--candidate", type=str, default="orig_convtx_robust3_aux")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    ORIG = load_orig_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        protocol_dir = PROTOCOL_ROOT / policy
        if not protocol_dir.exists():
            raise FileNotFoundError(f"Missing clean protocol: {protocol_dir}")
        if args.candidate not in ORIG.CANDIDATES:
            raise ValueError(f"Unknown candidate {args.candidate}; expected {sorted(ORIG.CANDIDATES)}")
        ORIG.SIGNALS_PATH = protocol_dir / "signals.npz"
        ORIG.ORIGINAL_ATLAS = protocol_dir / "original_region_atlas.csv"
        ORIG.NODE_ID = f"N17043_clean_{policy}"
        cfg = dict(ORIG.CANDIDATES[args.candidate])
        cfg["seed"] = int(cfg.get("seed", 20260683)) + abs(hash(policy)) % 10000
        run_args = argparse.Namespace(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)
        candidate_name = f"{policy}_{args.candidate}"
        summary = ORIG.train_candidate(candidate_name, cfg, run_args)
        summary["policy"] = policy
        summaries.append(summary)
        for row in summary["metric_rows"]:
            row = dict(row)
            row["policy"] = policy
            all_rows.append(row)

    metrics = pd.DataFrame(all_rows)
    metrics_path = OUT_DIR / "clean_but_protocol_transformer_sanity_metrics.csv"
    summary_path = OUT_DIR / "clean_but_protocol_transformer_sanity_summary.json"
    report_path = OUT_DIR / "clean_but_protocol_transformer_sanity_report.md"
    report_copy = REPORT_OUT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policies": [s["policy"] for s in summaries],
        "candidate": args.candidate,
        "epochs": int(args.epochs),
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "summaries": summaries,
    }
    write_json(summary_path, payload)
    report = render_report(metrics, summaries)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
