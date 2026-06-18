"""Evaluation-only balanced BUT 10s test reports.

The formal BUT 10s P1 test split is intentionally preserved, but its class
counts are highly uneven.  This utility adds a deterministic class-balanced
test subset report for already-trained runs without changing calibration,
training, or the original split files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    balanced_but_test_indices,
    ensure_but_10s,
    now_iso,
    read_json,
    update_state,
    write_balanced_but_index,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.run import (
    apply_but_thresholds,
    calibrate_but,
    load_uformer_model,
    multiclass_report,
    run_model_outputs,
)


DEFAULT_QRS_RUN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_real_noise_qrs_guided_10h_2026_06_04"
DEFAULT_QRS_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_real_noise_qrs_guided_10h_2026_06_04"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def existing_calibration(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "but_10s_eval" / "but_10s_eval_summary.json"
    if not summary_path.exists():
        return None
    payload = read_json(summary_path)
    cal = payload.get("calibration")
    if isinstance(cal, dict) and "t_good" in cal and "t_bad" in cal:
        return cal
    return None


def evaluate_balanced(args: argparse.Namespace, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    X, meta = ensure_but_10s(args)
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    balanced_idx, selection = balanced_but_test_indices(meta, seed=int(args.seed))
    out_dir = run_dir / "but_10s_eval"
    report_path = out_dir / "but_10s_balanced_test_report.json"
    if report_path.exists() and not args.force:
        return read_json(report_path)

    cal = existing_calibration(run_dir)
    val_idx = np.where(split == "val")[0]
    need_val = cal is None
    eval_idx = np.concatenate([val_idx, balanced_idx]) if need_val else balanced_idx
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_uformer_model(checkpoint, device, feature_set="full_tokens", with_head=True)
    outputs = run_model_outputs(model, X[eval_idx], int(args.batch_size_eval), device)
    probs_eval = outputs["probs"]
    if need_val:
        probs_val = probs_eval[: len(val_idx)]
        probs_balanced = probs_eval[len(val_idx) :]
        cal = calibrate_but(probs_val, y[val_idx])
    else:
        probs_balanced = probs_eval
    pred_raw = np.argmax(probs_balanced, axis=1).astype(np.int64)
    pred_cal = apply_but_thresholds(probs_balanced, float(cal["t_good"]), float(cal["t_bad"]))
    payload = {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "selection": selection,
        "calibration": cal,
        "raw_argmax_report": multiclass_report(y[balanced_idx], pred_raw, probs_balanced),
        "calibrated_report": multiclass_report(y[balanced_idx], pred_cal, probs_balanced),
        "elapsed_sec": float(outputs["elapsed_sec"]),
        "peak_cuda_memory_bytes": int(outputs["peak_cuda_memory_bytes"]),
        "generated_at": now_iso(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_path, payload)
    write_balanced_but_index(meta, balanced_idx, out_dir / "but_10s_balanced_test_index.csv")
    pred_df = meta.loc[balanced_idx, ["window_id", "record_id", "subject_id", "split", "start_sec", "end_sec", "y_class", "y"]].copy()
    pred_df["p_good"] = probs_balanced[:, 0]
    pred_df["p_medium"] = probs_balanced[:, 1]
    pred_df["p_bad"] = probs_balanced[:, 2]
    pred_df["pred_raw"] = pred_raw
    pred_df["pred_calibrated"] = pred_cal
    pred_df.to_csv(out_dir / "but_10s_balanced_test_predictions.csv", index=False)
    return payload


def update_run_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    run_root = Path(args.run_root)
    summary_path = run_root / args.summary_jsonl
    rows = load_jsonl(summary_path)
    if not rows:
        raise FileNotFoundError(f"No summary rows found at {summary_path}")
    state_path = run_root / "balanced_test_eval_state.json"
    update_state(state_path, status="running", updated_at=now_iso(), total=len(rows))
    out_rows: list[dict[str, Any]] = []
    done = 0
    for row in rows:
        run_dir_value = row.get("run_dir")
        if row.get("returncode") != 0 or not run_dir_value:
            out_rows.append(row)
            continue
        run_dir = Path(run_dir_value)
        checkpoint = run_dir / "ckpt_best.pt"
        if not checkpoint.exists():
            out_rows.append(row)
            continue
        if int(args.max_runs) > 0 and done >= int(args.max_runs):
            out_rows.append(row)
            continue
        update_state(state_path, status="running", current=str(run_dir.name), completed=done, total=len(rows), updated_at=now_iso())
        payload = evaluate_balanced(args, checkpoint, run_dir)
        row.setdefault("but_10s_eval", {})
        row["but_10s_eval"]["but_10s_balanced_test_report"] = payload["calibrated_report"]
        row["but_10s_eval"]["but_10s_balanced_test_selection"] = payload["selection"]
        row["but_10s_eval"]["but_10s_balanced_test_eval_path"] = str(run_dir / "but_10s_eval" / "but_10s_balanced_test_report.json")
        out_rows.append(row)
        done += 1
    if args.update_summary:
        backup = summary_path.with_suffix(summary_path.suffix + ".before_balanced_test.bak")
        if not backup.exists():
            backup.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_jsonl(summary_path, out_rows)
    update_state(state_path, status="complete", completed=done, total=len(rows), updated_at=now_iso())
    write_balanced_report(args, out_rows)
    return out_rows


def write_balanced_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        rep = row.get("but_10s_eval", {}).get("but_10s_balanced_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0, 0, 0])
        return (
            float(rep.get("macro_f1", 0.0)),
            float(rep.get("balanced_acc", 0.0)),
            min(float(rec[1]), float(rec[2])),
            float(rep.get("acc", 0.0)),
        )

    ranked = sorted(rows, key=key, reverse=True)
    lines = [
        "# BUT 10s Balanced-Test Evaluation",
        "",
        "This is an evaluation-only balanced subset of the formal BUT 10s P1 test split. It keeps validation-only calibration unchanged and samples the test split to equal class counts.",
        "",
        "| rank | mode | variant | balanced acc | balanced macro | recalls G/M/B | original acc | original macro | original recalls G/M/B |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for i, row in enumerate(ranked, start=1):
        bal = row.get("but_10s_eval", {}).get("but_10s_balanced_test_report", {})
        orig = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        b_rec = bal.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        o_rec = orig.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        if not bal:
            continue
        lines.append(
            f"| {i} | {row.get('mode')} | `{row.get('spec', {}).get('id')}` | "
            f"{float(bal.get('acc', 0.0)):.4f} | {float(bal.get('macro_f1', 0.0)):.4f} | "
            f"{float(b_rec[0]):.3f}/{float(b_rec[1]):.3f}/{float(b_rec[2]):.3f} | "
            f"{float(orig.get('acc', 0.0)):.4f} | {float(orig.get('macro_f1', 0.0)):.4f} | "
            f"{float(o_rec[0]):.3f}/{float(o_rec[1]):.3f}/{float(o_rec[2]):.3f} |"
        )
    (report_root / "but_10s_balanced_test_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "but_10s_balanced_test_summary.json", {"rows": ranked})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add deterministic class-balanced BUT 10s test metrics to existing runs.")
    parser.add_argument("--run_root", default=str(DEFAULT_QRS_RUN_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_QRS_REPORT_ROOT))
    parser.add_argument("--summary_jsonl", default="qrs_guided_training_summary.jsonl")
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--update_summary", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = update_run_summary(args)
    print(json.dumps({"status": "complete", "rows": len(rows), "report_root": args.report_root}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
