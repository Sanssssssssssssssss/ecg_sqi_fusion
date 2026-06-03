"""Full confirmation runs for the BUT 10s bad-boundary anchors.

The quick refinement pass showed that the original b10 mixed wearable rule is
still the best 10s bad-boundary anchor.  This runner performs a narrow full
recipe confirmation on b10 and the strongest softened alternative instead of
continuing a broad artifact grid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    evaluate_checkpoint_10s,
    now_iso,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)


RUN_TAG = "e311_but_bad_boundary_full_confirm_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG


def entries() -> list[dict[str, Any]]:
    return [
        {
            "id": "b10_all_bad_wearable_full_cw190",
            "source_artifact_dir": ROOT
            / "outputs"
            / "external_benchmarks"
            / "e311_but_bad_boundary_10s_2026_06_03"
            / "synthetic_variants"
            / "b10_all_bad_wearable",
            "class_weight": "1,1.40,1.90",
            "note": "First-grid winner; mixed wearable bad rule.",
        },
        {
            "id": "r08_bad_prior_mild_full_cw160",
            "source_artifact_dir": ROOT
            / "outputs"
            / "external_benchmarks"
            / "e311_but_bad_boundary_refine_10s_2026_06_03"
            / "synthetic_variants"
            / "r08_b10_bad_prior_mild",
            "class_weight": "1,1.42,1.60",
            "note": "Best refined softened bad-prior alternative.",
        },
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_one(args: argparse.Namespace, entry: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(args.out_root) / "runs" / entry["id"]
    log_dir = Path(args.out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(entry["source_artifact_dir"])
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source artifact for {entry['id']}: {source_dir}")
    cmd = [
        sys.executable,
        "-m",
        "src.transformer_pipeline.train_uformer_mainline",
        "--stage",
        "all",
        "--source_artifact_dir",
        str(source_dir),
        "--output_dir",
        str(run_dir),
        "--epochs_stage1",
        str(args.epochs_stage1),
        "--epochs_stage2",
        str(args.epochs_stage2),
        "--batch_size_stage1",
        str(args.batch_size_stage1),
        "--batch_size_stage2",
        str(args.batch_size_stage2),
        "--lambda_den_stage2",
        str(args.lambda_den_stage2),
        "--lambda_cls",
        str(args.lambda_cls),
        "--class_weight",
        str(entry["class_weight"]),
        "--seed",
        str(args.seed),
    ]
    start = time.perf_counter()
    with (log_dir / f"{entry['id']}.stdout.txt").open("w", encoding="utf-8") as out, (
        log_dir / f"{entry['id']}.stderr.txt"
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    payload: dict[str, Any] = {
        "id": entry["id"],
        "note": entry["note"],
        "class_weight": entry["class_weight"],
        "source_artifact_dir": str(source_dir),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "full_confirm_summary.json", payload)
    append_jsonl(Path(args.out_root) / "full_confirm_summary.jsonl", payload)
    return payload


def result_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    return (
        float(rep.get("balanced_acc", 0.0)),
        float(rep.get("macro_f1", 0.0)),
        float(rec[2]),
        float(rep.get("acc", 0.0)),
    )


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        path = out_root / "full_confirm_summary.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    ranked = sorted(rows, key=result_key, reverse=True)
    lines = [
        "# BUT 10s Bad-Boundary Full Confirmation",
        "",
        "Full recipe confirmation for the best quick-grid bad-boundary anchors.  Formal external protocol remains 10s P1; threshold calibration remains validation-only.",
        "",
        "| rank | id | class weight | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        den = row.get("ptb_denoise_metrics", {})
        lines.append(
            f"| {i} | {row.get('id')} | {row.get('class_weight')} | {row.get('returncode')} | "
            f"{float(rep.get('acc', 0.0)):.4f} | {float(rep.get('balanced_acc', 0.0)):.4f} | {float(rep.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} | "
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec)>2 else 0.0):.4f} | "
            f"{float(den.get('denoise_score', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If full b10 beats the quick b10 anchor or keeps bad recall high with better medium, continue b10-style generator adaptation.",
            "- If full b10 does not improve over quick b10, freeze the generator rule and switch to boundary calibration / BUT-supervised head-only adaptation.",
        ]
    )
    (report_root / "full_confirm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "full_confirm_summary.json", {"rows": rows, "ranked": ranked})


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "full_confirm_state.json"
    update_state(state_path, status="running", updated_at=now_iso())
    rows: list[dict[str, Any]] = []
    for i, entry in enumerate(entries()[: int(args.max_runs)], start=1):
        update_state(state_path, status="training", current=entry["id"], index=i, total=min(len(entries()), int(args.max_runs)), updated_at=now_iso())
        rows.append(train_one(args, entry))
        write_report(args)
    update_state(state_path, status="complete", updated_at=now_iso(), n_runs=len(rows))
    write_report(args, rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full confirmation for BUT 10s bad-boundary anchors.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=2)
    parser.add_argument("--epochs_stage1", type=int, default=10)
    parser.add_argument("--epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / "full_confirm_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
