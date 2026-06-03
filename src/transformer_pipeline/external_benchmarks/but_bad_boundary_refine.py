"""Refine BUT 10s bad-boundary synthetic rules around the useful b10 anchor.

This runner keeps the formal BUT protocol fixed to 10 seconds and focuses on
the failure mode from the first bad-boundary grid: b10 improves BUT bad recall,
but medium/good separation is still too blunt.  The refinement grid therefore
does not add arbitrary noise; it changes medium-local damage, wearable bad
mixtures, and bad class pressure separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import (
    ROOT,
    load_ptb_artifact,
)
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    apply_bad_spec,
    append_jsonl,
    evaluate_checkpoint_10s,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)


RUN_TAG = "e311_but_bad_boundary_refine_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG


def b10_base() -> dict[str, Any]:
    return {
        "qrs_attn_medium": 0.05,
        "qrs_attn_bad": 0.70,
        "tst_noise_medium": 0.10,
        "hf_medium": 0.04,
        "hf_bad": 0.16,
        "drift_medium": 0.05,
        "drift_bad": 0.12,
        "burst_bad": 0.18,
        "low_amp_medium": 0.94,
        "low_amp_bad": 0.56,
        "residual_scale_good": 0.70,
        "residual_scale_medium": 1.05,
        "residual_scale_bad": 1.30,
        "smooth_bad": 0.10,
        "contact_loss_bad": 0.36,
        "flatline_bad": 0.25,
        "dropout_bad": 0.0,
        "baseline_step_bad": 0.30,
        "clip_bad": 0.25,
        "spurious_peak_bad": 0.28,
        "family": "bad_boundary_refine_10s",
    }


def refined_runs() -> list[dict[str, Any]]:
    """Return spec + training pressure combos for the second bad-boundary pass."""
    base = b10_base()
    variants: list[tuple[str, dict[str, Any], str]] = [
        ("r01_b10_cw170", {}, "1,1.45,1.70"),
        ("r02_b10_cw180", {}, "1,1.45,1.80"),
        (
            "r03_b10_medium_detail",
            {"qrs_attn_medium": 0.09, "tst_noise_medium": 0.18, "drift_medium": 0.08, "hf_medium": 0.07},
            "1,1.55,1.75",
        ),
        (
            "r04_b10_less_bad_bias",
            {
                "qrs_attn_bad": 0.62,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.18,
                "baseline_step_bad": 0.22,
                "clip_bad": 0.18,
                "spurious_peak_bad": 0.18,
                "low_amp_bad": 0.62,
            },
            "1,1.45,1.75",
        ),
        (
            "r05_b10_dropout_medium_guard",
            {
                "qrs_attn_medium": 0.08,
                "tst_noise_medium": 0.17,
                "drift_medium": 0.08,
                "dropout_bad": 0.20,
                "baseline_step_bad": 0.25,
                "spurious_peak_bad": 0.12,
            },
            "1,1.55,1.75",
        ),
        (
            "r06_b04_soft_medium_guard",
            {
                "qrs_attn_medium": 0.09,
                "tst_noise_medium": 0.18,
                "drift_medium": 0.08,
                "qrs_attn_bad": 0.55,
                "contact_loss_bad": 0.20,
                "flatline_bad": 0.10,
                "dropout_bad": 0.28,
                "baseline_step_bad": 0.24,
                "clip_bad": 0.05,
                "spurious_peak_bad": 0.0,
                "burst_bad": 0.10,
                "low_amp_bad": 0.68,
            },
            "1,1.55,1.70",
        ),
        (
            "r07_b10_good_lenient",
            {"residual_scale_good": 0.88, "residual_scale_medium": 1.08, "qrs_attn_bad": 0.66, "low_amp_bad": 0.60},
            "1,1.45,1.70",
        ),
        (
            "r08_b10_bad_prior_mild",
            {"qrs_attn_bad": 0.66, "contact_loss_bad": 0.32, "flatline_bad": 0.22, "spurious_peak_bad": 0.20},
            "1,1.42,1.60",
        ),
    ]
    out = []
    for name, changes, class_weight in variants:
        spec = dict(base)
        spec.update(changes)
        spec["id"] = name
        out.append({"spec": spec, "class_weight": class_weight})
    return out


def result_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    ptb = row.get("ptb_test_report", {})
    den = row.get("ptb_denoise_metrics", {})
    # Bad recall is the focus, but medium must stay alive.
    return (
        float(rep.get("balanced_acc", 0.0)),
        float(rec[2]),
        float(rec[1]),
        float(ptb.get("acc", 0.0)) + 0.01 * float(den.get("denoise_score", 0.0)),
    )


def prepare_variant(args: argparse.Namespace, entry: dict[str, Any]) -> Path:
    spec = entry["spec"]
    variant_dir = Path(args.out_root) / "synthetic_variants" / spec["id"]
    if (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists():
        return variant_dir
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    ptb["source"] = str(args.source_artifact_dir)
    payload = apply_bad_spec(ptb, spec, variant_dir, seed=int(args.seed), sample_limit=0)
    clean = ptb["clean"].astype(np.float32, copy=False)
    plot_synthetic_gallery(
        clean,
        payload["X_noisy"],
        payload["labels"].reset_index(drop=True),
        variant_dir / "visuals" / "synthetic_gallery.png",
        str(spec["id"]),
    )
    return variant_dir


def train_one(args: argparse.Namespace, entry: dict[str, Any]) -> dict[str, Any]:
    spec = entry["spec"]
    spec_id = str(spec["id"])
    variant_dir = prepare_variant(args, entry)
    run_dir = Path(args.out_root) / "runs" / "quick" / spec_id
    log_dir = Path(args.out_root) / "logs" / "quick"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.transformer_pipeline.train_uformer_mainline",
        "--stage",
        "all",
        "--source_artifact_dir",
        str(variant_dir),
        "--output_dir",
        str(run_dir),
        "--epochs_stage1",
        str(args.quick_epochs_stage1),
        "--epochs_stage2",
        str(args.quick_epochs_stage2),
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
    with (log_dir / f"{spec_id}.stdout.txt").open("w", encoding="utf-8") as out, (
        log_dir / f"{spec_id}.stderr.txt"
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    payload: dict[str, Any] = {
        "spec": spec,
        "class_weight": entry["class_weight"],
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "bad_boundary_refine_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / "bad_boundary_refine_summary.jsonl", payload)
    return payload


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        summary_path = out_root / "bad_boundary_refine_summary.jsonl"
        rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if summary_path.exists() else []
    ranked = sorted(rows, key=result_key, reverse=True)
    lines = [
        "# BUT 10s Bad-Boundary Refinement",
        "",
        "This pass refines the first grid's useful `b10_all_bad_wearable` anchor.  The goal is to keep BUT bad recall high while recovering medium/good separation.",
        "",
        "| rank | spec | class weight | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        den = row.get("ptb_denoise_metrics", {})
        lines.append(
            f"| {i} | {row.get('spec', {}).get('id')} | {row.get('class_weight')} | {row.get('returncode')} | "
            f"{float(rep.get('acc', 0.0)):.4f} | {float(rep.get('balanced_acc', 0.0)):.4f} | {float(rep.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} | "
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec)>2 else 0.0):.4f} | "
            f"{float(den.get('denoise_score', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `r01/r02` isolate class-weight pressure on the successful b10 rule.",
            "- `r03/r05/r06` make medium more explicitly 'detail unreliable but QRS detectable', so medium should stop being swallowed by bad.",
            "- `r04/r08` soften bad severity to test whether b10 is over-triggering good/medium.",
            "- `r07` makes good less artificially pristine, testing whether BUT good is being rejected because our synthetic good is too clean.",
        ]
    )
    (report_root / "bad_boundary_refine_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "bad_boundary_refine_summary.json", {"rows": rows, "ranked": ranked})


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "bad_boundary_refine_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    rows: list[dict[str, Any]] = []
    entries = refined_runs()
    write_json(out_root / "bad_boundary_refine_specs.json", {"entries": entries})
    if args.stage in {"all", "quick_train"}:
        for i, entry in enumerate(entries[: int(args.max_runs)], start=1):
            update_state(state_path, status="quick_training", current=entry["spec"]["id"], index=i, total=min(len(entries), int(args.max_runs)), updated_at=now_iso())
            rows.append(train_one(args, entry))
            write_report(args)
    if args.stage in {"all", "report"}:
        write_report(args, rows or None)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso(), n_runs=len(rows) if rows else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine BUT 10s bad-boundary synthetic rules around b10.")
    parser.add_argument("--stage", choices=("quick_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=8)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
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
        update_state(Path(args.out_root) / "bad_boundary_refine_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
