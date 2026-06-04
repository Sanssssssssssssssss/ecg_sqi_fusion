"""BUT 10s morphology v3 generator search.

V2 confirmed the failure mode: stronger medium morphology tends to steal good
or weaken bad, while stronger bad pseudo-QRS collapses medium.  V3 is a smaller
visual-evidence pass: keep b10 as the anchor, avoid SNR changes, reduce
flatline reliance, and test only narrow morphology-margin changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_refine import b10_base
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    now_iso,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_generator_morph_sweet_grid import train_one
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)


RUN_TAG = "e311_but_generator_morph_v3_grid_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG


def entries() -> list[dict[str, Any]]:
    base = b10_base()
    base.update(
        {
            "qrs_attn_medium": 0.025,
            "tst_noise_medium": 0.065,
            "hf_medium": 0.032,
            "drift_medium": 0.042,
            "qrs_attn_bad": 0.70,
            "contact_loss_bad": 0.34,
            "flatline_bad": 0.18,
            "baseline_step_bad": 0.28,
            "clip_bad": 0.20,
            "spurious_peak_bad": 0.20,
            "low_amp_medium": 0.94,
            "low_amp_bad": 0.60,
        }
    )
    variants: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "v301_b10_micro_morph",
            {
                "residual_scale_good": 0.72,
                "residual_scale_medium": 1.03,
                "medium_event_prob": 0.48,
                "medium_pseudo_peak_rate": 0.05,
                "medium_inverse_peak_rate": 0.10,
                "medium_step_rate": 0.08,
                "medium_ramp_rate": 0.08,
                "medium_contact_rate": 0.025,
                "medium_qrs_preserve": 0.998,
                "bad_pseudo_peak_rate": 0.34,
                "bad_inverse_peak_rate": 0.24,
                "bad_qrs_confuse_rate": 0.26,
            },
            "1.00,1.54,1.82",
            "Conservative b10 continuation: tiny local morphology, do not disturb good.",
        ),
        (
            "v302_s02_bad_with_medium_floor",
            {
                "residual_scale_good": 0.78,
                "residual_scale_medium": 1.04,
                "medium_event_prob": 0.60,
                "medium_pseudo_peak_rate": 0.07,
                "medium_inverse_peak_rate": 0.13,
                "medium_step_rate": 0.12,
                "medium_ramp_rate": 0.10,
                "medium_contact_rate": 0.035,
                "medium_qrs_preserve": 0.995,
                "bad_pseudo_peak_rate": 0.50,
                "bad_inverse_peak_rate": 0.32,
                "bad_qrs_confuse_rate": 0.32,
            },
            "1.08,1.58,1.84",
            "S02 bad cue with a medium floor and less good pressure.",
        ),
        (
            "v303_mix05_medium_with_bad_floor",
            {
                "residual_scale_good": 0.72,
                "residual_scale_medium": 1.05,
                "medium_event_prob": 0.66,
                "medium_pseudo_peak_rate": 0.09,
                "medium_inverse_peak_rate": 0.16,
                "medium_step_rate": 0.18,
                "medium_ramp_rate": 0.14,
                "medium_contact_rate": 0.055,
                "medium_qrs_preserve": 0.988,
                "bad_pseudo_peak_rate": 0.48,
                "bad_inverse_peak_rate": 0.32,
                "bad_qrs_confuse_rate": 0.34,
                "contact_loss_bad": 0.36,
                "baseline_step_bad": 0.30,
            },
            "1.04,1.60,1.86",
            "Mix05-style medium, but without good-overlap and with bad floor restored.",
        ),
        (
            "v304_no_flatline_pseudo_qrs_bad",
            {
                "residual_scale_good": 0.74,
                "residual_scale_medium": 1.03,
                "flatline_bad": 0.08,
                "contact_loss_bad": 0.26,
                "baseline_step_bad": 0.38,
                "clip_bad": 0.14,
                "medium_event_prob": 0.56,
                "medium_pseudo_peak_rate": 0.06,
                "medium_inverse_peak_rate": 0.12,
                "medium_step_rate": 0.12,
                "medium_ramp_rate": 0.12,
                "medium_contact_rate": 0.03,
                "medium_qrs_preserve": 0.996,
                "bad_pseudo_peak_rate": 0.58,
                "bad_inverse_peak_rate": 0.38,
                "bad_qrs_confuse_rate": 0.42,
            },
            "1.06,1.54,1.88",
            "BUT bad often has pseudo-QRS confusion rather than pure flatline.",
        ),
        (
            "v305_medium_guard_bad_soft",
            {
                "residual_scale_good": 0.76,
                "residual_scale_medium": 1.06,
                "qrs_attn_bad": 0.66,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.14,
                "baseline_step_bad": 0.30,
                "medium_event_prob": 0.74,
                "medium_pseudo_peak_rate": 0.10,
                "medium_inverse_peak_rate": 0.18,
                "medium_step_rate": 0.20,
                "medium_ramp_rate": 0.16,
                "medium_contact_rate": 0.06,
                "medium_qrs_preserve": 0.992,
                "bad_pseudo_peak_rate": 0.42,
                "bad_inverse_peak_rate": 0.28,
                "bad_qrs_confuse_rate": 0.30,
            },
            "1.08,1.66,1.78",
            "Medium guard: lift medium with softer bad pressure.",
        ),
    ]
    out: list[dict[str, Any]] = []
    for name, changes, class_weight, note in variants:
        spec = dict(base)
        spec.update(changes)
        spec["id"] = name
        spec["family"] = "morph_v3_generator_10s"
        out.append({"spec": spec, "class_weight": class_weight, "note": note})
    return out


def result_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    ptb = row.get("ptb_test_report", {})
    return (
        min(float(rec[1]), float(rec[2])),
        float(rep.get("balanced_acc", 0.0)),
        float(rep.get("macro_f1", 0.0)),
        float(rec[0]),
        float(ptb.get("acc", 0.0)),
    )


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "morph_sweet_generator_summary.jsonl"
    if rows is None:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    ranked = sorted(rows, key=result_key, reverse=True)
    lines = [
        "# BUT 10s Morphology V3 Generator",
        "",
        "Narrow follow-up after v2.  It keeps 10s P1 and validation-only calibration.",
        "",
        "| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
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
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec) > 2 else 0.0):.4f} | "
            f"{float(den.get('denoise_score', 0.0)):.3f} | {row.get('note', '')} |"
        )
    (report_root / "morph_v3_generator_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "morph_v3_generator_summary.json", {"rows": rows, "ranked": ranked[:50]})


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    state_path = out_root / "morph_v3_generator_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    run_entries = entries()
    write_json(out_root / "morph_v3_generator_specs.json", {"entries": run_entries})
    rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    summary_path = out_root / "morph_sweet_generator_summary.jsonl"
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                completed_ids.add(str(json.loads(line).get("spec", {}).get("id", "")))
            except json.JSONDecodeError:
                continue
    if args.stage in {"all", "quick_train"}:
        total = min(len(run_entries), int(args.max_runs))
        for i, entry in enumerate(run_entries[:total], start=1):
            if str(entry["spec"]["id"]) in completed_ids:
                continue
            update_state(state_path, status="quick_training", current=entry["spec"]["id"], index=i, total=total, updated_at=now_iso())
            rows.append(train_one(args, entry))
            write_report(args)
    if args.stage in {"all", "report"}:
        write_report(args, rows or None)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso(), n_runs=len(rows) if rows else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s morphology v3 generator tuning.")
    parser.add_argument("--stage", choices=("quick_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=5)
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
        update_state(Path(args.out_root) / "morph_v3_generator_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
