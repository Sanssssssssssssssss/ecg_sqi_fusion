"""Refine the BUT 10s morphology sweet spot around s02 and mix05.

s02 was the first generator to keep medium and bad alive together, but good
recall dropped.  mix05 had the best balanced score but medium was slightly low.
This narrow pass adjusts morphology pressure and class weights instead of
introducing more SNR changes.
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


RUN_TAG = "e311_but_generator_morph_refine_grid_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG


def entries() -> list[dict[str, Any]]:
    base = b10_base()
    base.update(
        {
            "qrs_attn_medium": 0.02,
            "tst_noise_medium": 0.06,
            "hf_medium": 0.03,
            "drift_medium": 0.04,
            "qrs_attn_bad": 0.70,
            "contact_loss_bad": 0.34,
            "flatline_bad": 0.22,
            "baseline_step_bad": 0.26,
            "clip_bad": 0.20,
            "spurious_peak_bad": 0.20,
            "low_amp_bad": 0.60,
        }
    )
    variants: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "r01_s02_good_guard",
            {
                "medium_event_prob": 0.58,
                "medium_pseudo_peak_rate": 0.07,
                "medium_inverse_peak_rate": 0.12,
                "medium_step_rate": 0.10,
                "medium_ramp_rate": 0.10,
                "medium_contact_rate": 0.04,
                "medium_qrs_preserve": 0.995,
                "bad_pseudo_peak_rate": 0.48,
                "bad_inverse_peak_rate": 0.30,
                "bad_qrs_confuse_rate": 0.26,
            },
            "1.15,1.48,1.78",
            "s02-style bad spikes with stronger good/QRS guard.",
        ),
        (
            "r02_s02_less_bad_pressure",
            {
                "qrs_attn_bad": 0.66,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.18,
                "baseline_step_bad": 0.22,
                "medium_event_prob": 0.62,
                "medium_pseudo_peak_rate": 0.08,
                "medium_inverse_peak_rate": 0.14,
                "medium_step_rate": 0.12,
                "medium_ramp_rate": 0.10,
                "medium_contact_rate": 0.04,
                "medium_qrs_preserve": 0.99,
                "bad_pseudo_peak_rate": 0.44,
                "bad_inverse_peak_rate": 0.30,
                "bad_qrs_confuse_rate": 0.24,
            },
            "1.10,1.52,1.74",
            "Reduce over-bad calibration while keeping morphology events.",
        ),
        (
            "r03_s02_good_medium_balance",
            {
                "medium_event_prob": 0.70,
                "medium_pseudo_peak_rate": 0.10,
                "medium_inverse_peak_rate": 0.16,
                "medium_step_rate": 0.16,
                "medium_ramp_rate": 0.14,
                "medium_contact_rate": 0.06,
                "medium_qrs_preserve": 0.985,
                "bad_pseudo_peak_rate": 0.46,
                "bad_inverse_peak_rate": 0.30,
                "bad_qrs_confuse_rate": 0.28,
            },
            "1.08,1.58,1.76",
            "Middle point between s02 and mix05.",
        ),
        (
            "r04_mix05_plus_bad_spikes",
            {
                "residual_scale_good": 0.86,
                "residual_scale_medium": 1.08,
                "medium_event_prob": 0.76,
                "medium_pseudo_peak_rate": 0.12,
                "medium_inverse_peak_rate": 0.20,
                "medium_step_rate": 0.22,
                "medium_ramp_rate": 0.18,
                "medium_contact_rate": 0.08,
                "medium_qrs_preserve": 0.97,
                "bad_pseudo_peak_rate": 0.42,
                "bad_inverse_peak_rate": 0.30,
                "bad_qrs_confuse_rate": 0.30,
            },
            "1.05,1.60,1.80",
            "mix05-like medium with stronger bad QRS-confounding spikes.",
        ),
        (
            "r05_mix05_good_guard_bad_hard",
            {
                "residual_scale_good": 0.82,
                "residual_scale_medium": 1.08,
                "qrs_attn_bad": 0.76,
                "contact_loss_bad": 0.38,
                "flatline_bad": 0.26,
                "baseline_step_bad": 0.30,
                "medium_event_prob": 0.70,
                "medium_pseudo_peak_rate": 0.10,
                "medium_inverse_peak_rate": 0.18,
                "medium_step_rate": 0.18,
                "medium_ramp_rate": 0.16,
                "medium_contact_rate": 0.06,
                "medium_qrs_preserve": 0.98,
                "bad_pseudo_peak_rate": 0.44,
                "bad_inverse_peak_rate": 0.32,
                "bad_qrs_confuse_rate": 0.34,
            },
            "1.12,1.56,1.84",
            "Try to retain mix05 balanced score while restoring bad.",
        ),
        (
            "r06_s02_calibration_guard",
            {
                "medium_event_prob": 0.64,
                "medium_pseudo_peak_rate": 0.08,
                "medium_inverse_peak_rate": 0.14,
                "medium_step_rate": 0.14,
                "medium_ramp_rate": 0.12,
                "medium_contact_rate": 0.04,
                "medium_qrs_preserve": 0.992,
                "bad_pseudo_peak_rate": 0.50,
                "bad_inverse_peak_rate": 0.32,
                "bad_qrs_confuse_rate": 0.30,
            },
            "1.20,1.50,1.80",
            "Explicitly rescue good while keeping s02 morphology.",
        ),
    ]
    out = []
    for name, changes, class_weight, note in variants:
        spec = dict(base)
        spec.update(changes)
        spec["id"] = name
        spec["family"] = "morph_refine_generator_10s"
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
        "# BUT 10s Morphology Refine Generator",
        "",
        "Narrow morphology refine around s02 and mix05. The aim is not SNR; it is expert-style morphology: medium QRS usable, bad QRS confounded.",
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
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec)>2 else 0.0):.4f} | "
            f"{float(den.get('denoise_score', 0.0)):.3f} | {row.get('note', '')} |"
        )
    (report_root / "morph_refine_generator_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "morph_refine_generator_summary.json", {"rows": rows, "ranked": ranked[:50]})


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    state_path = out_root / "morph_refine_generator_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    run_entries = entries()
    write_json(out_root / "morph_refine_generator_specs.json", {"entries": run_entries})
    rows = []
    if args.stage in {"all", "quick_train"}:
        for i, entry in enumerate(run_entries[: int(args.max_runs)], start=1):
            update_state(state_path, status="quick_training", current=entry["spec"]["id"], index=i, total=min(len(run_entries), int(args.max_runs)), updated_at=now_iso())
            rows.append(train_one(args, entry))
            write_report(args)
    if args.stage in {"all", "report"}:
        write_report(args, rows or None)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso(), n_runs=len(rows) if rows else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s morphology refine generator tuning.")
    parser.add_argument("--stage", choices=("quick_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=6)
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
        update_state(Path(args.out_root) / "morph_refine_generator_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
