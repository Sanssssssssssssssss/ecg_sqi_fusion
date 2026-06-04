"""Continue BUT 10s synthetic generator tuning around the medium/bad boundary.

This runner is intentionally generator-first: it creates new PTB synthetic
variants and trains the Uformer mainline recipe on each variant.  Head-only BUT
adaptation is treated as diagnostic evidence only.
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
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT, load_ptb_artifact
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_refine import b10_base
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    apply_bad_spec,
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


RUN_TAG = "e311_but_generator_medium_bad_grid_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def medium_bad_entries() -> list[dict[str, Any]]:
    """New generator specs after the b10/r08/head-only diagnostics.

    The grid deliberately varies data rules, not final heads:
    - preserve medium as QRS-detectable but detail-unreliable
    - make bad mostly unreliable-QRS/contact-loss rather than merely noisy
    - avoid pushing every low-amplitude segment into bad
    """

    base = b10_base()
    variants: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "m01_medium_detail_bad_contact_bal",
            {
                "qrs_attn_medium": 0.11,
                "tst_noise_medium": 0.22,
                "drift_medium": 0.09,
                "hf_medium": 0.08,
                "qrs_attn_bad": 0.66,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.18,
                "baseline_step_bad": 0.22,
                "clip_bad": 0.14,
                "spurious_peak_bad": 0.12,
                "low_amp_bad": 0.62,
            },
            "1,1.58,1.72",
            "Medium detail corruption with softened b10 bad.",
        ),
        (
            "m02_medium_stronger_bad_qrs_missing",
            {
                "qrs_attn_medium": 0.13,
                "tst_noise_medium": 0.24,
                "drift_medium": 0.10,
                "hf_medium": 0.08,
                "qrs_attn_bad": 0.78,
                "contact_loss_bad": 0.38,
                "flatline_bad": 0.22,
                "dropout_bad": 0.18,
                "baseline_step_bad": 0.18,
                "clip_bad": 0.08,
                "spurious_peak_bad": 0.08,
                "low_amp_bad": 0.64,
            },
            "1,1.60,1.72",
            "Bad is QRS unreliable; medium is locally unreliable but detectable.",
        ),
        (
            "m03_bad_contact_no_spurious",
            {
                "qrs_attn_medium": 0.10,
                "tst_noise_medium": 0.20,
                "drift_medium": 0.08,
                "hf_medium": 0.07,
                "qrs_attn_bad": 0.72,
                "contact_loss_bad": 0.42,
                "flatline_bad": 0.25,
                "dropout_bad": 0.22,
                "baseline_step_bad": 0.26,
                "clip_bad": 0.08,
                "spurious_peak_bad": 0.0,
                "burst_bad": 0.10,
                "low_amp_bad": 0.66,
            },
            "1,1.58,1.78",
            "Remove spurious peaks to avoid confusing medium with artificial bad.",
        ),
        (
            "m04_bad_lowamp_guarded",
            {
                "qrs_attn_medium": 0.09,
                "tst_noise_medium": 0.20,
                "drift_medium": 0.08,
                "hf_medium": 0.07,
                "qrs_attn_bad": 0.70,
                "contact_loss_bad": 0.32,
                "flatline_bad": 0.16,
                "baseline_step_bad": 0.22,
                "clip_bad": 0.18,
                "spurious_peak_bad": 0.12,
                "low_amp_bad": 0.70,
            },
            "1,1.55,1.70",
            "Low-amplitude domain shift without making bad only low amplitude.",
        ),
        (
            "m05_good_lenient_medium_gap",
            {
                "residual_scale_good": 0.90,
                "residual_scale_medium": 1.15,
                "qrs_attn_medium": 0.12,
                "tst_noise_medium": 0.24,
                "drift_medium": 0.10,
                "hf_medium": 0.09,
                "qrs_attn_bad": 0.68,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.18,
                "baseline_step_bad": 0.22,
                "spurious_peak_bad": 0.12,
                "low_amp_bad": 0.64,
            },
            "1,1.62,1.70",
            "BUT good is not pristine; widen good-medium realism.",
        ),
        (
            "m06_medium_motion_bad_contact",
            {
                "qrs_attn_medium": 0.08,
                "tst_noise_medium": 0.18,
                "drift_medium": 0.12,
                "hf_medium": 0.09,
                "qrs_attn_bad": 0.74,
                "contact_loss_bad": 0.40,
                "flatline_bad": 0.24,
                "dropout_bad": 0.18,
                "baseline_step_bad": 0.32,
                "clip_bad": 0.12,
                "spurious_peak_bad": 0.08,
                "burst_bad": 0.14,
                "low_amp_bad": 0.62,
            },
            "1,1.55,1.80",
            "Wearable motion boundary: medium drift/HF, bad contact loss.",
        ),
        (
            "m07_bad_extreme_medium_protected",
            {
                "qrs_attn_medium": 0.06,
                "tst_noise_medium": 0.16,
                "drift_medium": 0.07,
                "hf_medium": 0.06,
                "qrs_attn_bad": 0.84,
                "contact_loss_bad": 0.48,
                "flatline_bad": 0.32,
                "dropout_bad": 0.25,
                "baseline_step_bad": 0.26,
                "clip_bad": 0.10,
                "spurious_peak_bad": 0.06,
                "low_amp_bad": 0.60,
            },
            "1,1.48,1.78",
            "A stricter bad class while protecting medium from over-damage.",
        ),
        (
            "m08_medium_qrs_visible_bad_flat",
            {
                "qrs_attn_medium": 0.07,
                "tst_noise_medium": 0.25,
                "drift_medium": 0.10,
                "hf_medium": 0.10,
                "qrs_attn_bad": 0.76,
                "contact_loss_bad": 0.36,
                "flatline_bad": 0.36,
                "dropout_bad": 0.22,
                "baseline_step_bad": 0.24,
                "clip_bad": 0.08,
                "spurious_peak_bad": 0.04,
                "low_amp_bad": 0.65,
            },
            "1,1.60,1.75",
            "Medium QRS stays visible; bad includes flat/contact loss.",
        ),
        (
            "m09_bad_burst_qrs_unreliable",
            {
                "qrs_attn_medium": 0.10,
                "tst_noise_medium": 0.20,
                "drift_medium": 0.08,
                "hf_medium": 0.08,
                "qrs_attn_bad": 0.78,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.14,
                "dropout_bad": 0.12,
                "baseline_step_bad": 0.18,
                "clip_bad": 0.08,
                "spurious_peak_bad": 0.22,
                "burst_bad": 0.24,
                "hf_bad": 0.20,
                "low_amp_bad": 0.66,
            },
            "1,1.58,1.78",
            "Noisy unusable bad, but still keyed to unreliable QRS.",
        ),
        (
            "m10_b10_medium_prior",
            {
                "qrs_attn_medium": 0.14,
                "tst_noise_medium": 0.26,
                "drift_medium": 0.10,
                "hf_medium": 0.10,
                "qrs_attn_bad": 0.66,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.20,
                "baseline_step_bad": 0.24,
                "clip_bad": 0.16,
                "spurious_peak_bad": 0.12,
                "low_amp_bad": 0.64,
            },
            "1,1.70,1.65",
            "Class pressure favors medium recovery while retaining b10-style bad.",
        ),
        (
            "m11_medium_bad_snr_overlap",
            {
                "residual_scale_medium": 1.22,
                "qrs_attn_medium": 0.11,
                "tst_noise_medium": 0.24,
                "drift_medium": 0.11,
                "hf_medium": 0.09,
                "qrs_attn_bad": 0.70,
                "contact_loss_bad": 0.34,
                "flatline_bad": 0.18,
                "baseline_step_bad": 0.22,
                "clip_bad": 0.14,
                "spurious_peak_bad": 0.10,
                "low_amp_bad": 0.66,
            },
            "1,1.62,1.68",
            "Let medium have worse SNR without losing QRS reliability.",
        ),
        (
            "m12_bad_contact_step_medium_cleaner",
            {
                "qrs_attn_medium": 0.05,
                "tst_noise_medium": 0.17,
                "drift_medium": 0.08,
                "hf_medium": 0.06,
                "qrs_attn_bad": 0.74,
                "contact_loss_bad": 0.44,
                "flatline_bad": 0.30,
                "dropout_bad": 0.18,
                "baseline_step_bad": 0.36,
                "clip_bad": 0.10,
                "spurious_peak_bad": 0.04,
                "low_amp_bad": 0.62,
            },
            "1,1.50,1.82",
            "A cleaner medium / contact-step bad contrast.",
        ),
    ]
    out: list[dict[str, Any]] = []
    for name, changes, class_weight, note in variants:
        spec = dict(base)
        spec.update(changes)
        spec["id"] = name
        spec["family"] = "medium_bad_generator_10s"
        out.append({"spec": spec, "class_weight": class_weight, "note": note})
    return out


def result_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    ptb = row.get("ptb_test_report", {})
    den = row.get("ptb_denoise_metrics", {})
    # Select for medium/bad coexistence, then balanced/macro-F1.
    medium_bad_floor = min(float(rec[1]), float(rec[2]))
    return (
        medium_bad_floor,
        float(rep.get("balanced_acc", 0.0)),
        float(rep.get("macro_f1", 0.0)),
        float(rec[2]),
        float(ptb.get("acc", 0.0)) + 0.01 * float(den.get("denoise_score", 0.0)),
    )


def prepare_variant(args: argparse.Namespace, entry: dict[str, Any]) -> Path:
    spec = entry["spec"]
    variant_dir = Path(args.out_root) / "synthetic_variants" / str(spec["id"])
    if (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists():
        return variant_dir
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    ptb["source"] = str(args.source_artifact_dir)
    payload = apply_bad_spec(ptb, spec, variant_dir, seed=int(args.seed), sample_limit=0)
    plot_synthetic_gallery(
        ptb["clean"],
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
        "note": entry["note"],
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
    write_json(run_dir / "medium_bad_generator_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / "medium_bad_generator_summary.jsonl", payload)
    return payload


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        summary_path = out_root / "medium_bad_generator_summary.jsonl"
        rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if summary_path.exists() else []
    ranked = sorted(rows, key=result_key, reverse=True)
    lines = [
        "# BUT 10s Medium/Bad Generator Continuation",
        "",
        "This pass continues data generation rather than head-only adaptation.  The formal BUT protocol remains 10s P1, and calibration is validation-only.",
        "",
        "Primary reading: a useful synthetic rule should keep bad recall high without swallowing medium.",
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If medium rises while bad stays above 0.85, continue this generator family.",
            "- If bad is high but medium collapses, the rule is still over-bad-biased.",
            "- If PTB acc or bad recall collapses, keep the rule as external-only diagnostic evidence.",
        ]
    )
    (report_root / "medium_bad_generator_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "medium_bad_generator_summary.json", {"rows": rows, "ranked": ranked[:50]})


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "medium_bad_generator_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    entries = medium_bad_entries()
    write_json(out_root / "medium_bad_generator_specs.json", {"entries": entries})
    rows: list[dict[str, Any]] = []
    if args.stage in {"all", "quick_train"}:
        for i, entry in enumerate(entries[: int(args.max_runs)], start=1):
            update_state(
                state_path,
                status="quick_training",
                current=entry["spec"]["id"],
                index=i,
                total=min(len(entries), int(args.max_runs)),
                updated_at=now_iso(),
            )
            rows.append(train_one(args, entry))
            write_report(args)
    if args.stage in {"all", "report"}:
        write_report(args, rows or None)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso(), n_runs=len(rows) if rows else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continue BUT 10s generator tuning for medium/bad boundary.")
    parser.add_argument("--stage", choices=("quick_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=12)
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
        update_state(Path(args.out_root) / "medium_bad_generator_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
