"""Large BUT 10s synthetic rule grid.

This runner consolidates the rule families discovered during BUT adaptation and
runs them under one comparable protocol.  It keeps the formal BUT protocol at
10s P1, uses validation-only calibration through the existing evaluator, and
separates quick, full, and seed-confirm stages.
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
    evaluate_checkpoint_10s,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_generator_morph_sweet_grid import apply_morph_spec
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)


RUN_TAG = "e311_but_large_rule_grid_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG

B10_ANCHOR = {
    "acc": 0.7735,
    "balanced_acc": 0.8045,
    "macro_f1": 0.7238,
    "recall_good_medium_bad": [0.824, 0.724, 0.866],
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def local_defaults() -> dict[str, float]:
    return {
        "medium_event_prob": 0.0,
        "medium_pseudo_peak_rate": 0.0,
        "medium_inverse_peak_rate": 0.0,
        "medium_step_rate": 0.0,
        "medium_ramp_rate": 0.0,
        "medium_contact_rate": 0.0,
        "medium_qrs_preserve": 0.99,
        "bad_event_prob": 1.0,
        "bad_pseudo_peak_rate": 0.0,
        "bad_inverse_peak_rate": 0.0,
        "bad_step_rate": 0.0,
        "bad_ramp_rate": 0.0,
        "bad_contact_rate": 0.0,
        "bad_qrs_confuse_rate": 0.0,
    }


def base_spec() -> dict[str, Any]:
    base = b10_base()
    base.update(local_defaults())
    return base


def make_entry(family: str, name: str, changes: dict[str, Any], class_weight: str, note: str) -> dict[str, Any]:
    spec = base_spec()
    spec.update(changes)
    spec["id"] = f"{family}_{name}"
    spec["family"] = family
    return {"spec": spec, "class_weight": class_weight, "note": note}


def entries() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Family 1: conservative b10 anchor variants.
    for i, (cw, bad_qrs, contact, flat, step, pseudo, good_scale) in enumerate(
        [
            ("1.00,1.54,1.82", 0.70, 0.36, 0.25, 0.30, 0.28, 0.70),
            ("1.04,1.54,1.84", 0.68, 0.34, 0.20, 0.28, 0.24, 0.74),
            ("1.08,1.56,1.84", 0.72, 0.38, 0.22, 0.32, 0.30, 0.76),
            ("1.12,1.52,1.86", 0.74, 0.34, 0.18, 0.34, 0.34, 0.78),
            ("1.00,1.58,1.80", 0.66, 0.32, 0.16, 0.26, 0.20, 0.72),
            ("1.06,1.60,1.82", 0.70, 0.30, 0.12, 0.36, 0.32, 0.78),
            ("1.10,1.50,1.88", 0.76, 0.40, 0.24, 0.30, 0.36, 0.82),
            ("1.04,1.62,1.78", 0.64, 0.26, 0.12, 0.24, 0.18, 0.76),
            ("1.12,1.58,1.82", 0.72, 0.34, 0.10, 0.40, 0.38, 0.78),
            ("1.06,1.56,1.88", 0.70, 0.42, 0.28, 0.24, 0.22, 0.72),
            ("1.16,1.52,1.80", 0.66, 0.30, 0.08, 0.34, 0.26, 0.84),
            ("1.00,1.50,1.90", 0.78, 0.44, 0.26, 0.34, 0.40, 0.72),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "b10_anchor_family",
                f"{i:02d}",
                {
                    "qrs_attn_bad": bad_qrs,
                    "contact_loss_bad": contact,
                    "flatline_bad": flat,
                    "baseline_step_bad": step,
                    "spurious_peak_bad": pseudo,
                    "residual_scale_good": good_scale,
                    "bad_pseudo_peak_rate": pseudo,
                    "bad_inverse_peak_rate": 0.20 + 0.4 * pseudo,
                    "bad_qrs_confuse_rate": 0.20 + 0.5 * pseudo,
                },
                cw,
                "Conservative perturbation around b10_all_bad_wearable.",
            )
        )

    # Family 2: medium is visibly imperfect, but QRS remains usable.
    for i, (event, inv, step, ramp, contact, preserve, cw) in enumerate(
        [
            (0.45, 0.08, 0.08, 0.08, 0.02, 0.998, "1.04,1.56,1.82"),
            (0.55, 0.10, 0.12, 0.10, 0.03, 0.996, "1.04,1.58,1.82"),
            (0.65, 0.12, 0.16, 0.12, 0.04, 0.994, "1.06,1.58,1.82"),
            (0.75, 0.16, 0.18, 0.16, 0.05, 0.990, "1.08,1.60,1.80"),
            (0.82, 0.20, 0.22, 0.18, 0.06, 0.985, "1.08,1.62,1.80"),
            (0.60, 0.18, 0.24, 0.20, 0.04, 0.992, "1.04,1.64,1.78"),
            (0.70, 0.22, 0.28, 0.24, 0.06, 0.980, "1.06,1.66,1.76"),
            (0.50, 0.14, 0.22, 0.14, 0.03, 0.996, "1.10,1.58,1.80"),
            (0.72, 0.10, 0.30, 0.26, 0.08, 0.988, "1.06,1.60,1.84"),
            (0.58, 0.24, 0.12, 0.26, 0.05, 0.990, "1.08,1.62,1.82"),
            (0.66, 0.18, 0.20, 0.18, 0.02, 0.996, "1.12,1.60,1.80"),
            (0.76, 0.26, 0.20, 0.22, 0.07, 0.982, "1.04,1.66,1.78"),
            (0.62, 0.16, 0.18, 0.32, 0.05, 0.990, "1.08,1.64,1.80"),
            (0.80, 0.12, 0.26, 0.16, 0.10, 0.980, "1.08,1.68,1.76"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "medium_qrs_visible_family",
                f"{i:02d}",
                {
                    "medium_event_prob": event,
                    "medium_pseudo_peak_rate": 0.06 + 0.08 * event,
                    "medium_inverse_peak_rate": inv,
                    "medium_step_rate": step,
                    "medium_ramp_rate": ramp,
                    "medium_contact_rate": contact,
                    "medium_qrs_preserve": preserve,
                    "residual_scale_good": 0.74,
                    "residual_scale_medium": 1.04,
                    "qrs_attn_bad": 0.68,
                    "contact_loss_bad": 0.32,
                    "flatline_bad": 0.16,
                    "baseline_step_bad": 0.28,
                    "bad_pseudo_peak_rate": 0.34,
                    "bad_inverse_peak_rate": 0.24,
                    "bad_qrs_confuse_rate": 0.26,
                },
                cw,
                "BUT class-2 style: visible QRS, unreliable P/T/ST or local baseline.",
            )
        )

    # Family 3: bad is QRS-unreliable, not just low amplitude/flatline.
    for i, (pseudo, inv, confuse, flat, contact, step, cw) in enumerate(
        [
            (0.36, 0.24, 0.24, 0.12, 0.28, 0.28, "1.04,1.54,1.86"),
            (0.44, 0.30, 0.30, 0.10, 0.30, 0.32, "1.04,1.56,1.88"),
            (0.52, 0.34, 0.36, 0.08, 0.28, 0.36, "1.06,1.56,1.90"),
            (0.60, 0.38, 0.42, 0.06, 0.24, 0.40, "1.08,1.54,1.92"),
            (0.46, 0.36, 0.46, 0.18, 0.34, 0.28, "1.06,1.58,1.88"),
            (0.54, 0.28, 0.50, 0.12, 0.40, 0.24, "1.10,1.54,1.90"),
            (0.68, 0.40, 0.52, 0.04, 0.20, 0.44, "1.08,1.52,1.94"),
            (0.40, 0.44, 0.34, 0.20, 0.42, 0.20, "1.06,1.56,1.86"),
            (0.58, 0.24, 0.58, 0.10, 0.30, 0.38, "1.10,1.50,1.94"),
            (0.50, 0.42, 0.40, 0.00, 0.16, 0.48, "1.06,1.58,1.90"),
            (0.62, 0.34, 0.48, 0.16, 0.36, 0.30, "1.08,1.56,1.92"),
            (0.48, 0.32, 0.60, 0.06, 0.22, 0.42, "1.12,1.52,1.90"),
            (0.42, 0.28, 0.44, 0.28, 0.46, 0.18, "1.04,1.54,1.88"),
            (0.56, 0.46, 0.56, 0.02, 0.18, 0.46, "1.08,1.50,1.96"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "bad_qrs_unreliable_family",
                f"{i:02d}",
                {
                    "qrs_attn_bad": 0.68 + 0.15 * confuse,
                    "flatline_bad": flat,
                    "contact_loss_bad": contact,
                    "baseline_step_bad": step,
                    "spurious_peak_bad": max(0.20, pseudo * 0.7),
                    "bad_pseudo_peak_rate": pseudo,
                    "bad_inverse_peak_rate": inv,
                    "bad_qrs_confuse_rate": confuse,
                    "medium_event_prob": 0.54,
                    "medium_pseudo_peak_rate": 0.06,
                    "medium_inverse_peak_rate": 0.12,
                    "medium_step_rate": 0.12,
                    "medium_ramp_rate": 0.10,
                    "medium_contact_rate": 0.03,
                    "medium_qrs_preserve": 0.996,
                },
                cw,
                "BUT class-3 style: QRS itself becomes unreliable through pseudo-peaks/contact.",
            )
        )

    # Family 4: good is wearable-good rather than pristine PTB.
    for i, (good_scale, med_event, med_preserve, bad_confuse, cw) in enumerate(
        [
            (0.80, 0.48, 0.998, 0.28, "1.16,1.52,1.84"),
            (0.84, 0.52, 0.996, 0.30, "1.18,1.50,1.86"),
            (0.88, 0.56, 0.996, 0.32, "1.20,1.50,1.86"),
            (0.92, 0.60, 0.994, 0.34, "1.22,1.48,1.88"),
            (0.76, 0.44, 0.998, 0.34, "1.14,1.54,1.86"),
            (0.86, 0.58, 0.992, 0.38, "1.18,1.52,1.88"),
            (0.90, 0.64, 0.990, 0.36, "1.20,1.54,1.84"),
            (0.82, 0.50, 0.996, 0.42, "1.16,1.56,1.88"),
            (0.78, 0.54, 0.994, 0.40, "1.14,1.58,1.86"),
            (0.94, 0.42, 0.998, 0.30, "1.24,1.46,1.86"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "good_not_pristine_family",
                f"{i:02d}",
                {
                    "residual_scale_good": good_scale,
                    "residual_scale_medium": 1.02,
                    "medium_event_prob": med_event,
                    "medium_pseudo_peak_rate": 0.06,
                    "medium_inverse_peak_rate": 0.10,
                    "medium_step_rate": 0.10,
                    "medium_ramp_rate": 0.10,
                    "medium_contact_rate": 0.025,
                    "medium_qrs_preserve": med_preserve,
                    "qrs_attn_bad": 0.70,
                    "contact_loss_bad": 0.34,
                    "flatline_bad": 0.18,
                    "baseline_step_bad": 0.30,
                    "bad_pseudo_peak_rate": 0.40,
                    "bad_inverse_peak_rate": 0.28,
                    "bad_qrs_confuse_rate": bad_confuse,
                },
                cw,
                "Good class gets wearable variation so BUT good is not over-rejected.",
            )
        )

    # Family 5: mixed expert boundary.
    for i, (med_event, med_inv, med_step, med_contact, med_preserve, bad_pseudo, bad_confuse, cw) in enumerate(
        [
            (0.58, 0.12, 0.12, 0.03, 0.996, 0.38, 0.30, "1.04,1.56,1.84"),
            (0.62, 0.14, 0.14, 0.04, 0.994, 0.42, 0.34, "1.06,1.58,1.84"),
            (0.66, 0.16, 0.16, 0.05, 0.992, 0.46, 0.36, "1.06,1.60,1.86"),
            (0.70, 0.18, 0.18, 0.06, 0.990, 0.50, 0.38, "1.08,1.60,1.86"),
            (0.74, 0.20, 0.20, 0.06, 0.988, 0.44, 0.42, "1.08,1.62,1.88"),
            (0.78, 0.22, 0.22, 0.08, 0.985, 0.48, 0.44, "1.10,1.62,1.88"),
            (0.64, 0.18, 0.24, 0.04, 0.994, 0.52, 0.32, "1.06,1.64,1.84"),
            (0.68, 0.24, 0.14, 0.06, 0.990, 0.40, 0.46, "1.08,1.60,1.90"),
            (0.54, 0.10, 0.20, 0.02, 0.998, 0.56, 0.36, "1.04,1.58,1.90"),
            (0.80, 0.18, 0.28, 0.10, 0.982, 0.36, 0.34, "1.08,1.68,1.80"),
            (0.60, 0.26, 0.10, 0.04, 0.996, 0.58, 0.50, "1.10,1.56,1.92"),
            (0.72, 0.12, 0.30, 0.06, 0.990, 0.46, 0.52, "1.08,1.66,1.88"),
            (0.56, 0.16, 0.16, 0.02, 0.998, 0.62, 0.44, "1.12,1.54,1.90"),
            (0.76, 0.28, 0.20, 0.08, 0.984, 0.50, 0.54, "1.08,1.64,1.92"),
            (0.66, 0.20, 0.18, 0.05, 0.994, 0.54, 0.46, "1.06,1.62,1.90"),
            (0.70, 0.14, 0.24, 0.04, 0.992, 0.48, 0.40, "1.10,1.60,1.86"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "mixed_expert_boundary_family",
                f"{i:02d}",
                {
                    "residual_scale_good": 0.76,
                    "residual_scale_medium": 1.05,
                    "medium_event_prob": med_event,
                    "medium_pseudo_peak_rate": 0.06 + med_event * 0.06,
                    "medium_inverse_peak_rate": med_inv,
                    "medium_step_rate": med_step,
                    "medium_ramp_rate": max(0.08, med_step * 0.75),
                    "medium_contact_rate": med_contact,
                    "medium_qrs_preserve": med_preserve,
                    "qrs_attn_bad": 0.68 + bad_confuse * 0.12,
                    "contact_loss_bad": 0.30 + bad_confuse * 0.16,
                    "flatline_bad": 0.10 + bad_confuse * 0.20,
                    "baseline_step_bad": 0.24 + bad_confuse * 0.22,
                    "bad_pseudo_peak_rate": bad_pseudo,
                    "bad_inverse_peak_rate": 0.24 + bad_pseudo * 0.30,
                    "bad_qrs_confuse_rate": bad_confuse,
                },
                cw,
                "Joint expert-boundary mixture: medium local unreliability plus bad QRS ambiguity.",
            )
        )

    # Family 6: negative controls.
    negative_controls = [
        ("snr_only_bad", {"qrs_attn_bad": 0.45, "hf_bad": 0.28, "drift_bad": 0.20, "burst_bad": 0.26}),
        ("flatline_only_bad", {"flatline_bad": 0.55, "contact_loss_bad": 0.20, "qrs_attn_bad": 0.50}),
        ("lowamp_only_bad", {"low_amp_bad": 0.38, "qrs_attn_bad": 0.40, "contact_loss_bad": 0.05}),
        ("medium_snr_only", {"hf_medium": 0.12, "drift_medium": 0.12, "tst_noise_medium": 0.10}),
        ("contact_only_bad", {"contact_loss_bad": 0.55, "qrs_attn_bad": 0.55, "flatline_bad": 0.10}),
        ("pseudo_only_bad", {"spurious_peak_bad": 0.65, "bad_pseudo_peak_rate": 0.65, "bad_inverse_peak_rate": 0.42, "bad_qrs_confuse_rate": 0.45}),
        ("medium_motion_only", {"medium_event_prob": 0.85, "medium_step_rate": 0.30, "medium_ramp_rate": 0.30, "medium_qrs_preserve": 0.985}),
        ("good_pristine_control", {"residual_scale_good": 0.60, "medium_event_prob": 0.0, "bad_pseudo_peak_rate": 0.20}),
    ]
    for name, changes in negative_controls:
        out.append(make_entry("negative_controls", name, changes, "1.00,1.55,1.85", "Negative-control rule family."))

    return out


def rep_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("but_10s_eval", {}).get("but_10s_test_report", {}) or {}


def rec_from_rep(rep: dict[str, Any]) -> list[float]:
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    return [float(rec[0]), float(rec[1]), float(rec[2])]


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    rep = rep_from_row(row)
    rec = rec_from_rep(rep)
    ptb = row.get("ptb_test_report", {}) or {}
    ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    den = row.get("ptb_denoise_metrics", {}) or {}
    return {
        "acc": float(rep.get("acc", 0.0)),
        "balanced_acc": float(rep.get("balanced_acc", 0.0)),
        "macro_f1": float(rep.get("macro_f1", 0.0)),
        "good_recall": rec[0],
        "medium_recall": rec[1],
        "bad_recall": rec[2],
        "min_medium_bad": min(rec[1], rec[2]),
        "ptb_acc": float(ptb.get("acc", 0.0)),
        "ptb_bad": float(ptb_rec[2] if len(ptb_rec) > 2 else 0.0),
        "denoise_score": float(den.get("denoise_score", 0.0)),
    }


def valid_for_promotion(row: dict[str, Any], args: argparse.Namespace) -> bool:
    m = row_metrics(row)
    return m["ptb_acc"] >= float(args.min_ptb_acc) and m["ptb_bad"] >= float(args.min_ptb_bad)


def result_key(row: dict[str, Any], args: argparse.Namespace | None = None) -> tuple[float, float, float, float, float, float]:
    m = row_metrics(row)
    valid = 1.0
    if args is not None and not valid_for_promotion(row, args):
        valid = 0.0
    return (
        valid,
        m["macro_f1"],
        m["balanced_acc"],
        m["min_medium_bad"],
        m["bad_recall"],
        m["ptb_acc"] + 0.01 * m["denoise_score"],
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        spec_id = str(row.get("spec", {}).get("id", ""))
        mode = str(row.get("mode", "quick"))
        seed = str(row.get("seed", 0))
        key = f"{mode}:{seed}:{spec_id}"
        by_id[key] = row
    return list(by_id.values())


def prepare_variant(args: argparse.Namespace, entry: dict[str, Any]) -> Path:
    spec = entry["spec"]
    variant_dir = Path(args.out_root) / "synthetic_variants" / str(spec["id"])
    if (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists():
        return variant_dir
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    ptb["source"] = str(args.source_artifact_dir)
    payload = apply_morph_spec(ptb, spec, variant_dir, seed=int(args.seed))
    plot_synthetic_gallery(
        ptb["clean"],
        payload["X_noisy"],
        payload["labels"].reset_index(drop=True),
        variant_dir / "visuals" / "synthetic_gallery.png",
        str(spec["id"]),
    )
    return variant_dir


def train_one(args: argparse.Namespace, entry: dict[str, Any], mode: str, seed: int) -> dict[str, Any]:
    spec = entry["spec"]
    spec_id = str(spec["id"])
    variant_dir = prepare_variant(args, entry)
    run_id = f"{spec_id}_seed{seed}"
    run_dir = Path(args.out_root) / "runs" / mode / run_id
    log_dir = Path(args.out_root) / "logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    epochs1 = args.quick_epochs_stage1 if mode == "quick" else args.full_epochs_stage1
    epochs2 = args.quick_epochs_stage2 if mode == "quick" else args.full_epochs_stage2
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
        str(epochs1),
        "--epochs_stage2",
        str(epochs2),
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
        str(seed),
    ]
    start = time.perf_counter()
    with (log_dir / f"{run_id}.stdout.txt").open("w", encoding="utf-8") as out, (
        log_dir / f"{run_id}.stderr.txt"
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    payload: dict[str, Any] = {
        "spec": spec,
        "class_weight": entry["class_weight"],
        "note": entry["note"],
        "mode": mode,
        "seed": seed,
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
    write_json(run_dir / "large_rule_grid_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / "large_rule_grid_summary.jsonl", payload)
    return payload


def completed_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {
        f"{row.get('mode', 'quick')}:{row.get('seed', 0)}:{row.get('spec', {}).get('id', '')}"
        for row in rows
        if int(row.get("returncode", 1)) == 0
    }


def select_entries(args: argparse.Namespace, mode: str, top_n: int) -> list[dict[str, Any]]:
    all_entries = {entry["spec"]["id"]: entry for entry in entries()}
    rows = dedupe_rows(load_rows(Path(args.out_root) / "large_rule_grid_summary.jsonl"))
    source_mode = "quick" if mode == "full" else "full"
    candidates = [
        row
        for row in rows
        if row.get("mode") == source_mode
        and int(row.get("returncode", 1)) == 0
        and row.get("spec", {}).get("id") in all_entries
    ]
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda row: result_key(row, args), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        spec_id = row["spec"]["id"]
        if spec_id in seen:
            continue
        selected.append(all_entries[spec_id])
        seen.add(spec_id)
        if len(selected) >= top_n:
            break
    return selected


def write_reports(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rows = dedupe_rows(load_rows(out_root / "large_rule_grid_summary.jsonl"))
    write_json(report_root / "large_rule_grid_rows.json", {"rows": rows})

    def fmt(x: float) -> str:
        return f"{x:.4f}"

    def table(table_rows: list[dict[str, Any]]) -> str:
        lines = [
            "| rank | mode | seed | spec | family | BUT acc | bal | macro | recalls G/M/B | minMB | PTB acc | PTB bad | denoise | note |",
            "|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
        for i, row in enumerate(table_rows, start=1):
            m = row_metrics(row)
            lines.append(
                f"| {i} | {row.get('mode')} | {row.get('seed')} | `{row.get('spec', {}).get('id')}` | "
                f"{row.get('spec', {}).get('family')} | {fmt(m['acc'])} | {fmt(m['balanced_acc'])} | "
                f"{fmt(m['macro_f1'])} | {fmt(m['good_recall'])}/{fmt(m['medium_recall'])}/{fmt(m['bad_recall'])} | "
                f"{fmt(m['min_medium_bad'])} | {fmt(m['ptb_acc'])} | {fmt(m['ptb_bad'])} | "
                f"{fmt(m['denoise_score'])} | {row.get('note', '')} |"
            )
        return "\n".join(lines)

    valid_rows = [row for row in rows if int(row.get("returncode", 1)) == 0 and rep_from_row(row)]
    quick_rows = [row for row in valid_rows if row.get("mode") == "quick"]
    full_rows = [row for row in valid_rows if row.get("mode") == "full"]
    seed_rows = [row for row in valid_rows if row.get("mode") == "seed"]
    quick_ranked = sorted(quick_rows, key=lambda row: result_key(row, args), reverse=True)
    full_ranked = sorted(full_rows, key=lambda row: result_key(row, args), reverse=True)
    seed_ranked = sorted(seed_rows, key=lambda row: result_key(row, args), reverse=True)

    family_stats: dict[str, dict[str, Any]] = {}
    for row in quick_rows:
        fam = str(row.get("spec", {}).get("family", "unknown"))
        m = row_metrics(row)
        stat = family_stats.setdefault(fam, {"n": 0, "best_balanced": 0.0, "best_macro": 0.0, "best_minmb": 0.0})
        stat["n"] += 1
        stat["best_balanced"] = max(stat["best_balanced"], m["balanced_acc"])
        stat["best_macro"] = max(stat["best_macro"], m["macro_f1"])
        stat["best_minmb"] = max(stat["best_minmb"], m["min_medium_bad"])
    write_json(report_root / "pareto.json", {"quick": quick_ranked, "full": full_ranked, "seed": seed_ranked})
    write_json(report_root / "rule_family_ablation.json", family_stats)

    md = [
        "# E3.11f BUT 10s Large Rule Grid",
        "",
        "Formal protocol: BUT 10s P1, validation-only calibration, test reporting only.",
        "",
        "## Anchor",
        "",
        "`b10_all_bad_wearable`: acc 0.7735, balanced 0.8045, macro-F1 0.7238, recalls 0.824/0.724/0.866.",
        "",
        "## Quick Stage Top 20",
        table(quick_ranked[:20]),
        "",
        "## Full Stage Top 20",
        table(full_ranked[:20]) if full_ranked else "_No full-stage rows yet._",
        "",
        "## Seed Confirmation",
        table(seed_ranked[:20]) if seed_ranked else "_No seed-confirmation rows yet._",
        "",
        "## Rule Family Summary",
        "",
        "| family | n | best balanced | best macro | best min(M,B) |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, stat in sorted(family_stats.items()):
        md.append(
            f"| {fam} | {stat['n']} | {stat['best_balanced']:.4f} | "
            f"{stat['best_macro']:.4f} | {stat['best_minmb']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Rank uses macro-F1, balanced accuracy, and medium/bad coexistence; raw accuracy alone is not the target.",
            "- Candidates that beat b10 balanced but lower macro-F1 should be treated as partial evidence, not automatic winners.",
            "- Negative controls are included to prove why SNR-only, flatline-only, and low-amplitude-only rules are insufficient.",
        ]
    )
    (report_root / "large_rule_grid_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run_stage(args: argparse.Namespace, stage: str) -> None:
    out_root = Path(args.out_root)
    state_path = out_root / "large_rule_grid_state.json"
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    all_entries = entries()
    write_json(out_root / "large_rule_grid_specs.json", {"entries": all_entries})
    existing = dedupe_rows(load_rows(out_root / "large_rule_grid_summary.jsonl"))
    done = completed_keys(existing)

    if stage == "quick_train":
        selected = all_entries[: int(args.max_quick_runs)]
        mode = "quick"
        seeds = [int(args.seed)]
    elif stage == "full_train":
        selected = select_entries(args, "full", int(args.top_full))
        mode = "full"
        seeds = [int(args.seed)]
    elif stage == "seed_confirm":
        selected = select_entries(args, "seed", int(args.top_seed_specs))
        mode = "seed"
        seeds = [int(s.strip()) for s in str(args.confirm_seeds).split(",") if s.strip()]
    else:
        raise ValueError(stage)

    total = len(selected) * len(seeds)
    idx = 0
    for entry in selected:
        for seed in seeds:
            idx += 1
            key = f"{mode}:{seed}:{entry['spec']['id']}"
            if key in done:
                continue
            update_state(
                state_path,
                status=f"{stage}_running",
                stage=stage,
                current=entry["spec"]["id"],
                mode=mode,
                seed=seed,
                index=idx,
                total=total,
                updated_at=now_iso(),
            )
            train_one(args, entry, mode=mode, seed=seed)
            write_reports(args)
            done.add(key)


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    state_path = out_root / "large_rule_grid_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"quick_train", "full_train", "seed_confirm"}:
        run_stage(args, args.stage)
    elif args.stage == "report":
        write_reports(args)
    elif args.stage == "all":
        run_stage(args, "quick_train")
        run_stage(args, "full_train")
        run_stage(args, "seed_confirm")
        write_reports(args)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a large BUT 10s synthetic rule grid.")
    parser.add_argument("--stage", choices=("quick_train", "full_train", "seed_confirm", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirm_seeds", default="0,1,2")
    parser.add_argument("--max_quick_runs", type=int, default=74)
    parser.add_argument("--top_full", type=int, default=10)
    parser.add_argument("--top_seed_specs", type=int, default=4)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--min_ptb_acc", type=float, default=0.975)
    parser.add_argument("--min_ptb_bad", type=float, default=0.985)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / "large_rule_grid_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
