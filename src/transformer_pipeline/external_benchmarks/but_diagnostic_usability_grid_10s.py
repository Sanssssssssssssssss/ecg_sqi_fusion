"""BUT 10s diagnostic-usability synthetic grid.

This pass follows the sample-level OR result: simply making bad multi-subtype
does not recover the BUT expert boundary.  The next hypothesis is that BUT
labels are closer to diagnostic usability than to noise severity:

* good: all critical dimensions are usable.
* medium: QRS remains usable, but local details/P/T/ST/baseline reliability fail.
* bad: one fatal dimension makes the strip not analyzable; some bad examples
  can still have visible QRS while global reliability is poor.

The runner intentionally defaults to smoke + quick only.  Full training was
shown to strengthen the synthetic boundary before the BUT boundary is right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.transformer_pipeline.external_benchmarks import but_sample_or_subtype_grid_10s as sample_or
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)


RUN_TAG = "e311_but_diagnostic_usability_grid_10s_2026_06_04"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
STATE_NAME = "diagnostic_usability_state.json"
SUMMARY_NAME = "diagnostic_usability_summary.jsonl"
VISIBLE_QRS_SUBTYPE = "visible_qrs_global_noise"
SUBTYPES = tuple(dict.fromkeys((*sample_or.SUBTYPES, VISIBLE_QRS_SUBTYPE)))


_ORIGINAL_APPLY_SUBTYPE = sample_or.apply_subtype


def apply_visible_qrs_global_noise(
    x: np.ndarray,
    clean: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    strength: float,
) -> None:
    """Make a strip globally unreliable while leaving QRS mostly visible."""

    std = sample_or.row_std(clean)
    protect = 1.0 - 0.36 * (qrs > 0.2).astype(np.float32)
    hf = sample_or.normalize_noise(sample_or.highpass_noise(rng, 1))[0]
    burst = sample_or.burst_noise(rng, 1, 1.0)[0]
    drift = sample_or.lowfreq_drift(rng, 1)[0]
    x += float(0.20 + 0.16 * strength) * std * hf * protect
    x += float(0.10 + 0.10 * strength) * std * burst * (1.0 - 0.24 * (qrs > 0.2))
    x += float(0.10 + 0.08 * strength) * std * drift

    # Small repeated distractors make the record feel unusable without making it
    # an obvious flatline/QRS-erasure case.
    for _ in range(int(rng.integers(2, 5))):
        center = int(rng.integers(sample_or.FS_TARGET // 3, sample_or.N_TARGET - sample_or.FS_TARGET // 3))
        width = int(rng.integers(5, 20))
        amp = float(rng.uniform(0.10, 0.32)) * std * strength
        sample_or.add_pulse(x, center, width, amp, float(rng.choice([-1.0, 1.0])))


def apply_subtype_with_visible_qrs(
    subtype: str,
    x: np.ndarray,
    clean: np.ndarray,
    qrs: np.ndarray,
    tst: np.ndarray,
    critical: np.ndarray,
    spec: dict[str, Any],
    rng: np.random.Generator,
    scale: float,
) -> None:
    if subtype == VISIBLE_QRS_SUBTYPE:
        strength = float(spec.get(f"{VISIBLE_QRS_SUBTYPE}_strength", 1.0)) * scale
        apply_visible_qrs_global_noise(x, clean, qrs, rng, strength)
        return
    _ORIGINAL_APPLY_SUBTYPE(subtype, x, clean, qrs, tst, critical, spec, rng, scale)


def make_entry(name: str, changes: dict[str, Any], class_weight: str, note: str) -> dict[str, Any]:
    spec = sample_or.mg.lrg.base_spec()
    spec.update(
        {
            "family": "diagnostic_usability",
            "id": f"diagnostic_usability_{name}",
            "residual_scale_good": 0.62,
            "residual_scale_medium": 1.03,
            "residual_scale_bad": 1.04,
            "medium_event_prob": 0.78,
            "medium_pseudo_peak_rate": 0.05,
            "medium_inverse_peak_rate": 0.18,
            "medium_step_rate": 0.24,
            "medium_ramp_rate": 0.20,
            "medium_contact_rate": 0.03,
            "medium_qrs_preserve": 0.999,
            "bad_base_low_amp": 0.98,
            "bad_base_residual_scale": 1.04,
            "secondary_prob": 0.16,
            "subtype_weights": {
                "visible_qrs_global_noise": 1.8,
                "qrs_confound": 0.55,
                "contact_flat": 0.95,
                "motion_burst": 1.15,
                "morph_break": 0.55,
                "clipping_lowamp": 0.55,
                "baseline_jump": 0.75,
            },
            "visible_qrs_global_noise_strength": 1.0,
            "qrs_confound_strength": 0.88,
            "contact_flat_strength": 0.86,
            "motion_burst_strength": 0.92,
            "morph_break_strength": 0.82,
            "clipping_lowamp_strength": 0.78,
            "baseline_jump_strength": 0.82,
        }
    )
    spec.update(changes)
    return {"spec": spec, "class_weight": class_weight, "note": note}


def entries() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    medium_profiles = [
        ("med_local_soft", 0.66, 0.14, 0.16, 0.14, 0.02, 0.999),
        ("med_local", 0.76, 0.18, 0.24, 0.20, 0.03, 0.999),
        ("med_detail", 0.84, 0.24, 0.20, 0.24, 0.035, 0.998),
        ("med_instability", 0.88, 0.20, 0.32, 0.28, 0.045, 0.998),
    ]
    bad_profiles = [
        (
            "bad_visible",
            {"visible_qrs_global_noise": 2.1, "qrs_confound": 0.35, "contact_flat": 0.65, "motion_burst": 1.25, "morph_break": 0.35, "clipping_lowamp": 0.35, "baseline_jump": 0.55},
            0.08,
            1.04,
        ),
        (
            "bad_mixed_visible",
            {"visible_qrs_global_noise": 1.6, "qrs_confound": 0.65, "contact_flat": 1.0, "motion_burst": 1.15, "morph_break": 0.55, "clipping_lowamp": 0.55, "baseline_jump": 0.85},
            0.16,
            1.00,
        ),
        (
            "bad_contact_visible",
            {"visible_qrs_global_noise": 1.35, "qrs_confound": 0.45, "contact_flat": 1.45, "motion_burst": 1.0, "morph_break": 0.45, "clipping_lowamp": 0.75, "baseline_jump": 1.2},
            0.18,
            0.96,
        ),
        (
            "bad_qrs_visible_floor",
            {"visible_qrs_global_noise": 1.2, "qrs_confound": 0.9, "contact_flat": 0.7, "motion_burst": 0.8, "morph_break": 0.75, "clipping_lowamp": 0.45, "baseline_jump": 0.65},
            0.10,
            1.06,
        ),
    ]
    good_profiles = [
        ("good_strict", 0.56),
        ("good_clean", 0.64),
    ]
    class_weights = [
        "1.00,1.72,1.70",
        "1.02,1.76,1.72",
        "1.04,1.68,1.78",
    ]
    idx = 1
    for g_name, good_scale in good_profiles:
        for m_name, event, inv, step, ramp, contact, qrs_preserve in medium_profiles:
            for b_name, weights, secondary_prob, bad_base in bad_profiles:
                cw = class_weights[(idx - 1) % len(class_weights)]
                out.append(
                    make_entry(
                        f"{idx:02d}_{g_name}_{m_name}_{b_name}",
                        {
                            "residual_scale_good": good_scale,
                            "medium_event_prob": event,
                            "medium_inverse_peak_rate": inv,
                            "medium_step_rate": step,
                            "medium_ramp_rate": ramp,
                            "medium_contact_rate": contact,
                            "medium_qrs_preserve": qrs_preserve,
                            "subtype_weights": weights,
                            "secondary_prob": secondary_prob,
                            "bad_base_residual_scale": bad_base,
                            "bad_base_low_amp": 0.99,
                        },
                        cw,
                        "Diagnostic-usability grid: good is all-critical-clean, medium is QRS-usable/detail-unreliable, bad includes visible-QRS global unusability.",
                    )
                )
                idx += 1
    return out


def install_patches() -> None:
    sample_or.RUN_TAG = RUN_TAG
    sample_or.DEFAULT_OUT_ROOT = DEFAULT_OUT_ROOT
    sample_or.DEFAULT_REPORT_ROOT = DEFAULT_REPORT_ROOT
    sample_or.STATE_NAME = STATE_NAME
    sample_or.SUMMARY_NAME = SUMMARY_NAME
    sample_or.SUBTYPES = SUBTYPES
    sample_or.entries = entries
    sample_or.apply_subtype = apply_subtype_with_visible_qrs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = sample_or.build_arg_parser()
    parser.description = "Run BUT 10s diagnostic-usability quick grid."
    parser.set_defaults(
        stage="all",
        out_root=str(DEFAULT_OUT_ROOT),
        report_root=str(DEFAULT_REPORT_ROOT),
        protocol_out_root=str(PROTOCOL_OUT_ROOT),
        protocol_report_root=str(PROTOCOL_REPORT_ROOT),
        max_runs=24,
        smoke_runs=2,
        top_full=0,
        min_ptb_acc=0.970,
        min_ptb_bad=0.985,
    )
    return parser


def write_report_alias(args: argparse.Namespace) -> None:
    report_root = Path(args.report_root)
    src_md = report_root / "sample_or_subtype_summary.md"
    src_csv = report_root / "sample_or_subtype_metric_join.csv"
    if src_md.exists():
        text = src_md.read_text(encoding="utf-8")
        text = text.replace(
            "# BUT 10s Sample-Level Fatal-Subtype OR Grid",
            "# BUT 10s Diagnostic-Usability Synthetic Grid",
        )
        text += (
            "\n\n## Interpretation\n\n"
            "This quick grid tests whether BUT labels behave like diagnostic usability: "
            "good requires all critical dimensions to be usable, medium is QRS-usable "
            "with local/detail unreliability, and bad may keep visible QRS while the "
            "strip is globally not analyzable. Full confirmation is intentionally "
            "disabled by default until the quick/probe boundary beats prior anchors.\n"
        )
        (report_root / "diagnostic_usability_summary.md").write_text(text, encoding="utf-8")
    if src_csv.exists():
        (report_root / "diagnostic_usability_metric_join.csv").write_bytes(src_csv.read_bytes())


def main() -> None:
    install_patches()
    args = build_arg_parser().parse_args()
    try:
        sample_or.run(args)
        write_report_alias(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        sample_or.update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=sample_or.now_iso())
        raise


if __name__ == "__main__":
    main()
