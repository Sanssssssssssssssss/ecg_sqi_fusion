"""BUT 10s morphology-guided synthetic rule grid.

This runner supersedes the stopped large-rule grid by using the completed BUT
morphology analysis as a prior.  It keeps the formal BUT protocol at 10s P1,
uses validation-only calibration through the existing evaluator, and performs a
small feature audit before every training run so synthetic rules are judged by
both waveform morphology and downstream BUT metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_large_rule_grid_10s as lrg
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT, load_ptb_artifact
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    evaluate_checkpoint_10s,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_generator_morph_sweet_grid import apply_morph_spec
from src.transformer_pipeline.external_benchmarks.but_morphology_analysis_10s import (
    ALL_FEATURES,
    CLASS_ORDER,
    compute_distances,
    extract_features_for_dataset,
    load_npz_array,
    summarize_features,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)


RUN_TAG = "e311_but_morphology_guided_grid_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
MORPH_ANALYSIS_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_morphology_analysis_10s_2026_06_03"
SUMMARY_NAME = "morphology_guided_summary.jsonl"
STATE_NAME = "morphology_guided_state.json"

B10_ANCHOR = {
    "acc": 0.7735,
    "balanced_acc": 0.8045,
    "macro_f1": 0.7238,
    "recall_good": 0.824,
    "recall_medium": 0.724,
    "recall_bad": 0.866,
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def make_entry(family: str, name: str, changes: dict[str, Any], class_weight: str, note: str) -> dict[str, Any]:
    spec = lrg.base_spec()
    spec.update(changes)
    spec["id"] = f"{family}_{name}"
    spec["family"] = family
    return {"spec": spec, "class_weight": class_weight, "note": note}


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    return lrg.row_metrics(row)


def rec_from_row(row: dict[str, Any]) -> list[float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {}) or {}
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    return [float(rec[0]), float(rec[1]), float(rec[2])]


def hypothesis_entries() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # H1: rescue medium by making P/T/ST and local baseline unreliable while QRS remains visible.
    for i, (event, inv, step, ramp, contact, preserve, bad_floor, cw) in enumerate(
        [
            (0.56, 0.10, 0.10, 0.10, 0.02, 0.998, 0.34, "1.04,1.58,1.82"),
            (0.66, 0.14, 0.16, 0.14, 0.04, 0.995, 0.38, "1.06,1.60,1.82"),
            (0.76, 0.18, 0.22, 0.18, 0.06, 0.990, 0.42, "1.08,1.62,1.82"),
            (0.62, 0.22, 0.14, 0.24, 0.04, 0.994, 0.46, "1.06,1.64,1.80"),
            (0.70, 0.12, 0.30, 0.20, 0.06, 0.992, 0.44, "1.08,1.62,1.84"),
            (0.82, 0.22, 0.26, 0.24, 0.08, 0.985, 0.48, "1.10,1.64,1.82"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "h_medium_rescue",
                f"{i:02d}",
                {
                    "residual_scale_good": 0.76,
                    "residual_scale_medium": 1.04,
                    "medium_event_prob": event,
                    "medium_pseudo_peak_rate": 0.05 + 0.10 * event,
                    "medium_inverse_peak_rate": inv,
                    "medium_step_rate": step,
                    "medium_ramp_rate": ramp,
                    "medium_contact_rate": contact,
                    "medium_qrs_preserve": preserve,
                    "qrs_attn_bad": 0.68,
                    "contact_loss_bad": 0.30,
                    "flatline_bad": 0.14,
                    "baseline_step_bad": 0.28,
                    "bad_pseudo_peak_rate": bad_floor,
                    "bad_inverse_peak_rate": 0.22 + 0.35 * bad_floor,
                    "bad_qrs_confuse_rate": 0.22 + 0.45 * bad_floor,
                },
                cw,
                "Hypothesis H1: class-2 BUT medium is QRS-visible but locally unreliable.",
            )
        )

    # H2: rescue bad by making QRS detectability fail through pseudo-peaks/contact, not pure flatline.
    for i, (pseudo, inv, confuse, contact, step, flat, cw) in enumerate(
        [
            (0.42, 0.28, 0.32, 0.24, 0.30, 0.06, "1.04,1.56,1.88"),
            (0.50, 0.34, 0.40, 0.26, 0.36, 0.04, "1.06,1.56,1.90"),
            (0.58, 0.40, 0.48, 0.22, 0.42, 0.02, "1.08,1.54,1.92"),
            (0.46, 0.42, 0.52, 0.32, 0.28, 0.08, "1.06,1.58,1.90"),
            (0.62, 0.30, 0.56, 0.18, 0.44, 0.00, "1.10,1.52,1.94"),
            (0.54, 0.46, 0.44, 0.36, 0.24, 0.10, "1.08,1.56,1.90"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "h_bad_rescue",
                f"{i:02d}",
                {
                    "residual_scale_good": 0.76,
                    "residual_scale_medium": 1.03,
                    "medium_event_prob": 0.58,
                    "medium_pseudo_peak_rate": 0.06,
                    "medium_inverse_peak_rate": 0.12,
                    "medium_step_rate": 0.12,
                    "medium_ramp_rate": 0.10,
                    "medium_contact_rate": 0.03,
                    "medium_qrs_preserve": 0.996,
                    "qrs_attn_bad": 0.68 + 0.12 * confuse,
                    "contact_loss_bad": contact,
                    "flatline_bad": flat,
                    "baseline_step_bad": step,
                    "bad_pseudo_peak_rate": pseudo,
                    "bad_inverse_peak_rate": inv,
                    "bad_qrs_confuse_rate": confuse,
                },
                cw,
                "Hypothesis H2: class-3 BUT bad is QRS-confounded rather than only low-amplitude.",
            )
        )

    # H3: coexistence around the empirically stable b10/s02/mix05/good-not-pristine neighborhood.
    for i, (good_scale, med_event, med_contact, bad_pseudo, bad_contact, bad_flat, cw) in enumerate(
        [
            (0.72, 0.52, 0.025, 0.34, 0.30, 0.14, "1.00,1.54,1.82"),
            (0.78, 0.60, 0.035, 0.48, 0.34, 0.12, "1.08,1.58,1.84"),
            (0.72, 0.66, 0.055, 0.48, 0.36, 0.14, "1.04,1.60,1.86"),
            (0.84, 0.56, 0.030, 0.40, 0.32, 0.12, "1.16,1.54,1.84"),
            (0.76, 0.70, 0.050, 0.42, 0.30, 0.10, "1.08,1.64,1.80"),
            (0.80, 0.62, 0.040, 0.54, 0.28, 0.08, "1.10,1.56,1.88"),
            (0.74, 0.58, 0.025, 0.58, 0.24, 0.04, "1.06,1.52,1.92"),
            (0.86, 0.68, 0.060, 0.36, 0.38, 0.16, "1.16,1.60,1.82"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "h_coexistence",
                f"{i:02d}",
                {
                    "residual_scale_good": good_scale,
                    "residual_scale_medium": 1.04,
                    "medium_event_prob": med_event,
                    "medium_pseudo_peak_rate": 0.06 + 0.05 * med_event,
                    "medium_inverse_peak_rate": 0.10 + 0.08 * med_event,
                    "medium_step_rate": 0.10 + 0.14 * med_event,
                    "medium_ramp_rate": 0.08 + 0.12 * med_event,
                    "medium_contact_rate": med_contact,
                    "medium_qrs_preserve": clamp(1.005 - 0.018 * med_event, 0.985, 0.999),
                    "qrs_attn_bad": 0.70,
                    "contact_loss_bad": bad_contact,
                    "flatline_bad": bad_flat,
                    "baseline_step_bad": 0.28 + 0.12 * bad_pseudo,
                    "bad_pseudo_peak_rate": bad_pseudo,
                    "bad_inverse_peak_rate": 0.18 + 0.36 * bad_pseudo,
                    "bad_qrs_confuse_rate": 0.20 + 0.50 * bad_pseudo,
                },
                cw,
                "Hypothesis H3: b10/s02/mix05 coexistence interpolation.",
            )
        )

    # H4: negative controls to test whether SNR/flatline-heavy rules are the wrong axis.
    for i, (flat, contact, qrs, pseudo, cw) in enumerate(
        [
            (0.42, 0.48, 0.82, 0.08, "1.00,1.40,2.00"),
            (0.52, 0.36, 0.90, 0.04, "1.00,1.36,2.08"),
            (0.34, 0.56, 0.86, 0.10, "1.00,1.42,2.04"),
            (0.60, 0.28, 0.92, 0.02, "1.00,1.34,2.12"),
        ],
        start=1,
    ):
        out.append(
            make_entry(
                "h_negative_control",
                f"{i:02d}",
                {
                    "residual_scale_good": 0.70,
                    "residual_scale_medium": 1.00,
                    "medium_event_prob": 0.12,
                    "medium_qrs_preserve": 0.999,
                    "qrs_attn_bad": qrs,
                    "contact_loss_bad": contact,
                    "flatline_bad": flat,
                    "baseline_step_bad": 0.12,
                    "bad_pseudo_peak_rate": pseudo,
                    "bad_inverse_peak_rate": 0.04,
                    "bad_qrs_confuse_rate": 0.04,
                },
                cw,
                "Negative control: flatline/SNR-heavy bad without morphology-rich medium.",
            )
        )
    return out


def default_template_entries() -> list[dict[str, Any]]:
    return [
        make_entry(
            "template_b10",
            "anchor",
            {
                "residual_scale_good": 0.74,
                "medium_event_prob": 0.50,
                "medium_pseudo_peak_rate": 0.05,
                "medium_inverse_peak_rate": 0.10,
                "medium_step_rate": 0.10,
                "medium_ramp_rate": 0.08,
                "medium_contact_rate": 0.025,
                "medium_qrs_preserve": 0.998,
                "qrs_attn_bad": 0.70,
                "contact_loss_bad": 0.34,
                "flatline_bad": 0.18,
                "baseline_step_bad": 0.28,
                "bad_pseudo_peak_rate": 0.34,
                "bad_inverse_peak_rate": 0.24,
                "bad_qrs_confuse_rate": 0.26,
            },
            "1.00,1.54,1.82",
            "Default b10-like template.",
        ),
        make_entry(
            "template_medium",
            "qrs_visible",
            {
                "residual_scale_good": 0.76,
                "medium_event_prob": 0.68,
                "medium_pseudo_peak_rate": 0.10,
                "medium_inverse_peak_rate": 0.16,
                "medium_step_rate": 0.20,
                "medium_ramp_rate": 0.16,
                "medium_contact_rate": 0.045,
                "medium_qrs_preserve": 0.992,
                "qrs_attn_bad": 0.68,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.12,
                "baseline_step_bad": 0.30,
                "bad_pseudo_peak_rate": 0.42,
                "bad_inverse_peak_rate": 0.30,
                "bad_qrs_confuse_rate": 0.34,
            },
            "1.06,1.62,1.82",
            "Default medium-QRS-visible template.",
        ),
        make_entry(
            "template_bad",
            "qrs_unreliable",
            {
                "residual_scale_good": 0.76,
                "medium_event_prob": 0.58,
                "medium_pseudo_peak_rate": 0.06,
                "medium_inverse_peak_rate": 0.12,
                "medium_step_rate": 0.12,
                "medium_ramp_rate": 0.10,
                "medium_contact_rate": 0.03,
                "medium_qrs_preserve": 0.996,
                "qrs_attn_bad": 0.72,
                "contact_loss_bad": 0.26,
                "flatline_bad": 0.04,
                "baseline_step_bad": 0.40,
                "bad_pseudo_peak_rate": 0.58,
                "bad_inverse_peak_rate": 0.38,
                "bad_qrs_confuse_rate": 0.46,
            },
            "1.06,1.54,1.90",
            "Default pseudo-QRS bad template.",
        ),
        make_entry(
            "template_good",
            "not_pristine",
            {
                "residual_scale_good": 0.84,
                "medium_event_prob": 0.58,
                "medium_pseudo_peak_rate": 0.08,
                "medium_inverse_peak_rate": 0.14,
                "medium_step_rate": 0.14,
                "medium_ramp_rate": 0.12,
                "medium_contact_rate": 0.03,
                "medium_qrs_preserve": 0.996,
                "qrs_attn_bad": 0.70,
                "contact_loss_bad": 0.32,
                "flatline_bad": 0.12,
                "baseline_step_bad": 0.32,
                "bad_pseudo_peak_rate": 0.40,
                "bad_inverse_peak_rate": 0.28,
                "bad_qrs_confuse_rate": 0.36,
            },
            "1.16,1.54,1.86",
            "Default wearable-good overlap template.",
        ),
    ]


def guided_entries_from_templates(templates: list[dict[str, Any]], max_entries: int = 96) -> list[dict[str, Any]]:
    medium_levels = [0.52, 0.62, 0.72, 0.82]
    bad_levels = [0.34, 0.46, 0.58]
    good_levels = [0.74, 0.84]
    out: list[dict[str, Any]] = []
    for t_i, template in enumerate(templates[:4], start=1):
        base = dict(template["spec"])
        for m_i, medium_strength in enumerate(medium_levels, start=1):
            for b_i, bad_pressure in enumerate(bad_levels, start=1):
                for g_i, good_scale in enumerate(good_levels, start=1):
                    spec = dict(base)
                    med_contact = clamp(float(spec.get("medium_contact_rate", 0.03)) + 0.03 * (medium_strength - 0.52), 0.015, 0.11)
                    med_step = clamp(float(spec.get("medium_step_rate", 0.12)) + 0.18 * (medium_strength - 0.52), 0.06, 0.34)
                    spec.update(
                        {
                            "id": f"g_t{t_i:02d}_m{m_i}_b{b_i}_g{g_i}",
                            "family": "guided_morphology_large",
                            "residual_scale_good": good_scale,
                            "residual_scale_medium": clamp(0.98 + 0.11 * medium_strength, 1.02, 1.09),
                            "medium_event_prob": medium_strength,
                            "medium_pseudo_peak_rate": clamp(0.04 + 0.09 * medium_strength, 0.05, 0.14),
                            "medium_inverse_peak_rate": clamp(0.07 + 0.16 * medium_strength, 0.09, 0.24),
                            "medium_step_rate": med_step,
                            "medium_ramp_rate": clamp(0.07 + 0.15 * medium_strength, 0.08, 0.24),
                            "medium_contact_rate": med_contact,
                            "medium_qrs_preserve": clamp(1.004 - 0.020 * medium_strength, 0.985, 0.999),
                            "qrs_attn_bad": clamp(0.64 + 0.18 * bad_pressure, 0.66, 0.78),
                            "contact_loss_bad": clamp(float(spec.get("contact_loss_bad", 0.30)) + 0.20 * (bad_pressure - 0.46), 0.18, 0.44),
                            "flatline_bad": clamp(float(spec.get("flatline_bad", 0.12)) + 0.12 * (0.46 - bad_pressure), 0.02, 0.22),
                            "baseline_step_bad": clamp(0.22 + 0.38 * bad_pressure, 0.24, 0.48),
                            "bad_pseudo_peak_rate": bad_pressure,
                            "bad_inverse_peak_rate": clamp(0.15 + 0.46 * bad_pressure, 0.22, 0.46),
                            "bad_qrs_confuse_rate": clamp(0.18 + 0.62 * bad_pressure, 0.26, 0.58),
                        }
                    )
                    cw_good = 1.00 + 0.10 * (good_scale > 0.80) + 0.04 * t_i
                    cw_medium = 1.52 + 0.18 * medium_strength
                    cw_bad = 1.78 + 0.22 * bad_pressure
                    out.append(
                        {
                            "spec": spec,
                            "class_weight": f"{cw_good:.2f},{cw_medium:.2f},{cw_bad:.2f}",
                            "note": (
                                "Guided expansion from hypothesis templates: medium P/T/ST unreliability, "
                                "bad pseudo-QRS/contact pressure, and wearable-good overlap."
                            ),
                        }
                    )
                    if len(out) >= max_entries:
                        return out
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def completed_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {
        f"{row.get('mode')}:{row.get('seed', 0)}:{row.get('spec', {}).get('id', '')}"
        for row in rows
        if int(row.get("returncode", 1)) == 0
    }


def result_key(row: dict[str, Any], args: argparse.Namespace | None = None) -> tuple[float, float, float, float, float, float, float]:
    m = row_metrics(row)
    valid = 1.0
    if args is not None and (m["ptb_acc"] < float(args.min_ptb_acc) or m["ptb_bad"] < float(args.min_ptb_bad)):
        valid = 0.0
    audit = row.get("feature_audit", {}) or {}
    target = float(audit.get("feature_target_score", 0.0))
    mismatch = 1.0 if audit.get("feature_mismatch") else 0.0
    return (
        valid,
        m["macro_f1"],
        m["balanced_acc"],
        m["min_medium_bad"],
        m["bad_recall"],
        target - 0.05 * mismatch,
        m["ptb_acc"] + 0.01 * m["denoise_score"],
    )


def top_hypothesis_templates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [r for r in load_rows(Path(args.out_root) / SUMMARY_NAME) if r.get("mode") == "hypothesis"]
    if not rows:
        return default_template_entries()
    ranked = sorted([r for r in rows if int(r.get("returncode", 1)) == 0], key=lambda r: result_key(r, args), reverse=True)
    selected: list[dict[str, Any]] = []
    seen_family: set[str] = set()
    for row in ranked:
        spec = dict(row.get("spec", {}))
        family = str(spec.get("family", "template"))
        if family in seen_family and len(selected) < 3:
            continue
        seen_family.add(family)
        selected.append(
            {
                "spec": spec,
                "class_weight": row.get("class_weight", "1.06,1.58,1.86"),
                "note": f"Template promoted from hypothesis row {spec.get('id')}.",
            }
        )
        if len(selected) >= 4:
            break
    defaults = default_template_entries()
    while len(selected) < 4:
        selected.append(defaults[len(selected)])
    return selected[:4]


def entries_for_stage(args: argparse.Namespace, stage: str) -> list[dict[str, Any]]:
    if stage == "hypothesis_train":
        return hypothesis_entries()
    if stage == "quick_train":
        return guided_entries_from_templates(top_hypothesis_templates(args), max_entries=int(args.max_quick_runs))
    return hypothesis_entries() + guided_entries_from_templates(top_hypothesis_templates(args), max_entries=int(args.max_quick_runs))


def load_but_feature_frame() -> pd.DataFrame:
    path = MORPH_ANALYSIS_ROOT / "but_morph_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing BUT morphology feature table: {path}")
    return pd.read_csv(path)


def class_summary(summary_df: pd.DataFrame, variant_id: str, cls: str, feature: str) -> float:
    row = summary_df[(summary_df["variant_id"] == variant_id) & (summary_df["y_class"] == cls)]
    if row.empty:
        return 0.0
    return float(row[f"{feature}_mean"].iloc[0])


def audit_variant(args: argparse.Namespace, variant_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    spec = entry["spec"]
    spec_id = str(spec["id"])
    audit_path = variant_dir / "synthetic_feature_profile.json"
    delta_path = variant_dir / "but_target_delta.json"
    if audit_path.exists() and delta_path.exists():
        payload = read_json(audit_path)
        payload.update(read_json(delta_path))
        return payload

    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    X = load_npz_array(variant_dir / "datasets" / "signals.npz")
    labels = pd.read_csv(labels_path)["y_class"]
    but_df = load_but_feature_frame()
    synth_df = extract_features_for_dataset(
        X,
        labels,
        source="PTB_synthetic_morphology_guided",
        variant_id=spec_id,
        family=str(spec.get("family", "guided")),
        max_per_class=int(args.audit_max_per_class),
        seed=int(args.seed),
    )
    synth_summary = summarize_features(synth_df, "synthetic_guided")
    distance_df, feature_distance_df = compute_distances(but_df, synth_df)
    dist = distance_df.iloc[0].to_dict() if not distance_df.empty else {}

    but_summary = summarize_features(but_df, "but_target")
    med_qrs = class_summary(synth_summary, spec_id, "medium", "qrs_reliable_proxy")
    med_ptst = class_summary(synth_summary, spec_id, "medium", "ptst_unreliable_proxy")
    bad_qrs = class_summary(synth_summary, spec_id, "bad", "qrs_reliable_proxy")
    bad_spurious = class_summary(synth_summary, spec_id, "bad", "spurious_peak_proxy")
    good_qrs = class_summary(synth_summary, spec_id, "good", "qrs_reliable_proxy")
    target = {
        "medium_qrs_reliable_target": class_summary(but_summary, "BUT_QDB", "medium", "qrs_reliable_proxy"),
        "medium_ptst_unreliable_target": class_summary(but_summary, "BUT_QDB", "medium", "ptst_unreliable_proxy"),
        "bad_qrs_reliable_target": class_summary(but_summary, "BUT_QDB", "bad", "qrs_reliable_proxy"),
        "bad_spurious_peak_target": class_summary(but_summary, "BUT_QDB", "bad", "spurious_peak_proxy"),
        "good_qrs_reliable_target": class_summary(but_summary, "BUT_QDB", "good", "qrs_reliable_proxy"),
    }
    target_deltas = {
        "medium_qrs_reliable_delta": med_qrs - target["medium_qrs_reliable_target"],
        "medium_ptst_unreliable_delta": med_ptst - target["medium_ptst_unreliable_target"],
        "bad_qrs_reliable_delta": bad_qrs - target["bad_qrs_reliable_target"],
        "bad_spurious_peak_delta": bad_spurious - target["bad_spurious_peak_target"],
        "good_qrs_reliable_delta": good_qrs - target["good_qrs_reliable_target"],
    }
    # The score is a tie-breaker: class-wise distance matters, but we do not let
    # this proxy overrule actual BUT metrics.
    feature_target_score = float(1.0 / (1.0 + float(dist.get("overall_but_like_score", 1.0))))
    medium_bad_direction = (
        med_qrs >= 0.55 * target["medium_qrs_reliable_target"]
        and med_ptst >= 0.55 * target["medium_ptst_unreliable_target"]
        and bad_qrs <= 2.8 * max(1e-6, target["bad_qrs_reliable_target"])
        and bad_spurious >= 0.25 * target["bad_spurious_peak_target"]
    )
    mismatch_reasons: list[str] = []
    if med_qrs < 0.55 * target["medium_qrs_reliable_target"]:
        mismatch_reasons.append("medium_qrs_too_weak")
    if med_ptst < 0.55 * target["medium_ptst_unreliable_target"]:
        mismatch_reasons.append("medium_ptst_too_clean")
    if bad_qrs > 2.8 * max(1e-6, target["bad_qrs_reliable_target"]):
        mismatch_reasons.append("bad_qrs_too_reliable")
    if bad_spurious < 0.25 * target["bad_spurious_peak_target"]:
        mismatch_reasons.append("bad_spurious_too_low")

    profile_payload = {
        "variant_id": spec_id,
        "family": spec.get("family"),
        "feature_target_score": feature_target_score,
        "feature_mismatch": not medium_bad_direction,
        "feature_mismatch_reasons": mismatch_reasons,
        "distance": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in dist.items()},
        "key_feature_means": {
            "good_qrs_reliable": good_qrs,
            "medium_qrs_reliable": med_qrs,
            "medium_ptst_unreliable": med_ptst,
            "bad_qrs_reliable": bad_qrs,
            "bad_spurious_peak": bad_spurious,
        },
    }
    delta_payload = {"but_targets": target, "target_deltas": target_deltas}
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_path, profile_payload)
    write_json(delta_path, delta_payload)
    synth_summary.to_csv(variant_dir / "synthetic_feature_summary.csv", index=False)
    feature_distance_df.to_csv(variant_dir / "synthetic_but_feature_distance.csv", index=False)
    return {**profile_payload, **delta_payload}


def prepare_variant(args: argparse.Namespace, entry: dict[str, Any]) -> Path:
    spec = entry["spec"]
    variant_dir = Path(args.out_root) / "synthetic_variants" / str(spec["id"])
    if not (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists():
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
    audit = audit_variant(args, variant_dir, entry)
    entry["_feature_audit"] = audit
    return variant_dir


def metric_row(payload: dict[str, Any]) -> dict[str, Any]:
    m = row_metrics(payload)
    rec = rec_from_row(payload)
    audit = payload.get("feature_audit", {}) or {}
    dist = audit.get("distance", {}) or {}
    return {
        "variant_id": payload.get("spec", {}).get("id"),
        "family": payload.get("spec", {}).get("family"),
        "mode": payload.get("mode"),
        "seed": payload.get("seed"),
        "acc": m["acc"],
        "balanced_acc": m["balanced_acc"],
        "macro_f1": m["macro_f1"],
        "good_recall": rec[0],
        "medium_recall": rec[1],
        "bad_recall": rec[2],
        "min_medium_bad": m["min_medium_bad"],
        "ptb_acc": m["ptb_acc"],
        "ptb_bad": m["ptb_bad"],
        "denoise_score": m["denoise_score"],
        "overall_but_like_score": float(dist.get("overall_but_like_score", 0.0)),
        "qrs_distance": float(dist.get("qrs_distance", 0.0)),
        "ptst_distance": float(dist.get("ptst_distance", 0.0)),
        "contact_motion_distance": float(dist.get("contact_motion_distance", 0.0)),
        "feature_target_score": float(audit.get("feature_target_score", 0.0)),
        "feature_mismatch": bool(audit.get("feature_mismatch", False)),
        "calibration_source": "but_val_only_evaluate_checkpoint_10s",
        "returncode": int(payload.get("returncode", 1)),
        "run_dir": payload.get("run_dir"),
        "variant_dir": payload.get("variant_dir"),
        "note": payload.get("note", ""),
    }


def write_visual_review(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = payload.get("feature_audit", {}) or {}
    m = row_metrics(payload)
    rec = rec_from_row(payload)
    lines = [
        "# Visual Review Stub",
        "",
        "Generated by morphology-guided grid. Inspect linked galleries before promotion.",
        "",
        f"- variant: `{payload.get('spec', {}).get('id')}`",
        f"- BUT acc/bal/macro: `{m['acc']:.4f}/{m['balanced_acc']:.4f}/{m['macro_f1']:.4f}`",
        f"- recalls G/M/B: `{rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f}`",
        f"- feature mismatch: `{audit.get('feature_mismatch', False)}` {audit.get('feature_mismatch_reasons', [])}",
        f"- synthetic gallery: `{Path(payload.get('variant_dir', '')) / 'visuals' / 'synthetic_gallery.png'}`",
        "",
        "Manual tags to check: good_overfiltered, qrs_low_prominence, qrs_confounded, ptst_unreliable, contact_loss, baseline_step.",
    ]
    (run_dir / "visual_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_one(args: argparse.Namespace, entry: dict[str, Any], mode: str, seed: int) -> dict[str, Any]:
    spec = entry["spec"]
    spec_id = str(spec["id"])
    variant_dir = prepare_variant(args, entry)
    run_id = f"{spec_id}_seed{seed}"
    run_dir = Path(args.out_root) / "runs" / mode / run_id
    log_dir = Path(args.out_root) / "logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    is_quick = mode in {"hypothesis", "quick"}
    epochs1 = args.quick_epochs_stage1 if is_quick else args.full_epochs_stage1
    epochs2 = args.quick_epochs_stage2 if is_quick else args.full_epochs_stage2
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
        "feature_audit": entry.get("_feature_audit", {}),
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "morphology_guided_run_summary.json", payload)
    write_json(run_dir / "morphology_guided_metric_row.json", metric_row(payload))
    write_visual_review(run_dir, payload)
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    return payload


def select_entries(args: argparse.Namespace, mode: str, top_n: int) -> list[dict[str, Any]]:
    rows = lrg.dedupe_rows(load_rows(Path(args.out_root) / SUMMARY_NAME))
    if mode == "full":
        source_modes = {"hypothesis", "quick"}
    else:
        source_modes = {"full"}
    candidates = [
        row
        for row in rows
        if row.get("mode") in source_modes
        and int(row.get("returncode", 1)) == 0
        and row.get("spec", {}).get("id")
    ]
    ranked = sorted(candidates, key=lambda row: result_key(row, args), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        spec = dict(row["spec"])
        spec_id = str(spec["id"])
        if spec_id in seen:
            continue
        selected.append(
            {
                "spec": spec,
                "class_weight": row.get("class_weight", "1.06,1.58,1.86"),
                "note": f"Promoted from {row.get('mode')} stage. {row.get('note', '')}",
            }
        )
        seen.add(spec_id)
        if len(selected) >= top_n:
            break
    return selected


def run_stage(args: argparse.Namespace, stage: str) -> None:
    out_root = Path(args.out_root)
    state_path = out_root / STATE_NAME
    if stage == "hypothesis_train":
        run_entries = entries_for_stage(args, stage)[: int(args.max_hypothesis_runs)]
        mode = "hypothesis"
        seeds = [int(args.seed)]
    elif stage == "quick_train":
        run_entries = entries_for_stage(args, stage)[: int(args.max_quick_runs)]
        mode = "quick"
        seeds = [int(args.seed)]
    elif stage == "full_train":
        run_entries = select_entries(args, "full", int(args.top_full))
        mode = "full"
        seeds = [int(args.seed)]
    elif stage == "seed_confirm":
        run_entries = select_entries(args, "seed", int(args.top_seed_specs))
        mode = "seed"
        seeds = [int(s.strip()) for s in str(args.confirm_seeds).split(",") if s.strip()]
    else:
        raise ValueError(f"Unknown training stage {stage}")

    rows = load_rows(out_root / SUMMARY_NAME)
    done = completed_keys(rows)
    total = len(run_entries) * len(seeds)
    index = 0
    for entry in run_entries:
        for seed in seeds:
            index += 1
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
                index=index,
                total=total,
                updated_at=now_iso(),
            )
            train_one(args, entry, mode=mode, seed=seed)
            write_reports(args)


def write_metric_csv(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    metric_rows = [metric_row(row) for row in rows if int(row.get("returncode", 1)) == 0]
    if not metric_rows:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    return metric_rows


def write_reports(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rows = lrg.dedupe_rows(load_rows(out_root / SUMMARY_NAME))
    write_json(report_root / "morphology_guided_rows.json", {"rows": rows})
    metrics = write_metric_csv(rows, report_root / "morphology_metric_join.csv")
    write_json(report_root / "family_ablation.json", family_stats(metrics))
    ranked = sorted(rows, key=lambda row: result_key(row, args), reverse=True)

    def fmt(x: Any) -> str:
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    def table(table_rows: list[dict[str, Any]]) -> str:
        lines = [
            "| rank | mode | seed | spec | family | BUT acc | bal | macro | recalls G/M/B | minMB | PTB acc | PTB bad | morph | feature | mismatch | note |",
            "|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for i, row in enumerate(table_rows, start=1):
            m = row_metrics(row)
            rec = rec_from_row(row)
            audit = row.get("feature_audit", {}) or {}
            dist = audit.get("distance", {}) or {}
            lines.append(
                f"| {i} | {row.get('mode')} | {row.get('seed')} | `{row.get('spec', {}).get('id')}` | "
                f"{row.get('spec', {}).get('family')} | {fmt(m['acc'])} | {fmt(m['balanced_acc'])} | "
                f"{fmt(m['macro_f1'])} | {fmt(rec[0])}/{fmt(rec[1])}/{fmt(rec[2])} | "
                f"{fmt(m['min_medium_bad'])} | {fmt(m['ptb_acc'])} | {fmt(m['ptb_bad'])} | "
                f"{fmt(dist.get('overall_but_like_score', 0.0))} | {fmt(audit.get('feature_target_score', 0.0))} | "
                f"{audit.get('feature_mismatch', False)} | {row.get('note', '')} |"
            )
        return "\n".join(lines)

    hypothesis_ranked = [r for r in ranked if r.get("mode") == "hypothesis"]
    quick_ranked = [r for r in ranked if r.get("mode") == "quick"]
    full_ranked = [r for r in ranked if r.get("mode") == "full"]
    seed_ranked = [r for r in ranked if r.get("mode") == "seed"]
    md = [
        "# BUT 10s Morphology-Guided Synthetic Grid",
        "",
        "Formal protocol: BUT 10s P1, validation-only calibration, test reporting only.",
        "",
        "## Anchor",
        "",
        "`b10_all_bad_wearable`: acc 0.7735, balanced 0.8045, macro-F1 0.7238, recalls 0.824/0.724/0.866.",
        "",
        "## Hypothesis Verifier Top 20",
        table(hypothesis_ranked[:20]) if hypothesis_ranked else "_No hypothesis rows yet._",
        "",
        "## Guided Quick Top 30",
        table(quick_ranked[:30]) if quick_ranked else "_No guided quick rows yet._",
        "",
        "## Full Confirmation Top 20",
        table(full_ranked[:20]) if full_ranked else "_No full rows yet._",
        "",
        "## Seed Confirmation",
        table(seed_ranked[:20]) if seed_ranked else "_No seed rows yet._",
        "",
        "## Interpretation Guard",
        "",
        "Feature target score is a tie-breaker only. Promotion requires BUT macro/balanced/min(M,B) and PTB sanity; a feature-only improvement without BUT metric movement is marked as proxy-insufficient.",
    ]
    (report_root / "guided_grid_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (report_root / "hypothesis_verifier_summary.md").write_text("\n".join(md[:13]) + "\n", encoding="utf-8")
    (report_root / "final_morphology_guided_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def family_stats(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        fam = str(row.get("family", "unknown"))
        stat = stats.setdefault(
            fam,
            {
                "n": 0,
                "best_macro_f1": 0.0,
                "best_balanced_acc": 0.0,
                "best_min_medium_bad": 0.0,
                "best_feature_target_score": 0.0,
            },
        )
        stat["n"] += 1
        stat["best_macro_f1"] = max(stat["best_macro_f1"], float(row.get("macro_f1", 0.0)))
        stat["best_balanced_acc"] = max(stat["best_balanced_acc"], float(row.get("balanced_acc", 0.0)))
        stat["best_min_medium_bad"] = max(stat["best_min_medium_bad"], float(row.get("min_medium_bad", 0.0)))
        stat["best_feature_target_score"] = max(stat["best_feature_target_score"], float(row.get("feature_target_score", 0.0)))
    return stats


def write_specs(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    specs = {
        "hypothesis": hypothesis_entries(),
        "guided_current": guided_entries_from_templates(top_hypothesis_templates(args), max_entries=int(args.max_quick_runs)),
        "default_templates": default_template_entries(),
    }
    write_json(out_root / "morphology_guided_specs.json", specs)


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / STATE_NAME
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    write_specs(args)
    if args.stage in {"hypothesis_train", "quick_train", "full_train", "seed_confirm"}:
        run_stage(args, args.stage)
    elif args.stage == "report":
        write_reports(args)
    elif args.stage == "all":
        run_stage(args, "hypothesis_train")
        write_specs(args)
        run_stage(args, "quick_train")
        run_stage(args, "full_train")
        run_stage(args, "seed_confirm")
        write_reports(args)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s morphology-guided synthetic grid.")
    parser.add_argument("--stage", choices=("hypothesis_train", "quick_train", "full_train", "seed_confirm", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirm_seeds", default="0,1,2")
    parser.add_argument("--max_hypothesis_runs", type=int, default=24)
    parser.add_argument("--max_quick_runs", type=int, default=96)
    parser.add_argument("--top_full", type=int, default=12)
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
    parser.add_argument("--audit_max_per_class", type=int, default=400)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
