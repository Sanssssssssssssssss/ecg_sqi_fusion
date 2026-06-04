"""BUT 10s sample-level fatal-subtype OR generator grid.

This runner follows the evidence from the fatal-OR pass: BUT bad does not look
like every bad sample has every artifact.  Instead, a sample can be bad because
one or two fatal usability dimensions fail.  The generator therefore assigns
each bad sample a primary and optional secondary subtype, while keeping medium
as a QRS-usable/detail-unreliable state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_morphology_guided_grid_10s as mg
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT, class_array, load_ptb_artifact
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    apply_bad_spec,
    burst_noise,
    evaluate_checkpoint_10s,
    highpass_noise,
    lowfreq_drift,
    normalize_noise,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    smooth_rows,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_generator_morph_sweet_grid import add_local_events, add_pulse
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.run import FS_TARGET, N_TARGET


RUN_TAG = "e311_but_sample_or_subtype_grid_10s_2026_06_04"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
STATE_NAME = "sample_or_subtype_state.json"
SUMMARY_NAME = "sample_or_subtype_summary.jsonl"
SUBTYPES = ("qrs_confound", "contact_flat", "motion_burst", "morph_break", "clipping_lowamp", "baseline_jump")

B10_ANCHOR = {
    "acc": 0.7735,
    "balanced_acc": 0.8045,
    "macro_f1": 0.7238,
    "recall_good": 0.824,
    "recall_medium": 0.724,
    "recall_bad": 0.866,
}
H_BAD_RESCUE_05 = {
    "acc": 0.8229,
    "balanced_acc": 0.8177,
    "macro_f1": 0.7454,
    "recall_good": 0.887,
    "recall_medium": 0.773,
    "recall_bad": 0.793,
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_entry(family: str, name: str, changes: dict[str, Any], class_weight: str, note: str) -> dict[str, Any]:
    spec = mg.lrg.base_spec()
    spec.update(
        {
            "residual_scale_good": 0.74,
            "residual_scale_medium": 1.04,
            "residual_scale_bad": 1.08,
            "medium_event_prob": 0.66,
            "medium_pseudo_peak_rate": 0.08,
            "medium_inverse_peak_rate": 0.14,
            "medium_step_rate": 0.16,
            "medium_ramp_rate": 0.14,
            "medium_contact_rate": 0.04,
            "medium_qrs_preserve": 0.997,
            "bad_base_low_amp": 0.96,
            "bad_base_residual_scale": 1.08,
            "secondary_prob": 0.30,
            "subtype_weights": {
                "qrs_confound": 1.0,
                "contact_flat": 1.0,
                "motion_burst": 1.0,
                "morph_break": 1.0,
                "clipping_lowamp": 1.0,
                "baseline_jump": 1.0,
            },
            "qrs_confound_strength": 1.0,
            "contact_flat_strength": 1.0,
            "motion_burst_strength": 1.0,
            "morph_break_strength": 1.0,
            "clipping_lowamp_strength": 1.0,
            "baseline_jump_strength": 1.0,
        }
    )
    spec.update(changes)
    spec["id"] = f"{family}_{name}"
    spec["family"] = family
    return {"spec": spec, "class_weight": class_weight, "note": note}


def entries() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    mixtures = [
        (
            "balanced",
            {"qrs_confound": 1.0, "contact_flat": 1.0, "motion_burst": 1.0, "morph_break": 1.0, "clipping_lowamp": 1.0, "baseline_jump": 1.0},
            "balanced fatal subtype mixture",
        ),
        (
            "contact_heavy",
            {"qrs_confound": 0.7, "contact_flat": 1.7, "motion_burst": 1.0, "morph_break": 0.7, "clipping_lowamp": 1.2, "baseline_jump": 1.5},
            "BUT-like contact/flat/baseline-heavy bad",
        ),
        (
            "qrs_heavy",
            {"qrs_confound": 1.8, "contact_flat": 0.8, "motion_burst": 0.7, "morph_break": 1.6, "clipping_lowamp": 0.6, "baseline_jump": 0.7},
            "QRS-confounding and morphology-break bad",
        ),
        (
            "wearable_heavy",
            {"qrs_confound": 0.8, "contact_flat": 1.5, "motion_burst": 1.7, "morph_break": 0.7, "clipping_lowamp": 1.4, "baseline_jump": 1.2},
            "wearable motion/contact/noisy bad",
        ),
    ]
    medium_shells = [
        ("m03", 0.72, 0.24, 0.22, 0.16, 0.06, 0.997, "medium-independent shell"),
        ("m04", 0.70, 0.28, 0.18, 0.20, 0.04, 0.997, "medium with local baseline/derivative instability"),
        ("soft", 0.58, 0.12, 0.10, 0.12, 0.03, 0.999, "soft medium, stronger bad contrast"),
    ]
    secondary_probs = [0.15, 0.35]
    cws = ["1.08,1.58,1.92", "1.10,1.64,1.90", "1.12,1.66,1.86"]
    idx = 1
    for mix_name, weights, mix_note in mixtures:
        for shell_name, event, step, ramp, inv, contact, qrs_preserve, shell_note in medium_shells:
            for secondary_prob in secondary_probs:
                cw = cws[(idx - 1) % len(cws)]
                out.append(
                    make_entry(
                        "sample_or",
                        f"{idx:02d}_{mix_name}_{shell_name}_s{int(secondary_prob * 100):02d}",
                        {
                            "subtype_weights": weights,
                            "secondary_prob": secondary_prob,
                            "medium_event_prob": event,
                            "medium_step_rate": step,
                            "medium_ramp_rate": ramp,
                            "medium_inverse_peak_rate": inv,
                            "medium_contact_rate": contact,
                            "medium_qrs_preserve": qrs_preserve,
                            "medium_pseudo_peak_rate": 0.08 if shell_name != "soft" else 0.05,
                        },
                        cw,
                        f"{mix_note}; {shell_note}; bad samples trigger 1-2 fatal subtypes.",
                    )
                )
                idx += 1

    # Focused controls around the best prior evidence.
    out.extend(
        [
            make_entry(
                "sample_or_control",
                "hbad05_like",
                {
                    "subtype_weights": {"qrs_confound": 1.4, "contact_flat": 0.8, "motion_burst": 0.8, "morph_break": 1.2, "clipping_lowamp": 0.5, "baseline_jump": 1.2},
                    "secondary_prob": 0.22,
                    "medium_event_prob": 0.58,
                    "medium_step_rate": 0.12,
                    "medium_ramp_rate": 0.10,
                    "medium_inverse_peak_rate": 0.12,
                    "medium_contact_rate": 0.03,
                    "qrs_confound_strength": 1.12,
                    "baseline_jump_strength": 1.08,
                },
                "1.08,1.54,1.92",
                "h_bad_rescue_05-style medium with sample-level fatal bad.",
            ),
            make_entry(
                "sample_or_control",
                "b10_like_or",
                {
                    "subtype_weights": {"qrs_confound": 1.0, "contact_flat": 1.4, "motion_burst": 1.2, "morph_break": 0.8, "clipping_lowamp": 1.1, "baseline_jump": 1.2},
                    "secondary_prob": 0.50,
                    "medium_event_prob": 0.50,
                    "medium_step_rate": 0.10,
                    "medium_ramp_rate": 0.08,
                    "medium_inverse_peak_rate": 0.10,
                    "medium_contact_rate": 0.03,
                },
                "1.04,1.56,1.86",
                "b10-style broad bad converted to sample-level OR subtypes.",
            ),
            make_entry(
                "sample_or_control",
                "bad_recall_guard",
                {
                    "subtype_weights": {"qrs_confound": 1.5, "contact_flat": 1.4, "motion_burst": 1.2, "morph_break": 1.2, "clipping_lowamp": 0.8, "baseline_jump": 1.2},
                    "secondary_prob": 0.60,
                    "medium_event_prob": 0.62,
                    "medium_step_rate": 0.12,
                    "medium_ramp_rate": 0.10,
                    "medium_inverse_peak_rate": 0.12,
                    "medium_contact_rate": 0.025,
                    "qrs_confound_strength": 1.18,
                    "contact_flat_strength": 1.12,
                },
                "1.06,1.50,1.98",
                "Bad-recall guard: strong fatal OR with soft medium.",
            ),
            make_entry(
                "sample_or_control",
                "medium_guard",
                {
                    "subtype_weights": {"qrs_confound": 0.9, "contact_flat": 1.0, "motion_burst": 0.8, "morph_break": 0.8, "clipping_lowamp": 0.7, "baseline_jump": 1.2},
                    "secondary_prob": 0.18,
                    "medium_event_prob": 0.76,
                    "medium_step_rate": 0.28,
                    "medium_ramp_rate": 0.20,
                    "medium_inverse_peak_rate": 0.20,
                    "medium_contact_rate": 0.045,
                    "medium_qrs_preserve": 0.998,
                },
                "1.12,1.68,1.84",
                "Medium guard: make medium independent while bad remains subtype OR.",
            ),
        ]
    )
    return out


def subtype_names_and_weights(spec: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    raw = spec.get("subtype_weights", {})
    weights = np.array([float(raw.get(s, 0.0)) for s in SUBTYPES], dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(SUBTYPES), dtype=np.float64)
    weights = weights / weights.sum()
    return list(SUBTYPES), weights


def row_std(clean_row: np.ndarray) -> float:
    return float(np.std(clean_row) + 1e-6)


def apply_qrs_confound(x: np.ndarray, clean: np.ndarray, qrs: np.ndarray, rng: np.random.Generator, strength: float) -> None:
    std = row_std(clean)
    x -= float(0.42 + 0.24 * strength) * clean * qrs
    q_positions = np.flatnonzero(qrs > 0.2)
    if len(q_positions):
        for _ in range(int(rng.integers(2, 6))):
            center = int(rng.choice(q_positions))
            add_pulse(x, center, int(rng.integers(4, 16)), float(rng.uniform(0.35, 0.95)) * std * strength, float(rng.choice([-1.0, 1.0])))
    for _ in range(int(rng.integers(1, 4))):
        center = int(rng.integers(FS_TARGET // 3, N_TARGET - FS_TARGET // 3))
        add_pulse(x, center, int(rng.integers(3, 13)), float(rng.uniform(0.22, 0.70)) * std * strength, float(rng.choice([-1.0, 1.0])))


def apply_contact_flat(x: np.ndarray, clean: np.ndarray, rng: np.random.Generator, strength: float) -> None:
    std = row_std(clean)
    for _ in range(int(rng.integers(1, 3))):
        width = int(rng.integers(FS_TARGET // 2, FS_TARGET * 4))
        start = int(rng.integers(0, max(1, N_TARGET - width)))
        end = start + width
        if rng.random() < 0.55:
            level = float(np.median(x[max(0, start - FS_TARGET // 2) : min(N_TARGET, end + FS_TARGET // 2)]))
            x[start:end] = level + rng.normal(0.0, 0.012 + 0.010 * strength, size=width).astype(np.float32)
        else:
            x[start:end] *= float(rng.uniform(0.03, 0.34 + 0.12 * (1.0 - min(strength, 1.0))))
    x += float(0.04 * strength) * std * lowfreq_drift(rng, 1)[0]


def apply_motion_burst(x: np.ndarray, clean: np.ndarray, rng: np.random.Generator, strength: float) -> None:
    std = row_std(clean)
    x += float(0.18 + 0.12 * strength) * std * burst_noise(rng, 1, 1.0)[0]
    x += float(0.14 + 0.10 * strength) * std * normalize_noise(highpass_noise(rng, 1))[0]
    x += float(0.08 + 0.08 * strength) * std * lowfreq_drift(rng, 1)[0]


def apply_morph_break(x: np.ndarray, clean: np.ndarray, qrs: np.ndarray, tst: np.ndarray, critical: np.ndarray, rng: np.random.Generator, strength: float) -> None:
    std = row_std(clean)
    sm = smooth_rows(x[None, :], int(7 + 22 * strength))[0]
    x[:] = x * (1.0 - critical * float(0.28 + 0.28 * strength)) + sm * critical * float(0.28 + 0.28 * strength)
    x -= float(0.25 + 0.18 * strength) * clean * qrs
    x += float(0.22 + 0.18 * strength) * std * normalize_noise(np.random.default_rng(int(rng.integers(0, 2**31 - 1))).normal(size=(1, N_TARGET)).astype(np.float32))[0] * tst
    for _ in range(int(rng.integers(1, 4))):
        center = int(rng.integers(FS_TARGET // 3, N_TARGET - FS_TARGET // 3))
        add_pulse(x, center, int(rng.integers(8, 28)), float(rng.uniform(0.20, 0.65)) * std * strength, -1.0)


def apply_clipping_lowamp(x: np.ndarray, clean: np.ndarray, rng: np.random.Generator, strength: float) -> None:
    hi = float(np.percentile(np.abs(x), rng.uniform(54, 72)))
    if hi > 1e-6:
        x[:] = np.clip(x, -hi, hi)
    x *= float(rng.uniform(0.42, 0.76) - 0.10 * min(strength, 1.2))
    if rng.random() < 0.45:
        width = int(rng.integers(FS_TARGET, FS_TARGET * 5))
        start = int(rng.integers(0, max(1, N_TARGET - width)))
        x[start : start + width] *= float(rng.uniform(0.08, 0.38))


def apply_baseline_jump(x: np.ndarray, clean: np.ndarray, rng: np.random.Generator, strength: float) -> None:
    std = row_std(clean)
    pos = int(rng.integers(FS_TARGET // 2, N_TARGET - FS_TARGET // 2))
    amp = float(rng.normal(0.0, 0.26 + 0.16 * strength)) * std
    x[pos:] += amp
    if rng.random() < 0.55:
        start = int(rng.integers(0, N_TARGET - FS_TARGET))
        width = int(rng.integers(FS_TARGET, FS_TARGET * 4))
        end = min(N_TARGET, start + width)
        x[start:end] += np.linspace(0.0, -0.75 * amp, end - start, dtype=np.float32)


def apply_subtype(
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
    strength = float(spec.get(f"{subtype}_strength", 1.0)) * scale
    if subtype == "qrs_confound":
        apply_qrs_confound(x, clean, qrs, rng, strength)
    elif subtype == "contact_flat":
        apply_contact_flat(x, clean, rng, strength)
    elif subtype == "motion_burst":
        apply_motion_burst(x, clean, rng, strength)
    elif subtype == "morph_break":
        apply_morph_break(x, clean, qrs, tst, critical, rng, strength)
    elif subtype == "clipping_lowamp":
        apply_clipping_lowamp(x, clean, rng, strength)
    elif subtype == "baseline_jump":
        apply_baseline_jump(x, clean, rng, strength)
    else:
        raise ValueError(f"Unknown fatal subtype: {subtype}")


def neutral_bad_apply_spec(spec: dict[str, Any]) -> dict[str, Any]:
    base = dict(spec)
    base.update(
        {
            "residual_scale_bad": float(spec.get("bad_base_residual_scale", 1.08)),
            "qrs_attn_bad": 0.0,
            "smooth_bad": 0.0,
            "drift_bad": 0.0,
            "hf_bad": 0.0,
            "burst_bad": 0.0,
            "contact_loss_bad": 0.0,
            "flatline_bad": 0.0,
            "dropout_bad": 0.0,
            "baseline_step_bad": 0.0,
            "clip_bad": 0.0,
            "spurious_peak_bad": 0.0,
            "low_amp_bad": float(spec.get("bad_base_low_amp", 0.96)),
        }
    )
    return base


def apply_sample_or_spec(ptb: dict[str, Any], spec: dict[str, Any], out_dir: Path, seed: int) -> dict[str, Any]:
    payload = apply_bad_spec(ptb, neutral_bad_apply_spec(spec), out_dir, seed=seed, sample_limit=0)
    x = payload["X_noisy"].astype(np.float32, copy=True)
    labels = payload["labels"].reset_index(drop=True)
    clean = ptb["clean"].astype(np.float32, copy=False)
    qrs = ptb["masks"].get("qrs_mask", np.zeros_like(clean)).astype(np.float32)
    tst = ptb["masks"].get("tst_mask", np.zeros_like(clean)).astype(np.float32)
    critical = ptb["masks"].get("critical_mask", qrs).astype(np.float32)
    y = class_array(labels)
    rng = np.random.default_rng(seed + 1701)

    medium = np.where(y == 1)[0]
    bad = np.where(y == 2)[0]
    add_local_events(x, clean, medium, qrs, spec, rng, "medium")

    names, weights = subtype_names_and_weights(spec)
    primary: list[str] = ["none"] * len(labels)
    secondary: list[str] = ["none"] * len(labels)
    strength: list[float] = [0.0] * len(labels)
    secondary_prob = float(spec.get("secondary_prob", 0.30))
    for idx in bad:
        i = int(idx)
        p = str(rng.choice(names, p=weights))
        primary[i] = p
        scale = float(rng.uniform(0.86, 1.18))
        strength[i] = scale
        apply_subtype(p, x[i], clean[i], qrs[i], tst[i], critical[i], spec, rng, scale)
        if rng.random() < secondary_prob:
            remaining = [n for n in names if n != p]
            rem_weights = np.array([weights[names.index(n)] for n in remaining], dtype=np.float64)
            rem_weights = rem_weights / rem_weights.sum()
            s = str(rng.choice(remaining, p=rem_weights))
            secondary[i] = s
            apply_subtype(s, x[i], clean[i], qrs[i], tst[i], critical[i], spec, rng, scale * float(rng.uniform(0.42, 0.68)))

    labels["fatal_subtype_primary"] = primary
    labels["fatal_subtype_secondary"] = secondary
    labels["fatal_subtype_strength"] = strength
    labels["sample_source"] = labels.get("sample_source", "ptb_synthetic").astype(str) + f"|sample_or:{spec['id']}"
    data_dir = out_dir / "datasets"
    np.savez_compressed(data_dir / "synth_10s_125hz_noisy.npz", X_noisy=x.astype(np.float32))
    np.savez_compressed(data_dir / "signals.npz", X=x[:, None, :].astype(np.float32))
    labels.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)
    labels.to_csv(data_dir / "synth_10s_125hz_labels_with_level.csv", index=False)

    subtype_audit = {
        "spec_id": spec["id"],
        "subtype_weights": spec.get("subtype_weights", {}),
        "secondary_prob": secondary_prob,
        "primary_counts": {str(k): int(v) for k, v in labels.loc[labels["y_class"].astype(str) == "bad", "fatal_subtype_primary"].value_counts().to_dict().items()},
        "secondary_counts": {str(k): int(v) for k, v in labels.loc[labels["y_class"].astype(str) == "bad", "fatal_subtype_secondary"].value_counts().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in labels["y_class"].value_counts().to_dict().items()},
        "split_counts": {str(k): int(v) for k, v in labels["split"].value_counts().to_dict().items()},
    }
    write_json(out_dir / "bad_subtype_audit.json", subtype_audit)
    audit = read_json(out_dir / "data_variant_audit.json")
    audit["sample_or_subtype_audit"] = subtype_audit
    write_json(out_dir / "data_variant_audit.json", audit)
    return {"audit": audit, "X_noisy": x.astype(np.float32), "labels": labels}


def plot_bad_subtype_gallery(clean: np.ndarray, noisy: np.ndarray, labels: pd.DataFrame, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    bad_labels = labels[labels["y_class"].astype(str) == "bad"].reset_index()
    positions: list[int] = []
    for subtype in SUBTYPES:
        idx = bad_labels.index[bad_labels["fatal_subtype_primary"].astype(str) == subtype].tolist()[:2]
        positions.extend([int(bad_labels.loc[i, "index"]) for i in idx])
    if not positions:
        return
    n = len(positions)
    fig, axes = plt.subplots(n, 1, figsize=(13, max(4, 1.25 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    t = np.arange(clean.shape[1]) / FS_TARGET
    for ax, pos in zip(axes, positions):
        ax.plot(t, clean[pos], color="#111827", lw=0.7, label="clean")
        ax.plot(t, noisy[pos], color="#dc2626", lw=0.55, alpha=0.9, label="synthetic noisy")
        ax.set_title(
            f"bad idx={labels.loc[pos, 'idx']} subtype={labels.loc[pos, 'fatal_subtype_primary']} + {labels.loc[pos, 'fatal_subtype_secondary']}",
            fontsize=8,
        )
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def prepare_variant(args: argparse.Namespace, entry: dict[str, Any]) -> Path:
    spec = entry["spec"]
    variant_dir = Path(args.out_root) / "synthetic_variants" / str(spec["id"])
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        ptb = load_ptb_artifact(Path(args.source_artifact_dir))
        ptb["source"] = str(args.source_artifact_dir)
        payload = apply_sample_or_spec(ptb, spec, variant_dir, seed=int(args.seed))
        plot_synthetic_gallery(
            ptb["clean"],
            payload["X_noisy"],
            payload["labels"].reset_index(drop=True),
            variant_dir / "visuals" / "synthetic_gallery.png",
            str(spec["id"]),
        )
        plot_bad_subtype_gallery(
            ptb["clean"],
            payload["X_noisy"],
            payload["labels"].reset_index(drop=True),
            variant_dir / "visuals" / "bad_subtype_gallery.png",
            str(spec["id"]),
        )
    entry["_feature_audit"] = mg.audit_variant(args, variant_dir, entry)
    return variant_dir


def train_one(args: argparse.Namespace, entry: dict[str, Any], mode: str) -> dict[str, Any]:
    spec = entry["spec"]
    spec_id = str(spec["id"])
    variant_dir = prepare_variant(args, entry)
    run_dir = Path(args.out_root) / "runs" / mode / spec_id
    log_dir = Path(args.out_root) / "logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        epochs1, epochs2 = int(args.smoke_epochs_stage1), int(args.smoke_epochs_stage2)
    elif mode == "quick":
        epochs1, epochs2 = int(args.quick_epochs_stage1), int(args.quick_epochs_stage2)
    else:
        epochs1, epochs2 = int(args.full_epochs_stage1), int(args.full_epochs_stage2)
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
        "mode": mode,
        "seed": int(args.seed),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
        "feature_audit": entry.get("_feature_audit", {}),
        "bad_subtype_audit": read_json(variant_dir / "bad_subtype_audit.json") if (variant_dir / "bad_subtype_audit.json").exists() else {},
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "sample_or_subtype_run_summary.json", payload)
    write_json(run_dir / "sample_or_subtype_metric_row.json", mg.metric_row(payload))
    mg.write_visual_review(run_dir, payload)
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    return payload


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def result_key(row: dict[str, Any], args: argparse.Namespace | None = None) -> tuple[float, ...]:
    metrics = mg.row_metrics(row)
    good, medium, bad = mg.rec_from_row(row)
    valid = 1.0
    if args is not None and (metrics["ptb_acc"] < float(args.min_ptb_acc) or metrics["ptb_bad"] < float(args.min_ptb_bad)):
        valid = 0.0
    return (
        valid,
        float(metrics["macro_f1"]),
        float(metrics["balanced_acc"]),
        min(float(medium), float(bad)),
        float(bad),
        float(medium),
        float(good),
    )


def select_full_entries(args: argparse.Namespace, top_n: int) -> list[dict[str, Any]]:
    rows = [r for r in load_rows(Path(args.out_root) / SUMMARY_NAME) if r.get("mode") == "quick" and int(r.get("returncode", 1)) == 0]
    ranked = sorted(rows, key=lambda r: result_key(r, args), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        spec = row.get("spec", {})
        spec_id = str(spec.get("id", ""))
        if not spec_id or spec_id in seen:
            continue
        selected.append({"spec": spec, "class_weight": row.get("class_weight", "1.10,1.64,1.90"), "note": row.get("note", "")})
        seen.add(spec_id)
        if len(selected) >= top_n:
            break
    return selected


def write_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out_root / SUMMARY_NAME)
    ranked = sorted([r for r in rows if int(r.get("returncode", 1)) == 0], key=lambda r: result_key(r, args), reverse=True)
    write_json(report_root / "sample_or_subtype_rows.json", {"rows": rows, "ranked": ranked[:80], "b10_anchor": B10_ANCHOR, "h_bad_rescue_05": H_BAD_RESCUE_05})

    lines = [
        "# BUT 10s Sample-Level Fatal-Subtype OR Grid",
        "",
        "Bad samples receive one primary and optional secondary fatal subtype instead of all bad artifacts at once.",
        "",
        f"b10 anchor: acc {B10_ANCHOR['acc']:.4f}, balanced {B10_ANCHOR['balanced_acc']:.4f}, macro-F1 {B10_ANCHOR['macro_f1']:.4f}, recalls {B10_ANCHOR['recall_good']:.3f}/{B10_ANCHOR['recall_medium']:.3f}/{B10_ANCHOR['recall_bad']:.3f}.",
        f"h_bad_rescue_05: acc {H_BAD_RESCUE_05['acc']:.4f}, balanced {H_BAD_RESCUE_05['balanced_acc']:.4f}, macro-F1 {H_BAD_RESCUE_05['macro_f1']:.4f}, recalls {H_BAD_RESCUE_05['recall_good']:.3f}/{H_BAD_RESCUE_05['recall_medium']:.3f}/{H_BAD_RESCUE_05['recall_bad']:.3f}.",
        "",
        "| rank | mode | spec | cw | BUT acc | bal | macro-F1 | recalls G/M/B | min M/B | PTB acc | PTB bad | denoise | primary subtype mix | note |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    metric_rows: list[dict[str, Any]] = []
    for i, row in enumerate(ranked, start=1):
        m = mg.row_metrics(row)
        good, medium, bad = mg.rec_from_row(row)
        audit = row.get("bad_subtype_audit", {}) or {}
        primary_counts = audit.get("primary_counts", {}) or {}
        subtype_mix = ",".join(f"{k}:{v}" for k, v in sorted(primary_counts.items()))
        spec = row.get("spec", {}) or {}
        metric_rows.append(
            {
                "rank": i,
                "mode": row.get("mode"),
                "spec_id": spec.get("id"),
                "family": spec.get("family"),
                "class_weight": row.get("class_weight"),
                "but_acc": m["acc"],
                "but_balanced_acc": m["balanced_acc"],
                "but_macro_f1": m["macro_f1"],
                "recall_good": good,
                "recall_medium": medium,
                "recall_bad": bad,
                "min_medium_bad": min(medium, bad),
                "ptb_acc": m["ptb_acc"],
                "ptb_bad": m["ptb_bad"],
                "denoise_score": m["denoise_score"],
                "primary_subtype_mix": subtype_mix,
                "run_dir": row.get("run_dir", ""),
            }
        )
        lines.append(
            f"| {i} | {row.get('mode')} | {spec.get('id')} | {row.get('class_weight')} | "
            f"{m['acc']:.4f} | {m['balanced_acc']:.4f} | {m['macro_f1']:.4f} | "
            f"{good:.3f}/{medium:.3f}/{bad:.3f} | {min(medium, bad):.3f} | "
            f"{m['ptb_acc']:.4f} | {m['ptb_bad']:.4f} | {m['denoise_score']:.3f} | "
            f"{subtype_mix} | {row.get('note', '')} |"
        )
    (report_root / "sample_or_subtype_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(report_root / "sample_or_subtype_metric_join.csv", index=False)


def run_entries(args: argparse.Namespace, run_entries_: list[dict[str, Any]], mode: str) -> None:
    out_root = Path(args.out_root)
    state_path = out_root / STATE_NAME
    completed = {str(r.get("spec", {}).get("id", "")) for r in load_rows(out_root / SUMMARY_NAME) if r.get("mode") == mode}
    for i, entry in enumerate(run_entries_, start=1):
        spec_id = str(entry["spec"]["id"])
        if spec_id in completed:
            continue
        update_state(state_path, status=f"{mode}_training", stage=mode, current=spec_id, index=i, total=len(run_entries_), updated_at=now_iso())
        train_one(args, entry, mode)
        write_report(args)


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / STATE_NAME
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    all_entries = entries()[: int(args.max_runs)]
    write_json(out_root / "sample_or_subtype_specs.json", {"entries": all_entries})
    if args.stage in {"smoke", "all"}:
        run_entries(args, all_entries[: int(args.smoke_runs)], "smoke")
    if args.stage in {"quick_train", "all"}:
        run_entries(args, all_entries, "quick")
    if args.stage in {"full_train", "all"}:
        run_entries(args, select_full_entries(args, int(args.top_full)), "full")
    if args.stage in {"report", "all"}:
        write_report(args)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s sample-level fatal-subtype OR grid.")
    parser.add_argument("--stage", choices=("smoke", "quick_train", "full_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=28)
    parser.add_argument("--smoke_runs", type=int, default=2)
    parser.add_argument("--top_full", type=int, default=6)
    parser.add_argument("--smoke_epochs_stage1", type=int, default=1)
    parser.add_argument("--smoke_epochs_stage2", type=int, default=1)
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
