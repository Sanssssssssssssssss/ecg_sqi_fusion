"""Medium-guard + bad-boundary synthetic grid for BUT 10s.

The grid starts from Sensors2025-style EM/MA SNR synthesis and optional BW
overlay, then adds prediction-independent sample-level bad subtypes.  The goal
is not to make every bad sample equally awful; each bad window gets one or two
fatal usability failures while medium keeps visible QRS and unreliable detail.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import real_noise_snr_but_match_grid_10s as rn
from src.transformer_pipeline.external_benchmarks import sensors2025_noise_synthesis_but_10s as s25
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    evaluate_checkpoint_10s,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_generator_morph_sweet_grid import add_local_events
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.but_sample_or_subtype_grid_10s import (
    SUBTYPES,
    apply_subtype,
    plot_bad_subtype_gallery,
    subtype_names_and_weights,
)
from src.transformer_pipeline.external_benchmarks.run import FS_TARGET, N_TARGET


ROOT = rn.ROOT
RUN_TAG = "e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_BUT_PROTOCOL = PROTOCOL_OUT_ROOT / "protocols" / "p1_current_10s_center"
DEFAULT_NSTDB_ROOT = ROOT / "data" / "physionet" / "nstdb"
DEFAULT_CINC_DIR = ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02" / "processed" / "cinc2017"
SENSORS_OUT = ROOT / "outputs" / "external_benchmarks" / "e311_sensors2025_noise_synthesis_but_10s_2026_06_05"

STATE_NAME = "medium_guard_bad_boundary_grid_state.json"
SPEC_NAME = "medium_guard_bad_boundary_specs.json"
SCAN_NAME = "medium_guard_bad_boundary_distance_leaderboard.csv"
SUMMARY_NAME = "medium_guard_bad_boundary_summary.jsonl"
CLASS_NAMES = ("good", "medium", "bad")

H_BAD_RESCUE_05 = {
    "acc": 0.8229,
    "balanced_acc": 0.8177,
    "macro_f1": 0.7454,
    "recall_good_medium_bad": [0.887, 0.773, 0.793],
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(text: str) -> str:
    return (
        text.lower()
        .replace("+", "_")
        .replace("-", "m")
        .replace(".", "p")
        .replace(",", "_")
        .replace(" ", "_")
    )


def paper_table_strict() -> dict[str, Any]:
    return {
        "name": "paper_table_strict",
        "good": [16.0, 18.0],
        "medium": [5.0, 14.0],
        "bad": [-5.0, -3.0],
        "sampler": "uniform",
        "note": "Sensors2025 Table-1-like bins; target SNR uses EM+MA only.",
    }


def mix_profiles() -> dict[str, dict[str, Any]]:
    return {
        "bw_badstrong": {
            "name": "em_ma_bw_badstrong",
            "em_weight": 0.45,
            "ma_weight": 0.55,
            "bw_overlay": {"good": [0.002, 0.012], "medium": [0.015, 0.045], "bad": [0.080, 0.160]},
        },
        "bw_mid": {
            "name": "em_ma_bw_mid",
            "em_weight": 0.50,
            "ma_weight": 0.50,
            "bw_overlay": {"good": [0.004, 0.020], "medium": [0.020, 0.060], "bad": [0.050, 0.120]},
        },
        "ma_heavy": {
            "name": "em_ma_ma_heavy",
            "em_weight": 0.30,
            "ma_weight": 0.70,
            "bw_overlay": None,
        },
    }


def base_spec(name: str, mix_key: str, class_weight: str, changes: dict[str, Any], note: str) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": "",
        "family": "medium_guard_bad_boundary",
        "snr_profile": paper_table_strict(),
        "mix": mix_profiles()[mix_key],
        "cinc_bad_fraction": 0.0,
        "class_weight": class_weight,
        "paper_constraints": {
            "target_snr_noise": "EM+MA only",
            "bw_participates_in_target_snr": False,
            "model_input_protocol": "project mainline 10s@125Hz, not paper 5s@200Hz",
        },
        "secondary_prob": 0.25,
        "medium_event_prob": 0.62,
        "medium_pseudo_peak_rate": 0.06,
        "medium_inverse_peak_rate": 0.12,
        "medium_step_rate": 0.14,
        "medium_ramp_rate": 0.12,
        "medium_contact_rate": 0.025,
        "medium_qrs_preserve": 0.998,
        "subtype_weights": {
            "qrs_confound": 1.0,
            "contact_flat": 1.0,
            "motion_burst": 1.0,
            "morph_break": 1.0,
            "clipping_lowamp": 0.8,
            "baseline_jump": 1.0,
        },
        "qrs_confound_strength": 1.0,
        "contact_flat_strength": 1.0,
        "motion_burst_strength": 1.0,
        "morph_break_strength": 1.0,
        "clipping_lowamp_strength": 1.0,
        "baseline_jump_strength": 1.0,
        "note": note,
    }
    spec.update(changes)
    spec["id"] = stable_id(f"mg_{name}__{mix_key}__cw{class_weight}")
    return spec


def make_specs(max_specs: int) -> list[dict[str, Any]]:
    bad_mixes = {
        "balanced_or": {"qrs_confound": 1.1, "contact_flat": 1.0, "motion_burst": 1.0, "morph_break": 1.0, "clipping_lowamp": 0.8, "baseline_jump": 1.0},
        "visible_unusable": {"qrs_confound": 0.8, "contact_flat": 0.8, "motion_burst": 1.1, "morph_break": 0.8, "clipping_lowamp": 0.7, "baseline_jump": 1.8},
        "qrs_confound": {"qrs_confound": 1.8, "contact_flat": 0.6, "motion_burst": 0.8, "morph_break": 1.5, "clipping_lowamp": 0.5, "baseline_jump": 0.8},
        "contact_lowamp": {"qrs_confound": 0.7, "contact_flat": 1.7, "motion_burst": 0.8, "morph_break": 0.7, "clipping_lowamp": 1.5, "baseline_jump": 1.2},
        "wearable_motion": {"qrs_confound": 0.9, "contact_flat": 1.2, "motion_burst": 1.8, "morph_break": 0.8, "clipping_lowamp": 1.0, "baseline_jump": 1.2},
    }
    medium_guards = {
        "m_guard": {"medium_event_prob": 0.70, "medium_step_rate": 0.20, "medium_ramp_rate": 0.16, "medium_inverse_peak_rate": 0.16, "medium_contact_rate": 0.025, "medium_qrs_preserve": 0.999},
        "m_strong_detail": {"medium_event_prob": 0.78, "medium_step_rate": 0.26, "medium_ramp_rate": 0.20, "medium_inverse_peak_rate": 0.20, "medium_contact_rate": 0.035, "medium_qrs_preserve": 0.998},
        "m_soft": {"medium_event_prob": 0.58, "medium_step_rate": 0.12, "medium_ramp_rate": 0.10, "medium_inverse_peak_rate": 0.10, "medium_contact_rate": 0.015, "medium_qrs_preserve": 0.999},
    }
    intensities = {
        "softbad": {"secondary_prob": 0.18, "qrs_confound_strength": 0.92, "contact_flat_strength": 0.92, "motion_burst_strength": 0.95, "morph_break_strength": 0.90, "baseline_jump_strength": 0.95},
        "midbad": {"secondary_prob": 0.30, "qrs_confound_strength": 1.05, "contact_flat_strength": 1.03, "motion_burst_strength": 1.05, "morph_break_strength": 1.00, "baseline_jump_strength": 1.08},
        "badguard": {"secondary_prob": 0.46, "qrs_confound_strength": 1.16, "contact_flat_strength": 1.12, "motion_burst_strength": 1.12, "morph_break_strength": 1.10, "baseline_jump_strength": 1.18},
    }
    class_weights = ["1.00,1.45,1.75", "1.00,1.55,1.70", "1.00,1.62,1.78"]
    mix_keys = ["bw_badstrong", "bw_mid", "ma_heavy"]
    specs: list[dict[str, Any]] = []
    for mix_key in mix_keys:
        for bad_name, bad_weights in bad_mixes.items():
            for med_name, med in medium_guards.items():
                for intensity_name, intensity in intensities.items():
                    cw = class_weights[len(specs) % len(class_weights)]
                    changes = {"subtype_weights": bad_weights, **med, **intensity}
                    name = f"{bad_name}_{med_name}_{intensity_name}"
                    note = f"Paper strict EM/MA SNR + {mix_key}; bad sample-level OR {bad_name}; medium guard {med_name}; intensity {intensity_name}."
                    specs.append(base_spec(name, mix_key, cw, changes, note))
                    if len(specs) >= int(max_specs):
                        return specs
    return specs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def wait_for_sensors(args: argparse.Namespace) -> None:
    if not bool(args.wait_for_sensors):
        return
    state_path = SENSORS_OUT / s25.STATE_NAME
    while True:
        active_state = False
        if state_path.exists():
            try:
                state = read_json(state_path)
                active_state = str(state.get("status", "")).lower() in {"running", "quick_training", "full_training"}
            except Exception:
                active_state = False
        if not active_state:
            return
        update_state(Path(args.out_root) / STATE_NAME, status="waiting_for_sensors2025", sensors_state=str(state_path), updated_at=now_iso())
        time.sleep(max(30, int(args.wait_poll_sec)))


def apply_medium_guard_or(spec: dict[str, Any], variant_dir: Path, seed: int) -> dict[str, Any]:
    data_dir = variant_dir / "datasets"
    clean = np.load(data_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    noisy = np.load(data_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    labels = pd.read_csv(data_dir / "synth_10s_125hz_labels_with_level.csv").reset_index(drop=True)
    masks_npz = np.load(data_dir / "synth_10s_125hz_local_mask.npz")
    qrs = masks_npz["qrs_mask"].astype(np.float32) if "qrs_mask" in masks_npz else np.zeros_like(clean)
    tst = masks_npz["tst_mask"].astype(np.float32) if "tst_mask" in masks_npz else np.zeros_like(clean)
    critical = masks_npz["critical_mask"].astype(np.float32) if "critical_mask" in masks_npz else qrs
    rng = np.random.default_rng(seed + 311)
    x = noisy.astype(np.float32, copy=True)
    y = labels["y_class"].map({"good": 0, "medium": 1, "bad": 2}).to_numpy(dtype=int)
    medium_idx = np.where(y == 1)[0]
    bad_idx = np.where(y == 2)[0]

    add_local_events(x, clean, medium_idx, qrs, spec, rng, "medium")

    names, weights = subtype_names_and_weights(spec)
    primary = ["none"] * len(labels)
    secondary = ["none"] * len(labels)
    strengths = [0.0] * len(labels)
    for idx in bad_idx:
        i = int(idx)
        p = str(rng.choice(names, p=weights))
        scale = float(rng.uniform(0.84, 1.18))
        primary[i] = p
        strengths[i] = scale
        apply_subtype(p, x[i], clean[i], qrs[i], tst[i], critical[i], spec, rng, scale)
        if rng.random() < float(spec.get("secondary_prob", 0.25)):
            remaining = [n for n in names if n != p]
            rem_w = np.array([weights[names.index(n)] for n in remaining], dtype=np.float64)
            rem_w = rem_w / rem_w.sum()
            s = str(rng.choice(remaining, p=rem_w))
            secondary[i] = s
            apply_subtype(s, x[i], clean[i], qrs[i], tst[i], critical[i], spec, rng, scale * float(rng.uniform(0.35, 0.66)))

    labels["fatal_subtype_primary"] = primary
    labels["fatal_subtype_secondary"] = secondary
    labels["fatal_subtype_strength"] = strengths
    labels["sample_source"] = labels.get("sample_source", "sensors2025").astype(str) + f"|medium_guard_or:{spec['id']}"
    np.savez_compressed(data_dir / "synth_10s_125hz_noisy.npz", X_noisy=x.astype(np.float32))
    np.savez_compressed(data_dir / "signals.npz", X=x[:, None, :].astype(np.float32))
    labels.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)
    labels.to_csv(data_dir / "synth_10s_125hz_labels_with_level.csv", index=False)
    audit = read_json(variant_dir / "data_variant_audit.json") if (variant_dir / "data_variant_audit.json").exists() else {}
    subtype_audit = {
        "spec_id": spec["id"],
        "subtype_weights": spec.get("subtype_weights", {}),
        "secondary_prob": float(spec.get("secondary_prob", 0.25)),
        "primary_counts": {str(k): int(v) for k, v in labels.loc[labels["y_class"].astype(str) == "bad", "fatal_subtype_primary"].value_counts().to_dict().items()},
        "secondary_counts": {str(k): int(v) for k, v in labels.loc[labels["y_class"].astype(str) == "bad", "fatal_subtype_secondary"].value_counts().to_dict().items()},
    }
    audit["medium_guard_bad_boundary"] = subtype_audit
    write_json(variant_dir / "data_variant_audit.json", audit)
    write_json(variant_dir / "bad_subtype_audit.json", subtype_audit)
    return {"X_noisy": x[:, None, :], "X_clean": clean[:, None, :], "labels": labels, "audit": audit}


def make_variant(
    args: argparse.Namespace,
    spec: dict[str, Any],
    sample_per_class: int,
    *,
    force_regenerate: bool = False,
) -> tuple[Path, dict[str, Any]]:
    variant_dir = Path(args.out_root) / "synthetic_variants" / str(spec["id"])
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if labels_path.exists() and not args.force and not force_regenerate:
        audit = read_json(variant_dir / "data_variant_audit.json") if (variant_dir / "data_variant_audit.json").exists() else {}
        return variant_dir, {"audit": audit}
    ptb = rn.load_ptb_artifact(Path(args.source_artifact_dir))
    tracks = rn.ensure_nstdb_tracks(Path(args.nstdb_root))["tracks"]
    cinc_pool = s25.load_cinc_bad_pool(Path(args.cinc_dir))
    made = s25.make_dataset_variant(ptb, spec, tracks, variant_dir, seed=int(args.seed), sample_per_class=sample_per_class, cinc_pool=cinc_pool)
    made2 = apply_medium_guard_or(spec, variant_dir, seed=int(args.seed))
    plot_synthetic_gallery(
        made["X_clean"][:, 0],
        made2["X_noisy"][:, 0],
        made2["labels"],
        variant_dir / "visuals" / "synthetic_gallery.png",
        str(spec["id"]),
    )
    plot_bad_subtype_gallery(
        made["X_clean"][:, 0],
        made2["X_noisy"][:, 0],
        made2["labels"],
        variant_dir / "visuals" / "bad_subtype_gallery.png",
        str(spec["id"]),
    )
    return variant_dir, made2


def score_one(args: argparse.Namespace, spec: dict[str, Any], but_X: np.ndarray, but_meta: pd.DataFrame, but_feat: pd.DataFrame) -> dict[str, Any]:
    variant_dir, made = make_variant(args, spec, int(args.scan_sample_per_class))
    report_path = variant_dir / "but_distance_report.json"
    if report_path.exists() and not args.force:
        return read_json(report_path)
    X_synth = np.load(variant_dir / "datasets" / "signals.npz")["X"].astype(np.float32)
    labels = pd.read_csv(variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv")
    meta = labels.copy()
    meta["dataset"] = "PTB medium-guard synthetic"
    meta["class_name"] = meta["y_class"].astype(str)
    synth_feat = rn.extract_features(X_synth, meta, variant_dir / "synthetic_morph_features.csv", max_rows=0)
    morph_dist = rn.classwise_ks_distance(but_feat, synth_feat, rn.FEATURE_COLUMNS)
    sqi_dist = rn.sqi_distance(
        but_X,
        X_synth,
        but_meta,
        meta,
        max_rows_per_class=int(args.sqi_rows_per_class),
        seed=int(args.seed) + 101,
    )
    domain_bal = rn.domain_separability(but_feat, synth_feat, rn.FEATURE_COLUMNS, seed=int(args.seed))
    medium_distance = float(morph_dist["by_class"].get("medium", math.nan))
    bad_distance = float(morph_dist["by_class"].get("bad", math.nan))
    overall = float(morph_dist["overall"])
    sqi_overall = float(sqi_dist["overall"])
    # Medium guard is the hard requirement, but bad class distance must not be
    # ignored; this score is only for pre-training ranking.
    target_score = float(0.34 * overall + 0.24 * medium_distance + 0.20 * bad_distance + 0.12 * sqi_overall + 0.10 * domain_bal)
    row = {
        "variant_id": str(spec["id"]),
        "family": spec.get("family", "medium_guard_bad_boundary"),
        "spec": spec,
        "variant_dir": str(variant_dir),
        "overall_but_like_score": target_score,
        "morph_overall_distance": overall,
        "morph_medium_distance": medium_distance,
        "morph_bad_distance": bad_distance,
        "morph_by_class": morph_dist["by_class"],
        "sqi_overall_distance": sqi_overall,
        "sqi_by_class": sqi_dist["by_class"],
        "domain_separability_bal_acc": domain_bal,
        "audit": made.get("audit", {}),
    }
    write_json(variant_dir / "synthetic_feature_profile.json", {"morph_features": synth_feat.describe().to_dict(), "audit": made.get("audit", {})})
    write_json(variant_dir / "but_distance_report.json", row)
    write_json(variant_dir / "morph_distance_rows.json", morph_dist["rows"])
    write_json(variant_dir / "sqi_distance_rows.json", sqi_dist["rows"])
    return row


def score_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="scan_running", stage="score_specs", updated_at=now_iso())
    specs = make_specs(int(args.max_specs))
    write_json(out_root / SPEC_NAME, {"rows": specs})
    write_json(report_root / SPEC_NAME, {"rows": specs})
    but_X, but_meta = rn.load_but(Path(args.but_protocol_dir))
    sample_idx = pd.Series(range(len(but_meta))).sample(
        n=min(len(but_meta), int(args.but_feature_max_rows) * len(CLASS_NAMES)),
        random_state=int(args.seed),
    ).sort_values()
    but_feat = rn.extract_features(
        but_X[sample_idx.to_numpy()],
        but_meta.iloc[sample_idx.to_numpy()].reset_index(drop=True),
        out_root / "but_morph_features_scan.csv",
        max_rows=0,
    )
    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status="scan_running", current=spec["id"], completed=i - 1, total=len(specs), updated_at=now_iso())
        rows.append(score_one(args, spec, but_X, but_meta, but_feat))
    rows = sorted(rows, key=lambda r: float(r["overall_but_like_score"]))
    df = pd.DataFrame(
        [
            {
                "rank": i,
                "variant_id": row["variant_id"],
                "overall_but_like_score": row["overall_but_like_score"],
                "morph_overall_distance": row["morph_overall_distance"],
                "morph_medium_distance": row["morph_medium_distance"],
                "morph_bad_distance": row["morph_bad_distance"],
                "sqi_overall_distance": row["sqi_overall_distance"],
                "domain_separability_bal_acc": row["domain_separability_bal_acc"],
                "variant_dir": row["variant_dir"],
            }
            for i, row in enumerate(rows, start=1)
        ]
    )
    df.to_csv(out_root / SCAN_NAME, index=False)
    df.to_csv(report_root / SCAN_NAME, index=False)
    write_json(out_root / "distance_leaderboard.json", {"rows": rows})
    write_json(report_root / "distance_leaderboard.json", {"rows": rows[: min(40, len(rows))]})
    write_scan_report(args, rows)
    update_state(out_root / STATE_NAME, status="scan_complete", completed=len(rows), total=len(rows), best=rows[0] if rows else None, updated_at=now_iso())
    return rows


def train_one(args: argparse.Namespace, row: dict[str, Any], mode: str, seed: int = 0) -> dict[str, Any]:
    spec = row["spec"]
    spec_id = str(spec["id"])
    variant_dir = Path(row["variant_dir"])
    # Regenerate full-size variant before training if scan used sampled rows.
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if int(args.train_regenerate_full) == 1:
        if labels_path.exists():
            labels = pd.read_csv(labels_path)
            if len(labels) < 1000:
                # Re-run full generation into the same variant directory.
                make_variant(args, spec, 0, force_regenerate=True)
    run_id = spec_id if mode != "seed" else f"{spec_id}_seed{seed}"
    run_dir = Path(args.out_root) / "runs" / mode / run_id
    summary_file = run_dir / "medium_guard_bad_boundary_run_summary.json"
    if summary_file.exists() and not args.rerun_existing:
        existing = read_json(summary_file)
        if int(existing.get("returncode", 1)) == 0 or args.skip_failed_existing:
            existing["skipped_existing"] = True
            return existing
    log_dir = Path(args.out_root) / "logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    epochs1, epochs2 = (
        (int(args.quick_epochs_stage1), int(args.quick_epochs_stage2))
        if mode == "quick"
        else (int(args.full_epochs_stage1), int(args.full_epochs_stage2))
    )
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
        "--hidden_dim",
        str(getattr(args, "hidden_dim", 128)),
        "--denoiser_width",
        str(getattr(args, "denoiser_width", 1.0)),
        "--lambda_den_stage2",
        str(args.lambda_den_stage2),
        "--lambda_cls",
        str(args.lambda_cls),
        "--class_weight",
        str(spec.get("class_weight", args.class_weight)),
        "--seed",
        str(seed),
    ]
    start = time.perf_counter()
    with (log_dir / f"{run_id}.stdout.txt").open("w", encoding="utf-8") as out, (log_dir / f"{run_id}.stderr.txt").open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    payload: dict[str, Any] = {
        "spec": spec,
        "mode": mode,
        "seed": int(seed),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
        "distance": {k: row.get(k) for k in ["overall_but_like_score", "morph_overall_distance", "morph_medium_distance", "morph_bad_distance", "sqi_overall_distance", "domain_separability_bal_acc"]},
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(summary_file, payload)
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    write_training_report(args)
    return payload


def candidate_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    ev = row.get("but_10s_eval", {})
    reps = [
        ev.get("but_10s_test_report", {}),
        ev.get("but_10s_raw_test_report", {}),
        ev.get("but_10s_balanced_test_report", {}),
        ev.get("but_10s_balanced_raw_report", {}),
    ]
    rep = max([r for r in reps if r], key=lambda r: float(r.get("macro_f1", 0.0)), default={})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    ptb = row.get("ptb_test_report", {})
    return (
        1.0 if float(rep.get("acc", 0.0)) >= 0.825 else 0.0,
        min(float(rec[1]), float(rec[2])),
        float(rep.get("macro_f1", 0.0)),
        float(rep.get("acc", 0.0)),
        float(ptb.get("acc", 0.0)),
    )


def select_for_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_json(Path(args.out_root) / "distance_leaderboard.json")["rows"] if (Path(args.out_root) / "distance_leaderboard.json").exists() else score_specs(args)
    selected = rows[: int(args.top_quick)]
    controls = [
        r
        for r in rows
        if any(token in str(r.get("variant_id", "")) for token in ["bw_badstrong", "ma_heavy"])
    ][:4]
    for row in controls:
        if row not in selected:
            selected.append(row)
    return selected[: int(args.top_quick)]


def run_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="quick_training", stage="quick_train", updated_at=now_iso())
    rows: list[dict[str, Any]] = []
    selected = select_for_quick(args)
    for i, row in enumerate(selected, start=1):
        update_state(out_root / STATE_NAME, status="quick_training", current=row["variant_id"], completed=i - 1, total=len(selected), updated_at=now_iso())
        rows.append(train_one(args, row, "quick", seed=int(args.seed)))
    update_state(out_root / STATE_NAME, status="quick_complete", n_quick=len(rows), updated_at=now_iso())
    return rows


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="full_training", stage="full_train", updated_at=now_iso())
    rows_all = load_jsonl(out_root / SUMMARY_NAME)
    quick = [r for r in rows_all if r.get("mode") == "quick" and int(r.get("returncode", 1)) == 0]
    selected = sorted(quick, key=candidate_score, reverse=True)[: int(args.top_full)]
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        update_state(out_root / STATE_NAME, status="full_training", current=row["spec"]["id"], completed=i - 1, total=len(selected), updated_at=now_iso())
        row2 = {"variant_id": row["spec"]["id"], "spec": row["spec"], "variant_dir": row["variant_dir"], **row.get("distance", {})}
        rows.append(train_one(args, row2, "full", seed=int(args.seed)))
    update_state(out_root / STATE_NAME, status="full_complete", n_full=len(rows), updated_at=now_iso())
    return rows


def run_seed(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="seed_training", stage="seed_confirm", updated_at=now_iso())
    rows_all = load_jsonl(out_root / SUMMARY_NAME)
    full = [r for r in rows_all if r.get("mode") == "full" and int(r.get("returncode", 1)) == 0]
    selected = sorted(full, key=candidate_score, reverse=True)[: int(args.top_seed)]
    rows: list[dict[str, Any]] = []
    seeds = [int(v) for v in str(args.seeds).split(",") if str(v).strip()]
    for i, row in enumerate(selected, start=1):
        for seed in seeds:
            update_state(out_root / STATE_NAME, status="seed_training", current=row["spec"]["id"], seed=seed, completed=i - 1, total=len(selected), updated_at=now_iso())
            row2 = {"variant_id": row["spec"]["id"], "spec": row["spec"], "variant_dir": row["variant_dir"], **row.get("distance", {})}
            rows.append(train_one(args, row2, "seed", seed=seed))
    update_state(out_root / STATE_NAME, status="seed_complete", n_seed=len(rows), updated_at=now_iso())
    return rows


def write_scan_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    lines = [
        "# Medium-Guard Bad-Boundary Distance Scan",
        "",
        "Lower score means closer to BUT by morphology/SQI/domain features. Training selection still depends on validation-only BUT metrics.",
        "",
        "| rank | variant | score | medium dist | bad dist | SQI | domain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows[:40], start=1):
        lines.append(
            f"| {i} | `{row['variant_id']}` | {row['overall_but_like_score']:.4f} | "
            f"{row['morph_medium_distance']:.4f} | {row['morph_bad_distance']:.4f} | "
            f"{row['sqi_overall_distance']:.4f} | {row['domain_separability_bal_acc']:.4f} |"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "medium_guard_bad_boundary_scan_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_report(args: argparse.Namespace) -> None:
    report_root = Path(args.report_root)
    rows = load_jsonl(Path(args.out_root) / SUMMARY_NAME)
    ranked = sorted(rows, key=candidate_score, reverse=True)
    lines = [
        "# Medium-Guard Bad-Boundary Training Summary",
        "",
        f"Target: original BUT acc >= 0.825, medium >= 0.825, bad >= 0.80. Reference h_bad_rescue_05 macro-F1 {H_BAD_RESCUE_05['macro_f1']:.4f}, recalls {H_BAD_RESCUE_05['recall_good_medium_bad']}.",
        "",
        "| rank | mode | variant | return | orig acc | orig macro | recalls G/M/B | balanced macro | PTB acc | PTB bad |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        ev = row.get("but_10s_eval", {})
        orig = ev.get("but_10s_test_report", {})
        bal = ev.get("but_10s_balanced_test_report", {})
        rec = orig.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        lines.append(
            f"| {i} | {row.get('mode')} | `{row.get('spec', {}).get('id')}` | {row.get('returncode')} | "
            f"{float(orig.get('acc', 0.0)):.4f} | {float(orig.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} | "
            f"{float(bal.get('macro_f1', 0.0)):.4f} | {float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec) > 2 else 0.0):.4f} |"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "medium_guard_bad_boundary_training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "medium_guard_bad_boundary_training_summary.json", {"rows": ranked, "reference": H_BAD_RESCUE_05})


def run_all(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "quick_train", "full_train", "seed_confirm"}:
        wait_for_sensors(args)
    update_state(out_root / STATE_NAME, status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"all", "score_specs"}:
        score_specs(args)
    if args.stage in {"all", "quick_train"}:
        run_quick(args)
    if args.stage in {"all", "full_train"}:
        run_full(args)
    if args.stage in {"all", "seed_confirm"} and int(args.top_seed) > 0:
        run_seed(args)
    if args.stage == "report":
        rows = read_json(out_root / "distance_leaderboard.json")["rows"] if (out_root / "distance_leaderboard.json").exists() else []
        if rows:
            write_scan_report(args, rows)
        write_training_report(args)
    update_state(out_root / STATE_NAME, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s medium-guard/bad-boundary grid.")
    parser.add_argument("--stage", choices=("score_specs", "quick_train", "full_train", "seed_confirm", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--nstdb_root", default=str(DEFAULT_NSTDB_ROOT))
    parser.add_argument("--cinc_dir", default=str(DEFAULT_CINC_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_specs", type=int, default=36)
    parser.add_argument("--scan_sample_per_class", type=int, default=60)
    parser.add_argument("--but_feature_max_rows", type=int, default=1500)
    parser.add_argument("--sqi_rows_per_class", type=int, default=24)
    parser.add_argument("--top_quick", type=int, default=24)
    parser.add_argument("--top_full", type=int, default=6)
    parser.add_argument("--top_seed", type=int, default=3)
    parser.add_argument("--train_regenerate_full", type=int, default=1)
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--skip_failed_existing", action="store_true", default=True)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--denoiser_width", type=float, default=1.0)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--class_weight", default="1.00,1.55,1.70")
    parser.add_argument("--wait_for_sensors", action="store_true", default=True)
    parser.add_argument("--wait_poll_sec", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_all(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
