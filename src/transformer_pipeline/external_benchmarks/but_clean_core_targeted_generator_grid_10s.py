"""CleanBUT-Core 64D target-driven PTB synthetic generator scan.

This runner is experiment-only.  It builds PTB synthetic candidates that target
the CleanBUT bad tight core in the same 64D feature/PCA space used by
``but_clean_target_visual_compare_10s.py``.  Scan/refine stages are CPU-only and
prediction-free; they never use BUT test predictions or model scores for data
selection.  Training is opt-in through ``--run_training``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_clean_label_core_atlas_10s as clean_atlas
from src.transformer_pipeline.external_benchmarks import but_clean_target_visual_compare_10s as visual64
from src.transformer_pipeline.external_benchmarks import but_medium_guard_bad_boundary_grid_10s as mg
from src.transformer_pipeline.external_benchmarks import real_noise_snr_but_match_grid_10s as rn
from src.transformer_pipeline.external_benchmarks import sensors2025_noise_synthesis_but_10s as s25
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    ensure_but_10s,
    evaluate_checkpoint_10s as base_evaluate_checkpoint_10s,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.but_sample_or_subtype_grid_10s import plot_bad_subtype_gallery
from src.transformer_pipeline.external_benchmarks.run import (
    apply_but_thresholds,
    load_uformer_model,
    multiclass_report,
    run_model_outputs,
)


ROOT = rn.ROOT
RUN_TAG = "e311_but_clean_core_targeted_grid_10s_2026_06_06"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_CLEAN_ROOT = clean_atlas.DEFAULT_OUT_ROOT
DEFAULT_BUT_PROTOCOL = PROTOCOL_OUT_ROOT / "protocols" / "p1_current_10s_center"
DEFAULT_NSTDB_ROOT = ROOT / "data" / "physionet" / "nstdb"
DEFAULT_CINC_DIR = ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02" / "processed" / "cinc2017"

STATE_NAME = "clean_core_targeted_state.json"
SPEC_NAME = "clean64_target_grid_specs.json"
LEADERBOARD_NAME = "clean64_distance_leaderboard.csv"
DETAIL_NAME = "clean64_distance_detail.csv"
PROJECTION_NAME = "clean64_pca_projection.csv"
SUBTYPE_AUDIT_NAME = "bad_core_subtype_audit.csv"
SUMMARY_NAME = "clean_core_targeted_training_summary.jsonl"

CLASS_NAMES = ("good", "medium", "bad")
COLORS = {"good": "#1b9e77", "medium": "#fdae61", "bad": "#d73027"}
FS = 125
N_TARGET = 1250
BASELINE_64D_BAD_DISTANCE = 0.7480
BASELINE_64D_MEDIUM_DISTANCE = 0.3108


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stable_id(text: str) -> str:
    slug = (
        text.lower()
        .replace("+", "_")
        .replace("-", "m")
        .replace(".", "p")
        .replace(",", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    if len(slug) > 70:
        digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:12]
        slug = f"{slug[:57]}_{digest}"
    return slug


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    lines = ["| " + " | ".join(map(str, view.columns)) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in view.columns) + " |")
    return "\n".join(lines)


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
    profiles = mg.mix_profiles()
    profiles["bw_core"] = {
        "name": "em_ma_bw_core",
        "em_weight": 0.42,
        "ma_weight": 0.58,
        "bw_overlay": {"good": [0.002, 0.010], "medium": [0.014, 0.036], "bad": [0.055, 0.125]},
    }
    profiles["quiet_core"] = {
        "name": "em_ma_quiet_core",
        "em_weight": 0.35,
        "ma_weight": 0.65,
        "bw_overlay": {"good": [0.001, 0.008], "medium": [0.010, 0.030], "bad": [0.020, 0.060]},
    }
    return profiles


def base_spec(family: str, mix_key: str, intensity: str, secondary: float, class_weight: str, extra: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": "",
        "family": family,
        "snr_profile": paper_table_strict(),
        "mix": mix_profiles()[mix_key],
        "cinc_bad_fraction": 0.0,
        "class_weight": class_weight,
        "paper_constraints": {
            "target_snr_noise": "EM+MA only",
            "bw_participates_in_target_snr": False,
            "model_input_protocol": "project mainline 10s@125Hz, not paper 5s@200Hz",
        },
        "medium_event_prob": 0.76,
        "medium_pseudo_peak_rate": 0.05,
        "medium_inverse_peak_rate": 0.17,
        "medium_step_rate": 0.22,
        "medium_ramp_rate": 0.18,
        "medium_contact_rate": 0.01,
        "medium_qrs_preserve": 0.999,
        "secondary_prob": float(secondary),
        "bad_primary_family": family,
        "bad_intensity_name": intensity,
        "bad_core_gain": 0.42,
        "bad_core_smooth": 15,
        "bad_core_flat_prob": 0.25,
        "bad_core_flat_len": [36, 120],
        "bad_qrs_attenuation": 0.72,
        "bad_qrs_smooth": 21,
        "bad_pseudo_peak_rate": 0.10,
        "bad_baseline_step_rate": 0.22,
        "bad_clip_quantile": 0.78,
        "good_mode": "g_amp_qrs",
        "good_target_std": 0.306,
        "good_std_jitter": 0.055,
        "good_qrs_boost": 0.28,
        "good_qrs_sharp": 0.08,
        "good_nonqrs_smooth": 17,
        "good_nonqrs_blend": 0.80,
        "good_residual_mix": 0.015,
        "good_baseline_mix": 0.035,
        "good_qrsband_noise": 0.006,
        "bad_alt_family": "none",
        "bad_alt_cluster_prob": 0.0,
        "selection_basis": "CleanBUT-Core 64D class-wise distance and PCA bad-core geometry; no BUT test metric used.",
    }
    if intensity == "tight":
        spec.update({"bad_core_gain": 0.34, "bad_core_smooth": 21, "bad_qrs_attenuation": 0.80, "bad_pseudo_peak_rate": 0.06, "bad_baseline_step_rate": 0.18, "bad_clip_quantile": 0.70})
    elif intensity == "center":
        spec.update({"bad_core_gain": 0.46, "bad_core_smooth": 17, "bad_qrs_attenuation": 0.68, "bad_pseudo_peak_rate": 0.11, "bad_baseline_step_rate": 0.24, "bad_clip_quantile": 0.78})
    elif intensity == "wide":
        spec.update({"bad_core_gain": 0.56, "bad_core_smooth": 11, "bad_qrs_attenuation": 0.58, "bad_pseudo_peak_rate": 0.18, "bad_baseline_step_rate": 0.30, "bad_clip_quantile": 0.86})
    spec.update(extra)
    good_suffix = str(spec.get("good_mode", "gbase"))
    alt_suffix = "badalt_none"
    if str(spec.get("bad_alt_family", "none")) != "none" and float(spec.get("bad_alt_cluster_prob", 0.0)) > 0:
        alt_suffix = f"badalt_{spec.get('bad_alt_family')}_{float(spec.get('bad_alt_cluster_prob', 0.0)):.2f}"
    spec["id"] = stable_id(f"cc_{family}_{mix_key}_{intensity}_sec{secondary:.2f}_cw{class_weight}_{good_suffix}_{alt_suffix}")
    spec["note"] = (
        f"CleanBUT-Core targeted {family}; bad samples use one primary fatal subtype plus weak secondary "
        f"probability {secondary:.2f}; medium keeps visible QRS and detail/local unreliability."
    )
    return spec


def make_specs(max_specs: int, *, refined: bool = False, parents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if refined and parents:
        specs: list[dict[str, Any]] = []
        for parent in parents:
            for gain_mult in (0.88, 1.02):
                for qrs_mult in (0.95, 1.10):
                    for sec in (0.0, 0.18):
                        for good_mult in (0.96, 1.12):
                            spec = json.loads(json.dumps(parent))
                            spec["id"] = stable_id(f"{parent['id']}_ref_g{gain_mult:.2f}_q{qrs_mult:.2f}_s{sec:.2f}_gm{good_mult:.2f}")
                            spec["family"] = str(spec.get("family", "")) + "_refine"
                            spec["secondary_prob"] = float(sec)
                            spec["bad_core_gain"] = float(spec.get("bad_core_gain", 0.42)) * float(gain_mult)
                            spec["bad_qrs_attenuation"] = min(0.92, float(spec.get("bad_qrs_attenuation", 0.72)) * float(qrs_mult))
                            spec["bad_pseudo_peak_rate"] = max(0.0, float(spec.get("bad_pseudo_peak_rate", 0.10)) * (0.75 if sec == 0 else 1.0))
                            spec["bad_alt_cluster_prob"] = float(np.clip(float(spec.get("bad_alt_cluster_prob", 0.0)) + (0.06 if sec > 0 else -0.04), 0.0, 0.34))
                            spec["good_qrs_boost"] = float(spec.get("good_qrs_boost", 0.52)) * float(good_mult)
                            spec["good_qrs_sharp"] = float(spec.get("good_qrs_sharp", 0.10)) * float(good_mult)
                            spec["good_nonqrs_blend"] = float(np.clip(float(spec.get("good_nonqrs_blend", 0.90)) + (0.04 if good_mult > 1.0 else -0.02), 0.76, 0.98))
                            spec["note"] = str(spec.get("note", "")) + " Refined around CPU scan leader."
                            specs.append(spec)
                            if len(specs) >= int(max_specs):
                                return specs
        return specs

    families = {
        "bad_qrs_visible_compact_core": {
            "bad_core_gain": 0.18,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.03,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_visible_target_std": 0.154,
            "bad_visible_pulse_amp": 0.74,
            "bad_visible_qrsband_amp": 0.34,
            "bad_visible_hf_amp": 0.035,
            "bad_visible_width": 5.5,
            "bad_visible_side_lobe": 0.46,
            "bad_neg_spike_rate": 0.05,
            "bad_neg_spike_amp": 0.12,
            "bad_hf_qrs_bias": 0.82,
        },
        "bad_1530_locked_lowpc2_core": {
            "bad_core_gain": 0.12,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.04,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_compact_target_std": 0.154,
            "bad_compact_osc_amp": 0.72,
            "bad_compact_f1": 21.6,
            "bad_compact_f2": 17.8,
            "bad_compact_f3": 27.4,
            "bad_compact_w2": 0.72,
            "bad_compact_w3": 0.54,
            "bad_compact_qrs_boost": 0.72,
            "bad_neg_spike_rate": 0.08,
            "bad_neg_spike_amp": 0.16,
            "bad_hf_qrs_bias": 0.72,
        },
        "bad_compact_pca_core": {
            "bad_core_gain": 0.34,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.08,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_compact_target_std": 0.154,
            "bad_compact_osc_amp": 0.66,
            "bad_compact_f1": 21.8,
            "bad_compact_f2": 16.6,
            "bad_compact_f3": 41.0,
            "bad_compact_w2": 0.26,
            "bad_compact_w3": 0.78,
            "bad_compact_qrs_boost": 0.42,
            "bad_neg_spike_rate": 0.32,
            "bad_neg_spike_amp": 0.34,
            "bad_hf_qrs_bias": 0.62,
        },
        "bad_1530_hfedge_spike_core": {
            "bad_core_gain": 0.16,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.14,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_osc_amp": 0.56,
            "bad_osc_f1": 16.9,
            "bad_osc_f2": 22.0,
            "bad_osc_f3": 38.0,
            "bad_osc_w2": 0.28,
            "bad_osc_w3": 0.44,
            "bad_osc_qrs_boost": 0.22,
            "bad_neg_spike_rate": 0.58,
            "bad_neg_spike_amp": 0.48,
            "bad_hf_qrs_bias": 0.38,
        },
        "bad_1530_spike_core": {
            "bad_core_gain": 0.18,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.10,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_osc_amp": 0.62,
            "bad_osc_f1": 16.8,
            "bad_osc_f2": 22.5,
            "bad_osc_f3": 36.0,
            "bad_osc_w2": 0.36,
            "bad_osc_w3": 0.16,
            "bad_osc_qrs_boost": 0.28,
            "bad_neg_spike_rate": 0.34,
            "bad_neg_spike_amp": 0.34,
            "bad_hf_qrs_bias": 0.48,
        },
        "bad_narrow_oscillatory_core": {
            "bad_core_gain": 0.22,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.12,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_osc_amp": 0.58,
            "bad_osc_f1": 16.4,
            "bad_osc_f2": 24.8,
            "bad_osc_f3": 33.0,
            "bad_osc_qrs_boost": 0.22,
            "bad_neg_spike_rate": 0.06,
            "bad_hf_qrs_bias": 0.44,
        },
        "bad_qrsband_disagree_core": {
            "bad_core_gain": 0.92,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.18,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_qrsband_amp": 0.72,
            "bad_hf_amp": 0.18,
            "bad_qrs_boost": 0.36,
            "bad_neg_spike_rate": 0.18,
            "bad_hf_qrs_bias": 0.54,
        },
        "bad_bandlimited_disagree_core": {
            "bad_core_gain": 0.62,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.20,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.15,
            "bad_hf_amp": 1.10,
            "bad_hf_lo": 17.0,
            "bad_hf_hi": 32.0,
            "bad_neg_spike_rate": 0.22,
            "bad_hf_qrs_bias": 0.35,
        },
        "bad_hf_detector_core": {
            "bad_core_gain": 0.94,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.42,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.05,
            "bad_hf_burst_rate": 0.72,
            "bad_hf_amp": 0.72,
            "bad_hf_qrs_bias": 0.72,
        },
        "bad_hf_qrs_core": {
            "bad_core_gain": 1.02,
            "bad_core_flat_prob": 0.00,
            "bad_pseudo_peak_rate": 0.28,
            "bad_baseline_step_rate": 0.00,
            "bad_qrs_attenuation": 0.00,
            "bad_hf_burst_rate": 0.52,
            "bad_hf_amp": 0.56,
            "bad_hf_qrs_bias": 0.88,
        },
        "bad_quiet_dead_core": {
            "bad_core_gain": 0.28,
            "bad_core_flat_prob": 0.55,
            "bad_pseudo_peak_rate": 0.02,
            "bad_baseline_step_rate": 0.10,
            "bad_qrs_attenuation": 0.82,
        },
        "bad_qrs_erased_core": {
            "bad_core_gain": 0.48,
            "bad_core_flat_prob": 0.18,
            "bad_pseudo_peak_rate": 0.04,
            "bad_qrs_attenuation": 0.90,
            "bad_qrs_smooth": 31,
        },
        "bad_detector_disagree_core": {
            "bad_core_gain": 0.62,
            "bad_core_flat_prob": 0.08,
            "bad_pseudo_peak_rate": 0.28,
            "bad_baseline_step_rate": 0.12,
            "bad_qrs_attenuation": 0.55,
        },
        "bad_bw_platform_core": {
            "bad_core_gain": 0.52,
            "bad_core_flat_prob": 0.18,
            "bad_pseudo_peak_rate": 0.05,
            "bad_baseline_step_rate": 0.48,
            "bad_qrs_attenuation": 0.74,
        },
        "bad_clipped_lowamp_core": {
            "bad_core_gain": 0.34,
            "bad_core_flat_prob": 0.42,
            "bad_pseudo_peak_rate": 0.03,
            "bad_baseline_step_rate": 0.18,
            "bad_clip_quantile": 0.55,
            "bad_qrs_attenuation": 0.76,
        },
        "hybrid_real_noise_core": {
            "bad_core_gain": 0.70,
            "bad_core_flat_prob": 0.04,
            "bad_pseudo_peak_rate": 0.00,
            "bad_baseline_step_rate": 0.20,
            "bad_qrs_attenuation": 0.66,
        },
    }
    mix_keys = ("quiet_core", "bw_core", "bw_badstrong", "bw_mid")
    intensities = ("tight", "center", "wide")
    secondary_probs = (0.0, 0.12, 0.24)
    class_weights = ("1.00,1.45,1.70", "1.00,1.55,1.70", "1.00,1.60,1.75")
    medium_modes = (
        {"medium_event_prob": 0.70, "medium_contact_rate": 0.00, "medium_step_rate": 0.18},
        {"medium_event_prob": 0.78, "medium_contact_rate": 0.01, "medium_step_rate": 0.24},
        {"medium_event_prob": 0.84, "medium_contact_rate": 0.015, "medium_step_rate": 0.28},
    )
    target_modes = (
        {
            "good_mode": "g_qrs_prominent_noalt",
            "good_target_std": 0.314,
            "good_qrs_boost": 0.92,
            "good_qrs_sharp": 0.22,
            "good_nonqrs_smooth": 27,
            "good_nonqrs_blend": 0.94,
            "good_residual_mix": 0.000,
            "good_baseline_mix": 0.018,
            "good_qrsband_noise": 0.000,
            "bad_alt_family": "none",
            "bad_alt_cluster_prob": 0.00,
        },
        {
            "good_mode": "g_peak_locked",
            "good_target_std": 0.306,
            "good_qrs_boost": 0.62,
            "good_qrs_sharp": 0.14,
            "good_nonqrs_smooth": 25,
            "good_nonqrs_blend": 0.92,
            "good_residual_mix": 0.000,
            "good_baseline_mix": 0.014,
            "good_qrsband_noise": 0.000,
            "bad_alt_family": "bad_1530_spike_core",
            "bad_alt_cluster_prob": 0.16,
        },
        {
            "good_mode": "g_sparse_highqrs",
            "good_target_std": 0.318,
            "good_qrs_boost": 0.78,
            "good_qrs_sharp": 0.18,
            "good_nonqrs_smooth": 31,
            "good_nonqrs_blend": 0.96,
            "good_residual_mix": 0.000,
            "good_baseline_mix": 0.006,
            "good_qrsband_noise": 0.000,
            "bad_alt_family": "bad_narrow_oscillatory_core",
            "bad_alt_cluster_prob": 0.22,
        },
        {
            "good_mode": "g_but_texture",
            "good_target_std": 0.300,
            "good_qrs_boost": 0.52,
            "good_qrs_sharp": 0.10,
            "good_nonqrs_smooth": 21,
            "good_nonqrs_blend": 0.86,
            "good_residual_mix": 0.004,
            "good_baseline_mix": 0.045,
            "good_qrsband_noise": 0.001,
            "bad_alt_family": "bad_bandlimited_disagree_core",
            "bad_alt_cluster_prob": 0.20,
        },
        {
            "good_mode": "g_qrs_prominent",
            "good_target_std": 0.314,
            "good_qrs_boost": 0.92,
            "good_qrs_sharp": 0.22,
            "good_nonqrs_smooth": 27,
            "good_nonqrs_blend": 0.94,
            "good_residual_mix": 0.000,
            "good_baseline_mix": 0.018,
            "good_qrsband_noise": 0.000,
            "bad_alt_family": "bad_1530_hfedge_spike_core",
            "bad_alt_cluster_prob": 0.18,
        },
    )

    specs = []
    for mix_key in mix_keys:
        for intensity in intensities:
            for sec in secondary_probs:
                for mi, med in enumerate(medium_modes):
                    for family, extra in families.items():
                        for tm in target_modes:
                            cw = class_weights[(len(specs) + mi) % len(class_weights)]
                            spec_extra = {**extra, **med, **tm, "medium_mode": f"m{mi}"}
                            specs.append(base_spec(family, mix_key, intensity, sec, cw, spec_extra))
                            if len(specs) >= int(max_specs):
                                return specs
    return specs


def _smooth(y: np.ndarray, width: int) -> np.ndarray:
    width = int(max(3, width))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(y.astype(np.float32), kernel, mode="same").astype(np.float32)


def _robust_scale(y: np.ndarray) -> float:
    return float(np.quantile(y, 0.90) - np.quantile(y, 0.10) + 1e-6)


def _insert_flat_segments(x: np.ndarray, rng: np.random.Generator, prob: float, length_range: list[int] | tuple[int, int], scale: float) -> int:
    if rng.random() >= float(prob):
        return 0
    n_seg = int(rng.integers(1, 3))
    count = 0
    lo, hi = int(length_range[0]), int(length_range[1])
    for _ in range(n_seg):
        length = int(rng.integers(max(8, lo), max(lo + 1, hi)))
        start = int(rng.integers(0, max(1, len(x) - length)))
        level = float(np.median(x[max(0, start - 20) : min(len(x), start + length + 20)]))
        x[start : start + length] = level + rng.normal(0.0, 0.002 * scale, size=length).astype(np.float32)
        count += 1
    return count


def _attenuate_qrs(x: np.ndarray, clean: np.ndarray, qrs: np.ndarray, attenuation: float, smooth_width: int) -> None:
    mask = np.clip(qrs.astype(np.float32), 0.0, 1.0)
    if float(mask.max()) <= 0:
        return
    base = _smooth(x, smooth_width)
    clean_base = _smooth(clean, max(9, smooth_width // 2))
    replacement = 0.70 * base + 0.30 * clean_base
    a = float(np.clip(attenuation, 0.0, 0.98))
    x[:] = x * (1.0 - a * mask) + replacement * (a * mask)


def _add_controlled_pseudo_peaks(x: np.ndarray, qrs: np.ndarray, rng: np.random.Generator, rate: float, scale: float, invert: bool = False) -> int:
    count = max(0, int(round(float(rate) * 10.0)))
    if count == 0:
        return 0
    qpos = np.where(qrs.reshape(-1) > 0.5)[0]
    amp = _robust_scale(x) * float(scale)
    placed = 0
    for _ in range(count):
        if len(qpos) and rng.random() < 0.70:
            center = int(rng.choice(qpos) + rng.integers(-55, 56))
        else:
            center = int(rng.integers(40, len(x) - 40))
        center = int(np.clip(center, 12, len(x) - 13))
        width = int(rng.integers(4, 10))
        t = np.arange(-width, width + 1, dtype=np.float32)
        pulse = np.exp(-0.5 * (t / max(1.5, width / 2.6)) ** 2)
        sign = -1.0 if invert or rng.random() < 0.40 else 1.0
        x[center - width : center + width + 1] += sign * amp * float(rng.uniform(0.25, 0.65)) * pulse
        placed += 1
    return placed


def _add_hf_detector_disagree_bursts(
    x: np.ndarray,
    clean: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    *,
    burst_rate: float,
    amp_scale: float,
    qrs_bias: float,
) -> int:
    """Add narrow high-frequency QRS-like bursts that split peak detectors."""

    qpos = np.where(qrs.reshape(-1) > 0.5)[0]
    n_bursts = max(2, int(round(float(burst_rate) * 18.0)))
    amp = _robust_scale(clean) * float(amp_scale)
    placed = 0
    for j in range(n_bursts):
        if len(qpos) and rng.random() < float(qrs_bias):
            center = int(rng.choice(qpos) + rng.integers(-28, 29))
        else:
            center = int(rng.integers(24, len(x) - 24))
        center = int(np.clip(center, 14, len(x) - 15))
        width = int(rng.integers(5, 12))
        samples = np.arange(-width, width + 1, dtype=np.float32)
        t = samples / FS
        freq = float(rng.uniform(18.0, 30.0))
        envelope = np.exp(-0.5 * (samples / max(2.0, width / 2.8)) ** 2)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        burst = np.sin(2.0 * np.pi * freq * t + phase) * envelope
        spike_width = max(1, width // 3)
        spike = np.exp(-0.5 * (samples / spike_width) ** 2)
        sign = -1.0 if (j % 2 == 0 or rng.random() < 0.55) else 1.0
        segment = amp * float(rng.uniform(0.45, 0.95)) * (0.65 * burst + sign * 0.55 * spike)
        x[center - width : center + width + 1] += segment.astype(np.float32)
        placed += 1
    return placed


def _band_limited_noise(n: int, rng: np.random.Generator, lo_hz: float, hi_hz: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(int(n), d=1.0 / FS)
    spec = np.zeros(len(freqs), dtype=np.complex64)
    mask = (freqs >= float(lo_hz)) & (freqs <= float(hi_hz))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(mask.sum()))
    mags = rng.uniform(0.75, 1.25, size=int(mask.sum()))
    spec[mask] = (mags * np.exp(1j * phases)).astype(np.complex64)
    y = np.fft.irfft(spec, n=int(n)).astype(np.float32)
    y = y - float(np.mean(y))
    y = y / float(np.std(y) + 1e-6)
    return y


def _add_negative_disagreement_spikes(
    x: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    *,
    rate: float,
    amp_scale: float,
    qrs_bias: float,
) -> int:
    qpos = np.where(qrs.reshape(-1) > 0.5)[0]
    n_spikes = max(1, int(round(float(rate) * 26.0)))
    amp = _robust_scale(x) * float(amp_scale)
    placed = 0
    for _ in range(n_spikes):
        if len(qpos) and rng.random() < float(qrs_bias):
            center = int(rng.choice(qpos) + rng.integers(-24, 25))
        else:
            center = int(rng.integers(8, len(x) - 8))
        center = int(np.clip(center, 5, len(x) - 6))
        width = int(rng.integers(1, 4))
        samples = np.arange(-width, width + 1, dtype=np.float32)
        pulse = np.exp(-0.5 * (samples / max(0.75, width / 1.7)) ** 2)
        sign = -1.0 if rng.random() < 0.78 else 1.0
        x[center - width : center + width + 1] += sign * amp * float(rng.uniform(0.10, 0.22)) * pulse
        placed += 1
    return placed


def _boost_qrs_band_core(
    x: np.ndarray,
    clean: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    *,
    qrsband_amp: float,
    hf_amp: float,
    qrs_boost: float,
    scale: float,
) -> None:
    """Pull bad windows toward CleanBUT bad: strong 5-18Hz, controlled HF."""

    centered_clean = clean - float(np.median(clean))
    qrs_mask = np.clip(qrs.reshape(-1).astype(np.float32), 0.0, 1.0)
    if float(qrs_mask.max()) > 0:
        qrs_mask = _smooth(qrs_mask, 9)
        qrs_mask = np.clip(qrs_mask / float(qrs_mask.max() + 1e-6), 0.0, 1.0)

    qrs_band = _band_limited_noise(len(x), rng, 6.0, 16.5)
    hf = _band_limited_noise(len(x), rng, 20.0, 29.0)
    robust = _robust_scale(clean) * float(scale)
    # Keep total spectrum concentrated in the QRS band.  A small HF component
    # gives the bad-core wavelet signature without throwing the point above
    # the medium cloud.
    x[:] = (
        float(np.median(clean))
        + centered_clean * (1.0 + float(qrs_boost) * qrs_mask)
        + qrs_band * robust * float(qrsband_amp)
        + hf * robust * float(hf_amp)
    ).astype(np.float32)


def _make_narrow_oscillatory_core(
    x: np.ndarray,
    clean: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    *,
    amp_scale: float,
    f1: float,
    f2: float,
    f3: float,
    w2: float,
    w3: float,
    qrs_boost: float,
    clean_gain: float,
    scale: float,
) -> None:
    """Create the tight bad-core spectral signature: 15-30Hz dominated."""

    n = len(x)
    t = np.arange(n, dtype=np.float32) / FS
    qrs_mask = np.clip(qrs.reshape(-1).astype(np.float32), 0.0, 1.0)
    if float(qrs_mask.max()) > 0:
        qrs_mask = _smooth(qrs_mask, 9)
        qrs_mask = np.clip(qrs_mask / float(qrs_mask.max() + 1e-6), 0.0, 1.0)
    phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase3 = float(rng.uniform(0.0, 2.0 * np.pi))
    f1 = float(f1) + float(rng.uniform(-0.45, 0.45))
    f2 = float(f2) + float(rng.uniform(-0.75, 0.75))
    f3 = float(f3) + float(rng.uniform(-1.2, 1.2))
    envelope = 0.82 + 0.18 * np.sin(2.0 * np.pi * float(rng.uniform(0.18, 0.42)) * t + phase2)
    carrier = (
        1.00 * np.sin(2.0 * np.pi * f1 * t + phase1)
        + float(w2) * np.sin(2.0 * np.pi * f2 * t + phase2)
        + float(w3) * np.sin(2.0 * np.pi * f3 * t + phase3)
    ) * envelope
    carrier = carrier - float(np.median(carrier))
    carrier = carrier / float(np.std(carrier) + 1e-6)
    centered_clean = clean - float(np.median(clean))
    robust = _robust_scale(clean) * float(amp_scale) * float(scale)
    x[:] = (
        float(np.median(clean))
        + centered_clean * float(clean_gain)
        + carrier.astype(np.float32) * robust * (1.0 + float(qrs_boost) * qrs_mask)
    ).astype(np.float32)


def _make_compact_pca_bad_core(
    x: np.ndarray,
    clean: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    spec: dict[str, Any],
    *,
    scale: float,
) -> None:
    """Target CleanBUT bad's tight right-lower PCA core, not a broad noise cloud."""

    n = len(x)
    t = np.arange(n, dtype=np.float32) / FS
    qrs_mask = np.clip(qrs.reshape(-1).astype(np.float32), 0.0, 1.0)
    if float(qrs_mask.max()) > 0:
        qrs_mask = _smooth(qrs_mask, 7)
        qrs_mask = np.clip(qrs_mask / float(qrs_mask.max() + 1e-6), 0.0, 1.0)
    clean_centered = clean - float(np.median(clean))
    clean_gain = float(spec.get("bad_core_gain", 0.34))
    qrs_boost = float(spec.get("bad_compact_qrs_boost", 0.30))
    f1 = float(spec.get("bad_compact_f1", 21.8)) + float(rng.uniform(-0.12, 0.12))
    f2 = float(spec.get("bad_compact_f2", 16.6)) + float(rng.uniform(-0.10, 0.10))
    f3 = float(spec.get("bad_compact_f3", 37.5)) + float(rng.uniform(-0.18, 0.18))
    w2 = float(spec.get("bad_compact_w2", 0.26))
    w3 = float(spec.get("bad_compact_w3", 0.24))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    carrier = (
        np.sin(2.0 * np.pi * f1 * t + phase)
        + w2 * np.sin(2.0 * np.pi * f2 * t + phase * 0.71)
        + w3 * np.sin(2.0 * np.pi * f3 * t + phase * 1.37)
    ).astype(np.float32)
    carrier = carrier / float(np.std(carrier) + 1e-6)
    robust = _robust_scale(clean) * float(spec.get("bad_compact_osc_amp", 0.74)) * float(scale)
    y = clean_centered * clean_gain * (1.0 + qrs_boost * qrs_mask) + carrier * robust * (0.88 + 0.12 * qrs_mask)
    target_std = float(spec.get("bad_compact_target_std", 0.154)) * float(rng.uniform(0.97, 1.03))
    y = y - float(np.median(y))
    y = y * (target_std / float(np.std(y) + 1e-6))
    x[:] = (float(np.median(clean)) + y).astype(np.float32)


def _qrs_segment_centers(qrs: np.ndarray) -> np.ndarray:
    mask = np.asarray(qrs).reshape(-1) > 0.5
    if not bool(mask.any()):
        return np.array([], dtype=int)
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    centers = ((starts + ends) / 2.0).astype(int)
    return centers[(centers >= 8) & (centers < len(mask) - 8)]


def _make_qrs_visible_compact_bad_core(
    x: np.ndarray,
    clean: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    spec: dict[str, Any],
    *,
    scale: float,
) -> None:
    """Bad core with CleanBUT-like low LF drift and concentrated QRS-band energy."""

    n = len(x)
    t = np.arange(n, dtype=np.float32) / FS
    qrs_mask = np.clip(qrs.reshape(-1).astype(np.float32), 0.0, 1.0)
    if float(qrs_mask.max()) > 0:
        qrs_mask = _smooth(qrs_mask, 7)
        qrs_mask = np.clip(qrs_mask / float(qrs_mask.max() + 1e-6), 0.0, 1.0)

    centers = _qrs_segment_centers(qrs)
    pulses = np.zeros(n, dtype=np.float32)
    pulse_amp = float(spec.get("bad_visible_pulse_amp", 0.72))
    width = float(spec.get("bad_visible_width", 5.5))
    side_lobe = float(spec.get("bad_visible_side_lobe", 0.44))
    for c in centers:
        lo = max(0, int(c - 22))
        hi = min(n, int(c + 23))
        samples = np.arange(lo - c, hi - c, dtype=np.float32)
        main = np.exp(-0.5 * (samples / width) ** 2)
        left = np.exp(-0.5 * ((samples + 8.0) / max(2.0, width * 0.52)) ** 2)
        right = np.exp(-0.5 * ((samples - 7.0) / max(2.0, width * 0.48)) ** 2)
        pulses[lo:hi] += (main - side_lobe * left + 0.34 * side_lobe * right).astype(np.float32) * pulse_amp
    if float(np.std(pulses)) > 1e-6:
        pulses = pulses / float(np.std(pulses) + 1e-6)

    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    qrs_band = (
        np.sin(2.0 * np.pi * 8.8 * t + phase)
        + 0.55 * np.sin(2.0 * np.pi * 13.4 * t + phase * 0.63)
        + 0.22 * np.sin(2.0 * np.pi * 16.2 * t + phase * 1.31)
    ).astype(np.float32)
    qrs_band = qrs_band / float(np.std(qrs_band) + 1e-6)
    hf = _band_limited_noise(n, rng, 20.0, 29.0)

    clean_centered = clean - float(np.median(clean))
    clean_gain = float(spec.get("bad_core_gain", 0.18))
    robust = _robust_scale(clean) * float(scale)
    y = (
        clean_centered * clean_gain
        + pulses * robust * 0.42
        + qrs_band * robust * float(spec.get("bad_visible_qrsband_amp", 0.34)) * (0.74 + 0.26 * qrs_mask)
        + hf * robust * float(spec.get("bad_visible_hf_amp", 0.035))
    ).astype(np.float32)

    # Remove the slow component because PC2 is dominated by LF/baseline terms;
    # CleanBUT bad's tight core sits low on PC2 with little baseline wander.
    y = y - _smooth(y, 151) * 0.88
    target_std = float(spec.get("bad_visible_target_std", 0.154)) * float(rng.uniform(0.985, 1.015))
    y = y - float(np.median(y))
    y = y * (target_std / float(np.std(y) + 1e-6))
    x[:] = (float(np.median(clean)) + y).astype(np.float32)


def _apply_good_core(
    x: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    qrs: np.ndarray,
    rng: np.random.Generator,
    spec: dict[str, Any],
) -> dict[str, float]:
    """Move synthetic good toward CleanBUT good: high-amplitude reliable QRS."""

    target_std = float(spec.get("good_target_std", 0.306))
    jitter = float(spec.get("good_std_jitter", 0.055))
    target_std *= float(rng.uniform(max(0.85, 1.0 - jitter), min(1.18, 1.0 + jitter)))
    qrs_boost = float(spec.get("good_qrs_boost", 0.28))
    qrs_sharp = float(spec.get("good_qrs_sharp", 0.08))
    nonqrs_smooth = int(spec.get("good_nonqrs_smooth", 17))
    nonqrs_blend_strength = float(spec.get("good_nonqrs_blend", 0.80))
    residual_mix = float(spec.get("good_residual_mix", 0.012))
    baseline_mix = float(spec.get("good_baseline_mix", 0.035))
    qrsband_noise = float(spec.get("good_qrsband_noise", 0.004))

    centered = clean - float(np.median(clean))
    qrs_mask = np.clip(qrs.reshape(-1).astype(np.float32), 0.0, 1.0)
    if float(qrs_mask.max()) > 0:
        qrs_mask = _smooth(qrs_mask, 11)
        qrs_mask = np.clip(qrs_mask / float(qrs_mask.max() + 1e-6), 0.0, 1.0)
    reliable = centered * (1.0 + qrs_boost * qrs_mask)
    if float(qrs_mask.max()) > 0 and qrs_sharp > 0:
        qrs_detail = centered - _smooth(centered, 31)
        detail_scale = target_std / float(np.std(qrs_detail * qrs_mask) + 1e-6)
        reliable = reliable + qrs_detail * qrs_mask * detail_scale * qrs_sharp
    std0 = float(np.std(reliable) + 1e-6)
    y = reliable * (target_std / std0)

    residual = noisy - clean
    slow = _smooth(residual, 95)
    fast = residual - _smooth(residual, 19)
    texture = _band_limited_noise(len(x), rng, 5.0, 13.5) * target_std * qrsband_noise
    y = y + slow * baseline_mix + fast * residual_mix + texture
    if float(qrs_mask.max()) > 0:
        smooth_y = _smooth(y, max(5, nonqrs_smooth | 1))
        non_qrs_blend = np.clip(1.0 - 1.10 * qrs_mask, 0.0, 1.0)
        blend = np.clip(nonqrs_blend_strength * non_qrs_blend, 0.0, 0.98)
        y = y * (1.0 - blend) + smooth_y * blend
    # Re-scale after adding texture so the class lands on CleanBUT good's
    # amplitude band instead of drifting back toward medium.
    y = y - float(np.median(y))
    y = y * (target_std / float(np.std(y) + 1e-6))
    x[:] = (float(np.median(clean)) + y).astype(np.float32)
    return {
        "target_std": target_std,
        "qrs_boost": qrs_boost,
        "qrs_sharp": qrs_sharp,
        "nonqrs_smooth": nonqrs_smooth,
        "nonqrs_blend": nonqrs_blend_strength,
        "residual_mix": residual_mix,
        "baseline_mix": baseline_mix,
        "qrsband_noise": qrsband_noise,
    }


def _add_baseline_platforms(x: np.ndarray, rng: np.random.Generator, rate: float, scale: float) -> int:
    n_seg = max(0, int(round(float(rate) * 4.0)))
    if n_seg == 0:
        return 0
    amp = _robust_scale(x) * 0.18 * float(scale)
    for _ in range(n_seg):
        length = int(rng.integers(80, 260))
        start = int(rng.integers(0, max(1, len(x) - length)))
        step = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 1.1) * amp)
        ramp = np.linspace(0.0, step, length, dtype=np.float32)
        if rng.random() < 0.45:
            x[start : start + length] += step
        else:
            x[start : start + length] += ramp
    return n_seg


def _clip_lowamp(x: np.ndarray, q: float, gain: float) -> None:
    med = float(np.median(x))
    y = (x - med) * float(gain)
    lim = float(np.quantile(np.abs(y), np.clip(q, 0.35, 0.95)) + 1e-5)
    x[:] = np.clip(y, -lim, lim) + med


def apply_medium_and_bad_core(spec: dict[str, Any], variant_dir: Path, seed: int) -> dict[str, Any]:
    data_dir = variant_dir / "datasets"
    clean = np.load(data_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    noisy = np.load(data_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    labels = pd.read_csv(data_dir / "synth_10s_125hz_labels_with_level.csv").reset_index(drop=True)
    masks = np.load(data_dir / "synth_10s_125hz_local_mask.npz")
    qrs = masks["qrs_mask"].astype(np.float32) if "qrs_mask" in masks else np.zeros_like(clean)
    tst = masks["tst_mask"].astype(np.float32) if "tst_mask" in masks else np.zeros_like(clean)
    critical = masks["critical_mask"].astype(np.float32) if "critical_mask" in masks else qrs
    rng = np.random.default_rng(seed + 6406)
    x = noisy.astype(np.float32, copy=True)
    y = labels["y_class"].map({"good": 0, "medium": 1, "bad": 2}).to_numpy(dtype=int)
    good_idx = np.where(y == 0)[0]
    medium_idx = np.where(y == 1)[0]
    bad_idx = np.where(y == 2)[0]

    good_rows: list[dict[str, Any]] = []
    for idx in good_idx:
        i = int(idx)
        before = x[i].copy()
        info = _apply_good_core(x[i], clean[i], noisy[i], qrs[i], rng, spec)
        good_rows.append(
            {
                "idx": i,
                "y_class": "good",
                "good_mode": str(spec.get("good_mode", "g_amp_qrs")),
                "rms_before": float(np.sqrt(np.mean(before * before))),
                "rms_after": float(np.sqrt(np.mean(x[i] * x[i]))),
                **info,
            }
        )

    mg.add_local_events(x, clean, medium_idx, qrs, spec, rng, "medium")

    primary_name = str(spec.get("bad_primary_family", spec.get("family", "bad_quiet_dead_core")))
    alt_family = str(spec.get("bad_alt_family", "none"))
    alt_prob = float(spec.get("bad_alt_cluster_prob", 0.0))
    secondary_prob = float(spec.get("secondary_prob", 0.0))
    subtype_rows: list[dict[str, Any]] = []
    for idx in bad_idx:
        i = int(idx)
        fam = primary_name
        if fam.endswith("_refine"):
            fam = fam.replace("_refine", "")
        if alt_family != "none" and alt_prob > 0.0 and rng.random() < alt_prob:
            fam = alt_family
            if fam.endswith("_refine"):
                fam = fam.replace("_refine", "")
        if fam in {"bad_compact_pca_core", "bad_1530_locked_lowpc2_core", "bad_qrs_visible_compact_core"}:
            scale = float(rng.uniform(0.98, 1.02))
        else:
            scale = float(rng.uniform(0.82, 1.12))
        n_flat = 0
        n_pseudo = 0
        n_platform = 0
        before = x[i].copy()

        if fam == "bad_qrs_visible_compact_core":
            _make_qrs_visible_compact_bad_core(x[i], clean[i], qrs[i], rng, spec, scale=scale)
            n_pseudo += _add_negative_disagreement_spikes(
                x[i],
                qrs[i],
                rng,
                rate=float(spec.get("bad_neg_spike_rate", 0.05)),
                amp_scale=float(spec.get("bad_neg_spike_amp", 0.12)) * scale,
                qrs_bias=float(spec.get("bad_hf_qrs_bias", 0.82)),
            )
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.03)), 0.08 * scale, invert=True)
        elif fam in {"bad_compact_pca_core", "bad_1530_locked_lowpc2_core"}:
            _make_compact_pca_bad_core(x[i], clean[i], qrs[i], rng, spec, scale=scale)
            n_pseudo += _add_negative_disagreement_spikes(
                x[i],
                qrs[i],
                rng,
                rate=float(spec.get("bad_neg_spike_rate", 0.18)),
                amp_scale=float(spec.get("bad_neg_spike_amp", 0.26)) * scale,
                qrs_bias=float(spec.get("bad_hf_qrs_bias", 0.62)),
            )
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.08)), 0.18 * scale, invert=True)
        elif fam in {"bad_narrow_oscillatory_core", "bad_1530_spike_core"}:
            _make_narrow_oscillatory_core(
                x[i],
                clean[i],
                qrs[i],
                rng,
                amp_scale=float(spec.get("bad_osc_amp", 0.58)),
                f1=float(spec.get("bad_osc_f1", 16.4)),
                f2=float(spec.get("bad_osc_f2", 24.8)),
                f3=float(spec.get("bad_osc_f3", 33.0)),
                w2=float(spec.get("bad_osc_w2", 0.72)),
                w3=float(spec.get("bad_osc_w3", 0.22)),
                qrs_boost=float(spec.get("bad_osc_qrs_boost", 0.22)),
                clean_gain=float(spec.get("bad_core_gain", 0.22)),
                scale=scale,
            )
            n_pseudo += _add_negative_disagreement_spikes(
                x[i],
                qrs[i],
                rng,
                rate=float(spec.get("bad_neg_spike_rate", 0.06)),
                amp_scale=float(spec.get("bad_neg_spike_amp", 0.20)) * scale,
                qrs_bias=float(spec.get("bad_hf_qrs_bias", 0.44)),
            )
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.12)), 0.22 * scale, invert=True)
        elif fam == "bad_quiet_dead_core":
            centered = x[i] - np.median(x[i])
            x[i] = np.median(x[i]) + _smooth(centered, int(spec.get("bad_core_smooth", 21))) * float(spec.get("bad_core_gain", 0.34)) * scale
            _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.80)), int(spec.get("bad_qrs_smooth", 21)))
            n_flat += _insert_flat_segments(x[i], rng, float(spec.get("bad_core_flat_prob", 0.5)), spec.get("bad_core_flat_len", [36, 120]), scale)
        elif fam == "bad_qrs_erased_core":
            _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.90)), int(spec.get("bad_qrs_smooth", 31)))
            x[i] = np.median(x[i]) + (x[i] - np.median(x[i])) * float(spec.get("bad_core_gain", 0.48)) * scale
            n_flat += _insert_flat_segments(x[i], rng, float(spec.get("bad_core_flat_prob", 0.18)), spec.get("bad_core_flat_len", [30, 90]), scale)
        elif fam == "bad_detector_disagree_core":
            _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.55)), int(spec.get("bad_qrs_smooth", 17)))
            x[i] = np.median(x[i]) + (x[i] - np.median(x[i])) * float(spec.get("bad_core_gain", 0.62)) * scale
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.28)), scale, invert=False)
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.28)) * 0.55, scale, invert=True)
        elif fam == "bad_qrsband_disagree_core":
            _boost_qrs_band_core(
                x[i],
                clean[i],
                qrs[i],
                rng,
                qrsband_amp=float(spec.get("bad_qrsband_amp", 0.72)),
                hf_amp=float(spec.get("bad_hf_amp", 0.18)),
                qrs_boost=float(spec.get("bad_qrs_boost", 0.36)),
                scale=scale,
            )
            n_pseudo += _add_negative_disagreement_spikes(
                x[i],
                qrs[i],
                rng,
                rate=float(spec.get("bad_neg_spike_rate", 0.18)),
                amp_scale=0.42 * scale,
                qrs_bias=float(spec.get("bad_hf_qrs_bias", 0.54)),
            )
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.18)), 0.28 * scale, invert=True)
        elif fam in {"bad_hf_detector_core", "bad_hf_qrs_core"}:
            residual = x[i] - clean[i]
            # CleanBUT bad-core keeps narrow high-frequency QRS-like structure
            # and has poor detector agreement; it is not a low-amplitude
            # smooth/dead signal. Remove slow residual drift, then add
            # polarity-mixed high-frequency bursts.
            x[i] = clean[i] + (residual - _smooth(residual, 65)) * 0.25
            if float(spec.get("bad_qrs_attenuation", 0.0)) > 0:
                _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.05)), int(spec.get("bad_qrs_smooth", 11)))
            n_pseudo += _add_hf_detector_disagree_bursts(
                x[i],
                clean[i],
                qrs[i],
                rng,
                burst_rate=float(spec.get("bad_hf_burst_rate", 0.65)),
                amp_scale=float(spec.get("bad_hf_amp", 0.65)) * scale,
                qrs_bias=float(spec.get("bad_hf_qrs_bias", 0.72)),
            )
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.32)), scale, invert=True)
        elif fam == "bad_bandlimited_disagree_core":
            hf = _band_limited_noise(
                len(x[i]),
                rng,
                float(spec.get("bad_hf_lo", 17.0)),
                float(spec.get("bad_hf_hi", 32.0)),
            )
            amp = _robust_scale(clean[i]) * float(spec.get("bad_hf_amp", 1.05)) * scale
            # Keep enough clean morphology for QRS visibility, but let the
            # 15-30Hz component dominate the bad-class spectral signature.
            centered_clean = clean[i] - float(np.median(clean[i]))
            x[i] = float(np.median(clean[i])) + centered_clean * float(spec.get("bad_core_gain", 0.62)) + hf * amp
            if float(spec.get("bad_qrs_attenuation", 0.0)) > 0:
                _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.15)), int(spec.get("bad_qrs_smooth", 7)))
            n_pseudo += _add_negative_disagreement_spikes(
                x[i],
                qrs[i],
                rng,
                rate=float(spec.get("bad_neg_spike_rate", 0.22)),
                amp_scale=0.55 * scale,
                qrs_bias=float(spec.get("bad_hf_qrs_bias", 0.35)),
            )
            n_pseudo += _add_controlled_pseudo_peaks(x[i], qrs[i], rng, float(spec.get("bad_pseudo_peak_rate", 0.20)), 0.35 * scale, invert=True)
        elif fam == "bad_bw_platform_core":
            _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.74)), int(spec.get("bad_qrs_smooth", 21)))
            n_platform += _add_baseline_platforms(x[i], rng, float(spec.get("bad_baseline_step_rate", 0.48)), scale)
            x[i] = np.median(x[i]) + _smooth(x[i] - np.median(x[i]), int(spec.get("bad_core_smooth", 17))) * float(spec.get("bad_core_gain", 0.52)) * scale
        elif fam == "bad_clipped_lowamp_core":
            _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.76)), int(spec.get("bad_qrs_smooth", 17)))
            _clip_lowamp(x[i], float(spec.get("bad_clip_quantile", 0.55)), float(spec.get("bad_core_gain", 0.34)) * scale)
            n_flat += _insert_flat_segments(x[i], rng, float(spec.get("bad_core_flat_prob", 0.42)), spec.get("bad_core_flat_len", [40, 140]), scale)
        elif fam == "hybrid_real_noise_core":
            _attenuate_qrs(x[i], clean[i], qrs[i], float(spec.get("bad_qrs_attenuation", 0.66)), int(spec.get("bad_qrs_smooth", 17)))
            x[i] = np.median(x[i]) + (x[i] - np.median(x[i])) * float(spec.get("bad_core_gain", 0.70)) * scale
            n_platform += _add_baseline_platforms(x[i], rng, float(spec.get("bad_baseline_step_rate", 0.20)), scale * 0.65)

        secondary = "none"
        if secondary_prob > 0.0 and rng.random() < secondary_prob:
            secondary = str(rng.choice(["qrs_erased", "lowamp_clip", "platform"]))
            if secondary == "qrs_erased":
                _attenuate_qrs(x[i], clean[i], qrs[i], 0.35, 13)
            elif secondary == "lowamp_clip":
                _clip_lowamp(x[i], 0.78, 0.75)
            else:
                n_platform += _add_baseline_platforms(x[i], rng, 0.18, 0.5)

        subtype_rows.append(
            {
                "idx": i,
                "y_class": "bad",
                "fatal_subtype_primary": fam,
                "fatal_subtype_secondary": secondary,
                "fatal_subtype_strength": scale,
                "flat_segments": n_flat,
                "pseudo_peaks": n_pseudo,
                "platform_segments": n_platform,
                "rms_before": float(np.sqrt(np.mean(before * before))),
                "rms_after": float(np.sqrt(np.mean(x[i] * x[i]))),
            }
        )

    audit_df = pd.DataFrame(subtype_rows)
    good_audit_df = pd.DataFrame(good_rows)
    labels["fatal_subtype_primary"] = "none"
    labels["fatal_subtype_secondary"] = "none"
    labels["fatal_subtype_strength"] = 0.0
    labels["good_core_mode"] = "none"
    if not audit_df.empty:
        labels.loc[audit_df["idx"].to_numpy(dtype=int), "fatal_subtype_primary"] = audit_df["fatal_subtype_primary"].to_numpy()
        labels.loc[audit_df["idx"].to_numpy(dtype=int), "fatal_subtype_secondary"] = audit_df["fatal_subtype_secondary"].to_numpy()
        labels.loc[audit_df["idx"].to_numpy(dtype=int), "fatal_subtype_strength"] = audit_df["fatal_subtype_strength"].to_numpy()
    if not good_audit_df.empty:
        labels.loc[good_audit_df["idx"].to_numpy(dtype=int), "good_core_mode"] = good_audit_df["good_mode"].to_numpy()
    labels["sample_source"] = labels.get("sample_source", "sensors2025").astype(str) + f"|clean_core_targeted:{spec['id']}"

    np.savez_compressed(data_dir / "synth_10s_125hz_noisy.npz", X_noisy=x.astype(np.float32))
    np.savez_compressed(data_dir / "signals.npz", X=x[:, None, :].astype(np.float32))
    labels.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)
    labels.to_csv(data_dir / "synth_10s_125hz_labels_with_level.csv", index=False)

    audit = read_json(variant_dir / "data_variant_audit.json") if (variant_dir / "data_variant_audit.json").exists() else {}
    subtype_summary = {
        "spec_id": spec["id"],
        "family": spec.get("family"),
        "secondary_prob": secondary_prob,
        "bad_alt_family": alt_family,
        "bad_alt_cluster_prob": alt_prob,
        "good_mode": str(spec.get("good_mode", "g_amp_qrs")),
        "good_rows": int(len(good_idx)),
        "mean_good_rms_after": float(good_audit_df["rms_after"].mean()) if not good_audit_df.empty else math.nan,
        "bad_rows": int(len(bad_idx)),
        "primary_counts": {str(k): int(v) for k, v in labels.loc[labels["y_class"].astype(str) == "bad", "fatal_subtype_primary"].value_counts().to_dict().items()},
        "secondary_counts": {str(k): int(v) for k, v in labels.loc[labels["y_class"].astype(str) == "bad", "fatal_subtype_secondary"].value_counts().to_dict().items()},
        "mean_bad_rms_after": float(audit_df["rms_after"].mean()) if not audit_df.empty else math.nan,
    }
    audit["clean_core_targeted"] = subtype_summary
    write_json(variant_dir / "data_variant_audit.json", audit)
    write_json(variant_dir / "bad_subtype_audit.json", subtype_summary)
    audit_df.to_csv(variant_dir / SUBTYPE_AUDIT_NAME, index=False)
    good_audit_df.to_csv(variant_dir / "good_core_audit.csv", index=False)
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
    made2 = apply_medium_and_bad_core(spec, variant_dir, seed=int(args.seed))
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
        variant_dir / "visuals" / "bad_core_gallery.png",
        str(spec["id"]),
    )
    return variant_dir, made2


def load_clean_target(clean_root: Path) -> tuple[pd.DataFrame, list[str]]:
    thresholds = read_json(clean_root / "clean_rule_thresholds.json")
    manifest = pd.read_csv(clean_root / "clean_subset_manifest.csv")
    target = manifest[manifest["is_clean_train_target"].astype(bool)].copy()
    features = [f for f in list(thresholds["features"]) if f in target.columns]
    if len(features) < 16:
        raise ValueError(f"CleanBUT target has too few feature columns: {len(features)}")
    return target, features


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def fit_clean_pca(clean_target: pd.DataFrame, features: list[str], seed: int) -> dict[str, Any]:
    fit_rows = clean_target[clean_target["split"].astype(str).eq("train")].copy()
    if fit_rows.empty:
        fit_rows = clean_target.copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=seed)
    X = imputer.fit_transform(_numeric(fit_rows, features))
    Z = scaler.fit_transform(X)
    pca.fit(Z)
    clean_xy = pca.transform(scaler.transform(imputer.transform(_numeric(clean_target, features))))
    clean_target = clean_target.copy()
    clean_target["pc1_common"] = clean_xy[:, 0]
    clean_target["pc2_common"] = clean_xy[:, 1]
    stats: dict[str, Any] = {}
    for cls in CLASS_NAMES:
        pts = clean_target.loc[clean_target["class_name"].eq(cls), ["pc1_common", "pc2_common"]].to_numpy(dtype=float)
        if len(pts) < 4:
            stats[cls] = {"centroid": [math.nan, math.nan], "cov": [[math.nan, 0.0], [0.0, math.nan]], "radius": math.nan}
        else:
            centroid = pts.mean(axis=0)
            cov = np.cov(pts.T) + np.eye(2) * 1e-6
            radius = float(np.median(np.linalg.norm(pts - centroid[None, :], axis=1)) + 1e-6)
            stats[cls] = {"centroid": centroid.tolist(), "cov": cov.tolist(), "radius": radius}
    return {"imputer": imputer, "scaler": scaler, "pca": pca, "clean_projected": clean_target, "class_stats": stats}


def project_features(df: pd.DataFrame, features: list[str], pca_bundle: dict[str, Any]) -> np.ndarray:
    X = pca_bundle["imputer"].transform(_numeric(df, features))
    Z = pca_bundle["scaler"].transform(X)
    return pca_bundle["pca"].transform(Z)


def pca_geometry_metrics(clean_stats: dict[str, Any], synth_full: pd.DataFrame, features: list[str], pca_bundle: dict[str, Any]) -> dict[str, float]:
    xy = project_features(synth_full, features, pca_bundle)
    synth = synth_full.copy()
    synth["pc1_common"] = xy[:, 0]
    synth["pc2_common"] = xy[:, 1]

    out: dict[str, float] = {}
    for cls in CLASS_NAMES:
        pts = synth.loc[synth["class_name"].eq(cls), ["pc1_common", "pc2_common"]].to_numpy(dtype=float)
        clean_cls = clean_stats[cls]
        clean_centroid_cls = np.asarray(clean_cls["centroid"], dtype=float)
        clean_cov_cls = np.asarray(clean_cls["cov"], dtype=float)
        if len(pts) < 4:
            out[f"{cls}_pca_centroid_distance"] = 1.0
            out[f"{cls}_pca_covariance_gap"] = 1.0
            out[f"{cls}_pca_range_gap"] = 1.0
            out[f"{cls}_pca_pc1_raw"] = math.nan
            out[f"{cls}_pca_pc2_raw"] = math.nan
            continue
        synth_centroid_cls = pts.mean(axis=0)
        synth_cov_cls = np.cov(pts.T) + np.eye(2) * 1e-6
        centroid_raw_cls = float(np.linalg.norm(synth_centroid_cls - clean_centroid_cls))
        clean_eig_cls = np.linalg.eigvalsh(clean_cov_cls)
        synth_eig_cls = np.linalg.eigvalsh(synth_cov_cls)
        cov_raw_cls = float(np.mean(np.abs(np.log((synth_eig_cls + 1e-6) / (clean_eig_cls + 1e-6)))))
        clean_proj_cls = pca_bundle["clean_projected"].loc[
            pca_bundle["clean_projected"]["class_name"].astype(str).eq(cls), ["pc1_common", "pc2_common"]
        ].to_numpy(dtype=float)
        clean_range = np.nanpercentile(clean_proj_cls, 95, axis=0) - np.nanpercentile(clean_proj_cls, 5, axis=0)
        synth_range = np.nanpercentile(pts, 95, axis=0) - np.nanpercentile(pts, 5, axis=0)
        range_raw_cls = float(np.mean(np.abs(np.log((synth_range + 1e-6) / (clean_range + 1e-6)))))
        out[f"{cls}_pca_centroid_distance"] = float(np.clip(centroid_raw_cls / 3.0, 0.0, 1.5))
        out[f"{cls}_pca_covariance_gap"] = float(np.clip(cov_raw_cls / 4.0, 0.0, 1.5))
        out[f"{cls}_pca_range_gap"] = float(np.clip(range_raw_cls / 4.0, 0.0, 1.5))
        out[f"{cls}_pca_pc1_raw"] = float(synth_centroid_cls[0])
        out[f"{cls}_pca_pc2_raw"] = float(synth_centroid_cls[1])

    bad_pts = synth.loc[synth["class_name"].eq("bad"), ["pc1_common", "pc2_common"]].to_numpy(dtype=float)
    if len(bad_pts) < 4:
        out.update({
            "bad_pca_centroid_distance": 1.0,
            "bad_pca_covariance_gap": 1.0,
            "bad_pca_pc2_distance": 1.0,
            "bad_pca_core_miss_rate": 1.0,
            "bad_pca_centroid_distance_raw": math.nan,
            "bad_pca_covariance_gap_raw": math.nan,
            "bad_pca_pc1_raw": math.nan,
            "bad_pca_pc2_raw": math.nan,
            "bad_pca_core_hit_rate": math.nan,
        })
        return out
    clean_bad = clean_stats["bad"]
    clean_centroid = np.asarray(clean_bad["centroid"], dtype=float)
    clean_cov = np.asarray(clean_bad["cov"], dtype=float)
    synth_centroid = bad_pts.mean(axis=0)
    synth_cov = np.cov(bad_pts.T) + np.eye(2) * 1e-6
    centroid_raw = float(np.linalg.norm(synth_centroid - clean_centroid))
    # CleanBUT bad is extremely tight, so radius-normalization saturates too
    # early and cannot distinguish "somewhat close" from "far away".  Use a
    # fixed PCA-space scale for selection, while still recording the raw gap.
    centroid_norm = float(np.clip(centroid_raw / 3.0, 0.0, 1.5))
    clean_eig = np.linalg.eigvalsh(clean_cov)
    synth_eig = np.linalg.eigvalsh(synth_cov)
    cov_raw = float(np.mean(np.abs(np.log((synth_eig + 1e-6) / (clean_eig + 1e-6)))))
    cov_norm = float(np.clip(cov_raw / 4.0, 0.0, 1.5))
    pc2_raw = float(abs(synth_centroid[1] - clean_centroid[1]))
    pc2_norm = float(np.clip(pc2_raw / 3.0, 0.0, 1.5))
    core_dist = np.linalg.norm(bad_pts - clean_centroid[None, :], axis=1)
    # The clean bad core radius is tiny.  Use a practical PCA radius so the
    # selector rewards compact coverage without requiring exact duplicates.
    core_hit_rate = float(np.mean(core_dist <= 0.55))
    out.update({
        "bad_pca_centroid_distance": centroid_norm,
        "bad_pca_covariance_gap": cov_norm,
        "bad_pca_pc2_distance": pc2_norm,
        "bad_pca_core_miss_rate": float(1.0 - core_hit_rate),
        "bad_pca_centroid_distance_raw": centroid_raw,
        "bad_pca_covariance_gap_raw": cov_raw,
        "bad_pca_pc1_raw": float(synth_centroid[0]),
        "bad_pca_pc2_raw": float(synth_centroid[1]),
        "bad_pca_core_hit_rate": core_hit_rate,
    })
    return out


def sample_by_class(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        sub = df[df["class_name"].astype(str).eq(cls)]
        if len(sub) > n_per_class:
            sub = sub.sample(n=n_per_class, random_state=seed + i)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else df.iloc[:0].copy()


def _filter_npz_rows(path: Path, row_pos: np.ndarray, n_old: int) -> None:
    if not path.exists():
        return
    with np.load(path) as npz:
        data = {}
        for key in npz.files:
            arr = npz[key]
            data[key] = arr[row_pos] if getattr(arr, "shape", (0,))[0] == n_old else arr
    np.savez_compressed(path, **data)


def _coverage_target_points(clean_projected: pd.DataFrame, cls: str, n: int, seed: int) -> np.ndarray:
    sub = clean_projected.loc[
        clean_projected["class_name"].astype(str).eq(cls), ["pc1_common", "pc2_common"]
    ].dropna()
    if sub.empty:
        return np.zeros((0, 2), dtype=float)
    if len(sub) <= int(n):
        return sub.to_numpy(dtype=float)
    work = sub.copy()
    q1 = max(2, min(5, int(np.sqrt(len(work)))))
    q2 = max(2, min(6, int(np.sqrt(len(work)))))
    work["_pc1_bin"] = pd.qcut(work["pc1_common"].rank(method="first"), q=q1, labels=False, duplicates="drop")
    work["_pc2_bin"] = pd.qcut(work["pc2_common"].rank(method="first"), q=q2, labels=False, duplicates="drop")
    groups = list(work.groupby(["_pc1_bin", "_pc2_bin"], dropna=False))
    rng = np.random.default_rng(seed)
    per_bin = max(1, int(math.ceil(float(n) / max(1, len(groups)))))
    parts: list[pd.DataFrame] = []
    for gi, (_, g) in enumerate(groups):
        take = min(len(g), per_bin)
        parts.append(g.sample(n=take, random_state=seed + gi) if len(g) > take else g)
    targets = pd.concat(parts, ignore_index=False)
    if len(targets) < int(n):
        remaining = work.drop(index=targets.index, errors="ignore")
        if remaining.empty:
            remaining = work
        fill_n = int(n) - len(targets)
        replace = len(remaining) < fill_n
        fill_idx = rng.choice(np.arange(len(remaining)), size=fill_n, replace=replace)
        targets = pd.concat([targets, remaining.iloc[fill_idx]], ignore_index=False)
    if len(targets) > int(n):
        targets = targets.sample(n=int(n), random_state=seed + 919)
    return targets[["pc1_common", "pc2_common"]].to_numpy(dtype=float)


def _greedy_match_targets(sub: pd.DataFrame, target_pts: np.ndarray, keep_n: int, centroid: np.ndarray) -> pd.DataFrame:
    if len(sub) <= int(keep_n) or len(target_pts) == 0:
        kept = sub.nsmallest(min(int(keep_n), len(sub)), "_core_dist").copy()
        kept["_target_pc1"] = float(centroid[0])
        kept["_target_pc2"] = float(centroid[1])
        return kept
    pts = sub[["_pc1_core", "_pc2_core"]].to_numpy(dtype=float)
    available = np.ones(len(sub), dtype=bool)
    selected: list[int] = []
    target_pc1: list[float] = []
    target_pc2: list[float] = []
    # Walk through coverage targets and pick the nearest unused synthetic row.
    # This preserves elongated good/medium geometry instead of collapsing to a centroid.
    for target in target_pts:
        if len(selected) >= int(keep_n) or not bool(available.any()):
            break
        dist = np.sum((pts - target[None, :]) ** 2, axis=1)
        dist[~available] = np.inf
        j = int(np.argmin(dist))
        if not math.isfinite(float(dist[j])):
            break
        available[j] = False
        selected.append(j)
        target_pc1.append(float(target[0]))
        target_pc2.append(float(target[1]))
    if len(selected) < int(keep_n):
        remaining = np.where(available)[0]
        if len(remaining):
            fill = sub.iloc[remaining].nsmallest(int(keep_n) - len(selected), "_core_dist").index
            for idx in fill:
                j = int(sub.index.get_loc(idx))
                if available[j]:
                    available[j] = False
                    selected.append(j)
                    target_pc1.append(float(centroid[0]))
                    target_pc2.append(float(centroid[1]))
    kept = sub.iloc[selected].copy()
    kept["_target_pc1"] = target_pc1[: len(kept)]
    kept["_target_pc2"] = target_pc2[: len(kept)]
    return kept


def apply_clean_core_feature_selection(
    args: argparse.Namespace,
    variant_dir: Path,
    synth_full: pd.DataFrame,
    clean_target: pd.DataFrame,
    common: list[str],
    pca_bundle: dict[str, Any],
    target_per_class: int,
) -> pd.DataFrame:
    """Filter synthetic pools to samples nearest CleanBUT class cores.

    This is generator-target selection only.  It uses labels plus waveform
    features/PCA fitted on CleanBUT train core; it never uses model predictions
    or BUT test outcomes.
    """

    if not bool(getattr(args, "feature_select_core", True)):
        return synth_full
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        return synth_full
    full = synth_full.copy().reset_index(drop=True)
    if len(full) <= int(target_per_class) * len(CLASS_NAMES):
        return full
    xy = project_features(full, common, pca_bundle)
    full["_row_pos"] = np.arange(len(full), dtype=int)
    full["_pc1_core"] = xy[:, 0]
    full["_pc2_core"] = xy[:, 1]

    selected_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    for cls in CLASS_NAMES:
        sub = full[full["class_name"].astype(str).eq(cls)].copy()
        if sub.empty:
            continue
        stats = pca_bundle["class_stats"][cls]
        centroid = np.asarray(stats["centroid"], dtype=float)
        pc = sub[["_pc1_core", "_pc2_core"]].to_numpy(dtype=float)
        core_dist = np.linalg.norm(pc - centroid[None, :], axis=1)
        sub["_core_dist"] = core_dist
        if cls == "bad":
            # For bad, PC2/core compactness is the failure mode we are fixing.
            sub["_select_score"] = (
                sub["_core_dist"]
                + 0.75 * np.abs(sub["_pc2_core"] - float(centroid[1]))
                + 0.10 * np.abs(sub["_pc1_core"] - float(centroid[0]))
            )
            selection_mode = "tight_bad_core"
            keep_n = min(int(target_per_class), len(sub))
            kept = sub.nsmallest(keep_n, "_select_score").copy()
            kept["_target_pc1"] = float(centroid[0])
            kept["_target_pc2"] = float(centroid[1])
        else:
            sub["_select_score"] = sub["_core_dist"]
            selection_mode = f"{cls}_coverage_match"
            keep_n = min(int(target_per_class), len(sub))
            targets = _coverage_target_points(
                pca_bundle["clean_projected"],
                cls,
                keep_n,
                int(args.seed) + {"good": 11, "medium": 23}.get(cls, 37),
            )
            kept = _greedy_match_targets(sub, targets, keep_n, centroid)
        kept["_feature_selected_rank"] = np.arange(1, len(kept) + 1, dtype=int)
        kept["_selection_mode"] = selection_mode
        selected_parts.append(kept)
        manifest_parts.append(
            kept[
                [
                    "_row_pos",
                    "idx",
                    "class_name",
                    "_pc1_core",
                    "_pc2_core",
                    "_core_dist",
                    "_select_score",
                    "_target_pc1",
                    "_target_pc2",
                    "_selection_mode",
                    "_feature_selected_rank",
                ]
            ].copy()
        )

    if not selected_parts:
        return full
    selected = pd.concat(selected_parts, ignore_index=True)
    row_pos = selected["_row_pos"].to_numpy(dtype=int)
    row_pos = np.sort(row_pos)
    n_old = len(full)
    data_dir = variant_dir / "datasets"
    for name in (
        "signals.npz",
        "synth_10s_125hz_noisy.npz",
        "synth_10s_125hz_clean.npz",
        "synth_10s_125hz_local_mask.npz",
    ):
        _filter_npz_rows(data_dir / name, row_pos, n_old)
    labels = pd.read_csv(labels_path).reset_index(drop=True)
    labels_selected = labels.iloc[row_pos].copy().reset_index(drop=True)
    labels_selected["feature_core_selected"] = 1
    labels_selected.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)
    labels_selected.to_csv(labels_path, index=False)

    selected_full = full.iloc[row_pos].copy().reset_index(drop=True)
    selected_full = selected_full.drop(columns=[c for c in selected_full.columns if c.startswith("_")], errors="ignore")
    selected_full.to_csv(variant_dir / "synthetic_atlas64_features.csv", index=False)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest.to_csv(variant_dir / "feature_core_selection_manifest.csv", index=False)
    write_json(
        variant_dir / "feature_core_selection_summary.json",
        {
            "enabled": True,
            "target_per_class": int(target_per_class),
            "pool_rows": int(n_old),
            "selected_rows": int(len(row_pos)),
            "selection_basis": "CleanBUT train-core PCA class centroid distance; bad additionally weights PC2/core compactness; no predictions.",
            "good_medium_selection": "coverage-aware greedy matching to stratified CleanBUT PCA target points",
            "bad_selection": "tight bad core nearest centroid with PC2 compactness weighting",
        },
    )
    return selected_full


def score_one(args: argparse.Namespace, spec: dict[str, Any], clean_target: pd.DataFrame, features: list[str], pca_bundle: dict[str, Any], sample_per_class: int) -> dict[str, Any]:
    pool_mult = max(1, int(getattr(args, "core_select_pool_multiplier", 1))) if bool(getattr(args, "feature_select_core", True)) else 1
    variant_dir, made = make_variant(args, spec, int(sample_per_class) * pool_mult)
    report_path = variant_dir / "clean64_distance_report.json"
    if report_path.exists() and not args.force:
        return read_json(report_path)

    labels = pd.read_csv(variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv")
    meta = labels.copy()
    meta["dataset"] = "PTB CleanBUT-core targeted synthetic"
    meta["class_name"] = meta["y_class"].astype(str)
    if bool(args.force):
        for stale in (variant_dir / "synthetic_morph_features.csv", variant_dir / "synthetic_atlas64_features.csv"):
            if stale.exists():
                stale.unlink()
    morph_path = variant_dir / "synthetic_morph_features.csv"
    morph_path.parent.mkdir(parents=True, exist_ok=True)
    X_pool = np.load(variant_dir / "datasets" / "signals.npz")["X"].astype(np.float32)
    rn.extract_features(X_pool, meta, morph_path, max_rows=0)
    synth_full = visual64.compute_full64_variant_features(variant_dir, features, force=bool(args.force))
    if "class_name" not in synth_full.columns and "y_class" in synth_full.columns:
        synth_full["class_name"] = synth_full["y_class"].astype(str)

    common = [f for f in features if f in clean_target.columns and f in synth_full.columns]
    if len(common) < 16:
        raise RuntimeError(f"{spec['id']}: only {len(common)} common 64D columns")
    synth_full = apply_clean_core_feature_selection(args, variant_dir, synth_full, clean_target, common, pca_bundle, int(sample_per_class))
    if "class_name" not in synth_full.columns and "y_class" in synth_full.columns:
        synth_full["class_name"] = synth_full["y_class"].astype(str)
    labels = pd.read_csv(variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv")
    meta = labels.copy()
    meta["dataset"] = "PTB CleanBUT-core targeted synthetic"
    meta["class_name"] = meta["y_class"].astype(str)
    X_synth = np.load(variant_dir / "datasets" / "signals.npz")["X"].astype(np.float32)
    try:
        if morph_path.exists() and bool(args.force):
            morph_path.unlink()
        synth_morph = rn.extract_features(X_synth, meta, morph_path, max_rows=0)
    except Exception as exc:
        synth_morph = pd.DataFrame()
        write_json(
            variant_dir / "synthetic_morph_features_warning.json",
            {
                "warning": "morphology feature extraction failed after CleanBUT core selection; 64D atlas features were still used for selection.",
                "error": str(exc),
            },
        )
    dist = clean_atlas.classwise_distance(clean_target, synth_full, common)
    good = float(dist["by_class"].get("good", math.nan))
    medium = float(dist["by_class"].get("medium", math.nan))
    bad = float(dist["by_class"].get("bad", math.nan))
    overall = float(dist["overall"])
    class_worst = float(np.nanmax([good, medium, bad]))
    geom = pca_geometry_metrics(pca_bundle["class_stats"], synth_full, common, pca_bundle)
    clean_domain = sample_by_class(clean_target, int(args.domain_rows_per_class), int(args.seed))
    synth_domain = sample_by_class(synth_full, int(args.domain_rows_per_class), int(args.seed) + 77)
    domain_bal = rn.domain_separability(clean_domain, synth_domain, common, seed=int(args.seed)) if bool(args.compute_domain) else math.nan
    domain_score = float(domain_bal if math.isfinite(domain_bal) else 1.0)
    score = float(
        0.11 * good
        + 0.12 * medium
        + 0.13 * bad
        + 0.06 * class_worst
        + 0.12 * geom["bad_pca_centroid_distance"]
        + 0.12 * geom["bad_pca_covariance_gap"]
        + 0.08 * geom["bad_pca_pc2_distance"]
        + 0.08 * geom["bad_pca_core_miss_rate"]
        + 0.08 * geom.get("medium_pca_covariance_gap", 1.0)
        + 0.05 * geom.get("medium_pca_range_gap", 1.0)
        + 0.04 * geom.get("good_pca_covariance_gap", 1.0)
        + 0.03 * geom.get("good_pca_range_gap", 1.0)
        + 0.03 * domain_score
    )
    xy = project_features(synth_full, common, pca_bundle)
    synth_proj = pd.DataFrame(
        {
            "variant_id": str(spec["id"]),
            "source": "PTB synthetic",
            "class_name": synth_full["class_name"].astype(str).to_numpy(),
            "idx": synth_full["idx"].to_numpy() if "idx" in synth_full.columns else np.arange(len(synth_full), dtype=int),
            "pc1_common": xy[:, 0],
            "pc2_common": xy[:, 1],
        }
    )
    synth_proj.to_csv(variant_dir / PROJECTION_NAME, index=False)

    row = {
        "variant_id": str(spec["id"]),
        "family": spec.get("family"),
        "spec": spec,
        "variant_dir": str(variant_dir),
        "score": score,
        "bad_64d_KS": bad,
        "medium_64d_KS": medium,
        "good_64d_KS": good,
        "class_worst_64d_KS": class_worst,
        "overall_64d_KS": overall,
        "bad_PCA_centroid_distance": float(geom["bad_pca_centroid_distance"]),
        "bad_PCA_covariance_gap": float(geom["bad_pca_covariance_gap"]),
        "bad_PCA_pc2_distance": float(geom["bad_pca_pc2_distance"]),
        "bad_PCA_core_miss_rate": float(geom["bad_pca_core_miss_rate"]),
        "bad_PCA_centroid_distance_raw": float(geom["bad_pca_centroid_distance_raw"]),
        "bad_PCA_covariance_gap_raw": float(geom["bad_pca_covariance_gap_raw"]),
        "bad_PCA_pc1_raw": float(geom["bad_pca_pc1_raw"]),
        "bad_PCA_pc2_raw": float(geom["bad_pca_pc2_raw"]),
        "bad_PCA_core_hit_rate": float(geom["bad_pca_core_hit_rate"]),
        "good_PCA_covariance_gap": float(geom.get("good_pca_covariance_gap", math.nan)),
        "good_PCA_range_gap": float(geom.get("good_pca_range_gap", math.nan)),
        "good_PCA_pc1_raw": float(geom.get("good_pca_pc1_raw", math.nan)),
        "good_PCA_pc2_raw": float(geom.get("good_pca_pc2_raw", math.nan)),
        "medium_PCA_covariance_gap": float(geom.get("medium_pca_covariance_gap", math.nan)),
        "medium_PCA_range_gap": float(geom.get("medium_pca_range_gap", math.nan)),
        "medium_PCA_pc1_raw": float(geom.get("medium_pca_pc1_raw", math.nan)),
        "medium_PCA_pc2_raw": float(geom.get("medium_pca_pc2_raw", math.nan)),
        "domain_separability": domain_score,
        "n_common_features": int(len(common)),
        "rows": int(len(synth_full)),
        "feature_select_core": bool(getattr(args, "feature_select_core", True)),
        "core_select_pool_multiplier": int(pool_mult),
        "bad_distance_improvement_vs_baseline": float((BASELINE_64D_BAD_DISTANCE - bad) / BASELINE_64D_BAD_DISTANCE),
        "medium_regression_vs_baseline": float((medium - BASELINE_64D_MEDIUM_DISTANCE) / max(BASELINE_64D_MEDIUM_DISTANCE, 1e-6)),
        "selection_score_note": "Coverage-aware CleanBUT score: KS 0.36, bad tight-core geometry 0.40, medium/good PCA coverage 0.20, domain 0.03.",
        "audit": made.get("audit", {}),
    }
    write_json(report_path, row)
    write_json(variant_dir / "clean64_distance_rows.json", dist["rows"])
    write_json(variant_dir / "synthetic_feature_profile.json", {"full64_features": synth_full.describe().to_dict(), "morph_features": synth_morph.describe().to_dict(), "audit": made.get("audit", {})})
    return row


def leaderboard_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "variant_id",
        "family",
        "score",
        "bad_64d_KS",
        "medium_64d_KS",
        "good_64d_KS",
        "class_worst_64d_KS",
        "overall_64d_KS",
        "bad_PCA_centroid_distance",
        "bad_PCA_covariance_gap",
        "bad_PCA_pc2_distance",
        "bad_PCA_core_miss_rate",
        "bad_PCA_pc1_raw",
        "bad_PCA_pc2_raw",
        "bad_PCA_core_hit_rate",
        "good_PCA_covariance_gap",
        "good_PCA_range_gap",
        "good_PCA_pc1_raw",
        "good_PCA_pc2_raw",
        "medium_PCA_covariance_gap",
        "medium_PCA_range_gap",
        "medium_PCA_pc1_raw",
        "medium_PCA_pc2_raw",
        "domain_separability",
        "bad_distance_improvement_vs_baseline",
        "medium_regression_vs_baseline",
        "n_common_features",
        "rows",
        "feature_select_core",
        "core_select_pool_multiplier",
        "variant_dir",
    ]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    if df.empty:
        return df
    df = df.sort_values(["score", "class_worst_64d_KS", "good_64d_KS", "bad_64d_KS"], na_position="last").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    return df


def score_specs(args: argparse.Namespace, *, stage_name: str = "scan") -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    clean_target, features = load_clean_target(Path(args.clean_root))
    pca_bundle = fit_clean_pca(clean_target, features, int(args.seed))
    sample_per_class = int(args.sample_per_class_refine if stage_name == "refine" else args.sample_per_class_scan)

    if stage_name == "refine":
        prior = read_json(out_root / "clean64_distance_leaderboard.json")["rows"] if (out_root / "clean64_distance_leaderboard.json").exists() else []
        parents = [r["spec"] for r in sorted(prior, key=lambda r: float(r.get("score", 999.0)))[: int(args.refine_parent_count)]]
        specs = make_specs(min(int(args.max_specs), int(args.max_refine_specs)), refined=True, parents=parents)
    else:
        specs = make_specs(int(args.max_specs))
    write_json(out_root / SPEC_NAME, {"rows": specs, "stage": stage_name})
    write_json(report_root / SPEC_NAME, {"rows": specs, "stage": stage_name})

    existing: list[dict[str, Any]] = []
    json_path = out_root / "clean64_distance_leaderboard.json"
    if stage_name == "refine" and json_path.exists():
        existing = read_json(json_path).get("rows", [])
    rows: list[dict[str, Any]] = []
    update_state(out_root / STATE_NAME, status=f"{stage_name}_running", stage=stage_name, total=len(specs), updated_at=now_iso())
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status=f"{stage_name}_running", stage=stage_name, current=spec["id"], completed=i - 1, total=len(specs), updated_at=now_iso())
        try:
            rows.append(score_one(args, spec, clean_target, features, pca_bundle, sample_per_class))
        except Exception as exc:
            rows.append({"variant_id": str(spec.get("id")), "family": spec.get("family"), "spec": spec, "status": "failed", "error": str(exc), "score": 999.0})
            append_jsonl(out_root / "clean64_scan_failures.jsonl", rows[-1])
        if i == 1 or i % 6 == 0:
            live = leaderboard_frame(existing + rows)
            live.to_csv(out_root / LEADERBOARD_NAME, index=False)
            live.to_csv(report_root / LEADERBOARD_NAME, index=False)

    combined = sorted(existing + rows, key=lambda r: float(r.get("score", 999.0)))
    write_json(json_path, {"rows": combined, "stage": stage_name, "baseline_bad_distance": BASELINE_64D_BAD_DISTANCE})
    lb = leaderboard_frame(combined)
    lb.to_csv(out_root / LEADERBOARD_NAME, index=False)
    lb.to_csv(report_root / LEADERBOARD_NAME, index=False)

    detail_rows: list[dict[str, Any]] = []
    subtype_rows: list[pd.DataFrame] = []
    for row in combined:
        vdir = Path(str(row.get("variant_dir", "")))
        dpath = vdir / "clean64_distance_rows.json"
        if dpath.exists():
            for d in read_json(dpath):
                detail_rows.append({"variant_id": row.get("variant_id"), **d})
        apath = vdir / SUBTYPE_AUDIT_NAME
        if apath.exists():
            try:
                tmp = pd.read_csv(apath)
                tmp["variant_id"] = row.get("variant_id")
                subtype_rows.append(tmp)
            except Exception:
                pass
    pd.DataFrame(detail_rows).to_csv(out_root / DETAIL_NAME, index=False)
    pd.DataFrame(detail_rows).to_csv(report_root / DETAIL_NAME, index=False)
    if subtype_rows:
        pd.concat(subtype_rows, ignore_index=True).to_csv(out_root / SUBTYPE_AUDIT_NAME, index=False)
        pd.concat(subtype_rows, ignore_index=True).to_csv(report_root / SUBTYPE_AUDIT_NAME, index=False)

    build_visuals(args, clean_target, features, pca_bundle, combined)
    write_report(args)
    best = combined[0] if combined else {}
    update_state(
        out_root / STATE_NAME,
        status=f"{stage_name}_complete",
        stage=stage_name,
        completed=len(rows),
        total=len(specs),
        best_variant=best.get("variant_id"),
        best_score=best.get("score"),
        best_bad_64d_KS=best.get("bad_64d_KS"),
        updated_at=now_iso(),
    )
    return combined


def build_visuals(args: argparse.Namespace, clean_target: pd.DataFrame, features: list[str], pca_bundle: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    seen_variants: set[str] = set()
    top_rows = []
    for r in rows:
        variant_id = str(r.get("variant_id", ""))
        if variant_id in seen_variants:
            continue
        if "variant_dir" in r and Path(str(r.get("variant_dir", ""))).exists() and float(r.get("score", 999.0)) < 999.0:
            top_rows.append(r)
            seen_variants.add(variant_id)
        if len(top_rows) >= min(9, int(args.visual_top_n)):
            break
    if not top_rows:
        return

    clean_sample = []
    for i, cls in enumerate(CLASS_NAMES):
        sub = pca_bundle["clean_projected"][pca_bundle["clean_projected"]["class_name"].astype(str).eq(cls)]
        if len(sub) > int(args.clean_plot_rows_per_class):
            sub = sub.sample(n=int(args.clean_plot_rows_per_class), random_state=int(args.seed) + i)
        clean_sample.append(sub)
    clean_plot = pd.concat(clean_sample, ignore_index=True)
    clean_out = clean_plot[["class_name", "pc1_common", "pc2_common"]].copy()
    clean_out["source"] = "CleanBUT-Core"
    clean_out["variant_id"] = "CleanBUT-Core"
    projected_rows = [clean_out]

    for row in top_rows:
        vdir = Path(str(row["variant_dir"]))
        proj_path = vdir / PROJECTION_NAME
        if proj_path.exists():
            proj = pd.read_csv(proj_path)
        else:
            full = pd.read_csv(vdir / "synthetic_atlas64_features.csv")
            xy = project_features(full, features, pca_bundle)
            proj = pd.DataFrame(
                {
                    "variant_id": row["variant_id"],
                    "source": "PTB synthetic",
                    "class_name": full["class_name"].astype(str),
                    "idx": full["idx"].to_numpy() if "idx" in full.columns else np.arange(len(full), dtype=int),
                    "pc1_common": xy[:, 0],
                    "pc2_common": xy[:, 1],
                }
            )
        for k in ["score", "good_64d_KS", "medium_64d_KS", "bad_64d_KS", "bad_PCA_centroid_distance", "bad_PCA_covariance_gap"]:
            proj[k] = row.get(k, math.nan)
        projected_rows.append(proj)
    projected = pd.concat(projected_rows, ignore_index=True)
    projected.to_csv(out_root / PROJECTION_NAME, index=False)
    projected.to_csv(report_root / PROJECTION_NAME, index=False)

    plot_overlay(projected, fig_dir / "top_rules_64d_overlay.png", top_rows)
    plot_best_rule(projected, fig_dir / "best_rule_64d_overlay.png", top_rows[0])
    plot_centroid_shift(projected, fig_dir / "bad_core_centroid_shift.png", top_rows)
    lb = leaderboard_frame(rows)
    plot_distance_bars(lb, fig_dir / "classwise_distance_bars.png", top_n=min(24, len(lb)))


def plot_overlay(projected: pd.DataFrame, out_path: Path, top_rows: list[dict[str, Any]]) -> None:
    clean = projected[projected["source"].eq("CleanBUT-Core")]
    variants = [r["variant_id"] for r in top_rows]
    ncols = 3
    nrows = int(math.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.4 * nrows), squeeze=False)
    for ax, vid in zip(axes.ravel(), variants):
        for cls in CLASS_NAMES:
            cdf = clean[clean["class_name"].eq(cls)]
            ax.scatter(cdf["pc1_common"], cdf["pc2_common"], s=6, c=COLORS[cls], alpha=0.14, linewidths=0)
        vf = projected[projected["variant_id"].eq(vid)]
        for cls in CLASS_NAMES:
            sdf = vf[vf["class_name"].eq(cls)]
            ax.scatter(sdf["pc1_common"], sdf["pc2_common"], s=24, c=COLORS[cls], alpha=0.75, marker="x", linewidths=1.0)
        row = next(r for r in top_rows if r["variant_id"] == vid)
        ax.set_title(f"{vid}\nscore={row.get('score', math.nan):.3f} bad={row.get('bad_64d_KS', math.nan):.3f}", fontsize=8)
        ax.set_xlabel("CleanBUT 64D-PC1")
        ax.set_ylabel("CleanBUT 64D-PC2")
        ax.grid(alpha=0.15)
    for ax in axes.ravel()[len(variants) :]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=f"Clean {c}", markerfacecolor=COLORS[c], markersize=7, alpha=0.65) for c in CLASS_NAMES]
    handles.append(plt.Line2D([0], [0], marker="x", color="#333333", label="PTB synthetic", markersize=7, linestyle="None"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("CleanBUT-Core 64D target scan: top PTB synthetic overlays", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_rule(projected: pd.DataFrame, out_path: Path, row: dict[str, Any]) -> None:
    clean = projected[projected["source"].eq("CleanBUT-Core")]
    vf = projected[projected["variant_id"].eq(row["variant_id"])]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for cls in CLASS_NAMES:
        cdf = clean[clean["class_name"].eq(cls)]
        ax.scatter(cdf["pc1_common"], cdf["pc2_common"], s=10, c=COLORS[cls], alpha=0.17, linewidths=0, label=f"Clean {cls}")
    for cls in CLASS_NAMES:
        sdf = vf[vf["class_name"].eq(cls)]
        ax.scatter(sdf["pc1_common"], sdf["pc2_common"], s=38, c=COLORS[cls], marker="x", alpha=0.8, linewidths=1.1, label=f"Synth {cls}")
    ax.set_title(f"Best CleanBUT bad-core targeted rule\n{row['variant_id']} | score={row.get('score', math.nan):.3f}, bad={row.get('bad_64d_KS', math.nan):.3f}")
    ax.set_xlabel("CleanBUT 64D-PC1")
    ax.set_ylabel("CleanBUT 64D-PC2")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_shift(projected: pd.DataFrame, out_path: Path, top_rows: list[dict[str, Any]]) -> None:
    clean = projected[projected["source"].eq("CleanBUT-Core")]
    clean_centers = clean.groupby("class_name")[["pc1_common", "pc2_common"]].mean()
    rows = []
    for row in top_rows:
        vf = projected[projected["variant_id"].eq(row["variant_id"])]
        for cls in CLASS_NAMES:
            pts = vf[vf["class_name"].eq(cls)][["pc1_common", "pc2_common"]]
            if pts.empty or cls not in clean_centers.index:
                continue
            center = pts.mean()
            delta = center.to_numpy() - clean_centers.loc[cls].to_numpy()
            rows.append({"variant_id": row["variant_id"], "class_name": cls, "centroid_distance": float(np.linalg.norm(delta))})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(top_rows) * 0.9), 5.2))
    variant_order = list(dict.fromkeys(str(r["variant_id"]) for r in top_rows))
    pivot = (
        df.pivot_table(index="variant_id", columns="class_name", values="centroid_distance", aggfunc="mean")
        .reindex(variant_order)
    )
    x = np.arange(len(pivot))
    width = 0.24
    for j, cls in enumerate(CLASS_NAMES):
        ax.bar(x + (j - 1) * width, pivot[cls].to_numpy(), width=width, color=COLORS[cls], alpha=0.75, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels([v[:38] for v in pivot.index], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("PCA centroid distance to CleanBUT core")
    ax.set_title("Centroid shift in CleanBUT 64D PCA space")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distance_bars(df: pd.DataFrame, out_path: Path, top_n: int) -> None:
    if df.empty:
        return
    view = df.head(top_n).copy()
    labels = [str(v)[:42] for v in view["variant_id"]]
    x = np.arange(len(view))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(12, len(view) * 0.62), 5.4))
    ax.bar(x - width, view["good_64d_KS"], width, label="good", color=COLORS["good"], alpha=0.75)
    ax.bar(x, view["medium_64d_KS"], width, label="medium", color=COLORS["medium"], alpha=0.75)
    ax.bar(x + width, view["bad_64d_KS"], width, label="bad", color=COLORS["bad"], alpha=0.75)
    ax.plot(x, view["score"], color="#333333", marker="o", lw=1.0, label="weighted score")
    ax.axhline(BASELINE_64D_BAD_DISTANCE, color=COLORS["bad"], linestyle="--", lw=1.0, alpha=0.55, label="old bad distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("64D CleanBUT distance")
    ax.set_title("CleanBUT 64D distance leaderboard: lower is closer")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def should_refine(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return True
    best = min(rows, key=lambda r: float(r.get("score", 999.0)))
    bad = float(best.get("bad_64d_KS", 999.0))
    medium = float(best.get("medium_64d_KS", 999.0))
    return not (bad <= BASELINE_64D_BAD_DISTANCE * 0.80 and medium <= BASELINE_64D_MEDIUM_DISTANCE * 1.15)


def select_training_rows(args: argparse.Namespace, n: int) -> list[dict[str, Any]]:
    path = Path(args.out_root) / "clean64_distance_leaderboard.json"
    rows = read_json(path)["rows"] if path.exists() else score_specs(args)
    rows = [r for r in rows if Path(str(r.get("variant_dir", ""))).exists()]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        variant_id = str(row.get("variant_id", ""))
        if not variant_id or variant_id in seen or len(selected) >= n:
            return
        selected.append(row)
        seen.add(variant_id)

    for row in sorted(rows, key=lambda r: float(r.get("score", 999.0))):
        add(row)
        if len(selected) >= max(1, int(math.ceil(n * 0.65))):
            break

    # Keep a few counterweights: best overall score is not always the best bad-core
    # or medium-coverage fit, and quick training is where that tradeoff is tested.
    for key in ("bad_64d_KS", "medium_64d_KS", "good_64d_KS"):
        for row in sorted(rows, key=lambda r, k=key: float(r.get(k, 999.0))):
            add(row)
            if len(selected) >= n:
                break
        if len(selected) >= n:
            break

    return selected


def ensure_training_artifact_reindexed(variant_dir: Path) -> None:
    data_dir = variant_dir / "datasets"
    labels_path = data_dir / "synth_10s_125hz_labels_with_level.csv"
    labels_simple_path = data_dir / "synth_10s_125hz_labels.csv"
    noisy_path = data_dir / "synth_10s_125hz_noisy.npz"
    if not labels_path.exists() or not noisy_path.exists():
        return
    labels = pd.read_csv(labels_path)
    n_rows = int(np.load(noisy_path)["X_noisy"].shape[0])
    if len(labels) != n_rows:
        labels = labels.iloc[:n_rows].copy().reset_index(drop=True)
    if "idx" not in labels.columns:
        labels["idx"] = np.arange(len(labels), dtype=int)
    old_idx = labels["idx"].to_numpy(dtype=np.int64, copy=True)
    needs_reindex = len(labels) != n_rows or int(np.nanmax(old_idx)) >= n_rows or int(np.nanmin(old_idx)) < 0
    if not needs_reindex:
        return
    if "original_idx" not in labels.columns:
        labels["original_idx"] = old_idx
    labels["idx"] = np.arange(len(labels), dtype=int)
    labels.to_csv(labels_path, index=False)
    labels.to_csv(labels_simple_path, index=False)
    write_json(
        variant_dir / "training_artifact_reindex_audit.json",
        {
            "rows": int(len(labels)),
            "array_rows": int(n_rows),
            "old_idx_min": int(np.nanmin(old_idx)),
            "old_idx_max": int(np.nanmax(old_idx)),
            "new_idx_min": 0,
            "new_idx_max": int(len(labels) - 1),
            "reason": "feature-space selected arrays were compacted, so labels idx must be compact for mainline loader",
        },
    )


def train_one(args: argparse.Namespace, row: dict[str, Any], mode: str) -> dict[str, Any]:
    if not bool(args.run_training):
        raise RuntimeError("Training is disabled. Re-run with --run_training after reviewing visual fit.")
    spec = row["spec"]
    spec_id = str(spec["id"])
    variant_dir = Path(row["variant_dir"])
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if int(args.train_regenerate_full) == 1:
        if not labels_path.exists() or len(pd.read_csv(labels_path)) < 1000:
            variant_dir, _ = make_variant(args, spec, 0, force_regenerate=True)
    ensure_training_artifact_reindexed(variant_dir)
    run_dir = Path(args.out_root) / "runs" / mode / spec_id
    summary_file = run_dir / "clean_core_targeted_run_summary.json"
    if summary_file.exists() and not args.rerun_existing:
        existing = read_json(summary_file)
        if int(existing.get("returncode", 1)) == 0 or bool(args.skip_failed_existing):
            existing["skipped_existing"] = True
            return existing
    e1, e2 = (int(args.quick_epochs_stage1), int(args.quick_epochs_stage2)) if mode == "quick" else (int(args.full_epochs_stage1), int(args.full_epochs_stage2))
    summary_path = run_dir / "mainline_summary.json"
    ckpt_path = run_dir / "ckpt_best.pt"
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
        str(e1),
        "--epochs_stage2",
        str(e2),
        "--batch_size_stage1",
        str(args.batch_size_stage1),
        "--batch_size_stage2",
        str(args.batch_size_stage2),
        "--hidden_dim",
        str(args.hidden_dim),
        "--denoiser_width",
        str(args.denoiser_width),
        "--lambda_den_stage2",
        str(args.lambda_den_stage2),
        "--lambda_cls",
        str(args.lambda_cls),
        "--cls_loss_variant",
        str(args.cls_loss_variant),
        "--class_weight",
        str(spec.get("class_weight", args.class_weight)),
        "--seed",
        str(args.seed),
    ]
    start = time.perf_counter()
    reused_existing_training = bool(ckpt_path.exists() and summary_path.exists())
    if reused_existing_training:
        returncode = 0
    else:
        log_dir = Path(args.out_root) / "logs" / mode
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"{spec_id}.stdout.txt").open("w", encoding="utf-8") as out, (log_dir / f"{spec_id}.stderr.txt").open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
        returncode = int(proc.returncode)
    payload: dict[str, Any] = {
        "spec": spec,
        "mode": mode,
        "returncode": int(returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
        "reused_existing_training": reused_existing_training,
        "clean64_distance": {k: row.get(k) for k in ["score", "bad_64d_KS", "medium_64d_KS", "good_64d_KS", "domain_separability"]},
    }
    if returncode == 0 and ckpt_path.exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, ckpt_path, run_dir)
    else:
        payload["status"] = "failed"
    write_json(summary_file, payload)
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    write_report(args)
    return payload


def run_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = select_training_rows(args, int(args.top_quick))
    update_state(Path(args.out_root) / STATE_NAME, status="quick_training", stage="quick", total=len(selected), updated_at=now_iso())
    rows = []
    for i, row in enumerate(selected, start=1):
        update_state(Path(args.out_root) / STATE_NAME, status="quick_training", stage="quick", current=row["variant_id"], completed=i - 1, total=len(selected), updated_at=now_iso())
        rows.append(train_one(args, row, "quick"))
    update_state(Path(args.out_root) / STATE_NAME, status="quick_complete", stage="quick", completed=len(rows), total=len(rows), updated_at=now_iso())
    return rows


def _metric_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    return (float(rep.get("macro_f1", 0.0)), float(rep.get("acc", 0.0)), min(float(rec[1]), float(rec[2])), float(rec[2]))


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(args.out_root) / SUMMARY_NAME)
    quick = [r for r in rows if r.get("mode") == "quick" and int(r.get("returncode", 1)) == 0]
    if not quick:
        quick = run_quick(args)
    selected = sorted(quick, key=_metric_score, reverse=True)[: int(args.top_full)]
    update_state(Path(args.out_root) / STATE_NAME, status="full_training", stage="full", total=len(selected), updated_at=now_iso())
    out = []
    for i, row in enumerate(selected, start=1):
        vid = row.get("spec", {}).get("id")
        update_state(Path(args.out_root) / STATE_NAME, status="full_training", stage="full", current=vid, completed=i - 1, total=len(selected), updated_at=now_iso())
        out.append(train_one(args, row, "full"))
    update_state(Path(args.out_root) / STATE_NAME, status="full_complete", stage="full", completed=len(out), total=len(out), updated_at=now_iso())
    return out


def _balanced_cleanbut_indices(manifest: pd.DataFrame, tier_values: set[str], *, seed: int = 20260607) -> tuple[np.ndarray, dict[str, Any]]:
    sub = manifest.loc[manifest["clean_tier"].astype(str).isin(tier_values)].copy()
    counts = {int(cls): int((sub["y"].astype(int) == cls).sum()) for cls in (0, 1, 2)}
    if min(counts.values()) <= 0:
        return np.asarray([], dtype=np.int64), {
            "source_counts_good_medium_bad": [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)],
            "n_per_class": 0,
            "n_total": 0,
            "warning": "missing at least one class",
        }
    n_per_class = int(min(counts.values()))
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for cls in (0, 1, 2):
        cls_idx = sub.loc[sub["y"].astype(int) == cls, "idx"].to_numpy(dtype=np.int64)
        selected.extend(int(v) for v in rng.choice(cls_idx, size=n_per_class, replace=False).tolist())
    selected_idx = np.asarray(sorted(selected), dtype=np.int64)
    return selected_idx, {
        "source_counts_good_medium_bad": [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)],
        "n_per_class": n_per_class,
        "selected_counts_good_medium_bad": [n_per_class, n_per_class, n_per_class],
        "n_total": int(len(selected_idx)),
        "seed": int(seed),
        "note": "CleanBUT-Core diagnostic only; not the formal BUT benchmark and not used for threshold selection.",
    }


def _cleanbut_formal_test_indices(manifest: pd.DataFrame, tier_values: set[str]) -> tuple[np.ndarray, dict[str, Any]]:
    sub = manifest.loc[
        manifest["clean_tier"].astype(str).isin(tier_values)
        & manifest["split"].astype(str).eq("test")
    ].copy()
    counts = {int(cls): int((sub["y"].astype(int) == cls).sum()) for cls in (0, 1, 2)}
    return sub["idx"].to_numpy(dtype=np.int64), {
        "source_counts_good_medium_bad": [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)],
        "n_total": int(len(sub)),
        "warning": "The formal CleanBUT clean-core test split is not class complete when bad count is zero.",
        "note": "Unbalanced clean-core test diagnostic only; original BUT test remains primary.",
    }


def add_cleanbut_core_diagnostics(args: argparse.Namespace, checkpoint: Path, run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(args.clean_root) / "clean_subset_manifest.csv"
    if not manifest_path.exists():
        report["clean_but_diagnostic_error"] = f"missing {manifest_path}"
        return report
    X, meta = ensure_but_10s(args)
    y = meta["y"].to_numpy(dtype=np.int64)
    manifest = pd.read_csv(manifest_path)
    manifest["idx"] = manifest["idx"].astype(int)
    manifest["y"] = manifest["y"].astype(int)
    subsets = {
        "clean_but_strict_balanced": _balanced_cleanbut_indices(manifest, {"clean_core_strict"}),
        "clean_but_train_target_balanced": _balanced_cleanbut_indices(manifest, {"clean_core_strict", "clean_core_train_target"}),
        "clean_but_formal_test_unbalanced": _cleanbut_formal_test_indices(manifest, {"clean_core_strict", "clean_core_train_target"}),
    }
    nonempty = [idx for idx, _ in subsets.values() if len(idx) > 0]
    if not nonempty:
        report["clean_but_diagnostic_error"] = "no non-empty CleanBUT diagnostic subset"
        return report

    cal = report.get("calibration", {})
    t_good = float(cal.get("t_good", 0.5))
    t_bad = float(cal.get("t_bad", 0.5))
    union_idx = np.unique(np.concatenate(nonempty))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_uformer_model(checkpoint, device, feature_set="full_tokens", with_head=True)
    outputs = run_model_outputs(model, X[union_idx], int(args.batch_size_eval), device)
    probs_union = outputs["probs"]
    assert isinstance(probs_union, np.ndarray)
    loc = {int(idx): pos for pos, idx in enumerate(union_idx.tolist())}
    out_dir = run_dir / "but_10s_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name, (idx, selection) in subsets.items():
        if len(idx) == 0:
            summary[name] = {"selection": selection, "calibrated_report": None, "raw_argmax_report": None}
            continue
        pos = np.asarray([loc[int(v)] for v in idx], dtype=np.int64)
        probs = probs_union[pos]
        raw_pred = np.argmax(probs, axis=1).astype(np.int64)
        cal_pred = apply_but_thresholds(probs, t_good, t_bad)
        calibrated = multiclass_report(y[idx], cal_pred, probs)
        raw = multiclass_report(y[idx], raw_pred, probs)
        report[f"{name}_report"] = calibrated
        report[f"{name}_raw_report"] = raw
        report[f"{name}_selection"] = selection
        summary[name] = {"selection": selection, "calibrated_report": calibrated, "raw_argmax_report": raw}

    # Backwards-compatible summary field used by the report table.
    if "clean_but_train_target_balanced_report" in report:
        report["clean_but_test_report"] = report["clean_but_train_target_balanced_report"]
        report["clean_but_raw_test_report"] = report["clean_but_train_target_balanced_raw_report"]
    write_json(out_dir / "clean_but_core_diagnostic_report.json", summary)
    write_json(out_dir / "but_10s_eval_summary.json", report)
    return report


def evaluate_checkpoint_10s(args: argparse.Namespace, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    report = base_evaluate_checkpoint_10s(args, checkpoint, run_dir)
    return add_cleanbut_core_diagnostics(args, checkpoint, run_dir, report)


def write_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    lb = pd.read_csv(out_root / LEADERBOARD_NAME) if (out_root / LEADERBOARD_NAME).exists() else pd.DataFrame()
    train_rows = []
    for r in load_jsonl(out_root / SUMMARY_NAME):
        rep = r.get("but_10s_eval", {}).get("but_10s_test_report", {})
        bal = r.get("but_10s_eval", {}).get("but_10s_balanced_test_report", {})
        clean_diag = r.get("but_10s_eval", {}).get("clean_but_test_report", {})
        rec = rep.get("recall_good_medium_bad", [math.nan, math.nan, math.nan])
        train_rows.append(
            {
                "mode": r.get("mode"),
                "variant_id": r.get("spec", {}).get("id"),
                "acc": rep.get("acc"),
                "macro_f1": rep.get("macro_f1"),
                "good_recall": rec[0],
                "medium_recall": rec[1],
                "bad_recall": rec[2],
                "balanced_macro": bal.get("macro_f1"),
                "clean_diag_macro": clean_diag.get("macro_f1"),
                **r.get("clean64_distance", {}),
            }
        )
    train_df = pd.DataFrame(train_rows)
    if not train_df.empty:
        train_df.to_csv(out_root / "clean_core_targeted_metric_join.csv", index=False)
        train_df.to_csv(report_root / "clean_core_targeted_metric_join.csv", index=False)
    best = lb.head(1).to_dict("records")[0] if not lb.empty else {}
    lines = [
        "# CleanBUT Bad-Core Targeted Synthetic Grid",
        "",
        "This is a generator-target scan. CleanBUT-Core is used only as a target/diagnostic subset; original BUT 10s P1 remains the benchmark.",
        "",
        "## Current Best CPU Fit",
        "",
        f"- Best variant: `{best.get('variant_id', 'n/a')}`",
        f"- Weighted score: `{float(best.get('score', math.nan)):.4f}`",
        f"- Bad 64D distance: `{float(best.get('bad_64d_KS', math.nan)):.4f}` vs prior baseline `~{BASELINE_64D_BAD_DISTANCE:.3f}`",
        f"- Medium 64D distance: `{float(best.get('medium_64d_KS', math.nan)):.4f}` vs prior baseline `~{BASELINE_64D_MEDIUM_DISTANCE:.3f}`",
        "",
        "## Figures",
        "",
        "- `figures/top_rules_64d_overlay.png`: CleanBUT-Core background with top targeted PTB rules.",
        "- `figures/best_rule_64d_overlay.png`: best current candidate in the same PCA space.",
        "- `figures/bad_core_centroid_shift.png`: class centroid gaps in CleanBUT 64D PCA.",
        "- `figures/classwise_distance_bars.png`: class-wise distance leaderboard.",
        "",
        "## Top No-Training Candidates",
        "",
        md_table(lb[["rank", "variant_id", "family", "score", "class_worst_64d_KS", "good_64d_KS", "medium_64d_KS", "bad_64d_KS", "bad_distance_improvement_vs_baseline", "medium_regression_vs_baseline"]] if not lb.empty else lb, max_rows=20),
        "",
        "## Training Results",
        "",
        md_table(train_df, max_rows=20) if not train_df.empty else "_No training has been run from this target scan yet._",
        "",
        "## Notes",
        "",
        "- Selection uses CleanBUT train-target core features and does not inspect BUT test predictions.",
        "- Synthetic `sqi_iSQI` remains a single-lead detector-agreement proxy.",
        "- `all` defaults to CPU distribution fitting only; pass `--run_training` for quick/full training after visual review.",
    ]
    (report_root / "final_clean_core_targeted_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / STATE_NAME
    if state_path.exists():
        try:
            state = read_json(state_path)
            if "error" in state:
                state.pop("error", None)
                write_json(state_path, state)
        except Exception:
            pass
    update_state(out_root / STATE_NAME, status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"scan", "all"}:
        rows = score_specs(args, stage_name="scan")
        if args.stage == "all" and should_refine(rows):
            score_specs(args, stage_name="refine")
    if args.stage == "refine":
        score_specs(args, stage_name="refine")
    if args.stage == "quick":
        run_quick(args)
    if args.stage == "full":
        run_full(args)
    if args.stage == "report":
        write_report(args)
    update_state(out_root / STATE_NAME, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CleanBUT bad-core targeted PTB synthetic generator grid.")
    p.add_argument("--stage", choices=("scan", "refine", "quick", "full", "report", "all"), default="all")
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    p.add_argument("--clean_root", default=str(DEFAULT_CLEAN_ROOT))
    p.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    p.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    p.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    p.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    p.add_argument("--nstdb_root", default=str(DEFAULT_NSTDB_ROOT))
    p.add_argument("--cinc_dir", default=str(DEFAULT_CINC_DIR))
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max_specs", type=int, default=360)
    p.add_argument("--max_refine_specs", type=int, default=108)
    p.add_argument("--refine_parent_count", type=int, default=12)
    p.add_argument("--sample_per_class_scan", type=int, default=256)
    p.add_argument("--sample_per_class_refine", type=int, default=1000)
    p.add_argument("--no_feature_select_core", dest="feature_select_core", action="store_false")
    p.set_defaults(feature_select_core=True)
    p.add_argument("--core_select_pool_multiplier", type=int, default=3)
    p.add_argument("--top_quick", type=int, default=12)
    p.add_argument("--top_full", type=int, default=4)
    p.add_argument("--feature_mode", choices=("full64",), default="full64")
    p.add_argument("--compute_domain", action="store_true", default=True)
    p.add_argument("--domain_rows_per_class", type=int, default=450)
    p.add_argument("--visual_top_n", type=int, default=9)
    p.add_argument("--clean_plot_rows_per_class", type=int, default=1500)
    p.add_argument("--run_training", action="store_true")
    p.add_argument("--train_regenerate_full", type=int, default=1)
    p.add_argument("--rerun_existing", action="store_true")
    p.add_argument("--skip_failed_existing", action="store_true", default=True)
    p.add_argument("--quick_epochs_stage1", type=int, default=5)
    p.add_argument("--quick_epochs_stage2", type=int, default=5)
    p.add_argument("--full_epochs_stage1", type=int, default=14)
    p.add_argument("--full_epochs_stage2", type=int, default=10)
    p.add_argument("--batch_size_stage1", type=int, default=24)
    p.add_argument("--batch_size_stage2", type=int, default=32)
    p.add_argument("--batch_size_eval", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--denoiser_width", type=float, default=1.5)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--lambda_den_stage2", type=float, default=0.5)
    p.add_argument("--lambda_cls", type=float, default=18.0)
    p.add_argument("--cls_loss_variant", choices=("hard_ce", "sample_weighted_ce", "soft_ce", "soft_sample_weighted_ce"), default="hard_ce")
    p.add_argument("--class_weight", default="1.00,1.55,1.70")
    return p


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
