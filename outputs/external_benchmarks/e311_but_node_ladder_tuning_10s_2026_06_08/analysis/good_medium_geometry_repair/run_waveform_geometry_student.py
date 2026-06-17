"""Waveform Geometry Student research runner.

External-only experiment.  The classifier sees waveform channels only.  The
47-column SQI/geometry table is used as a training-time teacher target, never as
an inference input.  Original BUT rows are report-only and are not used for
training, validation, threshold selection, or candidate selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ARCH_SCRIPT = ANALYSIS_DIR / "run_waveform_architecture_search.py"
NODE_ID = "N17043_gm_probe"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

REFERENCE_ROWS = [
    {
        "candidate": "47_feature_tabular_upper_bound",
        "bucket": "original_test_all_10s+",
        "acc": 0.963548,
        "macro_f1": 0.930683,
        "good_recall": 0.956319,
        "medium_recall": 0.972887,
        "bad_recall": 0.927007,
        "notes": "tabular-only upper bound; geometry features are inference inputs",
    },
    {
        "candidate": "current_best_waveform_transformer",
        "bucket": "original_test_all_10s+",
        "acc": 0.811018,
        "macro_f1": 0.730436,
        "good_recall": 0.960714,
        "medium_recall": 0.725938,
        "bad_recall": 0.401460,
        "notes": "aug_convtx_balanced_focal raw",
    },
    {
        "candidate": "stattoken_baseline",
        "bucket": "original_test_all_10s+",
        "acc": 0.813141,
        "macro_f1": 0.733720,
        "good_recall": 0.962088,
        "medium_recall": 0.725938,
        "bad_recall": 0.433090,
        "notes": "stattx_medium_guard raw",
    },
    {
        "candidate": "reduced_feature_auto_topk20",
        "bucket": "original_test_all_10s+",
        "acc": 0.913531,
        "macro_f1": np.nan,
        "good_recall": 0.973352,
        "medium_recall": 0.884546,
        "bad_recall": 0.695864,
        "notes": "reduced tabular model; not waveform-only",
    },
]

TEACHER_COLUMNS = [
    "pc1",
    "pc2",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "qrs_visibility",
    "qrs_prom_p90",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "diff_abs_p95",
    "template_corr",
    "amplitude_entropy",
    "low_amp_ratio",
    "sqi_bSQI",
    "sqi_sSQI",
    "sqi_kSQI",
]

CORE_COLUMNS = [
    "pc1",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "qrs_visibility",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
]

HARD_FEATURE_COLUMNS = [
    "pc2",
    "pc3",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "sqi_basSQI",
    "low_amp_ratio",
    "boundary_confidence",
    "knn_label_purity",
]

TOP14_FEATURE_COLUMNS = [
    "pc1",
    "qrs_visibility",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "pca_margin",
    "qrs_band_ratio",
    "template_corr",
    "amplitude_entropy",
    "sqi_bSQI",
    "region_confidence",
    "detector_agreement",
    "mean_abs",
    "sqi_basSQI",
]

CANDIDATES: dict[str, dict[str, Any]] = {
    "mapstudent_patchtx": {
        "arch": "mapstudent",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "lr": 5.0e-4,
        "class_weight": [1.00, 1.22, 2.00],
        "focal_gamma": 0.8,
        "aux_weight": 0.75,
        "core_aux_weight": 0.55,
        "supcon_weight": 0.10,
        "rank_weight": 0.18,
        "recon_weight": 0.0,
        "seed": 20260801,
    },
    "masked_geometry_mae": {
        "arch": "masked_mae",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "mask_ratio": 0.35,
        "lr": 4.5e-4,
        "class_weight": [1.00, 1.22, 2.00],
        "focal_gamma": 0.8,
        "aux_weight": 0.70,
        "core_aux_weight": 0.50,
        "supcon_weight": 0.08,
        "rank_weight": 0.18,
        "recon_weight": 0.20,
        "seed": 20260802,
    },
    "stattoken_v2": {
        "arch": "stattoken_v2",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.5e-4,
        "class_weight": [1.00, 1.25, 2.15],
        "focal_gamma": 1.0,
        "aux_weight": 0.65,
        "core_aux_weight": 0.50,
        "supcon_weight": 0.10,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "seed": 20260803,
    },
    "morphology_token_tx": {
        "arch": "morphology_token",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "lr": 5.0e-4,
        "class_weight": [1.00, 1.25, 2.05],
        "focal_gamma": 0.9,
        "aux_weight": 0.70,
        "core_aux_weight": 0.55,
        "supcon_weight": 0.12,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "seed": 20260804,
    },
    "stattoken_v2_badstress": {
        "arch": "stattoken_v2",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.18, 2.75],
        "focal_gamma": 1.1,
        "aux_weight": 0.62,
        "core_aux_weight": 0.48,
        "supcon_weight": 0.10,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "augment_strength": 0.35,
        "bad_outlier_aug_strength": 1.65,
        "seed": 20260805,
    },
    "morphology_token_badstress": {
        "arch": "morphology_token",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "lr": 4.5e-4,
        "class_weight": [1.00, 1.18, 2.55],
        "focal_gamma": 1.0,
        "aux_weight": 0.65,
        "core_aux_weight": 0.50,
        "supcon_weight": 0.12,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "augment_strength": 0.35,
        "bad_outlier_aug_strength": 1.55,
        "seed": 20260806,
    },
    "qrs_detail_token_tx": {
        "arch": "qrs_detail_token",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.8e-4,
        "class_weight": [1.00, 1.24, 2.05],
        "focal_gamma": 0.9,
        "aux_weight": 0.78,
        "core_aux_weight": 0.70,
        "supcon_weight": 0.12,
        "rank_weight": 0.25,
        "recon_weight": 0.0,
        "seed": 20260807,
    },
    "atlas_memory_tx": {
        "arch": "atlas_memory",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "lr": 4.4e-4,
        "class_weight": [1.00, 1.24, 2.10],
        "focal_gamma": 0.9,
        "aux_weight": 0.72,
        "core_aux_weight": 0.70,
        "supcon_weight": 0.18,
        "rank_weight": 0.30,
        "recon_weight": 0.0,
        "seed": 20260808,
    },
    "qrs_atlas_memory_tx": {
        "arch": "qrs_atlas_memory",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.24, 2.10],
        "focal_gamma": 0.9,
        "aux_weight": 0.76,
        "core_aux_weight": 0.76,
        "supcon_weight": 0.18,
        "rank_weight": 0.32,
        "recon_weight": 0.0,
        "seed": 20260809,
    },
    "teacher_atlas_student": {
        "arch": "teacher_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "teacher_prototypes_per_class": 10,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.24, 2.15],
        "focal_gamma": 0.9,
        "aux_weight": 0.72,
        "core_aux_weight": 0.80,
        "atlas_distill_weight": 0.55,
        "supcon_weight": 0.16,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "seed": 20260810,
    },
    "qrs_teacher_atlas_student": {
        "arch": "qrs_teacher_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "teacher_prototypes_per_class": 10,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.24, 2.15],
        "focal_gamma": 0.9,
        "aux_weight": 0.76,
        "core_aux_weight": 0.84,
        "atlas_distill_weight": 0.58,
        "supcon_weight": 0.18,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "seed": 20260811,
    },
    "hybrid_atlas_student": {
        "arch": "hybrid_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "teacher_prototypes_per_class": 10,
        "lr": 4.1e-4,
        "class_weight": [1.00, 1.24, 2.20],
        "focal_gamma": 0.9,
        "aux_weight": 0.76,
        "core_aux_weight": 0.84,
        "atlas_distill_weight": 0.58,
        "supcon_weight": 0.18,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "seed": 20260812,
    },
    "neighbor_stattoken_student": {
        "arch": "stattoken_v2",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.24, 2.35],
        "focal_gamma": 0.9,
        "aux_weight": 0.70,
        "core_aux_weight": 0.82,
        "neighbor_weight": 0.75,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.14,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "seed": 20260813,
    },
    "neighbor_hybrid_atlas_student": {
        "arch": "hybrid_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "teacher_prototypes_per_class": 10,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.24, 2.25],
        "focal_gamma": 0.9,
        "aux_weight": 0.74,
        "core_aux_weight": 0.84,
        "atlas_distill_weight": 0.52,
        "neighbor_weight": 0.70,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.16,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "seed": 20260814,
    },
    "statfed_patch_tx": {
        "arch": "statfed_patch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 160,
        "lr": 4.4e-4,
        "class_weight": [1.00, 1.24, 2.30],
        "focal_gamma": 0.9,
        "aux_weight": 0.72,
        "core_aux_weight": 0.84,
        "neighbor_weight": 0.45,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.14,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "augment_strength": 0.28,
        "bad_outlier_aug_strength": 1.30,
        "seed": 20260815,
    },
    "statfed_patch_atlas_tx": {
        "arch": "statfed_patch_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 160,
        "prototypes_per_class": 8,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.24, 2.35],
        "focal_gamma": 0.9,
        "aux_weight": 0.72,
        "core_aux_weight": 0.84,
        "neighbor_weight": 0.45,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.16,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "augment_strength": 0.28,
        "bad_outlier_aug_strength": 1.35,
        "seed": 20260816,
    },
    "statfed_patch_medium_guard": {
        "arch": "statfed_patch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 160,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.55, 2.15],
        "focal_gamma": 0.9,
        "aux_weight": 0.72,
        "core_aux_weight": 0.84,
        "neighbor_weight": 0.42,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.14,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "augment_strength": 0.26,
        "bad_outlier_aug_strength": 1.25,
        "seed": 20260817,
    },
    "statfed_patch_badstress_guard": {
        "arch": "statfed_patch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 160,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.34, 2.85],
        "focal_gamma": 1.0,
        "aux_weight": 0.72,
        "core_aux_weight": 0.84,
        "neighbor_weight": 0.40,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.16,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 2.20,
        "seed": 20260818,
    },
    "multiscale_statpatch_balanced": {
        "arch": "multiscale_stat_patch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 192,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.42, 2.45],
        "focal_gamma": 0.95,
        "aux_weight": 0.78,
        "core_aux_weight": 0.90,
        "neighbor_weight": 0.48,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.16,
        "rank_weight": 0.38,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.70,
        "seed": 20260819,
    },
    "multiscale_statpatch_medium_badguard": {
        "arch": "multiscale_stat_patch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 192,
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.55, 2.80],
        "focal_gamma": 1.05,
        "aux_weight": 0.78,
        "core_aux_weight": 0.92,
        "neighbor_weight": 0.48,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.18,
        "rank_weight": 0.38,
        "recon_weight": 0.0,
        "augment_strength": 0.34,
        "bad_outlier_aug_strength": 2.35,
        "seed": 20260820,
    },
    "multiscale_statpatch_stress_shell": {
        "arch": "multiscale_stat_patch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 192,
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.44, 2.95],
        "focal_gamma": 1.05,
        "aux_weight": 0.78,
        "core_aux_weight": 0.92,
        "neighbor_weight": 0.48,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.18,
        "rank_weight": 0.38,
        "recon_weight": 0.0,
        "augment_strength": 0.32,
        "bad_outlier_aug_strength": 1.15,
        "bad_stress_shell_strength": 1.35,
        "seed": 20260821,
    },
    "statfed_patch_stress_shell": {
        "arch": "statfed_patch",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 160,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.42, 2.95],
        "focal_gamma": 1.05,
        "aux_weight": 0.72,
        "core_aux_weight": 0.86,
        "neighbor_weight": 0.42,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.16,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.05,
        "bad_stress_shell_strength": 1.25,
        "seed": 20260822,
    },
    "statfed_patch_teacher_atlas": {
        "arch": "statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 160,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.42, 2.55],
        "focal_gamma": 1.0,
        "aux_weight": 0.76,
        "core_aux_weight": 0.98,
        "atlas_distill_weight": 0.85,
        "neighbor_weight": 0.52,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.18,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 0.65,
        "seed": 20260823,
    },
    "multiscale_statpatch_teacher_atlas": {
        "arch": "multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.48, 2.70],
        "focal_gamma": 1.05,
        "aux_weight": 0.80,
        "core_aux_weight": 1.00,
        "atlas_distill_weight": 0.90,
        "neighbor_weight": 0.54,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.20,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.32,
        "bad_outlier_aug_strength": 1.35,
        "bad_stress_shell_strength": 0.75,
        "seed": 20260824,
    },
    "statfed_patch_dualbad_atlas": {
        "arch": "statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 160,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.38, 2.45],
        "focal_gamma": 0.95,
        "aux_weight": 0.74,
        "core_aux_weight": 0.94,
        "atlas_distill_weight": 0.70,
        "bad_aux_weight": 0.55,
        "bad_aux_pos_weight": 2.2,
        "bad_logit_mix": 0.38,
        "neighbor_weight": 0.50,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.16,
        "rank_weight": 0.40,
        "recon_weight": 0.0,
        "augment_strength": 0.28,
        "bad_outlier_aug_strength": 1.10,
        "bad_stress_shell_strength": 0.85,
        "seed": 20260825,
    },
    "multiscale_statpatch_dualbad_atlas": {
        "arch": "multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.42, 2.55],
        "focal_gamma": 1.0,
        "aux_weight": 0.78,
        "core_aux_weight": 0.96,
        "atlas_distill_weight": 0.72,
        "bad_aux_weight": 0.55,
        "bad_aux_pos_weight": 2.4,
        "bad_logit_mix": 0.40,
        "neighbor_weight": 0.52,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.18,
        "rank_weight": 0.40,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.20,
        "bad_stress_shell_strength": 0.95,
        "seed": 20260826,
    },
    "stressbank_statfed_atlas": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 176,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.38, 2.75],
        "focal_gamma": 1.05,
        "aux_weight": 0.76,
        "core_aux_weight": 0.98,
        "atlas_distill_weight": 0.82,
        "bad_aux_weight": 0.70,
        "bad_aux_pos_weight": 2.8,
        "bad_logit_mix": 0.46,
        "neighbor_weight": 0.52,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.18,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.25,
        "seed": 20260827,
    },
    "stressbank_multiscale_atlas": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 208,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.6e-4,
        "class_weight": [1.00, 1.42, 2.90],
        "focal_gamma": 1.08,
        "aux_weight": 0.80,
        "core_aux_weight": 1.00,
        "atlas_distill_weight": 0.84,
        "bad_aux_weight": 0.72,
        "bad_aux_pos_weight": 3.0,
        "bad_logit_mix": 0.48,
        "neighbor_weight": 0.54,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.20,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.32,
        "bad_outlier_aug_strength": 1.35,
        "bad_stress_shell_strength": 1.35,
        "seed": 20260828,
    },
    "stresstoken_statfed_atlas": {
        "arch": "stresstoken_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 176,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.40, 2.85],
        "focal_gamma": 1.05,
        "aux_weight": 0.78,
        "core_aux_weight": 1.00,
        "atlas_distill_weight": 0.84,
        "bad_aux_weight": 0.76,
        "bad_aux_pos_weight": 3.0,
        "bad_logit_mix": 0.50,
        "neighbor_weight": 0.54,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.20,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.45,
        "seed": 20260829,
    },
    "stresstoken_multiscale_atlas": {
        "arch": "stresstoken_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 208,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.6e-4,
        "class_weight": [1.00, 1.44, 3.05],
        "focal_gamma": 1.10,
        "aux_weight": 0.82,
        "core_aux_weight": 1.02,
        "atlas_distill_weight": 0.86,
        "bad_aux_weight": 0.80,
        "bad_aux_pos_weight": 3.2,
        "bad_logit_mix": 0.52,
        "neighbor_weight": 0.56,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.20,
        "rank_weight": 0.44,
        "recon_weight": 0.0,
        "augment_strength": 0.32,
        "bad_outlier_aug_strength": 1.35,
        "bad_stress_shell_strength": 1.55,
        "seed": 20260830,
    },
    "stressbank_statfed_highstress": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 176,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.6e-4,
        "class_weight": [1.00, 1.36, 3.65],
        "focal_gamma": 1.15,
        "aux_weight": 0.76,
        "core_aux_weight": 0.98,
        "atlas_distill_weight": 0.80,
        "bad_aux_weight": 0.92,
        "bad_aux_pos_weight": 4.0,
        "bad_logit_mix": 0.72,
        "neighbor_weight": 0.50,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.18,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.10,
        "bad_stress_shell_strength": 2.35,
        "seed": 20260831,
    },
    "stressbank_multiscale_highstress": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [50, 100, 250, 500],
        "stat_hidden": 208,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.4e-4,
        "class_weight": [1.00, 1.40, 3.85],
        "focal_gamma": 1.18,
        "aux_weight": 0.80,
        "core_aux_weight": 1.00,
        "atlas_distill_weight": 0.82,
        "bad_aux_weight": 0.96,
        "bad_aux_pos_weight": 4.2,
        "bad_logit_mix": 0.78,
        "neighbor_weight": 0.52,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.20,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.32,
        "bad_outlier_aug_strength": 1.20,
        "bad_stress_shell_strength": 2.55,
        "seed": 20260832,
    },
    "statbank_token_balanced": {
        "arch": "statbank_token_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.42, 2.65],
        "focal_gamma": 1.0,
        "aux_weight": 0.78,
        "core_aux_weight": 1.00,
        "atlas_distill_weight": 0.86,
        "bad_aux_weight": 0.62,
        "bad_aux_pos_weight": 2.6,
        "bad_logit_mix": 0.42,
        "neighbor_weight": 0.56,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.18,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.28,
        "bad_outlier_aug_strength": 1.10,
        "bad_stress_shell_strength": 1.10,
        "seed": 20260833,
    },
    "statbank_token_badguard": {
        "arch": "statbank_token_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.40, 3.20],
        "focal_gamma": 1.10,
        "aux_weight": 0.78,
        "core_aux_weight": 1.00,
        "atlas_distill_weight": 0.84,
        "bad_aux_weight": 0.78,
        "bad_aux_pos_weight": 3.4,
        "bad_logit_mix": 0.58,
        "neighbor_weight": 0.56,
        "neighbor_tau": 0.40,
        "supcon_weight": 0.18,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.20,
        "bad_stress_shell_strength": 1.45,
        "seed": 20260834,
    },
    "retrieval_statfed_protoatlas": {
        "arch": "statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "embedding_prototypes_per_class": 18,
        "embedding_proto_weight": 0.72,
        "embedding_proto_temp": 0.35,
        "embedding_proto_logit_mix": 0.34,
        "lr": 3.5e-4,
        "class_weight": [1.00, 1.44, 2.85],
        "focal_gamma": 1.08,
        "aux_weight": 0.72,
        "core_aux_weight": 0.96,
        "atlas_distill_weight": 0.74,
        "bad_aux_weight": 0.70,
        "bad_aux_pos_weight": 3.0,
        "bad_logit_mix": 0.48,
        "neighbor_weight": 0.66,
        "neighbor_tau": 0.34,
        "supcon_weight": 0.22,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.45,
        "seed": 20260901,
    },
    "retrieval_stressbank_protoatlas": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "embedding_prototypes_per_class": 18,
        "embedding_proto_weight": 0.78,
        "embedding_proto_temp": 0.32,
        "embedding_proto_logit_mix": 0.38,
        "lr": 3.4e-4,
        "class_weight": [1.00, 1.42, 3.10],
        "focal_gamma": 1.12,
        "aux_weight": 0.76,
        "core_aux_weight": 1.00,
        "atlas_distill_weight": 0.78,
        "bad_aux_weight": 0.82,
        "bad_aux_pos_weight": 3.4,
        "bad_logit_mix": 0.56,
        "neighbor_weight": 0.70,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.24,
        "rank_weight": 0.44,
        "recon_weight": 0.0,
        "augment_strength": 0.34,
        "bad_outlier_aug_strength": 1.35,
        "bad_stress_shell_strength": 1.75,
        "seed": 20260902,
    },
    "retrieval_statbank_protoatlas": {
        "arch": "statbank_token_teacher_atlas",
        "channels": "robust3",
        "width": 128,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "embedding_prototypes_per_class": 18,
        "embedding_proto_weight": 0.68,
        "embedding_proto_temp": 0.34,
        "embedding_proto_logit_mix": 0.32,
        "lr": 3.8e-4,
        "class_weight": [1.00, 1.44, 2.90],
        "focal_gamma": 1.08,
        "aux_weight": 0.78,
        "core_aux_weight": 1.04,
        "atlas_distill_weight": 0.82,
        "bad_aux_weight": 0.72,
        "bad_aux_pos_weight": 3.0,
        "bad_logit_mix": 0.48,
        "neighbor_weight": 0.68,
        "neighbor_tau": 0.34,
        "supcon_weight": 0.20,
        "rank_weight": 0.42,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.45,
        "seed": 20260903,
    },
    "waveatlas_statfed_memory": {
        "arch": "statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "waveform_atlas_prototypes_per_class": 32,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.70,
        "waveform_atlas_temp": 0.70,
        "waveform_atlas_logit_mix": 0.42,
        "lr": 3.4e-4,
        "class_weight": [1.00, 1.44, 2.80],
        "focal_gamma": 1.05,
        "aux_weight": 0.62,
        "core_aux_weight": 0.88,
        "atlas_distill_weight": 0.64,
        "bad_aux_weight": 0.62,
        "bad_aux_pos_weight": 2.8,
        "bad_logit_mix": 0.42,
        "neighbor_weight": 0.52,
        "neighbor_tau": 0.38,
        "supcon_weight": 0.18,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "augment_strength": 0.28,
        "bad_outlier_aug_strength": 1.20,
        "bad_stress_shell_strength": 1.25,
        "seed": 20260911,
    },
    "waveatlas_stressbank_memory": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 192,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "waveform_atlas_prototypes_per_class": 36,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.78,
        "waveform_atlas_temp": 0.64,
        "waveform_atlas_logit_mix": 0.50,
        "lr": 3.3e-4,
        "class_weight": [1.00, 1.42, 3.25],
        "focal_gamma": 1.12,
        "aux_weight": 0.66,
        "core_aux_weight": 0.90,
        "atlas_distill_weight": 0.66,
        "bad_aux_weight": 0.86,
        "bad_aux_pos_weight": 3.6,
        "bad_logit_mix": 0.62,
        "neighbor_weight": 0.54,
        "neighbor_tau": 0.38,
        "supcon_weight": 0.20,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "augment_strength": 0.32,
        "bad_outlier_aug_strength": 1.35,
        "bad_stress_shell_strength": 1.90,
        "seed": 20260912,
    },
    "waveatlas_statbank_memory": {
        "arch": "statbank_token_teacher_atlas",
        "channels": "robust3",
        "width": 128,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 14,
        "teacher_prototype_mode": "farthest",
        "waveform_atlas_prototypes_per_class": 40,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.72,
        "waveform_atlas_temp": 0.68,
        "waveform_atlas_logit_mix": 0.46,
        "lr": 3.7e-4,
        "class_weight": [1.00, 1.44, 2.95],
        "focal_gamma": 1.08,
        "aux_weight": 0.68,
        "core_aux_weight": 0.94,
        "atlas_distill_weight": 0.70,
        "bad_aux_weight": 0.72,
        "bad_aux_pos_weight": 3.0,
        "bad_logit_mix": 0.50,
        "neighbor_weight": 0.54,
        "neighbor_tau": 0.38,
        "supcon_weight": 0.18,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.50,
        "seed": 20260913,
    },
    "featurefusion_statfed_teacher": {
        "arch": "statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "teacher",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 32,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.70,
        "waveform_atlas_logit_mix": 0.18,
        "lr": 3.2e-4,
        "class_weight": [1.00, 1.45, 2.85],
        "focal_gamma": 0.95,
        "aux_weight": 1.60,
        "core_aux_weight": 2.80,
        "atlas_distill_weight": 1.25,
        "bad_aux_weight": 0.64,
        "bad_aux_pos_weight": 3.0,
        "bad_logit_mix": 0.42,
        "neighbor_weight": 0.82,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.68,
        "recon_weight": 0.0,
        "augment_strength": 0.24,
        "bad_outlier_aug_strength": 1.15,
        "bad_stress_shell_strength": 1.35,
        "seed": 20260921,
    },
    "featurefusion_stressbank_teacher": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "teacher",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 36,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.48,
        "waveform_atlas_temp": 0.64,
        "waveform_atlas_logit_mix": 0.24,
        "lr": 3.1e-4,
        "class_weight": [1.00, 1.43, 3.15],
        "focal_gamma": 1.02,
        "aux_weight": 1.70,
        "core_aux_weight": 2.90,
        "atlas_distill_weight": 1.30,
        "bad_aux_weight": 0.84,
        "bad_aux_pos_weight": 3.6,
        "bad_logit_mix": 0.56,
        "neighbor_weight": 0.86,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.20,
        "rank_weight": 0.72,
        "recon_weight": 0.0,
        "augment_strength": 0.28,
        "bad_outlier_aug_strength": 1.30,
        "bad_stress_shell_strength": 1.80,
        "seed": 20260922,
    },
    "featurefusion_multiscale_core": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 36,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.50,
        "waveform_atlas_temp": 0.66,
        "waveform_atlas_logit_mix": 0.22,
        "lr": 2.9e-4,
        "class_weight": [1.00, 1.44, 3.05],
        "focal_gamma": 1.00,
        "aux_weight": 1.70,
        "core_aux_weight": 3.10,
        "atlas_distill_weight": 1.35,
        "bad_aux_weight": 0.78,
        "bad_aux_pos_weight": 3.4,
        "bad_logit_mix": 0.54,
        "neighbor_weight": 0.88,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.20,
        "rank_weight": 0.76,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.70,
        "seed": 20260923,
    },
    "gated_bad_statfed_teacher": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 36,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.68,
        "waveform_atlas_logit_mix": 0.12,
        "final_mode": "gated_bad",
        "bad_gate_bias_init": -1.25,
        "gm_pair_weight": 0.72,
        "lr": 3.1e-4,
        "class_weight": [1.00, 1.38, 2.70],
        "focal_gamma": 0.95,
        "aux_weight": 1.50,
        "core_aux_weight": 2.75,
        "atlas_distill_weight": 1.20,
        "bad_aux_weight": 0.86,
        "bad_aux_pos_weight": 3.8,
        "bad_specificity_weight": 0.80,
        "bad_specificity_margin": 0.35,
        "neighbor_weight": 0.82,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.68,
        "recon_weight": 0.0,
        "augment_strength": 0.26,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.75,
        "seed": 20260931,
    },
    "gated_bad_multiscale_teacher": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 36,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.46,
        "waveform_atlas_temp": 0.68,
        "waveform_atlas_logit_mix": 0.12,
        "final_mode": "gated_bad",
        "bad_gate_bias_init": -1.35,
        "gm_pair_weight": 0.76,
        "lr": 2.9e-4,
        "class_weight": [1.00, 1.40, 2.85],
        "focal_gamma": 0.98,
        "aux_weight": 1.56,
        "core_aux_weight": 2.95,
        "atlas_distill_weight": 1.25,
        "bad_aux_weight": 0.92,
        "bad_aux_pos_weight": 4.1,
        "bad_specificity_weight": 0.90,
        "bad_specificity_margin": 0.40,
        "neighbor_weight": 0.86,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.20,
        "rank_weight": 0.72,
        "recon_weight": 0.0,
        "augment_strength": 0.28,
        "bad_outlier_aug_strength": 1.35,
        "bad_stress_shell_strength": 1.90,
        "seed": 20260932,
    },
    "gated_bad_statfed_stablemix": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 36,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.36,
        "waveform_atlas_temp": 0.72,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.35,
        "bad_gate_bias_init": -1.45,
        "gm_pair_weight": 0.56,
        "lr": 2.1e-4,
        "class_weight": [1.00, 1.34, 2.35],
        "focal_gamma": 0.82,
        "aux_weight": 1.10,
        "core_aux_weight": 2.05,
        "atlas_distill_weight": 0.88,
        "bad_aux_weight": 0.64,
        "bad_aux_pos_weight": 3.0,
        "bad_specificity_weight": 1.05,
        "bad_specificity_margin": 0.45,
        "neighbor_weight": 0.62,
        "neighbor_tau": 0.34,
        "supcon_weight": 0.14,
        "rank_weight": 0.46,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "bad_outlier_aug_strength": 0.75,
        "bad_stress_shell_strength": 0.95,
        "seed": 20260933,
    },
    "gated_bad_multiscale_stablemix": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 36,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.38,
        "waveform_atlas_temp": 0.72,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.35,
        "bad_gate_bias_init": -1.55,
        "gm_pair_weight": 0.58,
        "lr": 2.0e-4,
        "class_weight": [1.00, 1.36, 2.45],
        "focal_gamma": 0.86,
        "aux_weight": 1.14,
        "core_aux_weight": 2.15,
        "atlas_distill_weight": 0.92,
        "bad_aux_weight": 0.68,
        "bad_aux_pos_weight": 3.1,
        "bad_specificity_weight": 1.10,
        "bad_specificity_margin": 0.45,
        "neighbor_weight": 0.64,
        "neighbor_tau": 0.34,
        "supcon_weight": 0.14,
        "rank_weight": 0.48,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "bad_outlier_aug_strength": 0.80,
        "bad_stress_shell_strength": 1.00,
        "seed": 20260934,
    },
    "gated_bad_flatcontact_statfed": {
        "arch": "stresstoken_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 40,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.70,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.45,
        "bad_gate_bias_init": -1.32,
        "gm_pair_weight": 0.50,
        "lr": 2.0e-4,
        "class_weight": [1.00, 1.34, 2.95],
        "focal_gamma": 0.90,
        "aux_weight": 1.10,
        "core_aux_weight": 2.20,
        "atlas_distill_weight": 0.95,
        "bad_aux_weight": 0.82,
        "bad_aux_pos_weight": 4.0,
        "bad_specificity_weight": 1.25,
        "bad_specificity_margin": 0.50,
        "neighbor_weight": 0.66,
        "neighbor_tau": 0.34,
        "supcon_weight": 0.14,
        "rank_weight": 0.50,
        "recon_weight": 0.0,
        "augment_strength": 0.16,
        "bad_outlier_aug_strength": 1.05,
        "bad_stress_shell_strength": 1.75,
        "seed": 20260935,
    },
    "gated_bad_flatcontact_multiscale": {
        "arch": "stresstoken_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 18,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 320,
        "waveform_atlas_prototypes_per_class": 40,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.44,
        "waveform_atlas_temp": 0.70,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.45,
        "bad_gate_bias_init": -1.36,
        "gm_pair_weight": 0.52,
        "lr": 1.9e-4,
        "class_weight": [1.00, 1.36, 3.10],
        "focal_gamma": 0.94,
        "aux_weight": 1.14,
        "core_aux_weight": 2.30,
        "atlas_distill_weight": 1.00,
        "bad_aux_weight": 0.88,
        "bad_aux_pos_weight": 4.2,
        "bad_specificity_weight": 1.35,
        "bad_specificity_margin": 0.52,
        "neighbor_weight": 0.68,
        "neighbor_tau": 0.34,
        "supcon_weight": 0.14,
        "rank_weight": 0.52,
        "recon_weight": 0.0,
        "augment_strength": 0.16,
        "bad_outlier_aug_strength": 1.15,
        "bad_stress_shell_strength": 1.90,
        "seed": 20260936,
    },
    "hardfeat_statfed_balanced": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 20,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 40,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.36,
        "waveform_atlas_temp": 0.70,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.32,
        "bad_gate_bias_init": -1.45,
        "gm_pair_weight": 0.58,
        "lr": 1.9e-4,
        "class_weight": [1.00, 1.38, 2.45],
        "focal_gamma": 0.86,
        "aux_weight": 0.90,
        "core_aux_weight": 1.80,
        "hard_aux_weight": 3.20,
        "atlas_distill_weight": 0.95,
        "bad_aux_weight": 0.70,
        "bad_aux_pos_weight": 3.2,
        "bad_specificity_weight": 1.15,
        "bad_specificity_margin": 0.48,
        "neighbor_weight": 0.66,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.58,
        "recon_weight": 0.0,
        "augment_strength": 0.20,
        "bad_outlier_aug_strength": 0.85,
        "bad_stress_shell_strength": 1.15,
        "seed": 20260937,
    },
    "hardfeat_multiscale_badguard": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 20,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 40,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.38,
        "waveform_atlas_temp": 0.70,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.36,
        "bad_gate_bias_init": -1.40,
        "gm_pair_weight": 0.56,
        "lr": 1.8e-4,
        "class_weight": [1.00, 1.38, 2.85],
        "focal_gamma": 0.90,
        "aux_weight": 0.92,
        "core_aux_weight": 1.85,
        "hard_aux_weight": 3.50,
        "atlas_distill_weight": 1.00,
        "bad_aux_weight": 0.78,
        "bad_aux_pos_weight": 3.6,
        "bad_specificity_weight": 1.25,
        "bad_specificity_margin": 0.50,
        "neighbor_weight": 0.68,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.62,
        "recon_weight": 0.0,
        "augment_strength": 0.20,
        "bad_outlier_aug_strength": 0.95,
        "bad_stress_shell_strength": 1.30,
        "seed": 20260938,
    },
    "primitive_badexpert_statfed": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 20,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.75,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 42,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.34,
        "waveform_atlas_temp": 0.72,
        "waveform_atlas_logit_mix": 0.06,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.42,
        "gm_pair_weight": 0.58,
        "lr": 1.8e-4,
        "class_weight": [1.00, 1.38, 2.65],
        "focal_gamma": 0.88,
        "aux_weight": 0.88,
        "core_aux_weight": 1.75,
        "hard_aux_weight": 3.20,
        "atlas_distill_weight": 0.95,
        "bad_aux_weight": 0.80,
        "bad_aux_pos_weight": 3.8,
        "bad_specificity_weight": 1.35,
        "bad_specificity_margin": 0.52,
        "neighbor_weight": 0.66,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.62,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "bad_outlier_aug_strength": 1.00,
        "bad_stress_shell_strength": 1.35,
        "seed": 20260941,
    },
    "primitive_badexpert_multiscale": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 20,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.85,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 42,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.36,
        "waveform_atlas_temp": 0.72,
        "waveform_atlas_logit_mix": 0.06,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.36,
        "bad_gate_bias_init": -1.38,
        "gm_pair_weight": 0.56,
        "lr": 1.7e-4,
        "class_weight": [1.00, 1.38, 2.95],
        "focal_gamma": 0.92,
        "aux_weight": 0.90,
        "core_aux_weight": 1.80,
        "hard_aux_weight": 3.40,
        "atlas_distill_weight": 1.00,
        "bad_aux_weight": 0.86,
        "bad_aux_pos_weight": 4.1,
        "bad_specificity_weight": 1.45,
        "bad_specificity_margin": 0.54,
        "neighbor_weight": 0.68,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.64,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "bad_outlier_aug_strength": 1.10,
        "bad_stress_shell_strength": 1.50,
        "seed": 20260942,
    },
    "qrsbank_badexpert_balanced": {
        "arch": "stress_statfed_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 20,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.90,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 42,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.34,
        "waveform_atlas_temp": 0.72,
        "waveform_atlas_logit_mix": 0.06,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.36,
        "gm_pair_weight": 0.58,
        "lr": 1.8e-4,
        "class_weight": [1.00, 1.38, 2.75],
        "focal_gamma": 0.90,
        "aux_weight": 0.86,
        "core_aux_weight": 1.72,
        "hard_aux_weight": 3.70,
        "atlas_distill_weight": 0.94,
        "bad_aux_weight": 0.88,
        "bad_aux_pos_weight": 4.1,
        "bad_specificity_weight": 1.45,
        "bad_specificity_margin": 0.54,
        "neighbor_weight": 0.66,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.64,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "bad_outlier_aug_strength": 1.10,
        "bad_stress_shell_strength": 1.55,
        "seed": 20260943,
    },
    "qrsbank_badexpert_badguard": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 20,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.05,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 42,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.36,
        "waveform_atlas_temp": 0.72,
        "waveform_atlas_logit_mix": 0.06,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.36,
        "bad_gate_bias_init": -1.28,
        "gm_pair_weight": 0.56,
        "lr": 1.7e-4,
        "class_weight": [1.00, 1.38, 3.15],
        "focal_gamma": 0.94,
        "aux_weight": 0.88,
        "core_aux_weight": 1.78,
        "hard_aux_weight": 3.90,
        "atlas_distill_weight": 1.00,
        "bad_aux_weight": 0.98,
        "bad_aux_pos_weight": 4.6,
        "bad_specificity_weight": 1.65,
        "bad_specificity_margin": 0.58,
        "neighbor_weight": 0.68,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.66,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "bad_outlier_aug_strength": 1.25,
        "bad_stress_shell_strength": 1.80,
        "seed": 20260944,
    },
    "qrsbank_badstress_hardneg_balanced": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 22,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.00,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 44,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.36,
        "waveform_atlas_temp": 0.74,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.34,
        "gm_pair_weight": 0.58,
        "lr": 1.6e-4,
        "class_weight": [1.00, 1.40, 3.05],
        "focal_gamma": 0.92,
        "aux_weight": 0.86,
        "core_aux_weight": 1.75,
        "hard_aux_weight": 4.00,
        "atlas_distill_weight": 1.00,
        "bad_aux_weight": 0.92,
        "bad_aux_pos_weight": 4.4,
        "bad_specificity_weight": 1.95,
        "bad_specificity_margin": 0.62,
        "neighbor_weight": 0.70,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.68,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 1.10,
        "bad_outlier_aug_strength": 1.20,
        "bad_stress_shell_strength": 1.95,
        "seed": 20260945,
    },
    "qrsbank_badstress_hardneg_wide": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.10,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.38,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.36,
        "bad_gate_bias_init": -1.22,
        "gm_pair_weight": 0.56,
        "lr": 1.55e-4,
        "class_weight": [1.00, 1.40, 3.35],
        "focal_gamma": 0.96,
        "aux_weight": 0.86,
        "core_aux_weight": 1.75,
        "hard_aux_weight": 4.20,
        "atlas_distill_weight": 1.05,
        "bad_aux_weight": 1.05,
        "bad_aux_pos_weight": 5.0,
        "bad_specificity_weight": 2.25,
        "bad_specificity_margin": 0.68,
        "neighbor_weight": 0.72,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.72,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 1.35,
        "bad_outlier_aug_strength": 1.45,
        "bad_stress_shell_strength": 2.40,
        "seed": 20260946,
    },
    "qrsbank_stattoken_balanced": {
        "arch": "qrsbank_stat_token_teacher_atlas",
        "channels": "robust3",
        "width": 128,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 256,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.95,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.44,
        "waveform_atlas_temp": 0.74,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.38,
        "bad_gate_bias_init": -1.30,
        "gm_pair_weight": 0.62,
        "lr": 1.8e-4,
        "class_weight": [1.00, 1.42, 3.05],
        "focal_gamma": 0.92,
        "aux_weight": 0.92,
        "core_aux_weight": 1.90,
        "hard_aux_weight": 4.00,
        "atlas_distill_weight": 1.10,
        "bad_aux_weight": 0.92,
        "bad_aux_pos_weight": 4.4,
        "bad_specificity_weight": 1.55,
        "bad_specificity_margin": 0.54,
        "neighbor_weight": 0.74,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.76,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.75,
        "bad_outlier_aug_strength": 1.20,
        "bad_stress_shell_strength": 1.65,
        "seed": 20260947,
    },
    "qrsbank_stattoken_badguard": {
        "arch": "qrsbank_stat_token_teacher_atlas",
        "channels": "robust3",
        "width": 128,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 256,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.10,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.46,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.40,
        "bad_gate_bias_init": -1.18,
        "gm_pair_weight": 0.58,
        "lr": 1.7e-4,
        "class_weight": [1.00, 1.42, 3.45],
        "focal_gamma": 0.96,
        "aux_weight": 0.92,
        "core_aux_weight": 1.90,
        "hard_aux_weight": 4.25,
        "atlas_distill_weight": 1.15,
        "bad_aux_weight": 1.05,
        "bad_aux_pos_weight": 5.0,
        "bad_specificity_weight": 1.90,
        "bad_specificity_margin": 0.60,
        "neighbor_weight": 0.76,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.80,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 1.35,
        "bad_stress_shell_strength": 1.95,
        "seed": 20260948,
    },
    "qrsbank_highamp_patch_balanced": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.00,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.38,
        "waveform_atlas_temp": 0.74,
        "waveform_atlas_logit_mix": 0.06,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.36,
        "bad_gate_bias_init": -1.26,
        "gm_pair_weight": 0.58,
        "lr": 1.6e-4,
        "class_weight": [1.00, 1.40, 3.25],
        "focal_gamma": 0.94,
        "aux_weight": 0.86,
        "core_aux_weight": 1.78,
        "hard_aux_weight": 4.10,
        "atlas_distill_weight": 1.05,
        "bad_aux_weight": 1.00,
        "bad_aux_pos_weight": 4.8,
        "bad_specificity_weight": 1.75,
        "bad_specificity_margin": 0.58,
        "neighbor_weight": 0.70,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.70,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.70,
        "bad_outlier_aug_strength": 0.95,
        "bad_stress_shell_strength": 1.20,
        "bad_highamp_stress_strength": 1.65,
        "seed": 20260949,
    },
    "qrsbank_highamp_stattoken_badguard": {
        "arch": "qrsbank_stat_token_teacher_atlas",
        "channels": "robust3",
        "width": 128,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 256,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.10,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.46,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.40,
        "bad_gate_bias_init": -1.12,
        "gm_pair_weight": 0.58,
        "lr": 1.65e-4,
        "class_weight": [1.00, 1.42, 3.55],
        "focal_gamma": 0.98,
        "aux_weight": 0.92,
        "core_aux_weight": 1.92,
        "hard_aux_weight": 4.35,
        "atlas_distill_weight": 1.18,
        "bad_aux_weight": 1.10,
        "bad_aux_pos_weight": 5.2,
        "bad_specificity_weight": 1.95,
        "bad_specificity_margin": 0.60,
        "neighbor_weight": 0.78,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.84,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.75,
        "bad_outlier_aug_strength": 1.00,
        "bad_stress_shell_strength": 1.25,
        "bad_highamp_stress_strength": 1.90,
        "seed": 20260950,
    },
    "qrsbank_highamp_mix_patch_guard": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.92,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.38,
        "waveform_atlas_temp": 0.74,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.34,
        "gm_pair_weight": 0.58,
        "lr": 1.6e-4,
        "class_weight": [1.00, 1.40, 3.05],
        "focal_gamma": 0.94,
        "aux_weight": 0.86,
        "core_aux_weight": 1.78,
        "hard_aux_weight": 4.10,
        "atlas_distill_weight": 1.05,
        "bad_aux_weight": 1.00,
        "bad_aux_pos_weight": 4.8,
        "bad_specificity_weight": 2.05,
        "bad_specificity_margin": 0.62,
        "neighbor_weight": 0.70,
        "neighbor_tau": 0.32,
        "supcon_weight": 0.16,
        "rank_weight": 0.70,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.05,
        "bad_highamp_stress_strength": 1.15,
        "bad_highamp_prob": 0.28,
        "seed": 20260951,
    },
    "qrsbank_highamp_mix_stattoken_guard": {
        "arch": "qrsbank_stat_token_teacher_atlas",
        "channels": "robust3",
        "width": 128,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 256,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.98,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.46,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.07,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.36,
        "bad_gate_bias_init": -1.24,
        "gm_pair_weight": 0.58,
        "lr": 1.65e-4,
        "class_weight": [1.00, 1.42, 3.25],
        "focal_gamma": 0.98,
        "aux_weight": 0.92,
        "core_aux_weight": 1.92,
        "hard_aux_weight": 4.35,
        "atlas_distill_weight": 1.18,
        "bad_aux_weight": 1.10,
        "bad_aux_pos_weight": 5.0,
        "bad_specificity_weight": 2.20,
        "bad_specificity_margin": 0.63,
        "neighbor_weight": 0.78,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.84,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.00,
        "bad_highamp_stress_strength": 1.25,
        "bad_highamp_prob": 0.25,
        "seed": 20260952,
    },
    "qrsbank_predmargin_badguard_patch": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.92,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.38,
        "waveform_atlas_temp": 0.74,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.32,
        "bad_gate_bias_init": -1.46,
        "bad_margin_guard": True,
        "bad_margin_alpha": 0.34,
        "bad_purity_alpha": 0.18,
        "bad_boundary_alpha": 0.10,
        "gm_pair_weight": 0.58,
        "lr": 1.55e-4,
        "class_weight": [1.00, 1.40, 2.95],
        "focal_gamma": 0.94,
        "aux_weight": 0.88,
        "core_aux_weight": 1.88,
        "hard_aux_weight": 4.35,
        "atlas_distill_weight": 1.10,
        "bad_aux_weight": 1.02,
        "bad_aux_pos_weight": 4.8,
        "bad_specificity_weight": 2.35,
        "bad_specificity_margin": 0.66,
        "neighbor_weight": 0.74,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.78,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.90,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.05,
        "bad_highamp_stress_strength": 1.15,
        "bad_highamp_prob": 0.28,
        "seed": 20260953,
    },
    "qrsbank_predmargin_badguard_stattoken": {
        "arch": "qrsbank_stat_token_teacher_atlas",
        "channels": "robust3",
        "width": 128,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "stat_hidden": 256,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.98,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.46,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.07,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.34,
        "bad_margin_guard": True,
        "bad_margin_alpha": 0.32,
        "bad_purity_alpha": 0.18,
        "bad_boundary_alpha": 0.10,
        "gm_pair_weight": 0.58,
        "lr": 1.6e-4,
        "class_weight": [1.00, 1.42, 3.10],
        "focal_gamma": 0.98,
        "aux_weight": 0.94,
        "core_aux_weight": 2.00,
        "hard_aux_weight": 4.50,
        "atlas_distill_weight": 1.20,
        "bad_aux_weight": 1.12,
        "bad_aux_pos_weight": 5.0,
        "bad_specificity_weight": 2.40,
        "bad_specificity_margin": 0.66,
        "neighbor_weight": 0.80,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.88,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.90,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.00,
        "bad_highamp_stress_strength": 1.20,
        "bad_highamp_prob": 0.25,
        "seed": 20260954,
    },
    "qrsbank_highamp_mix_patch_stressselect": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "core",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.94,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.40,
        "waveform_atlas_temp": 0.74,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.34,
        "gm_pair_weight": 0.58,
        "lr": 1.55e-4,
        "class_weight": [1.00, 1.40, 3.10],
        "focal_gamma": 0.94,
        "aux_weight": 0.88,
        "core_aux_weight": 1.86,
        "hard_aux_weight": 4.30,
        "atlas_distill_weight": 1.10,
        "bad_aux_weight": 1.04,
        "bad_aux_pos_weight": 4.9,
        "bad_specificity_weight": 2.05,
        "bad_specificity_margin": 0.62,
        "neighbor_weight": 0.74,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.18,
        "rank_weight": 0.80,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.05,
        "bad_highamp_stress_strength": 1.15,
        "bad_highamp_prob": 0.28,
        "selection_bad_stress_weight": 0.28,
        "selection_stress_nonbad_hardneg_strength": 0.70,
        "selection_stress_bad_outlier_aug_strength": 0.90,
        "selection_stress_bad_stress_shell_strength": 1.10,
        "selection_stress_bad_highamp_strength": 1.20,
        "selection_stress_bad_highamp_prob": 0.40,
        "seed": 20260955,
    },
    "qrsbank_top14_focus_patch": {
        "arch": "stress_multiscale_stat_patch_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.92,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.44,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.06,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.34,
        "gm_pair_weight": 0.60,
        "lr": 1.45e-4,
        "class_weight": [1.00, 1.44, 3.20],
        "focal_gamma": 0.96,
        "aux_weight": 0.78,
        "core_aux_weight": 1.70,
        "hard_aux_weight": 3.70,
        "top14_aux_weight": 3.20,
        "atlas_distill_weight": 1.12,
        "bad_aux_weight": 1.10,
        "bad_aux_pos_weight": 5.2,
        "bad_specificity_weight": 2.10,
        "bad_specificity_margin": 0.62,
        "neighbor_weight": 0.78,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.20,
        "rank_weight": 0.84,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.05,
        "bad_highamp_stress_strength": 1.15,
        "bad_highamp_prob": 0.28,
        "seed": 20260956,
    },
    "qrsbank_top14_focus_stattoken": {
        "arch": "qrsbank_stat_token_teacher_atlas",
        "channels": "robust3",
        "width": 112,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.94,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.06,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.34,
        "gm_pair_weight": 0.60,
        "lr": 1.40e-4,
        "class_weight": [1.00, 1.44, 3.25],
        "focal_gamma": 0.96,
        "aux_weight": 0.78,
        "core_aux_weight": 1.70,
        "hard_aux_weight": 3.70,
        "top14_aux_weight": 3.25,
        "atlas_distill_weight": 1.12,
        "bad_aux_weight": 1.12,
        "bad_aux_pos_weight": 5.3,
        "bad_specificity_weight": 2.15,
        "bad_specificity_margin": 0.62,
        "neighbor_weight": 0.78,
        "neighbor_tau": 0.30,
        "supcon_weight": 0.20,
        "rank_weight": 0.84,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.05,
        "bad_highamp_stress_strength": 1.15,
        "bad_highamp_prob": 0.28,
        "seed": 20260957,
    },
    "sqiquery_top14_boundary": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.88,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.32,
        "bad_gate_bias_init": -1.38,
        "gm_pair_weight": 0.68,
        "lr": 1.35e-4,
        "class_weight": [1.00, 1.48, 3.15],
        "focal_gamma": 0.96,
        "aux_weight": 0.74,
        "core_aux_weight": 1.85,
        "hard_aux_weight": 4.20,
        "top14_aux_weight": 4.20,
        "atlas_distill_weight": 1.15,
        "bad_aux_weight": 1.05,
        "bad_aux_pos_weight": 5.1,
        "bad_specificity_weight": 2.25,
        "bad_specificity_margin": 0.64,
        "neighbor_weight": 0.86,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.92,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.05,
        "bad_highamp_stress_strength": 1.15,
        "bad_highamp_prob": 0.28,
        "seed": 20260958,
    },
    "sqiquery_top14_badguard": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.94,
        "wide_aux_head": True,
        "aux_hidden": 384,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.32,
        "gm_pair_weight": 0.62,
        "lr": 1.35e-4,
        "class_weight": [1.00, 1.44, 3.45],
        "focal_gamma": 0.98,
        "aux_weight": 0.74,
        "core_aux_weight": 1.80,
        "hard_aux_weight": 4.10,
        "top14_aux_weight": 4.00,
        "atlas_distill_weight": 1.12,
        "bad_aux_weight": 1.22,
        "bad_aux_pos_weight": 5.8,
        "bad_specificity_weight": 2.45,
        "bad_specificity_margin": 0.66,
        "neighbor_weight": 0.82,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.90,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.90,
        "bad_outlier_aug_strength": 0.95,
        "bad_stress_shell_strength": 1.10,
        "bad_highamp_stress_strength": 1.20,
        "bad_highamp_prob": 0.34,
        "seed": 20260959,
    },
    "sqiquery_auxprimitive_boundary": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "aux_primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.88,
        "wide_aux_head": True,
        "aux_hidden": 448,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.32,
        "bad_gate_bias_init": -1.38,
        "gm_pair_weight": 0.68,
        "lr": 1.30e-4,
        "class_weight": [1.00, 1.48, 3.15],
        "focal_gamma": 0.96,
        "aux_weight": 0.68,
        "core_aux_weight": 1.70,
        "hard_aux_weight": 4.60,
        "top14_aux_weight": 4.80,
        "atlas_distill_weight": 1.15,
        "bad_aux_weight": 1.05,
        "bad_aux_pos_weight": 5.1,
        "bad_specificity_weight": 2.25,
        "bad_specificity_margin": 0.64,
        "neighbor_weight": 0.86,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.92,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.85,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 1.05,
        "bad_highamp_stress_strength": 1.15,
        "bad_highamp_prob": 0.28,
        "seed": 20260960,
    },
    "sqiquery_auxprimitive_badguard": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "aux_primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.94,
        "wide_aux_head": True,
        "aux_hidden": 448,
        "waveform_atlas_prototypes_per_class": 48,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.42,
        "waveform_atlas_temp": 0.76,
        "waveform_atlas_logit_mix": 0.05,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.32,
        "gm_pair_weight": 0.62,
        "lr": 1.30e-4,
        "class_weight": [1.00, 1.44, 3.45],
        "focal_gamma": 0.98,
        "aux_weight": 0.68,
        "core_aux_weight": 1.70,
        "hard_aux_weight": 4.55,
        "top14_aux_weight": 4.65,
        "atlas_distill_weight": 1.12,
        "bad_aux_weight": 1.22,
        "bad_aux_pos_weight": 5.8,
        "bad_specificity_weight": 2.45,
        "bad_specificity_margin": 0.66,
        "neighbor_weight": 0.82,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.90,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 0.90,
        "bad_outlier_aug_strength": 0.95,
        "bad_stress_shell_strength": 1.10,
        "bad_highamp_stress_strength": 1.20,
        "bad_highamp_prob": 0.34,
        "seed": 20260961,
    },
    "sqiquery_intermittentbad_balanced": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "aux_primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 0.96,
        "wide_aux_head": True,
        "aux_hidden": 448,
        "waveform_atlas_prototypes_per_class": 56,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.48,
        "waveform_atlas_temp": 0.70,
        "waveform_atlas_logit_mix": 0.07,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.36,
        "bad_gate_bias_init": -1.22,
        "gm_pair_weight": 0.62,
        "lr": 1.25e-4,
        "class_weight": [1.00, 1.45, 3.75],
        "focal_gamma": 1.00,
        "aux_weight": 0.68,
        "core_aux_weight": 1.70,
        "hard_aux_weight": 4.70,
        "top14_aux_weight": 4.75,
        "atlas_distill_weight": 1.12,
        "bad_aux_weight": 1.35,
        "bad_aux_pos_weight": 6.4,
        "bad_specificity_weight": 2.65,
        "bad_specificity_margin": 0.70,
        "neighbor_weight": 0.82,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.92,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 1.15,
        "bad_outlier_aug_strength": 0.90,
        "bad_stress_shell_strength": 0.85,
        "bad_highamp_stress_strength": 0.85,
        "bad_highamp_prob": 0.25,
        "bad_intermittent_contact_strength": 1.35,
        "seed": 20260962,
    },
    "sqiquery_intermittentbad_stress": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "aux_primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.04,
        "wide_aux_head": True,
        "aux_hidden": 448,
        "waveform_atlas_prototypes_per_class": 56,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.52,
        "waveform_atlas_temp": 0.68,
        "waveform_atlas_logit_mix": 0.09,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.40,
        "bad_gate_bias_init": -1.10,
        "gm_pair_weight": 0.58,
        "lr": 1.22e-4,
        "class_weight": [1.00, 1.42, 4.15],
        "focal_gamma": 1.05,
        "aux_weight": 0.66,
        "core_aux_weight": 1.65,
        "hard_aux_weight": 4.90,
        "top14_aux_weight": 4.85,
        "atlas_distill_weight": 1.08,
        "bad_aux_weight": 1.55,
        "bad_aux_pos_weight": 7.2,
        "bad_specificity_weight": 3.10,
        "bad_specificity_margin": 0.76,
        "neighbor_weight": 0.78,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.95,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 1.30,
        "bad_outlier_aug_strength": 0.85,
        "bad_stress_shell_strength": 0.80,
        "bad_highamp_stress_strength": 0.75,
        "bad_highamp_prob": 0.22,
        "bad_intermittent_contact_strength": 1.85,
        "seed": 20260963,
    },
    "sqiquery_intermitbad_auxmask_balanced": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "aux_primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.00,
        "wide_aux_head": True,
        "aux_hidden": 448,
        "waveform_atlas_prototypes_per_class": 56,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.50,
        "waveform_atlas_temp": 0.68,
        "waveform_atlas_logit_mix": 0.08,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.34,
        "bad_gate_bias_init": -1.18,
        "gm_pair_weight": 0.60,
        "lr": 1.20e-4,
        "class_weight": [1.00, 1.44, 3.90],
        "focal_gamma": 1.03,
        "aux_weight": 0.66,
        "core_aux_weight": 1.60,
        "hard_aux_weight": 4.55,
        "top14_aux_weight": 4.65,
        "atlas_distill_weight": 1.06,
        "bad_aux_weight": 1.45,
        "bad_aux_pos_weight": 6.8,
        "bad_specificity_weight": 3.35,
        "bad_specificity_margin": 0.78,
        "neighbor_weight": 0.78,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.92,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 1.45,
        "bad_outlier_aug_strength": 0.80,
        "bad_stress_shell_strength": 0.75,
        "bad_highamp_stress_strength": 0.70,
        "bad_highamp_prob": 0.20,
        "bad_intermittent_contact_strength": 1.85,
        "aug_aux_weight_bad": 0.08,
        "aug_aux_weight_nonbad": 0.55,
        "seed": 20260964,
    },
    "sqiquery_intermitbad_auxmask_stress": {
        "arch": "sqi_query_multiscale_patch_teacher_atlas",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "scales": [25, 50, 100, 250, 500],
        "stat_hidden": 224,
        "teacher_prototypes_per_class": 24,
        "teacher_prototype_mode": "farthest",
        "predicted_feature_fusion": "top14",
        "primitive_bank": "qrs_enhanced",
        "primitive_fusion": True,
        "aux_primitive_fusion": True,
        "primitive_bad_expert": True,
        "primitive_bad_mix": 1.10,
        "wide_aux_head": True,
        "aux_hidden": 448,
        "waveform_atlas_prototypes_per_class": 56,
        "waveform_atlas_mode": "farthest",
        "waveform_proto_weight": 0.52,
        "waveform_atlas_temp": 0.66,
        "waveform_atlas_logit_mix": 0.09,
        "final_mode": "gated_bad_mix",
        "gate_logit_alpha": 0.40,
        "bad_gate_bias_init": -1.12,
        "gm_pair_weight": 0.56,
        "lr": 1.18e-4,
        "class_weight": [1.00, 1.42, 4.35],
        "focal_gamma": 1.08,
        "aux_weight": 0.64,
        "core_aux_weight": 1.55,
        "hard_aux_weight": 4.60,
        "top14_aux_weight": 4.70,
        "atlas_distill_weight": 1.04,
        "bad_aux_weight": 1.72,
        "bad_aux_pos_weight": 7.6,
        "bad_specificity_weight": 3.85,
        "bad_specificity_margin": 0.84,
        "neighbor_weight": 0.74,
        "neighbor_tau": 0.28,
        "supcon_weight": 0.22,
        "rank_weight": 0.92,
        "recon_weight": 0.0,
        "augment_strength": 0.18,
        "nonbad_hardneg_strength": 1.65,
        "bad_outlier_aug_strength": 0.80,
        "bad_stress_shell_strength": 0.80,
        "bad_highamp_stress_strength": 0.70,
        "bad_highamp_prob": 0.22,
        "bad_intermittent_contact_strength": 2.25,
        "aug_aux_weight_bad": 0.06,
        "aug_aux_weight_nonbad": 0.60,
        "seed": 20260965,
    },
}


CANDIDATES["sqiquery_qrsdetector_auxmask_balanced"] = {
    **CANDIDATES["sqiquery_intermitbad_auxmask_balanced"],
    "primitive_bank": "qrs_enhanced_v2",
    "primitive_bad_mix": 1.08,
    "bad_aux_weight": 1.58,
    "bad_specificity_weight": 3.55,
    "bad_specificity_margin": 0.80,
    "nonbad_hardneg_strength": 1.55,
    "bad_intermittent_contact_strength": 2.00,
    "seed": 20260966,
}

CANDIDATES["sqiquery_qrsdetector_auxmask_stress"] = {
    **CANDIDATES["sqiquery_intermitbad_auxmask_stress"],
    "primitive_bank": "qrs_enhanced_v2",
    "primitive_bad_mix": 1.20,
    "bad_aux_weight": 1.86,
    "bad_specificity_weight": 4.10,
    "bad_specificity_margin": 0.86,
    "nonbad_hardneg_strength": 1.75,
    "bad_intermittent_contact_strength": 2.45,
    "seed": 20260967,
}

CANDIDATES["sqiquery_nonbad2badstress_balanced"] = {
    **CANDIDATES["sqiquery_intermitbad_auxmask_balanced"],
    "primitive_bad_mix": 1.06,
    "bad_aux_weight": 1.66,
    "bad_specificity_weight": 3.65,
    "bad_specificity_margin": 0.82,
    "nonbad_hardneg_strength": 1.70,
    "nonbad_to_bad_stress_prob": 0.08,
    "nonbad_to_bad_stress_strength": 2.25,
    "aug_aux_weight_bad": 0.04,
    "aug_aux_weight_nonbad": 0.62,
    "seed": 20260968,
}

CANDIDATES["sqiquery_nonbad2badstress_stress"] = {
    **CANDIDATES["sqiquery_intermitbad_auxmask_stress"],
    "primitive_bad_mix": 1.14,
    "bad_aux_weight": 1.92,
    "bad_specificity_weight": 4.35,
    "bad_specificity_margin": 0.88,
    "nonbad_hardneg_strength": 1.90,
    "nonbad_to_bad_stress_prob": 0.14,
    "nonbad_to_bad_stress_strength": 2.55,
    "aug_aux_weight_bad": 0.03,
    "aug_aux_weight_nonbad": 0.66,
    "seed": 20260969,
}

TOP20_INTERPRETABLE_FEATURE_COLUMNS = [
    "pca_margin",
    "sample_entropy_proxy",
    "flatline_ratio",
    "pc1",
    "higuchi_fd_proxy",
    "non_qrs_diff_p95",
    "diff_zero_crossing_rate",
    "zero_crossing_rate",
    "qrs_prom_p90",
    "sqi_fSQI",
    "amplitude_entropy",
    "sqi_kSQI",
    "sqi_sSQI",
    "low_amp_ratio",
    "non_qrs_rms_ratio",
    "baseline_step",
    "ptp_p99_p01",
    "band_30_45",
    "qrs_visibility",
    "band_15_30",
]

WAVEFORM_COMPUTABLE_TEACHER_COLUMNS = [
    "sample_entropy_proxy",
    "flatline_ratio",
    "higuchi_fd_proxy",
    "non_qrs_diff_p95",
    "diff_zero_crossing_rate",
    "zero_crossing_rate",
    "qrs_prom_p90",
    "sqi_fSQI",
    "amplitude_entropy",
    "sqi_kSQI",
    "sqi_sSQI",
    "low_amp_ratio",
    "non_qrs_rms_ratio",
    "baseline_step",
    "ptp_p99_p01",
    "band_30_45",
    "qrs_visibility",
    "band_15_30",
    "sqi_basSQI",
    "detector_agreement",
    "qrs_band_ratio",
    "template_corr",
    "mean_abs",
    "rms",
]

WAVEFORM_HARD_TEACHER_COLUMNS = [
    "flatline_ratio",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "non_qrs_diff_p95",
    "qrs_visibility",
    "qrs_visibility",
    "detector_agreement",
    "detector_agreement",
    "baseline_step",
    "baseline_step",
    "sqi_basSQI",
    "sqi_basSQI",
    "qrs_prom_p90",
    "band_15_30",
    "qrs_band_ratio",
    "template_corr",
]

TOP25_QRS_GEOMETRY_FEATURE_COLUMNS = TOP20_INTERPRETABLE_FEATURE_COLUMNS + [
    "detector_agreement",
    "boundary_confidence",
    "knn_label_purity",
    "pc2",
    "pc3",
]

P20_FAILURE_AXIS_FEATURE_COLUMNS = [
    "pc1",
    "pc2",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "sqi_basSQI",
    "qrs_prom_p90",
    "qrs_band_ratio",
    "band_15_30",
    "sample_entropy_proxy",
]

TOP23_HARD_FEATURE_COLUMNS = TOP20_INTERPRETABLE_FEATURE_COLUMNS + [
    "detector_agreement",
    "boundary_confidence",
    "knn_label_purity",
]

CANDIDATES["predtop20_qrsbank_patch_balanced"] = {
    **CANDIDATES["qrsbank_top14_focus_patch"],
    "predicted_feature_fusion": "configured",
    "predicted_feature_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "aux_pretrain_epochs": 3,
    "pretrain_classification_weight": 0.12,
    "pretrain_aux_weight": 1.20,
    "pretrain_core_aux_weight": 2.20,
    "pretrain_hard_aux_weight": 4.60,
    "pretrain_top14_aux_weight": 4.40,
    "pretrain_critical_aux_weight": 5.20,
    "pretrain_rank_weight": 1.10,
    "class_weight": [1.00, 1.42, 3.05],
    "bad_aux_weight": 0.92,
    "bad_aux_pos_weight": 4.6,
    "bad_specificity_weight": 2.35,
    "bad_specificity_margin": 0.68,
    "seed": 20260970,
}

CANDIDATES["predtop20_sqiquery_boundary"] = {
    **CANDIDATES["sqiquery_auxprimitive_boundary"],
    "predicted_feature_fusion": "configured",
    "predicted_feature_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "aux_pretrain_epochs": 3,
    "pretrain_classification_weight": 0.10,
    "pretrain_aux_weight": 1.15,
    "pretrain_core_aux_weight": 2.20,
    "pretrain_hard_aux_weight": 5.00,
    "pretrain_top14_aux_weight": 5.20,
    "pretrain_critical_aux_weight": 5.60,
    "pretrain_rank_weight": 1.18,
    "class_weight": [1.00, 1.46, 3.00],
    "bad_aux_weight": 0.90,
    "bad_aux_pos_weight": 4.4,
    "bad_specificity_weight": 2.55,
    "bad_specificity_margin": 0.70,
    "seed": 20260971,
}

CANDIDATES["predtop20_qrsbank_patch_pretrain"] = {
    **CANDIDATES["predtop20_qrsbank_patch_balanced"],
    "seed": 20260972,
}

CANDIDATES["predtop20_sqiquery_boundary_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary"],
    "seed": 20260973,
}

CANDIDATES["predtop20_sqiquery_goodguard_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "class_weight": [1.36, 1.16, 2.95],
    "gm_pair_weight": 0.64,
    "bad_aux_weight": 0.82,
    "bad_specificity_weight": 2.25,
    "bad_specificity_margin": 0.68,
    "pretrain_classification_weight": 0.14,
    "seed": 20260975,
}

CANDIDATES["predtop20_sqiquery_balancedguard_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "class_weight": [1.22, 1.28, 3.08],
    "gm_pair_weight": 0.72,
    "bad_aux_weight": 0.86,
    "bad_specificity_weight": 2.35,
    "bad_specificity_margin": 0.70,
    "pretrain_classification_weight": 0.13,
    "seed": 20260976,
}

CANDIDATES["predtop20_sqiquery_badguardlite_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "class_weight": [1.24, 1.18, 3.35],
    "gm_pair_weight": 0.68,
    "bad_aux_weight": 0.96,
    "bad_aux_pos_weight": 5.0,
    "bad_specificity_weight": 2.75,
    "bad_specificity_margin": 0.74,
    "pretrain_classification_weight": 0.12,
    "seed": 20260977,
}

CANDIDATES["predtop20_sqiquery_lowfreqbad_balanced_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "class_weight": [1.08, 1.34, 3.15],
    "gm_pair_weight": 0.70,
    "bad_aux_weight": 1.02,
    "bad_aux_pos_weight": 4.8,
    "bad_specificity_weight": 2.40,
    "bad_specificity_margin": 0.70,
    "nonbad_hardneg_strength": 0.80,
    "bad_outlier_aug_strength": 0.90,
    "bad_stress_shell_strength": 1.35,
    "bad_highamp_stress_strength": 1.10,
    "bad_highamp_prob": 0.35,
    "selection_bad_stress_weight": 0.24,
    "selection_stress_nonbad_hardneg_strength": 0.70,
    "selection_stress_bad_outlier_aug_strength": 0.90,
    "selection_stress_bad_stress_shell_strength": 1.45,
    "selection_stress_bad_highamp_strength": 1.15,
    "selection_stress_bad_highamp_prob": 0.42,
    "seed": 20260978,
}

CANDIDATES["predtop20_sqiquery_lowfreqbad_stress_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "class_weight": [1.05, 1.28, 3.45],
    "gm_pair_weight": 0.66,
    "bad_aux_weight": 1.12,
    "bad_aux_pos_weight": 5.4,
    "bad_specificity_weight": 2.85,
    "bad_specificity_margin": 0.76,
    "nonbad_hardneg_strength": 1.05,
    "bad_outlier_aug_strength": 1.05,
    "bad_stress_shell_strength": 1.80,
    "bad_highamp_stress_strength": 1.35,
    "bad_highamp_prob": 0.48,
    "selection_bad_stress_weight": 0.34,
    "selection_stress_nonbad_hardneg_strength": 0.95,
    "selection_stress_bad_outlier_aug_strength": 1.00,
    "selection_stress_bad_stress_shell_strength": 1.95,
    "selection_stress_bad_highamp_strength": 1.45,
    "selection_stress_bad_highamp_prob": 0.55,
    "seed": 20260979,
}

CANDIDATES["predtop20_sqiquery_hardaux_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    # Keep the classifier input clean: only the proven top20 predicted
    # features are fused.  Add harder targets only as auxiliary supervision.
    "predicted_feature_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "baseline_step",
        "boundary_confidence",
        "knn_label_purity",
        "sqi_basSQI",
        "pc2",
        "pc3",
    ],
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.09,
    "pretrain_core_aux_weight": 2.55,
    "pretrain_hard_aux_weight": 5.80,
    "pretrain_top14_aux_weight": 5.50,
    "pretrain_critical_aux_weight": 7.40,
    "pretrain_rank_weight": 1.28,
    "class_weight": [1.00, 1.44, 3.00],
    "gm_pair_weight": 0.74,
    "bad_aux_weight": 0.92,
    "bad_specificity_weight": 2.55,
    "bad_specificity_margin": 0.70,
    "seed": 20260980,
}

CANDIDATES["predtop23_sqiquery_hardfeat_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_hardaux_pretrain"],
    # Diagnostic contrast: fuse the newly predicted detector/boundary/purity
    # features too.  If this hurts, the feature predictions are not yet stable
    # enough to enter the classifier.
    "predicted_feature_columns": TOP23_HARD_FEATURE_COLUMNS,
    "critical_aux_columns": TOP23_HARD_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "baseline_step",
        "boundary_confidence",
        "boundary_confidence",
        "knn_label_purity",
        "sqi_basSQI",
        "pc2",
        "pc3",
    ],
    "seed": 20260981,
}

CANDIDATES["predtop20_sqiquery_featuredistill_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "feature_teacher_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "feature_teacher_distill_weight": 0.95,
    "pretrain_feature_teacher_distill_weight": 1.35,
    "feature_teacher_temperature": 1.8,
    "class_weight": [1.00, 1.42, 3.05],
    "gm_pair_weight": 0.72,
    "bad_aux_weight": 0.90,
    "bad_specificity_weight": 2.45,
    "bad_specificity_margin": 0.68,
    "seed": 20260982,
}

CANDIDATES["predtop23_sqiquery_featuredistill_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "feature_teacher_columns": TOP23_HARD_FEATURE_COLUMNS,
    "feature_teacher_distill_weight": 1.05,
    "pretrain_feature_teacher_distill_weight": 1.50,
    "feature_teacher_temperature": 1.8,
    # Keep classifier fusion on the stable top20; the extra top23 columns only
    # guide the soft teacher, avoiding the direct-fusion regression seen above.
    "predicted_feature_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + ["detector_agreement", "boundary_confidence", "knn_label_purity", "qrs_visibility", "baseline_step"],
    "aux_pretrain_epochs": 4,
    "class_weight": [1.00, 1.42, 3.10],
    "gm_pair_weight": 0.70,
    "bad_aux_weight": 0.92,
    "bad_specificity_weight": 2.50,
    "bad_specificity_margin": 0.70,
    "seed": 20260983,
}

CANDIDATES["predtop20_sqiquery_qrsstressv3_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    # Same SQI-query Transformer family as the current best, but the primitive
    # branch is expanded with detector agreement and a second explicit stress
    # bank.  This tests whether bad-stress failures come from missing internal
    # observables rather than class-weight tuning.
    "primitive_bank": "qrs_stress_v3",
    "predicted_feature_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "baseline_step",
        "boundary_confidence",
        "knn_label_purity",
        "flatline_ratio",
        "flatline_ratio",
        "sqi_basSQI",
    ],
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.09,
    "pretrain_core_aux_weight": 2.55,
    "pretrain_hard_aux_weight": 5.60,
    "pretrain_top14_aux_weight": 5.45,
    "pretrain_critical_aux_weight": 7.20,
    "pretrain_rank_weight": 1.24,
    "class_weight": [1.00, 1.42, 3.10],
    "gm_pair_weight": 0.70,
    "bad_aux_weight": 0.96,
    "bad_aux_pos_weight": 5.0,
    "bad_specificity_weight": 2.60,
    "bad_specificity_margin": 0.72,
    "primitive_bad_mix": 0.92,
    "gate_logit_alpha": 0.34,
    "seed": 20260984,
}

CANDIDATES["predtop20_sqiquery_qrsstressv3_stress_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_qrsstressv3_pretrain"],
    # Slightly stronger controlled bad-stress exposure.  Keep this paired with
    # the no-extra-stress candidate so the result tells us whether coverage or
    # architecture is the limiting factor.
    "class_weight": [1.00, 1.36, 3.35],
    "gm_pair_weight": 0.66,
    "bad_aux_weight": 1.08,
    "bad_aux_pos_weight": 5.6,
    "bad_specificity_weight": 2.95,
    "bad_specificity_margin": 0.76,
    "bad_outlier_aug_strength": 1.05,
    "bad_stress_shell_strength": 1.85,
    "bad_highamp_stress_strength": 1.30,
    "bad_highamp_prob": 0.48,
    "selection_bad_stress_weight": 0.32,
    "selection_stress_bad_outlier_aug_strength": 1.00,
    "selection_stress_bad_stress_shell_strength": 1.95,
    "selection_stress_bad_highamp_strength": 1.40,
    "selection_stress_bad_highamp_prob": 0.55,
    "seed": 20260985,
}

CANDIDATES["predtop20_sqiquery_primtree_teacher_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    # Training-only waveform primitive tree teacher.  The final model still
    # receives waveform channels only; the tree is used as a soft target because
    # primitive learnability shows it recovers original bad stress much better
    # than the neural bad head.
    "primitive_teacher_bank": "qrs_enhanced",
    "primitive_teacher_distill_weight": 0.82,
    "pretrain_primitive_teacher_distill_weight": 1.18,
    "primitive_teacher_temperature": 1.9,
    "class_weight": [1.00, 1.42, 3.18],
    "gm_pair_weight": 0.70,
    "bad_aux_weight": 0.96,
    "bad_aux_pos_weight": 5.0,
    "bad_specificity_weight": 2.70,
    "bad_specificity_margin": 0.72,
    "seed": 20260986,
}

CANDIDATES["predtop20_sqiquery_primtree_stress_teacher_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_primtree_teacher_pretrain"],
    # Same distillation idea, but the teacher uses the stress-expanded primitive
    # bank and the student sees mild controlled bad-stress augmentation.
    "primitive_teacher_bank": "qrs_stress_v3",
    "primitive_teacher_distill_weight": 0.92,
    "pretrain_primitive_teacher_distill_weight": 1.30,
    "class_weight": [1.00, 1.36, 3.35],
    "gm_pair_weight": 0.66,
    "bad_aux_weight": 1.06,
    "bad_aux_pos_weight": 5.4,
    "bad_specificity_weight": 2.95,
    "bad_specificity_margin": 0.76,
    "bad_outlier_aug_strength": 1.00,
    "bad_stress_shell_strength": 1.55,
    "bad_highamp_stress_strength": 1.20,
    "bad_highamp_prob": 0.42,
    "selection_bad_stress_weight": 0.24,
    "selection_stress_bad_outlier_aug_strength": 0.95,
    "selection_stress_bad_stress_shell_strength": 1.70,
    "selection_stress_bad_highamp_strength": 1.30,
    "selection_stress_bad_highamp_prob": 0.50,
    "seed": 20260987,
}

CANDIDATES["predtop20_sqiquery_thresholdtree_balanced_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    # Waveform-only differentiable threshold branch.  It gives the Transformer
    # head tree-like split capacity over internal primitive stats, avoiding the
    # failure mode where bad-stress evidence is averaged away.
    "primitive_threshold_fusion": True,
    "primitive_threshold_values": [-2.25, -1.45, -0.75, -0.25, 0.25, 0.75, 1.45, 2.25],
    "primitive_threshold_sharpness": 2.4,
    "class_weight": [1.00, 1.42, 3.18],
    "gm_pair_weight": 0.70,
    "bad_aux_weight": 0.98,
    "bad_aux_pos_weight": 5.2,
    "bad_specificity_weight": 2.75,
    "bad_specificity_margin": 0.74,
    "pretrain_classification_weight": 0.10,
    "seed": 20260988,
}

CANDIDATES["predtop20_sqiquery_thresholdtree_badguard_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_thresholdtree_balanced_pretrain"],
    "class_weight": [1.00, 1.34, 3.45],
    "gm_pair_weight": 0.66,
    "bad_aux_weight": 1.12,
    "bad_aux_pos_weight": 5.8,
    "bad_specificity_weight": 3.10,
    "bad_specificity_margin": 0.78,
    "bad_outlier_aug_strength": 1.00,
    "bad_stress_shell_strength": 1.45,
    "bad_highamp_stress_strength": 1.15,
    "bad_highamp_prob": 0.42,
    "selection_bad_stress_weight": 0.26,
    "selection_stress_bad_outlier_aug_strength": 0.95,
    "selection_stress_bad_stress_shell_strength": 1.65,
    "selection_stress_bad_highamp_strength": 1.25,
    "selection_stress_bad_highamp_prob": 0.50,
    "seed": 20260989,
}

CANDIDATES["predtop20_eventqrs_sqiquery_balanced_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "arch": "event_qrs_sqi_query_multiscale_patch_teacher_atlas",
    "event_count": 10,
    "event_window": 91,
    "class_weight": [1.00, 1.42, 3.15],
    "gm_pair_weight": 0.70,
    "bad_aux_weight": 0.98,
    "bad_aux_pos_weight": 5.2,
    "bad_specificity_weight": 2.70,
    "bad_specificity_margin": 0.74,
    "pretrain_classification_weight": 0.10,
    "seed": 20260990,
}

CANDIDATES["predtop20_eventqrs_sqiquery_badguard_pretrain"] = {
    **CANDIDATES["predtop20_eventqrs_sqiquery_balanced_pretrain"],
    "class_weight": [1.00, 1.34, 3.42],
    "gm_pair_weight": 0.66,
    "bad_aux_weight": 1.10,
    "bad_aux_pos_weight": 5.7,
    "bad_specificity_weight": 3.05,
    "bad_specificity_margin": 0.78,
    "bad_outlier_aug_strength": 1.00,
    "bad_stress_shell_strength": 1.35,
    "bad_highamp_stress_strength": 1.10,
    "bad_highamp_prob": 0.40,
    "selection_bad_stress_weight": 0.22,
    "selection_stress_bad_outlier_aug_strength": 0.95,
    "selection_stress_bad_stress_shell_strength": 1.55,
    "selection_stress_bad_highamp_strength": 1.20,
    "selection_stress_bad_highamp_prob": 0.48,
    "seed": 20260991,
}

CANDIDATES["predtop20_eventqrs_intermitbad_balanced_pretrain"] = {
    **CANDIDATES["predtop20_eventqrs_sqiquery_balanced_pretrain"],
    # Targets the verified original-test bad-outlier shell: high diff flat /
    # contact / low-amplitude ratios, huge baseline drift, low zcr, and low
    # local detail while keeping some inconsistent QRS-like spikes.
    "class_weight": [1.00, 1.38, 3.35],
    "gm_pair_weight": 0.68,
    "bad_aux_weight": 1.08,
    "bad_aux_pos_weight": 5.6,
    "bad_specificity_weight": 2.95,
    "bad_specificity_margin": 0.76,
    "nonbad_hardneg_strength": 1.00,
    "bad_outlier_aug_strength": 0.70,
    "bad_stress_shell_strength": 0.85,
    "bad_intermittent_contact_strength": 1.55,
    "bad_highamp_stress_strength": 0.55,
    "bad_highamp_prob": 0.28,
    "selection_bad_stress_weight": 0.30,
    "selection_stress_nonbad_hardneg_strength": 0.95,
    "selection_stress_bad_outlier_aug_strength": 0.70,
    "selection_stress_bad_stress_shell_strength": 0.95,
    "selection_stress_bad_intermittent_contact_strength": 1.75,
    "selection_stress_bad_highamp_strength": 0.65,
    "selection_stress_bad_highamp_prob": 0.32,
    "seed": 20260992,
}

CANDIDATES["predtop20_eventqrs_intermitbad_stress_pretrain"] = {
    **CANDIDATES["predtop20_eventqrs_intermitbad_balanced_pretrain"],
    "class_weight": [1.00, 1.30, 3.65],
    "gm_pair_weight": 0.62,
    "bad_aux_weight": 1.22,
    "bad_aux_pos_weight": 6.4,
    "bad_specificity_weight": 3.35,
    "bad_specificity_margin": 0.82,
    "nonbad_hardneg_strength": 1.15,
    "bad_outlier_aug_strength": 0.85,
    "bad_stress_shell_strength": 1.10,
    "bad_intermittent_contact_strength": 2.10,
    "bad_highamp_stress_strength": 0.75,
    "bad_highamp_prob": 0.36,
    "selection_bad_stress_weight": 0.40,
    "selection_stress_nonbad_hardneg_strength": 1.10,
    "selection_stress_bad_outlier_aug_strength": 0.85,
    "selection_stress_bad_stress_shell_strength": 1.20,
    "selection_stress_bad_intermittent_contact_strength": 2.25,
    "selection_stress_bad_highamp_strength": 0.85,
    "selection_stress_bad_highamp_prob": 0.42,
    "seed": 20260993,
}

CANDIDATES["predtop20_eventqrs_intermitbad_lowaux_pretrain"] = {
    **CANDIDATES["predtop20_eventqrs_intermitbad_stress_pretrain"],
    # The bad-stress augmentation intentionally changes waveform geometry, so
    # using the base synthetic aux target at full weight fights the new shell.
    # Lower aux weight only for augmented bad rows; keep non-bad aux supervision
    # intact for good/medium geometry.
    "aug_aux_weight_bad": 0.12,
    "selection_stress_aug_aux_weight_bad": 0.12,
    "class_weight": [1.00, 1.32, 3.55],
    "bad_aux_weight": 1.18,
    "bad_aux_pos_weight": 6.0,
    "bad_specificity_weight": 3.10,
    "bad_specificity_margin": 0.80,
    "selection_bad_stress_weight": 0.34,
    "seed": 20260994,
}

CANDIDATES["predtop20_eventqrs_intermitbad_lowaux_specific_pretrain"] = {
    **CANDIDATES["predtop20_eventqrs_intermitbad_lowaux_pretrain"],
    # Same bad-stress shell, but less bad-class pressure and more non-bad hard
    # negatives.  Goal: move the gain from badcal thresholding into raw logits
    # without sacrificing the good/medium boundary.
    "class_weight": [1.06, 1.46, 3.05],
    "gm_pair_weight": 0.74,
    "bad_aux_weight": 0.96,
    "bad_aux_pos_weight": 4.8,
    "bad_specificity_weight": 3.85,
    "bad_specificity_margin": 0.88,
    "nonbad_hardneg_strength": 1.65,
    "bad_outlier_aug_strength": 0.75,
    "bad_stress_shell_strength": 1.00,
    "bad_intermittent_contact_strength": 1.95,
    "bad_highamp_stress_strength": 0.60,
    "bad_highamp_prob": 0.30,
    "selection_bad_stress_weight": 0.22,
    "selection_stress_nonbad_hardneg_strength": 1.55,
    "selection_stress_bad_outlier_aug_strength": 0.75,
    "selection_stress_bad_stress_shell_strength": 1.10,
    "selection_stress_bad_intermittent_contact_strength": 2.05,
    "selection_stress_bad_highamp_strength": 0.70,
    "selection_stress_bad_highamp_prob": 0.34,
    "seed": 20260995,
}

CANDIDATES["predtop20_eventqrs_intermitbad_lowaux_specific_ft"] = {
    **CANDIDATES["predtop20_eventqrs_intermitbad_lowaux_specific_pretrain"],
    "init_from_candidate": "predtop20_eventqrs_intermitbad_lowaux_pretrain",
    "lr": 7.0e-5,
    "aux_pretrain_epochs": 0,
    "classification_weight": 1.05,
    "aux_weight": 0.42,
    "core_aux_weight": 1.10,
    "hard_aux_weight": 3.00,
    "top14_aux_weight": 3.20,
    "critical_aux_weight": 3.40,
    "bad_aux_weight": 0.88,
    "bad_specificity_weight": 4.20,
    "bad_specificity_margin": 0.92,
    "selection_bad_stress_weight": 0.18,
    "seed": 20260996,
}

CANDIDATES["predtop20_sqiquery_intermitbad_lowaux_specific_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    # Ablation against event-QRS: keep the strongest original SQIQuery backbone,
    # add the corrected low-aux bad-stress shell and specificity pressure.
    "aug_aux_weight_bad": 0.12,
    "selection_stress_aug_aux_weight_bad": 0.12,
    "class_weight": [1.06, 1.46, 3.05],
    "gm_pair_weight": 0.74,
    "bad_aux_weight": 0.96,
    "bad_aux_pos_weight": 4.8,
    "bad_specificity_weight": 3.85,
    "bad_specificity_margin": 0.88,
    "nonbad_hardneg_strength": 1.65,
    "bad_outlier_aug_strength": 0.75,
    "bad_stress_shell_strength": 1.00,
    "bad_intermittent_contact_strength": 1.95,
    "bad_highamp_stress_strength": 0.60,
    "bad_highamp_prob": 0.30,
    "selection_bad_stress_weight": 0.22,
    "selection_stress_nonbad_hardneg_strength": 1.55,
    "selection_stress_bad_outlier_aug_strength": 0.75,
    "selection_stress_bad_stress_shell_strength": 1.10,
    "selection_stress_bad_intermittent_contact_strength": 2.05,
    "selection_stress_bad_highamp_strength": 0.70,
    "selection_stress_bad_highamp_prob": 0.34,
    "seed": 20260997,
}

CANDIDATES["predtop20_eventqrs_stresshead_lowaux_specific_pretrain"] = {
    **CANDIDATES["predtop20_eventqrs_intermitbad_lowaux_specific_pretrain"],
    # Separate the learnable bad-stress subtype from the ordinary bad/non-bad
    # auxiliary head.  This tests whether the event-QRS encoder can recover the
    # contact/dropout/baseline shell without pushing all boundary medium to bad.
    "stress_aux_weight": 0.72,
    "stress_aux_pos_weight": 5.0,
    "pretrain_stress_aux_weight": 0.32,
    "stress_bad_logit_mix": 0.22,
    "stress_gate_bias_init": -2.25,
    "bad_aux_weight": 0.86,
    "bad_aux_pos_weight": 4.2,
    "bad_specificity_weight": 4.30,
    "bad_specificity_margin": 0.92,
    "selection_bad_stress_weight": 0.28,
    "seed": 20260998,
}

CANDIDATES["predtop20_sqiquery_stresshead_lowaux_specific_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_intermitbad_lowaux_specific_pretrain"],
    # Same stress-head split, but on the simpler SQI-query backbone to check
    # whether event tokens help stress transfer or mainly destabilize medium.
    "stress_aux_weight": 0.68,
    "stress_aux_pos_weight": 4.8,
    "pretrain_stress_aux_weight": 0.30,
    "stress_bad_logit_mix": 0.18,
    "stress_gate_bias_init": -2.30,
    "bad_aux_weight": 0.82,
    "bad_aux_pos_weight": 4.0,
    "bad_specificity_weight": 4.15,
    "bad_specificity_margin": 0.94,
    "selection_bad_stress_weight": 0.25,
    "seed": 20260999,
}

CANDIDATES["predtop20_eventqrs_longcontact_stresshead_pretrain"] = {
    **CANDIDATES["predtop20_eventqrs_stresshead_lowaux_specific_pretrain"],
    # Explicitly target the original-test bad-stress gap: long low-detail
    # contact/dropout plus baseline shell.  Keep stronger non-bad hard
    # negatives and only a modest stress-logit mix to protect medium.
    "bad_intermittent_contact_strength": 2.65,
    "selection_stress_bad_intermittent_contact_strength": 2.85,
    "bad_stress_shell_strength": 1.20,
    "selection_stress_bad_stress_shell_strength": 1.35,
    "bad_highamp_stress_strength": 0.45,
    "bad_highamp_prob": 0.22,
    "selection_stress_bad_highamp_strength": 0.55,
    "selection_stress_bad_highamp_prob": 0.25,
    "nonbad_hardneg_strength": 1.95,
    "selection_stress_nonbad_hardneg_strength": 1.95,
    "stress_aux_weight": 0.82,
    "stress_aux_pos_weight": 6.0,
    "pretrain_stress_aux_weight": 0.38,
    "stress_bad_logit_mix": 0.16,
    "bad_aux_weight": 0.78,
    "bad_specificity_weight": 4.85,
    "bad_specificity_margin": 1.02,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261000,
}

CANDIDATES["predtop20_sqiquery_subject111_shift_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    # Train-time PTB waveform shift for the original-test 111001-style
    # non-bad domain: low QRS slope/prominence, high LF baseline, and lower
    # detail tails.  This targets the medium->good failures without using
    # original rows as training examples.
    "subject111_shift_strength": 1.35,
    "aug_aux_weight_nonbad": 0.22,
    "nonbad_hardneg_strength": 0.85,
    "class_weight": [1.08, 1.62, 2.85],
    "gm_pair_weight": 0.86,
    "bad_aux_weight": 0.72,
    "bad_aux_pos_weight": 3.2,
    "bad_specificity_weight": 3.25,
    "bad_specificity_margin": 0.82,
    "seed": 20261001,
}

CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_pretrain"],
    # Same non-bad domain shift plus a mild stress specialist.  This is the
    # first attempt to improve medium transfer while not losing bad core.
    "bad_stress_shell_strength": 0.85,
    "bad_intermittent_contact_strength": 1.55,
    "bad_highamp_stress_strength": 0.35,
    "bad_highamp_prob": 0.18,
    "aug_aux_weight_bad": 0.18,
    "stress_aux_weight": 0.46,
    "stress_aux_pos_weight": 4.0,
    "pretrain_stress_aux_weight": 0.18,
    "stress_bad_logit_mix": 0.10,
    "selection_bad_stress_weight": 0.16,
    "selection_stress_nonbad_hardneg_strength": 1.25,
    "selection_stress_bad_stress_shell_strength": 0.95,
    "selection_stress_bad_intermittent_contact_strength": 1.75,
    "selection_subject111_shift_strength": 1.10,
    "seed": 20261002,
}

CANDIDATES["predtop20_sqiquery_subject111_shift_impulse_p12_nodual"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Follow-up from the p20 vs shift-stress exchange audit: shift-stress is the
    # best original-test transfer point, while p20's useful extra signal is a
    # small impulse/reset bad slice.  Keep the ordinary head and high specificity
    # guard so this tests data coverage, not a broad bad-collapse mechanism.
    "bad_impulse_reset_strength": 1.45,
    "bad_impulse_reset_prob": 0.12,
    "selection_stress_bad_impulse_reset_strength": 1.65,
    "selection_stress_bad_impulse_reset_prob": 0.55,
    "bad_intermittent_contact_prob": 0.54,
    "selection_stress_bad_intermittent_contact_prob": 0.66,
    "bad_aux_pos_weight": 3.45,
    "bad_specificity_weight": 3.05,
    "bad_specificity_margin": 0.78,
    "selection_bad_stress_weight": 0.22,
    "seed": 20261052,
}

CANDIDATES["predtop20_sqiquery_subject111_shift_v5badlite_nodual"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Same best transfer point, but expose the sparse reset/dropout primitive
    # bank only lightly.  The goal is to lift record111 bad-outlier sensitivity
    # without turning medium outlier rows into bad.
    "primitive_bank": "qrs_stress_v4",
    "primitive_bad_bank": "qrs_stress_v5",
    "primitive_bad_mix": 0.58,
    "bad_impulse_reset_strength": 1.30,
    "bad_impulse_reset_prob": 0.10,
    "selection_stress_bad_impulse_reset_strength": 1.55,
    "selection_stress_bad_impulse_reset_prob": 0.50,
    "bad_intermittent_contact_prob": 0.50,
    "selection_stress_bad_intermittent_contact_prob": 0.62,
    "bad_aux_pos_weight": 3.38,
    "bad_specificity_weight": 3.10,
    "bad_specificity_margin": 0.80,
    "selection_bad_stress_weight": 0.23,
    "seed": 20261053,
}

CANDIDATES["predtop20_sqiquery_subject111_mediumonly_shift_stress"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Coverage-gap audit: the dominant medium->good errors are far outside the
    # PTB primitive train shell, while good->medium errors are much closer and
    # should not be pushed further.  Keep the record111-style shift mostly on
    # medium rows instead of all non-bad rows.
    "subject111_shift_strength": 1.28,
    "selection_subject111_shift_strength": 1.08,
    "subject111_good_shift_scale": 0.18,
    "subject111_medium_shift_scale": 1.38,
    "aug_aux_weight_nonbad": 0.26,
    "nonbad_hardneg_strength": 0.62,
    "class_weight": [1.16, 1.68, 2.85],
    "gm_pair_weight": 0.94,
    "bad_specificity_weight": 3.45,
    "bad_specificity_margin": 0.86,
    "seed": 20261054,
}

CANDIDATES["predtop20_sqiquery_subject111_mediumonly_shift_impulse"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_mediumonly_shift_stress"],
    # Same medium-only shell plus a small bad impulse slice.  This is a narrow
    # test of whether bad-outlier recall can rise without repeating the
    # broad-v5 false-bad failure.
    "bad_impulse_reset_strength": 1.38,
    "bad_impulse_reset_prob": 0.10,
    "selection_stress_bad_impulse_reset_strength": 1.58,
    "selection_stress_bad_impulse_reset_prob": 0.50,
    "bad_aux_pos_weight": 3.38,
    "selection_bad_stress_weight": 0.20,
    "seed": 20261055,
}

CANDIDATES["predtop20_sqiquery_subject111_medium_gapheavy_mguard"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Larger-range coverage test from the primitive gap decomposition:
    # medium->good original-test errors are far outside the PTB primitive shell
    # in QRS-visibility, detector agreement, baseline/frequency and stress
    # groups.  Push only medium rows into that shell, keep good nearly untouched,
    # and guard selection with precision so this does not become a broad
    # medium-heavy shortcut.
    "subject111_shift_strength": 1.48,
    "selection_subject111_shift_strength": 1.28,
    "subject111_good_shift_scale": 0.02,
    "subject111_medium_shift_scale": 2.20,
    "aug_aux_weight_nonbad": 0.18,
    "nonbad_hardneg_strength": 1.05,
    "class_weight": [1.02, 1.92, 2.72],
    "gm_pair_weight": 1.12,
    "bad_specificity_weight": 3.35,
    "bad_specificity_margin": 0.82,
    "selection_precision_weight": 0.30,
    "selection_stress_precision_weight": 0.22,
    "selection_medium_stress_weight": 0.24,
    "selection_medium_subject111_shift_strength": 1.48,
    "selection_medium_subject111_good_shift_scale": 0.0,
    "selection_medium_subject111_medium_shift_scale": 2.10,
    "selection_medium_stress_nonbad_hardneg_strength": 1.05,
    "seed": 20261060,
}

CANDIDATES["predtop20_sqiquery_subject111_medium_gapheavy_balanced"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_medium_gapheavy_mguard"],
    # Slightly less medium-heavy and more good-preserving; use this to check if
    # the gap-heavy shell can help without collapsing good outlier rows.
    "subject111_shift_strength": 1.36,
    "subject111_good_shift_scale": 0.10,
    "subject111_medium_shift_scale": 1.85,
    "class_weight": [1.12, 1.78, 2.78],
    "gm_pair_weight": 1.00,
    "selection_medium_subject111_shift_strength": 1.36,
    "selection_medium_subject111_medium_shift_scale": 1.85,
    "selection_medium_stress_weight": 0.18,
    "seed": 20261061,
}

CANDIDATES["predtop20_sqiquery_subject111_shift_stress_mediumselect"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Selection-only experiment.  Keep the best transfer training recipe, but
    # choose the checkpoint with an additional PTB-generated medium-outlier
    # stress slice so we can test whether prior runs were choosing epochs that
    # look good globally but eat original-test medium rows.
    "selection_medium_stress_weight": 0.36,
    "selection_medium_subject111_shift_strength": 1.55,
    "selection_medium_subject111_good_shift_scale": 0.0,
    "selection_medium_subject111_medium_shift_scale": 1.55,
    "selection_medium_stress_nonbad_hardneg_strength": 0.82,
    "selection_medium_aug_aux_weight_nonbad": 0.30,
    "seed": 20261056,
}

CANDIDATES["predtop20_sqiquery_subject111_shift_stress_precisionselect"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Candidate-level audit shows original acc and medium transfer track
    # bad/medium precision more than raw recall.  Keep the same training data
    # and choose checkpoints with a precision-aware synthetic/stress score.
    "selection_precision_weight": 0.34,
    "selection_stress_precision_weight": 0.28,
    "selection_bad_stress_weight": 0.12,
    "seed": 20261058,
}

CANDIDATES["predtop20_sqiquery_subject111_shift_stress_originalcurve"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Diagnostic only: keep the best known training recipe and emit per-epoch
    # original BUT report-only metrics so we can see whether synthetic
    # checkpoint selection is missing a better epoch.  Original metrics do not
    # enter the selection score.
    "report_original_each_epoch": True,
    "seed": 20261062,
}

CANDIDATES["featurefirst_top20_shift_stress_a025"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Force the final decision to go mostly through internally predicted SQI /
    # geometry features. The waveform embedding remains as a small residual.
    "feature_first_logits": True,
    "feature_residual_alpha": 0.25,
    "aux_pretrain_epochs": 5,
    "pretrain_classification_weight": 0.06,
    "pretrain_hard_aux_weight": 6.40,
    "pretrain_top14_aux_weight": 6.20,
    "pretrain_critical_aux_weight": 7.80,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + ["qrs_visibility", "detector_agreement", "baseline_step", "sqi_basSQI", "pc2", "pc3"],
    "seed": 20261210,
}

CANDIDATES["featurefirst_top20_shift_stress_a050"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a025"],
    "feature_residual_alpha": 0.50,
    "seed": 20261211,
}

CANDIDATES["featurefirst_top20_shift_stress_a050_seed18"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    "seed": 20261218,
}

CANDIDATES["featurefirst_top20_shift_stress_a050_seed19"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    "seed": 20261219,
}

CANDIDATES["featurefirst_top20_shift_stress_a050_seed20"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    "seed": 20261220,
}

CANDIDATES["featurefirst_top20_shift_stress_a050_seed21"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    "seed": 20261221,
}

CANDIDATES["featurefirst_top20_shift_stress_a065"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a025"],
    "feature_residual_alpha": 0.65,
    "seed": 20261214,
}

CANDIDATES["featurefirst_top20_shift_stress_a080"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a025"],
    "feature_residual_alpha": 0.80,
    "seed": 20261215,
}

CANDIDATES["featurefirst_top20_shift_stress_goodguard_a065"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a025"],
    # a050 shifted the decision boundary toward medium and gained overall
    # transfer. This variant keeps feature-first routing but restores a little
    # good pressure so the gain is not only medium-heavy.
    "feature_residual_alpha": 0.65,
    "class_weight": [1.22, 1.50, 2.85],
    "gm_pair_weight": 0.80,
    "bad_aux_weight": 0.70,
    "bad_specificity_weight": 3.35,
    "bad_specificity_margin": 0.84,
    "seed": 20261216,
}

CANDIDATES["featurefirst_top20_shift_stress_mediumguard_a060"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a025"],
    # Control arm: slightly less good pressure and alpha near the observed
    # transfer sweet spot. This tests whether the a050 gain is a medium-stability
    # effect rather than an alpha accident.
    "feature_residual_alpha": 0.60,
    "class_weight": [1.04, 1.68, 2.85],
    "gm_pair_weight": 0.90,
    "seed": 20261217,
}

CANDIDATES["featuregate_top20_shift_stress_a050_bneg"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    # Dynamic routing: start by trusting the waveform embedding on easy rows,
    # then let the learned gate move toward feature-first only where it helps.
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": -0.65,
    "seed": 20261230,
}

CANDIDATES["featuregate_top20_shift_stress_a050_b0"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": 0.0,
    "seed": 20261231,
}

CANDIDATES["featuregate_top20_shift_stress_a050_bpos"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": 0.45,
    "seed": 20261232,
}

CANDIDATES["featuregate_top20_shift_stress_goodguard_a050_bneg"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_goodguard_a065"],
    "feature_residual_alpha": 0.50,
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": -0.65,
    "seed": 20261233,
}

CANDIDATES["featurefirst_top23_shift_stress_a050"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    # Exchange analysis showed the feature-first model wins low-confidence
    # boundary rows but hurts some high-confidence/easy rows.  Add the
    # boundary-confidence and KNN-purity targets to the predicted feature set so
    # the classifier can learn that routing signal from waveform only.
    "predicted_feature_columns": TOP23_HARD_FEATURE_COLUMNS,
    "critical_aux_columns": TOP23_HARD_FEATURE_COLUMNS
    + ["qrs_visibility", "detector_agreement", "baseline_step", "boundary_confidence", "knn_label_purity", "pc2", "pc3"],
    "pretrain_critical_aux_weight": 8.25,
    "seed": 20261234,
}

CANDIDATES["featuregate_top23_shift_stress_a050_bneg"] = {
    **CANDIDATES["featurefirst_top23_shift_stress_a050"],
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": -0.65,
    "seed": 20261235,
}

CANDIDATES["featuregate_top23_shift_stress_a050_b0"] = {
    **CANDIDATES["featurefirst_top23_shift_stress_a050"],
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": 0.0,
    "seed": 20261236,
}

CANDIDATES["featuregate_top25_shift_stress_a050_b0"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    "predicted_feature_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS,
    "critical_aux_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS
    + ["qrs_visibility", "detector_agreement", "baseline_step", "boundary_confidence", "knn_label_purity", "pc2", "pc3"],
    "pretrain_critical_aux_weight": 8.40,
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": 0.0,
    "seed": 20261237,
}

CANDIDATES["featuregate_supervised_top20_shift_stress_a050_q35"] = {
    **CANDIDATES["featuregate_top20_shift_stress_a050_b0"],
    "feature_gate_supervision_weight": 0.55,
    "pretrain_feature_gate_supervision_weight": 0.30,
    "feature_gate_boundary_quantile": 0.35,
    "feature_gate_purity_quantile": 0.35,
    "seed": 20261238,
}

CANDIDATES["featuregate_supervised_top20_shift_stress_a050_q45"] = {
    **CANDIDATES["featuregate_top20_shift_stress_a050_b0"],
    "feature_gate_supervision_weight": 0.75,
    "pretrain_feature_gate_supervision_weight": 0.40,
    "feature_gate_boundary_quantile": 0.45,
    "feature_gate_purity_quantile": 0.45,
    "seed": 20261239,
}

CANDIDATES["featuregate_supervised_top23_shift_stress_a050_q35"] = {
    **CANDIDATES["featuregate_top23_shift_stress_a050_b0"],
    "feature_gate_supervision_weight": 0.55,
    "pretrain_feature_gate_supervision_weight": 0.30,
    "feature_gate_boundary_quantile": 0.35,
    "feature_gate_purity_quantile": 0.35,
    "seed": 20261240,
}

CANDIDATES["featuregate_supervised_top23_shift_stress_a050_q45"] = {
    **CANDIDATES["featuregate_top23_shift_stress_a050_b0"],
    "feature_gate_supervision_weight": 0.75,
    "pretrain_feature_gate_supervision_weight": 0.40,
    "feature_gate_boundary_quantile": 0.45,
    "feature_gate_purity_quantile": 0.45,
    "seed": 20261241,
}

CANDIDATES["featurefirst_top20_hardrec_a050"] = {
    **CANDIDATES["featurefirst_top20_shift_stress_a050"],
    # Feature-recovery module test: keep the stable top20 classifier contract,
    # but force the encoder/aux head to recover the hard waveform-derived
    # geometry signals before classification dominates.
    "aux_pretrain_epochs": 8,
    "pretrain_classification_weight": 0.035,
    "pretrain_hard_aux_weight": 8.50,
    "pretrain_top14_aux_weight": 7.20,
    "pretrain_critical_aux_weight": 12.00,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "baseline_step",
        "sqi_basSQI",
        "sqi_basSQI",
        "boundary_confidence",
        "knn_label_purity",
        "pc2",
        "pc3",
    ],
    "hard_aux_weight": 5.40,
    "critical_aux_weight": 5.20,
    "seed": 20261242,
}

CANDIDATES["featurefirst_top20_hardrec_teacher_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # Same recovery pressure plus a soft feature-tree teacher.  This tests
    # whether better feature recovery must be tied to the quality boundary, not
    # only regressed as independent scalar targets.
    "feature_teacher_columns": TOP23_HARD_FEATURE_COLUMNS,
    "feature_teacher_distill_weight": 0.65,
    "pretrain_feature_teacher_distill_weight": 1.05,
    "feature_teacher_temperature": 1.8,
    "seed": 20261243,
}

CANDIDATES["featurefirst_top20_hardrec_longaux_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # Aux-pred upper-bound audit showed the final classifier is not the main
    # bottleneck: internally predicted qrs_visibility / detector_agreement /
    # baseline_step / pc2 remain weak.  This arm keeps the same waveform-only
    # inference contract but gives the encoder a longer feature-recovery phase
    # before normal classification pressure takes over.
    "aux_pretrain_epochs": 12,
    "pretrain_classification_weight": 0.018,
    "pretrain_gm_pair_weight": 0.035,
    "pretrain_hard_aux_weight": 11.50,
    "pretrain_top14_aux_weight": 9.40,
    "pretrain_critical_aux_weight": 15.50,
    "pretrain_rank_weight": 1.15,
    "pretrain_neighbor_weight": 0.26,
    "hard_aux_weight": 6.60,
    "critical_aux_weight": 6.70,
    "rank_weight": 0.96,
    "seed": 20261270,
}

CANDIDATES["featurefirst_top20_hardrec_wideaux_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # Capacity control: keep the same schedule as hardrec, but widen only the
    # auxiliary prediction path.  If this improves feature recovery without
    # hurting transfer, the previous head was too small for the hard axes.
    "wide_aux_head": True,
    "aux_hidden": 256,
    "hard_aux_weight": 6.10,
    "critical_aux_weight": 6.10,
    "pretrain_hard_aux_weight": 9.80,
    "pretrain_top14_aux_weight": 8.10,
    "pretrain_critical_aux_weight": 13.40,
    "seed": 20261271,
}

CANDIDATES["featurefirst_top20_hardrec_longaux_qrsbase_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_longaux_a050"],
    # Same long feature-first schedule, but concentrate repeated critical
    # supervision on the exact axes that stayed weak in the audit.  This tests
    # whether the model can learn the visual QRS/baseline cues when the loss is
    # no longer diluted across the easier pca_margin/non_qrs features.
    "wide_aux_head": True,
    "aux_hidden": 256,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "baseline_step",
        "baseline_step",
        "sqi_basSQI",
        "sqi_basSQI",
        "pc2",
        "pc2",
        "pc3",
        "boundary_confidence",
        "knn_label_purity",
    ],
    "pretrain_critical_aux_weight": 18.00,
    "critical_aux_weight": 7.40,
    "seed": 20261272,
}

CANDIDATES["featurefirst_top20_hardrec_qfeatbin_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # SmoothL1 did not teach the weak axes.  Add quantile-direction supervision
    # so the model only has to learn whether each hard SQI/geometry axis is
    # low/mid/high.  This matches the actual decision use more closely than
    # exact z-value regression.
    "quantile_aux_columns": [
        "qrs_visibility",
        "detector_agreement",
        "baseline_step",
        "sqi_basSQI",
        "flatline_ratio",
        "pc2",
        "pc3",
        "boundary_confidence",
        "knn_label_purity",
    ],
    "quantile_aux_quantiles": [0.25, 0.50, 0.75],
    "quantile_aux_sharpness": 2.65,
    "pretrain_quantile_aux_weight": 4.20,
    "quantile_aux_weight": 2.20,
    "seed": 20261273,
}

CANDIDATES["featurefirst_top20_hardrec_qfeatbin_qrsbase_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_longaux_qrsbase_a050"],
    # Stronger version of the same test: combine concentrated QRS/baseline
    # critical columns with quantile-direction supervision.  If this still
    # cannot lift qrs_visibility/detector/pc2, the limitation is in the
    # waveform/tokenizer representation rather than the loss shape.
    "quantile_aux_columns": [
        "qrs_visibility",
        "detector_agreement",
        "baseline_step",
        "sqi_basSQI",
        "flatline_ratio",
        "pc2",
        "pc3",
        "boundary_confidence",
        "knn_label_purity",
    ],
    "quantile_aux_quantiles": [0.25, 0.50, 0.75],
    "quantile_aux_sharpness": 2.85,
    "pretrain_quantile_aux_weight": 5.20,
    "quantile_aux_weight": 2.60,
    "seed": 20261274,
}

CANDIDATES["featurefirst_top20_hardrec_qrsfocus_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # QRS/baseline-focused control arm for the features that remain weakest on
    # both synthetic and original recovery.
    "primitive_bank": "qrs_enhanced_v2",
    "primitive_bad_mix": 0.76,
    "pretrain_critical_aux_weight": 13.00,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "qrs_band_ratio",
        "template_corr",
        "baseline_step",
        "sqi_basSQI",
        "flatline_ratio",
        "pc2",
        "pc3",
    ],
    "seed": 20261244,
}

CANDIDATES["featurefirst_top20_hardrec_qrslite_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # The qrsfocus arm recovered more bad-outlier stress rows, but paid for it
    # with too many good/medium regressions.  Keep only a small QRS-aware dose
    # and raise non-bad specificity so this tests a narrow bad-sensitivity gain.
    "primitive_bank": "qrs_enhanced_v2",
    "primitive_bad_mix": 0.42,
    "class_weight": [1.20, 1.58, 2.95],
    "gm_pair_weight": 0.88,
    "bad_aux_weight": 0.74,
    "bad_aux_pos_weight": 3.1,
    "bad_logit_mix": 0.38,
    "bad_specificity_weight": 3.85,
    "bad_specificity_margin": 0.88,
    "pretrain_critical_aux_weight": 12.50,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "qrs_band_ratio",
        "template_corr",
        "baseline_step",
        "sqi_basSQI",
        "flatline_ratio",
        "pc2",
        "pc3",
    ],
    "seed": 20261245,
}

CANDIDATES["featurefirst_top20_hardrec_badlite_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # Controlled bad-outlier expansion without the broad QRS push: learn a
    # little more record111-style high-amplitude/contact stress while preserving
    # the hardrec good/medium transfer point.
    "bad_outlier_aug_strength": 0.92,
    "bad_stress_shell_strength": 1.05,
    "bad_highamp_stress_strength": 1.10,
    "bad_highamp_prob": 0.24,
    "class_weight": [1.18, 1.57, 3.05],
    "bad_aux_weight": 0.82,
    "bad_aux_pos_weight": 3.4,
    "bad_logit_mix": 0.42,
    "bad_specificity_weight": 4.05,
    "bad_specificity_margin": 0.90,
    "selection_bad_stress_weight": 0.12,
    "selection_stress_nonbad_hardneg_strength": 0.75,
    "selection_stress_bad_outlier_aug_strength": 0.82,
    "selection_stress_bad_stress_shell_strength": 1.05,
    "selection_stress_bad_highamp_strength": 1.10,
    "selection_stress_bad_highamp_prob": 0.32,
    "seed": 20261246,
}

CANDIDATES["featurefirst_top20_hardrec_goodguard_badlite_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_badlite_a050"],
    # Guard against the common failure mode where bad stress recovery steals
    # good/medium boundary rows.
    "class_weight": [1.28, 1.54, 2.90],
    "gm_pair_weight": 0.98,
    "bad_aux_weight": 0.72,
    "bad_logit_mix": 0.34,
    "bad_specificity_weight": 4.30,
    "selection_bad_stress_weight": 0.08,
    "seed": 20261247,
}

CANDIDATES["featurefirst_top20_hardrec_badlite_marginguard_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_badlite_a050"],
    # The exchange audit shows bad-lite gains stress rows but regresses
    # low-confidence good/medium rows.  Margin guard uses internally predicted
    # geometry only, so it remains waveform-only at inference while discouraging
    # bad takeover on high-purity/high-margin non-bad rows.
    "bad_margin_guard": True,
    "bad_margin_alpha": 0.36,
    "bad_purity_alpha": 0.22,
    "bad_boundary_alpha": 0.14,
    "bad_logit_mix": 0.46,
    "bad_specificity_weight": 4.15,
    "selection_bad_stress_weight": 0.14,
    "seed": 20261248,
}

CANDIDATES["featurefirst_top20_hardrec_badlite_marginguard_soft_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_badlite_a050"],
    "bad_margin_guard": True,
    "bad_margin_alpha": 0.24,
    "bad_purity_alpha": 0.16,
    "bad_boundary_alpha": 0.10,
    "bad_logit_mix": 0.38,
    "bad_specificity_weight": 4.45,
    "class_weight": [1.22, 1.58, 2.88],
    "selection_bad_stress_weight": 0.10,
    "seed": 20261249,
}

CANDIDATES["featurefirst_top20_hardrec_shiftbad_marginguard_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # Bring back the old shift-stress model's useful bad sensitivity, but route
    # it through margin guard instead of broad bad expansion.
    "bad_margin_guard": True,
    "bad_margin_alpha": 0.32,
    "bad_purity_alpha": 0.20,
    "bad_boundary_alpha": 0.12,
    "class_weight": [1.16, 1.58, 3.15],
    "bad_aux_weight": 0.88,
    "bad_aux_pos_weight": 3.6,
    "bad_logit_mix": 0.50,
    "bad_specificity_weight": 4.25,
    "selection_bad_stress_weight": 0.16,
    "seed": 20261250,
}

CANDIDATES["featurefirst_top20_hardrec_record111lite_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # Target the original-test 111001 bad-stress shell without repeating the
    # broad bad-lite failure: rare record111 motion / burst-dropout / impulse
    # blocks, low stress-logit mix, and strong non-bad specificity.
    "bad_record111_motion_strength": 2.05,
    "bad_record111_motion_prob": 0.14,
    "bad_record111_burst_dropout_strength": 1.30,
    "bad_record111_burst_dropout_prob": 0.10,
    "bad_impulse_reset_strength": 1.20,
    "bad_impulse_reset_prob": 0.08,
    "bad_stress_shell_strength": 0.44,
    "bad_intermittent_contact_strength": 0.58,
    "bad_highamp_stress_strength": 0.16,
    "bad_highamp_prob": 0.10,
    "aug_aux_weight_bad": 0.10,
    "class_weight": [1.20, 1.62, 2.92],
    "gm_pair_weight": 0.94,
    "stress_aux_weight": 0.38,
    "stress_aux_pos_weight": 4.4,
    "pretrain_stress_aux_weight": 0.16,
    "stress_bad_logit_mix": 0.045,
    "bad_aux_weight": 0.72,
    "bad_aux_pos_weight": 3.2,
    "bad_specificity_weight": 4.55,
    "bad_specificity_margin": 0.94,
    "selection_bad_stress_weight": 0.12,
    "selection_stress_nonbad_hardneg_strength": 1.45,
    "selection_stress_bad_record111_motion_strength": 2.20,
    "selection_stress_bad_record111_motion_prob": 1.0,
    "selection_stress_bad_record111_burst_dropout_strength": 1.48,
    "selection_stress_bad_record111_burst_dropout_prob": 0.80,
    "selection_stress_bad_impulse_reset_strength": 1.35,
    "selection_stress_bad_impulse_reset_prob": 0.45,
    "selection_stress_bad_stress_shell_strength": 0.50,
    "selection_stress_bad_intermittent_contact_strength": 0.70,
    "selection_stress_bad_highamp_strength": 0.18,
    "selection_stress_bad_highamp_prob": 0.18,
    "seed": 20261251,
}

CANDIDATES["featurefirst_top20_hardrec_record111stress_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_record111lite_a050"],
    # Stronger coverage probe: if this helps bad-outlier but hurts GM, the gap is
    # data coverage plus specificity; if it still misses bad, the waveform model
    # is not learning the record111 stress primitive.
    "bad_record111_motion_strength": 2.55,
    "bad_record111_motion_prob": 0.22,
    "bad_record111_burst_dropout_strength": 1.72,
    "bad_record111_burst_dropout_prob": 0.18,
    "bad_impulse_reset_strength": 1.52,
    "bad_impulse_reset_prob": 0.12,
    "bad_stress_shell_strength": 0.34,
    "bad_intermittent_contact_strength": 0.42,
    "bad_highamp_stress_strength": 0.10,
    "class_weight": [1.18, 1.60, 3.08],
    "stress_aux_weight": 0.50,
    "stress_aux_pos_weight": 5.2,
    "pretrain_stress_aux_weight": 0.22,
    "stress_bad_logit_mix": 0.060,
    "bad_aux_weight": 0.78,
    "bad_aux_pos_weight": 3.55,
    "bad_specificity_weight": 4.90,
    "bad_specificity_margin": 1.02,
    "selection_bad_stress_weight": 0.18,
    "selection_stress_nonbad_hardneg_strength": 1.70,
    "selection_stress_bad_record111_motion_strength": 2.75,
    "selection_stress_bad_record111_burst_dropout_strength": 1.95,
    "selection_stress_bad_impulse_reset_strength": 1.72,
    "selection_stress_bad_impulse_reset_prob": 0.60,
    "seed": 20261252,
}

CANDIDATES["featurefirst_top20_hardrec_record111specific_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_record111stress_a050"],
    # Same wider bad-stress shell, but more conservative bad routing.  This is
    # the specificity control arm for the record111stress candidate.
    "class_weight": [1.26, 1.64, 2.82],
    "gm_pair_weight": 1.04,
    "stress_bad_logit_mix": 0.025,
    "bad_aux_weight": 0.62,
    "bad_aux_pos_weight": 2.85,
    "bad_specificity_weight": 5.35,
    "bad_specificity_margin": 1.08,
    "selection_bad_stress_weight": 0.10,
    "selection_precision_weight": 0.26,
    "selection_stress_precision_weight": 0.22,
    "selection_stress_nonbad_hardneg_strength": 1.95,
    "seed": 20261253,
}

CANDIDATES["featurefirst_top20_hardrec_eventqrs_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_a050"],
    # Architecture probe: keep the best feature-first hard-recovery objective,
    # but replace the plain SQI-query patch body with explicit local event/QRS
    # tokens.  The missing features audit points at qrs_visibility and
    # detector_agreement, so this candidate tests whether the encoder can learn
    # those from waveform-local QRS proposals instead of more bad data.
    "arch": "event_qrs_sqi_query_multiscale_patch_teacher_atlas",
    "event_count": 16,
    "event_window": 101,
    "width": 96,
    "layers": 4,
    "heads": 4,
    "patch": 25,
    "scales": [25, 50, 100, 250, 500],
    "stat_hidden": 224,
    "pretrain_critical_aux_weight": 13.25,
    "critical_aux_weight": 5.60,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "sqi_basSQI",
        "qrs_band_ratio",
        "template_corr",
        "pc2",
        "pc3",
    ],
    "seed": 20261254,
}

CANDIDATES["featurefirst_top20_hardrec_eventqrs_primctx_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_eventqrs_a050"],
    # Add a small primitive-token context path for detector-agreement and
    # baseline/sparse-event evidence.  This is still waveform-derived input only;
    # teacher features supervise the auxiliary head but are not inference inputs.
    "primitive_context_fusion": True,
    "aux_primitive_context_fusion": True,
    "primitive_context_bank": "qrs_stress_v5",
    "primitive_context_width": 48,
    "primitive_context_layers": 1,
    "primitive_context_heads": 4,
    "primitive_context_hidden": 192,
    "primitive_context_chunk": 32,
    "aux_hidden": 512,
    "wide_aux_head": True,
    "pretrain_critical_aux_weight": 13.75,
    "hard_aux_weight": 5.65,
    "critical_aux_weight": 5.85,
    "seed": 20261255,
}

CANDIDATES["featurefirst_top20_hardrec_eventqrs_gate_a050"] = {
    **CANDIDATES["featurefirst_top20_hardrec_eventqrs_primctx_a050"],
    # Guarded version: let the model reduce feature-first reliance on
    # low-confidence rows where previous bad-stress gains stole good/medium
    # boundary examples.  The gate target is built only from synthetic/node
    # boundary_confidence and KNN purity.
    "dynamic_feature_gate": True,
    "feature_gate_bias_init": -0.25,
    "feature_gate_supervision_weight": 0.55,
    "pretrain_feature_gate_supervision_weight": 0.35,
    "feature_gate_boundary_quantile": 0.40,
    "feature_gate_purity_quantile": 0.40,
    "class_weight": [1.20, 1.58, 2.95],
    "gm_pair_weight": 0.98,
    "bad_aux_weight": 0.68,
    "bad_specificity_weight": 4.45,
    "bad_specificity_margin": 0.96,
    "selection_bad_stress_weight": 0.08,
    "seed": 20261256,
}

CANDIDATES["featurefirst_top20_qrsstressv3_stress_a035"] = {
    **CANDIDATES["predtop20_sqiquery_qrsstressv3_stress_pretrain"],
    "feature_first_logits": True,
    "feature_residual_alpha": 0.35,
    "aux_pretrain_epochs": 5,
    "pretrain_classification_weight": 0.06,
    "pretrain_hard_aux_weight": 6.60,
    "pretrain_top14_aux_weight": 6.30,
    "pretrain_critical_aux_weight": 8.00,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + ["qrs_visibility", "detector_agreement", "baseline_step", "sqi_basSQI", "pc2", "pc3"],
    "seed": 20261213,
}

CANDIDATES["predtop20_sqiquery_subject111_goodsafe_stress_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # The first subject111 shift recovered medium but ate too much good.  This
    # variant lowers the non-bad shift and gives good more class pressure while
    # keeping a mild stress branch for bad core.
    "subject111_shift_strength": 1.02,
    "selection_subject111_shift_strength": 0.85,
    "aug_aux_weight_nonbad": 0.30,
    "nonbad_hardneg_strength": 0.55,
    "class_weight": [1.24, 1.52, 2.85],
    "gm_pair_weight": 0.80,
    "bad_stress_shell_strength": 0.72,
    "bad_intermittent_contact_strength": 1.35,
    "bad_highamp_stress_strength": 0.28,
    "stress_aux_weight": 0.40,
    "stress_bad_logit_mix": 0.08,
    "bad_specificity_weight": 3.55,
    "bad_specificity_margin": 0.86,
    "selection_bad_stress_weight": 0.14,
    "seed": 20261003,
}

CANDIDATES["predtop20_sqiquery_subject111_stressv3_mid_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Middle point between the medium-recovering subject111 shift and the
    # good-safe variant, with waveform-only qrs_stress_v3 primitives plus a
    # differentiable threshold branch.  The aim is to learn the low-QRS/LF
    # baseline and bad-stress axes without feeding the 47 teacher features at
    # inference time.
    "primitive_bank": "qrs_stress_v3",
    "primitive_threshold_fusion": True,
    "primitive_threshold_values": [-2.25, -1.45, -0.75, -0.25, 0.25, 0.75, 1.45, 2.25],
    "primitive_threshold_sharpness": 2.4,
    "subject111_shift_strength": 1.18,
    "selection_subject111_shift_strength": 0.95,
    "aug_aux_weight_nonbad": 0.25,
    "nonbad_hardneg_strength": 0.70,
    "class_weight": [1.15, 1.58, 2.95],
    "gm_pair_weight": 0.84,
    "bad_stress_shell_strength": 0.80,
    "bad_intermittent_contact_strength": 1.45,
    "bad_highamp_stress_strength": 0.32,
    "bad_highamp_prob": 0.16,
    "aug_aux_weight_bad": 0.18,
    "stress_aux_weight": 0.42,
    "stress_aux_pos_weight": 4.0,
    "pretrain_stress_aux_weight": 0.18,
    "stress_bad_logit_mix": 0.09,
    "bad_specificity_weight": 3.65,
    "bad_specificity_margin": 0.88,
    "selection_bad_stress_weight": 0.15,
    "selection_stress_nonbad_hardneg_strength": 1.12,
    "selection_stress_bad_stress_shell_strength": 0.90,
    "selection_stress_bad_intermittent_contact_strength": 1.60,
    "seed": 20261004,
}

CANDIDATES["predtop20_sqiquery_subject111_lfbank_stress_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Low-frequency/baseline bank without the threshold shortcut.  This tests
    # whether missing basSQI/baseline/detector observables, rather than split
    # capacity, are the remaining blocker for original-test transfer.
    "primitive_bank": "qrs_stress_v4",
    "primitive_threshold_fusion": False,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + [
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "baseline_step",
        "sqi_basSQI",
        "sqi_basSQI",
        "pc2",
        "pc3",
        "boundary_confidence",
        "qrs_band_ratio",
    ],
    "pretrain_core_aux_weight": 2.35,
    "pretrain_hard_aux_weight": 5.35,
    "pretrain_top14_aux_weight": 5.35,
    "pretrain_critical_aux_weight": 7.10,
    "pretrain_rank_weight": 1.20,
    "subject111_shift_strength": 1.25,
    "selection_subject111_shift_strength": 1.05,
    "aug_aux_weight_nonbad": 0.24,
    "nonbad_hardneg_strength": 0.78,
    "class_weight": [1.12, 1.62, 2.95],
    "gm_pair_weight": 0.86,
    "bad_stress_shell_strength": 0.86,
    "bad_intermittent_contact_strength": 1.55,
    "bad_highamp_stress_strength": 0.34,
    "bad_highamp_prob": 0.18,
    "stress_aux_weight": 0.44,
    "stress_bad_logit_mix": 0.08,
    "bad_specificity_weight": 3.75,
    "bad_specificity_margin": 0.90,
    "selection_bad_stress_weight": 0.15,
    "selection_stress_nonbad_hardneg_strength": 1.22,
    "selection_stress_bad_stress_shell_strength": 0.95,
    "selection_stress_bad_intermittent_contact_strength": 1.70,
    "seed": 20261005,
}

CANDIDATES["predtop20_sqiquery_subject111_recordbad_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Isolate the data question: keep the proven SQI-query architecture but
    # replace contact-heavy bad exposure with record-111-like low-QRS/LF motion
    # bad stress.  This should recover test bad outliers only if the missing
    # subtype, not extra model capacity, is the blocker.
    "subject111_shift_strength": 1.25,
    "selection_subject111_shift_strength": 1.05,
    "bad_record111_motion_strength": 1.45,
    "selection_stress_bad_record111_motion_strength": 1.70,
    "bad_stress_shell_strength": 0.42,
    "bad_intermittent_contact_strength": 0.72,
    "bad_highamp_stress_strength": 0.18,
    "bad_highamp_prob": 0.14,
    "selection_stress_bad_stress_shell_strength": 0.48,
    "selection_stress_bad_intermittent_contact_strength": 0.88,
    "class_weight": [1.08, 1.58, 3.25],
    "gm_pair_weight": 0.84,
    "stress_aux_weight": 0.54,
    "stress_bad_logit_mix": 0.11,
    "bad_aux_weight": 0.82,
    "bad_aux_pos_weight": 4.2,
    "bad_specificity_weight": 3.95,
    "bad_specificity_margin": 0.92,
    "selection_bad_stress_weight": 0.18,
    "seed": 20261006,
}

CANDIDATES["predtop20_sqiquery_subject111_recordbad_strong_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_recordbad_pretrain"],
    # Strong version after primitive/waveform gap audit: record111 bad-stress
    # is masked low-amplitude + smooth drift + sparse spikes, not noisy contact.
    "bad_record111_motion_strength": 2.35,
    "selection_stress_bad_record111_motion_strength": 2.55,
    "bad_stress_shell_strength": 0.18,
    "bad_intermittent_contact_strength": 0.22,
    "bad_highamp_stress_strength": 0.08,
    "selection_stress_bad_stress_shell_strength": 0.20,
    "selection_stress_bad_intermittent_contact_strength": 0.25,
    "class_weight": [1.05, 1.54, 3.55],
    "stress_aux_weight": 0.66,
    "stress_bad_logit_mix": 0.14,
    "bad_aux_weight": 0.92,
    "bad_aux_pos_weight": 4.8,
    "bad_specificity_weight": 4.20,
    "bad_specificity_margin": 1.00,
    "selection_bad_stress_weight": 0.24,
    "seed": 20261007,
}

CANDIDATES["predtop20_sqiquery_subject111_dual_stress_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_shift_stress_pretrain"],
    # Test the architecture hypothesis from the failed strong bad run: the
    # waveform encoder can learn a bad-stress signal, but the ordinary 3-class
    # head suppresses it.  Use explicit P(bad) and P(good/medium | non-bad)
    # composition instead of another class-weight sweep.
    "final_mode": "dual_bad_gm_replace",
    "bad_gate_bias_init": -0.72,
    "primitive_bank": "qrs_stress_v4",
    "primitive_threshold_fusion": False,
    "class_weight": [1.08, 1.55, 3.05],
    "gm_pair_weight": 0.92,
    "bad_aux_weight": 0.72,
    "bad_aux_pos_weight": 3.25,
    "stress_aux_weight": 0.46,
    "stress_aux_pos_weight": 3.2,
    "stress_bad_logit_mix": 0.04,
    "bad_specificity_weight": 1.15,
    "bad_specificity_margin": 0.50,
    "selection_bad_stress_weight": 0.28,
    "selection_stress_nonbad_hardneg_strength": 1.10,
    "selection_stress_bad_stress_shell_strength": 0.76,
    "selection_stress_bad_intermittent_contact_strength": 1.10,
    "seed": 20261008,
}

CANDIDATES["predtop20_sqiquery_subject111_recordbad_dual_pretrain"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_recordbad_strong_pretrain"],
    # Same explicit dual-head composition, but with the record-111-like
    # masked-low-amplitude/smooth-drift bad augmentation.  This is a targeted
    # follow-up to the primitive PCA gap audit, not a broad bad expansion.
    "final_mode": "dual_bad_gm_replace",
    "bad_gate_bias_init": -0.62,
    "primitive_bank": "qrs_stress_v4",
    "class_weight": [1.08, 1.52, 3.20],
    "gm_pair_weight": 0.94,
    "bad_aux_weight": 0.78,
    "bad_aux_pos_weight": 3.55,
    "stress_aux_weight": 0.52,
    "stress_aux_pos_weight": 3.6,
    "stress_bad_logit_mix": 0.05,
    "bad_specificity_weight": 1.30,
    "bad_specificity_margin": 0.55,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261009,
}

CANDIDATES["predtop20_sqiquery_subject111_mixedbad_dual_p18"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_dual_stress_pretrain"],
    # Preserve the successful dual-head/core-bad behavior, but expose a small
    # controlled slice of record111-like bad outlier so it does not erase the
    # right-island/near-boundary bad core.
    "bad_record111_motion_strength": 2.15,
    "bad_record111_motion_prob": 0.18,
    "selection_stress_bad_record111_motion_strength": 2.35,
    "selection_stress_bad_record111_motion_prob": 1.0,
    "bad_intermittent_contact_prob": 0.55,
    "selection_stress_bad_intermittent_contact_prob": 0.70,
    "bad_aux_weight": 0.74,
    "bad_aux_pos_weight": 3.35,
    "bad_specificity_weight": 1.28,
    "bad_specificity_margin": 0.55,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261010,
}

CANDIDATES["predtop20_sqiquery_subject111_mixedbad_dual_p28"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_mixedbad_dual_p18"],
    "bad_record111_motion_prob": 0.28,
    "bad_intermittent_contact_prob": 0.48,
    "bad_aux_pos_weight": 3.55,
    "bad_specificity_weight": 1.40,
    "selection_bad_stress_weight": 0.38,
    "seed": 20261011,
}

CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_dual_stress_pretrain"],
    # From original-error waveform audit: record111 bad outliers often show
    # abrupt impulse/reset/dropout events, not only smooth low-QRS drift.
    # Keep this block sparse so bad core/right-island remains learnable.
    "bad_impulse_reset_strength": 1.65,
    "bad_impulse_reset_prob": 0.20,
    "selection_stress_bad_impulse_reset_strength": 1.85,
    "selection_stress_bad_impulse_reset_prob": 1.0,
    "bad_intermittent_contact_prob": 0.58,
    "selection_stress_bad_intermittent_contact_prob": 0.72,
    "bad_aux_weight": 0.76,
    "bad_aux_pos_weight": 3.45,
    "bad_specificity_weight": 1.22,
    "bad_specificity_margin": 0.52,
    "selection_bad_stress_weight": 0.36,
    "seed": 20261012,
}

CANDIDATES["featurefirst_top20_p20_a035"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    "feature_first_logits": True,
    "feature_residual_alpha": 0.35,
    "aux_pretrain_epochs": 5,
    "pretrain_classification_weight": 0.06,
    "pretrain_hard_aux_weight": 6.40,
    "pretrain_top14_aux_weight": 6.20,
    "pretrain_critical_aux_weight": 7.80,
    "critical_aux_columns": TOP20_INTERPRETABLE_FEATURE_COLUMNS
    + ["qrs_visibility", "detector_agreement", "baseline_step", "sqi_basSQI", "pc2", "pc3"],
    "seed": 20261212,
}

CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20_mediumselect"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Same training recipe as the strong p20 baseline, with the same extra
    # medium-stress selection score used above.  This isolates selection from
    # the broader subject111 medium-only augmentation.
    "selection_medium_stress_weight": 0.34,
    "selection_medium_subject111_shift_strength": 1.50,
    "selection_medium_subject111_good_shift_scale": 0.0,
    "selection_medium_subject111_medium_shift_scale": 1.50,
    "selection_medium_stress_nonbad_hardneg_strength": 0.78,
    "selection_medium_aug_aux_weight_nonbad": 0.30,
    "seed": 20261057,
}

CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20_precisionselect"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Same p20 data/model recipe, but select for bad/medium precision.  This
    # tests whether the old p20 checkpoint was slightly too recall-biased.
    "selection_precision_weight": 0.32,
    "selection_stress_precision_weight": 0.30,
    "selection_bad_stress_weight": 0.28,
    "seed": 20261059,
}

CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p35"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    "bad_impulse_reset_strength": 1.85,
    "bad_impulse_reset_prob": 0.35,
    "selection_stress_bad_impulse_reset_strength": 2.05,
    "bad_aux_pos_weight": 3.65,
    "bad_specificity_weight": 1.36,
    "selection_bad_stress_weight": 0.42,
    "seed": 20261013,
}

CANDIDATES["predtop20_sqiquery_subject111_sparseevent_v5_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Structural test after p20/p35 original-error audit: bad-outlier misses
    # are sparse reset/dropout/cross-channel events, so expose those waveform
    # events to the primitive bad expert instead of globally increasing bad
    # weights.
    "primitive_bank": "qrs_stress_v5",
    "primitive_bad_mix": 0.94,
    "bad_aux_pos_weight": 3.50,
    "bad_specificity_weight": 1.30,
    "selection_bad_stress_weight": 0.38,
    "seed": 20261014,
}

CANDIDATES["predtop20_sqiquery_subject111_sparseevent_v5_guarded"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_sparseevent_v5_p20"],
    # Slightly stronger sparse-event bad expert, while keeping class/gm weights
    # at the p20 balance point to avoid the p35-style medium collapse.
    "primitive_bad_mix": 1.08,
    "bad_aux_pos_weight": 3.62,
    "bad_specificity_weight": 1.42,
    "selection_bad_stress_weight": 0.40,
    "seed": 20261015,
}

CANDIDATES["predtop20_sqiquery_subject111_sparseevent_v5_badonly"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_sparseevent_v5_p20"],
    # Keep the sparse reset/dropout bank away from the good/medium classifier.
    # It only contributes through the primitive bad expert, testing whether
    # record111-like bad outliers can be recovered without poisoning the GM
    # decision surface.
    "primitive_fusion": False,
    "aux_primitive_fusion": False,
    "primitive_bad_mix": 0.88,
    "bad_aux_pos_weight": 3.58,
    "bad_specificity_weight": 1.34,
    "selection_bad_stress_weight": 0.40,
    "seed": 20261016,
}

CANDIDATES["predtop20_sqiquery_subject111_dualbank_v4gm_v5bad"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Best-of-both structural test: stable v4 primitive bank remains in the
    # GM/classification path, while the v5 sparse reset/dropout bank is used
    # only by the primitive bad expert.
    "primitive_bank": "qrs_stress_v4",
    "primitive_bad_bank": "qrs_stress_v5",
    "primitive_fusion": True,
    "aux_primitive_fusion": True,
    "primitive_bad_mix": 0.94,
    "bad_aux_pos_weight": 3.52,
    "bad_specificity_weight": 1.30,
    "selection_bad_stress_weight": 0.38,
    "seed": 20261017,
}

CANDIDATES["predtop20_sqiquery_subject111_dualbank_v4gm_v5bad_light"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_dualbank_v4gm_v5bad"],
    "primitive_bad_mix": 0.76,
    "bad_aux_pos_weight": 3.42,
    "bad_specificity_weight": 1.20,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261018,
}

CANDIDATES["predtop20_sqiquery_subject111_burstdrop_dual_p16"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # After the v5 sparse-event audit: the useful bad signal is not generic
    # sparse activity, but record111-like long low-amplitude dropout with sparse
    # reset/burst edges.  Keep the block rare and retain p20's good/medium
    # balance so non-bad outlier rows are not swallowed as bad.
    "bad_record111_burst_dropout_strength": 1.45,
    "bad_record111_burst_dropout_prob": 0.16,
    "selection_stress_bad_record111_burst_dropout_strength": 1.70,
    "selection_stress_bad_record111_burst_dropout_prob": 1.0,
    "bad_impulse_reset_strength": 1.45,
    "bad_impulse_reset_prob": 0.14,
    "bad_intermittent_contact_prob": 0.50,
    "selection_stress_bad_impulse_reset_strength": 1.65,
    "selection_stress_bad_impulse_reset_prob": 0.70,
    "bad_aux_pos_weight": 3.42,
    "bad_specificity_weight": 1.28,
    "selection_bad_stress_weight": 0.38,
    "seed": 20261019,
}

CANDIDATES["predtop20_sqiquery_subject111_burstdrop_dual_p26"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_burstdrop_dual_p16"],
    # Medium-strength version.  This should only survive if it improves bad
    # outlier recall without the v5-style false-bad explosion.
    "bad_record111_burst_dropout_strength": 1.80,
    "bad_record111_burst_dropout_prob": 0.26,
    "selection_stress_bad_record111_burst_dropout_strength": 2.05,
    "bad_impulse_reset_strength": 1.55,
    "bad_impulse_reset_prob": 0.16,
    "bad_aux_pos_weight": 3.55,
    "bad_specificity_weight": 1.38,
    "selection_bad_stress_weight": 0.42,
    "seed": 20261020,
}

CANDIDATES["predtop20_sqiquery_subject111_burstdrop_p26_mguard"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_burstdrop_dual_p26"],
    # p26 preserved bad specificity and improved original_all, but lost too
    # many original-test medium rows to good.  This is a targeted GM correction,
    # not a new bad sweep.
    "class_weight": [1.04, 1.76, 2.95],
    "gm_pair_weight": 1.10,
    "bad_aux_pos_weight": 3.42,
    "bad_specificity_weight": 1.44,
    "selection_bad_stress_weight": 0.38,
    "seed": 20261021,
}

CANDIDATES["predtop20_sqiquery_subject111_burstdrop_p16_specific"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_burstdrop_dual_p16"],
    # p16 caught more bad outlier but created false-bad in non-bad outliers.
    # Tighten the bad expert while keeping the same rare burst/dropout exposure.
    "primitive_bad_mix": 0.78,
    "bad_gate_bias_init": -0.86,
    "bad_aux_pos_weight": 3.22,
    "bad_specificity_weight": 1.74,
    "bad_specificity_margin": 0.62,
    "stress_bad_logit_mix": 0.02,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261022,
}

CANDIDATES["predtop20_eventqrs_impulsebad_dual_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Structural probe: keep the current best p20 data/loss balance, but give
    # the Transformer explicit waveform-derived event tokens so QRS visibility,
    # detector agreement, pc2, and boundary confidence do not have to emerge
    # from uniform mean-pooled patches.
    "arch": "event_qrs_sqi_query_multiscale_patch_teacher_atlas",
    "event_count": 12,
    "event_window": 111,
    "predicted_feature_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS,
    "critical_aux_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "boundary_confidence",
        "pc2",
        "sqi_basSQI",
        "baseline_step",
    ],
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.08,
    "pretrain_aux_weight": 1.25,
    "pretrain_core_aux_weight": 2.45,
    "pretrain_hard_aux_weight": 5.70,
    "pretrain_top14_aux_weight": 5.20,
    "pretrain_critical_aux_weight": 7.30,
    "aux_weight": 0.92,
    "core_aux_weight": 1.75,
    "hard_aux_weight": 4.10,
    "seed": 20261023,
}

CANDIDATES["predtop20_eventqrs_impulsebad_dual_p20_qrsheavy"] = {
    **CANDIDATES["predtop20_eventqrs_impulsebad_dual_p20"],
    # Same p20 balance, but more event tokens and a wider event window.  This
    # tests whether the missing QRS/baseline axes need beat-scale evidence
    # instead of more class pressure.
    "event_count": 18,
    "event_window": 131,
    "pretrain_hard_aux_weight": 6.10,
    "pretrain_critical_aux_weight": 8.20,
    "hard_aux_weight": 4.60,
    "bad_aux_weight": 0.72,
    "bad_specificity_weight": 1.30,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261024,
}

CANDIDATES["predtop20_qrsbank_impulsebad_dual_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Same p20 data/loss balance as the current waveform best, but replace the
    # patch body with waveform-derived QRS reliability/stat tokens.  The error
    # audit says p20 already has the right broad GM tradeoff; the missing part
    # is recovering qrs_visibility/detector/baseline/pc2-like axes without
    # poisoning non-bad specificity.
    "arch": "qrsbank_stat_token_teacher_atlas",
    "width": 128,
    "layers": 4,
    "heads": 4,
    "stat_hidden": 256,
    "primitive_bank": "qrs_stress_v4",
    "predicted_feature_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS,
    "critical_aux_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS
    + [
        "qrs_visibility",
        "qrs_visibility",
        "detector_agreement",
        "detector_agreement",
        "baseline_step",
        "sqi_basSQI",
        "pc2",
        "boundary_confidence",
    ],
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.08,
    "pretrain_aux_weight": 1.25,
    "pretrain_core_aux_weight": 2.45,
    "pretrain_hard_aux_weight": 5.80,
    "pretrain_top14_aux_weight": 5.30,
    "pretrain_critical_aux_weight": 7.60,
    "aux_weight": 0.92,
    "core_aux_weight": 1.75,
    "hard_aux_weight": 4.40,
    "bad_aux_weight": 0.74,
    "bad_aux_pos_weight": 3.42,
    "bad_specificity_weight": 1.28,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261025,
}

CANDIDATES["predfailaxis_sqiquery_subject111_impulsebad_dual_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # P20 original-test failure audit: the large errors are separable by a
    # small geometry axis set.  Keep the proven SQI-query backbone and p20 bad
    # exposure, but force the classifier and aux loss to concentrate on those
    # axes instead of spreading capacity over the full top20.
    "predicted_feature_columns": P20_FAILURE_AXIS_FEATURE_COLUMNS,
    "critical_aux_columns": P20_FAILURE_AXIS_FEATURE_COLUMNS
    + [
        "pc1",
        "pc2",
        "pca_margin",
        "boundary_confidence",
        "flatline_ratio",
        "sqi_basSQI",
        "qrs_prom_p90",
        "band_15_30",
    ],
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.10,
    "pretrain_aux_weight": 1.20,
    "pretrain_core_aux_weight": 2.35,
    "pretrain_hard_aux_weight": 5.20,
    "pretrain_top14_aux_weight": 4.80,
    "pretrain_critical_aux_weight": 8.40,
    "aux_weight": 0.88,
    "core_aux_weight": 1.70,
    "hard_aux_weight": 4.60,
    "gm_pair_weight": 0.96,
    "bad_aux_weight": 0.74,
    "bad_aux_pos_weight": 3.42,
    "bad_specificity_weight": 1.30,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261026,
}

CANDIDATES["predtop14balanced_sqiquery_subject111_impulsebad_dual_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # The reduced-feature audit says this 14-axis set is the strongest compact
    # explanatory boundary on held-out BUT.  It includes region_confidence and
    # sqi_basSQI, which p20 did not emphasize enough.  They remain teacher
    # targets/predicted features only; inference still sees waveform-derived
    # channels.
    "predicted_feature_columns": TOP14_FEATURE_COLUMNS,
    "critical_aux_columns": TOP14_FEATURE_COLUMNS
    + [
        "region_confidence",
        "region_confidence",
        "pca_margin",
        "pca_margin",
        "sqi_basSQI",
        "detector_agreement",
        "boundary_confidence",
        "qrs_visibility",
    ],
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.10,
    "pretrain_aux_weight": 1.20,
    "pretrain_core_aux_weight": 2.35,
    "pretrain_hard_aux_weight": 5.20,
    "pretrain_top14_aux_weight": 6.40,
    "pretrain_critical_aux_weight": 8.80,
    "aux_weight": 0.90,
    "core_aux_weight": 1.72,
    "hard_aux_weight": 4.70,
    "top14_aux_weight": 5.60,
    "gm_pair_weight": 0.98,
    "bad_aux_weight": 0.74,
    "bad_aux_pos_weight": 3.42,
    "bad_specificity_weight": 1.30,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261027,
}

CANDIDATES["waveprimtoken_v5_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Full waveform-computable primitive bank as Transformer tokens.  This is
    # the cleanest follow-up to the feasibility audit: no atlas/neighborhood
    # features are fused or predicted, only statistics derivable from the input
    # waveform itself.  The p20 data/loss balance is kept because it is still
    # the best original-test waveform frontier.
    "arch": "primitive_token_teacher_atlas",
    "width": 128,
    "layers": 4,
    "heads": 4,
    "stat_hidden": 320,
    "primitive_token_bank": "qrs_stress_v5",
    "primitive_token_chunk": 16,
    "predicted_feature_fusion": "configured",
    "predicted_feature_columns": WAVEFORM_COMPUTABLE_TEACHER_COLUMNS,
    "critical_aux_columns": WAVEFORM_COMPUTABLE_TEACHER_COLUMNS + WAVEFORM_HARD_TEACHER_COLUMNS,
    "feature_teacher_columns": WAVEFORM_COMPUTABLE_TEACHER_COLUMNS,
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.10,
    "pretrain_aux_weight": 1.25,
    "pretrain_core_aux_weight": 2.30,
    "pretrain_hard_aux_weight": 5.40,
    "pretrain_top14_aux_weight": 0.00,
    "pretrain_critical_aux_weight": 7.60,
    "pretrain_rank_weight": 1.10,
    "aux_weight": 0.86,
    "core_aux_weight": 1.62,
    "hard_aux_weight": 4.35,
    "top14_aux_weight": 0.00,
    "bad_aux_weight": 0.72,
    "bad_aux_pos_weight": 3.40,
    "bad_specificity_weight": 1.34,
    "bad_specificity_margin": 0.72,
    "selection_bad_stress_weight": 0.34,
    "seed": 20261028,
}

CANDIDATES["waveprimtoken_v5_p20_badguard"] = {
    **CANDIDATES["waveprimtoken_v5_p20"],
    # Same primitive-token encoder, but with a little less bad pressure and a
    # stronger non-bad guard.  This tests whether the p20 bad-outlier recall can
    # improve without the false-bad explosion seen in prior stress variants.
    "class_weight": [1.04, 1.42, 2.82],
    "bad_aux_weight": 0.62,
    "bad_aux_pos_weight": 3.05,
    "bad_specificity_weight": 1.62,
    "bad_specificity_margin": 0.78,
    "selection_bad_stress_weight": 0.26,
    "seed": 20261029,
}

CANDIDATES["wavecomp_sqiquery_p20"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Keep the strongest p20 SQI-query patch Transformer, but stop asking it
    # to chase atlas/neighborhood geometry (pca_margin, pc1, knn purity) as
    # fused predicted features.  The teacher loss now focuses on quantities a
    # waveform-only encoder can plausibly recover.
    "predicted_feature_fusion": "configured",
    "predicted_feature_columns": WAVEFORM_COMPUTABLE_TEACHER_COLUMNS,
    "critical_aux_columns": WAVEFORM_COMPUTABLE_TEACHER_COLUMNS + WAVEFORM_HARD_TEACHER_COLUMNS,
    "feature_teacher_columns": WAVEFORM_COMPUTABLE_TEACHER_COLUMNS,
    "pretrain_top14_aux_weight": 0.00,
    "top14_aux_weight": 0.00,
    "pretrain_critical_aux_weight": 7.40,
    "pretrain_hard_aux_weight": 5.30,
    "hard_aux_weight": 4.25,
    "seed": 20261030,
}

CANDIDATES["wavecomp_sqiquery_v5_p20_badguard"] = {
    **CANDIDATES["wavecomp_sqiquery_p20"],
    # Same waveform-computable teacher target set, but expose the sparse
    # reset/dropout primitive bank only as the existing weak bad expert.  Keep
    # bad pressure guarded because earlier v5 variants bought bad recall by
    # spending too much good/medium specificity.
    "primitive_bank": "qrs_stress_v5",
    "primitive_bad_mix": 0.78,
    "bad_aux_weight": 0.66,
    "bad_aux_pos_weight": 3.18,
    "bad_specificity_weight": 1.56,
    "bad_specificity_margin": 0.76,
    "selection_bad_stress_weight": 0.28,
    "seed": 20261031,
}

CANDIDATES["p20_sqiquery_primctx_v5_light"] = {
    **CANDIDATES["predtop20_sqiquery_subject111_impulsebad_dual_p20"],
    # Hybrid test: keep the current best p20 patch/SQI-query morphology path,
    # but add a small primitive-token context Transformer over qrs_stress_v5.
    # This is not a replacement encoder; it is a late-fusion residual context
    # for sparse bad events and global baseline/detail evidence.
    "primitive_context_fusion": True,
    "aux_primitive_context_fusion": True,
    "primitive_context_bank": "qrs_enhanced_v2",
    "primitive_context_width": 48,
    "primitive_context_layers": 1,
    "primitive_context_heads": 4,
    "primitive_context_hidden": 160,
    "primitive_context_chunk": 32,
    "pretrain_critical_aux_weight": 8.20,
    "hard_aux_weight": 4.70,
    "bad_aux_weight": 0.72,
    "bad_aux_pos_weight": 3.35,
    "bad_specificity_weight": 1.34,
    "bad_specificity_margin": 0.66,
    "selection_bad_stress_weight": 0.32,
    "seed": 20261032,
}

CANDIDATES["p20_sqiquery_primctx_v5_badguard"] = {
    **CANDIDATES["p20_sqiquery_primctx_v5_light"],
    # More conservative bad path: keep context evidence but make false-bad
    # specificity more expensive.  This targets the previous p16/v5 failure
    # mode where bad outlier recall improved at the cost of many non-bad rows.
    "bad_aux_weight": 0.62,
    "bad_aux_pos_weight": 3.05,
    "bad_specificity_weight": 1.70,
    "bad_specificity_margin": 0.80,
    "selection_bad_stress_weight": 0.24,
    "primitive_context_bank": "qrs_stress_v3",
    "primitive_context_chunk": 48,
    "seed": 20261033,
}

CANDIDATES["predtop25_sqiquery_qrsfocus_goodguard"] = {
    **CANDIDATES["predtop20_sqiquery_boundary_pretrain"],
    "predicted_feature_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS,
    "critical_aux_columns": TOP25_QRS_GEOMETRY_FEATURE_COLUMNS
    + ["qrs_visibility", "qrs_visibility", "detector_agreement", "detector_agreement", "boundary_confidence", "pc2"],
    "primitive_bank": "qrs_enhanced_v2",
    "aux_pretrain_epochs": 4,
    "pretrain_classification_weight": 0.08,
    "pretrain_core_aux_weight": 2.60,
    "pretrain_hard_aux_weight": 5.50,
    "pretrain_top14_aux_weight": 5.50,
    "pretrain_critical_aux_weight": 7.00,
    "pretrain_rank_weight": 1.25,
    "class_weight": [1.18, 1.32, 3.05],
    "gm_pair_weight": 0.78,
    "bad_aux_weight": 0.94,
    "bad_specificity_weight": 2.70,
    "bad_specificity_margin": 0.74,
    "seed": 20260974,
}


def load_arch_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_architecture_search", ARCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ARCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCH = load_arch_module()
GEOM = ARCH.GEOM
FEATURE_COLUMNS = list(ARCH.FEATURE_COLUMNS)
TEACHER_IDX = [FEATURE_COLUMNS.index(c) for c in TEACHER_COLUMNS if c in FEATURE_COLUMNS]
CORE_IDX = [FEATURE_COLUMNS.index(c) for c in CORE_COLUMNS if c in FEATURE_COLUMNS]
HARD_IDX = [FEATURE_COLUMNS.index(c) for c in HARD_FEATURE_COLUMNS if c in FEATURE_COLUMNS]
TOP14_IDX = [FEATURE_COLUMNS.index(c) for c in TOP14_FEATURE_COLUMNS if c in FEATURE_COLUMNS]


class FeatureLogitTeacher:
    """Synthetic-feature teacher used only for training-time distillation."""

    def __init__(self, aux: np.ndarray, y: np.ndarray, columns: list[str], seed: int):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.columns = [c for c in columns if c in FEATURE_COLUMNS]
        if not self.columns:
            raise ValueError("feature teacher requested with no valid columns")
        self.indices = [FEATURE_COLUMNS.index(c) for c in self.columns]
        self.model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.055,
            max_leaf_nodes=31,
            l2_regularization=0.025,
            random_state=int(seed),
        )
        x = np.nan_to_num(aux[:, self.indices].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(x, y.astype(np.int64))
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)

    def predict_proba(self, aux: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(aux[:, self.indices].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        raw = self.model.predict_proba(x)
        out = np.zeros((x.shape[0], 3), dtype=np.float32)
        for j, cls in enumerate(self.classes_):
            if 0 <= int(cls) < 3:
                out[:, int(cls)] = raw[:, j]
        out = np.clip(out, 1e-5, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        return out.astype(np.float32)


class PrimitiveLogitTeacher:
    """Waveform-primitive teacher used only for training-time distillation.

    The teacher sees no BUT rows and no 47-column sidecar at inference time. It
    is fit on synthetic-train waveform-derived primitive stats, then provides a
    soft boundary target for each training batch from the batch waveform itself.
    This tests whether tree-like bad-stress splits can be transferred into the
    Transformer without making the tree a final classifier.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, bank: str, seed: int):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.preprocessing import StandardScaler

        self.bank = str(bank)
        self.scaler = StandardScaler()
        feats = self._collect_numpy(x, batch_size=512)
        feats_z = self.scaler.fit_transform(feats)
        self.model = ExtraTreesClassifier(
            n_estimators=96,
            max_features=0.55,
            max_depth=14,
            min_samples_leaf=2,
            class_weight={0: 1.0, 1: 1.35, 2: 3.2},
            random_state=int(seed),
            n_jobs=-1,
        )
        self.model.fit(feats_z, y.astype(np.int64))
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)

    def _collect_numpy(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32))
                rows.append(primitive_waveform_stats(xb, self.bank).cpu().numpy())
        return np.concatenate(rows, axis=0)

    def predict_proba_from_waveform(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            feats = primitive_waveform_stats(x.detach().cpu(), self.bank).cpu().numpy()
        feats_z = self.scaler.transform(feats)
        raw = self.model.predict_proba(feats_z)
        out = np.zeros((feats.shape[0], 3), dtype=np.float32)
        for j, cls in enumerate(self.classes_):
            if 0 <= int(cls) < 3:
                out[:, int(cls)] = raw[:, j]
        out = np.clip(out, 1e-5, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        return out.astype(np.float32)


@dataclass
class EvalPayload:
    report: dict[str, Any]
    probs: np.ndarray
    pred: np.ndarray
    aux_pred: np.ndarray
    embeddings: np.ndarray
    y: np.ndarray


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(tokens)
        score = torch.einsum("btd,d->bt", h, self.query) / math.sqrt(max(1, h.shape[-1]))
        weight = torch.softmax(score, dim=1)
        return torch.einsum("bt,btd->bd", weight, tokens)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


def make_encoder_layer(width: int, heads: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * 4,
        dropout=0.08,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class PatchStudentEncoder(nn.Module):
    """PatchTST-style channel-aware patch Transformer."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int):
        super().__init__()
        self.in_ch = in_ch
        self.patch = patch
        self.patch_proj = nn.Linear(patch, width)
        self.channel_embed = nn.Parameter(torch.randn(in_ch, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pos = PositionalEncoding(width, max_len=768)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 3

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        keep = (n // self.patch) * self.patch
        x = x[:, :, :keep].reshape(b, c, keep // self.patch, self.patch)
        tokens = self.patch_proj(x).reshape(b, c * (keep // self.patch), -1)
        ch = self.channel_embed[:, None, :].expand(c, keep // self.patch, -1).reshape(1, c * (keep // self.patch), -1)
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls, tokens + ch], dim=1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenize(x)
        return self.encoder(self.pos(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        cls = tokens[:, 0, :]
        body = tokens[:, 1:, :]
        return torch.cat([cls, self.pool(body), body.mean(dim=1)], dim=1)


class MaskedGeometryMAEEncoder(PatchStudentEncoder):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int):
        super().__init__(in_ch, width, layers, heads, patch)
        self.mask_token = nn.Parameter(torch.zeros(width))
        self.decoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, patch))

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        b, c, n = x.shape
        keep = (n // self.patch) * self.patch
        target = x[:, :, :keep].reshape(b, c * (keep // self.patch), self.patch)
        tokens = self.tokenize(x)
        patch_tokens = tokens[:, 1:, :]
        mask = None
        if self.training and mask_ratio > 0:
            mask = torch.rand(patch_tokens.shape[:2], device=x.device) < float(mask_ratio)
            patch_tokens = torch.where(mask[:, :, None], self.mask_token[None, None, :], patch_tokens)
            tokens = torch.cat([tokens[:, :1, :], patch_tokens], dim=1)
        encoded = self.encoder(self.pos(tokens))
        cls = encoded[:, 0, :]
        body = encoded[:, 1:, :]
        feat = torch.cat([cls, self.pool(body), body.mean(dim=1)], dim=1)
        recon = self.decoder(body)
        return feat, recon, mask if mask is not None else torch.ones(body.shape[:2], dtype=torch.bool, device=x.device)


def patch_stat_features(x: torch.Tensor, patch: int) -> torch.Tensor:
    eps = 1e-6
    b, c, n = x.shape
    keep = (n // patch) * patch
    chunks = x[:, :, :keep].reshape(b, c, keep // patch, patch)
    diff = chunks[..., 1:] - chunks[..., :-1]
    mean = chunks.mean(dim=3)
    std = chunks.std(dim=3)
    mean_abs = chunks.abs().mean(dim=3)
    rms = torch.sqrt((chunks * chunks).mean(dim=3) + eps)
    ptp = chunks.amax(dim=3) - chunks.amin(dim=3)
    diff_abs = diff.abs().mean(dim=3)
    flat = (diff.abs() < 0.015).float().mean(dim=3)
    low_amp = (chunks.abs() < 0.050).float().mean(dim=3)
    zcr = ((chunks[..., 1:] * chunks[..., :-1]) < 0).float().mean(dim=3)
    slope = (chunks[..., -1] - chunks[..., 0]).abs()
    return torch.stack([mean, std, mean_abs, rms, ptp, diff_abs, flat, low_amp, zcr, slope], dim=3)


class StatFedPatchTransformer(nn.Module):
    """Patch Transformer with waveform-derived local/global statistics injected at tokenization."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int, stat_hidden: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.patch = int(patch)
        self.patch_proj = nn.Linear(self.patch, width)
        self.patch_stat_proj = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, width), nn.GELU())
        self.channel_embed = nn.Parameter(torch.randn(in_ch, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.global_stat = nn.Sequential(
            nn.LayerNorm(in_ch * 31),
            nn.Linear(in_ch * 31, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, width),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=1024)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 4

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        keep = (n // self.patch) * self.patch
        patch_count = keep // self.patch
        chunks = x[:, :, :keep].reshape(b, c, patch_count, self.patch)
        raw_tokens = self.patch_proj(chunks).reshape(b, c * patch_count, -1)
        stat_tokens = self.patch_stat_proj(patch_stat_features(x, self.patch)).reshape(b, c * patch_count, -1)
        ch = self.channel_embed[:, None, :].expand(c, patch_count, -1).reshape(1, c * patch_count, -1)
        global_stats = torch.cat([waveform_stats_v2(x), qrs_detail_stats(x)], dim=1)
        global_token = self.global_stat(global_stats).unsqueeze(1)
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls, global_token, raw_tokens + stat_tokens + ch], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.pos(self.tokenize(x)))
        cls = tokens[:, 0, :]
        global_token = tokens[:, 1, :]
        body = tokens[:, 2:, :]
        return torch.cat([cls, global_token, self.pool(body), body.mean(dim=1)], dim=1)


class MultiScaleStatPatchTransformer(nn.Module):
    """Patch Transformer with raw patches plus multi-scale waveform statistic tokens."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int, stat_hidden: int, scales: list[int]):
        super().__init__()
        self.in_ch = int(in_ch)
        self.patch = int(patch)
        self.scales = [int(s) for s in scales if int(s) > 0]
        self.patch_proj = nn.Linear(self.patch, width)
        self.patch_stat_proj = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, width), nn.GELU())
        self.scale_stat_proj = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, width), nn.GELU(), nn.Linear(width, width))
        self.channel_embed = nn.Parameter(torch.randn(in_ch, width) * 0.02)
        self.raw_scale_embed = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.scale_embed = nn.Parameter(torch.randn(max(1, len(self.scales)), width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.global_stat = nn.Sequential(
            nn.LayerNorm(in_ch * 31),
            nn.Linear(in_ch * 31, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, width),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=2048)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 4

    def _channel_tokens(self, count: int) -> torch.Tensor:
        return self.channel_embed[:, None, :].expand(self.in_ch, count, -1).reshape(1, self.in_ch * count, -1)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        keep = (n // self.patch) * self.patch
        patch_count = keep // self.patch
        chunks = x[:, :, :keep].reshape(b, c, patch_count, self.patch)
        raw_tokens = self.patch_proj(chunks).reshape(b, c * patch_count, -1)
        stat_tokens = self.patch_stat_proj(patch_stat_features(x, self.patch)).reshape(b, c * patch_count, -1)
        raw_tokens = raw_tokens + stat_tokens + self._channel_tokens(patch_count) + self.raw_scale_embed
        parts = [raw_tokens]
        for i, scale in enumerate(self.scales):
            scale_stats = patch_stat_features(x, scale)
            count = scale_stats.shape[2]
            scale_tokens = self.scale_stat_proj(scale_stats).reshape(b, c * count, -1)
            scale_tokens = scale_tokens + self._channel_tokens(count) + self.scale_embed[i].view(1, 1, -1)
            parts.append(scale_tokens)
        global_stats = torch.cat([waveform_stats_v2(x), qrs_detail_stats(x)], dim=1)
        global_token = self.global_stat(global_stats).unsqueeze(1)
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls, global_token, *parts], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.pos(self.tokenize(x)))
        cls = tokens[:, 0, :]
        global_token = tokens[:, 1, :]
        body = tokens[:, 2:, :]
        return torch.cat([cls, global_token, self.pool(body), body.mean(dim=1)], dim=1)


def waveform_stress_stats(x: torch.Tensor) -> torch.Tensor:
    """Waveform-derived stress bank for flat/contact/drift shells.

    These are computed from the inference waveform itself, not from external
    SQI/geometry sidecars.  The bank targets the original bad-stress gaps:
    long low-derivative spans, low zero-crossing, large baseline drift, and
    low-detail/contact-like sections.
    """

    eps = 1e-6
    diff = x[:, :, 1:] - x[:, :, :-1]
    abs_x = x.abs()
    abs_d = diff.abs()
    flat005 = (abs_d < 0.005).float().mean(dim=2)
    flat015 = (abs_d < 0.015).float().mean(dim=2)
    flat030 = (abs_d < 0.030).float().mean(dim=2)
    low015 = (abs_x < 0.015).float().mean(dim=2)
    low050 = (abs_x < 0.050).float().mean(dim=2)
    low100 = (abs_x < 0.100).float().mean(dim=2)
    zcr = ((x[:, :, 1:] * x[:, :, :-1]) < 0).float().mean(dim=2)
    dzcr = ((diff[:, :, 1:] * diff[:, :, :-1]) < 0).float().mean(dim=2)

    flat_mask = (abs_d < 0.015).float()
    low_mask = (abs_x < 0.050).float()
    flat_run101 = F.avg_pool1d(flat_mask, kernel_size=101, stride=1, padding=50)
    flat_run251 = F.avg_pool1d(flat_mask, kernel_size=251, stride=1, padding=125)
    low_run101 = F.avg_pool1d(low_mask, kernel_size=101, stride=1, padding=50)
    low_run251 = F.avg_pool1d(low_mask, kernel_size=251, stride=1, padding=125)
    flat_run101_max = flat_run101.amax(dim=2)
    flat_run251_max = flat_run251.amax(dim=2)
    low_run101_max = low_run101.amax(dim=2)
    low_run251_max = low_run251.amax(dim=2)
    flat_run101_mean = flat_run101.mean(dim=2)
    low_run101_mean = low_run101.mean(dim=2)

    smooth101 = F.avg_pool1d(x, kernel_size=101, stride=1, padding=50)
    smooth251 = F.avg_pool1d(x, kernel_size=251, stride=1, padding=125)
    baseline_span101 = smooth101.amax(dim=2) - smooth101.amin(dim=2)
    baseline_span251 = smooth251.amax(dim=2) - smooth251.amin(dim=2)
    baseline_slope251 = (smooth251[:, :, -1] - smooth251[:, :, 0]).abs()
    baseline_curve251 = smooth251.std(dim=2)

    rms = torch.sqrt((x * x).mean(dim=2) + eps)
    high31 = x - F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
    high101 = x - smooth101
    detail_ratio31 = torch.sqrt((high31 * high31).mean(dim=2) + eps) / (rms + eps)
    detail_ratio101 = torch.sqrt((high101 * high101).mean(dim=2) + eps) / (rms + eps)
    k01 = max(1, int(abs_d.shape[2] * 0.01))
    k10 = max(1, int(abs_d.shape[2] * 0.10))
    diff_top01 = torch.topk(abs_d, k=k01, dim=2).values.mean(dim=2)
    diff_top10 = torch.topk(abs_d, k=k10, dim=2).values.mean(dim=2)
    ptp = x.amax(dim=2) - x.amin(dim=2)
    top_abs10 = torch.topk(abs_x, k=max(1, int(abs_x.shape[2] * 0.10)), dim=2).values.mean(dim=2)

    return torch.cat(
        [
            flat005,
            flat015,
            flat030,
            low015,
            low050,
            low100,
            zcr,
            dzcr,
            flat_run101_max,
            flat_run251_max,
            low_run101_max,
            low_run251_max,
            baseline_span101,
            baseline_span251,
            baseline_slope251,
            baseline_curve251,
            detail_ratio31,
            detail_ratio101,
            diff_top01,
            diff_top10,
            ptp,
            top_abs10,
            flat_run101_mean,
            low_run101_mean,
        ],
        dim=1,
    )


class StressBankWrapper(nn.Module):
    """Append a waveform-computed stress token bank to a Transformer encoder."""

    def __init__(self, base: nn.Module, base_dim: int, in_ch: int, stat_hidden: int):
        super().__init__()
        self.base = base
        self.stress_dim = int(in_ch) * 24
        self.stress_branch = nn.Sequential(
            nn.LayerNorm(self.stress_dim),
            nn.Linear(self.stress_dim, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, stat_hidden),
            nn.GELU(),
        )
        self.out_dim = int(base_dim) + int(stat_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.base(x), self.stress_branch(waveform_stress_stats(x))], dim=1)


class StressTokenStatFedPatchTransformer(StatFedPatchTransformer):
    """Stat-fed patch Transformer with stress bank inserted as an attention token."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int, stat_hidden: int):
        super().__init__(in_ch, width, layers, heads, patch, stat_hidden)
        self.stress_token = nn.Sequential(nn.LayerNorm(in_ch * 24), nn.Linear(in_ch * 24, width), nn.GELU())
        self.out_dim = width * 5

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        base_tokens = super().tokenize(x)
        stress = self.stress_token(waveform_stress_stats(x)).unsqueeze(1)
        return torch.cat([base_tokens[:, :2, :], stress, base_tokens[:, 2:, :]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.pos(self.tokenize(x)))
        cls = tokens[:, 0, :]
        global_token = tokens[:, 1, :]
        stress_token = tokens[:, 2, :]
        body = tokens[:, 3:, :]
        return torch.cat([cls, global_token, stress_token, self.pool(body), body.mean(dim=1)], dim=1)


class StressTokenMultiScaleStatPatchTransformer(MultiScaleStatPatchTransformer):
    """Multi-scale stat-patch Transformer with stress bank as an attention token."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int, stat_hidden: int, scales: list[int]):
        super().__init__(in_ch, width, layers, heads, patch, stat_hidden, scales)
        self.stress_token = nn.Sequential(nn.LayerNorm(in_ch * 24), nn.Linear(in_ch * 24, width), nn.GELU())
        self.out_dim = width * 5

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        base_tokens = super().tokenize(x)
        stress = self.stress_token(waveform_stress_stats(x)).unsqueeze(1)
        return torch.cat([base_tokens[:, :2, :], stress, base_tokens[:, 2:, :]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.pos(self.tokenize(x)))
        cls = tokens[:, 0, :]
        global_token = tokens[:, 1, :]
        stress_token = tokens[:, 2, :]
        body = tokens[:, 3:, :]
        return torch.cat([cls, global_token, stress_token, self.pool(body), body.mean(dim=1)], dim=1)


class SQIQueryMultiScaleTransformer(MultiScaleStatPatchTransformer):
    """Multi-scale patch Transformer with explicit SQI query tokens.

    The query tokens are learned from waveform-derived patch/stat tokens at
    inference time.  They are not external feature inputs; they are an
    architectural contract that gives the auxiliary heads dedicated capacity
    for QRS visibility, detector agreement, baseline/flatline, band/detail,
    bad stress, and good/medium boundary geometry.
    """

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int, stat_hidden: int, scales: list[int]):
        super().__init__(in_ch, width, layers, heads, patch, stat_hidden, scales)
        self.query_names = [
            "qrs_visibility",
            "detector_agreement",
            "baseline_step",
            "band_detail",
            "flatline_lowamp",
            "bad_stress",
            "gm_boundary",
        ]
        self.sqi_queries = nn.Parameter(torch.randn(1, len(self.query_names), width) * 0.02)
        self.query_type_embed = nn.Parameter(torch.randn(1, len(self.query_names), width) * 0.02)
        self.out_dim = width * (2 + len(self.query_names) + 3)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        base_tokens = super().tokenize(x)
        query = self.sqi_queries.expand(x.shape[0], -1, -1) + self.query_type_embed
        return torch.cat([base_tokens[:, :2, :], query, base_tokens[:, 2:, :]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.pos(self.tokenize(x)))
        cls = tokens[:, 0, :]
        global_token = tokens[:, 1, :]
        query = tokens[:, 2 : 2 + len(self.query_names), :].reshape(x.shape[0], -1)
        body = tokens[:, 2 + len(self.query_names) :, :]
        max_pool = body.amax(dim=1)
        return torch.cat([cls, global_token, query, self.pool(body), body.mean(dim=1), max_pool], dim=1)


def qrs_event_patch_features(x: torch.Tensor, event_count: int = 12, window: int = 81) -> torch.Tensor:
    """Top-k QRS-like event window features, waveform-only.

    Global qrs_visibility was the hardest target to recover.  These event
    features expose local peak-centered evidence: peak prominence, local detail,
    flat/low-amplitude ratios, baseline span, asymmetry, and normalized event
    position.  The top-k selection is a waveform-derived tokenization step; no
    external detector or SQI sidecar is used.
    """

    eps = 1e-6
    b, c, n = x.shape
    k = max(1, min(int(event_count), int(n)))
    w = max(9, int(window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    smooth = F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
    high = (x - smooth).abs()
    idx = torch.topk(high, k=k, dim=2).indices
    offsets = torch.arange(-pad, pad + 1, device=x.device).view(1, 1, 1, w)
    gather_idx = (idx.unsqueeze(-1) + offsets + pad).clamp(0, n + 2 * pad - 1)
    xp = F.pad(x, (pad, pad), mode="replicate")
    hp = F.pad(high, (pad, pad), mode="replicate")
    xw = torch.gather(xp.unsqueeze(2).expand(-1, -1, k, -1), 3, gather_idx)
    hw = torch.gather(hp.unsqueeze(2).expand(-1, -1, k, -1), 3, gather_idx)
    dw = xw[:, :, :, 1:] - xw[:, :, :, :-1]
    abs_w = xw.abs()
    abs_d = dw.abs()
    mean_abs = abs_w.mean(dim=3)
    rms = torch.sqrt((xw * xw).mean(dim=3) + eps)
    ptp = xw.amax(dim=3) - xw.amin(dim=3)
    high_rms = torch.sqrt((hw * hw).mean(dim=3) + eps)
    diff_rms = torch.sqrt((dw * dw).mean(dim=3) + eps)
    flat_ratio = (abs_d < 0.015).float().mean(dim=3)
    low_amp_ratio = (abs_w < 0.050).float().mean(dim=3)
    center_abs = xw[:, :, :, pad].abs()
    prom_ratio = center_abs / (mean_abs + eps)
    left = xw[:, :, :, :pad]
    right = xw[:, :, :, pad + 1 :]
    asym = (left.abs().mean(dim=3) - right.abs().mean(dim=3)).abs() / (mean_abs + eps)
    baseline_span = F.avg_pool1d(xw.reshape(b * c * k, 1, w), kernel_size=min(31, w), stride=1, padding=min(31, w) // 2)
    baseline_span = baseline_span.reshape(b, c, k, -1)
    baseline_span = baseline_span.amax(dim=3) - baseline_span.amin(dim=3)
    pos = idx.float() / max(1.0, float(n - 1))
    return torch.stack(
        [
            mean_abs,
            rms,
            ptp,
            high_rms,
            diff_rms,
            flat_ratio,
            low_amp_ratio,
            prom_ratio,
            asym,
            baseline_span,
            pos,
        ],
        dim=3,
    )


class EventQRSQueryMultiScaleTransformer(SQIQueryMultiScaleTransformer):
    """SQI-query Transformer with explicit local QRS/event tokens."""

    def __init__(
        self,
        in_ch: int,
        width: int,
        layers: int,
        heads: int,
        patch: int,
        stat_hidden: int,
        scales: list[int],
        event_count: int = 12,
        event_window: int = 81,
    ):
        super().__init__(in_ch, width, layers, heads, patch, stat_hidden, scales)
        self.event_count = int(event_count)
        self.event_window = int(event_window)
        self.event_proj = nn.Sequential(nn.LayerNorm(11), nn.Linear(11, width), nn.GELU(), nn.Linear(width, width))
        self.event_type_embed = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.event_rank_embed = nn.Parameter(torch.randn(1, max(1, self.event_count), width) * 0.02)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        base_tokens = super().tokenize(x)
        b = x.shape[0]
        event_feat = qrs_event_patch_features(x, self.event_count, self.event_window)
        event_tokens = self.event_proj(event_feat).reshape(b, self.in_ch * self.event_count, -1)
        event_tokens = event_tokens + self._channel_tokens(self.event_count) + self.event_type_embed
        rank = self.event_rank_embed[:, : self.event_count, :].repeat(1, self.in_ch, 1)
        event_tokens = event_tokens + rank
        # Insert event tokens right after SQI query tokens so the query tokens
        # can attend to localized QRS evidence before the dense patch body.
        q_end = 2 + len(self.query_names)
        return torch.cat([base_tokens[:, :q_end, :], event_tokens, base_tokens[:, q_end:, :]], dim=1)


class StatBankTokenTransformer(nn.Module):
    """Transformer over waveform-derived statistic groups.

    This is still waveform-only at inference: the three token groups are
    computed directly from the input trace and then contextualized by a small
    Transformer.  It tests whether high-value SQI-like summaries are easier to
    transfer than dense patch tokens.
    """

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, stat_hidden: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.wave_proj = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, width), nn.GELU())
        self.qrs_proj = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, width), nn.GELU())
        self.stress_proj = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, width), nn.GELU())
        self.channel_embed = nn.Parameter(torch.randn(in_ch, width) * 0.02)
        self.group_embed = nn.Parameter(torch.randn(3, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pos = PositionalEncoding(width, max_len=32)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.summary = nn.Sequential(
            nn.LayerNorm(in_ch * 55),
            nn.Linear(in_ch * 55, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, width),
            nn.GELU(),
        )
        self.out_dim = width * 4

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        wave = waveform_stats_v2(x).reshape(b, self.in_ch, 16)
        qrs = qrs_detail_stats(x).reshape(b, self.in_ch, 15)
        stress = waveform_stress_stats(x).reshape(b, self.in_ch, 24)
        pieces = [
            self.wave_proj(wave) + self.group_embed[0].view(1, 1, -1),
            self.qrs_proj(qrs) + self.group_embed[1].view(1, 1, -1),
            self.stress_proj(stress) + self.group_embed[2].view(1, 1, -1),
        ]
        tokens = []
        ch = self.channel_embed.view(1, self.in_ch, -1)
        for item in pieces:
            tokens.append(item + ch)
        return torch.cat(tokens, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stat_tokens = self._tokens(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = self.encoder(self.pos(torch.cat([cls, stat_tokens], dim=1)))
        summary = self.summary(torch.cat([waveform_stats_v2(x), qrs_detail_stats(x), waveform_stress_stats(x)], dim=1))
        body = tokens[:, 1:, :]
        return torch.cat([tokens[:, 0, :], self.pool(body), body.mean(dim=1), summary], dim=1)


class QRSBankStatTokenTransformer(nn.Module):
    """Transformer over waveform-derived SQI primitive tokens including QRS reliability.

    This intentionally tests whether the failure is in raw patch learning rather
    than in the final classifier.  All tokens are computed from the inference
    waveform; no atlas/SQI sidecar is used as input.
    """

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, stat_hidden: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.wave_proj = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, width), nn.GELU())
        self.qrs_proj = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, width), nn.GELU())
        self.stress_proj = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, width), nn.GELU())
        self.qrsbank_proj = nn.Sequential(nn.LayerNorm(19), nn.Linear(19, width), nn.GELU())
        self.channel_embed = nn.Parameter(torch.randn(in_ch, width) * 0.02)
        self.group_embed = nn.Parameter(torch.randn(4, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pos = PositionalEncoding(width, max_len=40)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.summary = nn.Sequential(
            nn.LayerNorm(in_ch * 74),
            nn.Linear(in_ch * 74, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, width),
            nn.GELU(),
        )
        self.out_dim = width * 4

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        wave = waveform_stats_v2(x).reshape(b, self.in_ch, 16)
        qrs = qrs_detail_stats(x).reshape(b, self.in_ch, 15)
        stress = waveform_stress_stats(x).reshape(b, self.in_ch, 24)
        qrsbank = qrs_visibility_bank_stats(x).reshape(b, self.in_ch, 19)
        pieces = [
            self.wave_proj(wave) + self.group_embed[0].view(1, 1, -1),
            self.qrs_proj(qrs) + self.group_embed[1].view(1, 1, -1),
            self.stress_proj(stress) + self.group_embed[2].view(1, 1, -1),
            self.qrsbank_proj(qrsbank) + self.group_embed[3].view(1, 1, -1),
        ]
        ch = self.channel_embed.view(1, self.in_ch, -1)
        return torch.cat([item + ch for item in pieces], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stat_tokens = self._tokens(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = self.encoder(self.pos(torch.cat([cls, stat_tokens], dim=1)))
        summary = self.summary(primitive_waveform_stats(x, "qrs_enhanced"))
        body = tokens[:, 1:, :]
        return torch.cat([tokens[:, 0, :], self.pool(body), body.mean(dim=1), summary], dim=1)


class PrimitiveTokenTransformer(nn.Module):
    """Transformer over a full bank of waveform-computable primitive statistics.

    Earlier stat-token variants exposed only a few grouped SQI-like summaries.
    The deterministic waveform proxy audit showed that a much wider bank of
    waveform-derived primitives carries most of the recoverable signal, while
    atlas/neighborhood geometry is not available from a single trace.  This
    encoder keeps inference waveform-only by computing those primitives from
    ``x`` and presenting them as chunked tokens to a Transformer.
    """

    def __init__(
        self,
        in_ch: int,
        width: int,
        layers: int,
        heads: int,
        stat_hidden: int,
        bank: str = "qrs_stress_v5",
        chunk: int = 16,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.bank = str(bank)
        self.chunk = max(4, int(chunk))
        self.primitive_dim = self.in_ch * primitive_bank_per_channel(self.bank)
        self.padded_dim = int(math.ceil(self.primitive_dim / self.chunk) * self.chunk)
        self.pad_dim = self.padded_dim - self.primitive_dim
        self.num_tokens = self.padded_dim // self.chunk
        self.chunk_norm = nn.LayerNorm(self.chunk)
        self.chunk_proj = nn.Sequential(nn.Linear(self.chunk, width), nn.GELU(), nn.Linear(width, width))
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.token_type = nn.Parameter(torch.randn(1, self.num_tokens, width) * 0.02)
        self.pos = PositionalEncoding(width, max_len=max(512, self.num_tokens + 4))
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.summary = nn.Sequential(
            nn.LayerNorm(self.primitive_dim),
            nn.Linear(self.primitive_dim, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, width),
            nn.GELU(),
        )
        self.out_dim = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = primitive_waveform_stats(x, self.bank)
        if self.pad_dim:
            padded = F.pad(stats, (0, self.pad_dim))
        else:
            padded = stats
        chunks = padded.reshape(x.shape[0], self.num_tokens, self.chunk)
        stat_tokens = self.chunk_proj(self.chunk_norm(chunks)) + self.token_type
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = self.encoder(self.pos(torch.cat([cls, stat_tokens], dim=1)))
        body = tokens[:, 1:, :]
        summary = self.summary(stats)
        return torch.cat([tokens[:, 0, :], self.pool(body), body.mean(dim=1), summary], dim=1)


class ConvSubsampleEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=256)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).transpose(1, 2)
        tokens = self.encoder(self.pos(tokens))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1)], dim=1)


class StatTokenTransformerV2(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int, stat_hidden: int):
        super().__init__()
        self.encoder = ConvSubsampleEncoder(in_ch, width, layers, heads)
        self.stat_dim = in_ch * 16
        self.stat_branch = nn.Sequential(
            nn.LayerNorm(self.stat_dim),
            nn.Linear(self.stat_dim, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, stat_hidden),
            nn.GELU(),
        )
        self.out_dim = self.encoder.out_dim + stat_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder(x), self.stat_branch(waveform_stats_v2(x))], dim=1)


class MorphologyTokenTransformer(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_ch, width, kernel_size=9, stride=4, padding=4),
                nn.Conv1d(in_ch, width, kernel_size=31, stride=8, padding=15),
                nn.Conv1d(in_ch, width, kernel_size=91, stride=16, padding=45),
                nn.Conv1d(in_ch, width, kernel_size=151, stride=25, padding=75),
            ]
        )
        self.branch_embed = nn.Parameter(torch.randn(len(self.branches), width) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.branches])
        self.pos = PositionalEncoding(width, max_len=768)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed morphology channels are computed inside the network from the
        # waveform input. They are not precomputed external features.
        diff = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        detail = torch.abs(diff)
        smooth = F.avg_pool1d(x, kernel_size=63, stride=1, padding=31)
        morph = torch.cat([x, diff, detail, smooth], dim=1)
        pieces = []
        for i, conv in enumerate(self.branches):
            # The conv expects the original channel count, so fold morphology
            # views back by averaging groups. This keeps parameter count stable.
            view = morph.reshape(morph.shape[0], 4, x.shape[1], morph.shape[2]).mean(dim=1)
            t = conv(view).transpose(1, 2)
            pieces.append(self.norms[i](t) + self.branch_embed[i][None, None, :])
        tokens = self.encoder(self.pos(torch.cat(pieces, dim=1)))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1)], dim=1)


class QRSDetailTokenTransformer(nn.Module):
    """High-resolution local token encoder for QRS visibility/prominence."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, stat_hidden: int):
        super().__init__()
        self.highres = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.detail = nn.Sequential(
            nn.Conv1d(in_ch * 3, width, kernel_size=11, stride=4, padding=5),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=768)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.stat_branch = nn.Sequential(
            nn.LayerNorm(in_ch * 15),
            nn.Linear(in_ch * 15, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, stat_hidden),
            nn.GELU(),
        )
        self.out_dim = width * 2 + stat_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        smooth = F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
        highpass = x - smooth
        detail_view = torch.cat([x, diff, highpass], dim=1)
        tokens = torch.cat([self.highres(x).transpose(1, 2), self.detail(detail_view).transpose(1, 2)], dim=1)
        tokens = self.encoder(self.pos(tokens))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1), self.stat_branch(qrs_detail_stats(x))], dim=1)


class AtlasMemoryTransformer(nn.Module):
    """Prototype-distance memory for boundary confidence / KNN-purity targets."""

    def __init__(self, base: nn.Module, base_dim: int, prototypes_per_class: int):
        super().__init__()
        self.base = base
        self.base_dim = int(base_dim)
        self.prototypes_per_class = int(prototypes_per_class)
        self.prototype_proj = nn.Sequential(nn.LayerNorm(base_dim), nn.Linear(base_dim, base_dim), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(3, self.prototypes_per_class, base_dim) * 0.04)
        self.out_dim = base_dim + 9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base(x)
        z = F.normalize(self.prototype_proj(feat), dim=1)
        p = F.normalize(self.prototypes, dim=2)
        dist = torch.sum((z[:, None, None, :] - p[None, :, :, :]) ** 2, dim=3)
        min_dist = dist.min(dim=2).values
        mean_dist = dist.mean(dim=2)
        purity_like = torch.softmax(-min_dist, dim=1)
        return torch.cat([feat, min_dist, mean_dist, purity_like], dim=1)


def qrs_detail_stats(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    diff = x[:, :, 1:] - x[:, :, :-1]
    smooth31 = F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
    smooth101 = F.avg_pool1d(x, kernel_size=101, stride=1, padding=50)
    highpass = x - smooth31
    local_peak = highpass.abs()
    k01 = max(1, int(x.shape[2] * 0.01))
    k03 = max(1, int(x.shape[2] * 0.03))
    k10 = max(1, int(x.shape[2] * 0.10))
    top01 = torch.topk(local_peak, k=k01, dim=2).values.mean(dim=2)
    top03 = torch.topk(local_peak, k=k03, dim=2).values.mean(dim=2)
    top10 = torch.topk(local_peak, k=k10, dim=2).values.mean(dim=2)
    prom_ratio = top03 / (top10 + eps)
    detail_rms = torch.sqrt((highpass * highpass).mean(dim=2) + eps)
    diff_top = torch.topk(diff.abs(), k=k03, dim=2).values.mean(dim=2)
    diff_bg = torch.topk(diff.abs(), k=k10, dim=2).values.mean(dim=2)
    diff_ratio = diff_top / (diff_bg + eps)
    baseline_span = smooth101.amax(dim=2) - smooth101.amin(dim=2)
    baseline_slope = (smooth101[:, :, -1] - smooth101[:, :, 0]).abs()
    flat = (diff.abs() < 0.015).float().mean(dim=2)
    low_amp = (x.abs() < 0.050).float().mean(dim=2)
    mean_abs = x.abs().mean(dim=2)
    rms = torch.sqrt((x * x).mean(dim=2) + eps)
    ptp = x.amax(dim=2) - x.amin(dim=2)
    zcr = ((x[:, :, 1:] * x[:, :, :-1]) < 0).float().mean(dim=2)
    return torch.cat([top01, top03, top10, prom_ratio, detail_rms, diff_top, diff_ratio, baseline_span, baseline_slope, flat, low_amp, mean_abs, rms, ptp, zcr], dim=1)


def waveform_atlas_stats(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([waveform_stats_v2(x), qrs_detail_stats(x), waveform_stress_stats(x)], dim=1)


def qrs_visibility_bank_stats(x: torch.Tensor) -> torch.Tensor:
    """Waveform-only QRS reliability bank.

    The hard original-test failures showed weak transfer for qrs_visibility and
    detector_agreement.  These summaries do not use external detectors; they
    expose local peak density, peak prominence, long no-peak spans, and coarse
    rhythm autocorrelation as internal Transformer inputs.
    """

    eps = 1e-6
    smooth15 = F.avg_pool1d(x, kernel_size=15, stride=1, padding=7)
    smooth51 = F.avg_pool1d(x, kernel_size=51, stride=1, padding=25)
    high = x - smooth51
    high_abs = high.abs()
    high_rms = torch.sqrt((high * high).mean(dim=2) + eps)
    high_mean_abs = high_abs.mean(dim=2)
    k005 = max(1, int(high.shape[2] * 0.005))
    k02 = max(1, int(high.shape[2] * 0.02))
    k10 = max(1, int(high.shape[2] * 0.10))
    top005 = torch.topk(high_abs, k=k005, dim=2).values.mean(dim=2)
    top02 = torch.topk(high_abs, k=k02, dim=2).values.mean(dim=2)
    top10 = torch.topk(high_abs, k=k10, dim=2).values.mean(dim=2)
    top_ratio = top02 / (top10 + eps)

    loc = high_abs[:, :, 1:-1]
    left = high_abs[:, :, :-2]
    right = high_abs[:, :, 2:]
    local_peak = ((loc >= left) & (loc >= right)).float() * loc
    local_top01 = torch.topk(local_peak, k=max(1, int(local_peak.shape[2] * 0.01)), dim=2).values.mean(dim=2)
    local_top03 = torch.topk(local_peak, k=max(1, int(local_peak.shape[2] * 0.03)), dim=2).values.mean(dim=2)
    peak_std = torch.topk(local_peak, k=max(2, int(local_peak.shape[2] * 0.03)), dim=2).values.std(dim=2)
    peak_cv = peak_std / (local_top03 + eps)

    scale = high_rms[:, :, None].clamp_min(0.01)
    peak08 = (local_peak > 0.80 * scale).float()
    peak12 = (local_peak > 1.20 * scale).float()
    peak18 = (local_peak > 1.80 * scale).float()
    peak_density08 = peak08.mean(dim=2)
    peak_density12 = peak12.mean(dim=2)
    peak_density18 = peak18.mean(dim=2)
    no_peak251 = 1.0 - F.avg_pool1d(peak12, kernel_size=251, stride=1, padding=125).amax(dim=2)

    def autocorr(lag: int) -> torch.Tensor:
        if high.shape[2] <= lag:
            return torch.zeros_like(high_rms)
        return (high[:, :, :-lag] * high[:, :, lag:]).mean(dim=2) / (high_rms * high_rms + eps)

    ac50 = autocorr(50)
    ac100 = autocorr(100)
    ac150 = autocorr(150)
    ac250 = autocorr(250)
    ac350 = autocorr(350)
    short_detail = (x - smooth15).abs().mean(dim=2)

    return torch.cat(
        [
            top005,
            top02,
            top10,
            top_ratio,
            local_top01,
            local_top03,
            peak_cv,
            peak_density08,
            peak_density12,
            peak_density18,
            no_peak251,
            ac50,
            ac100,
            ac150,
            ac250,
            ac350,
            high_rms,
            high_mean_abs,
            short_detail,
        ],
        dim=1,
    )


def qrs_detector_agreement_stats(x: torch.Tensor) -> torch.Tensor:
    """Detector-agreement surrogates from waveform only.

    BUT bad-stress misses often contain visible spikes plus dropout/baseline
    artifacts.  A single energy score confuses those with medium outliers, so
    this bank exposes agreement among amplitude, slope, and curvature peak
    proposals with a small tolerance window.
    """

    eps = 1e-6
    smooth51 = F.avg_pool1d(x, kernel_size=51, stride=1, padding=25)
    high = x - smooth51
    high_abs = high.abs()
    slope = F.pad(high[:, :, 1:] - high[:, :, :-1], (1, 0))
    curvature = F.pad(high[:, :, 2:] - 2.0 * high[:, :, 1:-1] + high[:, :, :-2], (1, 1))
    slope_abs = slope.abs()
    curv_abs = curvature.abs()

    def adaptive_mask(v: torch.Tensor, scale_mult: float, top_frac: float) -> torch.Tensor:
        rms = torch.sqrt((v * v).mean(dim=2, keepdim=True) + eps)
        k = max(1, int(v.shape[2] * top_frac))
        top = torch.topk(v, k=k, dim=2).values.mean(dim=2, keepdim=True)
        thr = torch.maximum(scale_mult * rms, 0.45 * top)
        return (v > thr).float()

    amp_mask = adaptive_mask(high_abs, 1.20, 0.020)
    slope_mask = adaptive_mask(slope_abs, 1.25, 0.020)
    curv_mask = adaptive_mask(curv_abs, 1.30, 0.020)

    def dilate(m: torch.Tensor, kernel: int = 25) -> torch.Tensor:
        return (F.avg_pool1d(m, kernel_size=kernel, stride=1, padding=kernel // 2) > 0.0).float()

    amp_d = dilate(amp_mask)
    slope_d = dilate(slope_mask)
    curv_d = dilate(curv_mask)
    union = ((amp_d + slope_d + curv_d) > 0.0).float()
    triple = amp_mask * slope_d * curv_d

    def jaccard(a: torch.Tensor, b: torch.Tensor, a_d: torch.Tensor, b_d: torch.Tensor) -> torch.Tensor:
        inter = (a * b_d).mean(dim=2)
        uni = ((a_d + b_d) > 0.0).float().mean(dim=2)
        return inter / (uni + eps)

    amp_density = amp_mask.mean(dim=2)
    slope_density = slope_mask.mean(dim=2)
    curv_density = curv_mask.mean(dim=2)
    union_density = union.mean(dim=2)
    triple_density = triple.mean(dim=2)
    amp_slope_j = jaccard(amp_mask, slope_mask, amp_d, slope_d)
    amp_curv_j = jaccard(amp_mask, curv_mask, amp_d, curv_d)
    slope_curv_j = jaccard(slope_mask, curv_mask, slope_d, curv_d)
    agreement_mean = (amp_slope_j + amp_curv_j + slope_curv_j) / 3.0
    disagreement_density = (
        (amp_density - slope_density).abs()
        + (amp_density - curv_density).abs()
        + (slope_density - curv_density).abs()
    ) / 3.0
    no_event200 = 1.0 - F.avg_pool1d(union, kernel_size=201, stride=1, padding=100).amax(dim=2)
    no_event400 = 1.0 - F.avg_pool1d(union, kernel_size=401, stride=1, padding=200).amax(dim=2)

    centered = union - union.mean(dim=2, keepdim=True)
    denom = (centered * centered).mean(dim=2) + eps

    def event_ac(lag: int) -> torch.Tensor:
        if centered.shape[2] <= lag:
            return torch.zeros_like(denom)
        return (centered[:, :, :-lag] * centered[:, :, lag:]).mean(dim=2) / denom

    ac80 = event_ac(80)
    ac160 = event_ac(160)
    ac260 = event_ac(260)
    k005 = max(1, int(high_abs.shape[2] * 0.005))
    k10 = max(1, int(high_abs.shape[2] * 0.10))
    peak_contrast = torch.topk(high_abs, k=k005, dim=2).values.mean(dim=2) / (
        torch.topk(high_abs, k=k10, dim=2).values.mean(dim=2) + eps
    )

    return torch.cat(
        [
            amp_density,
            slope_density,
            curv_density,
            union_density,
            triple_density,
            amp_slope_j,
            amp_curv_j,
            slope_curv_j,
            agreement_mean,
            disagreement_density,
            no_event200,
            no_event400,
            ac80,
            ac160,
            ac260,
            peak_contrast,
        ],
        dim=1,
    )


def baseline_frequency_stats(x: torch.Tensor) -> torch.Tensor:
    """Low-frequency and baseline-band waveform stats.

    These are computed from the inference waveform itself.  They give the
    Transformer explicit patch-time access to basSQI-like and baseline-wander
    evidence that has been hard to recover from raw patches alone.
    """

    eps = 1e-6
    smooth15 = F.avg_pool1d(x, kernel_size=15, stride=1, padding=7)
    smooth31 = F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
    smooth51 = F.avg_pool1d(x, kernel_size=51, stride=1, padding=25)
    smooth151 = F.avg_pool1d(x, kernel_size=151, stride=1, padding=75)
    smooth251 = F.avg_pool1d(x, kernel_size=251, stride=1, padding=125)
    smooth501 = F.avg_pool1d(x, kernel_size=501, stride=1, padding=250)
    smooth1001 = F.avg_pool1d(x, kernel_size=1001, stride=1, padding=500)

    rms = torch.sqrt((x * x).mean(dim=2) + eps)
    baseline501 = smooth501
    baseline1001 = smooth1001
    drift_band = smooth251 - smooth1001
    lf_band = smooth151 - smooth501
    mid_band = smooth51 - smooth251
    qrs_band = x - smooth51
    detail_band = x - smooth15
    high_band = x - smooth31

    def band_rms(v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((v * v).mean(dim=2) + eps)

    def top_abs_mean(v: torch.Tensor, frac: float) -> torch.Tensor:
        k = max(1, int(v.shape[2] * frac))
        return torch.topk(v.abs(), k=k, dim=2).values.mean(dim=2)

    base501_rms = band_rms(baseline501)
    base1001_rms = band_rms(baseline1001)
    drift_rms = band_rms(drift_band)
    lf_rms = band_rms(lf_band)
    mid_rms = band_rms(mid_band)
    qrs_rms = band_rms(qrs_band)
    detail_rms = band_rms(detail_band)
    high_rms = band_rms(high_band)
    base251_diff = (smooth251[:, :, 1:] - smooth251[:, :, :-1]).abs()
    base501_diff = (smooth501[:, :, 1:] - smooth501[:, :, :-1]).abs()

    return torch.cat(
        [
            baseline501.amax(dim=2) - baseline501.amin(dim=2),
            baseline1001.amax(dim=2) - baseline1001.amin(dim=2),
            baseline501.std(dim=2),
            baseline1001.std(dim=2),
            (baseline501[:, :, -1] - baseline501[:, :, 0]).abs(),
            (baseline1001[:, :, -1] - baseline1001[:, :, 0]).abs(),
            base501_rms / (rms + eps),
            base1001_rms / (rms + eps),
            drift_rms / (rms + eps),
            lf_rms / (rms + eps),
            mid_rms / (rms + eps),
            qrs_rms / (rms + eps),
            detail_rms / (rms + eps),
            high_rms / (rms + eps),
            qrs_rms / (base501_rms + eps),
            detail_rms / (base501_rms + eps),
            base501_rms / (detail_rms + eps),
            qrs_rms / (mid_rms + eps),
            top_abs_mean(baseline501, 0.05),
            top_abs_mean(lf_band, 0.10),
            top_abs_mean(qrs_band, 0.10),
            top_abs_mean(detail_band, 0.10),
            torch.topk(base251_diff, k=max(1, int(base251_diff.shape[2] * 0.01)), dim=2).values.mean(dim=2),
            torch.topk(base501_diff, k=max(1, int(base501_diff.shape[2] * 0.01)), dim=2).values.mean(dim=2),
        ],
        dim=1,
    )


def sparse_extreme_event_stats(x: torch.Tensor) -> torch.Tensor:
    """Sparse reset/dropout/impulse event bank, computed from waveform only.

    Original record111 bad-outlier misses are often not globally noisy.  They
    contain a few large vertical resets, short dropouts/plateaus, or channel
    disagreements that average pooling can dilute.  This bank exposes top-k
    jump evidence and cross-channel event disagreement without using any
    external SQI sidecar.
    """

    eps = 1e-6
    b, c, n = x.shape
    diff = x[:, :, 1:] - x[:, :, :-1]
    abs_x = x.abs()
    abs_d = diff.abs()
    rms_x = torch.sqrt((x * x).mean(dim=2) + eps)
    rms_d = torch.sqrt((diff * diff).mean(dim=2, keepdim=True) + eps)

    def top_abs_mean(v: torch.Tensor, frac: float) -> torch.Tensor:
        k = max(1, int(v.shape[2] * frac))
        return torch.topk(v.abs(), k=k, dim=2).values.mean(dim=2)

    diff_top001 = top_abs_mean(diff, 0.001)
    diff_top005 = top_abs_mean(diff, 0.005)
    diff_top02 = top_abs_mean(diff, 0.020)
    diff_top10 = top_abs_mean(diff, 0.100)
    diff_ratio001_10 = diff_top001 / (diff_top10 + eps)
    diff_ratio005_10 = diff_top005 / (diff_top10 + eps)

    abs_top001 = top_abs_mean(x, 0.001)
    abs_top005 = top_abs_mean(x, 0.005)
    abs_top02 = top_abs_mean(x, 0.020)
    abs_ratio001_10 = abs_top001 / (top_abs_mean(x, 0.100) + eps)

    jump3 = (abs_d > 3.0 * rms_d).float()
    jump5 = (abs_d > 5.0 * rms_d).float()
    jump8 = (abs_d > 8.0 * rms_d).float()
    jump_density3 = jump3.mean(dim=2)
    jump_density5 = jump5.mean(dim=2)
    jump_density8 = jump8.mean(dim=2)
    jump_run21 = F.avg_pool1d(jump3, kernel_size=21, stride=1, padding=10).amax(dim=2)
    jump_run101 = F.avg_pool1d(jump3, kernel_size=101, stride=1, padding=50).amax(dim=2)

    flat51 = F.avg_pool1d((abs_d < 0.015).float(), kernel_size=51, stride=1, padding=25).amax(dim=2)
    flat101 = F.avg_pool1d((abs_d < 0.015).float(), kernel_size=101, stride=1, padding=50).amax(dim=2)
    low51 = F.avg_pool1d((abs_x < 0.050).float(), kernel_size=51, stride=1, padding=25).amax(dim=2)
    low101 = F.avg_pool1d((abs_x < 0.050).float(), kernel_size=101, stride=1, padding=50).amax(dim=2)
    flat_jump_inter = flat51 * jump_run21
    low_jump_inter = low51 * jump_run21
    diff_flat_inter = diff_top001 * flat101 / (rms_x + eps)
    diff_low_inter = diff_top005 * low101 / (rms_x + eps)

    other_jump = ((jump5.sum(dim=1, keepdim=True) - jump5) > 0.0).float()
    cojump = jump5 * other_jump
    solojump = jump5 * (1.0 - other_jump)
    cojump_density = cojump.mean(dim=2)
    solojump_density = solojump.mean(dim=2)
    cojump_run51 = F.avg_pool1d(cojump, kernel_size=51, stride=1, padding=25).amax(dim=2)

    centered = x - x.mean(dim=1, keepdim=True)
    resid_abs = centered.abs()
    resid_top001 = top_abs_mean(centered, 0.001)
    resid_top005 = top_abs_mean(centered, 0.005)
    resid_rms = torch.sqrt((centered * centered).mean(dim=2) + eps)
    resid_ratio001_10 = resid_top001 / (top_abs_mean(centered, 0.100) + eps)

    cross_std = x.std(dim=1, unbiased=False)
    cross_diff_std = diff.std(dim=1, unbiased=False)
    cross_std_mean = cross_std.mean(dim=1)
    cross_std_top001 = torch.topk(cross_std, k=max(1, int(n * 0.001)), dim=1).values.mean(dim=1)
    cross_std_top005 = torch.topk(cross_std, k=max(1, int(n * 0.005)), dim=1).values.mean(dim=1)
    cross_diff_top005 = torch.topk(cross_diff_std, k=max(1, int(cross_diff_std.shape[1] * 0.005)), dim=1).values.mean(dim=1)

    def repeat_cross(v: torch.Tensor) -> torch.Tensor:
        return v[:, None].expand(-1, c)

    boundary_reset_score = diff_ratio001_10 * (resid_ratio001_10 / (resid_ratio001_10.mean(dim=1, keepdim=True) + eps))
    sparse_bad_score = jump_run21 * (diff_ratio005_10 + resid_ratio001_10) * 0.5

    return torch.cat(
        [
            diff_top001,
            diff_top005,
            diff_top02,
            diff_top10,
            diff_ratio001_10,
            diff_ratio005_10,
            abs_top001,
            abs_top005,
            abs_top02,
            abs_ratio001_10,
            jump_density3,
            jump_density5,
            jump_density8,
            jump_run21,
            jump_run101,
            flat_jump_inter,
            low_jump_inter,
            diff_flat_inter,
            diff_low_inter,
            cojump_density,
            solojump_density,
            cojump_run51,
            resid_top001,
            resid_top005,
            resid_rms,
            resid_ratio001_10,
            repeat_cross(cross_std_mean),
            repeat_cross(cross_std_top001),
            repeat_cross(cross_std_top005),
            repeat_cross(cross_diff_top005),
            boundary_reset_score,
            sparse_bad_score,
        ],
        dim=1,
    )


def primitive_bank_per_channel(bank: str) -> int:
    bank = str(bank)
    if bank == "qrs_stress_v5":
        return 170
    if bank == "qrs_stress_v4":
        return 138
    if bank == "qrs_stress_v3":
        return 114
    if bank == "qrs_enhanced_v2":
        return 90
    if bank == "qrs_enhanced":
        return 74
    return 55


def primitive_waveform_stats(x: torch.Tensor, bank: str) -> torch.Tensor:
    if str(bank) == "qrs_stress_v5":
        return torch.cat(
            [
                waveform_atlas_stats(x),
                qrs_visibility_bank_stats(x),
                qrs_detector_agreement_stats(x),
                baseline_frequency_stats(x),
                waveform_stress_stats(x),
                sparse_extreme_event_stats(x),
            ],
            dim=1,
        )
    if str(bank) == "qrs_stress_v4":
        return torch.cat(
            [
                waveform_atlas_stats(x),
                qrs_visibility_bank_stats(x),
                qrs_detector_agreement_stats(x),
                baseline_frequency_stats(x),
                waveform_stress_stats(x),
            ],
            dim=1,
        )
    if str(bank) == "qrs_stress_v3":
        return torch.cat(
            [
                waveform_atlas_stats(x),
                qrs_visibility_bank_stats(x),
                qrs_detector_agreement_stats(x),
                waveform_stress_stats(x),
            ],
            dim=1,
        )
    if str(bank) == "qrs_enhanced_v2":
        return torch.cat([waveform_atlas_stats(x), qrs_visibility_bank_stats(x), qrs_detector_agreement_stats(x)], dim=1)
    if str(bank) == "qrs_enhanced":
        return torch.cat([waveform_atlas_stats(x), qrs_visibility_bank_stats(x)], dim=1)
    return waveform_atlas_stats(x)


class GeometryStudent(nn.Module):
    def __init__(self, cfg: dict[str, Any], in_ch: int):
        super().__init__()
        arch = str(cfg["arch"])
        width = int(cfg["width"])
        layers = int(cfg["layers"])
        heads = int(cfg["heads"])
        self.arch = arch
        self.use_teacher_atlas = arch in {
            "teacher_atlas_student",
            "qrs_teacher_atlas_student",
            "hybrid_atlas_student",
            "statfed_patch_teacher_atlas",
            "multiscale_stat_patch_teacher_atlas",
            "stress_statfed_patch_teacher_atlas",
            "stress_multiscale_stat_patch_teacher_atlas",
            "stresstoken_statfed_patch_teacher_atlas",
            "stresstoken_multiscale_stat_patch_teacher_atlas",
            "statbank_token_teacher_atlas",
            "qrsbank_stat_token_teacher_atlas",
            "primitive_token_teacher_atlas",
        }
        if arch == "mapstudent":
            self.encoder = PatchStudentEncoder(in_ch, width, layers, heads, int(cfg["patch"]))
            out_dim = self.encoder.out_dim
        elif arch == "masked_mae":
            self.encoder = MaskedGeometryMAEEncoder(in_ch, width, layers, heads, int(cfg["patch"]))
            out_dim = self.encoder.out_dim
        elif arch == "stattoken_v2":
            self.encoder = StatTokenTransformerV2(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "morphology_token":
            self.encoder = MorphologyTokenTransformer(in_ch, width, layers, heads)
            out_dim = self.encoder.out_dim
        elif arch == "statbank_token_teacher_atlas":
            self.encoder = StatBankTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "qrsbank_stat_token_teacher_atlas":
            self.encoder = QRSBankStatTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "primitive_token_teacher_atlas":
            self.encoder = PrimitiveTokenTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["stat_hidden"]),
                str(cfg.get("primitive_token_bank", cfg.get("primitive_bank", "qrs_stress_v5"))),
                int(cfg.get("primitive_token_chunk", 16)),
            )
            out_dim = self.encoder.out_dim
        elif arch == "qrs_detail_token":
            self.encoder = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "atlas_memory":
            base = StatTokenTransformerV2(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        elif arch == "qrs_atlas_memory":
            base = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        elif arch == "teacher_atlas_student":
            self.encoder = StatTokenTransformerV2(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "qrs_teacher_atlas_student":
            self.encoder = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "hybrid_atlas_student":
            base = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        elif arch == "statfed_patch":
            self.encoder = StatFedPatchTransformer(in_ch, width, layers, heads, int(cfg["patch"]), int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "statfed_patch_teacher_atlas":
            self.encoder = StatFedPatchTransformer(in_ch, width, layers, heads, int(cfg["patch"]), int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "stress_statfed_patch_teacher_atlas":
            base = StatFedPatchTransformer(in_ch, width, layers, heads, int(cfg["patch"]), int(cfg["stat_hidden"]))
            self.encoder = StressBankWrapper(base, base.out_dim, in_ch, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "stresstoken_statfed_patch_teacher_atlas":
            self.encoder = StressTokenStatFedPatchTransformer(in_ch, width, layers, heads, int(cfg["patch"]), int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "statfed_patch_atlas":
            base = StatFedPatchTransformer(in_ch, width, layers, heads, int(cfg["patch"]), int(cfg["stat_hidden"]))
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        elif arch == "multiscale_stat_patch":
            self.encoder = MultiScaleStatPatchTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["patch"]),
                int(cfg["stat_hidden"]),
                list(cfg.get("scales", [50, 100, 250, 500])),
            )
            out_dim = self.encoder.out_dim
        elif arch == "multiscale_stat_patch_teacher_atlas":
            self.encoder = MultiScaleStatPatchTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["patch"]),
                int(cfg["stat_hidden"]),
                list(cfg.get("scales", [50, 100, 250, 500])),
            )
            out_dim = self.encoder.out_dim
        elif arch == "stress_multiscale_stat_patch_teacher_atlas":
            base = MultiScaleStatPatchTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["patch"]),
                int(cfg["stat_hidden"]),
                list(cfg.get("scales", [50, 100, 250, 500])),
            )
            self.encoder = StressBankWrapper(base, base.out_dim, in_ch, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "sqi_query_multiscale_patch_teacher_atlas":
            self.encoder = SQIQueryMultiScaleTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["patch"]),
                int(cfg["stat_hidden"]),
                list(cfg.get("scales", [25, 50, 100, 250, 500])),
            )
            out_dim = self.encoder.out_dim
        elif arch == "event_qrs_sqi_query_multiscale_patch_teacher_atlas":
            self.encoder = EventQRSQueryMultiScaleTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["patch"]),
                int(cfg["stat_hidden"]),
                list(cfg.get("scales", [25, 50, 100, 250, 500])),
                int(cfg.get("event_count", 12)),
                int(cfg.get("event_window", 81)),
            )
            out_dim = self.encoder.out_dim
        elif arch == "stresstoken_multiscale_stat_patch_teacher_atlas":
            self.encoder = StressTokenMultiScaleStatPatchTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["patch"]),
                int(cfg["stat_hidden"]),
                list(cfg.get("scales", [50, 100, 250, 500])),
            )
            out_dim = self.encoder.out_dim
        elif arch == "multiscale_stat_patch_atlas":
            base = MultiScaleStatPatchTransformer(
                in_ch,
                width,
                layers,
                heads,
                int(cfg["patch"]),
                int(cfg["stat_hidden"]),
                list(cfg.get("scales", [50, 100, 250, 500])),
            )
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        else:
            raise ValueError(f"unknown architecture {arch}")
        if self.use_teacher_atlas:
            proto_count = int(cfg["teacher_prototypes_per_class"])
            self.register_buffer("teacher_prototypes", torch.zeros(3, proto_count, len(CORE_IDX)), persistent=True)
            class_head_in = 192 + 9
        else:
            class_head_in = 192
        self.embedding_prototypes_per_class = int(cfg.get("embedding_prototypes_per_class", 0))
        self.use_embedding_prototypes = self.embedding_prototypes_per_class > 0
        self.embedding_proto_temp = float(cfg.get("embedding_proto_temp", 0.35))
        self.embedding_proto_logit_mix = float(cfg.get("embedding_proto_logit_mix", 0.0))
        if self.use_embedding_prototypes:
            self.embedding_proto_norm = nn.LayerNorm(192)
            self.embedding_prototypes = nn.Parameter(torch.randn(3, self.embedding_prototypes_per_class, 192) * 0.04)
            class_head_in += 9
        self.waveform_atlas_prototypes_per_class = int(cfg.get("waveform_atlas_prototypes_per_class", 0))
        self.use_waveform_atlas = self.waveform_atlas_prototypes_per_class > 0
        self.waveform_atlas_dim = int(in_ch) * 55
        self.waveform_atlas_temp = float(cfg.get("waveform_atlas_temp", 0.42))
        self.waveform_atlas_logit_mix = float(cfg.get("waveform_atlas_logit_mix", 0.0))
        if self.use_waveform_atlas:
            self.register_buffer("waveform_atlas_mean", torch.zeros(self.waveform_atlas_dim), persistent=True)
            self.register_buffer("waveform_atlas_std", torch.ones(self.waveform_atlas_dim), persistent=True)
            self.register_buffer(
                "waveform_atlas_prototypes",
                torch.zeros(3, self.waveform_atlas_prototypes_per_class, self.waveform_atlas_dim),
                persistent=True,
            )
            class_head_in += 9
        self.predicted_feature_fusion = str(cfg.get("predicted_feature_fusion", "none"))
        if self.predicted_feature_fusion == "core":
            self.predicted_feature_indices = CORE_IDX
            class_head_in += len(CORE_IDX)
        elif self.predicted_feature_fusion == "teacher":
            self.predicted_feature_indices = TEACHER_IDX
            class_head_in += len(TEACHER_IDX)
        elif self.predicted_feature_fusion == "top14":
            self.predicted_feature_indices = TOP14_IDX
            class_head_in += len(TOP14_IDX)
        elif self.predicted_feature_fusion == "configured":
            configured_cols = list(cfg.get("predicted_feature_columns", []))
            self.predicted_feature_indices = [FEATURE_COLUMNS.index(c) for c in configured_cols if c in FEATURE_COLUMNS]
            if len(self.predicted_feature_indices) != len(configured_cols):
                missing = [c for c in configured_cols if c not in FEATURE_COLUMNS]
                raise ValueError(f"configured predicted_feature_columns not found in schema: {missing}")
            class_head_in += len(self.predicted_feature_indices)
        else:
            self.predicted_feature_indices = []
        self.primitive_threshold_fusion = bool(cfg.get("primitive_threshold_fusion", False))
        self.use_primitive_branch = bool(
            cfg.get("primitive_fusion", False)
            or cfg.get("primitive_bad_expert", False)
            or self.primitive_threshold_fusion
        )
        self.primitive_fusion = bool(cfg.get("primitive_fusion", False))
        self.primitive_bad_mix = float(cfg.get("primitive_bad_mix", 0.0))
        self.primitive_bank = str(cfg.get("primitive_bank", "atlas"))
        if self.use_primitive_branch:
            primitive_per_channel = primitive_bank_per_channel(self.primitive_bank)
            primitive_dim = int(in_ch) * primitive_per_channel
            self.primitive_dim = primitive_dim
            self.primitive_branch = nn.Sequential(
                nn.LayerNorm(primitive_dim),
                nn.Linear(primitive_dim, 160),
                nn.GELU(),
                nn.Dropout(0.08),
                nn.Linear(160, 96),
                nn.GELU(),
            )
            if self.primitive_fusion:
                class_head_in += 96
            if self.primitive_threshold_fusion:
                threshold_values = cfg.get("primitive_threshold_values", [-1.75, -1.0, -0.45, 0.0, 0.45, 1.0, 1.75])
                threshold_tensor = torch.tensor([float(v) for v in threshold_values], dtype=torch.float32)
                self.register_buffer("primitive_threshold_values", threshold_tensor, persistent=True)
                self.primitive_threshold_sharpness = float(cfg.get("primitive_threshold_sharpness", 2.2))
                self.primitive_threshold_norm = nn.LayerNorm(primitive_dim)
                self.primitive_threshold_proj = nn.Sequential(
                    nn.LayerNorm(primitive_dim * int(threshold_tensor.numel())),
                    nn.Linear(primitive_dim * int(threshold_tensor.numel()), 192),
                    nn.GELU(),
                    nn.Dropout(0.08),
                    nn.Linear(192, 96),
                    nn.GELU(),
                )
                class_head_in += 96
            else:
                self.primitive_threshold_values = None
                self.primitive_threshold_norm = None
                self.primitive_threshold_proj = None
        else:
            self.primitive_branch = None
            self.primitive_dim = 0
            self.primitive_threshold_values = None
            self.primitive_threshold_norm = None
            self.primitive_threshold_proj = None
        self.primitive_bad_bank = str(cfg.get("primitive_bad_bank", self.primitive_bank))
        self.primitive_bad_branch = None
        if bool(cfg.get("primitive_bad_expert", False)):
            if self.primitive_bad_bank != self.primitive_bank:
                primitive_bad_dim = int(in_ch) * primitive_bank_per_channel(self.primitive_bad_bank)
                self.primitive_bad_branch = nn.Sequential(
                    nn.LayerNorm(primitive_bad_dim),
                    nn.Linear(primitive_bad_dim, 160),
                    nn.GELU(),
                    nn.Dropout(0.08),
                    nn.Linear(160, 96),
                    nn.GELU(),
                )
            self.primitive_bad_head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 64), nn.GELU(), nn.Dropout(0.08), nn.Linear(64, 1))
        else:
            self.primitive_bad_head = None
        self.project = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, 192), nn.GELU(), nn.Dropout(0.08))
        self.primitive_context_fusion = bool(cfg.get("primitive_context_fusion", False))
        self.aux_primitive_context_fusion = bool(cfg.get("aux_primitive_context_fusion", False))
        if self.primitive_context_fusion or self.aux_primitive_context_fusion:
            context_width = int(cfg.get("primitive_context_width", max(48, width // 2)))
            self.primitive_context_encoder = PrimitiveTokenTransformer(
                in_ch,
                context_width,
                int(cfg.get("primitive_context_layers", 2)),
                int(cfg.get("primitive_context_heads", max(1, min(heads, 4)))),
                int(cfg.get("primitive_context_hidden", max(128, int(cfg.get("stat_hidden", 128))))),
                str(cfg.get("primitive_context_bank", "qrs_stress_v5")),
                int(cfg.get("primitive_context_chunk", 16)),
            )
            self.primitive_context_project = nn.Sequential(
                nn.LayerNorm(self.primitive_context_encoder.out_dim),
                nn.Linear(self.primitive_context_encoder.out_dim, 96),
                nn.GELU(),
                nn.Dropout(0.08),
            )
            if self.primitive_context_fusion:
                class_head_in += 96
        else:
            self.primitive_context_encoder = None
            self.primitive_context_project = None
        self.aux_primitive_fusion = bool(cfg.get("aux_primitive_fusion", False))
        self.final_mode = str(cfg.get("final_mode", "plain"))
        self.gate_logit_alpha = float(cfg.get("gate_logit_alpha", 0.35))
        self.bad_margin_guard = bool(cfg.get("bad_margin_guard", False))
        self.bad_margin_alpha = float(cfg.get("bad_margin_alpha", 0.0))
        self.bad_purity_alpha = float(cfg.get("bad_purity_alpha", 0.0))
        self.bad_boundary_alpha = float(cfg.get("bad_boundary_alpha", 0.0))
        self.pca_margin_idx = FEATURE_COLUMNS.index("pca_margin") if "pca_margin" in FEATURE_COLUMNS else -1
        self.knn_purity_idx = FEATURE_COLUMNS.index("knn_label_purity") if "knn_label_purity" in FEATURE_COLUMNS else -1
        self.boundary_conf_idx = FEATURE_COLUMNS.index("boundary_confidence") if "boundary_confidence" in FEATURE_COLUMNS else -1
        self.class_head = nn.Sequential(nn.LayerNorm(class_head_in), nn.Linear(class_head_in, 96), nn.GELU(), nn.Dropout(0.08), nn.Linear(96, 3))
        self.feature_first_logits = bool(cfg.get("feature_first_logits", False))
        self.feature_residual_alpha = float(cfg.get("feature_residual_alpha", 0.35))
        if self.feature_first_logits and self.predicted_feature_indices:
            feature_head_in = len(self.predicted_feature_indices)
            self.feature_class_head = nn.Sequential(
                nn.LayerNorm(feature_head_in),
                nn.Linear(feature_head_in, 64),
                nn.GELU(),
                nn.Dropout(0.08),
                nn.Linear(64, 3),
            )
            self.dynamic_feature_gate = bool(cfg.get("dynamic_feature_gate", False))
            if self.dynamic_feature_gate:
                self.feature_gate_head = nn.Sequential(
                    nn.LayerNorm(feature_head_in),
                    nn.Linear(feature_head_in, 48),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(48, 1),
                )
                with torch.no_grad():
                    self.feature_gate_head[-1].bias.fill_(float(cfg.get("feature_gate_bias_init", -0.35)))
            else:
                self.feature_gate_head = None
        else:
            self.feature_class_head = None
            self.dynamic_feature_gate = False
            self.feature_gate_head = None
        self.dual_bad_modes = {"dual_bad_gm_replace", "dual_bad_gm_mix"}
        if self.final_mode in {"gated_bad", "gated_bad_mix", "gated_bad_replace"} | self.dual_bad_modes:
            self.gm_pair_head = nn.Sequential(nn.LayerNorm(class_head_in), nn.Linear(class_head_in, 96), nn.GELU(), nn.Dropout(0.08), nn.Linear(96, 2))
        else:
            self.gm_pair_head = None
        aux_hidden = int(cfg.get("aux_hidden", 128))
        aux_head_in = 192 + (96 if self.aux_primitive_fusion and self.primitive_branch is not None else 0)
        if self.aux_primitive_context_fusion and self.primitive_context_encoder is not None:
            aux_head_in += 96
        if bool(cfg.get("wide_aux_head", False)):
            self.aux_head = nn.Sequential(
                nn.LayerNorm(aux_head_in),
                nn.Linear(aux_head_in, aux_hidden),
                nn.GELU(),
                nn.Dropout(0.08),
                nn.Linear(aux_hidden, aux_hidden),
                nn.GELU(),
                nn.Linear(aux_hidden, len(FEATURE_COLUMNS)),
            )
        else:
            self.aux_head = nn.Sequential(nn.LayerNorm(aux_head_in), nn.Linear(aux_head_in, aux_hidden), nn.GELU(), nn.Linear(aux_hidden, len(FEATURE_COLUMNS)))
        self.bad_logit_mix = float(cfg.get("bad_logit_mix", 0.0))
        if float(cfg.get("bad_aux_weight", 0.0)) > 0 or self.bad_logit_mix != 0.0 or self.final_mode in self.dual_bad_modes:
            self.bad_head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 96), nn.GELU(), nn.Dropout(0.08), nn.Linear(96, 1))
            if self.final_mode in {"gated_bad", "gated_bad_mix", "gated_bad_replace"} | self.dual_bad_modes:
                with torch.no_grad():
                    self.bad_head[-1].bias.fill_(float(cfg.get("bad_gate_bias_init", -1.0)))
        else:
            self.bad_head = None
        self.stress_bad_logit_mix = float(cfg.get("stress_bad_logit_mix", 0.0))
        if float(cfg.get("stress_aux_weight", 0.0)) > 0 or self.stress_bad_logit_mix != 0.0:
            self.stress_head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 96), nn.GELU(), nn.Dropout(0.08), nn.Linear(96, 1))
            with torch.no_grad():
                self.stress_head[-1].bias.fill_(float(cfg.get("stress_gate_bias_init", -2.0)))
        else:
            self.stress_head = None

    def set_teacher_prototypes(self, prototypes: np.ndarray | torch.Tensor) -> None:
        if not self.use_teacher_atlas:
            return
        tensor = torch.as_tensor(prototypes, dtype=torch.float32, device=self.teacher_prototypes.device)
        if tuple(tensor.shape) != tuple(self.teacher_prototypes.shape):
            raise ValueError(f"prototype shape mismatch: got {tuple(tensor.shape)} expected {tuple(self.teacher_prototypes.shape)}")
        self.teacher_prototypes.copy_(tensor)

    def set_waveform_atlas(self, prototypes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        if not self.use_waveform_atlas:
            return
        proto = torch.as_tensor(prototypes, dtype=torch.float32, device=self.waveform_atlas_prototypes.device)
        mu = torch.as_tensor(mean, dtype=torch.float32, device=self.waveform_atlas_mean.device)
        sigma = torch.as_tensor(std, dtype=torch.float32, device=self.waveform_atlas_std.device)
        if tuple(proto.shape) != tuple(self.waveform_atlas_prototypes.shape):
            raise ValueError(f"waveform atlas shape mismatch: got {tuple(proto.shape)} expected {tuple(self.waveform_atlas_prototypes.shape)}")
        if tuple(mu.shape) != tuple(self.waveform_atlas_mean.shape):
            raise ValueError(f"waveform atlas mean mismatch: got {tuple(mu.shape)} expected {tuple(self.waveform_atlas_mean.shape)}")
        self.waveform_atlas_prototypes.copy_(proto)
        self.waveform_atlas_mean.copy_(mu)
        self.waveform_atlas_std.copy_(sigma.clamp_min(1e-4))

    def teacher_distance_features(self, aux_values: torch.Tensor) -> torch.Tensor:
        core = aux_values[:, CORE_IDX]
        dist = torch.mean((core[:, None, None, :] - self.teacher_prototypes[None, :, :, :]) ** 2, dim=3)
        min_dist = dist.min(dim=2).values
        mean_dist = dist.mean(dim=2)
        purity_like = torch.softmax(-min_dist, dim=1)
        return torch.cat([min_dist, mean_dist, purity_like], dim=1)

    def embedding_prototype_outputs(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.normalize(self.embedding_proto_norm(emb), dim=1)
        prototypes = F.normalize(self.embedding_prototypes, dim=2)
        sim = torch.einsum("bd,ckd->bck", z, prototypes)
        dist = 1.0 - sim
        min_dist = dist.min(dim=2).values
        mean_dist = dist.mean(dim=2)
        purity_like = torch.softmax(-min_dist / max(self.embedding_proto_temp, 1e-4), dim=1)
        features = torch.cat([min_dist, mean_dist, purity_like], dim=1)
        class_logits = -min_dist / max(self.embedding_proto_temp, 1e-4)
        proto_logits = sim.reshape(emb.shape[0], -1) / max(self.embedding_proto_temp, 1e-4)
        return features, class_logits, proto_logits

    def waveform_atlas_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stat = (waveform_atlas_stats(x) - self.waveform_atlas_mean[None, :]) / self.waveform_atlas_std[None, :]
        dist = torch.mean((stat[:, None, None, :] - self.waveform_atlas_prototypes[None, :, :, :]) ** 2, dim=3)
        min_dist = dist.min(dim=2).values
        mean_dist = dist.mean(dim=2)
        purity_like = torch.softmax(-min_dist / max(self.waveform_atlas_temp, 1e-4), dim=1)
        features = torch.cat([min_dist, mean_dist, purity_like], dim=1)
        class_logits = -min_dist / max(self.waveform_atlas_temp, 1e-4)
        proto_logits = (-dist / max(self.waveform_atlas_temp, 1e-4)).reshape(x.shape[0], -1)
        return features, class_logits, proto_logits

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> dict[str, torch.Tensor | None]:
        recon = None
        recon_mask = None
        if self.arch == "masked_mae":
            feat, recon, recon_mask = self.encoder(x, mask_ratio)
        else:
            feat = self.encoder(x)
        emb = self.project(feat)
        primitive_context_emb = None
        if self.primitive_context_encoder is not None and self.primitive_context_project is not None:
            primitive_context_emb = self.primitive_context_project(self.primitive_context_encoder(x))
        primitive_bad_logit = None
        primitive_emb = None
        primitive_threshold_emb = None
        if self.primitive_branch is not None:
            primitive_stats = primitive_waveform_stats(x, self.primitive_bank)
            primitive_emb = self.primitive_branch(primitive_stats)
            if self.primitive_bad_head is not None and self.primitive_bad_branch is None:
                primitive_bad_logit = self.primitive_bad_head(primitive_emb).squeeze(1)
            if self.primitive_threshold_fusion and self.primitive_threshold_proj is not None:
                z = self.primitive_threshold_norm(primitive_stats)
                thresholds = self.primitive_threshold_values.view(1, 1, -1)
                act = torch.sigmoid(self.primitive_threshold_sharpness * (z.unsqueeze(-1) - thresholds))
                primitive_threshold_emb = self.primitive_threshold_proj(act.flatten(start_dim=1))
        if self.primitive_bad_branch is not None and self.primitive_bad_head is not None:
            primitive_bad_stats = primitive_waveform_stats(x, self.primitive_bad_bank)
            primitive_bad_emb = self.primitive_bad_branch(primitive_bad_stats)
            primitive_bad_logit = self.primitive_bad_head(primitive_bad_emb).squeeze(1)
        aux_input = emb
        if self.aux_primitive_fusion and primitive_emb is not None:
            aux_input = torch.cat([aux_input, primitive_emb], dim=1)
        if self.aux_primitive_context_fusion and primitive_context_emb is not None:
            aux_input = torch.cat([aux_input, primitive_context_emb], dim=1)
        aux_pred = self.aux_head(aux_input)
        if self.use_teacher_atlas:
            atlas_pred = self.teacher_distance_features(aux_pred)
            head_input = torch.cat([emb, atlas_pred], dim=1)
        else:
            atlas_pred = None
            head_input = emb
        proto_logits = None
        proto_class_logits = None
        if self.use_embedding_prototypes:
            proto_features, proto_class_logits, proto_logits = self.embedding_prototype_outputs(emb)
            head_input = torch.cat([head_input, proto_features], dim=1)
        waveform_proto_logits = None
        waveform_proto_class_logits = None
        if self.use_waveform_atlas:
            waveform_features, waveform_proto_class_logits, waveform_proto_logits = self.waveform_atlas_outputs(x)
            head_input = torch.cat([head_input, waveform_features], dim=1)
        if self.predicted_feature_indices:
            head_input = torch.cat([head_input, aux_pred[:, self.predicted_feature_indices]], dim=1)
        if primitive_emb is not None:
            if self.primitive_fusion:
                head_input = torch.cat([head_input, primitive_emb], dim=1)
        if self.primitive_context_fusion and primitive_context_emb is not None:
            head_input = torch.cat([head_input, primitive_context_emb], dim=1)
        if primitive_threshold_emb is not None:
            head_input = torch.cat([head_input, primitive_threshold_emb], dim=1)
        logits = self.class_head(head_input)
        feature_gate = None
        if proto_class_logits is not None and self.embedding_proto_logit_mix != 0.0:
            logits = logits + self.embedding_proto_logit_mix * proto_class_logits
        if waveform_proto_class_logits is not None and self.waveform_atlas_logit_mix != 0.0:
            logits = logits + self.waveform_atlas_logit_mix * waveform_proto_class_logits
        if self.feature_class_head is not None and self.predicted_feature_indices:
            residual_logits = logits
            feature_values = aux_pred[:, self.predicted_feature_indices]
            feature_logits = self.feature_class_head(feature_values)
            feature_first_logits = feature_logits + self.feature_residual_alpha * residual_logits
            if self.feature_gate_head is not None:
                feature_gate = torch.sigmoid(self.feature_gate_head(feature_values)).clamp(0.02, 0.98)
                logits = feature_gate * feature_first_logits + (1.0 - feature_gate) * residual_logits
            else:
                logits = feature_first_logits
        bad_logit = self.bad_head(emb).squeeze(1) if self.bad_head is not None else None
        stress_logit = self.stress_head(emb).squeeze(1) if self.stress_head is not None else None
        if primitive_bad_logit is not None:
            if bad_logit is None:
                bad_logit = primitive_bad_logit
            else:
                bad_logit = bad_logit + self.primitive_bad_mix * primitive_bad_logit
        if stress_logit is not None and self.stress_bad_logit_mix != 0.0:
            if bad_logit is None:
                bad_logit = self.stress_bad_logit_mix * stress_logit
            else:
                bad_logit = bad_logit + self.stress_bad_logit_mix * stress_logit
        if self.bad_margin_guard and bad_logit is not None:
            margin_guard = torch.zeros_like(bad_logit)
            if self.pca_margin_idx >= 0:
                margin_guard = margin_guard - self.bad_margin_alpha * aux_pred[:, self.pca_margin_idx]
            if self.knn_purity_idx >= 0:
                margin_guard = margin_guard - self.bad_purity_alpha * aux_pred[:, self.knn_purity_idx]
            if self.boundary_conf_idx >= 0:
                margin_guard = margin_guard - self.bad_boundary_alpha * aux_pred[:, self.boundary_conf_idx]
            bad_logit = bad_logit + torch.clamp(margin_guard, min=-2.5, max=2.5)
        gm_pair_logits = self.gm_pair_head(head_input) if self.gm_pair_head is not None else None
        if self.final_mode in self.dual_bad_modes and bad_logit is not None and gm_pair_logits is not None:
            gm_log_probs = F.log_softmax(gm_pair_logits, dim=1)
            log_not_bad = F.logsigmoid(-bad_logit)
            log_bad = F.logsigmoid(bad_logit)
            dual_logits = torch.stack(
                [
                    gm_log_probs[:, CLASS_TO_INT["good"]] + log_not_bad,
                    gm_log_probs[:, CLASS_TO_INT["medium"]] + log_not_bad,
                    log_bad,
                ],
                dim=1,
            )
            dual_logits = torch.nan_to_num(dual_logits, nan=0.0, posinf=30.0, neginf=-30.0)
            if self.final_mode == "dual_bad_gm_replace":
                logits = dual_logits
            else:
                logits = (1.0 - self.gate_logit_alpha) * logits + self.gate_logit_alpha * dual_logits
        elif self.final_mode in {"gated_bad", "gated_bad_mix", "gated_bad_replace"} and bad_logit is not None and gm_pair_logits is not None:
            log_not_bad = F.logsigmoid(-bad_logit)
            gated_logits = torch.stack(
                [
                    gm_pair_logits[:, CLASS_TO_INT["good"]] + log_not_bad,
                    gm_pair_logits[:, CLASS_TO_INT["medium"]] + log_not_bad,
                    bad_logit,
                ],
                dim=1,
            )
            gated_logits = torch.nan_to_num(gated_logits, nan=0.0, posinf=30.0, neginf=-30.0)
            if self.final_mode == "gated_bad_replace":
                logits = gated_logits
            elif self.final_mode == "gated_bad_mix":
                logits = (1.0 - self.gate_logit_alpha) * logits + self.gate_logit_alpha * gated_logits
            else:
                logits = logits + gated_logits
        elif bad_logit is not None and self.bad_logit_mix != 0.0:
            logits = logits.clone()
            logits[:, CLASS_TO_INT["bad"]] = logits[:, CLASS_TO_INT["bad"]] + self.bad_logit_mix * bad_logit
        return {
            "logits": logits,
            "bad_logit": bad_logit,
            "stress_logit": stress_logit,
            "primitive_bad_logit": primitive_bad_logit,
            "gm_pair_logits": gm_pair_logits,
            "aux_pred": aux_pred,
            "atlas_pred": atlas_pred,
            "proto_logits": proto_logits,
            "proto_class_logits": proto_class_logits,
            "waveform_proto_logits": waveform_proto_logits,
            "waveform_proto_class_logits": waveform_proto_class_logits,
            "feature_gate": feature_gate,
            "embedding": emb,
            "recon": recon,
            "recon_mask": recon_mask,
        }


class AugmentedSyntheticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base: Any,
        strength: float,
        seed: int,
        bad_strength: float,
        bad_stress_strength: float = 0.0,
        nonbad_hardneg_strength: float = 0.0,
        bad_highamp_strength: float = 0.0,
        bad_highamp_prob: float = 1.0,
        bad_intermittent_contact_strength: float = 0.0,
        bad_intermittent_contact_prob: float = 1.0,
        subject111_shift_strength: float = 0.0,
        subject111_good_shift_scale: float = 0.72,
        subject111_medium_shift_scale: float = 1.15,
        bad_record111_motion_strength: float = 0.0,
        bad_record111_motion_prob: float = 1.0,
        bad_impulse_reset_strength: float = 0.0,
        bad_impulse_reset_prob: float = 1.0,
        bad_record111_burst_dropout_strength: float = 0.0,
        bad_record111_burst_dropout_prob: float = 1.0,
        aug_aux_weight_bad: float = 1.0,
        aug_aux_weight_nonbad: float = 1.0,
        nonbad_to_bad_stress_prob: float = 0.0,
        nonbad_to_bad_stress_strength: float = 0.0,
    ):
        self.base = base
        self.strength = float(strength)
        self.bad_strength = float(bad_strength)
        self.bad_stress_strength = float(bad_stress_strength)
        self.nonbad_hardneg_strength = float(nonbad_hardneg_strength)
        self.bad_highamp_strength = float(bad_highamp_strength)
        self.bad_highamp_prob = float(bad_highamp_prob)
        self.bad_intermittent_contact_strength = float(bad_intermittent_contact_strength)
        self.bad_intermittent_contact_prob = float(bad_intermittent_contact_prob)
        self.subject111_shift_strength = float(subject111_shift_strength)
        self.subject111_good_shift_scale = float(subject111_good_shift_scale)
        self.subject111_medium_shift_scale = float(subject111_medium_shift_scale)
        self.bad_record111_motion_strength = float(bad_record111_motion_strength)
        self.bad_record111_motion_prob = float(bad_record111_motion_prob)
        self.bad_impulse_reset_strength = float(bad_impulse_reset_strength)
        self.bad_impulse_reset_prob = float(bad_impulse_reset_prob)
        self.bad_record111_burst_dropout_strength = float(bad_record111_burst_dropout_strength)
        self.bad_record111_burst_dropout_prob = float(bad_record111_burst_dropout_prob)
        self.aug_aux_weight_bad = float(aug_aux_weight_bad)
        self.aug_aux_weight_nonbad = float(aug_aux_weight_nonbad)
        self.nonbad_to_bad_stress_prob = float(nonbad_to_bad_stress_prob)
        self.nonbad_to_bad_stress_strength = float(nonbad_to_bad_stress_strength)
        self.rng = np.random.default_rng(seed)
        self.x = base.x
        self.y = base.y
        self.aux = base.aux
        self.mean = base.mean
        self.std = base.std
        self.frame = base.frame

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self.base[i]
        x = item["x"].detach().cpu().numpy().astype(np.float32, copy=True)
        y = int(item["y"])
        converted_nonbad_to_bad = (
            y != CLASS_TO_INT["bad"]
            and self.nonbad_to_bad_stress_prob > 0
            and self.nonbad_to_bad_stress_strength > 0
            and self.rng.random() < self.nonbad_to_bad_stress_prob
        )
        stress_target = 0.0
        if converted_nonbad_to_bad:
            x = augment_synthetic_waveform(x, self.rng, self.strength, y, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            x = augment_bad_intermittent_contact_qrs(x, self.rng, self.nonbad_to_bad_stress_strength)
            y = CLASS_TO_INT["bad"]
            stress_target = 1.0
        else:
            effective_contact_strength = self.bad_intermittent_contact_strength
            effective_record111_strength = self.bad_record111_motion_strength
            effective_impulse_strength = self.bad_impulse_reset_strength
            effective_burstdrop_strength = self.bad_record111_burst_dropout_strength
            if y == CLASS_TO_INT["bad"] and effective_contact_strength > 0 and self.rng.random() > self.bad_intermittent_contact_prob:
                effective_contact_strength = 0.0
            if y == CLASS_TO_INT["bad"] and effective_record111_strength > 0 and self.rng.random() > self.bad_record111_motion_prob:
                effective_record111_strength = 0.0
            if y == CLASS_TO_INT["bad"] and effective_impulse_strength > 0 and self.rng.random() > self.bad_impulse_reset_prob:
                effective_impulse_strength = 0.0
            if y == CLASS_TO_INT["bad"] and effective_burstdrop_strength > 0 and self.rng.random() > self.bad_record111_burst_dropout_prob:
                effective_burstdrop_strength = 0.0
            x = augment_synthetic_waveform(
                x,
                self.rng,
                self.strength,
                y,
                self.bad_strength,
                self.bad_stress_strength,
                self.nonbad_hardneg_strength,
                self.bad_highamp_strength,
                self.bad_highamp_prob,
                effective_contact_strength,
                effective_record111_strength,
                effective_impulse_strength,
                effective_burstdrop_strength,
            )
            if y != CLASS_TO_INT["bad"] and self.subject111_shift_strength > 0:
                x = augment_subject111_nonbad_shift(
                    x,
                    self.rng,
                    self.subject111_shift_strength,
                    y,
                    self.subject111_good_shift_scale,
                    self.subject111_medium_shift_scale,
                )
            if y == CLASS_TO_INT["bad"] and (
                self.bad_stress_strength > 0
                or self.bad_highamp_strength > 0
                or effective_contact_strength > 0
                or effective_record111_strength > 0
                or effective_impulse_strength > 0
                or effective_burstdrop_strength > 0
            ):
                stress_target = 1.0
        aux_weight = 1.0
        if converted_nonbad_to_bad:
            aux_weight = self.aug_aux_weight_bad
        elif y == CLASS_TO_INT["bad"] and (
            self.bad_strength > 0
            or self.bad_stress_strength > 0
            or self.bad_highamp_strength > 0
            or self.bad_intermittent_contact_strength > 0
            or self.bad_record111_motion_strength > 0
            or self.bad_impulse_reset_strength > 0
            or self.bad_record111_burst_dropout_strength > 0
        ):
            aux_weight = self.aug_aux_weight_bad
        elif y != CLASS_TO_INT["bad"] and (self.nonbad_hardneg_strength > 0 or self.subject111_shift_strength > 0):
            aux_weight = self.aug_aux_weight_nonbad
        return {
            "x": torch.from_numpy(x),
            "aux": item["aux"],
            "aux_weight": torch.tensor(aux_weight, dtype=torch.float32),
            "stress_target": torch.tensor(stress_target, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }


def waveform_stats_v2(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    diff = x[:, :, 1:] - x[:, :, :-1]
    mean = x.mean(dim=2)
    std = x.std(dim=2)
    mean_abs = x.abs().mean(dim=2)
    rms = torch.sqrt((x * x).mean(dim=2) + eps)
    peak_to_peak = x.amax(dim=2) - x.amin(dim=2)
    diff_abs = diff.abs().mean(dim=2)
    diff_rms = torch.sqrt((diff * diff).mean(dim=2) + eps)
    flat = (diff.abs() < 0.015).float().mean(dim=2)
    low_amp = (x.abs() < 0.050).float().mean(dim=2)
    zcr = ((x[:, :, 1:] * x[:, :, :-1]) < 0).float().mean(dim=2)
    k10 = max(1, int(diff.shape[2] * 0.10))
    k02 = max(1, int(diff.shape[2] * 0.02))
    diff_tail10 = torch.topk(diff.abs(), k=k10, dim=2).values.mean(dim=2)
    diff_tail02 = torch.topk(diff.abs(), k=k02, dim=2).values.mean(dim=2)
    smooth_short = F.avg_pool1d(x, kernel_size=15, stride=1, padding=7)
    smooth_long = F.avg_pool1d(x, kernel_size=101, stride=1, padding=50)
    detail_short = (x - smooth_short).abs().mean(dim=2)
    detail_long = (x - smooth_long).abs().mean(dim=2)
    baseline_slope = (smooth_long[:, :, -1] - smooth_long[:, :, 0]).abs()
    contact = (x.abs() < 0.015).float().mean(dim=2)
    return torch.cat(
        [
            mean,
            std,
            mean_abs,
            rms,
            peak_to_peak,
            diff_abs,
            diff_rms,
            flat,
            low_amp,
            zcr,
            diff_tail10,
            diff_tail02,
            detail_short,
            detail_long,
            baseline_slope,
            contact,
        ],
        dim=1,
    )


def augment_synthetic_waveform(
    x: np.ndarray,
    rng: np.random.Generator,
    strength: float,
    y: int,
    bad_strength: float,
    bad_stress_strength: float = 0.0,
    nonbad_hardneg_strength: float = 0.0,
    bad_highamp_strength: float = 0.0,
    bad_highamp_prob: float = 1.0,
    bad_intermittent_contact_strength: float = 0.0,
    bad_record111_motion_strength: float = 0.0,
    bad_impulse_reset_strength: float = 0.0,
    bad_record111_burst_dropout_strength: float = 0.0,
) -> np.ndarray:
    if strength > 0:
        if rng.random() < 0.65:
            x = np.roll(x, int(rng.integers(-32, 33)), axis=1)
        if rng.random() < 0.75:
            x *= float(rng.uniform(0.88, 1.14))
        if rng.random() < 0.55:
            x += rng.normal(0.0, float(rng.uniform(0.005, 0.025) * strength), size=x.shape).astype(np.float32)
        if rng.random() < 0.35:
            width = int(rng.integers(18, 90))
            start = int(rng.integers(0, max(1, x.shape[1] - width)))
            x[:, start : start + width] *= float(rng.uniform(0.15, 0.70))
    if y == CLASS_TO_INT["bad"] and bad_strength > 0:
        x = augment_controlled_bad_outlier(x, rng, bad_strength)
    if y == CLASS_TO_INT["bad"] and bad_stress_strength > 0:
        x = augment_bad_stress_shell(x, rng, bad_stress_strength)
    if y == CLASS_TO_INT["bad"] and bad_highamp_strength > 0 and rng.random() < float(bad_highamp_prob):
        x = augment_bad_highamp_baseline_detail(x, rng, bad_highamp_strength)
    if y == CLASS_TO_INT["bad"] and bad_intermittent_contact_strength > 0:
        x = augment_bad_intermittent_contact_qrs(x, rng, bad_intermittent_contact_strength)
    if y == CLASS_TO_INT["bad"] and bad_record111_motion_strength > 0:
        x = augment_bad_record111_motion(x, rng, bad_record111_motion_strength)
    if y == CLASS_TO_INT["bad"] and bad_impulse_reset_strength > 0:
        x = augment_bad_impulse_reset(x, rng, bad_impulse_reset_strength)
    if y == CLASS_TO_INT["bad"] and bad_record111_burst_dropout_strength > 0:
        x = augment_bad_record111_burst_dropout(x, rng, bad_record111_burst_dropout_strength)
    if y != CLASS_TO_INT["bad"] and nonbad_hardneg_strength > 0:
        x = augment_nonbad_artifact_hard_negative(x, rng, nonbad_hardneg_strength)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_bad_highamp_baseline_detail(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """High-amplitude/detail bad outlier shell.

    Primitive-gap analysis showed original bad outliers have large positive
    baseline span/curve and QRS/detail-tail primitives, while our previous
    synthetic bad shell was mostly low-amplitude/flat.  This augmentation
    creates broad high-amplitude baseline drift plus detail bursts without
    erasing all signal into a flatline.
    """

    n = x.shape[1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    # Robust channel 1/2 carried the strongest missing shell in the gap report.
    preferred = [1, 2, 0] if x.shape[0] >= 3 else list(range(x.shape[0]))
    ch = int(preferred[int(rng.integers(0, len(preferred)))])
    if rng.random() < min(0.98, 0.55 + 0.20 * float(strength)):
        drift = float(rng.uniform(0.35, 1.25) * float(strength))
        curve = float(rng.uniform(0.20, 0.85) * float(strength))
        phase = float(rng.uniform(-np.pi, np.pi))
        x[ch] += drift * (0.55 * t + 0.45 * np.sin(np.pi * t + phase)) + curve * (t * t - 0.35)
    if x.shape[0] > 1 and rng.random() < 0.65:
        ch2 = int(preferred[int(rng.integers(0, len(preferred)))])
        x[ch2] *= float(rng.uniform(1.35, 2.60))
    if rng.random() < min(0.95, 0.45 + 0.22 * float(strength)):
        bursts = int(rng.integers(2, 5))
        for _ in range(bursts):
            width = int(rng.integers(max(18, n // 120), max(45, n // 18)))
            start = int(rng.integers(0, max(1, n - width)))
            local_t = np.linspace(-1.0, 1.0, width, dtype=np.float32)
            amp = float(rng.uniform(0.10, 0.38) * float(strength))
            carrier = np.sin(2 * np.pi * float(rng.uniform(8.0, 22.0)) * np.linspace(0, 1, width, dtype=np.float32))
            envelope = np.exp(-2.5 * local_t * local_t).astype(np.float32)
            channel = int(preferred[int(rng.integers(0, len(preferred)))])
            x[channel, start : start + width] += amp * carrier.astype(np.float32) * envelope
    if rng.random() < min(0.80, 0.30 + 0.20 * float(strength)):
        noise_scale = float(rng.uniform(0.025, 0.090) * float(strength))
        x += rng.normal(0.0, noise_scale, size=x.shape).astype(np.float32)
    if rng.random() < 0.35:
        clip = float(rng.uniform(1.05, 2.50) * max(1.0, float(strength) * 0.65))
        x = np.clip(x, -clip, clip)
    return x


def augment_nonbad_artifact_hard_negative(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Local artifacts that should remain good/medium, not bad.

    The original bad-frontier analysis shows aggressive bad thresholds confuse
    non-bad outlier/overlap windows with bad stress.  These PTB-only hard
    negatives teach the bad expert that short local dropouts, mild baseline
    drift, and moderate detail loss are not sufficient for the bad label.
    """

    n = x.shape[1]
    if rng.random() < min(0.90, 0.35 + 0.25 * float(strength)):
        width = int(rng.integers(max(24, n // 80), max(60, n // 9)))
        start = int(rng.integers(0, max(1, n - width)))
        x[:, start : start + width] *= float(rng.uniform(0.22, 0.78))
    if rng.random() < min(0.85, 0.30 + 0.22 * float(strength)):
        t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        drift = float(rng.uniform(0.025, 0.115) * float(strength))
        x += drift * (0.60 * t + 0.40 * np.sin(np.pi * t + float(rng.uniform(-1.5, 1.5))))[None, :]
    if rng.random() < min(0.70, 0.25 + 0.20 * float(strength)):
        x += rng.normal(0.0, float(rng.uniform(0.006, 0.030) * float(strength)), size=x.shape).astype(np.float32)
    if rng.random() < min(0.55, 0.18 + 0.16 * float(strength)):
        channel = int(rng.integers(0, x.shape[0]))
        width = int(rng.integers(max(20, n // 100), max(50, n // 12)))
        start = int(rng.integers(0, max(1, n - width)))
        x[channel, start : start + width] += float(rng.uniform(-0.12, 0.12))
    return x


def augment_subject111_nonbad_shift(
    x: np.ndarray,
    rng: np.random.Generator,
    strength: float,
    y: int,
    good_scale: float = 0.72,
    medium_scale: float = 1.15,
) -> np.ndarray:
    """Low-QRS/high-LF non-bad domain shift seen in original test record 111001.

    This PTB-only augmentation targets the train-vs-test split gap: lower
    qrs_slope/prominence and diff tails, stronger low-frequency/baseline
    content, and higher local RMS variability.  Labels are preserved; medium
    gets the stronger version so low QRS does not become a shortcut for good.
    """

    n = x.shape[1]
    s = float(strength) * (float(medium_scale) if y == CLASS_TO_INT["medium"] else float(good_scale))
    if s <= 0:
        return x
    kernel = int(rng.choice([21, 31, 51, 71]))
    filt = np.ones(kernel, dtype=np.float32) / float(kernel)
    for ch in range(x.shape[0]):
        if rng.random() < min(0.95, 0.35 + 0.25 * s):
            padded = np.pad(x[ch], (kernel // 2, kernel // 2), mode="edge")
            smooth = np.convolve(padded, filt, mode="valid")[:n]
            mix = float(rng.uniform(0.28, 0.68) * min(1.0, 0.55 + 0.20 * s))
            x[ch] = (1.0 - mix) * x[ch] + mix * smooth
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    lf_amp = float(rng.uniform(0.05, 0.22) * s)
    lf = lf_amp * (
        0.48 * np.sin(np.pi * t + float(rng.uniform(-1.4, 1.4)))
        + 0.32 * np.sin(2 * np.pi * float(rng.uniform(0.18, 0.42)) * (t + 1.0) + float(rng.uniform(-np.pi, np.pi)))
        + 0.20 * t
    )
    x += lf[None, :]
    if rng.random() < min(0.90, 0.30 + 0.25 * s):
        env = 1.0 + float(rng.uniform(0.10, 0.34) * s) * np.sin(np.pi * t + float(rng.uniform(-1.5, 1.5)))
        x *= env[None, :].astype(np.float32)
    if y == CLASS_TO_INT["medium"] and rng.random() < min(0.85, 0.25 + 0.22 * s):
        chunks = int(rng.integers(1, 4))
        for _ in range(chunks):
            width = int(rng.integers(max(36, n // 40), max(80, n // 8)))
            start = int(rng.integers(0, max(1, n - width)))
            end = start + width
            ch = int(rng.integers(0, x.shape[0]))
            x[ch, start:end] *= float(rng.uniform(0.15, 0.55))
            if rng.random() < 0.65:
                x[ch, start:end] += rng.normal(0.0, float(rng.uniform(0.004, 0.020) * s), size=width).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_bad_impulse_reset(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Bad outlier impulse/reset block seen in original test record 111001.

    The failed bad-outlier audit showed many test bad windows are not just
    smooth low-QRS drift.  They contain abrupt vertical resets, short dropouts,
    and large same-time impulses in one or two channels while another channel
    keeps a smooth LF trajectory.  This block is mixed in at low probability so
    it expands bad stress without replacing the bad core.
    """

    n = x.shape[1]
    s = float(strength)
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    if x.shape[0] >= 3:
        drift_ch = 2
        drift = float(rng.uniform(0.28, 0.95) * s)
        x[drift_ch] = (
            0.45 * x[drift_ch]
            + drift
            * (
                0.55 * np.sin(np.pi * t + float(rng.uniform(-1.2, 1.2)))
                + 0.30 * t
                + 0.15 * np.sin(2 * np.pi * float(rng.uniform(0.6, 1.6)) * (t + 1.0))
            )
        ).astype(np.float32)
    event_count = int(rng.integers(1, 4))
    event_channels = [0, 1] if x.shape[0] >= 2 else [0]
    for _ in range(event_count):
        center = int(rng.integers(max(8, n // 20), max(9, n - n // 20)))
        width = int(rng.integers(5, 32))
        start = max(0, center - width)
        end = min(n, center + width + 1)
        local = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
        spike = np.exp(-24.0 * local * local).astype(np.float32)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        for ch in event_channels:
            if rng.random() < 0.78:
                amp = float(rng.uniform(1.8, 7.5) * s)
                x[ch, start:end] += sign * amp * spike
                if rng.random() < 0.55:
                    step_len = int(rng.integers(20, 95))
                    step_end = min(n, center + step_len)
                    x[ch, center:step_end] += sign * float(rng.uniform(0.25, 1.15) * s)
        if rng.random() < 0.70:
            ch = int(event_channels[int(rng.integers(0, len(event_channels)))])
            drop_width = int(rng.integers(20, 110))
            drop_start = int(np.clip(center - drop_width // 2, 0, max(0, n - drop_width)))
            drop_end = min(n, drop_start + drop_width)
            x[ch, drop_start:drop_end] *= float(rng.uniform(0.0, 0.18))
            edge_amp = float(rng.uniform(0.8, 3.5) * s)
            x[ch, drop_start : min(n, drop_start + 4)] += edge_amp
            x[ch, max(0, drop_end - 4) : drop_end] -= edge_amp
    if rng.random() < 0.65:
        x += rng.normal(0.0, float(rng.uniform(0.003, 0.018) * s), size=x.shape).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_bad_record111_burst_dropout(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Record-111 bad stress with long low-amplitude dropout plus sparse resets.

    The v5 sparse-event branch recovered some held-out bad outliers but also
    marked many non-bad outlier rows as bad.  This generation block is narrower:
    one channel becomes low-amplitude/near-dropout for long spans, another keeps
    a weak broad drift, and short reset/burst edges are sparse and asymmetric.
    It avoids the strong all-channel motion used by record111_motion.
    """

    n = x.shape[1]
    s = float(strength)
    if s <= 0:
        return x
    t01 = np.linspace(0.0, 1.0, n, dtype=np.float32)
    t11 = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    channels = list(range(x.shape[0]))
    if x.shape[0] >= 3:
        low_ch = 1
        drift_ch = 2
        burst_ch = 0
    else:
        low_ch = int(channels[0])
        drift_ch = int(channels[-1])
        burst_ch = int(channels[0])

    # Long low-amplitude dropout/plateau, but with enough residual noise and
    # edge activity that it is not a trivial flatline/contact label shortcut.
    low_scale = float(rng.uniform(0.015, 0.085) / max(1.0, 0.35 * s))
    x[low_ch] = low_scale * x[low_ch] + rng.normal(0.0, float(rng.uniform(0.0015, 0.0060) * s), size=n).astype(np.float32)
    for _ in range(int(rng.integers(1, 4))):
        width = int(rng.integers(max(120, n // 7), max(180, n // 2)))
        start = int(rng.integers(0, max(1, n - width)))
        end = start + width
        plateau = float(rng.uniform(-0.018, 0.018) * s)
        x[low_ch, start:end] = plateau + rng.normal(0.0, float(rng.uniform(0.001, 0.006) * s), size=width).astype(np.float32)
        edge = float(rng.uniform(0.10, 0.46) * s)
        edge_width = int(rng.integers(3, 12))
        x[low_ch, start : min(n, start + edge_width)] += edge * (1.0 if rng.random() < 0.5 else -1.0)
        x[low_ch, max(0, end - edge_width) : end] -= edge * (1.0 if rng.random() < 0.5 else -1.0)

    # Weak drifting channel: enough low-frequency movement to match record111,
    # less than the previous strong motion block that overshot the PCA target.
    drift_amp = float(rng.uniform(0.22, 0.82) * s)
    drift = (
        0.46 * np.sin(np.pi * t11 + float(rng.uniform(-1.4, 1.4)))
        + 0.31 * t11
        + 0.23 * np.sin(2 * np.pi * float(rng.uniform(0.7, 2.4)) * t01 + float(rng.uniform(-np.pi, np.pi)))
    ).astype(np.float32)
    x[drift_ch] = (float(rng.uniform(0.20, 0.52)) * x[drift_ch] + drift_amp * drift).astype(np.float32)

    # A separate weak-QRS channel carries wide lurch/dropout events.  The
    # original record111 bad-stress examples have broad blue-channel excursions
    # plus orange-channel burst clusters, not a stationary noisy waveform.
    kernel = int(rng.choice([41, 61, 81]))
    filt = np.ones(kernel, dtype=np.float32) / float(kernel)
    padded = np.pad(x[burst_ch], (kernel // 2, kernel // 2), mode="edge")
    smooth = np.convolve(padded, filt, mode="valid")[:n]
    x[burst_ch] = (float(rng.uniform(0.12, 0.34)) * x[burst_ch] + float(rng.uniform(0.20, 0.45)) * smooth).astype(np.float32)
    for _ in range(int(rng.integers(1, 4))):
        width = int(rng.integers(max(55, n // 25), max(110, n // 7)))
        start = int(rng.integers(0, max(1, n - width)))
        end = start + width
        local = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        envelope = np.sin(np.pi * (local + 1.0) / 2.0).astype(np.float32)
        lurch = (
            float(rng.uniform(-0.85, 0.85) * s)
            * (0.62 * np.sign(local) + 0.38 * np.sin(np.pi * local + float(rng.uniform(-1.0, 1.0))))
            * envelope
        ).astype(np.float32)
        x[burst_ch, start:end] += lurch
        if rng.random() < 0.58:
            x[burst_ch, start:end] *= float(rng.uniform(0.08, 0.36))
        if rng.random() < 0.70:
            x[burst_ch, start:end] += rng.normal(0.0, float(rng.uniform(0.010, 0.045) * s), size=width).astype(np.float32)

    # Orange/low channel gets short noisy clusters separated by quiet spans.
    for _ in range(int(rng.integers(2, 6))):
        width = int(rng.integers(max(14, n // 140), max(34, n // 32)))
        start = int(rng.integers(0, max(1, n - width)))
        end = start + width
        local = np.linspace(0.0, 1.0, width, dtype=np.float32)
        carrier = np.sin(2 * np.pi * float(rng.uniform(8.0, 24.0)) * local + float(rng.uniform(-np.pi, np.pi)))
        cluster = (
            float(rng.uniform(0.06, 0.30) * s)
            * carrier.astype(np.float32)
            * np.sin(np.pi * local).astype(np.float32)
        )
        x[low_ch, start:end] += cluster.astype(np.float32)
        if rng.random() < 0.72:
            x[low_ch, start:end] += rng.normal(0.0, float(rng.uniform(0.006, 0.035) * s), size=width).astype(np.float32)

    for _ in range(int(rng.integers(3, 8))):
        center = int(rng.integers(8, max(9, n - 8)))
        width = int(rng.integers(3, 18))
        start = max(0, center - width)
        end = min(n, center + width + 1)
        local = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
        spike = np.exp(-16.0 * local * local).astype(np.float32)
        ch = int(rng.choice([low_ch, burst_ch, drift_ch]))
        amp = float(rng.uniform(0.08, 0.58) * s)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        x[ch, start:end] += sign * amp * spike
        if rng.random() < 0.38:
            step_len = int(rng.integers(16, 80))
            step_end = min(n, center + step_len)
            x[ch, center:step_end] += sign * float(rng.uniform(0.04, 0.24) * s)

    # Mild same-time edge disagreement across two channels, a signature the bad
    # expert can learn without treating every high-frequency burst as bad.
    if x.shape[0] >= 2 and rng.random() < 0.60:
        center = int(rng.integers(max(10, n // 20), max(11, n - n // 20)))
        edge_len = int(rng.integers(8, 32))
        step = float(rng.uniform(0.10, 0.42) * s)
        a, b = int(low_ch), int(drift_ch)
        x[a, center : min(n, center + edge_len)] += step
        x[b, center : min(n, center + edge_len)] -= float(rng.uniform(0.35, 0.90)) * step

    if rng.random() < 0.45:
        x += rng.normal(0.0, float(rng.uniform(0.002, 0.010) * s), size=x.shape).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_bad_record111_motion(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Record-111-like bad stress: low QRS-band, strong LF motion, non-flat.

    Previous bad augmentations over-emphasized contact/flatline.  Original-test
    bad outliers instead look like broad 1-5 Hz motion and low-QRS-band windows
    where the signal is not simply erased.  This remains PTB-only generation.
    """

    n = x.shape[1]
    s = float(strength)
    t01 = np.linspace(0.0, 1.0, n, dtype=np.float32)
    t11 = np.linspace(-1.0, 1.0, n, dtype=np.float32)

    if s >= 2.0 and x.shape[0] >= 3:
        # Stronger target-like mode from the gap audit: original record111 bad
        # has near-zero channel-1, low-amplitude channel-0 with sparse spikes,
        # and a large smooth channel-2 drift, not broadband noisy PTB.
        ch1_noise = rng.normal(0.0, float(rng.uniform(0.0025, 0.0090) * s), size=n).astype(np.float32)
        x[1] = ch1_noise
        for _ in range(int(rng.integers(3, 8))):
            width = int(rng.integers(3, 18))
            center = int(rng.integers(4, max(5, n - 4)))
            start = max(0, center - width)
            end = min(n, center + width + 1)
            local = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
            spike = np.exp(-14.0 * local * local).astype(np.float32)
            x[1, start:end] += float(rng.uniform(0.08, 0.65) * s) * spike * (1.0 if rng.random() < 0.5 else -1.0)

        kernel = int(rng.choice([51, 71, 101]))
        filt = np.ones(kernel, dtype=np.float32) / float(kernel)
        padded0 = np.pad(x[0], (kernel // 2, kernel // 2), mode="edge")
        smooth0 = np.convolve(padded0, filt, mode="valid")[:n]
        x[0] = float(rng.uniform(0.05, 0.22)) * x[0] + float(rng.uniform(0.18, 0.42)) * smooth0
        x[0] += (float(rng.uniform(0.035, 0.14) * s) * np.sin(2 * np.pi * float(rng.uniform(6.0, 16.0)) * t01 + float(rng.uniform(-np.pi, np.pi)))).astype(np.float32)

        drift = float(rng.uniform(0.75, 2.10) * s)
        x[2] = (
            0.22 * x[2]
            + drift
            * (
                0.42 * t11
                + 0.36 * np.sin(np.pi * t11 + float(rng.uniform(-1.4, 1.4)))
                + 0.22 * np.sin(2 * np.pi * float(rng.uniform(1.0, 3.5)) * t01 + float(rng.uniform(-np.pi, np.pi)))
            )
        ).astype(np.float32)

        for _ in range(int(rng.integers(2, 6))):
            center = int(rng.integers(6, max(7, n - 6)))
            width = int(rng.integers(4, 16))
            start = max(0, center - width)
            end = min(n, center + width + 1)
            local = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
            spike = np.exp(-18.0 * local * local).astype(np.float32)
            x[0, start:end] += float(rng.uniform(0.18, 0.90) * s) * spike * (1.0 if rng.random() < 0.5 else -1.0)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Suppress QRS-band/detail without turning the whole window into a flatline.
    for ch in range(x.shape[0]):
        if rng.random() < min(0.98, 0.45 + 0.18 * s):
            kernel = int(rng.choice([31, 51, 71, 101]))
            filt = np.ones(kernel, dtype=np.float32) / float(kernel)
            padded = np.pad(x[ch], (kernel // 2, kernel // 2), mode="edge")
            smooth = np.convolve(padded, filt, mode="valid")[:n]
            mix = float(rng.uniform(0.42, 0.82) * min(1.0, 0.62 + 0.10 * s))
            x[ch] = (1.0 - mix) * x[ch] + mix * smooth
        if rng.random() < min(0.88, 0.30 + 0.16 * s):
            x[ch] *= float(rng.uniform(0.42, 0.82))

    # Strong low-frequency motion/body movement band, phase-shifted per channel.
    for ch in range(x.shape[0]):
        if rng.random() < min(0.96, 0.50 + 0.16 * s):
            amp = float(rng.uniform(0.10, 0.34) * s)
            cycles1 = float(rng.uniform(6.0, 16.0))
            cycles2 = float(rng.uniform(16.0, 34.0))
            phase1 = float(rng.uniform(-np.pi, np.pi))
            phase2 = float(rng.uniform(-np.pi, np.pi))
            motion = (
                0.58 * np.sin(2 * np.pi * cycles1 * t01 + phase1)
                + 0.30 * np.sin(2 * np.pi * cycles2 * t01 + phase2)
                + 0.12 * np.sin(np.pi * t11 + float(rng.uniform(-1.2, 1.2)))
            )
            x[ch] += (amp * motion).astype(np.float32)

    # Slow baseline shell, usually strongest in the last robust channel.
    if x.shape[0] > 2 and rng.random() < min(0.95, 0.45 + 0.20 * s):
        drift = float(rng.uniform(0.28, 0.95) * s)
        phase = float(rng.uniform(-np.pi, np.pi))
        x[2] += drift * (0.45 * t11 + 0.35 * np.sin(np.pi * t11 + phase) + 0.20 * (t11 * t11 - 0.35))

    # Motion bursts are low-frequency oscillatory, not high-frequency noise.
    bursts = int(rng.integers(1, 4))
    for _ in range(bursts):
        width = int(rng.integers(max(80, n // 12), max(120, n // 4)))
        start = int(rng.integers(0, max(1, n - width)))
        end = start + width
        local = np.linspace(0.0, 1.0, width, dtype=np.float32)
        envelope = np.sin(np.pi * local).astype(np.float32)
        channel = int(rng.integers(0, x.shape[0]))
        amp = float(rng.uniform(0.10, 0.38) * s)
        cycles = float(rng.uniform(3.0, 10.0))
        burst = amp * np.sin(2 * np.pi * cycles * local + float(rng.uniform(-np.pi, np.pi))) * envelope
        x[channel, start:end] += burst.astype(np.float32)
        if rng.random() < 0.65:
            x[channel, start:end] *= float(rng.uniform(0.30, 0.72))

    # Sparse inconsistent spikes keep detector agreement poor but avoid a
    # trivially flat/contact signature.
    if rng.random() < min(0.85, 0.35 + 0.18 * s):
        spike_count = int(rng.integers(3, 9))
        for _ in range(spike_count):
            center = int(rng.integers(4, max(5, n - 4)))
            width = int(rng.integers(3, 10))
            start = max(0, center - width)
            end = min(n, center + width + 1)
            local = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
            spike = np.exp(-16.0 * local * local).astype(np.float32)
            ch = int(rng.integers(0, x.shape[0]))
            x[ch, start:end] += float(rng.uniform(0.08, 0.42) * s) * spike * (1.0 if rng.random() < 0.5 else -1.0)

    if rng.random() < 0.55:
        x += rng.normal(0.0, float(rng.uniform(0.004, 0.018) * s), size=x.shape).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_controlled_bad_outlier(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    # Controlled bad expansion: broad but not absurd contact/baseline/detail
    # perturbations.  This is synthetic PTB augmentation only.
    n = x.shape[1]
    if rng.random() < 0.80:
        width = int(rng.integers(120, 360))
        start = int(rng.integers(0, max(1, n - width)))
        x[:, start : start + width] *= float(rng.uniform(0.05, 0.45))
    if rng.random() < 0.75:
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        drift = float(rng.uniform(0.08, 0.24) * strength)
        x += drift * np.sin(2 * np.pi * float(rng.uniform(0.08, 0.35)) * t + float(rng.uniform(0, 2 * np.pi)))[None, :]
    if rng.random() < 0.65:
        x += rng.normal(0.0, float(rng.uniform(0.035, 0.110) * strength), size=x.shape).astype(np.float32)
    if rng.random() < 0.45:
        clip = float(rng.uniform(0.45, 1.05))
        x = np.clip(x, -clip, clip)
    return x


def augment_bad_stress_shell(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Mimic original bad-stress robust3 shell: flat/contact spans plus baseline drift."""

    n = x.shape[1]
    if rng.random() < min(0.95, 0.55 + 0.22 * float(strength)):
        max_frac = 0.62 if strength < 1.8 else min(0.86, 0.62 + 0.10 * (float(strength) - 1.8))
        min_frac = 0.20 if strength < 1.8 else 0.30
        width = int(rng.integers(max(80, int(n * min_frac)), max(120, int(n * max_frac))))
        start = int(rng.integers(0, max(1, n - width)))
        level = float(rng.uniform(-0.18, 0.18))
        x[0, start : start + width] = level + rng.normal(0.0, 0.006 * float(strength), size=width).astype(np.float32)
        if x.shape[0] > 1:
            x[1, start : start + width] = rng.normal(0.0, 0.004 * float(strength), size=width).astype(np.float32)
        if x.shape[0] > 2:
            plateau = float(rng.uniform(-1.8, 1.8) * float(strength))
            ramp = np.linspace(-0.35, 0.35, width, dtype=np.float32) * float(rng.uniform(0.4, 1.4))
            x[2, start : start + width] += plateau + ramp
    if x.shape[0] > 2 and rng.random() < 0.85:
        t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        drift_hi = 1.65 if strength < 1.8 else min(3.10, 1.65 + 0.55 * (float(strength) - 1.8))
        drift = float(rng.uniform(0.55, drift_hi) * float(strength))
        x[2] += drift * (0.55 * t + 0.45 * np.sin(np.pi * t + float(rng.uniform(-1.2, 1.2))))
    if rng.random() < 0.45:
        width = int(rng.integers(max(60, n // 8), max(90, n // 3)))
        start = int(rng.integers(0, max(1, n - width)))
        x[:, start : start + width] *= float(rng.uniform(0.02, 0.20))
    if strength >= 2.0 and rng.random() < 0.40:
        width = int(rng.integers(max(100, n // 4), max(140, int(n * 0.72))))
        start = int(rng.integers(0, max(1, n - width)))
        channel = int(rng.integers(0, x.shape[0]))
        contact_level = float(rng.uniform(-0.05, 0.05))
        x[channel, start : start + width] = contact_level + rng.normal(0.0, 0.0035 * float(strength), size=width).astype(np.float32)
    return x


def augment_bad_intermittent_contact_qrs(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Bad-stress shell seen in original test bad/outlier rows.

    The failure analysis showed a bad subtype that is not right-island noise:
    QRS-like spikes remain visible, but long chunks have contact/dropout,
    low high-frequency/detail content, strong baseline curvature, high
    flatline ratio, and weak detector agreement.  This PTB-only augmentation
    creates that shape while leaving non-bad hard negatives to guard
    specificity.
    """

    n = x.shape[1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    s = float(strength)

    # Preserve a small set of peak locations so the result is not simply
    # erased bad noise; this matches the original test bad-stress examples.
    base = x[0].copy()
    smooth = np.convolve(base, np.ones(31, dtype=np.float32) / 31.0, mode="same")
    high = np.abs(base - smooth)
    if np.isfinite(high).all() and high.size:
        k = max(3, min(10, n // 140))
        peak_idx = np.argpartition(high, -k)[-k:]
    else:
        peak_idx = rng.integers(0, n, size=max(3, n // 250))

    # Broad baseline drift concentrated in channel 2, with smaller coupling to
    # channel 0.  This targets high pc2 / high baseline_step / low basSQI.
    if x.shape[0] > 2 and rng.random() < min(0.98, 0.62 + 0.15 * s):
        drift = float(rng.uniform(0.65, 1.85) * s)
        phase = float(rng.uniform(-np.pi, np.pi))
        x[2] += drift * (0.45 * t + 0.35 * np.sin(np.pi * t + phase) + 0.20 * (t * t - 0.35))
    if rng.random() < 0.65:
        x[0] += float(rng.uniform(0.08, 0.28) * s) * np.sin(np.pi * t + float(rng.uniform(-1.0, 1.0)))

    # Intermittent contact/dropout chunks: flatten one channel, attenuate
    # another, and smooth detail instead of making the whole window high noise.
    chunks = int(rng.integers(2, 5))
    for _ in range(chunks):
        width = int(rng.integers(max(36, n // 18), max(70, n // 5)))
        start = int(rng.integers(0, max(1, n - width)))
        end = start + width
        contact_ch = int(rng.integers(0, min(2, x.shape[0])))
        contact_level = float(rng.uniform(-0.10, 0.10))
        x[contact_ch, start:end] = contact_level + rng.normal(0.0, 0.0045 * s, size=width).astype(np.float32)
        atten_ch = int(rng.integers(0, x.shape[0]))
        x[atten_ch, start:end] *= float(rng.uniform(0.04, 0.35))
        if rng.random() < 0.75:
            ch = int(rng.integers(0, x.shape[0]))
            kernel = int(rng.choice([9, 15, 21, 31]))
            filt = np.ones(kernel, dtype=np.float32) / float(kernel)
            padded = np.pad(x[ch, start:end], (kernel // 2, kernel // 2), mode="edge")
            x[ch, start:end] = np.convolve(padded, filt, mode="valid")[:width]

    # Strong 111001-like stress: long low-detail contact/dropout spans plus a
    # large baseline shell.  Enable only at high strength so older candidates
    # remain comparable.
    if s >= 2.25 and rng.random() < min(0.92, 0.35 + 0.18 * s):
        width = int(rng.integers(max(120, int(n * 0.38)), max(160, int(n * min(0.88, 0.52 + 0.08 * s)))))
        start = int(rng.integers(0, max(1, n - width)))
        end = start + width
        for ch in range(min(x.shape[0], 3)):
            mode = rng.random()
            if mode < 0.45:
                level = float(rng.uniform(-0.08, 0.08))
                x[ch, start:end] = level + rng.normal(0.0, 0.0035 * s, size=width).astype(np.float32)
            elif mode < 0.78:
                x[ch, start:end] *= float(rng.uniform(0.015, 0.16))
            else:
                kernel = int(rng.choice([31, 51, 71]))
                filt = np.ones(kernel, dtype=np.float32) / float(kernel)
                padded = np.pad(x[ch, start:end], (kernel // 2, kernel // 2), mode="edge")
                x[ch, start:end] = np.convolve(padded, filt, mode="valid")[:width]
        if x.shape[0] > 2:
            local_t = np.linspace(-1.0, 1.0, width, dtype=np.float32)
            x[2, start:end] += float(rng.uniform(1.1, 2.8) * s) * (
                0.55 * local_t + 0.30 * np.sin(np.pi * local_t + float(rng.uniform(-1.0, 1.0))) + 0.15
            )

    # Reinsert narrow inconsistent spikes around original peaks.  These keep
    # qrs_visibility ambiguous while detector agreement remains poor.
    for idx in peak_idx:
        center = int(np.clip(idx + int(rng.integers(-12, 13)), 2, n - 3))
        width = int(rng.integers(3, 9))
        start = max(0, center - width)
        end = min(n, center + width + 1)
        local = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
        spike = np.exp(-18.0 * local * local).astype(np.float32)
        amp = float(rng.uniform(0.35, 1.25) * s) * (1.0 if rng.random() < 0.5 else -1.0)
        ch = int(rng.integers(0, min(2, x.shape[0])))
        x[ch, start:end] += amp * spike
        if x.shape[0] > 1 and rng.random() < 0.55:
            x[1, start:end] += float(rng.uniform(0.10, 0.45) * s) * spike * (1.0 if rng.random() < 0.5 else -1.0)

    if rng.random() < 0.70:
        x += rng.normal(0.0, float(rng.uniform(0.006, 0.025) * s), size=x.shape).astype(np.float32)
    if rng.random() < 0.45:
        clip = float(rng.uniform(1.2, 3.2) * max(1.0, 0.65 * s))
        x = np.clip(x, -clip, clip)
    return x


def farthest_prototype_means(rows: np.ndarray, count: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    count = int(count)
    if len(rows) == 0:
        return np.zeros((count, rows.shape[1] if rows.ndim == 2 else len(CORE_IDX)), dtype=np.float32)
    finite = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    if len(finite) <= count:
        reps = np.zeros((count, finite.shape[1]), dtype=np.float32)
        reps[: len(finite)] = finite
        if len(finite) < count:
            reps[len(finite) :] = finite.mean(axis=0, keepdims=True)
        return reps
    selected = [int(np.argmin(np.sum((finite - np.median(finite, axis=0, keepdims=True)) ** 2, axis=1)))]
    min_dist = np.sum((finite - finite[selected[0]][None, :]) ** 2, axis=1)
    for _ in range(1, count):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        dist = np.sum((finite - finite[idx][None, :]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    seeds = finite[selected]
    dist = np.sum((finite[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
    assign = np.argmin(dist, axis=1)
    out = np.zeros_like(seeds)
    for i in range(count):
        chunk = finite[assign == i]
        out[i] = chunk.mean(axis=0) if len(chunk) else seeds[i]
    return out.astype(np.float32)


def build_teacher_prototypes(aux: np.ndarray, y: np.ndarray, prototypes_per_class: int, mode: str = "pc1_quantile") -> np.ndarray:
    core = np.asarray(aux[:, CORE_IDX], dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    out = np.zeros((3, int(prototypes_per_class), core.shape[1]), dtype=np.float32)
    pc1_col = 0
    if "pc1" in CORE_COLUMNS:
        pc1_col = CORE_COLUMNS.index("pc1")
    for cls in range(3):
        rows = core[labels == cls]
        if rows.size == 0:
            continue
        if str(mode) == "farthest":
            out[cls] = farthest_prototype_means(rows, int(prototypes_per_class))
            continue
        order = np.argsort(rows[:, pc1_col])
        chunks = np.array_split(rows[order], int(prototypes_per_class))
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0:
                out[cls, i] = rows.mean(axis=0)
            else:
                out[cls, i] = chunk.mean(axis=0)
    return out


@torch.no_grad()
def collect_waveform_atlas_matrix(dataset: Any, batch_size: int = 256) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(dtype=torch.float32)
        chunks.append(waveform_atlas_stats(x).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def build_waveform_atlas(dataset: Any, prototypes_per_class: int, mode: str = "farthest") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = collect_waveform_atlas_matrix(dataset)
    mean = stats.mean(axis=0).astype(np.float32)
    std = stats.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std).astype(np.float32)
    z = ((stats - mean[None, :]) / std[None, :]).astype(np.float32)
    labels = np.asarray(dataset.y, dtype=np.int64)
    out = np.zeros((3, int(prototypes_per_class), z.shape[1]), dtype=np.float32)
    for cls in range(3):
        rows = z[labels == cls]
        if rows.size == 0:
            continue
        if str(mode) == "pc1_quantile" and "pc1" in FEATURE_COLUMNS:
            pc1 = np.asarray(dataset.aux[labels == cls, FEATURE_COLUMNS.index("pc1")], dtype=np.float32)
            order = np.argsort(pc1)
            chunks = np.array_split(rows[order], int(prototypes_per_class))
            for i, chunk in enumerate(chunks):
                out[cls, i] = chunk.mean(axis=0) if len(chunk) else rows.mean(axis=0)
        else:
            out[cls] = farthest_prototype_means(rows, int(prototypes_per_class))
    return out, mean, std


def teacher_prototype_targets(aux: torch.Tensor, y: torch.Tensor, teacher_prototypes: torch.Tensor) -> torch.Tensor:
    if not CORE_IDX:
        return torch.zeros_like(y)
    core = aux[:, CORE_IDX]
    selected = teacher_prototypes[y]
    dist = torch.mean((core[:, None, :] - selected) ** 2, dim=2)
    nearest = dist.argmin(dim=1)
    return y * int(teacher_prototypes.shape[1]) + nearest


def waveform_prototype_targets(x: torch.Tensor, y: torch.Tensor, model: GeometryStudent) -> torch.Tensor:
    stat = (waveform_atlas_stats(x) - model.waveform_atlas_mean[None, :]) / model.waveform_atlas_std[None, :]
    selected = model.waveform_atlas_prototypes[y]
    dist = torch.mean((stat[:, None, :] - selected) ** 2, dim=2)
    nearest = dist.argmin(dim=1)
    return y * int(model.waveform_atlas_prototypes.shape[1]) + nearest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "search", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--top-k-search", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--but-report-mode", choices=["passed", "always", "none"], default="passed")
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}; expected {sorted(CANDIDATES)}")
    if args.stage == "smoke":
        epochs = args.epochs or 2
        run_set(names, epochs, args, run_label="smoke", but_report_mode="none")
        return
    if args.stage == "search":
        epochs = args.epochs or 12
        run_set(names, epochs, args, run_label="search", but_report_mode=args.but_report_mode)
        return
    smoke_payload = run_set(names, args.epochs or 2, args, run_label="smoke", but_report_mode="none")
    ranked = rank_candidates(smoke_payload["metrics"])
    selected = [x for x in ranked[: int(args.top_k_search)] if x in CANDIDATES]
    if not selected:
        selected = names[: int(args.top_k_search)]
    print(f"selected for long search: {selected}", flush=True)
    search_args = argparse.Namespace(**vars(args))
    run_set(selected, max(10, args.epochs or 12), search_args, run_label="search", but_report_mode=args.but_report_mode)


def run_set(names: list[str], epochs: int, args: argparse.Namespace, run_label: str, but_report_mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries = []
    for name in names:
        summary = train_candidate(name, CANDIDATES[name], epochs, args, run_label, but_report_mode)
        summaries.append(summary)
        rows.extend(summary["metric_rows"])
    metrics = pd.DataFrame(rows)
    prefix = f"waveform_geometry_student_{run_label}"
    metrics_path = ANALYSIS_DIR / f"{prefix}_metrics.csv"
    summary_path = ANALYSIS_DIR / f"{prefix}_summary.json"
    report_path = ANALYSIS_DIR / f"{prefix}_report.md"
    report_copy = REPORT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "run_label": run_label,
        "variant_id": ARCH.VARIANT_ID,
        "dirty_worktree_warning": git_status_short(),
        "input_contract": "waveform only; SQI/geometry columns are training teacher targets, never classifier inputs",
        "training_input": "PTB-generated synthetic waveform only",
        "original_used_for_training": False,
        "but_usage": "report-only held-out buckets",
        "teacher_columns": [FEATURE_COLUMNS[i] for i in TEACHER_IDX],
        "core_teacher_columns": [FEATURE_COLUMNS[i] for i in CORE_IDX],
        "hard_teacher_columns": [FEATURE_COLUMNS[i] for i in HARD_IDX],
        "top14_teacher_columns": [FEATURE_COLUMNS[i] for i in TOP14_IDX],
        "reference_rows": REFERENCE_ROWS,
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report, flush=True)
    return {"metrics": metrics, "payload": payload}


def train_candidate(
    name: str,
    cfg: dict[str, Any],
    epochs: int,
    args: argparse.Namespace,
    run_label: str,
    but_report_mode: str,
) -> dict[str, Any]:
    print(f"training {run_label}:{name}: {cfg}", flush=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_train_ds = ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    if (
        float(cfg.get("augment_strength", 0.0)) > 0
        or float(cfg.get("bad_outlier_aug_strength", 0.0)) > 0
        or float(cfg.get("bad_stress_shell_strength", 0.0)) > 0
        or float(cfg.get("nonbad_hardneg_strength", 0.0)) > 0
        or float(cfg.get("bad_highamp_stress_strength", 0.0)) > 0
        or float(cfg.get("bad_intermittent_contact_strength", 0.0)) > 0
        or float(cfg.get("bad_record111_motion_strength", 0.0)) > 0
        or float(cfg.get("bad_impulse_reset_strength", 0.0)) > 0
        or float(cfg.get("bad_record111_burst_dropout_strength", 0.0)) > 0
        or float(cfg.get("subject111_shift_strength", 0.0)) > 0
        or float(cfg.get("nonbad_to_bad_stress_prob", 0.0)) > 0
    ):
        train_ds = AugmentedSyntheticDataset(
            base_train_ds,
            float(cfg.get("augment_strength", 0.0)),
            int(cfg["seed"]),
            float(cfg.get("bad_outlier_aug_strength", 0.0)),
            float(cfg.get("bad_stress_shell_strength", 0.0)),
            float(cfg.get("nonbad_hardneg_strength", 0.0)),
            float(cfg.get("bad_highamp_stress_strength", 0.0)),
            float(cfg.get("bad_highamp_prob", 1.0)),
            float(cfg.get("bad_intermittent_contact_strength", 0.0)),
            float(cfg.get("bad_intermittent_contact_prob", 1.0)),
            float(cfg.get("subject111_shift_strength", 0.0)),
            float(cfg.get("subject111_good_shift_scale", 0.72)),
            float(cfg.get("subject111_medium_shift_scale", 1.15)),
            float(cfg.get("bad_record111_motion_strength", 0.0)),
            float(cfg.get("bad_record111_motion_prob", 1.0)),
            float(cfg.get("bad_impulse_reset_strength", 0.0)),
            float(cfg.get("bad_impulse_reset_prob", 1.0)),
            float(cfg.get("bad_record111_burst_dropout_strength", 0.0)),
            float(cfg.get("bad_record111_burst_dropout_prob", 1.0)),
            float(cfg.get("aug_aux_weight_bad", 1.0)),
            float(cfg.get("aug_aux_weight_nonbad", 1.0)),
            float(cfg.get("nonbad_to_bad_stress_prob", 0.0)),
            float(cfg.get("nonbad_to_bad_stress_strength", 0.0)),
        )
    else:
        train_ds = base_train_ds
    val_ds = ARCH.SyntheticWaveDataset("val", str(cfg["channels"]))
    test_ds = ARCH.SyntheticWaveDataset("test", str(cfg["channels"]))
    stress_val_ds = None
    medium_stress_val_ds = None
    if float(cfg.get("selection_bad_stress_weight", 0.0)) > 0:
        stress_val_ds = AugmentedSyntheticDataset(
            val_ds,
            0.0,
            int(cfg["seed"]) + 909,
            float(cfg.get("selection_stress_bad_outlier_aug_strength", cfg.get("bad_outlier_aug_strength", 0.0))),
            float(cfg.get("selection_stress_bad_stress_shell_strength", cfg.get("bad_stress_shell_strength", 0.0))),
            float(cfg.get("selection_stress_nonbad_hardneg_strength", cfg.get("nonbad_hardneg_strength", 0.0))),
            float(cfg.get("selection_stress_bad_highamp_strength", cfg.get("bad_highamp_stress_strength", 0.0))),
            float(cfg.get("selection_stress_bad_highamp_prob", cfg.get("bad_highamp_prob", 1.0))),
            float(cfg.get("selection_stress_bad_intermittent_contact_strength", cfg.get("bad_intermittent_contact_strength", 0.0))),
            float(cfg.get("selection_stress_bad_intermittent_contact_prob", cfg.get("bad_intermittent_contact_prob", 1.0))),
            float(cfg.get("selection_subject111_shift_strength", cfg.get("subject111_shift_strength", 0.0))),
            float(cfg.get("selection_subject111_good_shift_scale", cfg.get("subject111_good_shift_scale", 0.72))),
            float(cfg.get("selection_subject111_medium_shift_scale", cfg.get("subject111_medium_shift_scale", 1.15))),
            float(cfg.get("selection_stress_bad_record111_motion_strength", cfg.get("bad_record111_motion_strength", 0.0))),
            float(cfg.get("selection_stress_bad_record111_motion_prob", cfg.get("bad_record111_motion_prob", 1.0))),
            float(cfg.get("selection_stress_bad_impulse_reset_strength", cfg.get("bad_impulse_reset_strength", 0.0))),
            float(cfg.get("selection_stress_bad_impulse_reset_prob", cfg.get("bad_impulse_reset_prob", 1.0))),
            float(cfg.get("selection_stress_bad_record111_burst_dropout_strength", cfg.get("bad_record111_burst_dropout_strength", 0.0))),
            float(cfg.get("selection_stress_bad_record111_burst_dropout_prob", cfg.get("bad_record111_burst_dropout_prob", 1.0))),
            float(cfg.get("selection_stress_aug_aux_weight_bad", cfg.get("aug_aux_weight_bad", 1.0))),
            float(cfg.get("selection_stress_aug_aux_weight_nonbad", cfg.get("aug_aux_weight_nonbad", 1.0))),
            float(cfg.get("selection_stress_nonbad_to_bad_stress_prob", cfg.get("nonbad_to_bad_stress_prob", 0.0))),
            float(cfg.get("selection_stress_nonbad_to_bad_stress_strength", cfg.get("nonbad_to_bad_stress_strength", 0.0))),
        )
    if float(cfg.get("selection_medium_stress_weight", 0.0)) > 0:
        medium_stress_val_ds = AugmentedSyntheticDataset(
            val_ds,
            0.0,
            int(cfg["seed"]) + 1209,
            0.0,
            0.0,
            float(cfg.get("selection_medium_stress_nonbad_hardneg_strength", cfg.get("nonbad_hardneg_strength", 0.0))),
            0.0,
            1.0,
            0.0,
            1.0,
            float(cfg.get("selection_medium_subject111_shift_strength", cfg.get("selection_subject111_shift_strength", cfg.get("subject111_shift_strength", 0.0)))),
            float(cfg.get("selection_medium_subject111_good_shift_scale", 0.0)),
            float(cfg.get("selection_medium_subject111_medium_shift_scale", cfg.get("subject111_medium_shift_scale", 1.15))),
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            float(cfg.get("selection_medium_aug_aux_weight_nonbad", cfg.get("aug_aux_weight_nonbad", 1.0))),
            0.0,
            0.0,
        )
    norm = ARCH.FeatureNorm(mean=base_train_ds.mean, std=base_train_ds.std)
    original_ds = None
    original_loader = None
    if but_report_mode != "none":
        original_ds = ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    stress_val_loader = DataLoader(stress_val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin) if stress_val_ds is not None else None
    medium_stress_val_loader = DataLoader(medium_stress_val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin) if medium_stress_val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    if original_ds is not None:
        original_loader = DataLoader(original_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = GeometryStudent(cfg, int(train_ds.x.shape[1])).to(device)
    init_from_candidate = str(cfg.get("init_from_candidate", "")).strip()
    if init_from_candidate:
        init_path = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search" / init_from_candidate / "ckpt_best.pt"
        if not init_path.exists():
            raise FileNotFoundError(f"init_from_candidate checkpoint not found: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device)
        missing, unexpected = model.load_state_dict(init_ckpt["model_state"], strict=False)
        if missing or unexpected:
            print(
                f"loaded init_from_candidate={init_from_candidate} with missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
    if getattr(model, "use_teacher_atlas", False):
        prototypes = build_teacher_prototypes(
            base_train_ds.aux,
            base_train_ds.y,
            int(cfg["teacher_prototypes_per_class"]),
            str(cfg.get("teacher_prototype_mode", "pc1_quantile")),
        )
        model.set_teacher_prototypes(prototypes)
    if getattr(model, "use_waveform_atlas", False):
        wave_proto, wave_mean, wave_std = build_waveform_atlas(
            base_train_ds,
            int(cfg["waveform_atlas_prototypes_per_class"]),
            str(cfg.get("waveform_atlas_mode", "farthest")),
        )
        model.set_waveform_atlas(wave_proto, wave_mean, wave_std)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    class_weight = torch.tensor(cfg["class_weight"], dtype=torch.float32, device=device)
    critical_idx = [FEATURE_COLUMNS.index(c) for c in list(cfg.get("critical_aux_columns", [])) if c in FEATURE_COLUMNS]
    quantile_aux_idx = [FEATURE_COLUMNS.index(c) for c in list(cfg.get("quantile_aux_columns", [])) if c in FEATURE_COLUMNS]
    quantile_aux_cuts = None
    if quantile_aux_idx and (
        float(cfg.get("quantile_aux_weight", 0.0)) > 0
        or float(cfg.get("pretrain_quantile_aux_weight", 0.0)) > 0
    ):
        quantiles = [float(q) for q in cfg.get("quantile_aux_quantiles", [0.33, 0.50, 0.67])]
        cuts = []
        for q in quantiles:
            cuts.append(np.nanquantile(base_train_ds.aux[:, quantile_aux_idx], q, axis=0).astype(np.float32))
        quantile_aux_cuts = torch.tensor(np.stack(cuts, axis=0), dtype=torch.float32, device=device)
    gate_boundary_cut = None
    gate_purity_cut = None
    if float(cfg.get("feature_gate_supervision_weight", 0.0)) > 0 and "boundary_confidence" in FEATURE_COLUMNS and "knn_label_purity" in FEATURE_COLUMNS:
        boundary_i = FEATURE_COLUMNS.index("boundary_confidence")
        purity_i = FEATURE_COLUMNS.index("knn_label_purity")
        gate_boundary_cut = float(np.nanquantile(base_train_ds.aux[:, boundary_i], float(cfg.get("feature_gate_boundary_quantile", 0.35))))
        gate_purity_cut = float(np.nanquantile(base_train_ds.aux[:, purity_i], float(cfg.get("feature_gate_purity_quantile", 0.35))))
    aux_pretrain_epochs = int(cfg.get("aux_pretrain_epochs", 0))
    feature_teacher = None
    if float(cfg.get("feature_teacher_distill_weight", 0.0)) > 0:
        feature_teacher_cols = list(cfg.get("feature_teacher_columns", TOP20_INTERPRETABLE_FEATURE_COLUMNS))
        feature_teacher = FeatureLogitTeacher(base_train_ds.aux, base_train_ds.y, feature_teacher_cols, int(cfg["seed"]))
    primitive_teacher = None
    if float(cfg.get("primitive_teacher_distill_weight", 0.0)) > 0:
        primitive_teacher = PrimitiveLogitTeacher(
            np.asarray(base_train_ds.x, dtype=np.float32),
            np.asarray(base_train_ds.y, dtype=np.int64),
            str(cfg.get("primitive_teacher_bank", cfg.get("primitive_bank", "qrs_enhanced"))),
            int(cfg["seed"]) + 313,
        )
    best_state = None
    best_epoch = 0
    best_score = -1e9
    logs = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            if "aux_weight" in batch:
                aux_weight = batch["aux_weight"].to(device=device, dtype=torch.float32)
            else:
                aux_weight = torch.ones_like(y, dtype=torch.float32, device=device)
            if "stress_target" in batch:
                stress_target = batch["stress_target"].to(device=device, dtype=torch.float32)
            else:
                stress_target = torch.zeros_like(y, dtype=torch.float32, device=device)
            out = model(x, mask_ratio=float(cfg.get("mask_ratio", 0.0)))
            cls_loss = classification_loss(out["logits"], y, class_weight, float(cfg.get("focal_gamma", 0.0)))
            if out.get("gm_pair_logits") is not None and float(cfg.get("gm_pair_weight", 0.0)) > 0:
                gm_mask = y != CLASS_TO_INT["bad"]
                if gm_mask.any():
                    gm_pair_loss = F.cross_entropy(out["gm_pair_logits"][gm_mask], y[gm_mask])
                else:
                    gm_pair_loss = torch.zeros((), device=device)
            else:
                gm_pair_loss = torch.zeros((), device=device)
            aux_loss = weighted_smooth_l1_columns(out["aux_pred"], aux, TEACHER_IDX, aux_weight)
            core_loss = weighted_smooth_l1_columns(out["aux_pred"], aux, CORE_IDX, aux_weight)
            hard_loss = weighted_smooth_l1_columns(out["aux_pred"], aux, HARD_IDX, aux_weight)
            top14_loss = weighted_smooth_l1_columns(out["aux_pred"], aux, TOP14_IDX, aux_weight)
            critical_loss = weighted_smooth_l1_columns(out["aux_pred"], aux, critical_idx, aux_weight) if critical_idx else torch.zeros((), device=device)
            if out.get("atlas_pred") is not None:
                atlas_target = model.teacher_distance_features(aux)
                atlas_per = F.smooth_l1_loss(out["atlas_pred"], atlas_target, reduction="none").mean(dim=1)
                atlas_loss = (atlas_per * aux_weight).sum() / aux_weight.sum().clamp_min(1.0)
            else:
                atlas_loss = torch.zeros((), device=device)
            con_loss = supervised_contrastive_loss(out["embedding"], y)
            rank_loss = feature_rank_loss(out["aux_pred"], aux, aux_weight)
            neighbor_loss = teacher_neighborhood_loss(out["embedding"], aux, float(cfg.get("neighbor_tau", 0.45)), aux_weight)
            if (
                out.get("proto_logits") is not None
                and float(cfg.get("embedding_proto_weight", 0.0)) > 0
                and getattr(model, "use_teacher_atlas", False)
            ):
                proto_target = teacher_prototype_targets(aux, y, model.teacher_prototypes)
                proto_per = F.cross_entropy(out["proto_logits"], proto_target, reduction="none")
                embedding_proto_loss = (proto_per * aux_weight).sum() / aux_weight.sum().clamp_min(1.0)
            else:
                embedding_proto_loss = torch.zeros((), device=device)
            if out.get("waveform_proto_logits") is not None and float(cfg.get("waveform_proto_weight", 0.0)) > 0:
                waveform_target = waveform_prototype_targets(x, y, model)
                waveform_proto_loss = F.cross_entropy(out["waveform_proto_logits"], waveform_target)
            else:
                waveform_proto_loss = torch.zeros((), device=device)
            rec_loss = reconstruction_loss(out, x, int(cfg.get("patch", 25)))
            if out.get("bad_logit") is not None and float(cfg.get("bad_aux_weight", 0.0)) > 0:
                bad_target = (y == CLASS_TO_INT["bad"]).float()
                pos_weight = torch.tensor(float(cfg.get("bad_aux_pos_weight", 1.0)), dtype=torch.float32, device=device)
                bad_aux_loss = F.binary_cross_entropy_with_logits(out["bad_logit"], bad_target, pos_weight=pos_weight)
            else:
                bad_aux_loss = torch.zeros((), device=device)
            if out.get("stress_logit") is not None and float(cfg.get("stress_aux_weight", 0.0)) > 0:
                stress_pos_weight = torch.tensor(float(cfg.get("stress_aux_pos_weight", 1.0)), dtype=torch.float32, device=device)
                stress_aux_loss = F.binary_cross_entropy_with_logits(out["stress_logit"], stress_target, pos_weight=stress_pos_weight)
            else:
                stress_aux_loss = torch.zeros((), device=device)
            if out.get("bad_logit") is not None and float(cfg.get("bad_specificity_weight", 0.0)) > 0:
                bad_logit = out["bad_logit"]
                if out.get("gm_pair_logits") is not None:
                    nonbad_ref = out["gm_pair_logits"].max(dim=1).values
                else:
                    nonbad_ref = out["logits"][:, :2].max(dim=1).values
                margin = float(cfg.get("bad_specificity_margin", 0.35))
                nonbad_mask = y != CLASS_TO_INT["bad"]
                bad_mask = y == CLASS_TO_INT["bad"]
                parts = []
                if nonbad_mask.any():
                    parts.append(F.relu(bad_logit[nonbad_mask] - nonbad_ref[nonbad_mask] + margin).mean())
                if bad_mask.any():
                    parts.append(0.35 * F.relu(nonbad_ref[bad_mask] - bad_logit[bad_mask] + margin).mean())
                bad_specificity_loss = torch.stack(parts).sum() if parts else torch.zeros((), device=device)
            else:
                bad_specificity_loss = torch.zeros((), device=device)
            if feature_teacher is not None:
                teacher_probs_np = feature_teacher.predict_proba(aux.detach().cpu().numpy())
                teacher_probs = torch.tensor(teacher_probs_np, dtype=torch.float32, device=device)
                temp = float(cfg.get("feature_teacher_temperature", 1.6))
                feature_teacher_loss = F.kl_div(
                    F.log_softmax(out["logits"] / temp, dim=1),
                    teacher_probs,
                    reduction="batchmean",
                ) * (temp * temp)
            else:
                feature_teacher_loss = torch.zeros((), device=device)
            if (
                out.get("feature_gate") is not None
                and float(cfg.get("feature_gate_supervision_weight", 0.0)) > 0
                and gate_boundary_cut is not None
                and gate_purity_cut is not None
            ):
                boundary_i = FEATURE_COLUMNS.index("boundary_confidence")
                purity_i = FEATURE_COLUMNS.index("knn_label_purity")
                gate_target = ((aux[:, boundary_i] <= gate_boundary_cut) | (aux[:, purity_i] <= gate_purity_cut)).to(dtype=torch.float32)
                feature_gate_loss = F.binary_cross_entropy(out["feature_gate"].squeeze(1), gate_target)
            else:
                feature_gate_loss = torch.zeros((), device=device)
            if quantile_aux_cuts is not None and quantile_aux_idx:
                pred_sel = out["aux_pred"][:, quantile_aux_idx]
                target_sel = aux[:, quantile_aux_idx]
                cuts = quantile_aux_cuts
                sharpness = float(cfg.get("quantile_aux_sharpness", 2.25))
                q_logits = (pred_sel[:, None, :] - cuts[None, :, :]) * sharpness
                q_target = (target_sel[:, None, :] > cuts[None, :, :]).to(dtype=torch.float32)
                q_per = F.binary_cross_entropy_with_logits(q_logits, q_target, reduction="none").mean(dim=(1, 2))
                quantile_aux_loss = (q_per * aux_weight).sum() / aux_weight.sum().clamp_min(1.0)
            else:
                quantile_aux_loss = torch.zeros((), device=device)
            if primitive_teacher is not None:
                teacher_probs_np = primitive_teacher.predict_proba_from_waveform(x)
                teacher_probs = torch.tensor(teacher_probs_np, dtype=torch.float32, device=device)
                temp = float(cfg.get("primitive_teacher_temperature", 1.8))
                primitive_teacher_loss = F.kl_div(
                    F.log_softmax(out["logits"] / temp, dim=1),
                    teacher_probs,
                    reduction="batchmean",
                ) * (temp * temp)
            else:
                primitive_teacher_loss = torch.zeros((), device=device)
            in_aux_pretrain = epoch <= aux_pretrain_epochs
            cls_w = float(cfg.get("classification_weight", 1.0))
            gm_w = float(cfg.get("gm_pair_weight", 0.0))
            aux_w = float(cfg["aux_weight"])
            core_w = float(cfg["core_aux_weight"])
            hard_w = float(cfg.get("hard_aux_weight", 0.0))
            top14_w = float(cfg.get("top14_aux_weight", 0.0))
            critical_w = float(cfg.get("critical_aux_weight", 0.0))
            atlas_w = float(cfg.get("atlas_distill_weight", 0.0))
            bad_aux_w = float(cfg.get("bad_aux_weight", 0.0))
            stress_aux_w = float(cfg.get("stress_aux_weight", 0.0))
            bad_spec_w = float(cfg.get("bad_specificity_weight", 0.0))
            feature_teacher_w = float(cfg.get("feature_teacher_distill_weight", 0.0))
            feature_gate_w = float(cfg.get("feature_gate_supervision_weight", 0.0))
            quantile_aux_w = float(cfg.get("quantile_aux_weight", 0.0))
            primitive_teacher_w = float(cfg.get("primitive_teacher_distill_weight", 0.0))
            neighbor_w = float(cfg.get("neighbor_weight", 0.0))
            supcon_w = float(cfg["supcon_weight"])
            rank_w = float(cfg["rank_weight"])
            if in_aux_pretrain:
                cls_w = float(cfg.get("pretrain_classification_weight", min(cls_w, 0.20)))
                gm_w = float(cfg.get("pretrain_gm_pair_weight", min(gm_w, 0.10)))
                aux_w = float(cfg.get("pretrain_aux_weight", max(aux_w, 1.0)))
                core_w = float(cfg.get("pretrain_core_aux_weight", max(core_w, 1.6)))
                hard_w = float(cfg.get("pretrain_hard_aux_weight", max(hard_w, 2.5)))
                top14_w = float(cfg.get("pretrain_top14_aux_weight", max(top14_w, 3.0)))
                critical_w = float(cfg.get("pretrain_critical_aux_weight", max(critical_w, 3.0 if critical_idx else 0.0)))
                atlas_w = float(cfg.get("pretrain_atlas_distill_weight", atlas_w))
                bad_aux_w = float(cfg.get("pretrain_bad_aux_weight", min(bad_aux_w, 0.25)))
                stress_aux_w = float(cfg.get("pretrain_stress_aux_weight", min(stress_aux_w, 0.25)))
                bad_spec_w = float(cfg.get("pretrain_bad_specificity_weight", min(bad_spec_w, 0.25)))
                feature_teacher_w = float(cfg.get("pretrain_feature_teacher_distill_weight", feature_teacher_w))
                feature_gate_w = float(cfg.get("pretrain_feature_gate_supervision_weight", feature_gate_w))
                quantile_aux_w = float(cfg.get("pretrain_quantile_aux_weight", quantile_aux_w))
                primitive_teacher_w = float(cfg.get("pretrain_primitive_teacher_distill_weight", primitive_teacher_w))
                neighbor_w = float(cfg.get("pretrain_neighbor_weight", neighbor_w))
                supcon_w = float(cfg.get("pretrain_supcon_weight", min(supcon_w, 0.08)))
                rank_w = float(cfg.get("pretrain_rank_weight", max(rank_w, 0.70)))
            loss = (
                cls_w * cls_loss
                + gm_w * gm_pair_loss
                + aux_w * aux_loss
                + core_w * core_loss
                + hard_w * hard_loss
                + top14_w * top14_loss
                + critical_w * critical_loss
                + atlas_w * atlas_loss
                + bad_aux_w * bad_aux_loss
                + stress_aux_w * stress_aux_loss
                + bad_spec_w * bad_specificity_loss
                + feature_teacher_w * feature_teacher_loss
                + feature_gate_w * feature_gate_loss
                + quantile_aux_w * quantile_aux_loss
                + primitive_teacher_w * primitive_teacher_loss
                + neighbor_w * neighbor_loss
                + float(cfg.get("embedding_proto_weight", 0.0)) * embedding_proto_loss
                + float(cfg.get("waveform_proto_weight", 0.0)) * waveform_proto_loss
                + supcon_w * con_loss
                + rank_w * rank_loss
                + float(cfg["recon_weight"]) * rec_loss
            )
            if not torch.isfinite(loss):
                losses.append(
                    {
                        "loss": float("nan"),
                        "cls": float(cls_loss.detach().cpu()) if torch.isfinite(cls_loss) else float("nan"),
                        "gm_pair": float(gm_pair_loss.detach().cpu()) if torch.isfinite(gm_pair_loss) else float("nan"),
                        "aux": float(aux_loss.detach().cpu()) if torch.isfinite(aux_loss) else float("nan"),
                        "core": float(core_loss.detach().cpu()) if torch.isfinite(core_loss) else float("nan"),
                        "hard": float(hard_loss.detach().cpu()) if torch.isfinite(hard_loss) else float("nan"),
                        "top14": float(top14_loss.detach().cpu()) if torch.isfinite(top14_loss) else float("nan"),
                        "critical": float(critical_loss.detach().cpu()) if torch.isfinite(critical_loss) else float("nan"),
                        "atlas": float(atlas_loss.detach().cpu()) if torch.isfinite(atlas_loss) else float("nan"),
                        "bad_aux": float(bad_aux_loss.detach().cpu()) if torch.isfinite(bad_aux_loss) else float("nan"),
                        "stress_aux": float(stress_aux_loss.detach().cpu()) if torch.isfinite(stress_aux_loss) else float("nan"),
                        "bad_specificity": float(bad_specificity_loss.detach().cpu()) if torch.isfinite(bad_specificity_loss) else float("nan"),
                        "feature_teacher": float(feature_teacher_loss.detach().cpu()) if torch.isfinite(feature_teacher_loss) else float("nan"),
                        "feature_gate": float(feature_gate_loss.detach().cpu()) if torch.isfinite(feature_gate_loss) else float("nan"),
                        "primitive_teacher": float(primitive_teacher_loss.detach().cpu()) if torch.isfinite(primitive_teacher_loss) else float("nan"),
                        "neighbor": float(neighbor_loss.detach().cpu()) if torch.isfinite(neighbor_loss) else float("nan"),
                        "embedding_proto": float(embedding_proto_loss.detach().cpu()) if torch.isfinite(embedding_proto_loss) else float("nan"),
                        "waveform_proto": float(waveform_proto_loss.detach().cpu()) if torch.isfinite(waveform_proto_loss) else float("nan"),
                        "supcon": float(con_loss.detach().cpu()) if torch.isfinite(con_loss) else float("nan"),
                        "rank": float(rank_loss.detach().cpu()) if torch.isfinite(rank_loss) else float("nan"),
                        "recon": float(rec_loss.detach().cpu()) if torch.isfinite(rec_loss) else float("nan"),
                    }
                )
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "cls": float(cls_loss.detach().cpu()),
                    "gm_pair": float(gm_pair_loss.detach().cpu()),
                    "aux": float(aux_loss.detach().cpu()),
                    "core": float(core_loss.detach().cpu()),
                    "hard": float(hard_loss.detach().cpu()),
                    "top14": float(top14_loss.detach().cpu()),
                    "critical": float(critical_loss.detach().cpu()),
                    "atlas": float(atlas_loss.detach().cpu()),
                        "bad_aux": float(bad_aux_loss.detach().cpu()),
                        "stress_aux": float(stress_aux_loss.detach().cpu()),
                    "bad_specificity": float(bad_specificity_loss.detach().cpu()),
                    "feature_teacher": float(feature_teacher_loss.detach().cpu()),
                    "feature_gate": float(feature_gate_loss.detach().cpu()),
                    "primitive_teacher": float(primitive_teacher_loss.detach().cpu()),
                    "neighbor": float(neighbor_loss.detach().cpu()),
                    "embedding_proto": float(embedding_proto_loss.detach().cpu()),
                    "waveform_proto": float(waveform_proto_loss.detach().cpu()),
                    "supcon": float(con_loss.detach().cpu()),
                    "rank": float(rank_loss.detach().cpu()),
                    "recon": float(rec_loss.detach().cpu()),
                }
            )
        val_payload = eval_loader(model, val_loader, device)
        val_rep = val_payload.report
        score = (
            val_rep["acc"]
            + 0.25 * val_rep["macro_f1"]
            + 0.25 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
            - 0.02 * val_rep["teacher_core_mae"]
        )
        if float(cfg.get("selection_precision_weight", 0.0)) > 0:
            precision_score = (
                0.35 * val_rep["bad_precision"]
                + 0.30 * val_rep["medium_precision"]
                + 0.20 * val_rep["good_recall"]
                + 0.15 * val_rep["acc"]
                - 0.02 * val_rep["teacher_core_mae"]
            )
            score = score + float(cfg.get("selection_precision_weight", 0.0)) * precision_score
        stress_rep = None
        if stress_val_loader is not None:
            stress_payload = eval_loader(model, stress_val_loader, device)
            stress_rep = stress_payload.report
            stress_score = (
                stress_rep["acc"]
                + 0.25 * stress_rep["macro_f1"]
                + 0.45 * stress_rep["bad_recall"]
                + 0.15 * min(stress_rep["good_recall"], stress_rep["medium_recall"])
                - 0.02 * stress_rep["teacher_core_mae"]
            )
            score = score + float(cfg.get("selection_bad_stress_weight", 0.0)) * stress_score
            if float(cfg.get("selection_stress_precision_weight", 0.0)) > 0:
                stress_precision_score = (
                    0.35 * stress_rep["bad_precision"]
                    + 0.30 * stress_rep["medium_precision"]
                    + 0.20 * stress_rep["good_recall"]
                    + 0.15 * stress_rep["acc"]
                    - 0.02 * stress_rep["teacher_core_mae"]
                )
                score = score + float(cfg.get("selection_stress_precision_weight", 0.0)) * stress_precision_score
        medium_stress_rep = None
        if medium_stress_val_loader is not None:
            medium_stress_payload = eval_loader(model, medium_stress_val_loader, device)
            medium_stress_rep = medium_stress_payload.report
            medium_stress_score = (
                0.20 * medium_stress_rep["acc"]
                + 0.15 * medium_stress_rep["macro_f1"]
                + 0.20 * medium_stress_rep["good_recall"]
                + 0.45 * medium_stress_rep["medium_recall"]
                + 0.10 * min(medium_stress_rep["good_recall"], medium_stress_rep["bad_recall"])
                - 0.02 * medium_stress_rep["teacher_core_mae"]
            )
            score = score + float(cfg.get("selection_medium_stress_weight", 0.0)) * medium_stress_score
        loss_mean = pd.DataFrame(losses).mean(numeric_only=True).to_dict()
        logs.append({"epoch": epoch, **{f"train_{k}": v for k, v in loss_mean.items()}, **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        if stress_rep is not None:
            logs[-1].update({f"stress_val_{k}": v for k, v in stress_rep.items() if k != "confusion_3x3"})
            stress_msg = f" stress_acc={stress_rep['acc']:.4f} stress_bad={stress_rep['bad_recall']:.3f}"
        else:
            stress_msg = ""
        if medium_stress_rep is not None:
            logs[-1].update({f"medium_stress_val_{k}": v for k, v in medium_stress_rep.items() if k != "confusion_3x3"})
            stress_msg += f" medstress_acc={medium_stress_rep['acc']:.4f} medstress_mid={medium_stress_rep['medium_recall']:.3f}"
        if bool(cfg.get("report_original_each_epoch", False)) and original_loader is not None and original_ds is not None:
            # Report-only shadow curve.  These metrics are appended after the
            # synthetic selection score is computed and never affect best_state.
            epoch_original_payload = eval_loader(model, original_loader, device)
            epoch_original_test = ARCH.bucket_report(original_ds, epoch_original_payload.probs, "original_test_all_10s+")
            epoch_bad_core = ARCH.bucket_report(original_ds, epoch_original_payload.probs, "bad_core_nearboundary")
            epoch_bad_stress = ARCH.bucket_report(original_ds, epoch_original_payload.probs, "bad_outlier_stress")
            logs[-1].update(
                {
                    "report_original_test_acc": epoch_original_test["acc"],
                    "report_original_test_good_recall": epoch_original_test["good_recall"],
                    "report_original_test_medium_recall": epoch_original_test["medium_recall"],
                    "report_original_test_bad_recall": epoch_original_test["bad_recall"],
                    "report_original_bad_core_recall": epoch_bad_core["bad_recall"],
                    "report_original_bad_stress_recall": epoch_bad_stress["bad_recall"],
                }
            )
            stress_msg += (
                f" report_orig_acc={epoch_original_test['acc']:.4f}"
                f" report_orig={epoch_original_test['good_recall']:.3f}/{epoch_original_test['medium_recall']:.3f}/{epoch_original_test['bad_recall']:.3f}"
            )
        print(
            f"{run_label}:{name} epoch={epoch}/{epochs} loss={loss_mean['loss']:.4f} "
            f"val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} "
            f"teacher_core_mae={val_rep['teacher_core_mae']:.3f}{stress_msg}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_payload = eval_loader(model, val_loader, device)
    test_payload = eval_loader(model, test_loader, device)
    threshold = ARCH.calibrate_bad_threshold(val_ds.y, val_payload.probs)
    synthetic_badcal = GEOM.metric_report(test_ds.y, ARCH.apply_bad_threshold(test_payload.probs, threshold), test_payload.probs)
    metric_rows = [
        metric_row(name, run_label, "synthetic_val", val_payload.report),
        metric_row(name, run_label, "synthetic_test", test_payload.report),
        metric_row(f"{name}_badcal", run_label, "synthetic_test", synthetic_badcal),
    ]
    useful = (
        test_payload.report["acc"] >= 0.94
        and test_payload.report["good_recall"] >= 0.92
        and test_payload.report["medium_recall"] >= 0.90
        and test_payload.report["bad_recall"] >= 0.90
    )
    original_payload = None
    if but_report_mode == "always" or (but_report_mode == "passed" and useful):
        if original_ds is None or original_loader is None:
            raise FileNotFoundError("Original BUT report requested, but OriginalWaveDataset was not initialized.")
        original_payload = eval_loader(model, original_loader, device)
        for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
            metric_rows.append(metric_row(name, run_label, bucket, ARCH.bucket_report(original_ds, original_payload.probs, bucket)))
            metric_rows.append(metric_row(f"{name}_badcal", run_label, bucket, ARCH.bucket_report(original_ds, original_payload.probs, bucket, threshold)))
    run_dir = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / run_label / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "waveform_geometry_student",
            "input_contract": "waveform only; geometry/SQI teacher targets used only in training loss",
            "teacher_columns": [FEATURE_COLUMNS[i] for i in TEACHER_IDX],
            "core_teacher_columns": [FEATURE_COLUMNS[i] for i in CORE_IDX],
            "hard_teacher_columns": [FEATURE_COLUMNS[i] for i in HARD_IDX],
            "top14_teacher_columns": [FEATURE_COLUMNS[i] for i in TOP14_IDX],
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": base_train_ds.mean,
            "feature_train_std": base_train_ds.std,
            "teacher_prototypes": model.teacher_prototypes.detach().cpu().numpy() if getattr(model, "use_teacher_atlas", False) else None,
            "waveform_atlas_prototypes": model.waveform_atlas_prototypes.detach().cpu().numpy() if getattr(model, "use_waveform_atlas", False) else None,
            "waveform_atlas_mean": model.waveform_atlas_mean.detach().cpu().numpy() if getattr(model, "use_waveform_atlas", False) else None,
            "waveform_atlas_std": model.waveform_atlas_std.detach().cpu().numpy() if getattr(model, "use_waveform_atlas", False) else None,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
            "original_rows_used_for_training": False,
            "but_usage": "report-only",
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    write_diagnostics(name, run_label, run_dir, test_ds, test_payload, original_ds if original_payload else None, original_payload)
    return {
        "candidate": name,
        "run_label": run_label,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "useful_gate_passed": useful,
        "bad_threshold_trainval": threshold,
        "synthetic_test": test_payload.report,
        "original_report_ran": original_payload is not None,
        "metric_rows": metric_rows,
    }


def classification_loss(logits: torch.Tensor, y: torch.Tensor, class_weight: torch.Tensor, focal_gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, weight=class_weight, reduction="none")
    if focal_gamma <= 0:
        return ce.mean()
    pt = torch.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1).clamp(1e-5, 1.0)
    return (((1.0 - pt) ** float(focal_gamma)) * ce).mean()


def weighted_smooth_l1_columns(pred: torch.Tensor, target: torch.Tensor, cols: list[int], sample_weight: torch.Tensor) -> torch.Tensor:
    if not cols:
        return torch.zeros((), device=pred.device)
    loss = F.smooth_l1_loss(pred[:, cols], target[:, cols], reduction="none").mean(dim=1)
    weight = sample_weight.to(device=pred.device, dtype=pred.dtype).clamp_min(0.0)
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def supervised_contrastive_loss(emb: torch.Tensor, y: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    if emb.shape[0] < 4:
        return torch.zeros((), device=emb.device)
    z = F.normalize(emb, dim=1)
    logits = (z @ z.T) / temperature
    logits = logits - torch.eye(z.shape[0], device=z.device) * 1e9
    same = (y[:, None] == y[None, :]) & (~torch.eye(z.shape[0], dtype=torch.bool, device=z.device))
    if same.sum() == 0:
        return torch.zeros((), device=emb.device)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per = -(log_prob * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
    return per[same.any(dim=1)].mean()


def feature_rank_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    if pred.shape[0] < 8 or not CORE_IDX:
        return torch.zeros((), device=pred.device)
    idx = torch.arange(pred.shape[0], device=pred.device)
    shuffled = idx[torch.randperm(pred.shape[0], device=pred.device)]
    if sample_weight is None:
        pair_weight = torch.ones(pred.shape[0], dtype=pred.dtype, device=pred.device)
    else:
        sw = sample_weight.to(device=pred.device, dtype=pred.dtype).clamp_min(0.0)
        pair_weight = torch.minimum(sw, sw[shuffled])
    losses = []
    for col in CORE_IDX:
        td = target[:, col] - target[shuffled, col]
        valid = (td.abs() > 0.35) & (pair_weight > 0.25)
        if valid.any():
            sign = torch.sign(td[valid])
            pdiff = pred[:, col][valid] - pred[shuffled, col][valid]
            local = F.relu(0.05 - sign * pdiff)
            w = pair_weight[valid]
            losses.append((local * w).sum() / w.sum().clamp_min(1.0))
    if not losses:
        return torch.zeros((), device=pred.device)
    return torch.stack(losses).mean()


def teacher_neighborhood_loss(embedding: torch.Tensor, aux: torch.Tensor, tau: float, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    if embedding.shape[0] < 8 or not CORE_IDX:
        return torch.zeros((), device=embedding.device)
    if sample_weight is not None:
        keep = sample_weight.to(device=embedding.device).float() > 0.50
        if keep.sum() < 8:
            return torch.zeros((), device=embedding.device)
        embedding = embedding[keep]
        aux = aux[keep]
    teacher = F.normalize(aux[:, CORE_IDX], dim=1)
    student = F.normalize(embedding, dim=1)
    teacher_dist = torch.cdist(teacher, teacher, p=2)
    student_dist = torch.cdist(student, student, p=2)
    eye = torch.eye(teacher_dist.shape[0], dtype=torch.bool, device=teacher_dist.device)
    scale = max(float(tau), 1e-3)
    teacher_logits = (-teacher_dist / scale).masked_fill(eye, -1e4)
    student_logits = (-student_dist / scale).masked_fill(eye, -1e4)
    target = torch.softmax(teacher_logits, dim=1).detach()
    log_pred = torch.log_softmax(student_logits, dim=1)
    return F.kl_div(log_pred, target, reduction="batchmean")


def reconstruction_loss(out: dict[str, torch.Tensor | None], x: torch.Tensor, patch: int) -> torch.Tensor:
    recon = out.get("recon")
    mask = out.get("recon_mask")
    if recon is None or mask is None:
        return torch.zeros((), device=x.device)
    b, c, n = x.shape
    keep = (n // patch) * patch
    target = x[:, :, :keep].reshape(b, c * (keep // patch), patch)
    if mask.sum() == 0:
        return torch.zeros((), device=x.device)
    return F.smooth_l1_loss(recon[mask], target[mask])


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalPayload:
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    aux_pred: list[np.ndarray] = []
    aux_abs: list[np.ndarray] = []
    emb: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        aux = batch["aux"].to(device=device, dtype=torch.float32)
        out = model(x, mask_ratio=0.0)
        p = F.softmax(out["logits"], dim=1).detach().cpu().numpy()
        probs.append(p)
        pred_aux = out["aux_pred"].detach().cpu().numpy()
        aux_pred.append(pred_aux)
        aux_abs.append(torch.mean(torch.abs(out["aux_pred"] - aux), dim=0).detach().cpu().numpy())
        emb.append(out["embedding"].detach().cpu().numpy())
        ys.append(batch["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    pred = p.argmax(axis=1).astype(np.int64)
    rep = GEOM.metric_report(y, pred, p)
    aux_mae = np.stack(aux_abs).mean(axis=0)
    rep["teacher_mae"] = float(aux_mae.mean())
    rep["teacher_core_mae"] = float(aux_mae[CORE_IDX].mean()) if CORE_IDX else rep["teacher_mae"]
    return EvalPayload(rep, p, pred, np.concatenate(aux_pred), np.concatenate(emb), y)


def metric_row(candidate: str, run_label: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "run_label": run_label, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def write_diagnostics(
    name: str,
    run_label: str,
    run_dir: Path,
    synthetic_ds: Any,
    synthetic: EvalPayload,
    original_ds: Any | None,
    original: EvalPayload | None,
) -> None:
    feature_rows = feature_recovery_rows(name, run_label, "synthetic_test", synthetic, synthetic_ds.aux)
    pd.DataFrame(feature_rows).to_csv(run_dir / "feature_teacher_recovery_synthetic_test.csv", index=False)
    if original is not None and original_ds is not None:
        pd.DataFrame(feature_recovery_rows(name, run_label, "original_all_10s+", original, original_ds.aux)).to_csv(run_dir / "feature_teacher_recovery_original.csv", index=False)
    write_error_rows(run_dir / "synthetic_test_error_rows.csv", synthetic_ds, synthetic)
    plot_embedding_pca(run_dir / "synthetic_test_embedding_pca.png", synthetic.embeddings, synthetic.y, synthetic.pred, f"{name} synthetic embedding")
    plot_waveform_errors(run_dir / "synthetic_test_error_waveforms.png", synthetic_ds, synthetic, f"{name} synthetic mistakes")


def feature_recovery_rows(candidate: str, run_label: str, split: str, payload: EvalPayload, target_aux: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i, col in enumerate(FEATURE_COLUMNS):
        t = target_aux[:, i].astype(float)
        p = payload.aux_pred[:, i].astype(float)
        if np.std(t) > 1e-8 and np.std(p) > 1e-8:
            corr = float(np.corrcoef(t, p)[0, 1])
        else:
            corr = float("nan")
        rows.append(
            {
                "candidate": candidate,
                "run_label": run_label,
                "split": split,
                "feature": col,
                "mae_z": float(np.mean(np.abs(p - t))),
                "corr": corr,
                "is_teacher_core": col in CORE_COLUMNS,
                "is_teacher": col in TEACHER_COLUMNS,
                "is_teacher_hard": col in HARD_FEATURE_COLUMNS,
                "is_teacher_top14": col in TOP14_FEATURE_COLUMNS,
            }
        )
    return rows


def write_error_rows(path: Path, ds: Any, payload: EvalPayload) -> None:
    rows = []
    frame = getattr(ds, "frame", None)
    for i, (y, pred) in enumerate(zip(payload.y, payload.pred)):
        if int(y) == int(pred):
            continue
        base = {
            "row": i,
            "true": INT_TO_CLASS[int(y)],
            "pred": INT_TO_CLASS[int(pred)],
            "prob_good": float(payload.probs[i, 0]),
            "prob_medium": float(payload.probs[i, 1]),
            "prob_bad": float(payload.probs[i, 2]),
        }
        if frame is not None:
            for col in ["idx", "split", "y_class", "boundary_block", "source_variant"]:
                if col in frame.columns:
                    base[col] = frame.iloc[i][col]
        rows.append(base)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_embedding_pca(path: Path, emb: np.ndarray, y: np.ndarray, pred: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        z = emb - emb.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        xy = z @ vt[:2].T
        rng = np.random.default_rng(202608)
        take = np.arange(len(y))
        if len(take) > 3000:
            take = rng.choice(take, size=3000, replace=False)
        colors = np.array(["#2ca25f", "#3182bd", "#de2d26"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
        axes[0].scatter(xy[take, 0], xy[take, 1], s=6, c=colors[y[take]], alpha=0.55, linewidths=0)
        axes[0].set_title("true label")
        axes[1].scatter(xy[take, 0], xy[take, 1], s=6, c=colors[pred[take]], alpha=0.55, linewidths=0)
        axes[1].set_title("prediction")
        for ax in axes:
            ax.set_xlabel("embedding PC1")
            ax.set_ylabel("embedding PC2")
            ax.grid(alpha=0.15)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"embedding PCA plot failed: {exc}", encoding="utf-8")


def plot_waveform_errors(path: Path, ds: Any, payload: EvalPayload, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        mistake_idx = np.where(payload.y != payload.pred)[0]
        if len(mistake_idx) == 0:
            path.with_suffix(".txt").write_text("no mistakes to plot", encoding="utf-8")
            return
        selected = []
        for true_label, pred_label in [(0, 1), (1, 0), (2, 1), (1, 2)]:
            idx = mistake_idx[(payload.y[mistake_idx] == true_label) & (payload.pred[mistake_idx] == pred_label)]
            selected.extend(idx[:3].tolist())
        selected = selected[:12] or mistake_idx[:12].tolist()
        n = len(selected)
        cols = 3
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), dpi=160, squeeze=False)
        for ax, i in zip(axes.ravel(), selected):
            x = ds.x[i, 0]
            ax.plot(x, color="#222222", lw=0.9)
            ax.set_title(f"{INT_TO_CLASS[int(payload.y[i])]} -> {INT_TO_CLASS[int(payload.pred[i])]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(selected) :]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"waveform plot failed: {exc}", encoding="utf-8")


def rank_candidates(metrics: pd.DataFrame) -> list[str]:
    view = metrics[(metrics["bucket"] == "synthetic_test") & (~metrics["candidate"].astype(str).str.endswith("_badcal"))].copy()
    if view.empty:
        return []
    view["score"] = view["acc"].astype(float) + 0.25 * view["macro_f1"].astype(float) + 0.25 * view[["good_recall", "medium_recall", "bad_recall"]].astype(float).min(axis=1)
    return view.sort_values("score", ascending=False)["candidate"].tolist()


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform Geometry Student",
        "",
        "Input is waveform only. SQI/geometry features are teacher targets in the loss, not classifier inputs. Original BUT is report-only.",
        "",
        "## Dirty Worktree Warning",
        "",
        "Existing worktree changes were present before this runner. This experiment writes only external analysis/report/run outputs.",
        "",
        "```text",
        (payload.get("dirty_worktree_warning") or "(clean)").strip()[:4000],
        "```",
        "",
        "## Metrics",
        "",
        "| Candidate | Run | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher MAE | Core MAE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    show = metrics.copy()
    for _, row in show.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['run_label']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} | "
            f"{float(row.get('teacher_mae', np.nan)):.4f} | {float(row.get('teacher_core_mae', np.nan)):.4f} |"
        )
    lines.extend(["", "## Baselines", ""])
    for row in payload["reference_rows"]:
        lines.append(
            f"- `{row['candidate']}` {row['bucket']}: acc `{row['acc']:.6f}`, recalls "
            f"{row['good_recall']:.3f}/{row['medium_recall']:.3f}/{row['bad_recall']:.3f}`. {row['notes']}."
        )
    lines.extend(["", "## Candidates", ""])
    for item in payload["summaries"]:
        gate = "pass" if item.get("useful_gate_passed") else "fail"
        lines.append(
            f"- `{item['candidate']}` ({item['run_label']}): best_epoch={item['best_epoch']}, useful_gate={gate}, "
            f"original_report_ran={item['original_report_ran']}, checkpoint=`{item['checkpoint']}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Selection uses synthetic/node diagnostic only.",
            "- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.",
            "- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.",
            f"- Metrics CSV: `{payload['metrics_csv']}`",
            "",
        ]
    )
    return "\n".join(lines)


def git_status_short() -> str:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=str(ROOT), text=True, stderr=subprocess.STDOUT, timeout=15)
        return out
    except Exception as exc:
        return f"git status failed: {exc}"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    main()
