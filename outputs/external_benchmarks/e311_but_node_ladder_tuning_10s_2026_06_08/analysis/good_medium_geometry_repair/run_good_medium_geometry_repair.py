from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
TRIM_RUNNER_PATH = OUT_ROOT / "analysis" / "trim_bad_cleanup" / "run_n7000_trim_bad.py"
DEEP_DIVE_OUT = OUT_ROOT / "analysis" / "good_medium_overlap_deep_dive"
ANALYSIS_OUT = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
ANALYSIS_REPORT = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"

BASE_PROMOTED_VARIANT = "nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_873d37fc4791"
SOURCE_VARIANTS = (
    "nl_n6800_gm_trim_bad_refine_030_sc_overlap_narrow_oscilla_738286481ce6",
    "nl_n6800_gm_trim_bad_refine_medium_high_amplitude_detail__be341e2fdd63",
    "nl_n6800_gm_trim_bad_refine_medium_highdetail_lowqrs_mix0_1c4ef481f4bf",
    "nl_n6800_gm_trim_bad_refine_medium_isolated_lowqrs_baseli_e05c1215b526",
    "nl_n6800_gm_trim_bad_refine_medium_visibleqrs_mound_detai_0032d0df93dd",
)
N7000_GEOMETRY_BASE_VARIANT = "nl_n7000_gm_trim_bad_geom_pc1flat_g030_m038_g126_m168_b15_9160023afe86"
N7100_ADDED_RING_BASE_VARIANT = "nl_n7100_gm_trim_bad_geom_addedring_n7000base_g006_m030_g_f414c5d4968a"
N7110_BEST_BALANCED_VARIANT = "nl_n7110_gm_trim_bad_geom_addedring_n7100base_g003_m010_g_ff57b2cf1d57"
N7110_DIRECT_RULE_BEST_VARIANT = "nl_n7110_gm_trim_bad_geom_directrule_n7100base_g003_m008__69ab5b71cf7d"
N7125_ADDED_RING_BEST_VARIANT = "nl_n7125_gm_trim_bad_geom_addedring_n7100base_g004_m014_g_e48e7d59927b"
N7125_BOUNDARY_BLOCK_PROMOTED_VARIANT = "nl_n7125_gm_trim_bad_boundaryblocks_micro_bad_probe_n7125_83c93e7640e0"
N7150_HIGHGOOD_VARIANT = "nl_n7150_gm_trim_bad_geom_addedring_n7100base_g006_m020_g_e319672f8889"
N7150_BOUNDARY_BLOCK_PROMOTED_VARIANT = "nl_n7150_gm_trim_bad_boundaryblocks_micro_bad_probe_n7125_84064417a3c4"
N7155_BOUNDARY_BLOCK_PROMOTED_VARIANT = "nl_n7155_gm_trim_bad_boundaryblocks_breakthrough_softbala_5c50f0ee6d7a"
N7000_GEOMETRY_SOURCE_VARIANTS = (
    "nl_n7000_gm_trim_bad_geom_pc1flat_g018_m026_g122_m170_b15_5d375e2f0ace",
    "nl_n7000_gm_trim_bad_geom_pc1flat_g024_m032_g124_m170_b15_902479e72a11",
    "nl_n7000_gm_trim_bad_geom_pc1flat_g030_m038_g126_m168_b15_9160023afe86",
    "nl_n6800_gm_trim_bad_refine_medium_highdetail_lowqrs_mix0_1c4ef481f4bf",
    "nl_n6800_gm_trim_bad_refine_medium_visibleqrs_mound_detai_0032d0df93dd",
)
N7200_OLD_BEST_BASE_VARIANT = "nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good124_mid172_ec4f54fe7e3d"
N7200_TRI_BAND_SOURCE_VARIANTS = (
    N7200_OLD_BEST_BASE_VARIANT,
    "nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good126_mid170_0c33c60a0088",
    "nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good128_mid168_837d9498a6ae",
    "nl_n7200_gm_trim_bad_pair_a12_g18_m30_mid174_seed20260712",
    "nl_n7200_gm_trim_bad_pair_a12_g22_m12_mid172_seed20260711",
    "nl_n7200_gm_trim_bad_refine_medium_visibleqrs_highamp_ove_7f46689726e1",
)
BAD_STRESS_SOURCE_VARIANTS = (
    N7200_OLD_BEST_BASE_VARIANT,
    "nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good126_mid170_0c33c60a0088",
    "nl_n6800_gm_trim_bad_refine_030_sc_overlap_narrow_oscilla_738286481ce6",
    "nl_n6800_gm_trim_bad_refine_medium_highdetail_lowqrs_mix0_1c4ef481f4bf",
)
N7100_ADDED_RING_ROWS_PATH = ANALYSIS_OUT / "n7100_added_boundary_rows.csv"
N7110_ADDED_RING_ROWS_PATH = ANALYSIS_OUT / "n7110_added_boundary_rows.csv"
N7110_DISAGREEMENT_ROWS_PATH = ANALYSIS_OUT / "n7110_disagreement_boundary_rows.csv"
N7110_MEDIUM_RESCUE_ROWS_PATH = ANALYSIS_OUT / "n7110_medium_rescue_rows.csv"
N7110_DIRECT_ERROR_RULE_ROWS_PATH = ANALYSIS_OUT / "n7110_direct_error_rule_rows.csv"
N7125_ADDED_RING_ROWS_PATH = ANALYSIS_OUT / "n7125_added_boundary_rows.csv"
N7150_ADDED_RING_ROWS_PATH = ANALYSIS_OUT / "n7150_added_boundary_rows.csv"
N7150_MEDIUM_LOST_ROWS_PATH = ANALYSIS_OUT / "n7150_medium_lost_by_highgood_rows.csv"

GEOMETRY_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_pc1flat_g018_m026_g122_m170_b150",
        "good_aux_frac": 0.018,
        "medium_aux_frac": 0.026,
        "good_aux_weight": 1.04,
        "medium_aux_weight": 1.06,
        "class_weight": "1.22,1.70,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260901,
        "epochs": (5, 5),
    },
    {
        "name": "geom_pc1flat_g024_m032_g124_m170_b150",
        "good_aux_frac": 0.024,
        "medium_aux_frac": 0.032,
        "good_aux_weight": 1.04,
        "medium_aux_weight": 1.07,
        "class_weight": "1.24,1.70,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260902,
        "epochs": (5, 5),
    },
    {
        "name": "geom_pc1flat_g030_m038_g126_m168_b150",
        "good_aux_frac": 0.030,
        "medium_aux_frac": 0.038,
        "good_aux_weight": 1.03,
        "medium_aux_weight": 1.07,
        "class_weight": "1.26,1.68,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260903,
        "epochs": (5, 5),
    },
)
STACKED_7200_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_stack_n7000_g004_m034_g112_m180_b150",
        "base_variant_id": N7000_GEOMETRY_BASE_VARIANT,
        "source_variant_ids": N7000_GEOMETRY_SOURCE_VARIANTS,
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.034,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.09,
        "class_weight": "1.12,1.80,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260921,
        "epochs": (5, 5),
    },
    {
        "name": "geom_stack_n7000_g008_m044_g115_m184_b150",
        "base_variant_id": N7000_GEOMETRY_BASE_VARIANT,
        "source_variant_ids": N7000_GEOMETRY_SOURCE_VARIANTS,
        "good_aux_frac": 0.008,
        "medium_aux_frac": 0.044,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.10,
        "class_weight": "1.15,1.84,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260922,
        "epochs": (5, 5),
    },
    {
        "name": "geom_stack_n7000_g000_m052_g110_m188_b150",
        "base_variant_id": N7000_GEOMETRY_BASE_VARIANT,
        "source_variant_ids": N7000_GEOMETRY_SOURCE_VARIANTS,
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.052,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.10,
        "class_weight": "1.10,1.88,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260923,
        "epochs": (5, 5),
    },
)
TRI_BAND_7200_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_tri_atlaspc2_oldbest_g003_m010_g118_m172_b150",
        "geometry_profile": "n7200_tri_band",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": N7200_TRI_BAND_SOURCE_VARIANTS,
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.010,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.18,1.72,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260931,
        "epochs": (5, 5),
        "tri_good_gate": {
            "pc1_max": -3.60,
            "flatline_min": 0.20,
            "pc3_max": 0.40,
            "qrs_visibility_min": 0.45,
            "non_qrs_diff_max": 0.085,
        },
        "tri_medium_gate": {
            "pc1_min": -1.20,
            "flatline_max": 0.14,
            "pc3_min": 1.40,
            "non_qrs_diff_min": 0.085,
            "qrs_visibility_min": 0.18,
            "qrs_visibility_max": 0.58,
        },
    },
    {
        "name": "geom_tri_atlaspc2_oldbest_g005_m014_g120_m174_b150",
        "geometry_profile": "n7200_tri_band",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": N7200_TRI_BAND_SOURCE_VARIANTS,
        "good_aux_frac": 0.005,
        "medium_aux_frac": 0.014,
        "good_aux_weight": 1.02,
        "medium_aux_weight": 1.05,
        "class_weight": "1.20,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260932,
        "epochs": (5, 5),
        "tri_good_gate": {
            "pc1_max": -3.75,
            "flatline_min": 0.22,
            "pc3_max": 0.20,
            "qrs_visibility_min": 0.48,
            "non_qrs_diff_max": 0.080,
        },
        "tri_medium_gate": {
            "pc1_min": -1.05,
            "flatline_max": 0.13,
            "pc3_min": 1.60,
            "non_qrs_diff_min": 0.090,
            "qrs_visibility_min": 0.16,
            "qrs_visibility_max": 0.56,
        },
    },
    {
        "name": "geom_tri_atlaspc2_oldbest_g004_m018_g116_m176_b150",
        "geometry_profile": "n7200_tri_band",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": N7200_TRI_BAND_SOURCE_VARIANTS,
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.018,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.06,
        "class_weight": "1.16,1.76,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260933,
        "epochs": (5, 5),
        "tri_good_gate": {
            "pc1_max": -3.90,
            "flatline_min": 0.24,
            "pc3_max": 0.00,
            "qrs_visibility_min": 0.50,
            "non_qrs_diff_max": 0.075,
        },
        "tri_medium_gate": {
            "pc1_min": -0.90,
            "flatline_max": 0.12,
            "pc3_min": 1.80,
            "non_qrs_diff_min": 0.100,
            "qrs_visibility_min": 0.14,
            "qrs_visibility_max": 0.54,
        },
    },
    {
        "name": "geom_tri_atlaspc2_matchold_g002_m006_g124_m172_b170",
        "geometry_profile": "n7200_tri_band",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": N7200_TRI_BAND_SOURCE_VARIANTS,
        "good_aux_frac": 0.002,
        "medium_aux_frac": 0.006,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.01,
        "class_weight": "1.24,1.72,1.70",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260726,
        "epochs": (8, 8),
        "tri_good_gate": {
            "pc1_max": -3.60,
            "flatline_min": 0.20,
            "pc3_max": 0.40,
            "qrs_visibility_min": 0.45,
            "non_qrs_diff_max": 0.085,
        },
        "tri_medium_gate": {
            "pc1_min": -1.20,
            "flatline_max": 0.14,
            "pc3_min": 1.40,
            "non_qrs_diff_min": 0.085,
            "qrs_visibility_min": 0.18,
            "qrs_visibility_max": 0.58,
        },
    },
    {
        "name": "geom_tri_atlaspc2_matchold_g003_m010_g124_m172_b170",
        "geometry_profile": "n7200_tri_band",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": N7200_TRI_BAND_SOURCE_VARIANTS,
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.010,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.02,
        "class_weight": "1.24,1.72,1.70",
        "loss": "soft_sample_weighted_ce",
        "seed": 20260727,
        "epochs": (8, 8),
        "tri_good_gate": {
            "pc1_max": -3.60,
            "flatline_min": 0.20,
            "pc3_max": 0.40,
            "qrs_visibility_min": 0.45,
            "non_qrs_diff_max": 0.085,
        },
        "tri_medium_gate": {
            "pc1_min": -1.20,
            "flatline_max": 0.14,
            "pc3_min": 1.40,
            "non_qrs_diff_min": 0.085,
            "qrs_visibility_min": 0.18,
            "qrs_visibility_max": 0.58,
        },
    },
)
ADDED_RING_7100_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_addedring_n7000base_g006_m018_g124_m170_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7000_GEOMETRY_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "good_aux_frac": 0.006,
        "medium_aux_frac": 0.018,
        "good_aux_weight": 1.02,
        "medium_aux_weight": 1.06,
        "class_weight": "1.24,1.70,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261011,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7000base_g010_m024_g126_m170_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7000_GEOMETRY_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "good_aux_frac": 0.010,
        "medium_aux_frac": 0.024,
        "good_aux_weight": 1.03,
        "medium_aux_weight": 1.07,
        "class_weight": "1.26,1.70,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261012,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7000base_g006_m030_g122_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7000_GEOMETRY_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "good_aux_frac": 0.006,
        "medium_aux_frac": 0.030,
        "good_aux_weight": 1.02,
        "medium_aux_weight": 1.08,
        "class_weight": "1.22,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261013,
        "epochs": (5, 5),
    },
)
ADDED_RING_7110_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_addedring_n7100base_g003_m010_g124_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.010,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.24,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261061,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7100base_g004_m012_g122_m176_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.012,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.05,
        "class_weight": "1.22,1.76,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261062,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7100base_g004_m014_g122_m176_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.014,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.05,
        "class_weight": "1.22,1.76,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261063,
        "epochs": (5, 5),
    },
)
REFINED_7110_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_disagree_n7100base_g002_m006_g124_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DISAGREEMENT_ROWS_PATH),
        "good_aux_frac": 0.002,
        "medium_aux_frac": 0.006,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.02,
        "class_weight": "1.24,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261071,
        "epochs": (5, 5),
    },
    {
        "name": "geom_disagree_n7100base_g003_m008_g124_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DISAGREEMENT_ROWS_PATH),
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.008,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.03,
        "class_weight": "1.24,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261072,
        "epochs": (5, 5),
    },
    {
        "name": "geom_disagree_n7100base_g003_m010_g122_m176_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DISAGREEMENT_ROWS_PATH),
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.010,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.22,1.76,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261073,
        "epochs": (5, 5),
    },
)
MEDIUM_RESCUE_7110_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_mediumrescue_n7110best_g000_m004_g126_m170_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7110_BEST_BALANCED_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_MEDIUM_RESCUE_ROWS_PATH),
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.004,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.03,
        "class_weight": "1.26,1.70,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261081,
        "epochs": (5, 5),
    },
    {
        "name": "geom_mediumrescue_n7110best_g000_m006_g124_m172_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7110_BEST_BALANCED_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_MEDIUM_RESCUE_ROWS_PATH),
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.006,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.04,
        "class_weight": "1.24,1.72,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261082,
        "epochs": (5, 5),
    },
    {
        "name": "geom_mediumrescue_n7110best_g000_m008_g122_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7110_BEST_BALANCED_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_MEDIUM_RESCUE_ROWS_PATH),
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.008,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.05,
        "class_weight": "1.22,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261083,
        "epochs": (5, 5),
    },
)
DIRECT_RULE_7110_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_directrule_n7100base_g002_m006_g126_m170_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DIRECT_ERROR_RULE_ROWS_PATH),
        "good_aux_frac": 0.002,
        "medium_aux_frac": 0.006,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.03,
        "class_weight": "1.26,1.70,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261091,
        "epochs": (5, 5),
    },
    {
        "name": "geom_directrule_n7100base_g003_m008_g124_m172_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DIRECT_ERROR_RULE_ROWS_PATH),
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.008,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.24,1.72,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261092,
        "epochs": (5, 5),
    },
    {
        "name": "geom_directrule_n7100base_g004_m010_g122_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DIRECT_ERROR_RULE_ROWS_PATH),
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.010,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.22,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261093,
        "epochs": (5, 5),
    },
    {
        "name": "geom_directrule_micro_n7100base_g0032_m0085_g1235_m1725_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DIRECT_ERROR_RULE_ROWS_PATH),
        "good_aux_frac": 0.0032,
        "medium_aux_frac": 0.0085,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.235,1.725,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261094,
        "epochs": (5, 5),
    },
    {
        "name": "geom_directrule_micro_n7100base_g0035_m0090_g123_m173_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DIRECT_ERROR_RULE_ROWS_PATH),
        "good_aux_frac": 0.0035,
        "medium_aux_frac": 0.009,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.23,1.73,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261095,
        "epochs": (5, 5),
    },
    {
        "name": "geom_directrule_micro_n7100base_g0038_m0095_g1225_m1735_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7110_DIRECT_ERROR_RULE_ROWS_PATH),
        "good_aux_frac": 0.0038,
        "medium_aux_frac": 0.0095,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "class_weight": "1.225,1.735,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261096,
        "epochs": (5, 5),
    },
)
QRSLOW_MEDIUM_7110_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_qrslowmed_n7110best_g000_m003_g128_m168_b150",
        "geometry_profile": "n7110_qrslow_medium",
        "base_variant_id": N7110_DIRECT_RULE_BEST_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.003,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.10,
        "class_weight": "1.28,1.68,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261101,
        "epochs": (5, 5),
        "qrslow_medium_gate": {
            "qrs_visibility_max": 0.045,
            "pc1_min": -1.55,
            "pc3_min": 1.40,
            "flatline_max": 0.24,
            "non_qrs_diff_min": 0.045,
        },
    },
    {
        "name": "geom_qrslowmed_n7110best_g000_m005_g126_m170_b150",
        "geometry_profile": "n7110_qrslow_medium",
        "base_variant_id": N7110_DIRECT_RULE_BEST_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.005,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.12,
        "class_weight": "1.26,1.70,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261102,
        "epochs": (5, 5),
        "qrslow_medium_gate": {
            "qrs_visibility_max": 0.050,
            "pc1_min": -1.65,
            "pc3_min": 1.25,
            "flatline_max": 0.26,
            "non_qrs_diff_min": 0.040,
        },
    },
    {
        "name": "geom_qrslowmed_n7110best_g000_m007_g124_m172_b150",
        "geometry_profile": "n7110_qrslow_medium",
        "base_variant_id": N7110_DIRECT_RULE_BEST_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.007,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.14,
        "class_weight": "1.24,1.72,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261103,
        "epochs": (5, 5),
        "qrslow_medium_gate": {
            "qrs_visibility_max": 0.055,
            "pc1_min": -1.75,
            "pc3_min": 1.10,
            "flatline_max": 0.28,
            "non_qrs_diff_min": 0.035,
        },
    },
)
ADDED_RING_7125_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_addedring_n7100base_g002_m010_g122_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7125_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.002,
        "medium_aux_frac": 0.010,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.04,
        "class_weight": "1.22,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261021,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7100base_g004_m014_g122_m176_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7125_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.014,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.05,
        "class_weight": "1.22,1.76,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261022,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7100base_g002_m018_g118_m180_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7125_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.002,
        "medium_aux_frac": 0.018,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.07,
        "class_weight": "1.18,1.80,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261023,
        "epochs": (5, 5),
    },
)
ADDED_RING_7150_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_addedring_n7100base_g004_m014_g124_m172_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7150_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.014,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.05,
        "class_weight": "1.24,1.72,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261031,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7100base_g006_m020_g124_m174_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7150_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.006,
        "medium_aux_frac": 0.020,
        "good_aux_weight": 1.02,
        "medium_aux_weight": 1.06,
        "class_weight": "1.24,1.74,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261032,
        "epochs": (5, 5),
    },
    {
        "name": "geom_addedring_n7100base_g004_m026_g122_m176_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7100_ADDED_RING_BASE_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7150_ADDED_RING_ROWS_PATH),
        "good_aux_frac": 0.004,
        "medium_aux_frac": 0.026,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.08,
        "class_weight": "1.22,1.76,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261033,
        "epochs": (5, 5),
    },
)
MEDIUM_RESCUE_7150_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_mediumlost_n7150base_g000_m010_g118_m180_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7150_HIGHGOOD_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7150_MEDIUM_LOST_ROWS_PATH),
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.010,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.08,
        "class_weight": "1.18,1.80,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261041,
        "epochs": (5, 5),
    },
    {
        "name": "geom_mediumlost_n7150base_g000_m016_g116_m184_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7150_HIGHGOOD_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7150_MEDIUM_LOST_ROWS_PATH),
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.016,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.10,
        "class_weight": "1.16,1.84,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261042,
        "epochs": (5, 5),
    },
    {
        "name": "geom_mediumlost_n7150base_g000_m022_g114_m188_b150",
        "geometry_profile": "n7100_added_ring",
        "base_variant_id": N7150_HIGHGOOD_VARIANT,
        "source_variant_ids": SOURCE_VARIANTS,
        "added_ring_rows_path": str(N7150_MEDIUM_LOST_ROWS_PATH),
        "good_aux_frac": 0.000,
        "medium_aux_frac": 0.022,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.12,
        "class_weight": "1.14,1.88,1.50",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261043,
        "epochs": (5, 5),
    },
)

BAD_STRESS_7200_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "geom_badstress_controlled_b004_g003_m010_g118_m172_b154",
        "geometry_profile": "n7200_badstress",
        "badstress_mode": "controlled_qrsband",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": BAD_STRESS_SOURCE_VARIANTS,
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.010,
        "bad_aux_frac": 0.004,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "bad_aux_weight": 1.02,
        "class_weight": "1.18,1.72,1.54",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261141,
        "epochs": (5, 5),
        "tri_good_gate": {
            "pc1_max": -3.60,
            "flatline_min": 0.20,
            "pc3_max": 0.40,
            "qrs_visibility_min": 0.45,
            "non_qrs_diff_max": 0.085,
        },
        "tri_medium_gate": {
            "pc1_min": -1.20,
            "flatline_max": 0.14,
            "pc3_min": 1.40,
            "non_qrs_diff_min": 0.085,
            "qrs_visibility_min": 0.18,
            "qrs_visibility_max": 0.58,
        },
        "badstress_gate": {
            "qrs_band_ratio_min": 0.42,
            "band_15_30_min": 0.74,
            "baseline_step_max": 0.26,
            "band_30_45_max": 0.18,
            "fatal_max": 1.10,
            "contact_loss_max": 0.03,
        },
    },
    {
        "name": "geom_badstress_controlled_b008_g003_m010_g118_m172_b158",
        "geometry_profile": "n7200_badstress",
        "badstress_mode": "controlled_qrsband",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": BAD_STRESS_SOURCE_VARIANTS,
        "good_aux_frac": 0.003,
        "medium_aux_frac": 0.010,
        "bad_aux_frac": 0.008,
        "good_aux_weight": 1.01,
        "medium_aux_weight": 1.04,
        "bad_aux_weight": 1.03,
        "class_weight": "1.18,1.72,1.58",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261142,
        "epochs": (5, 5),
        "tri_good_gate": {
            "pc1_max": -3.60,
            "flatline_min": 0.20,
            "pc3_max": 0.40,
            "qrs_visibility_min": 0.45,
            "non_qrs_diff_max": 0.085,
        },
        "tri_medium_gate": {
            "pc1_min": -1.20,
            "flatline_max": 0.14,
            "pc3_min": 1.40,
            "non_qrs_diff_min": 0.085,
            "qrs_visibility_min": 0.18,
            "qrs_visibility_max": 0.58,
        },
        "badstress_gate": {
            "qrs_band_ratio_min": 0.42,
            "band_15_30_min": 0.74,
            "baseline_step_max": 0.26,
            "band_30_45_max": 0.18,
            "fatal_max": 1.10,
            "contact_loss_max": 0.03,
        },
    },
    {
        "name": "geom_badstress_corehf_b006_g002_m012_g116_m174_b156",
        "geometry_profile": "n7200_badstress",
        "badstress_mode": "moderate_baseline_hf",
        "base_variant_id": N7200_OLD_BEST_BASE_VARIANT,
        "source_variant_ids": BAD_STRESS_SOURCE_VARIANTS,
        "good_aux_frac": 0.002,
        "medium_aux_frac": 0.012,
        "bad_aux_frac": 0.006,
        "good_aux_weight": 1.00,
        "medium_aux_weight": 1.05,
        "bad_aux_weight": 1.03,
        "class_weight": "1.16,1.74,1.56",
        "loss": "soft_sample_weighted_ce",
        "seed": 20261143,
        "epochs": (5, 5),
        "tri_good_gate": {
            "pc1_max": -3.75,
            "flatline_min": 0.22,
            "pc3_max": 0.20,
            "qrs_visibility_min": 0.48,
            "non_qrs_diff_max": 0.080,
        },
        "tri_medium_gate": {
            "pc1_min": -1.05,
            "flatline_max": 0.13,
            "pc3_min": 1.60,
            "non_qrs_diff_min": 0.090,
            "qrs_visibility_min": 0.16,
            "qrs_visibility_max": 0.56,
        },
        "badstress_gate": {
            "baseline_step_min": 0.085,
            "baseline_step_max": 0.34,
            "band_30_45_min": 0.075,
            "qrs_band_ratio_max": 0.48,
            "fatal_max": 1.10,
            "contact_loss_max": 0.03,
        },
    },
)

BOUNDARY_BLOCK_SOURCE_VARIANTS = (
    N7100_ADDED_RING_BASE_VARIANT,
    N7110_DIRECT_RULE_BEST_VARIANT,
    N7200_OLD_BEST_BASE_VARIANT,
    "nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good126_mid170_0c33c60a0088",
    "nl_n6800_gm_trim_bad_refine_medium_highdetail_lowqrs_mix0_1c4ef481f4bf",
    "nl_n6800_gm_trim_bad_refine_medium_visibleqrs_mound_detai_0032d0df93dd",
)


def _boundary_block_cfg(
    name: str,
    *,
    seed: int,
    base_variant_id: str,
    class_weight: str,
    blocks: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "name": name,
        "geometry_profile": "boundary_blocks",
        "base_variant_id": base_variant_id,
        "source_variant_ids": BOUNDARY_BLOCK_SOURCE_VARIANTS,
        "boundary_blocks": blocks,
        "class_weight": class_weight,
        "loss": "soft_sample_weighted_ce",
        "seed": seed,
        "epochs": (5, 5),
        "tri_good_gate": {
            "pc1_max": -3.55,
            "flatline_min": 0.18,
            "pc3_max": 0.55,
            "qrs_visibility_min": 0.42,
            "non_qrs_diff_max": 0.095,
        },
        "tri_medium_gate": {
            "pc1_min": -1.30,
            "flatline_max": 0.16,
            "pc3_min": 1.25,
            "non_qrs_diff_min": 0.075,
            "qrs_visibility_min": 0.16,
            "qrs_visibility_max": 0.62,
        },
        "qrslow_medium_gate": {
            "qrs_visibility_max": 0.042,
            "pc1_min": -1.70,
            "pc3_min": 1.15,
            "flatline_max": 0.26,
            "non_qrs_diff_min": 0.040,
            "fatal_max": 1.20,
            "contact_loss_max": 0.08,
        },
        "badstress_mode": "controlled_qrsband",
        "badstress_gate": {
            "qrs_band_ratio_min": 0.40,
            "band_15_30_min": 0.70,
            "baseline_step_max": 0.28,
            "band_30_45_max": 0.20,
            "fatal_max": 1.10,
            "contact_loss_max": 0.035,
        },
    }


BOUNDARY_BLOCK_CONFIGS: tuple[dict[str, Any], ...] = (
    _boundary_block_cfg(
        "boundaryblocks_balanced_gm_controlbad_g004_m010_q004_b002",
        seed=20261201,
        base_variant_id=N7100_ADDED_RING_BASE_VARIANT,
        class_weight="1.22,1.72,1.50",
        blocks=(
            {
                "block": "gm_good_rescue_pc1flat_qrsvisible",
                "class_name": "good",
                "picker": "good_rescue_pc1flat_qrsvisible",
                "frac": 0.004,
                "weight": 1.02,
            },
            {
                "block": "gm_visible_qrs_medium_detail",
                "class_name": "medium",
                "picker": "medium_visible_qrs_detail",
                "frac": 0.010,
                "weight": 1.06,
            },
            {
                "block": "gm_medium_qrslow_hardneg",
                "class_name": "medium",
                "picker": "medium_qrslow_hardneg",
                "frac": 0.004,
                "weight": 1.04,
            },
            {
                "block": "bad_controlled_outlier",
                "class_name": "bad",
                "picker": "bad_controlled_outlier",
                "frac": 0.002,
                "weight": 1.01,
            },
        ),
    ),
    _boundary_block_cfg(
        "boundaryblocks_medium_repair_lightbad_g002_m014_q006_b002",
        seed=20261202,
        base_variant_id=N7100_ADDED_RING_BASE_VARIANT,
        class_weight="1.16,1.80,1.48",
        blocks=(
            {
                "block": "gm_good_rescue_pc1flat_qrsvisible",
                "class_name": "good",
                "picker": "good_rescue_pc1flat_qrsvisible",
                "frac": 0.002,
                "weight": 1.01,
            },
            {
                "block": "gm_visible_qrs_medium_detail",
                "class_name": "medium",
                "picker": "medium_visible_qrs_detail",
                "frac": 0.014,
                "weight": 1.08,
            },
            {
                "block": "gm_medium_qrslow_hardneg",
                "class_name": "medium",
                "picker": "medium_qrslow_hardneg",
                "frac": 0.006,
                "weight": 1.05,
            },
            {
                "block": "bad_controlled_outlier",
                "class_name": "bad",
                "picker": "bad_controlled_outlier",
                "frac": 0.002,
                "weight": 1.01,
            },
        ),
    ),
    _boundary_block_cfg(
        "boundaryblocks_good_rescue_lightbad_g008_m006_q003_b002",
        seed=20261203,
        base_variant_id=N7100_ADDED_RING_BASE_VARIANT,
        class_weight="1.30,1.66,1.48",
        blocks=(
            {
                "block": "gm_good_rescue_pc1flat_qrsvisible",
                "class_name": "good",
                "picker": "good_rescue_pc1flat_qrsvisible",
                "frac": 0.008,
                "weight": 1.03,
            },
            {
                "block": "gm_visible_qrs_medium_detail",
                "class_name": "medium",
                "picker": "medium_visible_qrs_detail",
                "frac": 0.006,
                "weight": 1.05,
            },
            {
                "block": "gm_medium_qrslow_hardneg",
                "class_name": "medium",
                "picker": "medium_qrslow_hardneg",
                "frac": 0.003,
                "weight": 1.03,
            },
            {
                "block": "bad_controlled_outlier",
                "class_name": "bad",
                "picker": "bad_controlled_outlier",
                "frac": 0.002,
                "weight": 1.01,
            },
        ),
    ),
    _boundary_block_cfg(
        "boundaryblocks_badstress_guard_only_g003_m008_q003_b004",
        seed=20261204,
        base_variant_id=N7100_ADDED_RING_BASE_VARIANT,
        class_weight="1.20,1.72,1.52",
        blocks=(
            {
                "block": "gm_good_rescue_pc1flat_qrsvisible",
                "class_name": "good",
                "picker": "good_rescue_pc1flat_qrsvisible",
                "frac": 0.003,
                "weight": 1.01,
            },
            {
                "block": "gm_visible_qrs_medium_detail",
                "class_name": "medium",
                "picker": "medium_visible_qrs_detail",
                "frac": 0.008,
                "weight": 1.06,
            },
            {
                "block": "gm_medium_qrslow_hardneg",
                "class_name": "medium",
                "picker": "medium_qrslow_hardneg",
                "frac": 0.003,
                "weight": 1.03,
            },
            {
                "block": "bad_controlled_outlier",
                "class_name": "bad",
                "picker": "bad_controlled_outlier",
                "frac": 0.004,
                "weight": 1.02,
            },
        ),
    ),
)

def _boundary_block_micro_configs(base_variant_id: str, label: str, seed_base: int) -> tuple[dict[str, Any], ...]:
    """Very small paired block probes around the current level-specific best body."""
    return (
        _boundary_block_cfg(
            f"boundaryblocks_micro_balanced_{label}_g0015_m003_q0015_b0005",
            seed=seed_base + 1,
            base_variant_id=base_variant_id,
            class_weight="1.24,1.74,1.50",
            blocks=(
                {
                    "block": "gm_good_rescue_pc1flat_qrsvisible",
                    "class_name": "good",
                    "picker": "good_rescue_pc1flat_qrsvisible",
                    "frac": 0.0015,
                    "weight": 1.005,
                },
                {
                    "block": "gm_visible_qrs_medium_detail",
                    "class_name": "medium",
                    "picker": "medium_visible_qrs_detail",
                    "frac": 0.003,
                    "weight": 1.012,
                },
                {
                    "block": "gm_medium_qrslow_hardneg",
                    "class_name": "medium",
                    "picker": "medium_qrslow_hardneg",
                    "frac": 0.0015,
                    "weight": 1.008,
                },
                {
                    "block": "bad_controlled_outlier",
                    "class_name": "bad",
                    "picker": "bad_controlled_outlier",
                    "frac": 0.0005,
                    "weight": 1.002,
                },
            ),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_micro_medium_pair_{label}_g001_m004_q0015_b000",
            seed=seed_base + 2,
            base_variant_id=base_variant_id,
            class_weight="1.24,1.74,1.50",
            blocks=(
                {
                    "block": "gm_good_rescue_pc1flat_qrsvisible",
                    "class_name": "good",
                    "picker": "good_rescue_pc1flat_qrsvisible",
                    "frac": 0.001,
                    "weight": 1.003,
                },
                {
                    "block": "gm_visible_qrs_medium_detail",
                    "class_name": "medium",
                    "picker": "medium_visible_qrs_detail",
                    "frac": 0.004,
                    "weight": 1.015,
                },
                {
                    "block": "gm_medium_qrslow_hardneg",
                    "class_name": "medium",
                    "picker": "medium_qrslow_hardneg",
                    "frac": 0.0015,
                    "weight": 1.008,
                },
            ),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_micro_good_pair_{label}_g002_m002_q001_b000",
            seed=seed_base + 3,
            base_variant_id=base_variant_id,
            class_weight="1.25,1.72,1.50",
            blocks=(
                {
                    "block": "gm_good_rescue_pc1flat_qrsvisible",
                    "class_name": "good",
                    "picker": "good_rescue_pc1flat_qrsvisible",
                    "frac": 0.002,
                    "weight": 1.008,
                },
                {
                    "block": "gm_visible_qrs_medium_detail",
                    "class_name": "medium",
                    "picker": "medium_visible_qrs_detail",
                    "frac": 0.002,
                    "weight": 1.008,
                },
                {
                    "block": "gm_medium_qrslow_hardneg",
                    "class_name": "medium",
                    "picker": "medium_qrslow_hardneg",
                    "frac": 0.001,
                    "weight": 1.005,
                },
            ),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_micro_bad_probe_{label}_g001_m002_q001_b001",
            seed=seed_base + 4,
            base_variant_id=base_variant_id,
            class_weight="1.24,1.73,1.50",
            blocks=(
                {
                    "block": "gm_good_rescue_pc1flat_qrsvisible",
                    "class_name": "good",
                    "picker": "good_rescue_pc1flat_qrsvisible",
                    "frac": 0.001,
                    "weight": 1.003,
                },
                {
                    "block": "gm_visible_qrs_medium_detail",
                    "class_name": "medium",
                    "picker": "medium_visible_qrs_detail",
                    "frac": 0.002,
                    "weight": 1.008,
                },
                {
                    "block": "gm_medium_qrslow_hardneg",
                    "class_name": "medium",
                    "picker": "medium_qrslow_hardneg",
                    "frac": 0.001,
                    "weight": 1.005,
                },
                {
                    "block": "bad_controlled_outlier",
                    "class_name": "bad",
                    "picker": "bad_controlled_outlier",
                    "frac": 0.001,
                    "weight": 1.004,
                },
            ),
        ),
    )


def _boundary_block_stability_configs(base_variant_id: str, label: str, seed_base: int) -> tuple[dict[str, Any], ...]:
    """Retrain the current best body without adding blocks, to estimate checkpoint stability."""
    return (
        _boundary_block_cfg(
            f"boundaryblocks_stability_{label}_reseed_a",
            seed=seed_base + 1,
            base_variant_id=base_variant_id,
            class_weight="1.24,1.72,1.50",
            blocks=(),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_stability_{label}_reseed_b",
            seed=seed_base + 2,
            base_variant_id=base_variant_id,
            class_weight="1.235,1.725,1.50",
            blocks=(),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_stability_{label}_reseed_c",
            seed=seed_base + 3,
            base_variant_id=base_variant_id,
            class_weight="1.23,1.73,1.50",
            blocks=(),
        ),
    )


def _boundary_block_breakthrough_configs(base_variant_id: str, label: str, seed_base: int) -> tuple[dict[str, Any], ...]:
    """Conservative probes after N7160 became good-heavy.

    N7160's previous good-pair profile reduced good->medium errors but created
    too many medium->good errors. These probes keep the N7150 body as the
    anchor, spend little or no budget on good rescue, and bias the added rows
    toward medium hard negatives plus a tiny controlled-bad guardrail.
    """
    return (
        _boundary_block_cfg(
            f"boundaryblocks_breakthrough_stability_{label}_g000_m000_q000_b000",
            seed=seed_base + 1,
            base_variant_id=base_variant_id,
            class_weight="1.235,1.735,1.50",
            blocks=(),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_breakthrough_mediumguard_{label}_g0005_m0035_q002_b0005",
            seed=seed_base + 2,
            base_variant_id=base_variant_id,
            class_weight="1.20,1.79,1.50",
            blocks=(
                {
                    "block": "gm_good_rescue_pc1flat_qrsvisible",
                    "class_name": "good",
                    "picker": "good_rescue_pc1flat_qrsvisible",
                    "frac": 0.0005,
                    "weight": 1.001,
                },
                {
                    "block": "gm_visible_qrs_medium_detail",
                    "class_name": "medium",
                    "picker": "medium_visible_qrs_detail",
                    "frac": 0.0035,
                    "weight": 1.014,
                },
                {
                    "block": "gm_medium_qrslow_hardneg",
                    "class_name": "medium",
                    "picker": "medium_qrslow_hardneg",
                    "frac": 0.002,
                    "weight": 1.010,
                },
                {
                    "block": "bad_controlled_outlier",
                    "class_name": "bad",
                    "picker": "bad_controlled_outlier",
                    "frac": 0.0005,
                    "weight": 1.002,
                },
            ),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_breakthrough_nogood_medium_{label}_g000_m003_q0025_b001",
            seed=seed_base + 3,
            base_variant_id=base_variant_id,
            class_weight="1.18,1.80,1.50",
            blocks=(
                {
                    "block": "gm_visible_qrs_medium_detail",
                    "class_name": "medium",
                    "picker": "medium_visible_qrs_detail",
                    "frac": 0.003,
                    "weight": 1.014,
                },
                {
                    "block": "gm_medium_qrslow_hardneg",
                    "class_name": "medium",
                    "picker": "medium_qrslow_hardneg",
                    "frac": 0.0025,
                    "weight": 1.012,
                },
                {
                    "block": "bad_controlled_outlier",
                    "class_name": "bad",
                    "picker": "bad_controlled_outlier",
                    "frac": 0.001,
                    "weight": 1.003,
                },
            ),
        ),
        _boundary_block_cfg(
            f"boundaryblocks_breakthrough_softbalance_{label}_g00075_m0025_q0015_b001",
            seed=seed_base + 4,
            base_variant_id=base_variant_id,
            class_weight="1.22,1.76,1.51",
            blocks=(
                {
                    "block": "gm_good_rescue_pc1flat_qrsvisible",
                    "class_name": "good",
                    "picker": "good_rescue_pc1flat_qrsvisible",
                    "frac": 0.00075,
                    "weight": 1.002,
                },
                {
                    "block": "gm_visible_qrs_medium_detail",
                    "class_name": "medium",
                    "picker": "medium_visible_qrs_detail",
                    "frac": 0.0025,
                    "weight": 1.011,
                },
                {
                    "block": "gm_medium_qrslow_hardneg",
                    "class_name": "medium",
                    "picker": "medium_qrslow_hardneg",
                    "frac": 0.0015,
                    "weight": 1.008,
                },
                {
                    "block": "bad_controlled_outlier",
                    "class_name": "bad",
                    "picker": "bad_controlled_outlier",
                    "frac": 0.001,
                    "weight": 1.003,
                },
            ),
        ),
    )


GATE_FEATURES = (
    "pc1",
    "flatline_ratio",
    "pc3",
    "qrs_visibility",
    "non_qrs_diff_p95",
    "qrs_prom_p90",
    "template_corr",
)
PCA_PROXY_FEATURES = (
    "rms",
    "std",
    "ptp_p99_p01",
    "mean_abs",
    "low_amp_ratio",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "baseline_step",
    "diff_abs_p95",
    "qrs_band_ratio",
    "spectral_entropy",
    "qrs_prom_p90",
    "periodicity",
    "qrs_visibility",
    "fatal_or_score",
    "medium_detail_unreliable_score",
    "sqi_bSQI",
    "sqi_pSQI",
    "sqi_sSQI",
    "sqi_kSQI",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "zero_crossing_rate",
    "diff_zero_crossing_rate",
    "amplitude_entropy",
    "template_corr",
    "detector_agreement",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "band_0p3_1",
    "band_1_5",
    "band_5_15",
    "band_15_30",
    "band_30_45",
    "wavelet_e4",
)
PCA_PROXY_TARGETS = ("pc1", "pc2", "pc3", "pc4")
_PCA_PROXY_MODEL: dict[str, Any] | None = None
_N7100_ADDED_RING_ROWS: pd.DataFrame | None = None
_ADDED_RING_ROWS_CACHE: dict[str, pd.DataFrame] = {}


def configs_for_level(level: int) -> tuple[dict[str, Any], ...]:
    if int(level) == 7100:
        return ADDED_RING_7100_CONFIGS
    if int(level) == 7110:
        return (
            ADDED_RING_7110_CONFIGS
            + REFINED_7110_CONFIGS
            + MEDIUM_RESCUE_7110_CONFIGS
            + DIRECT_RULE_7110_CONFIGS
            + QRSLOW_MEDIUM_7110_CONFIGS
        )
    if int(level) == 7125:
        return ADDED_RING_7125_CONFIGS
    if int(level) == 7150:
        return ADDED_RING_7150_CONFIGS + MEDIUM_RESCUE_7150_CONFIGS
    if int(level) >= 7200:
        return TRI_BAND_7200_CONFIGS
    return GEOMETRY_CONFIGS


def badstress_configs_for_level(level: int) -> tuple[dict[str, Any], ...]:
    if int(level) >= 7200:
        return BAD_STRESS_7200_CONFIGS
    raise ValueError(f"Bad-stress controlled expansion is only configured for N7200+, got N{int(level)}.")


def boundary_block_configs_for_level(level: int) -> tuple[dict[str, Any], ...]:
    level = int(level)
    if level == 7110:
        return _boundary_block_micro_configs(N7110_DIRECT_RULE_BEST_VARIANT, "n7110best", 20261300)
    if level == 7125:
        return _boundary_block_micro_configs(N7125_ADDED_RING_BEST_VARIANT, "n7125best", 20261320)
    if level == 7150:
        return _boundary_block_micro_configs(N7125_BOUNDARY_BLOCK_PROMOTED_VARIANT, "n7125bbbest", 20261340)
    if level == 7155:
        return _boundary_block_breakthrough_configs(N7150_BOUNDARY_BLOCK_PROMOTED_VARIANT, "n7150bb_to7155", 20261400)
    if level == 7160:
        return _boundary_block_breakthrough_configs(N7155_BOUNDARY_BLOCK_PROMOTED_VARIANT, "n7155bb_to7160", 20261420)
    if level >= 7175:
        return _boundary_block_micro_configs(N7150_BOUNDARY_BLOCK_PROMOTED_VARIANT, "n7150bbbest", 20261360)
    raise ValueError(f"Boundary block generator is configured for N7110/N7125/N7150/N7155/N7160+, got N{int(level)}.")


def _load_runner():
    spec = importlib.util.spec_from_file_location("geometry_trim_bad_runner", TRIM_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner: {TRIM_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.helper._patch_trim_score()
    module.helper._patch_medium_subtype_specs()
    module.helper._patch_medium_subtype_events()
    return module


runner = _load_runner()
ladder = runner.ladder
targeted = runner.targeted
now_iso = runner.now_iso
update_state = runner.update_state


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def configure_level(level: int) -> str:
    source_id = f"N{int(level)}_gm_probe"
    trim_id = f"N{int(level)}_gm_trim_bad"
    runner.SOURCE_NODE_ID = source_id
    runner.TRIM_NODE_ID = trim_id
    runner.SOURCE_NODE = {
        "node_id": source_id,
        "level": int(level),
        "role": "gm_geometry_repair_probe",
        "target_acc": 0.95,
        "target_good": 0.92,
        "target_medium": 0.90,
        "target_bad": 0.90,
        "train_default": False,
        "reason": f"N{int(level)} geometry repair probe before any wider expansion.",
    }
    runner.TRIM_NODE = {
        "node_id": trim_id,
        "level": int(level),
        "role": "gm_geometry_repair_trim_bad",
        "target_acc": 0.95,
        "target_good": 0.92,
        "target_medium": 0.90,
        "target_bad": 0.90,
        "train_default": False,
        "reason": (
            f"N{int(level)} good/medium geometry repair with bad outlier_low_confidence "
            "trimmed to core/right-island guardrail."
        ),
    }
    runner._patch_node_defs()
    return trim_id


def make_args(level: int, stage: str = "quick", run_training: bool = False) -> argparse.Namespace:
    configure_level(level)
    return runner._args(stage, run_training=run_training)


def _artifact(variant_id: str) -> dict[str, Any]:
    artifact = runner._load_blend_artifact_arrays(variant_id)
    labels = artifact["labels"].copy()
    if "original_idx" in labels.columns:
        original_idx = pd.to_numeric(labels["original_idx"], errors="coerce")
        labels["original_idx"] = original_idx.fillna(pd.to_numeric(labels["idx"], errors="coerce")).astype(int)
    else:
        fallback = pd.Series(np.arange(len(labels)), index=labels.index)
        labels["original_idx"] = pd.to_numeric(labels["idx"], errors="coerce").fillna(fallback).astype(int)
    artifact["labels"] = labels
    return artifact


def _source_counts(total: int, sources: tuple[str, ...]) -> list[int]:
    return runner._split_source_counts(int(total), sources)


def _robust_z(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = vals.median()
    iqr = vals.quantile(0.75) - vals.quantile(0.25)
    scale = float(iqr) if pd.notna(iqr) and abs(float(iqr)) > 1e-9 else float(vals.std(skipna=True) or 1.0)
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return ((vals - med) / scale).fillna(0.0)


def _class_split_counts(labels: pd.DataFrame, class_name: str, frac: float) -> dict[str, int]:
    counts = (
        labels.loc[labels["y_class"].astype(str).eq(class_name), "split"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {split: max(0, int(round(int(n) * float(frac)))) for split, n in counts.items()}


def _feature_merge(artifact: dict[str, Any], class_name: str, split: str) -> pd.DataFrame:
    labels = artifact["labels"].copy().reset_index().rename(columns={"index": "row_pos"})
    pool = labels.loc[
        labels["y_class"].astype(str).eq(class_name) & labels["split"].astype(str).eq(split)
    ].copy()
    atlas = artifact.get("atlas")
    if pool.empty or atlas is None or "idx" not in atlas.columns:
        return pool
    join_col = "original_idx" if "original_idx" in pool.columns else "idx"
    feature_cols = [
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pca_margin",
        "boundary_confidence",
        "flatline_ratio",
        "qrs_visibility",
        "qrs_band_ratio",
        "non_qrs_diff_p95",
        "qrs_prom_p90",
        "template_corr",
        "detector_agreement",
        "baseline_step",
        "non_qrs_rms_ratio",
        "diff_abs_p95",
        "spectral_entropy",
        "band_15_30",
        "band_30_45",
        "rms",
        "std",
        "mean_abs",
        "ptp_p99_p01",
        "amplitude_entropy",
        "low_amp_ratio",
        "sqi_sSQI",
        "sqi_kSQI",
        "fatal_or_score",
        "contact_loss_win_ratio",
    ]
    cols = ["idx", *[c for c in feature_cols if c in atlas.columns]]
    merged = pool.merge(atlas[cols], left_on=join_col, right_on="idx", how="left", suffixes=("_label", "_feat"))
    feature_cols_present = [c for c in feature_cols if c in merged.columns]
    if join_col != "idx" and "idx" in pool.columns and feature_cols_present:
        fallback = pool.merge(atlas[cols], left_on="idx", right_on="idx", how="left", suffixes=("_label", "_feat"))
        for col in feature_cols_present:
            if col in fallback.columns:
                merged[col] = merged[col].combine_first(fallback[col])
        if "idx_feat" in merged.columns and "idx_feat" in fallback.columns:
            merged["idx_feat"] = merged["idx_feat"].combine_first(fallback["idx_feat"])
    return merged


def _load_pca_proxy_model() -> dict[str, Any]:
    global _PCA_PROXY_MODEL
    if _PCA_PROXY_MODEL is not None:
        return _PCA_PROXY_MODEL
    manifest_path = OUT_ROOT / "nodes" / "N7200_gm_trim_bad" / "node_boundary_manifest.csv"
    if not manifest_path.exists():
        _PCA_PROXY_MODEL = {"features": [], "targets": {}, "note": f"missing manifest: {manifest_path}"}
        return _PCA_PROXY_MODEL
    manifest = pd.read_csv(manifest_path)
    features = [c for c in PCA_PROXY_FEATURES if c in manifest.columns]
    targets = [c for c in PCA_PROXY_TARGETS if c in manifest.columns]
    if not features or not targets:
        _PCA_PROXY_MODEL = {"features": [], "targets": {}, "note": "manifest lacks proxy features or targets"}
        return _PCA_PROXY_MODEL
    x = manifest[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(med)
    mean = x.mean(numeric_only=True)
    std = x.std(ddof=0, numeric_only=True).replace(0, 1.0).fillna(1.0)
    z = ((x - mean) / std).to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(z), dtype=np.float64), z])
    coefs: dict[str, np.ndarray] = {}
    for target in targets:
        y = pd.to_numeric(manifest[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = y.fillna(float(y.median()) if y.notna().any() else 0.0).to_numpy(dtype=np.float64)
        coefs[target] = np.linalg.lstsq(design, y, rcond=None)[0]
    _PCA_PROXY_MODEL = {
        "features": features,
        "targets": coefs,
        "median": med,
        "mean": mean,
        "std": std,
        "note": "linear feature-to-PC proxy fit on N7200 node_boundary_manifest; original BUT not used",
    }
    return _PCA_PROXY_MODEL


def _add_n7200_pca_proxy(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return merged
    model = _load_pca_proxy_model()
    features = list(model.get("features", []))
    targets = dict(model.get("targets", {}))
    if not features or not targets:
        merged = merged.copy()
        merged["pc_proxy_missing"] = True
        return merged
    out = merged.copy()
    x = out.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = model["median"].reindex(features).fillna(0.0)
    mean = model["mean"].reindex(features).fillna(0.0)
    std = model["std"].reindex(features).replace(0, 1.0).fillna(1.0)
    z = ((x.fillna(med) - mean) / std).to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(z), dtype=np.float64), z])
    for target, beta in targets.items():
        proxy = design @ np.asarray(beta, dtype=np.float64)
        out[f"{target}_proxy"] = proxy
        if target not in out.columns:
            out[target] = proxy
        else:
            vals = pd.to_numeric(out[target], errors="coerce")
            out[target] = vals.fillna(pd.Series(proxy, index=out.index))
    out["pc_proxy_missing"] = False
    return out


def _geometry_good_rescue_indices(
    artifact: dict[str, Any], split: str, n: int, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = _feature_merge(artifact, "good", split)
    if n <= 0 or merged.empty:
        return np.array([], dtype=int), pd.DataFrame()
    score = pd.Series(0.0, index=merged.index, dtype=float)
    for col, weight, direction in (
        ("pc1", 0.48, -1.0),
        ("flatline_ratio", 0.42, 1.0),
        ("pc3", 0.18, -1.0),
        ("qrs_visibility", 0.14, 1.0),
        ("template_corr", 0.08, 1.0),
        ("non_qrs_diff_p95", 0.10, -1.0),
        ("fatal_or_score", 0.16, -1.0),
        ("contact_loss_win_ratio", 0.10, -1.0),
    ):
        if col in merged.columns:
            score = score + direction * float(weight) * _robust_z(merged[col])
    merged["geometry_good_rescue_score"] = score
    gate = pd.Series(True, index=merged.index)
    if "pc1" in merged.columns:
        gate &= pd.to_numeric(merged["pc1"], errors="coerce") <= -1.561
    if "flatline_ratio" in merged.columns:
        gate &= pd.to_numeric(merged["flatline_ratio"], errors="coerce") >= 0.1537
    chosen = merged.loc[gate].sort_values("geometry_good_rescue_score", ascending=False).head(int(n))
    if len(chosen) < int(n):
        extra = merged.drop(chosen.index, errors="ignore").sort_values("geometry_good_rescue_score", ascending=False).head(
            int(n) - len(chosen)
        )
        chosen = pd.concat([chosen, extra], ignore_index=False)
    if len(chosen) < int(n):
        extra = merged.sample(n=int(n) - len(chosen), replace=True, random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=False)
    return chosen["row_pos"].to_numpy(dtype=int), chosen.copy()


def _geometry_medium_hard_negative_indices(
    artifact: dict[str, Any], split: str, n: int, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = _feature_merge(artifact, "medium", split)
    if n <= 0 or merged.empty:
        return np.array([], dtype=int), pd.DataFrame()
    score = pd.Series(0.0, index=merged.index, dtype=float)
    for col, weight, direction in (
        ("pc1", 0.40, 1.0),
        ("flatline_ratio", 0.36, -1.0),
        ("pc3", 0.20, 1.0),
        ("non_qrs_diff_p95", 0.24, 1.0),
        ("diff_abs_p95", 0.16, 1.0),
        ("band_15_30", 0.12, 1.0),
        ("band_30_45", 0.10, 1.0),
        ("qrs_visibility", 0.10, 1.0),
        ("qrs_prom_p90", 0.08, 1.0),
        ("template_corr", 0.06, 1.0),
        ("fatal_or_score", 0.10, -1.0),
    ):
        if col in merged.columns:
            score = score + direction * float(weight) * _robust_z(merged[col])
    merged["geometry_medium_hard_negative_score"] = score
    gate = pd.Series(True, index=merged.index)
    if "pc1" in merged.columns:
        gate &= pd.to_numeric(merged["pc1"], errors="coerce") > -1.561
    if "flatline_ratio" in merged.columns:
        gate &= pd.to_numeric(merged["flatline_ratio"], errors="coerce") < 0.1537
    soft_cols = []
    if "qrs_visibility" in merged.columns:
        soft_cols.append(pd.to_numeric(merged["qrs_visibility"], errors="coerce").between(0.25, 0.75))
    if "non_qrs_diff_p95" in merged.columns:
        soft_cols.append(pd.to_numeric(merged["non_qrs_diff_p95"], errors="coerce") >= 0.0792)
    if soft_cols:
        soft_gate = soft_cols[0]
        for col_gate in soft_cols[1:]:
            soft_gate = soft_gate | col_gate
        gate &= soft_gate
    chosen = merged.loc[gate].sort_values("geometry_medium_hard_negative_score", ascending=False).head(int(n))
    if len(chosen) < int(n):
        extra = merged.drop(chosen.index, errors="ignore").sort_values(
            "geometry_medium_hard_negative_score", ascending=False
        ).head(int(n) - len(chosen))
        chosen = pd.concat([chosen, extra], ignore_index=False)
    if len(chosen) < int(n):
        extra = merged.sample(n=int(n) - len(chosen), replace=True, random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=False)
    return chosen["row_pos"].to_numpy(dtype=int), chosen.copy()


def _n7110_qrslow_medium_indices(
    artifact: dict[str, Any], split: str, n: int, seed: int, cfg: dict[str, Any]
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = _add_n7200_pca_proxy(_feature_merge(artifact, "medium", split))
    if n <= 0 or merged.empty:
        return np.array([], dtype=int), pd.DataFrame()
    gate_cfg = dict(cfg.get("qrslow_medium_gate", {}))
    score = pd.Series(0.0, index=merged.index, dtype=float)
    for col, weight, direction in (
        ("qrs_visibility", 0.72, -1.0),
        ("pc3", 0.36, 1.0),
        ("non_qrs_diff_p95", 0.28, 1.0),
        ("diff_abs_p95", 0.20, 1.0),
        ("band_15_30", 0.14, 1.0),
        ("band_30_45", 0.12, 1.0),
        ("pc1", 0.12, 1.0),
        ("flatline_ratio", 0.10, -1.0),
        ("fatal_or_score", 0.22, -1.0),
        ("contact_loss_win_ratio", 0.18, -1.0),
    ):
        if col in merged.columns:
            score = score + direction * float(weight) * _robust_z(merged[col])
    merged["n7110_qrslow_medium_score"] = score
    gate = pd.Series(True, index=merged.index)
    if "qrs_visibility" in merged.columns:
        gate &= pd.to_numeric(merged["qrs_visibility"], errors="coerce") <= float(
            gate_cfg.get("qrs_visibility_max", 0.045)
        )
    if "pc1" in merged.columns:
        gate &= pd.to_numeric(merged["pc1"], errors="coerce") >= float(gate_cfg.get("pc1_min", -1.55))
    if "pc3" in merged.columns:
        gate &= pd.to_numeric(merged["pc3"], errors="coerce") >= float(gate_cfg.get("pc3_min", 1.40))
    if "flatline_ratio" in merged.columns:
        gate &= pd.to_numeric(merged["flatline_ratio"], errors="coerce") <= float(
            gate_cfg.get("flatline_max", 0.24)
        )
    if "non_qrs_diff_p95" in merged.columns:
        gate &= pd.to_numeric(merged["non_qrs_diff_p95"], errors="coerce") >= float(
            gate_cfg.get("non_qrs_diff_min", 0.045)
        )
    if "fatal_or_score" in merged.columns:
        gate &= pd.to_numeric(merged["fatal_or_score"], errors="coerce").fillna(0.0) < float(
            gate_cfg.get("fatal_max", 1.30)
        )
    if "contact_loss_win_ratio" in merged.columns:
        gate &= pd.to_numeric(merged["contact_loss_win_ratio"], errors="coerce").fillna(0.0) < float(
            gate_cfg.get("contact_loss_max", 0.10)
        )
    chosen = merged.loc[gate].sort_values("n7110_qrslow_medium_score", ascending=False).head(int(n))
    if len(chosen) < int(n):
        extra = merged.drop(chosen.index, errors="ignore").sort_values(
            "n7110_qrslow_medium_score", ascending=False
        ).head(int(n) - len(chosen))
        chosen = pd.concat([chosen, extra], ignore_index=False)
    if len(chosen) < int(n):
        extra = merged.sample(n=int(n) - len(chosen), replace=True, random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=False)
    return chosen["row_pos"].to_numpy(dtype=int), chosen.copy()


def _n7200_tri_good_rescue_indices(
    artifact: dict[str, Any], split: str, n: int, seed: int, cfg: dict[str, Any]
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = _add_n7200_pca_proxy(_feature_merge(artifact, "good", split))
    if n <= 0 or merged.empty:
        return np.array([], dtype=int), pd.DataFrame()
    gate_cfg = dict(cfg.get("tri_good_gate", {}))
    score = pd.Series(0.0, index=merged.index, dtype=float)
    for col, weight, direction in (
        ("pc1", 0.62, -1.0),
        ("flatline_ratio", 0.52, 1.0),
        ("pc3", 0.32, -1.0),
        ("qrs_visibility", 0.28, 1.0),
        ("non_qrs_diff_p95", 0.28, -1.0),
        ("template_corr", 0.10, 1.0),
        ("fatal_or_score", 0.20, -1.0),
        ("contact_loss_win_ratio", 0.14, -1.0),
    ):
        if col in merged.columns:
            score = score + direction * float(weight) * _robust_z(merged[col])
    merged["n7200_tri_good_rescue_score"] = score
    gate = pd.Series(True, index=merged.index)
    if "pc1" in merged.columns:
        gate &= pd.to_numeric(merged["pc1"], errors="coerce") <= float(gate_cfg.get("pc1_max", -3.60))
    if "flatline_ratio" in merged.columns:
        gate &= pd.to_numeric(merged["flatline_ratio"], errors="coerce") >= float(gate_cfg.get("flatline_min", 0.20))
    if "pc3" in merged.columns:
        gate &= pd.to_numeric(merged["pc3"], errors="coerce") <= float(gate_cfg.get("pc3_max", 0.40))
    if "qrs_visibility" in merged.columns:
        gate &= pd.to_numeric(merged["qrs_visibility"], errors="coerce") >= float(
            gate_cfg.get("qrs_visibility_min", 0.45)
        )
    if "non_qrs_diff_p95" in merged.columns:
        gate &= pd.to_numeric(merged["non_qrs_diff_p95"], errors="coerce") <= float(
            gate_cfg.get("non_qrs_diff_max", 0.085)
        )
    chosen = merged.loc[gate].sort_values("n7200_tri_good_rescue_score", ascending=False).head(int(n))
    if len(chosen) < int(n):
        extra = merged.drop(chosen.index, errors="ignore").sort_values(
            "n7200_tri_good_rescue_score", ascending=False
        ).head(int(n) - len(chosen))
        chosen = pd.concat([chosen, extra], ignore_index=False)
    if len(chosen) < int(n):
        extra = merged.sample(n=int(n) - len(chosen), replace=True, random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=False)
    return chosen["row_pos"].to_numpy(dtype=int), chosen.copy()


def _n7200_tri_medium_hard_negative_indices(
    artifact: dict[str, Any], split: str, n: int, seed: int, cfg: dict[str, Any]
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = _add_n7200_pca_proxy(_feature_merge(artifact, "medium", split))
    if n <= 0 or merged.empty:
        return np.array([], dtype=int), pd.DataFrame()
    gate_cfg = dict(cfg.get("tri_medium_gate", {}))
    score = pd.Series(0.0, index=merged.index, dtype=float)
    for col, weight, direction in (
        ("pc1", 0.60, 1.0),
        ("flatline_ratio", 0.50, -1.0),
        ("pc3", 0.42, 1.0),
        ("non_qrs_diff_p95", 0.42, 1.0),
        ("diff_abs_p95", 0.22, 1.0),
        ("band_15_30", 0.16, 1.0),
        ("band_30_45", 0.14, 1.0),
        ("qrs_visibility", 0.18, -1.0),
        ("qrs_prom_p90", 0.08, 1.0),
        ("fatal_or_score", 0.14, -1.0),
        ("contact_loss_win_ratio", 0.10, -1.0),
    ):
        if col in merged.columns:
            score = score + direction * float(weight) * _robust_z(merged[col])
    merged["n7200_tri_medium_hard_negative_score"] = score
    gate = pd.Series(True, index=merged.index)
    if "pc1" in merged.columns:
        gate &= pd.to_numeric(merged["pc1"], errors="coerce") >= float(gate_cfg.get("pc1_min", -1.20))
    if "flatline_ratio" in merged.columns:
        gate &= pd.to_numeric(merged["flatline_ratio"], errors="coerce") <= float(
            gate_cfg.get("flatline_max", 0.14)
        )
    if "pc3" in merged.columns:
        gate &= pd.to_numeric(merged["pc3"], errors="coerce") >= float(gate_cfg.get("pc3_min", 1.40))
    if "non_qrs_diff_p95" in merged.columns:
        gate &= pd.to_numeric(merged["non_qrs_diff_p95"], errors="coerce") >= float(
            gate_cfg.get("non_qrs_diff_min", 0.085)
        )
    if "qrs_visibility" in merged.columns:
        qrs = pd.to_numeric(merged["qrs_visibility"], errors="coerce")
        gate &= qrs.between(
            float(gate_cfg.get("qrs_visibility_min", 0.18)),
            float(gate_cfg.get("qrs_visibility_max", 0.58)),
        )
    chosen = merged.loc[gate].sort_values("n7200_tri_medium_hard_negative_score", ascending=False).head(int(n))
    if len(chosen) < int(n):
        extra = merged.drop(chosen.index, errors="ignore").sort_values(
            "n7200_tri_medium_hard_negative_score", ascending=False
        ).head(int(n) - len(chosen))
        chosen = pd.concat([chosen, extra], ignore_index=False)
    if len(chosen) < int(n):
        extra = merged.sample(n=int(n) - len(chosen), replace=True, random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=False)
    return chosen["row_pos"].to_numpy(dtype=int), chosen.copy()


def _n7200_bad_stress_indices(
    artifact: dict[str, Any], split: str, n: int, seed: int, cfg: dict[str, Any]
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = _add_n7200_pca_proxy(_feature_merge(artifact, "bad", split))
    if n <= 0 or merged.empty:
        return np.array([], dtype=int), pd.DataFrame()
    gate_cfg = dict(cfg.get("badstress_gate", {}))
    mode = str(cfg.get("badstress_mode", "controlled_qrsband"))
    score = pd.Series(0.0, index=merged.index, dtype=float)
    if mode == "moderate_baseline_hf":
        for col, weight, direction in (
            ("baseline_step", 0.60, 1.0),
            ("band_30_45", 0.48, 1.0),
            ("non_qrs_diff_p95", 0.26, 1.0),
            ("diff_abs_p95", 0.22, 1.0),
            ("qrs_band_ratio", 0.28, -1.0),
            ("qrs_visibility", 0.16, -1.0),
            ("pc4", 0.16, -1.0),
            ("fatal_or_score", 0.18, -1.0),
            ("contact_loss_win_ratio", 0.14, -1.0),
        ):
            if col in merged.columns:
                score = score + direction * float(weight) * _robust_z(merged[col])
    else:
        for col, weight, direction in (
            ("qrs_band_ratio", 0.58, 1.0),
            ("band_15_30", 0.44, 1.0),
            ("boundary_confidence", 0.30, 1.0),
            ("pca_margin", 0.18, 1.0),
            ("pc1", 0.18, 1.0),
            ("baseline_step", 0.20, -1.0),
            ("band_30_45", 0.10, 1.0),
            ("fatal_or_score", 0.18, -1.0),
            ("contact_loss_win_ratio", 0.14, -1.0),
        ):
            if col in merged.columns:
                score = score + direction * float(weight) * _robust_z(merged[col])
    merged["n7200_bad_stress_score"] = score
    gate = pd.Series(True, index=merged.index)

    def apply_min(col: str, key: str) -> None:
        nonlocal gate
        if col in merged.columns and key in gate_cfg:
            gate &= pd.to_numeric(merged[col], errors="coerce") >= float(gate_cfg[key])

    def apply_max(col: str, key: str) -> None:
        nonlocal gate
        if col in merged.columns and key in gate_cfg:
            gate &= pd.to_numeric(merged[col], errors="coerce") <= float(gate_cfg[key])

    apply_min("qrs_band_ratio", "qrs_band_ratio_min")
    apply_max("qrs_band_ratio", "qrs_band_ratio_max")
    apply_min("band_15_30", "band_15_30_min")
    apply_min("band_30_45", "band_30_45_min")
    apply_max("band_30_45", "band_30_45_max")
    apply_min("baseline_step", "baseline_step_min")
    apply_max("baseline_step", "baseline_step_max")
    apply_max("fatal_or_score", "fatal_max")
    apply_max("contact_loss_win_ratio", "contact_loss_max")
    chosen = merged.loc[gate].sort_values("n7200_bad_stress_score", ascending=False).head(int(n))
    if len(chosen) < int(n):
        extra = merged.drop(chosen.index, errors="ignore").sort_values(
            "n7200_bad_stress_score", ascending=False
        ).head(int(n) - len(chosen))
        chosen = pd.concat([chosen, extra], ignore_index=False)
    if len(chosen) < int(n):
        extra = merged.sample(n=int(n) - len(chosen), replace=True, random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=False)
    return chosen["row_pos"].to_numpy(dtype=int), chosen.copy()


def _load_n7100_added_ring_rows(path: Path | str | None = None) -> pd.DataFrame:
    global _N7100_ADDED_RING_ROWS
    rows_path = Path(path) if path is not None else N7100_ADDED_RING_ROWS_PATH
    cache_key = str(rows_path.resolve()) if rows_path.exists() else str(rows_path)
    if cache_key in _ADDED_RING_ROWS_CACHE:
        return _ADDED_RING_ROWS_CACHE[cache_key]
    if path is None and _N7100_ADDED_RING_ROWS is not None:
        return _N7100_ADDED_RING_ROWS
    if not rows_path.exists():
        df = pd.DataFrame()
        _ADDED_RING_ROWS_CACHE[cache_key] = df
        if path is None:
            _N7100_ADDED_RING_ROWS = df
        return df
    df = pd.read_csv(rows_path)
    if "slice" in df.columns:
        df = df.loc[df["slice"].astype(str).eq("added")].copy()
    df = df.reset_index(drop=True)
    _ADDED_RING_ROWS_CACHE[cache_key] = df
    if path is None:
        _N7100_ADDED_RING_ROWS = df
    return df


def _n7100_added_ring_targets(class_name: str, split: str, path: Path | str | None = None) -> pd.DataFrame:
    rows = _load_n7100_added_ring_rows(path)
    if rows.empty:
        return rows
    sub = rows.loc[rows["class_name"].astype(str).eq(class_name)].copy()
    if sub.empty or "split" not in sub.columns:
        return sub
    split_sub = sub.loc[sub["split"].astype(str).eq(str(split))].copy()
    return split_sub if not split_sub.empty else sub


def _n7100_added_ring_feature_weights(class_name: str) -> dict[str, float]:
    if class_name == "good":
        return {
            "pc1": 0.16,
            "pc3": 0.24,
            "pc4": 0.14,
            "flatline_ratio": 0.22,
            "qrs_visibility": 0.26,
            "detector_agreement": 0.20,
            "non_qrs_rms_ratio": 0.18,
            "diff_abs_p95": 0.18,
            "mean_abs": 0.16,
            "rms": 0.14,
            "std": 0.14,
            "baseline_step": 0.12,
            "amplitude_entropy": 0.12,
            "sqi_kSQI": 0.10,
        }
    return {
        "pc1": 0.30,
        "pc3": 0.24,
        "flatline_ratio": 0.18,
        "rms": 0.24,
        "std": 0.24,
        "ptp_p99_p01": 0.22,
        "amplitude_entropy": 0.22,
        "non_qrs_rms_ratio": 0.20,
        "qrs_visibility": 0.18,
        "diff_abs_p95": 0.18,
        "non_qrs_diff_p95": 0.18,
        "sqi_kSQI": 0.16,
        "qrs_band_ratio": 0.12,
    }


def _n7100_added_ring_indices(
    artifact: dict[str, Any],
    class_name: str,
    split: str,
    n: int,
    seed: int,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = _add_n7200_pca_proxy(_feature_merge(artifact, class_name, split))
    target_path = cfg.get("added_ring_rows_path", N7100_ADDED_RING_ROWS_PATH)
    targets = _n7100_added_ring_targets(class_name, split, target_path)
    if n <= 0 or merged.empty:
        return np.array([], dtype=int), pd.DataFrame()
    if targets.empty:
        if class_name == "good":
            return _geometry_good_rescue_indices(artifact, split, n, seed)
        return _geometry_medium_hard_negative_indices(artifact, split, n, seed)
    weights = _n7100_added_ring_feature_weights(class_name)
    feature_cols = [
        col
        for col in weights
        if col in merged.columns and col in targets.columns and pd.to_numeric(targets[col], errors="coerce").notna().any()
    ]
    score = pd.Series(0.0, index=merged.index, dtype=float)
    used_features: list[str] = []
    for col in feature_cols:
        target_vals = pd.to_numeric(targets[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        source_vals = pd.to_numeric(merged[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if target_vals.empty or source_vals.notna().sum() == 0:
            continue
        center = float(target_vals.median())
        scale = float(target_vals.quantile(0.75) - target_vals.quantile(0.25))
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = float(target_vals.std(ddof=0))
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = float(source_vals.std(skipna=True) or 1.0)
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        score = score + float(weights[col]) * ((source_vals - center).abs() / scale).fillna(3.0)
        used_features.append(col)
    if not used_features:
        if class_name == "good":
            return _geometry_good_rescue_indices(artifact, split, n, seed)
        return _geometry_medium_hard_negative_indices(artifact, split, n, seed)
    merged["n7100_added_ring_distance"] = score
    merged["n7100_added_ring_used_features"] = ",".join(used_features)
    gate = pd.Series(True, index=merged.index)
    if class_name == "good":
        if "fatal_or_score" in merged.columns:
            gate &= pd.to_numeric(merged["fatal_or_score"], errors="coerce").fillna(0.0) < 1.35
        if "contact_loss_win_ratio" in merged.columns:
            gate &= pd.to_numeric(merged["contact_loss_win_ratio"], errors="coerce").fillna(0.0) < 0.08
    else:
        if "fatal_or_score" in merged.columns:
            gate &= pd.to_numeric(merged["fatal_or_score"], errors="coerce").fillna(0.0) < 1.45
        if "contact_loss_win_ratio" in merged.columns:
            gate &= pd.to_numeric(merged["contact_loss_win_ratio"], errors="coerce").fillna(0.0) < 0.12
    chosen = merged.loc[gate].sort_values("n7100_added_ring_distance", ascending=True).head(int(n))
    if len(chosen) < int(n):
        extra = (
            merged.drop(chosen.index, errors="ignore")
            .sort_values("n7100_added_ring_distance", ascending=True)
            .head(int(n) - len(chosen))
        )
        chosen = pd.concat([chosen, extra], ignore_index=False)
    if len(chosen) < int(n):
        extra = merged.sample(n=int(n) - len(chosen), replace=True, random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=False)
    return chosen["row_pos"].to_numpy(dtype=int), chosen.copy()


def _append_rows(
    rows: list[pd.Series],
    arrays: dict[str, list[np.ndarray]],
    atlas_rows: list[pd.Series],
    artifact: dict[str, Any],
    indices: np.ndarray,
    *,
    source_label: str,
    sample_weight: float,
) -> None:
    runner._append_blend_rows(
        rows,
        arrays,
        atlas_rows,
        artifact,
        indices,
        source_label=source_label,
        sample_weight=float(sample_weight),
    )


def _write_geometry_artifact(level: int, cfg: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    trim_id = configure_level(level)
    spec_id = targeted.stable_id(f"nl_{trim_id}_{cfg['name']}_seed{int(cfg['seed'])}")
    variant_dir = OUT_ROOT / "synthetic_variants" / spec_id
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    audit_path = variant_dir / "data_variant_audit.json"
    if labels_path.exists():
        audit = ladder.read_json(audit_path) if audit_path.exists() else {}
        return spec_id, variant_dir, audit

    base_variant_id = str(cfg.get("base_variant_id", BASE_PROMOTED_VARIANT))
    source_variant_ids = tuple(str(v) for v in cfg.get("source_variant_ids", SOURCE_VARIANTS))
    base = _artifact(base_variant_id)
    sources = [_artifact(v) for v in source_variant_ids]
    rows: list[pd.Series] = []
    atlas_rows: list[pd.Series] = []
    arrays: dict[str, list[np.ndarray]] = {"clean": [], "noisy": []}
    base_source_label = f"base_{base_variant_id}_full_body"
    _append_rows(
        rows,
        arrays,
        atlas_rows,
        base,
        np.arange(len(base["labels"]), dtype=int),
        source_label=base_source_label,
        sample_weight=1.0,
    )

    selection_rows: list[pd.DataFrame] = []
    composition: list[dict[str, Any]] = []
    seed = int(cfg["seed"])
    profile = str(cfg.get("geometry_profile", "pc1flat"))
    if profile == "boundary_blocks":
        class_specs = [
            (
                str(block["class_name"]),
                float(block.get("frac", 0.0)),
                float(block.get("weight", 1.0)),
                str(block["block"]),
                str(block["picker"]),
            )
            for block in cfg.get("boundary_blocks", ())
        ]
    else:
        class_specs = [
            (
                "good",
                float(cfg.get("good_aux_frac", 0.0)),
                float(cfg.get("good_aux_weight", 1.0)),
                "aux_geometry_good_rescue_pc1low_flatlinehigh",
                "default_good",
            ),
            (
                "medium",
                float(cfg.get("medium_aux_frac", 0.0)),
                float(cfg.get("medium_aux_weight", 1.0)),
                "aux_geometry_medium_hard_negative_visibleqrs",
                "default_medium",
            ),
        ]
        if float(cfg.get("bad_aux_frac", 0.0)) > 0.0:
            class_specs.append(
                (
                    "bad",
                    float(cfg.get("bad_aux_frac", 0.0)),
                    float(cfg.get("bad_aux_weight", 1.0)),
                    "aux_n7200_badstress_controlled_boundary",
                    "bad_controlled_outlier",
                )
            )
    for class_name, frac, weight, source_label, picker_name in class_specs:
        if profile == "boundary_blocks" and picker_name == "good_rescue_pc1flat_qrsvisible":
            picker = lambda artifact, split, n, run_seed: _n7200_tri_good_rescue_indices(
                artifact, split, n, run_seed, cfg
            )
        elif profile == "boundary_blocks" and picker_name == "medium_qrslow_hardneg":
            picker = lambda artifact, split, n, run_seed: _n7110_qrslow_medium_indices(
                artifact, split, n, run_seed, cfg
            )
        elif profile == "boundary_blocks" and picker_name == "medium_visible_qrs_detail":
            picker = lambda artifact, split, n, run_seed: _n7200_tri_medium_hard_negative_indices(
                artifact, split, n, run_seed, cfg
            )
        elif profile == "boundary_blocks" and picker_name == "bad_controlled_outlier":
            picker = lambda artifact, split, n, run_seed: _n7200_bad_stress_indices(
                artifact, split, n, run_seed, cfg
            )
        elif profile == "n7200_tri_band" and class_name == "good":
            picker = lambda artifact, split, n, run_seed: _n7200_tri_good_rescue_indices(
                artifact, split, n, run_seed, cfg
            )
            source_label = "aux_n7200_tri_good_extreme_pc1flat"
        elif profile == "n7200_tri_band" and class_name == "medium":
            picker = lambda artifact, split, n, run_seed: _n7200_tri_medium_hard_negative_indices(
                artifact, split, n, run_seed, cfg
            )
            source_label = "aux_n7200_tri_medium_pc3detail_hardneg"
        elif profile == "n7100_added_ring":
            picker = lambda artifact, split, n, run_seed, cls=class_name: _n7100_added_ring_indices(
                artifact, cls, split, n, run_seed, cfg
            )
            source_label = f"aux_n7100_added_ring_{class_name}"
        elif profile == "n7110_qrslow_medium" and class_name == "medium":
            picker = lambda artifact, split, n, run_seed: _n7110_qrslow_medium_indices(
                artifact, split, n, run_seed, cfg
            )
            source_label = "aux_n7110_qrslow_medium_hard_negative"
        elif profile == "n7200_badstress" and class_name == "good":
            picker = lambda artifact, split, n, run_seed: _n7200_tri_good_rescue_indices(
                artifact, split, n, run_seed, cfg
            )
            source_label = "aux_n7200_badstress_tri_good_guard"
        elif profile == "n7200_badstress" and class_name == "medium":
            picker = lambda artifact, split, n, run_seed: _n7200_tri_medium_hard_negative_indices(
                artifact, split, n, run_seed, cfg
            )
            source_label = "aux_n7200_badstress_tri_medium_guard"
        elif profile == "n7200_badstress" and class_name == "bad":
            picker = lambda artifact, split, n, run_seed: _n7200_bad_stress_indices(
                artifact, split, n, run_seed, cfg
            )
            source_label = f"aux_n7200_badstress_{cfg.get('badstress_mode', 'controlled')}"
        elif class_name == "good":
            picker = _geometry_good_rescue_indices
        else:
            picker = _geometry_medium_hard_negative_indices
        split_counts = _class_split_counts(base["labels"], class_name, frac)
        for split, total_n in split_counts.items():
            counts = _source_counts(total_n, source_variant_ids)
            for src_i, (source_id, artifact, src_n) in enumerate(zip(source_variant_ids, sources, counts)):
                idx, selected = picker(artifact, split, int(src_n), seed + src_i + (0 if class_name == "good" else 97))
                _append_rows(rows, arrays, atlas_rows, artifact, idx, source_label=source_label, sample_weight=weight)
                if not selected.empty:
                    selected = selected.copy()
                    selected["selected_class"] = class_name
                    selected["selected_split"] = split
                    selected["aux_variant_id"] = source_id
                    selected["source_label"] = source_label
                    selection_rows.append(selected)
                composition.append(
                    {
                        "node_id": trim_id,
                        "split": split,
                        "class_name": class_name,
                        "source": source_label,
                        "aux_variant_id": source_id,
                        "n": int(len(idx)),
                    }
                )

    data_dir = variant_dir / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame(rows).reset_index(drop=True)
    labels["pre_geometry_idx"] = labels["idx"].to_numpy(dtype=np.int64, copy=True)
    labels["idx"] = np.arange(len(labels), dtype=int)
    labels["y"] = labels["y_class"].map({"good": 0, "medium": 1, "bad": 2}).astype(int)
    clean = np.stack(arrays["clean"]).astype(np.float32)
    noisy = np.stack(arrays["noisy"]).astype(np.float32)
    np.savez_compressed(data_dir / "synth_10s_125hz_clean.npz", X_clean=clean)
    np.savez_compressed(data_dir / "synth_10s_125hz_noisy.npz", X_noisy=noisy)
    np.savez_compressed(data_dir / "signals.npz", X=noisy[:, None, :].astype(np.float32))
    mask_payload = {
        key.split(":", 1)[1]: np.stack(value).astype(np.float32)
        for key, value in arrays.items()
        if key.startswith("mask:")
    }
    np.savez_compressed(data_dir / "synth_10s_125hz_local_mask.npz", **mask_payload)
    level_values = pd.to_numeric(
        labels.get("measured_snr_db", labels.get("measured_snr_all_noise_db", 0.0)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)
    np.savez_compressed(data_dir / "synth_10s_125hz_noise_level.npz", level=level_values)
    labels.to_csv(data_dir / "synth_10s_125hz_labels_with_level.csv", index=False)
    labels.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)

    if len(atlas_rows) == len(labels):
        atlas = pd.DataFrame(atlas_rows).reset_index(drop=True)
        atlas["idx"] = np.arange(len(atlas), dtype=int)
        atlas["split"] = labels["split"].astype(str).to_numpy()
        atlas["class_name"] = labels["y_class"].astype(str).to_numpy()
        atlas["y"] = labels["y"].astype(int).to_numpy()
        atlas.to_csv(variant_dir / "synthetic_atlas64_features.csv", index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_csv(variant_dir / "geometry_selected_aux_rows.csv", index=False)
    pd.DataFrame(composition).to_csv(variant_dir / "geometry_composition.csv", index=False)
    audit = {
        "created_at": now_iso(),
        "spec_id": spec_id,
        "node_id": trim_id,
        "node_level": int(level),
        "purpose": (
            "Good/medium geometry repair: preserve promoted N6800 full body, add paired pc1-low/flatline-high "
            "good rescue rows and visible-QRS medium hard negatives. Bad remains trim-bad guardrail."
        ),
        "base_variant_id": base_variant_id,
        "source_variant_ids": list(source_variant_ids),
        "config": cfg,
        "rows": int(len(labels)),
        "class_counts": labels["y_class"].astype(str).value_counts().sort_index().to_dict(),
        "split_counts": labels["split"].astype(str).value_counts().sort_index().to_dict(),
        "composition": composition,
        "selection_note": "Clean/SemiClean/node diagnostics only. Original BUT is report-only and bucketed.",
    }
    write_json(audit_path, audit)
    return spec_id, variant_dir, audit


def _threshold_candidates(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    rescue = train_df["disagreement_group"].astype(str).eq("good_rescued_by_dense")
    lost = train_df["disagreement_group"].astype(str).eq("medium_lost_by_dense")
    for feature in GATE_FEATURES:
        if feature not in train_df.columns:
            continue
        train_vals = pd.to_numeric(train_df[feature], errors="coerce")
        r_vals = train_vals[rescue].dropna()
        l_vals = train_vals[lost].dropna()
        if r_vals.empty or l_vals.empty:
            continue
        # Direction chooses the side that captures more rescued-good than medium-lost rows.
        candidates: list[tuple[str, float, float]] = []
        for q in np.linspace(0.05, 0.95, 91):
            thr = float(r_vals.quantile(float(q)))
            for op in ("<=", ">="):
                r_rate = float((r_vals <= thr).mean() if op == "<=" else (r_vals >= thr).mean())
                l_rate = float((l_vals <= thr).mean() if op == "<=" else (l_vals >= thr).mean())
                candidates.append((op, thr, r_rate - l_rate))
        op, thr, spread = max(candidates, key=lambda x: x[2])
        rows.append(
            {
                "feature": feature,
                "op": op,
                "threshold": thr,
                "train_rescue_pass_rate": float((r_vals <= thr).mean() if op == "<=" else (r_vals >= thr).mean()),
                "train_medium_lost_pass_rate": float((l_vals <= thr).mean() if op == "<=" else (l_vals >= thr).mean()),
                "train_spread": float(spread),
            }
        )
    cand_df = pd.DataFrame(rows).sort_values("train_spread", ascending=False)

    if test_df.empty:
        return cand_df, pd.DataFrame()
    default_pred = test_df["medium_strong_pred"].to_numpy(dtype=int)
    alt_pred = test_df["good_strong_pred"].to_numpy(dtype=int)
    y = test_df["y"].to_numpy(dtype=int)
    configs = []
    pc1 = cand_df.loc[cand_df["feature"].eq("pc1")].head(1)
    flat = cand_df.loc[cand_df["feature"].eq("flatline_ratio")].head(1)
    pc3 = cand_df.loc[cand_df["feature"].eq("pc3")].head(1)
    if not pc1.empty:
        configs.append(("pc1_only", [pc1.iloc[0].to_dict()]))
    if not pc1.empty and not flat.empty:
        configs.append(("pc1_flatline", [pc1.iloc[0].to_dict(), flat.iloc[0].to_dict()]))
    if not pc1.empty and not flat.empty and not pc3.empty:
        configs.append(("pc1_flatline_pc3", [pc1.iloc[0].to_dict(), flat.iloc[0].to_dict(), pc3.iloc[0].to_dict()]))
    for name, rules in configs:
        gate = test_df["medium_strong_pred"].ne(test_df["good_strong_pred"])
        for rule in rules:
            vals = pd.to_numeric(test_df[rule["feature"]], errors="coerce")
            if rule["op"] == "<=":
                gate &= vals <= float(rule["threshold"])
            else:
                gate &= vals >= float(rule["threshold"])
        pred = default_pred.copy()
        pred[gate.to_numpy(dtype=bool)] = alt_pred[gate.to_numpy(dtype=bool)]
        rep = _report(y, pred)
        eval_rows.append(
            {
                "gate_name": name,
                "split": "test",
                "gate_flips": int(gate.sum()),
                "acc": rep["acc"],
                "macro_f1": rep["macro_f1"],
                "good_recall": rep["good_recall"],
                "medium_recall": rep["medium_recall"],
                "bad_recall": rep["bad_recall"],
                "rules_json": json.dumps(rules, ensure_ascii=False, default=json_default),
            }
        )
    return cand_df, pd.DataFrame(eval_rows)


def _report(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in (0, 1, 2):
        tp = int(((y_true == cls) & (pred == cls)).sum())
        fp = int(((y_true != cls) & (pred == cls)).sum())
        fn = int(((y_true == cls) & (pred != cls)).sum())
        rec = tp / (tp + fn) if (tp + fn) else math.nan
        prec = tp / (tp + fp) if (tp + fp) else math.nan
        f1 = (2 * prec * rec / (prec + rec)) if math.isfinite(prec) and math.isfinite(rec) and (prec + rec) else math.nan
        recalls.append(rec)
        f1s.append(f1)
    return {
        "acc": float((y_true == pred).mean()) if len(y_true) else math.nan,
        "macro_f1": float(np.nanmean(f1s)),
        "good_recall": float(recalls[0]),
        "medium_recall": float(recalls[1]),
        "bad_recall": float(recalls[2]),
    }


def geometry_gate_analysis() -> dict[str, Any]:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORT.mkdir(parents=True, exist_ok=True)
    rows_path = DEEP_DIVE_OUT / "n7000_gm_focus_disagreement_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    df = pd.read_csv(rows_path)
    focus = df.loc[df["disagreement_group"].astype(str).isin(["good_rescued_by_dense", "medium_lost_by_dense"])].copy()
    train_df = focus.loc[focus["split"].astype(str).isin(["train", "val"])].copy()
    test_df = df.loc[df["split"].astype(str).eq("test")].copy()
    cand_df, eval_df = _threshold_candidates(train_df, test_df)
    cand_df.to_csv(ANALYSIS_OUT / "heldout_gate_thresholds.csv", index=False)
    cand_df.to_csv(ANALYSIS_REPORT / "heldout_gate_thresholds.csv", index=False)
    eval_df.to_csv(ANALYSIS_OUT / "heldout_gate_eval.csv", index=False)
    eval_df.to_csv(ANALYSIS_REPORT / "heldout_gate_eval.csv", index=False)
    summary = {
        "created_at": now_iso(),
        "focus_rows": int(len(focus)),
        "train_val_focus_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "top_thresholds": cand_df.head(8).to_dict("records"),
        "heldout_eval": eval_df.to_dict("records"),
        "note": "Thresholds estimated on train/val disagreement rows and evaluated on held-out test rows.",
    }
    write_json(ANALYSIS_OUT / "geometry_gate_analysis_summary.json", summary)
    write_json(ANALYSIS_REPORT / "geometry_gate_analysis_summary.json", summary)
    _write_analysis_report(summary)
    return summary


def _write_analysis_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Good/Medium Geometry Repair",
        "",
        "## Held-Out Gate Check",
        "",
        f"- train/val focus rows: {summary['train_val_focus_rows']}",
        f"- held-out test rows: {summary['test_rows']}",
        "- Thresholds are estimated on train/val only; original BUT is not used for selection.",
        "",
        "## Top Thresholds",
        "",
    ]
    for row in summary.get("top_thresholds", [])[:6]:
        lines.append(
            f"- `{row['feature']} {row['op']} {float(row['threshold']):.6g}`: "
            f"rescue pass {float(row['train_rescue_pass_rate']):.3f}, "
            f"medium-lost pass {float(row['train_medium_lost_pass_rate']):.3f}, "
            f"spread {float(row['train_spread']):.3f}"
        )
    lines += ["", "## Held-Out Simulations", ""]
    for row in summary.get("heldout_eval", []):
        lines.append(
            f"- `{row['gate_name']}`: acc {float(row['acc']):.4f}, macro-F1 {float(row['macro_f1']):.4f}, "
            f"recall good/medium/bad {float(row['good_recall']):.4f}/"
            f"{float(row['medium_recall']):.4f}/{float(row['bad_recall']):.4f}, flips {int(row['gate_flips'])}"
        )
    lines += [
        "",
        "## Generator Translation",
        "",
        "- Good aux: pc1-low + flatline-high rescue band, with guardrails against fatal/contact-like bad.",
        "- Medium aux: visible-QRS hard negatives outside the rescue band, with local detail/non-QRS derivative preserved.",
        "- Bad remains trim-bad core/right-island/near-boundary guardrail.",
    ]
    for root in (ANALYSIS_OUT, ANALYSIS_REPORT):
        (root / "good_medium_geometry_repair_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def geometry_gate_build(levels: list[int]) -> dict[str, Any]:
    built: list[dict[str, Any]] = []
    for level in levels:
        args = make_args(level, "report", run_training=False)
        runner.build_trim_node(args, force=False)
        for cfg in configs_for_level(level):
            spec_id, variant_dir, audit = _write_geometry_artifact(level, cfg)
            built.append({"level": int(level), "node_id": f"N{int(level)}_gm_trim_bad", "spec_id": spec_id, "variant_dir": str(variant_dir), "audit": audit})
    payload = {"created_at": now_iso(), "built": built}
    write_json(ANALYSIS_OUT / "geometry_gate_build_summary.json", payload)
    write_json(ANALYSIS_REPORT / "geometry_gate_build_summary.json", payload)
    return payload


def _training_row(level: int, spec_id: str, variant_dir: Path, cfg: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    node_id = f"N{int(level)}_gm_trim_bad"
    base_variant_id = str(cfg.get("base_variant_id", BASE_PROMOTED_VARIANT))
    return {
        "variant_id": spec_id,
        "variant_dir": str(variant_dir),
        "node_id": node_id,
        "node_level": int(level),
        "score": math.nan,
        "node_score": math.nan,
        "spec": {
            "id": spec_id,
            "base_variant_id": base_variant_id,
            "node_id": node_id,
            "node_role": "gm_geometry_repair_trim_bad",
            "class_weight": str(cfg["class_weight"]),
            "cls_loss_variant": str(cfg["loss"]),
            "geometry_gate_config": {k: v for k, v in cfg.items() if k not in {"loss", "class_weight"}},
            "refine_note": (
                "Geometry-guided paired good/medium repair. Good rescue is pc1-low/flatline-high; medium hard negatives "
                "stay outside that band. Bad remains trim-bad guardrail. Original BUT is report-only."
            ),
        },
        "geometry_gate_audit": audit,
    }


def geometry_gate_quick(levels: list[int]) -> dict[str, Any]:
    trained: list[dict[str, Any]] = []
    for level in levels:
        node_id = configure_level(level)
        args = make_args(level, "quick", run_training=True)
        runner.build_trim_node(args, force=False)
        existing_ids = {
            str(r.get("spec", {}).get("id", ""))
            for r in ladder.load_jsonl(OUT_ROOT / ladder.SUMMARY_NAME)
            if int(r.get("returncode", 1) or 1) == 0
        }
        update_state(
            OUT_ROOT / ladder.STATE_NAME,
            status="geometry_gate_quick_training",
            stage="geometry_gate_quick",
            node_id=node_id,
            total=len(configs_for_level(level)),
            completed=0,
            updated_at=now_iso(),
        )
        level_configs = configs_for_level(level)
        for done, cfg in enumerate(level_configs):
            spec_id, variant_dir, audit = _write_geometry_artifact(level, cfg)
            if spec_id in existing_ids:
                trained.append({"level": int(level), "spec_id": spec_id, "skipped": True, "reason": "existing_success"})
                continue
            train_row = _training_row(level, spec_id, variant_dir, cfg, audit)
            old_loss = args.cls_loss_variant
            old_seed = args.seed
            old_e1 = args.quick_epochs_stage1
            old_e2 = args.quick_epochs_stage2
            old_regen = args.train_regenerate_full
            args.cls_loss_variant = str(cfg["loss"])
            args.seed = int(cfg["seed"])
            args.quick_epochs_stage1 = int(cfg["epochs"][0])
            args.quick_epochs_stage2 = int(cfg["epochs"][1])
            args.train_regenerate_full = 0
            update_state(
                OUT_ROOT / ladder.STATE_NAME,
                status="geometry_gate_quick_training",
                stage="geometry_gate_quick",
                node_id=node_id,
                current=spec_id,
                completed=done,
                total=len(level_configs),
                updated_at=now_iso(),
            )
            try:
                result = targeted.train_one(args, train_row, "quick")
            finally:
                args.cls_loss_variant = old_loss
                args.seed = old_seed
                args.quick_epochs_stage1 = old_e1
                args.quick_epochs_stage2 = old_e2
                args.train_regenerate_full = old_regen
            result["node_id"] = node_id
            result["node_level"] = int(level)
            result["base_variant_id"] = str(cfg.get("base_variant_id", BASE_PROMOTED_VARIANT))
            result["cls_loss_variant"] = str(cfg["loss"])
            result["geometry_gate_config"] = cfg
            result["geometry_gate_audit"] = audit
            ladder.append_jsonl(OUT_ROOT / ladder.SUMMARY_NAME, result)
            ladder.append_jsonl(ladder.node_dir(args, node_id) / ladder.SUMMARY_NAME, result)
            trained.append(result)
            if int(result.get("returncode", 1) or 1) == 0:
                existing_ids.add(spec_id)
        update_state(
            OUT_ROOT / ladder.STATE_NAME,
            status="geometry_gate_quick_complete",
            stage="geometry_gate_quick",
            node_id=node_id,
            completed=len(level_configs),
            total=len(level_configs),
            updated_at=now_iso(),
        )
        ladder.write_report(args)
    payload = {"created_at": now_iso(), "trained": trained}
    write_json(ANALYSIS_OUT / "geometry_gate_quick_summary.json", payload)
    return payload


def geometry_gate_diagnostic_promote(levels: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for level in levels:
        node_id = configure_level(level)
        args = make_args(level, "diagnostic", run_training=False)
        metrics = runner.diagnostic_with_medium_guard(args)
        promo = ladder.promote_nodes(args)
        runner.helper.sync_registry_csv_from_json()
        metrics_path = OUT_ROOT / ladder.DIAGNOSTIC_NAME
        best = {}
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            sub = df.loc[df["node_id"].astype(str).eq(node_id)].copy()
            if not sub.empty:
                sub["acc_num"] = pd.to_numeric(sub["acc"], errors="coerce")
                best = sub.sort_values("acc_num", ascending=False).head(1).to_dict("records")[0]
        rows.append(
            {
                "level": int(level),
                "node_id": node_id,
                "diagnostic_rows": int(len(metrics)) if metrics is not None else 0,
                "promotion_rows": int(len(promo)) if promo is not None else 0,
                "best_by_acc": best,
            }
        )
    payload = {"created_at": now_iso(), "diagnostic_promote": rows}
    write_json(ANALYSIS_OUT / "geometry_gate_diagnostic_promote_summary.json", payload)
    write_json(ANALYSIS_REPORT / "geometry_gate_diagnostic_promote_summary.json", payload)
    return payload


def badstress_build(levels: list[int]) -> dict[str, Any]:
    built: list[dict[str, Any]] = []
    for level in levels:
        args = make_args(level, "report", run_training=False)
        runner.build_trim_node(args, force=False)
        for cfg in badstress_configs_for_level(level):
            spec_id, variant_dir, audit = _write_geometry_artifact(level, cfg)
            built.append(
                {
                    "level": int(level),
                    "node_id": f"N{int(level)}_gm_trim_bad",
                    "spec_id": spec_id,
                    "variant_dir": str(variant_dir),
                    "audit": audit,
                }
            )
    payload = {"created_at": now_iso(), "built": built}
    write_json(ANALYSIS_OUT / "badstress_build_summary.json", payload)
    write_json(ANALYSIS_REPORT / "badstress_build_summary.json", payload)
    return payload


def badstress_quick(levels: list[int]) -> dict[str, Any]:
    trained: list[dict[str, Any]] = []
    for level in levels:
        node_id = configure_level(level)
        args = make_args(level, "quick", run_training=True)
        runner.build_trim_node(args, force=False)
        level_configs = badstress_configs_for_level(level)
        existing_ids = {
            str(r.get("spec", {}).get("id", ""))
            for r in ladder.load_jsonl(OUT_ROOT / ladder.SUMMARY_NAME)
            if int(r.get("returncode", 1) or 1) == 0
        }
        update_state(
            OUT_ROOT / ladder.STATE_NAME,
            status="badstress_quick_training",
            stage="badstress_quick",
            node_id=node_id,
            total=len(level_configs),
            completed=0,
            updated_at=now_iso(),
        )
        for done, cfg in enumerate(level_configs):
            spec_id, variant_dir, audit = _write_geometry_artifact(level, cfg)
            if spec_id in existing_ids:
                trained.append({"level": int(level), "spec_id": spec_id, "skipped": True, "reason": "existing_success"})
                continue
            train_row = _training_row(level, spec_id, variant_dir, cfg, audit)
            train_row["spec"]["node_role"] = "gm_geometry_repair_trim_bad_badstress"
            train_row["spec"]["refine_note"] = (
                "Controlled N7200 bad-stress expansion from synthetic rows shaped by original report-only diagnostics. "
                "Extreme original bad outliers are excluded from the target idea; original is never used for selection."
            )
            old_loss = args.cls_loss_variant
            old_seed = args.seed
            old_e1 = args.quick_epochs_stage1
            old_e2 = args.quick_epochs_stage2
            old_regen = args.train_regenerate_full
            args.cls_loss_variant = str(cfg["loss"])
            args.seed = int(cfg["seed"])
            args.quick_epochs_stage1 = int(cfg["epochs"][0])
            args.quick_epochs_stage2 = int(cfg["epochs"][1])
            args.train_regenerate_full = 0
            update_state(
                OUT_ROOT / ladder.STATE_NAME,
                status="badstress_quick_training",
                stage="badstress_quick",
                node_id=node_id,
                current=spec_id,
                completed=done,
                total=len(level_configs),
                updated_at=now_iso(),
            )
            try:
                result = targeted.train_one(args, train_row, "quick")
            finally:
                args.cls_loss_variant = old_loss
                args.seed = old_seed
                args.quick_epochs_stage1 = old_e1
                args.quick_epochs_stage2 = old_e2
                args.train_regenerate_full = old_regen
            result["node_id"] = node_id
            result["node_level"] = int(level)
            result["base_variant_id"] = str(cfg.get("base_variant_id", BASE_PROMOTED_VARIANT))
            result["cls_loss_variant"] = str(cfg["loss"])
            result["geometry_gate_config"] = cfg
            result["geometry_gate_audit"] = audit
            ladder.append_jsonl(OUT_ROOT / ladder.SUMMARY_NAME, result)
            ladder.append_jsonl(ladder.node_dir(args, node_id) / ladder.SUMMARY_NAME, result)
            trained.append(result)
            if int(result.get("returncode", 1) or 1) == 0:
                existing_ids.add(spec_id)
        update_state(
            OUT_ROOT / ladder.STATE_NAME,
            status="badstress_quick_complete",
            stage="badstress_quick",
            node_id=node_id,
            completed=len(level_configs),
            total=len(level_configs),
            updated_at=now_iso(),
        )
        ladder.write_report(args)
    payload = {"created_at": now_iso(), "trained": trained}
    write_json(ANALYSIS_OUT / "badstress_quick_summary.json", payload)
    write_json(ANALYSIS_REPORT / "badstress_quick_summary.json", payload)
    return payload


def badstress_diagnostic_promote(levels: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for level in levels:
        node_id = configure_level(level)
        args = make_args(level, "diagnostic", run_training=False)
        metrics = runner.diagnostic_with_medium_guard(args)
        promo = ladder.promote_nodes(args)
        runner.helper.sync_registry_csv_from_json()
        metrics_path = OUT_ROOT / ladder.DIAGNOSTIC_NAME
        best = {}
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            sub = df.loc[df["node_id"].astype(str).eq(node_id)].copy()
            if not sub.empty:
                sub["acc_num"] = pd.to_numeric(sub["acc"], errors="coerce")
                best = sub.sort_values("acc_num", ascending=False).head(1).to_dict("records")[0]
        rows.append(
            {
                "level": int(level),
                "node_id": node_id,
                "diagnostic_rows": int(len(metrics)) if metrics is not None else 0,
                "promotion_rows": int(len(promo)) if promo is not None else 0,
                "best_by_acc": best,
            }
        )
    payload = {"created_at": now_iso(), "diagnostic_promote": rows}
    write_json(ANALYSIS_OUT / "badstress_diagnostic_promote_summary.json", payload)
    write_json(ANALYSIS_REPORT / "badstress_diagnostic_promote_summary.json", payload)
    return payload


def _boundary_block_label(row: pd.Series) -> str:
    source = str(row.get("blend_source", "unknown"))
    class_name = str(row.get("y_class", row.get("class_name", "")))
    if source.startswith("base_"):
        return "bad_core_guard" if class_name == "bad" else "gm_clean_overlap_body"
    return source


def _plot_boundary_pca(variant_dir: Path, spec_id: str, level: int) -> str | None:
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    atlas_path = variant_dir / "synthetic_atlas64_features.csv"
    selected_path = variant_dir / "geometry_selected_aux_rows.csv"
    if not labels_path.exists():
        return None
    labels = pd.read_csv(labels_path, usecols=lambda c: c in {"idx", "y_class", "split", "blend_source"})
    if atlas_path.exists():
        atlas = pd.read_csv(atlas_path)
        if "pc1" not in atlas.columns or "pc2" not in atlas.columns:
            return None
        df = labels.merge(atlas[["idx", "pc1", "pc2"]], on="idx", how="left")
        df["block_label"] = df.apply(_boundary_block_label, axis=1)
        title_suffix = "all rows"
    elif selected_path.exists():
        df = pd.read_csv(selected_path)
        if "pc1" not in df.columns or "pc2" not in df.columns or "source_label" not in df.columns:
            return None
        df["block_label"] = df["source_label"].astype(str)
        df["y_class"] = df.get("selected_class", "").astype(str)
        title_suffix = "selected aux rows"
    else:
        return None
    df = df.dropna(subset=["pc1", "pc2"]).copy()
    if len(df) > 8000:
        df = df.sample(n=8000, random_state=202612)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_OUT / "_mpl_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "gm_clean_overlap_body": "#A3BEFA",
        "gm_good_rescue_pc1flat_qrsvisible": "#FFE15B",
        "gm_visible_qrs_medium_detail": "#71B436",
        "gm_medium_qrslow_hardneg": "#F390CA",
        "bad_core_guard": "#C5CAD3",
        "bad_controlled_outlier": "#F0986E",
    }
    fig, ax = plt.subplots(figsize=(8.8, 6.4), dpi=140)
    for block, sub in df.groupby("block_label", sort=True):
        ax.scatter(
            pd.to_numeric(sub["pc1"], errors="coerce"),
            pd.to_numeric(sub["pc2"], errors="coerce"),
            s=8,
            alpha=0.55,
            linewidths=0,
            label=str(block),
            color=colors.get(str(block), "#7A828F"),
        )
    ax.set_title(f"N{int(level)} boundary block PCA overlay ({title_suffix})", loc="left", fontsize=13, color="#1F2430")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(color="#E6E8F0", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    out_path = ANALYSIS_OUT / f"boundary_blocks_n{int(level)}_{spec_id[-12:]}_pca.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _plot_boundary_waveforms(variant_dir: Path, spec_id: str, level: int) -> str | None:
    data_dir = variant_dir / "datasets"
    labels_path = data_dir / "synth_10s_125hz_labels_with_level.csv"
    clean_path = data_dir / "synth_10s_125hz_clean.npz"
    noisy_path = data_dir / "synth_10s_125hz_noisy.npz"
    if not labels_path.exists() or not clean_path.exists() or not noisy_path.exists():
        return None
    labels = pd.read_csv(labels_path, usecols=lambda c: c in {"idx", "y_class", "split", "blend_source"})
    labels["block_label"] = labels.apply(_boundary_block_label, axis=1)
    pick_rows: list[int] = []
    block_order = [
        "gm_clean_overlap_body",
        "gm_good_rescue_pc1flat_qrsvisible",
        "gm_visible_qrs_medium_detail",
        "gm_medium_qrslow_hardneg",
        "bad_core_guard",
        "bad_controlled_outlier",
    ]
    for block in block_order:
        sub = labels.loc[labels["block_label"].astype(str).eq(block)].copy()
        if sub.empty:
            continue
        sample_n = 2 if block not in {"gm_clean_overlap_body", "bad_core_guard"} else 1
        pick_rows.extend(sub.sample(n=min(sample_n, len(sub)), random_state=202612 + len(pick_rows))["idx"].astype(int).tolist())
    if not pick_rows:
        return None
    pick_rows = pick_rows[:12]
    clean = np.load(clean_path)["X_clean"]
    noisy = np.load(noisy_path)["X_noisy"]
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_OUT / "_mpl_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), dpi=140, sharex=True)
    axes_flat = axes.ravel()
    t = np.arange(clean.shape[1], dtype=float) / 125.0
    label_by_idx = labels.set_index("idx")
    for ax_i, ax in enumerate(axes_flat):
        if ax_i >= len(pick_rows):
            ax.axis("off")
            continue
        idx = int(pick_rows[ax_i])
        row = label_by_idx.loc[idx]
        ax.plot(t, noisy[idx], color="#A3BEFA", linewidth=0.75, alpha=0.85, label="noisy")
        ax.plot(t, clean[idx], color="#804126", linewidth=0.8, alpha=0.85, label="clean")
        ax.set_title(f"{row['block_label']} / {row['y_class']}", fontsize=8, loc="left")
        ax.grid(color="#E6E8F0", linewidth=0.45)
        ax.tick_params(labelsize=7)
    axes_flat[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(f"N{int(level)} boundary block clean/noisy waveforms", x=0.01, y=0.995, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = ANALYSIS_OUT / f"boundary_blocks_n{int(level)}_{spec_id[-12:]}_waveforms.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _boundary_variant_report_rows(level: int, spec_id: str, variant_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    labels = pd.read_csv(labels_path)
    labels["block_label"] = labels.apply(_boundary_block_label, axis=1)
    manifest = (
        labels.groupby(["block_label", "y_class", "split"], dropna=False)
        .size()
        .reset_index(name="n")
        .assign(level=int(level), spec_id=spec_id)
    )
    selected_path = variant_dir / "geometry_selected_aux_rows.csv"
    feature_summary = pd.DataFrame()
    feature_cols = [
        "pc1",
        "pc2",
        "pc3",
        "flatline_ratio",
        "qrs_visibility",
        "non_qrs_diff_p95",
        "qrs_band_ratio",
        "baseline_step",
        "band_15_30",
        "band_30_45",
        "fatal_or_score",
        "contact_loss_win_ratio",
    ]
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        if not selected.empty and "source_label" in selected.columns:
            present = [c for c in feature_cols if c in selected.columns]
            if present:
                feature_summary = (
                    selected.groupby(["source_label", "selected_class"], dropna=False)[present]
                    .median(numeric_only=True)
                    .reset_index()
                    .assign(level=int(level), spec_id=spec_id)
                )
    pca_path = _plot_boundary_pca(variant_dir, spec_id, level)
    waveform_path = _plot_boundary_waveforms(variant_dir, spec_id, level)
    return {
        "level": int(level),
        "spec_id": spec_id,
        "variant_dir": str(variant_dir),
        "rows": int(audit.get("rows", len(labels))),
        "class_counts": audit.get("class_counts", {}),
        "split_counts": audit.get("split_counts", {}),
        "manifest": manifest,
        "feature_summary": feature_summary,
        "pca_path": pca_path,
        "waveform_path": waveform_path,
    }


def _write_boundary_blocks_report(summary: dict[str, Any]) -> None:
    rows = summary.get("variants", [])
    lines = [
        "# Boundary Block Generator",
        "",
        "## What This Adds",
        "",
        "This run turns the boundary plan into explicit synthetic blocks: stable good/medium body, good rescue, visible-QRS medium detail, very-low-QRS medium hard negatives, bad core guard, and a small controlled bad-outlier block.",
        "",
        "Original BUT is report-only. These build and training stages use synthetic/Clean/SemiClean/node diagnostic data only.",
        "",
        "## Built Variants",
        "",
        "| level | spec | rows | class counts | PCA | waveforms |",
        "|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        pca = f"![pca]({row['pca_path']})" if row.get("pca_path") else ""
        wave = f"![waveforms]({row['waveform_path']})" if row.get("waveform_path") else ""
        lines.append(
            f"| N{int(row['level'])} | `{row['spec_id']}` | {int(row['rows'])} | `{json.dumps(row['class_counts'], sort_keys=True)}` | {pca} | {wave} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Manifest CSV: `{ANALYSIS_OUT / 'boundary_blocks_manifest.csv'}`",
            f"- Feature summary CSV: `{ANALYSIS_OUT / 'boundary_blocks_feature_summary.csv'}`",
            f"- Build summary JSON: `{ANALYSIS_OUT / 'boundary_blocks_build_summary.json'}`",
            "",
            "## Guardrails",
            "",
            "- `bad_controlled_outlier` is the only trained bad-outlier expansion block.",
            "- Extreme bad outliers remain report-only stress and are not selected for training.",
            "- Promotion gates remain acc>=0.95, good>=0.92, medium>=0.90, bad>=0.90.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (ANALYSIS_OUT / "boundary_blocks_report.md").write_text(text, encoding="utf-8")
    (ANALYSIS_REPORT / "boundary_blocks_report.md").write_text(text, encoding="utf-8")


def boundary_blocks_build(levels: list[int]) -> dict[str, Any]:
    built: list[dict[str, Any]] = []
    manifests: list[pd.DataFrame] = []
    feature_summaries: list[pd.DataFrame] = []
    for level in levels:
        args = make_args(level, "report", run_training=False)
        runner.build_trim_node(args, force=False)
        for cfg in boundary_block_configs_for_level(level):
            spec_id, variant_dir, audit = _write_geometry_artifact(level, cfg)
            report_row = _boundary_variant_report_rows(level, spec_id, variant_dir, audit)
            built.append({k: v for k, v in report_row.items() if k not in {"manifest", "feature_summary"}})
            manifests.append(report_row["manifest"])
            if not report_row["feature_summary"].empty:
                feature_summaries.append(report_row["feature_summary"])
    if manifests:
        pd.concat(manifests, ignore_index=True).to_csv(ANALYSIS_OUT / "boundary_blocks_manifest.csv", index=False)
        pd.concat(manifests, ignore_index=True).to_csv(ANALYSIS_REPORT / "boundary_blocks_manifest.csv", index=False)
    if feature_summaries:
        pd.concat(feature_summaries, ignore_index=True).to_csv(
            ANALYSIS_OUT / "boundary_blocks_feature_summary.csv", index=False
        )
        pd.concat(feature_summaries, ignore_index=True).to_csv(
            ANALYSIS_REPORT / "boundary_blocks_feature_summary.csv", index=False
        )
    payload = {"created_at": now_iso(), "variants": built}
    write_json(ANALYSIS_OUT / "boundary_blocks_build_summary.json", payload)
    write_json(ANALYSIS_REPORT / "boundary_blocks_build_summary.json", payload)
    _write_boundary_blocks_report(payload)
    return payload


def boundary_blocks_analyze(levels: list[int]) -> dict[str, Any]:
    return boundary_blocks_build(levels)


def boundary_blocks_quick(levels: list[int]) -> dict[str, Any]:
    trained: list[dict[str, Any]] = []
    for level in levels:
        node_id = configure_level(level)
        args = make_args(level, "quick", run_training=True)
        runner.build_trim_node(args, force=False)
        level_configs = boundary_block_configs_for_level(level)
        existing_ids = {
            str(r.get("spec", {}).get("id", ""))
            for r in ladder.load_jsonl(OUT_ROOT / ladder.SUMMARY_NAME)
            if int(r.get("returncode", 1) or 1) == 0
        }
        update_state(
            OUT_ROOT / ladder.STATE_NAME,
            status="boundary_blocks_quick_training",
            stage="boundary_blocks_quick",
            node_id=node_id,
            total=len(level_configs),
            completed=0,
            updated_at=now_iso(),
        )
        for done, cfg in enumerate(level_configs):
            spec_id, variant_dir, audit = _write_geometry_artifact(level, cfg)
            if spec_id in existing_ids:
                trained.append({"level": int(level), "spec_id": spec_id, "skipped": True, "reason": "existing_success"})
                continue
            train_row = _training_row(level, spec_id, variant_dir, cfg, audit)
            train_row["spec"]["node_role"] = "boundary_block_generator_trim_bad"
            train_row["spec"]["refine_note"] = (
                "Boundary block generator: stable good/medium body plus explicit good rescue, visible-QRS medium, "
                "very-low-QRS medium, and small controlled bad-outlier blocks. Original BUT remains report-only."
            )
            old_loss = args.cls_loss_variant
            old_seed = args.seed
            old_e1 = args.quick_epochs_stage1
            old_e2 = args.quick_epochs_stage2
            old_regen = args.train_regenerate_full
            args.cls_loss_variant = str(cfg["loss"])
            args.seed = int(cfg["seed"])
            args.quick_epochs_stage1 = int(cfg["epochs"][0])
            args.quick_epochs_stage2 = int(cfg["epochs"][1])
            args.train_regenerate_full = 0
            update_state(
                OUT_ROOT / ladder.STATE_NAME,
                status="boundary_blocks_quick_training",
                stage="boundary_blocks_quick",
                node_id=node_id,
                current=spec_id,
                completed=done,
                total=len(level_configs),
                updated_at=now_iso(),
            )
            try:
                result = targeted.train_one(args, train_row, "quick")
            finally:
                args.cls_loss_variant = old_loss
                args.seed = old_seed
                args.quick_epochs_stage1 = old_e1
                args.quick_epochs_stage2 = old_e2
                args.train_regenerate_full = old_regen
            result["node_id"] = node_id
            result["node_level"] = int(level)
            result["base_variant_id"] = str(cfg.get("base_variant_id", BASE_PROMOTED_VARIANT))
            result["cls_loss_variant"] = str(cfg["loss"])
            result["boundary_block_config"] = cfg
            result["geometry_gate_audit"] = audit
            ladder.append_jsonl(OUT_ROOT / ladder.SUMMARY_NAME, result)
            ladder.append_jsonl(ladder.node_dir(args, node_id) / ladder.SUMMARY_NAME, result)
            trained.append(result)
            if int(result.get("returncode", 1) or 1) == 0:
                existing_ids.add(spec_id)
        update_state(
            OUT_ROOT / ladder.STATE_NAME,
            status="boundary_blocks_quick_complete",
            stage="boundary_blocks_quick",
            node_id=node_id,
            completed=len(level_configs),
            total=len(level_configs),
            updated_at=now_iso(),
        )
        ladder.write_report(args)
    payload = {"created_at": now_iso(), "trained": trained}
    write_json(ANALYSIS_OUT / "boundary_blocks_quick_summary.json", payload)
    write_json(ANALYSIS_REPORT / "boundary_blocks_quick_summary.json", payload)
    return payload


def boundary_blocks_diagnostic_promote(levels: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for level in levels:
        node_id = configure_level(level)
        args = make_args(level, "diagnostic", run_training=False)
        metrics = runner.diagnostic_with_medium_guard(args)
        promo = ladder.promote_nodes(args)
        runner.helper.sync_registry_csv_from_json()
        metrics_path = OUT_ROOT / ladder.DIAGNOSTIC_NAME
        best = {}
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            sub = df.loc[df["node_id"].astype(str).eq(node_id)].copy()
            if not sub.empty:
                sub["acc_num"] = pd.to_numeric(sub["acc"], errors="coerce")
                best = sub.sort_values("acc_num", ascending=False).head(1).to_dict("records")[0]
        rows.append(
            {
                "level": int(level),
                "node_id": node_id,
                "diagnostic_rows": int(len(metrics)) if metrics is not None else 0,
                "promotion_rows": int(len(promo)) if promo is not None else 0,
                "best_by_acc": best,
            }
        )
    payload = {"created_at": now_iso(), "diagnostic_promote": rows}
    write_json(ANALYSIS_OUT / "boundary_blocks_diagnostic_promote_summary.json", payload)
    write_json(ANALYSIS_REPORT / "boundary_blocks_diagnostic_promote_summary.json", payload)
    return payload


def boundary_blocks_original_report(levels: list[int]) -> dict[str, Any]:
    decisions_path = OUT_ROOT / "node_promotion_decisions.csv"
    rows: list[dict[str, Any]] = []
    promoted_ids: set[str] = set()
    if decisions_path.exists():
        decisions = pd.read_csv(decisions_path)
        variant_col = "variant_id" if "variant_id" in decisions.columns else "best_variant"
        if "promoted" in decisions.columns and variant_col in decisions.columns:
            promoted_mask = decisions["promoted"].astype(str).str.lower().isin({"true", "1", "yes"})
            promoted_ids = set(decisions.loc[promoted_mask, variant_col].astype(str))
    for level in levels:
        for cfg in boundary_block_configs_for_level(level):
            spec_id = targeted.stable_id(f"nl_N{int(level)}_gm_trim_bad_{cfg['name']}_seed{int(cfg['seed'])}")
            rows.append(
                {
                    "level": int(level),
                    "spec_id": spec_id,
                    "status": "ready_for_bucketed_original_report" if spec_id in promoted_ids else "pending_clean_promotion",
                    "note": (
                        "Run bucketed original report-only after clean promotion; original is never used for selection."
                    ),
                }
            )
    payload = {"created_at": now_iso(), "original_report_gate": rows}
    write_json(ANALYSIS_OUT / "boundary_blocks_original_report_summary.json", payload)
    write_json(ANALYSIS_REPORT / "boundary_blocks_original_report_summary.json", payload)
    return payload


def parse_levels(value: str) -> list[int]:
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not levels:
        raise ValueError("At least one level is required.")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description="Good/medium geometry repair for N6900/N7000/N7200 trim-bad nodes.")
    parser.add_argument(
        "--stage",
        choices=(
            "geometry_gate_analysis",
            "geometry_gate_build",
            "geometry_gate_quick",
            "geometry_gate_diagnostic_promote",
            "geometry_gate_all",
            "badstress_build",
            "badstress_quick",
            "badstress_diagnostic_promote",
            "badstress_all",
            "boundary_blocks_analyze",
            "boundary_blocks_build",
            "boundary_blocks_quick",
            "boundary_blocks_diagnostic_promote",
            "boundary_blocks_original_report",
            "boundary_blocks_all",
        ),
        required=True,
    )
    parser.add_argument("--levels", default="6900,7000")
    args = parser.parse_args()
    levels = parse_levels(args.levels)
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORT.mkdir(parents=True, exist_ok=True)
    if args.stage == "geometry_gate_analysis":
        payload = geometry_gate_analysis()
    elif args.stage == "geometry_gate_build":
        payload = geometry_gate_build(levels)
    elif args.stage == "geometry_gate_quick":
        payload = geometry_gate_quick(levels)
    elif args.stage == "geometry_gate_diagnostic_promote":
        payload = geometry_gate_diagnostic_promote(levels)
    elif args.stage == "geometry_gate_all":
        analysis = geometry_gate_analysis()
        build = geometry_gate_build(levels)
        quick = geometry_gate_quick(levels)
        diag = geometry_gate_diagnostic_promote(levels)
        payload = {"analysis": analysis, "build": build, "quick": quick, "diagnostic_promote": diag}
    elif args.stage == "badstress_build":
        payload = badstress_build(levels)
    elif args.stage == "badstress_quick":
        payload = badstress_quick(levels)
    elif args.stage == "badstress_diagnostic_promote":
        payload = badstress_diagnostic_promote(levels)
    elif args.stage == "badstress_all":
        build = badstress_build(levels)
        quick = badstress_quick(levels)
        diag = badstress_diagnostic_promote(levels)
        payload = {"build": build, "quick": quick, "diagnostic_promote": diag}
    elif args.stage == "boundary_blocks_analyze":
        payload = boundary_blocks_analyze(levels)
    elif args.stage == "boundary_blocks_build":
        payload = boundary_blocks_build(levels)
    elif args.stage == "boundary_blocks_quick":
        payload = boundary_blocks_quick(levels)
    elif args.stage == "boundary_blocks_diagnostic_promote":
        payload = boundary_blocks_diagnostic_promote(levels)
    elif args.stage == "boundary_blocks_original_report":
        payload = boundary_blocks_original_report(levels)
    elif args.stage == "boundary_blocks_all":
        build = boundary_blocks_build(levels)
        quick = boundary_blocks_quick(levels)
        diag = boundary_blocks_diagnostic_promote(levels)
        original = boundary_blocks_original_report(levels)
        payload = {"build": build, "quick": quick, "diagnostic_promote": diag, "original_report": original}
    else:
        raise ValueError(args.stage)
    write_json(ANALYSIS_OUT / "geometry_gate_last_run.json", {"stage": args.stage, "levels": levels, "payload": payload})
    print(json.dumps({"stage": args.stage, "levels": levels, "payload": payload}, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
