from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ARTIFACT_DIR = ROOT / "outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph"
DEFAULT_OUT_ROOT = ROOT / "outputs/experiment/e311_sqi_research"
DEFAULT_INIT_CHECKPOINT = (
    ROOT / "outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt"
)

PACKAGE_REPORT = ROOT / "src/experiment/e311_sqi_research/reports/research_ablation_report.md"
TOP_REPORT = ROOT / "reports/transformer_e311_research_ablation_report.md"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {0: "good", 1: "medium", 2: "bad"}
NOISE_TO_INT = {"em": 0, "ma": 1, "bw": 2, "mix": 3}

BASELINE_ACC = 0.9464
CURRENT_BEST_ACC = 0.9505
CURRENT_BEST_MEDIUM_RECALL = 0.9496
CURRENT_BEST_BAD_RECALL = 0.9741
