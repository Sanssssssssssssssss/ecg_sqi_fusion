from __future__ import annotations

from typing import Any

CANDIDATE_NAME = "E31_wave_mechanism_conformer"


def official_config() -> dict[str, Any]:
    return {
        "factor_contract": "mechanism",
        "gm_mode": "direct",
        "factor_weight": 0.16,
        "local_weight": 0.14,
        "artifact_weight": 0.14,
        "subtype_weight": 0.02,
        "subtype_class_consistency_weight": 0.0,
        "subtype_class_fusion_alpha": 0.0,
        "use_pairrank": False,
        "pairrank_weight": 0.0,
        "hard_gm_weight": 0.0,
        "medium_bad_guard_weight": 0.0,
        "lr": 1.5e-4,
        "class_weights": [1.03, 1.05, 1.08],
    }
