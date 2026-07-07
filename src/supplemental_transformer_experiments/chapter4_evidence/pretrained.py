from __future__ import annotations

import os
from pathlib import Path

from .common import ROOT


PRETRAINED_ROOT = ROOT / "pretrained" / "chapter4"
SETA_E31_DIR = PRETRAINED_ROOT / "seta_e31_leadwise_shared"
BUT_E31_QUERY_MEAN_DIR = PRETRAINED_ROOT / "but_e31_query_mean_fused_conformer"
BUT_ARCHITECTURE_ABLATION_DIR = PRETRAINED_ROOT / "but_architecture_ablation"


def use_pretrained_on_cpu(device: str) -> bool:
    if str(device).lower() != "cpu":
        return False
    return os.environ.get("ECG_ALLOW_CPU_CONFORMER_TRAIN", "").strip() not in {"1", "true", "TRUE", "yes", "YES"}


def require_files(root: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing pretrained Conformer artifact(s) under {root}: {missing}")
