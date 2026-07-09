from __future__ import annotations

import os

from .common import POLICY, SUPPORT, py, run_or_print


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def allow_cpu_train() -> bool:
    return os.environ.get("ECG_ALLOW_CPU_CONFORMER_TRAIN", "").strip().lower() in {"1", "true", "yes"}


def command(model: str) -> list[str]:
    if model == "E4":
        return [
            py(),
            str(SUPPORT / "run_event_factorized_sqi_conformer.py"),
            "--stage",
            "phase1",
            "--policy",
            POLICY,
            "--folds",
            "1",
            "--seeds",
            "1",
            "--seed",
            "20260876",
            "--split-seed",
            "20260876",
            "--epochs",
            "8",
            "--batch-size",
            "96",
            "--candidates",
            "E4_query_highres_local_art",
            "--no-record-balanced-sampler",
        ]
    if model == "E24":
        return [
            py(),
            str(SUPPORT / "run_gm_mechanism_repair_suite.py"),
            "--policy",
            POLICY,
            "--folds",
            "1",
            "--seed",
            "20260876",
            "--split-seed",
            "20260876",
            "--epochs",
            "10",
            "--batch-size",
            "96",
            "--candidates",
            "E24_e6_subtype_fusion_pairrank",
            "--no-record-balanced-sampler",
        ]
    if model == "E31":
        return [
            py(),
            str(SUPPORT / "run_gm_mechanism_repair_suite.py"),
            "--policy",
            POLICY,
            "--folds",
            "1",
            "--seed",
            "20260876",
            "--split-seed",
            "20260876",
            "--epochs",
            "10",
            "--batch-size",
            "96",
            "--candidates",
            "E31_wave_mechanism_conformer",
            "--no-record-balanced-sampler",
        ]
    raise ValueError(f"unknown model: {model}")


def models(value: str) -> list[str]:
    if value == "both":
        return ["E4", "E24"]
    if value == "all":
        return ["E4", "E24", "E31"]
    return [value]


def main(*, model: str = "E31", run: bool = False) -> None:
    if run and not cuda_available() and not allow_cpu_train():
        print(
            "SKIP_GPU_REQUIRED: Conformer training was requested, but CUDA is not available. "
            "Set ECG_ALLOW_CPU_CONFORMER_TRAIN=1 to force slow CPU training."
        )
        return
    for name in models(model):
        run_or_print(command(name), run=run)
