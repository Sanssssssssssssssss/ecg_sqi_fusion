from __future__ import annotations

from .common import POLICY, SUPPORT, py, run_or_print


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
    raise ValueError(f"unknown model: {model}")


def models(value: str) -> list[str]:
    return ["E4", "E24"] if value == "both" else [value]


def main(*, model: str = "both", run: bool = False) -> None:
    for name in models(model):
        run_or_print(command(name), run=run)
