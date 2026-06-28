from __future__ import annotations

from .common import POLICY, SUPPORT, py, run_or_print


def command() -> list[str]:
    return [
        py(),
        str(SUPPORT / "run_event_factorized_sqi_conformer.py"),
        "--stage",
        "build_recordheldout_splits",
        "--policy",
        POLICY,
        "--folds",
        "1",
        "--seed",
        "20260876",
        "--split-seed",
        "20260876",
        "--no-record-balanced-sampler",
    ]


def main(*, run: bool = False) -> None:
    run_or_print(command(), run=run)
