from __future__ import annotations

from .common import SUPPORT, py, run_or_print


def command() -> list[str]:
    return [
        py(),
        str(SUPPORT / "run_v116_native_budget_repair.py"),
        "--stage",
        "all",
        "--balance-policy",
        "gap_fill",
        "--seed",
        "20260876",
        "--device",
        "auto",
        "--native-grid",
        "none",
        "--final-per-class",
        "10530",
        "--clean-candidates-per-class",
        "300",
        "--residual-per-subtype",
        "200",
        "--max-donors-per-subtype",
        "1600",
        "--native-morph-copies",
        "8",
        "--native-morph-strength",
        "0.10",
        "--gap-clean-cap",
        "0.01",
        "--gap-native-morph-min-frac",
        "0.99",
        "--gap-native-morph-selection",
        "random",
        "--max-ptb-carriers",
        "9000",
        "--floor-draws",
        "8",
        "--floor-max-rows-per-class",
        "2200",
        "--selector-swaps",
        "1000",
        "--support-max-target-rows",
        "2400",
        "--rff-dim",
        "1024",
    ]


def main(*, run: bool = False) -> None:
    run_or_print(command(), run=run)
