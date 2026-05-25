from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT / "outputs/transformer_e310_smooth_morph_mild_snr"
MODEL_ROOT = ARTIFACT_DIR / "models"
REPORT = ROOT / "reports/transformer_e310_visual_tuning_report.md"

BASE_RUNS = [
    ("D1 reference", ROOT / "outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/test_report.json"),
    ("E3.10 M0 baseline", MODEL_ROOT / "e310_smooth_morph_mild_snr_cls_raw/test_report.json"),
    ("E3.10 M1 D1 warm-start", MODEL_ROOT / "e310_m1_d1warm_lr3e5/test_report.json"),
    ("E3.10 M2 warm-start + SNR head", MODEL_ROOT / "e310_m2_d1warm_snr005/test_report.json"),
    ("E3.10 M3 low denoise", MODEL_ROOT / "e310_m3_d1warm_snr005_lowden/test_report.json"),
    ("E3.10 M4 noise type", MODEL_ROOT / "e310_m4_d1warm_snr005_lowden_ntype005/test_report.json"),
]

R1_RUNS = [
    ("R1 M2 seed 1", "e310_r1_m2_seed1"),
    ("R1 M2 seed 2", "e310_r1_m2_seed2"),
    ("R1 M2 seed 3", "e310_r1_m2_seed3"),
    ("R1 SNR lambda 0.02", "e310_r1_snr002"),
    ("R1 SNR lambda 0.075", "e310_r1_snr0075"),
    ("R1 medium weight 1.03", "e310_r1_medium103"),
    ("R1 medium weight 1.05", "e310_r1_medium105"),
    ("R1 label smoothing 0.005", "e310_r1_label_smoothing005"),
]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _recalls(cm: list[list[int]]) -> tuple[float, float, float]:
    vals = []
    for i in range(3):
        denom = sum(int(x) for x in cm[i])
        vals.append(float(cm[i][i] / denom) if denom else 0.0)
    return vals[0], vals[1], vals[2]


def _denoise(rep: dict[str, Any]) -> str:
    metrics = rep.get("denoise_metrics_by_class", {})
    out = []
    for cls in ("good", "medium", "bad"):
        val = metrics.get(cls, {}).get("snr_improve_db_mean")
        out.append("" if val is None else f"{float(val):.3f}")
    return "/".join(out)


def _row(label: str, path: Path) -> str:
    rep = _load(path)
    if rep is None:
        return f"| {label} | pending |  |  |  |  |  |"
    good, medium, bad = _recalls(rep["confusion_matrix_3x3"])
    return (
        f"| {label} | {float(rep['acc']):.4f} | {good:.4f} | {medium:.4f} | {bad:.4f} | "
        f"{_denoise(rep)} | `{rep['confusion_matrix_3x3']}` |"
    )


def _best() -> tuple[str, float] | None:
    candidates: list[tuple[str, Path]] = [(label, path) for label, path in BASE_RUNS if label.startswith("E3.10")]
    candidates.extend((label, MODEL_ROOT / run / "test_report.json") for label, run in R1_RUNS)
    best: tuple[str, float] | None = None
    for label, path in candidates:
        rep = _load(path)
        if rep is None:
            continue
        acc = float(rep["acc"])
        if best is None or acc > best[1]:
            best = (label, acc)
    return best


def main() -> None:
    lines = [
        "# E3.10 Visual Transformer Tuning",
        "",
        "Dataset: `e310_smooth_morph_mild_snr`.",
        "Goal: improve the current single-model visual benchmark while keeping the same simple CLS raw transformer family.",
        "",
        "## Baselines",
        "",
        "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(_row(label, path) for label, path in BASE_RUNS)
    lines.extend(
        [
            "",
            "## Round 1",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(_row(label, MODEL_ROOT / run / "test_report.json") for label, run in R1_RUNS)
    best = _best()
    if best:
        lines.extend(["", f"Best E3.10 visual result: `{best[0]}` = `{best[1]:.4f}`"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- M2 is the current single-model anchor because it already reaches the 0.94 visual benchmark threshold.",
            "- Round 1 keeps the architecture fixed and only tests seed robustness, SNR-head weight, light label smoothing, and small medium-class weighting.",
            "- If Round 1 does not improve M2, the next useful step is ensemble/error audit rather than adding heads.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
