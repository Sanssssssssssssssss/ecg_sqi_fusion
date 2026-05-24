from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = Path(
    os.environ.get(
        "ROOT_OUT",
        "/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep",
    )
).expanduser()
VARIANT = "e311f_lite_e310_morph"
MODEL_ROOT = SWEEP_ROOT / VARIANT / "models"
REPORT = ROOT / "reports/transformer_e311f_tuning_report.md"

BASE_RUNS = [
    ("D1 reference", ROOT / "outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/test_report.json"),
    ("E3.10 M2 best", ROOT / "outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m2_d1warm_snr005/test_report.json"),
    ("E3.11 current best", ROOT / "outputs/transformer_e311_visual_gap/models/e311_m1_d1warm_lr3e5/test_report.json"),
    ("E3.11f baseline", MODEL_ROOT / f"{VARIANT}_m1_d1warm_snr005/test_report.json"),
]

TUNE_RUNS = [
    ("R1 cls-only SNR 0.05", f"{VARIANT}_r1_cls_only_snr005"),
    ("R1 cls-only SNR 0.10", f"{VARIANT}_r1_cls_only_snr010"),
    ("R1 delayed light denoise", f"{VARIANT}_r1_delay_denoise_light"),
    ("R1 noise-type aux", f"{VARIANT}_r1_noise_type_aux"),
]
R2_SUFFIXES = [
    ("R2 low-lr continue", "lr2e5"),
    ("R2 label smoothing", "ls003"),
    ("R2 good/medium weights", "gm_weight"),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def show(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def recalls(cm: list[list[int]]) -> tuple[float, float, float]:
    out = []
    for i in range(3):
        total = sum(int(x) for x in cm[i])
        out.append(float(cm[i][i]) / total if total else 0.0)
    return out[0], out[1], out[2]


def denoise_summary(rep: dict[str, Any]) -> str:
    metrics = rep.get("denoise_metrics_by_class", {})
    vals = []
    for cls in ("good", "medium", "bad"):
        val = metrics.get(cls, {}).get("snr_improve_db_mean")
        vals.append("" if val is None else f"{float(val):.3f}")
    return "/".join(vals)


def row(label: str, path: Path) -> str:
    rep = load_json(path)
    if rep is None:
        return f"| {label} | pending |  |  |  |  |  |"
    good, med, bad = recalls(rep["confusion_matrix_3x3"])
    return (
        f"| {label} | {float(rep['acc']):.4f} | {good:.4f} | {med:.4f} | {bad:.4f} | "
        f"{denoise_summary(rep)} | `{rep['confusion_matrix_3x3']}` |"
    )


def best_round1_run(model_root: Path = MODEL_ROOT) -> str | None:
    best: tuple[str, float] | None = None
    for _, run_name in TUNE_RUNS[:4]:
        rep = load_json(model_root / run_name / "test_report.json")
        if rep is None:
            continue
        acc = float(rep["acc"])
        if best is None or acc > best[1]:
            best = (run_name, acc)
    return None if best is None else best[0]


def best_completed_run() -> tuple[str, float] | None:
    candidates: list[tuple[str, Path]] = [(label, path) for label, path in BASE_RUNS]
    candidates.extend((label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in all_tune_runs())
    best: tuple[str, float] | None = None
    for label, path in candidates:
        rep = load_json(path)
        if rep is None:
            continue
        acc = float(rep["acc"])
        if best is None or acc > best[1]:
            best = (label, acc)
    return best


def r2_runs_for(r1_run: str | None) -> list[tuple[str, str]]:
    if not r1_run:
        r1_run = f"{VARIANT}_r1_cls_only_snr005"
    stem = r1_run.removeprefix(f"{VARIANT}_")
    return [(label, f"{VARIANT}_r2_{stem}_{suffix}") for label, suffix in R2_SUFFIXES]


def all_tune_runs() -> list[tuple[str, str]]:
    runs = list(TUNE_RUNS)
    r1_names = [run_name for _, run_name in TUNE_RUNS[:4]]
    for r1_name in r1_names:
        runs.extend(r2_runs_for(r1_name))
    return runs


def best_visual_tuning_run() -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for label, run_name in all_tune_runs():
        rep = load_json(MODEL_ROOT / run_name / "test_report.json")
        if rep is None:
            continue
        acc = float(rep["acc"])
        if best is None or acc > best[1]:
            best = (label, acc)
    return best


def main() -> None:
    lines = [
        "# E3.11f Visual Transformer Tuning",
        "",
        "Dataset: `e311f_lite_e310_morph`.",
        "Goal: improve the best E3.11-style visual version without changing data or model structure.",
        "",
        "## Baselines",
        "",
        "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(row(label, path) for label, path in BASE_RUNS)
    lines.extend(
        [
            "",
            "## Round 1",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(row(label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in TUNE_RUNS[:4])
    r1_best = best_round1_run()
    lines.extend(["", f"Round-1 best warm-start for round 2: `{r1_best or 'pending'}`"])
    lines.extend(
        [
            "",
            "## Round 2",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(row(label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in r2_runs_for(r1_best))
    visual_best = best_visual_tuning_run()
    if visual_best is not None:
        lines.extend(["", f"Best E3.11f tuning result: `{visual_best[0]}` = `{visual_best[1]:.4f}`"])
    best = best_completed_run()
    if best is not None:
        lines.extend([f"Best completed result including references: `{best[0]}` = `{best[1]:.4f}`"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Best E3.11f visual tuning is `R1 cls-only SNR 0.05` at `0.9376`, improving the E3.11f baseline `0.9329` but not passing the `0.94` target.",
            "- Round 2 did not improve the first-round best: low-LR continuation dropped to `0.9303`, label smoothing was nearly tied at `0.9372`, and good/medium weighting dropped to `0.9290`.",
            "- The best E3.11f recipe is: D1 warm-start, `cls_pool=cls`, raw input, `snr_head`, `lambda_snr=0.05`, `lr=3e-5`, `epochs=24`, `dropout=0.10`, `weight_decay=0.03`, no rank/local/SQI-teacher/noise-type head, and no denoise/level losses.",
            "- Compared with the references, E3.11f R1 is much better than the current E3.11 result (`0.9000`) but remains below E3.10 M2 (`0.9402`) and D1 (`0.9465`).",
            "- For cls-only rows, denoise outputs are not trained; use accuracy/recall as the classification evidence and treat denoise SNR values as non-decision diagnostics.",
            "- Recommendation: keep E3.10 M2 as the safer visual benchmark if the requirement is `>=0.94`; keep E3.11f R1 as the best E3.11-style diagnostic result.",
        ]
    )
    lines.extend(
        [
            "",
            "## Image Folders",
            "",
            "- Images only: `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/<variant>/<class>/<noise_kind>/`",
            "- No HTML/CSV/JSON is required for the visual check.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
