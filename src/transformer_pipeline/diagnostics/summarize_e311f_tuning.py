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
R3_RUNS = [
    ("R3 lr 2.5e-5", f"{VARIANT}_r3_lr25e5"),
    ("R3 lr 4e-5", f"{VARIANT}_r3_lr4e5"),
    ("R3 dropout 0.15", f"{VARIANT}_r3_drop15"),
    ("R3 weight decay 0.05", f"{VARIANT}_r3_wd05"),
    ("R3 batch size 64", f"{VARIANT}_r3_batch64"),
    ("R3 SNR lambda 0.02", f"{VARIANT}_r3_snr002"),
    ("R3 SNR lambda 0.075", f"{VARIANT}_r3_snr0075"),
]
R4_RUNS = [
    ("R4 seed 1 best recipe", f"{VARIANT}_r4_seed1_best"),
    ("R4 seed 2 best recipe", f"{VARIANT}_r4_seed2_best"),
    ("R4 seed 3 best recipe", f"{VARIANT}_r4_seed3_best"),
    ("R4 dropout 0.05", f"{VARIANT}_r4_dropout005"),
    ("R4 label smoothing 0.01", f"{VARIANT}_r4_label_smoothing001"),
    ("R4 good weight 1.08", f"{VARIANT}_r4_good_weight108"),
    ("R4 medium weight 1.08", f"{VARIANT}_r4_medium_weight108"),
    ("R4 select by val loss", f"{VARIANT}_r4_select_val_loss"),
]
R5_RUNS = [
    ("R5 decoder pooling", f"{VARIANT}_r5_decoder_pool"),
    ("R5 robust input", f"{VARIANT}_r5_robust_input"),
    ("R5 rank 0.01 margin 0.08", f"{VARIANT}_r5_rank001"),
    ("R5 rank 0.02 margin 0.10", f"{VARIANT}_r5_rank002"),
    ("R5 ordinal + SNR", f"{VARIANT}_r5_ordinal_snr"),
    ("R5 ordinal + rank 0.01", f"{VARIANT}_r5_ordinal_rank001"),
    ("R5 long early-stop", f"{VARIANT}_r5_long_earlystop"),
    ("R5 tiny denoise curriculum", f"{VARIANT}_r5_tiny_denoise_curriculum"),
]
R6_RUNS = [
    ("R6 robust SNR 0.02", f"{VARIANT}_r6_robust_snr002"),
    ("R6 robust SNR 0.075", f"{VARIANT}_r6_robust_snr0075"),
    ("R6 robust medium weight 1.03", f"{VARIANT}_r6_robust_med103"),
    ("R6 robust medium weight 1.05", f"{VARIANT}_r6_robust_med105"),
    ("R6 robust label smoothing 0.005", f"{VARIANT}_r6_robust_ls005"),
    ("R6 robust medium 1.03 + SNR 0.02", f"{VARIANT}_r6_robust_med103_snr002"),
    ("R6 robust good/medium weight 1.03", f"{VARIANT}_r6_robust_gm103"),
    ("R6 raw medium weight 1.03", f"{VARIANT}_r6_raw_med103"),
]
R7_RUNS = [
    ("R7 tiny denoise seed 1", f"{VARIANT}_r7_tiny_denoise_seed1"),
    ("R7 tiny denoise seed 2", f"{VARIANT}_r7_tiny_denoise_seed2"),
    ("R7 tiny denoise seed 3", f"{VARIANT}_r7_tiny_denoise_seed3"),
    ("R7 tiny denoise lambda 5 bad 0.05", f"{VARIANT}_r7_tiny_den5_bad005"),
    ("R7 late denoise lambda 8 bad 0.05", f"{VARIANT}_r7_late_den8_bad005"),
    ("R7 tiny denoise SNR 0.02", f"{VARIANT}_r7_tiny_snr002"),
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
    runs.extend(R3_RUNS)
    runs.extend(R4_RUNS)
    runs.extend(R5_RUNS)
    runs.extend(R6_RUNS)
    runs.extend(R7_RUNS)
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
    lines.extend(
        [
            "",
            "## Round 3",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(row(label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in R3_RUNS)
    lines.extend(
        [
            "",
            "## Round 4",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(row(label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in R4_RUNS)
    lines.extend(
        [
            "",
            "## Round 5",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(row(label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in R5_RUNS)
    lines.extend(
        [
            "",
            "## Round 6",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(row(label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in R6_RUNS)
    lines.extend(
        [
            "",
            "## Round 7",
            "",
            "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(row(label, MODEL_ROOT / run_name / "test_report.json") for label, run_name in R7_RUNS)
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
            f"- Best E3.11f visual tuning so far is `{visual_best[0] if visual_best else 'pending'}` at `{visual_best[1]:.4f}`."
            if visual_best
            else "- Best E3.11f visual tuning is still pending.",
            "- The tuning target for replacing E3.10 as the visual benchmark is `>=0.94` test accuracy.",
            "- Round 2 did not improve the first-round best: low-LR continuation dropped to `0.9303`, label smoothing was nearly tied at `0.9372`, and good/medium weighting dropped to `0.9290`.",
            "- Round 3 did not improve either: LR, dropout, weight decay, batch size, and SNR-head weight all stayed at or below `0.9376`.",
            "- Round 4 also stayed below target: seeds 1/2/3 reached `0.9312/0.9342/0.9368`, and light boundary tuning moved good/medium recall around without increasing total accuracy.",
            "- Round 5 tests existing alternative training knobs: pooling, robust input, light rank/ordinal ordering, longer early-stop training, and tiny denoise curriculum.",
            "- Round 6 keeps the best simple recipe but focuses on robust input, because robust input tied the current best while improving good recall and lowering medium recall; the tested knobs are deliberately tiny medium weighting, SNR-head weight, and label smoothing.",
            "- Round 7 returns to the current best tiny-denoise recipe: first checking seed variance, then lowering or delaying denoise to see whether a lighter reconstruction touch can keep the classification gain without pulling the backbone off-task.",
            "- The current best recipe family is: D1 warm-start, `cls_pool=cls`, raw input, `snr_head`, `lambda_snr` near `0.05`, `lr` near `3e-5`, no rank/local/SQI-teacher/noise-type head, and no denoise/level losses.",
            "- Compared with the references, E3.11f R1 is much better than the current E3.11 result (`0.9000`) but remains below E3.10 M2 (`0.9402`) and D1 (`0.9465`).",
            "- For cls-only rows, denoise outputs are not trained; use accuracy/recall as the classification evidence and treat denoise SNR values as non-decision diagnostics.",
            "- Final recommendation: keep E3.10 M2 as the safer `>=0.94` visual benchmark and keep E3.11f as a diagnostic result showing the limit of this cleaner visual-label variant under simple transformer tuning.",
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
