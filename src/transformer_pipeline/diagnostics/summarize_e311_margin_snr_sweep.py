from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = Path(os.environ.get("ROOT_OUT", ROOT / "outputs/transformer_e311_margin_snr_sweep")).expanduser()
REPORT = ROOT / "reports/transformer_e311_margin_snr_sweep_report.md"

CLASS_ORDER = ("good", "medium", "bad")
VARIANTS = [
    ("E3.11b", "e311b_snr_gap_e310_morph", "wide SNR + E3.10 morphology"),
    ("E3.11c", "e311c_snr_gap_relaxed_morph", "wide SNR + relaxed morphology"),
    ("E3.11d", "e311d_snr_primary_good_guard", "wide SNR-primary + good guard"),
    ("E3.11e", "e311e_snr_only_visual", "wide SNR-only visual"),
    ("E3.11f", "e311f_lite_e310_morph", "lite SNR + E3.10 morphology"),
    ("E3.11g", "e311g_lite_snr_primary", "lite SNR-primary"),
]
TRAIN_RUNS = [
    ("E3.11b main", "e311b_snr_gap_e310_morph", "e311b_snr_gap_e310_morph_m1_d1warm_snr005"),
    ("E3.11c main", "e311c_snr_gap_relaxed_morph", "e311c_snr_gap_relaxed_morph_m1_d1warm_snr005"),
    ("E3.11d main", "e311d_snr_primary_good_guard", "e311d_snr_primary_good_guard_m1_d1warm_snr005"),
    ("E3.11e main", "e311e_snr_only_visual", "e311e_snr_only_visual_m1_d1warm_snr005"),
    ("E3.11f main", "e311f_lite_e310_morph", "e311f_lite_e310_morph_m1_d1warm_snr005"),
    ("E3.11g main", "e311g_lite_snr_primary", "e311g_lite_snr_primary_m1_d1warm_snr005"),
    ("E3.11b denoise-aware", "e311b_snr_gap_e310_morph", "e311b_snr_gap_e310_morph_m2_d1warm_snr005_denoise"),
    ("E3.11d denoise-aware", "e311d_snr_primary_good_guard", "e311d_snr_primary_good_guard_m2_d1warm_snr005_denoise"),
]
BASELINES = [
    ("D1 reference", ROOT / "outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/test_report.json"),
    ("E3.10 M2 best", ROOT / "outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m2_d1warm_snr005/test_report.json"),
    ("E3.11 M1 best", ROOT / "outputs/transformer_e311_visual_gap/models/e311_m1_d1warm_lr3e5/test_report.json"),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric(summary: dict[str, Any], name: str, cls: str, field: str = "mean") -> float | None:
    try:
        return float(summary["audit"]["class_metric_summary"][name][cls][field])
    except KeyError:
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def show(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def split_total(summary: dict[str, Any], split: str) -> int:
    counts = summary.get("split_y_class_counts", {}).get(split, {})
    return int(sum(int(counts.get(cls, 0)) for cls in CLASS_ORDER) / 3)


def sqi_accs(path: Path) -> tuple[float | None, float | None]:
    data = load_json(path)
    if data is None:
        return None, None
    metrics = data.get("metrics", {})
    svm = metrics.get("svm_rbf", {}).get("test", {}).get("acc")
    mlp = metrics.get("mlp", {}).get("test", {}).get("acc")
    return (float(svm) if svm is not None else None, float(mlp) if mlp is not None else None)


def recalls(cm: list[list[int]]) -> tuple[float, float, float]:
    out: list[float] = []
    for i, row in enumerate(cm):
        out.append(float(row[i]) / float(max(1, sum(row))))
    return out[0], out[1], out[2]


def denoise_snr_improve(report: dict[str, Any]) -> str:
    by_class = report.get("denoise_metrics_by_class", {})
    parts: list[str] = []
    for cls in CLASS_ORDER:
        value = by_class.get(cls, {}).get("snr_improve_db")
        if value is not None:
            parts.append(f"{cls}:{float(value):.2f}")
    return ", ".join(parts)


def baseline_rows() -> list[str]:
    rows: list[str] = []
    for label, path in BASELINES:
        rep = load_json(path)
        if rep is None:
            rows.append(f"| {label} | missing |  |  |  | `{show(path)}` |")
            continue
        good, medium, bad = recalls(rep["confusion_matrix_3x3"])
        rows.append(
            f"| {label} | {float(rep['acc']):.4f} | {good:.4f} | {medium:.4f} | {bad:.4f} | `{rep['confusion_matrix_3x3']}` |"
        )
    return rows


def audit_rows() -> list[str]:
    rows = [
        "| Variant | Design | Train Groups | Val Groups | Test Groups | SNR Oracle | SNR Mean Gap | Smooth Mean G/M/B | Global Mean G/M/B | Obs Margin p10/p50 | SQI-SVM | SQI-MLP | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for short, variant, design in VARIANTS:
        artifact = SWEEP_ROOT / variant
        summary = load_json(artifact / "datasets/morph_damage_triplet_summary.json")
        svm, mlp = sqi_accs(artifact / "sqi_ml_three_class/three_class_summary.json")
        if summary is None:
            rows.append(f"| {short} | {design} | pending |  |  |  |  |  |  |  | {fmt(svm)} | {fmt(mlp)} | pending |")
            continue
        train = split_total(summary, "train")
        val = split_total(summary, "val")
        test = split_total(summary, "test")
        smooth = "/".join(fmt(metric(summary, "smooth_morph_score", cls)) for cls in CLASS_ORDER)
        global_noise = "/".join(fmt(metric(summary, "global_noise_score", cls)) for cls in CLASS_ORDER)
        obs = summary.get("observable_margin", {}).get("observable_margin", {})
        gate = "pass" if train >= 3000 and val >= 600 and test >= 600 else "low-count"
        rows.append(
            f"| {short} | {design} | {train} | {val} | {test} | "
            f"{float(summary.get('measured_snr_oracle_acc', 0.0)):.4f} | "
            f"{float(summary['audit']['max_class_mean_gap'].get('measured_snr_db', 0.0)):.4f} | "
            f"{smooth} | {global_noise} | {fmt(obs.get('p10'))}/{fmt(obs.get('p50'))} | "
            f"{fmt(svm)} | {fmt(mlp)} | {gate} |"
        )
    return rows


def training_rows() -> list[str]:
    rows = [
        "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve | Confusion Matrix |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    best: tuple[str, float] | None = None
    for label, variant, run_name in TRAIN_RUNS:
        path = SWEEP_ROOT / variant / "models" / run_name / "test_report.json"
        rep = load_json(path)
        if rep is None:
            rows.append(f"| {label} | pending |  |  |  |  | `{show(path)}` |")
            continue
        good, medium, bad = recalls(rep["confusion_matrix_3x3"])
        acc = float(rep["acc"])
        if best is None or acc > best[1]:
            best = (label, acc)
        rows.append(
            f"| {label} | {acc:.4f} | {good:.4f} | {medium:.4f} | {bad:.4f} | "
            f"{denoise_snr_improve(rep)} | `{rep['confusion_matrix_3x3']}` |"
        )
    if best is not None:
        rows.extend(["", f"Best new run so far: `{best[0]}` = `{best[1]:.4f}`"])
    return rows


def figure_rows() -> list[str]:
    case_root = SWEEP_ROOT / "real_ecg_case_folders"
    rows = [
        "Combined visual galleries:",
        "",
        f"- Class x noise examples: `{show(SWEEP_ROOT / 'visual_gallery' / 'e311_margin_snr_class_noise_examples_gallery.png')}`",
        f"- Counterfactual triplets: `{show(SWEEP_ROOT / 'visual_gallery' / 'e311_margin_snr_counterfactual_triplets_gallery.png')}`",
        f"- Audit overview: `{show(SWEEP_ROOT / 'visual_gallery' / 'e311_margin_snr_audit_overview.png')}`",
        "",
        "Individual real ECG case folders:",
        "",
        f"- Folder root: `{show(case_root)}`",
        f"- Browser index: `{show(case_root / 'index.html')}`",
        f"- Metadata index: `{show(case_root / 'selected_cases.csv')}`",
        f"- Layout: `{show(case_root / 'by_variant')}/<variant>/<good|medium|bad>/<em|ma|mix>/`",
        "",
        "| Variant | Triplets | Class x Noise Examples |",
        "| --- | --- | --- |",
    ]
    for short, variant, _ in VARIANTS:
        fig_dir = SWEEP_ROOT / variant / "figs_label_samples"
        triplet = fig_dir / f"{variant}_counterfactual_triplets.png"
        class_noise = fig_dir / f"{variant}_class_noise_examples.png"
        rows.append(
            f"| {short} | `{show(triplet)}` | `{show(class_noise)}` |"
        )
    return rows


def main() -> None:
    lines = [
        "# E3.11 SNR/Morphology Learnability Sweep",
        "",
        "Goal: diagnose whether E3.11 is hard because the SNR gap is too wide, the morphology margin is too strict, or clean-reference morphology labels are weakly visible in raw noisy ECG.",
        "",
        "## Baselines",
        "",
        "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Confusion Matrix |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        *baseline_rows(),
        "",
        "## Data Audit",
        "",
        *audit_rows(),
        "",
        "## Transformer Runs",
        "",
        *training_rows(),
        "",
        "## Figures",
        "",
        *figure_rows(),
        "",
        "## Reading Guide",
        "",
        "- If E3.11b beats current E3.11 clearly, the strict E3.11 morphology margin was the main issue.",
        "- If E3.11d/e beat E3.11b, SNR/visual dirtiness is learnable while morphology labels are less visible.",
        "- If lite variants beat wide variants, the 3-6 dB bad range is too severe for this benchmark.",
        "- If SQI baselines are high, use that variant only as a visual diagnostic, not as the main non-shortcut benchmark.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
