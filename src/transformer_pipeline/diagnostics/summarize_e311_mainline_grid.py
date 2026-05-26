from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/transformer_e311_mainline_grid_report.md"

VARIANTS = [
    ("e311f_lite_e310_morph", "E3.11f lite SNR + E3.10 morphology"),
    ("e311h_lite_relaxed_morph", "E3.11h lite SNR + relaxed morphology"),
    ("e311i_wide_relaxed_morph", "E3.11i wide SNR + relaxed morphology"),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def recalls(cm: list[list[int]]) -> tuple[float, float, float]:
    vals = []
    for i in range(3):
        denom = sum(int(v) for v in cm[i])
        vals.append(float(cm[i][i] / denom) if denom else 0.0)
    return vals[0], vals[1], vals[2]


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def class_counts(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "missing"
    counts = summary.get("split_y_class_counts", {})
    out = []
    for sp in ("train", "val", "test"):
        d = counts.get(sp, {})
        out.append(f"{sp}: {d.get('good', 0)}/{d.get('medium', 0)}/{d.get('bad', 0)}")
    return "; ".join(out)


def metric_gap(summary: dict[str, Any] | None, metric: str) -> str:
    if not summary:
        return ""
    data = summary.get("audit", {}).get("class_metric_summary", {}).get(metric, {})
    vals = []
    for cls in ("good", "medium", "bad"):
        if cls in data:
            vals.append(fmt(data[cls].get("mean"), 3))
    return "/".join(vals)


def sqi_row(artifact_dir: Path) -> tuple[str, str]:
    summary = load_json(artifact_dir / "sqi_ml_three_class" / "three_class_summary.json")
    if not summary:
        return "", ""
    metrics = summary.get("metrics", {})
    svm = metrics.get("svm_rbf", {}).get("test", {}).get("acc")
    mlp = metrics.get("mlp", {}).get("test", {}).get("acc")
    return fmt(svm), fmt(mlp)


def run_rows(artifact_dir: Path) -> list[tuple[str, float, str]]:
    rows: list[tuple[str, float, str]] = []
    model_root = artifact_dir / "models"
    if not model_root.exists():
        return rows
    for report in sorted(model_root.glob("*/test_report.json")):
        rep = load_json(report)
        if not rep:
            continue
        cm = rep.get("confusion_matrix_3x3", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        good, medium, bad = recalls(cm)
        run_name = report.parent.name
        probe = load_json(report.parent / "probe_summary.json") or {}
        hp = probe.get("hyperparams", {})
        desc_bits = [
            f"acc={fmt(rep.get('acc'))}",
            f"rec={fmt(good)}/{fmt(medium)}/{fmt(bad)}",
            f"pool={hp.get('cls_pool', '')}",
            f"pos={hp.get('use_positional_embedding', False)}",
            f"snr={hp.get('lambda_snr', '')}",
            f"ord={hp.get('ordinal_head', False)}",
            f"noise={hp.get('noise_type_head', False)}",
            f"den={hp.get('lambda_den', '')}",
            f"lr={hp.get('lr', '')}",
            f"drop={hp.get('dropout', '')}",
        ]
        rows.append((run_name, float(rep.get("acc", 0.0)), ", ".join(desc_bits)))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    root_out = Path(args.root_out)
    if not root_out.is_absolute():
        root_out = ROOT / root_out

    lines = [
        "# E3.11 Mainline Grid",
        "",
        "Mainline decision: E3.11f-style visual data replaces E3.10 as the target line.",
        "The source clean pool uses strict PTB-XL filtering: any baseline/static/burst/electrode annotation in any lead is rejected.",
        "E3.10 remains only a historical reference, not an optimization target.",
        "",
        "Current historical references:",
        "",
        "- E3.10 best single visual: `0.9402`",
        "- E3.10 calibration/ensemble diagnostic: `0.9419`",
        "- Previous E3.11f best single visual: `0.9381`",
        "- Previous E3.11f ensemble diagnostic: `0.9402`",
        "",
        "Sweep goal:",
        "",
        "- keep E3.11f as the visual mainline and find a stable single model at or above `0.94`",
        "- screen whether relaxed morphology or wider SNR makes the task more learnable",
        "- prune any branch that is clearly worse before expanding the grid",
        "",
        "## Data Variants",
        "",
        "| Variant | Meaning | Class Counts G/M/B | measured SNR Mean G/M/B | smooth Morph Mean G/M/B | global Noise Mean G/M/B | SQI-SVM | SQI-MLP |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in VARIANTS:
        artifact_dir = root_out / name
        summary = load_json(artifact_dir / "datasets" / "morph_damage_triplet_summary.json")
        svm, mlp = sqi_row(artifact_dir)
        lines.append(
            f"| `{name}` | {label} | {class_counts(summary)} | "
            f"{metric_gap(summary, 'measured_snr_db')} | "
            f"{metric_gap(summary, 'smooth_morph_score')} | "
            f"{metric_gap(summary, 'global_noise_score')} | {svm} | {mlp} |"
        )

    lines.extend(
        [
            "",
            "## Transformer Results",
            "",
        ]
    )
    overall: list[tuple[str, str, float, str]] = []
    for name, label in VARIANTS:
        artifact_dir = root_out / name
        rows = run_rows(artifact_dir)
        lines.extend(
            [
                f"### {name}",
                "",
                f"{label}.",
                "",
                "| Rank | Run | Test Acc | Summary |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        if not rows:
            lines.append("|  | pending |  |  |")
        else:
            for rank, (run_name, acc, desc) in enumerate(rows[:12], start=1):
                lines.append(f"| {rank} | `{run_name}` | {acc:.4f} | {desc} |")
                overall.append((name, run_name, acc, desc))
        lines.append("")

    overall.sort(key=lambda item: item[2], reverse=True)
    lines.extend(
        [
            "## Current Best",
            "",
            "| Rank | Variant | Run | Test Acc | Summary |",
            "| ---: | --- | --- | ---: | --- |",
        ]
    )
    if not overall:
        lines.append("|  | pending | pending |  |  |")
    else:
        for rank, (variant, run_name, acc, desc) in enumerate(overall[:15], start=1):
            lines.append(f"| {rank} | `{variant}` | `{run_name}` | {acc:.4f} | {desc} |")

    lines.extend(
        [
            "",
            "## Stage Decisions",
            "",
            "- Keep `e311f_lite_e310_morph` as the main E3.11 data line: it is the only branch close to the historical E3.10 visual result.",
            "- Prune `e311h_lite_relaxed_morph`: the relaxed-morph screen stayed around `0.90-0.91`, far below E3.11f.",
            "- Prune `e311i_wide_relaxed_morph`: it has higher SQI baselines and early validation was clearly worse than E3.11f, so the wide-SNR branch is diagnostic only.",
            "- Round 2 crossed the target with `e311f_lite_e310_morph_r2_lr5_seed1_pos` at `0.9432`.",
            "- Keep the simple model recipe: CLS pooling, positional embedding, raw input, D1 warm-start, SNR head with `lambda_snr=0.05`, no denoise, no local head, no rank loss.",
            "- Drop the weak branches from the next sweep: relaxed morphology, wide SNR, raw_robust input, cls_mean pooling, val-loss checkpoint selection, noise-type head, and class-weight tweaks.",
            "- Round 3 should only probe LR/seed stability around `5e-5..6.25e-5` plus the light `dropout=0.075` hint.",
            "",
            "## Pruning Rules",
            "",
            "- If a variant's first 3-6 runs stay below `0.90`, stop expanding it.",
            "- If relaxed morphology beats E3.11f by `>=0.01`, expand that data branch.",
            "- If wide SNR is high but SQI baselines are also high, keep it as visual/diagnostic rather than the main benchmark.",
            "- If positional embedding helps the top E3.11f runs, keep it in round 2; otherwise remove it.",
            "- If denoise-aware runs lose more than `0.01` versus their paired classifier-only run, remove denoise from the next grid.",
        ]
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the E3.11 mainline data/model grid.")
    parser.add_argument("--root_out", default="outputs/transformer_e311_mainline_strict")
    return parser.parse_args()


if __name__ == "__main__":
    main()
