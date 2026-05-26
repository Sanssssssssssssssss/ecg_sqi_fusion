from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/transformer_e311_head_combo_grid_report.md"
VARIANT = "e311f_lite_e310_morph"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def recalls(cm: list[list[int]]) -> tuple[float, float, float]:
    out = []
    for i in range(3):
        denom = sum(int(v) for v in cm[i])
        out.append(float(cm[i][i] / denom) if denom else 0.0)
    return out[0], out[1], out[2]


def family_for(run_name: str) -> str:
    name = run_name.lower()
    if "no_snr" in name:
        return "no_snr_control"
    if "base" in name:
        return "snr_baseline"
    if "levelonly" in name:
        return "level_aux"
    if "den" in name:
        return "denoise_aux"
    bits = [bit for bit in ("ord", "noise", "mask", "rank") if bit in name]
    if len(bits) >= 2:
        return "+".join(bits)
    if bits:
        return f"{bits[0]}_aux"
    if "snr" in name:
        return "snr_lambda"
    if "good" in name or "medium" in name or "ls" in name or "wd" in name:
        return "regularization"
    return "other"


def run_rows(root_out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_root = root_out / VARIANT / "models"
    if not model_root.exists():
        return rows
    for report in sorted(model_root.glob("*/test_report.json")):
        run_name = report.parent.name
        if "_hc" not in run_name:
            continue
        rep = load_json(report)
        if not rep:
            continue
        probe = load_json(report.parent / "probe_summary.json") or {}
        hp = probe.get("hyperparams", {})
        cm = rep.get("confusion_matrix_3x3", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        good, medium, bad = recalls(cm)
        rows.append(
            {
                "run": run_name,
                "family": family_for(run_name),
                "acc": float(rep.get("acc", 0.0)),
                "good": good,
                "medium": medium,
                "bad": bad,
                "hp": hp,
                "cm": cm,
            }
        )
    rows.sort(key=lambda row: row["acc"], reverse=True)
    return rows


def hp_desc(hp: dict[str, Any]) -> str:
    return ", ".join(
        [
            f"seed={hp.get('seed', '')}",
            f"lr={hp.get('lr', '')}",
            f"drop={hp.get('dropout', '')}",
            f"snr={hp.get('lambda_snr', '') if hp.get('snr_head', False) else False}",
            f"ord={hp.get('lambda_ord', '') if hp.get('ordinal_head', False) else False}",
            f"noise={hp.get('lambda_noise_type', '') if hp.get('noise_type_head', False) else False}",
            f"mask={hp.get('lambda_local_mask', '') if hp.get('local_mask_head', False) else False}",
            f"rank={hp.get('lambda_rank', '')}",
            f"den={hp.get('lambda_den', '')}",
            f"lvl={hp.get('lambda_lvl', '')}",
            f"uncert={hp.get('e_uncert', '')}",
            f"gw/mw/bw={hp.get('class_weight_good', '')}/{hp.get('class_weight_medium', '')}/{hp.get('class_weight_bad', '')}",
            f"ls={hp.get('label_smoothing', '')}",
            f"wd={hp.get('weight_decay', '')}",
        ]
    )


def family_rows(rows: list[dict[str, Any]]) -> list[tuple[str, int, float, float, float, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(row)

    out = []
    for family, vals in grouped.items():
        accs = [float(v["acc"]) for v in vals]
        best = max(vals, key=lambda v: float(v["acc"]))
        out.append(
            (
                family,
                len(vals),
                statistics.mean(accs),
                statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                max(accs),
                str(best["run"]),
            )
        )
    out.sort(key=lambda item: (item[4], item[2]), reverse=True)
    return out


def main() -> None:
    args = parse_args()
    root_out = Path(args.root_out)
    if not root_out.is_absolute():
        root_out = ROOT / root_out

    rows = run_rows(root_out)
    lines = [
        "# E3.11 Head Combination Grid",
        "",
        "This grid searches task-head combinations on the E3.11f mainline data, without changing the backbone or labels.",
        "The fixed strong recipe is CLS pooling, positional embedding, raw input, D1 warm-start, and validation-accuracy checkpoint selection.",
        "",
        "Rationale references used for the grid design:",
        "",
        "- MTECG: ECG segment tokens with learnable positional embeddings and masked-autoencoder style representation learning: https://arxiv.org/abs/2309.07136",
        "- UniTS: time-series models can share parameters across classification, imputation, and related tasks: https://github.com/mims-harvard/UniTS",
        "- SwinDAE: ECG quality assessment can benefit from pairing Transformer features with denoising-autoencoder objectives: https://pubmed.ncbi.nlm.nih.gov/37698969/",
        "",
        "Historical anchors:",
        "",
        "- E3.10 best single visual: `0.9402`",
        "- E3.11f best before head-combo grid: `0.9464` (`r3_lr625_seed1`)",
        "- E3.11f stable basin: `lr=5.75e-5`, dropout `0.10`, mean about `0.9423` across prior seeds",
        "",
        "## Top Runs",
        "",
        "| Rank | Run | Family | Test Acc | Recall G/M/B | Head/Loss Summary |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    if not rows:
        lines.append("|  | pending | pending |  |  |  |")
    else:
        for i, row in enumerate(rows[:30], start=1):
            lines.append(
                f"| {i} | `{row['run']}` | {row['family']} | {row['acc']:.4f} | "
                f"{row['good']:.4f}/{row['medium']:.4f}/{row['bad']:.4f} | {hp_desc(row['hp'])} |"
            )

    lines.extend(
        [
            "",
            "## Family Summary",
            "",
            "| Family | N | Mean Acc | Std | Best Acc | Best Run |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    fams = family_rows(rows)
    if not fams:
        lines.append("| pending |  |  |  |  |  |")
    else:
        for family, n, mean, std, best_acc, best_run in fams:
            lines.append(f"| {family} | {n} | {mean:.4f} | {std:.4f} | {best_acc:.4f} | `{best_run}` |")

    lines.extend(
        [
            "",
            "## Live Decision Rules",
            "",
            "- If SNR-only remains best, keep the model simple and spend future effort on data/source audit.",
            "- If ordinal improves medium recall without hurting bad recall, expand ordinal around `lambda_ord=0.03-0.05`.",
            "- If local mask helps, keep it at low weight only; the current target is injected-noise envelope, so high weights may overfit synthetic placement.",
            "- If denoise/level auxiliary improves classification by at least `0.003`, rerun the best denoise setting across multiple seeds.",
            "- If noise-type head is flat or negative, drop it; E3.11f uses only `em/ma/mix`, so texture supervision may be a distraction.",
            "",
        ]
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E3.11 head-combination grid.")
    parser.add_argument("--root_out", default="outputs/transformer_e311_mainline_strict")
    return parser.parse_args()


if __name__ == "__main__":
    main()
