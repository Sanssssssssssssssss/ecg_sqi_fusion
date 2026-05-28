from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .constants import BASELINE_ACC, CURRENT_BEST_ACC, CURRENT_BEST_BAD_RECALL, CURRENT_BEST_MEDIUM_RECALL
from .constants import DEFAULT_OUT_ROOT, PACKAGE_REPORT, TOP_REPORT
from .recipes import RECIPES


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def decision(report: dict[str, Any]) -> str:
    test = report.get("test", {})
    acc = float(test.get("acc", 0.0))
    rec = test.get("per_class_recall", {})
    med = float(rec.get("medium", 0.0))
    bad = float(rec.get("bad", 0.0))
    if acc >= CURRENT_BEST_ACC:
        return "expand seeds: beats/equals local-mask candidate"
    if med >= CURRENT_BEST_MEDIUM_RECALL + 0.01 and bad >= CURRENT_BEST_BAD_RECALL - 0.005:
        return "expand seeds: improves medium without hurting bad"
    if acc >= BASELINE_ACC:
        return "watch: not below strong baseline"
    return "stop unless curve/grad norms explain a useful failure"


def read_rows(out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, recipes in RECIPES.items():
        for recipe in recipes:
            run_dir = out_root / group / recipe["name"]
            report = load_json(run_dir / "test_report.json")
            log = load_json(run_dir / "train_log.json")
            warm = load_json(run_dir / "warmstart_report.json")
            if report is None:
                rows.append(
                    {
                        "group": group,
                        "name": recipe["name"],
                        "status": "pending",
                        "description": recipe.get("description", ""),
                    }
                )
                continue
            test = report["test"]
            rec = test.get("per_class_recall", {})
            last_grad = {}
            if isinstance(log, list) and log:
                for row in reversed(log):
                    tr = row.get("train", {})
                    if "grad_cls" in tr:
                        last_grad = {
                            "cls": tr.get("grad_cls", 0.0),
                            "denoise": tr.get("grad_denoise", 0.0),
                            "level": tr.get("grad_level", 0.0),
                        }
                        break
            rows.append(
                {
                    "group": group,
                    "name": recipe["name"],
                    "status": "done",
                    "acc": float(test.get("acc", 0.0)),
                    "good": float(rec.get("good", 0.0)),
                    "medium": float(rec.get("medium", 0.0)),
                    "bad": float(rec.get("bad", 0.0)),
                    "best_epoch": report.get("best_epoch"),
                    "decision": decision(report),
                    "description": recipe.get("description", ""),
                    "grad": last_grad,
                    "warm_loaded": (warm or report.get("warmstart", {})).get("loaded"),
                    "warm_skipped": (warm or report.get("warmstart", {})).get("skipped_count"),
                }
            )
    return rows


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    if x is None:
        return ""
    return str(x)


def render(rows: list[dict[str, Any]]) -> str:
    done = [r for r in rows if r["status"] == "done"]
    best = max(done, key=lambda r: r.get("acc", -1.0), default=None)
    lines: list[str] = []
    lines.append("# E3.11 Isolated SQI Research Ablation")
    lines.append("")
    lines.append("This report is generated from the isolated package `src/experiment/e311_sqi_research/`.")
    lines.append("")
    lines.append("## Fixed References")
    lines.append("")
    lines.append(f"- Strong baseline clone target: `{BASELINE_ACC:.4f}`.")
    lines.append(f"- Current highest candidate: mainline local-mask+rank result `{CURRENT_BEST_ACC:.4f}`.")
    lines.append("- Dataset: `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph`.")
    lines.append("- Warm-start: D1 CLS checkpoint.")
    lines.append("")
    lines.append("## Queue And Promotion Plan")
    lines.append("")
    lines.append("Submitted on 2026-05-28 after the loss-scale fix:")
    lines.append("")
    lines.append("| job id | group | array | purpose |")
    lines.append("|---:|---|---|---|")
    lines.append("| `29751790` | `focused_tuning` | `0-5%1` | low-weight local/level follow-up around the strongest mainline signal |")
    lines.append("| `29751791` | `head_reimpl` | `0-4%1` | baseline, interpretable SQI head, local quality v2, combo head, multiscale |")
    lines.append("| `29754081` | `sqi_head_tuning` | `0-15%2` | projected deterministic/predicted SQI stats injected into the classifier head |")
    lines.append("| `29756103` | `sqi_head_mlp_tuning` | `0-11%2` | SQI-injected head plus a tiny one-hidden-layer classifier MLP |")
    lines.append("| `29767663` | `sqi_residual_tuning` | `0-9%2` | submitted then canceled before start to prioritize direct MIL SQI MLP tuning |")
    lines.append("| `29770007` | `sqi_mil_mlp_refined` | `0-15%2` | refined tuning around the best MIL SQI + small MLP head |")
    lines.append("| `29752152` | `loss_conflict` | `0-4%1` | CE-only and multi-task weighting conflict screen |")
    lines.append("| `29752153` | `target_gate_reimpl` | `0-5%1` | clean-RR targets, bad fallback target, denoise gates |")
    lines.append("| `29752154` | `generalization_loss` | `0-6%1` | label smoothing, focal, ordinal, R-Drop, SAM |")
    lines.append("")
    lines.append("A recipe can move into the mainline sweep only when it:")
    lines.append("")
    lines.append(f"- Beats the current candidate (`{CURRENT_BEST_ACC:.4f}` test acc), or")
    lines.append(f"- Improves medium recall by at least `0.01` while keeping bad recall within `0.005` of `{CURRENT_BEST_BAD_RECALL:.4f}`, or")
    lines.append(f"- Stays near the strong baseline (`>= {BASELINE_ACC:.4f}`) and clearly reduces multi-task gradient conflict, in which case it must be rerun with seeds `0/1/2/3` before promotion.")
    lines.append("")
    lines.append("Do not promote recipes that only improve auxiliary denoise/level metrics while lowering classification accuracy.")
    lines.append("")
    lines.append("## Implementation Audit Notes")
    lines.append("")
    lines.append("- `baseline_clone` and all fixed-weight recipes now match the mainline loss scale (`cls_weight=10`) and cosine LR schedule; earlier pre-fix research jobs over-weighted SNR/level/local auxiliaries relative to classification.")
    lines.append("- `lc_uncert_multitask` now follows Kendall-style uncertainty weighting more closely: after warmup it learns from raw CE/denoise/level losses instead of pre-scaled fixed weights.")
    lines.append("- `tg_patch_residual_level` now uses squared residual with a dataset-level p99 scale, so the target keeps absolute severity and no longer inverts bad-vs-medium supervision.")
    lines.append("- `sqi_interpretable` and `sqi_local` now include an estimated SNR-style feature from signal/residual power instead of a duplicated residual mean.")
    lines.append("- `multiscale_sqi_transformer` now uses PatchTST-style `unfold -> Linear(patch)` tokenizers for extra scales instead of padded Conv1d tokenizers.")
    lines.append("- `gl_sam` now freezes BatchNorm running-stat updates during the second SAM forward pass, matching common SAM practice for models with BatchNorm layers.")
    lines.append("- `sqi_head_tuning` keeps the model simple: it projects a small deterministic/predicted SQI-stat vector with one `LayerNorm -> Linear(32) -> GELU` block, then concatenates it with the CLS token.")
    lines.append("")
    if best:
        lines.append("## Current Best")
        lines.append("")
        lines.append(
            f"- `{best['group']}/{best['name']}`: acc `{best['acc']:.4f}`, "
            f"recall good/medium/bad `{best['good']:.4f}/{best['medium']:.4f}/{best['bad']:.4f}`."
        )
        lines.append(f"- Decision: {best['decision']}.")
        lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| group | run | status | acc | good | medium | bad | best epoch | decision |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if r["status"] == "pending":
            lines.append(f"| {r['group']} | {r['name']} | pending |  |  |  |  |  |  |")
        else:
            lines.append(
                f"| {r['group']} | {r['name']} | done | {fmt(r['acc'])} | {fmt(r['good'])} | "
                f"{fmt(r['medium'])} | {fmt(r['bad'])} | {fmt(r['best_epoch'])} | {r['decision']} |"
            )
    lines.append("")
    lines.append("## Gradient-Norm Snapshot")
    lines.append("")
    lines.append("| group | run | cls | denoise | level |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in done:
        g = r.get("grad", {})
        lines.append(f"| {r['group']} | {r['name']} | {fmt(g.get('cls'))} | {fmt(g.get('denoise'))} | {fmt(g.get('level'))} |")
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `loss_conflict` tests whether auxiliary dense losses help or fight classification.")
    lines.append("- `head_reimpl` tests whether classification improves when it sees local residual/level summaries instead of a token alone.")
    lines.append("- `sqi_head_tuning` is the focused version of that idea: deterministic input SQI stats, predicted residual/level stats, detached variants, and top-k patch MIL summaries.")
    lines.append("- `target_gate_reimpl` checks whether RR targets and denoise gates cover bad samples more reliably.")
    lines.append("- `generalization_loss` checks whether boundary-friendly losses improve medium without sacrificing bad.")
    lines.append("- `focused_tuning` follows up on early results by lowering local/level supervision weights instead of changing data or the mainline model.")
    lines.append("")
    lines.append("Runs only become candidates for seed expansion if they beat the current mainline best, improve medium recall with minimal bad loss, or match the strong baseline while reducing task-conflict evidence.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize isolated E3.11 SQI research experiments")
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--package_report", type=Path, default=PACKAGE_REPORT)
    parser.add_argument("--top_report", type=Path, default=TOP_REPORT)
    args = parser.parse_args()
    rows = read_rows(args.out_root)
    text = render(rows)
    args.package_report.parent.mkdir(parents=True, exist_ok=True)
    args.package_report.write_text(text, encoding="utf-8")
    args.top_report.parent.mkdir(parents=True, exist_ok=True)
    args.top_report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
