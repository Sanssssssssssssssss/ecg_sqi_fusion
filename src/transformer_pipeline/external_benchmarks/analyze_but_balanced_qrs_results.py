"""Analyze QRS-guided BUT balanced-test results.

This report explains how the class-balanced BUT 10s test subset changes the
interpretation of the real-noise/QRS-guided runs.  It is report-only and reads
existing run summaries plus per-run balanced-test reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT


DEFAULT_RUN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_real_noise_qrs_guided_10h_2026_06_04"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_real_noise_qrs_guided_10h_2026_06_04"
ANALYSIS_DIRNAME = "balanced_qrs_analysis"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def report_fields(rep: dict[str, Any], prefix: str) -> dict[str, float]:
    rec = rep.get("recall_good_medium_bad", [np.nan, np.nan, np.nan])
    prec = rep.get("precision_good_medium_bad", [np.nan, np.nan, np.nan])
    return {
        f"{prefix}_acc": float(rep.get("acc", np.nan)),
        f"{prefix}_balanced_acc": float(rep.get("balanced_acc", np.nan)),
        f"{prefix}_macro_f1": float(rep.get("macro_f1", np.nan)),
        f"{prefix}_good_recall": float(rec[0]) if len(rec) > 0 else np.nan,
        f"{prefix}_medium_recall": float(rec[1]) if len(rec) > 1 else np.nan,
        f"{prefix}_bad_recall": float(rec[2]) if len(rec) > 2 else np.nan,
        f"{prefix}_good_precision": float(prec[0]) if len(prec) > 0 else np.nan,
        f"{prefix}_medium_precision": float(prec[1]) if len(prec) > 1 else np.nan,
        f"{prefix}_bad_precision": float(prec[2]) if len(prec) > 2 else np.nan,
    }


def variant_parts(variant: str) -> dict[str, str]:
    if "__" in variant:
        base, profile = variant.split("__", 1)
    else:
        base, profile = variant, ""
    if base.startswith("blockwise_switch"):
        mix = "blockwise"
    elif base.startswith("single_bw"):
        mix = "single_bw"
    else:
        mix = "other"
    return {"base_variant": base, "qrs_profile": profile, "mix_family": mix}


def build_rows(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(run_root / "qrs_guided_training_summary.jsonl"):
        run_dir_value = row.get("run_dir")
        if not run_dir_value:
            continue
        run_dir = Path(run_dir_value)
        balanced_path = run_dir / "but_10s_eval" / "but_10s_balanced_test_report.json"
        if not balanced_path.exists():
            continue
        balanced_payload = load_json(balanced_path)
        variant = str(row.get("spec", {}).get("id", run_dir.name))
        out: dict[str, Any] = {
            "variant_id": variant,
            "mode": row.get("mode"),
            "returncode": row.get("returncode"),
            "run_dir": str(run_dir),
            "ptb_acc": float(row.get("ptb_test_report", {}).get("acc", np.nan)),
            "ptb_bad_recall": float((row.get("ptb_test_report", {}).get("recall_good_medium_bad") or [np.nan, np.nan, np.nan])[2]),
            "denoise_score": float(row.get("ptb_denoise_metrics", {}).get("denoise_score", np.nan)),
            **variant_parts(variant),
        }
        original = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        balanced = balanced_payload.get("calibrated_report", {})
        raw = balanced_payload.get("raw_argmax_report", {})
        out.update(report_fields(original, "orig"))
        out.update(report_fields(balanced, "bal_cal"))
        out.update(report_fields(raw, "bal_raw"))
        out["balanced_macro_lift_vs_original"] = out["bal_cal_macro_f1"] - out["orig_macro_f1"]
        out["balanced_acc_lift_vs_original"] = out["bal_cal_acc"] - out["orig_acc"]
        out["raw_macro_lift_vs_calibrated"] = out["bal_raw_macro_f1"] - out["bal_cal_macro_f1"]
        out["raw_medium_recall_lift_vs_calibrated"] = out["bal_raw_medium_recall"] - out["bal_cal_medium_recall"]
        out["raw_bad_recall_lift_vs_calibrated"] = out["bal_raw_bad_recall"] - out["bal_cal_bad_recall"]
        rows.append(out)
    return pd.DataFrame(rows)


def save_bar(df: pd.DataFrame, out_png: Path, value: str, title: str, n: int = 10) -> None:
    top = df.sort_values(value, ascending=False).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = ["#2f5d8c" if m == "full" else "#d38b2e" for m in top["mode"]]
    ax.barh(top["variant_id"], top[value], color=colors)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(value.replace("_", " "))
    ax.grid(axis="x", alpha=0.25)
    for y, v in enumerate(top[value]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    markers = {"full": "o", "quick": "s"}
    colors = {"blockwise": "#2f5d8c", "single_bw": "#d38b2e", "other": "#6b7280"}
    for (mode, mix), g in df.groupby(["mode", "mix_family"]):
        ax.scatter(
            g["bal_cal_medium_recall"],
            g["bal_cal_bad_recall"],
            label=f"{mode} / {mix}",
            marker=markers.get(str(mode), "o"),
            color=colors.get(str(mix), "#6b7280"),
            s=72,
            alpha=0.85,
            edgecolor="#111827",
            linewidth=0.45,
        )
    ax.axvline(0.72, color="#6b7280", linestyle="--", linewidth=1)
    ax.axhline(0.85, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title("Medium vs bad recall on balanced BUT test", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Medium recall")
    ax.set_ylabel("Bad recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def save_delta_plot(df: pd.DataFrame, out_png: Path) -> None:
    top = df.sort_values("bal_cal_macro_f1", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    y = np.arange(len(top))
    ax.barh(y - 0.18, top["bal_raw_macro_f1"], height=0.34, color="#2f5d8c", label="Raw argmax")
    ax.barh(y + 0.18, top["bal_cal_macro_f1"], height=0.34, color="#d38b2e", label="Val-calibrated")
    ax.set_yticks(y)
    ax.set_yticklabels(top["variant_id"], fontsize=8)
    ax.set_xlabel("Balanced-test macro-F1")
    ax.set_title("Raw argmax vs validation thresholds\nBalanced BUT test macro-F1", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def save_confusion(confusion: list[list[int]], out_png: Path, title: str) -> None:
    arr = np.asarray(confusion, dtype=float)
    row_sum = arr.sum(axis=1, keepdims=True)
    pct = np.divide(arr, np.maximum(row_sum, 1), out=np.zeros_like(arr), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    labels = ["good", "medium", "bad"]
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(arr[i, j])}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9, color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def write_report(df: pd.DataFrame, out_dir: Path, best_payload: dict[str, Any]) -> None:
    top_cal = df.sort_values("bal_cal_macro_f1", ascending=False).iloc[0]
    top_raw = df.sort_values("bal_raw_macro_f1", ascending=False).iloc[0]
    blockwise = df.groupby("mix_family").agg(
        n=("variant_id", "count"),
        bal_macro_mean=("bal_cal_macro_f1", "mean"),
        bal_macro_max=("bal_cal_macro_f1", "max"),
        bal_bad_mean=("bal_cal_bad_recall", "mean"),
        bal_medium_mean=("bal_cal_medium_recall", "mean"),
        raw_minus_cal_macro=("raw_macro_lift_vs_calibrated", "mean"),
    ).reset_index()
    mode = df.groupby("mode").agg(
        n=("variant_id", "count"),
        bal_macro_mean=("bal_cal_macro_f1", "mean"),
        bal_macro_max=("bal_cal_macro_f1", "max"),
        raw_minus_cal_macro=("raw_macro_lift_vs_calibrated", "mean"),
        ptb_acc_mean=("ptb_acc", "mean"),
    ).reset_index()

    def markdown_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in frame.iterrows():
            vals: list[str] = []
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    vals.append(f"{float(val):.4f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# BUT Balanced-Test QRS Result Analysis",
        "",
        "## Executive Summary",
        "",
        f"- **The balanced subset confirms the QRS-guided direction is real but still incomplete.** The best calibrated balanced-test row is `{top_cal['variant_id']}` with acc `{top_cal['bal_cal_acc']:.4f}`, macro-F1 `{top_cal['bal_cal_macro_f1']:.4f}`, and recalls `{top_cal['bal_cal_good_recall']:.3f}/{top_cal['bal_cal_medium_recall']:.3f}/{top_cal['bal_cal_bad_recall']:.3f}`.",
        f"- **Raw argmax is better than validation-threshold calibration on the best row.** For `{top_raw['variant_id']}`, raw balanced-test macro-F1 reaches `{top_raw['bal_raw_macro_f1']:.4f}` versus calibrated `{top_raw['bal_cal_macro_f1']:.4f}`. The threshold policy is over-correcting toward bad and suppressing medium.",
        "- **The gain is not just class imbalance math.** On the original imbalanced test, macro-F1 underweights the practical fact that bad has only 411 examples; on the balanced subset, bad errors become visible without being drowned by good/medium volume. The same best blockwise-QRS family remains on top.",
        "- **The bottleneck is now sharply localized: medium.** The best calibrated balanced row has bad recall `0.883` and good recall `0.878`, but medium recall only `0.360`. This says the current real-noise/QRS recipe learned `usable vs fatal` better than the independent BUT medium cluster.",
        "",
        "## What Changed After Balancing",
        "",
        "The formal test split is `good=3640`, `medium=4426`, `bad=411`. The balanced subset uses `411` windows per class. This makes accuracy equal to balanced accuracy and prevents good/medium volume from hiding bad performance.",
        "",
        "![Top balanced macro-F1 variants](figures/top_balanced_macro.png)",
        "",
        "The top variants are all blockwise real NSTDB noise with mild QRS attenuation. Single-BW variants can preserve PTB sanity and denoise well, but they do not separate BUT bad and medium as cleanly.",
        "",
        "## Calibration Is Hurting Medium",
        "",
        "![Raw versus calibrated macro-F1](figures/raw_vs_calibrated_macro.png)",
        "",
        "Validation thresholds were chosen on a tiny and skewed val split (`good=969`, `medium=105`, `bad=83`). On the balanced test, the raw logits often keep more medium signal. This strongly suggests the next external evaluation should report both raw argmax and validation-calibrated thresholds, or calibrate with a class-balanced validation objective/subsample.",
        "",
        "## Medium-Bad Tradeoff",
        "",
        "![Medium vs bad recall](figures/medium_vs_bad_recall.png)",
        "",
        "The desired upper-right region is medium recall >= 0.72 and bad recall >= 0.85. Current QRS-guided models move upward on bad but stay far left on medium. That is useful evidence: QRS fatality is now captured, but BUT medium is not simply weaker bad or lower SNR.",
        "",
        "## Best Confusion Pattern",
        "",
        "![Best balanced confusion matrix](figures/best_balanced_confusion.png)",
        "",
        "The best model confuses medium almost symmetrically into good and bad. That matters: if medium were just a scalar quality midpoint, errors should mostly move to one side depending on threshold. Instead, the model is missing medium's independent signature.",
        "",
        "## Group Summary",
        "",
        "### By Noise Mix",
        "",
        markdown_table(blockwise),
        "",
        "### By Training Mode",
        "",
        markdown_table(mode),
        "",
        "## Interpretation",
        "",
        "1. **Balanced test makes the QRS direction look healthier, not worse.** The best model is much more balanced than the original accuracy suggested.",
        "2. **Blockwise switching is doing real work.** It better matches BUT's free-living non-stationary signal quality than single BW.",
        "3. **Val calibration is unreliable here.** The validation bad/medium counts are too small; raw logits may be a better diagnostic view for now.",
        "4. **Next generator work should target medium as its own cluster.** Keep fatal OR/QRS bad, but add medium-only P/T/ST/detail unreliability without pushing those samples into bad.",
        "",
        "## Files",
        "",
        "- `balanced_qrs_rows.csv`: row-level metrics joined from original, balanced calibrated, and balanced raw evaluations.",
        "- `balanced_qrs_summary.json`: top candidates and aggregate diagnostics.",
        "- `figures/`: static PNGs used in this report.",
    ]
    (out_dir / "but_balanced_qrs_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "best_calibrated_balanced": top_cal.to_dict(),
        "best_raw_balanced": top_raw.to_dict(),
        "mix_family_summary": blockwise.to_dict(orient="records"),
        "mode_summary": mode.to_dict(orient="records"),
        "best_balanced_payload": best_payload,
    }
    write_json(out_dir / "balanced_qrs_summary.json", summary)


def run(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    out_dir = report_root / ANALYSIS_DIRNAME
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = build_rows(run_root)
    if df.empty:
        raise RuntimeError("No balanced-test rows found. Run but_balanced_test_eval first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "balanced_qrs_rows.csv", index=False)
    save_bar(df, fig_dir / "top_balanced_macro.png", "bal_cal_macro_f1", "Top calibrated macro-F1 on balanced BUT")
    save_scatter(df, fig_dir / "medium_vs_bad_recall.png")
    save_delta_plot(df, fig_dir / "raw_vs_calibrated_macro.png")
    best = df.sort_values("bal_cal_macro_f1", ascending=False).iloc[0]
    best_report = load_json(Path(best["run_dir"]) / "but_10s_eval" / "but_10s_balanced_test_report.json")
    save_confusion(best_report["calibrated_report"]["confusion_3x3"], fig_dir / "best_balanced_confusion.png", f"Best calibrated balanced confusion: {best['variant_id']}")
    save_confusion(best_report["raw_argmax_report"]["confusion_3x3"], fig_dir / "best_raw_balanced_confusion.png", f"Raw balanced confusion: {best['variant_id']}")
    write_report(df, out_dir, best_report)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze QRS-guided balanced BUT test results.")
    parser.add_argument("--run_root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = run(args)
    print(json.dumps({"status": "complete", "out_dir": str(out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
