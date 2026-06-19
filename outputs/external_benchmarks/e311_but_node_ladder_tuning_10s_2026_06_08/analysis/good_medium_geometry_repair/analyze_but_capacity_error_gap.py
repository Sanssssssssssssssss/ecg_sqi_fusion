"""Analyze BUT-only capacity prediction failures by record/region/features.

Report-only diagnostic. This script never trains a model and never uses BUT for
formal selection. It explains where a BUT-train capacity model fails on held-out
BUT test rows.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
SIGNALS_PATH = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_protocol_adaptation_2026_06_03"
    / "protocols"
    / "p1_current_10s_center"
    / "signals.npz"
)

SKIP_COLS = {
    "idx",
    "split",
    "y",
    "class_name",
    "record_id",
    "subject_id",
    "capacity_bucket",
    "true_int",
    "pred_int",
    "true_label",
    "pred_label",
    "correct",
    "prob_good",
    "prob_medium",
    "prob_bad",
    "nearest_other_class",
    "is_clean_strict",
    "is_clean_train_target",
    "clean_tier",
    "ambiguous_type",
    "original_region",
}
PANEL_FEATURES = [
    "sqi_basSQI",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "qrs_band_ratio",
    "non_qrs_diff_p95",
    "amplitude_entropy",
]


def metric_block(frame: pd.DataFrame) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {"n": int(len(frame))}
    if frame.empty:
        out.update({"acc": np.nan, "good_recall": np.nan, "medium_recall": np.nan, "bad_recall": np.nan})
        return out
    out["acc"] = float(frame["correct"].mean())
    for cls in ["good", "medium", "bad"]:
        sub = frame[frame["true_label"].eq(cls)]
        out[f"{cls}_n"] = int(len(sub))
        out[f"{cls}_recall"] = float(sub["correct"].mean()) if len(sub) else np.nan
    return out


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in SKIP_COLS:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() >= 30:
            cols.append(col)
    return cols


def feature_gap_rows(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    test = frame[frame["capacity_bucket"].eq("but_test")].copy()
    for cls in ["good", "medium", "bad"]:
        cls_frame = test[test["true_label"].eq(cls)]
        correct = cls_frame[cls_frame["correct"].astype(bool)]
        wrong = cls_frame[~cls_frame["correct"].astype(bool)]
        if len(correct) < 20 or len(wrong) < 20:
            continue
        for col in cols:
            a = pd.to_numeric(correct[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            b = pd.to_numeric(wrong[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            if len(a) < 20 or len(b) < 20:
                continue
            rows.append(
                {
                    "class_name": cls,
                    "feature": col,
                    "ks_correct_vs_wrong": float(ks_2samp(a, b).statistic),
                    "correct_median": float(np.median(a)),
                    "wrong_median": float(np.median(b)),
                    "delta_wrong_minus_correct": float(np.median(b) - np.median(a)),
                    "correct_p10": float(np.quantile(a, 0.10)),
                    "correct_p90": float(np.quantile(a, 0.90)),
                    "wrong_p10": float(np.quantile(b, 0.10)),
                    "wrong_p90": float(np.quantile(b, 0.90)),
                    "n_correct": int(len(a)),
                    "n_wrong": int(len(b)),
                }
            )
    return pd.DataFrame(rows).sort_values(["class_name", "ks_correct_vs_wrong"], ascending=[True, False])


def transition_counts(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["capacity_bucket", "record_id", "original_region", "true_label", "pred_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["capacity_bucket", "n"], ascending=[True, False])
    )


def render_table(frame: pd.DataFrame, limit: int = 20) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{float(x):.6g}")
        else:
            view[col] = view[col].astype(str)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_protocol_signals() -> np.ndarray | None:
    if not SIGNALS_PATH.exists():
        return None
    data = np.load(SIGNALS_PATH)
    key = "X" if "X" in data.files else data.files[0]
    signals = data[key].astype(np.float32)
    if signals.ndim == 3:
        signals = signals[:, 0, :]
    return signals


def short_feature_text(row: pd.Series) -> str:
    parts: list[str] = []
    for feat in PANEL_FEATURES:
        if feat not in row:
            continue
        val = pd.to_numeric(pd.Series([row[feat]]), errors="coerce").iloc[0]
        if pd.notna(val) and np.isfinite(float(val)):
            parts.append(f"{feat}={float(val):.3g}")
    return " | ".join(parts[:4])


def pick_panel_rows(frame: pd.DataFrame, max_per_group: int = 4) -> pd.DataFrame:
    test = frame[frame["capacity_bucket"].eq("but_test")].copy()
    test["transition"] = test["true_label"].astype(str) + "->" + test["pred_label"].astype(str)
    priority = [
        "medium->good",
        "good->medium",
        "bad->good",
        "bad->medium",
        "medium->bad",
        "good->bad",
    ]
    picked: list[pd.DataFrame] = []
    for trans in priority:
        sub = test[test["transition"].eq(trans)].copy()
        if sub.empty:
            continue
        if trans == "medium->good":
            sub = sub.sort_values(["record_id", "original_region", "prob_good"], ascending=[True, True, False])
        elif trans == "good->medium":
            sub = sub.sort_values(["record_id", "original_region", "prob_medium"], ascending=[True, True, False])
        elif trans.startswith("bad->"):
            sub = sub.sort_values(["record_id", "original_region", "prob_bad"], ascending=[True, True, True])
        else:
            sub = sub.sort_values(["record_id", "original_region"])
        picked.append(sub.head(max_per_group))
    if not picked:
        return pd.DataFrame()
    return pd.concat(picked, ignore_index=True)


def plot_error_waveforms(frame: pd.DataFrame, tag: str, out_dir: Path, report_dir: Path) -> tuple[Path | None, Path | None]:
    signals = load_protocol_signals()
    rows = pick_panel_rows(frame)
    if signals is None or rows.empty:
        return None, None
    n = len(rows)
    cols = 4
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.1, rows_n * 2.25), squeeze=False)
    t = np.linspace(0.0, 10.0, signals.shape[1], endpoint=False)
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        idx = int(row["idx"])
        if idx < 0 or idx >= len(signals):
            ax.axis("off")
            continue
        y = signals[idx].astype(float)
        y = y - np.nanmedian(y)
        scale = np.nanpercentile(np.abs(y), 95)
        if np.isfinite(scale) and scale > 0:
            y = y / scale
        ax.plot(t, y, color="#1f2937", lw=0.8)
        title = (
            f"{row['true_label']}->{row['pred_label']} | rec {row['record_id']} | {row['original_region']}\n"
            f"p={float(row['prob_good']):.2f}/{float(row['prob_medium']):.2f}/{float(row['prob_bad']):.2f}"
        )
        ax.set_title(title, fontsize=8)
        ax.text(
            0.01,
            0.02,
            short_feature_text(row),
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none", "pad": 1.5},
        )
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.15)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("BUT held-out error waveform panels (normalized for visual comparison)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = out_dir / f"{tag}_test_error_waveform_panels.png"
    report_path = report_dir / f"{tag}_test_error_waveform_panels.png"
    fig.savefig(out_path, dpi=160)
    fig.savefig(report_path, dpi=160)
    plt.close(fig)
    rows.to_csv(out_dir / f"{tag}_test_error_waveform_panel_rows.csv", index=False)
    return out_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    frame = pd.read_csv(pred_path)
    out_dir = ANALYSIS_DIR / "but_capacity_error_gap"
    report_dir = REPORT_DIR / "but_capacity_error_gap"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or pred_path.stem

    summary_rows = []
    for keys in [
        ["capacity_bucket"],
        ["capacity_bucket", "record_id"],
        ["capacity_bucket", "record_id", "original_region"],
        ["capacity_bucket", "record_id", "original_region", "true_label"],
    ]:
        for values, sub in frame.groupby(keys, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            row = {key: value for key, value in zip(keys, values)}
            row.update(metric_block(sub))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    transitions = transition_counts(frame)
    gaps = feature_gap_rows(frame, numeric_columns(frame))
    test_errors = frame[frame["capacity_bucket"].eq("but_test") & ~frame["correct"].astype(bool)].copy()
    test_errors = test_errors.sort_values(["true_label", "record_id", "original_region", "prob_bad"], ascending=[True, True, True, False])

    summary_path = out_dir / f"{tag}_summary_by_bucket_record_region.csv"
    transition_path = out_dir / f"{tag}_transitions.csv"
    gap_path = out_dir / f"{tag}_test_error_feature_gaps.csv"
    errors_path = out_dir / f"{tag}_test_errors.csv"
    summary.to_csv(summary_path, index=False)
    transitions.to_csv(transition_path, index=False)
    gaps.to_csv(gap_path, index=False)
    test_errors.to_csv(errors_path, index=False)
    waveform_path, waveform_report_path = plot_error_waveforms(frame, tag, out_dir, report_dir)

    top_test = summary[summary["capacity_bucket"].eq("but_test")].sort_values("n", ascending=False)
    report = "\n".join(
        [
            f"# BUT Capacity Error Gap: {tag}",
            "",
            "Report-only diagnostic. This is not a formal model-selection result.",
            "",
            "## Largest Test Buckets",
            "",
            render_table(top_test, 25),
            "",
            "## Top Test Error Transitions",
            "",
            render_table(transitions[transitions["capacity_bucket"].eq("but_test")], 25),
            "",
            "## Top Correct-vs-Wrong Feature Gaps",
            "",
            render_table(gaps, 35),
            "",
            f"Summary CSV: `{summary_path}`",
            f"Transitions CSV: `{transition_path}`",
            f"Feature gaps CSV: `{gap_path}`",
            f"Test errors CSV: `{errors_path}`",
            f"Waveform panel: `{waveform_report_path}`" if waveform_report_path else "Waveform panel: _not generated_",
            "",
        ]
    )
    report_path = report_dir / f"{tag}_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
