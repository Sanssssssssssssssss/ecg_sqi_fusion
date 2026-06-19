"""Analyze original BUT train/val/test split shift by class and feature.

This is report-only diagnostics. It does not train or select models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"


def numeric_cols(df: pd.DataFrame) -> list[str]:
    skip = {"idx", "split", "class_name", "original_region", "record", "record_id", "file", "source"}
    cols = []
    for col in df.columns:
        if col in skip:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 20:
            cols.append(col)
    return cols


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.copy()
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


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ATLAS)
    if "record_id" not in df.columns:
        for cand in ["record", "file", "source_record"]:
            if cand in df.columns:
                df["record_id"] = df[cand].astype(str)
                break
        else:
            df["record_id"] = "unknown"
    cols = numeric_cols(df)
    counts = (
        df.groupby(["split", "class_name", "original_region", "record_id"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["split", "class_name", "n"], ascending=[True, True, False])
    )
    rows = []
    for cls in ["good", "medium", "bad"]:
        train = df[(df["split"].eq("train")) & (df["class_name"].eq(cls))]
        test = df[(df["split"].eq("test")) & (df["class_name"].eq(cls))]
        for col in cols:
            a = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
            b = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
            if len(a) < 20 or len(b) < 20:
                continue
            ks = float(ks_2samp(a, b).statistic)
            rows.append(
                {
                    "class_name": cls,
                    "feature": col,
                    "ks_train_vs_test": ks,
                    "train_median": float(np.median(a)),
                    "test_median": float(np.median(b)),
                    "delta_median": float(np.median(b) - np.median(a)),
                    "train_p10": float(np.quantile(a, 0.10)),
                    "train_p90": float(np.quantile(a, 0.90)),
                    "test_p10": float(np.quantile(b, 0.10)),
                    "test_p90": float(np.quantile(b, 0.90)),
                    "n_train": int(len(a)),
                    "n_test": int(len(b)),
                }
            )
    gap = pd.DataFrame(rows).sort_values(["class_name", "ks_train_vs_test"], ascending=[True, False])
    counts_path = ANALYSIS_DIR / "original_split_shift_counts.csv"
    gap_path = ANALYSIS_DIR / "original_split_shift_feature_ks.csv"
    counts.to_csv(counts_path, index=False)
    gap.to_csv(gap_path, index=False)
    report_lines = [
        "# Original BUT Split Shift Diagnostic",
        "",
        "Report-only diagnostic. No model training or selection uses these results.",
        "",
        "## Counts By Split/Class/Region/Record",
        "",
        md_table(counts.head(40)),
        "",
        "## Top Train-vs-Test Feature Gaps",
        "",
    ]
    for cls in ["good", "medium", "bad"]:
        top = gap[gap["class_name"].eq(cls)].head(12)
        report_lines.extend([f"### {cls}", "", md_table(top), ""])
    report_lines.extend([f"Counts CSV: `{counts_path}`", f"Feature KS CSV: `{gap_path}`", ""])
    report = "\n".join(report_lines)
    (REPORT_DIR / "original_split_shift_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
