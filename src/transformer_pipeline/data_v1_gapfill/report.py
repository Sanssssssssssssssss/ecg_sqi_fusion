from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import POLICY, ROOT, protocol_dir, report_dir, split_dir


ORDER = ["good", "medium", "bad"]
TYPES = ["original_but", "but_native_morph", "ptb_morph", "clean_style"]
DRIFT_FEATURES = [
    "raw_rms",
    "raw_ptp_p99_p01",
    "raw_diff_abs_p95",
    "sqi_bSQI",
    "sqi_iSQI",
    "band_5_15",
    "band_15_30",
    "hf_ratio",
    "qrs_visibility",
    "template_corr",
    "detector_agreement",
    "pca_margin",
    "knn_label_purity",
]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs) if path.exists() else pd.DataFrame()


def table(df: pd.DataFrame) -> str:
    return "_missing_" if df.empty else df.to_string(index=True)


def composition(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return pd.crosstab(frame["class_name"], frame["v116_candidate_type"]).reindex(index=ORDER, columns=TYPES, fill_value=0)


def pct(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    return (comp.div(comp.sum(axis=1).replace(0, pd.NA), axis=0) * 100).round(2)


def generated_gap(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    gen = frame.loc[frame["class_name"].isin(["medium", "bad"]) & ~frame["v116_candidate_type"].eq("original_but")]
    comp = pd.crosstab(gen["class_name"], gen["v116_candidate_type"]).reindex(index=["medium", "bad"], columns=["but_native_morph", "ptb_morph", "clean_style"], fill_value=0)
    share = pct(comp)
    out = comp.copy()
    for col in comp.columns:
        out[f"{col}_pct"] = share[col]
    return out


def pool_capacity() -> pd.DataFrame:
    trace = read_csv(report_dir() / f"{POLICY}_selector_trace.csv")
    if trace.empty or "particle" not in trace.columns:
        return pd.DataFrame()
    final = trace.loc[pd.to_numeric(trace["particle"], errors="coerce").eq(-1)].copy()
    cols = ["class_name", "gap_fill_component", "selected_n", "pool_n", "rbf_mmd", "sym_domain_auc", "pca_density_overlap"]
    final = final[[c for c in cols if c in final.columns]]
    if "pool_n" in final.columns and "selected_n" in final.columns:
        final["pool_to_selected"] = (final["pool_n"] / final["selected_n"].replace(0, pd.NA)).round(2)
    return final


def metrics() -> pd.DataFrame:
    frame = read_csv(report_dir() / f"{POLICY}_global_distribution_metrics.csv")
    if frame.empty:
        return frame
    cols = ["scope", "but_n", "synthetic_n", "rbf_mmd", "sliced_wasserstein", "quantile_loss", "sym_domain_auc", "pca_density_overlap"]
    return frame[[c for c in cols if c in frame.columns]]


def raw_alias_drift(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for name, left, right in [
        ("raw_rms_vs_rms", "raw_rms", "rms"),
        ("raw_ptp_vs_ptp", "raw_ptp_p99_p01", "ptp_p99_p01"),
        ("raw_diff_vs_non_qrs_diff", "raw_diff_abs_p95", "non_qrs_diff_p95"),
    ]:
        if left not in frame.columns or right not in frame.columns:
            rows.append({"check": name, "max_abs_delta": float("inf")})
            continue
        delta = (pd.to_numeric(frame[left], errors="coerce") - pd.to_numeric(frame[right], errors="coerce")).abs()
        rows.append({"check": name, "max_abs_delta": float(delta.max(skipna=True))})
    return pd.DataFrame(rows)


def feature_drift(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    features = [c for c in DRIFT_FEATURES if c in frame.columns]
    rows: list[dict[str, Any]] = []
    for cls in ["medium", "bad"]:
        orig = frame.loc[frame["class_name"].eq(cls) & frame["v116_candidate_type"].eq("original_but")]
        if orig.empty:
            continue
        center = orig[features].median(numeric_only=True)
        scale = (orig[features].quantile(0.75) - orig[features].quantile(0.25)).replace(0, 1.0)
        for typ in ["but_native_morph", "ptb_morph", "clean_style"]:
            part = frame.loc[frame["class_name"].eq(cls) & frame["v116_candidate_type"].eq(typ)]
            if part.empty:
                continue
            drift = ((part[features].median(numeric_only=True) - center).abs() / scale).sort_values(ascending=False).head(5)
            rows.append(
                {
                    "class_name": cls,
                    "candidate_type": typ,
                    "rows": int(len(part)),
                    "top_drift": "; ".join(f"{name}={value:.3f}" for name, value in drift.items()),
                }
            )
    return pd.DataFrame(rows)


def split_summary(split: pd.DataFrame) -> pd.DataFrame:
    if split.empty:
        return pd.DataFrame()
    return (
        split.loc[split["split"].isin(["train", "val", "test"])]
        .groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "val", "test"], columns=ORDER, fill_value=0)
    )


def write_report() -> Path:
    atlas = read_csv(protocol_dir() / "original_region_atlas.csv")
    split = read_csv(split_dir() / "original_region_atlas.csv")
    comp = composition(atlas)
    train_comp = composition(split.loc[split["split"].eq("train")]) if not split.empty else pd.DataFrame()
    val_test_generated = 0
    if not split.empty:
        val_test_generated = int((split["split"].isin(["val", "test"]) & ~split["v116_candidate_type"].eq("original_but")).sum())
    out = ROOT / "docs" / "data_v1_fit_report.md"
    fig_dir = ROOT / "docs" / "data_v1_figures"
    lines = [
        "# Data v1 Fit Report",
        "",
        f"- policy: `{POLICY}`",
        f"- protocol path: `{protocol_dir()}`",
        f"- split path: `{split_dir()}`",
        f"- selector runtime: CPU numpy/sklearn; `--device auto` is recorded but not used by the current data selector.",
        "",
        "## Protocol Counts",
        "",
        table(comp),
        "",
        "## Protocol Percent By Class",
        "",
        table(pct(comp)),
        "",
        "## Generated Gap Composition",
        "",
        table(generated_gap(atlas)),
        "",
        "## Split Counts",
        "",
        table(split_summary(split)),
        "",
        "## Train Composition",
        "",
        table(train_comp),
        "",
        "## Train Percent By Class",
        "",
        table(pct(train_comp)),
        "",
        "## Pool Capacity And Selector Result",
        "",
        table(pool_capacity()),
        "",
        "## Distribution Metrics",
        "",
        table(metrics()),
        "",
        "## Raw Feature Alias Consistency",
        "",
        table(raw_alias_drift(atlas)),
        "",
        "## Top Feature Drift Versus Original",
        "",
        table(feature_drift(atlas)),
        "",
        "## Gates",
        "",
        f"- protocol rows: `{len(atlas) if not atlas.empty else 'missing'}`",
        f"- val/test generated rows: `{val_test_generated}`",
        "- final selected count is hard-gated by selector; insufficient pools raise instead of silently underfilling.",
        "",
        "## Figures",
        "",
    ]
    if fig_dir.exists():
        for path in sorted(fig_dir.glob("*.png")):
            lines.append(f"- `{path}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return out


def main() -> None:
    write_report()
