"""Visualize the clean-BUT policy distribution and interpretability contract.

Experiment-only reporting script. It reads the fixed-10s BUT margin table and
the original-region atlas, then writes static PNG figures plus a compact
markdown report. It does not modify source pipeline code or model checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "clean_but_policy_distribution"
MARGIN_PATH = ANALYSIS_DIR / "clean_but_window_policy" / "current_10s_window_segment_margins.csv"
ATLAS_PATH = REPORT_ROOT / "original_region_boundary" / "original_region_atlas.csv"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "good": "#5477C4",
    "medium": "#B8A037",
    "bad": "#CC6F47",
    "clean_core": "#5477C4",
    "good_medium_overlap": "#B8A037",
    "outlier_low_confidence": "#CC6F47",
    "right_bad_island": "#8A3A6F",
    "near_bad_boundary": "#804126",
    "medium_bad_overlap": "#71B436",
    "kept": "#5477C4",
    "dropped": "#CC6F47",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "text.color": TOKENS["ink"],
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.02, 0.952, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_data() -> pd.DataFrame:
    margin = pd.read_csv(MARGIN_PATH)
    atlas = pd.read_csv(ATLAS_PATH)
    atlas = atlas.drop_duplicates("idx").set_index("idx")
    df = margin.copy()
    df["idx"] = df["idx"].astype(int)
    feature_cols = [
        "pc1",
        "pc2",
        "qrs_visibility",
        "qrs_band_ratio",
        "sqi_basSQI",
        "baseline_step",
        "flatline_ratio",
        "non_qrs_diff_p95",
        "detector_agreement",
        "amplitude_entropy",
    ]
    for col in feature_cols:
        if col in atlas.columns:
            df[col] = df["idx"].map(atlas[col])
    df["matched_segment"] = df["matched_segment"].fillna(False).astype(bool)
    df["original_region"] = df["original_region"].fillna("unknown").astype(str)
    df["ambiguous_type"] = df["ambiguous_type"].fillna("unknown").astype(str)
    df["y_class"] = df["y_class"].fillna("unknown").astype(str)
    df["split"] = df["split"].fillna("unknown").astype(str)
    df["min_margin_sec"] = pd.to_numeric(df["min_margin_sec"], errors="coerce").fillna(-1.0)
    df["segment_duration_sec"] = pd.to_numeric(df["segment_duration_sec"], errors="coerce")
    return df


def policy_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    base = df["matched_segment"]
    return {
        "Full 10s protocol": base,
        "Margin >=5s, keep outlier": base & df["min_margin_sec"].ge(5.0),
        "Margin >=5s, drop outlier": base & df["min_margin_sec"].ge(5.0) & df["original_region"].ne("outlier_low_confidence"),
        "Margin >=10s, drop outlier": base & df["min_margin_sec"].ge(10.0) & df["original_region"].ne("outlier_low_confidence"),
        "Segment 10-60s, keep outlier": base
        & df["segment_duration_sec"].ge(10.0)
        & df["segment_duration_sec"].le(60.0),
    }


def count_summary(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, mask in masks.items():
        sub = df[mask]
        for split_name, frame in [("all", sub), ("test", sub[sub["split"].eq("test")])]:
            row = {
                "policy": policy,
                "split_scope": split_name,
                "n": int(len(frame)),
                "record_count": int(frame["record_id"].nunique()) if len(frame) else 0,
            }
            for cls in ["good", "medium", "bad"]:
                row[cls] = int(frame["y_class"].eq(cls).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def plot_policy_counts(summary: pd.DataFrame, out: Path) -> None:
    all_rows = summary[summary["split_scope"].eq("all")].copy()
    long = all_rows.melt(id_vars=["policy"], value_vars=["good", "medium", "bad"], var_name="class", value_name="count")
    order = list(all_rows["policy"])
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for cls in ["good", "medium", "bad"]:
        vals = long[long["class"].eq(cls)].set_index("policy").loc[order, "count"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=COLORS[cls], edgecolor="#FFFFFF", linewidth=0.8, label=cls)
        for i, v in enumerate(vals):
            if v > 800:
                ax.text(i, bottom[i] + v / 2, f"{int(v):,}", ha="center", va="center", fontsize=8, color="#1F2430")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Windows")
    ax.set_xlabel("")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    add_header(
        fig,
        "Clean BUT policy retention by class",
        "Stacked class counts; duration-only 10-60s is much smaller, while margin/drop-outlier keeps a trainable body.",
    )
    fig.subplots_adjust(top=0.86, bottom=0.28, left=0.08, right=0.98)
    savefig(fig, out)


def plot_region_composition(df: pd.DataFrame, masks: dict[str, pd.Series], out: Path) -> None:
    rows = []
    for policy, mask in masks.items():
        sub = df[mask]
        denom = max(len(sub), 1)
        for region, count in sub["original_region"].value_counts().items():
            rows.append({"policy": policy, "region": region, "share": count / denom, "n": int(count)})
    tab = pd.DataFrame(rows)
    order = list(masks.keys())
    regions = [
        "clean_core",
        "good_medium_overlap",
        "outlier_low_confidence",
        "right_bad_island",
        "near_bad_boundary",
        "medium_bad_overlap",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for region in regions:
        vals = (
            tab[tab["region"].eq(region)]
            .set_index("policy")
            .reindex(order)["share"]
            .fillna(0.0)
            .to_numpy()
        )
        ax.bar(x, vals, bottom=bottom, color=COLORS.get(region, "#C5CAD3"), edgecolor="#FFFFFF", linewidth=0.8, label=region)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_yticklabels([f"{int(t * 100)}%" for t in ax.get_yticks()])
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Share of windows")
    ax.set_xlabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    add_header(
        fig,
        "Region composition shows what the filter removes",
        "`drop outlier` removes the low-confidence region but preserves good/medium overlap and bad core/near-boundary.",
    )
    fig.subplots_adjust(top=0.86, bottom=0.36, left=0.08, right=0.98)
    savefig(fig, out)


def plot_margin_distribution(df: pd.DataFrame, out: Path) -> None:
    plot_df = df[df["matched_segment"]].copy()
    plot_df["min_margin_clipped"] = plot_df["min_margin_sec"].clip(upper=60)
    keep_regions = ["clean_core", "good_medium_overlap", "outlier_low_confidence", "right_bad_island", "near_bad_boundary"]
    plot_df = plot_df[plot_df["original_region"].isin(keep_regions)]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=plot_df,
        x="original_region",
        y="min_margin_clipped",
        order=keep_regions,
        palette={r: COLORS.get(r, "#C5CAD3") for r in keep_regions},
        showfliers=False,
        ax=ax,
    )
    ax.axhline(5, color=COLORS["dropped"], linestyle="--", linewidth=1.2)
    ax.text(0.02, 5.7, "5s margin rule", transform=ax.get_yaxis_transform(), fontsize=9, color=COLORS["dropped"])
    ax.set_ylabel("Nearest distance from 10s window to label-boundary, clipped at 60s")
    ax.set_xlabel("")
    ax.set_xticklabels([x.get_text().replace("_", "\n") for x in ax.get_xticklabels()])
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    add_header(
        fig,
        "Boundary margin separates label-contamination risk from outlier stress",
        "Low margin means a 10s crop sits near a label transition; this is different from low-confidence outlier geometry.",
    )
    fig.subplots_adjust(top=0.85, bottom=0.22, left=0.10, right=0.98)
    savefig(fig, out)


def plot_pca_geometry(df: pd.DataFrame, out: Path) -> None:
    rng = np.random.default_rng(20260619)
    full = df[df["matched_segment"] & df["pc1"].notna() & df["pc2"].notna()].copy()
    clean_mask = full["min_margin_sec"].ge(5.0) & full["original_region"].ne("outlier_low_confidence")
    full["policy_state"] = np.where(clean_mask, "kept by clean protocol", "held out / stress")
    if len(full) > 9000:
        idx = rng.choice(full.index.to_numpy(), size=9000, replace=False)
        full = full.loc[idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    class_palette = {"good": COLORS["good"], "medium": COLORS["medium"], "bad": COLORS["bad"]}
    sns.scatterplot(
        data=full,
        x="pc1",
        y="pc2",
        hue="y_class",
        hue_order=["good", "medium", "bad"],
        palette=class_palette,
        s=9,
        alpha=0.38,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Class geometry", fontsize=11, loc="left")
    state_palette = {"kept by clean protocol": COLORS["kept"], "held out / stress": COLORS["dropped"]}
    sns.scatterplot(
        data=full,
        x="pc1",
        y="pc2",
        hue="policy_state",
        hue_order=["kept by clean protocol", "held out / stress"],
        palette=state_palette,
        s=9,
        alpha=0.42,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Clean protocol state", fontsize=11, loc="left")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle=":", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    add_header(
        fig,
        "PCA view: outlier removal is a stress holdout, not a boundary deletion",
        "Good/medium overlap remains in the kept body; low-confidence points are held out for stress reporting.",
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.98, wspace=0.12)
    savefig(fig, out)


def plot_feature_boxes(df: pd.DataFrame, out: Path) -> None:
    features = [
        ("qrs_visibility", "QRS visibility"),
        ("sqi_basSQI", "basSQI"),
        ("baseline_step", "Baseline step"),
        ("flatline_ratio", "Flatline ratio"),
        ("non_qrs_diff_p95", "Non-QRS diff p95"),
        ("detector_agreement", "Detector agreement"),
    ]
    keep_regions = ["clean_core", "good_medium_overlap", "outlier_low_confidence", "right_bad_island"]
    plot_df = df[df["original_region"].isin(keep_regions)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, (col, label) in zip(axes.flat, features):
        local = plot_df[["original_region", col]].dropna().copy()
        if len(local) > 12000:
            local = local.sample(12000, random_state=20260619)
        sns.boxplot(
            data=local,
            x="original_region",
            y=col,
            order=keep_regions,
            palette={r: COLORS.get(r, "#C5CAD3") for r in keep_regions},
            showfliers=False,
            ax=ax,
        )
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([t.get_text().replace("_", "\n") for t in ax.get_xticklabels()], fontsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
    add_header(
        fig,
        "Waveform-computable features explain why regions are separable",
        "Outlier/stress regions differ in QRS reliability, baseline/flatline behavior, and non-QRS detail rather than only class labels.",
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.98, hspace=0.42, wspace=0.24)
    savefig(fig, out)


def plot_cleaning_flow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.18, 0.22, "Full fixed 10s BUT\n32,956 windows"),
        (0.25, 0.58, 0.20, 0.22, "Label-window reliability\nmatched segment + margin"),
        (0.50, 0.58, 0.20, 0.22, "Waveform geometry atlas\nSQI / QRS / baseline / detail"),
        (0.75, 0.58, 0.22, 0.22, "Clean learnable body\ncore + overlap + bad guard"),
        (0.75, 0.18, 0.22, 0.22, "Stress buckets\nlow-confidence + extreme outliers"),
    ]
    for x, y, w, h, text in boxes:
        color = COLORS["kept"] if "Clean" in text else COLORS["dropped"] if "Stress" in text else "#EAF1FE"
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="#2E4780", linewidth=1.2, alpha=0.35))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes, fontsize=10, color=TOKENS["ink"])
    arrows = [
        ((0.20, 0.69), (0.25, 0.69)),
        ((0.45, 0.69), (0.50, 0.69)),
        ((0.70, 0.69), (0.75, 0.69)),
        ((0.61, 0.58), (0.75, 0.29)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color=TOKENS["muted"], lw=1.4))
    ax.text(
        0.02,
        0.08,
        "Interpretation: the filter creates a reproducible high-confidence protocol and keeps stress/outlier rows visible as separate diagnostics.\n"
        "It should not be described as full BUT; full and stress buckets remain report-only.",
        transform=ax.transAxes,
        fontsize=10,
        color=TOKENS["muted"],
    )
    add_header(
        fig,
        "Closed-loop cleaning contract",
        "The rule is label-reliability + waveform-geometry confidence, not model-error cherry-picking.",
    )
    fig.subplots_adjust(top=0.84, bottom=0.06, left=0.02, right=0.98)
    savefig(fig, out)


def write_report(summary: pd.DataFrame, out_dir: Path, figures: Iterable[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "clean_but_policy_distribution_report.md"
    full = summary[summary["split_scope"].eq("all")]
    def row(policy: str) -> pd.Series:
        return full[full["policy"].eq(policy)].iloc[0]
    base = row("Full 10s protocol")
    clean5 = row("Margin >=5s, drop outlier")
    seg60 = row("Segment 10-60s, keep outlier")
    lines = [
        "# Clean BUT Policy Distribution",
        "",
        "## Main Takeaways",
        "",
        f"- Full fixed-10s BUT has **{int(base['n']):,}** windows: good {int(base['good']):,}, medium {int(base['medium']):,}, bad {int(base['bad']):,}.",
        f"- The current clean body (`margin>=5s` + drop `outlier_low_confidence`) keeps **{int(clean5['n']):,}** windows: good {int(clean5['good']):,}, medium {int(clean5['medium']):,}, bad {int(clean5['bad']):,}.",
        f"- The duration-only `10-60s` sanity slice keeps only **{int(seg60['n']):,}** windows and almost no stable bad body, so it is too small as a main protocol.",
        "- Good/medium overlap is intentionally retained; low-confidence outliers are held out as stress diagnostics rather than hidden.",
        "",
        "## Why This Is Defensible",
        "",
        "The clean protocol separates two concepts: label-window reliability and waveform-geometry confidence. "
        "A sample is not removed because the model gets it wrong. It is held out when the 10s window sits too close to a label transition or when the atlas marks it as low-confidence/outlier geometry. "
        "The correct claim is therefore `clean high-confidence BUT`, not full BUT.",
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        lines.append(f"![{fig.stem}]({fig})")
        lines.append("")
    lines.extend(
        [
            "## Policy Count Table",
            "",
            dataframe_to_markdown(summary),
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float) and float(val).is_integer():
                val = int(val)
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    set_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    masks = policy_masks(df)
    summary = count_summary(df, masks)
    summary.to_csv(REPORT_DIR / "clean_but_policy_distribution_counts.csv", index=False)
    figures = [
        REPORT_DIR / "fig1_policy_class_retention.png",
        REPORT_DIR / "fig2_region_composition.png",
        REPORT_DIR / "fig3_margin_distribution_by_region.png",
        REPORT_DIR / "fig4_pca_clean_vs_stress.png",
        REPORT_DIR / "fig5_waveform_feature_boxes.png",
        REPORT_DIR / "fig6_cleaning_contract.png",
    ]
    plot_policy_counts(summary, figures[0])
    plot_region_composition(df, masks, figures[1])
    plot_margin_distribution(df, figures[2])
    plot_pca_geometry(df, figures[3])
    plot_feature_boxes(df, figures[4])
    plot_cleaning_flow(figures[5])
    write_report(summary, REPORT_DIR, figures)
    payload = {
        "report": str(REPORT_DIR / "clean_but_policy_distribution_report.md"),
        "figures": [str(f) for f in figures],
        "counts_csv": str(REPORT_DIR / "clean_but_policy_distribution_counts.csv"),
    }
    (REPORT_DIR / "clean_but_policy_distribution_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
