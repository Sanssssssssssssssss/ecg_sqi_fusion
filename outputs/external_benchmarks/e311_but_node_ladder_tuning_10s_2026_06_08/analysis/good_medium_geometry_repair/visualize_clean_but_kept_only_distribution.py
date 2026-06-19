"""Simple kept-only visualizations for the clean BUT protocol.

This report intentionally omits dropped/stress samples. It shows only the
distribution retained by the current clean protocol:
matched_segment, min_margin_sec >= 5, and original_region != outlier_low_confidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "clean_but_kept_only_distribution"
MARGIN_PATH = ANALYSIS_DIR / "clean_but_window_policy" / "current_10s_window_segment_margins.csv"
ATLAS_PATH = REPORT_ROOT / "original_region_boundary" / "original_region_atlas.csv"

CLASS_COLORS = {"good": "#5477C4", "medium": "#B8A037", "bad": "#CC6F47"}
REGION_COLORS = {
    "good_medium_overlap": "#B8A037",
    "clean_core": "#5477C4",
    "right_bad_island": "#8A3A6F",
    "near_bad_boundary": "#804126",
    "medium_bad_overlap": "#71B436",
}


def style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#6F768A",
            "ytick.color": "#6F768A",
            "grid.color": "#E6E8F0",
            "text.color": "#1F2430",
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.952, subtitle, ha="left", va="top", fontsize=10, color="#6F768A")


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_kept() -> pd.DataFrame:
    margin = pd.read_csv(MARGIN_PATH)
    atlas = pd.read_csv(ATLAS_PATH).drop_duplicates("idx").set_index("idx")
    df = margin.copy()
    df["idx"] = df["idx"].astype(int)
    feature_cols = [
        "pc1",
        "pc2",
        "qrs_visibility",
        "sqi_basSQI",
        "baseline_step",
        "flatline_ratio",
        "non_qrs_diff_p95",
        "detector_agreement",
    ]
    for col in feature_cols:
        if col in atlas.columns:
            df[col] = df["idx"].map(atlas[col])
    df["matched_segment"] = df["matched_segment"].fillna(False).astype(bool)
    df["original_region"] = df["original_region"].fillna("unknown").astype(str)
    df["y_class"] = df["y_class"].fillna("unknown").astype(str)
    df["split"] = df["split"].fillna("unknown").astype(str)
    df["min_margin_sec"] = pd.to_numeric(df["min_margin_sec"], errors="coerce").fillna(-1.0)
    kept = df[
        df["matched_segment"]
        & df["min_margin_sec"].ge(5.0)
        & df["original_region"].ne("outlier_low_confidence")
    ].copy()
    return kept


def annotate_bars(ax: plt.Axes, fmt: str = "{:,.0f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2430",
        )


def plot_counts(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    class_counts = df["y_class"].value_counts().reindex(["good", "medium", "bad"]).fillna(0).astype(int)
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=CLASS_COLORS, ax=axes[0], hue=class_counts.index, legend=False)
    axes[0].set_title("Class counts", loc="left", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Windows")
    annotate_bars(axes[0])

    region_order = ["good_medium_overlap", "clean_core", "right_bad_island", "near_bad_boundary", "medium_bad_overlap"]
    region_counts = df["original_region"].value_counts().reindex(region_order).fillna(0).astype(int)
    sns.barplot(x=region_counts.values, y=region_counts.index, palette=REGION_COLORS, ax=axes[1], hue=region_counts.index, legend=False)
    axes[1].set_title("Region counts", loc="left", fontsize=11)
    axes[1].set_xlabel("Windows")
    axes[1].set_ylabel("")
    for patch in axes[1].patches:
        width = patch.get_width()
        if width > 0:
            axes[1].text(width, patch.get_y() + patch.get_height() / 2, f" {int(width):,}", va="center", fontsize=9)

    split_class = pd.crosstab(df["split"], df["y_class"]).reindex(columns=["good", "medium", "bad"]).fillna(0)
    split_class.plot(kind="bar", stacked=True, color=[CLASS_COLORS[c] for c in ["good", "medium", "bad"]], ax=axes[2], edgecolor="#FFFFFF")
    axes[2].set_title("Split composition", loc="left", fontsize=11)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Windows")
    axes[2].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    axes[2].tick_params(axis="x", rotation=0)

    for ax in axes:
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
    add_header(
        fig,
        "Kept clean-BUT distribution after filtering",
        "Only retained rows are shown: matched segment, margin >= 5s, and original_region != outlier_low_confidence.",
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.06, right=0.98, wspace=0.35)
    save(fig, out)


def plot_pca(df: pd.DataFrame, out: Path) -> None:
    local = df[df["pc1"].notna() & df["pc2"].notna()].copy()
    rng = np.random.default_rng(20260619)
    if len(local) > 12000:
        local = local.loc[rng.choice(local.index.to_numpy(), size=12000, replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharex=True, sharey=True)
    sns.scatterplot(
        data=local,
        x="pc1",
        y="pc2",
        hue="y_class",
        hue_order=["good", "medium", "bad"],
        palette=CLASS_COLORS,
        s=10,
        alpha=0.42,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Kept rows by label", loc="left", fontsize=11)
    region_order = ["good_medium_overlap", "clean_core", "right_bad_island", "near_bad_boundary", "medium_bad_overlap"]
    sns.scatterplot(
        data=local,
        x="pc1",
        y="pc2",
        hue="original_region",
        hue_order=[r for r in region_order if r in set(local["original_region"])],
        palette=REGION_COLORS,
        s=10,
        alpha=0.42,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Kept rows by retained region", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle=":", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, markerscale=2)
    add_header(
        fig,
        "Kept clean-BUT PCA distribution",
        "No dropped samples are plotted; this shows the training/evaluation body after the current clean filter.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.07, right=0.98, wspace=0.12)
    save(fig, out)


def plot_feature_distribution(df: pd.DataFrame, out: Path) -> None:
    features = [
        ("qrs_visibility", "QRS visibility"),
        ("sqi_basSQI", "basSQI"),
        ("baseline_step", "Baseline step"),
        ("flatline_ratio", "Flatline ratio"),
        ("non_qrs_diff_p95", "Non-QRS diff p95"),
        ("detector_agreement", "Detector agreement"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    for ax, (col, title) in zip(axes.flat, features):
        local = df[["y_class", col]].dropna().copy()
        if len(local) > 12000:
            local = local.sample(12000, random_state=20260619)
        sns.boxplot(
            data=local,
            x="y_class",
            y=col,
            order=["good", "medium", "bad"],
            hue="y_class",
            palette=CLASS_COLORS,
            showfliers=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
    add_header(
        fig,
        "Kept clean-BUT feature distribution by label",
        "Only retained rows are shown; these are waveform-computable SQI/QRS/baseline/detail features.",
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.06, right=0.98, hspace=0.35, wspace=0.25)
    save(fig, out)


def write_report(df: pd.DataFrame, figs: list[Path]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {
        "n": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["y_class"].value_counts().to_dict().items()},
        "region_counts": {str(k): int(v) for k, v in df["original_region"].value_counts().to_dict().items()},
        "split_class_counts": {
            str(split): {str(k): int(v) for k, v in sub["y_class"].value_counts().to_dict().items()}
            for split, sub in df.groupby("split")
        },
    }
    (REPORT_DIR / "kept_only_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    lines = [
        "# Kept Clean-BUT Distribution",
        "",
        "Filter shown here: `matched_segment == true`, `min_margin_sec >= 5`, `original_region != outlier_low_confidence`.",
        "",
        f"Retained windows: **{counts['n']:,}**.",
        "",
        "Class counts: "
        + ", ".join(f"{k} {v:,}" for k, v in counts["class_counts"].items()),
        "",
        "Region counts: "
        + ", ".join(f"{k} {v:,}" for k, v in counts["region_counts"].items()),
        "",
    ]
    for fig in figs:
        lines.append(f"![{fig.stem}]({fig})")
        lines.append("")
    (REPORT_DIR / "kept_only_distribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_kept()
    df.to_csv(REPORT_DIR / "kept_only_rows_summary.csv", index=False)
    figs = [
        REPORT_DIR / "kept_fig1_counts.png",
        REPORT_DIR / "kept_fig2_pca_distribution.png",
        REPORT_DIR / "kept_fig3_feature_distribution.png",
    ]
    plot_counts(df, figs[0])
    plot_pca(df, figs[1])
    plot_feature_distribution(df, figs[2])
    write_report(df, figs)
    print(json.dumps({"report": str(REPORT_DIR / "kept_only_distribution_report.md"), "figures": [str(f) for f in figs]}, indent=2))


if __name__ == "__main__":
    main()
