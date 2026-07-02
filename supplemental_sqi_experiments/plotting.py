from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0F4D92",
    "red": "#B64342",
    "gold": "#D6A124",
    "purple": "#6B4C9A",
    "teal": "#26877F",
    "ink": "#252525",
    "muted": "#737373",
    "grid": "#D9D9D9",
    "soft_blue": "#D8E6F3",
    "soft_red": "#F3D4D2",
    "soft_gold": "#F7E5B5",
    "soft_purple": "#E3DDF0",
    "soft_teal": "#CFE8E4",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )


def save_figure(fig: plt.Figure, base: str | Path, *, png_dpi: int = 600) -> list[Path]:
    b = Path(base)
    b.parent.mkdir(parents=True, exist_ok=True)
    outs = [b.with_suffix(".svg"), b.with_suffix(".pdf"), b.with_suffix(".png")]
    fig.savefig(outs[0], bbox_inches="tight")
    fig.savefig(outs[1], bbox_inches="tight")
    fig.savefig(outs[2], dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return outs


def plot_strict_table6(
    selected: pd.DataFrame,
    inclusion: pd.DataFrame,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(7.1, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    sel = selected.sort_values("cardinality")
    ax0.plot(sel["cardinality"], sel["val_Ac"], color=PALETTE["blue"], lw=1.4, marker="o", ms=4, label="validation")
    ax0.plot(sel["cardinality"], sel["test_Ac"], color=PALETTE["red"], lw=1.4, marker="s", ms=4, label="test")
    ax0.set_xlabel("Number of SQIs")
    ax0.set_ylabel("Accuracy")
    ax0.set_xticks(sel["cardinality"].astype(int).tolist())
    ax0.grid(axis="y", color=PALETTE["grid"], lw=0.45)
    ax0.legend(loc="lower right", handlelength=1.7)
    add_panel_label(ax0, "a")

    inc = inclusion[inclusion["rank_scope"].eq("top10_by_cardinality")].copy()
    if not inc.empty:
        pivot = inc.pivot(index="SQI", columns="cardinality", values="inclusion_frequency").fillna(0.0)
        pivot = pivot.reindex(["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI", "pSQI"])
        im = ax1.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax1.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        ax1.set_xticks(np.arange(len(pivot.columns)), labels=[str(int(x)) for x in pivot.columns])
        ax1.set_xlabel("Number of SQIs")
        ax1.set_ylabel("")
        cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
        cb.set_label("Inclusion frequency")
    add_panel_label(ax1, "b")
    return save_figure(fig, out_base)


def plot_model_diagnostics(
    score_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(7.35, 4.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], wspace=0.32, hspace=0.46)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[0, 1])
    ax_ci = fig.add_subplot(gs[1, :])

    from sklearn.metrics import roc_curve

    colors = {"MLP selected-five": PALETTE["blue"], "SVM selected-five": PALETTE["red"]}
    for model, color in colors.items():
        col = "mlp_selected5_score" if model.startswith("MLP") else "svm_selected5_score"
        if col not in score_df:
            continue
        fpr, tpr, _ = roc_curve(score_df["y01"].to_numpy(dtype=int), score_df[col].to_numpy(dtype=float))
        auc = float(
            model_summary.loc[
                model_summary["model"].eq(model) & model_summary["metric"].eq("AUC"),
                "estimate",
            ].iloc[0]
        )
        ax_roc.plot(fpr, tpr, color=color, lw=1.35, label=f"{model.split()[0]} AUC {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], color=PALETTE["muted"], lw=0.8, linestyle=":")
    ax_roc.set_xlabel("1-Sp")
    ax_roc.set_ylabel("Se")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(color=PALETTE["grid"], lw=0.35)
    add_panel_label(ax_roc, "a")

    groups = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]
    group_colors = [PALETTE["blue"], PALETTE["red"], PALETTE["gold"], PALETTE["purple"]]
    positions = np.arange(len(groups))
    for offset, col, color, label in [
        (-0.13, "mlp_selected5_score", PALETTE["blue"], "MLP"),
        (0.13, "svm_selected5_score", PALETTE["red"], "SVM"),
    ]:
        data = [score_df.loc[score_df["sample_group"].eq(g), col].to_numpy(dtype=float) for g in groups]
        parts = ax_score.violinplot(data, positions=positions + offset, widths=0.22, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.18)
        med = [np.median(x) if len(x) else np.nan for x in data]
        ax_score.scatter(positions + offset, med, s=12, color=color, label=label, zorder=3)
    ax_score.set_xticks(positions, labels=["acc.", "orig.\nunacc.", "em", "ma"])
    ax_score.set_ylabel("Model score")
    ax_score.set_ylim(-0.03, 1.03)
    ax_score.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    add_panel_label(ax_score, "b")

    ci = model_summary[model_summary["metric"].isin(["Ac", "Se", "Sp", "AUC"])].copy()
    metric_labels = {
        "Ac": "accuracy",
        "Se": "acceptable recall",
        "Sp": "poor recall",
        "AUC": "AUC",
    }
    ci["label"] = (
        ci["model"].str.replace(" selected-five", "", regex=False)
        + " "
        + ci["metric"].map(metric_labels).fillna(ci["metric"])
    )
    y = np.arange(len(ci))
    c = [PALETTE["blue"] if x.startswith("MLP") else PALETTE["red"] for x in ci["label"]]
    ax_ci.hlines(y, ci["ci_low"], ci["ci_high"], color=c, lw=1.2)
    ax_ci.scatter(ci["estimate"], y, color=c, s=18, zorder=3)
    ax_ci.set_yticks(y, labels=ci["label"])
    ax_ci.set_xlabel("Source-bootstrap estimate and 95% CI")
    ax_ci.set_xlim(0.80, 1.01)
    ax_ci.grid(axis="x", color=PALETTE["grid"], lw=0.35)
    add_panel_label(ax_ci, "c", x=-0.06)
    return save_figure(fig, out_base)


def plot_fsqi_mechanism(
    logdiff_summary: pd.DataFrame,
    threshold_scan: pd.DataFrame,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(7.1, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    groups = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["gold"], PALETTE["purple"]]
    data = [logdiff_summary.loc[logdiff_summary["sample_group"].eq(g), "median_log10_abs_diff"].to_numpy(dtype=float) for g in groups]
    parts = ax0.violinplot(data, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        rng = np.random.default_rng(100 + i)
        if len(vals):
            sample = vals if len(vals) <= 200 else rng.choice(vals, 200, replace=False)
            jitter = rng.normal(0, 0.035, size=len(sample))
            ax0.scatter(np.full(len(sample), i) + jitter, sample, s=3.5, color=color, alpha=0.32, linewidths=0)
            ax0.scatter([i], [np.median(vals)], s=16, color=color, edgecolor="#FFFFFF", linewidth=0.5, zorder=3)
    ax0.set_xticks(np.arange(1, len(groups) + 1), labels=["acc.", "orig.\nunacc.", "em", "ma"])
    ax0.set_ylabel("Median log10 abs difference")
    add_panel_label(ax0, "a")

    scan = threshold_scan[threshold_scan["score_name"].eq("flat_fraction")].copy()
    if not scan.empty:
        for metric, color in [("test_AUC", PALETTE["blue"]), ("test_Ac", PALETTE["red"])]:
            ax1.plot(scan["threshold_mv"], scan[metric], marker="o", ms=3, lw=1.2, color=color, label=metric.replace("test_", ""))
        ax1.set_xscale("log")
        ax1.set_xlabel("Flat threshold (mV)")
        ax1.set_ylabel("Univariate test metric")
        ax1.set_ylim(0.0, 1.02)
        ax1.grid(axis="y", color=PALETTE["grid"], lw=0.35)
        ax1.legend(loc="lower right")
    add_panel_label(ax1, "b")
    return save_figure(fig, out_base)


def _group_color_map() -> dict[str, str]:
    return {
        "original acceptable": PALETTE["blue"],
        "original unacceptable": PALETTE["red"],
        "synthetic em": PALETTE["gold"],
        "synthetic ma": PALETTE["purple"],
    }


def _add_cov_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    pts = np.column_stack([x, y]).astype(float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 4:
        return
    cov = np.cov(pts, rowvar=False)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = 1.7941225779941015  # sqrt(chi2.ppf(0.80, df=2))
    center = pts.mean(axis=0)
    ell = Ellipse(
        xy=center,
        width=2.0 * scale * np.sqrt(vals[0]),
        height=2.0 * scale * np.sqrt(vals[1]),
        angle=angle,
        facecolor="none",
        edgecolor=color,
        lw=1.0,
        alpha=0.85,
        zorder=2.5,
    )
    ax.add_patch(ell)
    ax.scatter([center[0]], [center[1]], marker="D", s=20, color=color, edgecolor="#FFFFFF", linewidth=0.55, zorder=3)


def _heat_text_color(cmap, norm, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return PALETTE["ink"] if luminance > 0.62 else "#FFFFFF"


def plot_domain_shift(
    pca_df: pd.DataFrame,
    per_sqi_auc: pd.DataFrame,
    cross_domain: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(11.3, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 1.08, 1.22], wspace=0.92)
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[0, 2])

    colors = _group_color_map()
    labels = {
        "original acceptable": "original acceptable",
        "original unacceptable": "original unacceptable",
        "synthetic em": "paper EM",
        "synthetic ma": "paper MA",
    }
    pc = pca_df[["PC1", "PC2"]].to_numpy(dtype=float)
    pc = pc[np.isfinite(pc).all(axis=1)]
    xlo, xhi = float(np.min(pc[:, 0])), float(np.max(pc[:, 0]))
    ylo, yhi = float(np.min(pc[:, 1])), float(np.max(pc[:, 1]))
    pad_x = 0.035 * max(1e-9, xhi - xlo)
    pad_y = 0.045 * max(1e-9, yhi - ylo)
    for group in ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]:
        g = pca_df[pca_df["sample_group"].eq(group)]
        if g.empty:
            continue
        rounded_xy = list(zip(g["PC1"].round(6), g["PC2"].round(6)))
        overlap_counts = pd.Series(rounded_xy).map(pd.Series(rounded_xy).value_counts()).to_numpy(dtype=float)
        marker_sizes = 11.0 + 3.2 * np.maximum(0.0, np.minimum(overlap_counts, 10.0) - 1.0)
        ax_pca.scatter(
            g["PC1"],
            g["PC2"],
            s=marker_sizes,
            color=colors[group],
            alpha=0.68,
            linewidths=0,
            label=labels[group],
        )
        _add_cov_ellipse(ax_pca, g["PC1"].to_numpy(dtype=float), g["PC2"].to_numpy(dtype=float), colors[group])
    ev1 = float(pca_df["explained_variance_PC1"].iloc[0]) * 100.0
    ev2 = float(pca_df["explained_variance_PC2"].iloc[0]) * 100.0
    if "projection_set" in pca_df.columns and pca_df["projection_set"].astype(str).str.contains("original_reference").any():
        ax_pca.margins(x=0.035, y=0.045)
        ax_pca.autoscale_view()
    else:
        ax_pca.set_xlim(xlo - pad_x, xhi + pad_x)
        ax_pca.set_ylim(ylo - pad_y, yhi + pad_y)
    ax_pca.set_xlabel(f"PC1 ({ev1:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({ev2:.1f}%)")
    ax_pca.grid(color=PALETTE["grid"], lw=0.3, alpha=0.65)
    handles, legend_labels = ax_pca.get_legend_handles_labels()
    ax_pca.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        fontsize=6.25,
        handletextpad=0.28,
        borderaxespad=0.0,
        labelspacing=0.42,
        markerscale=1.25,
    )
    add_panel_label(ax_pca, "a", x=-0.16)

    order = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI", "pSQI"]
    auc = per_sqi_auc.set_index("SQI").reindex(order).reset_index()
    vals = auc["domain_auc_original_vs_synthetic_poor"].to_numpy(dtype=float)
    bar_colors = [PALETTE["red"] if v >= 0.8 else PALETTE["blue"] for v in vals]
    y = np.arange(len(auc))
    ax_auc.barh(y, vals, color=bar_colors, alpha=0.88)
    if {"domain_auc_ci_low", "domain_auc_ci_high"}.issubset(auc.columns):
        lo = auc["domain_auc_ci_low"].to_numpy(dtype=float)
        hi = auc["domain_auc_ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([np.maximum(0.0, vals - lo), np.maximum(0.0, hi - vals)])
        ax_auc.errorbar(vals, y, xerr=xerr, fmt="none", ecolor=PALETTE["ink"], elinewidth=0.75, capsize=2.2, capthick=0.75)
    ax_auc.axvline(0.5, color=PALETTE["muted"], lw=0.8, linestyle=":")
    ax_auc.set_yticks(y, labels=auc["SQI"].tolist())
    ax_auc.invert_yaxis()
    ax_auc.set_xlim(0.45, 1.01)
    ax_auc.set_xlabel("Domain AUC")
    ax_auc.grid(axis="x", color=PALETTE["grid"], lw=0.3)
    try:
        overall = float(domain_metrics.loc[domain_metrics["metric"].eq("source_grouped_logistic_domain_auc"), "estimate"].iloc[0])
        mmd_p = float(domain_metrics.loc[domain_metrics["metric"].eq("RBF-MMD2"), "p_value_permutation"].iloc[0])
        ax_auc.text(
            0.98,
            0.98,
            f"84-SQI AUC {overall:.3f}\nMMD p {mmd_p:.3g}",
            transform=ax_auc.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            color=PALETTE["ink"],
        )
    except Exception:
        pass
    add_panel_label(ax_auc, "b", x=-0.23)

    heat = cross_domain.pivot(index="train_poor_domain", columns="test_poor_domain", values="test_AUC")
    row_order = ["original poor", "em", "ma", "synthetic poor"]
    col_order = ["original poor", "em", "ma"]
    heat = heat.reindex(index=row_order, columns=col_order)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = ax_heat.imshow(heat.to_numpy(dtype=float), cmap="RdBu_r", norm=norm, aspect="auto")
    ax_heat.set_xticks(np.arange(len(heat.columns)), labels=heat.columns, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(heat.index)), labels=heat.index)
    ax_heat.set_xlabel("Test poor domain")
    ax_heat.set_ylabel("")
    heatmap_text_effect = [pe.withStroke(linewidth=0.9, foreground=(0.0, 0.0, 0.0, 0.36))]
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.iloc[i, j]
            if pd.isna(v):
                continue
            ax_heat.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=_heat_text_color(im.cmap, im.norm, float(v)),
                path_effects=heatmap_text_effect,
            )
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.050, pad=0.025)
    cb.set_label("AUC")
    add_panel_label(ax_heat, "c", x=-0.25)
    return save_figure(fig, out_base)


def plot_fsqi_mechanism_updated(
    fsqi_summary: pd.DataFrame,
    threshold_scan: pd.DataFrame,
    subgroup_recall: pd.DataFrame,
    linear_vs_rbf: pd.DataFrame,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(8.75, 3.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.05, 1.08], wspace=0.58)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_scan = fig.add_subplot(gs[0, 1])
    ax_recall = fig.add_subplot(gs[0, 2])

    eps = 1e-5
    colors = _group_color_map()
    for group in ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]:
        g = fsqi_summary[fsqi_summary["sample_group"].eq(group)]
        if g.empty:
            continue
        ax_scatter.scatter(
            np.clip(g["median_fSQI"].to_numpy(dtype=float), eps, None),
            np.clip(g["max_fSQI"].to_numpy(dtype=float), eps, None),
            s=10,
            color=colors[group],
            alpha=0.62,
            linewidths=0,
            label=group,
        )
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Median fSQI")
    ax_scatter.set_ylabel("Max fSQI")
    ax_scatter.grid(color=PALETTE["grid"], lw=0.3, alpha=0.65)
    ax_scatter.legend(loc="lower right", fontsize=6.1, handletextpad=0.25, borderaxespad=0.2, markerscale=1.15)
    add_panel_label(ax_scatter, "a", x=-0.16)

    scan = threshold_scan.sort_values("threshold_mv")
    ax_scan.plot(scan["threshold_mv"], scan["test_AUC"], color=PALETTE["blue"], lw=1.25, marker="o", ms=3.2, label="AUC")
    ax_scan.plot(scan["threshold_mv"], scan["test_Ac"], color=PALETTE["red"], lw=1.25, marker="s", ms=3.0, label="Ac")
    ax_scan.plot(scan["threshold_mv"], scan["val_Ac"], color=PALETTE["muted"], lw=0.95, linestyle=":", label="val Ac")
    ax_scan.set_xscale("log")
    ax_scan.set_ylim(0.0, 1.03)
    ax_scan.set_xlabel("Flat threshold (mV)")
    ax_scan.set_ylabel("Fixed-RBF metric")
    ax_scan.grid(axis="y", color=PALETTE["grid"], lw=0.3)
    ax_scan.legend(loc="lower left", fontsize=6.4, handlelength=1.5)
    add_panel_label(ax_scan, "b", x=-0.18)

    if not threshold_scan.empty:
        best_thr = float(threshold_scan.sort_values(["val_Ac", "test_AUC"], ascending=[False, False]).iloc[0]["threshold_mv"])
        sub = subgroup_recall[
            subgroup_recall["threshold_mv"].astype(float).sub(best_thr).abs() < 1e-18
        ].copy()
    else:
        sub = subgroup_recall.copy()
        best_thr = float("nan")
    group_labels = ["orig. acc.", "orig. unacc.", "em", "ma"]
    group_order = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]
    sub = sub.set_index("sample_group").reindex(group_order)
    bar_colors = [colors[g] for g in group_order]
    ax_recall.bar(np.arange(len(group_order)), sub["recall"].to_numpy(dtype=float), color=bar_colors, alpha=0.86)
    ax_recall.set_xticks(np.arange(len(group_order)), labels=group_labels, rotation=25, ha="right")
    ax_recall.set_ylim(0.0, 1.03)
    ax_recall.set_ylabel("True-class recall")
    ax_recall.grid(axis="y", color=PALETTE["grid"], lw=0.3)
    try:
        best_auc = float(threshold_scan.sort_values(["val_Ac", "test_AUC"], ascending=[False, False]).iloc[0]["test_AUC"])
        ax_recall.text(
            0.98,
            0.98,
            f"selected flat {best_thr:g} mV\nAUC {best_auc:.3f}",
            transform=ax_recall.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
            color=PALETTE["ink"],
        )
    except Exception:
        pass
    add_panel_label(ax_recall, "c", x=-0.18)
    return save_figure(fig, out_base)


def plot_bassqi_domain_shift(
    tab: pd.DataFrame,
    delta: pd.DataFrame,
    recall: pd.DataFrame,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(8.85, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.05, 1.0], wspace=0.58)
    ax = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])
    ax_rec = fig.add_subplot(gs[0, 2])
    groups = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]
    colors = _group_color_map()
    data = [tab.loc[tab["sample_group"].eq(g), "mean_1_minus_basSQI"].to_numpy(dtype=float) for g in groups]
    parts = ax.violinplot(data, showmedians=False, showextrema=False)
    for body, group in zip(parts["bodies"], groups):
        body.set_facecolor(colors[group])
        body.set_edgecolor(colors[group])
        body.set_alpha(0.22)
    for i, (vals, group) in enumerate(zip(data, groups), start=1):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(700 + i)
        sample = vals if len(vals) <= 250 else rng.choice(vals, 250, replace=False)
        jitter = rng.normal(0, 0.035, size=len(sample))
        ax.scatter(np.full(len(sample), i) + jitter, sample, s=4, color=colors[group], alpha=0.34, linewidths=0)
        ax.scatter([i], [np.median(vals)], s=18, color=colors[group], edgecolor="#FFFFFF", linewidth=0.55, zorder=3)
    ax.set_xticks(np.arange(1, len(groups) + 1), labels=["orig. acc.", "orig. unacc.", "em", "ma"], rotation=20, ha="right")
    ax.set_ylabel("Mean 1-basSQI")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.3)
    add_panel_label(ax, "a", x=-0.16)

    if not delta.empty:
        positions = {"em": 0, "ma": 1}
        pivot = delta.pivot_table(index="source_record_id", columns="noise_type", values="delta_mean_1_minus_basSQI", aggfunc="mean")
        for _, row in pivot.dropna(how="any").iterrows():
            ax_delta.plot([positions["em"], positions["ma"]], [row["em"], row["ma"]], color=PALETTE["grid"], lw=0.45, alpha=0.65, zorder=1)
        for noise_type, color in [("em", PALETTE["gold"]), ("ma", PALETTE["purple"])]:
            vals = delta.loc[delta["noise_type"].eq(noise_type), "delta_mean_1_minus_basSQI"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            parts = ax_delta.violinplot([vals], positions=[positions[noise_type]], widths=0.52, showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.20)
            rng = np.random.default_rng(1100 + positions[noise_type])
            sample = vals if len(vals) <= 250 else rng.choice(vals, 250, replace=False)
            ax_delta.scatter(
                np.full(len(sample), positions[noise_type]) + rng.normal(0, 0.035, len(sample)),
                sample,
                s=4,
                color=color,
                alpha=0.34,
                linewidths=0,
                zorder=2,
            )
            ax_delta.scatter([positions[noise_type]], [np.median(vals)], s=18, color=color, edgecolor="#FFFFFF", linewidth=0.55, zorder=3)
    ax_delta.axhline(0, color=PALETTE["muted"], lw=0.75, linestyle=":")
    ax_delta.set_xticks([0, 1], labels=["em", "ma"])
    ax_delta.set_ylabel("Delta mean 1-basSQI")
    ax_delta.grid(axis="y", color=PALETTE["grid"], lw=0.3)
    add_panel_label(ax_delta, "b", x=-0.18)

    rec_order = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]
    rec = recall.set_index("sample_group").reindex(rec_order)
    ax_rec.bar(np.arange(len(rec_order)), rec["value"].to_numpy(dtype=float), color=[colors[g] for g in rec_order], alpha=0.86)
    ax_rec.set_xticks(np.arange(len(rec_order)), labels=["orig. acc.", "orig. unacc.", "em", "ma"], rotation=25, ha="right")
    ax_rec.set_ylim(0.0, 1.03)
    ax_rec.set_ylabel("True-class recall")
    ax_rec.grid(axis="y", color=PALETTE["grid"], lw=0.3)
    add_panel_label(ax_rec, "c", x=-0.18)
    return save_figure(fig, out_base)


def plot_sqi_subgroup_separability(
    separability: pd.DataFrame,
    subgroup_recall: pd.DataFrame,
    rescue: pd.DataFrame,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(9.2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.14, 1.16, 1.0], wspace=0.68)
    ax_sep = fig.add_subplot(gs[0, 0])
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_rescue = fig.add_subplot(gs[0, 2])

    sqi_order = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI", "pSQI"]
    domain_order = ["original poor", "em", "ma"]
    sep = separability.pivot(index="SQI", columns="poor_domain", values="test_AUC").reindex(index=sqi_order, columns=domain_order)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im0 = ax_sep.imshow(sep.to_numpy(dtype=float), cmap="RdBu_r", norm=norm, aspect="auto")
    ax_sep.set_yticks(np.arange(len(sqi_order)), labels=sqi_order)
    ax_sep.set_xticks(np.arange(len(domain_order)), labels=domain_order, rotation=35, ha="right")
    ax_sep.set_ylabel("SQI")
    ax_sep.set_xlabel("Poor subgroup")
    heatmap_text_effect = [pe.withStroke(linewidth=0.9, foreground=(0.0, 0.0, 0.0, 0.36))]
    for i in range(sep.shape[0]):
        for j in range(sep.shape[1]):
            v = sep.iloc[i, j]
            if pd.isna(v):
                continue
            ax_sep.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=_heat_text_color(im0.cmap, im0.norm, float(v)),
                path_effects=heatmap_text_effect,
            )
    cb0 = fig.colorbar(im0, ax=ax_sep, fraction=0.050, pad=0.025)
    cb0.set_label("AUC")
    add_panel_label(ax_sep, "a", x=-0.21)

    model_order = ["iSQI", "basSQI", "paper pair", "paper quintuplet", "all seven"]
    model_labels = ["iSQI", "basSQI", "paper pair", "paper quint.", "all seven"]
    group_order = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]
    rec = subgroup_recall.pivot(index="model", columns="sample_group", values="value").reindex(index=model_order, columns=group_order)
    im1 = ax_rec.imshow(rec.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax_rec.set_yticks(np.arange(len(model_order)), labels=model_labels)
    ax_rec.set_xticks(np.arange(len(group_order)), labels=["orig. acc.", "orig. poor", "em", "ma"], rotation=35, ha="right")
    ax_rec.set_xlabel("Test subgroup")
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            v = rec.iloc[i, j]
            if pd.isna(v):
                continue
            ax_rec.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=_heat_text_color(im1.cmap, im1.norm, float(v)),
                path_effects=heatmap_text_effect,
            )
    add_panel_label(ax_rec, "b", x=-0.21)

    colors = _group_color_map()
    res_order = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]
    res = rescue.set_index("sample_group").reindex(res_order)
    vals = res["rescue_rate"].to_numpy(dtype=float)
    ax_rescue.bar(np.arange(len(res_order)), vals, color=[colors[g] for g in res_order], alpha=0.86)
    for i, row in enumerate(res.itertuples()):
        n = getattr(row, "pair_wrong_n")
        if pd.isna(vals[i]):
            continue
        ax_rescue.text(i, min(1.02, vals[i] + 0.035), f"n={int(n)}", ha="center", va="bottom", fontsize=6.2, color=PALETTE["ink"])
    ax_rescue.set_xticks(np.arange(len(res_order)), labels=["orig. acc.", "orig. poor", "em", "ma"], rotation=30, ha="right")
    ax_rescue.set_ylim(0.0, 1.12)
    ax_rescue.set_ylabel("Rescue rate")
    ax_rescue.grid(axis="y", color=PALETTE["grid"], lw=0.3)
    add_panel_label(ax_rescue, "c", x=-0.18)
    return save_figure(fig, out_base)
