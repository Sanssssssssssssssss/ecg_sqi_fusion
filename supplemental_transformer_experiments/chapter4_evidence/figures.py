from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .common import Paths, dry, ensure_dirs, feature_cols, read_json
from .seta_sqi import arm_dir


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

PALETTE = {
    "original acceptable": "#4c78a8",
    "original unacceptable": "#c44e52",
    "fixed_synthetic": "#e6ab02",
    "quota_draw": "#8da0cb",
    "smc_gapfill": "#66a61e",
    "within_original_resample": "#8c8c8c",
    "SQI": "#7f7f7f",
    "Waveform": "#4c78a8",
}


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")


def _fig_d1(paths: Paths) -> Path:
    metrics = pd.read_csv(paths.tables / "seta_distribution_repair_metrics.csv")
    transfer = pd.read_csv(paths.tables / "seta_source_transfer.csv")
    smc = pd.read_csv(arm_dir(paths, "smc_gapfill") / "construction_features.csv")
    cols = feature_cols(smc)
    plot_df = smc[(smc["split"].eq("train")) & ((smc["generated"].eq(1)) | ((smc["generated"].eq(0)) & (smc["y"].eq(0))))].copy()
    plot_df["group"] = np.where(plot_df["generated"].eq(1), "smc_gapfill", "original unacceptable")
    if len(plot_df) > 700:
        plot_df = plot_df.groupby("group", group_keys=False).apply(lambda x: x.sample(min(len(x), 350), random_state=0))
    z = StandardScaler().fit_transform(np.nan_to_num(plot_df[cols].to_numpy(float)))
    pc = PCA(n_components=2, random_state=0).fit_transform(z)
    plot_df["PC1"] = pc[:, 0]
    plot_df["PC2"] = pc[:, 1]
    plot_df.to_csv(paths.source_data / "fig_D1_pca.csv", index=False)
    metrics.to_csv(paths.source_data / "fig_D1_metrics.csv", index=False)
    transfer.to_csv(paths.source_data / "fig_D1_transfer.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
    a, b, c, d = ax.ravel()
    for group, sub in plot_df.groupby("group"):
        a.scatter(sub["PC1"], sub["PC2"], s=10, alpha=0.65, label=group, color=PALETTE.get(group, "#555555"), linewidth=0)
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
    a.legend(loc="best", fontsize=6)
    _panel(a, "a")

    order = ["fixed_synthetic", "quota_draw", "smc_gapfill", "within_original_resample"]
    m = metrics.set_index("construction").reindex([x for x in order if x in set(metrics["construction"])])
    xs = np.arange(len(m))
    width = 0.25
    b.bar(xs - width, m["c2st_auc"], width, label="C2ST AUC", color="#8da0cb")
    b.bar(xs, m["rbf_mmd"], width, label="RBF-MMD", color="#66c2a5")
    b.bar(xs + width, m["swd"], width, label="SWD", color="#fc8d62")
    b.set_xticks(xs)
    b.set_xticklabels(m.index, rotation=25, ha="right")
    b.set_ylabel("Metric value")
    b.legend(fontsize=6)
    _panel(b, "b")

    heat = transfer.set_index("construction")[["test_original_poor_auc", "original_poor_recall", "acceptable_recall"]]
    im = c.imshow(heat.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    c.set_xticks(np.arange(heat.shape[1]))
    c.set_xticklabels(["orig poor AUC", "orig poor R", "accept R"], rotation=25, ha="right")
    c.set_yticks(np.arange(heat.shape[0]))
    c.set_yticklabels(heat.index)
    fig.colorbar(im, ax=c, fraction=0.046, pad=0.02)
    _panel(c, "c")

    d.bar(transfer["construction"], transfer["original_poor_recall"], color=[PALETTE.get(x, "#777777") for x in transfer["construction"]])
    d.set_ylim(0, 1)
    d.set_ylabel("Original-unacceptable recall")
    d.tick_params(axis="x", rotation=25)
    _panel(d, "d")
    fig.tight_layout()
    out = paths.figures / "fig_D1_distribution_repair_summary"
    _save(fig, out)
    return out.with_suffix(".png")


def _fig_d2(paths: Paths) -> Path:
    drift = pd.read_csv(paths.tables / "seta_top_drift_features.csv")
    top = drift.groupby("feature")["abs_z_mean_gap"].max().sort_values(ascending=False).head(20).reset_index()
    top.to_csv(paths.source_data / "fig_D2_top_drift.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.barh(top["feature"][::-1], top["abs_z_mean_gap"][::-1], color="#8da0cb")
    ax.set_xlabel("Maximum absolute z-mean gap")
    ax.set_ylabel("Feature")
    _panel(ax, "a")
    fig.tight_layout()
    out = paths.figures / "fig_D2_top_drift_features"
    _save(fig, out)
    return out.with_suffix(".png")


def _fig_m1(paths: Paths) -> Path:
    construction = pd.read_csv(paths.tables / "seta_construction_effect_models.csv")
    models = pd.read_csv(paths.tables / "seta_repaired_model_comparison.csv")
    construction.to_csv(paths.source_data / "fig_M1_construction.csv", index=False)
    models.to_csv(paths.source_data / "fig_M1_models.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    a, b = ax
    a.bar(construction["construction"], construction["original_unacceptable_recall"], color="#66a61e")
    a.set_ylim(0, 1)
    a.set_ylabel("Original-unacceptable recall")
    a.tick_params(axis="x", rotation=25)
    _panel(a, "a")
    model_colors = {
        "SQI SVM-RBF selected5": "#7f7f7f",
        "SQI SVM-RBF all84": "#bdbdbd",
        "SQI LM-MLP 84-6-1": "#66a61e",
        "12-lead E31-style waveform comparator": "#4c78a8",
    }
    markers = {
        "SQI SVM-RBF selected5": "o",
        "SQI SVM-RBF all84": "s",
        "SQI LM-MLP 84-6-1": "^",
        "12-lead E31-style waveform comparator": "D",
    }
    for _, row in models.iterrows():
        name = str(row["model"])
        b.scatter(
            row["acceptable_recall"],
            row["original_unacceptable_recall"],
            s=42,
            color=model_colors.get(name, "#555555"),
            marker=markers.get(name, "o"),
            label=name.replace("SQI ", "").replace("12-lead ", ""),
            linewidth=0,
        )
    b.set_xlim(0, 1.02)
    b.set_ylim(0, 1.02)
    b.set_xlabel("Acceptable recall")
    b.set_ylabel("Original-unacceptable recall")
    b.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    _panel(b, "b")
    fig.tight_layout()
    out = paths.figures / "fig_M1_seta_model_performance"
    _save(fig, out)
    return out.with_suffix(".png")


def _confusion(ax: plt.Axes, cm: np.ndarray, labels: list[str]) -> None:
    cm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    shown = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{shown[i, j]:.2f}\n{int(cm[i, j])}", ha="center", va="center", fontsize=6)
    return im


def _fig_m2(paths: Paths) -> Path:
    but = read_json(paths.but_models_json)
    svm_metrics = read_json(paths.but / "sqi_baseline" / "models" / "svm_84sqi_linear" / "svm_84sqi_linear_metrics_seed0.json")
    svm_cm = svm_metrics["test"]["confusion_matrix"]
    cm_svm = np.asarray([[svm_cm["tn"], svm_cm["fp"]], [svm_cm["fn"], svm_cm["tp"]]], dtype=int)
    cm_e31 = np.asarray(but["e31"]["three_class"]["confusion"], dtype=int)
    table = pd.read_csv(paths.tables / "but_model_comparison.csv")
    table.to_csv(paths.source_data / "fig_M2_but_models.csv", index=False)
    recall_rows = []
    model_labels = {
        "Linear SVM 84-SQI": "Linear SVM",
        "LM-MLP 84-J-1": "LM-MLP",
        "E31 wave-mechanism Conformer": "E31",
    }
    for _, row in table.iterrows():
        for cls, col in [("good", "good_recall"), ("medium", "intermediate_recall"), ("poor", "poor_recall")]:
            recall_rows.append(
                {
                    "model": model_labels.get(str(row["model"]), str(row["model"])),
                    "class": cls,
                    "recall": pd.to_numeric(row[col], errors="coerce"),
                }
            )
    rec = pd.DataFrame(recall_rows)
    rec.to_csv(paths.source_data / "fig_M2_class_recalls.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1, 1.25, 1.2]})
    _confusion(ax[0], cm_svm, ["medium/bad", "good"])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    _panel(ax[0], "a")
    _confusion(ax[1], cm_e31, ["good", "medium", "bad"])
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    _panel(ax[1], "b")
    classes = ["good", "medium", "poor"]
    models = ["Linear SVM", "LM-MLP", "E31"]
    colors = {"Linear SVM": "#8c8c8c", "LM-MLP": "#66a61e", "E31": "#4c78a8"}
    xs = np.arange(len(classes))
    width = 0.24
    for offset, model in zip([-width, 0, width], models):
        vals = [
            float(rec.loc[rec["model"].eq(model) & rec["class"].eq(cls), "recall"].iloc[0])
            for cls in classes
        ]
        ax[2].bar(xs + offset, vals, width, label=model, color=colors[model])
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(classes)
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("Recall")
    ax[2].legend(fontsize=6, loc="lower right")
    _panel(ax[2], "c")
    fig.tight_layout()
    out = paths.figures / "fig_M2_but_model_comparison"
    _save(fig, out)
    return out.with_suffix(".png")


def run(paths: Paths, *, execute: bool) -> dict[str, Any]:
    if not execute:
        dry("figures", paths)
        return {"step": "figures", "skipped": True}
    ensure_dirs(paths)
    figures = {
        "fig_D1_distribution_repair_summary": str(_fig_d1(paths)),
        "fig_D2_top_drift_features": str(_fig_d2(paths)),
        "fig_M1_seta_model_performance": str(_fig_m1(paths)),
        "fig_M2_but_model_comparison": str(_fig_m2(paths)),
    }
    (paths.reports / "figure_index.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    print(json.dumps(figures, indent=2))
    return {"step": "figures", "skipped": False, "outputs": figures}
