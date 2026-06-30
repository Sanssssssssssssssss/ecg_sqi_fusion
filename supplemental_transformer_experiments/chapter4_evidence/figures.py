from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
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

BUT_SHARED_PCA_FEATURES = [
    "raw_rms",
    "raw_ptp_p99_p01",
    "raw_diff_abs_p95",
    "rms",
    "std",
    "mean_abs",
    "ptp_p99_p01",
    "low_amp_ratio",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "baseline_step",
    "diff_abs_p95",
    "detail_instability",
    "qrs_band_ratio",
    "qrs_visibility",
    "qrs_prom_median",
    "qrs_prom_p90",
    "periodicity",
    "template_corr",
    "detector_agreement",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "band_0p3_1",
    "band_1_5",
    "band_5_15",
    "band_15_30",
    "band_30_45",
    "sqi_iSQI",
    "sqi_bSQI",
    "sqi_pSQI",
    "sqi_sSQI",
    "sqi_kSQI",
    "sqi_fSQI",
    "sqi_basSQI",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "sample_entropy_proxy",
    "amplitude_entropy",
    "wavelet_e0",
    "wavelet_e1",
    "wavelet_e2",
    "wavelet_e3",
    "wavelet_e4",
]


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")


def _ellipse(ax: plt.Axes, xy: np.ndarray, *, color: str, lw: float = 1.4, alpha: float = 0.95) -> None:
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 5:
        return
    center = np.median(xy, axis=0)
    scale_xy = np.median(np.abs(xy - center), axis=0)
    scale_xy = np.where(scale_xy > 1e-8, scale_xy, np.std(xy, axis=0) + 1e-8)
    robust_dist = np.sqrt((((xy - center) / scale_xy) ** 2).sum(axis=1))
    core = xy[robust_dist <= np.quantile(robust_dist, 0.9)]
    if len(core) >= 5:
        xy = core
    cov = np.cov(xy.T)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = 2.146  # sqrt(chi2.ppf(0.90, df=2)).
    width, height = 2.0 * scale * np.sqrt(vals)
    patch = Ellipse(
        xy=xy.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        lw=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)


def _sample_for_plot(df: pd.DataFrame, group_col: str, n: int = 350) -> pd.DataFrame:
    sampled = [g.sample(min(len(g), n), random_state=0) for _, g in df.groupby(group_col, sort=False)]
    return pd.concat(sampled, ignore_index=True)


def _sqi_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "__" in c and pd.api.types.is_numeric_dtype(df[c])]


def _feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    vals = []
    for col in features:
        if col in frame.columns:
            vals.append(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64))
        else:
            vals.append(np.zeros(len(frame), dtype=np.float64))
    x = np.vstack(vals).T if vals else np.zeros((len(frame), 0), dtype=np.float64)
    if x.size:
        med = np.nanmedian(x, axis=0)
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.take(med, np.where(bad)[1])
    return x


def _robust_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    scale = np.maximum(q75 - q25, np.nanstd(x, axis=0))
    return center, np.maximum(scale, 1e-5)


def _robust_z(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)


def _fig_d3_seta_ours_vs_paper(paths: Paths) -> Path:
    our_feat_path = paths.seta_arms / "smc_gapfill" / "features" / "record84_norm.parquet"
    our_split_path = paths.seta_arms / "smc_gapfill" / "splits" / "split.csv"
    paper_root = paths.out.parent / "sqi12_gapfill" / "sqi_full_rerun_clean"
    paper_feat_path = paper_root / "features" / "record84_norm.parquet"
    paper_split_path = paper_root / "splits" / "split_seta_seed0_paper_balanced.csv"
    if not (our_feat_path.exists() and our_split_path.exists() and paper_feat_path.exists() and paper_split_path.exists()):
        raise FileNotFoundError("missing Set-A SQI feature tables for Fig D3")

    our_feat = pd.read_parquet(our_feat_path)
    our_split = pd.read_csv(our_split_path)
    paper_feat = pd.read_parquet(paper_feat_path)
    paper_split = pd.read_csv(paper_split_path)
    for df in [our_feat, our_split, paper_feat, paper_split]:
        df["record_id"] = df["record_id"].astype(str)
    our = our_feat.merge(our_split[["record_id", "quality_record", "candidate_type", "is_augmented"]], on="record_id", how="inner")
    paper = paper_feat.merge(paper_split[["record_id", "quality_record", "noise_type", "is_augmented"]], on="record_id", how="inner")
    cols = [c for c in _sqi_feature_cols(paper) if c in our.columns]
    if len(cols) < 2:
        raise RuntimeError("not enough shared SQI columns for Fig D3")

    fit = paper.loc[paper["is_augmented"].eq(0), cols].to_numpy(dtype=float)
    scaler = StandardScaler().fit(np.nan_to_num(fit, nan=0.0, posinf=0.0, neginf=0.0))
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(np.nan_to_num(fit, nan=0.0, posinf=0.0, neginf=0.0)))

    def project(df: pd.DataFrame) -> np.ndarray:
        x = np.nan_to_num(df[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        return pca.transform(scaler.transform(x))

    our_pc = project(our)
    paper_pc = project(paper)
    our_plot = our[["record_id", "y", "quality_record", "candidate_type", "is_augmented"]].copy()
    our_plot["panel"] = "ours"
    our_plot["group"] = np.select(
        [
            our_plot["quality_record"].eq("acceptable"),
            our_plot["candidate_type"].ne("original_seta"),
        ],
        ["original acceptable", "synthesis"],
        default="original unacceptable",
    )
    our_plot["PC1"], our_plot["PC2"] = our_pc[:, 0], our_pc[:, 1]
    paper_plot = paper[["record_id", "y", "quality_record", "noise_type", "is_augmented"]].copy()
    paper_plot["panel"] = "paper"
    paper_plot["group"] = np.select(
        [
            paper_plot["quality_record"].eq("acceptable"),
            paper_plot["noise_type"].astype(str).str.contains("em", case=False, na=False),
            paper_plot["noise_type"].astype(str).str.contains("ma", case=False, na=False),
        ],
        ["original acceptable", "paper EM", "paper MA"],
        default="original unacceptable",
    )
    paper_plot["PC1"], paper_plot["PC2"] = paper_pc[:, 0], paper_pc[:, 1]
    plot_df = pd.concat([our_plot, paper_plot], ignore_index=True, sort=False)
    plot_df.to_csv(paths.source_data / "fig_D3_seta_ours_vs_paper_em_ma_pca.csv", index=False)
    summary = plot_df.groupby(["panel", "group"]).size().reset_index(name="n")
    summary.to_csv(paths.source_data / "fig_D3_seta_ours_vs_paper_em_ma_counts.csv", index=False)

    colors = {
        "original acceptable": "#4c78a8",
        "original unacceptable": "#c44e52",
        "synthesis": "#66a61e",
        "paper EM": "#e6ab02",
        "paper MA": "#8da0cb",
    }
    labels = {
        "original acceptable": "original acceptable",
        "original unacceptable": "original unacceptable",
        "synthesis": "synthesis",
        "paper EM": "paper EM",
        "paper MA": "paper MA",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    for ax, panel, label in zip(axes, ["ours", "paper"], ["a", "b"]):
        sub = _sample_for_plot(plot_df.loc[plot_df["panel"].eq(panel)].copy(), "group", n=260)
        for group, g in sub.groupby("group"):
            ax.scatter(g["PC1"], g["PC2"], s=8, alpha=0.62, color=colors[group], label=labels[group], linewidth=0)
        for group, g in plot_df.loc[plot_df["panel"].eq(panel)].groupby("group"):
            _ellipse(ax, g[["PC1", "PC2"]].to_numpy(), color=colors[group])
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.legend(fontsize=6, loc="best")
        _panel(ax, label)
    fig.tight_layout()
    out = paths.figures / "fig_D3_seta_ours_vs_paper_em_ma_distribution"
    _save(fig, out)
    return out.with_suffix(".png")


def _fig_d4_but_balanced_classes(paths: Paths) -> Path:
    but_meta = (
        paths.out.parent.parent
        / "v116_e31"
        / "analysis"
        / "good_medium_geometry_repair"
        / "clean_but_protocols"
        / "v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876"
        / "metadata.csv"
    )
    if not but_meta.exists():
        raise FileNotFoundError(f"missing BUT v116 metadata: {but_meta}")
    df = pd.read_csv(but_meta, low_memory=False)
    df = df.loc[df["split"].eq("train")].copy()
    df["domain"] = np.where(df["v116_candidate_type"].eq("original_but"), "original BUT", "synthesis")
    features = [f for f in BUT_SHARED_PCA_FEATURES if f in df.columns]
    if len(features) < 2:
        raise RuntimeError("not enough BUT shared-PCA feature columns for Fig D4")
    ref = df.loc[df["domain"].eq("original BUT")].copy()
    center, scale = _robust_fit(_feature_matrix(ref, features))
    z = _robust_z(_feature_matrix(df, features), center, scale)
    pca = PCA(n_components=2, random_state=19)
    pca.fit(_robust_z(_feature_matrix(ref, features), center, scale))
    pc = pca.transform(z)
    plot_df = df.loc[df["class_name"].isin(["medium", "bad"]), ["idx", "split", "class_name", "v116_candidate_type", "domain"]].copy()
    plot_df["PC1"] = pc[df["class_name"].isin(["medium", "bad"]), 0]
    plot_df["PC2"] = pc[df["class_name"].isin(["medium", "bad"]), 1]
    plot_df = plot_df[np.isfinite(plot_df[["PC1", "PC2"]]).all(axis=1)].copy()
    plot_df["coordinate_basis"] = "v116_shared_pca_44_features_fit_original_train"
    plot_df.to_csv(paths.source_data / "fig_D4_but_medium_bad_pca.csv", index=False)
    counts = plot_df.groupby(["class_name", "domain", "v116_candidate_type"]).size().reset_index(name="n")
    counts.to_csv(paths.source_data / "fig_D4_but_medium_bad_counts.csv", index=False)

    colors = {"original BUT": "#2b2b2b", "synthesis": "#4c78a8"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, cls, label in zip(axes, ["medium", "bad"], ["a", "b"]):
        cls_df = plot_df.loc[plot_df["class_name"].eq(cls)]
        sub = _sample_for_plot(cls_df, "domain", n=360)
        for domain, g in sub.groupby("domain"):
            ax.scatter(g["PC1"], g["PC2"], s=8, alpha=0.58, color=colors[domain], label=domain, linewidth=0)
        for domain, g in cls_df.groupby("domain"):
            _ellipse(ax, g[["PC1", "PC2"]].to_numpy(), color=colors[domain])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        for axis_col, setter in [("PC1", ax.set_xlim), ("PC2", ax.set_ylim)]:
            lo, hi = cls_df[axis_col].quantile([0.01, 0.99]).to_numpy(dtype=float)
            pad = max((hi - lo) * 0.12, 1e-3)
            setter(lo - pad, hi + pad)
        ax.text(0.02, 0.96, cls, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.legend(fontsize=6, loc="best")
        _panel(ax, label)
    fig.tight_layout()
    out = paths.figures / "fig_D4_but_medium_bad_gapfill_distribution"
    _save(fig, out)
    return out.with_suffix(".png")


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
        "fig_D3_seta_ours_vs_paper_em_ma_distribution": str(_fig_d3_seta_ours_vs_paper(paths)),
        "fig_D4_but_medium_bad_gapfill_distribution": str(_fig_d4_but_balanced_classes(paths)),
        "fig_M1_seta_model_performance": str(_fig_m1(paths)),
        "fig_M2_but_model_comparison": str(_fig_m2(paths)),
    }
    (paths.reports / "figure_index.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    print(json.dumps(figures, indent=2))
    return {"step": "figures", "skipped": False, "outputs": figures}
