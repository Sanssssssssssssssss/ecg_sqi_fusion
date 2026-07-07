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

from .common import ROOT, Paths, dry, ensure_dirs, feature_cols, read_json
from .seta_sqi import arm_dir
from src.transformer_pipeline.data_v1_gapfill.common import protocol_dir, split_dir

BUT_ALL7_MODEL = "SQI SVM-RBF single-lead all7"
BUT_E31_MODEL = "E31 query-mean fused Conformer"


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
    paper_roots = [paths.out.parent / "sqi12_gapfill" / "sqi_full_rerun_clean"]
    try:
        paper_roots.append(paths.out.parents[2] / "sqi_paper_aligned")
    except IndexError:
        pass
    paper_roots.append(ROOT / "outputs" / "sqi_paper_aligned")
    paper_feat_path = paper_split_path = None
    for paper_root in paper_roots:
        feat = paper_root / "features" / "record84_norm.parquet"
        split = paper_root / "splits" / "split_seta_seed0_paper_balanced.csv"
        if feat.exists() and split.exists():
            paper_feat_path, paper_split_path = feat, split
            break
    if not (our_feat_path.exists() and our_split_path.exists() and paper_feat_path is not None and paper_split_path is not None):
        checked = ", ".join(str(p) for p in paper_roots)
        raise FileNotFoundError(f"missing Set-A SQI feature tables for Fig D3; checked paper roots: {checked}")

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
    but_meta = protocol_dir() / "metadata.csv"
    if not but_meta.exists():
        raise FileNotFoundError(f"missing BUT v116 metadata: {but_meta}")
    df = pd.read_csv(but_meta, low_memory=False)
    official_split = split_dir() / "original_region_atlas.csv"
    if not official_split.exists():
        raise FileNotFoundError(f"missing official BUT fold split: {official_split}")
    fold = pd.read_csv(official_split, low_memory=False)
    fold_cols = [c for c in ["idx", "split", "class_name", "v116_candidate_type", "v116_generated"] if c in fold.columns]
    df["idx"] = pd.to_numeric(df["idx"], errors="raise").astype(int)
    fold = fold[fold_cols].copy()
    fold["idx"] = pd.to_numeric(fold["idx"], errors="raise").astype(int)
    df = df.drop(columns=[c for c in fold_cols if c != "idx" and c in df.columns], errors="ignore")
    df = df.merge(fold, on="idx", how="inner", validate="one_to_one")
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
    plot_df["split_source"] = str(official_split)
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
    transfer = pd.read_csv(paths.tables / "seta_construction_source_only_models.csv")
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
    transfer.to_csv(paths.source_data / "fig_D1_selected5_svm_transfer.csv", index=False)

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
    for bars in [
        b.bar(xs - width, m["c2st_auc"], width, label="C2ST AUC", color="#8da0cb"),
        b.bar(xs, m["rbf_mmd"], width, label="RBF-MMD", color="#66c2a5"),
        b.bar(xs + width, m["swd"], width, label="SWD", color="#fc8d62"),
    ]:
        b.bar_label(bars, fmt="%.2f", padding=1, fontsize=7)
    b.set_xticks(xs)
    b.set_xticklabels(m.index, rotation=25, ha="right")
    b.set_ylabel("Metric value")
    b.legend(fontsize=6)
    _panel(b, "b")

    heat = transfer.set_index("construction")[["auc", "original_unacceptable_recall", "acceptable_recall"]]
    im = c.imshow(heat.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    c.set_xticks(np.arange(heat.shape[1]))
    c.set_xticklabels(["model AUC", "poor R", "accept R"], rotation=25, ha="right")
    c.set_yticks(np.arange(heat.shape[0]))
    c.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = float(heat.iloc[i, j])
            c.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.45 else "black")
    fig.colorbar(im, ax=c, fraction=0.046, pad=0.02)
    _panel(c, "c")

    bars = d.bar(transfer["construction"], transfer["original_unacceptable_recall"], color=[PALETTE.get(x, "#777777") for x in transfer["construction"]])
    d.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
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
    table = pd.read_csv(paths.tables / "seta_repaired_model_comparison.csv")
    rows = []
    for label, part in [
        ("SVM", table.loc[table["model"].eq("SQI SVM-RBF selected5")]),
        ("MLP", table.loc[table["model"].astype(str).str.contains("MLP", na=False)].sort_values("auc", ascending=False)),
        ("Conformer", table.loc[table["model"].astype(str).str.contains("E31|waveform", case=False, na=False)]),
    ]:
        if part.empty:
            continue
        row = part.iloc[0].copy()
        row["figure_label"] = label
        rows.append(row)
    plot = pd.DataFrame(rows)
    plot.to_csv(paths.source_data / "fig_M1_seta_model_confusions.csv", index=False)
    recall_rows = []
    for _, row in plot.iterrows():
        recall_rows.extend(
            [
                {"model": row["figure_label"], "class": "unacceptable", "recall": float(row["original_unacceptable_recall"])},
                {"model": row["figure_label"], "class": "acceptable", "recall": float(row["acceptable_recall"])},
            ]
        )
    rec = pd.DataFrame(recall_rows)
    rec.to_csv(paths.source_data / "fig_M1_seta_model_recalls.csv", index=False)
    fig, ax = plt.subplots(1, 4, figsize=(9.2, 2.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    labels = ["unacceptable", "acceptable"]
    for axis, (_, row), panel in zip(ax[:3], plot.iterrows(), ["a", "b", "c"]):
        cm = json.loads(str(row["confusion"]))
        mat = np.asarray([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=int)
        _confusion(axis, mat, labels)
        axis.set_title(f"{row['figure_label']}  Acc {float(row['acc']):.2f}  AUC {float(row['auc']):.2f}", fontsize=7, pad=3)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        _panel(axis, panel)
    classes = ["unacceptable", "acceptable"]
    xs = np.arange(len(classes))
    models = plot["figure_label"].tolist()
    width = 0.22
    colors = {"SVM": "#8c8c8c", "MLP": "#66a61e", "Conformer": "#4c78a8"}
    offsets = np.linspace(-width, width, len(models))
    for offset, model in zip(offsets, models):
        vals = [float(rec.loc[rec["model"].eq(model) & rec["class"].eq(cls), "recall"].iloc[0]) for cls in classes]
        bars = ax[3].bar(xs + offset, vals, width, label=model, color=colors[model])
        ax[3].bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax[3].set_xticks(xs)
    ax[3].set_xticklabels(classes, rotation=20, ha="right")
    ax[3].set_ylim(0, 1.08)
    ax[3].set_ylabel("Recall")
    ax[3].legend(fontsize=6, loc="lower right")
    _panel(ax[3], "d")
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
            color = "white" if shown[i, j] > 0.55 else "black"
            ax.text(j, i, f"{shown[i, j]:.2f}", ha="center", va="center", fontsize=10, color=color)
    return im


def _fig_m2(paths: Paths) -> Path:
    table = pd.read_csv(paths.tables / "but_model_comparison.csv")
    table.to_csv(paths.source_data / "fig_M2_but_models.csv", index=False)
    mlp = table.loc[table["model"].astype(str).str.contains("LM-MLP", regex=False)].iloc[0]
    e31 = table.loc[table["model"].eq(BUT_E31_MODEL)].iloc[0]
    cm_mlp = np.asarray(json.loads(str(mlp["confusion"])), dtype=int)
    cm_e31 = np.asarray(json.loads(str(e31["confusion"])), dtype=int)
    recall_rows = []
    model_labels = {
        "SQI SVM-RBF selected5": "SVM-RBF selected5",
        BUT_ALL7_MODEL: "SVM-RBF all7",
        str(mlp["model"]): "LM-MLP",
        BUT_E31_MODEL: "Conformer",
    }
    for _, row in table.iterrows():
        for cls, col in [("good", "good_recall"), ("medium", "intermediate_recall"), ("poor", "poor_recall")]:
            recall_rows.append({"model": model_labels.get(str(row["model"]), str(row["model"])), "class": cls, "recall": float(row[col])})
    rec = pd.DataFrame(recall_rows)
    rec.to_csv(paths.source_data / "fig_M2_class_recalls.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.8), gridspec_kw={"width_ratios": [1.2, 1.2, 1.35]})
    _confusion(ax[0], cm_mlp, ["good", "medium", "bad"])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_title("LM-MLP", fontsize=7, pad=3)
    _panel(ax[0], "a")
    _confusion(ax[1], cm_e31, ["good", "medium", "bad"])
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    ax[1].set_title("Conformer", fontsize=7, pad=3)
    _panel(ax[1], "b")
    classes = ["good", "medium", "poor"]
    models = ["SVM-RBF selected5", "SVM-RBF all7", "LM-MLP", "Conformer"]
    colors = {"SVM-RBF selected5": "#bdbdbd", "SVM-RBF all7": "#8c8c8c", "LM-MLP": "#66a61e", "Conformer": "#4c78a8"}
    xs = np.arange(len(classes))
    width = 0.18
    label_lifts = [0.012, 0.032, 0.052, 0.072]
    for lift, offset, model in zip(label_lifts, [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width], models):
        vals = [float(rec.loc[rec["model"].eq(model) & rec["class"].eq(cls), "recall"].iloc[0]) for cls in classes]
        ax[2].bar(xs + offset, vals, width, label=model, color=colors[model])
        for x, v in zip(xs + offset, vals):
            ax[2].text(
                x,
                min(v + lift, 1.22),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=5.8,
                clip_on=False,
            )
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(classes)
    ax[2].set_ylim(0, 1.26)
    ax[2].set_ylabel("Recall")
    ax[2].legend(fontsize=6, loc="lower right")
    _panel(ax[2], "c")
    fig.tight_layout()
    out = paths.figures / "fig_M2_but_model_comparison"
    _save(fig, out)
    return out.with_suffix(".png")


def _fig_m3(paths: Paths) -> Path:
    boundary = pd.read_csv(paths.tables / "but_good_medium_boundary_audit.csv")
    records = pd.read_csv(paths.tables / "but_good_medium_boundary_records.csv")
    boundary.to_csv(paths.source_data / "fig_M3_boundary_counts.csv", index=False)
    score = records.loc[records["true_class"].isin(["good", "medium"])].copy()
    score["mlp_gm_margin"] = score["mlp_good_prob"] - score["mlp_medium_prob"]
    score["conformer_gm_margin"] = score["conformer_good_prob"] - score["conformer_medium_prob"]
    score.to_csv(paths.source_data / "fig_M3_score_overlap.csv", index=False)
    margins = score[["record_id", "true_class", "mlp_gm_margin", "conformer_gm_margin", "mlp_pred", "conformer_pred"]].copy()
    margins.to_csv(paths.source_data / "fig_M3_margin_distributions.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.45, 1, 1]})
    ax_a, ax_b, ax_c = ax

    metrics = ["good_to_medium", "medium_to_good", "good_to_bad", "bad_to_good"]
    labels = ["good to medium", "medium to good", "good to bad", "bad to good"]
    models = boundary["model"].tolist()
    colors = {BUT_ALL7_MODEL: "#8c8c8c", "SQI LM-MLP 7-8-1 OvR": "#66a61e", "Conformer": "#4c78a8"}
    xs = np.arange(len(metrics))
    width = 0.22
    offsets = np.linspace(-width, width, len(models))
    for offset, model in zip(offsets, models):
        row = boundary.loc[boundary["model"].eq(model)].iloc[0]
        vals = [int(row[m]) for m in metrics]
        bars = ax_a.bar(xs + offset, vals, width, color=colors.get(model, "#999999"), label=model)
        ax_a.bar_label(bars, labels=[str(v) for v in vals], padding=1, fontsize=7)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labels, rotation=18, ha="right")
    ax_a.set_ylabel("Count")
    ax_a.legend(fontsize=6)
    _panel(ax_a, "a")

    def overlap(axis: plt.Axes, column: str, title: str) -> None:
        bins = np.linspace(-1.0, 1.0, 42)
        for cls, color in [("good", "#4c78a8"), ("medium", "#c44e52")]:
            vals = pd.to_numeric(score.loc[score["true_class"].eq(cls), column], errors="coerce").dropna().to_numpy()
            hist, edges = np.histogram(vals, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2.0
            kernel = np.ones(3) / 3.0
            smooth = np.convolve(hist, kernel, mode="same")
            axis.fill_between(centers, smooth, alpha=0.28, color=color, linewidth=0)
            axis.plot(centers, smooth, color=color, lw=1.2, label=cls)
        axis.axvline(0, color="#333333", lw=0.8, ls=":")
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel("Good-minus-medium margin")
        axis.set_title(title, fontsize=7, pad=3)
        axis.set_ylabel("Density")
        axis.legend(fontsize=6)

    overlap(ax_b, "mlp_gm_margin", "LM-MLP")
    _panel(ax_b, "b")
    overlap(ax_c, "conformer_gm_margin", "Conformer")
    _panel(ax_c, "c")
    out = paths.figures / "fig_M3_but_good_medium_boundary_audit"
    _save(fig, out)
    return out.with_suffix(".png")


def _norm_wave(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    scale = max(float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5)), 1e-6)
    return np.clip((x - np.nanmedian(x)) / scale * 2.0, -2.5, 2.5)


def _but_test_sqi_features(paths: Paths) -> tuple[pd.DataFrame, list[str]]:
    feat_path = paths.but / "sqi_baseline" / "features" / "record84_norm.parquet"
    split_path = paths.but / "sqi_baseline" / "data" / "split_but_v116_good_vs_rest.csv"
    feat = pd.read_parquet(feat_path)
    split = pd.read_csv(split_path)
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    cols = [c for c in feat.columns if "__" in c]
    if len(cols) != 84:
        raise RuntimeError(f"expected 84 SQI columns, got {len(cols)}")
    test = split.loc[split["split"].eq("test"), ["record_id"]].merge(feat[["record_id", *cols]], on="record_id", how="inner")
    return test, cols


def _fig_m4(paths: Paths) -> Path:
    records = pd.read_csv(paths.tables / "but_good_medium_boundary_records.csv")
    feat, cols = _but_test_sqi_features(paths)
    records["record_id"] = records["record_id"].astype(str)
    df = records.merge(feat, on="record_id", how="inner")
    good = df.loc[df["true_class"].eq("good")].copy()
    medium = df.loc[df["true_class"].eq("medium") & df["mlp_pred"].eq("good") & df["conformer_pred"].eq("medium")].copy()
    if good.empty or medium.empty:
        raise FileNotFoundError("not enough MLP-wrong/Conformer-correct good-medium examples")

    scaler = StandardScaler().fit(df.loc[df["true_class"].isin(["good", "medium"]), cols].to_numpy(np.float64))
    xg = scaler.transform(good[cols].to_numpy(np.float64))
    xm = scaler.transform(medium[cols].to_numpy(np.float64))
    pairs = []
    for mi, row in enumerate(medium.itertuples(index=False)):
        d = np.linalg.norm(xg - xm[mi], axis=1)
        gi = int(np.argmin(d))
        grow = good.iloc[gi]
        pairs.append(
            {
                "good_record_id": grow["record_id"],
                "medium_record_id": getattr(row, "record_id"),
                "good_source_idx": int(grow["source_idx"]),
                "medium_source_idx": int(getattr(row, "source_idx")),
                "sqi_distance": float(d[gi]),
                "good_l_detail": float(grow["l_detail"]),
                "medium_l_detail": float(getattr(row, "l_detail")),
                "local_detail_delta": abs(float(grow["l_detail"]) - float(getattr(row, "l_detail"))),
                "good_mlp_margin": float(grow["mlp_good_prob"] - grow["mlp_medium_prob"]),
                "medium_mlp_margin": float(getattr(row, "mlp_good_prob") - getattr(row, "mlp_medium_prob")),
                "good_conformer_margin": float(grow["conformer_good_prob"] - grow["conformer_medium_prob"]),
                "medium_conformer_margin": float(getattr(row, "conformer_good_prob") - getattr(row, "conformer_medium_prob")),
                "good_conformer_medium_prob": float(grow["conformer_medium_prob"]),
                "medium_conformer_medium_prob": float(getattr(row, "conformer_medium_prob")),
            }
        )
    pair_df = pd.DataFrame(pairs)
    near = pair_df.loc[pair_df["sqi_distance"].le(pair_df["sqi_distance"].quantile(0.5))].copy()
    if len(near) < 3:
        near = pair_df.copy()
    near["rank_score"] = (
        near["medium_conformer_medium_prob"]
        - near["good_conformer_medium_prob"]
        - 0.15 * near["sqi_distance"]
        + 0.5 * near["local_detail_delta"]
    )
    chosen = near.sort_values(["rank_score", "sqi_distance"], ascending=[False, True]).head(3).reset_index(drop=True)
    sqi_names = ["iSQI", "bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI"]
    sqi_labels = [("i", "iSQI"), ("b", "bSQI"), ("p", "pSQI"), ("s", "sSQI"), ("k", "kSQI"), ("f", "fSQI"), ("bas", "basSQI")]
    sqi_cols = [f"I__{name}" for name in sqi_names]
    by_record = df.set_index("record_id")
    for kind in ["good", "medium"]:
        for name, col in zip(sqi_names, sqi_cols):
            chosen[f"{kind}_{name}"] = [float(by_record.loc[str(rid), col]) for rid in chosen[f"{kind}_record_id"]]
    chosen.to_csv(paths.source_data / "fig_M4_selection.csv", index=False)

    signals = np.load(Path("outputs") / "transformer" / "v116_e31" / "source" / "processed_butqdb" / "signals.npz", allow_pickle=True)["X"]
    time = np.arange(signals.shape[-1]) / 125.0
    wave_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(len(chosen), 2, figsize=(7.4, 5.4), sharex=True, sharey=True)
    ax = np.asarray(ax).reshape(len(chosen), 2)
    for i, row in chosen.iterrows():
        for j, kind in enumerate(["good", "medium"]):
            source_idx = int(row[f"{kind}_source_idx"])
            wave = _norm_wave(signals[source_idx, 0])
            axis = ax[i, j]
            color = "#4c78a8" if kind == "good" else "#c44e52"
            axis.plot(time, wave, color=color, lw=0.8)
            axis.set_xlim(0, 10)
            axis.set_ylim(-1.0, 2.8)
            axis.set_xticks([0, 2, 4, 6, 8, 10])
            axis.set_yticks([-1, 0, 1, 2])
            if i == 0:
                axis.text(
                    0.5,
                    1.34,
                    "true good\nSQI-nearest neighbour" if kind == "good" else "true medium\nMLP good, Conformer medium",
                    transform=axis.transAxes,
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    clip_on=False,
                )
            if j == 0:
                axis.set_ylabel(f"Pair {i + 1}\nAmplitude (a.u.)")
            if i == len(chosen) - 1:
                axis.set_xlabel("Time (s)")
            metric_1 = f"SQI distance={row['sqi_distance']:.2f}  Local detail={row[f'{kind}_l_detail']:.2f}"
            metric_2 = f"Conformer P(medium)={row[f'{kind}_conformer_medium_prob']:.2f}"
            sqi_line1 = " ".join(f"{label}={row[f'{kind}_{name}']:.2f}" for label, name in sqi_labels[:4])
            sqi_line2 = " ".join(f"{label}={row[f'{kind}_{name}']:.2f}" for label, name in sqi_labels[4:])
            axis.text(
                0.0,
                1.04,
                "\n".join([metric_1, metric_2, f"SQI {sqi_line1}", sqi_line2]),
                transform=axis.transAxes,
                fontsize=6.1,
                va="bottom",
                ha="left",
                linespacing=1.08,
                clip_on=False,
            )
            wave_rows.extend(
                {
                    "pair": int(i + 1),
                    "class": kind,
                    "record_id": row[f"{kind}_record_id"],
                    "source_idx": source_idx,
                    "time_s": float(t),
                    "z": float(v),
                    "sqi_distance": float(row["sqi_distance"]),
                    "l_detail": float(row[f"{kind}_l_detail"]),
                    "conformer_medium_prob": float(row[f"{kind}_conformer_medium_prob"]),
                    **{name: float(row[f"{kind}_{name}"]) for name in sqi_names},
                }
                for t, v in zip(time, wave)
            )
    pd.DataFrame(wave_rows).to_csv(paths.source_data / "fig_M4_mlp_error_conformer_correct_examples.csv", index=False)
    fig.tight_layout(h_pad=2.2)
    out = paths.figures / "fig_M4_but_mlp_error_conformer_correct_examples"
    _save(fig, out)
    return out.with_suffix(".png")


def run(paths: Paths, *, execute: bool, scope: str = "all") -> dict[str, Any]:
    if not execute:
        dry("figures", paths)
        return {"step": "figures", "skipped": True, "scope": scope}
    ensure_dirs(paths)
    if scope not in {"all", "seta", "but"}:
        raise ValueError(f"unknown figure scope: {scope}")
    figures: dict[str, str] = {}
    if scope in {"all", "seta"}:
        figures.update(
            {
                "fig_D1_distribution_repair_summary": str(_fig_d1(paths)),
                "fig_D2_top_drift_features": str(_fig_d2(paths)),
                "fig_D3_seta_ours_vs_paper_em_ma_distribution": str(_fig_d3_seta_ours_vs_paper(paths)),
                "fig_M1_seta_model_performance": str(_fig_m1(paths)),
            }
        )
    if scope in {"all", "but"}:
        figures["fig_D4_but_medium_bad_gapfill_distribution"] = str(_fig_d4_but_balanced_classes(paths))
        figures["fig_M2_but_model_comparison"] = str(_fig_m2(paths))
        figures["fig_M3_but_good_medium_boundary_audit"] = str(_fig_m3(paths))
        figures["fig_M4_but_mlp_error_conformer_correct_examples"] = str(_fig_m4(paths))
        m5 = paths.figures / "fig_M5_but_query_patching.png"
        if m5.exists():
            figures["fig_M5_but_query_patching"] = str(m5)
    (paths.reports / "figure_index.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    print(json.dumps(figures, indent=2))
    return {"step": "figures", "skipped": False, "scope": scope, "outputs": figures}
