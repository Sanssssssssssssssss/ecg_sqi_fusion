"""Visualize which PTB synthetic rules match CleanBUT-Core clusters.

This is CPU-only and prediction-free.  It fits a common-feature PCA on the
CleanBUT train-target core, projects synthetic audit samples into that same
space, and writes side-by-side figures for the closest rules.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.sqi_pipeline.features.sqi import (
    sqi_basSQI,
    sqi_bSQI_li2008_global,
    sqi_fSQI,
    sqi_kSQI,
    sqi_pSQI,
    sqi_sSQI,
)
from src.transformer_pipeline.external_benchmarks import but_clean_label_core_atlas_10s as clean_atlas
from src.transformer_pipeline.external_benchmarks import but_separability_atlas_10s as atlas


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEAN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_clean_label_core_10s_2026_06_06"
DEFAULT_FIT_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_clean_target_fit_grid_10s_2026_06_06"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_but_clean_target_fit_grid_10s_2026_06_06"

CLASS_ORDER = ["good", "medium", "bad"]
COLORS = {"good": "#1b9e77", "medium": "#fdae61", "bad": "#d73027"}
FS = 125
WELCH_KW = dict(fmax=40.0, window="hann", nperseg=256, noverlap=128, detrend="constant")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def detect_peak_pair(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(x, dtype=float).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - np.median(y)
    std = float(np.std(y))
    r1, _ = find_peaks(np.abs(y), distance=int(0.28 * FS), prominence=max(0.08 * std, 0.01))
    r2, _ = find_peaks(np.maximum(y, 0), distance=int(0.28 * FS), prominence=max(0.06 * std, 0.008))
    r3, _ = find_peaks(-np.minimum(y, 0), distance=int(0.28 * FS), prominence=max(0.06 * std, 0.008))
    return r1.astype(int), r2.astype(int), r3.astype(int)


def extract_synthetic_sqi_one(x: np.ndarray) -> dict[str, float]:
    y = np.asarray(x, dtype=float).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    r1, r2, r3 = detect_peak_pair(y)
    tol_samp = int(round(150 * FS / 1000.0))
    b12 = sqi_bSQI_li2008_global(r1, r2, tol_samp)
    b13 = sqi_bSQI_li2008_global(r1, r3, tol_samp)
    b23 = sqi_bSQI_li2008_global(r2, r3, tol_samp)
    # Synthetic PTB windows are single-lead, so true inter-lead iSQI is not
    # available. Use the strongest detector-agreement proxy and name the caveat
    # in the report rather than silently dropping the dimension.
    i_proxy = float(max(b12, b13, b23))
    return {
        "sqi_iSQI": i_proxy,
        "sqi_bSQI": float(b12),
        "sqi_pSQI": float(sqi_pSQI(y, fs=FS, welch_kwargs=WELCH_KW)),
        "sqi_sSQI": float(sqi_sSQI(y)),
        "sqi_kSQI": float(sqi_kSQI(y)),
        "sqi_fSQI": float(sqi_fSQI(y, flatline_eps=1e-4)),
        "sqi_basSQI": float(sqi_basSQI(y, fs=FS, welch_kwargs=WELCH_KW)),
    }


def compute_full64_variant_features(variant_dir: Path, feature_names: list[str], force: bool = False) -> pd.DataFrame:
    out_path = variant_dir / "synthetic_atlas64_features.csv"
    if out_path.exists() and not force:
        df = pd.read_csv(out_path)
        if all(f in df.columns for f in feature_names):
            return df

    morph_path = variant_dir / "synthetic_morph_features.csv"
    if not morph_path.exists():
        raise FileNotFoundError(morph_path)
    morph = pd.read_csv(morph_path)
    sig_path = variant_dir / "datasets" / "signals.npz"
    if not sig_path.exists():
        sig_path = variant_dir / "datasets" / "synth_10s_125hz_noisy.npz"
    X = np.load(sig_path)["X"].astype(np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    if len(X) != len(morph):
        raise ValueError(f"{variant_dir.name}: signal/features row mismatch {len(X)} vs {len(morph)}")

    extra_rows = [atlas.extract_extra_one(X[i, 0]) for i in range(len(X))]
    sqi_rows = [extract_synthetic_sqi_one(X[i, 0]) for i in range(len(X))]
    full = pd.concat([morph.reset_index(drop=True), pd.DataFrame(extra_rows), pd.DataFrame(sqi_rows)], axis=1)
    full = full.loc[:, ~full.columns.duplicated()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    return full


def load_variant_rows(variant_dir: Path, feature_names: list[str], feature_mode: str, force: bool = False) -> pd.DataFrame:
    path = variant_dir / "synthetic_morph_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if feature_mode == "full64":
        df = compute_full64_variant_features(variant_dir, feature_names, force=force)
    else:
        df = pd.read_csv(path)
    df["variant_id"] = variant_dir.name
    return df


def build_projection(clean_root: Path, fit_root: Path, top_n: int, seed: int, feature_mode: str, force_features: bool) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    thresholds = _read_json(clean_root / "clean_rule_thresholds.json")
    clean = pd.read_csv(clean_root / "clean_subset_manifest.csv")
    leaderboard = pd.read_csv(fit_root / "clean_target_fit_distance_leaderboard.csv").head(top_n)
    feature_names = list(thresholds["features"])

    variant_frames: list[pd.DataFrame] = []
    for _, row in leaderboard.iterrows():
        vdir = Path(str(row["variant_dir"]))
        if not vdir.exists():
            continue
        vf = load_variant_rows(vdir, feature_names, feature_mode=feature_mode, force=force_features)
        for k in [
            "clean_target_score",
            "clean_overall_distance",
            "clean_good_distance",
            "clean_medium_distance",
            "clean_bad_distance",
            "clean_domain_separability_bal_acc",
        ]:
            vf[k] = row.get(k, np.nan)
        variant_frames.append(vf)

    if not variant_frames:
        raise RuntimeError("No synthetic variant feature rows found from leaderboard.")

    synth = pd.concat(variant_frames, ignore_index=True)
    common_features = [c for c in feature_names if c in clean.columns and c in synth.columns]
    if len(common_features) < 8:
        raise RuntimeError(f"Too few common features for PCA: {len(common_features)}")

    clean_target = clean[clean["clean_tier"].isin(["clean_core_strict", "clean_core_train_target"])].copy()
    clean_train_target = clean[clean["is_clean_train_target"].astype(bool)].copy()
    distance_rows: list[dict[str, Any]] = []
    for vid in [v for v in synth["variant_id"].dropna().unique().tolist()]:
        sub = synth[synth["variant_id"].eq(vid)]
        dist = clean_atlas.classwise_distance(clean_train_target, sub, common_features)
        good = float(dist["by_class"].get("good", np.nan))
        medium = float(dist["by_class"].get("medium", np.nan))
        bad = float(dist["by_class"].get("bad", np.nan))
        overall = float(dist["overall"])
        score = float(0.36 * overall + 0.30 * medium + 0.26 * bad + 0.08 * good)
        distance_rows.append(
            {
                "variant_id": vid,
                "full_feature_score": score,
                "full_overall_distance": overall,
                "full_good_distance": good,
                "full_medium_distance": medium,
                "full_bad_distance": bad,
                "n_common_features": int(len(common_features)),
            }
        )
        mask = synth["variant_id"].eq(vid)
        synth.loc[mask, "clean_target_score"] = score
        synth.loc[mask, "clean_overall_distance"] = overall
        synth.loc[mask, "clean_good_distance"] = good
        synth.loc[mask, "clean_medium_distance"] = medium
        synth.loc[mask, "clean_bad_distance"] = bad

    distance_df = pd.DataFrame(distance_rows).sort_values("full_feature_score").reset_index(drop=True)
    distance_df.insert(0, "rank", np.arange(1, len(distance_df) + 1, dtype=int))
    order = distance_df["variant_id"].tolist()
    synth["variant_id"] = pd.Categorical(synth["variant_id"], categories=order, ordered=True)
    synth = synth.sort_values("variant_id").copy()
    synth["variant_id"] = synth["variant_id"].astype(str)

    fit_rows = clean_target[clean_target["split"].eq("train")].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=seed)
    x_fit = _safe_numeric(fit_rows, common_features)
    x_fit = imputer.fit_transform(x_fit)
    x_fit = scaler.fit_transform(x_fit)
    pca.fit(x_fit)

    clean_plot = []
    for cname in CLASS_ORDER:
        part = clean_target[clean_target["class_name"].eq(cname)]
        part = _sample(part, 1400, seed + CLASS_ORDER.index(cname))
        clean_plot.append(part)
    clean_plot_df = pd.concat(clean_plot, ignore_index=True)
    clean_xy = pca.transform(scaler.transform(imputer.transform(_safe_numeric(clean_plot_df, common_features))))
    clean_plot_df["pc1_common"] = clean_xy[:, 0]
    clean_plot_df["pc2_common"] = clean_xy[:, 1]
    clean_plot_df["source"] = "CleanBUT-Core"
    clean_plot_df["variant_id"] = "CleanBUT-Core"

    synth_xy = pca.transform(scaler.transform(imputer.transform(_safe_numeric(synth, common_features))))
    synth["pc1_common"] = synth_xy[:, 0]
    synth["pc2_common"] = synth_xy[:, 1]
    synth["source"] = "PTB synthetic"

    cols = [
        "source",
        "variant_id",
        "class_name",
        "pc1_common",
        "pc2_common",
        "clean_target_score",
        "clean_overall_distance",
        "clean_good_distance",
        "clean_medium_distance",
        "clean_bad_distance",
        "clean_domain_separability_bal_acc",
    ]
    clean_for_out = clean_plot_df.copy()
    for c in cols:
        if c not in clean_for_out.columns:
            clean_for_out[c] = np.nan
    projected = pd.concat([clean_for_out[cols], synth[cols]], ignore_index=True)

    meta = {
        "common_feature_count": len(common_features),
        "common_features": common_features,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "fit_rows": int(len(fit_rows)),
        "variants": order,
        "feature_mode": feature_mode,
        "synthetic_iSQI_note": "Synthetic PTB is single-lead; sqi_iSQI is a detector-agreement proxy, not true inter-lead iSQI." if feature_mode == "full64" else "",
    }
    return projected, meta, distance_df


def plot_overlay(projected: pd.DataFrame, out_path: Path, top_n: int) -> None:
    clean = projected[projected["source"].eq("CleanBUT-Core")]
    variants = [v for v in projected["variant_id"].dropna().unique().tolist() if v != "CleanBUT-Core"][:top_n]
    ncols = 3
    nrows = math.ceil(len(variants) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    for ax, vid in zip(axes.ravel(), variants):
        for cname in CLASS_ORDER:
            cdf = clean[clean["class_name"].eq(cname)]
            ax.scatter(cdf["pc1_common"], cdf["pc2_common"], s=6, c=COLORS[cname], alpha=0.16, linewidths=0)
        vf = projected[projected["variant_id"].eq(vid)]
        for cname in CLASS_ORDER:
            sdf = vf[vf["class_name"].eq(cname)]
            ax.scatter(
                sdf["pc1_common"],
                sdf["pc2_common"],
                s=24,
                c=COLORS[cname],
                alpha=0.75,
                marker="x",
                linewidths=1.0,
            )
        first = vf.iloc[0]
        title = (
            f"{vid}\n"
            f"score={first['clean_target_score']:.3f} "
            f"g/m/b={first['clean_good_distance']:.2f}/{first['clean_medium_distance']:.2f}/{first['clean_bad_distance']:.2f}"
        )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("CleanBUT common-PC1")
        ax.set_ylabel("CleanBUT common-PC2")
        ax.grid(alpha=0.15)
    for ax in axes.ravel()[len(variants) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"Clean {c}", markerfacecolor=COLORS[c], markersize=7, alpha=0.6)
        for c in CLASS_ORDER
    ] + [
        plt.Line2D([0], [0], marker="x", color="#333333", label="PTB synthetic", markersize=7, linestyle="None")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("PTB synthetic rules projected onto CleanBUT-Core morphology/QRS PCA", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_single(projected: pd.DataFrame, out_path: Path) -> None:
    clean = projected[projected["source"].eq("CleanBUT-Core")]
    variants = [v for v in projected["variant_id"].dropna().unique().tolist() if v != "CleanBUT-Core"]
    if not variants:
        return
    vid = variants[0]
    vf = projected[projected["variant_id"].eq(vid)]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for cname in CLASS_ORDER:
        cdf = clean[clean["class_name"].eq(cname)]
        ax.scatter(cdf["pc1_common"], cdf["pc2_common"], s=10, c=COLORS[cname], alpha=0.18, linewidths=0, label=f"Clean {cname}")
    for cname in CLASS_ORDER:
        sdf = vf[vf["class_name"].eq(cname)]
        ax.scatter(
            sdf["pc1_common"],
            sdf["pc2_common"],
            s=46,
            c=COLORS[cname],
            alpha=0.82,
            marker="x",
            linewidths=1.35,
            label=f"PTB {cname}",
        )
    first = vf.iloc[0]
    ax.set_title(
        f"Best current CleanBUT-distance rule: {vid}\n"
        f"score={first['clean_target_score']:.3f}; class distances good/medium/bad="
        f"{first['clean_good_distance']:.2f}/{first['clean_medium_distance']:.2f}/{first['clean_bad_distance']:.2f}",
        fontsize=11,
    )
    ax.set_xlabel("CleanBUT common-PC1")
    ax.set_ylabel("CleanBUT common-PC2")
    ax.grid(alpha=0.15)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_centroids(projected: pd.DataFrame, out_path: Path, top_n: int) -> pd.DataFrame:
    clean = projected[projected["source"].eq("CleanBUT-Core")]
    clean_cent = clean.groupby("class_name")[["pc1_common", "pc2_common"]].median()
    variants = [v for v in projected["variant_id"].dropna().unique().tolist() if v != "CleanBUT-Core"][:top_n]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    for cname in CLASS_ORDER:
        ax.scatter(clean_cent.loc[cname, "pc1_common"], clean_cent.loc[cname, "pc2_common"], s=140, c=COLORS[cname], marker="o", edgecolor="black", label=f"Clean {cname}")
    for vid in variants:
        vf = projected[projected["variant_id"].eq(vid)]
        score = float(vf["clean_target_score"].dropna().iloc[0])
        alpha = max(0.18, min(0.9, 1.2 - score))
        for cname in CLASS_ORDER:
            sc = vf[vf["class_name"].eq(cname)][["pc1_common", "pc2_common"]].median()
            cx, cy = clean_cent.loc[cname, ["pc1_common", "pc2_common"]]
            dist = float(np.linalg.norm(sc.to_numpy() - np.array([cx, cy])))
            rows.append({"variant_id": vid, "class_name": cname, "centroid_distance_common_pca": dist, "clean_target_score": score})
            ax.plot([cx, sc["pc1_common"]], [cy, sc["pc2_common"]], color=COLORS[cname], alpha=alpha * 0.4, linewidth=1)
            ax.scatter(sc["pc1_common"], sc["pc2_common"], c=COLORS[cname], marker="x", s=70, alpha=alpha)
    ax.set_title("Class centroid shifts: CleanBUT -> PTB synthetic")
    ax.set_xlabel("CleanBUT common-PC1")
    ax.set_ylabel("CleanBUT common-PC2")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)

    cdf = pd.DataFrame(rows)
    pivot = cdf.pivot_table(index="variant_id", columns="class_name", values="centroid_distance_common_pca", aggfunc="mean")
    pivot = pivot.reindex(variants)
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for cname in CLASS_ORDER:
        vals = pivot[cname].to_numpy()
        axes[1].bar(x, vals, bottom=bottom, color=COLORS[cname], alpha=0.75, label=cname)
        bottom += vals
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([v.replace("__", "\n")[:38] for v in pivot.index], rotation=70, ha="right", fontsize=7)
    axes[1].set_ylabel("Stacked median centroid distance")
    axes[1].set_title("Lower is closer to CleanBUT class centroids")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return cdf


def plot_distance_bars(distance_df: pd.DataFrame, out_path: Path, top_n: int) -> None:
    df = distance_df.head(top_n).copy()
    labels = [v.replace("__", "\n")[:45] for v in df["variant_id"]]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.65), 5.2))
    width = 0.22
    ax.bar(x - width, df["full_good_distance"], width, label="good", color=COLORS["good"], alpha=0.75)
    ax.bar(x, df["full_medium_distance"], width, label="medium", color=COLORS["medium"], alpha=0.75)
    ax.bar(x + width, df["full_bad_distance"], width, label="bad", color=COLORS["bad"], alpha=0.75)
    ax.plot(x, df["full_feature_score"], color="#333333", marker="o", label="weighted score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Class-wise CleanBUT distance")
    ax.set_title("CleanBUT target distance leaderboard: lower is closer")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(report_dir: Path, meta: dict[str, Any], centroid_df: pd.DataFrame, distance_df: pd.DataFrame, top_n: int) -> None:
    leaderboard = distance_df.head(top_n).copy()
    centroid_summary = (
        centroid_df.groupby("variant_id")["centroid_distance_common_pca"]
        .mean()
        .reset_index(name="mean_centroid_distance_common_pca")
    )
    joined = leaderboard.merge(centroid_summary, on="variant_id", how="left")
    joined.to_csv(report_dir / "clean_target_visual_fit_summary.csv", index=False)
    lines = [
        "# CleanBUT-Core Visual Fit",
        "",
        "This report projects PTB synthetic audit samples into a PCA fitted only on the CleanBUT train-target core.",
        "It is CPU-only and uses signal-derived morphology/QRS features, not model predictions.",
        "",
        f"- Common feature count: `{meta['common_feature_count']}`",
        f"- Feature mode: `{meta.get('feature_mode', 'common31')}`",
        f"- PCA explained variance: `{meta['pca_explained_variance_ratio'][0]:.3f}`, `{meta['pca_explained_variance_ratio'][1]:.3f}`",
        "",
        "## Figures",
        "",
        "- `figures/ptb_vs_cleanbut_pca_top_rules.png`: CleanBUT-Core background with top PTB synthetic variants overlaid.",
        "- `figures/ptb_vs_cleanbut_best_rule.png`: single best current rule in the same PCA space.",
        "- `figures/ptb_vs_cleanbut_centroid_shift.png`: class centroid shifts from CleanBUT to synthetic.",
        "- `figures/clean_target_distance_bars.png`: class-wise distance bars from the no-training scan.",
        "",
        "## Current Read",
        "",
        "The closest no-training candidates are mostly `qrs_confound` and `visible_unusable` bad-subtype rules with `bw_badstrong` overlay and a protected medium guard. The 64-feature view is stricter: it checks morphology, SQI proxies, frequency bands, wavelet energy, and complexity features together.",
        "",
        "## Top Variants",
        "",
        md_table(joined[
            [
                "rank",
                "variant_id",
                "full_feature_score",
                "full_good_distance",
                "full_medium_distance",
                "full_bad_distance",
                "n_common_features",
                "mean_centroid_distance_common_pca",
            ]
        ]),
        "",
        "## Caveat",
        "",
        meta.get("synthetic_iSQI_note", "") or "No additional caveats.",
    ]
    (report_dir / "clean_target_visual_fit_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean_root", default=str(DEFAULT_CLEAN_ROOT))
    p.add_argument("--fit_root", default=str(DEFAULT_FIT_ROOT))
    p.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    p.add_argument("--top_n", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--feature_mode", choices=["common31", "full64"], default="full64")
    p.add_argument("--force_features", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    clean_root = Path(args.clean_root)
    fit_root = Path(args.fit_root)
    report_root = Path(args.report_root) / ("clean_target_visual_fit_64d" if args.feature_mode == "full64" else "clean_target_visual_fit")
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    projected, meta, distance_df = build_projection(clean_root, fit_root, int(args.top_n), int(args.seed), str(args.feature_mode), bool(args.force_features))
    projected.to_csv(report_root / "cleanbut_ptb_common_pca_projection.csv", index=False)
    (report_root / "cleanbut_ptb_common_pca_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    distance_df.to_csv(report_root / "clean_target_fit_distance_leaderboard_64d.csv", index=False)
    if args.feature_mode == "full64":
        distance_df.to_csv(fit_root / "clean_target_fit_distance_leaderboard_64d.csv", index=False)

    plot_overlay(projected, fig_dir / "ptb_vs_cleanbut_pca_top_rules.png", int(args.top_n))
    plot_best_single(projected, fig_dir / "ptb_vs_cleanbut_best_rule.png")
    centroid_df = plot_centroids(projected, fig_dir / "ptb_vs_cleanbut_centroid_shift.png", int(args.top_n))
    centroid_df.to_csv(report_root / "cleanbut_ptb_centroid_shift.csv", index=False)
    plot_distance_bars(distance_df, fig_dir / "clean_target_distance_bars.png", max(15, int(args.top_n)))
    write_report(report_root, meta, centroid_df, distance_df, int(args.top_n))

    state = {
        "status": "complete",
        "top_n": int(args.top_n),
        "common_feature_count": meta["common_feature_count"],
        "report": str(report_root / "clean_target_visual_fit_report.md"),
        "overlay_figure": str(fig_dir / "ptb_vs_cleanbut_pca_top_rules.png"),
    }
    (fit_root / "clean_target_visual_fit_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
