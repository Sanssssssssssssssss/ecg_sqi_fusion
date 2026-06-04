"""BUT 10s medium-cluster analysis.

This experiment-only analysis tests whether BUT class-2/medium is merely an
intermediate point between good and bad, or a separate morphology cluster.  It
uses the existing BUT 10s P1 morphology feature table and joins the finding to
historical/current synthetic generator results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT
from src.transformer_pipeline.external_benchmarks.but_morphology_analysis_10s import ALL_FEATURES

NON_MORPH_FEATURES = {"row_index", "sample_order"}
ANALYSIS_FEATURES = [feat for feat in ALL_FEATURES if feat not in NON_MORPH_FEATURES]

RUN_TAG = "e311_but_medium_cluster_analysis_10s_2026_06_03"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG

MORPH_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_morphology_analysis_10s_2026_06_03"
BUT_PROCESSED = ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02" / "processed" / "butqdb"
CURRENT_GRID = ROOT / "outputs" / "external_benchmarks" / "e311_but_morphology_guided_grid_10s_2026_06_03"

CLASS_ORDER = ["good", "medium", "bad"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}


KEY_FEATURES = [
    "qrs_reliable_proxy",
    "prominence_p75",
    "spurious_peak_proxy",
    "peak_density",
    "rr_cv",
    "nonqrs_energy_ratio",
    "ptst_unreliable_proxy",
    "baseline_step_proxy",
    "contact_loss_proxy",
    "flatline_frac",
    "low_amp_frac",
    "baseline_wander",
    "hf_energy",
    "deriv_p95",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def ensure_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out = Path(args.out_root)
    rep = Path(args.report_root)
    vis = out / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    vis.mkdir(parents=True, exist_ok=True)
    return out, rep, vis


def load_but_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(MORPH_ROOT / "but_morph_features.csv")
    df = df[df["variant_id"] == "BUT_QDB"].copy()
    meta = pd.read_csv(BUT_PROCESSED / "metadata.csv")
    audit_path = BUT_PROCESSED / "preprocess_audit.json"
    audit = read_json(audit_path) if audit_path.exists() else {}
    audit["metadata_shape"] = [int(len(meta)), int(len(meta.columns))]
    audit["metadata_class_counts"] = meta["y_class"].value_counts().to_dict()
    audit["metadata_split_counts"] = meta["split"].value_counts().to_dict()
    return df, audit


def robust_matrix(df: pd.DataFrame) -> tuple[np.ndarray, RobustScaler]:
    X = df[ANALYSIS_FEATURES].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler(quantile_range=(25, 75))
    Xs = scaler.fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -8.0, 8.0)
    return Xs, scaler


def centroid_geometry(df: pd.DataFrame, Xs: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = df["y_class"].to_numpy(str)
    centroids = {cls: Xs[y == cls].mean(axis=0) for cls in CLASS_ORDER}
    g = centroids["good"]
    m = centroids["medium"]
    b = centroids["bad"]
    axis = b - g
    axis_norm = float(np.linalg.norm(axis))
    t = float(np.dot(m - g, axis) / (np.dot(axis, axis) + 1e-12))
    projected = g + t * axis
    perp = float(np.linalg.norm(m - projected))
    dgm = float(np.linalg.norm(g - m))
    dmb = float(np.linalg.norm(m - b))
    dgb = float(np.linalg.norm(g - b))
    rows = []
    for cls in CLASS_ORDER:
        c = centroids[cls]
        rows.append(
            {
                "class": cls,
                "distance_to_good": float(np.linalg.norm(c - g)),
                "distance_to_medium": float(np.linalg.norm(c - m)),
                "distance_to_bad": float(np.linalg.norm(c - b)),
                "distance_to_good_bad_axis": float(np.linalg.norm(c - (g + float(np.dot(c - g, axis) / (np.dot(axis, axis) + 1e-12)) * axis))),
            }
        )
    geom = {
        "medium_projection_t_on_good_bad_axis": t,
        "medium_perpendicular_distance": perp,
        "good_bad_centroid_distance": dgb,
        "medium_perpendicular_ratio": perp / (dgb + 1e-12),
        "good_medium_centroid_distance": dgm,
        "medium_bad_centroid_distance": dmb,
        "medium_is_between_on_axis": bool(0.0 <= t <= 1.0),
        "interpretation": "medium_has_independent_component" if perp / (dgb + 1e-12) >= 0.25 else "medium_mostly_intermediate_on_good_bad_axis",
    }
    return pd.DataFrame(rows), geom


def pairwise_feature_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stats = {}
    for cls in CLASS_ORDER:
        sub = df[df["y_class"] == cls]
        stats[cls] = {
            feat: {
                "median": float(sub[feat].median()),
                "iqr": float(sub[feat].quantile(0.75) - sub[feat].quantile(0.25)),
                "mean": float(sub[feat].mean()),
            }
            for feat in ANALYSIS_FEATURES
        }
    pooled_iqr = {}
    for feat in ANALYSIS_FEATURES:
        vals = [stats[c][feat]["iqr"] for c in CLASS_ORDER]
        pooled_iqr[feat] = float(np.nanmedian(vals) + 1e-6)
    for feat in ANALYSIS_FEATURES:
        g = stats["good"][feat]["median"]
        m = stats["medium"][feat]["median"]
        b = stats["bad"][feat]["median"]
        denom = pooled_iqr[feat]
        if abs(b - g) > 1e-9:
            medium_between_fraction = (m - g) / (b - g)
        else:
            medium_between_fraction = np.nan
        if np.isnan(medium_between_fraction):
            medium_shape = "undefined"
        elif 0.0 <= medium_between_fraction <= 1.0:
            medium_shape = "between_good_bad"
        else:
            medium_shape = "outside_good_bad_axis"
        rows.append(
            {
                "feature": feat,
                "good_median": g,
                "medium_median": m,
                "bad_median": b,
                "medium_between_fraction": float(medium_between_fraction) if not np.isnan(medium_between_fraction) else None,
                "medium_shape": medium_shape,
                "gm_robust_delta": float((m - g) / denom),
                "mb_robust_delta": float((b - m) / denom),
                "gb_robust_delta": float((b - g) / denom),
                "medium_unique_score": float(abs((m - g) / denom) + abs((b - m) / denom) - abs((b - g) / denom)),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_gm"] = out["gm_robust_delta"].abs()
    out["abs_mb"] = out["mb_robust_delta"].abs()
    out["abs_gb"] = out["gb_robust_delta"].abs()
    return out.sort_values(["medium_unique_score", "abs_gm", "abs_mb"], ascending=False)


def pca_and_models(df: pd.DataFrame, Xs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], PCA]:
    y = df["y_class"].map(CLASS_TO_INT).to_numpy(int)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(
        {
            "pc1": pcs[:, 0],
            "pc2": pcs[:, 1],
            "y_class": df["y_class"].to_numpy(str),
            "row_index": df["row_index"].to_numpy(int) if "row_index" in df.columns else np.arange(len(df)),
        }
    )
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=17, stratify=y)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    med = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    y_med = (y == CLASS_TO_INT["medium"]).astype(int)
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xs, y_med, test_size=0.25, random_state=23, stratify=y_med)
    med.fit(Xm_train, ym_train)
    med_pred = med.predict(Xm_test)
    coef = med.coef_[0]
    imp = pd.DataFrame({"feature": ANALYSIS_FEATURES, "medium_vs_rest_coef": coef, "abs_coef": np.abs(coef)})
    imp = imp.sort_values("abs_coef", ascending=False)
    try:
        sil = float(silhouette_score(Xs, y, metric="euclidean"))
    except Exception:
        sil = float("nan")
    report = {
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "multiclass_macro_f1_holdout": float(f1_score(y_test, pred, average="macro")),
        "medium_vs_rest_f1_holdout": float(f1_score(ym_test, med_pred)),
        "silhouette_by_expert_label": sil,
        "multiclass_report": classification_report(y_test, pred, target_names=CLASS_ORDER, output_dict=True),
    }
    return pca_df, imp, report, pca


def load_current_grid_rows() -> pd.DataFrame:
    rows = []
    for row in load_jsonl(CURRENT_GRID / "morphology_guided_summary.jsonl"):
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {}) or {}
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        audit = row.get("feature_audit", {}) or {}
        dist = audit.get("distance", {}) or {}
        rows.append(
            {
                "variant_id": row.get("spec", {}).get("id"),
                "family": row.get("spec", {}).get("family"),
                "mode": row.get("mode"),
                "acc": safe_float(rep.get("acc")),
                "balanced_acc": safe_float(rep.get("balanced_acc")),
                "macro_f1": safe_float(rep.get("macro_f1")),
                "good_recall": safe_float(rec[0] if len(rec) > 0 else None),
                "medium_recall": safe_float(rec[1] if len(rec) > 1 else None),
                "bad_recall": safe_float(rec[2] if len(rec) > 2 else None),
                "min_medium_bad": min(safe_float(rec[1] if len(rec) > 1 else None), safe_float(rec[2] if len(rec) > 2 else None)),
                "feature_target_score": safe_float(audit.get("feature_target_score")),
                "overall_but_like_score": safe_float(dist.get("overall_but_like_score")),
                "feature_mismatch": bool(audit.get("feature_mismatch", False)),
            }
        )
    return pd.DataFrame(rows)


def synthetic_alignment(but_df: pd.DataFrame, scaler: RobustScaler, but_centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    path = MORPH_ROOT / "synthetic_morph_feature_summary.csv"
    metric_path = MORPH_ROOT / "grid_metric_distance_join.csv"
    if not path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(path)
    metrics = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    rows = []
    g = but_centroids["good"]
    b = but_centroids["bad"]
    axis = b - g
    dgb = float(np.linalg.norm(axis) + 1e-12)
    for (variant_id, family), gdf in summary.groupby(["variant_id", "family"], dropna=False):
        if not set(CLASS_ORDER).issubset(set(gdf["y_class"])):
            continue
        centers = {}
        for cls in CLASS_ORDER:
            row = gdf[gdf["y_class"] == cls].iloc[0]
            means = np.asarray([safe_float(row.get(f"{feat}_mean")) for feat in ANALYSIS_FEATURES], dtype=float)
            transformed = scaler.transform(means[None, :])[0]
            centers[cls] = np.clip(np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
        m = centers["medium"]
        t = float(np.dot(m - g, axis) / (np.dot(axis, axis) + 1e-12))
        perp = float(np.linalg.norm(m - (g + t * axis)))
        rows.append(
            {
                "variant_id": variant_id,
                "family": family,
                "synthetic_medium_axis_t": t,
                "synthetic_medium_perp_ratio": perp / dgb,
                "synthetic_medium_to_but_medium": float(np.linalg.norm(m - but_centroids["medium"])),
                "synthetic_medium_to_but_good": float(np.linalg.norm(m - but_centroids["good"])),
                "synthetic_medium_to_but_bad": float(np.linalg.norm(m - but_centroids["bad"])),
            }
        )
    out = pd.DataFrame(rows)
    if not metrics.empty:
        out = out.merge(metrics, on=["variant_id", "family"], how="left")
    return out


def plot_pca(pca_df: pd.DataFrame, geom: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"good": "#2f6f9f", "medium": "#c8902f", "bad": "#b64c4c"}
    sample = pca_df.groupby("y_class", group_keys=False).apply(lambda g: g.sample(min(len(g), 900), random_state=3))
    for cls in CLASS_ORDER:
        sub = sample[sample["y_class"] == cls]
        ax.scatter(sub["pc1"], sub["pc2"], s=12, alpha=0.45, color=colors[cls], label=cls)
    ax.set_title("BUT 10s morphology features: PCA by expert class")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.text(
        0.02,
        0.02,
        f"medium projection t={geom['medium_projection_t_on_good_bad_axis']:.2f}, perp ratio={geom['medium_perpendicular_ratio']:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_box_features(df: pd.DataFrame, out: Path) -> None:
    feats = KEY_FEATURES[:10]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    colors = ["#2f6f9f", "#c8902f", "#b64c4c"]
    for ax, feat in zip(axes.ravel(), feats):
        data = [df[df["y_class"] == cls][feat].to_numpy(float) for cls in CLASS_ORDER]
        bp = ax.boxplot(data, labels=CLASS_ORDER, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(feat)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("BUT class distributions: medium is not always the midpoint")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_pairwise(deltas: pd.DataFrame, out: Path) -> None:
    top = deltas.copy()
    top["score"] = top[["abs_gm", "abs_mb", "abs_gb"]].max(axis=1)
    top = top.sort_values("score", ascending=False).head(12)
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y - 0.24, top["gm_robust_delta"], height=0.24, label="medium-good", color="#c8902f")
    ax.barh(y, top["mb_robust_delta"], height=0.24, label="bad-medium", color="#b64c4c")
    ax.barh(y + 0.24, top["gb_robust_delta"], height=0.24, label="bad-good", color="#5e6c84")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Robust median difference, normalized by class IQR")
    ax.set_title("Feature boundaries: medium differs from good and bad on different axes")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, out: Path) -> None:
    top = imp.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c8902f" if v > 0 else "#6d7f8d" for v in top["medium_vs_rest_coef"]]
    ax.barh(top["feature"], top["medium_vs_rest_coef"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Medium-vs-rest linear signature")
    ax.set_xlabel("Logistic coefficient after robust scaling")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_synthetic_alignment(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    metric = "macro_f1" if "macro_f1" in df.columns else "synthetic_medium_to_but_medium"
    families = list(df["family"].dropna().unique())[:12]
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(families))))
    fmap = {fam: palette[i] for i, fam in enumerate(families)}
    for fam, sub in df.groupby("family"):
        ax.scatter(
            sub["synthetic_medium_perp_ratio"],
            sub[metric],
            s=35,
            alpha=0.75,
            label=fam if fam in fmap else None,
            color=fmap.get(fam, "#999999"),
        )
    ax.set_xlabel("Synthetic medium perpendicular ratio to BUT good-bad axis")
    ax.set_ylabel(metric)
    ax.set_title("Synthetic medium geometry vs BUT performance")
    ax.grid(alpha=0.2)
    if len(families) <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_current_grid(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["medium_recall"], df["bad_recall"], s=60, c=df["macro_f1"], cmap="viridis", alpha=0.85)
    ax.axhline(0.866, color="#b64c4c", linestyle="--", linewidth=1, label="b10 bad recall")
    ax.axvline(0.724, color="#c8902f", linestyle="--", linewidth=1, label="b10 medium recall")
    ax.set_xlabel("BUT medium recall")
    ax.set_ylabel("BUT bad recall")
    ax.set_title("Current morphology-guided grid: medium/bad trade-off")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def build_markdown(
    out_root: Path,
    report_root: Path,
    audit: dict[str, Any],
    geom: dict[str, Any],
    deltas: pd.DataFrame,
    model_report: dict[str, Any],
    synthetic_df: pd.DataFrame,
    current_grid: pd.DataFrame,
) -> None:
    independent = deltas[deltas["medium_shape"] == "outside_good_bad_axis"].sort_values("medium_unique_score", ascending=False)
    top_ind = independent.head(8)
    top_med = deltas.sort_values("abs_gm", ascending=False).head(8)
    lines = [
        "# BUT Medium Cluster Analysis 10s",
        "",
        "## Technical Summary",
        "",
        f"- BUT processed data has `{audit.get('metadata_shape', ['?', '?'])[0]}` windows; this analysis uses the existing morphology feature sample with balanced class sampling from the formal 10s P1 protocol.",
        f"- Medium projects to `t={geom['medium_projection_t_on_good_bad_axis']:.3f}` on the good-bad centroid axis, but its perpendicular ratio is `{geom['medium_perpendicular_ratio']:.3f}`. Interpretation: `{geom['interpretation']}`.",
        f"- Holdout morphology-feature classifier macro-F1 is `{model_report['multiclass_macro_f1_holdout']:.3f}`; medium-vs-rest F1 is `{model_report['medium_vs_rest_f1_holdout']:.3f}`.",
        "- The current generator direction is correct only partly: feature audits move toward BUT, but current grid rows still trade medium and bad off rather than forming the medium-specific cluster.",
        "",
        "## Medium Is Not Just Halfway Between Good And Bad",
        "",
        "The good-bad axis test asks whether class 2 lies on the straight line from class 1 to class 3 in robust-scaled morphology feature space. A large perpendicular component means medium has its own morphology signature.",
        "",
        f"- good-medium centroid distance: `{geom['good_medium_centroid_distance']:.3f}`",
        f"- medium-bad centroid distance: `{geom['medium_bad_centroid_distance']:.3f}`",
        f"- good-bad centroid distance: `{geom['good_bad_centroid_distance']:.3f}`",
        f"- medium perpendicular / good-bad distance: `{geom['medium_perpendicular_ratio']:.3f}`",
        "",
        "## Features Where Medium Behaves Like An Independent State",
        "",
        "| feature | good median | medium median | bad median | medium position | unique score |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for _, r in top_ind.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['good_median']:.4f} | {r['medium_median']:.4f} | {r['bad_median']:.4f} | "
            f"{r['medium_between_fraction']:.3f} | {r['medium_unique_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Good-vs-Medium Separators",
            "",
            "| feature | robust medium-good delta | robust bad-medium delta |",
            "|---|---:|---:|",
        ]
    )
    for _, r in top_med.iterrows():
        lines.append(f"| `{r['feature']}` | {r['gm_robust_delta']:.3f} | {r['mb_robust_delta']:.3f} |")
    lines.extend(
        [
            "",
            "## Generator Implication",
            "",
            "- Medium should be generated as **QRS usable but measurement-details unreliable**, not as weaker bad or stronger good.",
            "- For medium, emphasize P/T/ST ambiguity, non-QRS energy, local derivative spikes, baseline micro-steps, and mild contact/ramp events while preserving QRS timing/prominence.",
            "- For bad, emphasize QRS-confounding pseudo-peaks/contact events; if bad pressure is increased through flatline/low-amplitude alone, medium collapses.",
            "- Next grid selection should penalize rows where medium recall only improves by sacrificing bad recall, and it should retain an explicit medium-cluster geometry score.",
            "",
            "## Visual Artifacts",
            "",
            f"- PCA by class: `{out_root / 'visuals' / 'but_pca_by_class.png'}`",
            f"- Key feature boxplots: `{out_root / 'visuals' / 'but_key_feature_boxplots.png'}`",
            f"- Pairwise feature boundaries: `{out_root / 'visuals' / 'pairwise_feature_boundaries.png'}`",
            f"- Medium feature importance: `{out_root / 'visuals' / 'medium_vs_rest_importance.png'}`",
            f"- Synthetic medium alignment: `{out_root / 'visuals' / 'synthetic_medium_alignment.png'}`",
            f"- Current grid trade-off: `{out_root / 'visuals' / 'current_grid_medium_bad_tradeoff.png'}`",
            "",
            "## Caveat",
            "",
            "This is descriptive morphology analysis. It explains the label geometry and generator failure mode, but it does not by itself prove a new synthetic rule will improve BUT test performance.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (out_root / "medium_cluster_analysis_report.md").write_text(text, encoding="utf-8")
    (report_root / "medium_cluster_analysis_report.md").write_text(text, encoding="utf-8")


def build_artifact_payload(
    report_root: Path,
    geom: dict[str, Any],
    deltas: pd.DataFrame,
    imp: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    current_grid: pd.DataFrame,
    model_report: dict[str, Any],
) -> None:
    independent = deltas[deltas["medium_shape"] == "outside_good_bad_axis"].sort_values("medium_unique_score", ascending=False).head(12)
    boundary = deltas.sort_values("abs_gm", ascending=False).head(14)
    imp_rows = imp.head(14)
    synth_rows = synthetic_df.sort_values("synthetic_medium_to_but_medium").head(80) if not synthetic_df.empty else pd.DataFrame()
    grid_rows = current_grid.sort_values("macro_f1", ascending=False).head(80) if not current_grid.empty else pd.DataFrame()
    snapshot = {
        "version": 1,
        "status": "ready",
        "datasets": {
            "medium_independent_features": independent.where(pd.notnull(independent), None).to_dict(orient="records"),
            "good_medium_boundaries": boundary.where(pd.notnull(boundary), None).to_dict(orient="records"),
            "medium_importance": imp_rows.where(pd.notnull(imp_rows), None).to_dict(orient="records"),
            "synthetic_alignment": synth_rows.where(pd.notnull(synth_rows), None).to_dict(orient="records"),
            "current_grid": grid_rows.where(pd.notnull(grid_rows), None).to_dict(orient="records"),
            "geometry": [
                {
                    "metric": "medium_projection_t_on_good_bad_axis",
                    "value": geom["medium_projection_t_on_good_bad_axis"],
                },
                {"metric": "medium_perpendicular_ratio", "value": geom["medium_perpendicular_ratio"]},
                {"metric": "multiclass_macro_f1_holdout", "value": model_report["multiclass_macro_f1_holdout"]},
                {"metric": "medium_vs_rest_f1_holdout", "value": model_report["medium_vs_rest_f1_holdout"]},
            ],
        },
    }
    source = {
        "id": "but_medium_cluster_analysis",
        "label": "BUT QDB 10s P1 morphology features",
        "path": str(MORPH_ROOT / "but_morph_features.csv"),
        "query": {
            "description": "Local Python analysis of BUT QDB 10s P1 morphology features and synthetic generator summaries.",
            "language": "python",
            "sql": "SELECT * FROM but_morph_features WHERE variant_id = 'BUT_QDB';",
            "tables_used": [
                str(MORPH_ROOT / "but_morph_features.csv"),
                str(MORPH_ROOT / "synthetic_morph_feature_summary.csv"),
                str(CURRENT_GRID / "morphology_guided_summary.jsonl"),
            ],
            "metric_definitions": [
                "medium_perpendicular_ratio = distance from medium centroid to good-bad centroid axis divided by good-bad centroid distance.",
                "medium_unique_score = |medium-good robust delta| + |bad-medium robust delta| - |bad-good robust delta|.",
            ],
        },
    }
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "BUT Medium Cluster Analysis 10s",
        "description": "Tests whether BUT medium is an independent morphology cluster and connects that to synthetic generator tuning.",
        "sources": [source],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# BUT Medium Cluster Analysis 10s"},
            {
                "id": "summary",
                "type": "markdown",
                "body": (
                    "## Technical Summary\n\n"
                    f"Medium is not well described as a simple midpoint between good and bad: its perpendicular ratio to the good-bad axis is `{geom['medium_perpendicular_ratio']:.3f}`. "
                    "The practical generator implication is to model medium as QRS-usable but detail-unreliable, rather than as a weaker bad class."
                ),
            },
            {"id": "geometry_chart_block", "type": "chart", "chartId": "geometry_chart"},
            {"id": "boundary_chart_block", "type": "chart", "chartId": "boundary_chart"},
            {"id": "importance_chart_block", "type": "chart", "chartId": "importance_chart"},
            {"id": "grid_chart_block", "type": "chart", "chartId": "grid_chart"},
            {"id": "independent_table_block", "type": "table", "tableId": "independent_table"},
            {
                "id": "implication",
                "type": "markdown",
                "body": (
                    "## Implication For The Running Grid\n\n"
                    "Rows that improve morphology distance without improving medium/bad recall should be treated as proxy-insufficient. "
                    "The next useful synthetic rules should preserve QRS for medium while adding P/T/ST unreliability, and reserve QRS-confounding pseudo-peaks/contact artifacts for bad."
                ),
            },
        ],
        "charts": [
            {
                "id": "geometry_chart",
                "title": "Medium has a measurable off-axis component",
                "type": "bar",
                "dataset": "geometry",
                "encodings": {"x": {"field": "metric"}, "y": {"field": "value"}},
                "source": source,
            },
            {
                "id": "boundary_chart",
                "title": "Strongest good-medium feature boundaries",
                "type": "bar",
                "dataset": "good_medium_boundaries",
                "encodings": {"x": {"field": "feature"}, "y": {"field": "gm_robust_delta"}},
                "source": source,
            },
            {
                "id": "importance_chart",
                "title": "Medium-vs-rest morphology signature",
                "type": "bar",
                "dataset": "medium_importance",
                "encodings": {"x": {"field": "feature"}, "y": {"field": "medium_vs_rest_coef"}},
                "source": source,
            },
            {
                "id": "grid_chart",
                "title": "Current grid still trades medium against bad",
                "type": "scatter",
                "dataset": "current_grid",
                "encodings": {
                    "x": {"field": "medium_recall"},
                    "y": {"field": "bad_recall"},
                    "color": {"field": "family"},
                },
                "source": source,
            },
        ],
        "tables": [
            {
                "id": "independent_table",
                "title": "Features where medium is not a simple midpoint",
                "dataset": "medium_independent_features",
                "columns": [
                    {"field": "feature", "label": "Feature", "type": "text"},
                    {"field": "medium_between_fraction", "label": "Position on good-bad axis", "type": "number"},
                    {"field": "medium_unique_score", "label": "Unique score", "type": "number"},
                    {"field": "gm_robust_delta", "label": "Medium-good delta", "type": "number"},
                    {"field": "mb_robust_delta", "label": "Bad-medium delta", "type": "number"},
                ],
                "source": source,
            }
        ],
    }
    write_json(report_root / "data_analytics_manifest.json", manifest)
    write_json(report_root / "data_analytics_snapshot.json", snapshot)


def run(args: argparse.Namespace) -> None:
    out_root, report_root, vis = ensure_dirs(args)
    df, audit = load_but_features()
    Xs, scaler = robust_matrix(df)
    centroid_df, geom = centroid_geometry(df, Xs)
    y = df["y_class"].to_numpy(str)
    centroids = {cls: Xs[y == cls].mean(axis=0) for cls in CLASS_ORDER}
    deltas = pairwise_feature_deltas(df)
    pca_df, imp, model_report, _pca = pca_and_models(df, Xs)
    synth = synthetic_alignment(df, scaler, centroids)
    current_grid = load_current_grid_rows()

    centroid_df.to_csv(out_root / "but_class_centroid_geometry.csv", index=False)
    deltas.to_csv(out_root / "feature_pairwise_robust_deltas.csv", index=False)
    pca_df.to_csv(out_root / "but_pca_projection.csv", index=False)
    imp.to_csv(out_root / "medium_one_vs_rest_feature_importance.csv", index=False)
    synth.to_csv(out_root / "synthetic_medium_alignment.csv", index=False)
    current_grid.to_csv(out_root / "current_guided_grid_medium_boundary.csv", index=False)
    write_json(out_root / "medium_cluster_geometry.json", geom)
    write_json(out_root / "model_separation_report.json", model_report)
    write_json(out_root / "data_audit.json", audit)

    # Mirror reader-facing tables into reports.
    deltas.to_csv(report_root / "feature_pairwise_robust_deltas.csv", index=False)
    imp.to_csv(report_root / "medium_one_vs_rest_feature_importance.csv", index=False)
    synth.to_csv(report_root / "synthetic_medium_alignment.csv", index=False)
    current_grid.to_csv(report_root / "current_guided_grid_medium_boundary.csv", index=False)

    plot_pca(pca_df, geom, vis / "but_pca_by_class.png")
    plot_box_features(df, vis / "but_key_feature_boxplots.png")
    plot_pairwise(deltas, vis / "pairwise_feature_boundaries.png")
    plot_importance(imp, vis / "medium_vs_rest_importance.png")
    plot_synthetic_alignment(synth, vis / "synthetic_medium_alignment.png")
    plot_current_grid(current_grid, vis / "current_grid_medium_bad_tradeoff.png")
    build_markdown(out_root, report_root, audit, geom, deltas, model_report, synth, current_grid)
    build_artifact_payload(report_root, geom, deltas, imp, synth, current_grid, model_report)
    print(json.dumps({"status": "complete", "out_root": str(out_root), "report_root": str(report_root)}, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether BUT medium is an independent morphology cluster.")
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--report_root", default=str(REPORT_ROOT))
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
