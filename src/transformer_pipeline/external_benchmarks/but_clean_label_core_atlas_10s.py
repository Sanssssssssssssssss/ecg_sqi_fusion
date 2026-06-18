"""Build a high-confidence CleanBUT-Core atlas for BUT 10s P1.

This analysis is deliberately prediction-independent.  It uses the same
feature bank as ``but_separability_atlas_10s.py`` and filters BUT windows by
class-label neighborhood purity plus class-centroid margin in train-fitted PCA
space.  The clean subsets are diagnostic/generator targets only; they do not
replace the formal BUT test split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_separability_atlas_10s as atlas


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_clean_label_core_10s_2026_06_06"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_SOURCE_ATLAS = ROOT / "outputs" / "external_benchmarks" / "e311_but_separability_atlas_10s_2026_06_06"
DEFAULT_SOURCE_ATLAS_REPORT = ROOT / "reports" / "external_benchmarks" / "e311_but_separability_atlas_10s_2026_06_06"
DEFAULT_BUT_PROTOCOL = atlas.DEFAULT_BUT_PROTOCOL
CLASS_NAMES = atlas.CLASS_NAMES


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy() if max_rows else df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4g}")
    lines = [
        "| " + " | ".join(map(str, view.columns)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in view.columns) + " |")
    return "\n".join(lines)


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(atlas.NON_FEATURE_COLUMNS) | {"label_raw", "split_code"}
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in blocked]


def load_feature_matrix(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    out_root = Path(args.out_root)
    src = Path(args.source_atlas_out) / "but_feature_matrix.csv"
    if src.exists():
        df = pd.read_csv(src)
    else:
        X, meta = atlas.load_but(Path(args.but_protocol_dir))
        tmp = argparse.Namespace(
            out_root=out_root,
            morph_cache=args.morph_cache,
            sqi_cache=args.sqi_cache,
        )
        df, _ = atlas.load_feature_matrix(tmp, X, meta)
    cols = feature_columns(df)
    if not cols:
        raise ValueError("No numeric atlas feature columns found.")
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "but_feature_matrix.csv", index=False)
    return df, cols


def _query_train_purity(
    P: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    train_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    train_pos = np.where(train_mask)[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(train_pos))).fit(P[train_pos])
    inds = nn.kneighbors(P, return_distance=False)
    purity = np.zeros(len(P), dtype=float)
    for i, refs_local in enumerate(inds):
        refs = train_pos[refs_local]
        if split[i] == "train":
            refs = refs[refs != i]
        refs = refs[:k]
        if len(refs) == 0:
            purity[i] = math.nan
        else:
            purity[i] = float(np.mean(labels[refs] == labels[i]))
    return purity


def build_clean_manifest(df: pd.DataFrame, features: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    split = work["split"].astype(str).to_numpy()
    labels = work["class_name"].astype(str).to_numpy()
    train_mask = split == "train"
    X = work[features].replace([np.inf, -np.inf], np.nan)
    med = X.loc[train_mask].median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X.loc[train_mask])
    pca = PCA(n_components=min(8, len(features)), random_state=int(args.seed))
    pca.fit(Z_train)
    P = pca.transform(scaler.transform(X))
    for i in range(min(8, P.shape[1])):
        work[f"pc{i + 1}"] = P[:, i]
    P8 = P[:, : min(8, P.shape[1])]

    centroids: dict[str, np.ndarray] = {}
    for cls in CLASS_NAMES:
        m = train_mask & (labels == cls)
        if not np.any(m):
            raise ValueError(f"No train rows for class {cls}")
        centroids[cls] = P8[m].mean(axis=0)
    D = np.column_stack([np.linalg.norm(P8 - centroids[cls], axis=1) for cls in CLASS_NAMES])
    own_idx = np.array([CLASS_NAMES.index(str(v)) for v in labels])
    own_d = D[np.arange(len(work)), own_idx]
    masked = D.copy()
    masked[np.arange(len(work)), own_idx] = np.inf
    nearest_other_idx = np.argmin(masked, axis=1)
    nearest_other_d = masked[np.arange(len(work)), nearest_other_idx]
    margin = nearest_other_d - own_d
    purity = _query_train_purity(P8, labels, split, train_mask, int(args.knn_k))

    work["pca_own_distance"] = own_d
    work["pca_nearest_other_distance"] = nearest_other_d
    work["pca_margin"] = margin
    work["knn_label_purity"] = purity
    work["nearest_other_class"] = [CLASS_NAMES[int(i)] for i in nearest_other_idx]

    thresholds: dict[str, Any] = {
        "generated_at": now_iso(),
        "feature_source": str(args.source_atlas_out),
        "fit_split": "train",
        "n_features": int(len(features)),
        "features": features,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "rules": {
            "clean_core_strict": {
                "knn_label_purity_min": float(args.strict_purity),
                "pca_margin_quantile_within_class_train": float(args.strict_margin_quantile),
            },
            "clean_core_train_target": {
                "knn_label_purity_min": float(args.target_purity),
                "pca_margin_quantile_within_class_train": float(args.target_margin_quantile),
            },
        },
        "class_thresholds": {},
    }

    is_strict = np.zeros(len(work), dtype=bool)
    is_target = np.zeros(len(work), dtype=bool)
    margin_pct = np.zeros(len(work), dtype=float)
    own_distance_pct = np.zeros(len(work), dtype=float)
    for cls in CLASS_NAMES:
        class_mask = labels == cls
        train_class = train_mask & class_mask
        strict_thr = float(np.quantile(margin[train_class], float(args.strict_margin_quantile)))
        target_thr = float(np.quantile(margin[train_class], float(args.target_margin_quantile)))
        own_train = own_d[train_class]
        thresholds["class_thresholds"][cls] = {
            "train_n": int(train_class.sum()),
            "strict_margin_min": strict_thr,
            "target_margin_min": target_thr,
            "own_distance_median_train": float(np.median(own_train)),
            "own_distance_q90_train": float(np.quantile(own_train, 0.90)),
        }
        is_strict |= class_mask & (purity >= float(args.strict_purity)) & (margin >= strict_thr)
        is_target |= class_mask & (purity >= float(args.target_purity)) & (margin >= target_thr)
        margin_pct[class_mask] = pd.Series(margin[class_mask]).rank(pct=True).to_numpy()
        own_distance_pct[class_mask] = pd.Series(-own_d[class_mask]).rank(pct=True).to_numpy()

    tier = np.full(len(work), "ambiguous_boundary", dtype=object)
    tier[is_target] = "clean_core_train_target"
    tier[is_strict] = "clean_core_strict"
    work["is_clean_strict"] = is_strict
    work["is_clean_train_target"] = is_target
    work["clean_tier"] = tier
    work["class_margin_percentile"] = margin_pct
    work["class_centrality_percentile"] = own_distance_pct

    ambiguous = []
    for cls, other, od_pct, pur, mar in zip(labels, work["nearest_other_class"], own_distance_pct, purity, margin):
        pair = "_".join(sorted([str(cls), str(other)], key=lambda x: CLASS_NAMES.index(x)))
        if str(cls) == "bad" and od_pct < 0.25:
            ambiguous.append("bad_outlier")
        elif str(cls) in {"good", "medium"} and od_pct < 0.20:
            ambiguous.append(f"isolated_{cls}")
        elif pur < 0.55:
            ambiguous.append(f"{pair}_low_purity")
        elif mar < 0:
            ambiguous.append(f"{pair}_negative_margin")
        else:
            ambiguous.append(f"{pair}_boundary")
    work["ambiguous_type"] = np.where(is_target, "clean_or_target", ambiguous)

    return work, thresholds


def target_distributions(manifest: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    target = manifest[manifest["is_clean_train_target"]].copy()
    for cls in CLASS_NAMES:
        sub = target[target["class_name"] == cls]
        stats_by_feat: dict[str, Any] = {}
        for feat in features:
            vals = pd.to_numeric(sub[feat], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            stats_by_feat[feat] = {
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "q10": float(np.quantile(vals, 0.10)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q90": float(np.quantile(vals, 0.90)),
                "iqr": float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)),
            }
        rows[cls] = {"n": int(len(sub)), "features": stats_by_feat}
    return rows


def classwise_distance(a: pd.DataFrame, b: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_class: dict[str, float] = {}
    for cls in CLASS_NAMES:
        av = a[a["class_name"] == cls]
        bv = b[b["class_name"] == cls]
        vals = []
        for feat in features:
            if feat not in av.columns or feat not in bv.columns:
                continue
            x = pd.to_numeric(av[feat], errors="coerce").dropna().to_numpy(dtype=float)
            y = pd.to_numeric(bv[feat], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) < 4 or len(y) < 4:
                continue
            iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25) + 1e-6)
            ks = float(stats.ks_2samp(x, y).statistic)
            med_delta = float((np.median(y) - np.median(x)) / iqr)
            vals.append(ks)
            rows.append(
                {
                    "class_name": cls,
                    "feature": feat,
                    "ks": ks,
                    "clean_median": float(np.median(x)),
                    "synthetic_median": float(np.median(y)),
                    "standardized_median_delta_synth_minus_clean": med_delta,
                    "n_clean": int(len(x)),
                    "n_synthetic": int(len(y)),
                }
            )
        by_class[cls] = float(np.mean(vals)) if vals else math.nan
    return {"overall": float(np.nanmean(list(by_class.values()))), "by_class": by_class, "rows": rows}


def synthetic_clean_target_distance(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    roots = [Path(p) for p in str(args.synthetic_roots).split(";") if str(p).strip()]
    clean = manifest[manifest["is_clean_train_target"]].copy()
    cols = [c for c in atlas.NON_FEATURE_COLUMNS if c in clean.columns]
    _ = cols  # keep explicit: no prediction columns are used.
    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("synthetic_morph_features.csv"):
            try:
                synth = pd.read_csv(path)
            except Exception:
                continue
            if "class_name" not in synth.columns:
                if "y_class" in synth.columns:
                    synth["class_name"] = synth["y_class"].astype(str)
                else:
                    continue
            common = [f for f in clean.select_dtypes(include=[np.number]).columns if f in synth.columns and f not in atlas.NON_FEATURE_COLUMNS]
            if len(common) < 6:
                continue
            dist = classwise_distance(clean, synth, common)
            variant_dir = path.parent
            variant_id = variant_dir.name
            row = {
                "variant_id": variant_id,
                "source_root": str(root),
                "variant_dir": str(variant_dir),
                "n_common_features": int(len(common)),
                "clean_target_distance": dist["overall"],
                "clean_good_distance": dist["by_class"].get("good", math.nan),
                "clean_medium_distance": dist["by_class"].get("medium", math.nan),
                "clean_bad_distance": dist["by_class"].get("bad", math.nan),
            }
            summaries.append(row)
            for d in dist["rows"]:
                detail_rows.append({"variant_id": variant_id, **d})
    out = pd.DataFrame(summaries)
    if not out.empty:
        out = out.sort_values(["clean_target_distance", "clean_medium_distance", "clean_bad_distance"], na_position="last")
    detail = pd.DataFrame(detail_rows)
    out_root = Path(args.out_root)
    detail.to_csv(out_root / "ptb_synthetic_clean_target_distance_detail.csv", index=False)
    return out


def plot_clean_pca(manifest: pd.DataFrame, report_root: Path) -> None:
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    strict = manifest[manifest["is_clean_strict"]].copy()
    if strict.empty:
        strict = manifest[manifest["is_clean_train_target"]].copy()
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=strict, x="pc1", y="pc2", hue="class_name", hue_order=list(CLASS_NAMES), s=11, alpha=0.72, linewidth=0)
    plt.title("CleanBUT-Core PCA by class")
    plt.tight_layout()
    plt.savefig(fig_dir / "clean_pca_class.png", dpi=180)
    plt.close()

    sample = manifest.sample(n=min(len(manifest), 12000), random_state=17).copy()
    sample["display_group"] = np.where(sample["is_clean_strict"], "clean_strict", np.where(sample["is_clean_train_target"], "clean_target", sample["ambiguous_type"]))
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=sample, x="pc1", y="pc2", hue="display_group", s=8, alpha=0.45, linewidth=0)
    plt.title("Clean core vs ambiguous BUT boundary regions")
    plt.tight_layout()
    plt.savefig(fig_dir / "clean_vs_ambiguous_pca.png", dpi=180)
    plt.close()


def plot_feature_ridges(manifest: pd.DataFrame, top_features: list[str], report_root: Path) -> None:
    fig_dir = report_root / "figures"
    clean = manifest[manifest["is_clean_train_target"]].copy()
    feats = [f for f in top_features if f in clean.columns][:9]
    if not feats:
        return
    ncols = 3
    nrows = int(math.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.3 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, feat in zip(axes, feats):
        sns.violinplot(data=clean, x="class_name", y=feat, order=list(CLASS_NAMES), ax=ax, inner="quartile", cut=0)
        ax.set_title(feat)
        ax.set_xlabel("")
    for ax in axes[len(feats) :]:
        ax.axis("off")
    fig.suptitle("CleanBUT target feature distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "clean_feature_ridges.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prototypes(manifest: pd.DataFrame, args: argparse.Namespace) -> None:
    X, _ = atlas.load_but(Path(args.but_protocol_dir))
    fig_dir = Path(args.report_root) / "figures" / "clean_prototype_galleries"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        sub = manifest[(manifest["class_name"] == cls) & (manifest["is_clean_strict"])].sort_values(
            ["class_centrality_percentile", "pca_margin"], ascending=[False, False]
        )
        if len(sub) == 0:
            sub = manifest[(manifest["class_name"] == cls) & (manifest["is_clean_train_target"])].sort_values(
                ["class_centrality_percentile", "pca_margin"], ascending=[False, False]
            )
        ids = sub["idx"].head(18).to_numpy(dtype=int)
        if len(ids):
            atlas.plot_wave_gallery(X, ids, f"CleanBUT {cls} prototypes", fig_dir / f"clean_{cls}_prototype_gallery.png")
    for kind in ["good_medium_boundary", "medium_bad_boundary", "bad_outlier", "isolated_good", "isolated_medium"]:
        sub = manifest[manifest["ambiguous_type"].astype(str).str.contains(kind, regex=False)]
        ids = sub.sort_values("knn_label_purity").head(18)["idx"].to_numpy(dtype=int)
        if len(ids):
            atlas.plot_wave_gallery(X, ids, f"BUT ambiguous {kind}", fig_dir / f"ambiguous_{kind}.png")


def write_report(
    manifest: pd.DataFrame,
    thresholds: dict[str, Any],
    target_stats: dict[str, Any],
    synthetic_dist: pd.DataFrame,
    top_features: list[str],
    args: argparse.Namespace,
) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    counts = (
        manifest.groupby(["clean_tier", "class_name"]).size().reset_index(name="n").pivot(index="clean_tier", columns="class_name", values="n").fillna(0).astype(int)
    )
    split_counts = manifest.groupby(["split", "clean_tier", "class_name"]).size().reset_index(name="n")
    ambiguous = manifest[~manifest["is_clean_train_target"]].groupby(["class_name", "ambiguous_type"]).size().reset_index(name="n").sort_values(["class_name", "n"], ascending=[True, False])
    lines = [
        "# BUT 10s Clean-Label Core Atlas",
        "",
        "## Executive read",
        "",
        "- CleanBUT-Core is a diagnostic and generator-target subset; it does not replace the original BUT 10s P1 test.",
        "- Selection uses only atlas signal features plus labels: train-fitted PCA margin and kNN label purity.",
        "- Original label-overlap regions are retained as ambiguous boundary material for analysis, not used as clean target.",
        "",
        "## Clean subset counts",
        "",
        md_table(counts.reset_index()),
        "",
        "## Split counts",
        "",
        md_table(split_counts, 18),
        "",
        "## Top clean target features",
        "",
        ", ".join(top_features[:15]),
        "",
        "## Ambiguous boundary summary",
        "",
        md_table(ambiguous, 30),
        "",
        "## Synthetic distance leaderboard",
        "",
        md_table(synthetic_dist.head(20), 20) if not synthetic_dist.empty else "_No synthetic morphology files found._",
        "",
        "## Rule thresholds",
        "",
        "```json",
        json.dumps({k: thresholds[k] for k in ["rules", "class_thresholds"]}, ensure_ascii=False, indent=2)[:12000],
        "```",
        "",
        "## Files",
        "",
        f"- Output root: `{out_root}`",
        f"- Report root: `{report_root}`",
        "- Figures: `figures/clean_pca_class.png`, `figures/clean_vs_ambiguous_pca.png`, `figures/clean_feature_ridges.png`",
    ]
    (report_root / "clean_label_core_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "clean_label_core_state.json", {"status": "running", "started_at": now_iso()})
    df, features = load_feature_matrix(args)
    manifest, thresholds = build_clean_manifest(df, features, args)
    manifest.to_csv(out_root / "clean_subset_manifest.csv", index=False)
    manifest.to_csv(report_root / "clean_subset_manifest.csv", index=False)
    write_json(out_root / "clean_rule_thresholds.json", thresholds)
    write_json(report_root / "clean_rule_thresholds.json", thresholds)
    target_stats = target_distributions(manifest, features)
    write_json(out_root / "clean_but_target_distributions.json", target_stats)
    write_json(report_root / "clean_but_target_distributions.json", target_stats)
    ambiguous = manifest[~manifest["is_clean_train_target"]].groupby(["class_name", "ambiguous_type"]).size().reset_index(name="n")
    ambiguous.to_csv(out_root / "label_noise_suspect_summary.csv", index=False)
    ambiguous.to_csv(report_root / "label_noise_suspect_summary.csv", index=False)

    feature_scores_path = Path(args.source_atlas_out) / "feature_separability_scores.csv"
    if feature_scores_path.exists():
        top_features = pd.read_csv(feature_scores_path).sort_values("separability_score", ascending=False)["feature"].tolist()
    else:
        top_features = features[:15]
    synthetic_dist = synthetic_clean_target_distance(manifest, args)
    synthetic_dist.to_csv(out_root / "ptb_synthetic_clean_target_distance.csv", index=False)
    synthetic_dist.to_csv(report_root / "ptb_synthetic_clean_target_distance.csv", index=False)
    plot_clean_pca(manifest, report_root)
    plot_feature_ridges(manifest, top_features, report_root)
    plot_prototypes(manifest, args)
    write_report(manifest, thresholds, target_stats, synthetic_dist, top_features, args)
    state = {
        "status": "complete",
        "updated_at": now_iso(),
        "n_rows": int(len(manifest)),
        "strict_counts": manifest[manifest["is_clean_strict"]]["class_name"].value_counts().to_dict(),
        "target_counts": manifest[manifest["is_clean_train_target"]]["class_name"].value_counts().to_dict(),
        "report": str(report_root / "clean_label_core_report.md"),
    }
    write_json(out_root / "clean_label_core_state.json", state)
    write_json(report_root / "clean_label_core_state.json", state)
    return state


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    p.add_argument("--source_atlas_out", default=str(DEFAULT_SOURCE_ATLAS))
    p.add_argument("--source_atlas_report", default=str(DEFAULT_SOURCE_ATLAS_REPORT))
    p.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    p.add_argument("--morph_cache", default=str(atlas.DEFAULT_MORPH_CACHE))
    p.add_argument("--sqi_cache", default=str(atlas.DEFAULT_SQI_CACHE))
    p.add_argument("--strict_purity", type=float, default=0.90)
    p.add_argument("--target_purity", type=float, default=0.85)
    p.add_argument("--strict_margin_quantile", type=float, default=0.85)
    p.add_argument("--target_margin_quantile", type=float, default=0.75)
    p.add_argument("--knn_k", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument(
        "--synthetic_roots",
        default=";".join(
            [
                str(ROOT / "outputs" / "external_benchmarks" / "e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05"),
                str(ROOT / "outputs" / "external_benchmarks" / "e311_but_big_uformer_long_search_10s_2026_06_06"),
                str(ROOT / "outputs" / "external_benchmarks" / "e311_sensors2025_noise_synthesis_but_10s_2026_06_05"),
                str(ROOT / "outputs" / "external_benchmarks" / "e311_but_diagnostic_usability_grid_10s_2026_06_04"),
            ]
        ),
    )
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
