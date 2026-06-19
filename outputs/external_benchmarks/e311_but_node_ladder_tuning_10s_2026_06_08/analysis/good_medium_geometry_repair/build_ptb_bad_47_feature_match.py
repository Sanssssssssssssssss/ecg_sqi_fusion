"""Build a PTB bad bank matched to BUT bad in the existing 47-feature space.

This is an external-only distribution matching artifact.  It uses the same
47-column SQI/geometry schema used by the N17043 feature sidecar.  The output
bank contains PTB synthetic waveforms only; BUT rows are used only as the target
distribution for report-only/domain-alignment diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
SYNTH_DATA_DIR = (
    OUT_ROOT
    / "synthetic_variants"
    / "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
    / "datasets"
)
BUT_ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
OUT_DIR = ANALYSIS_DIR / "ptb_bad_47_feature_match"
REPORT_OUT_DIR = REPORT_DIR / "ptb_bad_47_feature_match"

GEOMETRY_COLUMNS = [
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pca_margin",
    "boundary_confidence",
    "region_confidence",
    "knn_label_purity",
]


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.sort(a.astype(np.float64))
    y = np.sort(b.astype(np.float64))
    grid = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / max(1, len(x))
    cdf_y = np.searchsorted(y, grid, side="right") / max(1, len(y))
    return float(np.max(np.abs(cdf_x - cdf_y)))


def robust_scale(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(target, axis=0).astype(np.float32)
    q75 = np.nanpercentile(target, 75, axis=0).astype(np.float32)
    q25 = np.nanpercentile(target, 25, axis=0).astype(np.float32)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    return center, scale


def nearest_resample(
    cand_z: np.ndarray,
    target_z: np.ndarray,
    selected_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_pick = rng.choice(np.arange(target_z.shape[0]), size=int(selected_count), replace=True)
    selected = np.empty(int(selected_count), dtype=np.int64)
    distances = np.empty(int(selected_count), dtype=np.float32)
    for out_i, target_i in enumerate(target_pick):
        diff = cand_z - target_z[target_i][None, :]
        dist = np.mean(diff * diff, axis=1)
        best = int(np.argmin(dist))
        selected[out_i] = best
        distances[out_i] = float(np.sqrt(dist[best]))
    return selected, target_pick.astype(np.int64), distances


def feature_ks_table(cols: list[str], target: np.ndarray, base: np.ndarray, matched: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for j, col in enumerate(cols):
        for group, arr in [("base_ptb_bad", base), ("matched_ptb_bad", matched)]:
            rows.append(
                {
                    "feature": col,
                    "group": group,
                    "ks_vs_but_bad": ks_stat(target[:, j], arr[:, j]),
                    "target_median": float(np.nanmedian(target[:, j])),
                    "group_median": float(np.nanmedian(arr[:, j])),
                    "target_q10": float(np.nanpercentile(target[:, j], 10)),
                    "target_q90": float(np.nanpercentile(target[:, j], 90)),
                    "group_q10": float(np.nanpercentile(arr[:, j], 10)),
                    "group_q90": float(np.nanpercentile(arr[:, j], 90)),
                }
            )
    return pd.DataFrame(rows)


def pca2(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:2].T).astype(np.float32)


def plot_pca(target: np.ndarray, base: np.ndarray, matched: np.ndarray, out_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    groups = [
        ("BUT bad target", target, "#444444", 900),
        ("PTB bad base", base, "#d95f02", 700),
        ("PTB bad 47-match", matched, "#1b9e77", 700),
    ]
    all_x = np.concatenate([g[1] for g in groups], axis=0)
    z = pca2(all_x)
    offset = 0
    fig, ax = plt.subplots(figsize=(8, 6), dpi=170)
    for label, arr, color, max_n in groups:
        pts = z[offset : offset + len(arr)]
        offset += len(arr)
        if len(pts) > max_n:
            pts = pts[rng.choice(len(pts), size=max_n, replace=False)]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.35, label=label, color=color, edgecolors="none")
    ax.set_title("Bad distribution in existing 47-feature space")
    ax.set_xlabel("PCA-1 of 47-feature audit space")
    ax.set_ylabel("PCA-2 of 47-feature audit space")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_waveforms(signals: np.ndarray, manifest: pd.DataFrame, out_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    if len(signals) == 0:
        return
    take = rng.choice(len(signals), size=min(16, len(signals)), replace=False)
    t = np.linspace(0, 10, signals.shape[1], dtype=np.float32)
    fig, axes = plt.subplots(4, 4, figsize=(12, 7), dpi=160, sharex=True)
    for ax, idx in zip(axes.ravel(), take):
        ax.plot(t, signals[idx], lw=0.75, color="#1b9e77")
        region = str(manifest.iloc[idx].get("target_original_region", "target"))
        ax.set_title(region, fontsize=8)
        ax.set_yticks([])
        ax.grid(alpha=0.12)
    fig.suptitle("Selected PTB bad rows after 47-feature distribution matching")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="but_bad_all", choices=["but_bad_all", "but_bad_test"])
    parser.add_argument("--candidate-split", type=str, default="train", choices=["train", "all"])
    parser.add_argument("--match-space", type=str, default="all47", choices=["all47", "primitive39", "geometry8"])
    parser.add_argument("--selected-count", type=int, default=1816)
    parser.add_argument("--seed", type=int, default=2026061911)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    schema = json.loads((SYNTH_DATA_DIR / "tabular_feature_schema.json").read_text(encoding="utf-8"))
    all_cols = list(schema["feature_columns"])
    if args.match_space == "primitive39":
        cols = [c for c in all_cols if c not in GEOMETRY_COLUMNS]
    elif args.match_space == "geometry8":
        cols = list(GEOMETRY_COLUMNS)
    else:
        cols = all_cols
    col_idx = [all_cols.index(c) for c in cols]

    labels = pd.read_csv(SYNTH_DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    signals_all = np.load(SYNTH_DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    feat_npz = np.load(SYNTH_DATA_DIR / "tabular_features.npz")
    features_all = feat_npz["features"].astype(np.float32)
    rows = feat_npz["rows"].astype(np.int64)
    if not np.array_equal(rows, labels["idx"].to_numpy(dtype=np.int64)):
        order = {int(idx): i for i, idx in enumerate(rows)}
        feature_order = np.asarray([order[int(idx)] for idx in labels["idx"].to_numpy(dtype=np.int64)], dtype=np.int64)
        features_all = features_all[feature_order]

    cand_mask = labels["y_class"].astype(str).eq("bad").to_numpy()
    if args.candidate_split == "train":
        cand_mask &= labels["split"].astype(str).eq("train").to_numpy()
    cand_idx = np.flatnonzero(cand_mask)
    cand_features = features_all[cand_idx][:, col_idx]
    cand_signals = signals_all[labels.loc[cand_idx, "idx"].to_numpy(dtype=np.int64)]

    atlas_usecols = ["idx", "split", "class_name", "record_id", "original_region"] + all_cols
    atlas = pd.read_csv(BUT_ATLAS, usecols=atlas_usecols)
    target_mask = atlas["class_name"].astype(str).eq("bad")
    if args.target == "but_bad_test":
        target_mask &= atlas["split"].astype(str).eq("test")
    target = atlas.loc[target_mask].reset_index(drop=True)
    target_features = target[cols].to_numpy(dtype=np.float32)

    center, scale = robust_scale(target_features)
    cand_z = np.nan_to_num((cand_features - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    target_z = np.nan_to_num((target_features - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    selected, target_pick, distances = nearest_resample(cand_z, target_z, int(args.selected_count), rng)
    selected_source_rows = cand_idx[selected]
    matched_features = cand_features[selected]
    matched_signals = cand_signals[selected]

    manifest = pd.DataFrame(
        {
            "source_idx": labels.loc[selected_source_rows, "idx"].to_numpy(dtype=np.int64),
            "source_split": labels.loc[selected_source_rows, "split"].astype(str).to_numpy(),
            "source_block": labels.loc[selected_source_rows, "block"].astype(str).to_numpy() if "block" in labels.columns else "synthetic",
            "target_but_idx": target.loc[target_pick, "idx"].to_numpy(dtype=np.int64),
            "target_record_id": target.loc[target_pick, "record_id"].to_numpy(),
            "target_original_region": target.loc[target_pick, "original_region"].astype(str).to_numpy(),
            "match_distance": distances,
            "match_space": args.match_space,
        }
    )
    dist = feature_ks_table(cols, target_features, cand_features, matched_features)
    base_mean = float(dist.loc[dist["group"].eq("base_ptb_bad"), "ks_vs_but_bad"].mean())
    matched_mean = float(dist.loc[dist["group"].eq("matched_ptb_bad"), "ks_vs_but_bad"].mean())
    base_top10 = float(dist.loc[dist["group"].eq("base_ptb_bad")].nlargest(min(10, len(cols)), "ks_vs_but_bad")["ks_vs_but_bad"].mean())
    matched_top10 = float(dist.loc[dist["group"].eq("matched_ptb_bad")].nlargest(min(10, len(cols)), "ks_vs_but_bad")["ks_vs_but_bad"].mean())
    summary = {
        "target": args.target,
        "target_count": int(len(target)),
        "candidate_split": args.candidate_split,
        "candidate_bad_count": int(len(cand_idx)),
        "match_space": args.match_space,
        "matched_count": int(len(selected)),
        "unique_source_count": int(manifest["source_idx"].nunique()),
        "base_mean_ks": base_mean,
        "matched_mean_ks": matched_mean,
        "base_top10_mean_ks": base_top10,
        "matched_top10_mean_ks": matched_top10,
        "distance_q50": float(np.percentile(distances, 50)),
        "distance_q90": float(np.percentile(distances, 90)),
        "target_region_counts": manifest["target_original_region"].value_counts().to_dict(),
        "top_source_blocks": manifest["source_block"].value_counts().head(12).to_dict(),
    }

    stem = f"ptb_bad_47match_{args.target}_{args.candidate_split}_{args.match_space}"
    np.savez_compressed(OUT_DIR / f"{stem}_signals.npz", X=matched_signals)
    np.savez_compressed(OUT_DIR / f"{stem}_features.npz", features=matched_features, source_idx=manifest["source_idx"].to_numpy(dtype=np.int64))
    manifest.to_csv(OUT_DIR / f"{stem}_manifest.csv", index=False)
    dist.to_csv(OUT_DIR / f"{stem}_feature_ks.csv", index=False)
    (OUT_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pca_path = REPORT_OUT_DIR / f"{stem}_pca.png"
    wave_path = REPORT_OUT_DIR / f"{stem}_waveforms.png"
    plot_pca(target_features, cand_features, matched_features, pca_path, int(args.seed))
    plot_waveforms(matched_signals, manifest, wave_path, int(args.seed) + 17)
    worst = dist.loc[dist["group"].eq("matched_ptb_bad")].sort_values("ks_vs_but_bad", ascending=False).head(12)
    report = [
        "# PTB Bad 47-Feature Distribution Match",
        "",
        "This artifact matches PTB synthetic bad rows to the BUT bad distribution using the existing 47-column SQI/geometry schema. BUT rows are target-distribution diagnostics only; output signals remain PTB synthetic waveforms.",
        "",
        "## Summary",
        "",
        f"- Target: `{args.target}` rows `{len(target)}`",
        f"- Candidate: PTB synthetic bad `{args.candidate_split}` rows `{len(cand_idx)}`",
        f"- Match space: `{args.match_space}` columns `{len(cols)}`",
        f"- Matched rows: `{len(selected)}` using `{summary['unique_source_count']}` unique PTB bad sources",
        f"- Mean KS base -> BUT bad: `{base_mean:.4f}`",
        f"- Mean KS matched -> BUT bad: `{matched_mean:.4f}`",
        f"- Top10 KS base -> BUT bad: `{base_top10:.4f}`",
        f"- Top10 KS matched -> BUT bad: `{matched_top10:.4f}`",
        f"- Match distance q50/q90: `{summary['distance_q50']:.4f}` / `{summary['distance_q90']:.4f}`",
        "",
        "## Remaining Largest Matched KS",
        "",
        "| feature | matched KS | target median | matched median |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in worst.iterrows():
        report.append(f"| {row['feature']} | {float(row['ks_vs_but_bad']):.4f} | {float(row['target_median']):.4f} | {float(row['group_median']):.4f} |")
    report.extend(
        [
            "",
            "## Figures",
            "",
            f"![PCA]({pca_path})",
            "",
            f"![Waveforms]({wave_path})",
            "",
            "## Files",
            "",
            f"- Manifest: `{OUT_DIR / (stem + '_manifest.csv')}`",
            f"- Feature KS: `{OUT_DIR / (stem + '_feature_ks.csv')}`",
            f"- Matched PTB signals: `{OUT_DIR / (stem + '_signals.npz')}`",
        ]
    )
    report_path = OUT_DIR / f"{stem}_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    (REPORT_OUT_DIR / report_path.name).write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
