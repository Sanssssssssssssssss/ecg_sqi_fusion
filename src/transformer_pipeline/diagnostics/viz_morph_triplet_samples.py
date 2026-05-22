from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root


FS = 125
CLASS_ORDER = ("good", "medium", "bad")
NOISE_ORDER = ("em", "ma", "mix")


def main() -> None:
    args = parse_args()
    root = project_root()
    artifact_dir = resolve(root, args.artifact_dir)
    out_dir = resolve(root, args.out_dir) if args.out_dir else artifact_dir / "figs_label_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(artifact_dir / "datasets" / "synth_10s_125hz_labels.csv")
    clean = np.load(artifact_dir / "datasets" / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    noisy = np.load(artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    if len(labels) != clean.shape[0] or len(labels) != noisy.shape[0]:
        raise ValueError("labels/arrays row mismatch")

    selected_triplets = select_triplets(labels, k=args.triplets, split=args.split)
    selected_examples = select_class_noise_examples(labels, k=args.examples_per_cell, split=args.split)

    prefix = args.prefix
    triplet_png = out_dir / f"{prefix}_counterfactual_triplets.png"
    triplet_pdf = out_dir / f"{prefix}_counterfactual_triplets.pdf"
    grid_png = out_dir / f"{prefix}_class_noise_examples.png"
    grid_pdf = out_dir / f"{prefix}_class_noise_examples.pdf"
    csv_path = out_dir / f"{prefix}_selected_samples.csv"
    json_path = out_dir / f"{prefix}_visualization_summary.json"

    title_prefix = args.title or args.prefix
    plot_triplets(selected_triplets, clean, noisy, triplet_png, triplet_pdf, title_prefix=title_prefix)
    plot_class_noise_grid(selected_examples, clean, noisy, grid_png, grid_pdf, title_prefix=title_prefix)
    pd.concat([selected_triplets, selected_examples], ignore_index=True).drop_duplicates("idx").to_csv(csv_path, index=False)

    summary = {
        "artifact_dir": rel(artifact_dir, root),
        "out_dir": rel(out_dir, root),
        "split": args.split,
        "triplet_groups": [int(x) for x in selected_triplets["counterfactual_group"].drop_duplicates().tolist()],
        "selected_rows": int(len(pd.concat([selected_triplets, selected_examples], ignore_index=True).drop_duplicates("idx"))),
        "figures": [rel(triplet_png, root), rel(triplet_pdf, root), rel(grid_png, root), rel(grid_pdf, root)],
        "selected_csv": rel(csv_path, root),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {triplet_png}")
    print(f"Saved: {grid_png}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


def select_triplets(labels: pd.DataFrame, *, k: int, split: str) -> pd.DataFrame:
    df = labels[labels["split"].astype(str) == split].copy()
    groups: list[tuple[float, int, pd.DataFrame]] = []
    for gid, group in df.groupby("counterfactual_group"):
        classes = set(group["y_class"].astype(str))
        if not set(CLASS_ORDER).issubset(classes):
            continue
        rows = []
        for cls in CLASS_ORDER:
            rows.append(group[group["y_class"].astype(str) == cls].iloc[0])
        g = pd.DataFrame(rows)
        score = class_separation_score(g)
        groups.append((score, int(gid), g))
    groups.sort(key=lambda x: (-x[0], x[1]))
    chosen = [g for _, _, g in groups[:k]]
    if not chosen:
        raise ValueError(f"no complete triplets found for split={split}")
    return pd.concat(chosen, ignore_index=True)


def class_separation_score(group: pd.DataFrame) -> float:
    by_class = {str(r.y_class): r for r in group.itertuples(index=False)}
    good = by_class["good"]
    medium = by_class["medium"]
    bad = by_class["bad"]
    snr_gap = float(good.measured_snr_db) - float(bad.measured_snr_db)
    score_gap = float(bad.smooth_morph_score) - float(good.smooth_morph_score)
    medium_center = -abs(float(medium.smooth_morph_score) - 0.32)
    return snr_gap + 4.0 * score_gap + medium_center


def select_class_noise_examples(labels: pd.DataFrame, *, k: int, split: str) -> pd.DataFrame:
    df = labels[labels["split"].astype(str) == split].copy()
    rows = []
    for y_class in CLASS_ORDER:
        for noise_kind in NOISE_ORDER:
            sub = df[(df["y_class"].astype(str) == y_class) & (df["noise_kind"].astype(str) == noise_kind)].copy()
            if sub.empty:
                continue
            target = {"good": 0.08, "medium": 0.32, "bad": 0.65}[y_class]
            sub["_dist"] = (sub["smooth_morph_score"].astype(float) - target).abs()
            rows.append(sub.sort_values(["_dist", "idx"]).head(k).drop(columns=["_dist"]))
    if not rows:
        raise ValueError(f"no class/noise examples found for split={split}")
    return pd.concat(rows, ignore_index=True)


def plot_triplets(
    df: pd.DataFrame,
    clean: np.ndarray,
    noisy: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    *,
    title_prefix: str,
) -> None:
    groups = list(df.groupby("counterfactual_group", sort=False))
    t = np.arange(noisy.shape[1]) / FS
    fig, axes = plt.subplots(
        nrows=len(groups),
        ncols=3,
        figsize=(14, max(2.2, 2.0 * len(groups))),
        sharex=True,
        squeeze=False,
    )
    for r, (_, group) in enumerate(groups):
        by_class = {str(row.y_class): row for row in group.itertuples(index=False)}
        for c, y_class in enumerate(CLASS_ORDER):
            row = by_class[y_class]
            ax = axes[r, c]
            idx = int(row.idx)
            ax.plot(t, noisy[idx], linewidth=1.0, label="noisy")
            ax.plot(t, clean[idx], linewidth=0.8, linestyle="--", alpha=0.75, label="clean")
            title = (
                f"{y_class} | {row.noise_kind} | {row.placement}\n"
                f"SNR={float(row.measured_snr_db):.2f}, score={float(row.smooth_morph_score):.3f}, "
                f"qrs={float(row.qrs_nprd):.3f}, tst={float(row.tst_nprd):.3f}, corr={float(row.beat_corr):.3f}"
            )
            ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"group {int(row.counterfactual_group)}")
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=7)
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{title_prefix} representative counterfactual triplets", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_class_noise_grid(
    df: pd.DataFrame,
    clean: np.ndarray,
    noisy: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    *,
    title_prefix: str,
) -> None:
    t = np.arange(noisy.shape[1]) / FS
    ncols = max(1, int(df.groupby(["y_class", "noise_kind"]).size().max()))
    nrows = len(CLASS_ORDER) * len(NOISE_ORDER)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 1.8 * nrows), sharex=True, squeeze=False)
    row_i = 0
    for y_class in CLASS_ORDER:
        for noise_kind in NOISE_ORDER:
            sub = df[(df["y_class"].astype(str) == y_class) & (df["noise_kind"].astype(str) == noise_kind)].reset_index(drop=True)
            for c in range(ncols):
                ax = axes[row_i, c]
                if c >= len(sub):
                    ax.axis("off")
                    continue
                row = sub.iloc[c]
                idx = int(row["idx"])
                ax.plot(t, noisy[idx], linewidth=1.0, label="noisy")
                ax.plot(t, clean[idx], linewidth=0.8, linestyle="--", alpha=0.75, label="clean")
                ax.set_title(
                    f"{y_class}/{noise_kind} idx={idx}\n"
                    f"SNR={float(row['measured_snr_db']):.2f}, score={float(row['smooth_morph_score']):.3f}",
                    fontsize=8,
                )
                if c == 0:
                    ax.set_ylabel(f"{y_class}\n{noise_kind}")
                    ax.legend(loc="upper right", fontsize=7)
            row_i += 1
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(f"{title_prefix} representative samples by class and noise kind", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize morphology triplet labels.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_e310_smooth_morph_mild_snr")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--prefix", default="e310")
    parser.add_argument("--title", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--triplets", type=int, default=8)
    parser.add_argument("--examples_per_cell", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
