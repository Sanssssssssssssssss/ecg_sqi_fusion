from __future__ import annotations

from typing import Any

import pandas as pd

from .audit import load_atlas
from .common import ARTIFACTS, POLICY, protocol_dir, report_dir, split_dir


def bar(ax: Any, labels: list[str], values: list[float], title: str, ylabel: str = "rows") -> None:
    ax.bar(labels, values, color=["#4c78a8", "#59a14f", "#e15759", "#f28e2b"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)


def save_bar(labels: list[str], values: list[float], title: str, ylabel: str, path: Any) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bar(ax, labels, values, title, ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ARTIFACTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    atlas = load_atlas(protocol_dir() / "original_region_atlas.csv")
    split = load_atlas(split_dir() / "original_region_atlas.csv")
    original = atlas[atlas["v116_candidate_type"].astype(str).eq("original_but")]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    counts = original["class_name"].value_counts().reindex(["good", "medium", "bad"]).astype(int)
    bar(ax, counts.index.tolist(), counts.tolist(), "Original BUT gap5 rows")
    fig.tight_layout()
    fig.savefig(out / "original_but_class_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    split_counts = split[split["split"].isin(["train", "val", "test"])].groupby(["split", "class_name"]).size().unstack(fill_value=0).reindex(["train", "val", "test"])
    split_counts[["good", "medium", "bad"]].plot(kind="bar", ax=ax, color=["#4c78a8", "#59a14f", "#e15759"])
    ax.set_title("Data v1 split class counts")
    ax.set_ylabel("rows")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out / "split_class_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    comp = split[split["split"].eq("train")].groupby(["class_name", "v116_candidate_type"]).size().unstack(fill_value=0).reindex(["good", "medium", "bad"])
    comp.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Train composition after exact balance")
    ax.set_ylabel("rows")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out / "train_candidate_type_composition.png", dpi=180)
    plt.close(fig)

    gap = atlas.loc[atlas["class_name"].isin(["medium", "bad"]) & ~atlas["v116_candidate_type"].astype(str).eq("original_but")]
    if not gap.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        gap_comp = pd.crosstab(gap["class_name"], gap["v116_candidate_type"], normalize="index").reindex(["medium", "bad"]) * 100
        gap_comp[[c for c in ["but_native_morph", "ptb_morph", "clean_style"] if c in gap_comp.columns]].plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Generated gap component share")
        ax.set_ylabel("percent of generated gap")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(out / "generated_gap_component_share.png", dpi=180)
        plt.close(fig)

    metric_path = report_dir() / f"{POLICY}_global_distribution_metrics.csv"
    if metric_path.exists():
        m = pd.read_csv(metric_path)
        m = m[m["scope"].isin(["class_good", "class_medium", "class_bad"])].copy()
        m["scope"] = m["scope"].str.replace("class_", "", regex=False)
        for col, ylabel, filename in [
            ("rbf_mmd", "RBF MMD", "distribution_rbf_mmd.png"),
            ("sliced_wasserstein", "sliced Wasserstein", "distribution_sliced_wasserstein.png"),
            ("sym_domain_auc", "domain AUC", "distribution_domain_auc.png"),
            ("pca_density_overlap", "PCA overlap", "distribution_pca_overlap.png"),
        ]:
            if col in m.columns:
                save_bar(m["scope"].tolist(), m[col].astype(float).tolist(), f"{POLICY} {ylabel}", ylabel, out / filename)
    print(out)
