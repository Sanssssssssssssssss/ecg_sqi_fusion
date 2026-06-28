from __future__ import annotations

from typing import Any

import pandas as pd

from .audit import load_atlas
from .common import ROOT, protocol_dir, split_dir


def bar(ax: Any, labels: list[str], values: list[float], title: str, ylabel: str = "rows") -> None:
    ax.bar(labels, values, color=["#4c78a8", "#59a14f", "#e15759", "#f28e2b"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ROOT / "docs" / "data_v1_figures"
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

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    dual = pd.DataFrame({"scope": ["medium", "bad", "pooled"], "sym_auc": [0.510, 0.529, 0.549]})
    bar(ax, dual["scope"].tolist(), dual["sym_auc"].tolist(), "Dual-view generated-vs-original AUC", "sym AUC")
    ax.axhline(0.55, color="#666666", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out / "dual_auc_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    metrics = pd.DataFrame(
        [
            {"model": "E4", "split": "val", "acc": 0.9340, "macro_f1": 0.9385},
            {"model": "E4", "split": "test", "acc": 0.9389, "macro_f1": 0.9431},
            {"model": "E24", "split": "val", "acc": 0.9319, "macro_f1": 0.9363},
            {"model": "E24", "split": "test", "acc": 0.9362, "macro_f1": 0.9413},
        ]
    )
    metrics.pivot(index="model", columns="split", values="acc")[["val", "test"]].plot(kind="bar", ax=ax, color=["#9c755f", "#4e79a7"])
    ax.set_title("Exact-balanced model check accuracy")
    ax.set_ylim(0.88, 0.96)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out / "model_check_accuracy.png", dpi=180)
    plt.close(fig)
    print(out)
