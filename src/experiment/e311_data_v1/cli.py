from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
ANALYSIS = PROJECT_ROOT / "outputs" / "external_benchmarks" / RUN_TAG / "analysis" / "good_medium_geometry_repair"
POLICY = "v116_gapfill_dual_goodorig_nm99_ms10_rnd_s20260876"
SPLIT_ALIAS = "v116_gapfill_dual_goodorig_nm99__k1_s20260876"


def py() -> str:
    exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return str(exe if exe.exists() else Path(sys.executable))


def protocol_dir() -> Path:
    return ANALYSIS / "clean_but_protocols" / POLICY


def split_dir(base: str = "event_factorized_sqi_conformer") -> Path:
    return ANALYSIS / base / "rh_splits" / SPLIT_ALIAS / "fold0"


def command_build() -> list[str]:
    return [
        py(),
        str(ANALYSIS / "run_v116_native_budget_repair.py"),
        "--stage",
        "all",
        "--balance-policy",
        "gap_fill",
        "--seed",
        "20260876",
        "--device",
        "auto",
        "--native-grid",
        "none",
        "--final-per-class",
        "10530",
        "--clean-candidates-per-class",
        "300",
        "--residual-per-subtype",
        "200",
        "--max-donors-per-subtype",
        "1600",
        "--native-morph-copies",
        "8",
        "--native-morph-strength",
        "0.10",
        "--gap-clean-cap",
        "0.01",
        "--gap-native-morph-min-frac",
        "0.99",
        "--gap-native-morph-selection",
        "random",
        "--max-ptb-carriers",
        "9000",
        "--floor-draws",
        "8",
        "--floor-max-rows-per-class",
        "2200",
        "--selector-swaps",
        "1000",
        "--support-max-target-rows",
        "2400",
        "--rff-dim",
        "1024",
    ]


def command_split() -> list[str]:
    return [
        py(),
        str(ANALYSIS / "run_event_factorized_sqi_conformer.py"),
        "--stage",
        "build_recordheldout_splits",
        "--policy",
        POLICY,
        "--folds",
        "1",
        "--seed",
        "20260876",
        "--split-seed",
        "20260876",
        "--no-record-balanced-sampler",
    ]


def command_train(model: str) -> list[str]:
    if model == "E4":
        return [
            py(),
            str(ANALYSIS / "run_event_factorized_sqi_conformer.py"),
            "--stage",
            "phase1",
            "--policy",
            POLICY,
            "--folds",
            "1",
            "--seeds",
            "1",
            "--seed",
            "20260876",
            "--split-seed",
            "20260876",
            "--epochs",
            "8",
            "--batch-size",
            "96",
            "--candidates",
            "E4_query_highres_local_art",
            "--no-record-balanced-sampler",
        ]
    return [
        py(),
        str(ANALYSIS / "run_gm_mechanism_repair_suite.py"),
        "--policy",
        POLICY,
        "--folds",
        "1",
        "--seed",
        "20260876",
        "--split-seed",
        "20260876",
        "--epochs",
        "10",
        "--batch-size",
        "96",
        "--candidates",
        "E24_e6_subtype_fusion_pairrank",
        "--no-record-balanced-sampler",
    ]


def run_or_print(cmd: list[str], run: bool) -> None:
    print(" ".join(cmd))
    if run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def load_atlas(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def audit_payload() -> dict[str, Any]:
    atlas = load_atlas(protocol_dir() / "original_region_atlas.csv")
    split = load_atlas(split_dir() / "original_region_atlas.csv")
    original = atlas[atlas["v116_candidate_type"].astype(str).eq("original_but")]
    payload: dict[str, Any] = {
        "policy": POLICY,
        "protocol_rows": int(len(atlas)),
        "protocol_class_counts": atlas["class_name"].value_counts().sort_index().astype(int).to_dict(),
        "original_but_rows": int(len(original)),
        "original_but_class_counts": original["class_name"].value_counts().sort_index().astype(int).to_dict(),
        "train_class_counts": split.loc[split["split"].eq("train"), "class_name"].value_counts().sort_index().astype(int).to_dict(),
        "val_test_generated_rows": int(len(split[split["split"].isin(["val", "test"]) & ~split["v116_candidate_type"].astype(str).eq("original_but")])),
        "allowed_candidate_types": sorted(split.loc[split["split"].ne("unused"), "v116_candidate_type"].astype(str).unique().tolist()),
        "missing_class_rows": int(split.loc[split["split"].ne("unused"), "class_name"].isna().sum()),
        "missing_idx_rows": int(pd.to_numeric(split.loc[split["split"].ne("unused"), "idx"], errors="coerce").isna().sum()),
    }
    payload["train_generated_donor_split_problems"] = donor_split_problems(split)
    return payload


def donor_split_problems(split: pd.DataFrame) -> int:
    original = split[split["v116_candidate_type"].astype(str).eq("original_but")]
    source_to_split = dict(zip(original["source_idx"].astype(str), original["split"].astype(str)))
    idx_to_split = dict(zip(original["idx"].astype(str), original["split"].astype(str)))
    problems = 0
    generated_train = split[split["split"].eq("train") & ~split["v116_candidate_type"].astype(str).eq("original_but")]
    for _, row in generated_train.iterrows():
        linked = None
        for col in ["v116_native_donor_id", "v116_style_donor_id", "v114_donor_source_idx"]:
            if col not in row.index or pd.isna(row[col]):
                continue
            key = str(int(float(row[col])))
            linked = source_to_split.get(key, idx_to_split.get(key, linked))
        if "v116_residual_donor_id" in row.index and pd.notna(row["v116_residual_donor_id"]):
            record = str(int(float(row["v116_residual_donor_id"])))
            mode = original.loc[original["record_id"].astype(str).eq(record), "split"].mode()
            if len(mode):
                linked = str(mode.iloc[0])
        if linked not in (None, "train"):
            problems += 1
    return problems


def cmd_audit(_: argparse.Namespace) -> None:
    print(json.dumps(audit_payload(), indent=2, sort_keys=True))


def bar(ax: Any, labels: list[str], values: list[float], title: str, ylabel: str = "rows") -> None:
    ax.bar(labels, values, color=["#4c78a8", "#59a14f", "#e15759", "#f28e2b"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)


def cmd_plot(_: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = PROJECT_ROOT / "docs" / "data_v1_figures"
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


def cmd_build(args: argparse.Namespace) -> None:
    run_or_print(command_build(), args.run)
    run_or_print(command_split(), args.run)


def cmd_train(args: argparse.Namespace) -> None:
    models = ["E4", "E24"] if args.model == "both" else [args.model]
    for model in models:
        run_or_print(command_train(model), args.run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data v1 gap-fill reproducibility CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    sub.add_parser("plot").set_defaults(func=cmd_plot)
    build = sub.add_parser("build")
    build.add_argument("--run", action="store_true", help="run the printed build commands")
    build.set_defaults(func=cmd_build)
    train = sub.add_parser("train-check")
    train.add_argument("--model", choices=["E4", "E24", "both"], default="both")
    train.add_argument("--run", action="store_true", help="run the printed training command(s)")
    train.set_defaults(func=cmd_train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
