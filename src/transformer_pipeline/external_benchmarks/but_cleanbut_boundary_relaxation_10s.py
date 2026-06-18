"""CleanBUT filtered-boundary relaxation diagnostics.

This experiment treats CleanBUT-Core as a diagnostic target, not as the formal
BUT benchmark.  It ranks BUT windows by prediction-free label-confidence
features from the CleanBUT atlas, then gradually relaxes the selected subset
from pure class cores into ambiguous boundary regions.  Trained checkpoints are
evaluated on those filtered, class-balanced subsets so we can find the widest
boundary level that still reaches a target accuracy.
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
import torch
import torch.nn.functional as F

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_clean_core_targeted_generator_grid_10s as targeted
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import ensure_but_10s, read_json
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.run import (
    apply_but_thresholds,
    load_uformer_model,
    multiclass_report,
    plot_wave_gallery,
)


RUN_TAG = "e311_but_cleanbut_boundary_relaxation_10s_2026_06_07"
ROOT = targeted.ROOT
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_CLEAN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_clean_label_core_10s_2026_06_06"
DEFAULT_TRAIN_ROOT = targeted.DEFAULT_OUT_ROOT

CLASS_NAMES = ("good", "medium", "bad")
COLORS = {"good": "#1b9e77", "medium": "#fdae61", "bad": "#d73027"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def add_confidence_score(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest.copy()
    df["idx"] = df["idx"].astype(int)
    df["y"] = df["y"].astype(int)
    df["class_name"] = df["class_name"].astype(str)
    df["clean_tier"] = df["clean_tier"].astype(str)
    for col in [
        "knn_label_purity",
        "pca_margin",
        "class_margin_percentile",
        "class_centrality_percentile",
        "pca_own_distance",
        "pc1",
        "pc2",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pca_margin_rank"] = df.groupby("y")["pca_margin"].rank(pct=True, method="average")
    # Lower own-distance is more central; rank descending after negation.
    df["own_centrality_rank"] = df.groupby("y")["pca_own_distance"].transform(lambda s: (-s).rank(pct=True, method="average"))
    tier_bonus = df["clean_tier"].map({"clean_core_strict": 0.18, "clean_core_train_target": 0.10}).fillna(0.0)
    df["boundary_confidence"] = (
        0.35 * df["knn_label_purity"].fillna(0.0)
        + 0.25 * df["pca_margin_rank"].fillna(0.0)
        + 0.20 * df["class_margin_percentile"].fillna(0.0)
        + 0.12 * df["class_centrality_percentile"].fillna(0.0)
        + 0.08 * df["own_centrality_rank"].fillna(0.0)
        + tier_bonus
    )
    return df.sort_values(["y", "boundary_confidence"], ascending=[True, False]).reset_index(drop=True)


def select_top_n_per_class(manifest: pd.DataFrame, n_per_class: int) -> np.ndarray:
    selected: list[int] = []
    for cls in (0, 1, 2):
        sub = manifest.loc[manifest["y"].astype(int) == cls].sort_values("boundary_confidence", ascending=False)
        selected.extend(int(v) for v in sub.head(int(n_per_class))["idx"].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def n_grid(manifest: pd.DataFrame) -> list[int]:
    max_n = int(min((manifest["y"].astype(int) == cls).sum() for cls in (0, 1, 2)))
    base = [50, 100, 150, 200, 300, 400, 500, 719, 900, 1198, 1500, 1970, 2500, 3000, 3264, 3600, 4200, 4800, max_n]
    vals = sorted({int(v) for v in base if 10 <= int(v) <= max_n})
    return vals


def completed_checkpoints(train_root: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(train_root / targeted.SUMMARY_NAME)
    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        vid = str(row.get("spec", {}).get("id", ""))
        ckpt = row.get("but_10s_eval", {}).get("checkpoint")
        if vid and int(row.get("returncode", 1)) == 0 and ckpt and Path(ckpt).exists():
            by_variant[vid] = row
    # If a run finished after a previous summary rewrite, recover from per-run summaries.
    for summary_path in (train_root / "runs" / "quick").glob("*/clean_core_targeted_run_summary.json"):
        row = read_json(summary_path)
        vid = str(row.get("spec", {}).get("id", ""))
        ckpt = row.get("but_10s_eval", {}).get("checkpoint")
        if vid and int(row.get("returncode", 1)) == 0 and ckpt and Path(ckpt).exists():
            by_variant[vid] = row
    return list(by_variant.values())


@torch.no_grad()
def predict_probs(checkpoint: Path, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model = load_uformer_model(checkpoint, device, feature_set="full_tokens", with_head=True)
    probs_all: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(X), int(batch_size)):
        xb = torch.from_numpy(X[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
        out = model(xb)
        logits = out["logits"]
        if logits is None:
            logits = torch.zeros((xb.shape[0], 3), device=device)
        probs_all.append(F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(probs_all, axis=0)


def subset_composition(manifest: pd.DataFrame, selected_idx: np.ndarray) -> dict[str, Any]:
    sub = manifest.loc[manifest["idx"].isin(selected_idx)].copy()
    out: dict[str, Any] = {"n_total": int(len(sub))}
    for cls in CLASS_NAMES:
        s = sub.loc[sub["class_name"].eq(cls)]
        out[f"{cls}_n"] = int(len(s))
        out[f"{cls}_clean_core_strict"] = int(s["clean_tier"].eq("clean_core_strict").sum())
        out[f"{cls}_clean_core_train_target"] = int(s["clean_tier"].eq("clean_core_train_target").sum())
        out[f"{cls}_ambiguous_boundary"] = int(s["clean_tier"].eq("ambiguous_boundary").sum())
        out[f"{cls}_confidence_min"] = float(s["boundary_confidence"].min()) if len(s) else math.nan
        out[f"{cls}_confidence_median"] = float(s["boundary_confidence"].median()) if len(s) else math.nan
    out["ambiguous_fraction"] = float(sub["clean_tier"].eq("ambiguous_boundary").mean()) if len(sub) else math.nan
    return out


def evaluate_relaxation(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = add_confidence_score(pd.read_csv(Path(args.clean_root) / "clean_subset_manifest.csv"))
    manifest.to_csv(out_root / "boundary_confidence_manifest.csv", index=False)
    X, meta = ensure_but_10s(args)
    y = meta["y"].to_numpy(dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoints = completed_checkpoints(Path(args.train_root))
    if not checkpoints:
        raise FileNotFoundError(f"No completed checkpoints found under {args.train_root}")

    grid = n_grid(manifest)
    selected_by_n = {n: select_top_n_per_class(manifest, n) for n in grid}
    composition_rows = []
    for n, idx in selected_by_n.items():
        composition_rows.append({"n_per_class": n, **subset_composition(manifest, idx)})
    composition_df = pd.DataFrame(composition_rows)
    composition_df.to_csv(out_root / "boundary_subset_composition.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    pred_dir = out_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(checkpoints, start=1):
        vid = str(row.get("spec", {}).get("id"))
        ckpt = Path(row["but_10s_eval"]["checkpoint"])
        cal = row.get("but_10s_eval", {}).get("calibration", {})
        t_good = float(cal.get("t_good", 0.5))
        t_bad = float(cal.get("t_bad", 0.5))
        pred_cache = pred_dir / f"{vid}_probs.npz"
        if pred_cache.exists():
            probs = np.load(pred_cache)["probs"].astype(np.float32)
        else:
            probs = predict_probs(ckpt, X, int(args.batch_size_eval), device)
            np.savez_compressed(pred_cache, probs=probs.astype(np.float32))
        raw_pred_all = np.argmax(probs, axis=1).astype(np.int64)
        cal_pred_all = apply_but_thresholds(probs, t_good, t_bad)
        pred_df = pd.DataFrame(
            {
                "idx": np.arange(len(y), dtype=np.int64),
                "variant_id": vid,
                "true_y": y,
                "raw_pred": raw_pred_all,
                "calibrated_pred": cal_pred_all,
                "p_good": probs[:, 0],
                "p_medium": probs[:, 1],
                "p_bad": probs[:, 2],
            }
        )
        prediction_rows.append(pred_df)
        original = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        clean64 = row.get("clean64_distance", {})
        for n, idx in selected_by_n.items():
            for mode, pred_all in (("raw", raw_pred_all), ("calibrated", cal_pred_all)):
                rep = multiclass_report(y[idx], pred_all[idx], probs[idx])
                rec = rep["recall_good_medium_bad"]
                metric_rows.append(
                    {
                        "variant_id": vid,
                        "prediction_mode": mode,
                        "n_per_class": int(n),
                        "n_total": int(len(idx)),
                        "acc": rep["acc"],
                        "balanced_acc": rep["balanced_acc"],
                        "macro_f1": rep["macro_f1"],
                        "good_recall": rec[0],
                        "medium_recall": rec[1],
                        "bad_recall": rec[2],
                        "confusion_3x3": json.dumps(rep["confusion_3x3"]),
                        "original_but_acc": original.get("acc"),
                        "original_but_macro_f1": original.get("macro_f1"),
                        "clean64_score": clean64.get("score"),
                        "good_64d_KS": clean64.get("good_64d_KS"),
                        "medium_64d_KS": clean64.get("medium_64d_KS"),
                        "bad_64d_KS": clean64.get("bad_64d_KS"),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics = metrics.merge(composition_df, on=["n_per_class", "n_total"], how="left")
    metrics.to_csv(out_root / "boundary_relaxation_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out_root / "boundary_relaxation_predictions.csv", index=False)

    best_rows = []
    for (vid, mode), sub in metrics.groupby(["variant_id", "prediction_mode"]):
        above = sub.loc[sub["acc"] >= float(args.target_acc)].sort_values(["n_per_class", "acc"], ascending=[False, True])
        if above.empty:
            chosen = sub.iloc[(sub["acc"] - float(args.target_acc)).abs().argsort()[:1]].iloc[0]
            status = "closest_below_or_above"
        else:
            chosen = above.iloc[0]
            status = "max_coverage_at_or_above_target"
        d = chosen.to_dict()
        d["status"] = status
        best_rows.append(d)
    best_df = pd.DataFrame(best_rows).sort_values(["status", "n_per_class", "acc"], ascending=[True, False, False])
    best_df.to_csv(out_root / "boundary_relaxation_best_at_0p95.csv", index=False)
    return metrics, best_df, composition_df, manifest


def plot_curves(metrics: pd.DataFrame, best_df: pd.DataFrame, report_root: Path, target_acc: float) -> None:
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = metrics.loc[metrics["prediction_mode"].eq("raw")].copy()
    top_variants = (
        best_df.loc[best_df["prediction_mode"].eq("raw")]
        .sort_values(["n_per_class", "acc"], ascending=[False, False])
        .head(6)["variant_id"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for vid in top_variants:
        sub = raw.loc[raw["variant_id"].eq(vid)].sort_values("n_per_class")
        label = vid.replace("cc_bad_", "").replace("_quiet_core_tight_sec0p00", "")
        ax.plot(sub["n_per_class"], sub["acc"], marker="o", ms=3, lw=1.5, label=label[:48])
    ax.axhline(target_acc, color="#111827", ls="--", lw=1.0, label=f"target acc={target_acc:.2f}")
    ax.set_xlabel("selected windows per class (larger means more boundary/ambiguous samples)")
    ax.set_ylabel("filtered balanced accuracy")
    ax.set_ylim(0.25, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("CleanBUT filtered-boundary relaxation curve")
    fig.tight_layout()
    fig.savefig(fig_dir / "boundary_relaxation_curve.png", dpi=180)
    plt.close(fig)


def plot_boundary_pca(args: argparse.Namespace, metrics: pd.DataFrame, best_df: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    report_root = Path(args.report_root)
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    candidates = best_df.loc[
        best_df["prediction_mode"].eq("raw") & best_df["status"].eq("max_coverage_at_or_above_target")
    ].copy()
    if candidates.empty:
        candidates = best_df.loc[best_df["prediction_mode"].eq("raw")].copy()
    chosen = candidates.sort_values(["n_per_class", "acc"], ascending=[False, False]).iloc[0].to_dict()
    n = int(chosen["n_per_class"])
    vid = str(chosen["variant_id"])
    selected_idx = set(select_top_n_per_class(manifest, n).tolist())
    plot_df = manifest.copy()
    plot_df["selected"] = plot_df["idx"].isin(selected_idx)
    rng = np.random.default_rng(20260607)
    bg = plot_df.sample(min(len(plot_df), int(args.max_plot_rows)), random_state=20260607)
    sel = plot_df.loc[plot_df["selected"]].copy()
    if len(sel) > int(args.max_plot_rows):
        sel = sel.sample(int(args.max_plot_rows), random_state=20260608)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(bg["pc1"], bg["pc2"], s=3, c="#d1d5db", alpha=0.16, linewidths=0)
        ax.grid(alpha=0.18)
        ax.set_xlabel("CleanBUT 64D-PC1")
        ax.set_ylabel("CleanBUT 64D-PC2")
    for cls in CLASS_NAMES:
        sub = sel.loc[sel["class_name"].eq(cls)]
        axes[0].scatter(sub["pc1"], sub["pc2"], s=7, c=COLORS[cls], alpha=0.72, linewidths=0, label=cls)
    axes[0].set_title(f"Selected filtered diagnostic\nn/class={n}, acc={float(chosen['acc']):.3f}")
    axes[0].legend(fontsize=8)

    for tier, color in [("clean_core_strict", "#2563eb"), ("clean_core_train_target", "#22c55e"), ("ambiguous_boundary", "#ef4444")]:
        sub = sel.loc[sel["clean_tier"].eq(tier)]
        axes[1].scatter(sub["pc1"], sub["pc2"], s=7, c=color, alpha=0.65, linewidths=0, label=tier)
    axes[1].set_title("Same selected points by clean/ambiguous tier")
    axes[1].legend(fontsize=7)

    # Show the lowest-confidence included points as the current boundary shell.
    shell_parts = []
    for cls in CLASS_NAMES:
        sub = plot_df.loc[plot_df["selected"] & plot_df["class_name"].eq(cls)].sort_values("boundary_confidence", ascending=True).head(180)
        shell_parts.append(sub)
    shell = pd.concat(shell_parts, ignore_index=True)
    axes[2].scatter(sel["pc1"], sel["pc2"], s=4, c="#9ca3af", alpha=0.22, linewidths=0)
    for cls in CLASS_NAMES:
        sub = shell.loc[shell["class_name"].eq(cls)]
        axes[2].scatter(sub["pc1"], sub["pc2"], s=12, c=COLORS[cls], alpha=0.9, linewidths=0, label=f"{cls} boundary")
    axes[2].set_title("Lowest-confidence included shell")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"CleanBUT relaxed filtered subset: {vid}", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_0p95_boundary_pca.png", dpi=190)
    plt.close(fig)

    # Composition bars.
    comp_cols = [
        "clean_core_strict",
        "clean_core_train_target",
        "ambiguous_boundary",
    ]
    comp = []
    selected = plot_df.loc[plot_df["selected"]]
    for cls in CLASS_NAMES:
        s = selected.loc[selected["class_name"].eq(cls)]
        comp.append([int(s["clean_tier"].eq(c).sum()) for c in comp_cols])
    comp_arr = np.asarray(comp, dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bottom = np.zeros(len(CLASS_NAMES))
    colors = ["#2563eb", "#22c55e", "#ef4444"]
    for j, col in enumerate(comp_cols):
        ax.bar(CLASS_NAMES, comp_arr[:, j], bottom=bottom, label=col, color=colors[j], alpha=0.85)
        bottom += comp_arr[:, j]
    ax.set_ylabel("selected windows")
    ax.set_title("Clean vs ambiguous composition at selected boundary")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_0p95_boundary_composition.png", dpi=180)
    plt.close(fig)

    # Galleries for the boundary shell.
    X, meta = ensure_but_10s(args)
    gallery_dir = report_root / "boundary_waveform_galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        sub = shell.loc[shell["class_name"].eq(cls)].sort_values("boundary_confidence", ascending=True).head(12)
        pos = sub["idx"].astype(int).tolist()
        if pos:
            plot_wave_gallery(X, meta, pos, gallery_dir / f"{cls}_lowest_confidence_included.png", f"{cls} lowest-confidence included boundary")

    return chosen


def write_report(args: argparse.Namespace, metrics: pd.DataFrame, best_df: pd.DataFrame, chosen: dict[str, Any]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    best_raw = best_df.loc[best_df["prediction_mode"].eq("raw")].sort_values(["n_per_class", "acc"], ascending=[False, False])
    lines = [
        "# CleanBUT Filtered Boundary Relaxation",
        "",
        "This is a filtered diagnostic benchmark, not the formal BUT original test. Selection uses CleanBUT atlas label-confidence features only; model predictions are used only after the subset is fixed.",
        "",
        "## Main Finding",
        "",
        f"- Widest raw subset at/above target: `{chosen.get('variant_id')}`",
        f"- Selected per class: `{int(chosen.get('n_per_class', 0))}`; total `{int(chosen.get('n_total', 0))}`",
        f"- Filtered acc: `{float(chosen.get('acc', math.nan)):.4f}`; macro-F1 `{float(chosen.get('macro_f1', math.nan)):.4f}`",
        f"- Recalls good/medium/bad: `{float(chosen.get('good_recall', math.nan)):.3f}/{float(chosen.get('medium_recall', math.nan)):.3f}/{float(chosen.get('bad_recall', math.nan)):.3f}`",
        f"- Ambiguous fraction: `{float(chosen.get('ambiguous_fraction', math.nan)):.3f}`",
        "",
        "## Top Raw Operating Points",
        "",
        "| rank | variant | n/class | acc | macro-F1 | recalls good/medium/bad | ambiguous fraction | original BUT macro |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for rank, (_, row) in enumerate(best_raw.head(12).iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['variant_id']}` | {int(row['n_per_class'])} | {float(row['acc']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['good_recall']):.3f}/{float(row['medium_recall']):.3f}/{float(row['bad_recall']):.3f} | "
            f"{float(row.get('ambiguous_fraction', math.nan)):.3f} | {float(row.get('original_but_macro_f1', math.nan)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `boundary_relaxation_metrics.csv`: every model, raw/calibrated mode, and boundary level.",
            "- `boundary_relaxation_best_at_0p95.csv`: widest subset per model/mode at the target accuracy.",
            "- `figures/boundary_relaxation_curve.png`: accuracy as the boundary is relaxed.",
            "- `figures/best_0p95_boundary_pca.png`: selected subset, ambiguity tiers, and current boundary shell.",
            "- `figures/best_0p95_boundary_composition.png`: clean vs ambiguous composition.",
            "- `boundary_waveform_galleries/`: low-confidence included samples at the selected boundary.",
        ]
    )
    (report_root / "cleanbut_boundary_relaxation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics.to_csv(report_root / "boundary_relaxation_metrics.csv", index=False)
    best_df.to_csv(report_root / "boundary_relaxation_best_at_0p95.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CleanBUT filtered-boundary relaxation on trained checkpoints.")
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    p.add_argument("--clean_root", default=str(DEFAULT_CLEAN_ROOT))
    p.add_argument("--train_root", default=str(DEFAULT_TRAIN_ROOT))
    p.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    p.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    p.add_argument("--batch_size_eval", type=int, default=256)
    p.add_argument("--target_acc", type=float, default=0.95)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max_plot_rows", type=int, default=9000)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics, best_df, composition_df, manifest = evaluate_relaxation(args)
    report_root = Path(args.report_root)
    plot_curves(metrics, best_df, report_root, float(args.target_acc))
    chosen = plot_boundary_pca(args, metrics, best_df, manifest)
    write_report(args, metrics, best_df, chosen)
    write_json(
        Path(args.out_root) / "boundary_relaxation_state.json",
        {
            "status": "complete",
            "target_acc": float(args.target_acc),
            "chosen": chosen,
            "n_metric_rows": int(len(metrics)),
            "report": str(Path(args.report_root) / "cleanbut_boundary_relaxation_report.md"),
        },
    )
    print(json.dumps({"status": "complete", "chosen": chosen}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
