from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.utils.paths import project_root


PALETTE = {
    "blue": "#0F4D92",
    "blue_soft": "#D9E7F5",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "neutral": "#4D4D4D",
    "neutral_soft": "#CFCECE",
    "ink": "#272727",
    "muted": "#767676",
}


@dataclass(frozen=True)
class ModelTrace:
    name: str
    score_col: str
    color: str
    threshold: float
    auc: float
    ci_low: float
    ci_high: float
    fpr_op: float
    se_op: float
    sp_op: float
    acc_op: float


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["neutral"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )


def load_test_frame(artifacts_dir: Path) -> pd.DataFrame:
    feat = pd.read_parquet(artifacts_dir / "features" / "record84_norm.parquet")
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv")
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    split["source_record_id"] = split["source_record_id"].astype(str)

    df = feat.merge(
        split[["record_id", "split", "source_record_id", "is_augmented", "quality_record", "noise_type"]],
        on="record_id",
        how="inner",
    )
    df["y01"] = df["y"].astype(int).eq(1).astype(np.int32)
    test = df[df["split"].astype(str).eq("test")].copy()
    if test.empty:
        raise RuntimeError("No held-out test rows found in paper-aligned split.")

    for model, rel in [
        ("mlp_score", Path("models/lm_mlp/probs/Selected5_seed0.npz")),
        ("svm_score", Path("models/svm/probs/Selected5_seed0.npz")),
    ]:
        z = np.load(artifacts_dir / rel, allow_pickle=True)
        y_npz = z["y01_test"].astype(np.int32)
        y_test = test["y01"].to_numpy(dtype=np.int32)
        if not np.array_equal(y_npz, y_test):
            raise RuntimeError(f"{rel} y01_test no longer matches reconstructed test row order.")
        test[model] = z["p_test"].astype(np.float64)

    test["score_group"] = np.select(
        [
            test["y"].astype(int).eq(1) & test["is_augmented"].astype(int).eq(0),
            test["y"].astype(int).eq(-1) & test["is_augmented"].astype(int).eq(0),
            test["y"].astype(int).eq(-1) & test["is_augmented"].astype(int).eq(1),
        ],
        ["acceptable", "original unacceptable", "synthetic unacceptable"],
        default="other",
    )
    if test["score_group"].eq("other").any():
        raise RuntimeError("Unexpected score group in held-out test rows.")
    return test


def load_thresholds(artifacts_dir: Path) -> dict[str, float]:
    mlp = pd.read_csv(artifacts_dir / "models" / "lm_mlp" / "tables" / "table7_mlp_selected5_seed0.csv")
    svm = pd.read_csv(artifacts_dir / "models" / "svm" / "table7_svm_selected5_seed0.csv")
    return {"MLP": float(mlp["threshold"].iloc[0]), "SVM": float(svm["threshold"].iloc[0])}


def bootstrap_auc_by_source_group(
    y: np.ndarray,
    score: np.ndarray,
    source_group: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    source_group = np.asarray(source_group, dtype=str)
    auc = float(roc_auc_score(y, score))
    unique_groups = np.array(sorted(pd.unique(source_group)))
    group_to_idx = {g: np.flatnonzero(source_group == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(int(n_boot)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        boot.append(float(roc_auc_score(yy, score[idx])))
    if len(boot) < 100:
        raise RuntimeError("Too few valid bootstrap resamples for AUC CI.")
    low, high = np.percentile(np.asarray(boot), [2.5, 97.5])
    return auc, float(low), float(high)


def operating_point(y: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    pred = (np.asarray(score, dtype=float) > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y, dtype=int), pred, labels=[0, 1]).ravel()
    se = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return float(1.0 - sp), float(se), float(sp), float(acc)


def build_traces(test: pd.DataFrame, thresholds: dict[str, float], n_boot: int, seed: int) -> list[ModelTrace]:
    traces: list[ModelTrace] = []
    y = test["y01"].to_numpy(dtype=int)
    groups = test["source_record_id"].to_numpy(dtype=str)
    for i, (name, col, color) in enumerate(
        [
            ("MLP", "mlp_score", PALETTE["blue"]),
            ("SVM", "svm_score", PALETTE["red"]),
        ]
    ):
        score = test[col].to_numpy(dtype=float)
        auc, ci_low, ci_high = bootstrap_auc_by_source_group(y, score, groups, n_boot=n_boot, seed=seed + i)
        fpr, se, sp, acc = operating_point(y, score, thresholds[name])
        traces.append(
            ModelTrace(
                name=name,
                score_col=col,
                color=color,
                threshold=float(thresholds[name]),
                auc=auc,
                ci_low=ci_low,
                ci_high=ci_high,
                fpr_op=fpr,
                se_op=se,
                sp_op=sp,
                acc_op=acc,
            )
        )
    return traces


def draw_roc_panel(ax: plt.Axes, test: pd.DataFrame, traces: list[ModelTrace]) -> None:
    y = test["y01"].to_numpy(dtype=int)
    for trace, marker in zip(traces, ["o", "s"]):
        score = test[trace.score_col].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(
            fpr,
            tpr,
            color=trace.color,
            lw=1.35,
            label=f"{trace.name} AUC {trace.auc:.3f} [{trace.ci_low:.3f}, {trace.ci_high:.3f}]",
        )
        ax.scatter(
            [trace.fpr_op],
            [trace.se_op],
            marker=marker,
            s=44,
            facecolors="#FFFFFF",
            edgecolors=trace.color,
            linewidths=1.25,
            zorder=4,
        )
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], lw=0.9, linestyle=":", zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1-Sp")
    ax.set_ylabel("Se")
    ax.set_title("held-out test ROC", loc="left", fontsize=8.8, pad=6)
    ax.legend(title="test AUC, source-group 95% CI", loc="lower right", fontsize=7.4, title_fontsize=7.7, handlelength=1.8)


def draw_score_panel(ax: plt.Axes, test: pd.DataFrame, trace: ModelTrace, *, show_legend: bool) -> None:
    group_order = ["acceptable", "original unacceptable", "synthetic unacceptable"]
    colors = {
        "acceptable": PALETTE["blue"],
        "original unacceptable": PALETTE["red"],
        "synthetic unacceptable": PALETTE["neutral"],
    }
    fills = {
        "acceptable": PALETTE["blue_soft"],
        "original unacceptable": PALETTE["red_soft"],
        "synthetic unacceptable": PALETTE["neutral_soft"],
    }
    bins = np.linspace(0, 1, 34)
    for group in group_order:
        vals = test.loc[test["score_group"].eq(group), trace.score_col].to_numpy(dtype=float)
        pretty = {"acceptable": "acceptable", "original unacceptable": "orig. unacceptable", "synthetic unacceptable": "synthetic unacceptable"}[group]
        label = f"{pretty} (n={len(vals)})"
        ax.hist(vals, bins=bins, density=True, histtype="stepfilled", color=fills[group], alpha=0.35, linewidth=0)
        ax.hist(vals, bins=bins, density=True, histtype="step", color=colors[group], linewidth=1.1, label=label)
    ax.axvline(trace.threshold, color=trace.color, linestyle="--", lw=1.1)
    ymax = ax.get_ylim()[1]
    ax.text(
        trace.threshold,
        ymax * 0.93,
        f"thr {trace.threshold:.3f}",
        ha="right" if trace.threshold > 0.72 else "left",
        va="top",
        fontsize=7.4,
        color=trace.color,
        rotation=90,
        backgroundcolor="#FFFFFF",
    )
    ax.set_xlim(0, 1)
    ax.set_ylabel("Density")
    ax.set_title(f"{trace.name} scores", loc="left", fontsize=8.8, pad=4)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    if show_legend:
        draw_score_legend(ax, test)


def draw_score_legend(ax: plt.Axes, test: pd.DataFrame) -> None:
    ax.set_axis_off()
    groups = [
        ("acceptable", "acceptable", PALETTE["blue"], PALETTE["blue_soft"]),
        ("original unacceptable", "original unacc.", PALETTE["red"], PALETTE["red_soft"]),
        ("synthetic unacceptable", "synthetic unacc.", PALETTE["neutral"], PALETTE["neutral_soft"]),
    ]
    handles = [
        Patch(
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.9,
            label=f"{label} (n={int(test['score_group'].eq(group).sum())})",
        )
        for group, label, edge, fill in groups
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.50),
        ncol=1,
        fontsize=6.7,
        frameon=False,
        handlelength=1.15,
        labelspacing=0.28,
        borderaxespad=0.0,
    )


def save_stats(test: pd.DataFrame, traces: list[ModelTrace], out_csv: Path, n_boot: int) -> None:
    rows = []
    for trace in traces:
        rows.append(
            {
                "model": trace.name,
                "test_auc": trace.auc,
                "test_auc_ci_low_source_group_bootstrap": trace.ci_low,
                "test_auc_ci_high_source_group_bootstrap": trace.ci_high,
                "n_bootstrap": int(n_boot),
                "threshold_validation_selected": trace.threshold,
                "test_operating_point_1_minus_sp": trace.fpr_op,
                "test_operating_point_se": trace.se_op,
                "test_operating_point_sp": trace.sp_op,
                "test_operating_point_acc": trace.acc_op,
                "n_test_rows": int(len(test)),
                "n_test_source_groups": int(test["source_record_id"].nunique()),
                "n_acceptable": int(test["score_group"].eq("acceptable").sum()),
                "n_original_unacceptable": int(test["score_group"].eq("original unacceptable").sum()),
                "n_synthetic_unacceptable": int(test["score_group"].eq("synthetic unacceptable").sum()),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def plot_figure(artifacts_dir: Path, out_dir: Path, *, n_boot: int = 5000, seed: int = 0) -> Path:
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_test_frame(artifacts_dir)
    thresholds = load_thresholds(artifacts_dir)
    traces = build_traces(test, thresholds, n_boot=n_boot, seed=seed)

    fig = plt.figure(figsize=(7.7, 4.35))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.16, 1.0], height_ratios=[0.36, 1, 1], wspace=0.34, hspace=0.38)
    ax_roc = fig.add_subplot(gs[:, 0])
    ax_score_legend = fig.add_subplot(gs[0, 1])
    ax_mlp = fig.add_subplot(gs[1, 1])
    ax_svm = fig.add_subplot(gs[2, 1], sharex=ax_mlp)

    add_panel_label(ax_roc, "a", x=-0.13, y=1.02)
    add_panel_label(ax_score_legend, "b", x=-0.14, y=0.82)
    draw_roc_panel(ax_roc, test, traces)
    draw_score_legend(ax_score_legend, test)
    draw_score_panel(ax_mlp, test, traces[0], show_legend=False)
    draw_score_panel(ax_svm, test, traces[1], show_legend=False)
    ax_mlp.set_xlabel("")
    ax_mlp.tick_params(axis="x", labelbottom=False)
    ax_svm.set_xlabel("Model score")

    base = out_dir / "fig_09_optimised_fusion_diagnostics"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    save_stats(test, traces, base.with_name(base.name + "_stats.csv"), n_boot)
    return base.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Plot optimised-fusion diagnostics for paper-aligned SQI Table 7.")
    parser.add_argument("--artifacts_dir", default=str(root / "outputs" / "sqi_paper_aligned"))
    parser.add_argument("--out_dir", default=str(root / "reports" / "sqi_paper_aligned" / "images"))
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = plot_figure(Path(args.artifacts_dir), Path(args.out_dir), n_boot=int(args.n_boot), seed=int(args.seed))
    print(out)


if __name__ == "__main__":
    main()
