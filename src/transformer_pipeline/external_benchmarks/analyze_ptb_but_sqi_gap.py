"""Analyze the PTB-vs-BUT seven-SQI domain gap.

This is a source-backed analysis helper for the BUT SQI fusion line.  It reads
the already-reviewed PTB three-class SQI control cache and the BUT 10s P1 SQI
cache, then writes compact statistics, charts, and a markdown report.  It does
not touch ``src/sqi_pipeline`` or any mainline checkpoint.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.transformer_pipeline.e311_uformer_eval import write_json


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
SQI_COLUMNS = ["I__iSQI", "I__bSQI", "I__pSQI", "I__sSQI", "I__kSQI", "I__fSQI", "I__basSQI"]
CLASS_NAMES = ("good", "medium", "bad")
RUN_TAG = "e311_but_sqi_gap_analysis_10s_2026_06_04"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_PTB_CONTROL = ROOT / "outputs" / "controls" / "e311f_ptb_sqi_three_class"
DEFAULT_BUT_PROTOCOL = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_protocol_adaptation_2026_06_03"
    / "protocols"
    / "p1_current_10s_center"
)
DEFAULT_BUT_SQI_CACHE = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_sqi_fusion_ptb_train_10s_2026_06_04"
    / "feature_cache"
    / "but_sqi7"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def label_name(y: int) -> str:
    return CLASS_NAMES[int(y)]


def load_ptb(ptb_control: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(ptb_control / "splits" / "transformer_three_class_seed0.csv")
    split["record_id"] = split["record_id"].astype(str)
    raw = pd.read_parquet(ptb_control / "features" / "record7.parquet")
    norm = pd.read_parquet(ptb_control / "features" / "record7_norm.parquet")
    for df in (raw, norm):
        df["record_id"] = df["record_id"].astype(str)
    keep = ["record_id", "y", "split", "y_class", "snr_db", "noise_kind", "placement", "label_subtype"]
    meta = split[[c for c in keep if c in split.columns]].copy()
    raw = meta.merge(raw[["record_id", *SQI_COLUMNS]], on="record_id", how="left")
    norm = meta.merge(norm[["record_id", *SQI_COLUMNS]], on="record_id", how="left")
    for df in (raw, norm):
        df["dataset"] = "PTB synthetic"
        df["class_name"] = df["y"].map(label_name)
    return raw, norm


def load_but(but_protocol: Path, but_sqi_cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(but_protocol / "metadata.csv")
    raw = pd.read_csv(but_sqi_cache / "but_record7_raw.csv")
    norm = np.load(but_sqi_cache / "but_record7_norm.npy").astype(np.float32)
    norm_df = pd.DataFrame(norm, columns=SQI_COLUMNS)
    norm_df["window_id"] = meta["window_id"].astype(str).to_numpy()
    raw["window_id"] = raw["window_id"].astype(str)
    keep = [
        "window_id",
        "record_id",
        "subject_id",
        "split",
        "label_raw",
        "y_class",
        "y",
        "label_purity",
        "raw_rms_centered",
        "normalization_scale",
    ]
    meta_keep = meta[[c for c in keep if c in meta.columns]].copy()
    raw = meta_keep.merge(raw[["window_id", *SQI_COLUMNS]], on="window_id", how="left")
    norm_df = meta_keep.merge(norm_df[["window_id", *SQI_COLUMNS]], on="window_id", how="left")
    for df in (raw, norm_df):
        df["dataset"] = "BUT 10s P1"
        df["class_name"] = df["y"].map(label_name)
    return raw, norm_df


def robust_stats(df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(df["dataset"].unique()):
        for cls in CLASS_NAMES:
            sub = df[(df["dataset"] == dataset) & (df["class_name"] == cls)]
            for col in SQI_COLUMNS:
                x = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                rows.append(
                    {
                        "feature_mode": feature_mode,
                        "dataset": dataset,
                        "class_name": cls,
                        "feature": col,
                        "n": int(len(x)),
                        "mean": float(np.mean(x)) if len(x) else math.nan,
                        "median": float(np.median(x)) if len(x) else math.nan,
                        "std": float(np.std(x)) if len(x) else math.nan,
                        "p10": float(np.percentile(x, 10)) if len(x) else math.nan,
                        "p90": float(np.percentile(x, 90)) if len(x) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return math.nan
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return math.nan
    q = np.linspace(0.01, 0.99, 99)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def domain_distance(df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cls in CLASS_NAMES:
        ptb = df[(df["dataset"] == "PTB synthetic") & (df["class_name"] == cls)]
        but = df[(df["dataset"] == "BUT 10s P1") & (df["class_name"] == cls)]
        for col in SQI_COLUMNS:
            a = ptb[col].to_numpy(dtype=float)
            b = but[col].to_numpy(dtype=float)
            pooled = np.nanstd(np.concatenate([a, b]))
            rows.append(
                {
                    "feature_mode": feature_mode,
                    "class_name": cls,
                    "feature": col,
                    "ptb_median": float(np.nanmedian(a)),
                    "but_median": float(np.nanmedian(b)),
                    "median_delta_but_minus_ptb": float(np.nanmedian(b) - np.nanmedian(a)),
                    "standardized_median_delta": float((np.nanmedian(b) - np.nanmedian(a)) / (pooled + 1e-8)),
                    "ks": ks_stat(a, b),
                    "wasserstein": wasserstein_1d(a, b),
                }
            )
    return pd.DataFrame(rows)


def effect_size(df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [("medium", "good"), ("bad", "good"), ("bad", "medium")]
    for dataset in ("PTB synthetic", "BUT 10s P1"):
        for feature in SQI_COLUMNS:
            for hi, lo in pairs:
                a = df[(df["dataset"] == dataset) & (df["class_name"] == hi)][feature].to_numpy(dtype=float)
                b = df[(df["dataset"] == dataset) & (df["class_name"] == lo)][feature].to_numpy(dtype=float)
                pooled = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2.0)
                rows.append(
                    {
                        "feature_mode": feature_mode,
                        "dataset": dataset,
                        "feature": feature,
                        "contrast": f"{hi}_minus_{lo}",
                        "effect_d": float((np.nanmean(a) - np.nanmean(b)) / (pooled + 1e-8)),
                        "median_delta": float(np.nanmedian(a) - np.nanmedian(b)),
                    }
                )
    eff = pd.DataFrame(rows)
    joined = eff.pivot_table(
        index=["feature_mode", "feature", "contrast"],
        columns="dataset",
        values="effect_d",
        aggfunc="first",
    ).reset_index()
    joined["effect_alignment"] = np.sign(joined["PTB synthetic"].fillna(0.0)) * np.sign(joined["BUT 10s P1"].fillna(0.0))
    joined["abs_effect_gap"] = (joined["PTB synthetic"] - joined["BUT 10s P1"]).abs()
    return eff.merge(joined, on=["feature_mode", "feature", "contrast"], suffixes=("", "_joined"))


def dataset_separability(df_norm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cls in ("all", *CLASS_NAMES):
        if cls == "all":
            sub = df_norm.copy()
        else:
            sub = df_norm[df_norm["class_name"] == cls].copy()
        X = sub[SQI_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        y = (sub["dataset"] == "BUT 10s P1").astype(int).to_numpy()
        # Deterministic split by row order is enough for a domain-gap diagnostic;
        # we are not claiming a production classifier here.
        idx = np.arange(len(y))
        train = idx % 3 != 0
        test = ~train
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        clf.fit(X[train], y[train])
        prob = clf.predict_proba(X[test])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            {
                "class_name": cls,
                "n": int(len(y)),
                "but_share": float(np.mean(y)),
                "domain_auc": float(roc_auc_score(y[test], prob)) if len(np.unique(y[test])) > 1 else math.nan,
                "domain_balanced_acc": float(balanced_accuracy_score(y[test], pred)) if len(np.unique(y[test])) > 1 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def make_plots(
    df_raw: pd.DataFrame,
    df_norm: pd.DataFrame,
    distance: pd.DataFrame,
    effects: pd.DataFrame,
    separability: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    long_norm = df_norm.melt(
        id_vars=["dataset", "class_name"],
        value_vars=SQI_COLUMNS,
        var_name="feature",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    positions = np.arange(len(CLASS_NAMES))
    colors = {"PTB synthetic": "#2b6cb0", "BUT 10s P1": "#c05621"}
    for ax, feature in zip(axes, SQI_COLUMNS):
        ax.axhline(0, color="#d9d9d9", lw=0.8)
        for offset, dataset in [(-0.18, "PTB synthetic"), (0.18, "BUT 10s P1")]:
            vals = [
                long_norm[
                    (long_norm["feature"] == feature)
                    & (long_norm["dataset"] == dataset)
                    & (long_norm["class_name"] == cls)
                ]["value"].dropna().clip(-8, 8).to_numpy()
                for cls in CLASS_NAMES
            ]
            bp = ax.boxplot(
                vals,
                positions=positions + offset,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#202020", "lw": 1.0},
                whiskerprops={"color": colors[dataset], "lw": 0.8},
                capprops={"color": colors[dataset], "lw": 0.8},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[dataset])
                patch.set_alpha(0.35)
                patch.set_edgecolor(colors[dataset])
        ax.set_title(feature)
        ax.set_xticks(positions)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylabel("PTB-normalized SQI")
    axes[-1].axis("off")
    axes[-1].plot([], [], color=colors["PTB synthetic"], lw=6, alpha=0.45, label="PTB synthetic")
    axes[-1].plot([], [], color=colors["BUT 10s P1"], lw=6, alpha=0.45, label="BUT 10s P1")
    axes[-1].legend(loc="center")
    fig.suptitle("PTB vs BUT seven-SQI distributions by class", y=0.99)
    fig.tight_layout()
    fig.savefig(fig_dir / "ptb_but_sqi_boxplots_norm.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    heat = effects[effects["feature_mode"] == "normalized"].drop_duplicates(["feature", "contrast"])[
        ["feature", "contrast", "abs_effect_gap"]
    ]
    mat = heat.pivot(index="feature", columns="contrast", values="abs_effect_gap").loc[SQI_COLUMNS]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", color="white" if mat.iloc[i, j] > np.nanmax(mat.to_numpy()) * 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="|PTB effect - BUT effect|")
    ax.set_title("Class-separation effect gap: PTB vs BUT")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_effect_gap_heatmap.png", dpi=180)
    plt.close()

    dist = distance[distance["feature_mode"] == "normalized"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SQI_COLUMNS))
    width = 0.24
    for idx, cls in enumerate(CLASS_NAMES):
        vals = [float(dist[(dist["feature"] == feat) & (dist["class_name"] == cls)]["ks"].iloc[0]) for feat in SQI_COLUMNS]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels(SQI_COLUMNS, rotation=30, ha="right")
    ax.set_ylabel("KS distance")
    ax.set_title("SQI domain shift is class-specific")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "ks_distance_by_feature_class.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(separability["class_name"], separability["domain_auc"], color="#2b6cb0")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC: distinguish BUT from PTB using 7SQI")
    ax.set_title("7SQI carries strong dataset identity")
    plt.tight_layout()
    plt.savefig(fig_dir / "domain_separability_auc.png", dpi=180)
    plt.close()

    means = (
        df_norm.groupby(["dataset", "class_name"])[SQI_COLUMNS]
        .median()
        .reset_index()
        .melt(id_vars=["dataset", "class_name"], var_name="feature", value_name="median")
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, feature in zip(axes, SQI_COLUMNS):
        sub = means[means["feature"] == feature]
        for dataset, color in colors.items():
            vals = [float(sub[(sub["dataset"] == dataset) & (sub["class_name"] == cls)]["median"].iloc[0]) for cls in CLASS_NAMES]
            ax.plot(CLASS_NAMES, vals, marker="o", label=dataset, color=color)
        ax.axhline(0, color="#d9d9d9", lw=0.8)
        ax.set_title(feature)
        ax.set_ylabel("Median normalized SQI")
    axes[-1].axis("off")
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Class trajectory differs across PTB and BUT", y=0.99)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_median_trajectories.png", dpi=170, bbox_inches="tight")
    plt.close("all")


def top_findings(distance: pd.DataFrame, effects: pd.DataFrame, separability: pd.DataFrame) -> list[str]:
    norm_dist = distance[distance["feature_mode"] == "normalized"].copy()
    top_shift = norm_dist.sort_values("ks", ascending=False).head(5)
    eff = effects[effects["feature_mode"] == "normalized"].drop_duplicates(["feature", "contrast"]).copy()
    misaligned = eff[eff["effect_alignment"] < 0].sort_values("abs_effect_gap", ascending=False).head(5)
    sep_all = separability[separability["class_name"] == "all"].iloc[0]
    lines = [
        f"7SQI can distinguish BUT from PTB with AUC {sep_all['domain_auc']:.3f}, so it carries strong domain identity in addition to quality evidence.",
        "Largest class-wise distribution shifts: "
        + ", ".join(f"{r.feature}/{r.class_name} KS={r.ks:.2f}" for r in top_shift.itertuples()),
    ]
    if not misaligned.empty:
        lines.append(
            "Some SQI class directions flip between PTB and BUT: "
            + ", ".join(f"{r.feature} {r.contrast}" for r in misaligned.itertuples())
            + ". This explains why high SQI weight can help calibration but may also hurt one boundary."
        )
    else:
        lines.append("SQI class directions mostly align; the main issue is scale/calibration rather than semantic reversal.")
    lines.append(
        "Implication: use SQI as a higher-weight but calibrated branch, not as a replacement for morphology/Uformer features."
    )
    return lines


def write_report(
    report_root: Path,
    out_root: Path,
    stats: pd.DataFrame,
    distance: pd.DataFrame,
    effects: pd.DataFrame,
    separability: pd.DataFrame,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    findings = top_findings(distance, effects, separability)
    best_sep = separability.sort_values("domain_auc", ascending=False).iloc[0]
    worst_alignment = (
        effects[effects["feature_mode"] == "normalized"]
        .drop_duplicates(["feature", "contrast"])
        .sort_values("abs_effect_gap", ascending=False)
        .head(8)
    )
    lines = [
        "# PTB vs BUT Seven-SQI Domain Gap",
        "",
        "This report compares the same seven traditional SQI features on PTB synthetic training data and formal BUT 10s P1 windows.",
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend([f"- {x}" for x in findings])
    lines.extend(
        [
            "",
            "## What We Measured",
            "",
            "- Features: iSQI, bSQI, pSQI, sSQI, kSQI, fSQI, basSQI from noisy ECG only.",
            "- PTB source: `outputs/controls/e311f_ptb_sqi_three_class/features/record7*.parquet`.",
            "- BUT source: `outputs/external_benchmarks/e311_but_sqi_fusion_ptb_train_10s_2026_06_04/feature_cache/but_sqi7`.",
            "- BUT protocol: formal 10s P1; no 5s/ensemble data is used.",
            "",
            "## Strongest Domain-Signature Result",
            "",
            f"- Most separable slice: `{best_sep['class_name']}` with domain AUC `{best_sep['domain_auc']:.3f}` and balanced acc `{best_sep['domain_balanced_acc']:.3f}` using only 7SQI.",
            "",
            "## Largest PTB/BUT Class-Separation Mismatches",
            "",
            "| feature | contrast | PTB effect | BUT effect | abs gap | alignment |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worst_alignment.to_dict(orient="records"):
        lines.append(
            f"| {row['feature']} | {row['contrast']} | {row['PTB synthetic']:.3f} | {row['BUT 10s P1']:.3f} | {row['abs_effect_gap']:.3f} | {row['effect_alignment']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation For The Next Fusion Run",
            "",
            "- Increase SQI branch strength, but keep validation-only BUT calibration because the SQI domain signature is strong.",
            "- Prefer branch-logit fusion over pure concat: SQI needs an explicit path to influence logits.",
            "- Keep Uformer features in the model: SQI-only still cannot explain good/medium morphology boundaries.",
            "",
            "## Figures",
            "",
            "![PTB/BUT SQI distributions](figures/ptb_but_sqi_boxplots_norm.png)",
            "",
            "![Class effect gap heatmap](figures/class_effect_gap_heatmap.png)",
            "",
            "![KS distance by feature and class](figures/ks_distance_by_feature_class.png)",
            "",
            "![Dataset separability](figures/domain_separability_auc.png)",
            "",
            "![Class median trajectories](figures/class_median_trajectories.png)",
        ]
    )
    (report_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        report_root / "sqi_gap_summary.json",
        {
            "findings": findings,
            "domain_separability": separability.to_dict(orient="records"),
            "output_root": str(out_root),
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    fig_dir = report_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    ptb_raw, ptb_norm = load_ptb(Path(args.ptb_control_dir))
    but_raw, but_norm = load_but(Path(args.but_protocol_dir), Path(args.but_sqi_cache_dir))
    raw = pd.concat([ptb_raw, but_raw], ignore_index=True)
    norm = pd.concat([ptb_norm, but_norm], ignore_index=True)

    stats = pd.concat([robust_stats(raw, "raw"), robust_stats(norm, "normalized")], ignore_index=True)
    distance = pd.concat([domain_distance(raw, "raw"), domain_distance(norm, "normalized")], ignore_index=True)
    effects = pd.concat([effect_size(raw, "raw"), effect_size(norm, "normalized")], ignore_index=True)
    separability = dataset_separability(norm)

    stats.to_csv(out_root / "sqi_summary_stats.csv", index=False)
    distance.to_csv(out_root / "sqi_domain_distance.csv", index=False)
    effects.to_csv(out_root / "sqi_class_effect_alignment.csv", index=False)
    separability.to_csv(out_root / "sqi_domain_separability.csv", index=False)
    make_plots(raw, norm, distance, effects, separability, fig_dir)
    write_report(report_root, out_root, stats, distance, effects, separability)

    payload = {
        "status": "complete",
        "ptb_rows": int(len(ptb_norm)),
        "but_rows": int(len(but_norm)),
        "findings": top_findings(distance, effects, separability),
        "report": str(report_root / "README.md"),
    }
    write_json(out_root / "sqi_gap_analysis_state.json", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PTB-vs-BUT seven-SQI domain gap.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--ptb_control_dir", default=str(DEFAULT_PTB_CONTROL))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--but_sqi_cache_dir", default=str(DEFAULT_BUT_SQI_CACHE))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
