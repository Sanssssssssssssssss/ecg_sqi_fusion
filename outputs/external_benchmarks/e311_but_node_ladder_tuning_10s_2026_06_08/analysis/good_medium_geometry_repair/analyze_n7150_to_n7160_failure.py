from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path(
    r"E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08"
)
REPORT_ROOT = Path(
    r"E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08"
)
OUT_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"

N7150_VARIANT = "nl_n7150_gm_trim_bad_boundaryblocks_micro_bad_probe_n7125_84064417a3c4"
N7160_VARIANT = "nl_n7160_gm_trim_bad_boundaryblocks_micro_good_pair_n7150_6bd59f8d5ac2"
N7175_VARIANT = "nl_n7175_gm_trim_bad_boundaryblocks_micro_bad_probe_n7150_969a76661658"
N7200_VARIANT = "nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good124_mid172_ec4f54fe7e3d"

FALLBACK_METRICS = {
    "N7150_gm_trim_bad": {
        "node_id": "N7150_gm_trim_bad",
        "node_level": 7150,
        "variant_id": N7150_VARIANT,
        "prediction_mode": "raw",
        "n_total": 18384,
        "acc": 0.9542536988685814,
        "macro_f1": 0.9587565989,
        "good_recall": 0.9588811189,
        "medium_recall": 0.9402797203,
        "bad_recall": 0.9706170421,
        "confusion_3x3": "[[6856, 294, 0], [427, 6723, 0], [0, 120, 3964]]",
        "metric_note": "fallback_from_boundary_block_promotion_summary",
    },
    "N7175_gm_trim_bad": {
        "node_id": "N7175_gm_trim_bad",
        "node_level": 7175,
        "variant_id": N7175_VARIANT,
        "prediction_mode": "medium_guarded_pmed0005",
        "acc": 0.912499,
        "macro_f1": 0.922614,
        "good_recall": 0.973101,
        "medium_recall": 0.818676,
        "bad_recall": 0.970862,
        "confusion_3x3": "",
        "metric_note": "fallback_from_run_summary",
    },
    "N7200_gm_trim_bad": {
        "node_id": "N7200_gm_trim_bad",
        "node_level": 7200,
        "variant_id": N7200_VARIANT,
        "prediction_mode": "medium_guarded_pmed0005",
        "acc": 0.9364315,
        "macro_f1": 0.943622,
        "good_recall": 0.931389,
        "medium_recall": 0.922083,
        "bad_recall": 0.970617,
        "confusion_3x3": "",
        "metric_note": "fallback_from_run_summary",
    },
}

KEY_FEATURES = [
    "pc1",
    "pc3",
    "flatline_ratio",
    "qrs_visibility",
    "non_qrs_diff_p95",
    "qrs_band_ratio",
    "baseline_step",
    "amplitude_entropy",
    "low_amp_ratio",
    "sqi_sSQI",
    "sqi_kSQI",
    "template_corr",
    "detector_agreement",
    "rms",
    "std",
    "ptp_p99_p01",
    "band_15_30",
    "band_30_45",
]

LABEL_FEATURES = [
    "qrs_nprd",
    "tst_nprd",
    "beat_corr",
    "damage_score",
    "core_diagnostic_score",
    "proxy_rms",
    "proxy_basSQI",
    "proxy_kSQI",
    "proxy_sSQI",
    "rr_unacceptable_bins",
    "rr_snr_mean",
    "rr_corr_mean",
    "fatal_subtype_strength",
    "sample_weight",
]

CLASS_COLORS = {"good": "#20744a", "medium": "#bf7a00", "bad": "#9e2a2b"}
BLOCK_COLORS = {
    "gm_good_rescue_pc1flat_qrsvisible": "#2166ac",
    "gm_medium_qrslow_hardneg": "#b2182b",
    "gm_visible_qrs_medium_detail": "#ef8a62",
    "bad_controlled_outlier": "#542788",
    "bad_core_guard": "#7f3b08",
    "gm_clean_overlap_body": "#4d9221",
}


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _parse_confusion(value: object) -> np.ndarray:
    if isinstance(value, str) and value.strip():
        return np.asarray(ast.literal_eval(value), dtype=int)
    return np.zeros((3, 3), dtype=int)


def _copy_to_report(path: Path) -> Path:
    rel = path.relative_to(OUT_DIR)
    target = REPORT_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return target


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    show = show.replace({np.nan: ""})
    cols = list(show.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                text = str(val).replace("|", "\\|").replace("\n", " ")
                vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _read_best_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(OUT_ROOT / "node_ladder_diagnostic_metrics.csv")
    wanted = {
        ("N7150_gm_trim_bad", N7150_VARIANT),
        ("N7160_gm_trim_bad", N7160_VARIANT),
        ("N7175_gm_trim_bad", N7175_VARIANT),
        ("N7200_gm_trim_bad", N7200_VARIANT),
    }
    rows = []
    for node_id, variant_id in wanted:
        subset = metrics[(metrics["node_id"] == node_id) & (metrics["variant_id"] == variant_id)].copy()
        if subset.empty:
            best = dict(FALLBACK_METRICS.get(node_id, {}))
            if not best:
                continue
        else:
            subset["acc_num"] = subset["acc"].map(_safe_float)
            best = subset.sort_values("acc_num", ascending=False).iloc[0].to_dict()
            best.setdefault("metric_note", "diagnostic_metrics_csv")
        conf = _parse_confusion(best.get("confusion_3x3"))
        best["good_to_medium"] = int(conf[0, 1])
        best["medium_to_good"] = int(conf[1, 0])
        best["medium_to_bad"] = int(conf[1, 2])
        best["bad_to_good"] = int(conf[2, 0])
        best["bad_to_medium"] = int(conf[2, 1])
        best["gm_error_total"] = int(conf[0, 1] + conf[1, 0])
        rows.append(best)
    out = pd.DataFrame(rows)
    keep = [
        "node_id",
        "node_level",
        "variant_id",
        "prediction_mode",
        "n_total",
        "acc",
        "macro_f1",
        "good_recall",
        "medium_recall",
        "bad_recall",
        "good_to_medium",
        "medium_to_good",
        "bad_to_good",
        "bad_to_medium",
        "gm_error_total",
        "metric_note",
        "confusion_3x3",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("node_level")


def _metric_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["node_id"] == "N7150_gm_trim_bad"].iloc[0]
    rows = []
    for _, row in metrics.iterrows():
        rec = {"node_id": row["node_id"], "variant_id": row["variant_id"]}
        for col in [
            "acc",
            "macro_f1",
            "good_recall",
            "medium_recall",
            "bad_recall",
            "good_to_medium",
            "medium_to_good",
            "gm_error_total",
        ]:
            rec[f"{col}_delta_vs_n7150"] = _safe_float(row[col]) - _safe_float(base[col])
        rows.append(rec)
    return pd.DataFrame(rows)


def _read_manifest(level: int) -> pd.DataFrame:
    path = OUT_ROOT / "nodes" / f"N{level}_gm_trim_bad" / "node_boundary_manifest.csv"
    cols = ["idx", "split", "class_name", *KEY_FEATURES]
    available = pd.read_csv(path, nrows=1).columns.tolist()
    usecols = [c for c in cols if c in available]
    df = pd.read_csv(path, usecols=usecols)
    for col in KEY_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["level"] = level
    return df


def _feature_gap() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [_read_manifest(7150), _read_manifest(7160)]
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for level in [7150, 7160]:
        one = df[df["level"] == level]
        for class_name in ["good", "medium", "bad"]:
            sub = one[one["class_name"] == class_name]
            for feat in KEY_FEATURES:
                if feat not in sub.columns or sub.empty:
                    continue
                vals = pd.to_numeric(sub[feat], errors="coerce").dropna()
                if vals.empty:
                    continue
                rows.append(
                    {
                        "level": level,
                        "class_name": class_name,
                        "feature": feat,
                        "n": int(vals.shape[0]),
                        "p10": float(vals.quantile(0.10)),
                        "median": float(vals.median()),
                        "p90": float(vals.quantile(0.90)),
                    }
                )
    summary = pd.DataFrame(rows)
    deltas = []
    for feat in KEY_FEATURES:
        pivot = summary[(summary["feature"] == feat) & (summary["class_name"].isin(["good", "medium"]))].pivot(
            index="level", columns="class_name", values="median"
        )
        if not {7150, 7160}.issubset(set(pivot.index)) or not {"good", "medium"}.issubset(set(pivot.columns)):
            continue
        sep7150 = float(pivot.loc[7150, "medium"] - pivot.loc[7150, "good"])
        sep7160 = float(pivot.loc[7160, "medium"] - pivot.loc[7160, "good"])
        deltas.append(
            {
                "feature": feat,
                "gm_median_sep_n7150": sep7150,
                "gm_median_sep_n7160": sep7160,
                "sep_delta_n7160_minus_n7150": sep7160 - sep7150,
                "abs_sep_delta": abs(sep7160 - sep7150),
            }
        )
    return summary, pd.DataFrame(deltas).sort_values("abs_sep_delta", ascending=False)


def _variant_dir(variant_id: str) -> Path:
    return OUT_ROOT / "synthetic_variants" / variant_id / "datasets"


def _read_labels(variant_id: str) -> pd.DataFrame:
    labels = pd.read_csv(_variant_dir(variant_id) / "synth_10s_125hz_labels.csv")
    labels["idx"] = pd.to_numeric(labels["idx"], errors="coerce").astype(int)
    labels["variant_id"] = variant_id
    labels["is_block_append"] = ~labels["blend_source"].fillna("").str.startswith("base_")
    for col in LABEL_FEATURES:
        if col in labels.columns:
            labels[col] = pd.to_numeric(labels[col], errors="coerce")
    return labels


def _block_counts_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for level, variant_id in [(7150, N7150_VARIANT), (7160, N7160_VARIANT)]:
        labels = _read_labels(variant_id)
        labels["level"] = level
        frames.append(labels)
    labels = pd.concat(frames, ignore_index=True)
    block = labels[labels["is_block_append"]].copy()
    count_rows = (
        block.groupby(["level", "variant_id", "blend_source", "y_class", "split"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["level", "blend_source", "y_class", "split"])
    )
    feat_rows = []
    for keys, group in block.groupby(["level", "variant_id", "blend_source", "y_class"], dropna=False):
        rec = {
            "level": keys[0],
            "variant_id": keys[1],
            "block": keys[2],
            "y_class": keys[3],
            "n": int(group.shape[0]),
        }
        for feat in LABEL_FEATURES:
            if feat in group.columns:
                vals = pd.to_numeric(group[feat], errors="coerce").dropna()
                if not vals.empty:
                    rec[f"{feat}_median"] = float(vals.median())
                    rec[f"{feat}_p90"] = float(vals.quantile(0.90))
        feat_rows.append(rec)
    return count_rows, pd.DataFrame(feat_rows).sort_values(["level", "block", "y_class"])


def _plot_pca() -> Path:
    rng = np.random.default_rng(20260612)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, level in zip(axes, [7150, 7160]):
        df = _read_manifest(level)
        gm = df[df["class_name"].isin(["good", "medium"])].copy()
        if gm.shape[0] > 6000:
            gm = gm.iloc[rng.choice(gm.shape[0], size=6000, replace=False)]
        for class_name in ["good", "medium"]:
            sub = gm[gm["class_name"] == class_name]
            ax.scatter(
                sub["pc1"],
                sub["pc3"],
                s=5,
                alpha=0.25,
                label=class_name,
                c=CLASS_COLORS[class_name],
                linewidths=0,
            )
        ax.set_title(f"N{level} good/medium target shell")
        ax.set_xlabel("PC1")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("PC3")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("N7150 -> N7160: good/medium PCA geometry")
    fig.tight_layout()
    path = OUT_DIR / "n7150_to_n7160_failure_pca.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    _copy_to_report(path)
    return path


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data[key])


def _select_waveform_rows(labels: pd.DataFrame) -> pd.DataFrame:
    block = labels[labels["is_block_append"]].copy()
    samples = []
    wanted = [
        "gm_good_rescue_pc1flat_qrsvisible",
        "gm_visible_qrs_medium_detail",
        "gm_medium_qrslow_hardneg",
        "bad_controlled_outlier",
    ]
    for block_name in wanted:
        sub = block[block["blend_source"] == block_name].sort_values("idx")
        if sub.empty:
            continue
        take = sub.head(3)
        samples.append(take)
    if not samples:
        return block.head(12)
    return pd.concat(samples, ignore_index=True)


def _plot_waveforms() -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    time = np.arange(1250) / 125.0
    for row_i, (level, variant_id) in enumerate([(7150, N7150_VARIANT), (7160, N7160_VARIANT)]):
        labels = _read_labels(variant_id)
        selected = _select_waveform_rows(labels)
        clean = _load_npz_array(_variant_dir(variant_id) / "synth_10s_125hz_clean.npz", "X_clean")
        noisy = _load_npz_array(_variant_dir(variant_id) / "synth_10s_125hz_noisy.npz", "X_noisy")
        groups = list(selected.groupby("blend_source", sort=False))
        for col_i in range(4):
            ax = axes[row_i, col_i]
            if col_i >= len(groups):
                ax.axis("off")
                continue
            block_name, group = groups[col_i]
            for _, r in group.iterrows():
                idx = int(r["idx"])
                ax.plot(time, clean[idx], color="#2b8cbe", alpha=0.55, linewidth=0.8)
                ax.plot(time, noisy[idx], color="#d95f0e", alpha=0.45, linewidth=0.8)
            class_text = ",".join(sorted(group["y_class"].astype(str).unique()))
            ax.set_title(f"N{level} {block_name}\n{class_text}, n={len(group)}", fontsize=9)
            ax.grid(alpha=0.12)
            if col_i == 0:
                ax.set_ylabel("mV-ish")
            if row_i == 1:
                ax.set_xlabel("sec")
    handles = [
        plt.Line2D([0], [0], color="#2b8cbe", lw=1.5, label="clean"),
        plt.Line2D([0], [0], color="#d95f0e", lw=1.5, label="noisy"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle("Boundary-block raw waveform samples: clean vs noisy")
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    path = OUT_DIR / "n7150_to_n7160_failure_waveforms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    _copy_to_report(path)
    return path


def _write_report(
    metrics: pd.DataFrame,
    metric_delta: pd.DataFrame,
    feature_summary: pd.DataFrame,
    feature_delta: pd.DataFrame,
    block_counts: pd.DataFrame,
    block_features: pd.DataFrame,
    pca_path: Path,
    waveform_path: Path,
) -> Path:
    n7150 = metrics[metrics["node_id"] == "N7150_gm_trim_bad"].iloc[0]
    n7160 = metrics[metrics["node_id"] == "N7160_gm_trim_bad"].iloc[0]
    medium_loss = int(_safe_float(n7160["medium_to_good"]) - _safe_float(n7150["medium_to_good"]))
    good_gain = int(_safe_float(n7150["good_to_medium"]) - _safe_float(n7160["good_to_medium"]))
    top_features = feature_delta.head(8)
    count_pivot = block_counts.groupby(["level", "blend_source", "y_class"], dropna=False)["n"].sum().reset_index()

    lines = [
        "# N7150 to N7160 Boundary-Block Failure Analysis",
        "",
        "## Executive read",
        "",
        (
            f"N7150 is the current ordinary-checkpoint frontier: acc {float(n7150['acc']):.6f}, "
            f"good/medium/bad {float(n7150['good_recall']):.6f}/{float(n7150['medium_recall']):.6f}/"
            f"{float(n7150['bad_recall']):.6f}."
        ),
        (
            f"N7160 failed because the best pmed0005 mode became good-heavy: acc {float(n7160['acc']):.6f}, "
            f"good/medium/bad {float(n7160['good_recall']):.6f}/{float(n7160['medium_recall']):.6f}/"
            f"{float(n7160['bad_recall']):.6f}."
        ),
        (
            f"The trade was asymmetric: good->medium errors fell by {good_gain}, but medium->good errors rose by "
            f"{medium_loss}. Bad remained stable, so this is a local good/medium geometry failure, not a bad failure."
        ),
        "",
        "## Metrics",
        "",
        _md_table(metrics),
        "",
        "## Delta vs N7150",
        "",
        _md_table(metric_delta),
        "",
        "## Boundary Block Counts",
        "",
        _md_table(count_pivot),
        "",
        "## Top Good/Medium Feature Separation Changes",
        "",
        _md_table(top_features),
        "",
        "## Boundary Block Feature Medians",
        "",
        _md_table(block_features),
        "",
        "## Visuals",
        "",
        f"![N7150 to N7160 PCA]({pca_path.name})",
        "",
        f"![N7150 to N7160 waveforms]({waveform_path.name})",
        "",
        "## Recommended next move",
        "",
        (
            "Do not widen directly from N7150 to N7160 with the current good-pair profile. "
            "The next experiment should be a smaller N7155-style step or a N7160 medium-guarded profile that "
            "reduces the good-rescue fraction/weight and adds only medium rows adjacent to the failing shell. "
            "Bad can stay as controlled-outlier guardrail; the current evidence does not justify broad bad expansion."
        ),
    ]
    path = OUT_DIR / "n7150_to_n7160_failure_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    _copy_to_report(path)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = _read_best_metrics()
    metric_delta = _metric_delta(metrics)
    feature_summary, feature_delta = _feature_gap()
    block_counts, block_features = _block_counts_and_features()

    metrics_path = OUT_DIR / "n7150_to_n7160_metric_summary.csv"
    delta_path = OUT_DIR / "n7150_to_n7160_metric_delta.csv"
    feature_summary_path = OUT_DIR / "n7150_to_n7160_feature_summary.csv"
    feature_delta_path = OUT_DIR / "n7150_to_n7160_feature_gap.csv"
    block_counts_path = OUT_DIR / "n7150_to_n7160_block_counts.csv"
    block_features_path = OUT_DIR / "n7150_to_n7160_block_feature_summary.csv"

    metrics.to_csv(metrics_path, index=False)
    metric_delta.to_csv(delta_path, index=False)
    feature_summary.to_csv(feature_summary_path, index=False)
    feature_delta.to_csv(feature_delta_path, index=False)
    block_counts.to_csv(block_counts_path, index=False)
    block_features.to_csv(block_features_path, index=False)

    for path in [
        metrics_path,
        delta_path,
        feature_summary_path,
        feature_delta_path,
        block_counts_path,
        block_features_path,
    ]:
        _copy_to_report(path)

    pca_path = _plot_pca()
    waveform_path = _plot_waveforms()
    report_path = _write_report(
        metrics,
        metric_delta,
        feature_summary,
        feature_delta,
        block_counts,
        block_features,
        pca_path,
        waveform_path,
    )

    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "frontier": {
            "ordinary_checkpoint": N7150_VARIANT,
            "failed_next_level": N7160_VARIANT,
        },
        "metrics": metrics.to_dict(orient="records"),
        "top_feature_separation_changes": feature_delta.head(10).to_dict(orient="records"),
        "outputs": {
            "report": str(report_path),
            "pca": str(pca_path),
            "waveforms": str(waveform_path),
            "metrics_csv": str(metrics_path),
            "feature_gap_csv": str(feature_delta_path),
            "block_counts_csv": str(block_counts_path),
        },
    }
    summary_path = OUT_DIR / "n7150_to_n7160_failure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _copy_to_report(summary_path)
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
