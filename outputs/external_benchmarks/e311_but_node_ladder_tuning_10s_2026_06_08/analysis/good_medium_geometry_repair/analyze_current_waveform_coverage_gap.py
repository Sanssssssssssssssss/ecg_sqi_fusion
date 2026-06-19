"""Coverage-gap audit for the current best waveform-only student.

External-only analysis.  This script compares current original-test mistakes
against the PTB synthetic feature distribution used to train the waveform
student.  Original BUT is report-only: it is only analyzed here and is not used
for training, threshold selection, or checkpoint selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
SYNTH_ID = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
CURRENT_CANDIDATE = "featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050"
CLASS_ORDER = ["good", "medium", "bad"]


def robust_stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"q01": np.nan, "q05": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "q95": np.nan, "q99": np.nan, "iqr": np.nan}
    q01, q05, q25, med, q75, q95, q99 = np.nanpercentile(values, [1, 5, 25, 50, 75, 95, 99])
    iqr = max(float(q75 - q25), 1e-6)
    return {"q01": float(q01), "q05": float(q05), "q25": float(q25), "median": float(med), "q75": float(q75), "q95": float(q95), "q99": float(q99), "iqr": iqr}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    synth_dir = OUT_ROOT / "synthetic_variants" / SYNTH_ID / "datasets"
    schema = json.loads((synth_dir / "tabular_feature_schema.json").read_text())
    feature_cols = list(schema["feature_columns"])

    synth_npz = np.load(synth_dir / "tabular_features.npz")
    synth_features = pd.DataFrame(synth_npz["features"], columns=feature_cols)
    synth_labels = pd.read_csv(synth_dir / "synth_10s_125hz_labels.csv", usecols=["idx", "split", "y_class", "label_subtype", "sample_source", "blend_source", "sample_weight"])
    synth = pd.concat([synth_labels.reset_index(drop=True), synth_features.reset_index(drop=True)], axis=1)

    pred_path = REPORT_DIR / "original_candidate_error_audit" / CURRENT_CANDIDATE / "original_predictions.csv"
    pred = pd.read_csv(pred_path)
    atlas = pd.read_csv(OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv")
    atlas = atlas.reset_index(drop=True)
    atlas.insert(0, "row", np.arange(len(atlas), dtype=np.int64))
    original = pred.merge(atlas, on="row", how="left", suffixes=("_pred", "_atlas"))
    if "split_pred" in original.columns:
        original["split"] = original["split_pred"]
    if original[feature_cols].isna().any().any():
        missing = int(original[feature_cols].isna().any(axis=1).sum())
        raise RuntimeError(f"missing atlas features after row merge: {missing} rows")
    return synth, original, atlas, feature_cols


def build_support_tables(synth: pd.DataFrame, original_test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows: list[dict[str, object]] = []
    for cls in CLASS_ORDER:
        syn_cls = synth[(synth["split"] == "train") & (synth["y_class"] == cls)]
        orig_cls = original_test[original_test["true"] == cls]
        for feature in feature_cols:
            s = robust_stats(syn_cls[feature].to_numpy(dtype=float))
            stats_rows.append({"class_name": cls, "feature": feature, **s})
    stats = pd.DataFrame(stats_rows)

    bucket_defs = {
        "all_original_test": np.ones(len(original_test), dtype=bool),
        "correct_good": (original_test["true"] == "good") & (original_test["pred_badcal"] == "good"),
        "correct_medium": (original_test["true"] == "medium") & (original_test["pred_badcal"] == "medium"),
        "correct_bad": (original_test["true"] == "bad") & (original_test["pred_badcal"] == "bad"),
        "good_to_medium": (original_test["true"] == "good") & (original_test["pred_badcal"] == "medium"),
        "medium_to_good": (original_test["true"] == "medium") & (original_test["pred_badcal"] == "good"),
        "medium_to_bad": (original_test["true"] == "medium") & (original_test["pred_badcal"] == "bad"),
        "bad_to_good": (original_test["true"] == "bad") & (original_test["pred_badcal"] == "good"),
        "bad_to_medium": (original_test["true"] == "bad") & (original_test["pred_badcal"] == "medium"),
        "nonbad_to_bad": (original_test["true"].isin(["good", "medium"])) & (original_test["pred_badcal"] == "bad"),
    }

    support_rows: list[dict[str, object]] = []
    for bucket, mask in bucket_defs.items():
        rows = original_test[mask]
        if rows.empty:
            continue
        for cls in CLASS_ORDER:
            cls_rows = rows[rows["true"] == cls]
            if cls_rows.empty:
                continue
            stat_cls = stats[stats["class_name"] == cls].set_index("feature")
            correct_bucket = f"correct_{cls}"
            correct_rows = original_test[bucket_defs.get(correct_bucket, np.zeros(len(original_test), dtype=bool))]
            for feature in feature_cols:
                st = stat_cls.loc[feature]
                vals = cls_rows[feature].to_numpy(dtype=float)
                correct_vals = correct_rows[feature].to_numpy(dtype=float) if not correct_rows.empty else np.array([])
                low01 = vals < float(st["q01"])
                high99 = vals > float(st["q99"])
                low05 = vals < float(st["q05"])
                high95 = vals > float(st["q95"])
                med = float(np.nanmedian(vals))
                correct_med = float(np.nanmedian(correct_vals)) if correct_vals.size else np.nan
                support_rows.append(
                    {
                        "bucket": bucket,
                        "true": cls,
                        "feature": feature,
                        "n": int(len(vals)),
                        "bucket_median": med,
                        "correct_sameclass_median": correct_med,
                        "synthetic_train_median": float(st["median"]),
                        "delta_vs_synth_iqr": (med - float(st["median"])) / float(st["iqr"]),
                        "delta_vs_correct_iqr": (med - correct_med) / float(st["iqr"]) if np.isfinite(correct_med) else np.nan,
                        "oos_01_99_rate": float((low01 | high99).mean()),
                        "oos_05_95_rate": float((low05 | high95).mean()),
                        "below_q01_rate": float(low01.mean()),
                        "above_q99_rate": float(high99.mean()),
                        "below_q05_rate": float(low05.mean()),
                        "above_q95_rate": float(high95.mean()),
                    }
                )
    return stats, pd.DataFrame(support_rows)


def nearest_support_distances(synth: pd.DataFrame, original_test: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute same-class nearest synthetic-train distance for original-test rows."""
    out_frames: list[pd.DataFrame] = []
    eps = 1e-6
    for cls in CLASS_ORDER:
        syn_cls = synth[(synth["split"] == "train") & (synth["y_class"] == cls)]
        orig_idx = original_test.index[original_test["true"] == cls].to_numpy()
        if syn_cls.empty or orig_idx.size == 0:
            continue
        syn = syn_cls[feature_cols].to_numpy(dtype=np.float32)
        med = np.nanmedian(syn, axis=0).astype(np.float32)
        q25 = np.nanpercentile(syn, 25, axis=0).astype(np.float32)
        q75 = np.nanpercentile(syn, 75, axis=0).astype(np.float32)
        scale = np.maximum(q75 - q25, eps).astype(np.float32)
        syn_z = np.nan_to_num((syn - med) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        orig = original_test.loc[orig_idx, feature_cols].to_numpy(dtype=np.float32)
        orig_z = np.nan_to_num((orig - med) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mins = np.full(orig_z.shape[0], np.inf, dtype=np.float32)
        chunk = 256
        for start in range(0, orig_z.shape[0], chunk):
            part = orig_z[start : start + chunk]
            # squared Euclidean in robustly standardized feature space
            d = ((part[:, None, :] - syn_z[None, :, :]) ** 2).sum(axis=2)
            mins[start : start + chunk] = np.sqrt(d.min(axis=1))
        frame = original_test.loc[orig_idx, ["row", "split", "region", "true", "pred_raw", "pred_badcal", "prob_good", "prob_medium", "prob_bad", "is_test", "is_bad_core", "is_bad_outlier"]].copy()
        frame["sameclass_synth_nn_dist47"] = mins
        out_frames.append(frame)
    rows = pd.concat(out_frames, ignore_index=True)
    rows["bucket"] = "correct"
    rows.loc[(rows["true"] == "good") & (rows["pred_badcal"] == "medium"), "bucket"] = "good_to_medium"
    rows.loc[(rows["true"] == "medium") & (rows["pred_badcal"] == "good"), "bucket"] = "medium_to_good"
    rows.loc[(rows["true"] == "medium") & (rows["pred_badcal"] == "bad"), "bucket"] = "medium_to_bad"
    rows.loc[(rows["true"] == "bad") & (rows["pred_badcal"] == "good"), "bucket"] = "bad_to_good"
    rows.loc[(rows["true"] == "bad") & (rows["pred_badcal"] == "medium"), "bucket"] = "bad_to_medium"
    rows.loc[(rows["true"].isin(["good", "medium"])) & (rows["pred_badcal"] == "bad"), "bucket"] = "nonbad_to_bad"
    return rows


def summarize_row_oos(original_test: pd.DataFrame, stats: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    stat_lookup = {
        (r.class_name, r.feature): (float(r.q01), float(r.q99), float(r.q05), float(r.q95))
        for r in stats.itertuples(index=False)
    }
    rows = []
    for row in original_test.itertuples(index=False):
        cls = getattr(row, "true")
        oos01 = []
        oos05 = []
        for feature in feature_cols:
            val = float(getattr(row, feature))
            q01, q99, q05, q95 = stat_lookup[(cls, feature)]
            if val < q01 or val > q99:
                side = "low" if val < q01 else "high"
                oos01.append(f"{feature}:{side}")
            if val < q05 or val > q95:
                side = "low" if val < q05 else "high"
                oos05.append(f"{feature}:{side}")
        rows.append(
            {
                "row": int(getattr(row, "row")),
                "split": getattr(row, "split"),
                "region": getattr(row, "region"),
                "true": cls,
                "pred_raw": getattr(row, "pred_raw"),
                "pred_badcal": getattr(row, "pred_badcal"),
                "prob_good": float(getattr(row, "prob_good")),
                "prob_medium": float(getattr(row, "prob_medium")),
                "prob_bad": float(getattr(row, "prob_bad")),
                "is_bad_core": bool(getattr(row, "is_bad_core")),
                "is_bad_outlier": bool(getattr(row, "is_bad_outlier")),
                "oos_01_99_count": len(oos01),
                "oos_05_95_count": len(oos05),
                "top_oos_01_99": ";".join(oos01[:12]),
                "top_oos_05_95": ";".join(oos05[:12]),
            }
        )
    out = pd.DataFrame(rows)
    out["bucket"] = "correct"
    out.loc[(out["true"] == "good") & (out["pred_badcal"] == "medium"), "bucket"] = "good_to_medium"
    out.loc[(out["true"] == "medium") & (out["pred_badcal"] == "good"), "bucket"] = "medium_to_good"
    out.loc[(out["true"] == "medium") & (out["pred_badcal"] == "bad"), "bucket"] = "medium_to_bad"
    out.loc[(out["true"] == "bad") & (out["pred_badcal"] == "good"), "bucket"] = "bad_to_good"
    out.loc[(out["true"] == "bad") & (out["pred_badcal"] == "medium"), "bucket"] = "bad_to_medium"
    out.loc[(out["true"].isin(["good", "medium"])) & (out["pred_badcal"] == "bad"), "bucket"] = "nonbad_to_bad"
    return out


def write_markdown(out_dir: Path, support: pd.DataFrame, row_support: pd.DataFrame, nn_rows: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        shown = df.copy()
        for col in shown.columns:
            if pd.api.types.is_float_dtype(shown[col]):
                shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        headers = [str(c) for c in shown.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in shown.astype(str).itertuples(index=False):
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    bucket_counts = row_support.groupby(["bucket", "true", "pred_badcal"], dropna=False).size().reset_index(name="n").sort_values(["bucket", "true", "pred_badcal"])
    bucket_summary = row_support.groupby(["bucket", "true"], dropna=False).agg(
        n=("row", "size"),
        oos01_mean=("oos_01_99_count", "mean"),
        oos05_mean=("oos_05_95_count", "mean"),
    ).reset_index()
    nn_summary = nn_rows.groupby(["bucket", "true"], dropna=False).agg(
        n=("row", "size"),
        nn_median=("sameclass_synth_nn_dist47", "median"),
        nn_p90=("sameclass_synth_nn_dist47", lambda s: float(np.nanpercentile(s, 90))),
    ).reset_index()

    priority_buckets = ["good_to_medium", "medium_to_good", "medium_to_bad", "bad_to_good", "bad_to_medium", "nonbad_to_bad"]
    top = support[support["bucket"].isin(priority_buckets)].copy()
    top["gap_score"] = top["oos_01_99_rate"] * 3.0 + top["oos_05_95_rate"] + top["delta_vs_correct_iqr"].abs().fillna(0.0) * 0.25
    top = top.sort_values(["bucket", "gap_score"], ascending=[True, False]).groupby("bucket").head(8)

    lines = [
        "# Current Waveform Coverage Gap",
        "",
        f"Candidate: `{CURRENT_CANDIDATE}`",
        "",
        "This is report-only analysis of original BUT rows. It is not used for training or selection.",
        "",
        "## Bucket counts",
        "",
        md_table(bucket_counts),
        "",
        "## Row support summary",
        "",
        md_table(bucket_summary),
        "",
        "## Same-class PTB synthetic nearest-neighbor distance",
        "",
        md_table(nn_summary),
        "",
        "## Top feature support gaps by error bucket",
        "",
        md_table(top[[
            "bucket",
            "true",
            "feature",
            "n",
            "oos_01_99_rate",
            "oos_05_95_rate",
            "delta_vs_synth_iqr",
            "delta_vs_correct_iqr",
            "below_q01_rate",
            "above_q99_rate",
        ]]),
        "",
    ]
    (out_dir / "current_waveform_coverage_gap.md").write_text("\n".join(lines), encoding="utf-8")
    bucket_counts.to_csv(out_dir / "bucket_counts.csv", index=False)
    bucket_summary.to_csv(out_dir / "bucket_support_summary.csv", index=False)
    nn_summary.to_csv(out_dir / "nearest_synthetic_support_summary.csv", index=False)
    top.to_csv(out_dir / "top_feature_support_gaps.csv", index=False)


def main() -> None:
    out_dir = REPORT_DIR / "current_waveform_coverage_gap"
    out_dir.mkdir(parents=True, exist_ok=True)
    synth, original, _atlas, feature_cols = load_inputs()
    original_test = original[original["is_test"] == True].copy()  # noqa: E712

    stats, support = build_support_tables(synth, original_test, feature_cols)
    row_support = summarize_row_oos(original_test, stats, feature_cols)
    nn_rows = nearest_support_distances(synth, original_test, feature_cols)
    row_support = row_support.merge(nn_rows[["row", "sameclass_synth_nn_dist47"]], on="row", how="left")

    stats.to_csv(out_dir / "synthetic_train_feature_support.csv", index=False)
    support.to_csv(out_dir / "bucket_feature_support.csv", index=False)
    row_support.to_csv(out_dir / "row_support_original_test.csv", index=False)
    nn_rows.to_csv(out_dir / "nearest_synthetic_support_rows.csv", index=False)
    write_markdown(out_dir, support, row_support, nn_rows)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
