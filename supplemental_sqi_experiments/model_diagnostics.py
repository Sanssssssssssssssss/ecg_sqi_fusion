from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import (
    SELECTED_FIVE,
    SQIS,
    binary_metrics,
    calibration_summary,
    feature_cols_for_sqis,
    fit_fixed_svm,
    load_split_frame,
    max_accuracy_threshold,
    predict_score,
    source_bootstrap_ci,
    subgroup_rows,
    threshold_curve,
    write_table,
)
from .plotting import PALETTE, add_panel_label, apply_style, plot_model_diagnostics, save_figure


def _read_thresholds(art: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    p = art / "models" / "lm_mlp" / "tables" / "table7_mlp_selected5_seed0.csv"
    if p.exists():
        out["MLP selected-five"] = float(pd.read_csv(p)["threshold"].iloc[0])
    p = art / "models" / "svm" / "table7_svm_selected5_seed0.csv"
    if p.exists():
        out["SVM selected-five"] = float(pd.read_csv(p)["threshold"].iloc[0])
    tab5 = art / "models" / "svm" / "table5_12lead_single_sqi_seed0.csv"
    if tab5.exists():
        t5 = pd.read_csv(tab5)
        for _, row in t5.iterrows():
            out[f"SVM {row['SQI']}"] = float(row.get("threshold", row.get("maxAcc_thr_test", 0.5)))
    return out


def _attach_npz_score(test_df: pd.DataFrame, npz_path: Path, score_col: str) -> None:
    z = np.load(npz_path, allow_pickle=True)
    y_npz = z["y01_test"].astype(np.int32)
    y_df = test_df["y01"].to_numpy(dtype=np.int32)
    if not np.array_equal(y_npz, y_df):
        raise RuntimeError(f"{npz_path} y01_test does not match reconstructed test order")
    test_df[score_col] = z["p_test"].astype(np.float64)


def load_existing_scores(artifacts_dir: str | Path) -> pd.DataFrame:
    art = Path(artifacts_dir)
    df = load_split_frame(art, normalized=True)
    test = df[df["split"].astype(str).eq("test")].copy().reset_index(drop=True)
    _attach_npz_score(test, art / "models" / "lm_mlp" / "probs" / "Selected5_seed0.npz", "mlp_selected5_score")
    _attach_npz_score(test, art / "models" / "svm" / "probs" / "Selected5_seed0.npz", "svm_selected5_score")
    for sqi in SQIS:
        p = art / "models" / "svm" / "probs" / f"{sqi}_seed0.npz"
        if p.exists():
            _attach_npz_score(test, p, f"svm_{sqi}_score")
    return test


def _overall_and_ci(test: pd.DataFrame, score_col: str, model_name: str, threshold: float, n_boot: int, seed: int) -> pd.DataFrame:
    observed = binary_metrics(test["y01"].to_numpy(dtype=int), test[score_col].to_numpy(dtype=float), threshold)
    rows = [{"model": model_name, "metric": k, "estimate": float(observed[k]), "ci_low": np.nan, "ci_high": np.nan} for k in ["Ac", "Se", "Sp", "AUC"]]
    ci = source_bootstrap_ci(test.assign(_score=test[score_col]), "_score", threshold=threshold, n_boot=n_boot, seed=seed)
    for row in rows:
        c = ci[ci["metric"].eq(row["metric"])].iloc[0]
        row["ci_low"] = float(c["ci_low"])
        row["ci_high"] = float(c["ci_high"])
        row["n_bootstrap_valid"] = int(c["n_bootstrap_valid"])
        row["threshold"] = float(threshold)
    return pd.DataFrame(rows)


def _calibration_outputs(test: pd.DataFrame, score_col: str, model_name: str, out_dir: Path) -> dict[str, float]:
    cal, brier = calibration_summary(test["y01"].to_numpy(dtype=int), test[score_col].to_numpy(dtype=float), n_bins=10)
    safe = model_name.lower().replace(" ", "_").replace("-", "_")
    cal["model"] = model_name
    write_table(cal, out_dir / f"calibration_{safe}.csv")
    return {"model": model_name, "brier": float(brier)}


def _svm_validation_threshold_curve(art: Path, out_dir: Path) -> Path:
    df = load_split_frame(art, normalized=True)
    table7 = pd.read_csv(art / "models" / "svm" / "table7_svm_selected5_seed0.csv").iloc[0]
    C = float(table7["best_C"])
    gamma = float(table7["best_gamma"])
    cols = feature_cols_for_sqis(SELECTED_FIVE)
    tr = df["split"].astype(str).eq("train").to_numpy()
    va = df["split"].astype(str).eq("val").to_numpy()
    model = fit_fixed_svm(df.loc[tr, cols].to_numpy(dtype=np.float64), df.loc[tr, "y01"].to_numpy(dtype=int), C=C, gamma=gamma, seed=0)
    p_val = predict_score(model, df.loc[va, cols].to_numpy(dtype=np.float64))
    curve = threshold_curve(df.loc[va, "y01"].to_numpy(dtype=int), p_val, n_grid=501)
    out = out_dir / "validation_threshold_curve_svm_selected5.csv"
    write_table(curve, out)
    return out


def _load_sig125(art: Path, record_id: str) -> np.ndarray:
    z = np.load(art / "resampled_125" / f"{record_id}.npz", allow_pickle=True)
    return z["sig_125"].astype(np.float64)


def _load_qrs_counts(art: Path, record_id: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(art / "qrs" / f"{record_id}.npz", allow_pickle=True)
    r1 = np.array([len(x) for x in z["rpeaks_1"].tolist()], dtype=int)
    r2 = np.array([len(x) for x in z["rpeaks_2"].tolist()], dtype=int)
    return r1, r2


def _plot_case_gallery(
    art: Path,
    row: pd.Series,
    feature_row: pd.Series,
    out_base: Path,
    *,
    score_col: str,
    threshold: float,
) -> list[Path]:
    apply_style()
    sig = _load_sig125(art, str(row["record_id"]))
    t = np.arange(sig.shape[0], dtype=float) / 125.0
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fcols = [f"{lead}__fSQI" for lead in leads]
    fvals = feature_row[fcols].to_numpy(dtype=float)
    try:
        q1, q2 = _load_qrs_counts(art, str(row["record_id"]))
    except Exception:
        q1 = q2 = np.full(12, np.nan)

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 0.85], height_ratios=[1.0, 0.55], wspace=0.34, hspace=0.34)
    ax_wave = fig.add_subplot(gs[:, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, 1])

    y_offsets = np.arange(len(leads))[::-1] * 1.1
    for i, lead in enumerate(leads):
        x = sig[:, i]
        scale = np.nanpercentile(np.abs(x - np.nanmedian(x)), 95)
        scale = max(scale, 1e-6)
        y = (x - np.nanmedian(x)) / scale * 0.34 + y_offsets[i]
        ax_wave.plot(t, y, color=PALETTE["blue"], lw=0.55)
        ax_wave.text(-0.18, y_offsets[i], lead, ha="right", va="center", fontsize=6.8, color=PALETTE["ink"])
    lead_ii = sig[:, 1]
    flat = np.abs(np.diff(lead_ii)) < 1e-4
    flat_t = t[:-1][flat]
    if len(flat_t):
        ax_wave.scatter(flat_t, np.full(len(flat_t), -0.62), s=1.0, color=PALETTE["red"], alpha=0.6, linewidths=0)
    ax_wave.set_xlim(0, 10)
    ax_wave.set_ylim(-0.9, y_offsets[0] + 0.65)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_yticks([])
    add_panel_label(ax_wave, "a", x=-0.08, y=1.01)

    ax_f.bar(np.arange(12), fvals, color=PALETTE["soft_blue"], edgecolor=PALETTE["blue"], linewidth=0.8)
    ax_f.set_xticks(np.arange(12), labels=leads, rotation=90)
    ax_f.set_ylabel("fSQI")
    ax_f.set_ylim(0, max(1.0, float(np.nanmax(fvals)) * 1.05))
    add_panel_label(ax_f, "b", x=-0.16, y=1.02)

    ax_txt.set_axis_off()
    sqi_means = {
        sqi: float(np.nanmean([feature_row[f"{lead}__{sqi}"] for lead in leads]))
        for sqi in ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]
    }
    text = "\n".join(
        [
            f"{row['case_type']}",
            f"record {row['record_id']}",
            f"group {row['sample_group']}",
            f"score {row[score_col]:.3f} / thr {threshold:.3f}",
            f"wqrs/eplimited median {np.nanmedian(q1):.0f}/{np.nanmedian(q2):.0f}",
            "mean SQI " + ", ".join(f"{k} {v:.3g}" for k, v in list(sqi_means.items())[:3]),
            "         " + ", ".join(f"{k} {v:.3g}" for k, v in list(sqi_means.items())[3:]),
        ]
    )
    ax_txt.text(0.0, 0.98, text, ha="left", va="top", fontsize=7.2, color=PALETTE["ink"], linespacing=1.35)
    return save_figure(fig, out_base)


def make_high_confidence_gallery(
    art: Path,
    test: pd.DataFrame,
    out_dir: Path,
    report_dir: Path,
    *,
    score_col: str = "svm_selected5_score",
    threshold: float,
    n_per_case: int = 2,
) -> list[str]:
    feat = pd.read_parquet(art / "features" / "record84.parquet")
    feat["record_id"] = feat["record_id"].astype(str)
    feat = feat.set_index("record_id", drop=False)
    pred_accept = test[score_col].to_numpy(dtype=float) > float(threshold)
    tmp = test.copy()
    tmp["pred_accept"] = pred_accept
    cases = [
        ("original_poor_false_acceptance", tmp["sample_group"].eq("original unacceptable") & tmp["pred_accept"], False),
        ("synthetic_false_acceptance", tmp["sample_group"].isin(["synthetic em", "synthetic ma"]) & tmp["pred_accept"], False),
        ("acceptable_false_rejection", tmp["sample_group"].eq("original acceptable") & (~tmp["pred_accept"]), True),
        ("correct_acceptable_control", tmp["sample_group"].eq("original acceptable") & tmp["pred_accept"], True),
        ("correct_poor_control", tmp["y01"].eq(0) & (~tmp["pred_accept"]), False),
    ]
    rows = []
    for case_type, mask, high_score in cases:
        sub = tmp[mask].copy()
        already_ranked = False
        if sub.empty and case_type == "synthetic_false_acceptance":
            case_type = "synthetic_nearest_threshold_control"
            sub = tmp[tmp["sample_group"].isin(["synthetic em", "synthetic ma"])].copy()
            sub["_distance_to_threshold"] = np.abs(sub[score_col].to_numpy(dtype=float) - float(threshold))
            sub = sub.sort_values("_distance_to_threshold", ascending=True)
            already_ranked = True
        if not already_ranked:
            sub = sub.sort_values(score_col, ascending=not high_score)
        sub = sub.head(n_per_case)
        for _, r in sub.iterrows():
            rows.append(
                {
                    "case_type": case_type,
                    "record_id": str(r["record_id"]),
                    "source_record_id": str(r["source_record_id"]),
                    "sample_group": str(r["sample_group"]),
                    "y": int(r["y"]),
                    "y01": int(r["y01"]),
                    "score": float(r[score_col]),
                    "threshold": float(threshold),
                    "pred_accept": bool(r["pred_accept"]),
                    "mlp_selected5_score": float(r["mlp_selected5_score"]),
                    "svm_selected5_score": float(r["svm_selected5_score"]),
                }
            )
    cand = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = report_dir / "high_confidence_gallery"
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    cand_csv = out_dir / "high_confidence_error_gallery_candidates.csv"
    write_table(cand, cand_csv, md_path=report_dir / "high_confidence_error_gallery_candidates.md")
    outputs = [str(cand_csv)]
    plot_rows = pd.DataFrame(rows)
    # Rejoin full test rows for plotting, while keeping the saved candidate CSV compact.
    plot_rows = plot_rows.merge(tmp, on=["record_id", "source_record_id", "sample_group", "y", "y01"], how="left", suffixes=("", "_full"))
    for i, r in plot_rows.iterrows():
        rid = str(r["record_id"])
        if rid not in feat.index:
            continue
        safe = f"{i:02d}_{r['case_type']}_{rid}".replace("__", "_")
        outputs.extend(
            str(p)
            for p in _plot_case_gallery(
                art,
                r,
                feat.loc[rid],
                gallery_dir / safe,
                score_col=score_col,
                threshold=threshold,
            )
        )
    return outputs


def run_model_diagnostics(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    art = Path(artifacts_dir)
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    test = load_existing_scores(art)
    thresholds = _read_thresholds(art)

    test_csv = out / "test_scores_with_source_groups.csv"
    write_table(test, test_csv)

    summary_parts = []
    subgroup = []
    for model_name, score_col in [
        ("MLP selected-five", "mlp_selected5_score"),
        ("SVM selected-five", "svm_selected5_score"),
    ]:
        threshold = thresholds[model_name]
        summary_parts.append(_overall_and_ci(test, score_col, model_name, threshold, n_boot, seed))
        subgroup.extend(subgroup_rows(test, score_col, threshold=threshold, model_name=model_name))
    for sqi in SQIS:
        score_col = f"svm_{sqi}_score"
        model_name = f"SVM {sqi}"
        if score_col in test and model_name in thresholds:
            subgroup.extend(subgroup_rows(test, score_col, threshold=thresholds[model_name], model_name=model_name))

    summary = pd.concat(summary_parts, ignore_index=True)
    summary_csv = out / "selected5_source_bootstrap_metrics.csv"
    write_table(summary, summary_csv, md_path=rep / "selected5_source_bootstrap_metrics.md")
    subgroup_df = pd.DataFrame(subgroup)
    subgroup_csv = out / "stratified_score_summary.csv"
    write_table(subgroup_df, subgroup_csv, md_path=rep / "stratified_score_summary.md")

    briers = [
        _calibration_outputs(test, "mlp_selected5_score", "MLP selected-five", out),
        _calibration_outputs(test, "svm_selected5_score", "SVM selected-five", out),
    ]
    brier_csv = out / "brier_scores.csv"
    write_table(pd.DataFrame(briers), brier_csv, md_path=rep / "brier_scores.md")
    threshold_curve_csv = _svm_validation_threshold_curve(art, out)
    plot_paths = plot_model_diagnostics(test, summary, rep / "fig_supp_02_model_stratified_diagnostics")
    gallery_outputs = make_high_confidence_gallery(
        art,
        test,
        out / "error_gallery",
        rep / "error_gallery",
        threshold=thresholds["SVM selected-five"],
    )
    return {
        "outputs": [
            str(test_csv),
            str(summary_csv),
            str(subgroup_csv),
            str(brier_csv),
            str(threshold_curve_csv),
            *[str(p) for p in plot_paths],
            *gallery_outputs,
        ],
        "thresholds": thresholds,
    }
