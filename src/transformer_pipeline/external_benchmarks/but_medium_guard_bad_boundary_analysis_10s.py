"""BUT 10s medium-guard / bad-boundary diagnostic analysis.

This is an experiment-only analysis package.  It keeps the formal BUT 10s P1
test intact, adds prediction-independent balanced diagnostic subsets, and
joins model probabilities with SQI/morphology features so the next generator
grid can target the medium-vs-bad boundary without test leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import real_noise_snr_but_match_grid_10s as rn
from src.transformer_pipeline.external_benchmarks.analyze_but_morphology_clusters import (
    FEATURE_COLUMNS,
    extract_features,
)
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    balanced_but_test_indices,
    now_iso,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.but_sqi_fusion_ptb_train import (
    SQI_COLUMNS,
    sqi_for_signal,
)
from src.transformer_pipeline.external_benchmarks.run import (
    INT_TO_CLASS,
    apply_but_thresholds,
    calibrate_but,
    load_uformer_model,
    multiclass_report,
    run_model_outputs,
)


ROOT = rn.ROOT
RUN_TAG = "e311_but_medium_guard_bad_boundary_analysis_10s_2026_06_05"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_BUT_PROTOCOL = PROTOCOL_OUT_ROOT / "protocols" / "p1_current_10s_center"
SENSORS_OUT = ROOT / "outputs" / "external_benchmarks" / "e311_sensors2025_noise_synthesis_but_10s_2026_06_05"
H_BAD_SUMMARY = ROOT / "outputs" / "external_benchmarks" / "e311_but_morphology_guided_grid_10s_2026_06_03" / "morphology_guided_summary.jsonl"
STATE_NAME = "medium_guard_bad_boundary_analysis_state.json"
CLASS_NAMES = ("good", "medium", "bad")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def safe_id(text: str) -> str:
    return (
        text.lower()
        .replace("+", "_")
        .replace("-", "m")
        .replace(".", "p")
        .replace(",", "_")
        .replace(" ", "_")
        .replace("\\", "_")
        .replace("/", "_")
    )


def locate_anchor_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sensors_summary = SENSORS_OUT / "sensors2025_training_summary.jsonl"
    sensors = load_jsonl(sensors_summary)
    wanted = {
        "paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p40_1p70",
        "paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p55_1p70",
        "paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70",
        "paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70",
        "paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70",
        "paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70",
    }
    best_by_id: dict[str, dict[str, Any]] = {}
    for row in sensors:
        sid = str(row.get("spec", {}).get("id", ""))
        if sid not in wanted or int(row.get("returncode", 1)) != 0:
            continue
        # Prefer full rows when present, otherwise quick.
        old = best_by_id.get(sid)
        if old is None or (old.get("mode") != "full" and row.get("mode") == "full"):
            best_by_id[sid] = row
    for row in best_by_id.values():
        row = dict(row)
        row["anchor_group"] = "sensors2025"
        rows.append(row)

    for row in load_jsonl(H_BAD_SUMMARY):
        sid = str(row.get("spec", {}).get("id", ""))
        if sid == "h_bad_rescue_05" and int(row.get("returncode", 1)) == 0:
            row = dict(row)
            row["anchor_group"] = "history_h_bad_rescue"
            rows.append(row)
            break
    return rows


def load_but(protocol_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(protocol_dir / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(protocol_dir / "metadata.csv")
    if X.ndim == 2:
        X = X[:, None, :]
    if "class_name" not in meta.columns:
        meta["class_name"] = meta["y_class"].astype(str)
    if "y" not in meta.columns:
        meta["y"] = meta["y_class"].map({"good": 0, "medium": 1, "bad": 2}).astype(int)
    return X, meta


def compute_sqi_frame(X: np.ndarray, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows: list[list[float]] = []
    for i in range(len(X)):
        try:
            rows.append([float(v) for v in sqi_for_signal(X[i, 0])])
        except Exception:
            rows.append([0.0] * len(SQI_COLUMNS))
        if (i + 1) % 5000 == 0:
            print(f"SQI {i + 1}/{len(X)}", flush=True)
    df = pd.DataFrame(rows, columns=SQI_COLUMNS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def stratified_balanced_indices(meta: pd.DataFrame, features: pd.DataFrame, seed: int = 20260605) -> tuple[np.ndarray, dict[str, Any]]:
    test_idx = meta.index[meta["split"].astype(str) == "test"].to_numpy()
    test_meta = meta.loc[test_idx].copy()
    n_per_class = int(test_meta["y"].value_counts().min())
    key_features = [
        c
        for c in [
            "rms",
            "qrs_prominence",
            "qrs_like_peak_density",
            "baseline_drift_proxy",
            "hf_proxy",
            "flatline_ratio",
            "contact_loss_proxy",
            "low_amp_ratio",
        ]
        if c in features.columns
    ]
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    strata_audit: dict[str, Any] = {}
    feat_test = features.loc[test_idx, key_features].copy() if key_features else pd.DataFrame(index=test_idx)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_idx = test_meta.index[test_meta["y"].astype(int) == cls_id].to_numpy()
        if len(cls_idx) <= n_per_class:
            chosen = cls_idx
        elif key_features:
            # Build a compact morphology stratum key from tertiles of the most stable features.
            sub = feat_test.loc[cls_idx].replace([np.inf, -np.inf], np.nan).fillna(feat_test.median(numeric_only=True))
            keys = []
            for col in key_features[:6]:
                try:
                    q = pd.qcut(sub[col].rank(method="first"), q=3, labels=False, duplicates="drop")
                except Exception:
                    q = pd.Series(np.zeros(len(sub), dtype=int), index=sub.index)
                keys.append(q.astype(int).astype(str))
            stratum = keys[0]
            for k in keys[1:]:
                stratum = stratum + "_" + k
            buckets = pd.DataFrame({"idx": cls_idx, "stratum": stratum.to_numpy()})
            chosen_list: list[int] = []
            per_bucket = max(1, int(math.ceil(n_per_class / max(1, buckets["stratum"].nunique()))))
            for _, group in buckets.groupby("stratum"):
                ids = group["idx"].to_numpy()
                take = min(per_bucket, len(ids))
                chosen_list.extend(rng.choice(ids, size=take, replace=False).tolist())
            if len(chosen_list) < n_per_class:
                remaining = np.asarray([v for v in cls_idx if v not in set(chosen_list)], dtype=int)
                if len(remaining):
                    chosen_list.extend(rng.choice(remaining, size=min(n_per_class - len(chosen_list), len(remaining)), replace=False).tolist())
            chosen = np.asarray(chosen_list[:n_per_class], dtype=int)
        else:
            chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        selected.extend([int(v) for v in chosen])
        strata_audit[cls_name] = {"available": int(len(cls_idx)), "selected": int(len(chosen))}
    selected_arr = np.asarray(sorted(selected), dtype=int)
    manifest = {
        "name": "but_10s_p1_stratified_balanced_test",
        "seed": int(seed),
        "source_split": "test",
        "source_counts_good_medium_bad": [int((test_meta["y"] == i).sum()) for i in range(3)],
        "n_per_class": int(n_per_class),
        "selected_counts_good_medium_bad": [int((meta.loc[selected_arr, "y"].astype(int) == i).sum()) for i in range(3)],
        "features_used": key_features,
        "strata": strata_audit,
        "note": "Prediction-independent diagnostic subset; selected only from labels and morphology/SQI features.",
    }
    return selected_arr, manifest


def classify_errors(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.full(len(y), "correct", dtype=object)
    out[(y == 2) & (pred != 2)] = "bad_missed"
    out[(y == 2) & (pred == 2)] = "correct_bad"
    out[(y == 1) & (pred == 0)] = "medium_to_good"
    out[(y == 1) & (pred == 2)] = "medium_to_bad"
    out[(y == 0) & (pred == 2)] = "good_false_bad"
    out[(y == 0) & (pred == 1)] = "good_to_medium"
    return out


def infer_failure_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["low_amp_ratio", "flatline_ratio", "contact_loss_proxy", "baseline_drift_proxy", "hf_proxy", "qrs_prominence", "qrs_like_peak_density"]:
        if col not in out:
            out[col] = 0.0
    out["tag_low_amp_flat_contact"] = (
        (out["low_amp_ratio"] > out["low_amp_ratio"].quantile(0.75))
        | (out["flatline_ratio"] > out["flatline_ratio"].quantile(0.75))
        | (out["contact_loss_proxy"] > out["contact_loss_proxy"].quantile(0.75))
    )
    out["tag_baseline_platform_or_jump"] = out["baseline_drift_proxy"] > out["baseline_drift_proxy"].quantile(0.75)
    out["tag_qrs_confounding_noise"] = (
        (out["qrs_like_peak_density"] > out["qrs_like_peak_density"].quantile(0.75))
        | (out["qrs_prominence"] < out["qrs_prominence"].quantile(0.25))
    )
    out["tag_hf_motion_burst"] = out["hf_proxy"] > out["hf_proxy"].quantile(0.75)
    out["tag_visible_qrs_but_unusable"] = (
        (out["qrs_prominence"] >= out["qrs_prominence"].quantile(0.50))
        & ((out["baseline_drift_proxy"] > out["baseline_drift_proxy"].quantile(0.60)) | (out["hf_proxy"] > out["hf_proxy"].quantile(0.60)))
    )
    return out


def plot_wave_cases(
    X: np.ndarray,
    df: pd.DataFrame,
    case: str,
    out_png: Path,
    *,
    max_rows: int = 18,
) -> None:
    rows = df[df["error_type"] == case].head(max_rows)
    if rows.empty:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    cols = 3
    nrows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(15, max(3, 2.1 * nrows)), squeeze=False)
    t = np.arange(X.shape[-1]) / 125.0
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows.itertuples()):
        idx = int(row.idx)
        ax.plot(t, X[idx, 0], lw=0.75, color="#263238")
        ax.set_title(
            f"{row.y_class} -> {row.pred_calibrated_class} idx={idx}\n"
            f"Pg/M/B={row.p_good:.2f}/{row.p_medium:.2f}/{row.p_bad:.2f}",
            fontsize=8,
        )
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.18)
    fig.suptitle(case)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_class_profiles(X: np.ndarray, df: pd.DataFrame, out_dir: Path) -> None:
    for cls in CLASS_NAMES:
        rows = df[(df["split"] == "test") & (df["y_class"] == cls)].head(18)
        if rows.empty:
            continue
        tmp = rows.copy()
        tmp["error_type"] = cls
        plot_wave_cases(X, tmp, cls, out_dir / f"but_{cls}_profile.png")


def plot_embeddings(df: pd.DataFrame, out_dir: Path, prefix: str, cols: list[str]) -> None:
    use_cols = [c for c in cols if c in df.columns]
    if len(use_cols) < 2:
        return
    sample = df[df["split"] == "test"].copy()
    if len(sample) > 8000:
        sample = sample.sample(8000, random_state=7)
    X = sample[use_cols].replace([np.inf, -np.inf], np.nan).fillna(sample[use_cols].median(numeric_only=True)).to_numpy(dtype=np.float32)
    Z = StandardScaler().fit_transform(X)
    emb = PCA(n_components=2, random_state=7).fit_transform(Z)
    sample["pc1"] = emb[:, 0]
    sample["pc2"] = emb[:, 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    sample[["idx", "y_class", "pred_calibrated_class", "error_type", "pc1", "pc2", *use_cols]].to_csv(out_dir / f"{prefix}_embedding_sample.csv", index=False)
    for hue in ["y_class", "error_type"]:
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        groups = sample.groupby(hue)
        for name, group in groups:
            ax.scatter(group["pc1"], group["pc2"], s=9, alpha=0.55, label=str(name))
        ax.set_title(f"{prefix} PCA by {hue}")
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_pca_by_{hue}.png", dpi=180)
        plt.close(fig)


def plot_probability(df: pd.DataFrame, out_dir: Path) -> None:
    sample = df[df["split"] == "test"].copy()
    if len(sample) > 10000:
        sample = sample.sample(10000, random_state=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for cls in CLASS_NAMES:
        sub = sample[sample["y_class"] == cls]
        ax.scatter(sub["p_medium"], sub["p_bad"], s=8, alpha=0.5, label=cls)
    ax.set_xlabel("P(medium)")
    ax.set_ylabel("P(bad)")
    ax.set_title("BUT test probability geometry")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "probability_pmedium_pbad.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(sample["margin_bad_medium"], bins=80, color="#334155", alpha=0.82)
    ax.axvline(0, color="#dc2626", lw=1)
    ax.set_title("P(bad) - P(medium) margin")
    ax.set_xlabel("margin")
    fig.tight_layout()
    fig.savefig(out_dir / "margin_bad_medium_hist.png", dpi=180)
    plt.close(fig)


def feature_importance(df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    badish = test[test["y"].isin([1, 2])].copy()
    features = [c for c in [*FEATURE_COLUMNS, *SQI_COLUMNS, "margin_bad_medium", "entropy"] if c in badish.columns]
    if len(badish) < 200 or len(features) < 2:
        return pd.DataFrame()
    X = badish[features].replace([np.inf, -np.inf], np.nan).fillna(badish[features].median(numeric_only=True)).to_numpy(dtype=np.float32)
    y = (badish["y"].to_numpy(dtype=int) == 2).astype(int)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.30, random_state=13, stratify=y)
    clf = RandomForestClassifier(n_estimators=220, max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=13, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    imp = pd.DataFrame({"feature": features, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    payload = {
        "task": "bad_vs_medium_on_BUT_test",
        "confusion_2x2": confusion_matrix(y[te], pred, labels=[0, 1]).tolist(),
        "n": int(len(y)),
    }
    write_json(out_root / "feature_importance_audit.json", payload)
    imp.to_csv(out_root / "feature_importance_bad_vs_medium.csv", index=False)
    return imp


def analyze_anchor(args: argparse.Namespace, row: dict[str, Any], X: np.ndarray, meta: pd.DataFrame, morph: pd.DataFrame, sqi: pd.DataFrame) -> dict[str, Any]:
    run_dir = Path(row["run_dir"])
    ckpt = run_dir / "ckpt_best.pt"
    anchor_id = f"{row.get('anchor_group', 'anchor')}__{row.get('mode', 'run')}__{row.get('spec', {}).get('id', run_dir.name)}"
    anchor_id = safe_id(anchor_id)
    out_dir = Path(args.out_root) / "anchors" / anchor_id
    report_dir = Path(args.report_root) / "anchors" / anchor_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt.exists():
        payload = {"anchor_id": anchor_id, "status": "missing_checkpoint", "run_dir": str(run_dir)}
        write_json(out_dir / "anchor_analysis.json", payload)
        return payload

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_uformer_model(ckpt, device, feature_set="full_tokens", with_head=True)
    outputs = run_model_outputs(model, X, int(args.batch_size_eval), device)
    probs = outputs["probs"]
    logits = outputs["logits"]
    assert isinstance(probs, np.ndarray)
    assert isinstance(logits, np.ndarray)

    split = meta["split"].astype(str).to_numpy()
    y = meta["y"].to_numpy(dtype=np.int64)
    cal = calibrate_but(probs[split == "val"], y[split == "val"])
    pred_raw = np.argmax(probs, axis=1).astype(np.int64)
    pred_cal = apply_but_thresholds(probs, float(cal["t_good"]), float(cal["t_bad"]))
    balanced_idx, balanced_manifest = balanced_but_test_indices(meta, seed=int(args.seed))
    strat_idx, strat_manifest = stratified_balanced_indices(meta, morph, seed=int(args.seed) + 17)
    test_idx = np.where(split == "test")[0]

    df = meta.reset_index(drop=True).copy()
    df["idx"] = np.arange(len(df), dtype=int)
    df["p_good"] = probs[:, 0]
    df["p_medium"] = probs[:, 1]
    df["p_bad"] = probs[:, 2]
    df["logit_good"] = logits[:, 0]
    df["logit_medium"] = logits[:, 1]
    df["logit_bad"] = logits[:, 2]
    df["pred_raw"] = pred_raw
    df["pred_calibrated"] = pred_cal
    df["pred_raw_class"] = [INT_TO_CLASS[int(v)] for v in pred_raw]
    df["pred_calibrated_class"] = [INT_TO_CLASS[int(v)] for v in pred_cal]
    df["margin_bad_medium"] = df["p_bad"] - df["p_medium"]
    df["margin_good_medium"] = df["p_good"] - df["p_medium"]
    df["entropy"] = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)
    df["error_type"] = classify_errors(y, pred_cal)
    df["error_type_raw"] = classify_errors(y, pred_raw)
    for col in FEATURE_COLUMNS:
        if col in morph:
            df[col] = morph[col].to_numpy()
    for col in SQI_COLUMNS:
        df[col] = sqi[col].to_numpy()
    df = infer_failure_tags(df)

    df.to_csv(out_dir / "predictions_with_features.csv", index=False)
    df.loc[balanced_idx].to_csv(out_dir / "balanced_predictions_with_features.csv", index=False)
    df.loc[strat_idx].to_csv(out_dir / "stratified_balanced_predictions_with_features.csv", index=False)
    write_json(out_dir / "balanced_selection_manifest.json", balanced_manifest)
    write_json(out_dir / "stratified_balanced_selection_manifest.json", strat_manifest)
    shutil.copy2(out_dir / "balanced_selection_manifest.json", report_dir / "balanced_selection_manifest.json")
    shutil.copy2(out_dir / "stratified_balanced_selection_manifest.json", report_dir / "stratified_balanced_selection_manifest.json")

    reports = {
        "anchor_id": anchor_id,
        "spec_id": row.get("spec", {}).get("id"),
        "mode": row.get("mode"),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "calibration": cal,
        "original_test_calibrated": multiclass_report(y[test_idx], pred_cal[test_idx], probs[test_idx]),
        "original_test_raw": multiclass_report(y[test_idx], pred_raw[test_idx], probs[test_idx]),
        "balanced_test_calibrated": multiclass_report(y[balanced_idx], pred_cal[balanced_idx], probs[balanced_idx]),
        "balanced_test_raw": multiclass_report(y[balanced_idx], pred_raw[balanced_idx], probs[balanced_idx]),
        "stratified_balanced_test_calibrated": multiclass_report(y[strat_idx], pred_cal[strat_idx], probs[strat_idx]),
        "stratified_balanced_test_raw": multiclass_report(y[strat_idx], pred_raw[strat_idx], probs[strat_idx]),
        "runtime": {"elapsed_sec": float(outputs["elapsed_sec"]), "peak_cuda_memory_bytes": int(outputs["peak_cuda_memory_bytes"])},
    }
    write_json(out_dir / "anchor_analysis.json", reports)
    write_json(report_dir / "anchor_analysis.json", reports)

    error_summary = (
        df[df["split"] == "test"]
        .groupby(["y_class", "pred_calibrated_class", "error_type"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    error_summary.to_csv(out_dir / "error_taxonomy.csv", index=False)
    write_json(out_dir / "error_taxonomy.json", {"rows": error_summary.to_dict(orient="records")})
    error_summary.to_csv(report_dir / "error_taxonomy.csv", index=False)
    write_json(report_dir / "error_taxonomy.json", {"rows": error_summary.to_dict(orient="records")})

    figures = report_dir / "figures"
    plot_class_profiles(X, df, figures)
    for case in ["bad_missed", "correct_bad", "medium_to_good", "medium_to_bad", "good_false_bad"]:
        plot_wave_cases(X, df[df["split"] == "test"], case, figures / f"{case}.png")
    plot_embeddings(df, figures, "morphology", list(FEATURE_COLUMNS))
    plot_embeddings(df, figures, "seven_sqi", list(SQI_COLUMNS))
    plot_probability(df, figures)
    imp = feature_importance(df, out_dir)
    if not imp.empty:
        imp.to_csv(report_dir / "feature_importance_bad_vs_medium.csv", index=False)
    return reports


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: float(r.get("original_test_calibrated", {}).get("macro_f1", 0.0)), reverse=True)
    compact: list[dict[str, Any]] = []
    for row in ranked:
        rep = row.get("original_test_calibrated", {})
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        bal = row.get("stratified_balanced_test_calibrated", {})
        compact.append(
            {
                "anchor_id": row.get("anchor_id"),
                "spec_id": row.get("spec_id"),
                "mode": row.get("mode"),
                "acc": rep.get("acc"),
                "macro_f1": rep.get("macro_f1"),
                "balanced_acc": rep.get("balanced_acc"),
                "good_recall": rec[0] if len(rec) > 0 else None,
                "medium_recall": rec[1] if len(rec) > 1 else None,
                "bad_recall": rec[2] if len(rec) > 2 else None,
                "stratified_macro_f1": bal.get("macro_f1"),
            }
        )
    pd.DataFrame(compact).to_csv(report_root / "anchor_metric_summary.csv", index=False)
    lines = [
        "# BUT 10s Medium-Guard Bad-Boundary Analysis",
        "",
        "Formal result remains original BUT 10s P1 test. Balanced and stratified-balanced subsets are prediction-independent diagnostics.",
        "",
        "| rank | anchor | mode | acc | macro-F1 | recalls G/M/B | stratified macro |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for i, row in enumerate(compact, start=1):
        lines.append(
            f"| {i} | `{row['spec_id']}` | {row['mode']} | {float(row.get('acc') or 0):.4f} | "
            f"{float(row.get('macro_f1') or 0):.4f} | "
            f"{float(row.get('good_recall') or 0):.3f}/{float(row.get('medium_recall') or 0):.3f}/{float(row.get('bad_recall') or 0):.3f} | "
            f"{float(row.get('stratified_macro_f1') or 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Next-rule interpretation",
            "",
            "- If `bad_missed` is high with visible QRS, add `visible_qrs_but_unusable` and baseline/platform subtypes rather than only lower SNR.",
            "- If `medium_to_bad` is high, reduce medium contact/flat and keep medium QRS preserve high.",
            "- If SQI PCA separates bad better than morphology PCA, promote SQI fusion as an analysis branch only after checking medium recall.",
        ]
    )
    (report_root / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "analysis_report.json", {"anchors": rows, "compact": compact})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state = out_root / STATE_NAME
    update_state(state, status="running", updated_at=now_iso())
    X, meta = load_but(Path(args.but_protocol_dir))
    morph = extract_features(X, meta.assign(dataset="BUT 10s P1", class_name=meta["y_class"].astype(str)), out_root / "but_morph_features.csv", max_rows=0)
    sqi = compute_sqi_frame(X, out_root / "but_sqi_features.csv")
    anchors = locate_anchor_rows()
    if args.max_anchors > 0:
        anchors = anchors[: int(args.max_anchors)]
    write_json(out_root / "anchor_manifest.json", {"rows": anchors})
    write_json(report_root / "anchor_manifest.json", {"rows": anchors})
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(anchors, start=1):
        update_state(state, status="running", current=row.get("spec", {}).get("id", row.get("run_dir")), completed=i - 1, total=len(anchors), updated_at=now_iso())
        rows.append(analyze_anchor(args, row, X, meta, morph, sqi))
    write_summary(args, rows)
    update_state(state, status="complete", completed=len(rows), total=len(rows), updated_at=now_iso())
    return {"status": "complete", "anchors": len(rows), "report_root": str(report_root)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze BUT 10s medium-vs-bad boundary for selected anchors.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--max_anchors", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
