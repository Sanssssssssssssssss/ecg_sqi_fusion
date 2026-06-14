"""Reduced SQI/geometry feature candidates for the current N17043 node split.

This experiment keeps the 47-column feature branch logic but restricts each
candidate to a small, explicit feature subset. Feature ranking and model
selection use train/val only; the held-out BUT test split is report-only.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_OUT = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
ANALYSIS_REPORT = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
NODE_ID = "N17043_gm_probe"
VARIANT_ID = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
MANIFEST_PATH = OUT_ROOT / "nodes" / NODE_ID / "node_boundary_manifest.csv"
FEATURE_SCHEMA_PATH = OUT_ROOT / "synthetic_variants" / VARIANT_ID / "datasets" / "tabular_feature_schema.json"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
LABELS = [0, 1, 2]

TOP6_CORE = ["pc1", "qrs_visibility", "baseline_step", "flatline_ratio", "non_qrs_diff_p95", "pca_margin"]
TOP10_INTERPRETABLE = TOP6_CORE + ["qrs_band_ratio", "template_corr", "amplitude_entropy", "sqi_bSQI"]
TOP14_BALANCED = TOP10_INTERPRETABLE + ["region_confidence", "detector_agreement", "mean_abs", "sqi_basSQI"]

CLASS_WEIGHT = [1.03, 1.22, 2.05]
AUTO_TOPK_VALUES = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
AROUND20_TOPK_VALUES = [18, 20, 22, 24]
SEED = 20260614

REFERENCE_47 = {
    "name": "47-feature current best, threshold p_bad>=0.13",
    "original_test_all_10s+": {
        "n": 8477,
        "acc": 0.963548425150407,
        "macro_f1": 0.9306830954417679,
        "good_recall": 0.9563186813186814,
        "medium_recall": 0.9728874830546769,
        "bad_recall": 0.927007299270073,
    },
    "original_all_10s+": {
        "n": 32956,
        "acc": 0.9853744386454667,
        "macro_f1": 0.985233025715266,
        "good_recall": 0.9888517279821628,
        "medium_recall": 0.9753481369966127,
        "bad_recall": 0.9943235572374646,
    },
}


@dataclass
class FeatureNorm:
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class ReducedFeatureMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    manifest = load_manifest()
    feature_columns = load_feature_columns()
    sanity_check_features(manifest, feature_columns)

    ranking = build_feature_ranking(manifest, feature_columns)
    ranking_path = ANALYSIS_OUT / "current_reduced_feature_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    candidate_defs = {
        "top6_core": TOP6_CORE,
        "top10_interpretable": TOP10_INTERPRETABLE,
        "top14_balanced": TOP14_BALANCED,
    }
    ranked_features = ranking["feature"].tolist()
    for k in AROUND20_TOPK_VALUES:
        candidate_defs[f"ranked_top{k}"] = ranked_features[:k]
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, columns in candidate_defs.items():
        result = train_and_eval_candidate(name, manifest, columns, epochs=120)
        summaries.append(result)
        all_rows.extend(result["metrics"])

    auto_results = []
    for k in AUTO_TOPK_VALUES:
        cols = ranked_features[:k]
        result = train_and_eval_candidate(f"auto_top{k}", manifest, cols, epochs=120)
        auto_results.append(result)
    auto_best = select_best_auto(auto_results)
    original_auto_name = auto_best["candidate_name"]
    auto_best["candidate_name"] = "auto_topk"
    for row in auto_best["metrics"]:
        if row["candidate"] == original_auto_name:
            row["candidate"] = "auto_topk"
        if row["candidate"] == f"{original_auto_name}_badcal":
            row["candidate"] = "auto_topk_badcal"
    auto_best["selected_from"] = [r["candidate_name"] for r in auto_results]
    summaries.append(auto_best)
    all_rows.extend(auto_best["metrics"])

    selection_rows = []
    for result in summaries:
        val = metric_by_bucket(result["metrics"], "val")
        test = metric_by_bucket(result["metrics"], "original_test_all_10s+")
        selection_rows.append(
            {
                "candidate": result["candidate_name"],
                "n_features": len(result["feature_columns"]),
                "features": ", ".join(result["feature_columns"]),
                "val_acc": val["acc"],
                "val_macro_f1": val["macro_f1"],
                "val_good_recall": val["good_recall"],
                "val_medium_recall": val["medium_recall"],
                "val_bad_recall": val["bad_recall"],
                "val_score": result["selection_score"],
                "original_test_acc_report_only": test["acc"],
                "original_test_good_recall": test["good_recall"],
                "original_test_medium_recall": test["medium_recall"],
                "original_test_bad_recall": test["bad_recall"],
            }
        )

    metrics = pd.DataFrame(all_rows)
    selection = pd.DataFrame(selection_rows).sort_values(["val_score", "val_acc"], ascending=False)
    metrics_path = ANALYSIS_OUT / "current_reduced_feature_metrics.csv"
    selection_path = ANALYSIS_OUT / "current_reduced_feature_trainval_selection.csv"
    summary_path = ANALYSIS_OUT / "current_reduced_feature_summary.json"
    report_out = ANALYSIS_OUT / "current_reduced_feature_report.md"
    report_copy = ANALYSIS_REPORT / "current_reduced_feature_report.md"
    metrics.to_csv(metrics_path, index=False)
    selection.to_csv(selection_path, index=False)

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "node_id": NODE_ID,
        "variant_id": VARIANT_ID,
        "selection": "train/val only; held-out test is report-only",
        "original_used_for_training": False,
        "feature_ranking_csv": str(ranking_path),
        "metrics_csv": str(metrics_path),
        "selection_csv": str(selection_path),
        "report": str(report_copy),
        "reference_47": REFERENCE_47,
        "candidates": [
            {
                "candidate_name": s["candidate_name"],
                "feature_columns": s["feature_columns"],
                "selection_score": s["selection_score"],
                "model_path": s["model_path"],
                "normalization_path": s["normalization_path"],
                "bad_threshold_ablation": s["bad_threshold_ablation"],
            }
            for s in summaries
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = render_report(selection, metrics, ranking, payload)
    report_out.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_PATH)
    df["y"] = df["class_name"].astype(str).map(CLASS_TO_INT).astype(int)
    return df


def load_feature_columns() -> list[str]:
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return list(schema["feature_columns"])


def sanity_check_features(manifest: pd.DataFrame, feature_columns: list[str]) -> None:
    required = set(TOP14_BALANCED) | set(feature_columns)
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"node manifest missing required feature columns: {missing}")
    split_counts = manifest["split"].astype(str).value_counts().to_dict()
    expected = {"train", "val", "test"}
    if set(split_counts) != expected:
        raise ValueError(f"expected split set {expected}, got {split_counts}")


def build_feature_ranking(manifest: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    full_result = train_model(
        manifest,
        feature_columns,
        model_name="ranking_full47",
        epochs=80,
        save_artifacts=False,
    )
    model = full_result["model"]
    norm = full_result["norm"]
    x_val, y_val = make_xy(manifest, "val", feature_columns, norm)
    base_probs = predict_proba(model, x_val)
    base_pred = base_probs.argmax(axis=1)
    base = metric_report(y_val, base_pred)
    rng = np.random.default_rng(SEED + 47)
    rows = []
    train = manifest[manifest["split"].astype(str).eq("train")].reset_index(drop=True)
    val = manifest[manifest["split"].astype(str).eq("val")].reset_index(drop=True)
    for j, feature in enumerate(feature_columns):
        drops = []
        macro_drops = []
        for _ in range(5):
            x_perm = x_val.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            pred = predict_proba(model, x_perm).argmax(axis=1)
            rep = metric_report(y_val, pred)
            drops.append(base["acc"] - rep["acc"])
            macro_drops.append(base["macro_f1"] - rep["macro_f1"])
        uni_train = univariate_scores(train, feature)
        uni_val = univariate_scores(val, feature)
        score = (
            max(float(np.mean(drops)), 0.0) * 4.0
            + max(float(np.mean(macro_drops)), 0.0) * 2.0
            + uni_train["gm_auc"] * 0.45
            + uni_val["gm_auc"] * 0.45
            + uni_train["bad_auc"] * 0.55
            + uni_val["bad_auc"] * 0.55
            + uni_train["gm_ks"] * 0.25
            + uni_train["bad_ks"] * 0.25
        )
        rows.append(
            {
                "feature": feature,
                "ranking_score": float(score),
                "perm_acc_drop_val_mean": float(np.mean(drops)),
                "perm_macro_f1_drop_val_mean": float(np.mean(macro_drops)),
                "train_gm_auc": uni_train["gm_auc"],
                "val_gm_auc": uni_val["gm_auc"],
                "train_gm_ks": uni_train["gm_ks"],
                "val_gm_ks": uni_val["gm_ks"],
                "train_bad_auc": uni_train["bad_auc"],
                "val_bad_auc": uni_val["bad_auc"],
                "train_bad_ks": uni_train["bad_ks"],
                "val_bad_ks": uni_val["bad_ks"],
                "train_gm_direction": uni_train["gm_direction"],
                "train_bad_direction": uni_train["bad_direction"],
            }
        )
    out = pd.DataFrame(rows).sort_values("ranking_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def train_and_eval_candidate(name: str, manifest: pd.DataFrame, columns: list[str], epochs: int) -> dict[str, Any]:
    result = train_model(manifest, columns, model_name=name, epochs=epochs, save_artifacts=True)
    model = result["model"]
    norm = result["norm"]
    metrics: list[dict[str, Any]] = []
    for bucket, split in [("train", "train"), ("val", "val"), ("original_test_all_10s+", "test"), ("original_all_10s+", "all")]:
        x, y = make_xy(manifest, split, columns, norm)
        probs = predict_proba(model, x)
        metrics.append(metric_row(name, bucket, y, probs.argmax(axis=1)))
    meta = manifest
    test_mask = meta["split"].astype(str).eq("test").to_numpy()
    bad_mask = meta["class_name"].astype(str).eq("bad").to_numpy()
    outlier_mask = meta["original_region"].astype(str).eq("outlier_low_confidence").to_numpy()
    for bucket, mask in [
        ("original_test_without_bad_outlier_stress", test_mask & ~(bad_mask & outlier_mask)),
        ("bad_core_nearboundary", test_mask & bad_mask & ~outlier_mask),
        ("bad_outlier_stress", test_mask & bad_mask & outlier_mask),
    ]:
        x, y = make_xy_from_mask(manifest, mask, columns, norm)
        probs = predict_proba(model, x)
        metrics.append(metric_row(name, bucket, y, probs.argmax(axis=1)))
    threshold_info = calibrate_bad_threshold(model, manifest, columns, norm)
    badcal_name = f"{name}_badcal"
    for bucket, split in [("train", "train"), ("val", "val"), ("original_test_all_10s+", "test"), ("original_all_10s+", "all")]:
        x, y = make_xy(manifest, split, columns, norm)
        probs = predict_proba(model, x)
        metrics.append(metric_row(badcal_name, bucket, y, apply_bad_threshold(probs, threshold_info["threshold"])))
    for bucket, mask in [
        ("original_test_without_bad_outlier_stress", test_mask & ~(bad_mask & outlier_mask)),
        ("bad_core_nearboundary", test_mask & bad_mask & ~outlier_mask),
        ("bad_outlier_stress", test_mask & bad_mask & outlier_mask),
    ]:
        x, y = make_xy_from_mask(manifest, mask, columns, norm)
        probs = predict_proba(model, x)
        metrics.append(metric_row(badcal_name, bucket, y, apply_bad_threshold(probs, threshold_info["threshold"])))
    val = metric_by_bucket(metrics, "val")
    selection_score = val["acc"] + 0.40 * val["macro_f1"] + 0.25 * min(val["good_recall"], val["medium_recall"], val["bad_recall"])
    return {
        "candidate_name": name,
        "feature_columns": columns,
        "model": model,
        "norm": norm,
        "metrics": metrics,
        "selection_score": float(selection_score),
        "model_path": result["model_path"],
        "normalization_path": result["normalization_path"],
        "bad_threshold_ablation": threshold_info,
    }


def select_best_auto(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=lambda r: (r["selection_score"], metric_by_bucket(r["metrics"], "val")["acc"], -len(r["feature_columns"])))


def calibrate_bad_threshold(model: ReducedFeatureMLP, manifest: pd.DataFrame, columns: list[str], norm: FeatureNorm) -> dict[str, Any]:
    train_mask = manifest["split"].astype(str).isin(["train", "val"]).to_numpy()
    x, y = make_xy_from_mask(manifest, train_mask, columns, norm)
    probs = predict_proba(model, x)
    rows = []
    for threshold in np.round(np.linspace(0.01, 0.95, 95), 2):
        pred = apply_bad_threshold(probs, float(threshold))
        rep = metric_report(y, pred)
        rows.append(
            {
                "threshold": float(threshold),
                "acc": rep["acc"],
                "macro_f1": rep["macro_f1"],
                "good_recall": rep["good_recall"],
                "medium_recall": rep["medium_recall"],
                "bad_recall": rep["bad_recall"],
            }
        )
    sweep = pd.DataFrame(rows)
    valid = sweep[
        (sweep["acc"] >= 0.95)
        & (sweep["good_recall"] >= 0.92)
        & (sweep["medium_recall"] >= 0.90)
        & (sweep["bad_recall"] >= 0.90)
    ]
    if valid.empty:
        chosen = sweep.sort_values(["acc", "bad_recall", "macro_f1"], ascending=[False, False, False]).iloc[0]
    else:
        chosen = valid.sort_values(["acc", "macro_f1", "bad_recall"], ascending=[False, False, False]).iloc[0]
    return {
        "threshold": float(chosen["threshold"]),
        "trainval_acc": float(chosen["acc"]),
        "trainval_macro_f1": float(chosen["macro_f1"]),
        "trainval_good_recall": float(chosen["good_recall"]),
        "trainval_medium_recall": float(chosen["medium_recall"]),
        "trainval_bad_recall": float(chosen["bad_recall"]),
        "selection": "train+val only; report-only ablation, not pure-model candidate",
    }


def apply_bad_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    pred = probs.argmax(axis=1).astype(np.int64)
    pred[probs[:, CLASS_TO_INT["bad"]] >= float(threshold)] = CLASS_TO_INT["bad"]
    return pred


def train_model(
    manifest: pd.DataFrame,
    columns: list[str],
    *,
    model_name: str,
    epochs: int,
    save_artifacts: bool,
) -> dict[str, Any]:
    norm = fit_norm(manifest, columns)
    x_train, y_train = make_xy(manifest, "train", columns, norm)
    x_val, y_val = make_xy(manifest, "val", columns, norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReducedFeatureMLP(len(columns)).to(device)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-4)
    class_weight = torch.tensor(CLASS_WEIGHT, dtype=torch.float32, device=device)
    best_state = None
    best_score = -math.inf
    best_epoch = 0
    patience = 18
    stale = 0
    log_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_probs = predict_proba(model, x_val)
        val_rep = metric_report(y_val, val_probs.argmax(axis=1))
        score = val_rep["acc"] + 0.40 * val_rep["macro_f1"] + 0.25 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
        log_rows.append({"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        if score > best_score:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    model_path = ""
    norm_path = ""
    if save_artifacts:
        run_dir = OUT_ROOT / "runs" / "reduced_feature_nodecal" / NODE_ID / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(run_dir / "model.pt")
        norm_path = str(run_dir / "normalization.json")
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_columns": columns,
                "class_weight": CLASS_WEIGHT,
                "best_epoch": best_epoch,
                "model_kind": "reduced_feature_tabular_mlp",
                "selection": "train split for weights, val split for early stopping/model selection",
                "heldout_test_used_for_selection": False,
            },
            model_path,
        )
        pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
        (run_dir / "normalization.json").write_text(
            json.dumps(
                {
                    "feature_columns": columns,
                    "median": norm.median.tolist(),
                    "mean": norm.mean.tolist(),
                    "std": norm.std.tolist(),
                    "source": "train split only",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        joblib.dump({"feature_columns": columns, "median": norm.median, "mean": norm.mean, "std": norm.std}, run_dir / "normalization.joblib")
    return {"model": model, "norm": norm, "best_epoch": best_epoch, "model_path": model_path, "normalization_path": norm_path}


def fit_norm(manifest: pd.DataFrame, columns: list[str]) -> FeatureNorm:
    train = manifest[manifest["split"].astype(str).eq("train")]
    raw = train[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    median = np.nanmedian(raw, axis=0).astype(np.float32)
    filled = np.where(np.isfinite(raw), raw, median[None, :]).astype(np.float32)
    mean = np.nanmean(filled, axis=0).astype(np.float32)
    std = np.nanstd(filled, axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return FeatureNorm(columns=columns, median=median, mean=mean, std=std)


def make_xy(manifest: pd.DataFrame, split: str, columns: list[str], norm: FeatureNorm) -> tuple[np.ndarray, np.ndarray]:
    if split == "all":
        mask = np.ones(len(manifest), dtype=bool)
    else:
        mask = manifest["split"].astype(str).eq(split).to_numpy()
    return make_xy_from_mask(manifest, mask, columns, norm)


def make_xy_from_mask(manifest: pd.DataFrame, mask: np.ndarray, columns: list[str], norm: FeatureNorm) -> tuple[np.ndarray, np.ndarray]:
    frame = manifest.loc[mask]
    raw = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    raw = np.where(np.isfinite(raw), raw, norm.median[None, :]).astype(np.float32)
    x = ((raw - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32)
    y = frame["y"].to_numpy(dtype=np.int64)
    return x, y


def predict_proba(model: ReducedFeatureMLP, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).to(device=device, dtype=torch.float32)
            probs.append(torch.softmax(model(xb), dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 3), dtype=np.float32)


def univariate_scores(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    y = frame["y"].to_numpy(dtype=np.int64)
    x = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    med = np.nanmedian(x)
    x = np.where(np.isfinite(x), x, med)
    gm_mask = (y == CLASS_TO_INT["good"]) | (y == CLASS_TO_INT["medium"])
    x_gm = x[gm_mask]
    y_gm = y[gm_mask]
    good_values = x_gm[y_gm == CLASS_TO_INT["good"]]
    medium_values = x_gm[y_gm == CLASS_TO_INT["medium"]]
    bad_values = x[y == CLASS_TO_INT["bad"]]
    nonbad_values = x[y != CLASS_TO_INT["bad"]]
    gm_auc, gm_direction = binary_auc(x_gm, y_gm == CLASS_TO_INT["medium"])
    bad_auc, bad_direction = binary_auc(x, y == CLASS_TO_INT["bad"])
    return {
        "gm_auc": gm_auc,
        "gm_ks": ks_distance(good_values, medium_values),
        "gm_direction": gm_direction,
        "bad_auc": bad_auc,
        "bad_ks": ks_distance(nonbad_values, bad_values),
        "bad_direction": bad_direction,
    }


def binary_auc(scores: np.ndarray, target: np.ndarray) -> tuple[float, str]:
    if len(np.unique(target.astype(int))) < 2:
        return 0.5, "none"
    auc = float(roc_auc_score(target.astype(int), scores))
    if auc >= 0.5:
        return auc, "higher_positive"
    return 1.0 - auc, "lower_positive"


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return 0.0
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def metric_row(candidate: str, bucket: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(metric_report(y_true, y_pred))
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "n": 0,
            "acc": 0.0,
            "macro_f1": 0.0,
            "good_recall": 0.0,
            "medium_recall": 0.0,
            "bad_recall": 0.0,
            "good_precision": 0.0,
            "medium_precision": 0.0,
            "bad_precision": 0.0,
            "good_to_medium": 0,
            "medium_to_good": 0,
            "bad_to_medium": 0,
            "confusion_3x3": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    recalls = []
    precisions = []
    for i in LABELS:
        support = float(cm[i, :].sum())
        pred_n = float(cm[:, i].sum())
        tp = float(cm[i, i])
        recalls.append(tp / support if support else 0.0)
        precisions.append(tp / pred_n if pred_n else 0.0)
    return {
        "n": int(len(y_true)),
        "acc": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "good_recall": float(recalls[0]),
        "medium_recall": float(recalls[1]),
        "bad_recall": float(recalls[2]),
        "good_precision": float(precisions[0]),
        "medium_precision": float(precisions[1]),
        "bad_precision": float(precisions[2]),
        "good_to_medium": int(cm[0, 1]),
        "medium_to_good": int(cm[1, 0]),
        "bad_to_medium": int(cm[2, 1]),
        "confusion_3x3": cm.astype(int).tolist(),
    }


def metric_by_bucket(rows: list[dict[str, Any]], bucket: str) -> dict[str, Any]:
    for row in rows:
        if row["bucket"] == bucket:
            return row
    raise KeyError(bucket)


def render_report(selection: pd.DataFrame, metrics: pd.DataFrame, ranking: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Current Reduced SQI/Geometry Feature Candidates",
        "",
        "Reduced-feature tabular MLP candidates trained with train split only, selected on validation only, and reported on the held-out BUT test split.",
        "",
        "## Feature Ranking Top 16",
        "",
        markdown_table(
            ranking[
                [
                    "rank",
                    "feature",
                    "ranking_score",
                    "perm_acc_drop_val_mean",
                    "train_gm_auc",
                    "val_gm_auc",
                    "train_bad_auc",
                    "val_bad_auc",
                    "train_gm_direction",
                    "train_bad_direction",
                ]
            ].head(16)
        ),
        "",
        "## Train/Val Selection",
        "",
        markdown_table(selection),
        "",
        "## Held-Out Original BUT Test",
        "",
        "| Candidate | n features | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in selection.iterrows():
        cand = row["candidate"]
        test = metrics[(metrics["candidate"] == cand) & (metrics["bucket"] == "original_test_all_10s+")].iloc[0]
        lines.append(
            f"| {cand} | {int(row['n_features'])} | {test['acc']:.6f} | {test['macro_f1']:.6f} | "
            f"{test['good_recall']:.6f} | {test['medium_recall']:.6f} | {test['bad_recall']:.6f} | "
            f"{int(test['good_to_medium'])} | {int(test['medium_to_good'])} | {int(test['bad_to_medium'])} |"
        )
    ref = REFERENCE_47["original_test_all_10s+"]
    lines.append(
        f"| {REFERENCE_47['name']} | 47 | {ref['acc']:.6f} | {ref['macro_f1']:.6f} | "
        f"{ref['good_recall']:.6f} | {ref['medium_recall']:.6f} | {ref['bad_recall']:.6f} | 102 | 76 | 23 |"
    )
    sqi_path = ANALYSIS_OUT / "current_7sqi_baseline_metrics.csv"
    if sqi_path.exists():
        sqi = pd.read_csv(sqi_path)
        sqi_test = sqi[sqi["bucket"].astype(str).eq("original_test_all_10s+")]
        if not sqi_test.empty:
            r = sqi_test.iloc[0]
            lines.append(
                f"| Current 7SQI-only baseline | 7 | {r['acc']:.6f} | {r['macro_f1']:.6f} | "
                f"{r['good_recall']:.6f} | {r['medium_recall']:.6f} | {r['bad_recall']:.6f} | 2638 | 39 | 365 |"
            )
    lines.extend(
        [
            "",
            "## Train+Val Bad-Threshold Ablation",
            "",
            "This is a separate explanatory ablation, not the pure reduced-feature candidate. The threshold is selected on train+val only and then reported on held-out test.",
            "",
            "| Candidate | Threshold | Acc | Macro-F1 | Good R | Medium R | Bad R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in selection.iterrows():
        cand = f"{row['candidate']}_badcal"
        test_rows = metrics[(metrics["candidate"] == cand) & (metrics["bucket"] == "original_test_all_10s+")]
        if test_rows.empty:
            continue
        test = test_rows.iloc[0]
        threshold = ""
        for item in payload["candidates"]:
            if item["candidate_name"] == row["candidate"]:
                threshold = f"{item['bad_threshold_ablation']['threshold']:.2f}"
                break
        lines.append(
            f"| {cand} | {threshold} | {test['acc']:.6f} | {test['macro_f1']:.6f} | "
            f"{test['good_recall']:.6f} | {test['medium_recall']:.6f} | {test['bad_recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Original Buckets",
            "",
            markdown_table(metrics[metrics["bucket"].astype(str).str.startswith(("original", "bad_"))]),
            "",
            "## Outputs",
            "",
            f"- Ranking CSV: `{payload['feature_ranking_csv']}`",
            f"- Selection CSV: `{payload['selection_csv']}`",
            f"- Metrics CSV: `{payload['metrics_csv']}`",
            "",
            "## Notes",
            "",
            "- No held-out test rows are used for feature ranking, model training, early stopping, or candidate selection.",
            "- These are pure reduced-feature MLP candidates; no hand-written threshold/gate is added in v1.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{float(x):.6f}" if pd.notna(x) else "")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
