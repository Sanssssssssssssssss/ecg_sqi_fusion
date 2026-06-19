"""Waveform primitive learnability audit for hard SQI/geometry targets.

This is an external-only diagnostic. It uses waveform-derived primitive stats
from synthetic/PTB rows to ask which teacher targets are stable waveform facts
and which are mostly atlas/label-geometry proxies. Original BUT is report-only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"


def load_student_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student_audit", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STUDENT = load_student_module()
ARCH = STUDENT.ARCH
FEATURE_COLUMNS = list(STUDENT.FEATURE_COLUMNS)


TARGET_COLUMNS = [
    "pc1",
    "pc2",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "sqi_basSQI",
    "low_amp_ratio",
    "non_qrs_diff_p95",
    "qrs_band_ratio",
    "template_corr",
    "qrs_prom_p90",
    "sqi_bSQI",
    "sqi_sSQI",
    "sqi_kSQI",
    "amplitude_entropy",
    "mean_abs",
    "rms",
]


def now() -> str:
    import datetime as _dt

    return _dt.datetime.now().isoformat(timespec="seconds")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def primitive_column_names(bank: str, dim: int) -> list[str]:
    # Exact sub-bank names are intentionally coarse here: the point is to find
    # which waveform-derived evidence family can explain each teacher target.
    per_ch = STUDENT.primitive_bank_per_channel(bank)
    if dim % per_ch != 0:
        return [f"{bank}_{i:03d}" for i in range(dim)]
    groups = [
        ("atlas", 55),
        ("qrs_visibility_bank", 19),
        ("detector_agreement_bank", 16),
        ("baseline_frequency_bank", 24),
        ("stress_bank", 24),
        ("sparse_event_bank", 32),
    ]
    names: list[str] = []
    channels = dim // per_ch
    for ch in range(channels):
        pos = 0
        for group, width in groups:
            if pos >= per_ch:
                break
            use = min(width, per_ch - pos)
            for j in range(use):
                names.append(f"ch{ch}_{group}_{j:02d}")
            pos += use
        while pos < per_ch:
            names.append(f"ch{ch}_extra_{pos:02d}")
            pos += 1
    return names[:dim]


@torch.no_grad()
def collect_primitives(dataset: Any, bank: str, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(dtype=torch.float32)
        stats = STUDENT.primitive_waveform_stats(x, bank)
        chunks.append(stats.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = np.nanstd(train, axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    out = [((arr - mean[None, :]) / std[None, :]).astype(np.float32) for arr in (train, *others)]
    return tuple(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in out)


def max_abs_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    mask = np.isfinite(y)
    if mask.sum() < 4:
        return float("nan"), -1, float("nan")
    xm = x[mask].astype(np.float64)
    ym = y[mask].astype(np.float64)
    ym = ym - ym.mean()
    ystd = float(np.sqrt(np.mean(ym * ym)))
    if ystd < 1e-9:
        return 0.0, -1, 0.0
    x0 = xm - xm.mean(axis=0, keepdims=True)
    xstd = np.sqrt(np.mean(x0 * x0, axis=0))
    denom = np.where(xstd > 1e-9, xstd * ystd, np.inf)
    corr = np.mean(x0 * ym[:, None], axis=0) / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    idx = int(np.argmax(np.abs(corr)))
    return float(abs(corr[idx])), idx, float(corr[idx])


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 4:
        return float("nan")
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot < 1e-9:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def fit_predictors(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge

    mask_train = np.isfinite(y_train)
    mask_test = np.isfinite(y_test)
    if mask_train.sum() < 20 or mask_test.sum() < 10:
        return {"ridge_r2": np.nan, "trees_r2": np.nan, "tree_top_idx": -1, "tree_top_importance": np.nan}
    xt = x_train[mask_train]
    yt = y_train[mask_train]
    xv = x_test[mask_test]
    yv = y_test[mask_test]
    ridge = Ridge(alpha=4.0, random_state=int(seed))
    ridge.fit(xt, yt)
    ridge_pred = ridge.predict(xv)
    trees = ExtraTreesRegressor(
        n_estimators=96,
        max_depth=16,
        min_samples_leaf=3,
        max_features=0.55,
        random_state=int(seed),
        n_jobs=-1,
    )
    trees.fit(xt, yt)
    tree_pred = trees.predict(xv)
    importances = np.asarray(trees.feature_importances_, dtype=float)
    top_idx = int(np.argmax(importances)) if importances.size else -1
    return {
        "ridge_r2": r2_score_np(yv, ridge_pred),
        "trees_r2": r2_score_np(yv, tree_pred),
        "tree_top_idx": top_idx,
        "tree_top_importance": float(importances[top_idx]) if top_idx >= 0 else np.nan,
    }


def class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out["acc"] = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    for cls, name in [(0, "good"), (1, "medium"), (2, "bad")]:
        mask = y_true == cls
        out[f"{name}_recall"] = float(np.mean(y_pred[mask] == cls)) if mask.any() else float("nan")
    return out


def primitive_classifier_audit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_orig: np.ndarray | None,
    y_orig: np.ndarray | None,
    orig_split: np.ndarray | None,
    seed: int,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    rows: dict[str, Any] = {}
    logreg = LogisticRegression(
        C=0.7,
        max_iter=800,
        class_weight={0: 1.0, 1: 1.35, 2: 3.0},
        random_state=int(seed),
        n_jobs=1,
        multi_class="auto",
    )
    trees = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=16,
        min_samples_leaf=3,
        max_features=0.55,
        class_weight={0: 1.0, 1: 1.35, 2: 3.3},
        random_state=int(seed),
        n_jobs=-1,
    )
    for name, model in [("primitive_logreg", logreg), ("primitive_trees", trees)]:
        model.fit(x_train, y_train)
        rows[f"{name}_synthetic_test"] = class_report(y_test, model.predict(x_test))
        if x_orig is not None and y_orig is not None:
            pred_orig = model.predict(x_orig)
            rows[f"{name}_original_all_10s+_report_only"] = class_report(y_orig, pred_orig)
            if orig_split is not None:
                mask = np.asarray(orig_split).astype(str) == "test"
                if mask.any():
                    rows[f"{name}_original_test_all_10s+_report_only"] = class_report(y_orig[mask], pred_orig[mask])
    return rows


def render_markdown(summary: dict[str, Any], rows: pd.DataFrame) -> str:
    hard = rows[rows["target"].isin(["qrs_visibility", "detector_agreement", "baseline_step", "sqi_basSQI", "pc2", "pc3", "boundary_confidence", "knn_label_purity"])]
    table_cols = ["target", "category", "max_abs_pearson", "best_primitive", "ridge_r2", "trees_r2", "interpretation"]
    table_rows = hard.sort_values("trees_r2", ascending=False)[table_cols].copy()
    for col in ["max_abs_pearson", "ridge_r2", "trees_r2"]:
        table_rows[col] = table_rows[col].map(lambda x: "" if not np.isfinite(float(x)) else f"{float(x):.4f}")
    header = "| " + " | ".join(table_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(table_cols)) + " |"
    body = [
        "| " + " | ".join(str(row[col]).replace("|", "/") for col in table_cols) + " |"
        for _, row in table_rows.iterrows()
    ]
    hard_table = "\n".join([header, sep] + body)
    lines = [
        "# Waveform Primitive Learnability Audit",
        "",
        f"Created: {summary['created_at']}",
        "",
        "This audit uses synthetic/PTB waveform-derived primitive stats only. Original BUT rows are report-only.",
        "",
        "## Key Hard Targets",
        "",
        hard_table,
        "",
        "## Primitive Classifier Diagnostic",
        "",
        "These are diagnostic waveform-primitive baselines, not final models.",
        "",
        "```json",
        json.dumps(summary["primitive_classifier"], indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Outputs",
        "",
        f"- CSV: `{summary['csv']}`",
        f"- JSON: `{summary['json']}`",
    ]
    return "\n".join(lines) + "\n"


def target_category(name: str) -> str:
    if name in {"qrs_visibility", "detector_agreement", "qrs_band_ratio", "template_corr", "qrs_prom_p90"}:
        return "waveform_qrs_detector"
    if name in {"baseline_step", "flatline_ratio", "sqi_basSQI", "low_amp_ratio"}:
        return "waveform_baseline_flatline"
    if name in {"non_qrs_diff_p95", "mean_abs", "rms", "amplitude_entropy", "sqi_bSQI", "sqi_sSQI", "sqi_kSQI"}:
        return "waveform_detail_moment"
    if name in {"pc1", "pc2", "pc3", "pca_margin", "boundary_confidence", "knn_label_purity"}:
        return "atlas_geometry_proxy"
    return "other"


def interpretation(max_corr: float, trees_r2: float, category: str) -> str:
    if category == "atlas_geometry_proxy" and trees_r2 < 0.45:
        return "weak waveform fact; treat as geometry proxy, not primary aux target"
    if trees_r2 >= 0.65 or max_corr >= 0.70:
        return "strong waveform-computable target; architecture should learn it"
    if trees_r2 >= 0.35 or max_corr >= 0.50:
        return "partly learnable; use explicit task token/local head"
    return "weakly learnable from current primitives; redesign target or generator labels"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="qrs_stress_v5")
    parser.add_argument("--channels", default="robust3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--skip-original", action="store_true")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = ARCH.SyntheticWaveDataset("train", args.channels)
    val_ds = ARCH.SyntheticWaveDataset("val", args.channels)
    test_ds = ARCH.SyntheticWaveDataset("test", args.channels)
    norm = ARCH.FeatureNorm(train_ds.mean, train_ds.std)

    print("collecting synthetic primitive stats...", flush=True)
    x_train_raw = collect_primitives(train_ds, args.bank, args.batch_size)
    x_val_raw = collect_primitives(val_ds, args.bank, args.batch_size)
    x_test_raw = collect_primitives(test_ds, args.bank, args.batch_size)
    x_train, x_val, x_test = standardize(x_train_raw, x_val_raw, x_test_raw)
    x_trainval = np.concatenate([x_train, x_val], axis=0)
    y_trainval = np.concatenate([train_ds.y, val_ds.y], axis=0)
    aux_trainval = np.concatenate([train_ds.aux, val_ds.aux], axis=0)

    original_ds = None
    x_orig = None
    if not args.skip_original:
        try:
            original_ds = ARCH.OriginalWaveDataset(norm, args.channels)
            print("collecting original primitive stats (report-only)...", flush=True)
            x_orig_raw = collect_primitives(original_ds, args.bank, args.batch_size)
            x_orig = standardize(x_train_raw, x_orig_raw)[1]
        except Exception as exc:  # noqa: BLE001
            print(f"original primitive collection skipped: {exc}", flush=True)
            original_ds = None
            x_orig = None

    primitive_names = primitive_column_names(args.bank, x_train.shape[1])
    rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        if target not in FEATURE_COLUMNS:
            continue
        idx = FEATURE_COLUMNS.index(target)
        y_tr = aux_trainval[:, idx]
        y_te = test_ds.aux[:, idx]
        max_corr, corr_idx, signed_corr = max_abs_pearson(x_trainval, y_tr)
        pred = fit_predictors(x_trainval, x_test, y_tr, y_te, args.seed + idx)
        tree_idx = int(pred["tree_top_idx"])
        rows.append(
            {
                "target": target,
                "category": target_category(target),
                "max_abs_pearson": max_corr,
                "signed_pearson": signed_corr,
                "best_primitive_idx": corr_idx,
                "best_primitive": primitive_names[corr_idx] if 0 <= corr_idx < len(primitive_names) else "",
                "ridge_r2": pred["ridge_r2"],
                "trees_r2": pred["trees_r2"],
                "tree_top_idx": tree_idx,
                "tree_top_primitive": primitive_names[tree_idx] if 0 <= tree_idx < len(primitive_names) else "",
                "tree_top_importance": pred["tree_top_importance"],
                "interpretation": interpretation(max_corr, float(pred["trees_r2"]), target_category(target)),
            }
        )

    row_df = pd.DataFrame(rows).sort_values(["category", "trees_r2"], ascending=[True, False])
    classifier = primitive_classifier_audit(
        x_trainval,
        y_trainval,
        x_test,
        test_ds.y,
        x_orig,
        original_ds.y if original_ds is not None else None,
        original_ds.split if original_ds is not None else None,
        args.seed,
    )

    prefix = f"waveform_primitive_learnability_{args.bank}_{args.channels}"
    csv_path = ANALYSIS_DIR / f"{prefix}.csv"
    json_path = ANALYSIS_DIR / f"{prefix}.json"
    md_path = ANALYSIS_DIR / f"{prefix}.md"
    report_md_path = REPORT_DIR / md_path.name
    row_df.to_csv(csv_path, index=False)
    summary = {
        "created_at": now(),
        "bank": args.bank,
        "channels": args.channels,
        "variant_id": ARCH.VARIANT_ID,
        "synthetic_train_rows": int(len(train_ds)),
        "synthetic_val_rows": int(len(val_ds)),
        "synthetic_test_rows": int(len(test_ds)),
        "original_rows_report_only": int(len(original_ds)) if original_ds is not None else 0,
        "primitive_dim": int(x_train.shape[1]),
        "csv": str(csv_path),
        "json": str(json_path),
        "report": str(report_md_path),
        "primitive_classifier": classifier,
        "top_rows": row_df.head(12).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_markdown(summary, row_df)
    md_path.write_text(report, encoding="utf-8")
    report_md_path.write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
