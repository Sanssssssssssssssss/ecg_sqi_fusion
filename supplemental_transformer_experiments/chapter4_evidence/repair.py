from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import Paths, dry, ensure_dirs, feature_cols, write_json
from .seta_sqi import ARMS, arm_dir


def _xy(df_a: pd.DataFrame, df_b: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    a = df_a[cols].to_numpy(dtype=np.float64)
    b = df_b[cols].to_numpy(dtype=np.float64)
    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


def _c2st_auc(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> float:
    x, y = _xy(a, b, cols)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
        return float("nan")
    cv = StratifiedKFold(n_splits=min(5, int(min(np.bincount(y)))), shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
    score = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, score))
    return max(auc, 1.0 - auc)


def _rbf_mmd(a: np.ndarray, b: np.ndarray) -> float:
    x = np.vstack([a, b])
    sample = x[np.linspace(0, len(x) - 1, min(len(x), 500)).astype(int)]
    d = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(med, 1e-9)

    def k(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * ((u[:, None, :] - v[None, :, :]) ** 2).sum(axis=2))

    return float(k(a, a).mean() + k(b, b).mean() - 2.0 * k(a, b).mean())


def _swd(a: np.ndarray, b: np.ndarray, *, seed: int = 0, n_proj: int = 128) -> float:
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    out = []
    q = np.linspace(0.01, 0.99, 99)
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        qa = np.quantile(a @ v, q)
        qb = np.quantile(b @ v, q)
        out.append(float(np.mean(np.abs(qa - qb))))
    return float(np.mean(out))


def _cdf_gap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    gaps = []
    for j in range(a.shape[1]):
        vals = np.sort(np.unique(np.r_[a[:, j], b[:, j]]))
        if len(vals) == 0:
            continue
        ca = np.searchsorted(np.sort(a[:, j]), vals, side="right") / max(1, len(a))
        cb = np.searchsorted(np.sort(b[:, j]), vals, side="right") / max(1, len(b))
        gaps.append(float(np.max(np.abs(ca - cb))))
    return float(np.mean(gaps)), float(np.max(gaps))


def _standardize_against(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    aa = np.nan_to_num(a[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    bb = np.nan_to_num(b[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = aa.mean(axis=0)
    sd = aa.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (aa - mu) / sd, (bb - mu) / sd


def _pca_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(np.vstack([a, b]))
    za = z[: len(a)]
    zb = z[len(a) :]
    lo = np.quantile(za, 0.05, axis=0)
    hi = np.quantile(za, 0.95, axis=0)
    inside = ((zb >= lo) & (zb <= hi)).all(axis=1)
    return float(inside.mean()), float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])


def _metric_row(name: str, original: pd.DataFrame, generated: pd.DataFrame, cols: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    a, b = _standardize_against(original, generated, cols)
    mean_gap, max_gap = _cdf_gap(a, b)
    pca_overlap, pc1, pc2 = _pca_overlap(a, b)
    drift = np.abs(b.mean(axis=0) - a.mean(axis=0))
    top = pd.DataFrame({"construction": name, "feature": cols, "abs_z_mean_gap": drift}).sort_values("abs_z_mean_gap", ascending=False)
    row = {
        "construction": name,
        "n_original": int(len(original)),
        "n_generated": int(len(generated)),
        "c2st_auc": _c2st_auc(original, generated, cols),
        "rbf_mmd": _rbf_mmd(a, b),
        "swd": _swd(a, b, seed=7),
        "cdf_gap_mean": mean_gap,
        "cdf_gap_max": max_gap,
        "pca_overlap": pca_overlap,
        "pc1_var": pc1,
        "pc2_var": pc2,
    }
    return row, top.head(20)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    grid = np.quantile(score, np.linspace(0.01, 0.99, 199))
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        acc = float(accuracy_score(y, (score >= t).astype(int)))
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t


def _transfer_row(name: str, df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    acc_train = df[df["split"].eq("train") & df["y"].eq(1) & df["generated"].eq(0)]
    poor_train = df[df["split"].eq("train") & df["y"].eq(0)]
    if name not in {"native_imbalanced"}:
        poor_train = poor_train[poor_train["generated"].eq(1)]
    val = df[df["split"].eq("val") & df["generated"].eq(0)]
    test = df[df["split"].eq("test") & df["generated"].eq(0)]
    if len(acc_train) == 0 or len(poor_train) == 0:
        return {"construction": name, "status": "missing_train_source"}
    train_x = np.vstack([acc_train[cols].to_numpy(float), poor_train[cols].to_numpy(float)])
    train_y = np.r_[np.zeros(len(acc_train), dtype=int), np.ones(len(poor_train), dtype=int)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(np.nan_to_num(train_x), train_y)
    val_y = (val["y"].astype(int).to_numpy() == 0).astype(int)
    val_score = clf.predict_proba(np.nan_to_num(val[cols].to_numpy(float)))[:, 1]
    threshold = _best_threshold(val_y, val_score)
    test_y = (test["y"].astype(int).to_numpy() == 0).astype(int)
    test_score = clf.predict_proba(np.nan_to_num(test[cols].to_numpy(float)))[:, 1]
    pred = (test_score >= threshold).astype(int)
    poor_mask = test_y == 1
    return {
        "construction": name,
        "train_poor_source_n": int(len(poor_train)),
        "threshold_source": "validation_max_accuracy",
        "threshold": threshold,
        "test_original_poor_auc": float(roc_auc_score(test_y, test_score)),
        "original_poor_recall": float(pred[poor_mask].mean()) if np.any(poor_mask) else float("nan"),
        "acceptable_recall": float((pred[~poor_mask] == 0).mean()) if np.any(~poor_mask) else float("nan"),
    }


def _paired_mmd(name: str, original: pd.DataFrame, generated: pd.DataFrame, cols: list[str], *, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    a, b = _standardize_against(original, generated, cols)
    n = min(len(a), len(b), 128)
    within, cross = [], []
    for _ in range(100):
        ia = rng.choice(len(a), size=n, replace=True)
        ib = rng.choice(len(a), size=n, replace=True)
        ig = rng.choice(len(b), size=n, replace=True)
        within.append(_rbf_mmd(a[ia], a[ib]))
        cross.append(_rbf_mmd(a[ia], b[ig]))
    within = np.asarray(within)
    cross = np.asarray(cross)
    return {
        "comparison": name,
        "within_original_mmd_median": float(np.median(within)),
        "within_original_mmd_p05": float(np.quantile(within, 0.05)),
        "within_original_mmd_p95": float(np.quantile(within, 0.95)),
        "cross_mmd_median": float(np.median(cross)),
        "cross_mmd_p05": float(np.quantile(cross, 0.05)),
        "cross_mmd_p95": float(np.quantile(cross, 0.95)),
        "delta_gt_0_fraction": float(np.mean(cross > within)),
    }


def run(paths: Paths, *, execute: bool, force: bool) -> dict[str, Any]:
    if not execute:
        dry("seta-repair", paths)
        return {"step": "seta-repair", "skipped": True}
    ensure_dirs(paths)
    metrics_out = paths.tables / "seta_distribution_repair_metrics.csv"
    if metrics_out.exists() and not force:
        print(f"[seta-repair] exists: {metrics_out}")
        return {"step": "seta-repair", "skipped": True}
    metric_rows, drift_rows, transfer_rows, paired_rows = [], [], [], []
    for arm in ARMS:
        df = pd.read_csv(arm_dir(paths, arm) / "construction_features.csv")
        cols = feature_cols(df)
        original = df[df["split"].eq("train") & df["y"].eq(0) & df["generated"].eq(0)]
        generated = df[df["split"].eq("train") & df["y"].eq(0) & df["generated"].eq(1)]
        if arm == "native_imbalanced":
            generated = original.sample(n=383, replace=True, random_state=17).reset_index(drop=True)
            arm_name = "within_original_resample"
        else:
            arm_name = arm
        row, top = _metric_row(arm_name, original.reset_index(drop=True), generated.reset_index(drop=True), cols)
        metric_rows.append(row)
        drift_rows.append(top)
        paired_rows.append(_paired_mmd(arm_name, original.reset_index(drop=True), generated.reset_index(drop=True), cols))
        transfer_rows.append(_transfer_row(arm, df, cols))

    metrics = pd.DataFrame(metric_rows)
    drift = pd.concat(drift_rows, ignore_index=True)
    transfer = pd.DataFrame(transfer_rows)
    paired = pd.DataFrame(paired_rows)
    metrics.to_csv(metrics_out, index=False)
    drift.to_csv(paths.tables / "seta_top_drift_features.csv", index=False)
    transfer.to_csv(paths.tables / "seta_source_transfer.csv", index=False)
    paired.to_csv(paths.tables / "seta_paired_mmd_calibration.csv", index=False)
    out = {
        "distribution_metrics": metrics.to_dict(orient="records"),
        "transfer": transfer.to_dict(orient="records"),
        "paired_mmd": paired.to_dict(orient="records"),
    }
    write_json(paths.repair_json, out)
    print(metrics.to_string(index=False))
    return {"step": "seta-repair", "skipped": False, "outputs": out}

