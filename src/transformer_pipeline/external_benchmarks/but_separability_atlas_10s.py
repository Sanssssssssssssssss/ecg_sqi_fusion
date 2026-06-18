"""BUT 10s three-class separability atlas.

This is an analysis-only runner.  It does not train models, modify
``src/sqi_pipeline``, or touch mainline checkpoints.  The goal is to discover
which interpretable dimensions separate BUT good/medium/bad before designing
the next synthetic generator.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text

try:  # optional but present in the project venv
    import pywt
except Exception:  # pragma: no cover
    pywt = None

try:  # optional
    import umap
except Exception:  # pragma: no cover
    umap = None

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import analyze_but_morphology_clusters as morph
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import balanced_but_test_indices


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_separability_atlas_10s_2026_06_06"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_BUT_PROTOCOL = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_protocol_adaptation_2026_06_03"
    / "protocols"
    / "p1_current_10s_center"
)
DEFAULT_MORPH_CACHE = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_medium_guard_bad_boundary_analysis_10s_2026_06_05"
    / "but_morph_features.csv"
)
DEFAULT_SQI_CACHE = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_medium_guard_bad_boundary_analysis_10s_2026_06_05"
    / "but_sqi_features.csv"
)
DEFAULT_CLUSTER_OUT = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_morphology_cluster_analysis_10s_2026_06_04"
)
DEFAULT_SQI_GAP_OUT = ROOT / "outputs" / "external_benchmarks" / "e311_but_sqi_gap_analysis_10s_2026_06_04"
DEFAULT_DIRECTION_OUT = ROOT / "outputs" / "external_benchmarks" / "e311_but_big_uformer_long_search_10s_2026_06_06"
DEFAULT_MEDIUM_GUARD_OUT = (
    ROOT / "outputs" / "external_benchmarks" / "e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05"
)

CLASS_NAMES = ("good", "medium", "bad")
FS = 125.0
N_TARGET = 1250
PAIRWISE = (("good", "medium"), ("medium", "bad"), ("good", "bad"))
SQI_RENAME = {
    "I__iSQI": "sqi_iSQI",
    "I__bSQI": "sqi_bSQI",
    "I__pSQI": "sqi_pSQI",
    "I__sSQI": "sqi_sSQI",
    "I__kSQI": "sqi_kSQI",
    "I__fSQI": "sqi_fSQI",
    "I__basSQI": "sqi_basSQI",
}
NON_FEATURE_COLUMNS = {
    "idx",
    "y",
    "label_raw",
    "record_id",
    "subject_id",
    "window_id",
    "source_idx",
    "patient_id",
    "split_code",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4g}")
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in view.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_but(protocol_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(protocol_dir / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(protocol_dir / "metadata.csv")
    if X.ndim == 2:
        X = X[:, None, :]
    if X.shape != (32956, 1, N_TARGET):
        raise ValueError(f"BUT shape changed: expected (32956,1,{N_TARGET}), got {X.shape}")
    if "y" not in meta.columns:
        meta["y"] = meta["y_class"].map({"good": 0, "medium": 1, "bad": 2}).astype(int)
    if "class_name" not in meta.columns:
        if "y_class" in meta.columns:
            meta["class_name"] = meta["y_class"].astype(str)
        else:
            meta["class_name"] = meta["y"].map(lambda v: CLASS_NAMES[int(v)])
    split_counts = meta["split"].astype(str).value_counts().to_dict()
    expected = {"train": 23322, "val": 1157, "test": 8477}
    if {k: int(split_counts.get(k, 0)) for k in expected} != expected:
        raise ValueError(f"BUT split counts changed: expected {expected}, got {split_counts}")
    meta = meta.reset_index(drop=True)
    meta["idx"] = np.arange(len(meta), dtype=int)
    return X, meta


def _safe_entropy(x: np.ndarray, bins: int = 40) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(float)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-np.sum(p * np.log(p)) / math.log(max(2, len(p))))


def _zero_crossing_rate(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(x)) != 0))


def _higuchi_fd(x: np.ndarray, kmax: int = 6) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < kmax * 3:
        return 1.0
    lk: list[float] = []
    kk: list[float] = []
    for k in range(1, kmax + 1):
        vals = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            dist = np.sum(np.abs(np.diff(x[idx])))
            norm = (n - 1) / (len(idx) * k)
            vals.append(dist * norm / k)
        if vals:
            lk.append(float(np.mean(vals)))
            kk.append(float(1.0 / k))
    if len(lk) < 2 or min(lk) <= 0:
        return 1.0
    slope, _ = np.polyfit(np.log(kk), np.log(lk), 1)
    return float(max(0.0, min(3.0, slope)))


def _template_corr(y: np.ndarray, peaks: np.ndarray) -> float:
    half = int(0.18 * FS)
    beats = []
    for p in peaks[:30]:
        lo = int(p) - half
        hi = int(p) + half
        if lo >= 0 and hi <= len(y):
            b = y[lo:hi].astype(float)
            b = (b - np.mean(b)) / (np.std(b) + 1e-8)
            beats.append(b)
    if len(beats) < 3:
        return 0.0
    B = np.vstack(beats)
    tmpl = np.mean(B, axis=0)
    tmpl = (tmpl - np.mean(tmpl)) / (np.std(tmpl) + 1e-8)
    corr = np.mean(np.sum(B * tmpl[None, :], axis=1) / (B.shape[1] + 1e-8))
    return float(np.clip(corr, -1.0, 1.0))


def extract_extra_one(x: np.ndarray) -> dict[str, float]:
    y = np.asarray(x, dtype=float).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - np.median(y)
    dy = np.diff(y)
    ddy = np.diff(dy)
    eps = 1e-8
    var0 = float(np.var(y) + eps)
    var1 = float(np.var(dy) + eps)
    var2 = float(np.var(ddy) + eps)
    mobility = math.sqrt(var1 / var0)
    complexity = math.sqrt(var2 / var1) / (mobility + eps)

    abs_y = np.abs(y)
    peaks_a, _ = find_peaks(abs_y, distance=int(0.28 * FS), prominence=max(0.08 * np.std(y), 0.01))
    peaks_b, _ = find_peaks(np.maximum(y, 0), distance=int(0.28 * FS), prominence=max(0.06 * np.std(y), 0.008))
    peaks_c, _ = find_peaks(-np.minimum(y, 0), distance=int(0.28 * FS), prominence=max(0.06 * np.std(y), 0.008))
    detector_counts = np.asarray([len(peaks_a), len(peaks_b), len(peaks_c)], dtype=float)
    detector_agreement = float(1.0 / (1.0 + np.std(detector_counts)))

    freqs = np.fft.rfftfreq(len(y), d=1.0 / FS)
    power = np.abs(np.fft.rfft(y)) ** 2
    total = float(np.sum(power[(freqs >= 0.3) & (freqs <= 45.0)]) + eps)
    band = lambda lo, hi: float(np.sum(power[(freqs >= lo) & (freqs < hi)]) / total)

    if pywt is not None:
        coeffs = pywt.wavedec(y, "db4", level=4)
        energies = [float(np.mean(c**2)) for c in coeffs]
        denom = sum(energies) + eps
        wav = {f"wavelet_e{i}": float(e / denom) for i, e in enumerate(energies)}
    else:
        wav = {f"wavelet_e{i}": 0.0 for i in range(5)}

    qrs_mask = np.zeros(len(y), dtype=bool)
    half = int(0.12 * FS)
    for p in peaks_a:
        qrs_mask[max(0, int(p) - half) : min(len(y), int(p) + half)] = True
    non_qrs = y[~qrs_mask] if np.any(~qrs_mask) else y
    qrs = y[qrs_mask] if np.any(qrs_mask) else y

    return {
        "hjorth_activity": var0,
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
        "zero_crossing_rate": _zero_crossing_rate(y),
        "diff_zero_crossing_rate": _zero_crossing_rate(dy),
        "sample_entropy_proxy": _safe_entropy(dy),
        "amplitude_entropy": _safe_entropy(y),
        "higuchi_fd_proxy": _higuchi_fd(y),
        "template_corr": _template_corr(y, peaks_a),
        "detector_count_std": float(np.std(detector_counts)),
        "detector_agreement": detector_agreement,
        "rr_count_detector_a": float(len(peaks_a)),
        "rr_count_detector_b": float(len(peaks_b)),
        "rr_count_detector_c": float(len(peaks_c)),
        "non_qrs_rms_ratio": float(np.sqrt(np.mean(non_qrs**2)) / (np.sqrt(np.mean(qrs**2)) + eps)),
        "non_qrs_diff_p95": float(np.percentile(np.abs(np.diff(non_qrs)), 95)) if len(non_qrs) > 4 else 0.0,
        "band_0p3_1": band(0.3, 1.0),
        "band_1_5": band(1.0, 5.0),
        "band_5_15": band(5.0, 15.0),
        "band_15_30": band(15.0, 30.0),
        "band_30_45": band(30.0, 45.0),
        **wav,
    }


def extract_extra_features(X: np.ndarray, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = []
    for i in range(len(X)):
        rows.append(extract_extra_one(X[i, 0]))
        if (i + 1) % 5000 == 0:
            print(f"extra features {i + 1}/{len(X)}", flush=True)
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_feature_matrix(args: argparse.Namespace, X: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_root = Path(args.out_root)
    morph_path = Path(args.morph_cache)
    sqi_path = Path(args.sqi_cache)
    if morph_path.exists():
        morph_df = pd.read_csv(morph_path)
    else:
        morph_df = morph.extract_features(X, meta, out_root / "but_morph_features.csv")
    if sqi_path.exists():
        sqi_df = pd.read_csv(sqi_path).rename(columns=SQI_RENAME)
    else:
        raise FileNotFoundError(f"SQI cache not found: {sqi_path}")
    extra_df = extract_extra_features(X, out_root / "but_extra_features.csv")

    base = meta[["idx", "split", "y", "class_name", "record_id", "subject_id"]].copy()
    frames = [base]
    for df in (morph_df, sqi_df, extra_df):
        keep = df.select_dtypes(include=[np.number]).copy()
        for col in NON_FEATURE_COLUMNS:
            if col in keep.columns:
                keep = keep.drop(columns=[col])
        frames.append(keep.reset_index(drop=True))
    full = pd.concat(frames, axis=1)
    full = full.loc[:, ~full.columns.duplicated()]
    numeric = full.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if c not in NON_FEATURE_COLUMNS]
    feature_meta = build_feature_metadata(feature_cols)
    full.to_csv(out_root / "but_feature_matrix.csv", index=False)
    feature_meta.to_csv(out_root / "feature_metadata.csv", index=False)
    return full, feature_meta


def build_feature_metadata(features: list[str]) -> pd.DataFrame:
    def group(col: str) -> str:
        if col.startswith("sqi_"):
            return "SQI"
        if col in {
            "qrs_band_ratio",
            "qrs_peak_count",
            "aggressive_peak_count",
            "spurious_peak_density",
            "qrs_prom_median",
            "qrs_prom_p90",
            "qrs_width_median",
            "qrs_slope_median",
            "qrs_count_low",
            "qrs_count_high",
            "qrs_count_deviation",
            "periodicity",
            "qrs_visibility",
            "detector_count_std",
            "detector_agreement",
            "rr_count_detector_a",
            "rr_count_detector_b",
            "rr_count_detector_c",
            "template_corr",
        }:
            return "QRS detectability"
        if col in {
            "detail_instability",
            "medium_detail_unreliable_score",
            "diff_abs_median",
            "diff_abs_p95",
            "non_qrs_rms_ratio",
            "non_qrs_diff_p95",
            "sample_entropy_proxy",
            "amplitude_entropy",
            "higuchi_fd_proxy",
        }:
            return "Detail reliability"
        if col in {
            "low_amp_ratio",
            "flatline_ratio",
            "contact_loss_win_ratio",
            "clipping_like_ratio",
            "fatal_or_score",
            "local_rms_cv",
            "baseline_step",
        }:
            return "Contact/flat/fatal"
        if col.startswith("band_") or col.startswith("wavelet_") or col in {
            "lf_ratio",
            "hf_ratio",
            "spectral_entropy",
            "zero_crossing_rate",
            "diff_zero_crossing_rate",
            "hjorth_activity",
            "hjorth_mobility",
            "hjorth_complexity",
        }:
            return "Motion/frequency"
        if col in {"rms", "std", "ptp_p99_p01", "mean_abs"}:
            return "Amplitude/global"
        return "Other morphology"

    rows = []
    for col in features:
        g = group(col)
        rows.append(
            {
                "feature": col,
                "family": g,
                "scope": "inference_safe" if g != "Latent/model" else "diagnostic_only",
                "description": g,
            }
        )
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 2500) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return math.nan
    rng = np.random.default_rng(17)
    if len(a) > max_n:
        a = rng.choice(a, size=max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, size=max_n, replace=False)
    comp = a[:, None] - b[None, :]
    return float((np.sum(comp > 0) - np.sum(comp < 0)) / comp.size)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    pooled = math.sqrt(((len(a) - 1) * np.var(a) + (len(b) - 1) * np.var(b)) / (len(a) + len(b) - 2) + 1e-8)
    return float((np.mean(a) - np.mean(b)) / pooled)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    q[valid] = out
    return q


def pairwise_feature_tests(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    y_by_class = {cls: df[df["class_name"] == cls] for cls in CLASS_NAMES}
    rows: list[dict[str, Any]] = []
    for feat in features:
        groups = [y_by_class[cls][feat].to_numpy(dtype=float) for cls in CLASS_NAMES]
        try:
            kruskal_p = float(stats.kruskal(*[g[np.isfinite(g)] for g in groups]).pvalue)
        except Exception:
            kruskal_p = math.nan
        try:
            anova_p = float(stats.f_oneway(*[g[np.isfinite(g)] for g in groups]).pvalue)
        except Exception:
            anova_p = math.nan
        for a_cls, b_cls in PAIRWISE:
            a = y_by_class[a_cls][feat].to_numpy(dtype=float)
            b = y_by_class[b_cls][feat].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if len(a) == 0 or len(b) == 0:
                continue
            try:
                ks = stats.ks_2samp(a, b)
                ks_stat = float(ks.statistic)
                ks_p = float(ks.pvalue)
            except Exception:
                ks_stat = ks_p = math.nan
            try:
                mw_p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                mw_p = math.nan
            rows.append(
                {
                    "feature": feat,
                    "contrast": f"{a_cls}_vs_{b_cls}",
                    "class_a": a_cls,
                    "class_b": b_cls,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "median_a": float(np.median(a)),
                    "median_b": float(np.median(b)),
                    "iqr_a": float(np.percentile(a, 75) - np.percentile(a, 25)),
                    "iqr_b": float(np.percentile(b, 75) - np.percentile(b, 25)),
                    "median_delta_a_minus_b": float(np.median(a) - np.median(b)),
                    "cliffs_delta": cliffs_delta(a, b),
                    "cohens_d": cohens_d(a, b),
                    "ks": ks_stat,
                    "wasserstein": float(stats.wasserstein_distance(a, b)),
                    "mw_p": mw_p,
                    "ks_p": ks_p,
                    "kruskal_p_global": kruskal_p,
                    "anova_p_global": anova_p,
                }
            )
    out = pd.DataFrame(rows)
    for col in ["mw_p", "ks_p", "kruskal_p_global", "anova_p_global"]:
        out[f"{col}_fdr"] = fdr_bh(out[col].to_numpy(dtype=float))
    return out.sort_values(["contrast", "ks"], ascending=[True, False])


def add_mutual_info(df: pd.DataFrame, feature_tests: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sample = df.sample(min(len(df), 25000), random_state=29)
    X = sample[features].replace([np.inf, -np.inf], np.nan).fillna(sample[features].median(numeric_only=True)).to_numpy()
    y = sample["y"].to_numpy(dtype=int)
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=29)
    mi_df = pd.DataFrame({"feature": features, "mutual_info": mi})
    return feature_tests.merge(mi_df, on="feature", how="left")


def balanced_indices(df: pd.DataFrame, seed: int = 20260606) -> np.ndarray:
    test_mask = df["split"].astype(str).to_numpy() == "test"
    test_pos = np.flatnonzero(test_mask)
    test = df.iloc[test_pos]
    n = int(test["y"].value_counts().min())
    rng = np.random.default_rng(seed)
    ids: list[int] = []
    for cls in range(3):
        arr = test_pos[test["y"].astype(int).to_numpy() == cls]
        ids.extend(rng.choice(arr, size=n, replace=False).tolist())
    return np.asarray(sorted(ids), dtype=int)


def report_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "recall_good_medium_bad": [
            float(np.mean(y_pred[y_true == i] == i)) if np.any(y_true == i) else math.nan for i in labels
        ],
        "confusion_3x3": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def run_probes(df: pd.DataFrame, features: list[str], out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    use = df[["idx", "split", "y", "record_id", "subject_id", *features]].copy()
    use = use.loc[:, ~use.columns.duplicated()].reset_index(drop=True)
    features = [c for c in features if c in use.columns and c not in NON_FEATURE_COLUMNS]
    X_frame = use[features].replace([np.inf, -np.inf], np.nan)
    X_frame = X_frame.fillna(X_frame.median(numeric_only=True)).fillna(0.0)
    X_all = X_frame.to_numpy(dtype=np.float32)
    y_all = use["y"].to_numpy(dtype=int)
    split_arr = use["split"].astype(str).to_numpy()
    train_idx = np.flatnonzero(split_arr == "train")
    test_idx = np.flatnonzero(split_arr == "test")
    bal_idx = balanced_indices(use)
    groups = use["subject_id"].fillna(use["record_id"]).astype(str).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=23)
    g_train, g_test = next(gss.split(X_all, y_all, groups=groups))

    models: list[tuple[str, Any, bool]] = [
        (
            "elastic_net_logreg",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            l1_ratio=0.55,
                            class_weight="balanced",
                            max_iter=2500,
                            n_jobs=-1,
                            random_state=31,
                        ),
                    ),
                ]
            ),
            False,
        ),
        (
            "linear_svc",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LinearSVC(class_weight="balanced", C=0.6, max_iter=6000, random_state=32)),
                ]
            ),
            False,
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=320,
                max_depth=12,
                min_samples_leaf=18,
                class_weight="balanced",
                n_jobs=-1,
                random_state=33,
            ),
            True,
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=360,
                max_depth=14,
                min_samples_leaf=14,
                class_weight="balanced",
                n_jobs=-1,
                random_state=34,
            ),
            True,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(max_iter=180, l2_regularization=0.03, random_state=35),
            False,
        ),
        (
            "small_decision_tree",
            DecisionTreeClassifier(max_depth=4, min_samples_leaf=80, class_weight="balanced", random_state=36),
            True,
        ),
    ]
    rows: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []
    tree_rules = ""
    for name, model, has_imp in models:
        for split_name, tr, te in [
            ("protocol_train_to_test", train_idx, test_idx),
            ("grouped_subject_holdout", g_train, g_test),
        ]:
            tr = np.asarray(tr, dtype=int)
            te = np.asarray(te, dtype=int)
            tr = tr[(tr >= 0) & (tr < len(X_all))]
            te = te[(te >= 0) & (te < len(X_all))]
            safe_bal_idx = bal_idx[(bal_idx >= 0) & (bal_idx < len(X_all))]
            model.fit(X_all[tr], y_all[tr])
            pred_test = model.predict(X_all[te])
            pred_bal = model.predict(X_all[safe_bal_idx])
            rep = report_for(y_all[te], pred_test)
            bal = report_for(y_all[safe_bal_idx], pred_bal)
            rows.append(
                {
                    "model": name,
                    "split_mode": split_name,
                    "acc": rep["acc"],
                    "balanced_acc": rep["balanced_acc"],
                    "macro_f1": rep["macro_f1"],
                    "good_recall": rep["recall_good_medium_bad"][0],
                    "medium_recall": rep["recall_good_medium_bad"][1],
                    "bad_recall": rep["recall_good_medium_bad"][2],
                    "balanced_diag_macro_f1": bal["macro_f1"],
                    "balanced_diag_balanced_acc": bal["balanced_acc"],
                }
            )
            if split_name == "protocol_train_to_test":
                if name == "small_decision_tree":
                    tree_rules = export_text(model, feature_names=features, max_depth=4)
                imp = None
                if has_imp and hasattr(model, "feature_importances_"):
                    imp = np.asarray(model.feature_importances_, dtype=float)
                elif name == "elastic_net_logreg":
                    clf = model.named_steps["clf"]
                    imp = np.mean(np.abs(clf.coef_), axis=0)
                if imp is not None:
                    imp = np.asarray(imp, dtype=float)
                    if imp.ndim > 1 and imp.shape[-1] == len(features):
                        imp = np.mean(np.abs(imp), axis=0)
                    imp = np.ravel(imp)
                    if len(imp) == len(features):
                        importances.append(
                            pd.DataFrame({"model": name, "feature": features, "importance": imp}).sort_values(
                                "importance", ascending=False
                            )
                        )
    probe = pd.DataFrame(rows).sort_values(["split_mode", "macro_f1"], ascending=[True, False])
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    probe.to_csv(out_root / "model_probe_results.csv", index=False)
    imp_df.to_csv(out_root / "model_probe_importance.csv", index=False)
    (out_root / "decision_tree_surrogate.txt").write_text(tree_rules, encoding="utf-8")
    return probe, imp_df, tree_rules


def medium_independence(df: pd.DataFrame, features: list[str], out_root: Path, report_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    sample = df.sample(min(len(df), 12000), random_state=41).copy()
    X = sample[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median(numeric_only=True)).to_numpy()
    Z = StandardScaler().fit_transform(X)
    y = sample["y"].to_numpy(dtype=int)
    pca = PCA(n_components=8, random_state=41)
    P = pca.fit_transform(Z)
    sample["pc1"] = P[:, 0]
    sample["pc2"] = P[:, 1]

    centroids = {cls: P[y == i, :4].mean(axis=0) for i, cls in enumerate(CLASS_NAMES)}
    g = centroids["good"]
    m = centroids["medium"]
    b = centroids["bad"]
    gb = b - g
    t = float(np.dot(m - g, gb) / (np.dot(gb, gb) + 1e-8))
    closest = g + np.clip(t, 0.0, 1.0) * gb
    off_axis = float(np.linalg.norm(m - closest))
    good_bad_dist = float(np.linalg.norm(gb))
    medium_independence_ratio = float(off_axis / (good_bad_dist + 1e-8))

    nn = NearestNeighbors(n_neighbors=31)
    nn.fit(P[:, :8])
    neigh = nn.kneighbors(P[:, :8], return_distance=False)[:, 1:]
    local_medium_purity = float(np.mean(np.mean(y[neigh[y == 1]] == 1, axis=1)))
    local_same_label_purity = float(np.mean(np.mean(y[neigh] == y[:, None], axis=1)))

    cluster_rows: list[dict[str, Any]] = []
    labels_by_method: dict[str, np.ndarray] = {}
    for name, labels in [
        ("kmeans6", KMeans(n_clusters=6, random_state=41, n_init=20).fit_predict(Z)),
        ("gmm6", GaussianMixture(n_components=6, covariance_type="diag", random_state=42).fit_predict(Z)),
        ("agglomerative6", AgglomerativeClustering(n_clusters=6).fit_predict(P[:, :8])),
    ]:
        labels_by_method[name] = labels
        try:
            sil = float(silhouette_score(P[:, :8], labels))
        except Exception:
            sil = math.nan
        for c in sorted(np.unique(labels)):
            mask = labels == c
            comp = {cls: int(np.sum(y[mask] == i)) for i, cls in enumerate(CLASS_NAMES)}
            total = int(np.sum(mask))
            cluster_rows.append(
                {
                    "method": name,
                    "cluster": int(c),
                    "n": total,
                    "good_share": comp["good"] / max(1, total),
                    "medium_share": comp["medium"] / max(1, total),
                    "bad_share": comp["bad"] / max(1, total),
                    "dominant_class": max(comp, key=comp.get),
                    "silhouette_sample": sil,
                    "ari_vs_label": float(adjusted_rand_score(y, labels)),
                    "nmi_vs_label": float(normalized_mutual_info_score(y, labels)),
                }
            )
        sample[f"{name}_cluster"] = labels

    audit = {
        "pca_explained_var_first_2": [float(v) for v in pca.explained_variance_ratio_[:2]],
        "pca_explained_var_first_8": [float(v) for v in pca.explained_variance_ratio_[:8]],
        "medium_projection_on_good_bad_axis": t,
        "medium_off_axis_distance": off_axis,
        "good_bad_centroid_distance": good_bad_dist,
        "medium_independence_ratio": medium_independence_ratio,
        "local_medium_neighbor_purity_k30": local_medium_purity,
        "local_same_label_purity_k30": local_same_label_purity,
        "interpretation": (
            "medium_independent_or_mixed"
            if medium_independence_ratio > 0.12 or local_medium_purity > 0.55
            else "mostly_good_bad_continuum"
        ),
    }
    write_json(out_root / "medium_independence_tests.json", audit)
    cluster_df = pd.DataFrame(cluster_rows).sort_values(["method", "cluster"])
    sample.to_csv(out_root / "embedding_sample.csv", index=False)
    cluster_df.to_csv(out_root / "cluster_summary.csv", index=False)
    plot_embeddings(sample, report_root / "figures")
    plot_cluster_composition(cluster_df, report_root / "figures")
    return audit, cluster_df, sample


def prototype_manifest(df: pd.DataFrame, features: list[str], embed: pd.DataFrame, X: np.ndarray, report_root: Path, out_root: Path) -> pd.DataFrame:
    pcols = ["pc1", "pc2"]
    merged = df[["idx", "split", "y", "class_name", *features]].merge(embed[["idx", *pcols]], on="idx", how="left")
    rows: list[dict[str, Any]] = []
    scaler = StandardScaler()
    F = scaler.fit_transform(df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median(numeric_only=True)))
    for cls_id, cls in enumerate(CLASS_NAMES):
        idxs = df.index[df["y"].astype(int) == cls_id].to_numpy()
        center = F[idxs].mean(axis=0)
        dist = np.linalg.norm(F[idxs] - center[None, :], axis=1)
        for rank, local_pos in enumerate(np.argsort(dist)[:24], start=1):
            i = int(idxs[local_pos])
            rows.append(
                {
                    "prototype_type": "class_medoid",
                    "class_name": cls,
                    "rank": rank,
                    "idx": int(df.loc[i, "idx"]),
                    "split": str(df.loc[i, "split"]),
                    "distance": float(dist[local_pos]),
                }
            )
    proto = pd.DataFrame(rows)
    proto.to_csv(out_root / "prototype_manifest.csv", index=False)
    fig_dir = report_root / "figures" / "prototype_galleries"
    for cls in CLASS_NAMES:
        ids = proto[(proto["class_name"] == cls) & (proto["rank"] <= 18)]["idx"].to_numpy(dtype=int)
        plot_wave_gallery(X, ids, f"BUT {cls} medoid prototypes", fig_dir / f"but_{cls}_prototype_gallery.png")
    return proto


def model_results_summary(out_root: Path) -> pd.DataFrame:
    paths = [
        DEFAULT_MEDIUM_GUARD_OUT / "medium_guard_bad_boundary_summary.jsonl",
        DEFAULT_DIRECTION_OUT / "direction_validation_summary.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for r in read_jsonl(path):
            rep = (r.get("but_10s_eval") or {}).get("but_10s_test_report") or {}
            if not rep:
                continue
            rec = rep.get("recall_good_medium_bad") or [math.nan, math.nan, math.nan]
            spec = r.get("spec") or {}
            rows.append(
                {
                    "source_file": path.name,
                    "stage": r.get("direction_stage") or r.get("stage") or r.get("mode"),
                    "mode": r.get("mode"),
                    "seed": r.get("seed"),
                    "variant_id": spec.get("id") or r.get("variant_id"),
                    "acc": rep.get("acc"),
                    "balanced_acc": rep.get("balanced_acc"),
                    "macro_f1": rep.get("macro_f1"),
                    "good_recall": rec[0] if len(rec) > 0 else math.nan,
                    "medium_recall": rec[1] if len(rec) > 1 else math.nan,
                    "bad_recall": rec[2] if len(rec) > 2 else math.nan,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("macro_f1", ascending=False)
        df.to_csv(out_root / "model_result_context.csv", index=False)
    return df


def synthetic_gap(top_features: list[str], out_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    morph_dist = DEFAULT_CLUSTER_OUT / "morph_domain_distance.csv"
    sqi_dist = DEFAULT_SQI_GAP_OUT / "sqi_domain_distance.csv"
    if morph_dist.exists():
        m = pd.read_csv(morph_dist)
        m = m[m["feature"].isin(top_features)].copy()
        m["source"] = "morphology_cluster_analysis"
        rows.append(m)
    if sqi_dist.exists():
        s = pd.read_csv(sqi_dist)
        s["feature"] = s["feature"].map(lambda x: SQI_RENAME.get(str(x), str(x)))
        s = s[s["feature"].isin(top_features)].copy()
        s["source"] = "sqi_gap_analysis"
        rows.append(s)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_root / "synthetic_gap_on_separating_dimensions.csv", index=False)
    return out


def feature_family_scores(
    tests: pd.DataFrame, probe_imp: pd.DataFrame, feature_meta: pd.DataFrame, out_root: Path
) -> pd.DataFrame:
    base = (
        tests.groupby("feature")
        .agg(
            max_ks=("ks", "max"),
            max_abs_cliffs=("cliffs_delta", lambda x: float(np.nanmax(np.abs(x)))),
            max_abs_d=("cohens_d", lambda x: float(np.nanmax(np.abs(x)))),
            mutual_info=("mutual_info", "max"),
        )
        .reset_index()
    )
    if not probe_imp.empty:
        imp = probe_imp.groupby("feature")["importance"].mean().reset_index(name="mean_probe_importance")
        base = base.merge(imp, on="feature", how="left")
    else:
        base["mean_probe_importance"] = np.nan
    base = base.merge(feature_meta[["feature", "family", "scope"]], on="feature", how="left")
    for col in ["max_ks", "max_abs_cliffs", "max_abs_d", "mutual_info", "mean_probe_importance"]:
        vals = base[col].replace([np.inf, -np.inf], np.nan)
        denom = float(vals.max() - vals.min()) if vals.notna().any() else 0.0
        base[f"{col}_norm"] = (vals - vals.min()) / denom if denom > 0 else 0.0
    base["separability_score"] = (
        0.28 * base["max_ks_norm"].fillna(0)
        + 0.22 * base["max_abs_cliffs_norm"].fillna(0)
        + 0.18 * base["mutual_info_norm"].fillna(0)
        + 0.20 * base["mean_probe_importance_norm"].fillna(0)
        + 0.12 * base["max_abs_d_norm"].fillna(0)
    )
    fam = (
        base.groupby("family")
        .agg(
            n_features=("feature", "count"),
            mean_score=("separability_score", "mean"),
            max_score=("separability_score", "max"),
            top_features=("feature", lambda s: ", ".join(base.loc[s.index].sort_values("separability_score", ascending=False)["feature"].head(6))),
        )
        .reset_index()
        .sort_values("max_score", ascending=False)
    )
    base.sort_values("separability_score", ascending=False).to_csv(out_root / "feature_separability_scores.csv", index=False)
    fam.to_csv(out_root / "feature_family_scores.csv", index=False)
    return fam


def plot_wave_gallery(X: np.ndarray, indices: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(indices) == 0:
        return
    n = min(18, len(indices))
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.15 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    t = np.arange(N_TARGET) / FS
    for ax, idx in zip(axes, indices[:n]):
        ax.plot(t, X[int(idx), 0], lw=0.75, color="#2f4858")
        ax.set_title(f"idx {int(idx)}", fontsize=8)
        ax.grid(alpha=0.15)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, tests: pd.DataFrame, report_root: Path) -> None:
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = (
        tests.groupby("feature")["ks"]
        .max()
        .sort_values(ascending=False)
        .head(24)
        .index.tolist()
    )
    long = df.sample(min(len(df), 18000), random_state=51).melt(
        id_vars=["class_name"], value_vars=top, var_name="feature", value_name="value"
    )
    g = sns.catplot(
        data=long,
        x="class_name",
        y="value",
        col="feature",
        kind="box",
        col_wrap=4,
        sharey=False,
        showfliers=False,
        height=2.4,
        aspect=1.25,
        color="#8aa1b4",
    )
    g.fig.suptitle("Top BUT class-separating feature distributions", y=1.02)
    g.savefig(fig_dir / "top_feature_class_boxplots.png", dpi=170)
    plt.close(g.fig)

    heat = tests.pivot_table(index="feature", columns="contrast", values="cliffs_delta", aggfunc="max")
    order = tests.groupby("feature")["ks"].max().sort_values(ascending=False).head(35).index
    fig, ax = plt.subplots(figsize=(7.4, max(7.0, len(order) * 0.22)))
    sns.heatmap(heat.loc[order], cmap="vlag", center=0, ax=ax, linewidths=0.2)
    ax.set_title("Effect direction by class contrast (Cliff's delta)")
    fig.tight_layout()
    fig.savefig(fig_dir / "pairwise_effect_heatmap.png", dpi=180)
    plt.close(fig)


def plot_embeddings(sample: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sns.scatterplot(
        data=sample,
        x="pc1",
        y="pc2",
        hue="class_name",
        s=8,
        linewidth=0,
        alpha=0.6,
        palette={"good": "#2a9d8f", "medium": "#f4a261", "bad": "#c44536"},
        ax=ax,
    )
    ax.set_title("BUT PCA embedding by class")
    fig.tight_layout()
    fig.savefig(fig_dir / "but_pca_class.png", dpi=180)
    plt.close(fig)

    if umap is not None:
        feat_cols = [c for c in sample.columns if c not in {"idx", "split", "y", "class_name", "record_id", "subject_id"}]
        use_cols = [c for c in feat_cols if c not in {"pc1", "pc2"} and sample[c].dtype.kind in "fc"]
        if len(use_cols) > 4:
            X = sample[use_cols].replace([np.inf, -np.inf], np.nan).fillna(sample[use_cols].median(numeric_only=True))
            Z = StandardScaler().fit_transform(X)
            emb = umap.UMAP(n_neighbors=35, min_dist=0.08, metric="euclidean", random_state=52).fit_transform(Z)
            plot_df = sample[["class_name"]].copy()
            plot_df["umap1"] = emb[:, 0]
            plot_df["umap2"] = emb[:, 1]
            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            sns.scatterplot(
                data=plot_df,
                x="umap1",
                y="umap2",
                hue="class_name",
                s=8,
                linewidth=0,
                alpha=0.58,
                palette={"good": "#2a9d8f", "medium": "#f4a261", "bad": "#c44536"},
                ax=ax,
            )
            ax.set_title("BUT UMAP embedding by class")
            fig.tight_layout()
            fig.savefig(fig_dir / "but_umap_class.png", dpi=180)
            plt.close(fig)


def plot_cluster_composition(cluster_df: pd.DataFrame, fig_dir: Path) -> None:
    if cluster_df.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    for method in cluster_df["method"].unique():
        sub = cluster_df[cluster_df["method"] == method].copy()
        long = sub.melt(
            id_vars=["cluster"],
            value_vars=["good_share", "medium_share", "bad_share"],
            var_name="class_name",
            value_name="share",
        )
        long["class_name"] = long["class_name"].str.replace("_share", "", regex=False)
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        sns.barplot(data=long, x="cluster", y="share", hue="class_name", ax=ax)
        ax.set_title(f"{method} cluster composition")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{method}_cluster_composition.png", dpi=180)
        plt.close(fig)


def plot_probe_results(probe: pd.DataFrame, imp: pd.DataFrame, report_root: Path) -> None:
    fig_dir = report_root / "figures"
    if not probe.empty:
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        sns.barplot(data=probe, x="macro_f1", y="model", hue="split_mode", ax=ax)
        ax.set_title("Interpretable probe performance")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / "probe_macro_f1.png", dpi=180)
        plt.close(fig)
    if not imp.empty:
        top = imp.groupby("feature")["importance"].mean().sort_values(ascending=False).head(25).reset_index()
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.barplot(data=top, x="importance", y="feature", color="#6b8fb3", ax=ax)
        ax.set_title("Mean probe importance")
        fig.tight_layout()
        fig.savefig(fig_dir / "probe_feature_importance.png", dpi=180)
        plt.close(fig)


def plot_synthetic_gap(gap: pd.DataFrame, report_root: Path) -> None:
    if gap.empty:
        return
    fig_dir = report_root / "figures"
    metric = "ks" if "ks" in gap.columns else "standardized_median_delta"
    sub = gap.sort_values(metric, ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(8, max(5, len(sub) * 0.22)))
    sns.barplot(data=sub, x=metric, y="feature", hue="class_name" if "class_name" in sub.columns else None, ax=ax)
    ax.set_title("Synthetic-vs-BUT gap on separating dimensions")
    fig.tight_layout()
    fig.savefig(fig_dir / "synthetic_gap_top_features.png", dpi=180)
    plt.close(fig)


def write_rulebook(
    report_root: Path,
    tests: pd.DataFrame,
    feature_scores: pd.DataFrame,
    family_scores: pd.DataFrame,
    medium_audit: dict[str, Any],
    probe: pd.DataFrame,
    model_context: pd.DataFrame,
) -> None:
    top = feature_scores.sort_values("separability_score", ascending=False).head(20)
    med_bad = tests[tests["contrast"] == "medium_vs_bad"].sort_values("ks", ascending=False).head(12)
    good_med = tests[tests["contrast"] == "good_vs_medium"].sort_values("ks", ascending=False).head(12)
    good_bad = tests[tests["contrast"] == "good_vs_bad"].sort_values("ks", ascending=False).head(12)
    best_probe = probe.sort_values("macro_f1", ascending=False).head(1).to_dict("records")
    best_model = model_context.sort_values("macro_f1", ascending=False).head(1).to_dict("records") if not model_context.empty else []
    lines = [
        "# BUT 10s Three-Class Separability Atlas",
        "",
        "## Executive read",
        "",
        f"- Medium independence interpretation: `{medium_audit.get('interpretation')}`.",
        f"- Medium off-axis ratio from the good-bad centroid line: `{medium_audit.get('medium_independence_ratio', math.nan):.3f}`.",
        f"- Local medium neighbor purity k=30: `{medium_audit.get('local_medium_neighbor_purity_k30', math.nan):.3f}`.",
        f"- Best interpretable probe macro-F1: `{best_probe[0]['macro_f1']:.4f}` via `{best_probe[0]['model']}` / `{best_probe[0]['split_mode']}`." if best_probe else "- Best interpretable probe unavailable.",
        f"- Best current generator/model context macro-F1: `{best_model[0]['macro_f1']:.4f}` from `{best_model[0]['variant_id']}`." if best_model else "- Current generator/model context not found.",
        "",
        "## Top separating feature families",
        "",
        md_table(family_scores),
        "",
        "## Top individual dimensions",
        "",
        md_table(
            top[
                [
                    "feature",
                    "family",
                    "separability_score",
                    "max_ks",
                    "max_abs_cliffs",
                    "mutual_info",
                    "mean_probe_importance",
                ]
            ]
        ),
        "",
        "## Boundary-specific evidence",
        "",
        "### Good vs medium",
        "",
        md_table(good_med[["feature", "median_delta_a_minus_b", "cliffs_delta", "ks", "wasserstein", "mutual_info"]]),
        "",
        "### Medium vs bad",
        "",
        md_table(med_bad[["feature", "median_delta_a_minus_b", "cliffs_delta", "ks", "wasserstein", "mutual_info"]]),
        "",
        "### Good vs bad",
        "",
        md_table(good_bad[["feature", "median_delta_a_minus_b", "cliffs_delta", "ks", "wasserstein", "mutual_info"]]),
        "",
        "## Current answer",
        "",
        "- The classes are not best described by a single SNR line. BUT medium is at least mixed/partly independent: QRS-usable windows can be medium even when some global noise features overlap good or bad.",
        "- The most useful next generator target is a small set of axes: QRS detectability, non-QRS/detail reliability, fatal contact/flat events, baseline/HF motion, and SQI consistency.",
        "- SQI should remain a diagnostic branch because prior SQI gap analysis showed strong BUT-vs-PTB domain signature and class-direction flips.",
        "",
        "## Files",
        "",
        f"- Output root: `{DEFAULT_OUT_ROOT}`",
        f"- Report root: `{DEFAULT_REPORT_ROOT}`",
    ]
    (report_root / "separability_atlas_report.md").write_text("\n".join(lines), encoding="utf-8")

    rule_lines = [
        "# Dimension Rulebook For Next Synthetic Targets",
        "",
        "## Good: AND(all critical dimensions acceptable)",
        "",
        "- Strong QRS visibility and detector agreement.",
        "- Low fatal/contact/flat score.",
        "- Stable non-QRS morphology and low detail instability.",
        "- Baseline/HF motion not dominant.",
        "",
        "## Medium: QRS usable + details unreliable",
        "",
        "- QRS remains detectable, but P/T/ST or local baseline/detail features drift.",
        "- Avoid heavy flat/contact in medium generation; that pushes the sample toward bad.",
        "- Medium should be treated as an independent cluster family, not just midpoint SNR.",
        "",
        "## Bad: OR(any fatal dimension fails hard)",
        "",
        "- QRS detectability failure, severe spurious peaks, contact/flat/low amplitude, clipping, strong baseline jump/platform, or HF/motion burst can each be sufficient.",
        "- Do not make every bad sample fail on every dimension; use sample-level fatal subtype mixtures.",
        "",
        "## Candidate generator targets",
        "",
        "Use these top BUT-separating features as distribution targets before training:",
        "",
        md_table(top[["feature", "family", "separability_score"]].head(15)),
    ]
    (report_root / "dimension_rulebook.md").write_text("\n".join(rule_lines), encoding="utf-8")

    next_lines = [
        "# Next Generator Targets",
        "",
        "- First match BUT feature distributions on the top separating dimensions, class-wise.",
        "- Then run small validation grids only for candidate distributions whose medium and bad distances both move in the correct direction.",
        "- Treat feature target score as a gate, not as a final metric; BUT original test remains final.",
        "",
        "## Target table",
        "",
        md_table(top[["feature", "family", "separability_score", "max_ks", "max_abs_cliffs"]].head(20)),
    ]
    (report_root / "next_generator_targets.md").write_text("\n".join(next_lines), encoding="utf-8")


def make_artifact_payload(
    report_root: Path,
    family_scores: pd.DataFrame,
    feature_scores: pd.DataFrame,
    probe: pd.DataFrame,
    cluster_df: pd.DataFrame,
    medium_audit: dict[str, Any],
) -> dict[str, Any]:
    top_features = feature_scores.sort_values("separability_score", ascending=False).head(40)
    probe_small = probe.head(30)
    clusters = cluster_df.head(80)
    summary = [
        {
            "metric": "medium_independence_ratio",
            "value": float(medium_audit.get("medium_independence_ratio", math.nan)),
            "note": "Distance of medium centroid away from the good-bad axis.",
        },
        {
            "metric": "local_medium_neighbor_purity_k30",
            "value": float(medium_audit.get("local_medium_neighbor_purity_k30", math.nan)),
            "note": "How often medium's nearest neighbors are also medium.",
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "BUT 10s Three-Class Separability Atlas",
        "description": "Feature-level evidence for separating BUT good, medium, and bad ECG quality classes.",
        "sources": [
            {
                "id": "atlas_csv",
                "label": "Generated BUT separability atlas CSV tables",
                "path": "outputs/external_benchmarks/e311_but_separability_atlas_10s_2026_06_06",
            }
        ],
        "blocks": [
            {
                "id": "intro",
                "type": "markdown",
                "body": "# BUT 10s Three-Class Separability Atlas\n\nMedium is evaluated as its own usability cluster, not only as a midpoint between good and bad.",
            },
            {"id": "family_scores_block", "type": "chart", "chartId": "family_scores"},
            {"id": "top_features_block", "type": "chart", "chartId": "top_feature_scores"},
            {"id": "probe_results_block", "type": "table", "tableId": "probe_results"},
            {"id": "cluster_summary_block", "type": "table", "tableId": "cluster_summary"},
            {
                "id": "next_targets",
                "type": "markdown",
                "body": "The generator target should be class-wise distribution matching on QRS detectability, detail reliability, fatal/contact events, motion/frequency, and SQI consistency.",
            },
        ],
        "charts": [
            {
                "id": "family_scores",
                "title": "Separability by Feature Family",
                "type": "bar",
                "dataset": "family_scores",
                "sourceId": "atlas_csv",
                "encodings": {
                    "x": {"field": "family", "type": "nominal"},
                    "y": {"field": "max_score", "type": "quantitative"},
                },
            },
            {
                "id": "top_feature_scores",
                "title": "Top Separating Features",
                "type": "bar",
                "dataset": "top_feature_scores",
                "sourceId": "atlas_csv",
                "encodings": {
                    "x": {"field": "feature", "type": "nominal"},
                    "y": {"field": "separability_score", "type": "quantitative"},
                    "color": {"field": "family", "type": "nominal"},
                },
            },
        ],
        "tables": [
            {
                "id": "probe_results",
                "title": "Interpretable Probe Results",
                "dataset": "probe_results",
                "sourceId": "atlas_csv",
                "columns": [
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "split_mode", "label": "Split", "type": "text"},
                    {"field": "acc", "label": "Accuracy", "type": "number"},
                    {"field": "macro_f1", "label": "Macro-F1", "type": "number"},
                    {"field": "good_recall", "label": "Good recall", "type": "number"},
                    {"field": "medium_recall", "label": "Medium recall", "type": "number"},
                    {"field": "bad_recall", "label": "Bad recall", "type": "number"},
                ],
            },
            {
                "id": "cluster_summary",
                "title": "Cluster Composition",
                "dataset": "cluster_summary",
                "sourceId": "atlas_csv",
                "columns": [
                    {"field": "method", "label": "Method", "type": "text"},
                    {"field": "cluster", "label": "Cluster", "type": "number"},
                    {"field": "n", "label": "N", "type": "number"},
                    {"field": "good_share", "label": "Good share", "type": "number"},
                    {"field": "medium_share", "label": "Medium share", "type": "number"},
                    {"field": "bad_share", "label": "Bad share", "type": "number"},
                    {"field": "dominant_class", "label": "Dominant class", "type": "text"},
                ],
            },
        ],
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": now_iso(),
        "datasets": {
            "family_scores": family_scores.head(20).to_dict("records"),
            "top_feature_scores": top_features.to_dict("records"),
            "probe_results": probe_small.to_dict("records"),
            "cluster_summary": clusters.to_dict("records"),
            "summary_metrics": summary,
        },
    }
    sources = [
        {
            "id": "atlas_csv",
            "query": {
                "engine": "local-csv",
                "sql": "SELECT * FROM outputs.external_benchmarks.e311_but_separability_atlas_10s_2026_06_06.feature_family_scores",
                "description": "Loads aggregated BUT separability atlas tables from generated CSV outputs.",
            },
            "label": "Generated BUT separability atlas CSV tables",
            "path": "outputs/external_benchmarks/e311_but_separability_atlas_10s_2026_06_06",
        }
    ]
    payload = {"manifest": manifest, "snapshot": snapshot, "surface": "report", "sources": sources}
    write_json(report_root / "analytics_artifact_payload.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (report_root / "figures").mkdir(parents=True, exist_ok=True)
    write_json(out_root / "separability_atlas_state.json", {"status": "running", "started_at": now_iso()})

    X, meta = load_but(Path(args.but_protocol_dir))
    df, feature_meta = load_feature_matrix(args, X, meta)
    features = feature_meta["feature"].tolist()
    tests = pairwise_feature_tests(df, features)
    tests = add_mutual_info(df, tests, features)
    tests.to_csv(out_root / "pairwise_feature_tests.csv", index=False)
    plot_distributions(df, tests, report_root)

    probe, probe_imp, _tree = run_probes(df, features, out_root)
    medium_audit, cluster_df, embed = medium_independence(df, features, out_root, report_root)
    proto = prototype_manifest(df, features, embed, X, report_root, out_root)
    model_ctx = model_results_summary(out_root)
    feature_scores = pd.read_csv(out_root / "feature_separability_scores.csv") if (out_root / "feature_separability_scores.csv").exists() else pd.DataFrame()
    family_scores = feature_family_scores(tests, probe_imp, feature_meta, out_root)
    feature_scores = pd.read_csv(out_root / "feature_separability_scores.csv")
    gap = synthetic_gap(feature_scores["feature"].head(40).tolist(), out_root)
    plot_probe_results(probe, probe_imp, report_root)
    plot_synthetic_gap(gap, report_root)
    write_rulebook(report_root, tests, feature_scores, family_scores, medium_audit, probe, model_ctx)
    make_artifact_payload(report_root, family_scores, feature_scores, probe, cluster_df, medium_audit)

    # Copy core tables to report root for GitHub/lightweight browsing.
    for name in [
        "pairwise_feature_tests.csv",
        "feature_family_scores.csv",
        "feature_separability_scores.csv",
        "model_probe_results.csv",
        "cluster_summary.csv",
        "prototype_manifest.csv",
        "synthetic_gap_on_separating_dimensions.csv",
        "medium_independence_tests.json",
        "feature_metadata.csv",
    ]:
        src = out_root / name
        if src.exists():
            dst = report_root / name
            if src.suffix == ".json":
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                pd.read_csv(src).to_csv(dst, index=False)

    summary = {
        "status": "complete",
        "completed_at": now_iso(),
        "n_windows": int(len(df)),
        "n_features": int(len(features)),
        "top_family": family_scores.iloc[0].to_dict() if not family_scores.empty else {},
        "medium_independence": medium_audit,
        "best_probe": probe.iloc[0].to_dict() if not probe.empty else {},
        "report_root": str(report_root),
        "out_root": str(out_root),
    }
    write_json(out_root / "separability_atlas_state.json", summary)
    write_json(report_root / "separability_atlas_state.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build BUT 10s three-class separability atlas.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--morph_cache", default=str(DEFAULT_MORPH_CACHE))
    parser.add_argument("--sqi_cache", default=str(DEFAULT_SQI_CACHE))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
