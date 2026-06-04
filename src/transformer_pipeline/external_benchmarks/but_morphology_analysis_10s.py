"""BUT 10s morphology analysis and synthetic-rule distance audit.

This is an experiment-only analysis script.  It does not train models.  It
extracts a shared set of morphology proxies from real BUT QDB 10s windows and
from completed PTB synthetic variants, then joins class-wise domain distances
with the existing BUT evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:  # pragma: no cover - fallback for minimal environments.
    find_peaks = None
    ks_2samp = None
    wasserstein_distance = None


ROOT = Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_morphology_analysis_10s_2026_06_03"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG

BUT_PROCESSED = ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02" / "processed" / "butqdb"
LARGE_GRID_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_large_rule_grid_10s_2026_06_03"
LARGE_GRID_REPORT = ROOT / "reports" / "external_benchmarks" / "e311_but_large_rule_grid_10s_2026_06_03"
HISTORY_ROWS = ROOT / "reports" / "external_benchmarks" / "e311_but_generator_research_10s_2026_06_03" / "combined_generator_rows.json"

FS = 125
N_TARGET = 1250
CLASS_ORDER = ["good", "medium", "bad"]
CLASS_TO_Y = {"good": 0, "medium": 1, "bad": 2}

QRS_FEATURES = [
    "peak_density",
    "prominence_p75",
    "prominence_cv",
    "rr_cv",
    "spurious_peak_proxy",
    "missing_qrs_proxy",
    "qrs_reliable_proxy",
    "qrs_slope_proxy",
]
PTST_FEATURES = [
    "nonqrs_energy_ratio",
    "deriv_p95",
    "local_deriv_spike_frac",
    "baseline_step_proxy",
    "ptst_unreliable_proxy",
]
CONTACT_FEATURES = [
    "flatline_frac",
    "low_amp_frac",
    "clipping_frac",
    "contact_loss_proxy",
    "baseline_wander",
    "hf_energy",
]
GLOBAL_FEATURES = [
    "rms",
    "range95",
    "abs_mean",
    "deriv_rms",
]
ALL_FEATURES = QRS_FEATURES + PTST_FEATURES + CONTACT_FEATURES + GLOBAL_FEATURES


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz_array(path: Path, preferred: tuple[str, ...] = ("X", "X_noisy", "signals", "arr_0")) -> np.ndarray:
    z = np.load(path)
    for key in preferred:
        if key in z.files:
            arr = z[key]
            break
    else:
        arr = z[z.files[0]]
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal array from {path}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width <= 1:
        return x
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(x, kernel, mode="same")


def window_std_features(x: np.ndarray, width: int = 25, step: int = 12) -> tuple[float, float]:
    vals: list[float] = []
    for start in range(0, max(1, len(x) - width + 1), step):
        vals.append(float(np.std(x[start : start + width])))
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float32)
    flat_frac = float(np.mean(arr < 0.015))
    longest = 0
    current = 0
    for v in arr < 0.015:
        if bool(v):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return flat_frac, float(longest * step / max(1, len(x)))


def fallback_find_peaks(y: np.ndarray, distance: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if candidates.size == 0:
        return candidates, np.asarray([], dtype=np.float32)
    threshold = np.percentile(y, 75)
    candidates = candidates[y[candidates] >= threshold]
    if candidates.size == 0:
        return candidates, np.asarray([], dtype=np.float32)
    kept: list[int] = []
    for idx in candidates[np.argsort(y[candidates])[::-1]]:
        if all(abs(int(idx) - k) >= distance for k in kept):
            kept.append(int(idx))
    kept = sorted(kept)
    return np.asarray(kept, dtype=np.int64), y[kept].astype(np.float32)


def peak_features(xn: np.ndarray) -> dict[str, float | np.ndarray]:
    y = np.abs(xn)
    distance = max(1, int(0.25 * FS))
    if find_peaks is not None:
        peaks, props = find_peaks(y, distance=distance, prominence=0.08)
        prominences = props.get("prominences", np.asarray([], dtype=np.float32)).astype(np.float32)
    else:
        peaks, prominences = fallback_find_peaks(y, distance)
    peak_count = int(len(peaks))
    peak_density = peak_count / 10.0
    if peak_count >= 2:
        rr = np.diff(peaks) / float(FS)
        rr_cv = float(np.std(rr) / (np.mean(rr) + 1e-6))
    else:
        rr_cv = 2.0
    if prominences.size:
        prom_mean = float(np.mean(prominences))
        prom_p75 = float(np.percentile(prominences, 75))
        prom_cv = float(np.std(prominences) / (prom_mean + 1e-6))
        prom_max = float(np.max(prominences))
    else:
        prom_mean = prom_p75 = prom_max = 0.0
        prom_cv = 2.0
    spurious = max(0.0, peak_density - 2.4)
    missing = max(0.0, 0.7 - peak_density)
    reliable = prom_p75 / (1.0 + rr_cv + 0.5 * spurious + 1.5 * missing)
    mask = np.zeros_like(xn, dtype=bool)
    half = int(0.08 * FS)
    for p in peaks:
        start = max(0, int(p) - half)
        end = min(len(mask), int(p) + half + 1)
        mask[start:end] = True
    d = np.diff(xn, prepend=xn[0])
    qrs_slope = float(np.percentile(np.abs(d[mask]), 90)) if mask.any() else 0.0
    nonqrs = ~mask
    nonqrs_energy = float(np.sqrt(np.mean(xn[nonqrs] ** 2)) / (np.sqrt(np.mean(xn**2)) + 1e-6)) if nonqrs.any() else 0.0
    return {
        "peaks": peaks,
        "peak_density": float(peak_density),
        "prominence_mean": prom_mean,
        "prominence_p75": prom_p75,
        "prominence_max": prom_max,
        "prominence_cv": prom_cv,
        "rr_cv": rr_cv,
        "spurious_peak_proxy": float(spurious),
        "missing_qrs_proxy": float(missing),
        "qrs_reliable_proxy": float(reliable),
        "qrs_slope_proxy": qrs_slope,
        "nonqrs_energy_ratio": nonqrs_energy,
    }


def extract_one(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    centered = x - float(np.median(x))
    p05, p95 = np.percentile(centered, [5, 95])
    scale = float(max(p95 - p05, np.std(centered), 1e-4))
    xn = centered / scale
    d = np.diff(xn, prepend=xn[0])
    ma_long = moving_average(xn, FS)
    ma_short = moving_average(xn, max(3, int(0.06 * FS)))
    baseline = ma_long
    hf = xn - ma_short
    local_trend = moving_average(xn, max(3, int(0.20 * FS)))
    local_step = np.diff(local_trend, prepend=local_trend[0])
    flat_frac, contact_proxy = window_std_features(xn)
    pf = peak_features(xn)
    deriv_abs = np.abs(d)
    deriv_p95 = float(np.percentile(deriv_abs, 95))
    deriv_spike = float(np.mean(deriv_abs > max(0.30, np.percentile(deriv_abs, 90) * 1.5)))
    low_amp = float(np.mean(np.abs(xn) < 0.04))
    clipping = float(np.mean(np.abs(xn) > 2.5))
    baseline_wander = float(np.sqrt(np.mean(baseline**2)) / (np.sqrt(np.mean(xn**2)) + 1e-6))
    hf_energy = float(np.sqrt(np.mean(hf**2)) / (np.sqrt(np.mean(xn**2)) + 1e-6))
    baseline_step = float(np.percentile(np.abs(local_step), 95))
    ptst_unreliable = float(0.45 * pf["nonqrs_energy_ratio"] + 0.25 * deriv_spike + 0.20 * baseline_step + 0.10 * hf_energy)
    return {
        "rms": float(np.sqrt(np.mean(xn**2))),
        "abs_mean": float(np.mean(np.abs(xn))),
        "range95": float(p95 - p05),
        "deriv_rms": float(np.sqrt(np.mean(d**2))),
        "deriv_p95": deriv_p95,
        "local_deriv_spike_frac": deriv_spike,
        "baseline_step_proxy": baseline_step,
        "flatline_frac": flat_frac,
        "contact_loss_proxy": contact_proxy,
        "low_amp_frac": low_amp,
        "clipping_frac": clipping,
        "baseline_wander": baseline_wander,
        "hf_energy": hf_energy,
        "ptst_unreliable_proxy": ptst_unreliable,
        **{k: float(v) for k, v in pf.items() if k != "peaks"},
    }


def sample_indices_by_class(labels: pd.Series, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for cls in CLASS_ORDER:
        idx = np.flatnonzero(labels.to_numpy() == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(np.asarray(idx, dtype=np.int64))
    if not selected:
        return np.asarray([], dtype=np.int64)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def extract_features_for_dataset(
    X: np.ndarray,
    labels: pd.Series,
    source: str,
    variant_id: str,
    family: str,
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    idx = sample_indices_by_class(labels, max_per_class=max_per_class, seed=seed)
    rows: list[dict[str, Any]] = []
    for j, i in enumerate(idx):
        feats = extract_one(X[int(i)])
        feats.update(
            {
                "row_index": int(i),
                "sample_order": int(j),
                "source": source,
                "variant_id": variant_id,
                "family": family,
                "y_class": str(labels.iloc[int(i)]),
            }
        )
        rows.append(feats)
    return pd.DataFrame(rows)


def load_but_features(max_per_class: int, seed: int) -> pd.DataFrame:
    X = load_npz_array(BUT_PROCESSED / "signals.npz")
    meta = pd.read_csv(BUT_PROCESSED / "metadata.csv")
    if X.shape != (32956, N_TARGET):
        raise ValueError(f"Unexpected BUT shape {X.shape}; expected (32956, {N_TARGET})")
    split_counts = meta["split"].value_counts().to_dict()
    if split_counts.get("train") != 23322 or split_counts.get("val") != 1157 or split_counts.get("test") != 8477:
        raise ValueError(f"Unexpected BUT split counts {split_counts}")
    return extract_features_for_dataset(
        X,
        meta["y_class"],
        source="BUT_QDB_10s_P1",
        variant_id="BUT_QDB",
        family="real_but",
        max_per_class=max_per_class,
        seed=seed,
    )


def read_large_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jsonl = LARGE_GRID_ROOT / "large_rule_grid_summary.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rep = r.get("but_10s_eval", {}).get("but_10s_test_report", {})
            rec = rep.get("recall_good_medium_bad") or [math.nan, math.nan, math.nan]
            ptb_rec = r.get("ptb_test_report", {}).get("recall_good_medium_bad") or [math.nan, math.nan, math.nan]
            spec = r.get("spec", {})
            rows.append(
                {
                    "variant_id": spec.get("id"),
                    "family": spec.get("family", "large_rule_grid"),
                    "source_set": "large_rule_grid",
                    "acc": rep.get("acc"),
                    "balanced_acc": rep.get("balanced_acc"),
                    "macro_f1": rep.get("macro_f1"),
                    "good_recall": rec[0],
                    "medium_recall": rec[1],
                    "bad_recall": rec[2],
                    "min_medium_bad": min(rec[1], rec[2]) if len(rec) >= 3 else math.nan,
                    "ptb_acc": r.get("ptb_test_report", {}).get("acc"),
                    "ptb_bad": ptb_rec[2] if len(ptb_rec) >= 3 else math.nan,
                    "denoise_score": r.get("ptb_denoise_metrics", {}).get("denoise_score"),
                    "variant_dir": r.get("variant_dir"),
                    "run_dir": r.get("run_dir"),
                    "note": r.get("note"),
                    "mode": r.get("mode", "quick"),
                }
            )
    return rows


def read_history_rows() -> list[dict[str, Any]]:
    if not HISTORY_ROWS.exists():
        return []
    raw = read_json(HISTORY_ROWS)
    rows: list[dict[str, Any]] = []
    for r in raw:
        name = r.get("name")
        if not name:
            continue
        rows.append(
            {
                "variant_id": name,
                "family": r.get("family"),
                "source_set": "history_generator_research",
                "acc": r.get("acc"),
                "balanced_acc": r.get("balanced_acc"),
                "macro_f1": r.get("macro_f1"),
                "good_recall": r.get("good_recall"),
                "medium_recall": r.get("medium_recall"),
                "bad_recall": r.get("bad_recall"),
                "min_medium_bad": r.get("min_medium_bad"),
                "ptb_acc": r.get("ptb_acc"),
                "ptb_bad": r.get("ptb_bad"),
                "denoise_score": r.get("denoise_score"),
                "variant_dir": None,
                "run_dir": r.get("run_dir"),
                "note": r.get("note"),
                "mode": "history",
            }
        )
    return rows


def variant_dir_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for audit in (ROOT / "outputs" / "external_benchmarks").glob("e311_but*/*/*/data_variant_audit.json"):
        variant_dir = audit.parent
        out.setdefault(variant_dir.name, variant_dir)
        try:
            spec_id = read_json(audit).get("spec", {}).get("id")
            if spec_id:
                out.setdefault(str(spec_id), variant_dir)
        except Exception:
            pass
    for audit in LARGE_GRID_ROOT.glob("synthetic_variants/*/data_variant_audit.json"):
        variant_dir = audit.parent
        out.setdefault(variant_dir.name, variant_dir)
        try:
            spec_id = read_json(audit).get("spec", {}).get("id")
            if spec_id:
                out.setdefault(str(spec_id), variant_dir)
        except Exception:
            pass
    return out


def merge_metric_rows() -> pd.DataFrame:
    rows = read_history_rows() + read_large_rows()
    if not rows:
        raise FileNotFoundError("No generator metric rows found")
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["variant_id", "source_set"], keep="last")
    idx = variant_dir_index()
    resolved: list[str | None] = []
    for _, r in df.iterrows():
        existing = r.get("variant_dir")
        if isinstance(existing, str) and existing and Path(existing).exists():
            resolved.append(str(Path(existing)))
        else:
            p = idx.get(str(r["variant_id"]))
            resolved.append(str(p) if p else None)
    df["variant_dir"] = resolved
    df["has_variant_data"] = df["variant_dir"].notna()
    return df


def load_synthetic_variant_features(row: pd.Series, max_per_class: int, seed: int) -> pd.DataFrame | None:
    variant_dir = row.get("variant_dir")
    if not isinstance(variant_dir, str) or not variant_dir:
        return None
    datasets = Path(variant_dir) / "datasets"
    signals = datasets / "signals.npz"
    noisy = datasets / "synth_10s_125hz_noisy.npz"
    labels_path = datasets / "synth_10s_125hz_labels.csv"
    if not labels_path.exists():
        return None
    if signals.exists():
        X = load_npz_array(signals)
    elif noisy.exists():
        X = load_npz_array(noisy)
    else:
        return None
    labels = pd.read_csv(labels_path)
    label_col = "y_class" if "y_class" in labels.columns else "label"
    if label_col not in labels.columns:
        return None
    return extract_features_for_dataset(
        X,
        labels[label_col].astype(str),
        source=str(row.get("source_set", "synthetic")),
        variant_id=str(row["variant_id"]),
        family=str(row.get("family", "unknown")),
        max_per_class=max_per_class,
        seed=seed,
    )


def safe_ks(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return math.nan
    if ks_2samp is not None:
        return float(ks_2samp(a, b).statistic)
    qa = np.quantile(a, np.linspace(0, 1, 101))
    qb = np.quantile(b, np.linspace(0, 1, 101))
    return float(np.max(np.abs(qa - qb)) / (np.std(b) + 1e-6))


def safe_wasserstein_scaled(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return math.nan
    scale = float(np.std(b) + np.std(a) + 1e-6)
    if wasserstein_distance is not None:
        return float(wasserstein_distance(a, b) / scale)
    return float(abs(np.mean(a) - np.mean(b)) / scale)


def feature_distance(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ks = safe_ks(a, b)
    wass = safe_wasserstein_scaled(a, b)
    mean_z = float(abs(np.nanmean(a) - np.nanmean(b)) / (np.nanstd(b) + 1e-6))
    mean_z_capped = min(mean_z, 5.0) / 5.0
    combined = 0.45 * ks + 0.35 * min(wass, 3.0) / 3.0 + 0.20 * mean_z_capped
    return {"ks": float(ks), "wasserstein_scaled": float(wass), "mean_z": float(mean_z), "combined": float(combined)}


def compute_distances(but_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    but_by_class = {cls: but_df[but_df["y_class"] == cls] for cls in CLASS_ORDER}
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for (variant_id, family), vdf in synth_df.groupby(["variant_id", "family"], dropna=False):
        class_scores: dict[str, float] = {}
        group_scores: dict[str, list[float]] = {"qrs": [], "ptst": [], "contact_motion": [], "global": []}
        for cls in CLASS_ORDER:
            sdf = vdf[vdf["y_class"] == cls]
            bdf = but_by_class[cls]
            for feat in ALL_FEATURES:
                d = feature_distance(sdf[feat].to_numpy(float), bdf[feat].to_numpy(float))
                feature_rows.append(
                    {
                        "variant_id": variant_id,
                        "family": family,
                        "y_class": cls,
                        "feature": feat,
                        **d,
                    }
                )
                class_scores.setdefault(cls, 0.0)
            class_feature_df = pd.DataFrame([r for r in feature_rows if r["variant_id"] == variant_id and r["y_class"] == cls])
            class_scores[cls] = float(class_feature_df["combined"].mean())
            for group_name, feats in [
                ("qrs", QRS_FEATURES),
                ("ptst", PTST_FEATURES),
                ("contact_motion", CONTACT_FEATURES),
                ("global", GLOBAL_FEATURES),
            ]:
                g = class_feature_df[class_feature_df["feature"].isin(feats)]["combined"].mean()
                group_scores[group_name].append(float(g))
        qrs = float(np.nanmean(group_scores["qrs"]))
        ptst = float(np.nanmean(group_scores["ptst"]))
        contact = float(np.nanmean(group_scores["contact_motion"]))
        global_d = float(np.nanmean(group_scores["global"]))
        overall = float(np.nanmean([class_scores[c] for c in CLASS_ORDER]))
        rows.append(
            {
                "variant_id": variant_id,
                "family": family,
                "distance_good": class_scores.get("good"),
                "distance_medium": class_scores.get("medium"),
                "distance_bad": class_scores.get("bad"),
                "qrs_distance": qrs,
                "ptst_distance": ptst,
                "contact_motion_distance": contact,
                "global_distance": global_d,
                "overall_but_like_score": overall,
                "but_like_similarity": 1.0 / (1.0 + overall),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)


def summarize_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source, variant_id, family, y_class), g in df.groupby(["source", "variant_id", "family", "y_class"], dropna=False):
        row: dict[str, Any] = {
            "source": source,
            "variant_id": variant_id,
            "family": family,
            "y_class": y_class,
            "n": int(len(g)),
        }
        for feat in ALL_FEATURES:
            row[f"{feat}_mean"] = float(g[feat].mean())
            row[f"{feat}_p50"] = float(g[feat].median())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["summary_label"] = label
    return out


def save_table(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def plot_class_profiles(but_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    rng = np.random.default_rng(7)
    X = load_npz_array(BUT_PROCESSED / "signals.npz")
    meta = pd.read_csv(BUT_PROCESSED / "metadata.csv")
    for ax, cls in zip(axes, CLASS_ORDER):
        idx = np.flatnonzero(meta["y_class"].to_numpy() == cls)
        if len(idx) > 8:
            idx = rng.choice(idx, 8, replace=False)
        t = np.arange(N_TARGET) / FS
        for i in idx:
            x = X[int(i)]
            x = (x - np.median(x)) / (np.percentile(x, 95) - np.percentile(x, 5) + 1e-6)
            ax.plot(t, x, alpha=0.65, linewidth=0.9)
        ax.set_title(f"BUT {cls}: representative normalized 10s windows")
        ax.set_ylabel("norm amp")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_profile(but_summary: pd.DataFrame, synth_summary: pd.DataFrame, out_path: Path) -> None:
    key_feats = [
        "qrs_reliable_proxy",
        "spurious_peak_proxy",
        "missing_qrs_proxy",
        "ptst_unreliable_proxy",
        "baseline_step_proxy",
        "contact_loss_proxy",
        "flatline_frac",
        "baseline_wander",
        "hf_energy",
    ]
    but_rows = but_summary[but_summary["variant_id"] == "BUT_QDB"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    x = np.arange(len(key_feats))
    for ax, cls in zip(axes, CLASS_ORDER):
        vals = []
        for feat in key_feats:
            row = but_rows[but_rows["y_class"] == cls]
            vals.append(float(row[f"{feat}_mean"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x, vals)
        ax.set_title(f"BUT {cls}: morphology proxy means")
        ax.set_ylabel("mean")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(key_feats, rotation=35, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_distance_scatter(join_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    scatter_specs = [
        ("overall_but_like_score", "macro_f1", "Lower morphology distance vs macro-F1"),
        ("qrs_distance", "bad_recall", "QRS distance vs bad recall"),
        ("ptst_distance", "medium_recall", "P/T/ST distance vs medium recall"),
    ]
    for ax, (xcol, ycol, title) in zip(axes, scatter_specs):
        for family, g in join_df.groupby("family"):
            ax.scatter(g[xcol], g[ycol], label=family, alpha=0.75, s=36)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_variant_signal_and_labels(variant_dir: Path) -> tuple[np.ndarray, pd.DataFrame] | None:
    datasets = variant_dir / "datasets"
    labels_path = datasets / "synth_10s_125hz_labels.csv"
    if not labels_path.exists():
        return None
    signal_path = datasets / "signals.npz"
    if not signal_path.exists():
        signal_path = datasets / "synth_10s_125hz_noisy.npz"
    if not signal_path.exists():
        return None
    return load_npz_array(signal_path), pd.read_csv(labels_path)


def plot_nearest_farthest(join_df: pd.DataFrame, out_path: Path) -> None:
    candidates = join_df[join_df["variant_dir"].notna()].copy()
    if candidates.empty:
        return
    nearest = candidates.sort_values("overall_but_like_score").head(1)
    farthest = candidates.sort_values("overall_but_like_score", ascending=False).head(1)
    chosen = pd.concat([nearest, farthest], ignore_index=True)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 4.2 * len(chosen)), sharex=True)
    if len(chosen) == 1:
        axes = np.asarray([axes])
    rng = np.random.default_rng(11)
    for row_i, (_, row) in enumerate(chosen.iterrows()):
        loaded = load_variant_signal_and_labels(Path(str(row["variant_dir"])))
        if loaded is None:
            continue
        X, labels = loaded
        label_col = "y_class" if "y_class" in labels.columns else "label"
        for col_i, cls in enumerate(CLASS_ORDER):
            ax = axes[row_i, col_i]
            idx = np.flatnonzero(labels[label_col].astype(str).to_numpy() == cls)
            if len(idx) > 5:
                idx = rng.choice(idx, 5, replace=False)
            t = np.arange(N_TARGET) / FS
            for i in idx:
                x = X[int(i)]
                x = (x - np.median(x)) / (np.percentile(x, 95) - np.percentile(x, 5) + 1e-6)
                ax.plot(t, x, alpha=0.7, linewidth=0.85)
            ax.set_title(f"{row['variant_id']} {cls}")
            ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("seconds")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def correlation_rows(join_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("overall_but_like_score", "macro_f1"),
        ("overall_but_like_score", "balanced_acc"),
        ("qrs_distance", "bad_recall"),
        ("ptst_distance", "medium_recall"),
        ("contact_motion_distance", "bad_recall"),
        ("distance_good", "good_recall"),
        ("distance_medium", "medium_recall"),
        ("distance_bad", "bad_recall"),
    ]
    for x, y in pairs:
        sub = join_df[[x, y]].dropna()
        if len(sub) < 3:
            corr = math.nan
        else:
            corr = float(sub[x].corr(sub[y], method="spearman"))
        rows.append({"x": x, "y": y, "spearman": corr, "n": int(len(sub))})
    return rows


def markdown_summary(join_df: pd.DataFrame, corr: list[dict[str, Any]], output_paths: dict[str, str]) -> str:
    top_macro = join_df.sort_values(["macro_f1", "balanced_acc"], ascending=False).head(8)
    top_distance = join_df.sort_values("overall_but_like_score").head(8)
    anchor = join_df[join_df["variant_id"].eq("b10_all_bad_wearable")]
    lines = [
        "# BUT Morphology Analysis 10s",
        "",
        "Protocol: BUT 10s P1, expert consensus 1/2/3 mapped to good/medium/bad. Synthetic candidates are PTB-derived only; BUT test is not used for training or threshold selection.",
        "",
        "## Key Findings",
    ]
    if not anchor.empty:
        a = anchor.iloc[0]
        lines.append(
            f"- Strict b10 anchor remains the reference: acc `{a.acc:.4f}`, balanced `{a.balanced_acc:.4f}`, macro-F1 `{a.macro_f1:.4f}`, recalls `{a.good_recall:.3f}/{a.medium_recall:.3f}/{a.bad_recall:.3f}`."
        )
    best_macro = top_macro.iloc[0]
    lines.append(
        f"- Best completed large-grid macro row is `{best_macro.variant_id}`: macro-F1 `{best_macro.macro_f1:.4f}`, balanced `{best_macro.balanced_acc:.4f}`, recalls `{best_macro.good_recall:.3f}/{best_macro.medium_recall:.3f}/{best_macro.bad_recall:.3f}`."
    )
    best_dist = top_distance.iloc[0]
    lines.append(
        f"- Closest morphology-distance row is `{best_dist.variant_id}` with overall distance `{best_dist.overall_but_like_score:.4f}` and macro-F1 `{best_dist.macro_f1:.4f}`."
    )
    corr_overall = next((r for r in corr if r["x"] == "overall_but_like_score" and r["y"] == "macro_f1"), None)
    if corr_overall and np.isfinite(corr_overall["spearman"]):
        relation = "meaningful" if abs(corr_overall["spearman"]) >= 0.35 else "weak"
        lines.append(
            f"- Overall morphology distance vs macro-F1 Spearman is `{corr_overall['spearman']:.3f}` over `{corr_overall['n']}` rows, so this proxy currently has a `{relation}` relationship with external performance."
        )
    lines.extend(
        [
            "",
            "## Top By BUT Macro-F1",
            "| variant | family | acc | balanced | macro | G/M/B recall | morph distance |",
            "|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for _, r in top_macro.iterrows():
        lines.append(
            f"| `{r.variant_id}` | {r.family} | {r.acc:.4f} | {r.balanced_acc:.4f} | {r.macro_f1:.4f} | {r.good_recall:.3f}/{r.medium_recall:.3f}/{r.bad_recall:.3f} | {r.overall_but_like_score:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Closest Synthetic Morphology To BUT",
            "| variant | family | morph distance | qrs | ptst | contact | macro | G/M/B recall |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, r in top_distance.iterrows():
        lines.append(
            f"| `{r.variant_id}` | {r.family} | {r.overall_but_like_score:.4f} | {r.qrs_distance:.4f} | {r.ptst_distance:.4f} | {r.contact_motion_distance:.4f} | {r.macro_f1:.4f} | {r.good_recall:.3f}/{r.medium_recall:.3f}/{r.bad_recall:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- BUT medium is best described as QRS-visible but locally unreliable: P/T/ST and short baseline/contact events matter more than simple SNR.",
            "- BUT bad requires QRS detectability failure or QRS-confounding events. Rules that only strengthen flatline/low-amplitude often hurt medium or create unstable bad recall.",
            "- Rows that improve macro-F1 without improving morphology distance are likely exploiting calibration/head bias rather than truly matching BUT waveform morphology.",
            "",
            "## Artifacts",
        ]
    )
    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def make_dashboard_payload(
    join_df: pd.DataFrame,
    but_profile: pd.DataFrame,
    corr: list[dict[str, Any]],
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    top_rows = join_df.sort_values(["macro_f1", "balanced_acc"], ascending=False).head(30)
    dist_rows = join_df.sort_values("overall_but_like_score").head(30)
    family_rows = (
        join_df.groupby("family", dropna=False)
        .agg(
            n=("variant_id", "count"),
            best_macro=("macro_f1", "max"),
            best_balanced=("balanced_acc", "max"),
            best_bad=("bad_recall", "max"),
            mean_distance=("overall_but_like_score", "mean"),
        )
        .reset_index()
        .sort_values("best_macro", ascending=False)
    )
    profile_rows = but_profile[but_profile["variant_id"].eq("BUT_QDB")][
        [
            "y_class",
            "n",
            "qrs_reliable_proxy_mean",
            "spurious_peak_proxy_mean",
            "missing_qrs_proxy_mean",
            "ptst_unreliable_proxy_mean",
            "contact_loss_proxy_mean",
            "baseline_wander_mean",
            "hf_energy_mean",
        ]
    ].copy()
    for df in [top_rows, dist_rows, family_rows, profile_rows]:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float).round(6)
    snapshot = {
        "version": 1,
        "status": "ready",
        "datasets": {
            "top_metric_rows": top_rows.fillna("").to_dict(orient="records"),
            "top_distance_rows": dist_rows.fillna("").to_dict(orient="records"),
            "family_summary": family_rows.fillna("").to_dict(orient="records"),
            "but_profile": profile_rows.fillna("").to_dict(orient="records"),
            "correlations": pd.DataFrame(corr).fillna("").to_dict(orient="records"),
        },
    }
    table_columns = {
        "top_metric_table": [
            {"field": "variant_id", "label": "Variant", "type": "text"},
            {"field": "family", "label": "Family", "type": "text"},
            {"field": "acc", "label": "Acc", "type": "number"},
            {"field": "balanced_acc", "label": "Balanced", "type": "number"},
            {"field": "macro_f1", "label": "Macro-F1", "type": "number"},
            {"field": "good_recall", "label": "Good R", "type": "number"},
            {"field": "medium_recall", "label": "Medium R", "type": "number"},
            {"field": "bad_recall", "label": "Bad R", "type": "number"},
            {"field": "overall_but_like_score", "label": "Morph Dist", "type": "number"},
        ],
        "distance_table": [
            {"field": "variant_id", "label": "Variant", "type": "text"},
            {"field": "family", "label": "Family", "type": "text"},
            {"field": "overall_but_like_score", "label": "Morph Dist", "type": "number"},
            {"field": "qrs_distance", "label": "QRS Dist", "type": "number"},
            {"field": "ptst_distance", "label": "P/T/ST Dist", "type": "number"},
            {"field": "contact_motion_distance", "label": "Contact Dist", "type": "number"},
            {"field": "macro_f1", "label": "Macro-F1", "type": "number"},
        ],
        "profile_table": [
            {"field": "y_class", "label": "Class", "type": "text"},
            {"field": "n", "label": "N", "type": "number"},
            {"field": "qrs_reliable_proxy_mean", "label": "QRS Reliable", "type": "number"},
            {"field": "spurious_peak_proxy_mean", "label": "Spurious Peak", "type": "number"},
            {"field": "ptst_unreliable_proxy_mean", "label": "P/T/ST Unreliable", "type": "number"},
            {"field": "hf_energy_mean", "label": "HF Energy", "type": "number"},
        ],
        "corr_table": [
            {"field": "x", "label": "Distance Feature", "type": "text"},
            {"field": "y", "label": "Metric", "type": "text"},
            {"field": "spearman", "label": "Spearman", "type": "number"},
            {"field": "n", "label": "N", "type": "number"},
        ],
    }
    manifest = {
        "version": 1,
        "title": "BUT 10s Morphology Analysis",
        "surface": "report",
        "description": "Morphology-distance audit between BUT QDB expert labels and PTB synthetic generator rules.",
        "sources": [
            {"label": "Local markdown summary", "path": str(report_path)},
            {"label": "BUT processed data", "path": str(BUT_PROCESSED)},
            {"label": "Large rule grid rows", "path": str(LARGE_GRID_ROOT / "large_rule_grid_summary.jsonl")},
        ],
        "blocks": [
            {
                "id": "intro",
                "type": "markdown",
                "body": "# BUT 10s Morphology Analysis\nThis report compares real BUT expert classes with PTB synthetic generator variants using QRS reliability, local morphology, contact/motion, and global waveform proxies.",
            },
            {"id": "metric_scatter_block", "type": "chart", "chartId": "metric_scatter"},
            {"id": "top_metric_table_block", "type": "table", "tableId": "top_metric_table"},
            {"id": "family_bar_block", "type": "chart", "chartId": "family_bar"},
            {"id": "distance_table_block", "type": "table", "tableId": "distance_table"},
            {"id": "profile_table_block", "type": "table", "tableId": "profile_table"},
            {"id": "corr_table_block", "type": "table", "tableId": "corr_table"},
        ],
        "charts": [
            {
                "id": "metric_scatter",
                "title": "Morphology Distance vs BUT Macro-F1",
                "type": "scatter",
                "dataset": "top_metric_rows",
                "source": {
                    "label": "grid_metric_distance_join",
                    "query": {
                        "language": "sql",
                        "sql": "SELECT variant_id, family, macro_f1, overall_but_like_score FROM top_metric_rows",
                        "description": "Top completed synthetic variants joined with morphology distance and BUT metrics.",
                    },
                },
                "encodings": {
                    "x": {"field": "overall_but_like_score"},
                    "y": {"field": "macro_f1"},
                    "color": {"field": "family"},
                },
            },
            {
                "id": "family_bar",
                "title": "Best Macro-F1 by Rule Family",
                "type": "bar",
                "dataset": "family_summary",
                "source": {
                    "label": "family_summary",
                    "query": {
                        "language": "sql",
                        "sql": "SELECT family, best_macro FROM family_summary",
                        "description": "Rule-family aggregate metrics from completed synthetic variants.",
                    },
                },
                "encodings": {
                    "x": {"field": "family"},
                    "y": {"field": "best_macro"},
                },
                "options": {"orientation": "vertical"},
            },
        ],
        "tables": [
            {
                "id": "top_metric_table",
                "title": "Top Metric Rows",
                "dataset": "top_metric_rows",
                "source": {
                    "label": "top_metric_rows",
                    "query": {"language": "sql", "sql": "SELECT * FROM top_metric_rows", "description": "Top variants by BUT macro-F1."},
                },
                "columns": table_columns["top_metric_table"],
            },
            {
                "id": "distance_table",
                "title": "Closest Morphology Rows",
                "dataset": "top_distance_rows",
                "source": {
                    "label": "top_distance_rows",
                    "query": {
                        "language": "sql",
                        "sql": "SELECT * FROM top_distance_rows",
                        "description": "Variants ranked by closest class-wise morphology distance to BUT.",
                    },
                },
                "columns": table_columns["distance_table"],
            },
            {
                "id": "profile_table",
                "title": "BUT Class Morphology Profile",
                "dataset": "but_profile",
                "source": {
                    "label": "but_profile",
                    "query": {"language": "sql", "sql": "SELECT * FROM but_profile", "description": "Sampled BUT class-level morphology proxy means."},
                },
                "columns": table_columns["profile_table"],
            },
            {
                "id": "corr_table",
                "title": "Distance and Metric Correlations",
                "dataset": "correlations",
                "source": {
                    "label": "correlations",
                    "query": {
                        "language": "sql",
                        "sql": "SELECT * FROM correlations",
                        "description": "Spearman correlations between morphology-distance proxies and BUT metrics.",
                    },
                },
                "columns": table_columns["corr_table"],
            },
        ],
    }
    return manifest, snapshot


def run(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    visuals_dir = OUT_ROOT / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    metric_df = merge_metric_rows()
    completed_large = int((metric_df["source_set"] == "large_rule_grid").sum())
    if completed_large < 42:
        raise ValueError(f"Expected at least 42 completed large-rule rows, found {completed_large}")

    but_features = load_but_features(args.max_but_per_class, seed=args.seed)
    synth_feature_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    candidates = metric_df[metric_df["has_variant_data"]].copy()
    if args.max_variants and args.max_variants > 0:
        key_ids = {
            "medium_qrs_visible_family_13",
            "medium_qrs_visible_family_07",
            "bad_qrs_unreliable_family_11",
            "good_not_pristine_family_01",
            "b10_all_bad_wearable",
        }
        priority = candidates["variant_id"].isin(key_ids).astype(int)
        candidates = candidates.assign(_priority=priority).sort_values(["_priority", "macro_f1"], ascending=False).head(args.max_variants)
    for _, row in candidates.iterrows():
        try:
            f = load_synthetic_variant_features(row, max_per_class=args.max_synth_per_class, seed=args.seed)
            if f is None or f.empty:
                skipped.append({"variant_id": row["variant_id"], "reason": "missing_or_empty_variant_data"})
                continue
            synth_feature_parts.append(f)
        except Exception as exc:
            skipped.append({"variant_id": row["variant_id"], "reason": repr(exc)})

    if not synth_feature_parts:
        raise RuntimeError("No synthetic features were extracted")
    synth_features = pd.concat(synth_feature_parts, ignore_index=True)
    distance_df, feature_distance_df = compute_distances(but_features, synth_features)
    join_df = metric_df.merge(distance_df, on=["variant_id", "family"], how="left")
    join_df = join_df[join_df["overall_but_like_score"].notna()].copy()
    corr = correlation_rows(join_df)

    but_summary = summarize_features(but_features, "but")
    synth_summary = summarize_features(synth_features, "synthetic")
    save_table(but_features, OUT_ROOT / "but_morph_features")
    save_table(synth_features, OUT_ROOT / "synthetic_morph_features")
    save_table(but_summary, OUT_ROOT / "but_morph_feature_summary")
    save_table(synth_summary, OUT_ROOT / "synthetic_morph_feature_summary")
    save_table(distance_df, OUT_ROOT / "morph_distance_by_variant")
    save_table(feature_distance_df, OUT_ROOT / "morph_distance_by_feature")
    save_table(join_df, OUT_ROOT / "grid_metric_distance_join")
    write_json(OUT_ROOT / "morph_distance_by_variant.json", distance_df.to_dict(orient="records"))
    write_json(OUT_ROOT / "skipped_variants.json", skipped)
    write_json(OUT_ROOT / "distance_metric_correlations.json", corr)

    plot_class_profiles(but_features, visuals_dir / "but_class_profiles.png")
    plot_feature_profile(but_summary, synth_summary, visuals_dir / "but_feature_profiles_by_class.png")
    plot_distance_scatter(join_df, visuals_dir / "distance_vs_metrics.png")
    plot_nearest_farthest(join_df, visuals_dir / "synthetic_nearest_farthest.png")

    output_paths = {
        "BUT feature table": str(OUT_ROOT / "but_morph_features.csv"),
        "Synthetic feature table": str(OUT_ROOT / "synthetic_morph_features.csv"),
        "Distance join": str(OUT_ROOT / "grid_metric_distance_join.csv"),
        "BUT class gallery": str(visuals_dir / "but_class_profiles.png"),
        "Distance scatter": str(visuals_dir / "distance_vs_metrics.png"),
        "Nearest/farthest gallery": str(visuals_dir / "synthetic_nearest_farthest.png"),
    }
    summary = markdown_summary(join_df, corr, output_paths)
    (OUT_ROOT / "morphology_analysis_summary.md").write_text(summary, encoding="utf-8")
    (REPORT_ROOT / "morphology_analysis_summary.md").write_text(summary, encoding="utf-8")
    join_df.to_json(REPORT_ROOT / "grid_metric_distance_join.json", orient="records", indent=2, force_ascii=False)
    distance_df.to_json(REPORT_ROOT / "morph_distance_by_variant.json", orient="records", indent=2, force_ascii=False)

    manifest, snapshot = make_dashboard_payload(join_df, but_summary, corr, REPORT_ROOT / "morphology_analysis_summary.md")
    write_json(REPORT_ROOT / "data_analytics_manifest.json", manifest)
    write_json(REPORT_ROOT / "data_analytics_snapshot.json", snapshot)
    write_json(
        OUT_ROOT / "analysis_audit.json",
        {
            "but_features_rows": int(len(but_features)),
            "synthetic_features_rows": int(len(synth_features)),
            "variants_with_distance": int(len(distance_df)),
            "completed_large_rule_rows": completed_large,
            "skipped_variants": skipped,
            "outputs": output_paths,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_but_per_class", type=int, default=1600)
    parser.add_argument("--max_synth_per_class", type=int, default=450)
    parser.add_argument("--max_variants", type=int, default=0, help="0 means all variants with data")
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
