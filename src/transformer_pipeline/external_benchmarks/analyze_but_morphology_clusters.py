"""Deep BUT morphology/domain analysis against PTB synthetic.

This script is intentionally analysis-only.  It extracts interpretable 10s ECG
features from the formal BUT 10s P1 protocol and the current PTB synthetic
training artifact, then checks:

* which morphology/usability features separate BUT good/medium/bad,
* whether BUT medium behaves like an independent cluster rather than a midpoint,
* how far current PTB synthetic is from BUT on those features,
* which synthetic rules should be changed next.

No mainline checkpoints or ``src/sqi_pipeline`` files are modified.
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
from scipy.signal import find_peaks, peak_widths
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.transformer_pipeline.e311_uformer_eval import write_json

try:
    import umap
except Exception:  # pragma: no cover - optional runtime dependency
    umap = None


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_morphology_cluster_analysis_10s_2026_06_04"
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
DEFAULT_PTB_ARTIFACT = (
    ROOT
    / "outputs"
    / "experiment"
    / "e311_morph_denoise_gap5_7_grid"
    / "data"
    / "med6p25_badgap7_badcm0p75"
)
DEFAULT_PTB_SPLIT = ROOT / "outputs" / "controls" / "e311f_ptb_sqi_three_class" / "splits" / "transformer_three_class_seed0.csv"
CLASS_NAMES = ("good", "medium", "bad")
FS = 125.0
N_TARGET = 1250


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_but(protocol_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(protocol_dir / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(protocol_dir / "metadata.csv")
    if X.ndim == 2:
        X = X[:, None, :]
    if X.shape[1:] != (1, N_TARGET):
        raise ValueError(f"BUT expected (N,1,{N_TARGET}), got {X.shape}")
    meta["dataset"] = "BUT 10s P1"
    meta["class_name"] = meta["y"].map(lambda v: CLASS_NAMES[int(v)])
    return X, meta


def load_ptb(artifact_dir: Path, split_csv: Path) -> tuple[np.ndarray, pd.DataFrame]:
    noisy = np.load(artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    split = pd.read_csv(split_csv)
    split["record_id"] = split["record_id"].astype(str)
    X = noisy[split["source_idx"].to_numpy(dtype=np.int64)][:, None, :]
    if X.shape[1:] != (1, N_TARGET):
        raise ValueError(f"PTB expected (N,1,{N_TARGET}), got {X.shape}")
    split["dataset"] = "PTB synthetic"
    split["class_name"] = split["y"].map(lambda v: CLASS_NAMES[int(v)])
    return X.astype(np.float32), split


def _safe_entropy(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(p) == 0 or float(np.sum(p)) <= 0:
        return 0.0
    p = p / float(np.sum(p))
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12))


def extract_one(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y0 = y - np.median(y)
    eps = 1e-8
    rms = float(np.sqrt(np.mean(y0**2)) + eps)
    std = float(np.std(y0) + eps)
    abs_y = np.abs(y0)
    dy = np.diff(y0)
    abs_dy = np.abs(dy)
    p01, p05, p50, p95, p99 = np.percentile(y0, [1, 5, 50, 95, 99])
    ptp = float(p99 - p01)

    # Window-level contact/flat/motion proxies.
    win = 125
    n_win = len(y0) // win
    chunks = y0[: n_win * win].reshape(n_win, win) if n_win > 0 else y0.reshape(1, -1)
    win_std = np.std(chunks, axis=1)
    win_mean = np.mean(chunks, axis=1)
    win_rms = np.sqrt(np.mean(chunks**2, axis=1))
    flatline_ratio = float(np.mean(abs_dy < max(0.0025, 0.015 * std)))
    contact_loss_win_ratio = float(np.mean(win_std < max(0.01, 0.12 * std)))
    local_rms_cv = float(np.std(win_rms) / (np.mean(win_rms) + eps))
    baseline_step = float(np.max(np.abs(np.diff(win_mean))) / (std + eps)) if len(win_mean) > 1 else 0.0

    # Spectrum.  Simple FFT is enough at 10s and avoids per-row Welch overhead.
    freqs = np.fft.rfftfreq(len(y0), d=1.0 / FS)
    power = np.abs(np.fft.rfft(y0)) ** 2
    total = float(np.sum(power[(freqs >= 0.3) & (freqs <= 45.0)]) + eps)
    lf_ratio = float(np.sum(power[(freqs >= 0.3) & (freqs < 1.0)]) / total)
    hf_ratio = float(np.sum(power[(freqs >= 20.0) & (freqs <= 45.0)]) / total)
    qrs_band_ratio = float(np.sum(power[(freqs >= 5.0) & (freqs <= 18.0)]) / total)
    spectral_entropy = _safe_entropy(power[(freqs >= 0.3) & (freqs <= 45.0)])

    # Peak/QRS proxies.  We deliberately use generic peak geometry, not a
    # specialized detector, so the features apply to noisy unusable ECG too.
    min_distance = int(0.28 * FS)
    peaks, props = find_peaks(abs_y, distance=min_distance, prominence=max(0.08 * std, 0.01))
    aggr_peaks, aggr_props = find_peaks(abs_y, distance=int(0.08 * FS), prominence=max(0.04 * std, 0.006))
    if len(peaks):
        prom = props.get("prominences", np.zeros(len(peaks)))
        widths = peak_widths(abs_y, peaks, rel_height=0.5)[0] / FS
        qrs_prom_median = float(np.median(prom) / (rms + eps))
        qrs_prom_p90 = float(np.percentile(prom, 90) / (rms + eps))
        qrs_width_median = float(np.median(widths))
        slopes = []
        for p in peaks[:40]:
            lo = max(0, p - 3)
            hi = min(len(y0), p + 4)
            slopes.append(np.max(np.abs(np.diff(y0[lo:hi]))) if hi - lo > 1 else 0.0)
        qrs_slope_median = float(np.median(slopes) / (rms + eps)) if slopes else 0.0
    else:
        qrs_prom_median = 0.0
        qrs_prom_p90 = 0.0
        qrs_width_median = 0.0
        qrs_slope_median = 0.0
    qrs_peak_count = float(len(peaks))
    aggr_peak_count = float(len(aggr_peaks))
    spurious_peak_density = float(max(0.0, aggr_peak_count - qrs_peak_count) / 10.0)
    qrs_count_low = float(qrs_peak_count < 5)
    qrs_count_high = float(qrs_peak_count > 22)
    qrs_count_deviation = float(min(abs(qrs_peak_count - 11.0), 14.0) / 14.0)
    qrs_visibility = float(qrs_prom_median * qrs_band_ratio / (1.0 + spurious_peak_density))

    # Periodicity proxy from peak spacing.  This is intentionally cheap because
    # the analysis runs over tens of thousands of windows.
    if len(peaks) >= 3:
        rr = np.diff(peaks) / FS
        periodicity = float(1.0 / (1.0 + np.std(rr) / (np.mean(rr) + eps)))
    else:
        periodicity = 0.0

    # Usability dimensions inspired by the current hypothesis.
    low_amp_ratio = float(np.mean(abs_y < 0.12 * rms))
    clipping_like_ratio = float(np.mean(abs_y > np.percentile(abs_y, 98.5) - eps))
    detail_instability = float((np.percentile(abs_dy, 95) / (np.percentile(abs_dy, 50) + eps)) / 10.0)
    fatal_or_score = float(
        max(
            contact_loss_win_ratio,
            min(1.0, flatline_ratio * 1.8),
            min(1.0, hf_ratio * 8.0),
            min(1.0, spurious_peak_density / 4.0),
            qrs_count_low,
            qrs_count_high,
            min(1.0, baseline_step / 4.0),
        )
    )
    medium_detail_unreliable_score = float(
        qrs_visibility
        * min(1.0, 0.35 * detail_instability + 0.50 * hf_ratio * 8.0 + 0.35 * min(1.0, baseline_step / 3.0))
    )

    return {
        "rms": rms,
        "std": std,
        "ptp_p99_p01": ptp,
        "mean_abs": float(np.mean(abs_y)),
        "low_amp_ratio": low_amp_ratio,
        "flatline_ratio": flatline_ratio,
        "contact_loss_win_ratio": contact_loss_win_ratio,
        "local_rms_cv": local_rms_cv,
        "baseline_step": baseline_step,
        "diff_abs_median": float(np.median(abs_dy)),
        "diff_abs_p95": float(np.percentile(abs_dy, 95)),
        "detail_instability": detail_instability,
        "lf_ratio": lf_ratio,
        "hf_ratio": hf_ratio,
        "qrs_band_ratio": qrs_band_ratio,
        "spectral_entropy": spectral_entropy,
        "qrs_peak_count": qrs_peak_count,
        "aggressive_peak_count": aggr_peak_count,
        "spurious_peak_density": spurious_peak_density,
        "qrs_prom_median": qrs_prom_median,
        "qrs_prom_p90": qrs_prom_p90,
        "qrs_width_median": qrs_width_median,
        "qrs_slope_median": qrs_slope_median,
        "qrs_count_low": qrs_count_low,
        "qrs_count_high": qrs_count_high,
        "qrs_count_deviation": qrs_count_deviation,
        "periodicity": periodicity,
        "qrs_visibility": qrs_visibility,
        "clipping_like_ratio": clipping_like_ratio,
        "fatal_or_score": fatal_or_score,
        "medium_detail_unreliable_score": medium_detail_unreliable_score,
    }


FEATURE_COLUMNS = list(extract_one(np.zeros(N_TARGET)).keys())


def extract_features(X: np.ndarray, meta: pd.DataFrame, cache_path: Path, max_rows: int = 0) -> pd.DataFrame:
    if cache_path.exists() and max_rows <= 0:
        return pd.read_csv(cache_path)
    n = len(X) if max_rows <= 0 else min(len(X), int(max_rows))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row = extract_one(X[i, 0])
        m = meta.iloc[i]
        row.update(
            {
                "idx": int(i),
                "dataset": str(m.get("dataset", "")),
                "split": str(m.get("split", "")),
                "y": int(m.get("y")),
                "class_name": str(m.get("class_name")),
                "record_id": str(m.get("record_id", m.get("window_id", i))),
                "subject_id": str(m.get("subject_id", "")),
                "label_raw": str(m.get("label_raw", "")),
            }
        )
        rows.append(row)
        if (i + 1) % 5000 == 0:
            print(f"features {cache_path.name}: {i + 1}/{n}", flush=True)
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if max_rows <= 0:
        df.to_csv(cache_path, index=False)
    return df


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return math.nan
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def summarize_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(df["dataset"].unique()):
        for cls in CLASS_NAMES:
            sub = df[(df["dataset"] == dataset) & (df["class_name"] == cls)]
            for feat in FEATURE_COLUMNS:
                x = sub[feat].to_numpy(dtype=float)
                rows.append(
                    {
                        "dataset": dataset,
                        "class_name": cls,
                        "feature": feat,
                        "n": int(len(x)),
                        "mean": float(np.nanmean(x)),
                        "median": float(np.nanmedian(x)),
                        "p10": float(np.nanpercentile(x, 10)),
                        "p90": float(np.nanpercentile(x, 90)),
                        "std": float(np.nanstd(x)),
                    }
                )
    return pd.DataFrame(rows)


def domain_distance(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cls in CLASS_NAMES:
        ptb = df[(df["dataset"] == "PTB synthetic") & (df["class_name"] == cls)]
        but = df[(df["dataset"] == "BUT 10s P1") & (df["class_name"] == cls)]
        for feat in FEATURE_COLUMNS:
            a = ptb[feat].to_numpy(dtype=float)
            b = but[feat].to_numpy(dtype=float)
            pooled = np.nanstd(np.concatenate([a, b])) + 1e-8
            rows.append(
                {
                    "class_name": cls,
                    "feature": feat,
                    "ptb_median": float(np.nanmedian(a)),
                    "but_median": float(np.nanmedian(b)),
                    "standardized_median_delta": float((np.nanmedian(b) - np.nanmedian(a)) / pooled),
                    "ks": ks_stat(a, b),
                }
            )
    return pd.DataFrame(rows)


def effect_alignment(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contrasts = [("medium", "good"), ("bad", "good"), ("bad", "medium")]
    for dataset in sorted(df["dataset"].unique()):
        for feat in FEATURE_COLUMNS:
            for hi, lo in contrasts:
                a = df[(df["dataset"] == dataset) & (df["class_name"] == hi)][feat].to_numpy(dtype=float)
                b = df[(df["dataset"] == dataset) & (df["class_name"] == lo)][feat].to_numpy(dtype=float)
                pooled = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2.0) + 1e-8
                rows.append(
                    {
                        "dataset": dataset,
                        "feature": feat,
                        "contrast": f"{hi}_minus_{lo}",
                        "effect_d": float((np.nanmean(a) - np.nanmean(b)) / pooled),
                        "median_delta": float(np.nanmedian(a) - np.nanmedian(b)),
                    }
                )
    eff = pd.DataFrame(rows)
    piv = eff.pivot_table(index=["feature", "contrast"], columns="dataset", values="effect_d", aggfunc="first").reset_index()
    piv["effect_alignment"] = np.sign(piv["PTB synthetic"].fillna(0.0)) * np.sign(piv["BUT 10s P1"].fillna(0.0))
    piv["abs_effect_gap"] = (piv["PTB synthetic"] - piv["BUT 10s P1"]).abs()
    return piv


def rf_reports(df: pd.DataFrame, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y_class = df["y"].to_numpy(dtype=int)
    y_domain = (df["dataset"] == "BUT 10s P1").astype(int).to_numpy()
    ds = df["dataset"].to_numpy()

    but = ds == "BUT 10s P1"
    Xb = X[but]
    yb = y_class[but]
    idx = np.arange(len(yb))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=yb)
    clf = RandomForestClassifier(n_estimators=260, max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xb[train_idx], yb[train_idx])
    pred = clf.predict(Xb[test_idx])
    but_report = {
        "balanced_acc": float(balanced_accuracy_score(yb[test_idx], pred)),
        "confusion": confusion_matrix(yb[test_idx], pred).tolist(),
        "classification_report": classification_report(yb[test_idx], pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }
    but_imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)

    idx2 = np.arange(len(y_domain))
    tr, te = train_test_split(idx2, test_size=0.25, random_state=43, stratify=y_domain)
    dom = RandomForestClassifier(n_estimators=260, max_depth=12, min_samples_leaf=30, class_weight="balanced", random_state=43, n_jobs=-1)
    dom.fit(X[tr], y_domain[tr])
    pred2 = dom.predict(X[te])
    domain_report = {
        "balanced_acc": float(balanced_accuracy_score(y_domain[te], pred2)),
        "confusion": confusion_matrix(y_domain[te], pred2).tolist(),
    }
    dom_imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": dom.feature_importances_}).sort_values("importance", ascending=False)

    write_json(out_root / "but_rf_report.json", but_report)
    write_json(out_root / "domain_rf_report.json", domain_report)
    return but_imp, dom_imp, {"but": but_report, "domain": domain_report}


def cluster_but(df: pd.DataFrame, out_root: Path, n_clusters: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    but = df[df["dataset"] == "BUT 10s P1"].copy()
    X = but[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=7, n_init=20)
    labels = km.fit_predict(Z)
    but["cluster"] = labels
    comp = (
        but.groupby(["cluster", "class_name"])
        .size()
        .reset_index(name="n")
        .assign(share=lambda d: d["n"] / d.groupby("cluster")["n"].transform("sum"))
    )
    centers = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLUMNS)
    centers["cluster"] = np.arange(n_clusters)
    sample_idx = np.random.default_rng(7).choice(len(Z), size=min(6000, len(Z)), replace=False)
    sil = float(silhouette_score(Z[sample_idx], labels[sample_idx])) if len(sample_idx) > n_clusters else math.nan
    audit = {"n_clusters": n_clusters, "silhouette_sample": sil}
    but[["idx", "cluster", "class_name", "split", "record_id", "subject_id", *FEATURE_COLUMNS]].to_csv(out_root / "but_cluster_assignments.csv", index=False)
    comp.to_csv(out_root / "but_cluster_composition.csv", index=False)
    centers.to_csv(out_root / "but_cluster_centers_standardized.csv", index=False)
    write_json(out_root / "but_cluster_audit.json", audit)
    return but, comp, audit


def embedding(df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    parts = []
    for dataset in ("BUT 10s P1", "PTB synthetic"):
        for cls in CLASS_NAMES:
            sub = df[(df["dataset"] == dataset) & (df["class_name"] == cls)]
            take = min(len(sub), 900)
            parts.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
    sample = pd.concat(parts, ignore_index=True)
    X = sample[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=11)
    emb = pca.fit_transform(Z)
    sample["pca1"] = emb[:, 0]
    sample["pca2"] = emb[:, 1]
    if umap is not None:
        reducer = umap.UMAP(n_neighbors=35, min_dist=0.10, metric="euclidean", random_state=11)
        u = reducer.fit_transform(Z)
        sample["umap1"] = u[:, 0]
        sample["umap2"] = u[:, 1]
    else:
        sample["umap1"] = sample["pca1"]
        sample["umap2"] = sample["pca2"]
    sample.to_csv(out_root / "embedding_sample.csv", index=False)
    return sample


def plot_wave_gallery(X: np.ndarray, meta: pd.DataFrame, rows: pd.DataFrame, title: str, path: Path, n: int = 18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.head(n)
    cols = 3
    rows_n = int(math.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, max(3, rows_n * 2.1)), squeeze=False)
    t = np.arange(N_TARGET) / FS
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), rows.itertuples()):
        idx = int(item.idx)
        y = X[idx, 0]
        ax.plot(t, y, lw=0.8, color="#263238")
        ax.set_title(f"{getattr(item, 'class_name')} idx={idx} c={getattr(item, 'cluster', '-')}", fontsize=8)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    df: pd.DataFrame,
    but_clustered: pd.DataFrame,
    comp: pd.DataFrame,
    emb: pd.DataFrame,
    distance: pd.DataFrame,
    effects: pd.DataFrame,
    but_imp: pd.DataFrame,
    domain_imp: pd.DataFrame,
    X_but: np.ndarray,
    but_meta: pd.DataFrame,
    report_root: Path,
) -> None:
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=emb, x="umap1", y="umap2", hue="class_name", style="dataset", s=16, alpha=0.65, ax=ax)
    ax.set_title("Morphology embedding: BUT classes overlap PTB in uneven ways")
    fig.tight_layout()
    fig.savefig(fig_dir / "embedding_dataset_class.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    but_emb = emb[emb["dataset"] == "BUT 10s P1"]
    sns.scatterplot(data=but_emb, x="umap1", y="umap2", hue="class_name", s=18, alpha=0.75, ax=ax)
    ax.set_title("BUT-only morphology embedding")
    fig.tight_layout()
    fig.savefig(fig_dir / "but_embedding_class.png", dpi=180)
    plt.close(fig)

    pivot = comp.pivot(index="cluster", columns="class_name", values="share").fillna(0.0)[list(CLASS_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#2b6cb0", "#d69e2e", "#c05621"])
    ax.set_ylabel("class share")
    ax.set_title("BUT KMeans clusters show whether medium is its own mode")
    fig.tight_layout()
    fig.savefig(fig_dir / "but_cluster_composition.png", dpi=180)
    plt.close(fig)

    for name, imp in [("but_class_rf_importance", but_imp), ("domain_rf_importance", domain_imp)]:
        fig, ax = plt.subplots(figsize=(8, 5))
        top = imp.head(14).iloc[::-1]
        ax.barh(top["feature"], top["importance"], color="#2b6cb0" if name.startswith("but") else "#c05621")
        ax.set_title("BUT class feature importance" if name.startswith("but") else "PTB vs BUT domain feature importance")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=180)
        plt.close(fig)

    top_domain = distance.sort_values("ks", ascending=False).head(8)["feature"].drop_duplicates().head(6).tolist()
    long = df.melt(id_vars=["dataset", "class_name"], value_vars=top_domain, var_name="feature", value_name="value")
    fig = plt.figure(figsize=(14, 7))
    g = sns.catplot(data=long.sample(min(len(long), 18000), random_state=5), x="class_name", y="value", hue="dataset", col="feature", kind="box", col_wrap=3, sharey=False, showfliers=False, height=3.0, aspect=1.25)
    g.fig.suptitle("Top morphology domain mismatches: PTB synthetic vs BUT", y=1.02)
    g.savefig(fig_dir / "top_domain_mismatch_boxplots.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    med_feats = ["qrs_visibility", "medium_detail_unreliable_score", "fatal_or_score", "spurious_peak_density", "baseline_step", "contact_loss_win_ratio"]
    med_feats = [f for f in med_feats if f in FEATURE_COLUMNS]
    long2 = df[df["dataset"] == "BUT 10s P1"].melt(id_vars=["class_name"], value_vars=med_feats, var_name="feature", value_name="value")
    g = sns.catplot(data=long2.sample(min(len(long2), 20000), random_state=6), x="class_name", y="value", col="feature", kind="box", col_wrap=3, sharey=False, showfliers=False, height=3.0, aspect=1.25, color="#718096")
    g.fig.suptitle("BUT medium: QRS usable but detail/fatal dimensions vary independently", y=1.02)
    g.savefig(fig_dir / "but_medium_hypothesis_features.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    # Waveform galleries: class profiles and cluster profiles.
    galleries = report_root / "figures" / "galleries"
    rng = np.random.default_rng(17)
    class_rows = []
    for cls in CLASS_NAMES:
        sub = but_clustered[but_clustered["class_name"] == cls]
        class_rows.append(sub.iloc[rng.choice(len(sub), size=min(9, len(sub)), replace=False)])
    plot_wave_gallery(X_but, but_meta, pd.concat(class_rows), "BUT random class gallery", galleries / "but_class_gallery.png", n=27)
    for c in sorted(but_clustered["cluster"].unique())[:8]:
        sub = but_clustered[but_clustered["cluster"] == c].sort_values("fatal_or_score", ascending=False)
        plot_wave_gallery(X_but, but_meta, sub, f"BUT cluster {c} high fatal score gallery", galleries / f"but_cluster_{c}.png", n=15)


def write_report(
    report_root: Path,
    out_root: Path,
    rf_payload: dict[str, Any],
    cluster_audit: dict[str, Any],
    comp: pd.DataFrame,
    distance: pd.DataFrame,
    effects: pd.DataFrame,
    but_imp: pd.DataFrame,
    domain_imp: pd.DataFrame,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    medium_clusters = (
        comp[comp["class_name"] == "medium"].sort_values("share", ascending=False).head(3)[["cluster", "share"]].to_dict(orient="records")
    )
    top_but = ", ".join(f"{r.feature} ({r.importance:.3f})" for r in but_imp.head(8).itertuples())
    top_domain = ", ".join(f"{r.feature} ({r.importance:.3f})" for r in domain_imp.head(8).itertuples())
    top_shift = ", ".join(f"{r.feature}/{r.class_name} KS={r.ks:.2f}" for r in distance.sort_values("ks", ascending=False).head(8).itertuples())
    flipped = effects[effects["effect_alignment"] < 0].sort_values("abs_effect_gap", ascending=False).head(8)
    lines = [
        "# BUT Morphology Cluster Analysis vs PTB Synthetic",
        "",
        "## Executive Summary",
        "",
        f"- BUT good/medium/bad are separable from morphology/usability features with random-forest balanced accuracy `{rf_payload['but']['balanced_acc']:.3f}`. This means there is real signal beyond label noise.",
        f"- PTB synthetic vs BUT is also highly separable with balanced accuracy `{rf_payload['domain']['balanced_acc']:.3f}`, so our generated data still carries a strong synthetic-domain fingerprint.",
        f"- Medium appears partly independent: the most medium-heavy clusters are {medium_clusters}, not simply a clean interpolation between good and bad.",
        "- The next generator should model bad as sample-level OR subtypes, while medium should preserve QRS visibility and vary local detail/P-T/ST/baseline reliability independently.",
        "",
        "## Strongest BUT Class Features",
        "",
        f"{top_but}",
        "",
        "## Strongest PTB-vs-BUT Domain Features",
        "",
        f"{top_domain}",
        "",
        "## Biggest Synthetic Mismatches",
        "",
        f"{top_shift}",
        "",
        "## Effect-Direction Flips",
        "",
        "| feature | contrast | PTB effect | BUT effect | abs gap |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in flipped.to_dict(orient="records"):
        lines.append(f"| {row['feature']} | {row['contrast']} | {row['PTB synthetic']:.3f} | {row['BUT 10s P1']:.3f} | {row['abs_effect_gap']:.3f} |")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Embedding by dataset and class](figures/embedding_dataset_class.png)",
            "",
            "![BUT embedding by class](figures/but_embedding_class.png)",
            "",
            "![BUT cluster composition](figures/but_cluster_composition.png)",
            "",
            "![BUT class feature importance](figures/but_class_rf_importance.png)",
            "",
            "![Domain feature importance](figures/domain_rf_importance.png)",
            "",
            "![Top domain mismatch boxplots](figures/top_domain_mismatch_boxplots.png)",
            "",
            "![BUT medium hypothesis features](figures/but_medium_hypothesis_features.png)",
            "",
            "![BUT class waveform gallery](figures/galleries/but_class_gallery.png)",
            "",
            "## Output Tables",
            "",
            f"- Local output root: `{out_root}`",
            "- `morph_feature_summary.csv`",
            "- `morph_domain_distance.csv`",
            "- `morph_effect_alignment.csv`",
            "- `but_cluster_composition.csv`",
            "- `but_rf_importance.csv` and `domain_rf_importance.csv`",
        ]
    )
    (report_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        report_root / "morphology_cluster_summary.json",
        {
            "but_rf_balanced_acc": rf_payload["but"]["balanced_acc"],
            "domain_rf_balanced_acc": rf_payload["domain"]["balanced_acc"],
            "medium_heavy_clusters": medium_clusters,
            "top_but_features": but_imp.head(12).to_dict(orient="records"),
            "top_domain_features": domain_imp.head(12).to_dict(orient="records"),
            "cluster_audit": cluster_audit,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "morphology_cluster_state.json"
    write_json(state_path, {"status": "loading", "updated_at": now_iso()})

    X_but, meta_but = load_but(Path(args.but_protocol_dir))
    X_ptb, meta_ptb = load_ptb(Path(args.ptb_artifact_dir), Path(args.ptb_split_csv))
    write_json(state_path, {"status": "extracting_features", "but_n": int(len(X_but)), "ptb_n": int(len(X_ptb)), "updated_at": now_iso()})
    but_feat = extract_features(X_but, meta_but, out_root / "but_morph_features.csv", max_rows=int(args.max_but_rows))
    ptb_feat = extract_features(X_ptb, meta_ptb, out_root / "ptb_morph_features.csv", max_rows=int(args.max_ptb_rows))
    df = pd.concat([but_feat, ptb_feat], ignore_index=True)
    df.to_csv(out_root / "combined_morph_features.csv", index=False)

    write_json(state_path, {"status": "analyzing", "updated_at": now_iso()})
    summary = summarize_features(df)
    dist = domain_distance(df)
    eff = effect_alignment(df)
    but_imp, dom_imp, rf_payload = rf_reports(df, out_root)
    but_clustered, comp, cluster_audit = cluster_but(df, out_root, n_clusters=int(args.n_clusters))
    emb = embedding(df, out_root)
    summary.to_csv(out_root / "morph_feature_summary.csv", index=False)
    dist.to_csv(out_root / "morph_domain_distance.csv", index=False)
    eff.to_csv(out_root / "morph_effect_alignment.csv", index=False)
    but_imp.to_csv(out_root / "but_rf_importance.csv", index=False)
    dom_imp.to_csv(out_root / "domain_rf_importance.csv", index=False)

    write_json(state_path, {"status": "plotting", "updated_at": now_iso()})
    make_figures(df, but_clustered, comp, emb, dist, eff, but_imp, dom_imp, X_but, meta_but, report_root)
    write_report(report_root, out_root, rf_payload, cluster_audit, comp, dist, eff, but_imp, dom_imp)

    payload = {
        "status": "complete",
        "but_rows": int(len(but_feat)),
        "ptb_rows": int(len(ptb_feat)),
        "but_rf_balanced_acc": rf_payload["but"]["balanced_acc"],
        "domain_rf_balanced_acc": rf_payload["domain"]["balanced_acc"],
        "report": str(report_root / "README.md"),
        "updated_at": now_iso(),
    }
    write_json(state_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep BUT morphology cluster analysis vs PTB synthetic.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--ptb_artifact_dir", default=str(DEFAULT_PTB_ARTIFACT))
    parser.add_argument("--ptb_split_csv", default=str(DEFAULT_PTB_SPLIT))
    parser.add_argument("--max_but_rows", type=int, default=0)
    parser.add_argument("--max_ptb_rows", type=int, default=0)
    parser.add_argument("--n_clusters", type=int, default=7)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
