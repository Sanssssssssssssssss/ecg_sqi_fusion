from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from .common import SQIS, feature_cols_for_sqis, load_split_frame, write_json, write_table


POOR_GROUPS = ["original unacceptable", "synthetic em", "synthetic ma"]
SYNTHETIC_GROUPS = {
    "original poor vs em": ["synthetic em"],
    "original poor vs ma": ["synthetic ma"],
    "original poor vs combined synthetic": ["synthetic em", "synthetic ma"],
}


def _kernel_gamma_median_heuristic(X: np.ndarray) -> tuple[float, float]:
    d2 = squareform(pdist(X, metric="sqeuclidean"))
    upper = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    gamma = 1.0 / max(2.0 * med, 1e-12)
    return gamma, med


def _mmd2_from_kernel(K: np.ndarray, idx0: np.ndarray, idx1: np.ndarray) -> float:
    idx0 = np.asarray(idx0, dtype=int)
    idx1 = np.asarray(idx1, dtype=int)
    k00 = K[np.ix_(idx0, idx0)].mean()
    k11 = K[np.ix_(idx1, idx1)].mean()
    k01 = K[np.ix_(idx0, idx1)].mean()
    return float(k00 + k11 - 2.0 * k01)


def _source_sample_indices(
    frame: pd.DataFrame,
    groups: list[str],
    *,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = frame[frame["sample_group"].isin(groups)]
    by_source: dict[str, np.ndarray] = {
        str(source): g["poor_pos"].to_numpy(dtype=int)
        for source, g in pool.groupby("source_record_id", sort=True)
    }
    sources = np.asarray(sorted(by_source), dtype=object)
    if len(sources) < m:
        raise ValueError(f"Cannot sample {m} unique sources from groups {groups}; only {len(sources)} sources available.")
    chosen = rng.choice(sources, size=m, replace=False)
    out = np.empty(m, dtype=int)
    for i, source in enumerate(chosen):
        candidates = by_source[str(source)]
        out[i] = int(rng.choice(candidates))
    return out


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison, g in raw.groupby("comparison", sort=False):
        w = g["W_mmd2"].to_numpy(dtype=float)
        c = g["C_mmd2"].to_numpy(dtype=float)
        d = g["delta_mmd2"].to_numpy(dtype=float)
        rows.append(
            {
                "comparison": comparison,
                "n_resamples": int(len(g)),
                "m_per_side": int(g["m_per_side"].iloc[0]),
                "W_median": float(np.median(w)),
                "W_p05": float(np.percentile(w, 5)),
                "W_p95": float(np.percentile(w, 95)),
                "C_median": float(np.median(c)),
                "C_p05": float(np.percentile(c, 5)),
                "C_p95": float(np.percentile(c, 95)),
                "delta_median": float(np.median(d)),
                "delta_p05": float(np.percentile(d, 5)),
                "delta_p95": float(np.percentile(d, 95)),
                "p_delta_gt0": float(np.mean(d > 0.0)),
                "W_p95_lt_C_p05": bool(np.percentile(w, 95) < np.percentile(c, 5)),
                "C_median_over_W_median": float(np.median(c) / max(np.median(w), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def run_mmd_paired_resampling(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_split_frame(artifacts_dir, normalized=True)
    cols = feature_cols_for_sqis(SQIS)
    poor = df[df["sample_group"].isin(POOR_GROUPS)].copy().reset_index(drop=True)
    poor["poor_pos"] = np.arange(len(poor), dtype=int)

    X = StandardScaler().fit_transform(poor[cols].to_numpy(dtype=np.float64))
    gamma, median_sqdist = _kernel_gamma_median_heuristic(X)
    K = np.exp(-gamma * squareform(pdist(X, metric="sqeuclidean")))

    original_idx = poor.loc[poor["sample_group"].eq("original unacceptable"), "poor_pos"].to_numpy(dtype=int)
    n_original = int(len(original_idx))
    m = n_original // 2
    if m < 2:
        raise ValueError(f"Need at least four original poor records, found {n_original}.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for b in range(int(n_resamples)):
        shuffled_original = rng.permutation(original_idx)
        ref_idx = shuffled_original[:m]
        hold_idx = shuffled_original[m : 2 * m]
        W = _mmd2_from_kernel(K, ref_idx, hold_idx)
        for comparison, groups in SYNTHETIC_GROUPS.items():
            synth_idx = _source_sample_indices(poor, groups, m=m, rng=rng)
            C = _mmd2_from_kernel(K, ref_idx, synth_idx)
            rows.append(
                {
                    "iteration": b,
                    "comparison": comparison,
                    "m_per_side": int(m),
                    "W_mmd2": W,
                    "C_mmd2": C,
                    "delta_mmd2": float(C - W),
                    "gamma": float(gamma),
                    "median_sqdist": float(median_sqdist),
                    "n_original_total": n_original,
                    "n_synthetic_sources_available": int(
                        poor.loc[poor["sample_group"].isin(groups), "source_record_id"].astype(str).nunique()
                    ),
                    "synthetic_groups": ",".join(groups),
                    "sampling_unit": "source_record_id",
                    "seed": int(seed),
                }
            )

    raw = pd.DataFrame(rows)
    summary = _summarize(raw)

    full_rows: list[dict[str, Any]] = []
    for comparison, groups in SYNTHETIC_GROUPS.items():
        idx1 = poor.loc[poor["sample_group"].isin(groups), "poor_pos"].to_numpy(dtype=int)
        full_rows.append(
            {
                "comparison": comparison,
                "mmd2_full_sample_biased": _mmd2_from_kernel(K, original_idx, idx1),
                "n_original_poor": int(len(original_idx)),
                "n_synthetic_records": int(len(idx1)),
                "n_synthetic_sources": int(
                    poor.loc[poor["sample_group"].isin(groups), "source_record_id"].astype(str).nunique()
                ),
                "gamma": float(gamma),
                "median_sqdist": float(median_sqdist),
                "note": "Full-sample biased MMD is not directly compared with 112-vs-112 resampling ranges.",
            }
        )
    full = pd.DataFrame(full_rows)

    raw_csv = out / "mmd_paired_resampling_raw.csv"
    summary_csv = out / "mmd_paired_resampling_summary.csv"
    full_csv = out / "mmd_full_sample_reference.csv"
    write_table(raw, raw_csv)
    write_table(summary, summary_csv, md_path=out / "mmd_paired_resampling_summary.md")
    write_table(full, full_csv, md_path=out / "mmd_full_sample_reference.md")

    manifest = {
        "command": "mmd-paired-resampling",
        "artifacts_dir": str(Path(artifacts_dir)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "feature_source": "record84_norm.parquet",
        "feature_columns": "84 SQI features = 12 leads x 7 SQIs",
        "standardizer": "sklearn StandardScaler fit on all poor records, matching existing Figure 13 MMD implementation",
        "kernel": "RBF",
        "kernel_gamma_median_heuristic": float(gamma),
        "kernel_median_sqdist": float(median_sqdist),
        "m_per_side": int(m),
        "n_original_poor": n_original,
        "n_synthetic_em": int(poor["sample_group"].eq("synthetic em").sum()),
        "n_synthetic_ma": int(poor["sample_group"].eq("synthetic ma").sum()),
        "n_synthetic_combined": int(poor["sample_group"].isin(["synthetic em", "synthetic ma"]).sum()),
        "sampling_design": (
            "Each resample splits original poor into disjoint reference/holdout sets of size m, "
            "then samples m synthetic records by source_record_id. Combined synthetic first samples "
            "sources and then one available em/ma derivative per source."
        ),
        "outputs": [str(raw_csv), str(summary_csv), str(full_csv)],
    }
    manifest_path = write_json(out / "mmd_paired_resampling_manifest.json", manifest)
    return {
        "command": "mmd-paired-resampling",
        "outputs": [str(raw_csv), str(summary_csv), str(full_csv), str(manifest_path)],
        "summary": summary.to_dict(orient="records"),
        "full_sample_reference": full.to_dict(orient="records"),
    }
