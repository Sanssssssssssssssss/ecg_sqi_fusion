from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import (
    GROUP_ORDER,
    binary_metrics,
    load_split_frame,
    max_accuracy_threshold,
    write_table,
)
from .plotting import plot_fsqi_mechanism


THRESHOLDS_MV = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], dtype=float)
RUN_DURATIONS_SEC = (0.25, 0.5, 1.0)
FS = 125


def _load_sig125(art: Path, record_id: str) -> np.ndarray:
    z = np.load(art / "resampled_125" / f"{record_id}.npz", allow_pickle=True)
    return z["sig_125"].astype(np.float64)


def _flat_run_fraction_2d(abs_dx: np.ndarray, threshold: float, min_len: int) -> float:
    mask2 = np.asarray(abs_dx < float(threshold), dtype=bool)
    if mask2.size == 0:
        return 1.0
    if mask2.ndim == 1:
        mask2 = mask2[:, None]
    covered2 = np.zeros(mask2.shape, dtype=bool)
    for lead in range(mask2.shape[1]):
        mask = mask2[:, lead]
        padded = np.concatenate([[False], mask, [False]])
        starts = np.flatnonzero((~padded[:-1]) & padded[1:])
        ends = np.flatnonzero(padded[:-1] & (~padded[1:]))
        for start, end in zip(starts, ends):
            if end - start >= min_len:
                covered2[start:end, lead] = True
    return float(np.mean(covered2))


def build_logdiff_summary(artifacts_dir: str | Path, *, max_records: int | None = None) -> pd.DataFrame:
    art = Path(artifacts_dir)
    df = load_split_frame(art, normalized=False)
    if max_records is not None:
        df = df.groupby("sample_group", group_keys=False).head(int(max_records)).copy()
    rows = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        rid = str(row.record_id)
        sig = _load_sig125(art, rid)
        abs_dx_2d = np.abs(np.diff(sig, axis=0))
        abs_dx = abs_dx_2d.reshape(-1)
        logdx = np.log10(abs_dx + 1e-12)
        flat_fracs = {f"flat_fraction_lt_{thr:g}": float(np.mean(abs_dx_2d < thr)) for thr in THRESHOLDS_MV}
        run_fracs: dict[str, float] = {}
        for thr in THRESHOLDS_MV:
            for dur in RUN_DURATIONS_SEC:
                min_len = max(1, int(round(dur * FS)))
                run_fracs[f"flat_run_fraction_lt_{thr:g}_dur_{dur:g}s"] = _flat_run_fraction_2d(abs_dx_2d, thr, min_len)
        rows.append(
            {
                "record_id": rid,
                "source_record_id": str(row.source_record_id),
                "split": str(row.split),
                "y": int(row.y),
                "y01": int(row.y01),
                "sample_group": str(row.sample_group),
                "median_log10_abs_diff": float(np.median(logdx)),
                "mean_log10_abs_diff": float(np.mean(logdx)),
                "q05_log10_abs_diff": float(np.quantile(logdx, 0.05)),
                "q95_log10_abs_diff": float(np.quantile(logdx, 0.95)),
                **flat_fracs,
                **run_fracs,
            }
        )
    return pd.DataFrame(rows)


def _score_scan(summary: pd.DataFrame, score_col: str, score_name: str, threshold_mv: float, duration_s: float | None = None) -> dict[str, Any]:
    tr = summary["split"].astype(str).eq("train").to_numpy()
    va = summary["split"].astype(str).eq("val").to_numpy()
    te = summary["split"].astype(str).eq("test").to_numpy()
    yva = summary.loc[va, "y01"].to_numpy(dtype=int)
    yte = summary.loc[te, "y01"].to_numpy(dtype=int)
    # Test both orientations; fSQI can be poor-high for flatline or poor-low for noise.
    best_orientation = 1.0
    best_val = None
    for orientation in (1.0, -1.0):
        raw = summary.loc[va, score_col].to_numpy(dtype=float)
        vals = orientation * raw
        vals = (vals - np.nanmin(vals)) / max(1e-12, np.nanmax(vals) - np.nanmin(vals))
        cur = max_accuracy_threshold(yva, vals, n_grid=1001)
        if best_val is None or cur["Ac"] > best_val["Ac"]:
            best_val = cur
            best_orientation = orientation
    assert best_val is not None
    all_raw = summary[score_col].to_numpy(dtype=float)
    all_score = best_orientation * all_raw
    all_score = (all_score - np.nanmin(all_score)) / max(1e-12, np.nanmax(all_score) - np.nanmin(all_score))
    met = binary_metrics(yte, all_score[te], float(best_val["threshold"]))
    try:
        auc = float(roc_auc_score(yte, all_score[te]))
    except Exception:
        auc = float("nan")
    return {
        "score_name": score_name,
        "score_col": score_col,
        "threshold_mv": float(threshold_mv),
        "min_run_duration_s": np.nan if duration_s is None else float(duration_s),
        "orientation": "as_is" if best_orientation > 0 else "inverted",
        "val_Ac": float(best_val["Ac"]),
        "val_threshold": float(best_val["threshold"]),
        "test_Ac": float(met["Ac"]),
        "test_Se": float(met["Se"]),
        "test_Sp": float(met["Sp"]),
        "test_AUC": auc,
    }


def threshold_scan(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in THRESHOLDS_MV:
        rows.append(_score_scan(summary, f"flat_fraction_lt_{thr:g}", "flat_fraction", float(thr)))
        for dur in RUN_DURATIONS_SEC:
            rows.append(_score_scan(summary, f"flat_run_fraction_lt_{thr:g}_dur_{dur:g}s", "flat_run_fraction", float(thr), dur))
    return pd.DataFrame(rows)


def subgroup_distribution(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        g = summary[summary["sample_group"].eq(group)]
        if g.empty:
            continue
        rows.append(
            {
                "sample_group": group,
                "n": int(len(g)),
                "median_log10_abs_diff": float(g["median_log10_abs_diff"].median()),
                "q25_log10_abs_diff": float(g["median_log10_abs_diff"].quantile(0.25)),
                "q75_log10_abs_diff": float(g["median_log10_abs_diff"].quantile(0.75)),
                "flat_fraction_1e-4_median": float(g["flat_fraction_lt_0.0001"].median()),
            }
        )
    return pd.DataFrame(rows)


def run_fsqi_mechanism(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    report_dir: str | Path,
    max_records: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    rep = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    summary_csv = out / "fsqi_logdiff_record_summary.csv"
    if summary_csv.exists() and not force:
        summary = pd.read_csv(summary_csv)
    else:
        summary = build_logdiff_summary(artifacts_dir, max_records=max_records)
        write_table(summary, summary_csv)
    dist = subgroup_distribution(summary)
    dist_csv = out / "fsqi_logdiff_subgroup_distribution.csv"
    write_table(dist, dist_csv, md_path=rep / "fsqi_logdiff_subgroup_distribution.md")
    scan = threshold_scan(summary)
    scan_csv = out / "fsqi_threshold_scan.csv"
    write_table(scan, scan_csv, md_path=rep / "fsqi_threshold_scan.md")
    plot_paths = plot_fsqi_mechanism(summary, scan, rep / "fig_supp_03_fsqi_mechanism")
    return {"outputs": [str(summary_csv), str(dist_csv), str(scan_csv), *[str(p) for p in plot_paths]]}
