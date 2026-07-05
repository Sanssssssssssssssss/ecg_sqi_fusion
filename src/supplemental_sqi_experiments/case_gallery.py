from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.supplemental_sqi_experiments.common import (
    LEADS_12,
    SQIS,
    load_split_frame,
    project_root,
    write_table,
)
from src.supplemental_sqi_experiments.plotting import PALETTE, apply_style, save_figure


def _add_record_summaries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for sqi in SQIS:
        cols = [f"{lead}__{sqi}" for lead in LEADS_12]
        out[f"mean_{sqi}"] = out[cols].mean(axis=1)
        out[f"median_{sqi}"] = out[cols].median(axis=1)
    out["mean_1_minus_basSQI"] = 1.0 - out["mean_basSQI"]
    out["mean_abs_sSQI"] = out[[f"{lead}__sSQI" for lead in LEADS_12]].abs().mean(axis=1)
    return out


def _pick_first(
    frame: pd.DataFrame,
    *,
    selected_record_ids: set[str],
    selected_source_ids: set[str] | None = None,
    sort_col: str,
    ascending: bool,
    role: str,
    mask: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> dict[str, object] | None:
    pool = frame.copy()
    if mask is not None:
        pool = pool[mask(pool)]
    pool = pool[~pool["record_id"].astype(str).isin(selected_record_ids)]
    if selected_source_ids is not None:
        distinct = pool[~pool["source_record_id"].astype(str).isin(selected_source_ids)]
        if not distinct.empty:
            pool = distinct
    if pool.empty:
        return None
    row = pool.sort_values(sort_col, ascending=ascending).iloc[0].to_dict()
    row["selection_role"] = role
    return row


def _lead_wave_metrics(artifacts_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        rid = str(row.record_id)
        sig, leads, _fs = _load_signal(artifacts_dir, rid)
        for i, lead in enumerate(leads):
            x = np.asarray(sig[:, i], dtype=float)
            finite = x[np.isfinite(x)]
            if finite.size == 0:
                continue
            dx = np.diff(finite)
            q = np.nanpercentile(finite, [1, 5, 50, 95, 99])
            p2p99 = float(q[4] - q[0])
            p2p90 = float(q[3] - q[1])
            rows.append(
                {
                    "record_id": rid,
                    "source_record_id": str(row.source_record_id),
                    "split": str(row.split),
                    "lead": str(lead),
                    "p2p99": p2p99,
                    "p2p90": p2p90,
                    "median": float(q[2]),
                    "flat_frac_wave": float(np.mean(np.abs(dx) < 1e-4)) if dx.size else 1.0,
                    "zero_frac_wave": float(np.mean(np.abs(finite) < 1e-4)),
                    "sat80_frac_wave": float(np.mean(np.abs(np.abs(finite) - 80.0) < 0.5)),
                    "diff_mad_wave": float(np.median(np.abs(dx - np.median(dx)))) if dx.size else 0.0,
                    "tv_wave": float(np.mean(np.abs(dx))) if dx.size else 0.0,
                    "bSQI": float(getattr(row, f"{lead}__bSQI")),
                    "iSQI": float(getattr(row, f"{lead}__iSQI")),
                    "pSQI": float(getattr(row, f"{lead}__pSQI")),
                    "kSQI": float(getattr(row, f"{lead}__kSQI")),
                    "fSQI": float(getattr(row, f"{lead}__fSQI")),
                    "basSQI": float(getattr(row, f"{lead}__basSQI")),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["const_extreme_wave"] = (out["p2p90"].lt(0.05) & out["median"].abs().gt(20)).astype(int)
    return out


def _select_visual_original_examples(original: pd.DataFrame, artifacts_dir: Path) -> list[dict[str, object]]:
    lead_metrics = _lead_wave_metrics(artifacts_dir, original)
    if lead_metrics.empty:
        return []
    by_record = original.set_index("record_id")
    rows: list[dict[str, object]] = []
    used_records: set[str] = set()

    specs = [
        ("flatline", lambda x: x["zero_frac_wave"] > 0.99, "zero_frac_wave", False, "fSQI"),
        ("saturation", lambda x: (x["const_extreme_wave"] > 0) | (x["sat80_frac_wave"] > 0.80), "sat80_frac_wave", False, "basSQI"),
        ("large artifact", lambda x: (x["p2p99"] > 20.0) & (x["zero_frac_wave"] < 0.80), "p2p99", False, "p2p99"),
        ("high-frequency noise", lambda x: (x["diff_mad_wave"] > 0.03) & (x["p2p99"] < 8.0) & (x["zero_frac_wave"] < 0.50), "diff_mad_wave", False, "diff_mad_wave"),
        ("low pSQI nonflat", lambda x: (x["pSQI"] < 0.45) & (x["zero_frac_wave"] < 0.50) & (x["const_extreme_wave"] == 0), "pSQI", True, "pSQI"),
    ]
    for role, mask_fn, sort_col, ascending, metric in specs:
        pool = lead_metrics[mask_fn(lead_metrics)].copy()
        pool = pool[~pool["record_id"].isin(used_records)]
        if pool.empty:
            continue
        lead_row = pool.sort_values(sort_col, ascending=ascending).iloc[0]
        rid = str(lead_row["record_id"])
        record = by_record.loc[rid].to_dict()
        record["record_id"] = rid
        record["selection_role"] = role
        record["selected_lead"] = str(lead_row["lead"])
        record["lead_selection_metric"] = metric
        record["lead_selection_value"] = float(lead_row[sort_col])
        used_records.add(rid)
        rows.append(record)
        if len(rows) == 5:
            break
    return rows


def select_gallery_examples(df: pd.DataFrame, artifacts_dir: str | Path | None = None) -> pd.DataFrame:
    test = df[df["split"].astype(str).eq("test")].copy()
    original = test[test["sample_group"].eq("original unacceptable")].copy()
    synthetic = test[test["sample_group"].isin(["synthetic em", "synthetic ma"])].copy()
    if len(original) < 5 or len(synthetic) < 5:
        raise RuntimeError("Not enough held-out original/synthetic poor records to build the gallery.")

    rows: list[dict[str, object]] = []
    if artifacts_dir is not None:
        rows.extend(_select_visual_original_examples(original, Path(artifacts_dir)))

    selected_records = {str(row["record_id"]) for row in rows}
    if len(rows) < 5:
        for _, row in original.sort_values("record_id").iterrows():
            rid = str(row["record_id"])
            if rid not in selected_records:
                item = row.to_dict()
                item["selection_role"] = "fallback diverse original"
                selected_records.add(rid)
                rows.append(item)
            if len(rows) == 5:
                break

    selected_records = set()
    selected_sources: set[str] = set()
    synthetic_specs = [
        ("em low-frequency", "synthetic em", "mean_1_minus_basSQI", False),
        ("em QRS disagreement", "synthetic em", "mean_bSQI", True),
        ("em shape outlier", "synthetic em", "mean_kSQI", False),
        ("ma low-frequency", "synthetic ma", "mean_1_minus_basSQI", False),
        ("ma QRS disagreement", "synthetic ma", "mean_bSQI", True),
    ]
    for role, group, sort_col, ascending in synthetic_specs:
        row = _pick_first(
            synthetic[synthetic["sample_group"].eq(group)],
            selected_record_ids=selected_records,
            selected_source_ids=selected_sources,
            sort_col=sort_col,
            ascending=ascending,
            role=role,
        )
        if row is not None:
            selected_records.add(str(row["record_id"]))
            selected_sources.add(str(row["source_record_id"]))
            rows.append(row)

    if len(rows) < 10:
        for _, row in synthetic.sort_values(["sample_group", "record_id"]).iterrows():
            rid = str(row["record_id"])
            if rid not in selected_records:
                item = row.to_dict()
                item["selection_role"] = "fallback diverse synthetic"
                selected_records.add(rid)
                rows.append(item)
            if len(rows) == 10:
                break

    selected = pd.DataFrame(rows[:10]).copy()
    selected["gallery_column"] = np.where(selected["sample_group"].eq("original unacceptable"), "original", "synthetic")
    selected["gallery_index"] = selected.groupby("gallery_column").cumcount() + 1
    selected["display_id"] = np.where(
        selected["gallery_column"].eq("original"),
        "O" + selected["gallery_index"].astype(str),
        "S" + selected["gallery_index"].astype(str),
    )
    lead_info = selected.apply(_select_problem_lead, axis=1, result_type="expand")
    for col in ["selected_lead", "lead_selection_metric", "lead_selection_value"]:
        if col not in selected.columns:
            selected[col] = lead_info[col]
        else:
            selected[col] = selected[col].where(selected[col].notna(), lead_info[col])
    return selected


def _lead_values(row: pd.Series, sqi: str) -> pd.Series:
    return pd.Series({lead: float(row[f"{lead}__{sqi}"]) for lead in LEADS_12})


def _select_problem_lead(row: pd.Series) -> dict[str, object]:
    role = str(row.get("selection_role", "")).lower()
    if "flat" in role:
        metric = "fSQI"
        vals = _lead_values(row, metric)
        lead = str(vals.idxmax())
        value = float(vals.loc[lead])
    elif "baseline" in role or "low-frequency" in role or "saturated" in role:
        metric = "basSQI"
        vals = _lead_values(row, metric)
        lead = str(vals.idxmin())
        value = float(vals.loc[lead])
    elif "qrs" in role:
        metric = "bSQI"
        vals = _lead_values(row, metric)
        lead = str(vals.idxmin())
        value = float(vals.loc[lead])
    elif "lead disagreement" in role:
        metric = "iSQI"
        vals = _lead_values(row, metric)
        lead = str(vals.idxmin())
        value = float(vals.loc[lead])
    elif "psqi" in role:
        metric = "pSQI"
        vals = _lead_values(row, metric)
        nonflat = _lead_values(row, "fSQI") < 0.80
        if bool(nonflat.any()):
            vals = vals[nonflat]
        lead = str(vals.idxmin())
        value = float(vals.loc[lead])
    elif "shape" in role:
        metric = "kSQI"
        vals = _lead_values(row, metric)
        fvals = _lead_values(row, "fSQI")
        bas = _lead_values(row, "basSQI")
        physiologic = (fvals < 0.20) & (bas > 0.10)
        if bool(physiologic.any()):
            vals = vals[physiologic]
        lead = str(vals.idxmax())
        value = float(vals.loc[lead])
    else:
        metric = "pSQI"
        vals = _lead_values(row, metric)
        lead = str(vals.idxmin())
        value = float(vals.loc[lead])
    return {
        "selected_lead": lead,
        "lead_selection_metric": metric,
        "lead_selection_value": value,
    }


def _load_signal(artifacts_dir: Path, record_id: str) -> tuple[np.ndarray, list[str], float]:
    z = np.load(artifacts_dir / "resampled_125" / f"{record_id}.npz", allow_pickle=True)
    sig = z["sig_125"].astype(np.float64)
    leads = [str(x) for x in z["leads"]]
    fs = float(z["fs"])
    return sig, leads, fs


def _robust_ylim(x: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(finite, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    span = float(hi - lo)
    if span < 1e-6:
        center = float(np.nanmedian(finite))
        return center - 0.5, center + 0.5
    margin = max(0.08 * span, 0.02)
    return float(lo - margin), float(hi + margin)


def _format_sqi_value(value: object) -> str:
    v = float(value)
    if not np.isfinite(v):
        return "NA"
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _sqi_line(row: pd.Series) -> str:
    keys = [("b", "bSQI"), ("i", "iSQI"), ("p", "pSQI"), ("s", "sSQI"), ("k", "kSQI"), ("f", "fSQI"), ("bas", "basSQI")]
    return " ".join(f"{label}={_format_sqi_value(row[f'mean_{sqi}'])}" for label, sqi in keys)


def _plot_single_lead(
    ax: plt.Axes,
    sig: np.ndarray,
    leads: list[str],
    fs: float,
    *,
    lead: str,
    color: str,
    label: str,
    sqi_line: str,
    show_x: bool,
    show_ylabel: bool,
) -> None:
    n = min(sig.shape[0], int(round(10.0 * fs)))
    li = leads.index(lead) if lead in leads else 0
    x = sig[:n, li]
    t = np.arange(n, dtype=float) / fs
    ax.plot(t, x, color=color, lw=0.78, alpha=0.98)
    ax.set_xlim(0, 10)
    ax.set_ylim(*_robust_ylim(x))
    ax.tick_params(axis="both", labelsize=6.4, length=2.2, pad=1.5)
    if not show_x:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("mV", fontsize=7)
    else:
        ax.set_yticklabels([])
    ax.text(
        0.012,
        0.94,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=color,
        fontweight="bold",
        bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
    )
    ax.text(
        0.0,
        1.035,
        sqi_line,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=PALETTE["ink"],
        clip_on=False,
    )
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.55)
    ax.grid(False)


def plot_original_vs_synthetic_bad_gallery(
    selected: pd.DataFrame,
    *,
    artifacts_dir: str | Path,
    out_base: str | Path,
) -> list[Path]:
    apply_style()
    art = Path(artifacts_dir)
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(7.45, 6.35),
        sharex=True,
        gridspec_kw={"wspace": 0.16, "hspace": 0.66},
    )
    columns = [
        ("original", "Original unacceptable", PALETTE["red"]),
        ("synthetic", "Synthetic poor", PALETTE["blue"]),
    ]
    for col_idx, (col_name, header, color) in enumerate(columns):
        sub = selected[selected["gallery_column"].eq(col_name)].sort_values("gallery_index").head(5)
        axes[0, col_idx].set_title(header, loc="left", fontsize=8.5, fontweight="bold", color=PALETTE["ink"], pad=15)
        for row_idx, (_, row) in enumerate(sub.iterrows()):
            rid = str(row["record_id"])
            sig, leads, fs = _load_signal(art, rid)
            lead = str(row["selected_lead"])
            if col_name == "original":
                label = f"{row['display_id']}  {lead}"
            else:
                label = f"{row['display_id']}  {str(row['noise_type']).lower()}  {lead}"
            _plot_single_lead(
                axes[row_idx, col_idx],
                sig,
                leads,
                fs,
                lead=lead,
                color=color,
                label=label,
                sqi_line=_sqi_line(row),
                show_x=(row_idx == 4),
                show_ylabel=(col_idx == 0),
            )
    return save_figure(fig, out_base)


def run_case_gallery(
    *,
    artifacts_dir: str | Path,
    out_dir: str | Path,
    shared_images_dir: str | Path,
) -> dict[str, list[str]]:
    art = Path(artifacts_dir)
    out = Path(out_dir)
    img = Path(shared_images_dir)
    out.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)
    df = _add_record_summaries(load_split_frame(art, normalized=False))
    selected = select_gallery_examples(df, artifacts_dir=art)
    keep_cols = [
        "display_id",
        "gallery_column",
        "gallery_index",
        "record_id",
        "source_record_id",
        "split",
        "sample_group",
        "noise_type",
        "snr_db",
        "selection_role",
        "selected_lead",
        "lead_selection_metric",
        "lead_selection_value",
        "mean_bSQI",
        "mean_iSQI",
        "mean_pSQI",
        "mean_sSQI",
        "mean_kSQI",
        "mean_fSQI",
        "mean_basSQI",
        "mean_1_minus_basSQI",
    ]
    selected = selected[[c for c in keep_cols if c in selected.columns]]
    manifest_csv = out / "fig_16_original_vs_synthetic_bad_examples_manifest.csv"
    write_table(selected, manifest_csv)
    paths = plot_original_vs_synthetic_bad_gallery(
        selected,
        artifacts_dir=art,
        out_base=img / "fig_16_original_vs_synthetic_bad_examples",
    )
    return {"outputs": [str(manifest_csv), *[str(p) for p in paths]]}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Draw original-vs-synthetic poor ECG example gallery.")
    parser.add_argument("--artifacts-dir", default=str(root / "outputs" / "sqi_paper_aligned"))
    parser.add_argument("--out-dir", default=str(root / "outputs" / "sqi_supplemental" / "existing_seed0" / "case_gallery"))
    parser.add_argument("--shared-images-dir", default=str(root / "outputs" / "reports" / "sqi_paper_aligned" / "images"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_case_gallery(
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
        shared_images_dir=args.shared_images_dir,
    )
    for item in out["outputs"]:
        print(item)


if __name__ == "__main__":
    main()
