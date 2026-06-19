"""Plot representative PTB-before, PTB-after, and BUT waveforms by label.

Outputs 10 examples per label for:
- PTB base synthetic pool before target-aware replay/matching.
- PTB target-aware combo bank after boundary/bad matching.
- Clean BUT retained protocol.

This is a report-only visual inspection helper. It does not train models or
modify any pipeline/checkpoint files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "representative_waveform_examples"

PTB_BASE_DIR = (
    OUT_ROOT
    / "synthetic_variants"
    / "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
    / "datasets"
)
PTB_BASE_SIGNALS = PTB_BASE_DIR / "synth_10s_125hz_noisy.npz"
PTB_BASE_LABELS = PTB_BASE_DIR / "synth_10s_125hz_labels_with_level.csv"

PTB_AFTER_SIGNALS = ANALYSIS_DIR / "ptb_combo_boundary_bank" / "ptb_combo_cleanbad5s_gm_cvfold2_wavefact_v3_signals.npz"
PTB_AFTER_MANIFEST = ANALYSIS_DIR / "ptb_combo_boundary_bank" / "ptb_combo_cleanbad5s_gm_cvfold2_wavefact_v3_manifest.csv"

BUT_SIGNALS = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier" / "signals.npz"
BUT_ATLAS = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier" / "original_region_atlas.csv"

LABELS = ["good", "medium", "bad"]
LABEL_COLORS = {"good": "#5477C4", "medium": "#B8A037", "bad": "#CC6F47"}
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "text.color": INK,
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        }
    )


def load_npz_signal(path: Path) -> np.ndarray:
    z = np.load(path)
    key = "X" if "X" in z.files else "X_noisy" if "X_noisy" in z.files else z.files[0]
    x = np.asarray(z[key], dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    return x


def robust_wave(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.nanmedian(x))
    q25, q75 = np.nanpercentile(x, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(x) + 1e-6)
    z = (x - med) / scale
    return np.clip(z, -4.0, 4.0)


def rolling_mean_abs(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.abs(x)
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(np.abs(x), kernel, mode="same")


def primitive_features(x: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in x:
        y = robust_wave(row)
        dy = np.diff(y, prepend=y[0])
        baseline = np.convolve(y, np.ones(251, dtype=np.float32) / 251.0, mode="same")
        detail = rolling_mean_abs(y - baseline, 31)
        rows.append(
            {
                "rms": float(np.sqrt(np.mean(y**2))),
                "std": float(np.std(y)),
                "mean_abs": float(np.mean(np.abs(y))),
                "ptp": float(np.percentile(y, 99) - np.percentile(y, 1)),
                "diff_p95": float(np.percentile(np.abs(dy), 95)),
                "baseline_range": float(np.percentile(baseline, 95) - np.percentile(baseline, 5)),
                "flatline_ratio": float(np.mean(np.abs(dy) < 0.005)),
                "detail_mean": float(np.mean(detail)),
                "detail_p95": float(np.percentile(detail, 95)),
            }
        )
    return pd.DataFrame(rows)


def representative_indices(x: np.ndarray, meta: pd.DataFrame, label_col: str, label: str, source_col: str | None = None) -> list[int]:
    group_idx = np.flatnonzero(meta[label_col].astype(str).to_numpy() == label)
    if len(group_idx) <= 10:
        return group_idx.tolist()
    feats = primitive_features(x[group_idx])
    arr = feats.to_numpy(dtype=np.float32)
    center = np.nanmedian(arr, axis=0)
    scale = np.nanpercentile(arr, 75, axis=0) - np.nanpercentile(arr, 25, axis=0)
    scale = np.where(scale < 1e-6, np.nanstd(arr, axis=0) + 1e-6, scale)
    z = np.nan_to_num((arr - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.sqrt(np.mean(z**2, axis=1))
    order = np.argsort(dist)
    cutoff = max(10, int(len(order) * 0.80))
    ordered = order[:cutoff]
    picks_local: list[int] = []
    seen_sources: set[Any] = set()
    targets = np.linspace(0, len(ordered) - 1, 10).round().astype(int)
    for pos in targets:
        for candidate_pos in range(int(pos), len(ordered)):
            loc = int(ordered[candidate_pos])
            global_idx = int(group_idx[loc])
            source_val: Any = None
            if source_col and source_col in meta.columns:
                source_val = meta.iloc[global_idx][source_col]
            key = source_val if source_val is not None and not pd.isna(source_val) else global_idx
            if key in seen_sources:
                continue
            picks_local.append(global_idx)
            seen_sources.add(key)
            break
        if len(picks_local) >= 10:
            break
    if len(picks_local) < 10:
        for loc in ordered:
            global_idx = int(group_idx[int(loc)])
            if global_idx not in picks_local:
                picks_local.append(global_idx)
            if len(picks_local) >= 10:
                break
    return picks_local[:10]


def make_sources() -> dict[str, dict[str, Any]]:
    ptb_base_x = load_npz_signal(PTB_BASE_SIGNALS)
    ptb_base_meta = pd.read_csv(PTB_BASE_LABELS).sort_values("idx").reset_index(drop=True)
    # Keep train split so "before" matches the source pool used by target-aware banks.
    base_mask = ptb_base_meta["split"].astype(str).eq("train").to_numpy()
    ptb_base = {
        "name": "PTB base synthetic before target-aware replay",
        "short": "ptb_before",
        "x": ptb_base_x[base_mask],
        "meta": ptb_base_meta.loc[base_mask].reset_index(drop=True),
        "label_col": "y_class",
        "source_col": "idx",
        "note_cols": ["label_subtype", "snr_db", "noise_kind"],
    }

    ptb_after = {
        "name": "PTB target-aware synthetic after replay/matching",
        "short": "ptb_after",
        "x": load_npz_signal(PTB_AFTER_SIGNALS),
        "meta": pd.read_csv(PTB_AFTER_MANIFEST),
        "label_col": "y_class",
        "source_col": "source_idx",
        "note_cols": ["bank_role", "combo_source_bank", "mode", "target_original_region"],
    }

    but_meta = pd.read_csv(BUT_ATLAS)
    but_meta["y_class"] = but_meta["class_name"].astype(str)
    but = {
        "name": "BUT clean retained protocol",
        "short": "but_clean",
        "x": load_npz_signal(BUT_SIGNALS),
        "meta": but_meta,
        "label_col": "y_class",
        "source_col": "source_idx",
        "note_cols": ["original_region", "ambiguous_type", "record_id", "split"],
    }
    return {"ptb_before": ptb_base, "ptb_after": ptb_after, "but_clean": but}


def item_title(meta: pd.Series, label: str, note_cols: list[str]) -> str:
    parts = [label]
    for col in note_cols:
        if col not in meta.index:
            continue
        val = meta[col]
        if pd.isna(val):
            continue
        if isinstance(val, float):
            if abs(val) > 100:
                sval = f"{val:.0f}"
            else:
                sval = f"{val:.2f}"
        else:
            sval = str(val)
        if len(sval) > 26:
            sval = sval[:23] + "..."
        parts.append(f"{col}={sval}")
        if len(parts) >= 3:
            break
    return "\n".join(parts)


def plot_label_grid(source: dict[str, Any], label: str, picks: list[int], out: Path) -> None:
    x = source["x"]
    meta = source["meta"]
    t = np.linspace(0, 10, x.shape[1], dtype=np.float32)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.3), sharex=True, sharey=True)
    color = LABEL_COLORS[label]
    for ax, idx in zip(axes.ravel(), picks):
        y = robust_wave(x[idx])
        ax.plot(t, y, lw=0.9, color=color)
        ax.axhline(0, color="#D7DBE7", lw=0.6)
        ax.set_ylim(-4.1, 4.1)
        ax.set_title(item_title(meta.iloc[idx], label, source["note_cols"]), fontsize=7.5, loc="left")
        ax.set_yticks([])
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.65)
    for ax in axes[-1, :]:
        ax.set_xlabel("seconds", fontsize=8)
    fig.text(0.02, 0.985, f"{source['name']} - {label} examples", ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.952, "10 representative 10s windows; each waveform is robust-normalized for visual comparison.", ha="left", va="top", fontsize=9, color=MUTED)
    fig.subplots_adjust(top=0.86, left=0.04, right=0.99, bottom=0.10, wspace=0.14, hspace=0.38)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_contact_sheet(sources: dict[str, dict[str, Any]], selections: dict[str, dict[str, list[int]]], out: Path) -> None:
    fig, axes = plt.subplots(9, 10, figsize=(20, 13.5), sharex=True, sharey=True)
    t: np.ndarray | None = None
    row = 0
    for source_key in ["ptb_before", "ptb_after", "but_clean"]:
        source = sources[source_key]
        if t is None:
            t = np.linspace(0, 10, source["x"].shape[1], dtype=np.float32)
        for label in LABELS:
            picks = selections[source_key][label]
            for col, idx in enumerate(picks):
                ax = axes[row, col]
                y = robust_wave(source["x"][idx])
                ax.plot(t, y, lw=0.62, color=LABEL_COLORS[label])
                ax.set_ylim(-4.1, 4.1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.45)
                if col == 0:
                    ax.set_ylabel(f"{source['short']}\n{label}", fontsize=8, rotation=0, ha="right", va="center", labelpad=36)
            row += 1
    fig.text(0.02, 0.99, "Representative waveform contact sheet", ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.968, "Rows are source x label; columns are 10 representative windows. Robust-normalized per window.", ha="left", va="top", fontsize=10, color=MUTED)
    fig.subplots_adjust(top=0.94, left=0.08, right=0.995, bottom=0.03, wspace=0.05, hspace=0.10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def counts_for_source(source: dict[str, Any]) -> dict[str, int]:
    meta = source["meta"]
    col = source["label_col"]
    return {label: int(meta[col].astype(str).eq(label).sum()) for label in LABELS}


def main() -> None:
    set_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sources = make_sources()
    selections: dict[str, dict[str, list[int]]] = {}
    selection_rows: list[dict[str, Any]] = []
    figures: list[Path] = []
    for source_key, source in sources.items():
        selections[source_key] = {}
        for label in LABELS:
            picks = representative_indices(source["x"], source["meta"], source["label_col"], label, source.get("source_col"))
            selections[source_key][label] = picks
            for rank, idx in enumerate(picks, start=1):
                row = {"source": source_key, "label": label, "rank": rank, "row_idx": int(idx)}
                for col in source["note_cols"]:
                    if col in source["meta"].columns:
                        val = source["meta"].iloc[idx][col]
                        row[col] = None if pd.isna(val) else val
                selection_rows.append(row)
            out = REPORT_DIR / f"{source_key}_{label}_10_examples.png"
            plot_label_grid(source, label, picks, out)
            figures.append(out)
    contact = REPORT_DIR / "all_sources_labels_contact_sheet.png"
    plot_contact_sheet(sources, selections, contact)
    figures.insert(0, contact)
    pd.DataFrame(selection_rows).to_csv(REPORT_DIR / "representative_waveform_selection_manifest.csv", index=False)
    summary = {
        key: {
            "name": src["name"],
            "n": int(len(src["meta"])),
            "class_counts": counts_for_source(src),
        }
        for key, src in sources.items()
    }
    (REPORT_DIR / "representative_waveform_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Representative PTB/BUT Waveform Examples",
        "",
        "Each source has 10 representative 10s windows per label. Waveforms are robust-normalized per window for visual comparison.",
        "",
        "Sources:",
        "- `ptb_before`: base PTB synthetic train pool before target-aware replay/matching.",
        "- `ptb_after`: PTB target-aware combo bank after good/medium boundary replay and bad waveform matching.",
        "- `but_clean`: retained clean BUT protocol (`margin>=5s`, drop `outlier_low_confidence`).",
        "",
        "## Contact Sheet",
        "",
        f"![contact sheet]({contact})",
        "",
        "## Per-Label Panels",
        "",
    ]
    for fig in figures[1:]:
        lines.append(f"![{fig.stem}]({fig})")
        lines.append("")
    (REPORT_DIR / "representative_waveform_examples_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_DIR / "representative_waveform_examples_report.md"),
                "contact_sheet": str(contact),
                "figures": [str(f) for f in figures],
                "manifest": str(REPORT_DIR / "representative_waveform_selection_manifest.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
