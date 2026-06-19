from __future__ import annotations

from pathlib import Path
import json
import textwrap

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import wfdb
from scipy.signal import decimate, filtfilt, firwin, kaiserord, welch
from sklearn.metrics import roc_auc_score

from src.utils.paths import project_root


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SQI_TYPES = ["iSQI", "bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI"]
BOUNDED_SQIS = {"iSQI", "bSQI", "pSQI", "fSQI", "basSQI"}
FS_125 = 125
FS_500 = 500

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

COLORS = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}
SCI = {
    "purple": "#1F4E79",
    "purple_light": "#9FBDD3",
    "gold": "#8B1E3F",
    "gold_light": "#D99AAA",
    "teal": "#247C7A",
    "red": "#A23B52",
}


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial"],
            "font.monospace": ["Consolas", "DejaVu Sans Mono", "monospace"],
        },
    )


def add_header(fig: plt.Figure, title: str, subtitle: str, *, left: float = 0.075, top: float = 0.985) -> None:
    title = textwrap.fill(title.strip(), width=92, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=128, break_long_words=False)
    fig.text(left, top, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, top - 0.04, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])


def finish(fig: plt.Figure, out_base: Path, manifest: list[dict], description: str, report_candidate: str) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)
    manifest.append(
        {
            "file_png": str(png),
            "file_pdf": str(pdf),
            "name": out_base.name,
            "description": description,
            "report_candidate": report_candidate,
        }
    )


def label_name(y: int | float | str) -> str:
    return "acceptable" if int(y) == 1 else "poor"


def rid_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def read_inputs(root: Path) -> dict[str, pd.DataFrame]:
    base = root / "outputs" / "sqi_paper_aligned"
    return {
        "record84": pd.read_parquet(base / "features" / "record84.parquet"),
        "record84_norm": pd.read_parquet(base / "features" / "record84_norm.parquet"),
        "lead7": pd.read_parquet(base / "features" / "lead7.parquet"),
        "split": pd.read_csv(base / "splits" / "split_seta_seed0_paper_balanced.csv"),
        "audit": pd.read_csv(base / "splits" / "split_seta_seed0_paper_balanced.audit.csv"),
        "qrs_summary": pd.read_csv(base / "qrs" / "qrs_summary_seed0.csv"),
    }


def feature_columns() -> list[str]:
    return [f"{lead}__{sqi}" for lead in LEADS_12 for sqi in SQI_TYPES]


def lead_sqi_col(lead: str, sqi: str) -> str:
    return f"{lead}__{sqi}"


def plot_split_balance(split: pd.DataFrame, audit: pd.DataFrame, out_dir: Path, manifest: list[dict]) -> None:
    df = split.copy()
    df["label"] = df["y"].map(lambda v: "acceptable" if int(v) == 1 else "poor")
    df["mechanism"] = np.where(
        df["y"].astype(int).eq(1),
        "original acceptable",
        np.where(df["is_augmented"].astype(int).eq(1), "synthetic poor", "original poor"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.24)
    add_header(
        fig,
        "Split, balance, and augmentation checks",
        "These panels verify the Set-a-only balanced protocol, synthetic noisy-poor construction, and source-record group split.",
    )

    ax = axes[0, 0]
    label_order = ["acceptable", "poor"]
    split_order = ["train", "val", "test"]
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(split_order)
    bottom = np.zeros(len(counts))
    color_map = {"acceptable": COLORS["blue"]["base"], "poor": COLORS["orange"]["base"]}
    for lab in label_order:
        values = counts[lab].to_numpy()
        bars = ax.bar(counts.index, values, bottom=bottom, color=color_map[lab], edgecolor=TOKENS["ink"], linewidth=0.8, label=lab)
        for rect, val, base in zip(bars, values, bottom):
            ax.text(rect.get_x() + rect.get_width() / 2, base + val / 2, f"{int(val)}", ha="center", va="center", fontsize=8, color=TOKENS["ink"])
        bottom += values
    ax.set_ylabel("Records")
    ax.set_xlabel("Split")
    ax.set_title("Class counts by split", loc="left", fontsize=10, fontweight="semibold")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    mech_order = ["original acceptable", "original poor", "synthetic poor"]
    mech_counts = df["mechanism"].value_counts().reindex(mech_order).fillna(0)
    colors = [COLORS["blue"]["base"], COLORS["orange"]["light"], COLORS["orange"]["base"]]
    bars = ax.barh(mech_counts.index[::-1], mech_counts.values[::-1], color=colors[::-1], edgecolor=TOKENS["ink"], linewidth=0.8)
    for rect, val in zip(bars, mech_counts.values[::-1]):
        ax.text(val + 8, rect.get_y() + rect.get_height() / 2, f"{int(val)}", va="center", fontsize=9)
    ax.set_xlabel("Records")
    ax.set_title("Sample mechanism after balancing", loc="left", fontsize=10, fontweight="semibold")
    ax.set_xlim(0, max(mech_counts.max() * 1.22, 1))

    ax = axes[1, 0]
    noise_counts = audit["noise_type"].value_counts().reindex(["em", "ma"]).fillna(0)
    bars = ax.bar(noise_counts.index, noise_counts.values, color=[COLORS["pink"]["base"], COLORS["olive"]["base"]], edgecolor=TOKENS["ink"], linewidth=0.8)
    for rect, val in zip(bars, noise_counts.values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 4, f"{int(val)}", ha="center", fontsize=9)
    ax.set_ylabel("Synthetic records")
    ax.set_xlabel("NSTDB noise type")
    ax.set_title("Noisy-poor construction", loc="left", fontsize=10, fontweight="semibold")
    ax.set_ylim(0, max(noise_counts.max() * 1.22, 1))

    ax = axes[1, 1]
    source_split = df.assign(source_record_id=df["source_record_id"].map(rid_str)).groupby("source_record_id")["split"].nunique()
    leakage = int((source_split > 1).sum())
    augmented_sources = int((df.groupby("source_record_id")["is_augmented"].sum() > 0).sum())
    text = (
        f"Rows: {len(df):,}\n"
        f"Acceptable / poor: {(df['y'].astype(int) == 1).sum():,} / {(df['y'].astype(int) == -1).sum():,}\n"
        f"Synthetic poor: {int(df['is_augmented'].sum()):,}\n"
        f"Augmented source records: {augmented_sources:,}\n"
        f"Source IDs crossing splits: {leakage:,}\n"
        f"Audit rows: {len(audit):,}"
    )
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.04, 0.08), 0.92, 0.78, transform=ax.transAxes, facecolor=NEUTRAL["xlight"], edgecolor=NEUTRAL["base"], linewidth=1))
    ax.text(0.10, 0.78, "Group-split integrity", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="semibold", color=TOKENS["ink"])
    ax.text(0.10, 0.66, text, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=TOKENS["ink"], linespacing=1.55, family="monospace")

    for ax in axes.flat:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_01_split_balance", manifest, "Set-a-only class balance, synthetic noisy-poor counts, and source-record leakage check.", "Strong report candidate for methods/QC appendix.")


def plot_qrs_counts(qrs: pd.DataFrame, lead7: pd.DataFrame, out_dir: Path, manifest: list[dict]) -> None:
    q = qrs.copy()
    q["count_diff"] = q["n_detector1"] - q["n_detector2"]
    q["abs_count_diff"] = q["count_diff"].abs()
    long = q.melt(
        id_vars=["record_id", "lead"],
        value_vars=["n_detector1", "n_detector2"],
        var_name="detector",
        value_name="n_beats",
    )
    long["detector"] = long["detector"].map({"n_detector1": "wqrs", "n_detector2": "eplimited"})
    long["lead"] = pd.Categorical(long["lead"], LEADS_12, ordered=True)

    lead7 = lead7.copy()
    lead7["label"] = lead7["y"].map(label_name)
    lead7["lead"] = pd.Categorical(lead7["lead"], LEADS_12, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(top=0.86, hspace=0.40, wspace=0.26)
    add_header(
        fig,
        "QRS detector cache checks",
        "Beat counts and bSQI distributions verify that wqrs and EP Limited were both called and returned plausible 10 s annotations.",
    )

    ax = axes[0, 0]
    sns.boxplot(
        data=long,
        x="lead",
        y="n_beats",
        hue="detector",
        palette={"wqrs": COLORS["blue"]["base"], "eplimited": COLORS["gold"]["base"]},
        fliersize=0,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_ylabel("Detected beats per 10 s")
    ax.set_xlabel("Lead")
    ax.set_title("Beat count by lead and detector", loc="left", fontsize=10, fontweight="semibold")
    ax.legend(frameon=False, loc="upper right", title=None)

    ax = axes[0, 1]
    sample = q.sample(n=min(3500, len(q)), random_state=0)
    ax.scatter(sample["n_detector1"], sample["n_detector2"], s=12, alpha=0.28, color=COLORS["blue"]["mid"], edgecolors="none", rasterized=True)
    max_n = int(max(q["n_detector1"].max(), q["n_detector2"].max(), 1))
    ax.plot([0, max_n], [0, max_n], color=NEUTRAL["dark"], linewidth=1, linestyle="--")
    ax.set_xlabel("wqrs beats")
    ax.set_ylabel("eplimited beats")
    ax.set_title("Detector count agreement", loc="left", fontsize=10, fontweight="semibold")

    ax = axes[1, 0]
    bins = np.arange(q["count_diff"].min() - 0.5, q["count_diff"].max() + 1.5, 1)
    ax.hist(q["count_diff"], bins=bins, color=COLORS["orange"]["base"], edgecolor=COLORS["orange"]["dark"], linewidth=0.8)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1, linestyle="--")
    ax.set_xlabel("wqrs beats minus eplimited beats")
    ax.set_ylabel("Lead-records")
    ax.set_title("Count difference distribution", loc="left", fontsize=10, fontweight="semibold")

    ax = axes[1, 1]
    bmed = lead7.groupby(["lead", "label"], observed=False)["bSQI"].median().reset_index()
    sns.pointplot(
        data=bmed,
        x="lead",
        y="bSQI",
        hue="label",
        palette={"acceptable": COLORS["blue"]["mid"], "poor": COLORS["orange"]["mid"]},
        markers=["o", "s"],
        linestyles=["-", "--"],
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Median bSQI")
    ax.set_xlabel("Lead")
    ax.set_title("bSQI by label and lead", loc="left", fontsize=10, fontweight="semibold")
    ax.legend(frameon=False, loc="lower left", title=None)

    for ax in axes.flat:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_02_qrs_detector_counts", manifest, "wqrs/eplimited count plausibility, detector agreement, and bSQI lead-level behavior.", "Good appendix candidate when discussing QRS alignment risk.")


def plot_feature_integrity(record84: pd.DataFrame, out_dir: Path, manifest: list[dict]) -> None:
    feat_cols = feature_columns()
    x = record84[feat_cols].to_numpy(float)
    finite = np.isfinite(x)

    missing = []
    range_bad = []
    for lead in LEADS_12:
        for sqi in SQI_TYPES:
            col = lead_sqi_col(lead, sqi)
            vals = record84[col].to_numpy(float)
            missing.append({"lead": lead, "sqi": sqi, "nonfinite": int((~np.isfinite(vals)).sum())})
            if sqi in BOUNDED_SQIS:
                bad = np.isfinite(vals) & ((vals < -1e-9) | (vals > 1 + 1e-9))
                range_bad.append({"lead": lead, "sqi": sqi, "out_of_range": int(bad.sum())})
            else:
                range_bad.append({"lead": lead, "sqi": sqi, "out_of_range": np.nan})

    missing_df = pd.DataFrame(missing).pivot(index="sqi", columns="lead", values="nonfinite").reindex(SQI_TYPES)[LEADS_12]
    range_df = pd.DataFrame(range_bad).pivot(index="sqi", columns="lead", values="out_of_range").reindex(SQI_TYPES)[LEADS_12]
    p99_rows = []
    for sqi in SQI_TYPES:
        vals = record84[[lead_sqi_col(lead, sqi) for lead in LEADS_12]].to_numpy(float).ravel()
        vals = vals[np.isfinite(vals)]
        p99_rows.append({"sqi": sqi, "p99_abs": float(np.nanpercentile(np.abs(vals), 99)) if len(vals) else np.nan})
    p99 = pd.DataFrame(p99_rows).sort_values("p99_abs")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8))
    fig.subplots_adjust(top=0.86, hspace=0.44, wspace=0.27)
    add_header(
        fig,
        "Feature table integrity checks",
        "The 84 SQI matrix is checked for non-finite values, bounded-SQI range violations, feature scale, and duplicate record IDs.",
    )

    ax = axes[0, 0]
    sns.heatmap(missing_df, annot=True, fmt=".0f", cmap="Blues", cbar=False, linewidths=0.4, linecolor=TOKENS["grid"], ax=ax)
    ax.set_xlabel("Lead")
    ax.set_ylabel("SQI")
    ax.set_title("Non-finite counts", loc="left", fontsize=10, fontweight="semibold")

    ax = axes[0, 1]
    sns.heatmap(range_df, annot=True, fmt=".0f", cmap="Oranges", cbar=False, linewidths=0.4, linecolor=TOKENS["grid"], ax=ax)
    ax.set_xlabel("Lead")
    ax.set_ylabel("SQI")
    ax.set_title("Out-of-range counts for bounded SQIs", loc="left", fontsize=10, fontweight="semibold")

    ax = axes[1, 0]
    bars = ax.barh(p99["sqi"], p99["p99_abs"], color=COLORS["olive"]["base"], edgecolor=COLORS["olive"]["dark"], linewidth=0.8)
    ax.set_xlabel("99th percentile of absolute value")
    ax.set_ylabel("SQI")
    ax.set_title("Scale sanity by SQI type", loc="left", fontsize=10, fontweight="semibold")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    for rect, val in zip(bars, p99["p99_abs"]):
        ax.text(val + max(p99["p99_abs"].max() * 0.015, 0.01), rect.get_y() + rect.get_height() / 2, f"{val:.3g}", va="center", fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    duplicate_ids = int(record84["record_id"].astype(str).duplicated().sum())
    finite_pct = float(finite.mean() * 100.0)
    text = (
        f"Rows: {len(record84):,}\n"
        f"Feature columns: {len(feat_cols):,}\n"
        f"Finite cells: {finite_pct:.3f}%\n"
        f"Total non-finite cells: {int((~finite).sum()):,}\n"
        f"Duplicate record IDs: {duplicate_ids:,}\n"
        f"Labels: {record84['y'].value_counts().to_dict()}"
    )
    ax.add_patch(plt.Rectangle((0.04, 0.08), 0.92, 0.78, transform=ax.transAxes, facecolor=NEUTRAL["xlight"], edgecolor=NEUTRAL["base"], linewidth=1))
    ax.text(0.10, 0.78, "84-feature table summary", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="semibold")
    ax.text(0.10, 0.66, text, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, family="monospace", linespacing=1.55)

    for ax in axes.flat:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_03_feature_integrity", manifest, "84-feature nonfinite/range/scale checks.", "Good methods appendix candidate.")


def _clipped_values(values: np.ndarray) -> tuple[np.ndarray, str]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return values, ""
    lo, hi = np.nanpercentile(values, [1, 99])
    if np.isclose(lo, hi):
        return values, ""
    clipped = np.clip(values, lo, hi)
    return clipped, f"1-99% clipped [{lo:.3g}, {hi:.3g}]"


def plot_pooled_sqi_distributions(lead7: pd.DataFrame, out_dir: Path, manifest: list[dict]) -> None:
    df = lead7.copy()
    df["label"] = df["y"].map(label_name)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.4))
    fig.subplots_adjust(top=0.84, hspace=0.46, wspace=0.30)
    add_header(
        fig,
        "Pooled SQI distributions across all 12 leads",
        "Each panel pools record-lead rows and compares acceptable against poor after light 1-99% clipping for readability.",
    )

    for ax, sqi in zip(axes.flat, SQI_TYPES):
        notes = []
        for lab, color, edge in [
            ("acceptable", COLORS["blue"]["base"], COLORS["blue"]["dark"]),
            ("poor", COLORS["orange"]["base"], COLORS["orange"]["dark"]),
        ]:
            vals = df.loc[df["label"] == lab, sqi].to_numpy(float)
            vals, note = _clipped_values(vals)
            if note:
                notes.append(note)
            ax.hist(vals, bins=50, density=True, histtype="stepfilled", alpha=0.32, color=color, edgecolor=edge, linewidth=1.0, label=lab)
            med = float(np.nanmedian(vals)) if len(vals) else np.nan
            ax.axvline(med, color=edge, linewidth=1, linestyle="-")
        ax.set_title(sqi, loc="left", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.ticklabel_format(axis="x", style="plain")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.945, 0.92), frameon=False, ncol=2)
    for ax in axes.flat[:-1]:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_04_pooled_sqi_distributions", manifest, "Pooled feature distributions by label for all seven SQIs.", "Useful appendix candidate; can support explanations of fSQI/basSQI gaps.")


def plot_lead_median_profiles(lead7: pd.DataFrame, out_dir: Path, manifest: list[dict]) -> None:
    df = lead7.copy()
    df["label"] = df["y"].map(label_name)
    df["lead"] = pd.Categorical(df["lead"], LEADS_12, ordered=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))
    fig.subplots_adjust(top=0.84, hspace=0.46, wspace=0.28)
    add_header(
        fig,
        "Lead-wise median SQI profiles",
        "Median acceptable and poor values by lead reveal lead-order issues, detector artifacts, and SQIs that separate labels consistently.",
    )

    for ax, sqi in zip(axes.flat, SQI_TYPES):
        med = df.groupby(["lead", "label"], observed=False)[sqi].median().reset_index()
        sns.pointplot(
            data=med,
            x="lead",
            y=sqi,
            hue="label",
            palette={"acceptable": COLORS["blue"]["mid"], "poor": COLORS["orange"]["mid"]},
            errorbar=None,
            markers=["o", "s"],
            linestyles=["-", "--"],
            ax=ax,
        )
        ax.set_title(sqi, loc="left", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Lead")
        ax.set_ylabel("Median value")
        ax.tick_params(axis="x", rotation=0)
        leg = ax.get_legend()
        if leg:
            leg.remove()
        sns.despine(ax=ax)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.945, 0.92), frameon=False, ncol=2)
    finish(fig, out_dir / "diag_05_lead_median_profiles", manifest, "Median SQI values by lead and label.", "Useful for appendix or internal validation.")


def plot_normalization_check(record84: pd.DataFrame, record84_norm: pd.DataFrame, split: pd.DataFrame, out_dir: Path, manifest: list[dict]) -> None:
    split_small = split[["record_id", "split"]].copy()
    split_small["record_id"] = split_small["record_id"].astype(str)
    raw = record84.copy()
    norm = record84_norm.copy()
    raw["record_id"] = raw["record_id"].astype(str)
    norm["record_id"] = norm["record_id"].astype(str)
    raw = raw.merge(split_small, on="record_id", how="left")
    norm = norm.merge(split_small, on="record_id", how="left")

    checks = [("raw", raw, "II__sSQI"), ("normalized", norm, "II__sSQI"), ("raw", raw, "II__kSQI"), ("normalized", norm, "II__kSQI")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    fig.subplots_adjust(top=0.84, hspace=0.42, wspace=0.24)
    add_header(
        fig,
        "Normalization train/test checks",
        "Representative unbounded SQIs on lead II are compared before and after train-derived normalization.",
    )

    for ax, (stage, df, col) in zip(axes.flat, checks):
        for split_name, color, edge in [("train", COLORS["blue"]["base"], COLORS["blue"]["dark"]), ("test", COLORS["orange"]["base"], COLORS["orange"]["dark"])]:
            vals = df.loc[df["split"] == split_name, col].to_numpy(float)
            vals, _ = _clipped_values(vals)
            ax.hist(vals, bins=44, density=True, histtype="stepfilled", alpha=0.32, color=color, edgecolor=edge, linewidth=1.0, label=split_name)
            ax.axvline(float(np.nanmedian(vals)), color=edge, linewidth=1)
        ax.set_title(f"{stage}: {col}", loc="left", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        sns.despine(ax=ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.94, 0.92), frameon=False, ncol=2)
    finish(fig, out_dir / "diag_06_normalization_train_test", manifest, "Train/test distribution check before and after normalization for lead-II sSQI and kSQI.", "Internal QC; mention only if normalization needs defense.")


def load_case_500(cases_dir: Path, record_id: str) -> np.ndarray:
    z = np.load(cases_dir / f"{record_id}.npz", allow_pickle=True)
    return z["sig_500"].astype(float)


def compute_snr_per_lead(clean12: np.ndarray, noisy12: np.ndarray) -> np.ndarray:
    noise = noisy12 - clean12
    px = np.mean(clean12**2, axis=0) + 1e-12
    pn = np.mean(noise**2, axis=0) + 1e-12
    return 10.0 * np.log10(px / pn)


def plot_noise_allocation(audit: pd.DataFrame, out_dir: Path, manifest: list[dict], root: Path) -> None:
    cases_dir = root / "outputs" / "sqi_paper_aligned" / "cases_500"
    a = audit.copy()
    a["record_id"] = a["record_id"].map(rid_str)
    a["source_record_id"] = a["source_record_id"].map(rid_str)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.7))
    fig.subplots_adjust(top=0.85, hspace=0.42, wspace=0.25)
    add_header(
        fig,
        "Noise allocation and SNR checks",
        "Audit rows confirm balanced em/ma assignment, globally unique NSTDB offsets, and per-lead SNR near the -6 dB target.",
    )

    ax = axes[0, 0]
    counts = a["noise_type"].value_counts().reindex(["em", "ma"]).fillna(0)
    bars = ax.bar(counts.index, counts.values, color=[COLORS["pink"]["base"], COLORS["olive"]["base"]], edgecolor=TOKENS["ink"], linewidth=0.8)
    for rect, val in zip(bars, counts.values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 4, f"{int(val)}", ha="center", fontsize=9)
    ax.set_xlabel("Noise type")
    ax.set_ylabel("Synthetic records")
    ax.set_title("Balanced noise types", loc="left", fontsize=10, fontweight="semibold")

    ax = axes[0, 1]
    for nt, color in [("em", COLORS["pink"]["mid"]), ("ma", COLORS["olive"]["mid"])]:
        vals = a.loc[a["noise_type"] == nt, "noise_start_360"].to_numpy(float)
        ax.hist(vals, bins=32, histtype="step", linewidth=1.5, color=color, label=nt)
    ax.set_xlabel("NSTDB start sample at 360 Hz")
    ax.set_ylabel("Audit rows")
    ax.set_title("Noise segment allocation", loc="left", fontsize=10, fontweight="semibold")
    ax.legend(frameon=False)

    examples = []
    for nt in ["em", "ma"]:
        row = a.loc[a["noise_type"] == nt].iloc[0]
        clean = load_case_500(cases_dir, row["source_record_id"])
        noisy = load_case_500(cases_dir, row["record_id"])
        examples.append((nt, row["record_id"], row["source_record_id"], compute_snr_per_lead(clean, noisy)))

    for ax, (nt, rid, src, snr) in zip(axes[1], examples):
        color = COLORS["pink"]["base"] if nt == "em" else COLORS["olive"]["base"]
        edge = COLORS["pink"]["dark"] if nt == "em" else COLORS["olive"]["dark"]
        bars = ax.bar(LEADS_12, snr, color=color, edgecolor=edge, linewidth=0.8)
        ax.axhline(-6.0, color=TOKENS["ink"], linewidth=1, linestyle="--", label="target -6 dB")
        ax.set_ylabel("SNR (dB)")
        ax.set_xlabel("Lead")
        ax.set_title(f"{nt} example: {rid}", loc="left", fontsize=10, fontweight="semibold")
        ax.text(0.02, 0.08, f"source={src}\nmean={np.mean(snr):.2f} dB", transform=ax.transAxes, fontsize=8.5, color=TOKENS["ink"], bbox=dict(facecolor=TOKENS["panel"], edgecolor=NEUTRAL["base"], pad=4))
        ax.legend(frameon=False, loc="lower right")
    for ax in axes.flat:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_07_noise_allocation_snr", manifest, "NSTDB segment allocation, em/ma balance, and example per-lead SNR checks.", "Good methods/QC appendix candidate.")


def ensure_lead_order(sig: np.ndarray, sig_name: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(sig_name)}
    missing = [lead for lead in LEADS_12 if lead not in idx]
    if missing:
        raise ValueError(f"Missing leads: {missing}; available={sig_name}")
    return sig[:, [idx[lead] for lead in LEADS_12]]


def pick_clean_example(split: pd.DataFrame, root: Path) -> str:
    qrs_dir = root / "outputs" / "sqi_paper_aligned" / "qrs"
    resampled_dir = root / "outputs" / "sqi_paper_aligned" / "resampled_125"
    df = split.loc[(split["y"].astype(int) == 1) & (split["is_augmented"].astype(int) == 0)].copy()
    for rid in df["record_id"].astype(str).sort_values():
        if (qrs_dir / f"{rid}.npz").exists() and (resampled_dir / f"{rid}.npz").exists():
            return rid
    raise FileNotFoundError("No clean acceptable record with qrs and resampled cache found.")


def plot_resample_qc(split: pd.DataFrame, out_dir: Path, manifest: list[dict], root: Path, rid: str) -> None:
    data_dir = root / "data" / "physionet" / "challenge-2011" / "set-a"
    resampled_dir = root / "outputs" / "sqi_paper_aligned" / "resampled_125"
    rec = wfdb.rdrecord(str(data_dir / rid), physical=True)
    sig500 = ensure_lead_order(rec.p_signal, list(rec.sig_name))
    z = np.load(resampled_dir / f"{rid}.npz", allow_pickle=True)
    sig125 = z["sig_125"].astype(float)
    lead_idx = LEADS_12.index("II")
    x500 = sig500[:, lead_idx]
    x125 = sig125[:, lead_idx]

    mid = len(x500) // 2
    half500 = int(1.0 * FS_500)
    s0, s1 = max(0, mid - half500), min(len(x500), mid + half500)
    t500 = np.arange(s0, s1) / FS_500
    s0_125, s1_125 = int(round(s0 / 4)), int(round(s1 / 4))
    t125 = np.arange(s0_125, s1_125) / FS_125

    f500, p500 = welch(x500, fs=FS_500, nperseg=1024, noverlap=512, detrend="constant")
    f125, p125 = welch(x125, fs=FS_125, nperseg=256, noverlap=128, detrend="constant")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    fig.subplots_adjust(top=0.90, wspace=0.28)

    ax = axes[0]
    ax.plot(t500, x500[s0:s1], color=SCI["purple"], linewidth=1.05, label="500 Hz")
    ax.plot(t125, x125[s0_125:s1_125], color=SCI["gold"], linewidth=1.20, label="125 Hz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title("Waveform", loc="left", fontsize=10, fontweight="semibold")
    ax.text(0.02, 0.96, f"record {rid}, lead II", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=TOKENS["muted"])
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    m500 = (f500 >= 0) & (f500 <= 60)
    m125 = (f125 >= 0) & (f125 <= 60)
    ax.semilogy(f500[m500], p500[m500], color=SCI["purple"], linewidth=1.05, label="500 Hz")
    ax.semilogy(f125[m125], p125[m125], color=SCI["gold"], linewidth=1.20, label="125 Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Welch PSD", loc="left", fontsize=10, fontweight="semibold")
    ax.legend(frameon=False)

    for ax in axes:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_08_resampling_wave_psd", manifest, "Raw 500 Hz versus resampled 125 Hz waveform and PSD comparison.", "Internal QC; useful if preprocessing details are challenged.")


def qrs_peak_lists(z: np.lib.npyio.NpzFile, key: str) -> list[np.ndarray]:
    arr = z[key].tolist()
    return [np.asarray(x, dtype=int) for x in arr]


def plot_qrs_overlay(out_dir: Path, manifest: list[dict], root: Path, rid: str) -> None:
    resampled_dir = root / "outputs" / "sqi_paper_aligned" / "resampled_125"
    qrs_dir = root / "outputs" / "sqi_paper_aligned" / "qrs"
    sig = np.load(resampled_dir / f"{rid}.npz", allow_pickle=True)["sig_125"].astype(float)
    z = np.load(qrs_dir / f"{rid}.npz", allow_pickle=True)
    r1 = qrs_peak_lists(z, "rpeaks_1")
    r2 = qrs_peak_lists(z, "rpeaks_2")
    n = sig.shape[0]
    t = np.arange(n) / FS_125

    fig, axes = plt.subplots(12, 1, figsize=(12.2, 12.4), sharex=True)
    fig.subplots_adjust(top=0.965, hspace=0.09)

    for i, ax in enumerate(axes):
        ax.plot(t, sig[:, i], color=NEUTRAL["dark"], linewidth=0.75)
        p1 = r1[i][(r1[i] >= 0) & (r1[i] < n)]
        p2 = r2[i][(r2[i] >= 0) & (r2[i] < n)]
        if len(p1):
            ax.scatter(p1 / FS_125, sig[p1, i], s=20, facecolors=SCI["gold_light"], edgecolors=SCI["gold"], linewidths=0.8, label="wqrs" if i == 0 else None, zorder=4, rasterized=True)
        if len(p2):
            ax.scatter(p2 / FS_125, sig[p2, i], s=26, marker="x", color=SCI["purple"], linewidths=1.0, label="eplimited" if i == 0 else None, zorder=5, rasterized=True)
        ax.set_ylabel(LEADS_12[i], rotation=0, labelpad=18, va="center", fontsize=8.5)
        ax.grid(True, axis="x", color=TOKENS["grid"], linewidth=0.6)
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", labelleft=False, length=0)
        sns.despine(ax=ax, left=True, bottom=i != len(axes) - 1)
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", frameon=False, ncol=2, fontsize=8.5)
    finish(fig, out_dir / "diag_09_qrs_overlay_12lead", manifest, "12-lead ECG with cached wqrs and EP Limited detections.", "Good appendix candidate for detector validation.")


def plot_raw_seta_qc(out_dir: Path, manifest: list[dict], root: Path, rid: str) -> None:
    data_dir = root / "data" / "physionet" / "challenge-2011" / "set-a"
    rec = wfdb.rdrecord(str(data_dir / rid), physical=True)
    sig12 = ensure_lead_order(rec.p_signal, list(rec.sig_name))
    fs = float(rec.fs)
    n_plot = min(int(round(10.0 * fs)), sig12.shape[0])
    t = np.arange(n_plot) / fs
    x = sig12[:n_plot]

    amp_range = x.max(axis=0) - x.min(axis=0)

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], top=0.90, bottom=0.12, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    scale = np.nanpercentile(np.abs(x), 98)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    offset = 3.2 * scale
    ytick_pos = []
    for i, lead in enumerate(LEADS_12):
        pos = (len(LEADS_12) - 1 - i) * offset
        ytick_pos.append(pos)
        ax.plot(t, x[:, i] + pos, color=SCI["purple"], linewidth=0.72)
    ax.set_xlabel("Time (s), 500 Hz")
    ax.set_ylabel("Lead order")
    ax.set_yticks(ytick_pos, LEADS_12)
    ax.set_xlim(0, 10)
    ax.set_title("Raw 12-lead ECG (10 s)", loc="left", fontsize=10, fontweight="semibold")
    ax.text(0.01, 0.985, f"record {rid}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=TOKENS["muted"])
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    ypos = np.arange(len(LEADS_12))
    ax.barh(ypos, amp_range, color=SCI["gold_light"], edgecolor=SCI["gold"], linewidth=0.9)
    ax.set_yticks(ypos, LEADS_12)
    ax.invert_yaxis()
    ax.set_xlabel("Amplitude range (mV)")
    ax.set_ylabel("Lead")
    ax.set_title("Per-lead range", loc="left", fontsize=10, fontweight="semibold")

    for ax in fig.axes:
        sns.despine(ax=ax, left=False)
    finish(fig, out_dir / "diag_10_raw_seta_qc", manifest, "Raw Set-a 500 Hz waveform, per-lead amplitude spread, and PSD sanity check.", "Good preprocessing appendix candidate.")


def _design_reference_lpf(fs: float) -> np.ndarray:
    pass_hz = 40.0
    stop_hz = 62.5
    width = (stop_hz - pass_hz) / (fs / 2.0)
    numtaps, beta = kaiserord(90.0, width)
    if numtaps % 2 == 0:
        numtaps += 1
    numtaps = int(min(max(numtaps, 401), 5001))
    return firwin(numtaps=numtaps, cutoff=pass_hz, window=("kaiser", beta), fs=fs, pass_zero="lowpass").astype(float)


def _rel_rmse(ref: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - ref) ** 2)) / (np.sqrt(np.mean(ref**2)) + 1e-12))


def plot_resample_alias_check(out_dir: Path, manifest: list[dict], root: Path, rid: str) -> None:
    data_dir = root / "data" / "physionet" / "challenge-2011" / "set-a"
    resampled_dir = root / "outputs" / "sqi_paper_aligned" / "resampled_125"
    rec = wfdb.rdrecord(str(data_dir / rid), physical=True)
    sig12 = ensure_lead_order(rec.p_signal, list(rec.sig_name))
    x500 = sig12[:, LEADS_12.index("II")].astype(float)
    x_cache = np.load(resampled_dir / f"{rid}.npz", allow_pickle=True)["sig_125"][:, LEADS_12.index("II")].astype(float)

    taps = _design_reference_lpf(FS_500)
    x_filt500 = filtfilt(taps, [1.0], x500)
    x_ref = x_filt500[::4]
    x_dec = decimate(x500, q=4, ftype="fir", zero_phase=True)
    n = min(len(x_ref), len(x_cache), len(x_dec))
    x_ref = x_ref[:n]
    x_cache = x_cache[:n]
    x_dec = x_dec[:n]

    f_ref, p_ref = welch(x_ref, fs=FS_125, nperseg=min(512, n), noverlap=min(256, max(0, n // 2)), detrend="constant")
    f_cache, p_cache = welch(x_cache, fs=FS_125, nperseg=min(512, n), noverlap=min(256, max(0, n // 2)), detrend="constant")
    f_raw500, p_raw500 = welch(x500, fs=FS_500, nperseg=min(2048, len(x500)), noverlap=min(1024, max(0, len(x500) // 2)), detrend="constant")
    f_filt500, p_filt500 = welch(x_filt500, fs=FS_500, nperseg=min(2048, len(x_filt500)), noverlap=min(1024, max(0, len(x_filt500) // 2)), detrend="constant")
    m = (f_ref >= 0) & (f_ref <= 60)
    psd_rel = _rel_rmse(p_ref[m], p_cache[m])
    time_rel = _rel_rmse(x_ref, x_cache)
    p_relerr = np.abs(p_cache - p_ref) / (np.abs(p_ref) + 1e-12)

    mid = n // 2
    half = FS_125
    s0, s1 = max(0, mid - half), min(n, mid + half)
    t = np.arange(s0, s1) / FS_125

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6))
    fig.subplots_adjust(top=0.92, hspace=0.42, wspace=0.30)

    ax = axes[0, 0]
    ax.plot(t, x_ref[s0:s1], color=SCI["purple"], linewidth=1.05, label="reference")
    ax.plot(t, x_cache[s0:s1], color=SCI["gold"], linewidth=1.10, label="current")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title(f"Waveform, rel RMSE={time_rel:.4f}", loc="left", fontsize=10, fontweight="semibold")
    ax.text(0.02, 0.96, f"record {rid}, lead II", transform=ax.transAxes, ha="left", va="top", fontsize=8.4, color=TOKENS["muted"])
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(t, (x_cache - x_ref)[s0:s1], color=SCI["red"], linewidth=1.0)
    ax.axhline(0, color=TOKENS["ink"], linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title("Difference", loc="left", fontsize=10, fontweight="semibold")

    ax = axes[1, 0]
    m500 = (f_raw500 >= 0) & (f_raw500 <= 250)
    ax.semilogy(f_raw500[m500], p_raw500[m500], color=NEUTRAL["mid"], linewidth=0.95, label="raw")
    ax.semilogy(f_filt500[m500], p_filt500[m500], color=SCI["purple"], linewidth=1.05, label="anti-alias LPF")
    ax.axvspan(62.5, 250, color=NEUTRAL["xlight"], alpha=0.80, zorder=0)
    ax.axvline(40, color=NEUTRAL["dark"], linestyle=":", linewidth=0.9, label="40 Hz")
    ax.axvline(62.5, color=TOKENS["ink"], linestyle="--", linewidth=0.9, label="62.5 Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_xlim(0, 250)
    ax.set_title("Anti-aliasing PSD", loc="left", fontsize=10, fontweight="semibold")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    m_pass = (f_ref >= 0) & (f_ref <= 40)
    ax.plot(f_ref[m_pass], p_relerr[m_pass], color=SCI["red"], linewidth=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Pcache - Pref| / |Pref|")
    ax.set_title(f"Passband error, PSD RMSE={psd_rel:.4f}", loc="left", fontsize=10, fontweight="semibold")
    for ax in axes.flat:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_11_resample_alias_reference", manifest, "Current 125 Hz cache versus steep reference resampling error plots.", "Strong preprocessing/QC appendix candidate.")


def plot_noise_wave12_gallery(audit: pd.DataFrame, out_dir: Path, manifest: list[dict], root: Path) -> None:
    cases_dir = root / "outputs" / "sqi_paper_aligned" / "cases_500"
    a = audit.copy()
    a["record_id"] = a["record_id"].map(rid_str)
    a["source_record_id"] = a["source_record_id"].map(rid_str)
    rows = [a.loc[a["noise_type"] == nt].iloc[0] for nt in ["em", "ma"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.5), sharex=True)
    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.18)

    colors = {"em": SCI["purple"], "ma": SCI["gold"]}
    residuals = []
    for row in rows:
        clean = load_case_500(cases_dir, row["source_record_id"])
        noisy = load_case_500(cases_dir, row["record_id"])
        residuals.append(noisy - clean)
    scale = np.nanpercentile(np.abs(np.concatenate([r[: FS_500 * 10].ravel() for r in residuals])), 98)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    offset = 3.0 * scale
    t = np.arange(FS_500 * 10) / FS_500
    ytick_pos = [(len(LEADS_12) - 1 - i) * offset for i in range(len(LEADS_12))]

    for col, row in enumerate(rows):
        nt = str(row["noise_type"])
        residual = residuals[col][: FS_500 * 10]
        ax = axes[col]
        for i, lead in enumerate(LEADS_12):
            pos = ytick_pos[i]
            ax.plot(t, residual[:, i] + pos, color=colors[nt], linewidth=0.72)
        ax.set_title(f"Synthetic noise: {nt}", loc="left", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Time (s), 500 Hz")
        ax.set_yticks(ytick_pos, LEADS_12 if col == 0 else [""] * len(LEADS_12))
        ax.set_xlim(0, 10)
        ax.grid(True, axis="x", linewidth=0.55)
        ax.grid(False, axis="y")
        ax.text(0.02, 0.97, f"source {row['source_record_id']}", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=TOKENS["muted"])
        sns.despine(ax=ax, left=col != 0)
    axes[0].set_ylabel("Lead order, offset")
    finish(fig, out_dir / "diag_12_noise_wave12_gallery", manifest, "Full 12-lead paper-aligned em/ma synthetic noise residual gallery.", "Strong augmentation appendix candidate.")


def plot_single_feature_auc(record84: pd.DataFrame, out_dir: Path, manifest: list[dict]) -> None:
    y01 = (record84["y"].astype(int).to_numpy() == 1).astype(int)
    rows = []
    for col in feature_columns():
        x = record84[col].to_numpy(float)
        m = np.isfinite(x)
        if m.sum() < 10 or len(np.unique(y01[m])) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y01[m], x[m]))
        sep = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
        direction = "higher in acceptable" if np.isfinite(auc) and auc >= 0.5 else "higher in poor"
        rows.append({"feature": col, "auc": auc, "separability_auc": sep, "direction": direction})
    auc_df = pd.DataFrame(rows).sort_values("separability_auc", ascending=False)
    auc_df.to_csv(out_dir / "single_feature_auc_current.csv", index=False)
    top = auc_df.head(20).iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.subplots_adjust(top=0.82, left=0.27)
    add_header(
        fig,
        "Top single-feature SQI separability",
        "Top 20 of 84 record-level features, ranked by max(AUC, 1-AUC) so both high-good and high-poor features are visible.",
        left=0.27,
    )
    palette = {"higher in acceptable": COLORS["blue"]["base"], "higher in poor": COLORS["orange"]["base"]}
    edge = {"higher in acceptable": COLORS["blue"]["dark"], "higher in poor": COLORS["orange"]["dark"]}
    bars = ax.barh(top["feature"], top["separability_auc"], color=[palette[d] for d in top["direction"]], edgecolor=[edge[d] for d in top["direction"]], linewidth=0.8)
    ax.axvline(0.5, color=TOKENS["ink"], linestyle="--", linewidth=1)
    ax.set_xlim(0.48, 1.01)
    ax.set_xlabel("Single-feature separability AUC")
    ax.set_ylabel("Feature")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))
    for rect, sep, auc in zip(bars, top["separability_auc"], top["auc"]):
        ax.text(sep + 0.006, rect.get_y() + rect.get_height() / 2, f"{sep:.3f}", va="center", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[k], edgecolor=edge[k], label=k)
        for k in ["higher in acceptable", "higher in poor"]
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        fontsize=8.5,
        handlelength=1.4,
        borderaxespad=0.6,
    )
    sns.despine(ax=ax)
    finish(fig, out_dir / "diag_13_single_feature_auc_top20", manifest, "Top-20 single-feature SQI separability across the current paper-aligned 84-feature table.", "Good analysis figure for feature behavior.")


def _compute_hr(rpeaks: np.ndarray, fs: int = FS_125) -> float:
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        return np.nan
    rr = np.diff(rpeaks) / float(fs)
    rr = rr[rr > 0]
    if len(rr) == 0:
        return np.nan
    return float(60.0 / np.median(rr))


def plot_qrs_hr_by_lead(qrs_summary: pd.DataFrame, out_dir: Path, manifest: list[dict], root: Path) -> None:
    qrs_dir = root / "outputs" / "sqi_paper_aligned" / "qrs"
    rows = []
    for rid in qrs_summary["record_id"].astype(str).drop_duplicates():
        path = qrs_dir / f"{rid}.npz"
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        for det_key, det_name in [("rpeaks_1", "wqrs"), ("rpeaks_2", "eplimited")]:
            peaks = qrs_peak_lists(z, det_key)
            for lead, arr in zip(LEADS_12, peaks):
                rows.append({"record_id": rid, "lead": lead, "detector": det_name, "hr_bpm": _compute_hr(arr)})
    hr = pd.DataFrame(rows)
    hr["lead"] = pd.Categorical(hr["lead"], LEADS_12, ordered=True)
    finite = hr[np.isfinite(hr["hr_bpm"])].copy()
    finite.to_csv(out_dir / "qrs_hr_by_lead_current.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(top=0.78, wspace=0.27)
    add_header(
        fig,
        "QRS-derived heart-rate sanity checks",
        "Median RR-derived HR by lead and detector, computed from the current paper-aligned QRS cache.",
    )
    ax = axes[0]
    sns.boxplot(
        data=finite,
        x="lead",
        y="hr_bpm",
        hue="detector",
        palette={"wqrs": COLORS["blue"]["base"], "eplimited": COLORS["gold"]["base"]},
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("Lead")
    ax.set_ylabel("HR (bpm)")
    ax.set_title("HR by lead and detector", loc="left", fontsize=10, fontweight="semibold")
    ax.set_ylim(20, min(260, max(160, np.nanpercentile(finite["hr_bpm"], 99) * 1.1)))
    ax.legend(frameon=False, title=None)

    ax = axes[1]
    for det, color, edge in [("wqrs", COLORS["blue"]["base"], COLORS["blue"]["dark"]), ("eplimited", COLORS["gold"]["base"], COLORS["gold"]["dark"])]:
        vals = finite.loc[finite["detector"] == det, "hr_bpm"].to_numpy(float)
        vals = vals[(vals >= 20) & (vals <= 260)]
        ax.hist(vals, bins=45, density=True, histtype="stepfilled", alpha=0.32, color=color, edgecolor=edge, linewidth=1.0, label=f"{det} median={np.median(vals):.1f}")
    ax.set_xlabel("HR (bpm)")
    ax.set_ylabel("Density")
    ax.set_title("Pooled HR distribution", loc="left", fontsize=10, fontweight="semibold")
    ax.legend(frameon=False)
    for ax in axes:
        sns.despine(ax=ax)
    finish(fig, out_dir / "diag_14_qrs_hr_by_lead", manifest, "HR-by-lead boxplots and pooled HR distributions from the current QRS cache.", "Good detector sanity appendix candidate.")


def plot_sqi_12lead_jitter(record84: pd.DataFrame, sqi: str, out_dir: Path, manifest: list[dict]) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.6), sharex=True)
    fig.subplots_adjust(top=0.865, hspace=0.34, wspace=0.23)
    add_header(
        fig,
        f"{sqi} per-lead label distribution",
        "Record-level values are jittered by class for all 12 leads; horizontal ticks mark medians.",
    )

    for ax, lead in zip(axes.flat, LEADS_12):
        col = lead_sqi_col(lead, sqi)
        for y, lab, x0, color, edge in [
            (1, "acceptable", 0.0, COLORS["blue"]["base"], COLORS["blue"]["dark"]),
            (-1, "poor", 1.0, COLORS["orange"]["base"], COLORS["orange"]["dark"]),
        ]:
            vals = record84.loc[record84["y"].astype(int) == y, col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            x = x0 + rng.normal(0, 0.075, size=len(vals))
            ax.scatter(x, vals, s=7, alpha=0.30, color=color, edgecolors="none", rasterized=True, label=lab if lead == "I" else None)
            med = float(np.nanmedian(vals)) if len(vals) else np.nan
            ax.hlines(med, x0 - 0.22, x0 + 0.22, color=edge, linewidth=1.2)
        ax.set_title(lead, fontsize=9, loc="left", fontweight="semibold")
        ax.set_xticks([0, 1], ["acc", "poor"])
        ax.set_ylabel(sqi)
        sns.despine(ax=ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.94, 0.925), frameon=False, ncol=2)
    finish(fig, out_dir / "individual_sqi" / f"diag_sqi_{sqi}_12lead_jitter", manifest, f"{sqi} jitter distribution by class across all 12 leads.", "Exploratory diagnostic; use only if a specific SQI behavior needs illustration.")


def plot_lead_7sqi_jitter(record84: pd.DataFrame, lead: str, out_dir: Path, manifest: list[dict]) -> None:
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.7), sharex=True)
    fig.subplots_adjust(top=0.82, hspace=0.40, wspace=0.30)
    add_header(
        fig,
        f"{lead} lead profile across seven SQIs",
        "Each panel compares acceptable and poor record-level values for one SQI on a fixed lead; median ticks help spot separation and outliers.",
    )
    for ax, sqi in zip(axes.flat, SQI_TYPES):
        col = lead_sqi_col(lead, sqi)
        for y, lab, x0, color, edge in [
            (1, "acceptable", 0.0, COLORS["blue"]["base"], COLORS["blue"]["dark"]),
            (-1, "poor", 1.0, COLORS["orange"]["base"], COLORS["orange"]["dark"]),
        ]:
            vals = record84.loc[record84["y"].astype(int) == y, col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            x = x0 + rng.normal(0, 0.075, size=len(vals))
            ax.scatter(x, vals, s=7, alpha=0.30, color=color, edgecolors="none", rasterized=True, label=lab if sqi == "iSQI" else None)
            med = float(np.nanmedian(vals)) if len(vals) else np.nan
            ax.hlines(med, x0 - 0.22, x0 + 0.22, color=edge, linewidth=1.2)
        ax.set_title(sqi, fontsize=9, loc="left", fontweight="semibold")
        ax.set_xticks([0, 1], ["acc", "poor"])
        ax.set_ylabel("Value")
        sns.despine(ax=ax)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.94, 0.895), frameon=False, ncol=2)
    finish(fig, out_dir / "individual_lead" / f"diag_lead_{lead}_7sqi_jitter", manifest, f"{lead} lead profile across all seven SQIs.", "Exploratory diagnostic; useful for lead-specific implementation checks.")


def make_contact_sheet(out_dir: Path, manifest: list[dict]) -> None:
    selected_names = [
        "diag_01_split_balance",
        "diag_02_qrs_detector_counts",
        "diag_03_feature_integrity",
        "diag_04_pooled_sqi_distributions",
        "diag_05_lead_median_profiles",
        "diag_06_normalization_train_test",
        "diag_07_noise_allocation_snr",
        "diag_08_resampling_wave_psd",
        "diag_09_qrs_overlay_12lead",
        "diag_10_raw_seta_qc",
        "diag_11_resample_alias_reference",
        "diag_12_noise_wave12_gallery",
        "diag_13_single_feature_auc_top20",
        "diag_14_qrs_hr_by_lead",
    ]
    selected = []
    by_name = {row["name"]: row for row in manifest}
    for name in selected_names:
        if name in by_name:
            selected.append(by_name[name])

    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(12.5, 3.55 * nrows)))
    axes = np.asarray(axes).reshape(nrows, ncols)
    fig.subplots_adjust(top=0.91, hspace=0.30, wspace=0.08)
    fig.text(0.045, 0.975, "SQI exploratory diagnostic contact sheet", ha="left", va="top", fontsize=16, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.045, 0.945, "Implementation-check figures generated from the paper-aligned SQI intermediate artifacts.", ha="left", va="top", fontsize=10, color=TOKENS["muted"])

    for ax, row in zip(axes.flat, selected):
        img = mpimg.imread(row["file_png"])
        ax.imshow(img)
        ax.set_axis_off()
        ax.text(
            0.0,
            -0.04,
            row["name"].replace("diag_", ""),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=TOKENS["ink"],
            fontweight="semibold",
        )
    for ax in axes.flat[len(selected) :]:
        ax.axis("off")

    out_base = out_dir / "diagnostic_contact_sheet"
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def write_index(out_dir: Path, manifest: list[dict]) -> None:
    rows = sorted(manifest, key=lambda x: x["name"])
    pd.DataFrame(rows).to_csv(out_dir / "diagnostic_manifest.csv", index=False)
    payload = {
        "n_figures": len(rows),
        "output_dir": str(out_dir),
        "figures": rows,
    }
    (out_dir / "diagnostic_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# SQI exploratory diagnostics",
        "",
        "These figures are implementation-check diagnostics for the paper-aligned SQI reproduction line. They are not the final report figure set.",
        "",
        "| Figure | What it checks | Report use |",
        "|---|---|---|",
    ]
    for row in rows:
        png_name = Path(row["file_png"]).name
        lines.append(f"| `{png_name}` | {row['description']} | {row['report_candidate']} |")
    lines.extend(
        [
            "",
            "Contact sheet:",
            "",
            "- `diagnostic_contact_sheet.png`",
            "- `diagnostic_contact_sheet.pdf`",
        ]
    )
    (out_dir / "diagnostic_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    use_chart_theme()
    root = project_root()
    out_dir = root / "reports" / "sqi_paper_aligned" / "exploratory_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    data = read_inputs(root)
    example_rid = pick_clean_example(data["split"], root)

    plot_split_balance(data["split"], data["audit"], out_dir, manifest)
    plot_qrs_counts(data["qrs_summary"], data["lead7"], out_dir, manifest)
    plot_feature_integrity(data["record84"], out_dir, manifest)
    plot_pooled_sqi_distributions(data["lead7"], out_dir, manifest)
    plot_lead_median_profiles(data["lead7"], out_dir, manifest)
    plot_normalization_check(data["record84"], data["record84_norm"], data["split"], out_dir, manifest)
    plot_noise_allocation(data["audit"], out_dir, manifest, root)
    plot_resample_qc(data["split"], out_dir, manifest, root, example_rid)
    plot_qrs_overlay(out_dir, manifest, root, example_rid)
    plot_raw_seta_qc(out_dir, manifest, root, example_rid)
    plot_resample_alias_check(out_dir, manifest, root, example_rid)
    plot_noise_wave12_gallery(data["audit"], out_dir, manifest, root)
    plot_single_feature_auc(data["record84"], out_dir, manifest)
    plot_qrs_hr_by_lead(data["qrs_summary"], out_dir, manifest, root)

    for sqi in SQI_TYPES:
        plot_sqi_12lead_jitter(data["record84"], sqi, out_dir, manifest)
    for lead in LEADS_12:
        plot_lead_7sqi_jitter(data["record84"], lead, out_dir, manifest)

    make_contact_sheet(out_dir, manifest)
    write_index(out_dir, manifest)
    print(f"[done] wrote {len(manifest)} diagnostic figures to {out_dir}")
    print(f"[done] contact sheet: {out_dir / 'diagnostic_contact_sheet.png'}")
    print(f"[done] index: {out_dir / 'diagnostic_index.md'}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
