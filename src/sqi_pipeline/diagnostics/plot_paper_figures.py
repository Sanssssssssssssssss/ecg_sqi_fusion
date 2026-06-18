from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from scipy.signal import butter, filtfilt
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.sqi_pipeline.config import LEADS_12
from src.sqi_pipeline.diagnostics.paper_extra_experiments import (
    SELECTED5,
    _load_challenge_lead_proxy_data,
    _max_acc_threshold,
)
from src.sqi_pipeline.models.lm_mlp import LMConfig, LMMLP
from src.sqi_pipeline.qrs.paper_detectors import resolve_paper_qrs_executables, run_eplimited_multilead
from src.utils.paths import project_root


FS_RAW = 500
FS_RESAMPLED = 125
SNR_DB = -6.0
FIGURE_NAMES = [
    "fig_01_reproduction_pipeline.pdf",
    "fig_02_noise_generation_examples.pdf",
    "fig_03_bassqi_examples.pdf",
    "fig_04_single_lead_proxy_roc.pdf",
    "fig_05a_single_sqi_dumbbell.pdf",
    "fig_05b_sqi_combination_paired.pdf",
    "fig_06_fsqi_by_mechanism.pdf",
    "fig_07_mitbih_acceptance.pdf",
    "fig_08_runtime_breakdown.pdf",
    "fig_A1_eplimited_warmup.pdf",
]

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E7E9EF",
    "axis": "#B8BDC8",
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


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.monospace": ["Consolas", "DejaVu Sans Mono", "monospace"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def finish(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_header(ax: plt.Axes, text: str, *, x: float = 0.0, y: float = 1.045) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=TOKENS["ink"],
        clip_on=False,
    )


def pct_label(value: float, *, signed: bool = False) -> str:
    if signed:
        return f"{value * 100:+.1f}"
    return f"{value * 100:.1f}"


def metric_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    center: float | None = None,
    fmt_signed: bool = False,
    cbar: bool = False,
) -> None:
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar=cbar,
        linewidths=0.7,
        linecolor="#FFFFFF",
        annot=np.vectorize(lambda x: pct_label(float(x), signed=fmt_signed))(data.to_numpy()),
        fmt="",
        annot_kws={"fontsize": 8.2, "color": TOKENS["ink"]},
    )
    ax.tick_params(axis="x", rotation=0, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)
    ax.set_xlabel("")
    ax.set_ylabel("")


def outline_heatmap_row(ax: plt.Axes, row_idx: int, *, color: str = "#1F2430") -> None:
    n_cols = len(ax.get_xticklabels())
    ax.add_patch(Rectangle((0, row_idx), n_cols, 1, fill=False, edgecolor=color, linewidth=1.2, clip_on=False))


def load_case(cases_dir: Path, record_id: str) -> tuple[np.ndarray, list[str]]:
    z = np.load(cases_dir / f"{record_id}.npz", allow_pickle=True)
    return z["sig_500"].astype(float), [str(x) for x in z["leads"]]


def load_resampled(resampled_dir: Path, record_id: str) -> tuple[np.ndarray, list[str]]:
    z = np.load(resampled_dir / f"{record_id}.npz", allow_pickle=True)
    return z["sig_125"].astype(float), [str(x) for x in z["leads"]]


def lowpass_baseline(x: np.ndarray, fs: float, cutoff_hz: float = 1.0) -> np.ndarray:
    b, a = butter(N=2, Wn=cutoff_hz / (fs / 2.0), btype="lowpass")
    return filtfilt(b, a, x)


def draw_node(ax: plt.Axes, xy: tuple[float, float], text: str, *, w: float = 1.65, h: float = 0.58, fc: str = "#FFFFFF", ec: str = "#464C55", dashed: bool = False) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.045,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.05,
        linestyle=(0, (4, 3)) if dashed else "solid",
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=TOKENS["ink"], fontsize=8.8, linespacing=1.08)


def draw_arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], *, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=NEUTRAL["dark"],
            linestyle=(0, (4, 3)) if dashed else "solid",
        )
    )


def fig_01_pipeline(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)

    nodes = {
        "labels": (1.0, 4.55, "Set-a labels\nacceptable / unacceptable"),
        "clean": (1.0, 3.1, "Set-a ECG\n10 s, 12 leads"),
        "nstdb": (1.0, 1.65, "NSTDB\nem + ma"),
        "pca": (3.0, 1.65, "PCA\n2 noise leads"),
        "dower": (4.75, 1.65, "third component\n+ inverse Dower"),
        "snr": (6.45, 1.65, "-6 dB\nsynthetic poor"),
        "balance": (6.45, 3.1, "balanced Set-a\n773 / 773"),
        "split": (8.15, 3.1, "group split\nsource_record_id"),
        "resample": (8.15, 4.55, "125 Hz\nresampling"),
        "qrs": (6.45, 4.55, "QRS\nwqrs + eplimited"),
        "sqi": (4.75, 4.55, "84 SQIs\n12 leads x 7"),
        "models": (3.0, 4.55, "MLP / SVM\nvalidation threshold"),
        "sub": (8.7, 1.05, "Set-a-only substitution\n(no Set-b, no single-lead labels)"),
    }
    fills = {
        "labels": COLORS["blue"]["xlight"],
        "clean": COLORS["blue"]["xlight"],
        "nstdb": COLORS["gold"]["xlight"],
        "pca": COLORS["gold"]["xlight"],
        "dower": COLORS["gold"]["xlight"],
        "snr": COLORS["orange"]["xlight"],
        "balance": COLORS["orange"]["xlight"],
        "split": COLORS["olive"]["xlight"],
        "resample": COLORS["olive"]["xlight"],
        "qrs": COLORS["pink"]["xlight"],
        "sqi": COLORS["pink"]["xlight"],
        "models": COLORS["blue"]["xlight"],
        "sub": TOKENS["panel"],
    }
    for key, (x, y, text) in nodes.items():
        draw_node(ax, (x, y), text, fc=fills[key], dashed=(key == "sub"), w=2.0 if key == "sub" else 1.62)

    arrow_pairs = [
        ("nstdb", "pca"),
        ("pca", "dower"),
        ("dower", "snr"),
        ("clean", "balance"),
        ("snr", "balance"),
        ("balance", "split"),
        ("split", "resample"),
        ("resample", "qrs"),
        ("qrs", "sqi"),
        ("sqi", "models"),
        ("labels", "models"),
        ("sub", "split"),
    ]
    for a_key, b_key in arrow_pairs:
        a = nodes[a_key]
        b = nodes[b_key]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        ux = dx / length
        uy = dy / length
        draw_arrow(
            ax,
            (a[0] + 0.9 * ux, a[1] + 0.36 * uy),
            (b[0] - 0.9 * ux, b[1] - 0.36 * uy),
            dashed=(a_key == "sub"),
        )

    ax.text(0.6, 0.35, "Dashed path marks the paper-aligned Set-a-only replacement for unavailable Set-b and single-lead labels.", fontsize=8.5, color=TOKENS["muted"])
    finish(fig, out_dir / "fig_01_reproduction_pipeline.pdf")


def choose_noise_source(split: pd.DataFrame) -> tuple[str, dict[str, str]]:
    aug = split[split["is_augmented"].astype(int).eq(1)].copy()
    grouped = aug.groupby("source_record_id")["noise_type"].apply(lambda s: set(s.dropna().astype(str)))
    source = sorted([str(k) for k, v in grouped.items() if {"em", "ma"}.issubset(v)])[0]
    ids: dict[str, str] = {}
    for noise in ["em", "ma"]:
        row = aug[(aug["source_record_id"].astype(str) == source) & (aug["noise_type"].astype(str) == noise)].iloc[0]
        ids[noise] = str(row["record_id"])
    return source, ids


def fig_02_noise_examples(out_dir: Path, artifacts_dir: Path) -> None:
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv")
    source, noisy_ids = choose_noise_source(split)
    cases_dir = artifacts_dir / "cases_500"
    clean, leads = load_case(cases_dir, source)
    lead = "II" if "II" in leads else leads[0]
    li = leads.index(lead)
    n = int(10 * FS_RAW)
    t = np.arange(n) / FS_RAW

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.5), sharex=True)
    for ax, noise, color in zip(axes, ["em", "ma"], [COLORS["orange"]["mid"], COLORS["olive"]["mid"]]):
        noisy, _ = load_case(cases_dir, noisy_ids[noise])
        ax.plot(t, clean[:n, li], color=NEUTRAL["dark"], linewidth=0.9, label="clean")
        ax.plot(t, noisy[:n, li], color=color, linewidth=0.8, alpha=0.92, label=f"clean + {noise}")
        panel_header(ax, f"record {source}, lead {lead}; SNR {SNR_DB:.0f} dB; noise {noise}")
        ax.set_ylabel("mV")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), frameon=False, fontsize=8.5, ncol=2, borderaxespad=0)
        ax.grid(True, axis="x", color=TOKENS["grid"])
    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.34)
    finish(fig, out_dir / "fig_02_noise_generation_examples.pdf")


def fig_03_bassqi_examples(out_dir: Path, artifacts_dir: Path) -> None:
    lead = pd.read_parquet(artifacts_dir / "features" / "lead7.parquet")
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv").rename(columns={"y": "record_y"})
    lead["record_id"] = lead["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    df = lead.merge(split[["record_id", "record_y", "is_augmented", "quality_record"]], on="record_id", how="left")
    cases_dir = artifacts_dir / "cases_500"

    def visual_candidate(frame: pd.DataFrame, *, ascending: bool, p2p_min: float, p2p_max: float) -> pd.Series:
        for _, row in frame.sort_values("basSQI", ascending=ascending).iterrows():
            sig, leads = load_case(cases_dir, str(row["record_id"]))
            li = leads.index(str(row["lead"]))
            x = sig[: int(10 * FS_RAW), li]
            q1, q99 = np.nanpercentile(x, [1, 99])
            robust_p2p = float(q99 - q1)
            if p2p_min <= robust_p2p <= p2p_max and abs(float(np.nanmedian(x))) < 5.0 and float(np.nanstd(x)) > 0.04:
                return row
        return frame.sort_values("basSQI", ascending=ascending).iloc[0]

    high = visual_candidate(df[(df["record_y"].eq(1)) & (df["is_augmented"].eq(0))], ascending=False, p2p_min=0.5, p2p_max=4.0)
    low = visual_candidate(df[(df["record_y"].eq(-1)) & (df["is_augmented"].eq(0))], ascending=True, p2p_min=0.5, p2p_max=10.0)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.5), sharex=True)
    for ax, row, label, color in [
        (axes[0], high, "acceptable high-basSQI", COLORS["blue"]["mid"]),
        (axes[1], low, "unacceptable low-basSQI", COLORS["orange"]["mid"]),
    ]:
        sig, leads = load_case(cases_dir, str(row["record_id"]))
        li = leads.index(str(row["lead"]))
        n = int(10 * FS_RAW)
        t = np.arange(n) / FS_RAW
        x = sig[:n, li]
        baseline = lowpass_baseline(x, FS_RAW)
        ax.plot(t, x, color=color, linewidth=0.8, label="raw ECG")
        ax.plot(t, baseline, color=TOKENS["ink"], linestyle="--", linewidth=1.0, label="1 Hz baseline")
        panel_header(ax, f"{label}; record {row['record_id']}, lead {row['lead']}; basSQI={float(row['basSQI']):.4f}")
        ax.set_ylabel("mV")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), frameon=False, fontsize=8.5, ncol=2, borderaxespad=0)
    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.34)
    finish(fig, out_dir / "fig_03_bassqi_examples.pdf")


def train_proxy_scores(artifacts_dir: Path, seed: int = 0) -> dict[str, Any]:
    df = _load_challenge_lead_proxy_data(artifacts_dir)
    tr = df["split"].eq("train")
    va = df["split"].eq("val")
    te = df["split"].eq("test")
    Xtr = df.loc[tr, SELECTED5].to_numpy(dtype=float)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, SELECTED5].to_numpy(dtype=float)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, SELECTED5].to_numpy(dtype=float)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)

    probe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=25.0, gamma=0.03125, probability=True, random_state=seed))])
    probe.fit(Xtr, ytr)
    svm_thr = float(_max_acc_threshold(yva, probe.predict_proba(Xva)[:, 1])["threshold"])
    svm = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=25.0, gamma=0.03125, probability=True, random_state=seed))])
    svm.fit(np.concatenate([Xtr, Xva], axis=0), np.concatenate([ytr, yva], axis=0))
    p_svm = svm.predict_proba(Xte)[:, 1].astype(float)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)
    dtype = torch.float64
    mlp = LMMLP(J=5, D=len(SELECTED5), device=torch.device("cpu"), dtype=dtype, seed=seed)
    mlp.fit_lm(
        torch.tensor(Xtr_s, dtype=dtype),
        torch.tensor(ytr.astype(float), dtype=dtype),
        LMConfig(epochs_max=100),
        X_val=torch.tensor(Xva_s, dtype=dtype),
        y_val=torch.tensor(yva.astype(float), dtype=dtype),
        model_select_metric="val_auc",
        patience=15,
        threshold=0.5,
    )
    p_mlp_val = mlp.predict_proba(torch.tensor(Xva_s, dtype=dtype))
    mlp_thr = float(_max_acc_threshold(yva, p_mlp_val)["threshold"])
    p_mlp = mlp.predict_proba(torch.tensor(Xte_s, dtype=dtype))
    return {"y_test": yte, "svm": p_svm, "svm_threshold": svm_thr, "mlp": p_mlp, "mlp_threshold": mlp_thr}


def fig_04_roc(out_dir: Path, artifacts_dir: Path) -> None:
    scores = train_proxy_scores(artifacts_dir)
    metrics = pd.read_csv(artifacts_dir / "extra_experiments" / "challenge_singlelead_proxy_metrics.csv")
    auc_svm_label = float(metrics.loc[metrics["model"].eq("singlelead_weak_svm"), "test_AUC"].iloc[0])
    auc_mlp_label = float(metrics.loc[metrics["model"].eq("singlelead_weak_lm_mlp_J5"), "test_AUC"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    y = scores["y_test"]
    for name, key, thr, color, auc_label in [
        ("MLP", "mlp", scores["mlp_threshold"], COLORS["blue"]["mid"], auc_mlp_label),
        ("SVM", "svm", scores["svm_threshold"], COLORS["orange"]["mid"], auc_svm_label),
    ]:
        fpr, tpr, _ = roc_curve(y, scores[key])
        ax.plot(fpr, tpr, color=color, linewidth=1.2, label=f"{name} AUC {auc_label:.3f}")
        pred = (scores[key] > thr).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        op_fpr = fp / max(1, fp + tn)
        op_tpr = tp / max(1, tp + fn)
        ax.scatter([op_fpr], [op_tpr], s=46, facecolors=TOKENS["panel"], edgecolors=color, linewidths=1.4, zorder=4)
    ax.plot([0, 1], [0, 1], color=NEUTRAL["mid"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", linestyle="None", markerfacecolor=TOKENS["panel"], markeredgecolor=TOKENS["ink"], label="validation-selected operating point"))
    labels.append("validation-selected operating point")
    ax.legend(handles=handles, labels=labels, loc="lower right", frameon=False, fontsize=8.8)
    finish(fig, out_dir / "fig_04_single_lead_proxy_roc.pdf")


def fig_05a_single_sqi(out_dir: Path, reports_dir: Path) -> None:
    df = pd.read_csv(reports_dir / "table_trend_comparison" / "paper_table5_comparison.csv")
    df = df.sort_values("paper_rank", ascending=True).set_index("SQI")
    acc = df[["paper_Ac_test", "run_Ac_test"]].rename(columns={"paper_Ac_test": "Paper Ac", "run_Ac_test": "Run Ac"})
    delta = df[["delta_Ac_test"]].rename(columns={"delta_Ac_test": "Run-Paper Ac"})
    profile = df[["run_Se_test", "run_Sp_test"]].rename(columns={"run_Se_test": "Run Se", "run_Sp_test": "Run Sp"})

    fig, axes = plt.subplots(1, 3, figsize=(9.7, 4.8), gridspec_kw={"width_ratios": [2.1, 1.15, 2.1], "wspace": 0.08})
    metric_heatmap(axes[0], acc, cmap=sns.light_palette(COLORS["blue"]["mid"], as_cmap=True), vmin=0.55, vmax=0.96)
    metric_heatmap(axes[1], delta, cmap="vlag", vmin=-0.18, vmax=0.08, center=0.0, fmt_signed=True)
    metric_heatmap(axes[2], profile, cmap=sns.light_palette(COLORS["olive"]["mid"], as_cmap=True), vmin=0.0, vmax=1.0)
    panel_header(axes[0], "accuracy (%)")
    panel_header(axes[1], "accuracy delta (pp)")
    panel_header(axes[2], "run operating profile (%)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fsqi_y = list(df.index).index("fSQI")
    for ax in axes:
        outline_heatmap_row(ax, fsqi_y, color=COLORS["orange"]["dark"])
    finish(fig, out_dir / "fig_05a_single_sqi_dumbbell.pdf")


def fig_05b_combo(out_dir: Path, reports_dir: Path) -> None:
    df = pd.read_csv(reports_dir / "table_trend_comparison" / "paper_table6_comparison.csv")
    order = ["Pairs", "Triplets", "Quadruplets", "Quintuplets", "Sextuplets", "All SQI"]
    df["Group"] = pd.Categorical(df["Group"], order, ordered=True)
    df = df.sort_values("Group", ascending=True).set_index("Group")
    acc = df[["paper_Ac_test", "run_Ac_test"]].rename(columns={"paper_Ac_test": "Paper Ac", "run_Ac_test": "Run Ac"})
    delta = df[["delta_Ac_test"]].rename(columns={"delta_Ac_test": "Run-Paper Ac"})

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.55), gridspec_kw={"width_ratios": [2.1, 1.15, 3.1], "wspace": 0.08})
    metric_heatmap(axes[0], acc, cmap=sns.light_palette(COLORS["blue"]["mid"], as_cmap=True), vmin=0.89, vmax=0.955)
    metric_heatmap(axes[1], delta, cmap="vlag", vmin=-0.05, vmax=0.01, center=0.0, fmt_signed=True)
    panel_header(axes[0], "accuracy (%)")
    panel_header(axes[1], "accuracy delta (pp)")
    axes[1].set_yticklabels([])
    q_y = list(df.index.astype(str)).index("Quintuplets")
    outline_heatmap_row(axes[0], q_y, color=COLORS["olive"]["dark"])
    outline_heatmap_row(axes[1], q_y, color=COLORS["olive"]["dark"])

    axes[2].set_axis_off()
    axes[2].set_ylim(len(df), 0)
    axes[2].set_xlim(0, 1)
    panel_header(axes[2], "selected SQIs")
    for i, (_, row) in enumerate(df.iterrows()):
        txt = textwrap.fill(str(row["paper_Selected_SQI"]), width=34, break_long_words=False)
        color = COLORS["olive"]["dark"] if i == q_y else TOKENS["ink"]
        weight = "semibold" if i == q_y else "normal"
        axes[2].text(0.0, i + 0.5, txt, ha="left", va="center", fontsize=8.0, color=color, fontweight=weight)
    finish(fig, out_dir / "fig_05b_sqi_combination_paired.pdf")


def fig_06_fsqi(out_dir: Path, artifacts_dir: Path) -> None:
    lead = pd.read_parquet(artifacts_dir / "features" / "lead7.parquet")
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv").rename(columns={"y": "record_y"})
    lead["record_id"] = lead["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    df = lead.merge(split[["record_id", "record_y", "is_augmented"]], on="record_id", how="left")
    rec = df.groupby(["record_id", "record_y", "is_augmented"], as_index=False)["fSQI"].mean()
    rec["group"] = np.select(
        [
            rec["record_y"].eq(1) & rec["is_augmented"].eq(0),
            rec["record_y"].eq(-1) & rec["is_augmented"].eq(0),
            rec["record_y"].eq(-1) & rec["is_augmented"].eq(1),
        ],
        ["original acceptable", "original unacceptable", "synthetic poor"],
        default="other",
    )
    rec = rec[rec["group"].ne("other")].copy()
    eps = 1e-4
    rec["fSQI_plot"] = np.log10(rec["fSQI"].astype(float) + eps)
    order = ["original acceptable", "original unacceptable", "synthetic poor"]
    palette = {
        "original acceptable": COLORS["blue"]["mid"],
        "original unacceptable": COLORS["orange"]["mid"],
        "synthetic poor": COLORS["olive"]["mid"],
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sns.boxplot(
        data=rec,
        x="group",
        y="fSQI_plot",
        hue="group",
        order=order,
        palette=palette,
        width=0.42,
        linewidth=0.9,
        showfliers=False,
        legend=False,
        ax=ax,
    )
    rng = np.random.default_rng(0)
    for i, group in enumerate(order):
        vals = rec.loc[rec["group"].eq(group), "fSQI"].to_numpy(dtype=float)
        sample = vals if len(vals) <= 700 else rng.choice(vals, 700, replace=False)
        x = rng.normal(i, 0.055, size=len(sample))
        ax.scatter(x, np.log10(sample + eps), s=7, facecolors=palette[group], edgecolors="none", alpha=0.22)
        med = float(np.median(vals))
        ax.text(i, np.log10(med + eps) + 0.12, f"median {med:.4f}", ha="center", va="bottom", fontsize=8.4, color=TOKENS["ink"])
    ax.set_xlabel("")
    ax.set_ylabel("Record mean fSQI (log10 scale)")
    ticks = np.asarray([0.0, 0.001, 0.01, 0.1, 1.0])
    ax.set_yticks(np.log10(ticks + eps), ["0", "0.001", "0.01", "0.1", "1"])
    ax.set_ylim(np.log10(eps), np.log10(1.0 + eps) + 0.08)
    ax.tick_params(axis="x", labelrotation=8)
    finish(fig, out_dir / "fig_06_fsqi_by_mechanism.pdf")


def fig_07_mitbih(out_dir: Path, artifacts_dir: Path) -> None:
    overall = pd.read_csv(artifacts_dir / "extra_experiments" / "mitbih_transfer_overall_summary.csv")
    rows = []
    for _, row in overall.iterrows():
        model = "SVM" if "svm" in str(row["model"]) else "MLP"
        rows.append({"model": model, "status": "accepted", "share": float(row["acceptance_rate"])})
        rows.append({"model": model, "status": "rejected", "share": float(row["false_rejection_rate_proxy"])})
    df = pd.DataFrame(rows)
    order = ["MLP", "SVM"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    left = np.zeros(len(order))
    status_order = ["accepted", "rejected"]
    colors = {"accepted": "#D8E3F7", "rejected": "#F2C1A5"}
    edges = {"accepted": COLORS["blue"]["dark"], "rejected": COLORS["orange"]["dark"]}
    for status in status_order:
        vals = [float(df[(df["model"].eq(m)) & (df["status"].eq(status))]["share"].iloc[0]) for m in order]
        bars = ax.barh(order, vals, left=left, color=colors[status], edgecolor=edges[status], linewidth=1.0, label=status)
        for bar, val, lft in zip(bars, vals, left):
            ax.text(lft + val / 2, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", ha="center", va="center", fontsize=8.5, color=TOKENS["ink"])
        left += np.asarray(vals)
    ax.set_xlabel("MIT-BIH lead-window share")
    ax.set_ylabel("Proxy model")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2, frameon=False, borderaxespad=0)
    ax.text(0.0, -0.42, "MIT-BIH has no signal-quality ground truth; rejected share is proxy false-rejection.", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=TOKENS["muted"], clip_on=False)
    finish(fig, out_dir / "fig_07_mitbih_acceptance.pdf")


def fig_08_runtime(out_dir: Path, artifacts_dir: Path) -> None:
    end = pd.read_csv(artifacts_dir / "extra_experiments" / "end_to_end_timing_summary.csv")
    qrs = float(end.loc[end["component"].eq("qrs_ms"), "mean_ms"].iloc[0])
    total = float(end.loc[end["component"].eq("total_ms"), "mean_ms"].iloc[0])
    other = total - qrs
    share = qrs / total
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.barh(["12-lead end-to-end"], [qrs], color="#E8D0E4", edgecolor=COLORS["pink"]["dark"], linewidth=1.0, label=f"QRS {qrs:.1f} ms")
    ax.barh(["12-lead end-to-end"], [other], left=[qrs], color="#D8E3F7", edgecolor=COLORS["blue"]["dark"], linewidth=1.0, label=f"Other {other:.1f} ms")
    ax.text(qrs / 2, 0, f"QRS {qrs:.1f} ms\n{share:.1%}", ha="center", va="center", fontsize=9.0, color=TOKENS["ink"])
    ax.annotate(f"other {other:.1f} ms", xy=(qrs + other, 0), xytext=(total * 1.08, 0.18), ha="left", va="center", arrowprops={"arrowstyle": "-", "color": NEUTRAL["dark"], "lw": 0.8}, fontsize=8.8, color=TOKENS["ink"])
    ax.set_xlabel("Mean elapsed time (ms)")
    ax.set_ylabel("")
    ax.set_xlim(0, total * 1.18)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, frameon=False, borderaxespad=0)
    finish(fig, out_dir / "fig_08_runtime_breakdown.pdf")


def unmatched_count(a: np.ndarray, b: np.ndarray, tol: int) -> int:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.size == 0:
        return int(b.size)
    if b.size == 0:
        return int(a.size)
    count = 0
    for arr, other in [(a, b), (b, a)]:
        for t in arr:
            idx = int(np.searchsorted(other, t))
            ok = False
            if idx < len(other) and abs(int(other[idx]) - int(t)) <= tol:
                ok = True
            if idx > 0 and abs(int(other[idx - 1]) - int(t)) <= tol:
                ok = True
            if not ok:
                count += 1
    return count


def find_warmup_example(artifacts_dir: Path, seed: int = 0) -> dict[str, Any]:
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv")
    clean_ids = split[~split["record_id"].astype(str).str.contains("__paper_", regex=False)]["record_id"].astype(str).to_list()
    clean_ids = clean_ids[:80]
    executables = resolve_paper_qrs_executables({}, artifacts_dir / "qrs")
    resampled_dir = artifacts_dir / "resampled_125"
    work_dir = artifacts_dir / "figure_qrs_tmp"
    best: dict[str, Any] | None = None
    for rid in clean_ids:
        sig, leads = load_resampled(resampled_dir, rid)
        lead = "II" if "II" in leads else leads[0]
        li = leads.index(lead)
        one = sig[:, [li]]
        det0 = run_eplimited_multilead(record_id=f"figA1_{rid}_w0", sig=one, fs=FS_RESAMPLED, leads=[lead], executable=executables.eplimited, work_dir=work_dir, eplimited_warmup_sec=0.0)[0]
        det8 = run_eplimited_multilead(record_id=f"figA1_{rid}_w8", sig=one, fs=FS_RESAMPLED, leads=[lead], executable=executables.eplimited, work_dir=work_dir, eplimited_warmup_sec=8.0)[0]
        score = unmatched_count(det0, det8, tol=int(round(0.15 * FS_RESAMPLED))) + abs(len(det0) - len(det8)) * 2
        early_score = unmatched_count(det0[det0 < 5 * FS_RESAMPLED], det8[det8 < 5 * FS_RESAMPLED], tol=int(round(0.15 * FS_RESAMPLED)))
        score += early_score * 2
        item = {"record_id": rid, "lead": lead, "sig": sig[:, li], "det0": det0, "det8": det8, "score": score}
        if best is None or score > best["score"]:
            best = item
    if best is None:
        raise RuntimeError("No eplimited warm-up example candidates found")
    return best


def fig_A1_warmup(out_dir: Path, artifacts_dir: Path) -> None:
    ex = find_warmup_example(artifacts_dir)
    x = ex["sig"]
    t = np.arange(len(x)) / FS_RESAMPLED
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.2), sharex=True, sharey=True)
    configs = [
        (axes[0], "no warm-up", ex["det0"], COLORS["orange"]["mid"]),
        (axes[1], "8 s warm-up", ex["det8"], COLORS["blue"]["mid"]),
    ]
    for ax, label, det, color in configs:
        ax.plot(t, x, color=NEUTRAL["dark"], linewidth=0.8, label="ECG")
        det_t = np.asarray(det, dtype=float) / FS_RESAMPLED
        in_view = det_t[(det_t >= 0) & (det_t <= 10.0)]
        ymin, ymax = np.nanpercentile(x, [1, 99])
        ax.vlines(in_view, ymin, ymax, color=color, linewidth=0.9, alpha=0.95, label=f"{label} detections")
        ax.text(0.01, 0.92, f"record {ex['record_id']}, lead {ex['lead']}; {label}; n={len(det)}", transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=TOKENS["ink"])
        ax.legend(loc="upper right", frameon=False, fontsize=8.4)
        ax.set_ylabel("mV")
        ax.set_xlim(0, 10)
    axes[-1].set_xlabel("Time (s)")
    finish(fig, out_dir / "fig_A1_eplimited_warmup.pdf")


def build_all(artifacts_dir: Path, reports_dir: Path, out_dir: Path, seed: int = 0) -> None:
    use_theme()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_01_pipeline(out_dir)
    fig_02_noise_examples(out_dir, artifacts_dir)
    fig_03_bassqi_examples(out_dir, artifacts_dir)
    fig_04_roc(out_dir, artifacts_dir)
    fig_05a_single_sqi(out_dir, reports_dir)
    fig_05b_combo(out_dir, reports_dir)
    fig_06_fsqi(out_dir, artifacts_dir)
    fig_07_mitbih(out_dir, artifacts_dir)
    fig_08_runtime(out_dir, artifacts_dir)
    fig_A1_warmup(out_dir, artifacts_dir)
    missing = [name for name in FIGURE_NAMES if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing figures: {missing}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    p = argparse.ArgumentParser(description="Generate paper-aligned SQI report figures as vector PDFs.")
    p.add_argument("--artifacts_dir", default=str(root / "outputs" / "sqi_paper_aligned"))
    p.add_argument("--reports_dir", default=str(root / "reports" / "sqi_paper_aligned"))
    p.add_argument("--out_dir", default=str(root / "reports" / "sqi_paper_aligned" / "figures"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_all(Path(args.artifacts_dir), Path(args.reports_dir), Path(args.out_dir), seed=int(args.seed))
    print(Path(args.out_dir))


if __name__ == "__main__":
    main()
