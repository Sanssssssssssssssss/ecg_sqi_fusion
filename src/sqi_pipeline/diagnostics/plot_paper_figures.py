from __future__ import annotations

import argparse
import math
import textwrap
import string
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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
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
from src.sqi_pipeline.qrs.paper_detectors import resolve_paper_qrs_executables, run_paper_qrs_12lead
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
    "ink": "#272727",
    "muted": "#767676",
    "grid": "#E6E6E6",
    "axis": "#4D4D4D",
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
    "purple": "#0F4D92",
    "purple_light": "#D9E7F5",
    "gold": "#B64342",
    "gold_light": "#F6CFCB",
    "teal": "#247C7A",
    "red": "#A23B52",
}
SCI_BLUE = SCI["purple"]
SCI_BLUE_LIGHT = SCI["purple_light"]
SCI_RED = SCI["gold"]
SCI_RED_LIGHT = SCI["gold_light"]
SCI_NEUTRAL = "#4A535D"
CMAP_BLUE = LinearSegmentedColormap.from_list("sqi_blue", ["#FFFFFF", SCI_BLUE_LIGHT, SCI_BLUE])
CMAP_RED = LinearSegmentedColormap.from_list("sqi_red", ["#FFFFFF", SCI_RED_LIGHT, SCI_RED])
CMAP_DIVERGE = LinearSegmentedColormap.from_list("sqi_diverge", [SCI_RED, "#FFFFFF", SCI_BLUE])


def use_theme() -> None:
    sns.set_theme(
        style="white",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.monospace": ["Consolas", "DejaVu Sans Mono", "monospace"],
            "font.size": 8,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        },
    )
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def finish(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.075, y: float = 1.055) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=TOKENS["ink"],
        clip_on=False,
    )


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
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar=cbar,
        linewidths=0.7,
        linecolor="#FFFFFF",
        annot=False,
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else Normalize(vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data.iat[i, j])
            r, g, b, _ = cmap_obj(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "#FFFFFF" if lum < 0.42 else TOKENS["ink"]
            ax.text(j + 0.5, i + 0.5, pct_label(val, signed=fmt_signed), ha="center", va="center", fontsize=8.2, color=color)
    ax.tick_params(axis="x", rotation=0, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)
    ax.set_xlabel("")
    ax.set_ylabel("")


def rank_heatmap(ax: plt.Axes, data: pd.DataFrame, *, cmap: str | Any) -> None:
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap_obj,
        vmin=1,
        vmax=max(2, float(np.nanmax(data.to_numpy(dtype=float)))),
        cbar=False,
        linewidths=0.7,
        linecolor="#FFFFFF",
        annot=False,
    )
    norm = Normalize(vmin=1, vmax=max(2, float(np.nanmax(data.to_numpy(dtype=float)))))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data.iat[i, j])
            r, g, b, _ = cmap_obj(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "#FFFFFF" if lum < 0.42 else TOKENS["ink"]
            ax.text(j + 0.5, i + 0.5, f"{int(val)}", ha="center", va="center", fontsize=8.2, color=color)
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
    fig, ax = plt.subplots(figsize=(11.7, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 11.55)
    ax.set_ylim(1.45, 6.0)

    nodes = {
        "paper_labels": (1.15, 5.2, "paper labels\nSet-a + Set-b"),
        "paper_single": (3.05, 5.2, "single-lead\nmanual labels"),
        "paper_eval": (5.0, 5.2, "paper eval\nSet-b / MIT-BIH"),
        "seta": (0.8, 3.65, "Set-a\nlabels"),
        "clean": (2.05, 3.65, "12-lead ECG\n10 s, 500 Hz"),
        "balance": (3.45, 3.65, "balanced Set-a\n773 / 773"),
        "split": (4.85, 3.65, "group split\nsource_record_id"),
        "resample": (6.25, 3.65, "125 Hz\nresampling"),
        "qrs": (7.65, 3.65, "QRS\nwqrs + eplimited"),
        "sqi": (9.05, 3.65, "84 SQIs\n12 leads x 7"),
        "models": (10.25, 3.65, "MLP / SVM\nval threshold"),
        "nstdb": (0.8, 2.25, "NSTDB\nem + ma"),
        "pca": (2.05, 2.25, "PCA noise\n+ third axis"),
        "dower": (3.45, 2.25, "inverse Dower\n12-lead noise"),
        "snr": (4.85, 2.25, "-6 dB\nsynthetic poor"),
    }
    fills = {
        "paper_labels": NEUTRAL["xlight"],
        "paper_single": NEUTRAL["xlight"],
        "paper_eval": NEUTRAL["xlight"],
        "seta": SCI_BLUE_LIGHT,
        "clean": SCI_BLUE_LIGHT,
        "nstdb": SCI_RED_LIGHT,
        "pca": SCI_RED_LIGHT,
        "dower": SCI_RED_LIGHT,
        "snr": SCI_RED_LIGHT,
        "balance": SCI_BLUE_LIGHT,
        "split": SCI_BLUE_LIGHT,
        "resample": SCI_BLUE_LIGHT,
        "qrs": SCI_BLUE_LIGHT,
        "sqi": SCI_BLUE_LIGHT,
        "models": SCI_BLUE_LIGHT,
    }
    widths = {
        "seta": 1.05,
        "clean": 1.3,
        "balance": 1.25,
        "split": 1.25,
        "resample": 1.2,
        "qrs": 1.25,
        "sqi": 1.25,
        "models": 1.15,
        "nstdb": 1.05,
        "pca": 1.25,
        "dower": 1.25,
        "snr": 1.15,
    }
    for key, (x, y, text) in nodes.items():
        draw_node(ax, (x, y), text, fc=fills[key], dashed=key.startswith("paper"), w=widths.get(key, 1.68))

    arrow_pairs = [
        ("paper_labels", "paper_single"),
        ("paper_single", "paper_eval"),
        ("seta", "clean"),
        ("clean", "balance"),
        ("balance", "split"),
        ("split", "resample"),
        ("resample", "qrs"),
        ("qrs", "sqi"),
        ("sqi", "models"),
        ("nstdb", "pca"),
        ("pca", "dower"),
        ("dower", "snr"),
        ("snr", "balance"),
    ]
    for a_key, b_key in arrow_pairs:
        a = nodes[a_key]
        b = nodes[b_key]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        ux = dx / length
        uy = dy / length
        x_pad = min(0.44, length * 0.28)
        y_pad = min(0.28, length * 0.20)
        draw_arrow(
            ax,
            (a[0] + x_pad * ux, a[1] + y_pad * uy),
            (b[0] - x_pad * ux, b[1] - y_pad * uy),
            dashed=a_key.startswith("paper"),
        )

    ax.plot([0.35, 10.25], [4.65, 4.65], color=NEUTRAL["light"], linewidth=1.0)
    ax.text(0.35, 5.78, "paper branch unavailable in this repository", fontsize=9.0, color=NEUTRAL["dark"], ha="left")
    ax.text(0.35, 4.48, "implemented 12-lead paper-aligned branch", fontsize=9.0, color=TOKENS["ink"], ha="left")
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

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 4.9), sharex=True)
    fig.subplots_adjust(hspace=0.28, top=0.93, bottom=0.13)
    for ax, noise, color, title in zip(
        axes,
        ["em", "ma"],
        [SCI_BLUE, SCI_RED],
        ["Electrode motion (em)", "Muscle artifact (ma)"],
    ):
        noisy, _ = load_case(cases_dir, noisy_ids[noise])
        ax.plot(t, clean[:n, li], color=NEUTRAL["dark"], linewidth=0.9, label="clean")
        ax.plot(t, noisy[:n, li], color=color, linewidth=0.82, alpha=0.96, label=f"clean + {noise}")
        ax.set_title(title, loc="left", fontsize=10, fontweight="semibold")
        ax.text(0.01, 0.94, f"record {source}, lead {lead}, SNR {SNR_DB:.0f} dB", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=TOKENS["muted"])
        ax.set_ylabel("mV")
        ax.legend(loc="upper right", frameon=False, fontsize=8.4, ncol=2)
        ax.grid(False)
    for label, ax in zip("ab", axes):
        add_panel_label(ax, label)
    axes[-1].set_xlabel("Time (s)")
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

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 4.9), sharex=True)
    fig.subplots_adjust(hspace=0.30, top=0.93, bottom=0.13)
    for ax, row, label, color in [
        (axes[0], high, "High basSQI", SCI_BLUE),
        (axes[1], low, "Low basSQI", SCI_RED),
    ]:
        sig, leads = load_case(cases_dir, str(row["record_id"]))
        li = leads.index(str(row["lead"]))
        n = int(10 * FS_RAW)
        t = np.arange(n) / FS_RAW
        x = sig[:n, li]
        baseline = lowpass_baseline(x, FS_RAW)
        ax.plot(t, x, color=color, linewidth=0.8, label="raw ECG")
        ax.plot(t, baseline, color=TOKENS["ink"], linestyle="--", linewidth=1.0, label="1 Hz baseline")
        ax.set_title(f"{label} = {float(row['basSQI']):.4f}", loc="left", fontsize=10, fontweight="semibold")
        ax.set_ylabel("mV")
        ax.legend(loc="upper right", frameon=False, fontsize=8.0, ncol=2, handlelength=1.6)
    for label, ax in zip("ab", axes):
        add_panel_label(ax, label)
    axes[-1].set_xlabel("Time (s)")
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

    fig, ax = plt.subplots(figsize=(4.15, 3.4))
    y = scores["y_test"]
    for name, key, thr, color, auc_label in [
        ("MLP", "mlp", scores["mlp_threshold"], SCI_BLUE, auc_mlp_label),
        ("SVM", "svm", scores["svm_threshold"], SCI_RED, auc_svm_label),
    ]:
        fpr, tpr, _ = roc_curve(y, scores[key])
        ax.plot(fpr, tpr, color=color, linewidth=1.25, drawstyle="steps-post", label=f"{name} AUC {auc_label:.3f}")
        pred = (scores[key] > thr).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        op_fpr = fp / max(1, fp + tn)
        op_tpr = tp / max(1, tp + fn)
        ax.scatter([op_fpr], [op_tpr], s=34, facecolors=TOKENS["panel"], edgecolors=color, linewidths=1.15, zorder=4)
    ax.plot([0, 1], [0, 1], color=NEUTRAL["mid"], linestyle=(0, (1.2, 2.0)), linewidth=0.8, zorder=0)
    ax.set_xlabel("1-Sp", fontsize=8.0)
    ax.set_ylabel("Se", fontsize=8.0)
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.tick_params(axis="both", labelsize=7.3, length=3.0, width=0.75)
    ax.grid(color=TOKENS["grid"], linewidth=0.45, alpha=0.72)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=4.8,
            markerfacecolor=TOKENS["panel"],
            markeredgecolor=TOKENS["ink"],
            markeredgewidth=1.1,
        )
    )
    labels.append("validation point")
    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        frameon=False,
        fontsize=7.0,
        handlelength=1.55,
        handletextpad=0.48,
        labelspacing=0.32,
        borderaxespad=0.22,
    )
    finish(fig, out_dir / "fig_04_single_lead_proxy_roc.pdf")


def fig_05a_single_sqi(out_dir: Path, reports_dir: Path) -> None:
    df = pd.read_csv(reports_dir / "table_trend_comparison" / "paper_table5_comparison.csv")
    df = df.sort_values("paper_rank", ascending=True).set_index("SQI")
    acc = df[["paper_Ac_test", "run_Ac_test"]].rename(columns={"paper_Ac_test": "Paper Ac", "run_Ac_test": "Run Ac"})
    delta = df[["delta_Ac_test"]].rename(columns={"delta_Ac_test": "Run-Paper Ac"})
    ranks = df[["paper_rank", "run_rank"]].rename(columns={"paper_rank": "Paper rank", "run_rank": "Run rank"}).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.8), gridspec_kw={"width_ratios": [2.15, 1.2, 1.65], "wspace": 0.08})
    for label, ax in zip("abc", axes):
        add_panel_label(ax, label, y=1.08)
    metric_heatmap(axes[0], acc, cmap=CMAP_BLUE, vmin=0.55, vmax=0.96)
    metric_heatmap(axes[1], delta, cmap=CMAP_DIVERGE, vmin=-0.18, vmax=0.08, center=0.0, fmt_signed=True)
    rank_heatmap(axes[2], ranks, cmap=sns.light_palette(NEUTRAL["dark"], as_cmap=True))
    panel_header(axes[0], "accuracy (%)")
    panel_header(axes[1], "accuracy delta (pp)")
    panel_header(axes[2], "rank (1 = best)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fsqi_y = list(df.index).index("fSQI")
    for ax in axes:
        outline_heatmap_row(ax, fsqi_y, color=SCI_RED)
    finish(fig, out_dir / "fig_05a_single_sqi_dumbbell.pdf")


def fig_05b_combo(out_dir: Path, reports_dir: Path) -> None:
    df = pd.read_csv(reports_dir / "table_trend_comparison" / "paper_table6_comparison.csv")
    order = ["Pairs", "Triplets", "Quadruplets", "Quintuplets", "Sextuplets", "All SQI"]
    df["Group"] = pd.Categorical(df["Group"], order, ordered=True)
    df = df.sort_values("Group", ascending=True).set_index("Group")
    acc = df[["paper_Ac_test", "run_Ac_test"]].rename(columns={"paper_Ac_test": "Paper Ac", "run_Ac_test": "Run Ac"})
    delta = df[["delta_Ac_test"]].rename(columns={"delta_Ac_test": "Run-Paper Ac"})
    ranks = df[["paper_rank", "run_rank"]].rename(columns={"paper_rank": "Paper rank", "run_rank": "Run rank"}).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(11.7, 4.55), gridspec_kw={"width_ratios": [2.05, 1.15, 1.55, 3.25], "wspace": 0.08})
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label, y=1.08)
    metric_heatmap(axes[0], acc, cmap=CMAP_BLUE, vmin=0.89, vmax=0.955)
    metric_heatmap(axes[1], delta, cmap=CMAP_DIVERGE, vmin=-0.05, vmax=0.01, center=0.0, fmt_signed=True)
    rank_heatmap(axes[2], ranks, cmap=sns.light_palette(NEUTRAL["dark"], as_cmap=True))
    panel_header(axes[0], "accuracy (%)")
    panel_header(axes[1], "accuracy delta (pp)")
    panel_header(axes[2], "rank (1 = best)")
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    q_y = list(df.index.astype(str)).index("Quintuplets")
    outline_heatmap_row(axes[0], q_y, color=SCI_BLUE)
    outline_heatmap_row(axes[1], q_y, color=SCI_BLUE)
    outline_heatmap_row(axes[2], q_y, color=SCI_BLUE)

    axes[3].set_axis_off()
    axes[3].set_ylim(len(df), 0)
    axes[3].set_xlim(0, 1)
    panel_header(axes[3], "selected SQIs")
    for i, (_, row) in enumerate(df.iterrows()):
        txt = textwrap.fill(str(row["paper_Selected_SQI"]), width=34, break_long_words=False)
        color = SCI_BLUE if i == q_y else TOKENS["ink"]
        weight = "semibold" if i == q_y else "normal"
        axes[3].text(0.0, i + 0.5, txt, ha="left", va="center", fontsize=8.0, color=color, fontweight=weight)
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
    rec["fSQI_plot"] = rec["fSQI"].astype(float).clip(lower=eps)
    order = ["original acceptable", "original unacceptable", "synthetic poor"]
    palette = {
        "original acceptable": SCI_BLUE,
        "original unacceptable": SCI_RED,
        "synthetic poor": SCI_NEUTRAL,
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    add_panel_label(ax, "a")
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
        ax.scatter(x, np.clip(sample, eps, None), s=7, facecolors=palette[group], edgecolors="none", alpha=0.22)
        med = float(np.median(vals))
        q75 = float(np.quantile(vals, 0.75))
        label_y = min(0.35, max(med * 1.45, q75 * 1.08, 0.0018))
        ax.text(
            i,
            label_y,
            f"median {med:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=TOKENS["ink"],
            bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        )
    ax.set_xlabel("")
    ax.set_ylabel("Record mean fSQI")
    ax.set_yscale("log")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax.set_ylim(1e-4, 1.15)
    ax.tick_params(axis="x", labelrotation=8)
    finish(fig, out_dir / "fig_06_fsqi_by_mechanism.pdf")


def fig_07_mitbih(out_dir: Path, artifacts_dir: Path) -> None:
    overall = pd.read_csv(artifacts_dir / "extra_experiments" / "mitbih_transfer_overall_summary.csv")
    per_record = pd.read_csv(artifacts_dir / "extra_experiments" / "mitbih_transfer_per_record_summary.csv")
    rows = []
    for _, row in overall.iterrows():
        model = "SVM" if "svm" in str(row["model"]) else "MLP"
        rows.append({"model": model, "status": "accepted", "share": float(row["acceptance_rate"])})
        rows.append({"model": model, "status": "rejected", "share": float(row["false_rejection_rate_proxy"])})
    df = pd.DataFrame(rows)
    order = ["MLP", "SVM"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.85), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28})
    for label, ax_panel in zip("ab", axes):
        add_panel_label(ax_panel, label, y=1.10)
    ax = axes[0]
    left = np.zeros(len(order))
    status_order = ["accepted", "rejected"]
    colors = {"accepted": SCI_BLUE_LIGHT, "rejected": SCI_RED_LIGHT}
    edges = {"accepted": SCI_BLUE, "rejected": SCI_RED}
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
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.18), ncol=2, frameon=False, borderaxespad=0)
    panel_header(ax, "overall lead-window acceptance", y=1.08)

    dist = pd.DataFrame(
        {
            "SVM": per_record["svm_acceptance_rate"].astype(float),
            "MLP": per_record["mlp_acceptance_rate"].astype(float),
        }
    ).melt(var_name="model", value_name="acceptance_rate")
    ax2 = axes[1]
    sns.boxplot(
        data=dist,
        x="model",
        y="acceptance_rate",
        order=order,
        ax=ax2,
        width=0.42,
        showfliers=False,
        linewidth=0.9,
        palette={"MLP": SCI_BLUE_LIGHT, "SVM": SCI_RED_LIGHT},
        hue="model",
        legend=False,
    )
    rng = np.random.default_rng(0)
    for i, model in enumerate(order):
        vals = dist.loc[dist["model"].eq(model), "acceptance_rate"].to_numpy()
        ax2.scatter(rng.normal(i, 0.045, size=len(vals)), vals, s=14, color=SCI_BLUE if model == "MLP" else SCI_RED, alpha=0.55, linewidths=0)
        ax2.text(i, min(1.02, np.median(vals) + 0.035), f"median {np.median(vals):.1%}", ha="center", va="bottom", fontsize=8.0, color=TOKENS["ink"])
    ax2.set_ylim(0, 1.04)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_xlabel("Proxy model")
    ax2.set_ylabel("Per-record acceptance rate")
    panel_header(ax2, "48 MIT-BIH records", y=1.08)
    fig.text(0.08, -0.02, "MIT-BIH has no signal-quality ground truth; rejected share is proxy false-rejection, not accuracy.", ha="left", va="top", fontsize=8.3, color=TOKENS["muted"])
    finish(fig, out_dir / "fig_07_mitbih_acceptance.pdf")


def fig_08_runtime(out_dir: Path, artifacts_dir: Path) -> None:
    comp = pd.read_csv(artifacts_dir / "extra_experiments" / "component_timing_summary.csv")
    end = pd.read_csv(artifacts_dir / "extra_experiments" / "end_to_end_timing_summary.csv")
    paper_ref = pd.read_csv(artifacts_dir / "extra_experiments" / "paper_table8_reference.csv")
    qrs = float(end.loc[end["component"].eq("qrs_ms"), "mean_ms"].iloc[0])
    feature = float(end.loc[end["component"].eq("feature84_ms"), "mean_ms"].iloc[0])
    predict = float(end.loc[end["component"].isin(["svm_predict_ms", "mlp_predict_ms"]), "mean_ms"].sum())
    total = float(end.loc[end["component"].eq("total_ms"), "mean_ms"].iloc[0])
    remainder = max(0.0, total - qrs - feature - predict)
    share = qrs / total

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.2), gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.34})
    for label, ax_panel in zip("ab", axes):
        add_panel_label(ax_panel, label, y=1.10)
    ax = axes[0]
    detectors = pd.DataFrame(
        [
            {
                "detector": "wqrs",
                "run_ms": float(comp.loc[comp["component"].eq("wqrs"), "mean_ms"].iloc[0]),
                "paper_ms": float(paper_ref.loc[paper_ref["component"].eq("wqrs"), "paper_ms"].iloc[0]),
            },
            {
                "detector": "eplimited / P&T",
                "run_ms": float(comp.loc[comp["component"].eq("eplimited_PandT"), "mean_ms"].iloc[0]),
                "paper_ms": float(paper_ref.loc[paper_ref["component"].eq("P&T/eplimited"), "paper_ms"].iloc[0]),
            },
        ]
    )
    y = np.arange(len(detectors))
    ax.barh(y, detectors["run_ms"], color=SCI_BLUE_LIGHT, edgecolor=SCI_BLUE, linewidth=1.0)
    ax.scatter(detectors["paper_ms"], y, marker="D", s=34, facecolors=TOKENS["panel"], edgecolors=TOKENS["ink"], linewidths=1.0)
    for yi, row in enumerate(detectors.itertuples()):
        ax.text(row.run_ms + 2.0, yi, f"{row.run_ms:.1f}", va="center", ha="left", fontsize=8.2, color=TOKENS["ink"])
        ax.text(row.paper_ms + 2.0, yi + 0.16, f"{row.paper_ms:.1f}", va="center", ha="left", fontsize=7.8, color=NEUTRAL["dark"])
    ax.set_yticks(y, detectors["detector"])
    ax.set_xlabel("Per-lead mean time (ms)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(detectors["run_ms"].max(), detectors["paper_ms"].max()) * 1.35)
    ax.legend(
        handles=[
            Patch(facecolor=SCI_BLUE_LIGHT, edgecolor=SCI_BLUE, label="this run"),
            Line2D([], [], marker="D", linestyle="None", markerfacecolor=TOKENS["panel"], markeredgecolor=TOKENS["ink"], label="paper Table 8"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        handlelength=1.2,
        borderaxespad=0.0,
    )
    panel_header(ax, "detector comparison", y=1.08)

    ax2 = axes[1]
    non_qrs = feature + predict + remainder
    parts = [
        ("QRS", qrs, SCI_RED_LIGHT, SCI_RED),
        ("84 SQIs", feature, SCI_BLUE_LIGHT, SCI_BLUE),
        ("predict", predict, "#DCE3E8", SCI_NEUTRAL),
        ("overhead", remainder, NEUTRAL["light"], NEUTRAL["dark"]),
    ]
    left = 0.0
    for label, value, fill, edge in parts:
        if value <= 0:
            continue
        ax2.barh(["12-lead"], [value], left=[left], color=fill, edgecolor=edge, linewidth=1.0, label=f"{label} {value:.1f} ms")
        if label == "QRS":
            ax2.text(left + value / 2, 0, f"{label}\n{value:.1f}", ha="center", va="center", fontsize=8.5, color=TOKENS["ink"])
        left += value
    ax2.text(qrs / 2, -0.36, f"QRS share {share:.1%}", ha="center", va="top", fontsize=8.6, color=TOKENS["ink"], clip_on=False)
    ax2.annotate(
        f"non-QRS {non_qrs:.1f} ms",
        xy=(qrs + non_qrs, 0),
        xytext=(total * 1.03, 0.28),
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": NEUTRAL["dark"], "lw": 0.8},
        fontsize=8.5,
        color=TOKENS["ink"],
    )
    ax2.set_xlabel("Mean elapsed time (ms)")
    ax2.set_ylabel("")
    ax2.set_xlim(0, total * 1.18)
    panel_header(ax2, "12-lead runtime decomposition", y=1.08)
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
    artifacts_dir = artifacts_dir.resolve()
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
        _, epl0 = run_paper_qrs_12lead(record_id=f"{rid}w0", sig12=sig, fs=FS_RESAMPLED, leads=leads, executables=executables, work_dir=work_dir, eplimited_warmup_sec=0.0)
        _, epl8 = run_paper_qrs_12lead(record_id=f"{rid}w8", sig12=sig, fs=FS_RESAMPLED, leads=leads, executables=executables, work_dir=work_dir, eplimited_warmup_sec=8.0)
        det0 = epl0[li]
        det8 = epl8[li]
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
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.28, top=0.93, bottom=0.13)
    configs = [
        (axes[0], "No warm-up", ex["det0"], SCI_RED),
        (axes[1], "8 s warm-up", ex["det8"], SCI_BLUE),
    ]
    for ax, label, det, color in configs:
        ax.plot(t, x, color=NEUTRAL["dark"], linewidth=0.8, label="ECG")
        det_t = np.asarray(det, dtype=float) / FS_RESAMPLED
        in_view = det_t[(det_t >= 0) & (det_t <= 10.0)]
        ymin, ymax = np.nanpercentile(x, [1, 99])
        ax.vlines(in_view, ymin, ymax, color=color, linewidth=1.45, alpha=0.98, label=f"{label} detections")
        if len(in_view):
            ax.scatter(
                in_view,
                np.full_like(in_view, ymax),
                marker="v",
                s=28,
                color=color,
                edgecolors="#FFFFFF",
                linewidths=0.35,
                zorder=5,
                clip_on=False,
            )
        ax.set_title(label, loc="left", fontsize=10, fontweight="semibold")
        ax.text(0.01, 0.94, f"record {ex['record_id']}, lead {ex['lead']}, n={len(det)}", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=TOKENS["muted"])
        ax.legend(loc="upper right", frameon=False, fontsize=8.4)
        ax.set_ylabel("mV")
        ax.set_xlim(0, 10)
    for label, ax in zip("ab", axes):
        add_panel_label(ax, label)
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
    p.add_argument("--reports_dir", default=str(root / "outputs" / "reports" / "sqi_paper_aligned"))
    p.add_argument("--out_dir", default=str(root / "outputs" / "reports" / "sqi_paper_aligned" / "images"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_all(Path(args.artifacts_dir), Path(args.reports_dir), Path(args.out_dir), seed=int(args.seed))
    print(Path(args.out_dir))


if __name__ == "__main__":
    main()
