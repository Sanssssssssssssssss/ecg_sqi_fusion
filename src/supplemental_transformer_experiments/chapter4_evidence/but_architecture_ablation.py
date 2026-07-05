from __future__ import annotations

import ast
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.transformer_pipeline.data_v1_gapfill.common import POLICY
from src.transformer_pipeline.data_v1_gapfill.support import run_gm_mechanism_repair_suite as GM

from .common import ROOT, Paths, dry, ensure_dirs, rel, table_to_md, write_json
from .figures import _save


SEEDS = [20260876, 20260877, 20260878]
SPLIT_SEED = 20260876
LABELS = ["good", "medium", "bad"]
DISPLAY = {
    "full_conformer": "Full Conformer",
    "no_hires_cross_attention": "No hi-res cross attention",
    "mean_h_fusion": "Mean-H fusion",
    "pooled_h_only": "Pooled-H only",
}
SHORT_DISPLAY = {
    "full_conformer": "Full\nConformer",
    "no_hires_cross_attention": "No hi-res\ncross-attn",
    "mean_h_fusion": "Mean-H\nfusion",
    "pooled_h_only": "Pooled-H\nonly",
}
ORDER = list(DISPLAY)
ORIGINAL_SUITE_CANDIDATES = GM.suite_candidates


def _patched_forward_tokens(self: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    cfg = GM.ACTIVE_CFG
    hi_ch = self.hi_stem(x)
    hi_tokens = hi_ch.transpose(1, 2)
    ctx = self.ctx_down(hi_ch).transpose(1, 2)
    b = x.shape[0]
    if bool(cfg.get("pooled_h_only", False)):
        context_tokens = self.blocks(self.pos(ctx))
        pooled = hi_tokens.mean(dim=1)
        query_tokens = pooled[:, None, :].expand(-1, len(self.query_names), -1)
        return {
            "hi_ch": hi_ch,
            "hi_tokens": hi_tokens,
            "context_tokens": context_tokens,
            "query_tokens_pre_hires": query_tokens,
            "query_tokens": query_tokens,
        }
    if self.use_queries:
        q = self.query[None, :, :].expand(b, -1, -1)
        seq = torch.cat([q, ctx], dim=1)
        seq = self.blocks(self.pos(seq))
        query_tokens = seq[:, : len(self.query_names), :]
        context_tokens = seq[:, len(self.query_names) :, :]
    else:
        context_tokens = self.blocks(self.pos(ctx))
        pooled = self.pool(context_tokens)
        query_tokens = pooled[:, None, :].expand(-1, len(self.query_names), -1)
    query_tokens = self._maybe_patch_query(query_tokens, "Z_Q")
    query_tokens_pre_hires = query_tokens
    attn_h = hi_tokens
    if str(cfg.get("hires_ablation", "")) == "mean_h":
        attn_h = hi_tokens.mean(dim=1, keepdim=True).expand_as(hi_tokens)
    if self.use_highres_fusion:
        query_delta, _ = self.hi_cross_attn(query_tokens, attn_h, attn_h, need_weights=False)
        query_tokens = query_tokens + query_delta
    query_tokens = self._maybe_patch_query(query_tokens, "Q_e")
    if self.branch_upper_block:
        phys = self.physiology_blocks(torch.cat([query_tokens[:, :6, :], context_tokens], dim=1))[:, :6, :]
        art = self.artifact_blocks(torch.cat([query_tokens[:, 6:, :], context_tokens], dim=1))[:, :2, :]
        query_tokens = torch.cat([phys, art], dim=1)
    return {
        "hi_ch": hi_ch,
        "hi_tokens": attn_h,
        "context_tokens": context_tokens,
        "query_tokens_pre_hires": query_tokens_pre_hires,
        "query_tokens": query_tokens,
    }


def _base_cfg() -> dict[str, Any]:
    for name, cfg in ORIGINAL_SUITE_CANDIDATES():
        if name == "E31_wave_mechanism_conformer":
            return copy.deepcopy(cfg)
    raise RuntimeError("E31_wave_mechanism_conformer config not found")


def _candidate_cfgs() -> list[tuple[str, dict[str, Any]]]:
    base = _base_cfg()
    return [
        ("full_conformer", copy.deepcopy(base)),
        ("no_hires_cross_attention", {**copy.deepcopy(base), "use_highres_fusion": False}),
        ("mean_h_fusion", {**copy.deepcopy(base), "hires_ablation": "mean_h"}),
        (
            "pooled_h_only",
            {
                **copy.deepcopy(base),
                "pooled_h_only": True,
                "use_queries": False,
                "use_highres_fusion": False,
            },
        ),
    ]


def _args(seed: int, candidate: str, *, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        policy=POLICY,
        folds=1,
        seed=int(seed),
        split_seed=SPLIT_SEED,
        epochs=10,
        batch_size=96,
        suite="all",
        candidates=candidate,
        max_train_rows=0,
        max_val_rows=0,
        max_test_rows=0,
        record_balanced_sampler=False,
        audit_gradients=False,
        gradient_batches=30,
        cpu=str(device).lower() == "cpu",
        num_workers=0,
    )


def _set_suite_paths(root: Path) -> None:
    GM.SUITE_OUT_DIR = root / "analysis"
    GM.SUITE_REPORT_DIR = root / "outputs" / "reports"
    GM.SUITE_RUN_DIR = root / "runs"


def _run_candidate(paths: Paths, seed: int, candidate: str, *, device: str) -> Path:
    seed_root = paths.out / "but_architecture_ablation" / f"s{seed}"
    _set_suite_paths(seed_root)
    GM.GMMechanismConformer.forward_tokens = _patched_forward_tokens
    old_candidates = GM.suite_candidates
    try:
        cfgs = dict(_candidate_cfgs())
        GM.suite_candidates = lambda: [(candidate, cfgs[candidate])]
        GM.run_suite(_args(seed, candidate, device=device))
    finally:
        GM.suite_candidates = old_candidates
    return seed_root


def _confusion_from(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=int)
    return np.asarray(value, dtype=int)


def _test_rows(seed_root: Path, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in ORDER:
        pred_path = seed_root / "runs" / f"{candidate}_fold0_seed0" / "test_predictions.npz"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        z = np.load(pred_path, allow_pickle=True)
        y = z["y"].astype(int)
        probs = z["probs"].astype(np.float64)
        pred = probs.argmax(axis=1)
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        if cm.shape != (3, 3):
            raise RuntimeError(f"{candidate} seed {seed}: bad confusion shape {cm.shape}")
        if int(cm.sum()) != 1849:
            raise RuntimeError(f"{candidate} seed {seed}: expected 1849 test rows, got {int(cm.sum())}")
        if cm.sum(axis=1).tolist() != [1053, 632, 164]:
            raise RuntimeError(f"{candidate} seed {seed}: wrong test class counts {cm.sum(axis=1).tolist()}")
        g2m, m2g = int(cm[0, 1]), int(cm[1, 0])
        rows.append(
            {
                "seed": int(seed),
                "candidate": candidate,
                "model": DISPLAY.get(candidate, candidate),
                "acc": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "good_recall": float(recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[0]),
                "medium_recall": float(recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[1]),
                "bad_recall": float(recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[2]),
                "boundary_error_rate": float((g2m + m2g) / (1053 + 632)),
                "good_to_medium": g2m,
                "medium_to_good": m2g,
                "confusion": json.dumps(cm.astype(int).tolist()),
                "run_path": str(pred_path.parent.resolve()),
            }
        )
    return pd.DataFrame(rows)


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["acc", "macro_f1", "good_recall", "medium_recall", "bad_recall", "boundary_error_rate"]
    rows: list[dict[str, Any]] = []
    for candidate in ORDER:
        part = raw.loc[raw["candidate"].eq(candidate)]
        if part.empty:
            continue
        item = {"candidate": candidate, "model": DISPLAY[candidate], "n_seeds": int(part["seed"].nunique())}
        for metric in metrics:
            item[f"{metric}_mean"] = float(part[metric].mean())
            item[f"{metric}_std"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(item)
    summary = pd.DataFrame(rows)
    full = summary.loc[summary["candidate"].eq("full_conformer")]
    if not full.empty:
        base = full.iloc[0]
        for metric in metrics:
            summary[f"delta_{metric}_vs_full"] = summary[f"{metric}_mean"] - float(base[f"{metric}_mean"])
    return summary


def _mean_confusion(raw: pd.DataFrame, candidate: str) -> np.ndarray:
    mats = [_confusion_from(v) for v in raw.loc[raw["candidate"].eq(candidate), "confusion"]]
    if not mats:
        return np.zeros((3, 3), dtype=float)
    return np.mean(np.stack(mats, axis=0), axis=0)


def _annotate_bars(ax: plt.Axes, xs: np.ndarray, vals: np.ndarray, errs: np.ndarray | None = None) -> None:
    for i, v in zip(xs, vals):
        bump = 0.004 if errs is None else float(errs[i] if i < len(errs) else 0.0) + 0.004
        ax.text(float(i), float(v) + bump, f"{float(v):.3f}", ha="center", va="bottom", fontsize=7)


def _figure(paths: Paths, raw: pd.DataFrame, summary: pd.DataFrame) -> Path:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    raw.to_csv(paths.source_data / "fig_M7_architecture_ablation_seed_metrics.csv", index=False)
    summary.to_csv(paths.source_data / "fig_M7_architecture_ablation_summary.csv", index=False)
    fig = plt.figure(figsize=(7.6, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.15], width_ratios=[1, 1, 1, 1, 1])
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:5])
    ax_c1 = fig.add_subplot(gs[1, 0:2])
    ax_c2 = fig.add_subplot(gs[1, 2:5])

    colors = ["#4c78a8", "#c44e52", "#bdbdbd", "#8da0cb"]
    labels = [SHORT_DISPLAY[c] for c in ORDER if c in set(summary["candidate"])]
    x = np.arange(len(labels))
    vals = summary["boundary_error_rate_mean"].to_numpy(float)
    errs = summary["boundary_error_rate_std"].to_numpy(float)
    ax_a.bar(x, vals, yerr=errs, color=colors[: len(labels)], width=0.72, capsize=2)
    ax_a.set_xticks(x, labels, fontsize=6)
    ax_a.set_ylabel("Boundary error rate")
    ax_a.set_ylim(0, max(0.12, float(np.nanmax(vals + errs)) * 1.35))
    _annotate_bars(ax_a, x, vals, errs)
    ax_a.text(-0.17, 1.08, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")

    w = 0.35
    good = summary["good_recall_mean"].to_numpy(float)
    med = summary["medium_recall_mean"].to_numpy(float)
    good_err = summary["good_recall_std"].to_numpy(float)
    med_err = summary["medium_recall_std"].to_numpy(float)
    ax_b.bar(x - w / 2, good, yerr=good_err, width=w, color="#4c78a8", capsize=2, label="Good R")
    ax_b.bar(x + w / 2, med, yerr=med_err, width=w, color="#66a61e", capsize=2, label="Medium R")
    ax_b.set_xticks(x, labels, fontsize=6)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("Recall")
    ax_b.legend(loc="lower left")
    ax_b.text(-0.17, 1.08, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")

    for ax, candidate, letter in [
        (ax_c1, "full_conformer", "c"),
        (ax_c2, "no_hires_cross_attention", ""),
    ]:
        cm = _mean_confusion(raw, candidate)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if candidate == "full_conformer" else "")
        ax.set_title(DISPLAY[candidate], fontsize=7)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.2f}\n{cm[i, j]:.0f}", ha="center", va="center", fontsize=6)
        if letter:
            ax.text(-0.17, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
    out = paths.figures / "fig_M7_but_architecture_ablation"
    _save(fig, out)
    return out.with_suffix(".png")


def _mirror_figure(paths: Paths) -> None:
    target = ROOT / "outputs" / "transformer" / "supplemental" / "chapter4_evidence" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    base = paths.figures / "fig_M7_but_architecture_ablation"
    for ext in [".png", ".svg", ".pdf", ".tiff"]:
        src = base.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, target / src.name)


def _write_report(paths: Paths, raw: pd.DataFrame, summary: pd.DataFrame, fig: Path) -> None:
    full = summary.loc[summary["candidate"].eq("full_conformer")].iloc[0]
    no_hires = summary.loc[summary["candidate"].eq("no_hires_cross_attention")].iloc[0]
    mean_h = summary.loc[summary["candidate"].eq("mean_h_fusion")].iloc[0]
    pooled = summary.loc[summary["candidate"].eq("pooled_h_only")].iloc[0]
    full_err = float(full["boundary_error_rate_mean"])
    no_hires_err = float(no_hires["boundary_error_rate_mean"])
    mean_h_err = float(mean_h["boundary_error_rate_mean"])
    pooled_err = float(pooled["boundary_error_rate_mean"])
    if no_hires_err >= full_err + 0.01 and mean_h_err >= full_err + 0.005:
        verdict = "supports high-resolution query-fusion as a necessary architecture component for good/medium boundary recovery"
    elif pooled_err >= full_err + 0.01 and no_hires_err > full_err:
        verdict = "partial support: the full Conformer is best and pooled-H is clearly worse, but no-hires is only modestly worse"
    elif mean_h_err > full_err:
        verdict = "weak support for time-resolved H; effects are too small for a necessary-structure claim"
    else:
        verdict = "weak or negative evidence for a necessary hi-res query-fusion contribution"
    cols = [
        "model",
        "n_seeds",
        "acc_mean",
        "macro_f1_mean",
        "good_recall_mean",
        "medium_recall_mean",
        "boundary_error_rate_mean",
    ]
    lines = [
        "# BUT Matched Architecture Ablation Retraining",
        "",
        f"- Seeds: `{', '.join(str(s) for s in sorted(raw['seed'].unique()))}`",
        "- Data: v116 official split; train balanced, val/test original-only.",
        f"- Verdict: **{verdict}**.",
        f"- Figure: `{rel(fig)}`",
        "",
        "## Summary",
        "",
        table_to_md(summary[cols], floatfmt=".4f"),
        "",
        "## Seed-level rows",
        "",
        table_to_md(
            raw[
                [
                    "seed",
                    "model",
                    "acc",
                    "macro_f1",
                    "good_recall",
                    "medium_recall",
                    "boundary_error_rate",
                ]
            ],
            floatfmt=".4f",
        ),
        "",
    ]
    (paths.reports / "but_architecture_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        paths.reports / "but_architecture_ablation_summary.json",
        {
            "seeds": sorted(int(s) for s in raw["seed"].unique()),
            "verdict": verdict,
            "figure": rel(fig),
        },
    )


def _validate_split(seed_root: Path) -> dict[str, Any]:
    atlas = next((seed_root / "analysis" / "rh_splits").glob("*/fold0/original_region_atlas.csv"))
    cols = ["split", "class_name", "v116_generated"]
    df = pd.read_csv(atlas, usecols=cols)
    train = df.loc[df["split"].eq("train"), "class_name"].value_counts().to_dict()
    test = df.loc[df["split"].eq("test"), "class_name"].value_counts().to_dict()
    generated_test = int(df.loc[df["split"].eq("test"), "v116_generated"].sum())
    if train != {"good": 8310, "medium": 8310, "bad": 8310}:
        raise RuntimeError(f"wrong train split counts: {train}")
    if test != {"good": 1053, "medium": 632, "bad": 164}:
        raise RuntimeError(f"wrong test split counts: {test}")
    if generated_test != 0:
        raise RuntimeError(f"test generated rows must be 0, got {generated_test}")
    return {"train": train, "test": test, "test_generated_rows": generated_test}


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-architecture-ablation", paths)
        return {"step": "but-architecture-ablation", "skipped": True}
    ensure_dirs(paths)
    rows: list[pd.DataFrame] = []
    split_audits: dict[str, Any] = {}
    for seed in SEEDS:
        seed_root = paths.out / "but_architecture_ablation" / f"s{seed}"
        for candidate in ORDER:
            done = seed_root / "runs" / f"{candidate}_fold0_seed0" / "test_predictions.npz"
            if force or not done.exists():
                print(f"[architecture-ablation] running seed={seed} candidate={candidate}", flush=True)
                _run_candidate(paths, seed, candidate, device=device)
        split_audits[str(seed)] = _validate_split(seed_root)
        rows.append(_test_rows(seed_root, seed))
    raw = pd.concat(rows, ignore_index=True)
    raw["candidate"] = pd.Categorical(raw["candidate"], ORDER, ordered=True)
    raw = raw.sort_values(["candidate", "seed"]).reset_index(drop=True)
    summary = _summarize(raw)
    raw.to_csv(paths.tables / "but_architecture_ablation_retrain.csv", index=False)
    summary.to_csv(paths.tables / "but_architecture_ablation_summary.csv", index=False)
    ranking = summary.sort_values("delta_boundary_error_rate_vs_full", ascending=False).reset_index(drop=True)
    ranking.to_csv(paths.tables / "but_mechanism_contribution_ranking.csv", index=False)
    fig = _figure(paths, raw, summary)
    _mirror_figure(paths)
    _write_report(paths, raw, summary, fig)
    index_path = paths.reports / "figure_index.json"
    figure_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    figure_index["fig_M7_but_architecture_ablation"] = str(fig.resolve())
    write_json(index_path, figure_index)
    out = {
        "raw": rel(paths.tables / "but_architecture_ablation_retrain.csv"),
        "summary": rel(paths.tables / "but_architecture_ablation_summary.csv"),
        "ranking": rel(paths.tables / "but_mechanism_contribution_ranking.csv"),
        "figure": rel(fig),
        "report": rel(paths.reports / "but_architecture_ablation_summary.md"),
        "split_audits": split_audits,
    }
    print(json.dumps(out, indent=2))
    return {"step": "but-architecture-ablation", "skipped": False, "outputs": out}
