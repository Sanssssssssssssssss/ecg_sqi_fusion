from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.transformer_pipeline.data_v1_gapfill.support import run_event_factorized_sqi_conformer as EVT

from .but_query_patching import (
    _align_clean_predictions,
    _ensure_boundary_records,
    _float,
    _forward,
    _load_model,
    _pred,
    _rescue_pairs,
)
from .common import ROOT, Paths, dry, ensure_dirs, rel, table_to_md, write_json
from .figures import _save


LABELS = ["good", "medium", "bad"]
WINDOW = 31
TOP_K = 3
INTERVENTIONS = ["top-local", "random", "same-class", "donor global-mean"]


def _patchable_forward_tokens(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    hi_ch = self.hi_stem(x)
    hi_tokens = hi_ch.transpose(1, 2)
    ctx = self.ctx_down(hi_ch).transpose(1, 2)
    b = x.shape[0]
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
    patch = getattr(self, "_hires_patch", None)
    patched_hi = _apply_hires_patch(hi_tokens, patch)
    if self.use_highres_fusion and not (patch and patch.get("mode") == "no_hires"):
        query_delta, _ = self.hi_cross_attn(query_tokens, patched_hi, patched_hi, need_weights=False)
        query_tokens = query_tokens + query_delta
    query_tokens = self._maybe_patch_query(query_tokens, "Q_e")
    if self.branch_upper_block:
        phys = self.physiology_blocks(torch.cat([query_tokens[:, :6, :], context_tokens], dim=1))[:, :6, :]
        art = self.artifact_blocks(torch.cat([query_tokens[:, 6:, :], context_tokens], dim=1))[:, :2, :]
        query_tokens = torch.cat([phys, art], dim=1)
    return {
        "hi_ch": hi_ch,
        "hi_tokens": patched_hi,
        "context_tokens": context_tokens,
        "query_tokens_pre_hires": query_tokens_pre_hires,
        "query_tokens": query_tokens,
    }


def _apply_hires_patch(hi_tokens: torch.Tensor, patch: dict[str, Any] | None) -> torch.Tensor:
    if not patch:
        return hi_tokens
    mode = str(patch.get("mode", ""))
    if mode == "shuffle":
        return hi_tokens.flip(1)
    if mode == "mean":
        return hi_tokens.mean(dim=1, keepdim=True).expand_as(hi_tokens)
    if mode != "replace":
        return hi_tokens
    windows = patch.get("windows", [])
    if not windows:
        return hi_tokens
    donor = patch["donor"].to(device=hi_tokens.device, dtype=hi_tokens.dtype)
    if donor.ndim == 2:
        donor = donor.unsqueeze(0)
    out = hi_tokens.clone()
    for start, stop in windows:
        start = max(0, min(int(start), out.shape[1]))
        stop = max(start, min(int(stop), out.shape[1]))
        if stop <= start:
            continue
        if patch.get("replacement") == "global_mean":
            repl = donor.mean(dim=1, keepdim=True).expand(out.shape[0], stop - start, out.shape[2])
        else:
            repl = donor[:, start:stop, :]
            if repl.shape[1] != stop - start:
                continue
            if repl.shape[0] == 1 and out.shape[0] != 1:
                repl = repl.expand(out.shape[0], -1, -1)
        out[:, start:stop, :] = repl
    return out


def _install_hires_patch(model: torch.nn.Module) -> None:
    model.forward_tokens = MethodType(_patchable_forward_tokens, model)


@torch.no_grad()
def _forward_h(model: torch.nn.Module, x: torch.Tensor, patch: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    old = getattr(model, "_hires_patch", None)
    if patch is None:
        if hasattr(model, "_hires_patch"):
            delattr(model, "_hires_patch")
    else:
        model._hires_patch = patch
    try:
        return model(x)
    finally:
        if old is None:
            if hasattr(model, "_hires_patch"):
                delattr(model, "_hires_patch")
        else:
            model._hires_patch = old


def _nonoverlap_top(score: np.ndarray, window: int = WINDOW, k: int = TOP_K) -> list[tuple[int, int]]:
    score = np.asarray(score, dtype=np.float64)
    if score.size < window:
        return [(0, int(score.size))]
    kernel = np.ones(window, dtype=np.float64) / float(window)
    rolled = np.convolve(score, kernel, mode="valid")
    work = rolled.copy()
    out: list[tuple[int, int]] = []
    for _ in range(k):
        if not np.isfinite(work).any():
            break
        start = int(np.nanargmax(work))
        out.append((start, start + window))
        lo = max(0, start - window + 1)
        hi = min(work.size, start + window)
        work[lo:hi] = -np.inf
    return out


def _random_windows(t: int, pair_id: int, window: int = WINDOW, k: int = TOP_K) -> list[tuple[int, int]]:
    rng = np.random.default_rng(20260702 + int(pair_id))
    max_start = max(1, int(t) - window + 1)
    starts = list(rng.permutation(max_start))
    out: list[tuple[int, int]] = []
    for start in starts:
        if all(start + window <= a or start >= b for a, b in out):
            out.append((int(start), int(start + window)))
            if len(out) == k:
                break
    return out


def _local_windows(out: dict[str, torch.Tensor]) -> list[tuple[int, int]]:
    idx = [EVT.LOCAL_MAP_NAMES.index(name) for name in ["baseline", "contact", "reset", "flatline", "detail"]]
    maps = torch.sigmoid(out["local_logits"][0, idx, :]).detach().cpu().numpy()
    q25 = np.percentile(maps, 25, axis=1, keepdims=True)
    q75 = np.percentile(maps, 75, axis=1, keepdims=True)
    scale = np.maximum(q75 - q25, 1.0e-4)
    score = ((maps - np.median(maps, axis=1, keepdims=True)) / scale).mean(axis=0)
    return _nonoverlap_top(score)


def _row(pair_id: int, direction: str, intervention: str, clean: dict[str, torch.Tensor], patched: dict[str, torch.Tensor], windows: list[tuple[int, int]]) -> dict[str, Any]:
    if direction == "necessity":
        delta = _float(clean, "medium_logit") - _float(patched, "medium_logit")
        bad_delta = _float(clean, "bad_logit") - _float(patched, "bad_logit")
        flip = int(_pred(clean) == 1 and _pred(patched) == 0)
    else:
        delta = _float(patched, "medium_logit") - _float(clean, "medium_logit")
        bad_delta = _float(patched, "bad_logit") - _float(clean, "bad_logit")
        flip = int(_pred(clean) == 0 and _pred(patched) == 1)
    return {
        "pair_id": int(pair_id),
        "direction": direction,
        "intervention": intervention,
        "clean_pred": LABELS[_pred(clean)],
        "patched_pred": LABELS[_pred(patched)],
        "delta_h_med": float(delta),
        "delta_bad_logit": float(bad_delta),
        "flip": flip,
        "windows": json.dumps([[int(a), int(b)] for a, b in windows]),
    }


def _transplant_rows(model: torch.nn.Module, test_ds: Any, rescue: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    max_self_err = 0.0
    n = len(rescue)
    for pair_id, rec in rescue.iterrows():
        next_rec = rescue.iloc[(pair_id + 1) % n]
        xm = torch.from_numpy(test_ds.x[int(rec["medium_pos"]) : int(rec["medium_pos"]) + 1]).to(device=device, dtype=torch.float32)
        xg = torch.from_numpy(test_ds.x[int(rec["good_pos"]) : int(rec["good_pos"]) + 1]).to(device=device, dtype=torch.float32)
        xmm = torch.from_numpy(test_ds.x[int(next_rec["medium_pos"]) : int(next_rec["medium_pos"]) + 1]).to(device=device, dtype=torch.float32)
        xgg = torch.from_numpy(test_ds.x[int(next_rec["good_pos"]) : int(next_rec["good_pos"]) + 1]).to(device=device, dtype=torch.float32)
        clean_m = _forward_h(model, xm)
        clean_g = _forward_h(model, xg)
        donor_m = _forward_h(model, xmm)
        donor_g = _forward_h(model, xgg)
        top = _local_windows(clean_m)
        random = _random_windows(clean_m["hi_tokens"].shape[1], pair_id)

        for windows in [[], top]:
            own = _forward_h(model, xm, {"mode": "replace", "windows": windows, "donor": clean_m["hi_tokens"]})
            max_self_err = max(max_self_err, float(torch.max(torch.abs(own["logits"] - clean_m["logits"])).detach().cpu()))

        specs = [
            ("top-local", top, clean_g["hi_tokens"], "window"),
            ("random", random, clean_g["hi_tokens"], "window"),
            ("same-class", top, donor_m["hi_tokens"], "window"),
            ("donor global-mean", top, clean_g["hi_tokens"], "global_mean"),
        ]
        for name, windows, donor, replacement in specs:
            patched = _forward_h(model, xm, {"mode": "replace", "windows": windows, "donor": donor, "replacement": replacement})
            rows.append(_row(pair_id, "necessity", name, clean_m, patched, windows))

        specs = [
            ("top-local", top, clean_m["hi_tokens"], "window"),
            ("random", random, clean_m["hi_tokens"], "window"),
            ("same-class", top, donor_g["hi_tokens"], "window"),
            ("donor global-mean", top, clean_m["hi_tokens"], "global_mean"),
        ]
        for name, windows, donor, replacement in specs:
            patched = _forward_h(model, xg, {"mode": "replace", "windows": windows, "donor": donor, "replacement": replacement})
            rows.append(_row(pair_id, "sufficiency", name, clean_g, patched, windows))
    return pd.DataFrame(rows), max_self_err


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    out = (
        raw.groupby(["direction", "intervention"], as_index=False)
        .agg(
            n=("pair_id", "count"),
            mean_delta_h_med=("delta_h_med", "mean"),
            sem_delta_h_med=("delta_h_med", lambda s: float(s.std(ddof=1) / np.sqrt(max(1, len(s))))),
            flip_rate=("flip", "mean"),
            mean_delta_bad_logit=("delta_bad_logit", "mean"),
        )
    )
    out["intervention"] = pd.Categorical(out["intervention"], INTERVENTIONS, ordered=True)
    return out.sort_values(["direction", "intervention"]).reset_index(drop=True)


@torch.no_grad()
def _path_ablation(model: torch.nn.Module, test_ds: Any, rescue: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, patch in [
        ("full", None),
        ("no hi-res cross attention", {"mode": "no_hires"}),
        ("shuffled H", {"mode": "shuffle"}),
        ("mean H", {"mode": "mean"}),
    ]:
        good_ok = med_ok = errors = 0
        bad_delta: list[float] = []
        for _, rec in rescue.iterrows():
            xm = torch.from_numpy(test_ds.x[int(rec["medium_pos"]) : int(rec["medium_pos"]) + 1]).to(device=device, dtype=torch.float32)
            xg = torch.from_numpy(test_ds.x[int(rec["good_pos"]) : int(rec["good_pos"]) + 1]).to(device=device, dtype=torch.float32)
            clean_m = _forward_h(model, xm)
            clean_g = _forward_h(model, xg)
            out_m = _forward_h(model, xm, patch)
            out_g = _forward_h(model, xg, patch)
            pm, pg = _pred(out_m), _pred(out_g)
            med_ok += int(pm == 1)
            good_ok += int(pg == 0)
            errors += int(pm != 1) + int(pg != 0)
            bad_delta.extend([_float(out_m, "bad_logit") - _float(clean_m, "bad_logit"), _float(out_g, "bad_logit") - _float(clean_g, "bad_logit")])
        n = int(len(rescue))
        rows.append(
            {
                "condition": condition,
                "n_pairs": n,
                "boundary_error_rate": float(errors / max(1, 2 * n)),
                "medium_recall": float(med_ok / max(1, n)),
                "good_recall": float(good_ok / max(1, n)),
                "mean_delta_bad_logit": float(np.mean(bad_delta)),
            }
        )
    return pd.DataFrame(rows)


def _figure(paths: Paths, summary: pd.DataFrame, ablation: pd.DataFrame) -> Path:
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
    summary.to_csv(paths.source_data / "fig_M6_time_local_transplant_summary.csv", index=False)
    ablation.to_csv(paths.source_data / "fig_M6_time_local_path_ablation.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), gridspec_kw={"width_ratios": [1, 1, 1]})
    colors = {
        "top-local": "#66a61e",
        "random": "#8da0cb",
        "same-class": "#8c8c8c",
        "donor global-mean": "#bdbdbd",
    }
    for ax, direction, label, title in [
        (axes[0], "necessity", "a", "medium local H replaced by nearest good"),
        (axes[1], "sufficiency", "b", "good local H replaced by rescued medium"),
    ]:
        data = summary.loc[summary["direction"].eq(direction)].copy()
        vals = data["mean_delta_h_med"].to_numpy(float)
        x = np.arange(len(data))
        ax.bar(x, vals, color=[colors[str(v)] for v in data["intervention"]], width=0.72)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [str(v).replace("donor ", "") for v in data["intervention"]], rotation=35, ha="right")
        ax.set_ylabel("Delta medium logit")
        ax.set_title(title, fontsize=7, pad=9)
        lo = float(min(0.0, vals.min()) - 0.02)
        hi = float(max(0.04, vals.max()) + 0.035)
        ax.set_ylim(lo, hi)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold")
    x = np.arange(len(ablation))
    vals = ablation["boundary_error_rate"].to_numpy(float)
    axes[2].bar(x, vals, color=["#4c78a8", "#c44e52", "#8da0cb", "#bdbdbd"], width=0.72)
    axes[2].set_xticks(x, ablation["condition"], rotation=35, ha="right")
    axes[2].set_ylabel("Boundary error rate")
    axes[2].set_title("hi-res path ablation", fontsize=7, pad=3)
    axes[2].set_ylim(0, max(0.12, float(vals.max()) * 1.35))
    for i, v in enumerate(vals):
        axes[2].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    axes[2].text(-0.16, 1.08, "c", transform=axes[2].transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    out = paths.figures / "fig_M6_but_time_local_latent_transplant"
    _save(fig, out)
    return out.with_suffix(".png")


def _mirror_figures(paths: Paths) -> None:
    target = ROOT / "outputs" / "transformer" / "supplemental" / "chapter4_evidence" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    base = paths.figures / "fig_M6_but_time_local_latent_transplant"
    for ext in [".png", ".svg", ".pdf", ".tiff"]:
        src = base.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, target / src.name)


def _write_report(paths: Paths, raw: pd.DataFrame, summary: pd.DataFrame, ablation: pd.DataFrame, max_self_err: float, fig: Path) -> None:
    wide = summary.pivot(index="direction", columns="intervention", values="mean_delta_h_med")
    top_beats_controls = all(
        float(wide.loc[d, "top-local"]) > max(float(wide.loc[d, c]) for c in ["random", "same-class", "donor global-mean"])
        for d in ["necessity", "sufficiency"]
    )
    full_err = float(ablation.loc[ablation["condition"].eq("full"), "boundary_error_rate"].iloc[0])
    no_hires_err = float(ablation.loc[ablation["condition"].eq("no hi-res cross attention"), "boundary_error_rate"].iloc[0])
    mean_h_err = float(ablation.loc[ablation["condition"].eq("mean H"), "boundary_error_rate"].iloc[0])
    path_support = no_hires_err > full_err and mean_h_err > full_err
    if top_beats_controls and path_support:
        verdict = "supports local high-resolution temporal evidence"
    elif path_support:
        verdict = "supports the high-resolution query-fusion path, but not a uniquely top-local transplant effect"
    else:
        verdict = "does not isolate local high-resolution temporal evidence"
    lines = [
        "# BUT Time-Local Latent Transplant",
        "",
        f"- Rescue pairs: `{raw['pair_id'].nunique()}`",
        f"- Self/empty H-patch max logit error: `{max_self_err:.3g}`",
        f"- Verdict: **{verdict}**.",
        f"- Figure: `{rel(fig)}`",
        "",
        "## Transplant summary",
        "",
        table_to_md(summary),
        "",
        "## Hi-res path ablation",
        "",
        table_to_md(ablation),
        "",
    ]
    (paths.reports / "but_time_local_latent_transplant_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        paths.reports / "but_time_local_latent_transplant_summary.json",
        {
            "rescue_pairs": int(raw["pair_id"].nunique()),
            "self_patch_max_logit_error": float(max_self_err),
            "verdict": verdict,
            "top_local_beats_controls": bool(top_beats_controls),
            "path_ablation_support": bool(path_support),
            "figure": rel(fig),
        },
    )


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-time-local-transplant", paths)
        return {"step": "but-time-local-transplant", "skipped": True}
    ensure_dirs(paths)
    records = _ensure_boundary_records(paths, force=False, device=device)
    model, _, test_ds, torch_device = _load_model(device)
    _install_hires_patch(model)
    _align_clean_predictions(model, test_ds, torch_device)
    rescue = _rescue_pairs(records, test_ds)
    raw, max_self_err = _transplant_rows(model, test_ds, rescue, torch_device)
    if max_self_err > 1.0e-5:
        raise RuntimeError(f"self/empty H-patch gate failed: max logit err {max_self_err:.3g}")
    summary = _summary(raw)
    ablation = _path_ablation(model, test_ds, rescue, torch_device)
    raw.to_csv(paths.tables / "but_time_local_transplant_raw.csv", index=False)
    summary.to_csv(paths.tables / "but_time_local_transplant_summary.csv", index=False)
    ablation.to_csv(paths.tables / "but_time_local_path_ablation.csv", index=False)
    fig = _figure(paths, summary, ablation)
    _mirror_figures(paths)
    index_path = paths.reports / "figure_index.json"
    figure_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    figure_index["fig_M6_but_time_local_latent_transplant"] = str(fig.resolve())
    write_json(index_path, figure_index)
    _write_report(paths, raw, summary, ablation, max_self_err, fig)
    out = {
        "raw": rel(paths.tables / "but_time_local_transplant_raw.csv"),
        "summary": rel(paths.tables / "but_time_local_transplant_summary.csv"),
        "path_ablation": rel(paths.tables / "but_time_local_path_ablation.csv"),
        "figure": rel(fig),
        "report": rel(paths.reports / "but_time_local_latent_transplant_summary.md"),
    }
    print(json.dumps(out, indent=2))
    return {"step": "but-time-local-transplant", "skipped": False, "outputs": out}
