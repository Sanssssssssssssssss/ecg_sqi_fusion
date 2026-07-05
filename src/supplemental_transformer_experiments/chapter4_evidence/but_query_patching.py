from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.transformer_pipeline.data_v1_gapfill.support import run_event_factorized_sqi_conformer as EVT
from src.transformer_pipeline.data_v1_gapfill.support import run_gm_mechanism_repair_suite as GM

from .but_models import _find_e31_predictions
from .common import ROOT, Paths, dry, ensure_dirs, rel, table_to_md, write_json
from .figures import _save


LABELS = ["good", "medium", "bad"]
QUERY_NAMES = EVT.EventFactorizedSQIConformer.query_names
LOCATIONS = ["Z_Q", "Q_e"]
SPLIT_ROOT = (
    ROOT
    / "outputs"
    / "transformer"
    / "v116_e31"
    / "analysis"
    / "good_medium_geometry_repair"
    / "gm_mechanism_repair_suite"
    / "rh_splits"
    / "v116_gapfill_dual_goodorig_nm40__k1_s20260876"
)


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        policy="v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876",
        max_train_rows=0,
        max_val_rows=0,
        max_test_rows=0,
        batch_size=256,
        num_workers=0,
        seed=20260620,
        record_balanced_sampler=False,
        cpu=str(device).lower() == "cpu",
    )


def _load_model(device_name: str):
    run_dir = _find_e31_predictions().parent
    if not (run_dir / "ckpt_best.pt").exists():
        raise FileNotFoundError(f"missing E31 checkpoint: {run_dir / 'ckpt_best.pt'}")
    if not (SPLIT_ROOT / "fold0" / "source_protocol.json").exists():
        raise FileNotFoundError(f"missing E31 split root: {SPLIT_ROOT}")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    GM.install_patches()
    ckpt = _load_ckpt(run_dir / "ckpt_best.pt", device)
    cfg = dict(ckpt["candidate_config"])
    GM.ACTIVE_CFG = dict(cfg)
    train_loader, _, test_loader, _ = GM.EVT.make_loaders(_args(device_name), SPLIT_ROOT, 0)
    model = GM.EVT.EventFactorizedSQIConformer(
        in_ch=test_loader.dataset.x.shape[1],
        factor_dim=len(GM.EVT.FACTOR_COLUMNS),
        width=int(cfg.get("width", 96)),
        layers=int(cfg.get("layers", 3)),
        heads=int(cfg.get("heads", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, train_loader.dataset, test_loader.dataset, device


@torch.no_grad()
def _forward(model: torch.nn.Module, x: torch.Tensor, *, location: str | None = None, query_index: int | None = None, values: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    old = getattr(model, "_query_patch", None)
    if location is not None:
        model._query_patch = {"location": location, "query_index": int(query_index), "values": values.detach()}
    elif hasattr(model, "_query_patch"):
        delattr(model, "_query_patch")
    try:
        return model(x)
    finally:
        if old is None:
            if hasattr(model, "_query_patch"):
                delattr(model, "_query_patch")
        else:
            model._query_patch = old


def _pred(out: dict[str, torch.Tensor]) -> int:
    return int(out["probs"].detach().cpu().numpy()[0].argmax())


def _float(out: dict[str, torch.Tensor], key: str) -> float:
    return float(out[key].detach().cpu().numpy()[0])


def _ensure_boundary_records(paths: Paths, *, force: bool, device: str) -> pd.DataFrame:
    path = paths.tables / "but_good_medium_boundary_records.csv"
    if not path.exists() or force:
        from . import but_boundary_audit

        but_boundary_audit.run(paths, execute=True, force=force, device=device)
    records = pd.read_csv(path)
    test = records.loc[records["split"].astype(str).eq("test")]
    counts = test["true_class"].astype(str).value_counts().to_dict()
    generated = pd.to_numeric(test["v116_generated"], errors="coerce").fillna(0).astype(int)
    if len(test) != 1849 or counts != {"good": 1053, "medium": 632, "bad": 164} or int(generated.sum()) != 0:
        raise RuntimeError(f"bad BUT test scope for query patching: n={len(test)}, counts={counts}, generated={int(generated.sum())}")
    return records


def _rescue_pairs(records: pd.DataFrame, test_ds: Any) -> pd.DataFrame:
    source = pd.to_numeric(test_ds.frame["source_idx"], errors="raise").astype(int)
    source_to_pos = {int(src): i for i, src in enumerate(source)}
    records = records.copy()
    records["record_id"] = records["record_id"].astype(str)
    records["sqi_nearest_good_id"] = records["sqi_nearest_good_id"].astype(str)
    records["source_idx"] = pd.to_numeric(records["source_idx"], errors="raise").astype(int)
    record_to_source = dict(zip(records["record_id"], records["source_idx"]))
    rescue = records.loc[
        records["true_class"].astype(str).eq("medium")
        & records["mlp_pred"].astype(str).eq("good")
        & records["conformer_pred"].astype(str).eq("medium")
        & records["source_idx"].isin(source_to_pos)
        & records["sqi_nearest_good_id"].map(record_to_source).isin(source_to_pos)
    ].copy()
    true_by_id = dict(zip(records["record_id"].astype(str), records["true_class"].astype(str)))
    bad = [rid for rid in rescue["sqi_nearest_good_id"] if true_by_id.get(str(rid)) != "good"]
    if bad:
        raise RuntimeError(f"nearest-good gate failed for {len(bad)} rows")
    if rescue.empty:
        raise RuntimeError("empty rescue set")
    rescue["medium_pos"] = rescue["source_idx"].map(source_to_pos).astype(int)
    rescue["good_source_idx"] = rescue["sqi_nearest_good_id"].map(record_to_source).astype(int)
    rescue["good_pos"] = rescue["good_source_idx"].map(source_to_pos).astype(int)
    return rescue.reset_index(drop=True)


def _align_clean_predictions(model: torch.nn.Module, test_ds: Any, device: torch.device) -> None:
    probs = []
    for i in range(0, len(test_ds), 512):
        x = torch.from_numpy(test_ds.x[i : i + 512]).to(device=device, dtype=torch.float32)
        probs.append(_forward(model, x)["probs"].detach().cpu().numpy())
    got = np.concatenate(probs, axis=0)
    ref = np.load(_find_e31_predictions(), allow_pickle=True)
    if not np.array_equal(ref["y"].astype(int), test_ds.y.astype(int)):
        raise RuntimeError("E31 y does not align with official test split")
    err = float(np.max(np.abs(got - ref["probs"].astype(np.float32))))
    if err > 1.0e-5:
        raise RuntimeError(f"clean forward does not match saved E31 predictions: max_abs_err={err:.3g}")


def _patch_rows(model: torch.nn.Module, test_ds: Any, rescue: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    max_self_err = 0.0
    state_key = {"Z_Q": "query_tokens_pre_hires", "Q_e": "query_tokens"}
    for pair_id, row in rescue.iterrows():
        xm = torch.from_numpy(test_ds.x[int(row["medium_pos"]) : int(row["medium_pos"]) + 1]).to(device=device, dtype=torch.float32)
        xg = torch.from_numpy(test_ds.x[int(row["good_pos"]) : int(row["good_pos"]) + 1]).to(device=device, dtype=torch.float32)
        clean_m = _forward(model, xm)
        clean_g = _forward(model, xg)
        for location in LOCATIONS:
            for qi, query_name in enumerate(QUERY_NAMES):
                own = _forward(model, xm, location=location, query_index=qi, values=clean_m[state_key[location]][:, qi, :])
                max_self_err = max(max_self_err, float(torch.max(torch.abs(own["logits"] - clean_m["logits"])).detach().cpu()))

                patched_m = _forward(model, xm, location=location, query_index=qi, values=clean_g[state_key[location]][:, qi, :])
                patched_g = _forward(model, xg, location=location, query_index=qi, values=clean_m[state_key[location]][:, qi, :])
                rows.append(
                    {
                        "pair_id": int(pair_id),
                        "direction": "rescued_medium_to_nearest_good_query",
                        "location": location,
                        "query": query_name,
                        "medium_record_id": row["record_id"],
                        "good_record_id": row["sqi_nearest_good_id"],
                        "sqi_distance": float(row["sqi_distance"]),
                        "clean_pred": LABELS[_pred(clean_m)],
                        "patched_pred": LABELS[_pred(patched_m)],
                        "delta_h_int": _float(clean_m, "medium_logit") - _float(patched_m, "medium_logit"),
                        "delta_bad_logit": _float(clean_m, "bad_logit") - _float(patched_m, "bad_logit"),
                        "flip": int(_pred(clean_m) == 1 and _pred(patched_m) == 0),
                    }
                )
                rows.append(
                    {
                        "pair_id": int(pair_id),
                        "direction": "nearest_good_to_rescued_medium_query",
                        "location": location,
                        "query": query_name,
                        "medium_record_id": row["record_id"],
                        "good_record_id": row["sqi_nearest_good_id"],
                        "sqi_distance": float(row["sqi_distance"]),
                        "clean_pred": LABELS[_pred(clean_g)],
                        "patched_pred": LABELS[_pred(patched_g)],
                        "delta_h_int": _float(patched_g, "medium_logit") - _float(clean_g, "medium_logit"),
                        "delta_bad_logit": _float(patched_g, "bad_logit") - _float(clean_g, "bad_logit"),
                        "flip": int(_pred(clean_g) == 0 and _pred(patched_g) == 1),
                    }
                )
    return pd.DataFrame(rows), max_self_err


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["direction", "location", "query"], as_index=False)
        .agg(
            n=("pair_id", "count"),
            mean_delta_h_int=("delta_h_int", "mean"),
            sem_delta_h_int=("delta_h_int", lambda s: float(s.std(ddof=1) / np.sqrt(max(1, len(s))))),
            flip_rate=("flip", "mean"),
            mean_delta_bad_logit=("delta_bad_logit", "mean"),
        )
        .sort_values(["direction", "location", "mean_delta_h_int"], ascending=[True, True, False])
    )


@torch.no_grad()
def _query_embeddings(model: torch.nn.Module, ds: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    embs = []
    for i in range(0, len(ds), 512):
        x = torch.from_numpy(ds.x[i : i + 512]).to(device=device, dtype=torch.float32)
        embs.append(_forward(model, x)["query_tokens"].detach().cpu().numpy())
    return np.concatenate(embs, axis=0), ds.y.astype(int)


def _probe_summary(model: torch.nn.Module, train_ds: Any, test_ds: Any, device: torch.device) -> pd.DataFrame:
    train_q, ytr = _query_embeddings(model, train_ds, device)
    test_q, yte = _query_embeddings(model, test_ds, device)
    train_orig = pd.to_numeric(train_ds.frame.get("v116_generated", 0), errors="coerce").fillna(0).astype(int).to_numpy() == 0
    rows = []
    for qi, query_name in enumerate(QUERY_NAMES):
        for task in ["good_vs_medium", "bad_vs_nonbad"]:
            if task == "good_vs_medium":
                tr = train_orig & np.isin(ytr, [0, 1])
                te = np.isin(yte, [0, 1])
                yy_tr = (ytr[tr] == 1).astype(int)
                yy_te = (yte[te] == 1).astype(int)
            else:
                tr = train_orig
                te = np.ones_like(yte, dtype=bool)
                yy_tr = (ytr[tr] == 2).astype(int)
                yy_te = (yte[te] == 2).astype(int)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
            clf.fit(train_q[tr, qi, :], yy_tr)
            score = clf.predict_proba(test_q[te, qi, :])[:, 1]
            rows.append({"query": query_name, "task": task, "auc": float(roc_auc_score(yy_te, score)), "train_n": int(tr.sum()), "test_n": int(te.sum())})
    return pd.DataFrame(rows)


def _figure(paths: Paths, summary: pd.DataFrame) -> Path:
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    forward_qe = summary.loc[
        summary["direction"].eq("rescued_medium_to_nearest_good_query") & summary["location"].eq("Q_e")
    ].copy()
    reverse_qe = summary.loc[
        summary["direction"].eq("nearest_good_to_rescued_medium_query") & summary["location"].eq("Q_e")
    ].copy()
    line = summary.loc[
        summary["direction"].eq("rescued_medium_to_nearest_good_query") & summary["query"].isin(["GM_BOUNDARY", "BAD_STRESS"])
    ].copy()
    forward_qe.to_csv(paths.source_data / "fig_M5_forward_qe.csv", index=False)
    reverse_qe.to_csv(paths.source_data / "fig_M5_reverse_qe.csv", index=False)
    line.to_csv(paths.source_data / "fig_M5_layer_line.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    colors = {"GM_BOUNDARY": "#5AA02C", "BAD_STRESS": "#777777"}
    for ax, data, title in [
        (axes[0], forward_qe, "medium query replaced by nearest good"),
        (axes[2], reverse_qe, "good query replaced by rescued medium"),
    ]:
        data = data.set_index("query").loc[QUERY_NAMES].reset_index()
        vals = data["mean_delta_h_int"].to_numpy(float)
        bar_colors = [colors.get(q, "#8DA0CB") for q in data["query"]]
        ax.bar(np.arange(len(data)), vals, color=bar_colors, width=0.75)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(data)), data["query"], rotation=45, ha="right")
        ax.set_ylabel("Mean delta medium logit")
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for query, data in line.groupby("query"):
        data = data.set_index("location").loc[LOCATIONS].reset_index()
        axes[1].plot(data["location"], data["mean_delta_h_int"], marker="o", lw=1.8, color=colors.get(query, "#4C78A8"), label=query)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Mean delta medium logit")
    axes[1].set_title("patch location", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = paths.figures / "fig_M5_but_query_patching"
    _save(fig, out)
    return out.with_suffix(".png")


def _write_report(paths: Paths, raw: pd.DataFrame, summary: pd.DataFrame, probes: pd.DataFrame, max_self_err: float) -> None:
    qe = summary.loc[summary["direction"].eq("rescued_medium_to_nearest_good_query") & summary["location"].eq("Q_e")].copy()
    gm = qe.loc[qe["query"].eq("GM_BOUNDARY")].iloc[0]
    gm_z = summary.loc[
        summary["direction"].eq("rescued_medium_to_nearest_good_query")
        & summary["location"].eq("Z_Q")
        & summary["query"].eq("GM_BOUNDARY")
    ].iloc[0]
    rank = int(qe["mean_delta_h_int"].rank(ascending=False, method="min")[qe["query"].eq("GM_BOUNDARY")].iloc[0])
    bad = qe.loc[qe["query"].eq("BAD_STRESS")].iloc[0]
    top = qe.sort_values("mean_delta_h_int", ascending=False).iloc[0]
    verdict = (
        "supports a GM_BOUNDARY-mediated medium decision path"
        if rank <= 2 and float(gm["mean_delta_h_int"]) > float(bad["mean_delta_h_int"])
        else "supports a distributed boundary decision path, not a GM_BOUNDARY-only claim"
    )
    lines = [
        "# BUT Query Activation Patching",
        "",
        f"- Rescue pairs: `{raw['pair_id'].nunique()}`",
        f"- Self-patch max logit error: `{max_self_err:.3g}`",
        f"- Verdict: **{verdict}**.",
        f"- Interpretation boundary: this is a causal final-query intervention. The largest Q_e mean delta is `{str(top['query'])}` at `{float(top['mean_delta_h_int']):.4f}`; probe AUCs below only show information content, not causality.",
        f"- Patch-location readout: GM_BOUNDARY mean delta is `{float(gm_z['mean_delta_h_int']):.4f}` at `Z_Q` and `{float(gm['mean_delta_h_int']):.4f}` at `Q_e`.",
        "",
        "## Final-query forward patch",
        "",
        table_to_md(qe[["query", "n", "mean_delta_h_int", "sem_delta_h_int", "flip_rate", "mean_delta_bad_logit"]]),
        "",
        "## Query probe AUC",
        "",
        table_to_md(probes),
        "",
    ]
    (paths.reports / "but_query_patching_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        paths.reports / "but_query_patching_summary.json",
        {
            "rescue_pairs": int(raw["pair_id"].nunique()),
            "self_patch_max_logit_error": max_self_err,
            "gm_boundary_qe_rank": rank,
            "gm_boundary_qe_mean_delta_h_int": float(gm["mean_delta_h_int"]),
            "bad_stress_qe_mean_delta_h_int": float(bad["mean_delta_h_int"]),
            "verdict": verdict,
        },
    )


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-query-patching", paths)
        return {"step": "but-query-patching", "skipped": True}
    ensure_dirs(paths)
    records = _ensure_boundary_records(paths, force=False, device=device)
    model, train_ds, test_ds, torch_device = _load_model(device)
    _align_clean_predictions(model, test_ds, torch_device)
    rescue = _rescue_pairs(records, test_ds)
    raw, max_self_err = _patch_rows(model, test_ds, rescue, torch_device)
    if max_self_err > 1.0e-5:
        raise RuntimeError(f"self-patch gate failed: max logit err {max_self_err:.3g}")
    summary = _summary(raw)
    probes = _probe_summary(model, train_ds, test_ds, torch_device)
    raw.to_csv(paths.tables / "but_query_patching_raw.csv", index=False)
    summary.to_csv(paths.tables / "but_query_patching_summary.csv", index=False)
    probes.to_csv(paths.tables / "but_query_probe_summary.csv", index=False)
    fig = _figure(paths, summary)
    index_path = paths.reports / "figure_index.json"
    figure_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    figure_index["fig_M5_but_query_patching"] = str(fig.resolve())
    write_json(index_path, figure_index)
    _write_report(paths, raw, summary, probes, max_self_err)
    out = {
        "raw": rel(paths.tables / "but_query_patching_raw.csv"),
        "summary": rel(paths.tables / "but_query_patching_summary.csv"),
        "probes": rel(paths.tables / "but_query_probe_summary.csv"),
        "figure": rel(fig),
        "report": rel(paths.reports / "but_query_patching_summary.md"),
    }
    print(json.dumps(out, indent=2))
    return {"step": "but-query-patching", "skipped": False, "outputs": out}
