"""Joint PTB+BUT training for interpretable waveform-only SQI Conformer.

This runner answers a different question from one-way cross-dataset transfer:
can the interpretable Event-Factorized SQI Conformer perform well when trained
directly on the cleaned PTB synthetic protocol plus cleaned BUT train split?

Inference remains waveform-derived channels only.  SQI/factor columns are used
only as training teacher targets and diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
EVENT_SCRIPT = ANALYSIS_DIR / "run_event_factorized_sqi_conformer.py"
BASE_SCRIPT = ANALYSIS_DIR / "run_event_factorized_cross_dataset_top3.py"
ALIGNED_SCRIPT = ANALYSIS_DIR / "run_event_factorized_cross_dataset_aligned_ptb.py"

DEFAULT_BUT_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier"
DEFAULT_PTB_PROTOCOL = ANALYSIS_DIR / "event_xds_aligned_v11_buttrain_style_replay" / "protocol_ptb_buttrain_aligned_pc3000_s20260620"
DEFAULT_PTB_BAD_EXTRA = [
    ANALYSIS_DIR / "event_xds_aligned_v1" / "protocol_ptb_buttrain_aligned_pc3000_s20260620",
    ANALYSIS_DIR / "event_xds_aligned_v10_butmedium_extreme" / "protocol_ptb_buttrain_aligned_pc3000_s20260620",
]

JOINT_OUT = ANALYSIS_DIR / "event_joint_ptb_but_interpretable"
JOINT_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_joint_ptb_but_report.md"
JOINT_RUN_DIR = OUT_ROOT / "runs" / "event_joint_ptb_but_interpretable"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVENT = load_module(EVENT_SCRIPT, "event_factorized_for_joint_ptb_but")
BASE = load_module(BASE_SCRIPT, "event_cross_base_for_joint_ptb_but")
ALIGNED = load_module(ALIGNED_SCRIPT, "event_aligned_for_joint_ptb_but")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    JOINT_OUT.mkdir(parents=True, exist_ok=True)
    JOINT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JOINT_RUN_DIR.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_protocol(protocol: Path) -> tuple[np.ndarray, pd.DataFrame]:
    z = np.load(protocol / "signals.npz")
    key = "X" if "X" in z.files else z.files[0]
    x = np.asarray(z[key], dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    atlas = pd.read_csv(protocol / "original_region_atlas.csv").copy()
    return x.astype(np.float32), atlas


def prefix_atlas(atlas: pd.DataFrame, domain: str, idx_offset: int) -> pd.DataFrame:
    out = atlas.copy()
    out["idx"] = np.arange(idx_offset, idx_offset + len(out), dtype=np.int64)
    out["source_idx"] = np.arange(idx_offset, idx_offset + len(out), dtype=np.int64)
    out["source_domain"] = domain
    out["dataset"] = domain
    out["record_id"] = domain + "_" + out.get("record_id", pd.Series(["rec"] * len(out))).fillna("rec").astype(str)
    out["subject_id"] = domain + "_" + out.get("subject_id", out["record_id"]).fillna("subj").astype(str)
    out["joint_origin_idx"] = atlas.get("idx", pd.Series(np.arange(len(atlas)))).to_numpy()
    return out


def materialize_joint_protocol(args: argparse.Namespace) -> Path:
    ensure_dirs()
    protocol = JOINT_OUT / f"protocol_joint_v11_badextra{int(args.bad_extra_per_source)}_s{int(args.seed)}"
    atlas_path = protocol / "original_region_atlas.csv"
    signals_path = protocol / "signals.npz"
    if atlas_path.exists() and signals_path.exists() and not bool(args.force_protocol):
        return protocol
    protocol.mkdir(parents=True, exist_ok=True)

    # Ensure the aligned protocols exist. This is harmless if already materialized.
    ALIGNED.configure_variant("v11_buttrain_style_replay")
    if not DEFAULT_PTB_PROTOCOL.exists() or bool(args.force_protocol):
        tmp = argparse.Namespace(
            per_class=3000,
            seed=20260620,
            force_protocol=False,
            but_protocol=str(DEFAULT_BUT_PROTOCOL),
            variant="v11_buttrain_style_replay",
        )
        ALIGNED.materialize_aligned_protocol(tmp)
    for variant, path in [("v1", DEFAULT_PTB_BAD_EXTRA[0]), ("v10_butmedium_extreme", DEFAULT_PTB_BAD_EXTRA[1])]:
        if not path.exists() or bool(args.force_protocol):
            ALIGNED.configure_variant(variant)
            tmp = argparse.Namespace(
                per_class=3000,
                seed=20260620,
                force_protocol=False,
                but_protocol=str(DEFAULT_BUT_PROTOCOL),
                variant=variant,
            )
            ALIGNED.materialize_aligned_protocol(tmp)

    xs: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    offset = 0

    ptb_x, ptb_atlas = load_protocol(DEFAULT_PTB_PROTOCOL)
    ptb_frame = prefix_atlas(ptb_atlas, "ptb_v11_style_replay", offset)
    xs.append(ptb_x)
    frames.append(ptb_frame)
    offset += len(ptb_frame)

    but_x, but_atlas = load_protocol(DEFAULT_BUT_PROTOCOL)
    but_frame = prefix_atlas(but_atlas, "clean_but", offset)
    xs.append(but_x)
    frames.append(but_frame)
    offset += len(but_frame)

    rng = np.random.default_rng(int(args.seed) + 3900)
    for extra_i, extra_protocol in enumerate(DEFAULT_PTB_BAD_EXTRA):
        extra_x, extra_atlas = load_protocol(extra_protocol)
        mask = extra_atlas["split"].astype(str).eq("train") & extra_atlas["class_name"].astype(str).eq("bad")
        idx = np.flatnonzero(mask.to_numpy())
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(int(args.bad_extra_per_source), idx.size), replace=False)
        part_x = extra_x[take]
        part_atlas = extra_atlas.iloc[take].copy().reset_index(drop=True)
        part_atlas["split"] = "train"
        part_atlas["bad_extra_source_protocol"] = str(extra_protocol)
        part_frame = prefix_atlas(part_atlas, f"ptb_bad_extra_{extra_i}", offset)
        xs.append(part_x)
        frames.append(part_frame)
        offset += len(part_frame)

    x_all = np.concatenate(xs, axis=0).astype(np.float32)
    atlas = pd.concat(frames, ignore_index=True)
    atlas["idx"] = np.arange(len(atlas), dtype=np.int64)
    atlas["source_idx"] = np.arange(len(atlas), dtype=np.int64)

    for col in EVENT.FORMAL_FEATURE_COLUMNS:
        if col not in atlas.columns:
            atlas[col] = 0.0
    y = atlas["class_name"].astype(str).map(EVENT.CLASS_TO_INT).fillna(0).astype(int).to_numpy()
    atlas["y"] = y

    np.savez_compressed(signals_path, X=x_all)
    atlas.to_csv(atlas_path, index=False)
    counts = atlas.groupby(["source_domain", "split", "class_name"]).size().reset_index(name="n")
    counts.to_csv(protocol / "joint_source_split_counts.csv", index=False)
    summary = {
        "created_at": now(),
        "n": int(len(atlas)),
        "ptb_protocol": str(DEFAULT_PTB_PROTOCOL),
        "but_protocol": str(DEFAULT_BUT_PROTOCOL),
        "bad_extra_protocols": [str(p) for p in DEFAULT_PTB_BAD_EXTRA],
        "bad_extra_per_source": int(args.bad_extra_per_source),
        "class_counts": atlas["class_name"].value_counts().to_dict(),
        "source_split_counts": counts.to_dict(orient="records"),
        "input_contract": "waveform-derived channels only; SQI/factors are teacher targets and diagnostics",
    }
    (protocol / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return protocol


def make_dataset(protocol: Path, split: str, ref_ds: Any | None = None, max_rows: int = 0, seed: int = 0, all_rows: bool = False) -> Any:
    atlas_frame = None
    use_split = split
    if all_rows:
        atlas_frame = pd.read_csv(protocol / "original_region_atlas.csv")
        atlas_frame["split"] = "test"
        use_split = "test"
    return EVENT.CleanProtocolDataset(
        protocol,
        use_split,
        atlas_frame=atlas_frame,
        channel_stats=None if ref_ds is None else ref_ds.channel_stats,
        feature_norm=None if ref_ds is None else ref_ds.feature_norm,
        factor_transform=None if ref_ds is None else ref_ds.factor_transform,
        artifact_thresholds=None if ref_ds is None else ref_ds.artifact_thresholds,
        max_rows=max_rows,
        seed=seed,
    )


def loader(ds: Any, batch_size: int, shuffle: bool = False, sampler: Any | None = None, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def make_model(cfg: dict[str, Any], in_ch: int, device: torch.device) -> torch.nn.Module:
    return EVENT.EventFactorizedSQIConformer(
        in_ch=int(in_ch),
        factor_dim=len(EVENT.FACTOR_COLUMNS),
        width=int(cfg["width"]),
        layers=int(cfg["layers"]),
        heads=int(cfg["heads"]),
        use_queries=bool(cfg.get("use_queries", True)),
        use_highres_fusion=bool(cfg.get("use_highres_fusion", True)),
        use_local_supervision=bool(cfg.get("use_local_supervision", True)),
        use_artifact_aux=bool(cfg.get("use_artifact_aux", True)),
        use_hierarchical_head=bool(cfg.get("use_hierarchical_head", True)),
        branch_upper_block=bool(cfg.get("branch_upper_block", False)),
    ).to(device)


def train_joint(candidate: str, cfg: dict[str, Any], joint_protocol: Path, args: argparse.Namespace, seed_index: int) -> dict[str, Any]:
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = make_dataset(joint_protocol, "train", max_rows=int(args.max_train_rows), seed=seed)
    val_ds = make_dataset(joint_protocol, "val", ref_ds=train_ds, max_rows=int(args.max_val_rows), seed=seed + 1)
    joint_test_ds = make_dataset(joint_protocol, "test", ref_ds=train_ds, max_rows=int(args.max_test_rows), seed=seed + 2)
    ptb_test_ds = make_dataset(DEFAULT_PTB_PROTOCOL, "test", ref_ds=train_ds, max_rows=int(args.max_eval_rows), seed=seed + 3)
    but_test_ds = make_dataset(DEFAULT_BUT_PROTOCOL, "test", ref_ds=train_ds, max_rows=int(args.max_eval_rows), seed=seed + 4)
    ptb_all_ds = make_dataset(DEFAULT_PTB_PROTOCOL, "test", ref_ds=train_ds, max_rows=int(args.max_eval_all_rows), seed=seed + 5, all_rows=True)
    but_all_ds = make_dataset(DEFAULT_BUT_PROTOCOL, "test", ref_ds=train_ds, max_rows=int(args.max_eval_all_rows), seed=seed + 6, all_rows=True)

    sampler = EVENT.record_balanced_sampler(train_ds, seed) if bool(args.record_balanced_sampler) else None
    train_loader = loader(train_ds, int(args.batch_size), shuffle=True, sampler=sampler, num_workers=int(args.num_workers))
    val_loader = loader(val_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))
    eval_loaders = {
        "joint_val": (val_ds, val_loader),
        "joint_test": (joint_test_ds, loader(joint_test_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))),
        "ptb_test": (ptb_test_ds, loader(ptb_test_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))),
        "but_test": (but_test_ds, loader(but_test_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))),
        "ptb_all": (ptb_all_ds, loader(ptb_all_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))),
        "but_all": (but_all_ds, loader(but_all_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))),
    }

    model = make_model(cfg, int(train_ds.x.shape[1]), device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    pre_logs = EVENT.pretrain_model(model, train_loader, cfg, device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        parts_rows = []
        for batch in train_loader:
            total, parts, _ = EVENT.model_loss(model, batch, cfg, device)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(total.detach().cpu()))
            parts_rows.append(parts)
        val_rep, *_ = EVENT.eval_loader(model, val_loader, device, candidate)
        score = float(val_rep["record_macro_supported_f1"]) + 0.20 * min(
            float(val_rep["good_recall"]), float(val_rep["medium_recall"]), float(val_rep["bad_recall"])
        ) - 0.05 * float(val_rep["bad_fpr_nonbad"])
        part_mean = pd.DataFrame(parts_rows).mean(numeric_only=True).to_dict() if parts_rows else {}
        row = {
            "phase": "finetune",
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"train_{k}": v for k, v in part_mean.items()},
            **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"},
        }
        logs.append(row)
        print(
            f"joint {candidate} seed={seed_index} epoch={epoch}/{args.epochs} "
            f"loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} "
            f"val_rec={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    metrics: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    preds: dict[str, Any] = {}
    for bucket, (eval_ds, eval_loader_obj) in eval_loaders.items():
        rep, probs, factor_pred, factor_true, y, artifact, rec_rows, _ = EVENT.eval_loader(model, eval_loader_obj, device, candidate)
        row = EVENT.metric_row(candidate, bucket, rep)
        row["seed_index"] = int(seed_index)
        row["stage"] = str(cfg.get("stage", "phase1"))
        row["n_rows"] = int(len(eval_ds))
        metrics.append(row)
        for rec in rec_rows:
            rec = dict(rec)
            rec["bucket"] = bucket
            rec["seed_index"] = int(seed_index)
            records.append(rec)
        for frec in EVENT.factor_recovery_rows(candidate, bucket, factor_pred, factor_true, y):
            frec["seed_index"] = int(seed_index)
            recovery.append(frec)
        preds[bucket] = {"probs": probs, "y": y, "artifact_presence": artifact}

    run_path = JOINT_RUN_DIR / f"joint_{candidate}_seed{seed_index}"
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "factor_columns": EVENT.FACTOR_COLUMNS,
            "local_map_names": EVENT.LOCAL_MAP_NAMES,
            "input_contract": "waveform-derived channels only; no tabular feature input at inference",
            "model_kind": "EventFactorizedSQIConformer",
            "train_protocol": str(joint_protocol),
            "eval_protocols": {"ptb": str(DEFAULT_PTB_PROTOCOL), "but": str(DEFAULT_BUT_PROTOCOL)},
            "no_warm_start": True,
        },
        run_path / "ckpt_best.pt",
    )
    pd.DataFrame(pre_logs + logs).to_csv(run_path / "train_log.csv", index=False)
    for bucket, payload in preds.items():
        np.savez_compressed(run_path / f"{bucket}_predictions.npz", **payload)
    return {"metrics": metrics, "records": records, "recovery": recovery, "run_path": str(run_path)}


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "acc",
        "macro_f1",
        "good_recall",
        "medium_recall",
        "bad_recall",
        "record_macro_acc",
        "record_macro_supported_f1",
        "bad_containing_record_bad_recall_mean",
        "artifact_positive_nonbad_bad_fpr",
    ]
    rows = []
    for (candidate, bucket), sub in metrics.groupby(["candidate", "bucket"], dropna=False):
        row = {"candidate": candidate, "bucket": bucket, "runs": int(len(sub))}
        for col in numeric_cols:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce")
                row[col] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, metrics_path: Path, records_path: Path, recovery_path: Path, joint_protocol: Path) -> None:
    key = summary[summary["bucket"].isin(["joint_test", "ptb_test", "but_test", "ptb_all", "but_all"])].copy()
    cols = [
        "candidate",
        "bucket",
        "runs",
        "acc",
        "macro_f1",
        "good_recall",
        "medium_recall",
        "bad_recall",
        "record_macro_supported_f1",
        "artifact_positive_nonbad_bad_fpr",
    ]
    cols = [c for c in cols if c in key.columns]
    lines = [
        "# Joint PTB+BUT Interpretable Waveform Model",
        "",
        f"- Generated: {now()}",
        "- Scope: external-only experiment; no `src/sqi_pipeline` changes.",
        "- Training protocol: PTB v11 style replay + clean BUT train/val + controlled PTB bad morphology extras.",
        "- Formal model input remains waveform-derived channels only.",
        "- SQI/factor columns are teacher targets and diagnostics only.",
        "",
        "## Protocol",
        "",
        f"- Joint protocol: `{joint_protocol}`",
        f"- PTB eval protocol: `{DEFAULT_PTB_PROTOCOL}`",
        f"- BUT eval protocol: `{DEFAULT_BUT_PROTOCOL}`",
        "",
        "## Held-Out Results",
        "",
        md_table(key[cols].sort_values(["bucket", "acc"], ascending=[True, False])),
        "",
        "## Output Files",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Summary: `{JOINT_OUT / 'joint_ptb_but_summary.csv'}`",
        f"- Record metrics: `{records_path}`",
        f"- Feature recovery: `{recovery_path}`",
        f"- Checkpoints/logs: `{JOINT_RUN_DIR}`",
    ]
    JOINT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    joint_protocol = materialize_joint_protocol(args)
    seeds = [int(args.seed) + 1000 * i for i in range(int(args.seeds))]
    candidates = []
    for stage, candidate in BASE.TOP3:
        if args.candidates and candidate not in set(args.candidates.split(",")):
            continue
        candidates.append((stage, candidate))

    metric_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        for stage, candidate in candidates:
            cfg = EVENT.candidate_grid(stage, int(seed))[candidate]
            cfg["stage"] = stage
            res = train_joint(candidate, dict(cfg), joint_protocol, args, seed_index)
            metric_rows.extend(res["metrics"])
            record_rows.extend(res["records"])
            recovery_rows.extend(res["recovery"])
            metrics = pd.DataFrame(metric_rows)
            records = pd.DataFrame(record_rows)
            recovery = pd.DataFrame(recovery_rows)
            metrics_path = JOINT_OUT / "joint_ptb_but_metrics.csv"
            records_path = JOINT_OUT / "joint_ptb_but_record_metrics.csv"
            recovery_path = JOINT_OUT / "joint_ptb_but_feature_recovery.csv"
            metrics.to_csv(metrics_path, index=False)
            records.to_csv(records_path, index=False)
            recovery.to_csv(recovery_path, index=False)
            summary = summarize(metrics)
            summary.to_csv(JOINT_OUT / "joint_ptb_but_summary.csv", index=False)
            write_report(summary, metrics_path, records_path, recovery_path, joint_protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="run", choices=["materialize_protocol", "run"])
    parser.add_argument("--force-protocol", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--record-balanced-sampler", action="store_true", default=True)
    parser.add_argument("--bad-extra-per-source", type=int, default=1200)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--max-eval-all-rows", type=int, default=0)
    parser.add_argument("--candidates", type=str, default="E1_query_only,E2_query_highres")
    args = parser.parse_args()
    ensure_dirs()
    if args.stage == "materialize_protocol":
        print(materialize_joint_protocol(args))
    else:
        run(args)
        print(JOINT_REPORT)


if __name__ == "__main__":
    main()
