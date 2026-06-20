"""Cross-dataset test for the top Event-Factorized SQI Conformer candidates.

This is an external-only experiment helper.  It does not modify mainline SQI
pipeline code and it does not warm-start from any previous checkpoint.

Directions:
- PTB block synthetic train/val/test -> clean BUT report-only test.
- Clean BUT train/val/test -> PTB block synthetic report-only test.

Inference input remains waveform-derived channels only.  Interpretable
waveform-computable factors are training teacher targets and diagnostics.
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
GEOM_SCRIPT = ANALYSIS_DIR / "uformer_geometry_branch_experiment.py"

PTB_BANK_DIR = ANALYSIS_DIR / "ptb_combo_boundary_bank"
PTB_BANK_STEM = "ptb_combo_balanced2500_block_corebad_ctl10_gm_keepout_blocks_v3"
DEFAULT_BUT_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_drop_outlier"
CROSS_OUT = ANALYSIS_DIR / "event_xds_top3"
CROSS_REPORT = REPORT_DIR / "event_factorized_sqi_conformer" / "event_factorized_cross_dataset_top3_report.md"
CROSS_RUN_DIR = OUT_ROOT / "runs" / "event_xds_top3"

TOP3 = [
    ("phase1", "E1_query_only"),
    ("phase1", "E0_noquery_nohi_nolocal_noart"),
    ("phase1", "E2_query_highres"),
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVENT = load_module(EVENT_SCRIPT, "event_factorized_for_cross_dataset")
GEOM_FEATURES = load_module(GEOM_SCRIPT, "uformer_geometry_for_cross_dataset")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    CROSS_OUT.mkdir(parents=True, exist_ok=True)
    CROSS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CROSS_RUN_DIR.mkdir(parents=True, exist_ok=True)


def split_groups_stratified(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Assign train/val/test by source_idx groups, with class balance as far as possible."""

    rng = np.random.default_rng(int(seed))
    groups = frame[["source_idx", "y_class"]].copy()
    groups["source_idx"] = groups["source_idx"].astype(str)
    # A source can be replayed into multiple labels; place all its rows together
    # to avoid clean-source leakage across PTB train/test.
    source_major = (
        groups.groupby("source_idx")["y_class"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
        .rename(columns={"y_class": "major_class"})
    )
    group_split: dict[str, str] = {}
    for cls in ["good", "medium", "bad"]:
        keys = source_major.loc[source_major["major_class"].eq(cls), "source_idx"].to_numpy()
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        for key in keys[:n_train]:
            group_split[str(key)] = "train"
        for key in keys[n_train : n_train + n_val]:
            group_split[str(key)] = "val"
        for key in keys[n_train + n_val :]:
            group_split[str(key)] = "test"
    return frame["source_idx"].astype(str).map(group_split).fillna("train").to_numpy()


def materialize_ptb_protocol(seed: int = 20260620, force: bool = False) -> Path:
    ensure_dirs()
    protocol = CROSS_OUT / f"protocol_ptb_bal2500_blocks_v3_s{seed}"
    atlas_path = protocol / "original_region_atlas.csv"
    signals_path = protocol / "signals.npz"
    if atlas_path.exists() and signals_path.exists() and not force:
        return protocol

    protocol.mkdir(parents=True, exist_ok=True)
    manifest_path = PTB_BANK_DIR / f"{PTB_BANK_STEM}_manifest.csv"
    bank_signals_path = PTB_BANK_DIR / f"{PTB_BANK_STEM}_signals.npz"
    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    x = np.load(bank_signals_path)["X"].astype(np.float32)
    if x.ndim == 2:
        x_save = x[:, None, :]
    elif x.ndim == 3:
        x_save = x
        x = x[:, 0, :]
    else:
        raise ValueError(f"unexpected PTB signal shape {x.shape}")

    if "source_idx" not in manifest.columns:
        manifest["source_idx"] = np.arange(len(manifest))
    manifest["split"] = split_groups_stratified(manifest, seed)
    y = manifest["y_class"].astype(str).map(EVENT.CLASS_TO_INT).to_numpy(dtype=np.int64)
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()

    primitives = GEOM_FEATURES.compute_primitives(x)
    feature_state = GEOM_FEATURES.fit_feature_state(primitives, y, train_mask)
    features = GEOM_FEATURES.assemble_features(primitives, feature_state)

    atlas = pd.DataFrame(
        {
            "idx": np.arange(len(manifest), dtype=np.int64),
            "source_idx": manifest["source_idx"].astype(int).to_numpy(),
            "split": manifest["split"].astype(str).to_numpy(),
            "y": y,
            "class_name": manifest["y_class"].astype(str).to_numpy(),
            "record_id": ["ptb_src_" + str(v) for v in manifest["source_idx"].astype(str).to_numpy()],
            "subject_id": ["ptb_src_" + str(v) for v in manifest["source_idx"].astype(str).to_numpy()],
            "dataset": "ptb_synthetic_block",
            "original_region": manifest.get("profile_block", manifest.get("target_original_region", pd.Series(["ptb_block"] * len(manifest)))).fillna("ptb_block").astype(str).to_numpy(),
            "bank_role": manifest.get("bank_role", pd.Series([""] * len(manifest))).fillna("").astype(str).to_numpy(),
            "profile_block": manifest.get("profile_block", pd.Series([""] * len(manifest))).fillna("").astype(str).to_numpy(),
            "source_bank_manifest": str(manifest_path),
        }
    )
    for col in EVENT.FORMAL_FEATURE_COLUMNS:
        if col in features.columns:
            atlas[col] = pd.to_numeric(features[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        elif col in primitives.columns:
            atlas[col] = pd.to_numeric(primitives[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        else:
            atlas[col] = 0.0

    np.savez_compressed(signals_path, X=x_save.astype(np.float32))
    atlas.to_csv(atlas_path, index=False)
    manifest.to_csv(protocol / "metadata.csv", index=False)
    split_counts = atlas.groupby(["split", "class_name"]).size().reset_index(name="n")
    split_counts.to_csv(protocol / "split_counts.csv", index=False)
    summary = {
        "created_at": now(),
        "source_manifest": str(manifest_path),
        "source_signals": str(bank_signals_path),
        "split_seed": int(seed),
        "n": int(len(atlas)),
        "class_counts": atlas["class_name"].value_counts().to_dict(),
        "split_counts": split_counts.to_dict(orient="records"),
        "split_rule": "source_idx-grouped 70/15/15 stratified by majority class",
        "feature_note": "waveform-computable primitives plus train-split PCA/atlas diagnostics; formal model input remains waveform channels only",
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


def train_cross(
    direction: str,
    candidate: str,
    cfg: dict[str, Any],
    train_protocol: Path,
    test_protocol: Path,
    args: argparse.Namespace,
    seed_index: int,
) -> dict[str, Any]:
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = make_dataset(train_protocol, "train", max_rows=int(args.max_train_rows), seed=seed)
    val_ds = make_dataset(train_protocol, "val", ref_ds=train_ds, max_rows=int(args.max_val_rows), seed=seed + 1)
    train_test_ds = make_dataset(train_protocol, "test", ref_ds=train_ds, max_rows=int(args.max_test_rows), seed=seed + 2)
    cross_test_ds = make_dataset(test_protocol, "test", ref_ds=train_ds, max_rows=int(args.max_cross_rows), seed=seed + 3)
    cross_all_ds = make_dataset(test_protocol, "test", ref_ds=train_ds, max_rows=int(args.max_cross_all_rows), seed=seed + 4, all_rows=True)

    sampler = EVENT.record_balanced_sampler(train_ds, seed) if bool(args.record_balanced_sampler) else None
    train_loader = loader(train_ds, int(args.batch_size), shuffle=True, sampler=sampler, num_workers=int(args.num_workers))
    val_loader = loader(val_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))
    train_test_loader = loader(train_test_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))
    cross_loader = loader(cross_test_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))
    cross_all_loader = loader(cross_all_ds, int(args.batch_size) * 2, num_workers=int(args.num_workers))

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
        score = float(val_rep["record_macro_supported_f1"]) + 0.20 * min(float(val_rep["good_recall"]), float(val_rep["medium_recall"]), float(val_rep["bad_recall"])) - 0.05 * float(val_rep["bad_fpr_nonbad"])
        part_mean = pd.DataFrame(parts_rows).mean(numeric_only=True).to_dict() if parts_rows else {}
        row = {"phase": "finetune", "epoch": epoch, "train_loss": float(np.mean(losses)), **{f"train_{k}": v for k, v in part_mean.items()}, **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}}
        logs.append(row)
        print(
            f"{direction} {candidate} seed={seed_index} epoch={epoch}/{args.epochs} "
            f"loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} "
            f"val_rec={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    buckets = []
    records = []
    recovery = []
    preds = {}
    for bucket_name, eval_ds, eval_loader_obj in [
        ("source_val", val_ds, val_loader),
        ("source_test", train_test_ds, train_test_loader),
        ("cross_test", cross_test_ds, cross_loader),
        ("cross_all", cross_all_ds, cross_all_loader),
    ]:
        rep, probs, factor_pred, factor_true, y, artifact, rec_rows, _ = EVENT.eval_loader(model, eval_loader_obj, device, candidate)
        row = EVENT.metric_row(candidate, bucket_name, rep)
        row["direction"] = direction
        row["seed_index"] = int(seed_index)
        row["stage"] = str(cfg.get("stage", "phase1"))
        row["n_rows"] = int(len(eval_ds))
        buckets.append(row)
        for rec in rec_rows:
            rec = dict(rec)
            rec["direction"] = direction
            rec["bucket"] = bucket_name
            rec["seed_index"] = int(seed_index)
            records.append(rec)
        for frec in EVENT.factor_recovery_rows(candidate, bucket_name, factor_pred, factor_true, y):
            frec["direction"] = direction
            frec["seed_index"] = int(seed_index)
            recovery.append(frec)
        preds[bucket_name] = {"probs": probs, "y": y, "artifact_presence": artifact}

    run_path = CROSS_RUN_DIR / f"{direction}_{candidate}_seed{seed_index}"
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "factor_columns": EVENT.FACTOR_COLUMNS,
            "local_map_names": EVENT.LOCAL_MAP_NAMES,
            "input_contract": "waveform-derived channels only; no tabular feature input at inference",
            "model_kind": "EventFactorizedSQIConformer",
            "direction": direction,
            "train_protocol": str(train_protocol),
            "test_protocol": str(test_protocol),
            "no_warm_start": True,
        },
        run_path / "ckpt_best.pt",
    )
    pd.DataFrame(pre_logs + logs).to_csv(run_path / "train_log.csv", index=False)
    for bucket_name, payload in preds.items():
        np.savez_compressed(run_path / f"{bucket_name}_predictions.npz", **payload)
    return {"metrics": buckets, "records": records, "recovery": recovery, "run_path": str(run_path)}


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
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
    for (direction, candidate, bucket), sub in metrics.groupby(["direction", "candidate", "bucket"], dropna=False):
        row = {"direction": direction, "candidate": candidate, "bucket": bucket, "runs": int(len(sub))}
        for col in numeric_cols:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce")
                row[col] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, metrics_path: Path, records_path: Path, recovery_path: Path, ptb_protocol: Path, but_protocol: Path) -> None:
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
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    key = summary.loc[summary["bucket"].isin(["cross_test", "cross_all"])].copy()
    cols = [
        "direction",
        "candidate",
        "bucket",
        "runs",
        "acc",
        "macro_f1",
        "good_recall",
        "medium_recall",
        "bad_recall",
        "record_macro_supported_f1",
    ]
    cols = [c for c in cols if c in key.columns]
    report = [
        "# Event-Factorized Cross-Dataset Top-3 Report",
        "",
        f"- Generated: {now()}",
        "- Scope: external-only experiment; no `src/sqi_pipeline` changes.",
        "- Formal input: waveform-derived channels only.",
        "- Factor/SQI targets: training teacher/diagnostic only, not inference input.",
        "- Top-3 definition: pooled clean BUT internal ranking from Event-Factorized full report: `E1`, `E0`, `E2`.",
        "",
        "## Protocols",
        "",
        f"- PTB synthetic protocol: `{ptb_protocol}`",
        f"- Clean BUT protocol: `{but_protocol}`",
        "",
        "## Cross-Dataset Summary",
        "",
        md_table(key[cols].sort_values(["direction", "bucket", "acc"], ascending=[True, True, False])),
        "",
        "## Output Files",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Summary: `{CROSS_OUT / 'cross_dataset_top3_summary.csv'}`",
        f"- Record metrics: `{records_path}`",
        f"- Feature recovery: `{recovery_path}`",
        f"- Checkpoints/logs: `{CROSS_RUN_DIR}`",
    ]
    CROSS_REPORT.write_text("\n".join(report), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    ptb_protocol = materialize_ptb_protocol(seed=int(args.ptb_split_seed), force=bool(args.force_protocol))
    but_protocol = Path(args.but_protocol)
    seeds = [int(args.seed) + 1000 * i for i in range(int(args.seeds))]
    metric_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        for stage, candidate in TOP3:
            cfg = EVENT.candidate_grid(stage, int(seed))[candidate]
            cfg["stage"] = stage
            for direction, train_protocol, test_protocol in [
                ("ptb_to_but", ptb_protocol, but_protocol),
                ("but_to_ptb", but_protocol, ptb_protocol),
            ]:
                res = train_cross(direction, candidate, dict(cfg), train_protocol, test_protocol, args, seed_index)
                metric_rows.extend(res["metrics"])
                record_rows.extend(res["records"])
                recovery_rows.extend(res["recovery"])
                metrics_df = pd.DataFrame(metric_rows)
                records_df = pd.DataFrame(record_rows)
                recovery_df = pd.DataFrame(recovery_rows)
                metrics_path = CROSS_OUT / "cross_dataset_top3_metrics.csv"
                records_path = CROSS_OUT / "cross_dataset_top3_record_metrics.csv"
                recovery_path = CROSS_OUT / "cross_dataset_top3_feature_recovery.csv"
                metrics_df.to_csv(metrics_path, index=False)
                records_df.to_csv(records_path, index=False)
                recovery_df.to_csv(recovery_path, index=False)
                summary = summarize_metrics(metrics_df)
                summary.to_csv(CROSS_OUT / "cross_dataset_top3_summary.csv", index=False)
                write_report(summary, metrics_path, records_path, recovery_path, ptb_protocol, but_protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="run", choices=["materialize_protocol", "run"])
    parser.add_argument("--ptb-split-seed", type=int, default=20260620)
    parser.add_argument("--but-protocol", type=str, default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--force-protocol", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--record-balanced-sampler", action="store_true", default=True)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--max-cross-rows", type=int, default=0)
    parser.add_argument("--max-cross-all-rows", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()
    if args.stage == "materialize_protocol":
        print(materialize_ptb_protocol(seed=int(args.ptb_split_seed), force=bool(args.force_protocol)))
    else:
        run(args)
        print(CROSS_REPORT)


if __name__ == "__main__":
    main()
