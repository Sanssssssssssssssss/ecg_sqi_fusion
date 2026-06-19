"""Interpretable fixed-10s PTB/BUT cross-dataset Transformer checks.

The model remains waveform-only at inference.  Auxiliary targets are limited to
waveform-computable SQI/QRS/baseline/detail features.  This script compares:

* PTB target-aware synthetic train -> PTB synthetic test + curated clean BUT.
* Clean BUT train -> clean BUT test + PTB synthetic test.

No variable-length inputs are used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
DUAL_SCRIPT = ANALYSIS_DIR / "run_clean_but_dualview_hier_transformer.py"
ARCH_SCRIPT = ANALYSIS_DIR / "run_waveform_architecture_search.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DUAL = load_module("clean_dualview_hier_for_cross", DUAL_SCRIPT)
ARCH = load_module("wave_arch_for_cross", ARCH_SCRIPT)
GEOM = DUAL.GEOM

CLASS_TO_INT = DUAL.CLASS_TO_INT
FORMAL_FEATURE_COLUMNS = list(DUAL.FORMAL_FEATURE_COLUMNS)
FORMAL_FEATURE_IDX = [ARCH.FEATURE_COLUMNS.index(c) for c in FORMAL_FEATURE_COLUMNS]


class SyntheticDualviewDataset(Dataset):
    def __init__(
        self,
        split: str,
        channel_stats: Any | None = None,
        feature_norm: Any | None = None,
    ) -> None:
        labels = pd.read_csv(ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
        mask = labels["split"].astype(str).eq(split).to_numpy()
        self.frame = labels.loc[mask].reset_index(drop=True)
        rows = self.frame["idx"].to_numpy(dtype=np.int64)
        signals = np.load(ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        feat_npz = np.load(ARCH.DATA_DIR / "tabular_features.npz")
        features = feat_npz["features"].astype(np.float32)[:, FORMAL_FEATURE_IDX]
        if channel_stats is None:
            train_rows = labels.loc[labels["split"].astype(str).eq("train"), "idx"].to_numpy(dtype=np.int64)
            channel_stats = DUAL.compute_channel_stats(signals[train_rows])
        if feature_norm is None:
            train_rows = labels.loc[labels["split"].astype(str).eq("train"), "idx"].to_numpy(dtype=np.int64)
            train_raw = features[train_rows]
            mean = train_raw.mean(axis=0).astype(np.float32)
            std = train_raw.std(axis=0).astype(np.float32)
            std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
            feature_norm = DUAL.FeatureNorm(mean=mean, std=std)
        self.channel_stats = channel_stats
        self.feature_norm = feature_norm
        self.x = DUAL.make_dualview_channels(signals[rows], channel_stats)
        self.aux = ((features[rows] - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.y = self.frame["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.split = self.frame["split"].astype(str).to_numpy()
        self.region = self.frame.get("block", pd.Series("synthetic", index=self.frame.index)).astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


@torch.no_grad()
def eval_dataset(model: torch.nn.Module, ds: Dataset, batch_size: int, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    rep, probs, _, _ = DUAL.eval_loader(model, loader, device)
    return rep, probs


def bucket_report(ds: Any, probs: np.ndarray, bucket: str) -> dict[str, Any]:
    if bucket == "all":
        mask = np.ones(len(ds.y), dtype=bool)
    elif bucket == "test":
        mask = ds.split == "test"
    elif bucket == "bad_core_nearboundary":
        mask = (ds.split == "test") & (ds.y == CLASS_TO_INT["bad"]) & (ds.region != "outlier_low_confidence")
    elif bucket == "bad_outlier_stress":
        mask = (ds.split == "test") & (ds.y == CLASS_TO_INT["bad"]) & (ds.region == "outlier_low_confidence")
    else:
        raise ValueError(bucket)
    return GEOM.metric_report(ds.y[mask], probs.argmax(axis=1)[mask], probs[mask])


def train_domain(run_name: str, train_domain: str, policy: str, cfg: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"cross train {run_name}: domain={train_domain} policy={policy}", flush=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_domain == "ptb_synthetic":
        train_ds = SyntheticDualviewDataset("train")
        val_ds = SyntheticDualviewDataset("val", train_ds.channel_stats, train_ds.feature_norm)
        ptb_test = SyntheticDualviewDataset("test", train_ds.channel_stats, train_ds.feature_norm)
        clean_test = DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, "test", train_ds.channel_stats, train_ds.feature_norm)
        full_but = DUAL.FullButDataset(train_ds.channel_stats, train_ds.feature_norm) if bool(args.include_full_diagnostic) else None
    elif train_domain == "but_clean":
        train_ds = DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, "train")
        val_ds = DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, "val", train_ds.channel_stats, train_ds.feature_norm)
        clean_test = DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, "test", train_ds.channel_stats, train_ds.feature_norm)
        ptb_test = SyntheticDualviewDataset("test", train_ds.channel_stats, train_ds.feature_norm)
        full_but = DUAL.FullButDataset(train_ds.channel_stats, train_ds.feature_norm) if bool(args.include_full_diagnostic) else None
    else:
        raise ValueError(train_domain)
    model = DUAL.HierTransformer(cfg, int(train_ds.x.shape[1]), len(FORMAL_FEATURE_COLUMNS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -1e9
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(x)
            loss = DUAL.classification_losses(out, y, aux, cfg, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, _, _, _ = DUAL.eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.15 * val_rep["macro_f1"] + 0.10 * min(val_rep["good_recall"], val_rep["medium_recall"]) - 0.02 * val_rep["aux_mae"]
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"{run_name} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_rep, _ = eval_dataset(model, val_ds, args.batch_size, device)
    ptb_rep, ptb_probs = eval_dataset(model, ptb_test, args.batch_size, device)
    clean_rep, clean_probs = eval_dataset(model, clean_test, args.batch_size, device)
    full_probs = None
    if full_but is not None:
        _, full_probs = eval_dataset(model, full_but, args.batch_size, device)
    rows = [
        metric_row(run_name, "train_domain_val", val_rep),
        metric_row(run_name, "ptb_synthetic_test", ptb_rep),
        metric_row(run_name, f"but_clean_{policy}_test", clean_rep),
    ]
    if full_but is not None and full_probs is not None:
        rows.extend(
            [
                metric_row(run_name, "legacy_full_test_all_10s+", bucket_report(full_but, full_probs, "test")),
                metric_row(run_name, "legacy_full_all_10s+", bucket_report(full_but, full_probs, "all")),
                metric_row(run_name, "legacy_full_bad_core_nearboundary", bucket_report(full_but, full_probs, "bad_core_nearboundary")),
                metric_row(run_name, "legacy_full_bad_outlier_stress", bucket_report(full_but, full_probs, "bad_outlier_stress")),
            ]
        )
    run_dir = DUAL.OUT_ROOT / "runs" / "interpretable_clean_cross_dataset" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "fixed_10s_interpretable_cross_dataset_transformer",
            "train_domain": train_domain,
            "policy": policy,
            "input_contract": "fixed 10s waveform-derived channels only; no feature input at inference",
            "formal_feature_columns": FORMAL_FEATURE_COLUMNS,
            "best_epoch": best_epoch,
            "channel_stats": train_ds.channel_stats.__dict__,
            "feature_train_mean": train_ds.feature_norm.mean,
            "feature_train_std": train_ds.feature_norm.std,
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "ptb_test_probs.npz", probs=ptb_probs)
    np.savez_compressed(run_dir / "but_clean_test_probs.npz", probs=clean_probs)
    if full_probs is not None:
        np.savez_compressed(run_dir / "legacy_full_but_probs.npz", probs=full_probs)
    return {"candidate": run_name, "checkpoint": str(ckpt_path), "best_epoch": best_epoch, "train_domain": train_domain, "policy": policy}, rows


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Interpretable Clean PTB/BUT Cross-Dataset Checks",
        "",
        "Fixed 10s only. No variable-length inputs. Classifier input is waveform-derived channels; interpretable features supervise training only.",
        "",
        "## Metrics",
        "",
        "| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} |"
        )
    lines.extend(["", "## Checkpoints", ""])
    for item in payload["summaries"]:
        lines.append(f"- `{item['candidate']}` ({item['train_domain']}): best_epoch={item['best_epoch']}, checkpoint=`{item['checkpoint']}`")
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "- `ptb_synthetic_test` measures target-aware PTB synthetic learnability.",
            "- `but_clean_*_test` measures cleaned fixed-10s BUT learnability.",
            "- `legacy_full_*` buckets are emitted only when explicitly requested and are not the modeling target.",
            "- Large asymmetry between within-domain and cross-domain performance is evidence of domain/split/label-policy shift, not proof that variable-length is required.",
            "",
            f"Metrics CSV: `{payload['metrics_csv']}`",
        ]
    )
    return "\n".join(lines)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="margin_ge_5s_drop_outlier")
    parser.add_argument("--train-domains", type=str, default="ptb_synthetic,but_clean")
    parser.add_argument("--candidate", type=str, default="dualview_convtx_hier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-full-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.candidate not in DUAL.CANDIDATES:
        raise ValueError(f"unknown candidate {args.candidate}; expected {sorted(DUAL.CANDIDATES)}")
    out_dir = ANALYSIS_DIR / "interpretable_clean_cross_dataset"
    report_dir = REPORT_DIR / "interpretable_clean_cross_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for domain in [d.strip() for d in args.train_domains.split(",") if d.strip()]:
        run_name = f"{domain}_{args.policy}_{args.candidate}"
        summary, metric_rows = train_domain(run_name, domain, args.policy, dict(DUAL.CANDIDATES[args.candidate]), args)
        summaries.append(summary)
        rows.extend(metric_rows)
    metrics = pd.DataFrame(rows)
    stem = f"interpretable_clean_cross_dataset_{args.policy}_{args.candidate}"
    metrics_path = out_dir / f"{stem}_metrics.csv"
    summary_path = out_dir / f"{stem}_summary.json"
    report_path = out_dir / f"{stem}_report.md"
    report_copy = report_dir / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": args.policy,
        "candidate": args.candidate,
        "epochs": int(args.epochs),
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
