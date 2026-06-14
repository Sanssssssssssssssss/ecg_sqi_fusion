"""Train UFormer + reduced SQI/geometry feature candidates.

This is the counterpart to the reduced-feature tabular MLP experiment: the
UFormer waveform feature extractor is kept, but the tabular branch only sees a
small selected feature subset. Training and model selection use node train/val
only; held-out BUT test buckets are report-only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
GEOM_SCRIPT = ANALYSIS_DIR / "uformer_geometry_branch_experiment.py"
NODE_ID = "N17043_gm_probe"
VARIANT_ID = "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
MANIFEST_PATH = OUT_ROOT / "nodes" / NODE_ID / "node_boundary_manifest.csv"
SIGNALS_PATH = ROOT / "outputs" / "external_benchmarks" / "e311_but_protocol_adaptation_2026_06_03" / "protocols" / "p1_current_10s_center" / "signals.npz"
BASE_CKPT = OUT_ROOT / "runs" / "quick" / VARIANT_ID / "stage1_denoiser_pretrain" / "ckpt_best_stage1_denoiser.pt"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
LABELS = [0, 1, 2]
CLASS_WEIGHT = [1.03, 1.22, 2.05]
SEED = 20260614

AUTO_TOP20 = [
    "pca_margin",
    "sample_entropy_proxy",
    "flatline_ratio",
    "pc1",
    "higuchi_fd_proxy",
    "non_qrs_diff_p95",
    "diff_zero_crossing_rate",
    "zero_crossing_rate",
    "qrs_prom_p90",
    "sqi_fSQI",
    "amplitude_entropy",
    "sqi_kSQI",
    "sqi_sSQI",
    "low_amp_ratio",
    "non_qrs_rms_ratio",
    "baseline_step",
    "ptp_p99_p01",
    "band_30_45",
    "qrs_visibility",
    "band_15_30",
]
RANKED_TOP22 = AUTO_TOP20 + ["hjorth_mobility", "wavelet_e4"]
TOP14_BALANCED = [
    "pc1",
    "qrs_visibility",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "pca_margin",
    "qrs_band_ratio",
    "template_corr",
    "amplitude_entropy",
    "sqi_bSQI",
    "region_confidence",
    "detector_agreement",
    "mean_abs",
    "sqi_basSQI",
]

CANDIDATES = {
    "uformer_top20": AUTO_TOP20,
    "uformer_top22": RANKED_TOP22,
    "uformer_top14": TOP14_BALANCED,
}

REFERENCE_47 = {
    "name": "47-feature current best, threshold p_bad>=0.13",
    "n_features": 47,
    "acc": 0.963548425150407,
    "macro_f1": 0.9306830954417679,
    "good_recall": 0.9563186813186814,
    "medium_recall": 0.9728874830546769,
    "bad_recall": 0.927007299270073,
}


def load_geom_module() -> Any:
    spec = importlib.util.spec_from_file_location("uformer_geometry_branch_experiment", GEOM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GEOM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEOM = load_geom_module()


@dataclass
class FeatureNorm:
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class ReducedUformerDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, signals: np.ndarray, split: str, norm: FeatureNorm | None, columns: list[str]):
        if split == "all":
            mask = np.ones(len(manifest), dtype=bool)
        else:
            mask = manifest["split"].astype(str).eq(split).to_numpy()
        self.frame = manifest.loc[mask].reset_index(drop=True)
        idx = self.frame["idx"].to_numpy(dtype=np.int64)
        self.signals = signals[idx].astype(np.float32)
        if self.signals.ndim == 2:
            self.signals = self.signals[:, None, :]
        raw = self.frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        if norm is None:
            raise ValueError("norm is required")
        raw = np.where(np.isfinite(raw), raw, norm.median[None, :]).astype(np.float32)
        self.features = ((raw - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32)
        self.y = self.frame["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "noisy": torch.from_numpy(self.signals[i]),
            "tabular": torch.from_numpy(self.features[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifest = pd.read_csv(MANIFEST_PATH)
    for name, cols in CANDIDATES.items():
        missing = sorted(set(cols) - set(manifest.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    signals = np.load(SIGNALS_PATH)["X"].astype(np.float32)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, cols in CANDIDATES.items():
        summary = train_candidate(name, cols, manifest, signals, epochs=10, batch_size=128)
        summaries.append(summary)
        rows.extend(summary["metrics"])
    metrics = pd.DataFrame(rows)
    metrics_path = ANALYSIS_DIR / "current_reduced_uformer_metrics.csv"
    summary_path = ANALYSIS_DIR / "current_reduced_uformer_summary.json"
    report_out = ANALYSIS_DIR / "current_reduced_uformer_report.md"
    report_copy = REPORT_DIR / "current_reduced_uformer_report.md"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "node_id": NODE_ID,
        "variant_id": VARIANT_ID,
        "training_source": "node train only; val for model selection; test report-only",
        "original_used_for_training": False,
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "reference_47": REFERENCE_47,
        "candidates": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(metrics, payload)
    report_out.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


def train_candidate(name: str, columns: list[str], manifest: pd.DataFrame, signals: np.ndarray, epochs: int, batch_size: int) -> dict[str, Any]:
    print(f"training {name} features={len(columns)}", flush=True)
    norm = fit_norm(manifest, columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ReducedUformerDataset(manifest, signals, "train", norm, columns)
    val_ds = ReducedUformerDataset(manifest, signals, "val", norm, columns)
    test_ds = ReducedUformerDataset(manifest, signals, "test", norm, columns)
    all_ds = ReducedUformerDataset(manifest, signals, "all", norm, columns)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    all_loader = DataLoader(all_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    base_cfg = GEOM.load_base_config(BASE_CKPT)
    cfg = GEOM.Uformer1DConfig(
        noise_scale=base_cfg.noise_scale,
        dropout=0.08,
        feature_set=base_cfg.feature_set,
        detach_encoder_features=base_cfg.detach_encoder_features,
        head_hidden_dim=base_cfg.head_hidden_dim,
        denoiser_width=base_cfg.denoiser_width,
    )
    model = GEOM.UformerGeometryBranch(
        cfg,
        feature_dim=len(columns),
        fusion_mode="waveform_tabular",
    ).to(device)
    load_report = GEOM.load_denoiser(model, BASE_CKPT, device)
    model.attach_head(device=device)
    for p in model.backbone.denoiser.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(
        [
            {"params": model.tabular_branch.parameters(), "lr": 7e-4},
            {"params": model.head.parameters(), "lr": 7e-4},
        ],
        weight_decay=1e-4,
    )
    class_weight = torch.tensor(CLASS_WEIGHT, dtype=torch.float32, device=device)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    train_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            noisy = batch["noisy"].to(device=device, dtype=torch.float32)
            tab = batch["tabular"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            logits = model(noisy, tab)["logits"]
            loss = F.cross_entropy(logits, y, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rep, _ = eval_loader(model, val_loader, device)
        score = val_rep["acc"] + 0.25 * val_rep["macro_f1"] + 0.15 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
        train_log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(f"{name} epoch={epoch}/{epochs} loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}", flush=True)
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_rep, val_probs = eval_loader(model, val_loader, device)
    test_rep, test_probs = eval_loader(model, test_loader, device)
    all_rep, all_probs = eval_loader(model, all_loader, device)
    threshold_info = calibrate_bad_threshold(model, manifest, signals, columns, norm, batch_size, device)

    metrics = [
        metric_row(name, "val", val_rep),
        metric_row(name, "original_test_all_10s+", test_rep),
        metric_row(name, "original_all_10s+", all_rep),
    ]
    test_frame = test_ds.frame
    test_y = test_ds.y
    test_pred = np.argmax(test_probs, axis=1)
    test_pred_badcal = apply_bad_threshold(test_probs, threshold_info["threshold"])
    all_pred_badcal = apply_bad_threshold(all_probs, threshold_info["threshold"])
    metrics.extend(bucket_rows(name, test_frame, test_y, test_pred))
    metrics.append(metric_row(f"{name}_badcal", "val", metric_report(val_ds.y, apply_bad_threshold(val_probs, threshold_info["threshold"]))))
    metrics.append(metric_row(f"{name}_badcal", "original_test_all_10s+", metric_report(test_y, test_pred_badcal)))
    metrics.append(metric_row(f"{name}_badcal", "original_all_10s+", metric_report(all_ds.y, all_pred_badcal)))
    metrics.extend(bucket_rows(f"{name}_badcal", test_frame, test_y, test_pred_badcal))

    run_dir = OUT_ROOT / "runs" / "reduced_uformer_nodecal" / NODE_ID / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "denoiser_state": model.backbone.denoiser.state_dict(),
            "config": cfg.__dict__,
            "feature_columns": columns,
            "feature_train_median": norm.median,
            "feature_train_mean": norm.mean,
            "feature_train_std": norm.std,
            "feature_source": "node_manifest_reduced_features",
            "branch_config": {"tab_hidden": 64, "head_hidden": 128, "fusion_mode": "waveform_tabular", "candidate": name},
            "class_weight": CLASS_WEIGHT,
            "base_variant": VARIANT_ID,
            "best_epoch": best_epoch,
            "load_report": load_report,
            "bad_threshold_ablation": threshold_info,
            "original_rows_used_for_training": False,
        },
        ckpt_path,
    )
    pd.DataFrame(train_log).to_csv(run_dir / "train_log.csv", index=False)
    (run_dir / "normalization.json").write_text(
        json.dumps(
            {
                "feature_columns": columns,
                "median": norm.median.tolist(),
                "mean": norm.mean.tolist(),
                "std": norm.std.tolist(),
                "source": "node train split only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "test_probs.npz", probs=test_probs)
    return {
        "candidate_name": name,
        "feature_columns": columns,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "val": val_rep,
        "original_test_all_10s+": test_rep,
        "original_all_10s+": all_rep,
        "bad_threshold_ablation": threshold_info,
        "metrics": metrics,
    }


def fit_norm(manifest: pd.DataFrame, columns: list[str]) -> FeatureNorm:
    train = manifest[manifest["split"].astype(str).eq("train")]
    raw = train[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    median = np.nanmedian(raw, axis=0).astype(np.float32)
    filled = np.where(np.isfinite(raw), raw, median[None, :]).astype(np.float32)
    mean = np.nanmean(filled, axis=0).astype(np.float32)
    std = np.nanstd(filled, axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return FeatureNorm(columns=columns, median=median, mean=mean, std=std)


@torch.no_grad()
def eval_loader(model: Any, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    ys = []
    probs = []
    for batch in loader:
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        tab = batch["tabular"].to(device=device, dtype=torch.float32)
        logits = model(noisy, tab)["logits"]
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        ys.append(batch["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    return metric_report(y, p.argmax(axis=1)), p


def calibrate_bad_threshold(model: Any, manifest: pd.DataFrame, signals: np.ndarray, columns: list[str], norm: FeatureNorm, batch_size: int, device: torch.device) -> dict[str, Any]:
    trainval = pd.concat(
        [
            manifest[manifest["split"].astype(str).eq("train")],
            manifest[manifest["split"].astype(str).eq("val")],
        ],
        ignore_index=True,
    )
    tmp = manifest.copy()
    tmp["_tmp_trainval"] = False
    mask = manifest["split"].astype(str).isin(["train", "val"]).to_numpy()
    ds = ReducedUformerDataset(manifest, signals, "all", norm, columns)
    probs = predict_dataset(model, ds, batch_size, device)
    y = ds.y
    probs = probs[mask]
    y = y[mask]
    best = None
    for threshold in np.round(np.linspace(0.01, 0.95, 95), 2):
        pred = apply_bad_threshold(probs, float(threshold))
        rep = metric_report(y, pred)
        row = {
            "threshold": float(threshold),
            "acc": rep["acc"],
            "macro_f1": rep["macro_f1"],
            "good_recall": rep["good_recall"],
            "medium_recall": rep["medium_recall"],
            "bad_recall": rep["bad_recall"],
        }
        score = (row["acc"], row["macro_f1"], min(row["good_recall"], row["medium_recall"], row["bad_recall"]))
        if best is None or score > best[0]:
            best = (score, row)
    assert best is not None
    row = best[1]
    row["selection"] = "train+val only"
    return row


@torch.no_grad()
def predict_dataset(model: Any, ds: ReducedUformerDataset, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    _, probs = eval_loader(model, loader, device)
    return probs


def apply_bad_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    pred = probs.argmax(axis=1).astype(np.int64)
    pred[probs[:, CLASS_TO_INT["bad"]] >= float(threshold)] = CLASS_TO_INT["bad"]
    return pred


def bucket_rows(candidate: str, frame: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    region = frame["original_region"].astype(str).to_numpy()
    bad = y == CLASS_TO_INT["bad"]
    outlier = region == "outlier_low_confidence"
    rows = []
    for bucket, mask in [
        ("original_test_without_bad_outlier_stress", ~(bad & outlier)),
        ("bad_core_nearboundary", bad & ~outlier),
        ("bad_outlier_stress", bad & outlier),
    ]:
        rows.append(metric_row(candidate, bucket, metric_report(y[mask], pred[mask])))
    return rows


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        cm = np.zeros((3, 3), dtype=int)
    else:
        cm = GEOM.confusion_matrix(y_true, y_pred, labels=LABELS)
    recalls = []
    precisions = []
    for i in LABELS:
        support = float(cm[i, :].sum())
        pred_n = float(cm[:, i].sum())
        tp = float(cm[i, i])
        recalls.append(tp / support if support else 0.0)
        precisions.append(tp / pred_n if pred_n else 0.0)
    return {
        "n": int(len(y_true)),
        "acc": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(GEOM.f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "good_recall": float(recalls[0]),
        "medium_recall": float(recalls[1]),
        "bad_recall": float(recalls[2]),
        "good_precision": float(precisions[0]),
        "medium_precision": float(precisions[1]),
        "bad_precision": float(precisions[2]),
        "good_to_medium": int(cm[0, 1]),
        "medium_to_good": int(cm[1, 0]),
        "bad_to_medium": int(cm[2, 1]),
        "confusion_3x3": cm.astype(int).tolist(),
    }


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Reduced UFormer + SQI/Geometry Feature Candidates",
        "",
        "Frozen UFormer waveform feature extractor plus a reduced tabular branch. Model selection uses train/val only; held-out BUT test is report-only.",
        "",
        "## Held-Out BUT Test",
        "",
        "| Candidate | n features | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["candidates"]:
        cand = item["candidate_name"]
        row = metrics[(metrics["candidate"] == cand) & (metrics["bucket"] == "original_test_all_10s+")].iloc[0]
        lines.append(
            f"| {cand} | {len(item['feature_columns'])} | {row['acc']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['good_recall']:.6f} | {row['medium_recall']:.6f} | {row['bad_recall']:.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} |"
        )
        badcal = f"{cand}_badcal"
        row = metrics[(metrics["candidate"] == badcal) & (metrics["bucket"] == "original_test_all_10s+")].iloc[0]
        lines.append(
            f"| {badcal} | {len(item['feature_columns'])} | {row['acc']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['good_recall']:.6f} | {row['medium_recall']:.6f} | {row['bad_recall']:.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} |"
        )
    ref = REFERENCE_47
    lines.append(
        f"| {ref['name']} | 47 | {ref['acc']:.6f} | {ref['macro_f1']:.6f} | {ref['good_recall']:.6f} | {ref['medium_recall']:.6f} | {ref['bad_recall']:.6f} | 102 | 76 | 23 |"
    )
    lines.extend(
        [
            "",
            "## Original Buckets",
            "",
            markdown_table(metrics[metrics["bucket"].astype(str).str.startswith(("original", "bad_"))]),
            "",
            "## Candidate Feature Sets",
            "",
        ]
    )
    for item in payload["candidates"]:
        lines.append(f"- `{item['candidate_name']}` ({len(item['feature_columns'])}): " + ", ".join(item["feature_columns"]))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Metrics CSV: `{payload['metrics_csv']}`",
            "- Checkpoints: `outputs/.../runs/reduced_uformer_nodecal/N17043_gm_probe/<candidate>/ckpt_best.pt`",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{float(x):.6f}" if pd.notna(x) else "")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


if __name__ == "__main__":
    main()
