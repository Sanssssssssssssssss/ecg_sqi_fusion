"""Cross-domain hard feature learnability probe.

This is diagnostic only. It does not train or select a classifier. It asks:

1. Are the hard waveform facts recoverable from waveform-derived primitive
   token banks inside each source domain?
2. Do regressors trained on one source domain recover the same facts on the
   opposite domain, or does recovery collapse because the feature distributions
   are not aligned?

The final model is still waveform-only. This script uses simple source-trained
linear regressors only to audit tokenizer evidence and target consistency.
"""

from __future__ import annotations

import argparse
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


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
PTB_ALIGN_SCRIPT = ANALYSIS_DIR / "run_ptb_bad_alignment_cross_dataset.py"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"
GEOMETRY_SCRIPT = ANALYSIS_DIR / "uformer_geometry_branch_experiment.py"

HARD_TARGETS = [
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "sqi_basSQI",
    "flatline_ratio",
    "qrs_band_ratio",
    "template_corr",
    "non_qrs_diff_p95",
    "band_30_45",
    "sqi_bSQI",
    "sqi_pSQI",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ALIGN = load_module(PTB_ALIGN_SCRIPT, "ptb_align_for_hard_feature_probe")
DUAL = ALIGN.DUAL
STUDENT = load_module(STUDENT_SCRIPT, "waveform_student_for_hard_feature_probe")
GEOMETRY = load_module(GEOMETRY_SCRIPT, "uformer_geometry_for_hard_feature_probe")
FEATURE_COLUMNS = list(DUAL.FORMAL_FEATURE_COLUMNS)


@dataclass
class DomainData:
    name: str
    x: np.ndarray
    y: np.ndarray
    aux_raw: np.ndarray
    frame: pd.DataFrame


def robust_raw_features(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    med = np.median(arr, axis=1, keepdims=True)
    q75 = np.percentile(arr, 75, axis=1, keepdims=True)
    q25 = np.percentile(arr, 25, axis=1, keepdims=True)
    scale = np.maximum((q75 - q25) / 1.349, 1e-6)
    z = (arr - med) / scale
    dz = np.diff(z, axis=1, prepend=z[:, :1])
    return np.stack(
        [
            np.sqrt(np.mean(arr * arr, axis=1)),
            np.std(arr, axis=1),
            np.mean(np.abs(arr), axis=1),
            np.percentile(arr, 99, axis=1) - np.percentile(arr, 1, axis=1),
            np.sqrt(np.mean(z * z, axis=1)),
            np.percentile(np.abs(dz), 95, axis=1),
            np.mean(np.abs(dz) < 1e-3, axis=1),
            np.mean(np.abs(z) < 0.05, axis=1),
            np.mean(np.abs(z) > 7.5, axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def _torch_part(fn: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            arr = np.asarray(x[start : start + batch_size], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, None, :]
            xb = torch.from_numpy(arr)
            rows.append(fn(xb).detach().cpu().numpy().reshape(xb.shape[0], -1).astype(np.float32))
    return np.concatenate(rows, axis=0)


def primitive_bank(x: np.ndarray, batch_size: int, mode: str) -> np.ndarray:
    """Build a compact waveform-derived primitive bank.

    The bank contains fixed differentiable/statistical views that a Transformer
    tokenizer could expose as tokens: waveform moments, QRS/detail banks,
    detector/RR-like summaries, baseline/frequency, and sparse artifact/stress.
    """

    parts = [robust_raw_features(x)]
    funcs = [
        STUDENT.waveform_stats_v2,
        STUDENT.qrs_detail_stats,
        STUDENT.qrs_visibility_bank_stats,
        STUDENT.qrs_detector_agreement_stats,
        STUDENT.rr_consistency_stats,
        STUDENT.baseline_frequency_stats,
        STUDENT.waveform_stress_stats,
    ]
    if mode == "full":
        funcs.extend([STUDENT.sparse_extreme_event_stats, STUDENT.qrs_event_patch_features])
    for fn in funcs:
        arr = _torch_part(fn, x, batch_size)
        parts.append(arr)
    bank = np.concatenate(parts, axis=1)
    bank = np.nan_to_num(bank, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return bank


def waveform_fact_aux(raw: np.ndarray) -> np.ndarray:
    primitives = GEOMETRY.compute_primitives(raw)
    for col in FEATURE_COLUMNS:
        if col not in primitives.columns:
            primitives[col] = 0.0
    return (
        primitives[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def load_ptb(split: str, channel_stats: Any, feature_norm: Any, target_source: str) -> DomainData:
    ds = ALIGN.BadAlignedSyntheticDualviewDataset(
        split,
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        channel_stats,
        feature_norm,
        False,
        "",
        0.0,
        20260619,
    )
    raw_all = np.load(ALIGN.ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    raw = raw_all[ds.frame["idx"].to_numpy(dtype=np.int64)]
    if target_source == "manifest":
        feat_npz = np.load(ALIGN.ARCH.DATA_DIR / "tabular_features.npz")
        features_all = feat_npz["features"].astype(np.float32)[:, ALIGN.FORMAL_FEATURE_IDX]
        aux_raw = features_all[ds.frame["idx"].to_numpy(dtype=np.int64)]
    elif target_source == "waveform_primitives":
        aux_raw = waveform_fact_aux(raw)
    else:
        raise ValueError(f"unknown PTB target source: {target_source}")
    frame = ds.frame.copy()
    frame["label"] = pd.Series(ds.y).map({0: "good", 1: "medium", 2: "bad"}).to_numpy()
    return DomainData(f"ptb_{split}", raw, ds.y.astype(np.int64), aux_raw, frame)


def load_clean(policy: str, split: str, channel_stats: Any, feature_norm: Any, target_source: str) -> DomainData:
    ds = DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, split, channel_stats, feature_norm)
    npz = np.load(DUAL.PROTOCOL_ROOT / policy / "signals.npz")
    key = "X" if "X" in npz.files else npz.files[0]
    raw_all = np.asarray(npz[key], dtype=np.float32)
    if raw_all.ndim == 3:
        raw_all = raw_all[:, 0, :]
    raw = raw_all[ds.frame["idx"].to_numpy(dtype=np.int64)]
    if target_source == "atlas":
        aux_raw = ds.aux_raw.astype(np.float32)
    elif target_source == "waveform_primitives":
        aux_raw = waveform_fact_aux(raw)
    else:
        raise ValueError(f"unknown clean target source: {target_source}")
    frame = ds.frame.copy()
    frame["label"] = pd.Series(ds.y).map({0: "good", 1: "medium", 2: "bad"}).to_numpy()
    return DomainData(f"clean_{split}", raw, ds.y.astype(np.int64), aux_raw, frame)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xm, xs = standardize_fit(x)
    ym, ys = standardize_fit(y)
    xz = standardize_apply(x, xm, xs)
    yz = standardize_apply(y, ym, ys)
    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float32)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    w = np.linalg.solve(x_aug.T @ x_aug + float(alpha) * eye, x_aug.T @ yz).astype(np.float32)
    return w, np.stack([xm, xs]), np.stack([ym, ys])


def ridge_predict(x: np.ndarray, w: np.ndarray, x_stats: np.ndarray, y_stats: np.ndarray) -> np.ndarray:
    xz = standardize_apply(x, x_stats[0], x_stats[1])
    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float32)], axis=1)
    pred_z = x_aug @ w
    return pred_z * y_stats[1][None, :] + y_stats[0][None, :]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3 or float(np.std(aa[mask])) < 1e-9 or float(np.std(bb[mask])) < 1e-9:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def eval_rows(
    train_name: str,
    eval_name: str,
    pred: np.ndarray,
    true: np.ndarray,
    y: np.ndarray,
    target_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for j, feat in enumerate(target_names):
        rows.append(
            {
                "train_domain": train_name,
                "eval_domain": eval_name,
                "feature": feat,
                "class": "all",
                "corr": corr(pred[:, j], true[:, j]),
                "mae": float(np.nanmean(np.abs(pred[:, j] - true[:, j]))),
            }
        )
        for cls, label in [(0, "good"), (1, "medium"), (2, "bad")]:
            mask = y == cls
            if int(mask.sum()) >= 8:
                rows.append(
                    {
                        "train_domain": train_name,
                        "eval_domain": eval_name,
                        "feature": feat,
                        "class": label,
                        "corr": corr(pred[mask, j], true[mask, j]),
                        "mae": float(np.nanmean(np.abs(pred[mask, j] - true[mask, j]))),
                    }
                )
    return rows


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-policy", default="margin_ge_5s_drop_outlier")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bank-mode", choices=["compact", "full"], default="compact")
    parser.add_argument("--alpha", type=float, default=25.0)
    parser.add_argument("--ptb-target-source", choices=["manifest", "waveform_primitives"], default="manifest")
    parser.add_argument("--clean-target-source", choices=["atlas", "waveform_primitives"], default="atlas")
    args = parser.parse_args()

    ckpt = torch.load(OUT_ROOT / "runs" / "ptb_bad_alignment_cross_dataset" / "pba_run_0af4de989b083919" / "ckpt_best.pt", map_location="cpu")
    channel_stats = DUAL.ChannelStats(**ckpt["channel_stats"])
    feature_norm = DUAL.FeatureNorm(
        mean=np.asarray(ckpt["feature_train_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_train_std"], dtype=np.float32),
    )
    target_names = [t for t in HARD_TARGETS if t in FEATURE_COLUMNS]
    target_idx = [FEATURE_COLUMNS.index(t) for t in target_names]

    domains = {
        "ptb_train": load_ptb("train", channel_stats, feature_norm, args.ptb_target_source),
        "ptb_test": load_ptb("test", channel_stats, feature_norm, args.ptb_target_source),
        "clean_train": load_clean(args.clean_policy, "train", channel_stats, feature_norm, args.clean_target_source),
        "clean_test": load_clean(args.clean_policy, "test", channel_stats, feature_norm, args.clean_target_source),
    }

    banks = {
        name: primitive_bank(data.x, int(args.batch_size), args.bank_mode)
        for name, data in domains.items()
    }
    rows: list[dict[str, Any]] = []
    for train_name, eval_names in {
        "ptb_train": ["ptb_test", "clean_test"],
        "clean_train": ["clean_test", "ptb_test"],
    }.items():
        train = domains[train_name]
        w, x_stats, y_stats = ridge_fit(banks[train_name], train.aux_raw[:, target_idx], float(args.alpha))
        for eval_name in eval_names:
            data = domains[eval_name]
            pred = ridge_predict(banks[eval_name], w, x_stats, y_stats)
            rows.extend(eval_rows(train_name, eval_name, pred, data.aux_raw[:, target_idx], data.y, target_names))

    results = pd.DataFrame(rows)
    out_dir = ANALYSIS_DIR / "cross_domain_hard_feature_learnability"
    report_dir = REPORT_DIR / "cross_domain_hard_feature_learnability"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    short_source = {"manifest": "mf", "waveform_primitives": "wf", "atlas": "atlas"}
    suffix = f"{args.bank_mode}_ptb-{short_source[args.ptb_target_source]}_clean-{short_source[args.clean_target_source]}"
    csv_path = out_dir / f"hard_feature_learnability_{suffix}.csv"
    summary_path = out_dir / f"hard_feature_learnability_{suffix}_summary.json"
    report_path = out_dir / f"hard_feature_learnability_{suffix}_report.md"
    results.to_csv(csv_path, index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bank_mode": args.bank_mode,
        "ptb_target_source": args.ptb_target_source,
        "clean_target_source": args.clean_target_source,
        "alpha": float(args.alpha),
        "targets": target_names,
        "csv": str(csv_path),
        "report": str(report_dir / report_path.name),
        "contract": "diagnostic only; no classifier selection; no target-domain fitting",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Cross-Domain Hard Feature Learnability",
        "",
        "Diagnostic only. A source-trained linear ridge probe is fitted on waveform-derived primitive banks, then evaluated in-source and cross-domain.",
        "",
        f"- Bank mode: `{args.bank_mode}`",
        f"- PTB target source: `{args.ptb_target_source}`",
        f"- Clean target source: `{args.clean_target_source}`",
        f"- Ridge alpha: `{float(args.alpha):g}`",
        f"- Targets: `{', '.join(target_names)}`",
        "",
        "## All-Class Recovery",
        "",
    ]
    all_rows = results[results["class"].eq("all")].copy()
    all_rows["corr"] = all_rows["corr"].astype(float)
    lines.append(
        md_table(
            all_rows.sort_values(["train_domain", "eval_domain", "corr"])[
                ["train_domain", "eval_domain", "feature", "corr", "mae"]
            ]
        )
    )
    lines.extend(["", "## Minimum Per-Class Correlation", ""])
    cls_rows = results[~results["class"].eq("all")].copy()
    min_rows = (
        cls_rows.groupby(["train_domain", "eval_domain", "feature"], as_index=False)
        .agg(min_class_corr=("corr", "min"), mean_class_mae=("mae", "mean"))
        .sort_values(["train_domain", "eval_domain", "min_class_corr"])
    )
    lines.append(md_table(min_rows))
    lines.extend(["", f"CSV: `{csv_path}`"])
    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    (report_dir / report_path.name).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
