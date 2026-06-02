"""E3.11f BUT-like synthetic adaptation grid.

This runner is intentionally experiment-only.  It explores whether changing the
PTB synthetic data rules can better match BUT QDB expert SQI labels while keeping
the Uformer mainline intact.  It also runs BUT-supervised probes as a separate
adaptation track.  Nothing here touches ``src/sqi_pipeline`` or overwrites the
mainline checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.run import (
    CLASS_TO_INT,
    FS_TARGET,
    INT_TO_CLASS,
    N_TARGET,
    apply_but_thresholds,
    basic_sqi_features,
    calibrate_but,
    extract_uformer_features,
    export_eval_visuals,
    load_uformer_model,
    multiclass_report,
    plot_wave_gallery,
    run_model_outputs,
)


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_adaptation_14h_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_REALDATA_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02"
DEFAULT_SOURCE_ARTIFACT = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid" / "data" / "med6p25_badgap7_badcm0p75"
DEFAULT_MAINLINE_CKPT = ROOT / "outputs" / "mainline" / "e311_uformer_full_tokens_detach_seed0" / "ckpt_best.pt"
CLASS_NAMES = ("good", "medium", "bad")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return {"status": "initialized", "started_at": now_iso(), "steps": []}


def update_state(path: Path, **kwargs: Any) -> dict[str, Any]:
    state = load_state(path)
    state.update(kwargs)
    write_json(path, state)
    return state


def ensure_processed_but(out_root: Path, realdata_root: Path) -> None:
    src = realdata_root / "processed" / "butqdb"
    dst = out_root / "processed" / "butqdb"
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def class_array(labels: pd.DataFrame) -> np.ndarray:
    return labels["y_class"].map(CLASS_TO_INT).to_numpy(dtype=np.int64)


def moving_average_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    xp = np.pad(x, ((0, 0), (pad_l, pad_r)), mode="edge")
    return np.stack([np.convolve(row, kernel, mode="valid") for row in xp], axis=0)


def signal_features(X: np.ndarray) -> pd.DataFrame:
    x = X[:, 0] if X.ndim == 3 else X
    x = x.astype(np.float64, copy=False)
    dx = np.diff(x, axis=1)
    drift = moving_average_same(x, FS_TARGET)
    rows: list[dict[str, float]] = []
    for row, drow, drift_row in zip(x, dx, drift):
        f, pxx = welch(row, fs=FS_TARGET, nperseg=min(256, len(row)))

        def band(lo: float, hi: float) -> float:
            mask = (f >= lo) & (f < hi)
            if not np.any(mask):
                return 0.0
            df = float(np.median(np.diff(f))) if len(f) > 1 else 1.0
            return float(np.sum(pxx[mask]) * df)

        total = band(0.5, 40.0) + 1e-12
        abs_row = np.abs(row)
        rows.append(
            {
                "rms": float(np.sqrt(np.mean(row * row) + 1e-12)),
                "amp_p95": float(np.percentile(abs_row, 95.0)),
                "amp_p99": float(np.percentile(abs_row, 99.0)),
                "qrs_prominence": float(np.percentile(abs_row, 98.0) / (np.percentile(abs_row, 70.0) + 1e-6)),
                "hf_ratio": float(band(20.0, 40.0) / total),
                "drift_ratio": float(band(0.5, 1.0) / total),
                "deriv_rms": float(np.sqrt(np.mean(drow * drow) + 1e-12)),
                "flat_fraction": float(np.mean(abs_row < (np.percentile(abs_row, 20.0) + 1e-6))),
                "drift_rms": float(np.sqrt(np.mean((drift_row - np.mean(drift_row)) ** 2) + 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_class(features: pd.DataFrame, labels: np.ndarray | pd.Series) -> dict[str, Any]:
    lab = np.asarray(labels)
    out: dict[str, Any] = {}
    for cls_idx, cls in enumerate(CLASS_NAMES):
        if lab.dtype.kind in {"U", "S", "O"}:
            mask = lab.astype(str) == cls
        else:
            mask = lab.astype(int) == cls_idx
        sub = features.loc[mask]
        out[cls] = {"n": int(len(sub))}
        for col in features.columns:
            vals = sub[col].to_numpy(dtype=np.float64)
            if len(vals) == 0:
                out[cls][col] = {"mean": None, "p10": None, "p50": None, "p90": None}
            else:
                out[cls][col] = {
                    "mean": float(np.mean(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                }
    return out


def load_ptb_artifact(source: Path) -> dict[str, Any]:
    data_dir = source / "datasets"
    labels_path = data_dir / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = pd.read_csv(labels_path).sort_values("idx").reset_index(drop=True)
    clean = np.load(data_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    noisy = np.load(data_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    masks_npz = np.load(data_dir / "synth_10s_125hz_local_mask.npz")
    masks = {key: masks_npz[key].astype(np.float32) for key in masks_npz.files}
    return {"labels": labels, "clean": clean, "noisy": noisy, "masks": masks}


def normalize_noise(noise: np.ndarray) -> np.ndarray:
    noise = noise.astype(np.float32, copy=False)
    return noise / (np.std(noise, axis=1, keepdims=True) + 1e-6)


def smooth_rows(x: np.ndarray, window: int) -> np.ndarray:
    return moving_average_same(x.astype(np.float64), window).astype(np.float32)


def highpass_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=(n, N_TARGET)).astype(np.float32)
    b, a = butter(2, 18.0, btype="highpass", fs=FS_TARGET)
    return filtfilt(b, a, white, axis=1).astype(np.float32)


def lowfreq_drift(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(N_TARGET, dtype=np.float32) / float(FS_TARGET)
    out = np.zeros((n, N_TARGET), dtype=np.float32)
    for i in range(n):
        f1 = float(rng.uniform(0.08, 0.35))
        f2 = float(rng.uniform(0.35, 0.90))
        p1 = float(rng.uniform(0.0, 2 * np.pi))
        p2 = float(rng.uniform(0.0, 2 * np.pi))
        out[i] = np.sin(2 * np.pi * f1 * t + p1) + 0.4 * np.sin(2 * np.pi * f2 * t + p2)
    return normalize_noise(out)


def burst_noise(rng: np.random.Generator, n: int, strength: float) -> np.ndarray:
    out = np.zeros((n, N_TARGET), dtype=np.float32)
    for i in range(n):
        k = int(rng.integers(1, 4))
        for _ in range(k):
            width = int(rng.integers(FS_TARGET // 5, FS_TARGET * 2))
            start = int(rng.integers(0, max(1, N_TARGET - width)))
            out[i, start : start + width] += rng.normal(0.0, strength, size=width)
    return out


def make_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    families = [
        ("qrs_visibility", [0.18, 0.28, 0.38], [0.05, 0.12]),
        ("medium_expert_boundary", [0.08, 0.15, 0.22], [0.06, 0.12]),
        ("bad_unreliable_qrs", [0.35, 0.48, 0.60], [0.16, 0.28]),
        ("wearable_noise", [0.10, 0.20, 0.32], [0.14, 0.24]),
        ("low_amp_domain", [0.18, 0.30, 0.42], [0.10, 0.22]),
        ("mixed_expert", [0.20, 0.34, 0.50], [0.10, 0.20]),
    ]
    idx = 0
    for family, a_values, b_values in families:
        for a in a_values:
            for b in b_values:
                idx += 1
                spec = {
                    "id": f"{idx:02d}_{family}_a{a:.2f}_b{b:.2f}".replace(".", "p"),
                    "family": family,
                    "qrs_attn_medium": 0.06,
                    "qrs_attn_bad": 0.22,
                    "tst_noise_medium": 0.05,
                    "hf_medium": 0.03,
                    "hf_bad": 0.08,
                    "drift_medium": 0.03,
                    "drift_bad": 0.06,
                    "burst_bad": 0.03,
                    "low_amp_medium": 1.0,
                    "low_amp_bad": 1.0,
                    "residual_scale_good": 0.75,
                    "residual_scale_medium": 1.05,
                    "residual_scale_bad": 1.20,
                    "smooth_bad": 0.0,
                    "contact_loss_bad": 0.0,
                }
                if family == "qrs_visibility":
                    spec.update({"qrs_attn_medium": a * 0.45, "qrs_attn_bad": a, "smooth_bad": b})
                elif family == "medium_expert_boundary":
                    spec.update({"tst_noise_medium": a, "drift_medium": b, "qrs_attn_medium": 0.04})
                elif family == "bad_unreliable_qrs":
                    spec.update({"qrs_attn_bad": a, "contact_loss_bad": b, "burst_bad": 0.05})
                elif family == "wearable_noise":
                    spec.update({"drift_medium": a * 0.7, "drift_bad": a, "hf_bad": b, "burst_bad": b})
                elif family == "low_amp_domain":
                    spec.update({"low_amp_medium": 1.0 - a * 0.35, "low_amp_bad": 1.0 - a, "qrs_attn_bad": b})
                elif family == "mixed_expert":
                    spec.update(
                        {
                            "qrs_attn_medium": a * 0.25,
                            "qrs_attn_bad": a,
                            "tst_noise_medium": b,
                            "drift_bad": b,
                            "hf_bad": b * 0.65,
                            "low_amp_bad": max(0.55, 1.0 - a * 0.7),
                            "contact_loss_bad": b * 0.8,
                        }
                    )
                specs.append(spec)
    return specs


def apply_spec_to_ptb(ptb: dict[str, Any], spec: dict[str, Any], out_dir: Path, seed: int, sample_limit: int = 0) -> dict[str, Any]:
    labels = ptb["labels"].copy()
    clean = ptb["clean"].astype(np.float32, copy=False)
    noisy = ptb["noisy"].astype(np.float32, copy=False)
    masks = ptb["masks"]
    if sample_limit > 0:
        labels = labels.iloc[:sample_limit].copy()
        clean = clean[:sample_limit]
        noisy = noisy[:sample_limit]
        masks = {k: v[:sample_limit] for k, v in masks.items()}
    rng = np.random.default_rng(seed)
    y = class_array(labels)
    residual = noisy - clean
    qrs = masks.get("qrs_mask", np.zeros_like(clean)).astype(np.float32)
    tst = masks.get("tst_mask", np.zeros_like(clean)).astype(np.float32)
    critical = masks.get("critical_mask", qrs).astype(np.float32)
    x = clean.copy()
    scale = np.ones(len(x), dtype=np.float32)
    scale[y == 0] = float(spec["residual_scale_good"])
    scale[y == 1] = float(spec["residual_scale_medium"])
    scale[y == 2] = float(spec["residual_scale_bad"])
    x = x + residual * scale[:, None]

    medium = y == 1
    bad = y == 2
    if np.any(medium):
        x[medium] -= float(spec["qrs_attn_medium"]) * clean[medium] * qrs[medium]
        x[medium] += float(spec["tst_noise_medium"]) * np.std(clean[medium], axis=1, keepdims=True) * normalize_noise(rng.normal(size=(medium.sum(), N_TARGET)).astype(np.float32)) * tst[medium]
        x[medium] += float(spec["drift_medium"]) * np.std(clean[medium], axis=1, keepdims=True) * lowfreq_drift(rng, int(medium.sum()))
        x[medium] += float(spec["hf_medium"]) * np.std(clean[medium], axis=1, keepdims=True) * normalize_noise(highpass_noise(rng, int(medium.sum())))
        x[medium] *= float(spec["low_amp_medium"])
    if np.any(bad):
        x[bad] -= float(spec["qrs_attn_bad"]) * clean[bad] * qrs[bad]
        if float(spec["smooth_bad"]) > 0:
            sm = smooth_rows(x[bad], int(5 + 28 * float(spec["smooth_bad"])))
            x[bad] = x[bad] * (1.0 - critical[bad] * float(spec["smooth_bad"])) + sm * critical[bad] * float(spec["smooth_bad"])
        x[bad] += float(spec["drift_bad"]) * np.std(clean[bad], axis=1, keepdims=True) * lowfreq_drift(rng, int(bad.sum()))
        x[bad] += float(spec["hf_bad"]) * np.std(clean[bad], axis=1, keepdims=True) * normalize_noise(highpass_noise(rng, int(bad.sum())))
        x[bad] += float(spec["burst_bad"]) * np.std(clean[bad], axis=1, keepdims=True) * burst_noise(rng, int(bad.sum()), 1.0)
        if float(spec["contact_loss_bad"]) > 0:
            gate = np.clip(qrs[bad] + critical[bad], 0.0, 1.0)
            x[bad] *= 1.0 - float(spec["contact_loss_bad"]) * gate
        x[bad] *= float(spec["low_amp_bad"])

    x = x.astype(np.float32)
    labels["adaptation_spec_id"] = str(spec["id"])
    labels["adaptation_family"] = str(spec["family"])
    labels["sample_source"] = labels.get("sample_source", "ptb_synthetic").astype(str) + f"|but_adapt:{spec['id']}"
    data_dir = out_dir / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir = Path(ptb["source"]) / "datasets"
    if sample_limit <= 0:
        link_or_copy(source_data_dir / "synth_10s_125hz_clean.npz", data_dir / "synth_10s_125hz_clean.npz")
        link_or_copy(source_data_dir / "synth_10s_125hz_local_mask.npz", data_dir / "synth_10s_125hz_local_mask.npz")
    else:
        np.savez_compressed(data_dir / "synth_10s_125hz_clean.npz", X_clean=clean)
        np.savez_compressed(data_dir / "synth_10s_125hz_local_mask.npz", **masks)
    np.savez_compressed(data_dir / "synth_10s_125hz_noisy.npz", X_noisy=x)
    labels.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)
    labels.to_csv(data_dir / "synth_10s_125hz_labels_with_level.csv", index=False)
    if "synth_10s_125hz_noise_level.npz" in [p.name for p in (Path(ptb["source"]) / "datasets").glob("*.npz")]:
        src_level = source_data_dir / "synth_10s_125hz_noise_level.npz"
        if src_level.exists():
            link_or_copy(src_level, data_dir / src_level.name)
    audit = {
        "spec": spec,
        "signals_shape": list(x.shape),
        "split_counts": {str(k): int(v) for k, v in labels["split"].value_counts().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in labels["y_class"].value_counts().to_dict().items()},
    }
    write_json(out_dir / "data_variant_audit.json", audit)
    return audit


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def domain_distance(candidate_summary: dict[str, Any], but_summary: dict[str, Any]) -> float:
    cols = ["rms", "amp_p95", "qrs_prominence", "hf_ratio", "drift_ratio", "deriv_rms"]
    total = 0.0
    n = 0
    for cls in CLASS_NAMES:
        for col in cols:
            a = candidate_summary[cls][col]["p50"]
            b = but_summary[cls][col]["p50"]
            if a is None or b is None:
                continue
            scale = abs(float(b)) + 1e-4
            total += min(5.0, abs(float(a) - float(b)) / scale)
            n += 1
    return float(total / max(1, n))


def make_domain_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    ensure_processed_but(out_root, Path(args.realdata_root))
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    ptb["source"] = str(args.source_artifact_dir)
    but_dir = out_root / "processed" / "butqdb"
    but_x = np.load(but_dir / "signals.npz")["X"].astype(np.float32)
    but_meta = pd.read_csv(but_dir / "metadata.csv")
    but_feat = signal_features(but_x)
    ptb_feat = signal_features(ptb["noisy"][:, None, :])
    but_summary = summarize_by_class(but_feat, but_meta["y_class"])
    ptb_summary = summarize_by_class(ptb_feat, class_array(ptb["labels"]))
    audit = {
        "status": "completed",
        "but_shape": list(but_x.shape),
        "ptb_shape": list(ptb["noisy"][:, None, :].shape),
        "but_summary": but_summary,
        "current_ptb_summary": ptb_summary,
        "initial_domain_distance": domain_distance(ptb_summary, but_summary),
    }
    write_json(out_root / "domain_gap_metrics.json", audit)
    write_json(report_root / "domain_gap_metrics.json", audit)
    lines = [
        "# BUT Domain Gap Audit",
        "",
        f"- Initial synthetic-vs-BUT distance: `{audit['initial_domain_distance']:.4f}`",
        "- BUT target labels are expert consensus `1/2/3 -> good/medium/bad`.",
        "- Key hypothesis: BUT bad/medium are more about QRS reliability, low amplitude/contact loss, and wearable drift than our original SNR/morph thresholds.",
        "",
        "## Median Feature Snapshot",
        "",
        "| class | BUT rms | PTB rms | BUT qrs_prom | PTB qrs_prom | BUT drift | PTB drift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in CLASS_NAMES:
        lines.append(
            f"| {cls} | {but_summary[cls]['rms']['p50']:.4f} | {ptb_summary[cls]['rms']['p50']:.4f} | "
            f"{but_summary[cls]['qrs_prominence']['p50']:.3f} | {ptb_summary[cls]['qrs_prominence']['p50']:.3f} | "
            f"{but_summary[cls]['drift_ratio']['p50']:.4f} | {ptb_summary[cls]['drift_ratio']['p50']:.4f} |"
        )
    (report_root / "but_domain_gap_report.md").parent.mkdir(parents=True, exist_ok=True)
    (report_root / "but_domain_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    visuals = out_root / "visuals" / "domain_gap"
    visuals.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        idx_but = but_meta.index[but_meta["y_class"].astype(str) == cls].to_numpy()[:12]
        plot_wave_gallery(but_x, but_meta, idx_but.tolist(), visuals / f"but_{cls}_gallery.png", f"BUT {cls}")
    return audit


def score_specs(args: argparse.Namespace, specs: list[dict[str, Any]], limit: int = 2400) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    ptb["source"] = str(args.source_artifact_dir)
    but_dir = out_root / "processed" / "butqdb"
    but_x = np.load(but_dir / "signals.npz")["X"].astype(np.float32)
    but_meta = pd.read_csv(but_dir / "metadata.csv")
    but_summary = summarize_by_class(signal_features(but_x), but_meta["y_class"])
    rows: list[dict[str, Any]] = []
    tmp_root = out_root / "_domain_score_tmp"
    for spec in specs:
        variant_dir = tmp_root / str(spec["id"])
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        audit = apply_spec_to_ptb(ptb, spec, variant_dir, seed=int(args.seed), sample_limit=limit)
        x = np.load(variant_dir / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        labels = pd.read_csv(variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv")
        summary = summarize_by_class(signal_features(x[:, None, :]), class_array(labels))
        dist = domain_distance(summary, but_summary)
        rows.append({"spec_id": spec["id"], "family": spec["family"], "domain_distance": dist, "spec": spec, "audit": audit})
        shutil.rmtree(variant_dir, ignore_errors=True)
    rows.sort(key=lambda r: float(r["domain_distance"]))
    write_json(out_root / "grid_specs_ranked.json", rows)
    write_json(Path(args.report_root) / "grid_specs_ranked.json", rows)
    return rows


def generate_grid(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    specs = make_specs()
    write_json(out_root / "grid_specs.json", specs)
    ranked = score_specs(args, specs, limit=int(args.domain_score_sample_limit))
    payload = {"status": "completed", "n_specs": len(specs), "top12": ranked[:12]}
    write_json(out_root / "make_grid_summary.json", payload)
    return payload


def run_training_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, text=True)
    return int(proc.returncode)


def evaluate_but_checkpoint(args: argparse.Namespace, checkpoint: Path, run_dir: Path, force: bool = True) -> dict[str, Any]:
    result_dir = run_dir / "butqdb_eval"
    if (result_dir / "but_test_report.json").exists() and not force:
        return read_json(result_dir / "but_test_report.json")
    but_dir = Path(args.out_root) / "processed" / "butqdb"
    X = np.load(but_dir / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(but_dir / "metadata.csv")
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_uformer_model(checkpoint, device, feature_set="full_tokens", with_head=True)
    outputs = run_model_outputs(model, X, int(args.batch_size_eval), device)
    probs = outputs["probs"]
    assert isinstance(probs, np.ndarray)
    cal = calibrate_but(probs[split == "val"], y[split == "val"])
    pred_val = apply_but_thresholds(probs[split == "val"], cal["t_good"], cal["t_bad"])
    pred_test = apply_but_thresholds(probs[split == "test"], cal["t_good"], cal["t_bad"])
    pred_all = apply_but_thresholds(probs, cal["t_good"], cal["t_bad"])
    report = {
        "checkpoint": str(checkpoint),
        "threshold_calibration": cal,
        "but_val_report": multiclass_report(y[split == "val"], pred_val, probs[split == "val"]),
        "but_test_report": multiclass_report(y[split == "test"], pred_test, probs[split == "test"]),
        "but_all_report": multiclass_report(y, pred_all, probs),
        "runtime_hardware": {
            "platform": platform.platform(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "elapsed_sec": float(outputs["elapsed_sec"]),
            "peak_cuda_memory_bytes": int(outputs["peak_cuda_memory_bytes"]),
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "threshold_calibration.json", cal)
    write_json(result_dir / "but_val_report.json", report["but_val_report"])
    write_json(result_dir / "but_test_report.json", report["but_test_report"])
    write_json(result_dir / "but_eval_summary.json", report)
    denoise = outputs["denoise"]
    assert isinstance(denoise, np.ndarray)
    export_eval_visuals("butqdb", X, meta, denoise, probs, pred_all, result_dir / "visuals")
    make_but_error_galleries(X, meta, denoise, probs, pred_all, result_dir / "visuals")
    return report


def make_but_error_galleries(X: np.ndarray, meta: pd.DataFrame, denoise: np.ndarray, probs: np.ndarray, pred: np.ndarray, out_dir: Path) -> None:
    meta2 = meta.copy()
    meta2["pred_class"] = [INT_TO_CLASS[int(v)] for v in pred]
    meta2["pred_idx"] = pred
    cases = {
        "good_misrejected": (meta2["y"].to_numpy() == 0) & (pred != 0),
        "medium_confused": (meta2["y"].to_numpy() == 1) & (pred != 1),
        "bad_missed": (meta2["y"].to_numpy() == 2) & (pred != 2),
    }
    for name, mask in cases.items():
        positions = np.where(mask)[0][:16].tolist()
        plot_wave_gallery(X, meta2, positions, out_dir / f"{name}_gallery.png", f"BUT {name}", denoise=denoise, probs=probs)


def training_summary_from_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "mainline_summary.json"
    if not summary_path.exists():
        return {"status": "missing", "run_dir": str(run_dir)}
    summary = read_json(summary_path)
    stage2 = summary.get("stage2", {})
    return {
        "status": "completed",
        "ptb_test_report": stage2.get("test_report", {}),
        "ptb_denoise_metrics": stage2.get("denoise_metrics", {}),
        "output_dir": str(run_dir),
    }


def train_spec(args: argparse.Namespace, spec: dict[str, Any], mode: str) -> dict[str, Any]:
    out_root = Path(args.out_root)
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    ptb["source"] = str(args.source_artifact_dir)
    spec_root = out_root / "synthetic_variants" / str(spec["id"])
    if not (spec_root / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists() or bool(args.force_data):
        apply_spec_to_ptb(ptb, spec, spec_root, seed=int(args.seed))
    run_dir = out_root / "runs" / mode / str(spec["id"])
    log_dir = out_root / "logs" / mode
    if mode == "quick":
        e1, e2 = int(args.quick_epochs_stage1), int(args.quick_epochs_stage2)
        bs1, bs2 = int(args.quick_batch_size_stage1), int(args.quick_batch_size_stage2)
    else:
        e1, e2 = int(args.full_epochs_stage1), int(args.full_epochs_stage2)
        bs1, bs2 = int(args.full_batch_size_stage1), int(args.full_batch_size_stage2)
    command = [
        sys.executable,
        "-m",
        "src.transformer_pipeline.train_uformer_mainline",
        "--stage",
        "all",
        "--source_artifact_dir",
        str(spec_root),
        "--output_dir",
        str(run_dir),
        "--epochs_stage1",
        str(e1),
        "--epochs_stage2",
        str(e2),
        "--batch_size_stage1",
        str(bs1),
        "--batch_size_stage2",
        str(bs2),
        "--lambda_den_stage2",
        str(args.lambda_den_stage2),
        "--lambda_cls",
        str(args.lambda_cls),
        "--class_weight",
        str(args.class_weight),
        "--seed",
        str(args.seed),
    ]
    start = time.perf_counter()
    rc = run_training_command(command, ROOT, log_dir / f"{spec['id']}.stdout.txt", log_dir / f"{spec['id']}.stderr.txt")
    elapsed = float(time.perf_counter() - start)
    payload: dict[str, Any] = {"spec": spec, "mode": mode, "returncode": rc, "elapsed_sec": elapsed, "run_dir": str(run_dir)}
    if rc == 0 and (run_dir / "ckpt_best.pt").exists():
        payload.update(training_summary_from_run(run_dir))
        but = evaluate_but_checkpoint(args, run_dir / "ckpt_best.pt", run_dir)
        payload["but_val_report"] = but["but_val_report"]
        payload["but_test_report"] = but["but_test_report"]
        payload["threshold_calibration"] = but["threshold_calibration"]
        write_visual_review(run_dir, payload)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "adaptation_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / "adaptation_summary.jsonl", payload)
    return payload


def selection_score(row: dict[str, Any]) -> float:
    if row.get("status") == "failed":
        return -1e9
    but = row.get("but_val_report", {})
    ptb = row.get("ptb_test_report", {})
    den = row.get("ptb_denoise_metrics", {})
    acc = float(but.get("acc", 0.0))
    macro = float(but.get("macro_f1", 0.0))
    bad = float((but.get("recall_good_medium_bad") or [0, 0, 0])[2])
    ptb_acc = float(ptb.get("acc", 0.0))
    den_score = float(den.get("overall", den).get("denoise_score", den.get("denoise_score", 0.0)) if isinstance(den, dict) else 0.0)
    penalty = 0.0
    if ptb_acc < 0.975:
        penalty += 0.15
    if den_score < 3.5:
        penalty += 0.05
    return acc + 0.20 * macro + 0.15 * bad - penalty


def write_visual_review(run_dir: Path, payload: dict[str, Any]) -> None:
    but = payload.get("but_test_report", {})
    ptb = payload.get("ptb_test_report", {})
    rec = but.get("recall_good_medium_bad") or [0, 0, 0]
    lines = [
        "# Visual Review",
        "",
        f"- BUT test acc: `{but.get('acc')}`",
        f"- BUT bad recall: `{rec[2] if len(rec) > 2 else None}`",
        f"- PTB test acc: `{ptb.get('acc')}`",
        "",
        "## Tags",
    ]
    if len(rec) > 2 and rec[2] < 0.5:
        lines.append("- `bad_missed`: bad is still too medium-like or threshold remains conservative.")
    if len(rec) > 0 and rec[0] < 0.8:
        lines.append("- `good_over_rejected`: good class is not separated cleanly.")
    lines.extend(
        [
            "",
            "## Required Galleries",
            "",
            "- `butqdb_eval/visuals/processed_good_gallery.png` equivalent prediction galleries",
            "- `butqdb_eval/visuals/bad_missed_gallery.png`",
            "- `butqdb_eval/visuals/good_misrejected_gallery.png`",
        ]
    )
    (run_dir / "visual_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quick_train(args: argparse.Namespace) -> dict[str, Any]:
    ranked = read_json(Path(args.out_root) / "grid_specs_ranked.json")
    chosen = [row["spec"] for row in ranked[: int(args.quick_top_n)]]
    results = []
    for spec in chosen:
        results.append(train_spec(args, spec, "quick"))
        write_round_summary(args, "quick", results)
    return {"status": "completed", "n": len(results), "results": results}


def run_full_train(args: argparse.Namespace) -> dict[str, Any]:
    summaries = read_summary_rows(Path(args.out_root) / "adaptation_summary.jsonl")
    quick = [r for r in summaries if r.get("mode") == "quick" and r.get("returncode") == 0]
    quick.sort(key=selection_score, reverse=True)
    chosen = [r["spec"] for r in quick[: int(args.full_top_n)]]
    results = []
    for spec in chosen:
        results.append(train_spec(args, spec, "full"))
        write_round_summary(args, "full", results)
    return {"status": "completed", "n": len(results), "results": results}


def read_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def classifier_score(clf: Pipeline, X: np.ndarray) -> np.ndarray:
    model = clf.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(model, "decision_function"):
        z = clf.decision_function(X)
        if z.ndim == 1:
            z = np.stack([-z, z], axis=1)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)
    pred = clf.predict(X)
    out = np.zeros((len(pred), 3), dtype=np.float32)
    out[np.arange(len(pred)), pred.astype(int)] = 1.0
    return out


def run_but_supervised(args: argparse.Namespace) -> dict[str, Any]:
    summaries = read_summary_rows(Path(args.out_root) / "adaptation_summary.jsonl")
    full = [r for r in summaries if r.get("mode") == "full" and r.get("returncode") == 0]
    full.sort(key=selection_score, reverse=True)
    checkpoints = [(Path(args.mainline_checkpoint), "current_mainline")]
    for row in full[: int(args.supervised_top_n)]:
        checkpoints.append((Path(row["run_dir"]) / "ckpt_best.pt", str(row["spec"]["id"])))
    but_dir = Path(args.out_root) / "processed" / "butqdb"
    X = np.load(but_dir / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(but_dir / "metadata.csv")
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"
    models = {
        "logreg": lambda: LogisticRegression(max_iter=1600, class_weight="balanced", solver="lbfgs"),
        "linear_svm": lambda: LinearSVC(class_weight="balanced", max_iter=10000),
        "small_mlp": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, max_iter=800, early_stopping=True, random_state=0),
    }
    feature_sets = ["full_tokens", "bottleneck_only", "summary_only", "handcrafted_sqi"]
    rows = []
    for ckpt, tag in checkpoints:
        for feature_set in feature_sets:
            if feature_set == "handcrafted_sqi":
                feats = basic_sqi_features(X)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                feats = extract_uformer_features(ckpt, X, feature_set, int(args.batch_size_eval), device)
            for name, factory in models.items():
                clf = Pipeline([("scaler", StandardScaler()), ("model", factory())])
                clf.fit(feats[train], y[train])
                probs_val = classifier_score(clf, feats[val])
                probs_test = classifier_score(clf, feats[test])
                cal = calibrate_but(probs_val, y[val])
                pred_test = apply_but_thresholds(probs_test, cal["t_good"], cal["t_bad"])
                rep = multiclass_report(y[test], pred_test, probs_test)
                row = {"checkpoint_tag": tag, "feature_set": feature_set, "model": name, "threshold_calibration": cal, "but_test_report": rep}
                rows.append(row)
                out = Path(args.out_root) / "but_supervised" / tag / feature_set
                write_json(out / f"{name}_report.json", row)
    rows.sort(key=lambda r: float(r["but_test_report"]["acc"]), reverse=True)
    payload = {"status": "completed", "runs": rows, "best": rows[0] if rows else None}
    write_json(Path(args.out_root) / "but_supervised_summary.json", payload)
    write_json(Path(args.report_root) / "but_supervised_summary.json", payload)
    return payload


def write_round_summary(args: argparse.Namespace, stage: str, rows: list[dict[str, Any]] | None = None) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    all_rows = rows or read_summary_rows(out_root / "adaptation_summary.jsonl")
    relevant = [r for r in all_rows if rows is not None or r.get("mode") == stage]
    relevant.sort(key=selection_score, reverse=True)
    lines = [
        f"# Round Summary: {stage}",
        "",
        "| rank | spec | family | BUT acc | BUT macro-F1 | BUT bad recall | PTB acc | denoise_score |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    pareto = []
    for i, row in enumerate(relevant[:20], 1):
        but = row.get("but_test_report", {})
        ptb = row.get("ptb_test_report", {})
        den = row.get("ptb_denoise_metrics", {})
        rec = but.get("recall_good_medium_bad") or [0, 0, 0]
        den_score = den.get("denoise_score") if isinstance(den, dict) else None
        lines.append(
            f"| {i} | `{row.get('spec', {}).get('id', '')}` | {row.get('spec', {}).get('family', '')} | "
            f"{float(but.get('acc', 0.0)):.4f} | {float(but.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[2] if len(rec) > 2 else 0.0):.4f} | {float(ptb.get('acc', 0.0)):.4f} | "
            f"{float(den_score or 0.0):.3f} |"
        )
        pareto.append(
            {
                "spec": row.get("spec", {}),
                "mode": row.get("mode"),
                "but_acc": but.get("acc"),
                "but_macro_f1": but.get("macro_f1"),
                "but_bad_recall": rec[2] if len(rec) > 2 else None,
                "ptb_acc": ptb.get("acc"),
                "denoise_score": den_score,
            }
        )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{stage}_round_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_root / "pareto.json", pareto)
    write_json(report_root / "pareto.json", pareto)
    write_reflection(args, relevant[:8])


def write_reflection(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Domain Gap Reflection",
        "",
        "Current hypothesis: BUT expert labels emphasize QRS reliability, low-amplitude/contact loss, and wearable drift more than the original PTB synthetic rule.",
        "",
        "## Top Observations",
    ]
    for row in rows[:5]:
        spec = row.get("spec", {})
        but = row.get("but_test_report", {})
        rec = but.get("recall_good_medium_bad") or [0, 0, 0]
        lines.append(
            f"- `{spec.get('id')}` ({spec.get('family')}): acc={float(but.get('acc', 0.0)):.4f}, "
            f"bad_recall={float(rec[2] if len(rec) > 2 else 0.0):.4f}."
        )
    lines.extend(
        [
            "",
            "If low-amplitude/QRS-unreliable specs improve bad recall without destroying good, the original rule was under-modeling expert unusable morphology.",
            "If wearable-noise specs improve only medium but not bad, BUT bad likely needs explicit QRS detectability damage rather than more noise.",
        ]
    )
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    (Path(args.report_root) / "domain_gap_reflection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_final_report(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    rows = read_summary_rows(out_root / "adaptation_summary.jsonl")
    rows.sort(key=selection_score, reverse=True)
    supervised = read_json(out_root / "but_supervised_summary.json") if (out_root / "but_supervised_summary.json").exists() else {"best": None}
    lines = [
        "# E3.11f BUT-Like Adaptation 14h Report",
        "",
        "This report separates strict synthetic zero-shot adaptation from BUT-supervised adaptation.",
        "",
        "## Strict Synthetic Track",
        "",
        "| rank | mode | spec | family | BUT acc | BUT macro-F1 | BUT bad recall | PTB acc |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows[:15], 1):
        but = row.get("but_test_report", {})
        ptb = row.get("ptb_test_report", {})
        rec = but.get("recall_good_medium_bad") or [0, 0, 0]
        lines.append(
            f"| {i} | {row.get('mode')} | `{row.get('spec', {}).get('id', '')}` | {row.get('spec', {}).get('family', '')} | "
            f"{float(but.get('acc', 0.0)):.4f} | {float(but.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[2] if len(rec) > 2 else 0.0):.4f} | {float(ptb.get('acc', 0.0)):.4f} |"
        )
    lines.extend(["", "## BUT-Supervised Track", ""])
    best = supervised.get("best")
    if best:
        rep = best["but_test_report"]
        rec = rep.get("recall_good_medium_bad") or [0, 0, 0]
        lines.append(
            f"Best supervised adaptation: `{best['checkpoint_tag']}` / `{best['feature_set']}` / `{best['model']}` "
            f"with BUT acc `{rep['acc']:.4f}`, macro-F1 `{rep['macro_f1']:.4f}`, bad recall `{rec[2]:.4f}`."
        )
    else:
        lines.append("BUT-supervised adaptation has not run yet.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Synthetic zero-shot gains support a better PTB data rule.",
            "- BUT-supervised gains support deployable transfer of the Uformer representation.",
            "- A >0.95 supervised result is not claimed as zero-shot unless the strict track also reaches it.",
        ]
    )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"status": "completed", "n_rows": len(rows), "best_synthetic": rows[0] if rows else None, "best_supervised": best}
    write_json(out_root / "final_report_summary.json", payload)
    write_json(report_root / "final_report_summary.json", payload)
    return payload


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "but_adaptation_state.json"
    update_state(state_path, status="running", started_at=now_iso(), args=vars(args))
    try:
        steps = []
        if args.stage in {"all", "audit"}:
            steps.append({"audit": make_domain_audit(args)})
            update_state(state_path, status="running", steps=steps)
        if args.stage in {"all", "make_grid"}:
            steps.append({"make_grid": generate_grid(args)})
            update_state(state_path, status="running", steps=steps)
        if args.stage in {"all", "quick_train"}:
            steps.append({"quick_train": run_quick_train(args)})
            update_state(state_path, status="running", steps=steps)
        if args.stage in {"all", "full_train"}:
            steps.append({"full_train": run_full_train(args)})
            update_state(state_path, status="running", steps=steps)
        if args.stage in {"all", "but_supervised"}:
            steps.append({"but_supervised": run_but_supervised(args)})
            update_state(state_path, status="running", steps=steps)
        if args.stage in {"all", "report"}:
            steps.append({"report": render_final_report(args)})
        state = update_state(state_path, status="completed", completed_at=now_iso(), steps=steps)
        return state
    except Exception as exc:
        update_state(state_path, status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=now_iso())
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT-like synthetic adaptation grid for E3.11f Uformer.")
    parser.add_argument("--stage", choices=("audit", "make_grid", "quick_train", "full_train", "but_supervised", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--realdata_root", default=str(DEFAULT_REALDATA_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--mainline_checkpoint", default=str(DEFAULT_MAINLINE_CKPT))
    parser.add_argument("--budget_hours", type=float, default=14.0)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--domain_score_sample_limit", type=int, default=2400)
    parser.add_argument("--quick_top_n", type=int, default=12)
    parser.add_argument("--full_top_n", type=int, default=4)
    parser.add_argument("--supervised_top_n", type=int, default=2)
    parser.add_argument("--quick_epochs_stage1", type=int, default=5)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--quick_batch_size_stage1", type=int, default=24)
    parser.add_argument("--quick_batch_size_stage2", type=int, default=32)
    parser.add_argument("--full_batch_size_stage1", type=int, default=24)
    parser.add_argument("--full_batch_size_stage2", type=int, default=32)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--class_weight", default="1,1.40,1.70")
    parser.add_argument("--force_data", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    state = run_all(args)
    print(json.dumps({"status": state["status"], "state": str(Path(args.out_root) / "but_adaptation_state.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
