"""10s BUT bad-boundary synthetic tuning.

This experiment keeps BUT evaluation on the 10 second protocol and changes only
the PTB synthetic bad/medium artifact rules.  The aim is to improve the bad
boundary without claiming 5s/ensemble protocol gains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import (
    CLASS_NAMES,
    ROOT,
    burst_noise,
    class_array,
    domain_distance,
    highpass_noise,
    link_or_copy,
    load_ptb_artifact,
    lowfreq_drift,
    normalize_noise,
    signal_features,
    smooth_rows,
    summarize_by_class,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_MAINLINE_CKPT,
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.run import (
    FS_TARGET,
    INT_TO_CLASS,
    N_TARGET,
    apply_but_thresholds,
    calibrate_but,
    load_uformer_model,
    multiclass_report,
    plot_wave_gallery,
    run_model_outputs,
)


RUN_TAG = "e311_but_bad_boundary_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_SOURCE_ARTIFACT = (
    ROOT
    / "outputs"
    / "experiment"
    / "e311_morph_denoise_gap5_7_grid"
    / "data"
    / "med6p25_badgap7_badcm0p75"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_state(path: Path, **kwargs: Any) -> None:
    state = read_json(path) if path.exists() else {"started_at": now_iso()}
    state.update(kwargs)
    write_json(path, state)


def bad_specs() -> list[dict[str, Any]]:
    base = {
        "qrs_attn_medium": 0.05,
        "qrs_attn_bad": 0.45,
        "tst_noise_medium": 0.10,
        "hf_medium": 0.04,
        "hf_bad": 0.10,
        "drift_medium": 0.05,
        "drift_bad": 0.12,
        "burst_bad": 0.08,
        "low_amp_medium": 0.94,
        "low_amp_bad": 0.72,
        "residual_scale_good": 0.70,
        "residual_scale_medium": 1.05,
        "residual_scale_bad": 1.30,
        "smooth_bad": 0.10,
        "contact_loss_bad": 0.18,
        "flatline_bad": 0.0,
        "dropout_bad": 0.0,
        "baseline_step_bad": 0.0,
        "clip_bad": 0.0,
        "spurious_peak_bad": 0.0,
    }
    variants = [
        ("b01_qrs_contact_mild", {"qrs_attn_bad": 0.55, "contact_loss_bad": 0.25, "low_amp_bad": 0.68}),
        ("b02_qrs_contact_strong", {"qrs_attn_bad": 0.72, "contact_loss_bad": 0.38, "low_amp_bad": 0.60, "smooth_bad": 0.22}),
        ("b03_flatline_contact", {"qrs_attn_bad": 0.62, "contact_loss_bad": 0.34, "flatline_bad": 0.35, "dropout_bad": 0.25, "low_amp_bad": 0.62}),
        ("b04_baseline_step_dropout", {"baseline_step_bad": 0.35, "dropout_bad": 0.45, "drift_bad": 0.20, "qrs_attn_bad": 0.58}),
        ("b05_clipping_lowamp", {"clip_bad": 0.45, "low_amp_bad": 0.48, "qrs_attn_bad": 0.50, "contact_loss_bad": 0.24}),
        ("b06_spurious_qrs_burst", {"spurious_peak_bad": 0.45, "burst_bad": 0.22, "hf_bad": 0.18, "qrs_attn_bad": 0.52}),
        ("b07_but_mixed_bad", {"qrs_attn_bad": 0.68, "contact_loss_bad": 0.32, "flatline_bad": 0.22, "baseline_step_bad": 0.22, "clip_bad": 0.20, "low_amp_bad": 0.58}),
        ("b08_bad_extreme_guarded_medium", {"qrs_attn_medium": 0.03, "tst_noise_medium": 0.13, "qrs_attn_bad": 0.78, "contact_loss_bad": 0.42, "flatline_bad": 0.30, "low_amp_bad": 0.55}),
        ("b09_medium_partial_bad_contact", {"qrs_attn_medium": 0.08, "tst_noise_medium": 0.18, "drift_medium": 0.08, "qrs_attn_bad": 0.65, "contact_loss_bad": 0.36, "dropout_bad": 0.22}),
        ("b10_all_bad_wearable", {"qrs_attn_bad": 0.70, "contact_loss_bad": 0.36, "flatline_bad": 0.25, "baseline_step_bad": 0.30, "clip_bad": 0.25, "spurious_peak_bad": 0.28, "burst_bad": 0.18, "hf_bad": 0.16, "low_amp_bad": 0.56}),
    ]
    out = []
    for name, changes in variants:
        spec = dict(base)
        spec.update(changes)
        spec["id"] = name
        spec["family"] = "bad_boundary_10s"
        out.append(spec)
    return out


def apply_bad_spec(ptb: dict[str, Any], spec: dict[str, Any], out_dir: Path, seed: int, sample_limit: int = 0) -> dict[str, Any]:
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
    x = clean + residual
    scale = np.ones(len(x), dtype=np.float32)
    scale[y == 0] = float(spec["residual_scale_good"])
    scale[y == 1] = float(spec["residual_scale_medium"])
    scale[y == 2] = float(spec["residual_scale_bad"])
    x = clean + residual * scale[:, None]
    medium = y == 1
    bad = y == 2
    if np.any(medium):
        x[medium] -= float(spec["qrs_attn_medium"]) * clean[medium] * qrs[medium]
        x[medium] += float(spec["tst_noise_medium"]) * np.std(clean[medium], axis=1, keepdims=True) * normalize_noise(rng.normal(size=(medium.sum(), N_TARGET)).astype(np.float32)) * tst[medium]
        x[medium] += float(spec["drift_medium"]) * np.std(clean[medium], axis=1, keepdims=True) * lowfreq_drift(rng, int(medium.sum()))
        x[medium] += float(spec["hf_medium"]) * np.std(clean[medium], axis=1, keepdims=True) * normalize_noise(highpass_noise(rng, int(medium.sum())))
        x[medium] *= float(spec["low_amp_medium"])
    if np.any(bad):
        xb = x[bad].copy()
        cb = clean[bad]
        qb = qrs[bad]
        crit = critical[bad]
        std = np.std(cb, axis=1, keepdims=True) + 1e-6
        xb -= float(spec["qrs_attn_bad"]) * cb * qb
        if float(spec["smooth_bad"]) > 0:
            sm = smooth_rows(xb, int(5 + 32 * float(spec["smooth_bad"])))
            xb = xb * (1.0 - crit * float(spec["smooth_bad"])) + sm * crit * float(spec["smooth_bad"])
        xb += float(spec["drift_bad"]) * std * lowfreq_drift(rng, int(bad.sum()))
        xb += float(spec["hf_bad"]) * std * normalize_noise(highpass_noise(rng, int(bad.sum())))
        xb += float(spec["burst_bad"]) * std * burst_noise(rng, int(bad.sum()), 1.0)
        if float(spec["contact_loss_bad"]) > 0:
            gate = np.clip(qb + crit, 0.0, 1.0)
            xb *= 1.0 - float(spec["contact_loss_bad"]) * gate
        if float(spec["dropout_bad"]) > 0 or float(spec["flatline_bad"]) > 0:
            for i in range(xb.shape[0]):
                if rng.random() < max(float(spec["dropout_bad"]), float(spec["flatline_bad"])):
                    width = int(rng.integers(FS_TARGET // 2, FS_TARGET * 4))
                    start = int(rng.integers(0, max(1, N_TARGET - width)))
                    level = float(np.median(xb[i]))
                    if rng.random() < float(spec["flatline_bad"]):
                        xb[i, start : start + width] = level + rng.normal(0.0, 0.015, size=width)
                    else:
                        xb[i, start : start + width] *= float(rng.uniform(0.05, 0.35))
        if float(spec["baseline_step_bad"]) > 0:
            for i in range(xb.shape[0]):
                if rng.random() < float(spec["baseline_step_bad"]):
                    pos = int(rng.integers(FS_TARGET, N_TARGET - FS_TARGET))
                    amp = float(rng.normal(0.0, 0.28)) * float(std[i, 0])
                    xb[i, pos:] += amp
        if float(spec["clip_bad"]) > 0:
            for i in range(xb.shape[0]):
                if rng.random() < float(spec["clip_bad"]):
                    hi = float(np.percentile(np.abs(xb[i]), rng.uniform(58, 78)))
                    xb[i] = np.clip(xb[i], -hi, hi)
        if float(spec["spurious_peak_bad"]) > 0:
            for i in range(xb.shape[0]):
                n_peaks = int(rng.integers(1, 5))
                for _ in range(n_peaks):
                    if rng.random() < float(spec["spurious_peak_bad"]):
                        width = int(rng.integers(3, 14))
                        center = int(rng.integers(width + 1, N_TARGET - width - 1))
                        pulse = np.hanning(width * 2 + 1).astype(np.float32)
                        amp = float(rng.normal(0.0, 0.65)) * float(std[i, 0])
                        xb[i, center - width : center + width + 1] += amp * pulse
        xb *= float(spec["low_amp_bad"])
        x[bad] = xb
    x = x.astype(np.float32)
    labels["adaptation_spec_id"] = str(spec["id"])
    labels["adaptation_family"] = str(spec["family"])
    labels["sample_source"] = labels.get("sample_source", "ptb_synthetic").astype(str) + f"|bad_boundary:{spec['id']}"
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
    level = source_data_dir / "synth_10s_125hz_noise_level.npz"
    if level.exists():
        link_or_copy(level, data_dir / level.name)
    audit = {
        "spec": spec,
        "signals_shape": list(x.shape),
        "split_counts": {str(k): int(v) for k, v in labels["split"].value_counts().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in labels["y_class"].value_counts().to_dict().items()},
    }
    write_json(out_dir / "data_variant_audit.json", audit)
    return {"audit": audit, "X_noisy": x, "labels": labels}


def plot_synthetic_gallery(clean: np.ndarray, noisy: np.ndarray, labels: pd.DataFrame, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(9, 1, figsize=(13, 12), sharex=True)
    t = np.arange(clean.shape[1]) / FS_TARGET
    positions: list[int] = []
    for cls in CLASS_NAMES:
        idx = labels.index[labels["y_class"].astype(str) == cls].tolist()[:3]
        positions.extend(idx)
    for ax, pos in zip(axes, positions):
        ax.plot(t, clean[pos], color="#111827", lw=0.7, label="clean")
        ax.plot(t, noisy[pos], color="#dc2626", lw=0.55, alpha=0.9, label="synthetic noisy")
        ax.set_title(f"{labels.loc[pos, 'y_class']} idx={labels.loc[pos, 'idx']}", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def ensure_but_10s(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    but_dir = Path(args.protocol_out_root) / "protocols" / "p1_current_10s_center"
    if not but_dir.exists():
        raise FileNotFoundError(f"Missing 10s protocol output: {but_dir}")
    X = np.load(but_dir / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(but_dir / "metadata.csv")
    return X, meta


def score_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    update_state(out_root / "bad_boundary_state.json", status="scoring_specs", updated_at=now_iso())
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    ptb["source"] = str(args.source_artifact_dir)
    but_X, but_meta = ensure_but_10s(args)
    but_summary = summarize_by_class(signal_features(but_X), but_meta["y_class"])
    clean = ptb["clean"].astype(np.float32, copy=False)
    rows = []
    for spec in bad_specs():
        variant_dir = out_root / "synthetic_variants" / spec["id"]
        payload = apply_bad_spec(ptb, spec, variant_dir, seed=int(args.seed), sample_limit=int(args.score_sample_limit))
        X = payload["X_noisy"]
        labels = payload["labels"]
        cand_summary = summarize_by_class(signal_features(X), labels["y_class"])
        dist = domain_distance(cand_summary, but_summary)
        row = {
            "spec": spec,
            "domain_distance": float(dist),
            "bad_qrs_prominence_p50": float(cand_summary["bad"]["qrs_prominence"]["p50"]),
            "bad_rms_p50": float(cand_summary["bad"]["rms"]["p50"]),
            "bad_drift_p50": float(cand_summary["bad"]["drift_ratio"]["p50"]),
            "variant_dir": str(variant_dir),
        }
        rows.append(row)
        append_jsonl(out_root / "bad_boundary_specs_scored.jsonl", row)
        plot_synthetic_gallery(clean[: len(X)], X, labels.reset_index(drop=True), variant_dir / "visuals" / "synthetic_gallery.png", spec["id"])
    rows.sort(key=lambda r: r["domain_distance"])
    write_json(out_root / "bad_boundary_specs_ranked.json", {"rows": rows})
    report_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BUT 10s Bad-Boundary Synthetic Specs",
        "",
        "Lower domain distance means the generated PTB synthetic statistics are closer to BUT 10s class statistics.",
        "",
        "| rank | spec | distance | bad qrs p50 | bad rms p50 | bad drift p50 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row['spec']['id']} | {row['domain_distance']:.4f} | "
            f"{row['bad_qrs_prominence_p50']:.3f} | {row['bad_rms_p50']:.3f} | {row['bad_drift_p50']:.4f} |"
        )
    (report_root / "bad_boundary_specs_ranked.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "bad_boundary_specs_ranked.json", {"rows": rows})
    update_state(out_root / "bad_boundary_state.json", status="specs_scored", updated_at=now_iso(), n_specs=len(rows))
    return rows


def evaluate_checkpoint_10s(args: argparse.Namespace, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    X, meta = ensure_but_10s(args)
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_uformer_model(checkpoint, device, feature_set="full_tokens", with_head=True)
    outputs = run_model_outputs(model, X, int(args.batch_size_eval), device)
    probs = outputs["probs"]
    denoise = outputs["denoise"]
    assert isinstance(probs, np.ndarray)
    assert isinstance(denoise, np.ndarray)
    cal = calibrate_but(probs[split == "val"], y[split == "val"])
    pred_test = apply_but_thresholds(probs[split == "test"], cal["t_good"], cal["t_bad"])
    pred_all = apply_but_thresholds(probs, cal["t_good"], cal["t_bad"])
    report = {
        "checkpoint": str(checkpoint),
        "calibration": cal,
        "but_10s_test_report": multiclass_report(y[split == "test"], pred_test, probs[split == "test"]),
        "but_10s_all_report": multiclass_report(y, pred_all, probs),
        "elapsed_sec": float(outputs["elapsed_sec"]),
        "peak_cuda_memory_bytes": int(outputs["peak_cuda_memory_bytes"]),
    }
    out_dir = run_dir / "but_10s_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "but_10s_eval_summary.json", report)
    meta2 = meta.copy()
    meta2["pred_class"] = [INT_TO_CLASS[int(v)] for v in pred_all]
    cases = {
        "bad_missed": (meta2["y"].to_numpy() == 2) & (pred_all != 2),
        "correct_bad": (meta2["y"].to_numpy() == 2) & (pred_all == 2),
        "medium_confused": (meta2["y"].to_numpy() == 1) & (pred_all != 1),
        "good_false_bad": (meta2["y"].to_numpy() == 0) & (pred_all == 2),
    }
    for name, mask in cases.items():
        pos = np.where(mask)[0][:18].tolist()
        if pos:
            plot_wave_gallery(X, meta2, pos, out_dir / "visuals" / f"{name}.png", f"BUT 10s {name}", denoise=denoise, probs=probs)
    return report


def train_one(args: argparse.Namespace, row: dict[str, Any], mode: str) -> dict[str, Any]:
    spec = row["spec"]
    spec_id = str(spec["id"])
    variant_dir = Path(row["variant_dir"])
    if not (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists():
        ptb = load_ptb_artifact(Path(args.source_artifact_dir))
        ptb["source"] = str(args.source_artifact_dir)
        apply_bad_spec(ptb, spec, variant_dir, seed=int(args.seed))
    run_dir = Path(args.out_root) / "runs" / mode / spec_id
    log_dir = Path(args.out_root) / "logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    if mode == "quick":
        e1, e2 = int(args.quick_epochs_stage1), int(args.quick_epochs_stage2)
    else:
        e1, e2 = int(args.full_epochs_stage1), int(args.full_epochs_stage2)
    cmd = [
        sys.executable,
        "-m",
        "src.transformer_pipeline.train_uformer_mainline",
        "--stage",
        "all",
        "--source_artifact_dir",
        str(variant_dir),
        "--output_dir",
        str(run_dir),
        "--epochs_stage1",
        str(e1),
        "--epochs_stage2",
        str(e2),
        "--batch_size_stage1",
        str(args.batch_size_stage1),
        "--batch_size_stage2",
        str(args.batch_size_stage2),
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
    with (log_dir / f"{spec_id}.stdout.txt").open("w", encoding="utf-8") as out, (log_dir / f"{spec_id}.stderr.txt").open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    payload: dict[str, Any] = {
        "spec": spec,
        "mode": mode,
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "bad_boundary_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / "bad_boundary_training_summary.jsonl", payload)
    return payload


def run_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    state = out_root / "bad_boundary_state.json"
    update_state(state, status="quick_training", updated_at=now_iso())
    ranked_path = out_root / "bad_boundary_specs_ranked.json"
    rows = read_json(ranked_path)["rows"] if ranked_path.exists() else score_specs(args)
    # Train both the top domain-match specs and explicit bad-heavy specs.
    selected = rows[: int(args.top_by_distance)]
    by_id = {r["spec"]["id"]: r for r in rows}
    for spec_id in ["b03_flatline_contact", "b07_but_mixed_bad", "b08_bad_extreme_guarded_medium", "b10_all_bad_wearable"]:
        if spec_id in by_id and by_id[spec_id] not in selected:
            selected.append(by_id[spec_id])
    selected = selected[: int(args.max_quick_runs)]
    results = [train_one(args, row, "quick") for row in selected]
    update_state(state, status="quick_complete", updated_at=now_iso(), n_quick=len(results))
    write_report(args, results)
    return results


def result_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    ptb = row.get("ptb_test_report", {})
    ptb_acc = float(ptb.get("acc", 0.0))
    return (float(rec[2]), float(rep.get("balanced_acc", 0.0)), float(rep.get("macro_f1", 0.0)), ptb_acc)


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    report_root = Path(args.report_root)
    out_root = Path(args.out_root)
    if rows is None:
        summary_path = out_root / "bad_boundary_training_summary.jsonl"
        rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if summary_path.exists() else []
    report_root.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=result_score, reverse=True)
    lines = [
        "# BUT 10s Bad-Boundary Tuning",
        "",
        "Formal protocol is fixed to 10s (`p1_current_10s_center`).  These runs change PTB synthetic bad/medium artifact rules, then train Uformer quick recipes.",
        "",
        "## Results Ranked By Bad Recall",
        "",
        "| rank | spec | return | BUT acc | BUT bal | BUT macro-F1 | BUT recalls good/medium/bad | PTB acc | PTB bad recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for i, row in enumerate(sorted_rows, start=1):
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        lines.append(
            f"| {i} | {row.get('spec', {}).get('id')} | {row.get('returncode')} | "
            f"{float(rep.get('acc', 0.0)):.4f} | {float(rep.get('balanced_acc', 0.0)):.4f} | {float(rep.get('macro_f1', 0.0)):.4f} | "
            f"{rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f} | {float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec)>2 else 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- A useful direction must beat the 10s balanced baseline bad recall 0.805 without collapsing medium.",
            "- If BUT bad recall improves only by making every flat/low-amplitude segment bad, check good false-bad galleries before promotion.",
            "- If PTB acc falls below 0.975 or PTB bad recall below 0.985, keep the spec as diagnostic only.",
        ]
    )
    (report_root / "bad_boundary_tuning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "bad_boundary_tuning_summary.json", {"rows": rows, "ranked": sorted_rows})


def run_all(args: argparse.Namespace) -> None:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    update_state(Path(args.out_root) / "bad_boundary_state.json", status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"all", "score_specs"}:
        score_specs(args)
    if args.stage in {"all", "quick_train"}:
        run_quick(args)
    if args.stage in {"all", "report"}:
        write_report(args)
    update_state(Path(args.out_root) / "bad_boundary_state.json", status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s bad-boundary synthetic tuning.")
    parser.add_argument("--stage", choices=("score_specs", "quick_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MAINLINE_CKPT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score_sample_limit", type=int, default=0)
    parser.add_argument("--top_by_distance", type=int, default=4)
    parser.add_argument("--max_quick_runs", type=int, default=6)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=3)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--class_weight", default="1,1.40,1.90")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_all(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / "bad_boundary_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
