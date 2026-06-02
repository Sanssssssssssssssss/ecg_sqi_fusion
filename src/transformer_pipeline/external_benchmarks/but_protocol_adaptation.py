"""Protocol-first BUT QDB adaptation audit for E3.11f.

This is experiment-only code.  It checks whether BUT QDB performance is being
limited by window protocol and label purity before launching another synthetic
generator grid.  It never touches ``src/sqi_pipeline`` or the mainline Uformer
checkpoint.
"""

from __future__ import annotations

import argparse
import json
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
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.run import (
    CLASS_TO_INT,
    FS_TARGET,
    N_TARGET,
    apply_but_thresholds,
    basic_sqi_features,
    calibrate_but,
    extract_uformer_features,
    load_uformer_model,
    multiclass_report,
    plot_wave_gallery,
    run_model_outputs,
)


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_protocol_adaptation_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_REALDATA_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02"
DEFAULT_MAINLINE_CKPT = ROOT / "outputs" / "mainline" / "e311_uformer_full_tokens_detach_seed0" / "ckpt_best.pt"
DEFAULT_SOURCE_ARTIFACT = (
    ROOT
    / "outputs"
    / "experiment"
    / "e311_morph_denoise_gap5_7_grid"
    / "data"
    / "med6p25_badgap7_badcm0p75"
)
CLASS_NAMES = ("good", "medium", "bad")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_state(path: Path, **kwargs: Any) -> dict[str, Any]:
    state = read_json(path) if path.exists() else {"started_at": now_iso(), "steps": []}
    state.update(kwargs)
    write_json(path, state)
    return state


def split_counts(meta: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": int(len(meta)),
        "split_counts": {str(k): int(v) for k, v in meta["split"].value_counts().sort_index().items()},
        "class_counts": {str(k): int(v) for k, v in meta["y_class"].value_counts().sort_index().items()},
        "split_class_counts": {
            str(split): {str(k): int(v) for k, v in sub["y_class"].value_counts().sort_index().items()}
            for split, sub in meta.groupby("split")
        },
    }


def ensure_but_processed(out_root: Path, realdata_root: Path) -> Path:
    src = realdata_root / "processed" / "butqdb"
    if not src.exists():
        raise FileNotFoundError(f"Processed BUT directory is missing: {src}")
    dst = out_root / "processed" / "butqdb_current_10s"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    return dst


def pad_5s_to_10s(x5: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Pad a 5 second, 625 sample crop back to the 10 second model input."""

    if x5.shape[-1] != N_TARGET // 2:
        raise ValueError(f"Expected 5s crop length {N_TARGET // 2}, got {x5.shape[-1]}")
    if mode == "repeat":
        return np.concatenate([x5, x5], axis=-1)
    if mode == "zero":
        out = np.zeros((x5.shape[0], N_TARGET), dtype=x5.dtype)
        out[:, : x5.shape[-1]] = x5
        return out
    return np.pad(x5, ((0, 0), (0, N_TARGET - x5.shape[-1])), mode="reflect")


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    kind: str
    description: str
    crop_offsets: tuple[int, ...] = ()
    ensemble_offsets: tuple[int, ...] = ()
    purity_threshold: float | None = None
    pad_mode: str = "reflect"


PROTOCOLS = [
    ProtocolSpec("p1_current_10s_center", "current_10s", "Current 10s consensus-window protocol."),
    ProtocolSpec(
        "p2_10s_purity90",
        "current_10s",
        "10s with purity >=0.90. Existing processed BUT windows are segment-pure by construction.",
        purity_threshold=0.90,
    ),
    ProtocolSpec("p3_5s_center_pad10", "crop_5s", "Center 5s crop padded back to 10s.", crop_offsets=(N_TARGET // 4,)),
    ProtocolSpec(
        "p4_5s_purity90_pad10",
        "crop_5s",
        "5s center crop with purity >=0.90, padded to 10s. Purity inherited from consensus 10s base.",
        crop_offsets=(N_TARGET // 4,),
        purity_threshold=0.90,
    ),
    ProtocolSpec(
        "p5_5s_stride2p5_pad10",
        "crop_5s",
        "5s crops at 0/2.5/5s stride, padded to 10s and evaluated as windows.",
        crop_offsets=(0, N_TARGET // 4, N_TARGET // 2),
        purity_threshold=0.90,
    ),
    ProtocolSpec(
        "p6_two_5s_crops_ensemble",
        "ensemble_5s",
        "Two 5s crops from each 10s window; features/probabilities are averaged at the original-window level.",
        ensemble_offsets=(0, N_TARGET // 2),
        purity_threshold=0.90,
    ),
]


def load_current_but(processed_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(processed_dir / "signals.npz")["X"].astype(np.float32)
    meta = pd.read_csv(processed_dir / "metadata.csv")
    meta["y"] = meta["y"].astype(int)
    return X, meta


def make_protocol_dataset(
    spec: ProtocolSpec,
    X_base: np.ndarray,
    meta_base: pd.DataFrame,
    protocol_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = protocol_root / spec.name
    signals_path = out_dir / "signals.npz"
    meta_path = out_dir / "metadata.csv"
    audit_path = out_dir / "protocol_audit.json"
    if signals_path.exists() and meta_path.exists() and audit_path.exists() and not force:
        return read_json(audit_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_base.copy()
    meta["protocol"] = spec.name
    meta["protocol_description"] = spec.description
    meta["label_purity"] = 1.0
    if spec.purity_threshold is not None:
        meta = meta[meta["label_purity"] >= float(spec.purity_threshold)].copy()
    base_idx = meta.index.to_numpy()
    X_sel = X_base[base_idx]
    if spec.kind == "current_10s":
        meta["crop_offset_sec"] = 0.0
        np.savez_compressed(signals_path, X=X_sel.astype(np.float32))
    elif spec.kind == "crop_5s":
        rows: list[np.ndarray] = []
        metas: list[pd.DataFrame] = []
        for offset in spec.crop_offsets:
            crop = X_sel[:, :, offset : offset + N_TARGET // 2]
            padded = np.stack([pad_5s_to_10s(c, spec.pad_mode) for c in crop], axis=0)
            rows.append(padded.astype(np.float32))
            part = meta.copy()
            part["window_id"] = part["window_id"].astype(str) + f"_crop{offset}"
            part["crop_offset_sec"] = float(offset / FS_TARGET)
            part["padded"] = True
            metas.append(part)
        X_out = np.concatenate(rows, axis=0)
        meta = pd.concat(metas, ignore_index=True)
        np.savez_compressed(signals_path, X=X_out)
    elif spec.kind == "ensemble_5s":
        if len(spec.ensemble_offsets) != 2:
            raise ValueError("Only two-crop ensemble is supported.")
        offset_a, offset_b = spec.ensemble_offsets
        crop_a = X_sel[:, :, offset_a : offset_a + N_TARGET // 2]
        crop_b = X_sel[:, :, offset_b : offset_b + N_TARGET // 2]
        X_a = np.stack([pad_5s_to_10s(c, spec.pad_mode) for c in crop_a], axis=0).astype(np.float32)
        X_b = np.stack([pad_5s_to_10s(c, spec.pad_mode) for c in crop_b], axis=0).astype(np.float32)
        meta["crop_offset_sec"] = "ensemble_0_5"
        meta["padded"] = True
        np.savez_compressed(signals_path, X_first=X_a, X_second=X_b)
    else:
        raise ValueError(f"Unsupported protocol kind: {spec.kind}")
    meta.to_csv(meta_path, index=False)
    audit = {
        "protocol": spec.name,
        "kind": spec.kind,
        "description": spec.description,
        "purity_threshold": spec.purity_threshold,
        "note": "Purity is inherited from existing BUT consensus segments; this stage tests crop/window protocol before regenerating raw BUT windows.",
        "signals_npz": str(signals_path),
        **split_counts(meta),
    }
    write_json(audit_path, audit)
    gallery_dir = out_dir / "visuals"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    if spec.kind == "ensemble_5s":
        X_gallery = np.load(signals_path)["X_first"].astype(np.float32)
    else:
        X_gallery = np.load(signals_path)["X"].astype(np.float32)
    for label in CLASS_NAMES:
        positions = meta.index[meta["y_class"] == label].to_list()[:18]
        if positions:
            plot_wave_gallery(X_gallery, meta.reset_index(drop=True), positions, gallery_dir / f"{label}_processed_gallery.png", f"{spec.name} {label}")
    return audit


def load_protocol_signals(protocol_dir: Path) -> tuple[np.ndarray | tuple[np.ndarray, np.ndarray], pd.DataFrame, bool]:
    meta = pd.read_csv(protocol_dir / "metadata.csv")
    npz = np.load(protocol_dir / "signals.npz")
    if "X" in npz:
        return npz["X"].astype(np.float32), meta, False
    return (npz["X_first"].astype(np.float32), npz["X_second"].astype(np.float32)), meta, True


def classifier_probs(clf: Pipeline, X: np.ndarray) -> np.ndarray:
    model = clf.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return clf.predict_proba(X).astype(np.float32)
    if hasattr(model, "decision_function"):
        z = clf.decision_function(X)
        if z.ndim == 1:
            z = np.stack([-z, z, np.zeros_like(z)], axis=1)
        return softmax(z, axis=1).astype(np.float32)
    pred = clf.predict(X).astype(np.int64)
    probs = np.zeros((len(pred), 3), dtype=np.float32)
    probs[np.arange(len(pred)), pred] = 1.0
    return probs


def feature_matrix_for_protocol(
    protocol_dir: Path,
    checkpoint: Path,
    feature_set: str,
    batch_size: int,
    device: torch.device,
    force: bool = False,
) -> np.ndarray:
    feature_dir = protocol_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{feature_set}.npy"
    if path.exists() and not force:
        return np.load(path).astype(np.float32)
    X_obj, _, is_ensemble = load_protocol_signals(protocol_dir)
    if feature_set == "handcrafted_sqi":
        if is_ensemble:
            X_a, X_b = X_obj  # type: ignore[misc]
            feats = 0.5 * (basic_sqi_features(X_a) + basic_sqi_features(X_b))
        else:
            feats = basic_sqi_features(X_obj)  # type: ignore[arg-type]
    else:
        if is_ensemble:
            X_a, X_b = X_obj  # type: ignore[misc]
            feats = 0.5 * (
                extract_uformer_features(checkpoint, X_a, feature_set, batch_size, device)
                + extract_uformer_features(checkpoint, X_b, feature_set, batch_size, device)
            )
        else:
            feats = extract_uformer_features(checkpoint, X_obj, feature_set, batch_size, device)  # type: ignore[arg-type]
    np.save(path, feats.astype(np.float32))
    return feats.astype(np.float32)


def zero_shot_for_protocol(
    protocol_dir: Path,
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
    force: bool = False,
) -> dict[str, Any]:
    result_dir = protocol_dir / "zero_shot"
    summary_path = result_dir / "zero_shot_report.json"
    if summary_path.exists() and not force:
        return read_json(summary_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    X_obj, meta, is_ensemble = load_protocol_signals(protocol_dir)
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    model = load_uformer_model(checkpoint, device, feature_set="full_tokens", with_head=True)
    if is_ensemble:
        X_a, X_b = X_obj  # type: ignore[misc]
        out_a = run_model_outputs(model, X_a, batch_size, device)
        out_b = run_model_outputs(model, X_b, batch_size, device)
        probs = 0.5 * (out_a["probs"] + out_b["probs"])  # type: ignore[operator]
        denoise = out_a["denoise"]
    else:
        out = run_model_outputs(model, X_obj, batch_size, device)  # type: ignore[arg-type]
        probs = out["probs"]  # type: ignore[assignment]
        denoise = out["denoise"]
    raw_pred = np.asarray(probs).argmax(axis=1)
    val = split == "val"
    test = split == "test"
    cal = calibrate_but(np.asarray(probs)[val], y[val])
    cal_pred = apply_but_thresholds(np.asarray(probs)[test], cal["t_good"], cal["t_bad"])
    payload = {
        "protocol": protocol_dir.name,
        "mode": "zero_shot_current_mainline",
        "raw_test_report": multiclass_report(y[test], raw_pred[test], np.asarray(probs)[test]),
        "calibration": cal,
        "calibrated_test_report": multiclass_report(y[test], cal_pred, np.asarray(probs)[test]),
    }
    write_json(summary_path, payload)
    np.savez_compressed(result_dir / "zero_shot_outputs.npz", probs=np.asarray(probs, dtype=np.float32), denoise=np.asarray(denoise, dtype=np.float32))
    return payload


def run_protocol_probe(
    protocol_dir: Path,
    checkpoint: Path,
    feature_set: str,
    model_name: str,
    batch_size: int,
    device: torch.device,
    force: bool = False,
) -> dict[str, Any]:
    result_dir = protocol_dir / "probe_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / f"{feature_set}_{model_name}.json"
    if report_path.exists() and not force:
        return read_json(report_path)
    _, meta, _ = load_protocol_signals(protocol_dir)
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    train = split == "train"
    val = split == "val"
    test = split == "test"
    feats = feature_matrix_for_protocol(protocol_dir, checkpoint, feature_set, batch_size, device, force=force)
    factories = {
        "logreg": lambda: LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs"),
        "linear_svm": lambda: LinearSVC(class_weight="balanced", max_iter=10000),
        "small_mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(96,),
            activation="relu",
            alpha=1e-4,
            max_iter=600,
            early_stopping=True,
            random_state=0,
        ),
    }
    clf = Pipeline([("scaler", StandardScaler()), ("model", factories[model_name]())])
    clf.fit(feats[train], y[train])
    probs_val = classifier_probs(clf, feats[val])
    probs_test = classifier_probs(clf, feats[test])
    raw_pred = probs_test.argmax(axis=1)
    cal = calibrate_but(probs_val, y[val])
    cal_pred = apply_but_thresholds(probs_test, cal["t_good"], cal["t_bad"])
    payload = {
        "protocol": protocol_dir.name,
        "feature_set": feature_set,
        "model": model_name,
        "n_features": int(feats.shape[1]),
        "raw_test_report": multiclass_report(y[test], raw_pred, probs_test),
        "calibration": cal,
        "calibrated_test_report": multiclass_report(y[test], cal_pred, probs_test),
    }
    write_json(report_path, payload)
    np.savez_compressed(result_dir / f"{feature_set}_{model_name}_predictions.npz", probs=probs_test.astype(np.float32), y=y[test])
    return payload


def best_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rep = row.get("calibrated_test_report", {})
    recalls = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    return (
        float(rep.get("acc", 0.0)),
        float(rep.get("macro_f1", 0.0)),
        float(rep.get("balanced_acc", 0.0)),
        float(recalls[2] if len(recalls) > 2 else 0.0),
    )


def write_protocol_sweep_summary(rows: list[dict[str, Any]], report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    best = sorted(rows, key=best_key, reverse=True)
    lines = [
        "# E3.11f BUT Protocol Sweep",
        "",
        "This sweep is protocol-first: no Uformer denoiser training is performed here.  Calibration uses validation split only.",
        "",
        "| rank | protocol | mode | feature/model | acc | bal acc | macro-F1 | recalls good/medium/bad |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(best[:30], start=1):
        rep = row["calibrated_test_report"]
        rec = rep["recall_good_medium_bad"]
        fm = row.get("feature_set", "mainline") + "/" + row.get("model", "zero_shot")
        lines.append(
            f"| {rank} | {row['protocol']} | {row.get('mode', 'probe')} | {fm} | "
            f"{rep['acc']:.4f} | {rep['balanced_acc']:.4f} | {rep['macro_f1']:.4f} | "
            f"{rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f} |"
        )
    by_protocol = {}
    for row in rows:
        by_protocol.setdefault(row["protocol"], []).append(row)
    lines.extend(["", "## Best By Protocol", "", "| protocol | best feature/model | acc | bal acc | macro-F1 | bad recall |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for protocol, items in sorted(by_protocol.items()):
        row = max(items, key=best_key)
        rep = row["calibrated_test_report"]
        rec = rep["recall_good_medium_bad"]
        fm = row.get("feature_set", "mainline") + "/" + row.get("model", "zero_shot")
        lines.append(f"| {protocol} | {fm} | {rep['acc']:.4f} | {rep['balanced_acc']:.4f} | {rep['macro_f1']:.4f} | {rec[2]:.4f} |")
    (report_root / "protocol_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "protocol_sweep_summary.json", {"runs": rows, "best": best[:20]})


def run_protocol_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    state_path = out_root / "but_protocol_state.json"
    update_state(state_path, status="protocol_sweep_running", updated_at=now_iso())
    processed = ensure_but_processed(out_root, Path(args.realdata_root))
    X_base, meta_base = load_current_but(processed)
    protocol_root = out_root / "protocols"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows: list[dict[str, Any]] = []
    feature_sets = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
    models = [s.strip() for s in args.probe_models.split(",") if s.strip()]
    for spec in PROTOCOLS:
        audit = make_protocol_dataset(spec, X_base, meta_base, protocol_root, force=args.force)
        write_json(report_root / "protocol_audits" / f"{spec.name}.json", audit)
        protocol_dir = protocol_root / spec.name
        z = zero_shot_for_protocol(protocol_dir, Path(args.checkpoint), int(args.batch_size), device, force=args.force)
        rows.append(z)
        append_jsonl(out_root / "but_protocol_summary.jsonl", z)
        for feature_set in feature_sets:
            for model_name in models:
                row = run_protocol_probe(
                    protocol_dir,
                    Path(args.checkpoint),
                    feature_set,
                    model_name,
                    int(args.batch_size),
                    device,
                    force=args.force,
                )
                rows.append(row)
                append_jsonl(out_root / "but_protocol_summary.jsonl", row)
    write_protocol_sweep_summary(rows, report_root)
    update_state(state_path, status="protocol_sweep_complete", updated_at=now_iso(), protocol_rows=len(rows))
    return {"runs": rows}


def signal_metrics(X: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for sample in X[:, 0]:
        y = sample.astype(np.float64)
        dy = np.diff(y)
        centered = y - np.median(y)
        rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
        low_amp_frac = float(np.mean(np.abs(centered) < 0.05))
        flatline = float(np.max([np.mean(np.abs(centered[i : i + 125]) < 0.03) for i in range(0, max(1, len(centered) - 125), 25)]))
        qrs_prom = float((np.percentile(np.abs(y), 99) - np.percentile(np.abs(y), 75)) / (np.std(y) + 1e-6))
        jump = float(np.max(np.abs(np.diff(pd.Series(y).rolling(63, min_periods=1, center=True).mean().to_numpy()))))
        hf = float(np.std(dy) / (np.std(y) + 1e-6))
        clip = float(max(np.mean(np.isclose(y, np.max(y), atol=0.02)), np.mean(np.isclose(y, np.min(y), atol=0.02))))
        rows.append(
            {
                "rms": rms,
                "low_amp_frac": low_amp_frac,
                "flatline_frac": flatline,
                "qrs_prominence_proxy": qrs_prom,
                "baseline_jump_proxy": jump,
                "hf_ratio_proxy": hf,
                "clipping_frac": clip,
            }
        )
    return pd.DataFrame(rows)


def failure_tags(metrics: pd.Series, class_thresholds: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if metrics["baseline_jump_proxy"] > class_thresholds["baseline_jump"]:
        tags.append("baseline_jump")
    if metrics["rms"] < class_thresholds["low_rms"]:
        tags.append("low_amplitude")
    if metrics["flatline_frac"] > 0.35 or metrics["low_amp_frac"] > 0.60:
        tags.append("flatline")
    if metrics["clipping_frac"] > 0.02:
        tags.append("clipping")
    if metrics["hf_ratio_proxy"] > class_thresholds["hf_ratio"]:
        tags.append("burst_emg")
    if metrics["qrs_prominence_proxy"] < class_thresholds["qrs_prom"]:
        tags.append("qrs_low_prominence")
    if metrics["flatline_frac"] > 0.50 and metrics["rms"] < class_thresholds["low_rms"] * 1.2:
        tags.append("contact_loss")
    if not tags:
        tags.append("unclassified_visual")
    return tags


def pick_positions(mask: np.ndarray, limit: int = 24) -> list[int]:
    idx = np.flatnonzero(mask)
    return [int(i) for i in idx[:limit]]


def run_failure_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    state_path = out_root / "but_protocol_state.json"
    update_state(state_path, status="failure_audit_running", updated_at=now_iso())
    protocol_dir = out_root / "protocols" / args.audit_protocol
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Run protocol_sweep first; missing {protocol_dir}")
    X_obj, meta, is_ensemble = load_protocol_signals(protocol_dir)
    X = X_obj[0] if is_ensemble else X_obj  # type: ignore[index]
    zero = read_json(protocol_dir / "zero_shot" / "zero_shot_report.json")
    npz = np.load(protocol_dir / "zero_shot" / "zero_shot_outputs.npz")
    probs = npz["probs"]
    denoise = npz["denoise"]
    pred = apply_but_thresholds(probs, zero["calibration"]["t_good"], zero["calibration"]["t_bad"])
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    test = split == "test"
    metrics = signal_metrics(X)
    thresholds = {
        "baseline_jump": float(metrics["baseline_jump_proxy"].quantile(0.90)),
        "low_rms": float(metrics["rms"].quantile(0.12)),
        "hf_ratio": float(metrics["hf_ratio_proxy"].quantile(0.90)),
        "qrs_prom": float(metrics["qrs_prominence_proxy"].quantile(0.20)),
    }
    tag_rows: list[dict[str, Any]] = []
    for i, row in metrics.iterrows():
        tag_rows.append({"index": int(i), "tags": failure_tags(row, thresholds), **{k: float(v) for k, v in row.items()}})
    failures = {
        "bad_missed": test & (y == 2) & (pred != 2),
        "medium_confusion": test & (y == 1) & (pred != 1),
        "good_false_bad": test & (y == 0) & (pred == 2),
    }
    visual_dir = out_root / "visuals" / "failure_audit" / args.audit_protocol
    visual_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in failures.items():
        positions = pick_positions(mask, limit=24)
        if positions:
            plot_wave_gallery(X, meta, positions, visual_dir / f"{name}.png", f"{args.audit_protocol} {name}", denoise=denoise, probs=probs)
    tag_summary: dict[str, int] = {}
    for item in tag_rows:
        if not test[item["index"]]:
            continue
        for tag in item["tags"]:
            tag_summary[tag] = tag_summary.get(tag, 0) + 1
    payload = {
        "protocol": args.audit_protocol,
        "thresholds": thresholds,
        "failure_counts": {name: int(mask.sum()) for name, mask in failures.items()},
        "tag_summary_test": tag_summary,
        "galleries": {name: str(visual_dir / f"{name}.png") for name in failures},
    }
    write_json(out_root / "failure_taxonomy.json", {"summary": payload, "rows": tag_rows})
    write_json(report_root / "failure_taxonomy.json", {"summary": payload, "rows": tag_rows[:2000]})
    lines = [
        "# BUT Failure Taxonomy",
        "",
        f"Protocol: `{args.audit_protocol}`",
        "",
        "## Failure Counts",
        "",
    ]
    for name, count in payload["failure_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Tag Summary On Test", ""])
    for tag, count in sorted(tag_summary.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## Visuals", ""])
    for name, path in payload["galleries"].items():
        lines.append(f"- `{name}`: `{path}`")
    (report_root / "but_failure_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_state(state_path, status="failure_audit_complete", updated_at=now_iso())
    return payload


def write_generator_plan(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    specs = [
        {
            "name": "g0_current_generator_baseline",
            "purpose": "Baseline PTB synthetic generator; no BUT-style protocol changes.",
            "expected_value": "Separates protocol effects from generator effects.",
        },
        {
            "name": "g1_but_snr_table1",
            "purpose": "Good 16-18dB, medium 5-14dB, bad <= -3dB using BUT-like severity bands.",
            "expected_value": "Tests whether logit mismatch is mostly SNR calibration.",
        },
        {
            "name": "g2_good_lenient",
            "purpose": "Good can include mild noise if P/QRS/T remain clear.",
            "expected_value": "Reduces over-rejection of BUT good under real wearable noise.",
        },
        {
            "name": "g3_medium_partial_local",
            "purpose": "QRS detectable but P/T/ST local details unreliable.",
            "expected_value": "Targets BUT class-2 expert definition instead of middle SNR only.",
        },
        {
            "name": "g4_bad_qrs_unreliable",
            "purpose": "Bad has unreliable QRS: missing/spurious peaks, attenuation, broadening.",
            "expected_value": "Directly attacks current bad false negatives.",
        },
        {
            "name": "g5_bad_contact_loss",
            "purpose": "Adds flatline, clipping, contact loss, baseline step and dropout.",
            "expected_value": "Models BUT free-living wearable failures absent in PTB synthetic.",
        },
        {
            "name": "g6_cinc_noisy_bad",
            "purpose": "Adds burst/HF/motion unusable noisy patterns.",
            "expected_value": "Keeps CinC-style too-noisy rejection aligned with BUT bad.",
        },
        {
            "name": "g7_mixed_but_style",
            "purpose": "Mixture of G2-G6 with PTB balanced split preserved.",
            "expected_value": "Candidate for full Uformer training after head-only validation.",
        },
    ]
    payload = {
        "status": "planned_protocol_first",
        "note": "Stage 2 starts only after protocol_sweep_summary and failure_taxonomy are reviewed.",
        "source_artifact": str(Path(args.source_artifact_dir)),
        "specs": specs,
    }
    write_json(out_root / "generator_head_only_specs.json", payload)
    write_json(report_root / "generator_head_only_specs.json", payload)
    lines = [
        "# Generator / Head-Only Minimal Validation Plan",
        "",
        "This is deliberately held behind the protocol sweep.  The old v0 grid is superseded because it skipped this audit.",
        "",
        "| spec | purpose | value |",
        "| --- | --- | --- |",
    ]
    for spec in specs:
        lines.append(f"| {spec['name']} | {spec['purpose']} | {spec['expected_value']} |")
    (report_root / "generator_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def render_readme(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    lines = [
        "# E3.11f BUT Protocol-First Adaptation",
        "",
        "This run supersedes the previous 14h BUT grid until the BUT window protocol is audited.",
        "",
        "## Stages",
        "",
        "- Stage 0: protocol sweep for current 10s, 10s purity, 5s crops, 5s stride, and two-crop ensemble.",
        "- Stage 1: false-negative visual/failure taxonomy.",
        "- Stage 2: generator/head-only plan, gated on Stage 0/1 findings.",
        "",
        "## Guardrails",
        "",
        "- BUT mapping remains `1/2/3 -> good/medium/bad`.",
        "- Calibration is validation-only; test is never used for threshold selection.",
        "- `src/sqi_pipeline` and mainline checkpoints are not modified.",
        "",
        f"Output root: `{out_root}`",
    ]
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    render_readme(args)
    state_path = out_root / "but_protocol_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    result: dict[str, Any] = {}
    if args.stage in {"all", "protocol_sweep"}:
        result["protocol_sweep"] = run_protocol_sweep(args)
    if args.stage in {"all", "failure_audit"}:
        result["failure_audit"] = run_failure_audit(args)
    if args.stage in {"all", "head_only_plan", "report"}:
        result["head_only_plan"] = write_generator_plan(args)
    if args.stage in {"all", "report"}:
        render_readme(args)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso())
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run protocol-first BUT QDB adaptation audit.")
    parser.add_argument("--stage", choices=("protocol_sweep", "failure_audit", "head_only_plan", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--realdata_root", default=str(DEFAULT_REALDATA_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MAINLINE_CKPT))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_sets", default="full_tokens,bottleneck_only,summary_only,handcrafted_sqi")
    parser.add_argument("--probe_models", default="logreg,linear_svm,small_mlp")
    parser.add_argument("--audit_protocol", default="p1_current_10s_center")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_root = Path(args.out_root)
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        result = run_all(args)
        print(json.dumps(result if args.stage != "all" else {"status": "complete"}, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        update_state(out_root / "but_protocol_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
