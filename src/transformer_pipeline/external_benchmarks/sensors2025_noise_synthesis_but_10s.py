"""Sensors 2025-style ECG noise synthesis, then BUT 10s evaluation.

This runner reproduces only the data-generation idea from the Sensors 2025
paper: clean ECG is mixed with real NSTDB electrode-motion (EM), muscle
artifact (MA), and optional baseline wander (BW).  The target SNR is computed
against EM+MA only; BW is allowed as an extra overlay and is reported in the
all-noise measured SNR, but it does not participate in the target SNR
calibration.

The model and BUT protocol are unchanged.  This is experiment-only and never
touches src/sqi_pipeline or mainline checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
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

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import real_noise_snr_but_match_grid_10s as rn
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    evaluate_checkpoint_10s,
    now_iso,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.run import FS_TARGET, N_TARGET


RUN_TAG = "e311_sensors2025_noise_synthesis_but_10s_2026_06_05"
DEFAULT_OUT_ROOT = rn.ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = rn.ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_BUT_PROTOCOL = PROTOCOL_OUT_ROOT / "protocols" / "p1_current_10s_center"
DEFAULT_NSTDB_ROOT = rn.ROOT / "data" / "physionet" / "nstdb"
DEFAULT_CINC_DIR = rn.ROOT / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02" / "processed" / "cinc2017"

STATE_NAME = "sensors2025_state.json"
SPEC_NAME = "sensors2025_specs.json"
SCAN_NAME = "distance_leaderboard.csv"
SUMMARY_NAME = "sensors2025_training_summary.jsonl"
NOISE_NAMES = ("em", "ma", "bw")

RULE_ANCHOR = {
    "variant": "h_bad_rescue_05",
    "acc": 0.8229,
    "balanced_acc": 0.8177,
    "macro_f1": 0.7454,
    "recall_good_medium_bad": [0.887, 0.773, 0.793],
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(text: str) -> str:
    return (
        text.lower()
        .replace("+", "_")
        .replace("-", "m")
        .replace(".", "p")
        .replace(",", "_")
        .replace(" ", "_")
    )


def sample_snr(spec: dict[str, Any], y_class: str, rng: np.random.Generator) -> float:
    profile = spec["snr_profile"]
    lo, hi = [float(v) for v in profile[y_class]]
    sampler = str(profile.get("sampler", "uniform"))
    if sampler == "fixed_jitter":
        center = 0.5 * (lo + hi)
        return float(np.clip(rng.normal(center, max(0.05, (hi - lo) / 6.0)), lo, hi))
    if sampler == "triangular":
        return float(rng.triangular(lo, 0.5 * (lo + hi), hi))
    return float(rng.uniform(lo, hi))


def paper_snr_profiles() -> list[dict[str, Any]]:
    return [
        {
            "name": "paper_table_strict",
            "good": [16.0, 18.0],
            "medium": [5.0, 14.0],
            "bad": [-5.0, -3.0],
            "sampler": "uniform",
            "note": "Table-1-like bins: good >=16, medium 5-14, bad <=-3.",
        },
        {
            "name": "paper_denoise_interval",
            "good": [16.0, 18.0],
            "medium": [3.0, 16.0],
            "bad": [-5.0, 3.0],
            "sampler": "uniform",
            "note": "Denoising interval in the paper: roughly -5 to 18 dB with broad medium/bad overlap.",
        },
        {
            "name": "paper_fig2_like",
            "good": [17.5, 18.5],
            "medium": [5.5, 6.5],
            "bad": [-2.5, -1.5],
            "sampler": "fixed_jitter",
            "note": "Figure-2-style examples: good near 18 dB, medium near 6 dB, bad near -2 dB.",
        },
    ]


def paper_mix_profiles() -> list[dict[str, Any]]:
    bw_light = {"good": [0.002, 0.010], "medium": [0.010, 0.035], "bad": [0.020, 0.065]}
    bw_mid = {"good": [0.004, 0.020], "medium": [0.020, 0.060], "bad": [0.040, 0.100]}
    bw_strong_bad = {"good": [0.002, 0.012], "medium": [0.015, 0.045], "bad": [0.080, 0.160]}
    return [
        {"name": "em_only", "em_weight": 1.0, "ma_weight": 0.0, "bw_overlay": None},
        {"name": "ma_only", "em_weight": 0.0, "ma_weight": 1.0, "bw_overlay": None},
        {"name": "em_ma_equal", "em_weight": 0.5, "ma_weight": 0.5, "bw_overlay": None},
        {"name": "em_ma_ma_heavy", "em_weight": 0.30, "ma_weight": 0.70, "bw_overlay": None},
        {"name": "em_ma_bw_light", "em_weight": 0.5, "ma_weight": 0.5, "bw_overlay": bw_light},
        {"name": "em_ma_bw_mid", "em_weight": 0.5, "ma_weight": 0.5, "bw_overlay": bw_mid},
        {"name": "em_ma_bw_badstrong", "em_weight": 0.45, "ma_weight": 0.55, "bw_overlay": bw_strong_bad},
    ]


def make_specs(max_specs: int = 96) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    class_weights = ["1.00,1.40,1.70", "1.00,1.55,1.70"]
    cinc_fracs = [0.0, 0.10, 0.25]
    for snr in paper_snr_profiles():
        for mix in paper_mix_profiles():
            for cinc_frac in cinc_fracs:
                for cw in class_weights:
                    # Keep the pure paper controls early, then include CinC controls.
                    if cinc_frac > 0.0 and mix["name"] in {"em_only", "ma_only"}:
                        continue
                    variant = f"{snr['name']}__{mix['name']}__cinc{cinc_frac:.2f}__cw{cw}"
                    specs.append(
                        {
                            "id": stable_id(variant),
                            "family": "sensors2025_noise_synthesis",
                            "snr_profile": snr,
                            "mix": mix,
                            "cinc_bad_fraction": float(cinc_frac),
                            "class_weight": cw,
                            "paper_constraints": {
                                "target_snr_noise": "EM+MA only",
                                "bw_participates_in_target_snr": False,
                                "model_input_protocol": "project mainline 10s@125Hz, not paper 5s@200Hz",
                            },
                        }
                    )
    return specs[: max(1, int(max_specs))]


def load_cinc_bad_pool(cinc_dir: Path) -> dict[str, Any]:
    signals_path = cinc_dir / "signals.npz"
    meta_path = cinc_dir / "metadata.csv"
    if not signals_path.exists() or not meta_path.exists():
        return {"available": False, "signals": None, "metadata": None, "reason": f"missing {cinc_dir}"}
    X = np.load(signals_path)["X"].astype(np.float32)
    meta = pd.read_csv(meta_path)
    noisy = meta["is_noisy"].astype(int).to_numpy() == 1 if "is_noisy" in meta.columns else meta["label_raw"].astype(str).to_numpy() == "~"
    if X.ndim != 3 or X.shape[1:] != (1, N_TARGET) or not np.any(noisy):
        return {"available": False, "signals": None, "metadata": None, "reason": f"bad shape/no noisy rows: {X.shape}"}
    return {"available": True, "signals": X[noisy, 0], "metadata": meta.loc[noisy].reset_index(drop=True), "reason": "ok"}


def sample_track_noise(name: str, split_name: str, tracks: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    item = tracks[name]
    return rn.normalize_1d(rn.sample_segment(item["signal"], split_name, item["ranges"], rng))


def sample_cinc_noise(cinc_pool: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    X = cinc_pool.get("signals")
    if not isinstance(X, np.ndarray) or len(X) == 0:
        raise ValueError("CinC noisy pool is not available")
    pos = int(rng.integers(0, len(X)))
    return rn.normalize_1d(X[pos].astype(np.float32))


def make_em_ma_noise(
    spec: dict[str, Any],
    split_name: str,
    tracks: dict[str, Any],
    rng: np.random.Generator,
    *,
    use_cinc_bad: bool,
    cinc_pool: dict[str, Any],
) -> tuple[np.ndarray, str, dict[str, float]]:
    if use_cinc_bad and bool(cinc_pool.get("available")):
        return sample_cinc_noise(cinc_pool, rng), "cinc2017_noisy_as_bad_control", {"em": 0.0, "ma": 0.0, "bw": 0.0}
    mix = spec["mix"]
    ew = float(mix.get("em_weight", 0.0))
    mw = float(mix.get("ma_weight", 0.0))
    denom = ew + mw
    if denom <= 0:
        raise ValueError(f"Spec has no EM/MA SNR noise: {spec['id']}")
    ew /= denom
    mw /= denom
    noise = ew * sample_track_noise("em", split_name, tracks, rng) + mw * sample_track_noise("ma", split_name, tracks, rng)
    weights = {"em": ew, "ma": mw, "bw": 0.0}
    return rn.normalize_1d(noise), f"em:{ew:.2f}+ma:{mw:.2f}", weights


def scale_noise_component(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> tuple[np.ndarray, float]:
    px = float(np.mean(clean * clean)) + 1e-12
    pv = float(np.mean(noise * noise)) + 1e-12
    scale = float(np.sqrt(px / (pv * (10.0 ** (target_snr_db / 10.0)))))
    return (scale * noise).astype(np.float32), scale


def measured_snr_for_component(clean: np.ndarray, noise_component: np.ndarray) -> float:
    px = float(np.mean(clean * clean)) + 1e-12
    pn = float(np.mean(noise_component * noise_component)) + 1e-12
    return float(10.0 * np.log10(px / pn))


def sample_bw_overlay(
    spec: dict[str, Any],
    y_class: str,
    clean: np.ndarray,
    split_name: str,
    tracks: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    overlay = spec["mix"].get("bw_overlay")
    if not overlay:
        return np.zeros_like(clean, dtype=np.float32), 0.0
    lo, hi = [float(v) for v in overlay[y_class]]
    bw_scale = float(rng.uniform(lo, hi))
    bw = sample_track_noise("bw", split_name, tracks, rng)
    # BW is a separately sampled waveform overlay.  It does not participate in
    # EM+MA target SNR calibration.
    amp = float(np.sqrt(np.mean(clean * clean)) + 1e-6)
    return (bw_scale * amp * bw).astype(np.float32), bw_scale


def qrs_interval_slices(qrs_mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(qrs_mask, dtype=np.float32) > 0.35
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return [(0, N_TARGET)]
    groups: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value - prev > 2:
            groups.append((start, prev))
            start = value
        prev = value
    groups.append((start, prev))
    centers = [int(0.5 * (a + b)) for a, b in groups if b - a >= 2]
    if len(centers) < 2:
        return [(0, N_TARGET)]
    bounds = [0]
    for a, b in zip(centers[:-1], centers[1:]):
        bounds.append(int(0.5 * (a + b)))
    bounds.append(N_TARGET)
    return [(max(0, bounds[i]), min(N_TARGET, bounds[i + 1])) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] >= FS_TARGET // 2]


def rr_noise_level_graph(clean: np.ndarray, noisy: np.ndarray, emma_component: np.ndarray, qrs_mask: np.ndarray, y_class: str) -> tuple[np.ndarray, dict[str, float]]:
    level = np.zeros(N_TARGET, dtype=np.float32)
    snrs: list[float] = []
    corrs: list[float] = []
    bad_bins = 0
    for start, end in qrs_interval_slices(qrs_mask):
        c = clean[start:end]
        x = noisy[start:end]
        n = emma_component[start:end]
        if len(c) < 4:
            continue
        snr = measured_snr_for_component(c, n)
        corr = float(np.corrcoef(c, x)[0, 1]) if np.std(c) > 1e-6 and np.std(x) > 1e-6 else 0.0
        unacceptable = snr < 5.0 or corr < 0.82 or (y_class == "bad" and snr < 10.0)
        level[start:end] = 1.0 if unacceptable else 0.0
        snrs.append(float(snr))
        corrs.append(float(corr))
        bad_bins += int(unacceptable)
    return level, {
        "rr_bins": float(len(snrs)),
        "rr_unacceptable_bins": float(bad_bins),
        "rr_snr_mean": float(np.mean(snrs)) if snrs else math.nan,
        "rr_corr_mean": float(np.mean(corrs)) if corrs else math.nan,
    }


def make_dataset_variant(
    ptb: dict[str, Any],
    spec: dict[str, Any],
    tracks: dict[str, Any],
    out_dir: Path,
    seed: int,
    sample_per_class: int = 0,
    cinc_pool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels_all = ptb["labels"].copy().sort_values("idx").reset_index(drop=True)
    clean_all = ptb["clean"].astype(np.float32, copy=False)
    masks_all = {k: v.astype(np.float32, copy=False) for k, v in ptb["masks"].items()}
    if sample_per_class > 0:
        rng_pick = np.random.default_rng(seed + 11)
        idxs: list[int] = []
        for cls in rn.CLASS_NAMES:
            cls_idx = labels_all.index[labels_all["y_class"].astype(str) == cls].to_numpy()
            take = min(int(sample_per_class), len(cls_idx))
            idxs.extend(rng_pick.choice(cls_idx, size=take, replace=False).tolist())
        idxs = sorted(idxs)
        labels = labels_all.iloc[idxs].copy().reset_index(drop=True)
        clean = clean_all[idxs]
        masks = {k: v[idxs] for k, v in masks_all.items()}
    else:
        labels = labels_all.copy()
        clean = clean_all
        masks = masks_all

    rng = np.random.default_rng(seed)
    cinc_pool = cinc_pool or {"available": False}
    noisy = np.zeros_like(clean, dtype=np.float32)
    emma_components = np.zeros_like(clean, dtype=np.float32)
    bw_components = np.zeros_like(clean, dtype=np.float32)
    noise_level_graph = np.zeros_like(clean, dtype=np.float32)
    qrs = masks.get("qrs_mask", np.zeros_like(clean))
    tst = masks.get("tst_mask", np.zeros_like(clean))
    critical = masks.get("critical_mask", qrs)

    rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, float]] = []
    for i, row in enumerate(labels.itertuples(index=False)):
        y_class = str(row.y_class)
        split_name = str(row.split)
        use_cinc_bad = y_class == "bad" and rng.random() < float(spec.get("cinc_bad_fraction", 0.0))
        raw_emma, noise_label, weights = make_em_ma_noise(spec, split_name, tracks, rng, use_cinc_bad=use_cinc_bad, cinc_pool=cinc_pool)
        target_snr = sample_snr(spec, y_class, rng)
        emma_component, emma_scale = scale_noise_component(clean[i], raw_emma, target_snr)
        bw_component, bw_scale = sample_bw_overlay(spec, y_class, clean[i], split_name, tracks, rng)
        x = (clean[i] + emma_component + bw_component).astype(np.float32)
        noisy[i] = x
        emma_components[i] = emma_component
        bw_components[i] = bw_component
        graph, graph_report = rr_noise_level_graph(clean[i], x, emma_component, qrs[i], y_class)
        noise_level_graph[i] = graph
        graph_rows.append(graph_report)
        rows.append(
            {
                "noise_kind": noise_label + ("+bw_overlay" if spec["mix"].get("bw_overlay") else ""),
                "target_snr_em_ma_db": float(target_snr),
                "measured_snr_em_ma_db": measured_snr_for_component(clean[i], emma_component),
                "measured_snr_all_noise_db": measured_snr_for_component(clean[i], emma_component + bw_component),
                "measured_snr_db": measured_snr_for_component(clean[i], emma_component + bw_component),
                "original_measured_snr_db": measured_snr_for_component(clean[i], emma_component + bw_component),
                "emma_scale": float(emma_scale),
                "bw_overlay_scale": float(bw_scale),
                "cinc_noisy_used": int(use_cinc_bad and bool(cinc_pool.get("available"))),
                "mix_weight_em": float(weights.get("em", 0.0)),
                "mix_weight_ma": float(weights.get("ma", 0.0)),
                "mix_weight_bw": 0.0,
            }
        )

    residual = noisy - clean
    qrs_nprd = np.asarray([rn.masked_nrmse(c, r, m) for c, r, m in zip(clean, residual, qrs)], dtype=np.float32)
    tst_nprd = np.asarray([rn.masked_nrmse(c, r, m) for c, r, m in zip(clean, residual, tst)], dtype=np.float32)
    critical_nprd = np.asarray([rn.masked_nrmse(c, r, m) for c, r, m in zip(clean, residual, critical)], dtype=np.float32)
    global_nprd = np.sqrt(np.mean(residual * residual, axis=1) / (np.mean(clean * clean, axis=1) + 1e-6)).astype(np.float32)

    labels = labels.copy()
    labels["idx"] = np.arange(len(labels), dtype=int)
    labels["y"] = labels["y_class"].map({"good": 0, "medium": 1, "bad": 2}).astype(int)
    for key in rows[0].keys() if rows else []:
        labels[key] = [r[key] for r in rows]
    labels["snr_db"] = labels["target_snr_em_ma_db"]
    labels["placement"] = "sensors2025_global_emma_snr_bw_overlay"
    labels["sample_source"] = f"sensors2025_noise_synthesis|{spec['id']}"
    labels["sensors2025_spec_id"] = str(spec["id"])
    labels["sensors2025_snr_profile"] = str(spec["snr_profile"]["name"])
    labels["sensors2025_mix_profile"] = str(spec["mix"]["name"])
    labels["qrs_nprd"] = qrs_nprd
    labels["tst_nprd"] = tst_nprd
    labels["critical_damage_score"] = critical_nprd
    labels["diagnostic_damage_score"] = np.maximum(critical_nprd, 0.5 * (qrs_nprd + tst_nprd))
    labels["smooth_morph_score"] = global_nprd
    for key in ("rr_bins", "rr_unacceptable_bins", "rr_snr_mean", "rr_corr_mean"):
        labels[key] = [float(r.get(key, math.nan)) for r in graph_rows]

    data_dir = out_dir / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "synth_10s_125hz_clean.npz", X_clean=clean.astype(np.float32))
    np.savez_compressed(data_dir / "synth_10s_125hz_noisy.npz", X_noisy=noisy.astype(np.float32))
    np.savez_compressed(data_dir / "synth_10s_125hz_local_mask.npz", **masks)
    labels.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)
    labels.to_csv(data_dir / "synth_10s_125hz_labels_with_level.csv", index=False)
    np.savez_compressed(data_dir / "synth_10s_125hz_noise_level.npz", level=global_nprd)
    np.savez_compressed(
        out_dir / "noise_level_graph.npz",
        level_graph=noise_level_graph.astype(np.float32),
        emma_noise=emma_components.astype(np.float32),
        bw_noise=bw_components.astype(np.float32),
    )

    audit = {
        "spec": spec,
        "rows": int(len(labels)),
        "sample_per_class": int(sample_per_class),
        "split_counts": {str(k): int(v) for k, v in labels["split"].value_counts().sort_index().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in labels["y_class"].value_counts().sort_index().to_dict().items()},
        "cinc_available": bool(cinc_pool.get("available")),
        "cinc_bad_fraction_requested": float(spec.get("cinc_bad_fraction", 0.0)),
        "cinc_bad_rows_used": int(labels["cinc_noisy_used"].sum()) if "cinc_noisy_used" in labels else 0,
        "snr_by_class": {
            cls: {
                "target_em_ma_mean": float(labels.loc[labels["y_class"] == cls, "target_snr_em_ma_db"].mean()),
                "measured_em_ma_mean": float(labels.loc[labels["y_class"] == cls, "measured_snr_em_ma_db"].mean()),
                "measured_all_mean": float(labels.loc[labels["y_class"] == cls, "measured_snr_all_noise_db"].mean()),
            }
            for cls in rn.CLASS_NAMES
        },
        "paper_note": "Target SNR is computed from EM+MA only. BW overlay changes measured_snr_all_noise_db but not target_snr_em_ma_db.",
    }
    write_json(out_dir / "data_variant_audit.json", audit)
    write_json(out_dir / "sensors2025_spec.json", spec)
    write_json(out_dir / "noise_level_graph_audit.json", {"summary": labels[["y_class", "rr_bins", "rr_unacceptable_bins", "rr_snr_mean", "rr_corr_mean"]].groupby("y_class").mean(numeric_only=True).to_dict()})
    return {"X_noisy": noisy[:, None, :], "X_clean": clean[:, None, :], "labels": labels, "audit": audit}


def plot_class_gallery(X_clean: np.ndarray, X_noisy: np.ndarray, labels: pd.DataFrame, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    positions: list[int] = []
    for cls in rn.CLASS_NAMES:
        positions.extend(labels.index[labels["y_class"].astype(str) == cls].tolist()[:4])
    if not positions:
        return
    t = np.arange(N_TARGET, dtype=np.float32) / FS_TARGET
    fig, axes = plt.subplots(len(positions), 1, figsize=(13, max(4, len(positions) * 1.1)), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, pos in zip(axes_arr, positions):
        row = labels.iloc[pos]
        ax.plot(t, X_clean[pos, 0], color="#111827", lw=0.65, label="clean")
        ax.plot(t, X_noisy[pos, 0], color="#b45309", lw=0.55, alpha=0.95, label="paper-style synthetic")
        ax.set_ylabel(
            f"{row['y_class']}\nT {row['target_snr_em_ma_db']:.1f}\nAll {row['measured_snr_all_noise_db']:.1f}",
            fontsize=7,
            rotation=0,
            labelpad=42,
        )
        ax.grid(alpha=0.18)
    axes_arr[0].legend(loc="upper right", fontsize=8)
    axes_arr[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def score_one_spec(
    args: argparse.Namespace,
    spec: dict[str, Any],
    ptb: dict[str, Any],
    tracks: dict[str, Any],
    cinc_pool: dict[str, Any],
    but_X: np.ndarray,
    but_meta: pd.DataFrame,
    but_feat: pd.DataFrame,
) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    spec_id = str(spec["id"])
    variant_dir = out_root / "scan_variants" / spec_id
    if (variant_dir / "but_distance_report.json").exists() and not args.force:
        return read_json(variant_dir / "but_distance_report.json")
    made = make_dataset_variant(
        ptb,
        spec,
        tracks,
        variant_dir,
        seed=int(args.seed),
        sample_per_class=int(args.scan_sample_per_class),
        cinc_pool=cinc_pool,
    )
    X_synth = made["X_noisy"]
    meta = made["labels"].copy()
    meta["dataset"] = "PTB sensors2025-style synthetic"
    meta["class_name"] = meta["y_class"].astype(str)
    synth_feat = rn.extract_features(X_synth, meta, variant_dir / "synthetic_morph_features.csv", max_rows=0)
    morph_dist = rn.classwise_ks_distance(but_feat, synth_feat, rn.FEATURE_COLUMNS)
    sqi_dist = rn.sqi_distance(
        but_X,
        X_synth,
        but_meta,
        meta,
        max_rows_per_class=int(args.sqi_rows_per_class),
        seed=int(args.seed) + 101,
    )
    domain_bal = rn.domain_separability(but_feat, synth_feat, rn.FEATURE_COLUMNS, seed=int(args.seed))
    medium_distance = float(morph_dist["by_class"].get("medium", math.nan))
    overall = float(morph_dist["overall"])
    sqi_overall = float(sqi_dist["overall"])
    # Medium and SQI are weighted a little more heavily because the paper-style
    # claim is mainly about weak SQI labels, not only waveform noise energy.
    but_like_score = float(0.42 * overall + 0.25 * medium_distance + 0.20 * sqi_overall + 0.13 * domain_bal)
    row = {
        "variant_id": spec_id,
        "family": spec.get("family", "sensors2025_noise_synthesis"),
        "spec": spec,
        "variant_dir": str(variant_dir),
        "morph_overall_distance": overall,
        "morph_medium_distance": medium_distance,
        "morph_by_class": morph_dist["by_class"],
        "sqi_overall_distance": sqi_overall,
        "sqi_by_class": sqi_dist["by_class"],
        "domain_separability_bal_acc": domain_bal,
        "overall_but_like_score": but_like_score,
        "audit": made["audit"],
    }
    write_json(variant_dir / "synthetic_feature_profile.json", {"morph_features": synth_feat.describe().to_dict(), "audit": made["audit"]})
    write_json(variant_dir / "but_distance_report.json", row)
    write_json(variant_dir / "morph_distance_rows.json", morph_dist["rows"])
    write_json(variant_dir / "sqi_distance_rows.json", sqi_dist["rows"])
    plot_class_gallery(made["X_clean"], X_synth, meta, variant_dir / "visuals" / "synthetic_class_gallery.png", spec_id)
    report_copy = report_root / "scan_variants" / spec_id
    report_copy.mkdir(parents=True, exist_ok=True)
    rn.link_or_copy(variant_dir / "visuals" / "synthetic_class_gallery.png", report_copy / "synthetic_class_gallery.png")
    write_json(report_copy / "but_distance_report.json", row)
    return row


def score_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="scan_running", stage="score_specs", updated_at=now_iso())
    nstdb = rn.ensure_nstdb_tracks(Path(args.nstdb_root))
    write_json(out_root / "nstdb_noise_audit.json", nstdb["audit"])
    cinc_pool = load_cinc_bad_pool(Path(args.cinc_dir))
    write_json(out_root / "cinc_bad_pool_audit.json", {k: v for k, v in cinc_pool.items() if k not in {"signals", "metadata"}})
    ptb = rn.load_ptb_artifact(Path(args.source_artifact_dir))
    specs = make_specs(int(args.max_specs))
    write_json(out_root / SPEC_NAME, {"rows": specs})
    write_json(report_root / SPEC_NAME, {"rows": specs})
    but_X, but_meta = rn.load_but(Path(args.but_protocol_dir))
    rng = pd.Series(range(len(but_meta))).sample(
        n=min(len(but_meta), int(args.but_feature_max_rows) * len(rn.CLASS_NAMES)),
        random_state=int(args.seed),
    ).sort_values()
    but_feat = rn.extract_features(
        but_X[rng.to_numpy()],
        but_meta.iloc[rng.to_numpy()].reset_index(drop=True),
        out_root / "but_morph_features_scan.csv",
        max_rows=0,
    )
    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status="scan_running", current=spec["id"], completed=i - 1, total=len(specs), updated_at=now_iso())
        rows.append(score_one_spec(args, spec, ptb, nstdb["tracks"], cinc_pool, but_X, but_meta, but_feat))
    rows = sorted(rows, key=lambda row: float(row["overall_but_like_score"]))
    df = pd.DataFrame(
        [
            {
                "rank": i,
                "variant_id": row["variant_id"],
                "snr_profile": row["spec"]["snr_profile"]["name"],
                "mix_profile": row["spec"]["mix"]["name"],
                "cinc_bad_fraction": row["spec"].get("cinc_bad_fraction", 0.0),
                "class_weight": row["spec"].get("class_weight"),
                "overall_but_like_score": row["overall_but_like_score"],
                "morph_overall_distance": row["morph_overall_distance"],
                "morph_medium_distance": row["morph_medium_distance"],
                "sqi_overall_distance": row["sqi_overall_distance"],
                "domain_separability_bal_acc": row["domain_separability_bal_acc"],
                "variant_dir": row["variant_dir"],
            }
            for i, row in enumerate(rows, start=1)
        ]
    )
    df.to_csv(out_root / SCAN_NAME, index=False)
    df.to_csv(report_root / SCAN_NAME, index=False)
    write_json(out_root / "distance_leaderboard.json", {"rows": rows})
    write_json(report_root / "distance_leaderboard.json", {"rows": rows[: min(30, len(rows))]})
    write_scan_report(args, rows)
    update_state(out_root / STATE_NAME, status="scan_complete", completed=len(rows), total=len(rows), best=rows[0] if rows else None, updated_at=now_iso())
    return rows


def train_one(args: argparse.Namespace, row: dict[str, Any], mode: str) -> dict[str, Any]:
    spec = row["spec"]
    spec_id = str(spec["id"])
    variant_dir = Path(row["variant_dir"])
    run_dir = Path(args.out_root) / "runs" / mode / spec_id
    summary_file = run_dir / "sensors2025_run_summary.json"
    if summary_file.exists() and not bool(args.rerun_existing):
        existing = read_json(summary_file)
        if int(existing.get("returncode", 1)) == 0 or bool(args.skip_failed_existing):
            existing["skipped_existing"] = True
            return existing
    if not (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists() or int(args.train_regenerate_full) == 1:
        ptb = rn.load_ptb_artifact(Path(args.source_artifact_dir))
        tracks = rn.ensure_nstdb_tracks(Path(args.nstdb_root))["tracks"]
        cinc_pool = load_cinc_bad_pool(Path(args.cinc_dir))
        make_dataset_variant(ptb, spec, tracks, variant_dir, seed=int(args.seed), sample_per_class=0, cinc_pool=cinc_pool)
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
        str(spec.get("class_weight", args.class_weight)),
        "--seed",
        str(args.seed),
    ]
    start = time.perf_counter()
    with (log_dir / f"{spec_id}.stdout.txt").open("w", encoding="utf-8") as out, (log_dir / f"{spec_id}.stderr.txt").open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(rn.ROOT), stdout=out, stderr=err, text=True)
    payload: dict[str, Any] = {
        "spec": spec,
        "mode": mode,
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
        "distance": {k: row.get(k) for k in ["overall_but_like_score", "morph_overall_distance", "morph_medium_distance", "sqi_overall_distance", "domain_separability_bal_acc"]},
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(summary_file, payload)
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    write_train_report(args)
    return payload


def select_rows_for_training(args: argparse.Namespace, n: int) -> list[dict[str, Any]]:
    path = Path(args.out_root) / "distance_leaderboard.json"
    ranked = read_json(path)["rows"] if path.exists() else score_specs(args)
    selected = ranked[: int(n)]
    by_id = {str(row["variant_id"]): row for row in ranked}
    controls = [
        "paper_table_strict__em_only__cinc0p00__cw1p00_1p40_1p70",
        "paper_table_strict__ma_only__cinc0p00__cw1p00_1p40_1p70",
        "paper_table_strict__em_ma_equal__cinc0p00__cw1p00_1p40_1p70",
        "paper_fig2_like__em_ma_bw_mid__cinc0p00__cw1p00_1p40_1p70",
    ]
    for cid in controls:
        if cid in by_id and by_id[cid] not in selected:
            selected.append(by_id[cid])
    return selected[: int(n)]


def run_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="quick_training", stage="quick_train", updated_at=now_iso())
    selected = select_rows_for_training(args, int(args.top_quick))
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        update_state(out_root / STATE_NAME, status="quick_training", current=row["variant_id"], completed=i - 1, total=len(selected), updated_at=now_iso())
        rows.append(train_one(args, row, "quick"))
    update_state(out_root / STATE_NAME, status="quick_complete", n_quick=len(rows), updated_at=now_iso())
    return rows


def result_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    ev = row.get("but_10s_eval", {})
    reps = [ev.get("but_10s_test_report", {}), ev.get("but_10s_raw_test_report", {}), ev.get("but_10s_balanced_test_report", {}), ev.get("but_10s_balanced_raw_report", {})]
    rep = max([r for r in reps if r], key=lambda r: float(r.get("macro_f1", 0.0)), default={})
    rec = rep.get("recall_good_medium_bad", [0, 0, 0])
    ptb = row.get("ptb_test_report", {})
    return (
        float(rep.get("macro_f1", 0.0)),
        min(float(rec[1]), float(rec[2])),
        float(rep.get("balanced_acc", 0.0)),
        float(rec[2]),
        float(ptb.get("acc", 0.0)),
    )


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="full_training", stage="full_train", updated_at=now_iso())
    summary_path = out_root / SUMMARY_NAME
    quick_rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if summary_path.exists() else []
    selected_quick = sorted([r for r in quick_rows if r.get("mode") == "quick" and r.get("returncode") == 0], key=result_key, reverse=True)[: int(args.top_full)]
    if not selected_quick:
        selected = select_rows_for_training(args, int(args.top_full))
    else:
        selected = [dict(r, variant_id=r["spec"]["id"]) for r in selected_quick]
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        spec_id = row["spec"]["id"]
        update_state(out_root / STATE_NAME, status="full_training", current=spec_id, completed=i - 1, total=len(selected), updated_at=now_iso())
        row2 = {
            "variant_id": spec_id,
            "spec": row["spec"],
            "variant_dir": row["variant_dir"],
            **row.get("distance", {}),
        }
        rows.append(train_one(args, row2, "full"))
    update_state(out_root / STATE_NAME, status="full_complete", n_full=len(rows), updated_at=now_iso())
    return rows


def write_scan_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sensors 2025-Style Noise Synthesis Scan vs BUT 10s P1",
        "",
        "This scan reproduces the paper's data idea only: real NSTDB EM/MA noise determines target SNR; BW may be overlaid but is excluded from target SNR.",
        "Our model input remains 10s@125Hz. The paper's 5s@200Hz model/protocol is documented but not used here.",
        "",
        "| rank | variant | SNR profile | mix | CinC bad frac | score | morph | medium | SQI | domain bal |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows[:30], start=1):
        spec = row["spec"]
        lines.append(
            f"| {i} | `{row['variant_id']}` | {spec['snr_profile']['name']} | {spec['mix']['name']} | "
            f"{float(spec.get('cinc_bad_fraction', 0.0)):.2f} | {row['overall_but_like_score']:.4f} | "
            f"{row['morph_overall_distance']:.4f} | {row['morph_medium_distance']:.4f} | "
            f"{row['sqi_overall_distance']:.4f} | {row['domain_separability_bal_acc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- `target_snr_em_ma_db` and `measured_snr_all_noise_db` are both recorded in generated labels.",
            "- BW overlay is reported via `bw_overlay_scale`; it is not used in target SNR scaling.",
            "- Lower distance score only selects candidates for training; BUT test thresholds are never selected on test.",
        ]
    )
    (report_root / "real_noise_snr_scan_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_train_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    rows = [json.loads(line) for line in (out_root / SUMMARY_NAME).read_text(encoding="utf-8").splitlines() if line.strip()] if (out_root / SUMMARY_NAME).exists() else []
    ranked = sorted(rows, key=result_key, reverse=True)
    lines = [
        "# Sensors 2025-Style Noise Synthesis BUT Model Validation",
        "",
        f"Rule-based reference `{RULE_ANCHOR['variant']}`: acc `{RULE_ANCHOR['acc']:.4f}`, balanced `{RULE_ANCHOR['balanced_acc']:.4f}`, macro-F1 `{RULE_ANCHOR['macro_f1']:.4f}`, recalls `{RULE_ANCHOR['recall_good_medium_bad']}`.",
        "",
        "| rank | mode | variant | return | orig macro | orig recalls G/M/B | bal cal macro | bal raw macro | PTB acc | PTB bad | distance |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        ev = row.get("but_10s_eval", {})
        orig = ev.get("but_10s_test_report", {})
        bal = ev.get("but_10s_balanced_test_report", {})
        bal_raw = ev.get("but_10s_balanced_raw_report", {})
        rec = orig.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        lines.append(
            f"| {i} | {row.get('mode')} | `{row.get('spec', {}).get('id')}` | {row.get('returncode')} | "
            f"{float(orig.get('macro_f1', 0.0)):.4f} | {float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} | "
            f"{float(bal.get('macro_f1', 0.0)):.4f} | {float(bal_raw.get('macro_f1', 0.0)):.4f} | "
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec) > 2 else 0.0):.4f} | "
            f"{float(row.get('distance', {}).get('overall_but_like_score', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "- If paper-style strict SNR loses to `h_bad_rescue_05`, keep it as a natural-noise control rather than forcing it into the mainline.",
            "- If it improves medium while bad stays >=0.80, it supports adding paper-like weak SQI labels to the next generator.",
            "- If BW overlay reduces all-noise SNR but does not improve BUT, that supports the paper's claim that BW should not define SQI by itself.",
        ]
    )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "sensors2025_noise_synthesis_training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "sensors2025_noise_synthesis_training_summary.json", {"rows": rows, "ranked": ranked})


def run_all(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"all", "score_specs"}:
        score_specs(args)
    if args.stage in {"all", "quick_train"}:
        run_quick(args)
    if args.stage in {"all", "full_train"}:
        run_full(args)
    if args.stage in {"all", "report"}:
        rows = read_json(out_root / "distance_leaderboard.json")["rows"] if (out_root / "distance_leaderboard.json").exists() else []
        if rows:
            write_scan_report(args, rows)
        write_train_report(args)
    update_state(out_root / STATE_NAME, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sensors 2025-style noise synthesis grid and BUT 10s validation.")
    parser.add_argument("--stage", choices=("score_specs", "quick_train", "full_train", "report", "all"), default="score_specs")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--nstdb_root", default=str(DEFAULT_NSTDB_ROOT))
    parser.add_argument("--cinc_dir", default=str(DEFAULT_CINC_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_specs", type=int, default=96)
    parser.add_argument("--scan_sample_per_class", type=int, default=60)
    parser.add_argument("--but_feature_max_rows", type=int, default=1500)
    parser.add_argument("--sqi_rows_per_class", type=int, default=24)
    parser.add_argument("--top_quick", type=int, default=12)
    parser.add_argument("--top_full", type=int, default=4)
    parser.add_argument("--train_regenerate_full", type=int, default=1)
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--skip_failed_existing", action="store_true", default=True)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--class_weight", default="1.00,1.40,1.70")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_all(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
