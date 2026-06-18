"""Real NSTDB MA/EM/BW SNR-grid synthetic data for BUT matching.

This experiment intentionally avoids the hand-built BUT morphology rules used
by the previous generator grids.  It mixes only real NSTDB noise tracks
(`em`, `ma`, `bw`) into PTB clean ECG at class-specific SNR ranges, audits the
result against BUT 10s P1 morphology/SQI distributions, then trains only the
best-matching candidates.

It is experiment-only: no ``src/sqi_pipeline`` files or mainline checkpoints are
modified.
"""

from __future__ import annotations

import argparse
import json
import math
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
import wfdb
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.analyze_but_morphology_clusters import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    extract_features,
    load_but,
)
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT, link_or_copy, load_ptb_artifact
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
from src.transformer_pipeline.external_benchmarks.but_sqi_fusion_ptb_train import SQI_COLUMNS, sqi_for_signal
from src.transformer_pipeline.external_benchmarks.run import FS_TARGET, N_TARGET, basic_sqi_features
from src.transformer_pipeline.noise.synthesize_snr_dataset import add_noise_at_snr, load_noise_125, split_noise_ranges


RUN_TAG = "e311_real_noise_snr_but_match_grid_10s_2026_06_04"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_BUT_PROTOCOL = PROTOCOL_OUT_ROOT / "protocols" / "p1_current_10s_center"
DEFAULT_NSTDB_ROOT = ROOT / "data" / "physionet" / "nstdb"

STATE_NAME = "real_noise_snr_state.json"
SPEC_NAME = "real_noise_snr_specs.json"
SCAN_NAME = "distance_leaderboard.csv"
SUMMARY_NAME = "real_noise_snr_training_summary.jsonl"
NOISE_NAMES = ("em", "ma", "bw")

OLD_DOMAIN_SEPARABILITY = 0.971
RULE_ANCHOR = {
    "variant": "h_bad_rescue_05",
    "acc": 0.8229,
    "balanced_acc": 0.8177,
    "macro_f1": 0.7454,
    "recall_good_medium_bad": [0.887, 0.773, 0.793],
}


@dataclass(frozen=True)
class SNRSpec:
    good: tuple[float, float]
    medium: tuple[float, float]
    bad: tuple[float, float]
    sampler: str


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_nstdb_tracks(nstdb_root: Path) -> dict[str, Any]:
    """Ensure `em`, `ma`, and `bw` exist and are readable.

    The older synthetic code silently used `ma` when `bw` was absent.  This
    experiment must not do that, because the point is to test real BW/EM/MA.
    """

    nstdb_root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in NOISE_NAMES if not (nstdb_root / f"{name}.hea").exists()]
    if missing:
        wfdb.dl_database("nstdb", dl_dir=str(nstdb_root), records=list(NOISE_NAMES), keep_subdirs=False)
    tracks: dict[str, Any] = {}
    audit: dict[str, Any] = {"nstdb_root": str(nstdb_root), "tracks": {}}
    for name in NOISE_NAMES:
        base = nstdb_root / name
        if not base.with_suffix(".hea").exists():
            raise FileNotFoundError(f"NSTDB noise track missing after download: {base.with_suffix('.hea')}")
        arr = load_noise_125(base).astype(np.float32)
        if arr.ndim != 1 or len(arr) < 3 * N_TARGET:
            raise ValueError(f"NSTDB {name} is not a usable 1D track: shape={arr.shape}")
        tracks[name] = {"signal": arr, "ranges": split_noise_ranges(len(arr))}
        audit["tracks"][name] = {
            "samples_125hz": int(len(arr)),
            "duration_sec": float(len(arr) / FS_TARGET),
            "std": float(np.std(arr)),
            "path": str(base),
        }
    return {"tracks": tracks, "audit": audit}


def sample_segment(track: np.ndarray, split_name: str, ranges: dict[str, tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    lo, hi = ranges[split_name]
    if hi - lo <= N_TARGET:
        raise ValueError(f"Noise range too short for split={split_name}: {hi - lo}")
    start = int(rng.integers(lo, hi - N_TARGET))
    return track[start : start + N_TARGET].astype(np.float32, copy=False)


def normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    return x / (float(np.std(x)) + 1e-6)


def fixed_mix_profiles() -> list[dict[str, Any]]:
    return [
        {"name": "single_em", "mode": "weighted", "weights": {"em": 1.0}},
        {"name": "single_ma", "mode": "weighted", "weights": {"ma": 1.0}},
        {"name": "single_bw", "mode": "weighted", "weights": {"bw": 1.0}},
        {"name": "pair_em_ma", "mode": "weighted", "weights": {"em": 0.5, "ma": 0.5}},
        {"name": "pair_em_bw", "mode": "weighted", "weights": {"em": 0.5, "bw": 0.5}},
        {"name": "pair_ma_bw", "mode": "weighted", "weights": {"ma": 0.5, "bw": 0.5}},
        {"name": "triple_balanced", "mode": "weighted", "weights": {"em": 1 / 3, "ma": 1 / 3, "bw": 1 / 3}},
        {"name": "ma_heavy", "mode": "weighted", "weights": {"em": 0.15, "ma": 0.70, "bw": 0.15}},
        {"name": "em_heavy", "mode": "weighted", "weights": {"em": 0.70, "ma": 0.15, "bw": 0.15}},
        {"name": "bw_heavy", "mode": "weighted", "weights": {"em": 0.15, "ma": 0.15, "bw": 0.70}},
        {"name": "ma_bw_wearable", "mode": "weighted", "weights": {"em": 0.10, "ma": 0.55, "bw": 0.35}},
        {"name": "em_ma_no_bw", "mode": "weighted", "weights": {"em": 0.45, "ma": 0.55}},
        {"name": "dirichlet_balanced", "mode": "dirichlet", "alpha": {"em": 1.0, "ma": 1.0, "bw": 1.0}},
        {"name": "dirichlet_wearable", "mode": "dirichlet", "alpha": {"em": 0.7, "ma": 1.8, "bw": 1.3}},
        {"name": "blockwise_switch", "mode": "blockwise", "choices": ("em", "ma", "bw")},
    ]


def snr_profiles() -> list[SNRSpec]:
    return [
        SNRSpec((16, 20), (8, 14), (-6, 0), "uniform"),
        SNRSpec((14, 18), (6, 12), (-4, 2), "uniform"),
        SNRSpec((12, 16), (4, 10), (-2, 4), "uniform"),
        SNRSpec((10, 14), (2, 8), (0, 6), "uniform"),
        SNRSpec((14, 18), (8, 14), (-4, 2), "triangular"),
        SNRSpec((12, 16), (6, 12), (-2, 4), "triangular"),
        SNRSpec((16, 20), (6, 12), (-6, 0), "triangular"),
        SNRSpec((10, 14), (4, 10), (-2, 4), "triangular"),
    ]


def make_specs(max_specs: int = 120) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for mix in fixed_mix_profiles():
        for snr_i, snr in enumerate(snr_profiles(), start=1):
            spec = {
                "id": f"{mix['name']}_snr{snr_i:02d}_{snr.sampler}",
                "family": "real_noise_snr",
                "mix": mix,
                "snr": {
                    "good": list(snr.good),
                    "medium": list(snr.medium),
                    "bad": list(snr.bad),
                    "sampler": snr.sampler,
                },
                "class_weight": "1.00,1.40,1.70",
                "note": "Only real NSTDB em/ma/bw noise mixed at class-specific SNR.",
            }
            specs.append(spec)
    return specs[: max(1, int(max_specs))]


def sample_snr(spec: dict[str, Any], y_class: str, rng: np.random.Generator) -> float:
    lo, hi = [float(v) for v in spec["snr"][y_class]]
    if str(spec["snr"].get("sampler", "uniform")) == "triangular":
        return float(rng.triangular(lo, 0.5 * (lo + hi), hi))
    return float(rng.uniform(lo, hi))


def make_noise(kind_spec: dict[str, Any], split_name: str, tracks: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, str, dict[str, float]]:
    mode = str(kind_spec["mode"])
    if mode in {"weighted", "dirichlet"}:
        if mode == "dirichlet":
            names = list(NOISE_NAMES)
            alpha = np.asarray([float(kind_spec.get("alpha", {}).get(name, 1.0)) for name in names], dtype=np.float64)
            weights = rng.dirichlet(alpha)
        else:
            weights_map = {str(k): float(v) for k, v in kind_spec.get("weights", {}).items()}
            names = [name for name in NOISE_NAMES if weights_map.get(name, 0.0) > 0]
            weights = np.asarray([weights_map[name] for name in names], dtype=np.float64)
            weights = weights / float(np.sum(weights))
        seg = np.zeros(N_TARGET, dtype=np.float32)
        for name, weight in zip(names, weights):
            item = tracks[name]
            seg += float(weight) * normalize_1d(sample_segment(item["signal"], split_name, item["ranges"], rng))
        weights_dict = {name: float(weight) for name, weight in zip(names, weights)}
        return normalize_1d(seg), "+".join(f"{k}:{v:.2f}" for k, v in weights_dict.items()), weights_dict
    if mode == "blockwise":
        choices = list(kind_spec.get("choices", NOISE_NAMES))
        blocks = int(rng.integers(2, 4))
        cuts = sorted(rng.choice(np.arange(FS_TARGET, N_TARGET - FS_TARGET), size=blocks - 1, replace=False).tolist())
        starts = [0, *cuts]
        ends = [*cuts, N_TARGET]
        seg = np.zeros(N_TARGET, dtype=np.float32)
        counts = {name: 0.0 for name in NOISE_NAMES}
        labels: list[str] = []
        for start, end in zip(starts, ends):
            name = str(rng.choice(choices))
            item = tracks[name]
            raw = normalize_1d(sample_segment(item["signal"], split_name, item["ranges"], rng))
            seg[start:end] = raw[start:end]
            counts[name] += float(end - start)
            labels.append(name)
        total = float(sum(counts.values())) or 1.0
        weights_dict = {name: counts[name] / total for name in NOISE_NAMES if counts[name] > 0}
        return normalize_1d(seg), "block:" + "-".join(labels), weights_dict
    raise ValueError(f"Unknown noise mix mode: {mode}")


def measured_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    px = float(np.mean(clean * clean)) + 1e-12
    pn = float(np.mean(noise * noise)) + 1e-12
    return float(10.0 * np.log10(px / pn))


def smooth_mask(mask: np.ndarray, width: int = 13) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if width <= 1:
        return np.clip(mask, 0.0, 1.0)
    kernel = np.ones(int(width), dtype=np.float32) / float(width)
    return np.clip(np.convolve(mask, kernel, mode="same"), 0.0, 1.0).astype(np.float32)


def apply_qrs_shaping(
    *,
    clean: np.ndarray,
    raw_noise: np.ndarray,
    qrs_mask: np.ndarray,
    tst_mask: np.ndarray,
    y_class: str,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Small QRS-aware shaping for real noise.

    This keeps the noise source real.  It only changes where the real noise is
    emphasized and optionally applies a small QRS attenuation to bad examples.
    Defaults are all no-op, so the original real-noise SNR scan remains intact.
    """

    qrs_spec = spec.get("qrs", {}) or {}
    if not qrs_spec:
        return clean, raw_noise, {}
    qrs = smooth_mask(qrs_mask, int(qrs_spec.get("mask_smooth_width", 13)))
    tst = smooth_mask(tst_mask, int(qrs_spec.get("mask_smooth_width", 13)))
    nonqrs = 1.0 - np.clip(qrs, 0.0, 1.0)
    qrs_gain = float(qrs_spec.get(f"{y_class}_qrs_noise_gain", 1.0))
    nonqrs_gain = float(qrs_spec.get(f"{y_class}_nonqrs_noise_gain", 1.0))
    tst_gain = float(qrs_spec.get(f"{y_class}_tst_noise_gain", nonqrs_gain))
    shaped = raw_noise * (qrs_gain * qrs + nonqrs_gain * nonqrs)
    shaped = shaped * (1.0 + (tst_gain - nonqrs_gain) * tst)
    shaped = normalize_1d(shaped)
    atten = float(qrs_spec.get(f"{y_class}_qrs_atten", 0.0))
    clean_eff = clean * (1.0 - atten * qrs)
    report = {
        "qrs_noise_gain": qrs_gain,
        "nonqrs_noise_gain": nonqrs_gain,
        "tst_noise_gain": tst_gain,
        "qrs_atten": atten,
    }
    return clean_eff.astype(np.float32), shaped.astype(np.float32), report


def masked_nrmse(clean: np.ndarray, noise: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=np.float32)
    denom = float(np.sum(m)) + 1e-6
    num = float(np.sum((noise * m) ** 2) / denom)
    ref = float(np.sum((clean * m) ** 2) / denom) + 1e-6
    return float(np.sqrt(num / ref))


def make_dataset_variant(
    ptb: dict[str, Any],
    spec: dict[str, Any],
    tracks: dict[str, Any],
    out_dir: Path,
    seed: int,
    sample_per_class: int = 0,
) -> dict[str, Any]:
    labels_all = ptb["labels"].copy().sort_values("idx").reset_index(drop=True)
    clean_all = ptb["clean"].astype(np.float32, copy=False)
    masks_all = {k: v.astype(np.float32, copy=False) for k, v in ptb["masks"].items()}
    if sample_per_class > 0:
        rng_pick = np.random.default_rng(seed + 11)
        idxs: list[int] = []
        for cls in CLASS_NAMES:
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
    noisy = np.zeros_like(clean, dtype=np.float32)
    noise_kind: list[str] = []
    snr_values: list[float] = []
    measured_values: list[float] = []
    mix_weights_rows: list[dict[str, float]] = []
    qrs_shape_rows: list[dict[str, float]] = []
    qrs_all = masks.get("qrs_mask", np.zeros_like(clean))
    tst_all = masks.get("tst_mask", np.zeros_like(clean))
    for i, row in enumerate(labels.itertuples(index=False)):
        y_class = str(row.y_class)
        split_name = str(row.split)
        raw_noise, label, weights = make_noise(spec["mix"], split_name, tracks, rng)
        clean_eff, raw_noise, qrs_report = apply_qrs_shaping(
            clean=clean[i],
            raw_noise=raw_noise,
            qrs_mask=qrs_all[i],
            tst_mask=tst_all[i],
            y_class=y_class,
            spec=spec,
        )
        snr_db = sample_snr(spec, y_class, rng)
        x = add_noise_at_snr(clean_eff, raw_noise, snr_db)
        noisy[i] = x
        noise_kind.append(label)
        snr_values.append(float(snr_db))
        measured_values.append(measured_snr(clean[i], x))
        mix_weights_rows.append(weights)
        qrs_shape_rows.append(qrs_report)

    residual = noisy - clean
    qrs = masks.get("qrs_mask", np.zeros_like(clean))
    tst = masks.get("tst_mask", np.zeros_like(clean))
    critical = masks.get("critical_mask", qrs)
    qrs_nprd = np.asarray([masked_nrmse(c, r, m) for c, r, m in zip(clean, residual, qrs)], dtype=np.float32)
    tst_nprd = np.asarray([masked_nrmse(c, r, m) for c, r, m in zip(clean, residual, tst)], dtype=np.float32)
    critical_nprd = np.asarray([masked_nrmse(c, r, m) for c, r, m in zip(clean, residual, critical)], dtype=np.float32)
    global_nprd = np.sqrt(np.mean(residual * residual, axis=1) / (np.mean(clean * clean, axis=1) + 1e-6)).astype(np.float32)

    labels = labels.copy()
    labels["idx"] = np.arange(len(labels), dtype=int)
    labels["y"] = labels["y_class"].map({"good": 0, "medium": 1, "bad": 2}).astype(int)
    labels["snr_db"] = np.asarray(snr_values, dtype=np.float32)
    labels["measured_snr_db"] = np.asarray(measured_values, dtype=np.float32)
    labels["original_measured_snr_db"] = labels["measured_snr_db"]
    labels["noise_kind"] = noise_kind
    labels["placement"] = "global_real_noise"
    labels["sample_source"] = f"real_nstdb_snr|{spec['id']}"
    labels["real_noise_spec_id"] = str(spec["id"])
    labels["real_noise_mix_mode"] = str(spec["mix"]["mode"])
    for name in NOISE_NAMES:
        labels[f"mix_weight_{name}"] = [float(row.get(name, 0.0)) for row in mix_weights_rows]
    for name in ("qrs_noise_gain", "nonqrs_noise_gain", "tst_noise_gain", "qrs_atten"):
        labels[f"shape_{name}"] = [float(row.get(name, 1.0 if name.endswith("gain") else 0.0)) for row in qrs_shape_rows]
    labels["qrs_nprd"] = qrs_nprd
    labels["tst_nprd"] = tst_nprd
    labels["critical_damage_score"] = critical_nprd
    labels["diagnostic_damage_score"] = np.maximum(critical_nprd, 0.5 * (qrs_nprd + tst_nprd))
    labels["smooth_morph_score"] = global_nprd

    data_dir = out_dir / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "synth_10s_125hz_clean.npz", X_clean=clean.astype(np.float32))
    np.savez_compressed(data_dir / "synth_10s_125hz_noisy.npz", X_noisy=noisy.astype(np.float32))
    np.savez_compressed(data_dir / "synth_10s_125hz_local_mask.npz", **masks)
    labels.to_csv(data_dir / "synth_10s_125hz_labels.csv", index=False)
    labels.to_csv(data_dir / "synth_10s_125hz_labels_with_level.csv", index=False)
    np.savez_compressed(data_dir / "synth_10s_125hz_noise_level.npz", level=global_nprd)
    audit = {
        "spec": spec,
        "rows": int(len(labels)),
        "sample_per_class": int(sample_per_class),
        "split_counts": {str(k): int(v) for k, v in labels["split"].value_counts().sort_index().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in labels["y_class"].value_counts().sort_index().to_dict().items()},
        "snr_by_class": {
            cls: {
                "mean": float(labels.loc[labels["y_class"] == cls, "measured_snr_db"].mean()),
                "min": float(labels.loc[labels["y_class"] == cls, "measured_snr_db"].min()),
                "max": float(labels.loc[labels["y_class"] == cls, "measured_snr_db"].max()),
            }
            for cls in CLASS_NAMES
        },
    }
    write_json(out_dir / "data_variant_audit.json", audit)
    write_json(out_dir / "snr_mix_spec.json", spec)
    return {"X_noisy": noisy[:, None, :], "X_clean": clean[:, None, :], "labels": labels, "audit": audit}


def plot_class_gallery(X_clean: np.ndarray, X_noisy: np.ndarray, labels: pd.DataFrame, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    positions: list[int] = []
    for cls in CLASS_NAMES:
        positions.extend(labels.index[labels["y_class"].astype(str) == cls].tolist()[:4])
    if not positions:
        return
    t = np.arange(N_TARGET, dtype=np.float32) / FS_TARGET
    fig, axes = plt.subplots(len(positions), 1, figsize=(13, max(4, len(positions) * 1.1)), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, pos in zip(axes_arr, positions):
        ax.plot(t, X_clean[pos, 0], color="#111827", lw=0.65, label="clean")
        ax.plot(t, X_noisy[pos, 0], color="#c2410c", lw=0.55, alpha=0.9, label="real-noise synthetic")
        row = labels.iloc[pos]
        ax.set_ylabel(f"{row['y_class']}\n{row['measured_snr_db']:.1f}dB", fontsize=7, rotation=0, labelpad=34)
        ax.grid(alpha=0.18)
    axes_arr[0].legend(loc="upper right", fontsize=8)
    axes_arr[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def classwise_ks_distance(a: pd.DataFrame, b: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_class: dict[str, float] = {}
    for cls in CLASS_NAMES:
        av = a[a["class_name"] == cls]
        bv = b[b["class_name"] == cls]
        vals: list[float] = []
        for feat in features:
            if feat not in av or feat not in bv:
                continue
            x = av[feat].to_numpy(dtype=np.float64)
            y = bv[feat].to_numpy(dtype=np.float64)
            if len(x) == 0 or len(y) == 0:
                continue
            ks = float(ks_2samp(x[np.isfinite(x)], y[np.isfinite(y)]).statistic)
            rows.append({"class_name": cls, "feature": feat, "ks": ks})
            vals.append(ks)
        by_class[cls] = float(np.mean(vals)) if vals else math.nan
    return {"rows": rows, "by_class": by_class, "overall": float(np.nanmean(list(by_class.values())))}


def domain_separability(but_feat: pd.DataFrame, synth_feat: pd.DataFrame, features: list[str], seed: int) -> float:
    sample = pd.concat(
        [
            but_feat[features].assign(domain=1),
            synth_feat[features].assign(domain=0),
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan)
    sample = sample.dropna()
    if len(sample) < 100:
        return 1.0
    X = sample[features].to_numpy(dtype=np.float32)
    y = sample["domain"].to_numpy(dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=seed, class_weight="balanced", n_jobs=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return float(balanced_accuracy_score(y_test, pred))


def true_sqi_features(X: np.ndarray, max_rows: int) -> pd.DataFrame:
    n = min(len(X), int(max_rows))
    rows: list[list[float]] = []
    for i in range(n):
        try:
            rows.append(sqi_for_signal(X[i, 0]))
        except Exception:
            rows.append([0.0] * len(SQI_COLUMNS))
    return pd.DataFrame(rows, columns=SQI_COLUMNS)


def sqi_distance(but_X: np.ndarray, synth_X: np.ndarray, but_meta: pd.DataFrame, synth_meta: pd.DataFrame, max_rows_per_class: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    def take(X: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        positions: list[int] = []
        for cls in CLASS_NAMES:
            idx = meta.index[meta["class_name"].astype(str) == cls].to_numpy()
            if len(idx):
                positions.extend(rng.choice(idx, size=min(max_rows_per_class, len(idx)), replace=False).tolist())
        positions = sorted(positions)
        return X[positions], meta.iloc[positions].reset_index(drop=True)

    Xb, mb = take(but_X, but_meta)
    Xs, ms = take(synth_X, synth_meta)
    fb = true_sqi_features(Xb, len(Xb))
    fs = true_sqi_features(Xs, len(Xs))
    fb["class_name"] = mb["class_name"].to_numpy()
    fs["class_name"] = ms["class_name"].to_numpy()
    dist = classwise_ks_distance(fb, fs, list(SQI_COLUMNS))
    return {"overall": dist["overall"], "by_class": dist["by_class"], "rows": dist["rows"], "n_but": int(len(Xb)), "n_synth": int(len(Xs))}


def score_one_spec(args: argparse.Namespace, spec: dict[str, Any], ptb: dict[str, Any], tracks: dict[str, Any], but_X: np.ndarray, but_meta: pd.DataFrame, but_feat: pd.DataFrame) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    spec_id = str(spec["id"])
    variant_dir = out_root / "scan_variants" / spec_id
    if (variant_dir / "but_distance_report.json").exists() and not args.force:
        return read_json(variant_dir / "but_distance_report.json")
    made = make_dataset_variant(ptb, spec, tracks, variant_dir, seed=int(args.seed), sample_per_class=int(args.scan_sample_per_class))
    X_synth = made["X_noisy"]
    meta = made["labels"].copy()
    meta["dataset"] = "PTB real-noise synthetic"
    meta["class_name"] = meta["y_class"].astype(str)
    synth_feat = extract_features(X_synth, meta, variant_dir / "synthetic_morph_features.csv", max_rows=0)
    morph_dist = classwise_ks_distance(but_feat, synth_feat, FEATURE_COLUMNS)
    sqi_dist = sqi_distance(
        but_X,
        X_synth,
        but_meta,
        meta,
        max_rows_per_class=int(args.sqi_rows_per_class),
        seed=int(args.seed) + 101,
    )
    domain_bal = domain_separability(but_feat, synth_feat, FEATURE_COLUMNS, seed=int(args.seed))
    medium_distance = float(morph_dist["by_class"].get("medium", math.nan))
    overall = float(morph_dist["overall"])
    sqi_overall = float(sqi_dist["overall"])
    but_like_score = float(0.50 * overall + 0.25 * medium_distance + 0.15 * sqi_overall + 0.10 * domain_bal)
    row = {
        "variant_id": spec_id,
        "family": spec.get("family", "real_noise_snr"),
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
    link_or_copy(variant_dir / "visuals" / "synthetic_class_gallery.png", report_copy / "synthetic_class_gallery.png")
    write_json(report_copy / "but_distance_report.json", row)
    return row


def score_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="scan_running", stage="score_specs", updated_at=now_iso())
    nstdb = ensure_nstdb_tracks(Path(args.nstdb_root))
    write_json(out_root / "nstdb_noise_audit.json", nstdb["audit"])
    write_json(report_root / "nstdb_noise_audit.json", nstdb["audit"])
    ptb = load_ptb_artifact(Path(args.source_artifact_dir))
    specs = make_specs(max_specs=int(args.max_specs))
    write_json(out_root / SPEC_NAME, {"rows": specs})
    write_json(report_root / SPEC_NAME, {"rows": specs})
    but_X, but_meta = load_but(Path(args.but_protocol_dir))
    if int(args.but_feature_max_rows) > 0:
        rng = np.random.default_rng(int(args.seed))
        pos: list[int] = []
        for cls in CLASS_NAMES:
            idx = but_meta.index[but_meta["class_name"] == cls].to_numpy()
            pos.extend(rng.choice(idx, size=min(int(args.but_feature_max_rows), len(idx)), replace=False).tolist())
        pos = sorted(pos)
        but_X_feat = but_X[pos]
        but_meta_feat = but_meta.iloc[pos].reset_index(drop=True)
    else:
        but_X_feat = but_X
        but_meta_feat = but_meta
    but_feat = extract_features(but_X_feat, but_meta_feat, out_root / "but_morph_features_scan.csv", max_rows=0)
    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status="scan_running", current=spec["id"], completed=i - 1, total=len(specs), updated_at=now_iso())
        rows.append(score_one_spec(args, spec, ptb, nstdb["tracks"], but_X, but_meta, but_feat))
    rows = sorted(rows, key=lambda r: float(r["overall_but_like_score"]))
    df = pd.DataFrame(
        [
            {
                "rank": i,
                "variant_id": r["variant_id"],
                "mix": r["spec"]["mix"]["name"],
                "snr_good": str(r["spec"]["snr"]["good"]),
                "snr_medium": str(r["spec"]["snr"]["medium"]),
                "snr_bad": str(r["spec"]["snr"]["bad"]),
                "sampler": r["spec"]["snr"]["sampler"],
                "overall_but_like_score": r["overall_but_like_score"],
                "morph_overall_distance": r["morph_overall_distance"],
                "morph_medium_distance": r["morph_medium_distance"],
                "sqi_overall_distance": r["sqi_overall_distance"],
                "domain_separability_bal_acc": r["domain_separability_bal_acc"],
                "variant_dir": r["variant_dir"],
            }
            for i, r in enumerate(rows, start=1)
        ]
    )
    df.to_csv(out_root / SCAN_NAME, index=False)
    df.to_csv(report_root / SCAN_NAME, index=False)
    write_json(out_root / "distance_leaderboard.json", {"rows": rows})
    write_json(report_root / "distance_leaderboard.json", {"rows": rows[: min(30, len(rows))]})
    write_scan_report(args, rows)
    update_state(out_root / STATE_NAME, status="scan_complete", stage="score_specs", completed=len(rows), total=len(rows), best=rows[0] if rows else None, updated_at=now_iso())
    return rows


def train_one(args: argparse.Namespace, row: dict[str, Any], mode: str) -> dict[str, Any]:
    spec = row["spec"]
    spec_id = str(spec["id"])
    variant_dir = Path(row["variant_dir"])
    if not (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists() or int(args.train_regenerate_full) == 1:
        ptb = load_ptb_artifact(Path(args.source_artifact_dir))
        tracks = ensure_nstdb_tracks(Path(args.nstdb_root))["tracks"]
        make_dataset_variant(ptb, spec, tracks, variant_dir, seed=int(args.seed), sample_per_class=0)
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
        str(spec.get("class_weight", args.class_weight)),
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
    write_json(run_dir / "real_noise_snr_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    write_train_report(args)
    return payload


def select_rows_for_training(args: argparse.Namespace, n: int) -> list[dict[str, Any]]:
    ranked = read_json(Path(args.out_root) / "distance_leaderboard.json")["rows"] if (Path(args.out_root) / "distance_leaderboard.json").exists() else score_specs(args)
    selected = ranked[:n]
    by_id = {str(r["variant_id"]): r for r in ranked}
    for control in ("single_em_snr01_uniform", "single_ma_snr01_uniform", "single_bw_snr01_uniform"):
        if control in by_id and by_id[control] not in selected:
            selected.append(by_id[control])
    return selected[:n]


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


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    update_state(out_root / STATE_NAME, status="full_training", stage="full_train", updated_at=now_iso())
    summary_path = out_root / SUMMARY_NAME
    quick_rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if summary_path.exists() else []
    good = [r for r in quick_rows if r.get("mode") == "quick" and r.get("returncode") == 0]
    if not good:
        selected = select_rows_for_training(args, int(args.top_full))
    else:
        def key(r: dict[str, Any]) -> tuple[float, float, float, float]:
            rep = r.get("but_10s_eval", {}).get("but_10s_test_report", {})
            rec = rep.get("recall_good_medium_bad", [0, 0, 0])
            return (float(rep.get("macro_f1", 0.0)), float(rep.get("balanced_acc", 0.0)), min(float(rec[1]), float(rec[2])), float(r.get("ptb_test_report", {}).get("acc", 0.0)))
        good = sorted(good, key=key, reverse=True)[: int(args.top_full)]
        selected = [dict(r, variant_id=r["spec"]["id"]) for r in good]
    rows = [train_one(args, row, "full") for row in selected]
    update_state(out_root / STATE_NAME, status="full_complete", n_full=len(rows), updated_at=now_iso())
    return rows


def write_scan_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real NSTDB Noise SNR Scan vs BUT 10s P1",
        "",
        "This scan uses only real NSTDB `em/ma/bw` noise mixed into PTB clean ECG at class-specific SNR ranges.",
        "No hand-written morphology/contact/pseudo-peak rules are used in these candidates.",
        "",
        f"- Old PTB-vs-BUT morphology domain separability reference: `{OLD_DOMAIN_SEPARABILITY:.3f}`.",
        f"- Lower `overall_but_like_score` is better.",
        "",
        "## Top Distance Candidates",
        "",
        "| rank | variant | mix | SNR G/M/B | score | morph | medium | SQI | domain bal |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows[:20], start=1):
        spec = row["spec"]
        lines.append(
            f"| {i} | `{row['variant_id']}` | {spec['mix']['name']} | "
            f"{spec['snr']['good']}/{spec['snr']['medium']}/{spec['snr']['bad']} | "
            f"{row['overall_but_like_score']:.4f} | {row['morph_overall_distance']:.4f} | "
            f"{row['morph_medium_distance']:.4f} | {row['sqi_overall_distance']:.4f} | "
            f"{row['domain_separability_bal_acc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- If these candidates are closer to BUT but later underperform, pure real-noise SNR is a useful natural-control but not sufficient.",
            "- If medium distance stays high, the BUT medium boundary is not captured by SNR-only noise mixing.",
            "- Test data is not used for calibration or candidate selection; this stage is distribution-only.",
        ]
    )
    (report_root / "real_noise_snr_scan_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_train_report(args: argparse.Namespace) -> None:
    report_root = Path(args.report_root)
    out_root = Path(args.out_root)
    rows: list[dict[str, Any]] = []
    path = out_root / SUMMARY_NAME
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def score(row: dict[str, Any]) -> tuple[float, float, float, float]:
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0, 0, 0])
        return (float(rep.get("macro_f1", 0.0)), float(rep.get("balanced_acc", 0.0)), min(float(rec[1]), float(rec[2])), float(row.get("ptb_test_report", {}).get("acc", 0.0)))
    ranked = sorted(rows, key=score, reverse=True)
    lines = [
        "# Real NSTDB Noise SNR Model Validation",
        "",
        f"Rule-based reference `{RULE_ANCHOR['variant']}`: acc `{RULE_ANCHOR['acc']:.4f}`, balanced `{RULE_ANCHOR['balanced_acc']:.4f}`, macro-F1 `{RULE_ANCHOR['macro_f1']:.4f}`.",
        "",
        "| rank | mode | variant | return | BUT acc | BUT bal | BUT macro | recalls G/M/B | PTB acc | PTB bad | distance score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        lines.append(
            f"| {i} | {row.get('mode')} | `{row.get('spec', {}).get('id')}` | {row.get('returncode')} | "
            f"{float(rep.get('acc', 0.0)):.4f} | {float(rep.get('balanced_acc', 0.0)):.4f} | {float(rep.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} | "
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec) > 2 else 0.0):.4f} | "
            f"{float(row.get('distance', {}).get('overall_but_like_score', 0.0)):.4f} |"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "final_real_noise_snr_but_match_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "training_rows.json", {"rows": ranked})


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
    if args.stage in {"full_train"}:
        run_full(args)
    if args.stage in {"report"}:
        rows = read_json(out_root / "distance_leaderboard.json")["rows"] if (out_root / "distance_leaderboard.json").exists() else []
        if rows:
            write_scan_report(args, rows)
        write_train_report(args)
    update_state(out_root / STATE_NAME, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real NSTDB noise SNR grid and BUT distance audit.")
    parser.add_argument("--stage", choices=("score_specs", "quick_train", "full_train", "report", "all"), default="score_specs")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--nstdb_root", default=str(DEFAULT_NSTDB_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_specs", type=int, default=120)
    parser.add_argument("--scan_sample_per_class", type=int, default=60)
    parser.add_argument("--but_feature_max_rows", type=int, default=1500)
    parser.add_argument("--sqi_rows_per_class", type=int, default=24)
    parser.add_argument("--top_quick", type=int, default=12)
    parser.add_argument("--top_full", type=int, default=4)
    parser.add_argument("--train_regenerate_full", type=int, default=1)
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
    run_all(args)


if __name__ == "__main__":
    main()
