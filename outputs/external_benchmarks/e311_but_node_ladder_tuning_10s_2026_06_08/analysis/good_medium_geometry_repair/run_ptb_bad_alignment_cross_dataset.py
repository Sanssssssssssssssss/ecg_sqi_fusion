"""PTB bad-alignment cross-dataset checks.

This external-only experiment keeps inference waveform-only.  It tests whether
small, interpretable PTB bad augmentations can bring synthetic bad closer to the
curated clean-BUT protocol without destroying PTB self-test or good/medium
behavior.

The augmentations are used only for PTB synthetic training rows.  The default
reporting target is the curated clean-BUT protocol.  Legacy full-BUT buckets can
be enabled only as an explicit diagnostic and must not be used for claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
CROSS_SCRIPT = ANALYSIS_DIR / "run_interpretable_clean_cross_dataset.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CROSS = load_module("interpretable_cross_for_badalign", CROSS_SCRIPT)
DUAL = CROSS.DUAL
ARCH = CROSS.ARCH
GEOM = CROSS.GEOM
CLASS_TO_INT = CROSS.CLASS_TO_INT
FORMAL_FEATURE_COLUMNS = CROSS.FORMAL_FEATURE_COLUMNS
FORMAL_FEATURE_IDX = CROSS.FORMAL_FEATURE_IDX

OUT_DIR = ANALYSIS_DIR / "ptb_bad_alignment_cross_dataset"
REPORT_OUT_DIR = REPORT_DIR / "ptb_bad_alignment_cross_dataset"
SCALE_MATCH_CACHE: dict[str, dict[str, Any]] = {}


def compute_waveform_formal_features(signals: np.ndarray) -> np.ndarray:
    primitives = GEOM.compute_primitives(signals)
    for col in FORMAL_FEATURE_COLUMNS:
        if col not in primitives.columns:
            primitives[col] = 0.0
    return (
        primitives[FORMAL_FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def load_ptb_feature_matrix(feature_source: str) -> np.ndarray:
    feature_source = str(feature_source)
    if feature_source == "manifest":
        feat_npz = np.load(ARCH.DATA_DIR / "tabular_features.npz")
        return feat_npz["features"].astype(np.float32)[:, FORMAL_FEATURE_IDX]
    if feature_source != "waveform_primitives":
        raise ValueError(f"unknown PTB feature source: {feature_source}")
    cache_path = ARCH.DATA_DIR / "tabular_features_waveform_primitives.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cols = [str(x) for x in cached["feature_columns"].tolist()]
        if cols == list(FORMAL_FEATURE_COLUMNS):
            return cached["features"].astype(np.float32)
    raw = np.load(ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    features = compute_waveform_formal_features(raw)
    np.savez_compressed(
        cache_path,
        features=features,
        feature_columns=np.asarray(FORMAL_FEATURE_COLUMNS, dtype=object),
        source="computed_from_synth_10s_125hz_noisy",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    return features


def _as_2d_signal(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    return arr


def _rms_by_class(signals: np.ndarray, class_names: np.ndarray) -> dict[str, float]:
    x = _as_2d_signal(signals)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    out: dict[str, float] = {}
    for cls in ["good", "medium", "bad"]:
        vals = rms[class_names.astype(str) == cls]
        vals = vals[np.isfinite(vals)]
        out[cls] = float(np.median(vals)) if vals.size else 1.0
    return out


def classwise_scale_factors(policy: str, labels_all: pd.DataFrame, signals_all: np.ndarray) -> dict[str, Any]:
    if not policy:
        return {"enabled": False, "factors": {"good": 1.0, "medium": 1.0, "bad": 1.0}}
    if policy in SCALE_MATCH_CACHE:
        return SCALE_MATCH_CACHE[policy]
    protocol_dir = DUAL.PROTOCOL_ROOT / policy
    atlas = pd.read_csv(protocol_dir / "original_region_atlas.csv")
    clean = _as_2d_signal(np.load(protocol_dir / "signals.npz")["X"].astype(np.float32))
    train_mask = atlas["split"].astype(str).eq("train").to_numpy()
    train_rows = atlas.loc[train_mask, "idx"].astype(int).to_numpy()
    target_classes = atlas.loc[train_mask, "class_name"].astype(str).to_numpy()
    source_classes = labels_all["y_class"].astype(str).to_numpy()
    source_rms = _rms_by_class(signals_all, source_classes)
    target_rms = _rms_by_class(clean[train_rows], target_classes)
    factors: dict[str, float] = {}
    for cls in ["good", "medium", "bad"]:
        denom = max(float(source_rms.get(cls, 1.0)), 1e-6)
        factors[cls] = float(np.clip(float(target_rms.get(cls, denom)) / denom, 0.02, 8.0))
    payload = {
        "enabled": True,
        "policy": policy,
        "source_rms_median": source_rms,
        "target_rms_median": target_rms,
        "factors": factors,
    }
    SCALE_MATCH_CACHE[policy] = payload
    return payload


def apply_classwise_signal_scale(signals: np.ndarray, labels: np.ndarray, factors: dict[str, float]) -> np.ndarray:
    x = np.asarray(signals, dtype=np.float32).copy()
    names = np.asarray(labels)
    for cls, factor in factors.items():
        if names.dtype.kind in {"i", "u"}:
            mask = names.astype(int) == CLASS_TO_INT[cls]
        else:
            mask = names.astype(str) == cls
        if np.any(mask):
            x[mask] *= float(factor)
    return x.astype(np.float32)


def bank_train_holdout_indices(
    manifest: pd.DataFrame,
    labels_all: pd.DataFrame,
    holdout_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "source_idx" not in manifest.columns:
        raise ValueError("matched bank manifest lacks source_idx")
    source_idx = manifest["source_idx"].to_numpy(dtype=np.int64)
    split_lookup = labels_all.set_index("idx")["split"].astype(str)
    source_split = np.asarray([split_lookup.get(int(idx), "") for idx in source_idx], dtype=object)
    train_allowed = np.flatnonzero(source_split == "train")
    if train_allowed.size == 0:
        raise ValueError("matched bank has no train-split source rows")
    frac = float(holdout_frac)
    if frac <= 0.0:
        return train_allowed, np.asarray([], dtype=np.int64)
    frac = min(frac, 0.50)
    rng = np.random.default_rng(int(seed) + 314159)
    holdout: list[int] = []
    if "bank_role" in manifest.columns:
        roles = manifest.iloc[train_allowed]["bank_role"].astype(str).to_numpy()
    elif "combo_source_bank" in manifest.columns:
        roles = manifest.iloc[train_allowed]["combo_source_bank"].astype(str).to_numpy()
    else:
        roles = np.asarray(["matched_bank"] * train_allowed.size, dtype=object)
    for role in sorted(set(roles.tolist())):
        role_pos = np.flatnonzero(roles == str(role))
        if role_pos.size == 0:
            continue
        n_role = max(1, int(round(role_pos.size * frac)))
        chosen = rng.choice(train_allowed[role_pos], size=min(n_role, role_pos.size), replace=False)
        holdout.extend(int(v) for v in chosen.tolist())
    holdout_arr = np.asarray(sorted(set(holdout)), dtype=np.int64)
    holdout_set = set(int(v) for v in holdout_arr.tolist())
    train_arr = np.asarray([int(v) for v in train_allowed.tolist() if int(v) not in holdout_set], dtype=np.int64)
    if train_arr.size == 0:
        raise ValueError("matched bank holdout consumed all train rows")
    return train_arr, holdout_arr


def augment_record111_bad_raw(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Bad stress: long contact/dropout, broad baseline shell, sparse reset edges.

    This targets the observed BUT bad-outlier gaps: high flat/contact/low-amp
    ratios, low zcr/detail, and large baseline span/curve.  It is intentionally
    not pure high-SNR noise.
    """
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    s = float(strength)

    if rng.random() < min(0.98, 0.58 + 0.16 * s):
        drift = float(rng.uniform(0.45, 1.55) * s)
        curve = float(rng.uniform(0.16, 0.72) * s)
        phase = float(rng.uniform(-np.pi, np.pi))
        y += drift * (0.48 * t + 0.36 * np.sin(np.pi * t + phase)) + curve * (t * t - 0.35)

    for _ in range(int(rng.integers(2, 5))):
        width = int(rng.integers(max(42, n // 18), max(80, n // 4)))
        start = int(rng.integers(0, max(1, n - width)))
        end = start + width
        mode = rng.random()
        if mode < 0.45:
            level = float(rng.uniform(-0.12, 0.12))
            y[start:end] = level + rng.normal(0.0, 0.006 * s, size=width).astype(np.float32)
        elif mode < 0.80:
            y[start:end] *= float(rng.uniform(0.015, 0.18))
        else:
            kernel = int(rng.choice([21, 31, 51]))
            filt = np.ones(kernel, dtype=np.float32) / float(kernel)
            padded = np.pad(y[start:end], (kernel // 2, kernel // 2), mode="edge")
            y[start:end] = np.convolve(padded, filt, mode="valid")[:width]

        # Transition edges / reset spikes.  These matter for BUT 111001 stress.
        for edge in (start, end - 1):
            if rng.random() < 0.55:
                lo = max(0, edge - int(rng.integers(3, 10)))
                hi = min(n, edge + int(rng.integers(4, 14)))
                local = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
                spike = np.exp(-14.0 * local * local).astype(np.float32)
                y[lo:hi] += float(rng.uniform(0.10, 0.55) * s) * spike * (1.0 if rng.random() < 0.5 else -1.0)

    if rng.random() < min(0.80, 0.30 + 0.18 * s):
        # Low-detail contact rows are not high-zcr white noise, but a small
        # amount of residual roughness avoids a trivial constant-window shortcut.
        y += rng.normal(0.0, float(rng.uniform(0.004, 0.025) * s), size=n).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_nonbad_guard_raw(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Mild artifacts that should stay good/medium, used to guard specificity."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    s = float(strength)
    if rng.random() < min(0.85, 0.32 + 0.20 * s):
        drift = float(rng.uniform(0.025, 0.12) * s)
        y += drift * (0.60 * t + 0.40 * np.sin(np.pi * t + float(rng.uniform(-1.5, 1.5))))
    if rng.random() < min(0.70, 0.24 + 0.16 * s):
        width = int(rng.integers(max(24, n // 80), max(50, n // 12)))
        start = int(rng.integers(0, max(1, n - width)))
        y[start : start + width] *= float(rng.uniform(0.45, 0.82))
    if rng.random() < 0.45:
        y += rng.normal(0.0, float(rng.uniform(0.004, 0.018) * s), size=n).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_good_flatline_lowdetail_raw(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Good guard: flat/low-detail shell while preserving readable QRS.

    PTB->clean-BUT errors showed clean-BUT good->medium rows are often high
    flatline, low non-QRS detail and low 15-45 Hz energy, but with visible QRS
    and acceptable template shape.  This augmentation teaches PTB training that
    this shell can still be good.
    """
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    s = float(strength)
    k = int(rng.integers(41, 121))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(y, kernel, mode="same").astype(np.float32)
    mix = float(rng.uniform(0.12, 0.30) * np.clip(s, 0.5, 1.6))
    y = (1.0 - mix) * y + mix * smooth
    if rng.random() < min(0.85, 0.35 + 0.20 * s):
        width = int(rng.integers(max(32, n // 95), max(72, n // 20)))
        start = int(rng.integers(0, max(1, n - width)))
        shelf = np.linspace(float(y[start]), float(y[min(n - 1, start + width - 1)]), width, dtype=np.float32)
        y[start : start + width] = 0.72 * y[start : start + width] + 0.28 * shelf
    if rng.random() < 0.65:
        t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        y += float(rng.uniform(0.018, 0.060) * s) * (0.6 * t + 0.4 * np.sin(np.pi * t + float(rng.uniform(-1.2, 1.2))))
    y *= float(rng.uniform(0.82, 1.04))
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_medium_lowqrs_hardneg_raw(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Medium guard: lower QRS visibility/detail without becoming bad stress."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    s = float(strength)
    k = int(rng.integers(21, 71))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(y, kernel, mode="same").astype(np.float32)
    y = (0.72 - 0.08 * min(s, 1.5)) * y + (0.28 + 0.08 * min(s, 1.5)) * smooth
    if rng.random() < min(0.90, 0.40 + 0.20 * s):
        width = int(rng.integers(max(48, n // 70), max(90, n // 12)))
        start = int(rng.integers(0, max(1, n - width)))
        y[start : start + width] *= float(rng.uniform(0.55, 0.82))
    if rng.random() < 0.65:
        t = np.arange(n, dtype=np.float32) / 125.0
        y += (float(rng.uniform(0.006, 0.018) * s) * np.sin(2.0 * np.pi * float(rng.uniform(8.0, 18.0)) * t + float(rng.uniform(-np.pi, np.pi)))).astype(np.float32)
    y += rng.normal(0.0, float(rng.uniform(0.002, 0.008) * s), size=n).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class BadAlignedSyntheticDualviewDataset(Dataset):
    def __init__(
        self,
        split: str,
        aug_mode: str,
        bad_aug_prob: float,
        bad_aug_strength: float,
        nonbad_guard_prob: float,
        nonbad_guard_strength: float,
        channel_stats: Any | None = None,
        feature_norm: Any | None = None,
        stats_on_augmented_train: bool = False,
        matched_bank_path: str = "",
        matched_bank_frac: float = 0.0,
        matched_bank_select: str = "random",
        matched_bank_holdout_frac: float = 0.0,
        seed: int = 2026061901,
        ptb_feature_source: str = "manifest",
        scale_match_policy: str = "",
    ) -> None:
        labels_all = pd.read_csv(ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
        signals_all = np.load(ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        scale_match = classwise_scale_factors(scale_match_policy, labels_all, signals_all) if str(scale_match_policy) else {"enabled": False, "factors": {"good": 1.0, "medium": 1.0, "bad": 1.0}}
        if scale_match.get("enabled"):
            signals_all = apply_classwise_signal_scale(
                signals_all,
                labels_all["y_class"].astype(str).to_numpy(),
                scale_match["factors"],
            )
        features_all = load_ptb_feature_matrix(ptb_feature_source)

        split_mask = labels_all["split"].astype(str).eq(split).to_numpy()
        frame = labels_all.loc[split_mask].reset_index(drop=True)
        rows = frame["idx"].to_numpy(dtype=np.int64)
        signals = signals_all[rows].copy()
        features = features_all[rows].copy()
        y = frame["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        stress = np.zeros(len(y), dtype=np.float32)
        artifact = np.zeros(len(y), dtype=np.float32)

        self.augmentation_summary: dict[str, Any] = {
            "split": split,
            "aug_mode": aug_mode,
            "bad_aug_prob": float(bad_aug_prob),
            "bad_aug_strength": float(bad_aug_strength),
            "nonbad_guard_prob": float(nonbad_guard_prob),
            "nonbad_guard_strength": float(nonbad_guard_strength),
            "matched_bank_path": str(matched_bank_path),
            "matched_bank_frac": float(matched_bank_frac),
            "matched_bank_select": str(matched_bank_select),
            "matched_bank_holdout_frac": float(matched_bank_holdout_frac),
            "scale_match": scale_match,
            "original_rows": int(len(y)),
            "bad_augmented_added": 0,
            "nonbad_guard_added": 0,
            "matched_bank_added": 0,
            "artifact_positive_rows": 0,
        }

        if split == "train" and str(aug_mode).startswith("distribution_bank"):
            bank_path = Path(matched_bank_path)
            if not bank_path.exists():
                raise FileNotFoundError(f"matched bad distribution bank not found: {bank_path}")
            rng = np.random.default_rng(seed)
            bank = np.load(bank_path)["X"].astype(np.float32)
            manifest_path = bank_path.with_name("ptb_bad_distribution_matched_manifest.csv")
            if not manifest_path.exists() and bank_path.name.endswith("_signals.npz"):
                manifest_path = bank_path.with_name(bank_path.name.replace("_signals.npz", "_manifest.csv"))
            if not manifest_path.exists():
                raise FileNotFoundError(f"matched bank manifest not found: {manifest_path}")
            manifest = pd.read_csv(manifest_path)
            source_idx = manifest["source_idx"].to_numpy(dtype=np.int64)
            allowed, holdout = bank_train_holdout_indices(
                manifest,
                labels_all,
                float(matched_bank_holdout_frac),
                int(seed),
            )
            if float(matched_bank_frac) <= 0:
                n_take = 0
            elif float(matched_bank_frac) <= 1.0:
                n_take = max(1, int(round(allowed.size * float(matched_bank_frac))))
            else:
                n_take = min(allowed.size, int(round(float(matched_bank_frac))))
            if n_take > 0:
                select_mode = str(matched_bank_select)
                if select_mode == "random":
                    take = rng.choice(allowed, size=n_take, replace=False)
                elif select_mode == "order":
                    take = allowed[:n_take]
                elif select_mode == "stratified_order":
                    if "target_original_region" not in manifest.columns:
                        take = allowed[:n_take]
                    else:
                        regions = manifest.iloc[allowed]["target_original_region"].astype(str).to_numpy()
                        taken: list[int] = []
                        used = np.zeros(allowed.size, dtype=bool)
                        unique_regions, counts = np.unique(regions, return_counts=True)
                        quotas: dict[str, int] = {}
                        remaining = int(n_take)
                        for region, count in zip(unique_regions, counts):
                            quota = int(np.floor(float(n_take) * float(count) / float(allowed.size)))
                            quotas[str(region)] = min(int(count), quota)
                            remaining -= quotas[str(region)]
                        if remaining > 0:
                            frac_parts = [
                                (
                                    (float(n_take) * float(count) / float(allowed.size)) - quotas[str(region)],
                                    str(region),
                                )
                                for region, count in zip(unique_regions, counts)
                                if quotas[str(region)] < int(count)
                            ]
                            for _, region in sorted(frac_parts, reverse=True)[:remaining]:
                                quotas[region] += 1
                        for region in unique_regions:
                            region_pos = np.flatnonzero(regions == str(region))
                            quota = quotas.get(str(region), 0)
                            if quota > 0:
                                chosen_pos = region_pos[:quota]
                                taken.extend(allowed[chosen_pos].tolist())
                                used[chosen_pos] = True
                        if len(taken) < n_take:
                            fill = allowed[~used][: n_take - len(taken)]
                            taken.extend(fill.tolist())
                        take = np.asarray(taken[:n_take], dtype=np.int64)
                else:
                    raise ValueError(f"unknown matched_bank_select={select_mode}")
                add_x = bank[take].astype(np.float32)
                add_src = source_idx[take]
                if str(ptb_feature_source) == "waveform_primitives":
                    add_f = compute_waveform_formal_features(add_x)
                else:
                    add_f = features_all[add_src].astype(np.float32)
                if "y_class" in manifest.columns:
                    add_label_names = manifest.iloc[take]["y_class"].astype(str).to_numpy()
                    add_y = np.asarray([CLASS_TO_INT[name] for name in add_label_names], dtype=np.int64)
                else:
                    add_label_names = np.asarray(["bad"] * n_take, dtype=object)
                    add_y = np.full(n_take, CLASS_TO_INT["bad"], dtype=np.int64)
                if "stress" in manifest.columns:
                    add_s = manifest.iloc[take]["stress"].astype(float).to_numpy(dtype=np.float32)
                else:
                    add_s = (add_y == CLASS_TO_INT["bad"]).astype(np.float32)
                if "artifact" in manifest.columns:
                    add_a = manifest.iloc[take]["artifact"].astype(float).to_numpy(dtype=np.float32)
                else:
                    role_text = np.asarray([""] * n_take, dtype=object)
                    for col in ["bank_role", "combo_source_bank", "profile_block", "target_original_region"]:
                        if col in manifest.columns:
                            role_text = np.char.add(
                                np.char.add(role_text.astype(str), "|"),
                                manifest.iloc[take][col].fillna("").astype(str).to_numpy(),
                            )
                    role_lower = np.char.lower(role_text.astype(str))
                    text_artifact = (
                        np.char.find(role_lower, "controlled") >= 0
                    ) | (
                        np.char.find(role_lower, "contact") >= 0
                    ) | (
                        np.char.find(role_lower, "mild_noise") >= 0
                    ) | (
                        np.char.find(role_lower, "outlier") >= 0
                    ) | (
                        np.char.find(role_lower, "bad_distribution") >= 0
                    )
                    add_a = np.where((add_y == CLASS_TO_INT["bad"]) | text_artifact, 1.0, 0.0).astype(np.float32)
                if scale_match.get("enabled"):
                    add_x = apply_classwise_signal_scale(add_x, add_y, scale_match["factors"])
                add_rows = labels_all.set_index("idx").loc[add_src].reset_index()
                add_rows["split"] = "train"
                add_rows["y_class"] = add_label_names
                add_rows["block"] = add_rows.get("block", pd.Series("synthetic", index=add_rows.index)).astype(str) + "|distribution_bank"
                signals = np.concatenate([signals, add_x], axis=0)
                features = np.concatenate([features, add_f], axis=0)
                y = np.concatenate([y, add_y], axis=0)
                stress = np.concatenate([stress, add_s], axis=0)
                artifact = np.concatenate([artifact, add_a], axis=0)
                frame = pd.concat([frame, add_rows], ignore_index=True)
            self.augmentation_summary["matched_bank_added"] = int(n_take)
            self.augmentation_summary["matched_bank_train_pool"] = int(allowed.size)
            self.augmentation_summary["matched_bank_holdout"] = int(holdout.size)

        if split == "train" and str(aug_mode) != "none" and (
            not str(aug_mode).startswith("distribution_bank") or str(aug_mode) == "distribution_bank_artifact"
        ):
            rng = np.random.default_rng(seed)
            extra_signals: list[np.ndarray] = []
            extra_features: list[np.ndarray] = []
            extra_y: list[int] = []
            extra_stress: list[float] = []
            extra_artifact: list[float] = []
            extra_rows: list[pd.Series] = []
            for i, label in enumerate(y):
                if label == CLASS_TO_INT["bad"] and rng.random() < float(bad_aug_prob):
                    extra_signals.append(augment_record111_bad_raw(signals[i], rng, bad_aug_strength))
                    extra_features.append(features[i].copy())
                    extra_y.append(int(label))
                    extra_stress.append(1.0)
                    extra_artifact.append(1.0)
                    row = frame.iloc[i].copy()
                    row["block"] = f"{str(row.get('block', 'synthetic'))}|bad_align_record111"
                    extra_rows.append(row)
                elif label != CLASS_TO_INT["bad"] and rng.random() < float(nonbad_guard_prob):
                    if str(aug_mode) == "gm_overlap_guard" and label == CLASS_TO_INT["good"]:
                        extra_signals.append(augment_good_flatline_lowdetail_raw(signals[i], rng, nonbad_guard_strength))
                    elif str(aug_mode) == "gm_overlap_guard" and label == CLASS_TO_INT["medium"]:
                        extra_signals.append(augment_medium_lowqrs_hardneg_raw(signals[i], rng, nonbad_guard_strength))
                    else:
                        extra_signals.append(augment_nonbad_guard_raw(signals[i], rng, nonbad_guard_strength))
                    extra_features.append(features[i].copy())
                    extra_y.append(int(label))
                    extra_stress.append(0.0)
                    extra_artifact.append(1.0)
                    row = frame.iloc[i].copy()
                    row["block"] = f"{str(row.get('block', 'synthetic'))}|{str(aug_mode) if str(aug_mode) == 'gm_overlap_guard' else 'nonbad_guard'}"
                    extra_rows.append(row)
            if extra_signals:
                add_x = np.stack(extra_signals).astype(np.float32)
                if str(ptb_feature_source) == "waveform_primitives":
                    add_f = compute_waveform_formal_features(add_x)
                else:
                    add_f = np.stack(extra_features).astype(np.float32)
                add_y = np.asarray(extra_y, dtype=np.int64)
                add_s = np.asarray(extra_stress, dtype=np.float32)
                add_a = np.asarray(extra_artifact, dtype=np.float32)
                signals = np.concatenate([signals, add_x], axis=0)
                features = np.concatenate([features, add_f], axis=0)
                y = np.concatenate([y, add_y], axis=0)
                stress = np.concatenate([stress, add_s], axis=0)
                artifact = np.concatenate([artifact, add_a], axis=0)
                frame = pd.concat([frame, pd.DataFrame(extra_rows)], ignore_index=True)
            self.augmentation_summary["bad_augmented_added"] = int(sum(v == CLASS_TO_INT["bad"] for v in extra_y))
            self.augmentation_summary["nonbad_guard_added"] = int(sum(v != CLASS_TO_INT["bad"] for v in extra_y))

        if channel_stats is None:
            if split == "train" and stats_on_augmented_train:
                channel_stats = DUAL.compute_channel_stats(signals)
            else:
                train_rows = labels_all.loc[labels_all["split"].astype(str).eq("train"), "idx"].to_numpy(dtype=np.int64)
                channel_stats = DUAL.compute_channel_stats(signals_all[train_rows])
        if feature_norm is None:
            if split == "train" and stats_on_augmented_train:
                train_raw = features
            else:
                train_rows = labels_all.loc[labels_all["split"].astype(str).eq("train"), "idx"].to_numpy(dtype=np.int64)
                train_raw = features_all[train_rows]
            mean = train_raw.mean(axis=0).astype(np.float32)
            std = train_raw.std(axis=0).astype(np.float32)
            std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
            feature_norm = DUAL.FeatureNorm(mean=mean, std=std)

        self.channel_stats = channel_stats
        self.feature_norm = feature_norm
        self.frame = frame.reset_index(drop=True)
        self.x = DUAL.make_dualview_channels(signals, channel_stats)
        self.aux = ((features - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.y = y.astype(np.int64)
        self.stress = stress.astype(np.float32)
        self.artifact = artifact.astype(np.float32)
        self.split = self.frame["split"].astype(str).to_numpy()
        self.region = self.frame.get("block", pd.Series("synthetic", index=self.frame.index)).astype(str).to_numpy()
        self.augmentation_summary["artifact_positive_rows"] = int(np.sum(self.artifact > 0.5))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
            "stress": torch.tensor(float(self.stress[i]), dtype=torch.float32),
            "artifact": torch.tensor(float(self.artifact[i]), dtype=torch.float32),
        }


class MatchedBankStressValidationDataset(Dataset):
    def __init__(
        self,
        matched_bank_path: str,
        holdout_frac: float,
        channel_stats: Any,
        feature_norm: Any,
        seed: int,
        ptb_feature_source: str = "manifest",
        scale_match_policy: str = "",
    ) -> None:
        if float(holdout_frac) <= 0.0:
            raise ValueError("MatchedBankStressValidationDataset requires holdout_frac > 0")
        labels_all = pd.read_csv(ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
        signals_all = np.load(ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        scale_match = classwise_scale_factors(scale_match_policy, labels_all, signals_all) if str(scale_match_policy) else {"enabled": False, "factors": {"good": 1.0, "medium": 1.0, "bad": 1.0}}
        features_all = load_ptb_feature_matrix(ptb_feature_source)
        bank_path = Path(matched_bank_path)
        if not bank_path.exists():
            raise FileNotFoundError(f"matched bank not found: {bank_path}")
        bank = np.load(bank_path)["X"].astype(np.float32)
        manifest_path = bank_path.with_name("ptb_bad_distribution_matched_manifest.csv")
        if not manifest_path.exists() and bank_path.name.endswith("_signals.npz"):
            manifest_path = bank_path.with_name(bank_path.name.replace("_signals.npz", "_manifest.csv"))
        if not manifest_path.exists():
            raise FileNotFoundError(f"matched bank manifest not found: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        _, holdout = bank_train_holdout_indices(manifest, labels_all, float(holdout_frac), int(seed))
        if holdout.size == 0:
            raise ValueError("matched bank stress validation holdout is empty")
        source_idx = manifest["source_idx"].to_numpy(dtype=np.int64)
        x = bank[holdout].astype(np.float32)
        if "y_class" in manifest.columns:
            label_names = manifest.iloc[holdout]["y_class"].astype(str).to_numpy()
        else:
            label_names = np.asarray(["bad"] * holdout.size, dtype=object)
        y = np.asarray([CLASS_TO_INT[name] for name in label_names], dtype=np.int64)
        if scale_match.get("enabled"):
            x = apply_classwise_signal_scale(x, y, scale_match["factors"])
        if str(ptb_feature_source) == "waveform_primitives":
            features = compute_waveform_formal_features(x)
        else:
            features = features_all[source_idx[holdout]].astype(np.float32)
        self.channel_stats = channel_stats
        self.feature_norm = feature_norm
        self.frame = manifest.iloc[holdout].reset_index(drop=True).copy()
        self.frame["split"] = "stress_val"
        self.frame["y_class"] = label_names
        self.x = DUAL.make_dualview_channels(x, channel_stats)
        self.aux = ((features - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.y = y.astype(np.int64)
        self.stress = (self.y == CLASS_TO_INT["bad"]).astype(np.float32)
        if "artifact" in self.frame.columns:
            self.artifact = self.frame["artifact"].astype(float).to_numpy(dtype=np.float32)
        else:
            role_text = np.asarray([""] * len(self.y), dtype=object)
            for col in ["bank_role", "combo_source_bank", "profile_block", "target_original_region"]:
                if col in self.frame.columns:
                    role_text = np.char.add(
                        np.char.add(role_text.astype(str), "|"),
                        self.frame[col].fillna("").astype(str).to_numpy(),
                    )
            role_lower = np.char.lower(role_text.astype(str))
            text_artifact = (
                np.char.find(role_lower, "controlled") >= 0
            ) | (
                np.char.find(role_lower, "contact") >= 0
            ) | (
                np.char.find(role_lower, "mild_noise") >= 0
            ) | (
                np.char.find(role_lower, "outlier") >= 0
            ) | (
                np.char.find(role_lower, "bad_distribution") >= 0
            )
            self.artifact = np.where((self.y == CLASS_TO_INT["bad"]) | text_artifact, 1.0, 0.0).astype(np.float32)
        self.split = np.asarray(["stress_val"] * len(self.y), dtype=object)
        if "bank_role" in self.frame.columns:
            self.region = self.frame["bank_role"].astype(str).to_numpy()
        elif "combo_source_bank" in self.frame.columns:
            self.region = self.frame["combo_source_bank"].astype(str).to_numpy()
        else:
            self.region = np.asarray(["matched_bank_stress_val"] * len(self.y), dtype=object)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
            "stress": torch.tensor(float(self.stress[i]), dtype=torch.float32),
            "artifact": torch.tensor(float(self.artifact[i]), dtype=torch.float32),
        }


class StressAwareHierTransformer(nn.Module):
    def __init__(self, cfg: dict[str, Any], in_ch: int, aux_dim: int):
        super().__init__()
        self.base = DUAL.HierTransformer(cfg, in_ch, aux_dim)
        dim = int(self.base.encoder.out_dim)
        self.stress_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 1),
        )
        self.artifact_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base(x)
        out["stress_logit"] = self.stress_head(out["features"]).squeeze(1)
        out["artifact_logit"] = self.artifact_head(out["features"]).squeeze(1)
        return out


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def recall_spread(rep: dict[str, Any]) -> float:
    vals = [
        float(rep["good_recall"]),
        float(rep["medium_recall"]),
        float(rep["bad_recall"]),
    ]
    return float(max(vals) - min(vals))


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


INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


def prediction_frame(ds: Any, probs: np.ndarray, bucket: str) -> pd.DataFrame:
    frame = ds.frame.copy().reset_index(drop=True)
    pred = probs.argmax(axis=1).astype(np.int64)
    frame["capacity_bucket"] = bucket
    frame["true_int"] = ds.y.astype(np.int64)
    frame["pred_int"] = pred
    frame["true_label"] = [INT_TO_CLASS.get(int(v), str(int(v))) for v in ds.y]
    frame["pred_label"] = [INT_TO_CLASS.get(int(v), str(int(v))) for v in pred]
    frame["correct"] = frame["true_int"].eq(frame["pred_int"])
    frame["prob_good"] = probs[:, CLASS_TO_INT["good"]]
    frame["prob_medium"] = probs[:, CLASS_TO_INT["medium"]]
    frame["prob_bad"] = probs[:, CLASS_TO_INT["bad"]]
    if hasattr(ds, "aux_raw"):
        raw = np.asarray(ds.aux_raw, dtype=np.float32)
        for j, col in enumerate(FORMAL_FEATURE_COLUMNS):
            if col not in frame.columns and j < raw.shape[1]:
                frame[col] = raw[:, j]
    return frame


def apply_input_view(ds: Any, input_view: str) -> None:
    view = str(input_view)
    if view == "full":
        return
    if view == "robust_no_physical":
        # Drop physical and physical_trend channels, which can encode a
        # PTB-vs-BUT amplitude-domain shortcut.  Retain per-window robust
        # morphology, derivative, baseline, highpass, and local detail.
        keep = [1, 2, 3, 4, 5, 6]
    elif view == "robust_detail_only":
        keep = [1, 2, 4, 5, 6]
    else:
        raise ValueError(f"unknown input_view={input_view}")
    ds.x = ds.x[:, keep, :].astype(np.float32)


def plot_aug_examples(ds: BadAlignedSyntheticDualviewDataset, out_path: Path) -> None:
    mask = np.array(["bad_align_record111" in r for r in ds.region])
    idx = np.where(mask & (ds.y == CLASS_TO_INT["bad"]))[0][:12]
    if len(idx) == 0:
        return
    fig, axes = plt.subplots(3, 4, figsize=(12, 6), dpi=150, sharex=True)
    t = np.arange(ds.x.shape[-1]) / 125.0
    for ax, row_idx in zip(axes.ravel(), idx):
        # Plot the first physical/robust channels from dual-view input.
        for ch, color in zip([0, 1, 3], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            if ch < ds.x.shape[1]:
                ax.plot(t, ds.x[row_idx, ch] + 2.2 * (ch != 0), lw=0.7, color=color)
        ax.set_title("PTB bad-align aug", fontsize=8)
        ax.set_yticks([])
        ax.grid(alpha=0.12)
    fig.suptitle("PTB synthetic bad rows after record111-like stress augmentation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def train_badalign_candidate(name: str, policy: str, cfg: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"bad-align train {name}: policy={policy} mode={args.aug_mode}", flush=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = BadAlignedSyntheticDualviewDataset(
        "train",
        args.aug_mode,
        args.bad_aug_prob,
        args.bad_aug_strength,
        args.nonbad_guard_prob,
        args.nonbad_guard_strength,
        stats_on_augmented_train=args.stats_on_augmented_train,
        matched_bank_path=args.matched_bank,
        matched_bank_frac=args.matched_bank_frac,
        matched_bank_select=args.matched_bank_select,
        matched_bank_holdout_frac=args.matched_bank_holdout_frac,
        ptb_feature_source=args.ptb_feature_source,
        scale_match_policy=args.scale_match_policy,
        seed=int(cfg["seed"]) + 991,
    )
    val_ds = BadAlignedSyntheticDualviewDataset(
        "val",
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        train_ds.channel_stats,
        train_ds.feature_norm,
        ptb_feature_source=args.ptb_feature_source,
        scale_match_policy=args.scale_match_policy,
    )
    ptb_test = BadAlignedSyntheticDualviewDataset(
        "test",
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        train_ds.channel_stats,
        train_ds.feature_norm,
        ptb_feature_source=args.ptb_feature_source,
        scale_match_policy=args.scale_match_policy,
    )
    clean_val = DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, "val", train_ds.channel_stats, train_ds.feature_norm)
    clean_test = DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, "test", train_ds.channel_stats, train_ds.feature_norm)
    stress_val = None
    if str(args.aug_mode).startswith("distribution_bank") and float(args.matched_bank_holdout_frac) > 0:
        stress_val = MatchedBankStressValidationDataset(
            args.matched_bank,
            float(args.matched_bank_holdout_frac),
            train_ds.channel_stats,
            train_ds.feature_norm,
            int(cfg["seed"]) + 991,
            ptb_feature_source=args.ptb_feature_source,
            scale_match_policy=args.scale_match_policy,
        )
    full_but = None
    if bool(args.include_full_diagnostic):
        full_but = DUAL.FullButDataset(train_ds.channel_stats, train_ds.feature_norm)

    for ds in [train_ds, val_ds, ptb_test, clean_val, clean_test] + ([stress_val] if stress_val is not None else []) + ([full_but] if full_but is not None else []):
        apply_input_view(ds, args.input_view)

    if float(args.stress_aux_weight) > 0 or float(args.artifact_aux_weight) > 0:
        model = StressAwareHierTransformer(cfg, int(train_ds.x.shape[1]), len(FORMAL_FEATURE_COLUMNS)).to(device)
    else:
        model = DUAL.HierTransformer(cfg, int(train_ds.x.shape[1]), len(FORMAL_FEATURE_COLUMNS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    stress_val_loader = None
    if stress_val is not None:
        stress_val_loader = DataLoader(stress_val, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    clean_train_loader: DataLoader | None = None
    if int(args.pretrain_clean_epochs) > 0:
        clean_train_ds = BadAlignedSyntheticDualviewDataset(
            "train",
            "none",
            0.0,
            0.0,
            0.0,
            0.0,
            train_ds.channel_stats,
            train_ds.feature_norm,
            ptb_feature_source=args.ptb_feature_source,
            scale_match_policy=args.scale_match_policy,
        )
        apply_input_view(clean_train_ds, args.input_view)
        clean_train_loader = DataLoader(clean_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -1e9
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    logs: list[dict[str, Any]] = []
    def train_epoch(loader: DataLoader) -> float:
        model.train()
        losses: list[float] = []
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(x)
            loss = DUAL.classification_losses(out, y, aux, cfg, device)
            if float(args.stress_aux_weight) > 0:
                stress = batch["stress"].to(device=device, dtype=torch.float32)
                pos = stress.sum().clamp_min(1.0)
                neg = (1.0 - stress).sum().clamp_min(1.0)
                pos_weight = (neg / pos).clamp(1.0, 12.0)
                stress_loss = F.binary_cross_entropy_with_logits(out["stress_logit"], stress, pos_weight=pos_weight)
                nonstress_prob = torch.sigmoid(out["stress_logit"][stress < 0.5])
                nonstress_penalty = nonstress_prob.mean() if nonstress_prob.numel() else torch.zeros((), device=device)
                loss = loss + float(args.stress_aux_weight) * stress_loss + float(args.stress_nonbad_penalty) * nonstress_penalty
            if float(args.artifact_aux_weight) > 0:
                artifact = batch["artifact"].to(device=device, dtype=torch.float32)
                pos = artifact.sum().clamp_min(1.0)
                neg = (1.0 - artifact).sum().clamp_min(1.0)
                pos_weight = (neg / pos).clamp(1.0, 12.0)
                artifact_loss = F.binary_cross_entropy_with_logits(out["artifact_logit"], artifact, pos_weight=pos_weight)
                loss = loss + float(args.artifact_aux_weight) * artifact_loss
                nonbad_artifact = (artifact > 0.5) & (y != CLASS_TO_INT["bad"])
                if float(args.artifact_nonbad_bad_penalty) > 0 and torch.any(nonbad_artifact):
                    bad_prob = torch.softmax(out["logits"][nonbad_artifact], dim=1)[:, CLASS_TO_INT["bad"]]
                    loss = loss + float(args.artifact_nonbad_bad_penalty) * bad_prob.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    for pre_epoch in range(1, int(args.pretrain_clean_epochs) + 1):
        assert clean_train_loader is not None
        loss_value = train_epoch(clean_train_loader)
        val_rep, _, _, _ = DUAL.eval_loader(model, val_loader, device)
        logs.append({"phase": "pretrain_clean", "epoch": pre_epoch, "train_loss": loss_value, **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"{name} pretrain={pre_epoch}/{args.pretrain_clean_epochs} loss={loss_value:.4f} val_acc={val_rep['acc']:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
    if int(args.pretrain_clean_epochs) > 0 and float(args.finetune_lr_scale) != 1.0:
        for group in opt.param_groups:
            group["lr"] = float(cfg["lr"]) * float(args.finetune_lr_scale)
        best_score = -1e9
        best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        loss_value = train_epoch(train_loader)
        val_rep, _, _, _ = DUAL.eval_loader(model, val_loader, device)
        clean_val_rep, _, _, _ = DUAL.eval_loader(model, clean_val_loader, device)
        stress_val_rep = None
        if stress_val_loader is not None:
            stress_val_rep, _, _, _ = DUAL.eval_loader(model, stress_val_loader, device)
        score = (
            val_rep["acc"]
            + 0.15 * val_rep["macro_f1"]
            + 0.10 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
            - 0.02 * val_rep["aux_mae"]
        )
        if str(args.selection_mode) == "ptb_stress_joint":
            if stress_val_rep is None:
                raise ValueError("selection_mode=ptb_stress_joint requires --aug-mode distribution_bank and --matched-bank-holdout-frac > 0")
            stress_guard = min(stress_val_rep["good_recall"], stress_val_rep["medium_recall"], stress_val_rep["bad_recall"])
            score = (
                0.45 * val_rep["acc"]
                + 0.10 * val_rep["macro_f1"]
                + 0.25 * stress_val_rep["acc"]
                + 0.20 * stress_guard
                - 0.02 * val_rep["aux_mae"]
            )
        if str(args.selection_mode) == "ptb_stress_balance":
            if stress_val_rep is None:
                raise ValueError("selection_mode=ptb_stress_balance requires --aug-mode distribution_bank and --matched-bank-holdout-frac > 0")
            val_guard = min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
            stress_guard = min(stress_val_rep["good_recall"], stress_val_rep["medium_recall"], stress_val_rep["bad_recall"])
            score = (
                0.35 * val_rep["acc"]
                + 0.20 * val_guard
                + 0.25 * stress_val_rep["acc"]
                + 0.30 * stress_guard
                - 0.55 * recall_spread(stress_val_rep)
                - 0.25 * recall_spread(val_rep)
                - 0.02 * val_rep["aux_mae"]
            )
        if str(args.selection_mode) == "ptb_stress_badguard":
            if stress_val_rep is None:
                raise ValueError("selection_mode=ptb_stress_badguard requires --aug-mode distribution_bank and --matched-bank-holdout-frac > 0")
            val_guard = min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
            stress_guard = min(stress_val_rep["good_recall"], stress_val_rep["medium_recall"], stress_val_rep["bad_recall"])
            val_bad_short = max(0.0, 0.985 - float(val_rep["bad_recall"]))
            stress_bad_short = max(0.0, 0.985 - float(stress_val_rep["bad_recall"]))
            score = (
                0.30 * val_rep["acc"]
                + 0.15 * val_guard
                + 0.25 * stress_val_rep["acc"]
                + 0.20 * stress_guard
                + 0.10 * val_rep["bad_recall"]
                + 0.15 * stress_val_rep["bad_recall"]
                - 0.55 * recall_spread(stress_val_rep)
                - 0.30 * recall_spread(val_rep)
                - 1.50 * val_bad_short
                - 2.50 * stress_bad_short
                - 0.02 * val_rep["aux_mae"]
            )
        if str(args.selection_mode) == "clean_val_joint":
            clean_gm_guard = min(clean_val_rep["good_recall"], clean_val_rep["medium_recall"])
            score = (
                0.60 * clean_val_rep["acc"]
                + 0.15 * clean_val_rep["macro_f1"]
                + 0.10 * clean_gm_guard
                + 0.20 * val_rep["acc"]
                + 0.05 * val_rep["macro_f1"]
                - 0.02 * val_rep["aux_mae"]
            )
        logs.append({"phase": "badalign_finetune", "epoch": epoch, "train_loss": loss_value, **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        logs[-1].update({f"clean_val_{k}": v for k, v in clean_val_rep.items() if k != "confusion_3x3"})
        if stress_val_rep is not None:
            logs[-1].update({f"stress_val_{k}": v for k, v in stress_val_rep.items() if k != "confusion_3x3"})
        print(
            f"{name} epoch={epoch}/{args.epochs} loss={loss_value:.4f} val_acc={val_rep['acc']:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}"
            + (
                f" stress_acc={stress_val_rep['acc']:.4f} stress_recalls={stress_val_rep['good_recall']:.3f}/{stress_val_rep['medium_recall']:.3f}/{stress_val_rep['bad_recall']:.3f}"
                if stress_val_rep is not None
                else ""
            ),
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if str(args.average_epochs).strip():
            epoch_states[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    averaged_epochs: list[int] = []
    if str(args.average_epochs).strip():
        averaged_epochs = [int(x.strip()) for x in str(args.average_epochs).split(",") if x.strip()]
        missing = [e for e in averaged_epochs if e not in epoch_states]
        if missing:
            raise ValueError(f"--average-epochs requested missing epochs {missing}; available={sorted(epoch_states)}")
        avg_state: dict[str, torch.Tensor] = {}
        for key in epoch_states[averaged_epochs[0]]:
            tensors = [epoch_states[e][key].to(dtype=torch.float32) for e in averaged_epochs]
            avg = torch.stack(tensors, dim=0).mean(dim=0)
            avg_state[key] = avg.to(dtype=epoch_states[averaged_epochs[0]][key].dtype)
        best_state = avg_state
        best_epoch = -1
    model.load_state_dict(best_state)

    val_rep, _ = eval_dataset(model, val_ds, args.batch_size, device)
    stress_val_rep = None
    if stress_val is not None:
        stress_val_rep, _ = eval_dataset(model, stress_val, args.batch_size, device)
    clean_val_rep, clean_val_probs = eval_dataset(model, clean_val, args.batch_size, device)
    ptb_rep, ptb_probs = eval_dataset(model, ptb_test, args.batch_size, device)
    clean_rep, clean_probs = eval_dataset(model, clean_test, args.batch_size, device)
    full_probs = None
    if full_but is not None:
        _, full_probs = eval_dataset(model, full_but, args.batch_size, device)

    rows = [
        metric_row(name, "ptb_val", val_rep),
        metric_row(name, f"but_clean_{policy}_val", clean_val_rep),
        metric_row(name, "ptb_synthetic_test", ptb_rep),
        metric_row(name, f"but_clean_{policy}_test", clean_rep),
    ]
    if stress_val_rep is not None:
        rows.insert(1, metric_row(name, "ptb_combo_stress_val", stress_val_rep))
    if full_but is not None and full_probs is not None:
        rows.extend(
            [
                metric_row(name, "legacy_full_test_all_10s+", bucket_report(full_but, full_probs, "test")),
                metric_row(name, "legacy_full_all_10s+", bucket_report(full_but, full_probs, "all")),
                metric_row(name, "legacy_full_bad_core_nearboundary", bucket_report(full_but, full_probs, "bad_core_nearboundary")),
                metric_row(name, "legacy_full_bad_outlier_stress", bucket_report(full_but, full_probs, "bad_outlier_stress")),
            ]
        )
    safe_name = name
    if len(safe_name) > 96:
        safe_name = "pba_run_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    run_dir = DUAL.OUT_ROOT / "runs" / "ptb_bad_alignment_cross_dataset" / safe_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "fixed_10s_ptb_bad_alignment_transformer",
            "input_contract": "fixed 10s waveform-derived channels only; no feature input at inference",
            "train_domain": "ptb_synthetic_bad_aligned",
            "policy": policy,
            "augmentation_summary": train_ds.augmentation_summary,
            "stress_aux_weight": float(args.stress_aux_weight),
            "stress_nonbad_penalty": float(args.stress_nonbad_penalty),
            "artifact_aux_weight": float(args.artifact_aux_weight),
            "artifact_nonbad_bad_penalty": float(args.artifact_nonbad_bad_penalty),
            "selection_mode": str(args.selection_mode),
            "pretrain_clean_epochs": int(args.pretrain_clean_epochs),
            "finetune_lr_scale": float(args.finetune_lr_scale),
            "ptb_feature_source": str(args.ptb_feature_source),
            "input_view": str(args.input_view),
            "scale_match_policy": str(args.scale_match_policy),
            "best_epoch": best_epoch,
            "averaged_epochs": averaged_epochs,
            "channel_stats": train_ds.channel_stats.__dict__,
            "feature_train_mean": train_ds.feature_norm.mean,
            "feature_train_std": train_ds.feature_norm.std,
            "formal_feature_columns": FORMAL_FEATURE_COLUMNS,
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    np.savez_compressed(run_dir / "ptb_test_probs.npz", probs=ptb_probs)
    np.savez_compressed(run_dir / "but_clean_test_probs.npz", probs=clean_probs)
    pred_path = run_dir / "but_clean_val_test_predictions.csv"
    pd.concat(
        [
            prediction_frame(clean_val, clean_val_probs, "but_val"),
            prediction_frame(clean_test, clean_probs, "but_test"),
        ],
        ignore_index=True,
    ).to_csv(pred_path, index=False)
    if full_probs is not None:
        np.savez_compressed(run_dir / "legacy_full_but_probs.npz", probs=full_probs)
    short_name = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    plot_aug_examples(train_ds, REPORT_OUT_DIR / f"{short_name}_augmented_bad_examples.png")

    return {
        "candidate": name,
        "checkpoint": str(ckpt_path),
        "predictions": str(pred_path),
        "best_epoch": best_epoch,
        "averaged_epochs": averaged_epochs,
        "augmentation_summary": train_ds.augmentation_summary,
    }, rows


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# PTB Bad-Alignment Cross-Dataset Check",
        "",
        "Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.",
        "",
        "## Why This Exists",
        "",
        "The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.",
        "",
        "## Metrics",
        "",
        "| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} | {int(row.get('nonbad_to_bad', 0))} |"
        )
    lines.extend(["", "## Candidate Summaries", ""])
    for item in payload["summaries"]:
        aug = item["augmentation_summary"]
        lines.append(
            f"- `{item['candidate']}`: best_epoch={item['best_epoch']}, "
            f"averaged_epochs={item.get('averaged_epochs', [])}, "
            f"bad_aug_added={aug['bad_augmented_added']}, nonbad_guard_added={aug['nonbad_guard_added']}, "
            f"matched_bank_added={aug.get('matched_bank_added', 0)}, "
            f"checkpoint=`{item['checkpoint']}`, "
            f"predictions=`{item.get('predictions', '')}`"
        )
    lines.extend(
        [
            "",
            "## Decision Contract",
            "",
            "- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.",
            "- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.",
            "- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.",
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
    parser.add_argument("--policy", type=str, default="margin_ge_5s_keep_outlier")
    parser.add_argument("--candidate", type=str, default="dualview_convtx_hier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--aug-mode", type=str, default="record111_bad_align")
    parser.add_argument("--bad-aug-prob", type=float, default=0.60)
    parser.add_argument("--bad-aug-strengths", type=str, default="0.75,1.15")
    parser.add_argument("--nonbad-guard-prob", type=float, default=0.10)
    parser.add_argument("--nonbad-guard-strength", type=float, default=0.55)
    parser.add_argument("--stress-aux-weight", type=float, default=0.0)
    parser.add_argument("--stress-nonbad-penalty", type=float, default=0.0)
    parser.add_argument("--artifact-aux-weight", type=float, default=0.0)
    parser.add_argument("--artifact-nonbad-bad-penalty", type=float, default=0.0)
    parser.add_argument("--pretrain-clean-epochs", type=int, default=0)
    parser.add_argument("--finetune-lr-scale", type=float, default=1.0)
    parser.add_argument("--stats-on-augmented-train", action="store_true")
    parser.add_argument("--matched-bank", type=str, default="")
    parser.add_argument("--matched-bank-frac", type=float, default=0.0)
    parser.add_argument("--matched-bank-select", type=str, default="random", choices=["random", "order", "stratified_order"])
    parser.add_argument("--matched-bank-holdout-frac", type=float, default=0.0)
    parser.add_argument("--include-full-diagnostic", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--selection-mode", type=str, default="ptb_val", choices=["ptb_val", "clean_val_joint", "ptb_stress_joint", "ptb_stress_balance", "ptb_stress_badguard"])
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="")
    parser.add_argument("--gm-weight", type=float, default=-1.0)
    parser.add_argument("--bad-weight", type=float, default=-1.0)
    parser.add_argument("--ptb-feature-source", type=str, default="manifest", choices=["manifest", "waveform_primitives"])
    parser.add_argument("--input-view", type=str, default="full", choices=["full", "robust_no_physical", "robust_detail_only"])
    parser.add_argument("--scale-match-policy", type=str, default="")
    parser.add_argument("--average-epochs", type=str, default="")
    args = parser.parse_args()

    if args.candidate not in DUAL.CANDIDATES:
        raise ValueError(f"unknown candidate {args.candidate}; expected {sorted(DUAL.CANDIDATES)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for strength_text in [s.strip() for s in args.bad_aug_strengths.split(",") if s.strip()]:
        strength = float(strength_text)
        cfg = dict(DUAL.CANDIDATES[args.candidate])
        # Augmented waveforms do not have perfectly recomputed tabular targets,
        # so keep the SQI auxiliary heads as a light regularizer rather than the
        # dominant loss in this stress-alignment check.
        cfg["aux_weight"] = min(float(cfg.get("aux_weight", 0.0)), 0.12)
        cfg["core_aux_weight"] = min(float(cfg.get("core_aux_weight", 0.0)), 0.10)
        cfg["bad_weight"] = max(float(cfg.get("bad_weight", 1.0)), 2.05)
        cfg["false_bad_penalty"] = max(float(cfg.get("false_bad_penalty", 0.0)), 0.55)
        cfg["lr"] = float(cfg.get("lr", 3e-4)) * float(args.lr_scale)
        if str(args.class_weight).strip():
            weights = [float(x.strip()) for x in str(args.class_weight).split(",") if x.strip()]
            if len(weights) != 3:
                raise ValueError("--class-weight must have exactly three comma-separated values")
            cfg["class_weight"] = weights
        if float(args.gm_weight) > 0:
            cfg["gm_weight"] = float(args.gm_weight)
        if float(args.bad_weight) > 0:
            cfg["bad_weight"] = float(args.bad_weight)
        cfg["seed"] = int(cfg["seed"]) + int(round(strength * 1000)) + int(args.seed_offset)
        stress_suffix = ""
        if float(args.stress_aux_weight) > 0:
            stress_suffix = f"_stressaux{str(args.stress_aux_weight).replace('.', 'p')}"
        if float(args.artifact_aux_weight) > 0:
            stress_suffix += f"_artaux{str(args.artifact_aux_weight).replace('.', 'p')}"
        if float(args.artifact_nonbad_bad_penalty) > 0:
            stress_suffix += f"_artnbbad{str(args.artifact_nonbad_bad_penalty).replace('.', 'p')}"
        curriculum_suffix = ""
        if int(args.pretrain_clean_epochs) > 0:
            curriculum_suffix = f"_pre{int(args.pretrain_clean_epochs)}_lr{str(args.finetune_lr_scale).replace('.', 'p')}"
        aug_suffix = ""
        if str(args.aug_mode) != "record111_bad_align":
            aug_suffix = f"_{str(args.aug_mode)}"
        if str(args.aug_mode).startswith("distribution_bank"):
            bank_hash = hashlib.md5(str(args.matched_bank).encode("utf-8")).hexdigest()[:6] if str(args.matched_bank) else "nobank"
            aug_suffix = f"_distbank{bank_hash}_b{str(args.matched_bank_frac).replace('.', 'p')}"
        if str(args.matched_bank_select) != "random":
            aug_suffix += f"_{str(args.matched_bank_select)}"
        if float(args.matched_bank_holdout_frac) > 0:
            aug_suffix += f"_hold{str(args.matched_bank_holdout_frac).replace('.', 'p')}"
        seed_suffix = f"_seed{int(args.seed_offset)}" if int(args.seed_offset) != 0 else ""
        select_suffix = "_selclean" if str(args.selection_mode) == "clean_val_joint" else ("_selbadguard" if str(args.selection_mode) == "ptb_stress_badguard" else ("_selstressbal" if str(args.selection_mode) == "ptb_stress_balance" else ("_selstress" if str(args.selection_mode) == "ptb_stress_joint" else "")))
        hp_suffix = ""
        if float(args.lr_scale) != 1.0:
            hp_suffix += f"_lrscale{str(args.lr_scale).replace('.', 'p')}"
        if str(args.class_weight).strip():
            hp_suffix += "_cw" + str(args.class_weight).replace(",", "x").replace(".", "p")
        if float(args.gm_weight) > 0:
            hp_suffix += f"_gm{str(args.gm_weight).replace('.', 'p')}"
        if float(args.bad_weight) > 0:
            hp_suffix += f"_bad{str(args.bad_weight).replace('.', 'p')}"
        if str(args.average_epochs).strip():
            hp_suffix += "_avgE" + str(args.average_epochs).replace(",", "x")
        feat_suffix = "_featwf" if str(args.ptb_feature_source) == "waveform_primitives" else ""
        view_suffix = "" if str(args.input_view) == "full" else f"_{str(args.input_view)}"
        scale_suffix = "" if not str(args.scale_match_policy) else "_scalematch"
        run_name = (
            f"ptb_badalign_{args.policy}_{args.candidate}_"
            f"p{int(args.bad_aug_prob*100):02d}_s{str(strength).replace('.', 'p')}{stress_suffix}{curriculum_suffix}{aug_suffix}{select_suffix}{hp_suffix}{seed_suffix}{feat_suffix}{view_suffix}{scale_suffix}"
        )
        args.bad_aug_strength = strength
        summary, metric_rows = train_badalign_candidate(run_name, args.policy, cfg, args)
        summaries.append(summary)
        rows.extend(metric_rows)

    metrics = pd.DataFrame(rows)
    stress_stem = ""
    if float(args.stress_aux_weight) > 0:
        stress_stem = f"_stressaux{str(args.stress_aux_weight).replace('.', 'p')}"
    if float(args.artifact_aux_weight) > 0:
        stress_stem += f"_artaux{str(args.artifact_aux_weight).replace('.', 'p')}"
    if float(args.artifact_nonbad_bad_penalty) > 0:
        stress_stem += f"_artnbbad{str(args.artifact_nonbad_bad_penalty).replace('.', 'p')}"
    curriculum_stem = ""
    if int(args.pretrain_clean_epochs) > 0:
        curriculum_stem = f"_pre{int(args.pretrain_clean_epochs)}_lr{str(args.finetune_lr_scale).replace('.', 'p')}"
    aug_stem = ""
    if str(args.aug_mode) != "record111_bad_align":
        aug_stem = f"_{str(args.aug_mode)}"
    if str(args.aug_mode).startswith("distribution_bank"):
        bank_hash = hashlib.md5(str(args.matched_bank).encode("utf-8")).hexdigest()[:6] if str(args.matched_bank) else "nobank"
        aug_stem = f"_distbank{bank_hash}_b{str(args.matched_bank_frac).replace('.', 'p')}"
        if str(args.matched_bank_select) != "random":
            aug_stem += f"_{str(args.matched_bank_select)}"
        if float(args.matched_bank_holdout_frac) > 0:
            aug_stem += f"_hold{str(args.matched_bank_holdout_frac).replace('.', 'p')}"
    seed_stem = f"_seed{int(args.seed_offset)}" if int(args.seed_offset) != 0 else ""
    select_stem = "_selclean" if str(args.selection_mode) == "clean_val_joint" else ("_selbadguard" if str(args.selection_mode) == "ptb_stress_badguard" else ("_selstressbal" if str(args.selection_mode) == "ptb_stress_balance" else ("_selstress" if str(args.selection_mode) == "ptb_stress_joint" else "")))
    hp_stem = ""
    if float(args.lr_scale) != 1.0:
        hp_stem += f"_lrscale{str(args.lr_scale).replace('.', 'p')}"
    if str(args.class_weight).strip():
        hp_stem += "_cw" + str(args.class_weight).replace(",", "x").replace(".", "p")
    if float(args.gm_weight) > 0:
        hp_stem += f"_gm{str(args.gm_weight).replace('.', 'p')}"
    if float(args.bad_weight) > 0:
        hp_stem += f"_bad{str(args.bad_weight).replace('.', 'p')}"
    if str(args.average_epochs).strip():
        hp_stem += "_avgE" + str(args.average_epochs).replace(",", "x")
    feat_stem = "_featwf" if str(args.ptb_feature_source) == "waveform_primitives" else ""
    view_stem = "" if str(args.input_view) == "full" else f"_{str(args.input_view)}"
    scale_stem = "" if not str(args.scale_match_policy) else "_scalematch"
    if str(args.aug_mode).startswith("distribution_bank"):
        cand_short = str(args.candidate).replace("dualview_", "").replace("_hier", "")
        bank_hash = hashlib.md5(str(args.matched_bank).encode("utf-8")).hexdigest()[:6] if str(args.matched_bank) else "nobank"
        bank_select = ""
        if str(args.matched_bank_select) != "random":
            bank_select = f"_{str(args.matched_bank_select)}"
        bank_hold = ""
        if float(args.matched_bank_holdout_frac) > 0:
            bank_hold = f"_hold{str(args.matched_bank_holdout_frac).replace('.', 'p')}"
        stem = f"pba_{cand_short}_{bank_hash}_b{str(args.matched_bank_frac).replace('.', 'p')}{bank_select}{bank_hold}{stress_stem}{curriculum_stem}{select_stem}{hp_stem}{seed_stem}{feat_stem}{view_stem}{scale_stem}"
    else:
        stem = f"ptb_bad_alignment_{args.policy}_{args.candidate}_p{int(args.bad_aug_prob*100):02d}{stress_stem}{curriculum_stem}{aug_stem}{select_stem}{hp_stem}{seed_stem}{feat_stem}{view_stem}{scale_stem}"
    if len(stem) > 74:
        stem = stem[:52] + "_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    metrics_path = OUT_DIR / f"{stem}_metrics.csv"
    summary_path = OUT_DIR / f"{stem}_summary.json"
    report_path = OUT_DIR / f"{stem}_report.md"
    report_copy = REPORT_OUT_DIR / report_path.name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": args.policy,
        "candidate": args.candidate,
        "epochs": int(args.epochs),
        "ptb_feature_source": str(args.ptb_feature_source),
        "input_view": str(args.input_view),
        "scale_match_policy": str(args.scale_match_policy),
        "average_epochs": str(args.average_epochs),
        "matched_bank_holdout_frac": float(args.matched_bank_holdout_frac),
        "selection_mode": str(args.selection_mode),
        "stress_aux_weight": float(args.stress_aux_weight),
        "stress_nonbad_penalty": float(args.stress_nonbad_penalty),
        "artifact_aux_weight": float(args.artifact_aux_weight),
        "artifact_nonbad_bad_penalty": float(args.artifact_nonbad_bad_penalty),
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
