from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

try:
    from src.transformer_pipeline.noise.make_rr_noise_level import safe_corr
    from src.transformer_pipeline.noise.synthesize_local_counterfactual import (
        PLACEMENTS,
        critical_region_masks,
        placement_envelope,
    )
    from src.transformer_pipeline.noise.synthesize_snr_dataset import (
        FS,
        N,
        WIN_SEC,
        add_noise_at_snr,
        load_noise_125,
        make_noise_segment,
        parse_noise_kinds,
        split_noise_ranges,
    )
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.transformer_pipeline.noise.make_rr_noise_level import safe_corr
    from src.transformer_pipeline.noise.synthesize_local_counterfactual import (
        PLACEMENTS,
        critical_region_masks,
        placement_envelope,
    )
    from src.transformer_pipeline.noise.synthesize_snr_dataset import (
        FS,
        N,
        WIN_SEC,
        add_noise_at_snr,
        load_noise_125,
        make_noise_segment,
        parse_noise_kinds,
        split_noise_ranges,
    )
    from src.utils.paths import project_root


logger = logging.getLogger(__name__)

CLASS_ORDER = ("good", "medium", "bad")
CLASS_TARGET_DAMAGE = {"good": 0.08, "medium": 0.32, "bad": 0.70}
CLASS_TARGET_SEVERITY = {"good": 0.22, "medium": 0.68, "bad": 1.12}
LABEL_VERSIONS = (
    "e35_morph_damage",
    "e36_critical_damage",
    "e37_diagnostic_damage",
    "e38_core_diagnostic_damage",
)
SNR_OFFSETS = (-0.25, 0.0, 0.25)


@dataclass(frozen=True)
class Candidate:
    y_class: str | None
    label_subtype: str
    noisy: np.ndarray
    envelope: np.ndarray
    metrics: dict[str, float | str]
    placement: str
    target_snr_db: float
    measured_snr_db: float


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)

    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer_e35_morph_damage_triplet")
    source_artifact_dir = _path(params.get("source_artifact_dir"), root / "outputs/transformer")
    nstdb = _path(params.get("nstdb_root"), root / "data" / "physionet" / "nstdb")
    seed = int(params.get("seed", 0))
    force = bool(params.get("force", False))
    max_train_clean = _none_or_int(params.get("max_train_clean"), 4000)
    max_val_clean = _none_or_int(params.get("max_val_clean"), 800)
    max_test_clean = _none_or_int(params.get("max_test_clean"), 800)
    noise_kinds = parse_noise_kinds(params.get("noise_kinds", "em,ma,bw,mix"))
    snr_min = float(params.get("snr_min", 6.0))
    snr_max = float(params.get("snr_max", 12.0))
    group_retries = int(params.get("group_retries", 8))
    label_version = str(params.get("label_version", "e35_morph_damage"))
    if label_version not in LABEL_VERSIONS:
        raise ValueError(f"label_version must be one of: {', '.join(LABEL_VERSIONS)}")

    x_npz = source_artifact_dir / "segments" / "ptbxl_leadI_x_10s_125hz.npz"
    split_csv = source_artifact_dir / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
    out_dir = artifact_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_clean = out_dir / "synth_10s_125hz_clean.npz"
    out_noisy = out_dir / "synth_10s_125hz_noisy.npz"
    out_mask = out_dir / "synth_10s_125hz_local_mask.npz"
    out_lbl = out_dir / "synth_10s_125hz_labels.csv"
    out_gray = out_dir / "morph_damage_gray_zone_candidates.csv"
    out_summary = out_dir / "morph_damage_triplet_summary.json"
    outputs = [out_clean, out_noisy, out_mask, out_lbl, out_gray, out_summary]
    if all(path.exists() for path in outputs) and not force:
        logger.info("%s morphology-damage triplet outputs exist -> skip (set --force to rerun)", label_version)
        return {"step": "morph_damage_triplet", "skipped": True, "outputs": [str(p) for p in outputs]}

    X_all = np.load(x_npz)["X"].astype(np.float32)
    df_split = pd.read_csv(split_csv)
    need_cols = {"seg_id", "ecg_id", "npz_index", "split"}
    if not need_cols.issubset(df_split.columns):
        raise ValueError(f"split csv missing columns: {sorted(need_cols - set(df_split.columns))}")

    rng = np.random.default_rng(seed)
    tracks = {"em": load_noise_125(nstdb / "em"), "ma": load_noise_125(nstdb / "ma")}
    bw_path = nstdb / "bw"
    tracks["bw"] = load_noise_125(bw_path) if bw_path.with_suffix(".hea").exists() else tracks["ma"]
    ranges = {name: split_noise_ranges(len(track)) for name, track in tracks.items()}

    x_clean_rows: list[np.ndarray] = []
    x_noisy_rows: list[np.ndarray] = []
    local_masks: list[np.ndarray] = []
    critical_masks: list[np.ndarray] = []
    qrs_masks: list[np.ndarray] = []
    tst_masks: list[np.ndarray] = []
    label_rows: list[dict[str, object]] = []
    gray_rows: list[dict[str, object]] = []

    split_limits = {"train": max_train_clean, "val": max_val_clean, "test": max_test_clean}
    group_id = 0
    skipped_groups = 0

    for split_name in ("train", "val", "test"):
        split_rows = df_split[df_split["split"].astype(str) == split_name].reset_index(drop=True)
        split_rows = sample_rows(split_rows, split_limits[split_name], rng)
        accepted_in_split = 0
        for row_i, row in enumerate(split_rows.itertuples(index=False), start=1):
            if verbose and row_i % 250 == 0:
                logger.info("%s synth %s progress: %d/%d", label_version, split_name, row_i, len(split_rows))

            clean = X_all[int(row.npz_index)].astype(np.float32, copy=False)
            qrs_mask, tst_mask, critical_mask, peaks = critical_region_masks(clean)
            noise_kind = noise_kinds[group_id % len(noise_kinds)]
            triplet, gray = build_triplet_candidates(
                clean=clean,
                qrs_mask=qrs_mask,
                tst_mask=tst_mask,
                critical_mask=critical_mask,
                peaks=peaks,
                noise_kind=noise_kind,
                split_name=split_name,
                tracks=tracks,
                ranges=ranges,
                rng=rng,
                snr_min=snr_min,
                snr_max=snr_max,
                group_retries=group_retries,
                label_version=label_version,
            )
            for item in gray:
                gray_rows.append(
                    {
                        "split": split_name,
                        "counterfactual_group": int(group_id),
                        "seg_id": int(row.seg_id),
                        "ecg_id": int(row.ecg_id),
                        "noise_kind": noise_kind,
                        **item,
                    }
                )
            if triplet is None:
                skipped_groups += 1
                group_id += 1
                continue

            noise_window_id = str(triplet["noise_window_id"])
            matched_snr_db = float(triplet["matched_snr_db"])
            for y_class in CLASS_ORDER:
                cand = triplet[y_class]
                assert isinstance(cand, Candidate)
                proxies = sqi_proxies(cand.noisy)
                x_clean_rows.append(clean.astype(np.float32, copy=False))
                x_noisy_rows.append(cand.noisy.astype(np.float32, copy=False))
                local_masks.append(cand.envelope.astype(np.float32, copy=False))
                critical_masks.append(critical_mask.astype(np.float32, copy=False))
                qrs_masks.append(qrs_mask.astype(np.float32, copy=False))
                tst_masks.append(tst_mask.astype(np.float32, copy=False))
                label_rows.append(
                    {
                        "idx": len(label_rows),
                        "seg_id": int(row.seg_id),
                        "ecg_id": int(row.ecg_id),
                        "split": split_name,
                        "y_class": y_class,
                        "label_subtype": cand.label_subtype,
                        "snr_db": cand.target_snr_db,
                        "snr_profile": "matched_6_12dB",
                        "matched_snr_db": matched_snr_db,
                        "measured_snr_db": cand.measured_snr_db,
                        "noise_kind": noise_kind,
                        "noise_window_id": noise_window_id,
                        "placement": cand.placement,
                        "counterfactual_group": int(group_id),
                        "qrs_nprd": cand.metrics["qrs_nprd"],
                        "tst_nprd": cand.metrics["tst_nprd"],
                        "beat_corr": cand.metrics["beat_corr"],
                        "max_beat_nprd": cand.metrics["max_beat_nprd"],
                        "damage_score": (
                            cand.metrics["core_diagnostic_score"]
                            if label_version == "e38_core_diagnostic_damage"
                            else (
                                cand.metrics["diagnostic_damage_score"]
                                if label_version == "e37_diagnostic_damage"
                                else (
                                    cand.metrics["critical_damage_score"]
                                    if label_version == "e36_critical_damage"
                                    else cand.metrics["legacy_damage_score"]
                                )
                            )
                        ),
                        "critical_damage_score": cand.metrics["critical_damage_score"],
                        "diagnostic_damage_score": cand.metrics["diagnostic_damage_score"],
                        "core_diagnostic_score": cand.metrics["core_diagnostic_score"],
                        "beat_axis": cand.metrics["beat_axis"],
                        "dominant_axis": cand.metrics["dominant_axis"],
                        "global_noise_score": cand.metrics["global_noise_score"],
                        "global_nprd": cand.metrics["global_nprd"],
                        "legacy_damage_score": cand.metrics["legacy_damage_score"],
                        "proxy_pSQI": proxies["pSQI"],
                        "proxy_basSQI": proxies["basSQI"],
                        "proxy_kSQI": proxies["kSQI"],
                        "proxy_sSQI": proxies["sSQI"],
                        "proxy_rms": proxies["rms"],
                        "fs": FS,
                        "window_sec": WIN_SEC,
                        "source_npz_index": int(row.npz_index),
                        "sample_source": label_version,
                    }
                )
            accepted_in_split += 1
            group_id += 1
        logger.info("%s accepted %s groups: %d/%d", label_version, split_name, accepted_in_split, len(split_rows))

    if not label_rows:
        raise RuntimeError(f"{label_version} generated no triplets; relax margins or inspect candidate damage metrics")

    X_clean = np.stack(x_clean_rows, axis=0).astype(np.float32)
    X_noisy = np.stack(x_noisy_rows, axis=0).astype(np.float32)
    M = np.stack(local_masks, axis=0).astype(np.float32)
    C = np.stack(critical_masks, axis=0).astype(np.float32)
    Q = np.stack(qrs_masks, axis=0).astype(np.float32)
    T = np.stack(tst_masks, axis=0).astype(np.float32)
    labels = pd.DataFrame(label_rows)
    labels["idx"] = np.arange(len(labels), dtype=int)

    np.savez(out_clean, X_clean=X_clean)
    np.savez(out_noisy, X_noisy=X_noisy)
    np.savez(out_mask, M=M, critical_mask=C, qrs_mask=Q, tst_mask=T)
    labels.to_csv(out_lbl, index=False)
    pd.DataFrame(gray_rows).to_csv(out_gray, index=False)

    summary = build_summary(
        labels,
        X_clean,
        X_noisy,
        skipped_groups=skipped_groups,
        gray_rows=len(gray_rows),
        label_version=label_version,
    )
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("%s rows=%d groups=%d skipped_groups=%d", label_version, len(labels), labels["counterfactual_group"].nunique(), skipped_groups)
    logger.info("measured-SNR oracle acc: %.6f", summary["measured_snr_oracle_acc"])
    logger.info("max class SNR mean gap: %.6f dB", summary["audit"]["max_class_mean_gap"]["measured_snr_db"])
    for path in outputs:
        logger.info("Saved: %s", display(path, root))

    return {
        "step": "morph_damage_triplet",
        "skipped": False,
        "outputs": [str(p) for p in outputs],
        "rows": int(len(labels)),
        "groups": int(labels["counterfactual_group"].nunique()),
        "measured_snr_oracle_acc": summary["measured_snr_oracle_acc"],
    }


def build_triplet_candidates(
    *,
    clean: np.ndarray,
    qrs_mask: np.ndarray,
    tst_mask: np.ndarray,
    critical_mask: np.ndarray,
    peaks: np.ndarray,
    noise_kind: str,
    split_name: str,
    tracks: dict[str, np.ndarray],
    ranges: dict[str, dict[str, tuple[int, int]]],
    rng: np.random.Generator,
    snr_min: float,
    snr_max: float,
    group_retries: int,
    label_version: str,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    gray_rows: list[dict[str, object]] = []
    for _ in range(group_retries):
        matched_snr_db = float(rng.uniform(snr_min, snr_max))
        raw_noise, noise_window_id = sample_noise_with_id(noise_kind, split_name, tracks, ranges, rng)
        candidates: list[Candidate] = []
        for snr_offset in SNR_OFFSETS:
            target_snr = float(np.clip(matched_snr_db + snr_offset, snr_min, snr_max))
            for placement in PLACEMENTS:
                envelope = placement_envelope(
                    placement=placement,
                    qrs_mask=qrs_mask,
                    tst_mask=tst_mask,
                    critical_mask=critical_mask,
                    rng=rng,
                )
                placed_noise = raw_noise * envelope
                if float(np.mean(placed_noise * placed_noise)) < 1e-12:
                    placed_noise = raw_noise.copy()
                    envelope = np.ones(N, dtype=np.float32)
                noisy = add_noise_at_snr(clean, placed_noise, target_snr)
                metrics = damage_metrics(clean, noisy, qrs_mask, tst_mask, peaks)
                measured_snr = measured_snr_db(clean, noisy)
                y_class, label_subtype = assign_margin_label(metrics, placement=placement, label_version=label_version)
                cand = Candidate(
                    y_class=y_class,
                    label_subtype=label_subtype,
                    noisy=noisy,
                    envelope=envelope,
                    metrics=metrics,
                    placement=placement,
                    target_snr_db=target_snr,
                    measured_snr_db=measured_snr,
                )
                candidates.append(cand)
                if y_class is None:
                    gray_rows.append(
                        {
                            "placement": placement,
                            "label_subtype": label_subtype,
                            "target_snr_db": target_snr,
                            "measured_snr_db": measured_snr,
                            **metrics,
                        }
                    )
        picked = pick_triplet(candidates, matched_snr_db, label_version=label_version)
        if picked is not None:
            picked["matched_snr_db"] = matched_snr_db
            picked["noise_window_id"] = noise_window_id
            return picked, gray_rows
    return None, gray_rows


def pick_triplet(candidates: list[Candidate], matched_snr_db: float, *, label_version: str) -> dict[str, object] | None:
    picked: dict[str, object] = {}
    if label_version == "e38_core_diagnostic_damage":
        damage_key = "core_diagnostic_score"
        class_targets = CLASS_TARGET_SEVERITY
    elif label_version == "e37_diagnostic_damage":
        damage_key = "diagnostic_damage_score"
        class_targets = CLASS_TARGET_SEVERITY
    elif label_version == "e36_critical_damage":
        damage_key = "critical_damage_score"
        class_targets = CLASS_TARGET_DAMAGE
    else:
        damage_key = "legacy_damage_score"
        class_targets = CLASS_TARGET_DAMAGE
    for y_class in CLASS_ORDER:
        pool = [c for c in candidates if c.y_class == y_class]
        if not pool:
            return None
        target_damage = class_targets[y_class]
        if label_version == "e38_core_diagnostic_damage":
            pool_snr = [c for c in pool if abs(c.measured_snr_db - matched_snr_db) <= 0.20]
            if not pool_snr:
                pool_snr = [c for c in pool if abs(c.measured_snr_db - matched_snr_db) <= 0.35]
            if not pool_snr:
                pool_snr = pool
            picked[y_class] = min(
                pool_snr,
                key=lambda c: (
                    abs(c.measured_snr_db - matched_snr_db),
                    abs(float(c.metrics[damage_key]) - target_damage),
                ),
            )
        elif label_version == "e37_diagnostic_damage":
            pool_snr = [c for c in pool if abs(c.measured_snr_db - matched_snr_db) <= 0.50]
            if not pool_snr:
                pool_snr = pool
            picked[y_class] = min(
                pool_snr,
                key=lambda c: (
                    abs(float(c.metrics[damage_key]) - target_damage),
                    abs(c.measured_snr_db - matched_snr_db),
                ),
            )
        else:
            picked[y_class] = min(
                pool,
                key=lambda c: (
                    abs(c.measured_snr_db - matched_snr_db),
                    abs(float(c.metrics[damage_key]) - target_damage),
                ),
            )
    snrs = [float(picked[name].measured_snr_db) for name in CLASS_ORDER]  # type: ignore[union-attr]
    max_snr_gap = 0.25 if label_version == "e38_core_diagnostic_damage" else 0.75
    if max(snrs) - min(snrs) > max_snr_gap:
        return None
    return picked


def damage_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    qrs_mask: np.ndarray,
    tst_mask: np.ndarray,
    peaks: np.ndarray,
) -> dict[str, float | str]:
    qrs = region_nprd(clean, noisy, qrs_mask)
    tst = region_nprd(clean, noisy, tst_mask)
    beat_corr, max_beat = beat_template_corr_and_max_nprd(clean, noisy, peaks)
    legacy_damage = 0.45 * qrs + 0.25 * tst + 0.20 * (1.0 - beat_corr) + 0.10 * max_beat
    critical_damage = 0.45 * qrs + 0.35 * tst + 0.15 * (1.0 - beat_corr) + 0.05 * max_beat
    qrs_axis = qrs / 0.35
    tst_axis = tst / 0.45
    crit_axis = critical_damage / 0.55
    corr_axis = max(0.0, (0.94 - beat_corr) / (0.94 - 0.70))
    beat_axis = max_beat / 0.60
    diagnostic_components = {
        "qrs": qrs_axis,
        "tst": 0.85 * tst_axis,
        "crit": crit_axis,
        "corr": corr_axis,
        "beat": 0.75 * beat_axis,
    }
    diagnostic_damage = max(diagnostic_components.values())
    core_diagnostic_score = max(qrs_axis, tst_axis, crit_axis, corr_axis)
    dominant_axis = max(diagnostic_components, key=diagnostic_components.get)
    global_nprd = region_nprd(clean, noisy, np.ones_like(clean, dtype=np.float32))
    return {
        "qrs_nprd": float(qrs),
        "tst_nprd": float(tst),
        "beat_corr": float(beat_corr),
        "max_beat_nprd": float(max_beat),
        "damage_score": float(legacy_damage),
        "legacy_damage_score": float(legacy_damage),
        "critical_damage_score": float(critical_damage),
        "diagnostic_damage_score": float(diagnostic_damage),
        "core_diagnostic_score": float(core_diagnostic_score),
        "beat_axis": float(beat_axis),
        "dominant_axis": str(dominant_axis),
        "global_noise_score": float(global_nprd),
        "global_nprd": float(global_nprd),
    }


def assign_margin_label(metrics: dict[str, float | str], *, placement: str, label_version: str) -> tuple[str | None, str]:
    if label_version == "e36_critical_damage":
        return assign_e36_label(metrics, placement=placement)
    if label_version == "e37_diagnostic_damage":
        return assign_e37_label(metrics)
    if label_version == "e38_core_diagnostic_damage":
        return assign_e38_label(metrics)
    y_class = assign_e35_label(metrics)
    return y_class, f"{y_class}_legacy" if y_class is not None else "gray_zone"


def assign_e35_label(metrics: dict[str, float | str]) -> str | None:
    damage = float(metrics["damage_score"])
    qrs = float(metrics["qrs_nprd"])
    beat_corr = float(metrics["beat_corr"])
    if damage <= 0.12 and qrs <= 0.10 and beat_corr >= 0.95:
        return "good"
    if 0.25 <= damage <= 0.40 and qrs < 0.35 and beat_corr >= 0.80:
        return "medium"
    if damage >= 0.55 or qrs >= 0.45 or beat_corr <= 0.70:
        return "bad"
    return None


def assign_e36_label(metrics: dict[str, float | str], *, placement: str) -> tuple[str | None, str]:
    critical = float(metrics["critical_damage_score"])
    global_noise = float(metrics["global_noise_score"])
    qrs = float(metrics["qrs_nprd"])
    tst = float(metrics["tst_nprd"])
    beat_corr = float(metrics["beat_corr"])

    if critical <= 0.12 and qrs <= 0.10 and tst <= 0.12 and beat_corr >= 0.94:
        if placement == "noncritical" and global_noise >= 0.35:
            return "good", "good_noncritical_hard"
        return "good", "good_critical_clean"

    if qrs >= 0.35 or beat_corr <= 0.70 or critical >= 0.55:
        if qrs >= 0.35:
            return "bad", "bad_qrs_damage"
        if beat_corr <= 0.70:
            return "bad", "bad_low_beat_corr"
        return "bad", "bad_critical_severe"

    if 0.12 < critical < 0.22:
        return None, "gray_low_critical_boundary"
    if 0.42 < critical < 0.55:
        return None, "gray_high_critical_boundary"

    medium_critical = 0.22 <= critical <= 0.42
    medium_tst = placement != "noncritical" and 0.20 <= tst <= 0.45
    medium_uniform_or_tst = placement in {"uniform", "tst_overlap"} and medium_critical and tst >= 0.14
    if qrs <= 0.25 and beat_corr >= 0.80 and (medium_critical or medium_tst or medium_uniform_or_tst):
        if 0.20 < qrs <= 0.25:
            return "medium", "medium_qrs_mild"
        if placement == "uniform":
            return "medium", "medium_uniform_moderate"
        if tst >= 0.20 or placement == "tst_overlap":
            return "medium", "medium_tst_damage"
        return "medium", "medium_critical_damage"

    return None, "gray_zone"


def assign_e37_label(metrics: dict[str, float | str]) -> tuple[str | None, str]:
    diagnostic = float(metrics["diagnostic_damage_score"])
    qrs = float(metrics["qrs_nprd"])
    tst = float(metrics["tst_nprd"])
    beat_corr = float(metrics["beat_corr"])

    if diagnostic <= 0.32 and qrs <= 0.10 and tst <= 0.12 and beat_corr >= 0.94:
        return "good", "good_diagnostic_low"
    if 0.55 <= diagnostic <= 0.82 and qrs < 0.35 and beat_corr >= 0.75:
        return "medium", "medium_diagnostic_moderate"
    if diagnostic >= 1.00:
        return "bad", "bad_diagnostic_severe"
    return None, "gray_diagnostic_boundary"


def assign_e38_label(metrics: dict[str, float | str]) -> tuple[str | None, str]:
    core = float(metrics["core_diagnostic_score"])
    qrs = float(metrics["qrs_nprd"])
    tst = float(metrics["tst_nprd"])
    beat_corr = float(metrics["beat_corr"])
    dominant_axis = str(metrics["dominant_axis"])

    if core <= 0.32 and qrs <= 0.10 and tst <= 0.12 and beat_corr >= 0.94:
        return "good", "good_core_low"
    if 0.45 <= core <= 0.85 and qrs < 0.35 and beat_corr >= 0.75 and dominant_axis != "beat":
        return "medium", "medium_core_moderate"
    if core >= 1.00:
        return "bad", "bad_core_severe"
    return None, "gray_core_boundary"


def region_nprd(clean: np.ndarray, noisy: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if int(m.sum()) <= 0:
        return 0.0
    err = noisy[m].astype(np.float64) - clean[m].astype(np.float64)
    ref = clean[m].astype(np.float64)
    return float(np.sqrt(np.sum(err * err) / (np.sum(ref * ref) + 1e-12)))


def beat_template_corr_and_max_nprd(clean: np.ndarray, noisy: np.ndarray, peaks: np.ndarray) -> tuple[float, float]:
    left = int(0.18 * FS)
    right = int(0.42 * FS)
    clean_beats: list[np.ndarray] = []
    noisy_beats: list[np.ndarray] = []
    beat_nprd: list[float] = []
    for r in peaks:
        a = int(r) - left
        b = int(r) + right
        if a < 0 or b > len(clean) or b <= a:
            continue
        c = clean[a:b].astype(np.float64)
        n = noisy[a:b].astype(np.float64)
        clean_beats.append(c)
        noisy_beats.append(n)
        beat_nprd.append(float(np.sqrt(np.sum((n - c) ** 2) / (np.sum(c * c) + 1e-12))))
    if len(clean_beats) < 2:
        return safe_corr(clean, noisy), region_nprd(clean, noisy, np.ones_like(clean, dtype=np.float32))
    clean_template = np.mean(np.stack(clean_beats, axis=0), axis=0)
    noisy_template = np.mean(np.stack(noisy_beats, axis=0), axis=0)
    return safe_corr(clean_template, noisy_template), float(max(beat_nprd))


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    sig = float(np.mean(clean.astype(np.float64) ** 2)) + 1e-12
    err = float(np.mean((noisy.astype(np.float64) - clean.astype(np.float64)) ** 2)) + 1e-12
    return float(10.0 * np.log10(sig / err))


def sqi_proxies(x: np.ndarray) -> dict[str, float]:
    y = x.astype(np.float64)
    f, pxx = welch(y, fs=FS, nperseg=min(256, len(y)))
    p_total = band_power(f, pxx, 0.5, 40.0) + 1e-12
    p_qrs = band_power(f, pxx, 5.0, 15.0)
    p_wide = band_power(f, pxx, 5.0, 40.0) + 1e-12
    p_base = band_power(f, pxx, 0.5, 1.0)
    return {
        "pSQI": float(p_qrs / p_wide),
        "basSQI": float(p_base / p_total),
        "kSQI": float(kurtosis(y, fisher=False, bias=False)),
        "sSQI": float(skew(y, bias=False)),
        "rms": float(np.sqrt(np.mean(y * y))),
    }


def band_power(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    if not np.any(m):
        return 0.0
    df = float(np.median(np.diff(f))) if len(f) > 1 else 1.0
    return float(np.sum(pxx[m]) * df)


def sample_noise_with_id(
    kind: str,
    split_name: str,
    tracks: dict[str, np.ndarray],
    ranges: dict[str, dict[str, tuple[int, int]]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if kind in {"em", "ma", "bw"}:
        seg, start = sample_track_window(tracks[kind], split_name, ranges[kind], rng)
        return seg, f"{kind}:{split_name}:{start}"
    if kind == "mix":
        em, em_start = sample_track_window(tracks["em"], split_name, ranges["em"], rng)
        ma, ma_start = sample_track_window(tracks["ma"], split_name, ranges["ma"], rng)
        return (em + ma).astype(np.float32), f"mix:{split_name}:em{em_start}_ma{ma_start}"
    return make_noise_segment(kind, split_name, tracks, ranges, rng), f"{kind}:{split_name}:synthetic"


def sample_track_window(
    track: np.ndarray,
    split_name: str,
    ranges: dict[str, tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    lo, hi = ranges[split_name]
    if hi - lo <= N:
        raise ValueError(f"Noise range too short for split={split_name}")
    start = int(rng.integers(lo, hi - N))
    return track[start : start + N].astype(np.float32), start


def build_summary(
    labels: pd.DataFrame,
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    *,
    skipped_groups: int,
    gray_rows: int,
    label_version: str,
) -> dict[str, Any]:
    y = labels["y_class"].astype(str).to_numpy()
    snr_measured = 10.0 * np.log10(
        (np.mean(X_clean * X_clean, axis=1) + 1e-12)
        / (np.mean((X_noisy - X_clean) ** 2, axis=1) + 1e-12)
    )
    snr_oracle = np.where(snr_measured >= 10.0, "good", np.where(snr_measured >= 0.0, "medium", "bad"))
    metric_cols = [
        "measured_snr_db",
        "proxy_pSQI",
        "proxy_basSQI",
        "proxy_kSQI",
        "proxy_sSQI",
        "proxy_rms",
        "qrs_nprd",
        "tst_nprd",
        "beat_corr",
        "max_beat_nprd",
        "damage_score",
        "critical_damage_score",
        "diagnostic_damage_score",
        "core_diagnostic_score",
        "beat_axis",
        "global_noise_score",
        "global_nprd",
        "legacy_damage_score",
    ]
    class_summary = {
        col: {
            cls: describe(labels.loc[labels["y_class"] == cls, col].to_numpy(dtype=np.float64))
            for cls in CLASS_ORDER
        }
        for col in metric_cols
    }
    return {
        "rows": int(len(labels)),
        "label_version": label_version,
        "counterfactual_groups": int(labels["counterfactual_group"].nunique()),
        "variants_per_group": 3,
        "skipped_groups": int(skipped_groups),
        "gray_zone_candidate_rows": int(gray_rows),
        "measured_snr_oracle_acc": float(np.mean(snr_oracle == y)),
        "split_y_class_counts": split_counts(labels, "y_class"),
        "split_noise_kind_counts": split_counts(labels, "noise_kind"),
        "split_placement_counts": split_counts(labels, "placement"),
        "split_label_subtype_counts": split_counts(labels, "label_subtype"),
        "label_by_placement": nested_counts(labels, "placement", "y_class"),
        "dominant_axis_by_class": nested_counts(labels, "y_class", "dominant_axis"),
        "dominant_axis_by_placement_class": nested_counts(
            labels,
            ["y_class", "placement"],
            "dominant_axis",
        ),
        "label_subtype_by_class": nested_counts(labels, "y_class", "label_subtype"),
        "audit": {
            "class_metric_summary": class_summary,
            "max_class_mean_gap": {
                col: max_class_mean_gap(class_summary[col])
                for col in ["measured_snr_db", "proxy_pSQI", "proxy_basSQI", "proxy_kSQI", "proxy_sSQI", "proxy_rms"]
            },
        },
    }


def describe(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "n": float(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def max_class_mean_gap(summary: dict[str, dict[str, float]]) -> float:
    means = [summary[cls]["mean"] for cls in CLASS_ORDER if summary[cls]["n"] > 0]
    if not means:
        return 0.0
    return float(max(means) - min(means))


def split_counts(labels: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    return {
        str(sp): {str(k): int(v) for k, v in d[column].value_counts().sort_index().to_dict().items()}
        for sp, d in labels.groupby("split")
    }


def nested_counts(labels: pd.DataFrame, by: str, value: str) -> dict[str, dict[str, int]]:
    return {
        str(name): {str(k): int(v) for k, v in d[value].value_counts().sort_index().to_dict().items()}
        for name, d in labels.groupby(by)
    }


def sample_rows(df: pd.DataFrame, limit: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    idx = rng.choice(np.arange(len(df)), size=limit, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def _none_or_int(value: object, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    run(vars(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the morphology-damage triplet benchmark.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_e35_morph_damage_triplet")
    parser.add_argument("--source_artifact_dir", default="outputs/transformer")
    parser.add_argument("--nstdb_root", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_clean", type=int, default=4000)
    parser.add_argument("--max_val_clean", type=int, default=800)
    parser.add_argument("--max_test_clean", type=int, default=800)
    parser.add_argument("--noise_kinds", default="em,ma,bw,mix")
    parser.add_argument("--snr_min", type=float, default=6.0)
    parser.add_argument("--snr_max", type=float, default=12.0)
    parser.add_argument("--group_retries", type=int, default=8)
    parser.add_argument("--label_version", choices=LABEL_VERSIONS, default="e35_morph_damage")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
