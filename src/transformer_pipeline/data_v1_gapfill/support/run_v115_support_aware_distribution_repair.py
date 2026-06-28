from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for CPU-only hosts.
    torch = None


HERE = Path(__file__).resolve().parent


def import_local(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v114 = import_local("v114_hybrid", HERE / "build_v114_but_style_residual_hybrid.py")
abc114 = import_local("v114_clean_abc", HERE / "run_v114_cleanonly_abc_smc_generator.py")


REPORT_ROOT = v114.REPORT_ROOT / "v115_support_aware_distribution_repair"
PROTOCOL_ROOT = v114.PROTOCOL_ROOT

REGIMES = [
    "midband_periodic",
    "pseudo_qrs_train",
    "qrs_atten_gain",
    "baseline_step_ramp",
    "contact_reset_flatline",
    "quantization_clipping",
    "motion_am",
]

KEY_GAP_FEATURES = [
    "detector_agreement",
    "sqi_iSQI",
    "sqi_bSQI",
    "qrs_visibility",
    "qrs_band_ratio",
    "template_corr",
    "band_30_45",
    "non_qrs_rms_ratio",
    "amplitude_entropy",
    "flatline_ratio",
    "baseline_step",
]


def lp(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def now() -> str:
    return v114.now()


def safe_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(name, default))
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def sample_frame(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= int(n):
        return frame.copy()
    return frame.iloc[rng.choice(np.arange(len(frame)), size=int(n), replace=False)].copy()


def robust_z_against(target: pd.DataFrame, other: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tm = v114.safe_feature_matrix(target, features)
    om = v114.safe_feature_matrix(other, features)
    center, scale = v114.V81.robust_fit(tm)
    return v114.V81.robust_z(tm, center, scale), v114.V81.robust_z(om, center, scale)


def metric_row(tag: str, scope: str, target: pd.DataFrame, synth: pd.DataFrame, features: list[str], rng: np.random.Generator) -> dict[str, Any]:
    tz, sz = robust_z_against(target, synth, features)
    auc = v114.V81.domain_auc(tz, sz, rng)
    sym_auc = max(float(auc), 1.0 - float(auc)) if math.isfinite(float(auc)) else np.nan
    return {
        "tag": tag,
        "scope": scope,
        "but_n": int(len(target)),
        "synthetic_n": int(len(synth)),
        "rbf_mmd": float(v114.V81.rbf_mmd(tz, sz, rng)),
        "sliced_wasserstein": float(v114.V81.sliced_wasserstein(tz, sz, rng)),
        "quantile_loss": float(v114.V81.quantile_loss(tz, sz)),
        "domain_auc": float(auc),
        "sym_domain_auc": float(sym_auc),
        "pca_density_overlap": float(v114.V81.pca_density_overlap(tz, sz)),
    }


def compute_v110_metrics(target: pd.DataFrame, synth: pd.DataFrame, tag: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    features = v114.available_features(target, synth, v114.V81.MATCH_FEATURES)
    target_tv = target.loc[target["split"].astype(str).isin(["train", "val"])].copy()
    sub_t = v114.subtype_col(target_tv)
    sub_s = v114.subtype_col(synth)
    rows: list[dict[str, Any]] = []
    for cls in v114.CLASS_ORDER:
        for subtype in sorted(sub_t.loc[target_tv["class_name"].astype(str).eq(cls)].unique()):
            t = target_tv.loc[target_tv["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)]
            s = synth.loc[synth["class_name"].astype(str).eq(cls) & sub_s.eq(subtype)]
            if len(t) >= 5 and len(s) >= 5:
                row = metric_row(tag, f"{cls}/{subtype}", t, s, features, rng)
                row.update({"class_name": cls, "subtype": subtype, "ptb_n": int(len(s))})
                rows.append(row)
    metric = pd.DataFrame(rows)
    if metric.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            metric.groupby("class_name", as_index=False)
            .agg(
                subtype_rows=("subtype", "count"),
                but_n=("but_n", "sum"),
                ptb_n=("synthetic_n", "sum"),
                rbf_mmd_median=("rbf_mmd", "median"),
                rbf_mmd_mean=("rbf_mmd", "mean"),
                sliced_wasserstein_median=("sliced_wasserstein", "median"),
                quantile_loss_median=("quantile_loss", "median"),
                sym_domain_auc_median=("sym_domain_auc", "median"),
                pca_density_overlap_median=("pca_density_overlap", "median"),
                pca_density_overlap_mean=("pca_density_overlap", "mean"),
            )
            .assign(tag=tag)
        )
    global_rows = []
    global_rows.append(metric_row(tag, "all_labels", target_tv, synth, features, rng))
    for cls in v114.CLASS_ORDER:
        t = target_tv.loc[target_tv["class_name"].astype(str).eq(cls)]
        s = synth.loc[synth["class_name"].astype(str).eq(cls)]
        if len(t) >= 5 and len(s) >= 5:
            global_rows.append(metric_row(tag, f"class_{cls}", t, s, features, rng))
    return metric, summary, pd.DataFrame(global_rows)


def support_floor_and_coverage(
    target: pd.DataFrame,
    synth: pd.DataFrame,
    *,
    features: list[str],
    rng: np.random.Generator,
    max_target_rows: int = 2500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_tv = target.loc[target["split"].astype(str).isin(["train", "val"])].copy()
    sub_t = v114.subtype_col(target_tv)
    sub_s = v114.subtype_col(synth)
    rows: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []
    for cls in v114.CLASS_ORDER:
        t_cls = target_tv.loc[target_tv["class_name"].astype(str).eq(cls)].copy()
        s_cls = synth.loc[synth["class_name"].astype(str).eq(cls)].copy()
        if len(t_cls) < 10 or len(s_cls) < 5:
            continue
        t_eval = sample_frame(t_cls, max_target_rows, rng).reset_index(drop=True)
        tz, sz = robust_z_against(t_cls, s_cls, features)
        # Map sampled rows into z-space via robust params from full t_cls.
        center, scale = v114.V81.robust_fit(v114.safe_feature_matrix(t_cls, features))
        te_z = v114.V81.robust_z(v114.safe_feature_matrix(t_eval, features), center, scale)
        # BUT-vs-BUT cross-record nearest floor.
        floor_dist = np.full(len(t_eval), np.nan, dtype=np.float32)
        t_full_z = tz
        t_records = t_cls["record_id"].astype(str).to_numpy()
        t_eval_records = t_eval["record_id"].astype(str).to_numpy()
        nn_floor = NearestNeighbors(n_neighbors=min(15, len(t_full_z)), metric="euclidean").fit(t_full_z)
        dists, idxs = nn_floor.kneighbors(te_z, return_distance=True)
        for i in range(len(t_eval)):
            for d, j in zip(dists[i], idxs[i]):
                if t_records[int(j)] != t_eval_records[i]:
                    floor_dist[i] = float(d)
                    break
        floor_valid = floor_dist[np.isfinite(floor_dist)]
        q90 = float(np.percentile(floor_valid, 90)) if len(floor_valid) else np.nan
        q95 = float(np.percentile(floor_valid, 95)) if len(floor_valid) else np.nan
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(sz)
        cand_dist, cand_idx = nn.kneighbors(te_z, return_distance=True)
        cand_dist = cand_dist[:, 0]
        cand_idx = cand_idx[:, 0]
        rows.append(
            {
                "class_name": cls,
                "subtype": "__class__",
                "target_n": int(len(t_cls)),
                "synthetic_n": int(len(s_cls)),
                "eval_n": int(len(t_eval)),
                "but_but_q90": q90,
                "but_but_q95": q95,
                "target_to_synth_median": float(np.median(cand_dist)),
                "target_to_synth_q90": float(np.percentile(cand_dist, 90)),
                "target_to_synth_q95": float(np.percentile(cand_dist, 95)),
                "coverage_q90": float(np.mean(cand_dist <= q90)) if math.isfinite(q90) else np.nan,
                "coverage_q95": float(np.mean(cand_dist <= q95)) if math.isfinite(q95) else np.nan,
                "candidate_support_insufficient": bool(cls == "bad" and (not math.isfinite(q95) or float(np.mean(cand_dist <= q95)) < 0.50)),
            }
        )
        sub_eval = v114.subtype_col(t_eval)
        sub_s_cls = v114.subtype_col(s_cls)
        for subtype in sorted(sub_eval.unique()):
            mask = sub_eval.eq(subtype)
            cd = cand_dist[np.asarray(mask, dtype=bool)]
            if len(cd) == 0:
                continue
            rows.append(
                {
                    "class_name": cls,
                    "subtype": subtype,
                    "target_n": int(mask.sum()),
                    "synthetic_n": int(sub_s_cls.eq(subtype).sum()) if len(s_cls) else 0,
                    "eval_n": int(len(cd)),
                    "but_but_q90": q90,
                    "but_but_q95": q95,
                    "target_to_synth_median": float(np.median(cd)),
                    "target_to_synth_q90": float(np.percentile(cd, 90)),
                    "target_to_synth_q95": float(np.percentile(cd, 95)),
                    "coverage_q90": float(np.mean(cd <= q90)) if math.isfinite(q90) else np.nan,
                    "coverage_q95": float(np.mean(cd <= q95)) if math.isfinite(q95) else np.nan,
                    "candidate_support_insufficient": bool(cls == "bad" and (not math.isfinite(q95) or float(np.mean(cd <= q95)) < 0.50)),
                }
            )
        s_idx = s_cls.index.to_numpy()
        for k in np.argsort(-cand_dist)[: min(80, len(cand_dist))]:
            target_row = t_eval.iloc[int(k)]
            synth_row = synth.loc[int(s_idx[int(cand_idx[int(k)])])]
            nearest_rows.append(
                {
                    "class_name": cls,
                    "target_subtype": str(v114.subtype_col(t_eval).iloc[int(k)]),
                    "target_source_idx": str(target_row.get("source_idx", "")),
                    "target_record_id": str(target_row.get("record_id", "")),
                    "nearest_distance": float(cand_dist[int(k)]),
                    "but_but_q95": q95,
                    "nearest_synth_subtype": str(synth_row.get("display_subtype", synth_row.get("transport_subtype", ""))),
                    "nearest_synth_regime": str(synth_row.get("v115_regime", "")),
                    "nearest_synth_source_idx": str(synth_row.get("source_idx", "")),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(nearest_rows)


def spectral_noise(n: int, rng: np.random.Generator, bands: list[tuple[float, float, float]]) -> np.ndarray:
    freqs = np.fft.rfftfreq(n, d=1.0 / 125.0)
    spec = np.zeros(len(freqs), dtype=np.complex128)
    for low, high, gain in bands:
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            continue
        phase = rng.uniform(-np.pi, np.pi, int(mask.sum()))
        mag = rng.rayleigh(max(float(gain), 1e-6), int(mask.sum()))
        spec[mask] += mag * np.exp(1j * phase)
    x = np.fft.irfft(spec, n=n).astype(np.float32)
    _, scale = v114.robust_stats(x)
    return (x / max(scale, 1e-5)).astype(np.float32)


def add_pseudo_qrs_train(n: int, rng: np.random.Generator, amp: float, period_range: tuple[float, float], width_range: tuple[int, int]) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    pos = float(rng.uniform(5, 35))
    period = float(rng.uniform(*period_range))
    while pos < n - 10:
        width = int(rng.integers(width_range[0], width_range[1] + 1))
        p = int(np.clip(pos + rng.normal(0.0, 5.5), 2, n - width - 2))
        pulse = np.hanning(width).astype(np.float32)
        if rng.random() < 0.20:
            pulse = -pulse
        x[p : p + width] += float(amp) * pulse
        pos += period + rng.normal(0.0, 7.0)
    return x


def quantize_clip(x: np.ndarray, rng: np.random.Generator, levels: int, clip_q: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    lo, hi = np.percentile(y, [100 - clip_q, clip_q])
    y = np.clip(y, lo, hi)
    bins = np.linspace(lo, hi, max(int(levels), 3), dtype=np.float32)
    idx = np.searchsorted(bins, y, side="left")
    idx = np.clip(idx, 0, len(bins) - 1)
    y = bins[idx]
    _, scale = v114.robust_stats(y)
    y += rng.normal(0.0, 0.006 * max(scale, 1e-5), len(y)).astype(np.float32)
    return y.astype(np.float32)


def regime_bad_transform(base: np.ndarray, target: pd.Series, subtype: str, regime: str, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(base, dtype=np.float32).copy()
    n = len(y)
    med, scale = v114.robust_stats(y)
    scale = max(scale, 1e-5)
    target_rms = max(safe_float(target, "raw_rms", safe_float(target, "rms", 0.16)), 1e-5)
    target_ptp = max(safe_float(target, "raw_ptp_p99_p01", safe_float(target, "ptp_p99_p01", 0.65)), 1e-5)
    target_base = max(safe_float(target, "baseline_step", 0.04), 0.0)
    target_diff = max(safe_float(target, "non_qrs_diff_p95", safe_float(target, "raw_diff_abs_p95", 0.08)), 0.005)
    if regime == "midband_periodic":
        tex = spectral_noise(n, rng, [(0.3, 5.0, 0.12), (5.0, 12.0, 0.48), (12.0, 22.0, 0.30), (22.0, 45.0, 0.02)])
        t = np.arange(n, dtype=np.float32) / 125.0
        tex += rng.uniform(0.15, 0.45) * np.sin(2 * np.pi * rng.uniform(8.5, 18.0) * t + rng.uniform(-np.pi, np.pi)).astype(np.float32)
        y = med + rng.uniform(0.12, 0.38) * (y - med) + target_rms * tex
    elif regime == "pseudo_qrs_train":
        pulse = add_pseudo_qrs_train(n, rng, amp=rng.uniform(0.35, 1.20) * scale, period_range=(30.0, 95.0), width_range=(4, 18))
        tex = spectral_noise(n, rng, [(0.4, 5.0, 0.16), (5.0, 15.0, 0.36), (15.0, 30.0, 0.08)])
        y = med + rng.uniform(0.08, 0.28) * (y - med) + pulse + target_rms * rng.uniform(0.20, 0.55) * tex
    elif regime == "qrs_atten_gain":
        y = v114.attenuate_peak_windows(y, rng, factor=rng.uniform(0.08, 0.42), q=rng.uniform(82.0, 92.0), radius=int(rng.integers(6, 18)))
        y = med + rng.uniform(0.22, 0.72) * (y - med)
        y += target_rms * rng.uniform(0.04, 0.18) * spectral_noise(n, rng, [(0.3, 5.0, 0.35), (5.0, 16.0, 0.25), (16.0, 45.0, 0.03)])
    elif regime == "baseline_step_ramp":
        t = np.linspace(0, 1, n, dtype=np.float32)
        drift = np.sin(2 * np.pi * rng.uniform(0.08, 0.55) * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        ramp = (t - 0.5).astype(np.float32)
        step = np.zeros(n, dtype=np.float32)
        step[int(rng.integers(n // 5, max(n // 5 + 1, 4 * n // 5))) :] = rng.normal(0, 1)
        y = y + scale * ((0.18 + 3.0 * target_base) * drift + rng.uniform(-0.25, 0.35) * ramp + rng.uniform(0.04, 0.24) * step)
    elif regime == "contact_reset_flatline":
        y = v114.inject_contact_segments(y, rng, n_segments=int(rng.integers(1, 6)), max_len=int(rng.integers(45, 230)))
        if rng.random() < 0.7:
            pos = int(rng.integers(8, max(9, n - 8)))
            y[pos:] += rng.normal(0.0, rng.uniform(0.08, 0.35) * scale)
    elif regime == "quantization_clipping":
        y = quantize_clip(y, rng, levels=int(rng.integers(8, 36)), clip_q=float(rng.uniform(82, 98)))
        y = v114.attenuate_peak_windows(y, rng, factor=rng.uniform(0.22, 0.72), q=90.0, radius=int(rng.integers(4, 11)))
    elif regime == "motion_am":
        t = np.linspace(0, 1, n, dtype=np.float32)
        env = 1.0 + rng.uniform(0.20, 0.75) * np.sin(2 * np.pi * rng.uniform(0.4, 3.5) * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        y = med + env * (y - med)
        y += scale * rng.uniform(0.05, 0.22) * spectral_noise(n, rng, [(0.2, 4.0, 0.45), (4.0, 14.0, 0.28), (14.0, 35.0, 0.04)])
    # Match coarse amplitude and derivative without making a high-frequency wall.
    y = y - float(np.nanmedian(y))
    cur_rms = max(float(np.sqrt(np.nanmean(y * y))), 1e-6)
    cur_ptp = max(float(np.nanpercentile(y, 99) - np.nanpercentile(y, 1)), 1e-6)
    gain = 0.65 * target_rms / cur_rms + 0.35 * target_ptp / cur_ptp
    y = y * float(np.clip(gain, 0.05, 4.0))
    cur_diff = max(float(np.nanpercentile(np.abs(np.diff(y)), 95)), 1e-6)
    if cur_diff < 0.70 * target_diff:
        y += float(np.clip(target_diff - cur_diff, 0.002, 0.05)) * spectral_noise(n, rng, [(5.0, 18.0, 0.45), (18.0, 35.0, 0.04)])
    return v114.soft_clip_like_reference(y.astype(np.float32), base, widen=2.2)


def make_v115_record(cls: str, subtype: str, target: pd.Series, ptb_meta: pd.Series, style_row: pd.Series, regime: str, source_line: str) -> dict[str, Any]:
    return {
        "source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "split": "train",
        "y": v114.CLASS_TO_INT[cls],
        "class_name": cls,
        "record_id": f"v115_{source_line}_{v114.clean_str(ptb_meta.get('record_id', 'ptb'), 'ptb')}",
        "subject_id": v114.clean_str(ptb_meta.get("subject_id", "ptb"), "ptb"),
        "transport_subtype": subtype,
        "display_subtype": subtype,
        "original_region": subtype,
        "clean_policy": "v115_support_aware_distribution_repair",
        "v115_regime": regime,
        "v115_source_line": source_line,
        "v115_style_source_idx": v114.clean_str(style_row.get("source_idx", ""), ""),
        "v115_target_source_idx": v114.clean_str(target.get("source_idx", ""), ""),
        "v115_native_replay": 0,
        "ptbxl_source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "ptbxl_record_id": v114.clean_str(ptb_meta.get("record_id", ""), ""),
    }


def build_cleanonly_regime_bank(
    *,
    but_tv: pd.DataFrame,
    but_x: np.ndarray,
    ptb: pd.DataFrame,
    ptb_x: np.ndarray,
    rng: np.random.Generator,
    max_candidates_per_class: int,
    bad_regimes: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    anchors = v114.select_clean_anchors(but_tv, min_rows=120)
    anchor_pos = anchors["_row_pos"].astype(int).to_numpy()
    ptb_pos = ptb["_row_pos"].astype(int).to_numpy()
    sub = v114.subtype_col(but_tv)
    rows: list[dict[str, Any]] = []
    xs: list[np.ndarray] = []
    for cls in v114.CLASS_ORDER:
        target_cls = but_tv.loc[but_tv["class_name"].astype(str).eq(cls)].copy()
        if len(target_cls) == 0:
            continue
        n_total = int(max_candidates_per_class)
        if cls == "bad":
            regimes = bad_regimes
        else:
            regimes = ["abc_cleanonly"]
        for i in range(n_total):
            target = target_cls.iloc[int(rng.integers(0, len(target_cls)))]
            subtype = str(target.get("display_subtype", target.get("transport_subtype", target.get("original_region", cls))))
            ppos = int(ptb_pos[int(rng.integers(0, len(ptb_pos)))])
            apos = int(anchor_pos[int(rng.integers(0, len(anchor_pos)))])
            ptb_meta = ptb.loc[ptb["_row_pos"].eq(ppos)].iloc[0]
            style_row = but_tv.loc[but_tv["_row_pos"].eq(apos)].iloc[0]
            base = v114.style_ptb_to_but(ptb_x[ppos], but_x[apos], rng)
            if cls == "bad":
                regime = str(regimes[i % len(regimes)])
                sig = regime_bad_transform(base, target, subtype, regime, rng)
            else:
                regime = "abc_cleanonly"
                spec = abc114.param_spec(cls, subtype)
                params = spec.sample_uniform(rng, 1)[0]
                sig = abc114.abc_transform(base, cls, subtype, target, params, spec, rng)
            rows.append(make_v115_record(cls, subtype, target, ptb_meta, style_row, regime, "cleanonly_regime"))
            xs.append(sig.astype(np.float32))
    frame = pd.DataFrame(rows)
    x = np.vstack(xs).astype(np.float32)
    frame["idx"] = np.arange(len(frame), dtype=int)
    frame["_row_pos"] = np.arange(len(frame), dtype=int)
    frame, x = v114.normalize_frame(frame, x, "v115 clean-only regime candidate bank")
    return frame, x


def set_objective(target: pd.DataFrame, selected: pd.DataFrame, coverage: float, features: list[str], rng: np.random.Generator, baseline: dict[str, float]) -> dict[str, float]:
    row = metric_row("objective", "class", target, selected, features, rng)
    mmd = row["rbf_mmd"] / max(baseline.get("mmd", 1.0), 1e-6)
    swd = row["sliced_wasserstein"] / max(baseline.get("swd", 1.0), 1e-6)
    cdf = row["quantile_loss"] / max(baseline.get("cdf", 1.0), 1e-6)
    dom = max(0.0, row["sym_domain_auc"] - 0.60) ** 2
    overlap = 1.0 - row["pca_density_overlap"]
    cov_pen = max(0.0, 0.80 - float(coverage)) ** 2
    j = 0.45 * mmd + 0.16 * swd + 0.12 * cdf + 0.10 * dom + 0.10 * overlap + 0.07 * cov_pen
    row.update({"coverage_q95": float(coverage), "objective": float(j)})
    return row


def resolve_torch_device(requested: str) -> Any:
    if torch is None:
        return None
    req = str(requested or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(req)


def torch_rff_embedding(z: Any, rff_dim: int, seed: int, sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)) -> Any:
    """Multi-kernel random Fourier features for fast set-level MMD proposals."""
    assert torch is not None
    dim = int(z.shape[1])
    per = max(16, int(rff_dim) // len(sigmas))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    outs = []
    for sigma in sigmas:
        omega = torch.randn((dim, per), generator=gen, dtype=torch.float32) / float(sigma)
        bias = 2.0 * math.pi * torch.rand((per,), generator=gen, dtype=torch.float32)
        omega = omega.to(device=z.device)
        bias = bias.to(device=z.device)
        outs.append(torch.cos(z @ omega + bias))
    out = torch.cat(outs, dim=1)
    return out * math.sqrt(2.0 / float(out.shape[1]))


def gpu_rff_mcmc_select(
    *,
    target_cls: pd.DataFrame,
    pool_cls: pd.DataFrame,
    pool_x: np.ndarray,
    final_n: int,
    features: list[str],
    rng: np.random.Generator,
    swaps: int,
    class_name: str,
    device_request: str,
    rff_dim: int,
    seed: int,
    support_max_target_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """GPU-accelerated set MCMC. Exact v110 metrics are still computed after selection."""
    device = resolve_torch_device(device_request)
    if torch is None or device is None or device.type != "cuda":
        raise RuntimeError("GPU selector requested without CUDA")
    n = min(int(final_n), len(pool_cls))
    if n <= 0:
        return pool_cls.iloc[[]].copy(), pool_x[:0], pd.DataFrame()
    tz, pz = robust_z_against(target_cls, pool_cls, features)
    # Initial seed is diverse nearest-target coverage in robust-z space, then GPU MCMC refines the set.
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(pz)
    _, idx = nn.kneighbors(tz, return_distance=True)
    selected = list(dict.fromkeys(idx[:, 0].tolist()))
    if len(selected) < n:
        selected.extend([int(i) for i in rng.choice(np.arange(len(pool_cls)), size=n - len(selected), replace=False) if int(i) not in selected])
    if len(selected) < n:
        selected = list(dict.fromkeys(selected + rng.choice(np.arange(len(pool_cls)), size=n, replace=False).astype(int).tolist()))
    selected = np.asarray(selected[:n], dtype=np.int64)
    selected_mask = np.zeros(len(pool_cls), dtype=bool)
    selected_mask[selected] = True

    z_target = torch.as_tensor(tz, dtype=torch.float32, device=device)
    z_pool = torch.as_tensor(pz, dtype=torch.float32, device=device)
    with torch.no_grad():
        emb_target = torch_rff_embedding(z_target, int(rff_dim), int(seed) + 17)
        emb_pool = torch_rff_embedding(z_pool, int(rff_dim), int(seed) + 17)
        target_mu = emb_target.mean(dim=0)
        target_mean = z_target.mean(dim=0)
        target_s2 = (z_target * z_target).mean(dim=0)
        sel_t = torch.as_tensor(selected, dtype=torch.long, device=device)
        cur_mu = emb_pool.index_select(0, sel_t).mean(dim=0)
        cur_mean = z_pool.index_select(0, sel_t).mean(dim=0)
        cur_s2 = (z_pool.index_select(0, sel_t) ** 2).mean(dim=0)

        def energy(mu: Any, mean: Any, s2: Any) -> Any:
            mmd = torch.sum((mu - target_mu) ** 2)
            mean_gap = torch.mean(torch.abs(mean - target_mean))
            var_gap = torch.mean(torch.abs(torch.sqrt(torch.clamp(s2, min=1e-6)) - torch.sqrt(torch.clamp(target_s2, min=1e-6))))
            return 0.72 * mmd + 0.16 * mean_gap + 0.12 * var_gap

        best_energy = float(energy(cur_mu, cur_mean, cur_s2).detach().cpu())
        best_selected = selected.copy()
        cur_energy = best_energy
        trace: list[dict[str, Any]] = [{
            "step": 0,
            "class_name": class_name,
            "gpu_energy": cur_energy,
            "gpu_best_energy": best_energy,
            "accepted": 0,
            "device": str(device),
        }]
        accepted = 0
        temperature0 = 0.035
        for step in range(1, int(swaps) + 1):
            out_pos = int(rng.integers(0, n))
            out_idx = int(selected[out_pos])
            in_idx = int(rng.integers(0, len(pool_cls)))
            tries = 0
            while selected_mask[in_idx] and tries < 32:
                in_idx = int(rng.integers(0, len(pool_cls)))
                tries += 1
            if selected_mask[in_idx]:
                continue
            delta_emb = (emb_pool[in_idx] - emb_pool[out_idx]) / float(n)
            delta_z = (z_pool[in_idx] - z_pool[out_idx]) / float(n)
            delta_z2 = ((z_pool[in_idx] ** 2) - (z_pool[out_idx] ** 2)) / float(n)
            prop_mu = cur_mu + delta_emb
            prop_mean = cur_mean + delta_z
            prop_s2 = cur_s2 + delta_z2
            prop_energy = float(energy(prop_mu, prop_mean, prop_s2).detach().cpu())
            temp = temperature0 * max(0.03, 1.0 - step / max(float(swaps), 1.0))
            if prop_energy < cur_energy or rng.random() < math.exp(-(prop_energy - cur_energy) / max(temp, 1e-6)):
                selected_mask[out_idx] = False
                selected_mask[in_idx] = True
                selected[out_pos] = in_idx
                cur_mu, cur_mean, cur_s2 = prop_mu, prop_mean, prop_s2
                cur_energy = prop_energy
                accepted += 1
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_selected = selected.copy()
            if step % max(1, int(swaps) // 20) == 0 or step == swaps:
                trace.append({
                    "step": int(step),
                    "class_name": class_name,
                    "gpu_energy": float(cur_energy),
                    "gpu_best_energy": float(best_energy),
                    "accepted": int(accepted),
                    "accept_rate": float(accepted / max(step, 1)),
                    "device": str(device),
                })
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    out = pool_cls.iloc[best_selected].copy().reset_index(drop=True)
    out_x = pool_x[best_selected].astype(np.float32)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    support, _ = support_floor_and_coverage(
        target_cls.assign(split="train"),
        out,
        features=features,
        rng=rng,
        max_target_rows=max(150, int(support_max_target_rows)),
    )
    cov = float(support.loc[support["subtype"].eq("__class__"), "coverage_q95"].iloc[0]) if not support.empty else 0.0
    exact = set_objective(target_cls, out, cov, features, rng, {"mmd": 1.0, "swd": 1.0, "cdf": 1.0})
    trace.append({"step": int(swaps), "class_name": class_name, "coverage_q95": cov, **exact, "device": str(device)})
    return out, out_x, pd.DataFrame(trace)


def greedy_set_select(
    *,
    target_cls: pd.DataFrame,
    pool_cls: pd.DataFrame,
    pool_x: np.ndarray,
    final_n: int,
    features: list[str],
    rng: np.random.Generator,
    swaps: int,
    class_name: str,
    support_max_target_rows: int = 900,
    device_request: str = "cpu",
    rff_dim: int = 1024,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    n = min(int(final_n), len(pool_cls))
    if n <= 0:
        return pool_cls.iloc[[]].copy(), pool_x[:0], pd.DataFrame()
    device = resolve_torch_device(device_request)
    if torch is not None and device is not None and device.type == "cuda":
        return gpu_rff_mcmc_select(
            target_cls=target_cls,
            pool_cls=pool_cls,
            pool_x=pool_x,
            final_n=final_n,
            features=features,
            rng=rng,
            swaps=swaps,
            class_name=class_name,
            device_request=device_request,
            rff_dim=rff_dim,
            seed=seed,
            support_max_target_rows=support_max_target_rows,
        )
    tz, pz = robust_z_against(target_cls, pool_cls, features)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(pz)
    d, idx = nn.kneighbors(tz, return_distance=True)
    # Seed from nearest targets plus low-distance diverse supplement.
    selected = list(dict.fromkeys(idx[:, 0].tolist()))
    if len(selected) < n:
        pool_order = np.argsort(np.min(np.linalg.norm(pz[:, None, :] - tz[None, : min(len(tz), 800), :], axis=2), axis=1))
        for j in pool_order:
            if int(j) not in selected:
                selected.append(int(j))
            if len(selected) >= n:
                break
    selected = np.asarray(selected[:n], dtype=int)
    all_idx = np.arange(len(pool_cls), dtype=int)
    not_selected = np.setdiff1d(all_idx, selected, assume_unique=False)
    baseline = {
        "mmd": max(v114.V81.rbf_mmd(tz, pz[selected], rng), 1e-6),
        "swd": max(v114.V81.sliced_wasserstein(tz, pz[selected], rng), 1e-6),
        "cdf": max(v114.V81.quantile_loss(tz, pz[selected]), 1e-6),
    }
    trace: list[dict[str, Any]] = []
    target_tmp = target_cls.copy()
    best_frame = pool_cls.iloc[selected].copy()
    support, _ = support_floor_and_coverage(
        target_tmp.assign(split="train"),
        best_frame,
        features=features,
        rng=rng,
        max_target_rows=int(support_max_target_rows),
    )
    class_cov = float(support.loc[support["subtype"].eq("__class__"), "coverage_q95"].iloc[0]) if not support.empty else 0.0
    best = set_objective(target_tmp, best_frame, class_cov, features, rng, baseline)
    best_obj = float(best["objective"])
    trace.append({"step": 0, "class_name": class_name, **best})
    temperature0 = 0.025
    for step in range(1, int(swaps) + 1):
        if len(not_selected) == 0:
            break
        out_pos = int(rng.integers(0, len(selected)))
        in_pos = int(not_selected[int(rng.integers(0, len(not_selected)))])
        proposal = selected.copy()
        proposal[out_pos] = in_pos
        prop_frame = pool_cls.iloc[proposal].copy()
        if step % 25 == 0:
            support, _ = support_floor_and_coverage(
                target_tmp.assign(split="train"),
                prop_frame,
                features=features,
                rng=rng,
                max_target_rows=max(150, int(support_max_target_rows) // 2),
            )
            cov = float(support.loc[support["subtype"].eq("__class__"), "coverage_q95"].iloc[0]) if not support.empty else 0.0
        else:
            cov = class_cov
        cur = set_objective(target_tmp, prop_frame, cov, features, rng, baseline)
        obj = float(cur["objective"])
        temp = temperature0 * max(0.05, 1.0 - step / max(float(swaps), 1.0))
        if obj < best_obj or rng.random() < math.exp(-(obj - best_obj) / max(temp, 1e-6)):
            old = selected[out_pos]
            selected = proposal
            not_selected = np.setdiff1d(all_idx, selected, assume_unique=False)
            best_obj = obj
            best = cur
            best_frame = prop_frame
            class_cov = cov
        if step % max(1, int(swaps) // 20) == 0 or step == swaps:
            trace.append({"step": int(step), "class_name": class_name, **best})
    out = pool_cls.iloc[selected].copy().reset_index(drop=True)
    out_x = pool_x[selected].astype(np.float32)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    return out, out_x, pd.DataFrame(trace)


def build_semisynth_pool(
    but_tv: pd.DataFrame,
    but_x: np.ndarray,
    clean_pool: pd.DataFrame,
    clean_x: np.ndarray,
    native_fraction: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    if native_fraction <= 0:
        return clean_pool.copy(), clean_x.copy()
    native = but_tv.loc[but_tv["split"].astype(str).eq("train")].copy()
    if len(native) == 0:
        native = but_tv.copy()
    native_x = but_x[native["_row_pos"].astype(int).to_numpy()]
    n_native = max(1, int(round(len(clean_pool) * float(native_fraction) / max(1.0 - float(native_fraction), 1e-6))))
    take = rng.choice(np.arange(len(native)), size=min(n_native, len(native)), replace=len(native) < n_native)
    nframe = native.iloc[take].copy().reset_index(drop=True)
    nx = native_x[take].astype(np.float32)
    nframe["v115_source_line"] = "semi_synthetic_native_train_only"
    nframe["v115_native_replay"] = 1
    nframe["v115_regime"] = "but_train_native"
    combo = pd.concat([clean_pool.copy(), nframe], ignore_index=True)
    combo_x = np.vstack([clean_x, nx]).astype(np.float32)
    combo["idx"] = np.arange(len(combo), dtype=int)
    combo["_row_pos"] = np.arange(len(combo), dtype=int)
    return combo, combo_x


def select_with_native_quota(
    *,
    target_cls: pd.DataFrame,
    pool_cls: pd.DataFrame,
    pool_x: np.ndarray,
    final_n: int,
    native_fraction: float,
    features: list[str],
    rng: np.random.Generator,
    swaps: int,
    class_name: str,
    support_max_target_rows: int,
    device_request: str,
    rff_dim: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    native_flag = pd.to_numeric(pool_cls.get("v115_native_replay", pd.Series(0, index=pool_cls.index)), errors="coerce").fillna(0).to_numpy() > 0.5
    if float(native_fraction) <= 0.0 or native_flag.sum() == 0:
        return greedy_set_select(
            target_cls=target_cls,
            pool_cls=pool_cls.reset_index(drop=True),
            pool_x=pool_x,
            final_n=final_n,
            features=features,
            rng=rng,
            swaps=swaps,
            class_name=class_name,
            support_max_target_rows=support_max_target_rows,
            device_request=device_request,
            rff_dim=rff_dim,
            seed=seed,
        )
    n_native = min(int(round(int(final_n) * float(native_fraction))), int(native_flag.sum()))
    n_synth = max(0, int(final_n) - n_native)
    if n_synth > int((~native_flag).sum()):
        n_synth = int((~native_flag).sum())
        n_native = min(int(final_n) - n_synth, int(native_flag.sum()))
    outs: list[pd.DataFrame] = []
    xs: list[np.ndarray] = []
    traces: list[pd.DataFrame] = []
    if n_synth > 0:
        sframe = pool_cls.loc[~native_flag].reset_index(drop=True)
        sx = pool_x[np.where(~native_flag)[0]]
        sel, sel_x, trace = greedy_set_select(
            target_cls=target_cls,
            pool_cls=sframe,
            pool_x=sx,
            final_n=n_synth,
            features=features,
            rng=rng,
            swaps=max(50, swaps // 2),
            class_name=f"{class_name}_synthetic_quota",
            support_max_target_rows=support_max_target_rows,
            device_request=device_request,
            rff_dim=rff_dim,
            seed=seed + 11,
        )
        outs.append(sel)
        xs.append(sel_x)
        traces.append(trace)
    if n_native > 0:
        nframe = pool_cls.loc[native_flag].reset_index(drop=True)
        nx = pool_x[np.where(native_flag)[0]]
        sel, sel_x, trace = greedy_set_select(
            target_cls=target_cls,
            pool_cls=nframe,
            pool_x=nx,
            final_n=n_native,
            features=features,
            rng=rng,
            swaps=max(50, swaps // 2),
            class_name=f"{class_name}_native_quota",
            support_max_target_rows=support_max_target_rows,
            device_request=device_request,
            rff_dim=rff_dim,
            seed=seed + 23,
        )
        outs.append(sel)
        xs.append(sel_x)
        traces.append(trace)
    out = pd.concat(outs, ignore_index=True) if outs else pool_cls.iloc[[]].copy()
    out_x = np.vstack(xs).astype(np.float32) if xs else pool_x[:0]
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    trace["quota_native_fraction"] = float(native_fraction)
    return out, out_x, trace


def save_selected_protocol(name: str, frame: pd.DataFrame, x: np.ndarray, seed: int, summary: dict[str, Any]) -> Path:
    out = frame.copy().reset_index(drop=True)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    out["split"] = v114.assign_protocol_splits(out, int(seed) + 131)
    path = PROTOCOL_ROOT / name
    v114.save_protocol(path, out, x.astype(np.float32), summary)
    return path


def write_markdown_report(path: Path, title: str, sections: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "smoke", "support_audit_cleanonly", "set_level_smc_objective", "regime_conditioned_bad_generator", "semi_synthetic_native_fraction_grid"], default="all")
    parser.add_argument("--seed", type=int, default=20260840)
    parser.add_argument("--max-ptb-carriers", type=int, default=6000)
    parser.add_argument("--candidates-per-class", type=int, default=1600)
    parser.add_argument("--final-per-class", type=int, default=900)
    parser.add_argument("--smc-swaps", type=int, default=1500)
    parser.add_argument("--support-max-target-rows", type=int, default=2500)
    parser.add_argument("--rff-dim", type=int, default=1024)
    parser.add_argument("--native-fractions", default="0,0.10,0.25,0.55")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-protocol-write", action="store_true")
    args = parser.parse_args()
    if args.stage == "smoke":
        args.candidates_per_class = min(args.candidates_per_class, 256)
        args.final_per_class = min(args.final_per_class, 96)
        args.smc_swaps = min(args.smc_swaps, 200)
        args.support_max_target_rows = min(args.support_max_target_rows, 350)
        args.rff_dim = min(args.rff_dim, 512)
        args.native_fractions = "0,0.25"
    rng = np.random.default_rng(int(args.seed))
    tag = f"s{args.seed}"
    report_root = REPORT_ROOT / tag
    report_root.mkdir(parents=True, exist_ok=True)

    print(f"{now()} loading BUT reference", flush=True)
    but0, butx0 = v114.V81.load_protocol(v114.DEFAULT_BUT_PROTOCOL)
    but, but_x = v114.normalize_frame(but0, butx0, "v115 BUT reference")
    but_tv = v114.split_train_val(but)
    print(f"{now()} loading PTB clean carriers", flush=True)
    ptb0, ptbx0 = v114.V81.load_protocol(v114.DEFAULT_PTB_CARRIER_PROTOCOL)
    ptb, ptb_x = v114.normalize_frame(ptb0, ptbx0, "v115 PTB clean carrier")
    if len(ptb) > int(args.max_ptb_carriers):
        ids = rng.choice(np.arange(len(ptb)), size=int(args.max_ptb_carriers), replace=False)
        ptb = ptb.iloc[ids].reset_index(drop=True)
        ptb_x = ptb_x[ids]
        ptb["_row_pos"] = np.arange(len(ptb), dtype=int)

    bad_regimes = REGIMES if args.stage != "smoke" else REGIMES[:2]
    print(f"{now()} building clean-only regime candidate bank", flush=True)
    clean_pool, clean_x = build_cleanonly_regime_bank(
        but_tv=but_tv,
        but_x=but_x,
        ptb=ptb,
        ptb_x=ptb_x,
        rng=rng,
        max_candidates_per_class=int(args.candidates_per_class),
        bad_regimes=bad_regimes,
    )
    features = v114.available_features(but_tv, clean_pool, v114.V81.MATCH_FEATURES)

    print(f"{now()} support audit clean-only", flush=True)
    support, nearest = support_floor_and_coverage(
        but,
        clean_pool,
        features=features,
        rng=rng,
        max_target_rows=int(args.support_max_target_rows),
    )
    support.to_csv(lp(report_root / "support_coverage_by_class_subtype.csv"), index=False)
    nearest.to_csv(lp(report_root / "nearest_farthest_support_gaps.csv"), index=False)
    bad_class = support.loc[(support["class_name"].eq("bad")) & (support["subtype"].eq("__class__"))]
    bad_cov = float(bad_class["coverage_q95"].iloc[0]) if not bad_class.empty else 0.0
    support_insufficient = bool(bad_cov < 0.50)
    write_markdown_report(
        report_root / "v115_support_audit_report.md",
        "v115 Support Audit Report",
        [
            f"Clean-only candidate bank rows: {len(clean_pool)}.",
            f"Bad class q95 support coverage: {bad_cov:.4f}.",
            f"candidate_support_insufficient: {support_insufficient}.",
            "Coverage threshold uses BUT-vs-BUT cross-record nearest-neighbor q95 in 47D robust-z space.",
        ],
    )

    if args.stage == "support_audit_cleanonly":
        print(json.dumps({"report": str(report_root), "bad_q95_coverage": bad_cov, "support_insufficient": support_insufficient}, indent=2), flush=True)
        return

    print(f"{now()} set-level SMC selection", flush=True)
    selected_frames: list[pd.DataFrame] = []
    selected_xs: list[np.ndarray] = []
    traces: list[pd.DataFrame] = []
    for cls in v114.CLASS_ORDER:
        target_cls = but_tv.loc[but_tv["class_name"].astype(str).eq(cls)].copy()
        pool_cls = clean_pool.loc[clean_pool["class_name"].astype(str).eq(cls)].copy()
        pool_idx = pool_cls.index.to_numpy(dtype=int)
        sel, sel_x, trace = greedy_set_select(
            target_cls=target_cls,
            pool_cls=pool_cls.reset_index(drop=True),
            pool_x=clean_x[pool_idx],
            final_n=int(args.final_per_class),
            features=features,
            rng=rng,
            swaps=int(args.smc_swaps),
            class_name=cls,
            support_max_target_rows=int(args.support_max_target_rows),
            device_request=str(args.device),
            rff_dim=int(args.rff_dim),
            seed=int(args.seed) + v114.CLASS_ORDER.index(cls) * 1009,
        )
        selected_frames.append(sel)
        selected_xs.append(sel_x)
        traces.append(trace)
    selected = pd.concat(selected_frames, ignore_index=True)
    selected_x = np.vstack(selected_xs).astype(np.float32)
    selected["idx"] = np.arange(len(selected), dtype=int)
    selected["_row_pos"] = np.arange(len(selected), dtype=int)
    trace_df = pd.concat(traces, ignore_index=True)
    trace_df.to_csv(lp(report_root / "set_level_objective_trace.csv"), index=False)
    clean_name = f"ptb_v115_cleanonly_regime_bad_s{args.seed}"
    if not args.skip_protocol_write:
        save_selected_protocol(
            clean_name,
            selected,
            selected_x,
            int(args.seed),
            {
                "protocol": clean_name,
                "seed": int(args.seed),
                "line": "clean-only",
                "contract": "PTB clean carrier + BUT good style anchors + BUT non-clean feature targets; no BUT medium/bad waveform donor.",
                "candidate_support_insufficient": support_insufficient,
                "bad_q95_coverage": bad_cov,
                "final_per_class": int(args.final_per_class),
                "smc_swaps": int(args.smc_swaps),
                "bad_regimes": bad_regimes,
            },
        )
    metric, summary, global_metric = compute_v110_metrics(but, selected, clean_name, int(args.seed) + 21)
    metric.to_csv(lp(report_root / "v110_distribution_metrics.csv"), index=False)
    summary.to_csv(lp(report_root / "v110_class_subtype_median_summary.csv"), index=False)
    global_metric.to_csv(lp(report_root / "v110_global_distribution_metrics.csv"), index=False)
    v114.V81.plot_metric_heatmap(metric, report_root, clean_name, "rbf_mmd")
    v114.V81.plot_shared_pca(but, selected, report_root, clean_name)
    write_markdown_report(
        report_root / "v115_set_level_smc_report.md",
        "v115 Set-Level SMC Report",
        [
            "Selection optimizes a class-level set objective instead of subtype quota or row-distance-only scoring.",
            "See `set_level_objective_trace.csv`, `v110_distribution_metrics.csv`, and shared PCA output.",
            global_metric.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        ],
    )
    reg_counts = selected.groupby(["class_name", "v115_regime"], dropna=False).size().reset_index(name="n")
    reg_counts.to_csv(lp(report_root / "regime_parameter_posterior.csv"), index=False)
    write_markdown_report(
        report_root / "v115_regime_bad_generator_report.md",
        "v115 Regime-Conditioned Bad Generator Report",
        [
            "Bad regimes are signal-processing mechanisms, not white-noise/spike-only corruptions.",
            reg_counts.to_string(index=False),
            "Audit-only signal-processing feature expansion is reserved for the next iteration if these regimes still miss support.",
        ],
    )

    if args.stage in {"set_level_smc_objective", "regime_conditioned_bad_generator"}:
        print(json.dumps({"report": str(report_root), "clean_name": clean_name, "bad_q95_coverage": bad_cov}, indent=2), flush=True)
        return

    print(f"{now()} semi-synthetic native fraction grid", flush=True)
    semi_rows: list[pd.DataFrame] = []
    native_fracs = [float(x.strip()) for x in str(args.native_fractions).split(",") if x.strip()]
    for frac in native_fracs:
        combo_pool, combo_x = build_semisynth_pool(but_tv, but_x, clean_pool, clean_x, frac, rng)
        combo_selected_frames: list[pd.DataFrame] = []
        combo_selected_xs: list[np.ndarray] = []
        for cls in v114.CLASS_ORDER:
            target_cls = but_tv.loc[but_tv["class_name"].astype(str).eq(cls)].copy()
            pool_cls = combo_pool.loc[combo_pool["class_name"].astype(str).eq(cls)].copy()
            pool_idx = pool_cls.index.to_numpy(dtype=int)
            sel, sel_x, _ = select_with_native_quota(
                target_cls=target_cls,
                pool_cls=pool_cls.reset_index(drop=True),
                pool_x=combo_x[pool_idx],
                final_n=int(args.final_per_class),
                native_fraction=float(frac),
                features=features,
                rng=rng,
                swaps=max(100, int(args.smc_swaps) // 2),
                class_name=cls,
                support_max_target_rows=max(150, int(args.support_max_target_rows) // 2),
                device_request=str(args.device),
                rff_dim=int(args.rff_dim),
                seed=int(args.seed) + int(round(frac * 1000)) + v114.CLASS_ORDER.index(cls) * 1009,
            )
            combo_selected_frames.append(sel)
            combo_selected_xs.append(sel_x)
        combo_sel = pd.concat(combo_selected_frames, ignore_index=True)
        combo_sel_x = np.vstack(combo_selected_xs).astype(np.float32)
        name = f"hybrid_v115_semisynth_native{int(round(frac * 100)):02d}_s{args.seed}"
        if not args.skip_protocol_write:
            save_selected_protocol(
                name,
                combo_sel,
                combo_sel_x,
                int(args.seed) + int(round(frac * 1000)),
                {
                    "protocol": name,
                    "seed": int(args.seed),
                    "line": "controlled_semi_synthetic",
                    "native_fraction_requested": float(frac),
                    "leakage_statement": "BUT train-only native support may be used; BUT test is excluded from donor, target optimization, MMD selection, and model selection.",
                    "final_per_class": int(args.final_per_class),
                },
            )
        _, ss, gg = compute_v110_metrics(but, combo_sel, name, int(args.seed) + 44)
        out = gg.copy()
        out["native_fraction_requested"] = float(frac)
        out["actual_native_fraction"] = float(pd.to_numeric(combo_sel.get("v115_native_replay", pd.Series(0, index=combo_sel.index)), errors="coerce").fillna(0).mean())
        semi_rows.append(out)
        ss.to_csv(lp(report_root / f"{name}_class_subtype_median_summary.csv"), index=False)
    semi = pd.concat(semi_rows, ignore_index=True) if semi_rows else pd.DataFrame()
    semi.to_csv(lp(report_root / "semisynthetic_fraction_grid_metrics.csv"), index=False)
    write_markdown_report(
        report_root / "v115_semisynthetic_fraction_grid_report.md",
        "v115 Semi-Synthetic Native Fraction Grid Report",
        [
            "This is not a clean-only claim. It is controlled target-domain-informed distribution repair.",
            "BUT train-only native support is permitted here; BUT test remains excluded.",
            semi.to_string(index=False, float_format=lambda x: f"{x:.4f}") if not semi.empty else "No semi-synthetic rows generated.",
        ],
    )
    print(json.dumps({"report": str(report_root), "clean_name": clean_name, "bad_q95_coverage": bad_cov}, indent=2), flush=True)


if __name__ == "__main__":
    main()
