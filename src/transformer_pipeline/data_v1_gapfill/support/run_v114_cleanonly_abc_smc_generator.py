from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V114_PATH = HERE / "build_v114_but_style_residual_hybrid.py"
spec = importlib.util.spec_from_file_location("v114_hybrid", V114_PATH)
v114 = importlib.util.module_from_spec(spec)
sys.modules["v114_hybrid"] = v114
assert spec.loader is not None
spec.loader.exec_module(v114)


KEY_FEATURES = [
    "sqi_basSQI",
    "qrs_visibility",
    "qrs_band_ratio",
    "detector_agreement",
    "template_corr",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "non_qrs_diff_p95",
    "raw_diff_abs_p95",
    "band_0p3_1",
    "band_15_30",
    "band_30_45",
    "amplitude_entropy",
    "low_amp_ratio",
]


@dataclass
class ParamSpec:
    names: list[str]
    lo: np.ndarray
    hi: np.ndarray

    def sample_uniform(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=(int(n), len(self.names))).astype(np.float32)

    def sample_elite(self, rng: np.random.Generator, elites: np.ndarray, n: int, shrink: float) -> np.ndarray:
        if len(elites) == 0:
            return self.sample_uniform(rng, n)
        center = elites[rng.integers(0, len(elites), size=int(n))]
        span = np.maximum(np.nanpercentile(elites, 90, axis=0) - np.nanpercentile(elites, 10, axis=0), (self.hi - self.lo) * 0.08)
        noise = rng.normal(0.0, span * float(shrink), size=center.shape)
        # Keep a little global exploration so bad support can jump islands.
        jump = self.sample_uniform(rng, n)
        mix = rng.random(size=(int(n), 1)) < 0.16
        out = np.where(mix, jump, center + noise)
        return np.clip(out, self.lo, self.hi).astype(np.float32)


def lp(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def fval(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(name, default))
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def param_spec(cls: str, subtype: str) -> ParamSpec:
    st = subtype.lower()
    names = [
        "baseline_mult",
        "baseline_add",
        "baseline_freq",
        "ramp_amp",
        "detail_mult",
        "detail_add",
        "hf_bias",
        "lowfreq_add",
        "qrs_extra",
        "qrs_floor",
        "spike_count",
        "spike_amp",
        "spike_width",
        "contact_extra",
        "contact_max_len",
        "gain_compress",
        "smooth_mix",
        "white_amp",
        "texture_mix",
    ]
    if cls == "good":
        lo = [0.05, 0.00, 0.04, -0.04, 0.02, 0.00, 0.00, 0.00, 0.00, 0.68, 0, 0.00, 2, 0.00, 18, 0.78, 0.00, 0.00, 0.00]
        hi = [0.75, 0.04, 0.35, 0.06, 0.55, 0.03, 0.08, 0.04, 0.25, 0.98, 4, 0.12, 5, 0.04, 55, 1.08, 0.14, 0.025, 0.00]
    elif cls == "medium":
        lo = [0.25, 0.00, 0.05, -0.10, 0.20, 0.00, 0.00, 0.00, 0.05, 0.35, 0, 0.00, 2, 0.00, 28, 0.55, 0.00, 0.00, 0.00]
        hi = [1.55, 0.14, 0.65, 0.16, 1.45, 0.10, 0.24, 0.16, 0.65, 0.92, 10, 0.35, 8, 0.16, 130, 1.05, 0.32, 0.055, 0.00]
    else:
        lo = [0.08, 0.00, 0.05, -0.12, 0.02, 0.00, 0.00, 0.00, 0.02, 0.06, 0, 0.00, 2, 0.00, 35, 0.45, 0.00, 0.00, 0.55]
        hi = [1.35, 0.18, 0.90, 0.20, 0.95, 0.10, 0.16, 0.22, 0.55, 0.88, 12, 0.32, 10, 0.42, 230, 1.05, 0.34, 0.050, 1.00]
    lo_a = np.asarray(lo, dtype=np.float32)
    hi_a = np.asarray(hi, dtype=np.float32)
    if "baseline" in st or "lowfreq" in st:
        lo_a[0] *= 1.2
        hi_a[0] *= 1.45
        hi_a[1] += 0.12
        hi_a[7] += 0.14
    if "highfreq" in st or "detail" in st:
        lo_a[4] *= 1.15
        hi_a[4] *= 1.25
        hi_a[6] += 0.08
        hi_a[17] += 0.018
    if "contact" in st or "flatline" in st or "reset" in st:
        hi_a[13] += 0.28
        hi_a[14] += 100.0
        hi_a[15] = min(hi_a[15], 0.88)
    if "low_qrs" in st or "lowqrs" in st or "visibility" in st:
        hi_a[8] += 0.35
        lo_a[9] = min(lo_a[9], 0.03)
        hi_a[9] = min(hi_a[9], 0.68)
    if "detector" in st or "template" in st or "dense" in st or "other" in st:
        hi_a[10] += 4
        hi_a[11] += 0.10
        hi_a[8] += 0.25
        lo_a[18] = max(lo_a[18], 0.82)
    return ParamSpec(names=names, lo=lo_a, hi=hi_a)


def p(params: np.ndarray, spec: ParamSpec, name: str) -> float:
    return float(params[spec.names.index(name)])


def add_baseline(y: np.ndarray, amp: float, freq: float, ramp_amp: float, rng: np.random.Generator) -> np.ndarray:
    if amp <= 1e-7 and abs(ramp_amp) <= 1e-7:
        return y
    n = len(y)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    drift = np.sin(2.0 * math.pi * float(freq) * t + rng.uniform(0.0, 2.0 * math.pi)).astype(np.float32)
    ramp = (t - 0.5).astype(np.float32)
    _, scale = v114.robust_stats(y)
    return (y + float(amp) * scale * drift + float(ramp_amp) * scale * ramp).astype(np.float32)


def add_spikes(y: np.ndarray, count: int, amp: float, width: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or amp <= 1e-7:
        return y
    out = y.copy()
    _, scale = v114.robust_stats(out)
    scale = max(scale, 1e-5)
    n = len(out)
    for _ in range(int(count)):
        center = int(rng.integers(6, max(7, n - 6)))
        w = max(2, int(width))
        lo = max(0, center - w)
        hi = min(n, center + w + 1)
        out[lo:hi] += rng.normal(0.0, float(amp) * scale) * np.hanning(hi - lo).astype(np.float32)
    return out.astype(np.float32)


def smooth_toward_lowpass(y: np.ndarray, mix: float) -> np.ndarray:
    if mix <= 1e-7:
        return y
    low = v114.moving_average(y, 9)
    return ((1.0 - float(mix)) * y + float(mix) * low).astype(np.float32)


def abc_transform(base: np.ndarray, cls: str, subtype: str, target: pd.Series, params: np.ndarray, spec: ParamSpec, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(base, dtype=np.float32).copy()
    bad_texture_mode = cls == "bad" and hasattr(v114.V81, "synthesize_bad_like_but")
    if bad_texture_mode:
        # Formal clean-only: the carrier is PTB clean; the BUT non-clean row is
        # used only as a feature/severity anchor inside the mechanism generator.
        texture = v114.V81.synthesize_bad_like_but(
            y,
            target,
            subtype,
            rng,
            variant=int(rng.integers(0, 100000)),
        ).astype(np.float32)
        mix = p(params, spec, "texture_mix")
        y = ((1.0 - mix) * y + mix * texture).astype(np.float32)
    med, scale = v114.robust_stats(y)
    scale = max(scale, 1e-5)
    y = med + float(p(params, spec, "gain_compress")) * (y - med)

    baseline_target = 1.75 * fval(target, "baseline_step", 0.0) + 0.35 * fval(target, "band_0p3_1", 0.0)
    baseline_amp = p(params, spec, "baseline_mult") * baseline_target + p(params, spec, "baseline_add")
    if bad_texture_mode:
        baseline_amp *= 0.45
    y = add_baseline(y, baseline_amp, p(params, spec, "baseline_freq"), p(params, spec, "ramp_amp"), rng)

    detail_target = (
        0.025 * fval(target, "non_qrs_diff_p95", 0.0)
        + 0.015 * fval(target, "raw_diff_abs_p95", 0.0)
        + 1.8 * fval(target, "band_30_45", 0.0)
        + 0.42 * fval(target, "band_15_30", 0.0)
        + 0.12 * fval(target, "detail_instability", 0.0)
    )
    detail_amp = p(params, spec, "detail_mult") * detail_target + p(params, spec, "detail_add")
    if bad_texture_mode:
        detail_amp *= 0.18
    if detail_amp > 1e-7:
        hf = v114.fft_band_noise(len(y), rng, 22.0, 45.0)
        mf = v114.fft_band_noise(len(y), rng, 5.0, 22.0)
        y = y + float(detail_amp) * scale * (0.55 * hf + 0.45 * mf)
    hf_bias = p(params, spec, "hf_bias") * (0.20 if bad_texture_mode else 1.0)
    if hf_bias > 1e-7:
        y = y + hf_bias * scale * v114.fft_band_noise(len(y), rng, 30.0, 45.0)
    if p(params, spec, "lowfreq_add") > 1e-7:
        y = y + p(params, spec, "lowfreq_add") * scale * v114.fft_band_noise(len(y), rng, 0.25, 5.0)
    white_amp = p(params, spec, "white_amp") * (0.25 if bad_texture_mode else 1.0)
    if white_amp > 1e-7:
        y = y + white_amp * scale * rng.normal(size=len(y)).astype(np.float32)

    qrs_vis = fval(target, "qrs_visibility", 1.5)
    qrs_band = fval(target, "qrs_band_ratio", 0.8)
    template = fval(target, "template_corr", 0.8)
    qrs_severity = np.clip(
        0.45 * max(0.0, 1.35 - qrs_vis)
        + 0.24 * max(0.0, 0.78 - qrs_band)
        + 0.20 * max(0.0, 0.82 - template)
        + p(params, spec, "qrs_extra") * (0.35 if bad_texture_mode else 1.0),
        0.0,
        1.0,
    )
    if qrs_severity > 0.035:
        factor = float(np.clip(1.0 - qrs_severity, p(params, spec, "qrs_floor"), 0.98))
        y = v114.attenuate_peak_windows(y, rng, factor=factor, q=float(np.clip(91.0 - 8.0 * qrs_severity, 80.0, 94.0)), radius=int(5 + 11 * qrs_severity))

    spike_scale = 0.25 if bad_texture_mode else 1.0
    y = add_spikes(y, int(round(p(params, spec, "spike_count") * spike_scale)), p(params, spec, "spike_amp") * spike_scale, int(round(p(params, spec, "spike_width"))), rng)

    contact_base = 0.55 * fval(target, "flatline_ratio", 0.0) + 0.55 * fval(target, "contact_loss_win_ratio", 0.0) + 0.05 * fval(target, "low_amp_ratio", 0.0)
    contact_ratio = float(np.clip(contact_base + p(params, spec, "contact_extra"), 0.0, 0.72))
    if contact_ratio > 0.015:
        n_segments = int(np.clip(round(1 + 10 * contact_ratio), 1, 8))
        max_len = int(np.clip(p(params, spec, "contact_max_len") * (0.55 + contact_ratio), 14, 260))
        y = v114.inject_contact_segments(y, rng, n_segments=n_segments, max_len=max_len)

    y = smooth_toward_lowpass(y, p(params, spec, "smooth_mix"))
    widen = 1.30 if cls == "good" else (1.55 if cls == "medium" else 2.15)
    return v114.soft_clip_like_reference(y, base, widen=widen).astype(np.float32)


def make_record(cls: str, subtype: str, target: pd.Series, ptb_meta: pd.Series, style_row: pd.Series, round_id: int) -> dict[str, object]:
    return {
        "source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "split": "train",
        "y": v114.CLASS_TO_INT[cls],
        "class_name": cls,
        "record_id": f"v114_cleanabc_{v114.clean_str(ptb_meta.get('record_id', 'ptb'), 'ptb')}",
        "subject_id": v114.clean_str(ptb_meta.get("subject_id", "ptb"), "ptb"),
        "transport_subtype": subtype,
        "display_subtype": subtype,
        "original_region": subtype,
        "clean_policy": "v114_cleanonly_abc_smc",
        "v114_source_line": "D2_cleanonly_ptb_style_abc_smc",
        "v114_style_subject_id": v114.clean_str(style_row.get("subject_id", ""), ""),
        "v114_style_source_idx": v114.clean_str(style_row.get("source_idx", ""), ""),
        "v114_target_source_idx": v114.clean_str(target.get("source_idx", ""), ""),
        "v114_native_replay": 0,
        "v114_subtype": subtype,
        "v114_abc_round": int(round_id),
        "ptbxl_source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "ptbxl_record_id": v114.clean_str(ptb_meta.get("record_id", ""), ""),
    }


def score_candidates(target: pd.DataFrame, cand: pd.DataFrame, features: list[str], rng: np.random.Generator) -> np.ndarray:
    tm = v114.safe_feature_matrix(target, features)
    cm = v114.safe_feature_matrix(cand, features)
    center, scale = v114.V81.robust_fit(tm)
    tz = v114.V81.robust_z(tm, center, scale)
    cz = v114.V81.robust_z(cm, center, scale)
    w = np.ones(len(features), dtype=np.float32)
    for i, name in enumerate(features):
        if name in KEY_FEATURES:
            w[i] = 2.5
        if any(k in name for k in ["qrs", "baseline", "flatline", "contact", "band_30_45", "entropy"]):
            w[i] = max(w[i], 1.8)
    target_ids = rng.choice(np.arange(len(tz)), size=len(cz), replace=True)
    dz = (cz - tz[target_ids]) * w[None, :]
    row = np.sqrt(np.nanmean(dz * dz, axis=1))
    # Add a weak center penalty so nearest-target matching does not ignore the
    # subtype cloud as a whole.
    center_penalty = np.sqrt(np.nanmean(((cz - np.nanmedian(tz, axis=0)) * w[None, :]) ** 2, axis=1))
    return (0.82 * row + 0.18 * center_penalty).astype(np.float32)


def generate_subtype_pool(
    *,
    cls: str,
    subtype: str,
    target: pd.DataFrame,
    but_tv: pd.DataFrame,
    but_x: np.ndarray,
    ptb: pd.DataFrame,
    ptb_x: np.ndarray,
    anchors: pd.DataFrame,
    rounds: int,
    round_size: int,
    elite_frac: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    spec = param_spec(cls, subtype)
    anchor_pos = anchors["_row_pos"].astype(int).to_numpy()
    ptb_pos = ptb["_row_pos"].astype(int).to_numpy()
    target_rows = target.reset_index(drop=True)
    all_frames: list[pd.DataFrame] = []
    all_x: list[np.ndarray] = []
    all_params: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    elite_params = np.empty((0, len(spec.names)), dtype=np.float32)
    shrink = 0.55
    for r in range(int(rounds)):
        params = spec.sample_uniform(rng, round_size) if r == 0 else spec.sample_elite(rng, elite_params, round_size, shrink)
        rows: list[dict[str, object]] = []
        xs: list[np.ndarray] = []
        for i in range(int(round_size)):
            target_row = target_rows.iloc[int(rng.integers(0, len(target_rows)))]
            ppos = int(ptb_pos[int(rng.integers(0, len(ptb_pos)))])
            apos = int(anchor_pos[int(rng.integers(0, len(anchor_pos)))])
            ptb_meta = ptb.loc[ptb["_row_pos"].eq(ppos)].iloc[0]
            style_row = but_tv.loc[but_tv["_row_pos"].eq(apos)].iloc[0]
            base = v114.style_ptb_to_but(ptb_x[ppos], but_x[apos], rng)
            x = abc_transform(base, cls, subtype, target_row, params[i], spec, rng)
            rows.append(make_record(cls, subtype, target_row, ptb_meta, style_row, r))
            xs.append(x)
        frame0 = pd.DataFrame(rows)
        x0 = np.vstack(xs).astype(np.float32)
        frame, x = v114.normalize_frame(frame0, x0, f"clean-only ABC {cls}/{subtype} round{r}")
        features = v114.available_features(target, frame, v114.V81.MATCH_FEATURES)
        dist = score_candidates(target, frame, features, rng)
        frame["abc_distance"] = dist
        keep_n = max(8, int(round(float(elite_frac) * len(frame))))
        keep_local = np.argsort(dist)[:keep_n]
        all_frames.append(frame.iloc[keep_local].copy())
        all_x.append(x[keep_local])
        all_params.append(params[keep_local])
        elite_params = np.vstack(all_params)
        # Keep elite memory bounded while preserving the best rounds.
        if len(elite_params) > max(keep_n * 4, 128):
            all_dist = np.concatenate([f["abc_distance"].to_numpy(dtype=np.float32) for f in all_frames])
            all_param_cat = np.vstack(all_params)
            best = np.argsort(all_dist)[: max(keep_n * 4, 128)]
            elite_params = all_param_cat[best]
        audit_rows.append(
            {
                "class_name": cls,
                "subtype": subtype,
                "round": int(r),
                "round_size": int(round_size),
                "elite_n": int(keep_n),
                "median_distance": float(np.median(dist)),
                "elite_median_distance": float(np.median(dist[keep_local])),
                "elite_p90_distance": float(np.percentile(dist[keep_local], 90)),
                "shrink": float(shrink),
            }
        )
        shrink = max(0.20, shrink * 0.78)
    pool = pd.concat(all_frames, ignore_index=True)
    pool_x = np.vstack(all_x).astype(np.float32)
    pool["idx"] = np.arange(len(pool), dtype=int)
    pool["_row_pos"] = np.arange(len(pool), dtype=int)
    return pool, pool_x, pd.DataFrame(audit_rows)


def select_final(
    target: pd.DataFrame,
    pool: pd.DataFrame,
    pool_x: np.ndarray,
    final_n: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    n = min(int(final_n), len(pool))
    if n <= 0:
        return pool.iloc[[]].copy(), pool_x[:0], {"selected_n": 0}
    features = v114.available_features(target, pool, v114.V81.MATCH_FEATURES)
    tm = v114.safe_feature_matrix(target, features)
    pm = v114.safe_feature_matrix(pool, features)
    center, scale = v114.V81.robust_fit(tm)
    tz = v114.V81.robust_z(tm, center, scale)
    pz = v114.V81.robust_z(pm, center, scale)
    target_ids = rng.choice(np.arange(len(tz)), size=n, replace=len(tz) < n)
    remaining = set(range(len(pool)))
    chosen: list[int] = []
    dists: list[float] = []
    # Half nearest-to-target, half best ABC distance keeps tails without losing
    # the subtype-level empirical CDF completely.
    for ti in target_ids[: n // 2]:
        rem = np.fromiter(remaining, dtype=int)
        dz = pz[rem] - tz[int(ti)]
        dist = np.nanmean(dz * dz, axis=1)
        j = int(rem[int(np.argmin(dist))])
        remaining.remove(j)
        chosen.append(j)
        dists.append(float(np.sqrt(np.min(dist))))
        if not remaining:
            break
    if remaining and len(chosen) < n:
        rem = np.fromiter(remaining, dtype=int)
        abc = pd.to_numeric(pool.iloc[rem]["abc_distance"], errors="coerce").fillna(9999.0).to_numpy()
        for j in rem[np.argsort(abc)[: n - len(chosen)]]:
            chosen.append(int(j))
    out = pool.iloc[np.asarray(chosen, dtype=int)].copy().reset_index(drop=True)
    x = pool_x[np.asarray(chosen, dtype=int)]
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    return out, x.astype(np.float32), {
        "selected_n": int(len(out)),
        "pool_n": int(len(pool)),
        "median_nn_dist": float(np.median(dists)) if dists else np.nan,
        "p90_nn_dist": float(np.percentile(dists, 90)) if dists else np.nan,
        "selected_abc_median": float(pd.to_numeric(out["abc_distance"], errors="coerce").median()) if len(out) else np.nan,
    }


def per_class_final_counts(but_tv: pd.DataFrame, final_per_class: int, allocation: str) -> dict[tuple[str, str], int]:
    sub = v114.subtype_col(but_tv)
    counts: dict[tuple[str, str], int] = {}
    for cls in v114.CLASS_ORDER:
        mask = but_tv["class_name"].astype(str).eq(cls)
        subs = sorted(sub.loc[mask].unique())
        if not subs:
            continue
        if allocation == "equal":
            base = int(final_per_class) // len(subs)
            rem = int(final_per_class) - base * len(subs)
            for i, subtype in enumerate(subs):
                counts[(cls, str(subtype))] = base + (1 if i < rem else 0)
            continue
        natural = sub.loc[mask].value_counts().reindex(subs).fillna(0).astype(float)
        raw = natural / max(float(natural.sum()), 1.0) * int(final_per_class)
        base_counts = np.floor(raw).astype(int)
        # Keep tiny but present subtypes represented, while preserving the
        # natural within-class subtype proportions as closely as possible.
        min_present = 1 if int(final_per_class) < 1000 else 4
        base_counts = base_counts.mask(natural > 0, np.maximum(base_counts, min_present))
        diff = int(final_per_class) - int(base_counts.sum())
        frac = (raw - np.floor(raw)).to_numpy()
        order = np.argsort(-frac)
        if diff > 0:
            for i in order[:diff]:
                base_counts.iloc[int(i)] += 1
        elif diff < 0:
            removable = np.argsort(frac)
            need = -diff
            for i in removable:
                idx = int(i)
                can = int(base_counts.iloc[idx] - min_present)
                if can <= 0:
                    continue
                take = min(can, need)
                base_counts.iloc[idx] -= take
                need -= take
                if need <= 0:
                    break
        for subtype, n in base_counts.items():
            counts[(cls, str(subtype))] = int(n)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-size", type=int, default=420)
    parser.add_argument("--elite-frac", type=float, default=0.30)
    parser.add_argument("--final-per-class", type=int, default=2400)
    parser.add_argument("--subtype-allocation", choices=["natural", "equal"], default="natural")
    parser.add_argument("--selection-scope", choices=["subtype", "class"], default="class")
    parser.add_argument("--max-ptb-carriers", type=int, default=9000)
    parser.add_argument("--bad-round-mult", type=float, default=1.35)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(int(args.seed))

    print(f"{v114.now()} loading BUT reference", flush=True)
    but0, butx0 = v114.V81.load_protocol(v114.DEFAULT_BUT_PROTOCOL)
    but, but_x = v114.normalize_frame(but0, butx0, "BUT reference")
    but_tv = v114.split_train_val(but)
    anchors = v114.select_clean_anchors(but_tv, min_rows=120)

    print(f"{v114.now()} loading PTB clean carriers", flush=True)
    ptb0, ptbx0 = v114.V81.load_protocol(v114.DEFAULT_PTB_CARRIER_PROTOCOL)
    ptb, ptb_x = v114.normalize_frame(ptb0, ptbx0, "PTB clean carrier")
    if len(ptb) > int(args.max_ptb_carriers):
        ids = rng.choice(np.arange(len(ptb)), size=int(args.max_ptb_carriers), replace=False)
        ptb = ptb.iloc[ids].reset_index(drop=True)
        ptb_x = ptb_x[ids]
        ptb["_row_pos"] = np.arange(len(ptb), dtype=int)

    sub = v114.subtype_col(but_tv)
    final_counts = per_class_final_counts(but_tv, int(args.final_per_class), str(args.subtype_allocation))
    selected_frames: list[pd.DataFrame] = []
    selected_xs: list[np.ndarray] = []
    audits: list[pd.DataFrame] = []
    select_audits: list[dict[str, object]] = []

    for cls in v114.CLASS_ORDER:
        class_pools: list[pd.DataFrame] = []
        class_xs: list[np.ndarray] = []
        subtypes = sorted(sub.loc[but_tv["class_name"].astype(str).eq(cls)].unique())
        for subtype in subtypes:
            target = but_tv.loc[but_tv["class_name"].astype(str).eq(cls) & sub.eq(subtype)].copy()
            if len(target) == 0:
                continue
            rs = int(round(int(args.round_size) * (float(args.bad_round_mult) if cls == "bad" else 1.0)))
            print(f"{v114.now()} ABC-SMC {cls}/{subtype} target={len(target)} round_size={rs}", flush=True)
            pool, pool_x, audit = generate_subtype_pool(
                cls=cls,
                subtype=str(subtype),
                target=target,
                but_tv=but_tv,
                but_x=but_x,
                ptb=ptb,
                ptb_x=ptb_x,
                anchors=anchors,
                rounds=int(args.rounds),
                round_size=rs,
                elite_frac=float(args.elite_frac),
                rng=rng,
            )
            if str(args.selection_scope) == "class":
                class_pools.append(pool)
                class_xs.append(pool_x)
            else:
                final_n = int(final_counts[(cls, str(subtype))])
                sel, sel_x, sa = select_final(target, pool, pool_x, final_n, rng)
                sa.update({"class_name": cls, "subtype": str(subtype), "target_n": int(len(target)), "final_target_n": int(final_n), "selection_scope": "subtype"})
                select_audits.append(sa)
                selected_frames.append(sel)
                selected_xs.append(sel_x)
            audits.append(audit)
        if str(args.selection_scope) == "class":
            class_pool = pd.concat(class_pools, ignore_index=True)
            class_x = np.vstack(class_xs).astype(np.float32)
            class_pool["idx"] = np.arange(len(class_pool), dtype=int)
            class_pool["_row_pos"] = np.arange(len(class_pool), dtype=int)
            class_target = but_tv.loc[but_tv["class_name"].astype(str).eq(cls)].copy()
            sel, sel_x, sa = select_final(class_target, class_pool, class_x, int(args.final_per_class), rng)
            sa.update({"class_name": cls, "subtype": "__class_level__", "target_n": int(len(class_target)), "final_target_n": int(args.final_per_class), "selection_scope": "class"})
            select_audits.append(sa)
            selected_frames.append(sel)
            selected_xs.append(sel_x)

    selected = pd.concat(selected_frames, ignore_index=True)
    selected_x = np.vstack(selected_xs).astype(np.float32)
    selected["idx"] = np.arange(len(selected), dtype=int)
    selected["_row_pos"] = np.arange(len(selected), dtype=int)
    selected["split"] = v114.assign_protocol_splits(selected, int(args.seed) + 77)
    name = f"ptb_v114_cleanonly_abc_smc_s{args.seed}"
    out_path = v114.PROTOCOL_ROOT / name
    v114.save_protocol(
        out_path,
        selected,
        selected_x,
        {
            "protocol": name,
            "seed": int(args.seed),
            "rows": int(len(selected)),
            "rounds": int(args.rounds),
            "round_size": int(args.round_size),
            "elite_frac": float(args.elite_frac),
            "final_per_class": int(args.final_per_class),
            "subtype_allocation": str(args.subtype_allocation),
            "selection_scope": str(args.selection_scope),
            "contract": "Formal clean-only line: PTB clean carriers + BUT clean style anchors; BUT non-clean rows only provide feature-distribution targets, never waveform donors.",
        },
    )

    report_dir = v114.REPORT_ROOT / "v114_cleanonly_abc_smc" / f"s{args.seed}" / name
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(audits, ignore_index=True).to_csv(lp(report_dir / f"{name}_abc_round_audit.csv"), index=False)
    pd.DataFrame(select_audits).to_csv(lp(report_dir / f"{name}_selection_audit.csv"), index=False)

    if not args.skip_audit:
        q_features = v114.available_features(but, but, v114.QUALITY_DELTA_FEATURES)
        baselines, global_base = v114.subject_baselines(anchors, q_features)
        v114.audit_protocol(
            but=but,
            but_x=but_x,
            synth=selected,
            synth_x=selected_x,
            tag=name,
            report_dir=report_dir,
            seed=int(args.seed) + 99,
            baselines=baselines,
            global_base=global_base,
        )

    print(
        json.dumps(
            {
                "protocol": str(out_path),
                "report": str(report_dir),
                "rows": int(len(selected)),
                "contract": "clean-only PTB carrier; BUT clean style only; BUT non-clean feature targets only",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
