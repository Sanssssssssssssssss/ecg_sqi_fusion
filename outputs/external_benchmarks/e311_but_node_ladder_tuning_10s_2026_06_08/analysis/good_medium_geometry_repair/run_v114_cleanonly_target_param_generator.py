from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
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


def lp(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def val(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(name, default))
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def add_target_baseline(y: np.ndarray, target: pd.Series, rng: np.random.Generator, scale: float) -> np.ndarray:
    n = len(y)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    baseline = val(target, "baseline_step", 0.0)
    band01 = val(target, "band_0p3_1", 0.0)
    amp = np.clip(1.6 * baseline + 0.45 * band01, 0.0, 0.75) * scale
    if amp <= 1e-6:
        return y
    drift = np.sin(2.0 * math.pi * rng.uniform(0.07, 0.45) * t + rng.uniform(0.0, 2.0 * math.pi))
    ramp = (t - 0.5) * rng.uniform(-1.0, 1.0)
    return y + (amp * rng.uniform(0.65, 1.25) * (0.78 * drift + 0.22 * ramp)).astype(np.float32)


def add_target_detail(y: np.ndarray, target: pd.Series, rng: np.random.Generator, scale: float) -> np.ndarray:
    diff = val(target, "non_qrs_diff_p95", 0.0)
    rawdiff = val(target, "raw_diff_abs_p95", diff)
    band30 = val(target, "band_30_45", 0.0)
    band15 = val(target, "band_15_30", 0.0)
    detail = val(target, "detail_instability", 0.0)
    amp = np.clip(0.025 * diff + 0.015 * rawdiff + 2.2 * band30 + 0.45 * band15 + 0.12 * detail, 0.0, 0.45) * scale
    if amp <= 1e-6:
        return y
    hf = v114.fft_band_noise(len(y), rng, 18.0, 45.0)
    mf = v114.fft_band_noise(len(y), rng, 5.0, 18.0)
    return (y + rng.uniform(0.60, 1.20) * amp * (0.68 * hf + 0.32 * mf)).astype(np.float32)


def apply_target_qrs(y: np.ndarray, target: pd.Series, rng: np.random.Generator) -> np.ndarray:
    qrs_vis = val(target, "qrs_visibility", 1.6)
    qrs_band = val(target, "qrs_band_ratio", 0.8)
    template = val(target, "template_corr", 0.82)
    detector = val(target, "detector_agreement", 0.95)
    # Lower target visibility/agreement means stronger peak attenuation and
    # irregular local events.  High-quality targets keep QRS mostly intact.
    severity = np.clip(0.55 * max(0.0, 1.35 - qrs_vis) + 0.28 * max(0.0, 0.80 - qrs_band) + 0.22 * max(0.0, 0.85 - template), 0.0, 1.0)
    if severity > 0.04:
        factor = float(np.clip(1.0 - rng.uniform(0.40, 0.85) * severity, 0.10, 0.94))
        y = v114.attenuate_peak_windows(y, rng, factor=factor, q=88.0 + 6.0 * rng.random(), radius=int(rng.integers(5, 13)))
    if detector < 0.85:
        _, scale = v114.robust_stats(y)
        for _ in range(int(rng.integers(2, 10))):
            center = int(rng.integers(8, max(9, len(y) - 8)))
            width = int(rng.integers(2, 7))
            amp = rng.normal(0.0, np.clip(0.10 + 0.38 * (0.85 - detector), 0.08, 0.45) * max(scale, 1e-5))
            lo = max(0, center - width)
            hi = min(len(y), center + width + 1)
            y[lo:hi] += amp * np.hanning(hi - lo).astype(np.float32)
    return y.astype(np.float32)


def apply_target_contact(y: np.ndarray, target: pd.Series, rng: np.random.Generator) -> np.ndarray:
    flat = val(target, "flatline_ratio", 0.0)
    contact = val(target, "contact_loss_win_ratio", 0.0)
    lowamp = val(target, "low_amp_ratio", 0.0)
    ratio = float(np.clip(0.65 * flat + 0.55 * contact + 0.08 * lowamp, 0.0, 0.45))
    if ratio <= 0.015:
        return y
    n_segments = int(np.clip(round(1 + 8 * ratio), 1, 6))
    max_len = int(np.clip(round(len(y) * ratio / max(n_segments, 1) * 1.65), 18, 180))
    return v114.inject_contact_segments(y, rng, n_segments=n_segments, max_len=max_len)


def target_guided_transform(base: np.ndarray, cls: str, subtype: str, target: pd.Series, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(base, dtype=np.float32).copy()
    _, scale = v114.robust_stats(y)
    scale = max(scale, 1e-5)
    if cls == "good":
        # Good targets get only gentle, feature-shaped perturbations.
        y = add_target_baseline(y, target, rng, scale * 0.28)
        y = add_target_detail(y, target, rng, scale * 0.18)
        y = apply_target_qrs(y, target, rng) if "low" in subtype.lower() or "artifact" in subtype.lower() else y
    elif cls == "medium":
        y = add_target_baseline(y, target, rng, scale * 0.62)
        y = add_target_detail(y, target, rng, scale * 0.55)
        y = apply_target_qrs(y, target, rng)
        y = apply_target_contact(y, target, rng)
    else:
        y = add_target_baseline(y, target, rng, scale * 1.10)
        y = add_target_detail(y, target, rng, scale * 1.35)
        y = apply_target_qrs(y, target, rng)
        y = apply_target_contact(y, target, rng)
        # Bad subtype priors only add mechanism shape; magnitude is still driven
        # by target features above.
        if any(k in subtype.lower() for k in ["highfreq", "detail", "detector", "dense", "other"]):
            y = add_target_detail(y, target, rng, scale * 0.55)
        if any(k in subtype.lower() for k in ["baseline", "lowfreq"]):
            y = add_target_baseline(y, target, rng, scale * 0.75)
        if any(k in subtype.lower() for k in ["contact", "flatline", "reset"]):
            y = apply_target_contact(y, target, rng)
    return v114.soft_clip_like_reference(y, base, widen=1.55).astype(np.float32)


def synthesize_one(
    cls: str,
    subtype: str,
    target: pd.Series,
    ptb_signal: np.ndarray,
    style_signal: np.ndarray,
    ptb_meta: pd.Series,
    style_row: pd.Series,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    base = v114.style_ptb_to_but(ptb_signal, style_signal, rng)
    x = target_guided_transform(base, cls, subtype, target, rng)
    rec = {
        "source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "split": "train",
        "y": v114.CLASS_TO_INT[cls],
        "class_name": cls,
        "record_id": f"v114_cleanparam_{v114.clean_str(ptb_meta.get('record_id', 'ptb'), 'ptb')}",
        "subject_id": v114.clean_str(ptb_meta.get("subject_id", "ptb"), "ptb"),
        "transport_subtype": subtype,
        "display_subtype": subtype,
        "original_region": subtype,
        "clean_policy": "v114_cleanonly_target_param",
        "v114_source_line": "D2_cleanonly_target_param",
        "v114_style_subject_id": v114.clean_str(style_row.get("subject_id", ""), ""),
        "v114_style_source_idx": v114.clean_str(style_row.get("source_idx", ""), ""),
        "v114_target_source_idx": v114.clean_str(target.get("source_idx", ""), ""),
        "v114_native_replay": 0,
        "v114_subtype": subtype,
        "ptbxl_source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "ptbxl_record_id": v114.clean_str(ptb_meta.get("record_id", ""), ""),
    }
    return x, rec


def build_pool(
    but_tv: pd.DataFrame,
    but_x: np.ndarray,
    ptb: pd.DataFrame,
    ptb_x: np.ndarray,
    pool_per_subtype: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    anchors = v114.select_clean_anchors(but_tv, min_rows=120)
    anchor_pos = anchors["_row_pos"].astype(int).to_numpy()
    ptb_pos = ptb["_row_pos"].astype(int).to_numpy()
    sub = v114.subtype_col(but_tv)
    rows: list[dict[str, object]] = []
    signals: list[np.ndarray] = []
    for cls in v114.CLASS_ORDER:
        for subtype in sorted(sub.loc[but_tv["class_name"].astype(str).eq(cls)].unique()):
            targets = but_tv.loc[but_tv["class_name"].astype(str).eq(cls) & sub.eq(subtype)]
            if len(targets) == 0:
                continue
            for _ in range(int(pool_per_subtype)):
                target = targets.iloc[int(rng.integers(0, len(targets)))]
                ppos = int(ptb_pos[int(rng.integers(0, len(ptb_pos)))])
                apos = int(anchor_pos[int(rng.integers(0, len(anchor_pos)))])
                ptb_meta = ptb.loc[ptb["_row_pos"].eq(ppos)].iloc[0]
                style_row = but_tv.loc[but_tv["_row_pos"].eq(apos)].iloc[0]
                sig, rec = synthesize_one(
                    cls=str(cls),
                    subtype=str(subtype),
                    target=target,
                    ptb_signal=ptb_x[ppos],
                    style_signal=but_x[apos],
                    ptb_meta=ptb_meta,
                    style_row=style_row,
                    rng=rng,
                )
                rows.append(rec)
                signals.append(sig)
    frame = pd.DataFrame(rows)
    x = np.vstack(signals).astype(np.float32)
    frame["idx"] = np.arange(len(frame), dtype=int)
    frame["_row_pos"] = np.arange(len(frame), dtype=int)
    return frame, x


def select_nearest(
    target: pd.DataFrame,
    pool: pd.DataFrame,
    pool_x: np.ndarray,
    final_per_subtype: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    features = v114.available_features(target, pool, v114.V81.MATCH_FEATURES)
    sub_t = v114.subtype_col(target)
    sub_p = v114.subtype_col(pool)
    selected: list[int] = []
    audit: list[dict[str, object]] = []
    for cls in v114.CLASS_ORDER:
        for subtype in sorted(sub_t.loc[target["class_name"].astype(str).eq(cls)].unique()):
            t = target.loc[target["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)]
            p = pool.loc[pool["class_name"].astype(str).eq(cls) & sub_p.eq(subtype)]
            n = min(int(final_per_subtype), len(p))
            if len(t) == 0 or n == 0:
                continue
            tm = v114.safe_feature_matrix(t, features)
            pm = v114.safe_feature_matrix(p, features)
            center, scale = v114.V81.robust_fit(tm)
            tz = v114.V81.robust_z(tm, center, scale)
            pz = v114.V81.robust_z(pm, center, scale)
            target_ids = rng.choice(np.arange(len(tz)), size=n, replace=len(tz) < n)
            remaining = set(range(len(pz)))
            p_index = p.index.to_numpy(dtype=int)
            dists: list[float] = []
            for ti in target_ids:
                rem = np.fromiter(remaining, dtype=int)
                dz = pz[rem] - tz[int(ti)]
                dist = np.sum(dz * dz, axis=1)
                local = int(rem[int(np.argmin(dist))])
                remaining.remove(local)
                selected.append(int(p_index[local]))
                dists.append(float(np.sqrt(np.min(dist))))
                if not remaining:
                    break
            audit.append(
                {
                    "class_name": cls,
                    "subtype": subtype,
                    "target_n": int(len(t)),
                    "pool_n": int(len(p)),
                    "selected_n": int(len(dists)),
                    "median_nn_dist": float(np.median(dists)) if dists else np.nan,
                    "p90_nn_dist": float(np.percentile(dists, 90)) if dists else np.nan,
                }
            )
    out = pool.loc[np.asarray(selected, dtype=int)].copy().reset_index(drop=True)
    out_x = pool_x[np.asarray(selected, dtype=int)].astype(np.float32)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    return out, out_x, pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--pool-per-subtype", type=int, default=1800)
    parser.add_argument("--final-per-subtype", type=int, default=500)
    parser.add_argument("--max-ptb-carriers", type=int, default=9000)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(int(args.seed))
    print(f"{v114.now()} loading BUT target + clean style anchors", flush=True)
    but0, butx0 = v114.V81.load_protocol(v114.DEFAULT_BUT_PROTOCOL)
    but, but_x = v114.normalize_frame(but0, butx0, "BUT reference")
    but_tv = v114.split_train_val(but)
    print(f"{v114.now()} loading PTB clean carriers", flush=True)
    ptb0, ptbx0 = v114.V81.load_protocol(v114.DEFAULT_PTB_CARRIER_PROTOCOL)
    ptb, ptb_x = v114.normalize_frame(ptb0, ptbx0, "PTB raw carrier")
    if len(ptb) > int(args.max_ptb_carriers):
        ids = rng.choice(np.arange(len(ptb)), size=int(args.max_ptb_carriers), replace=False)
        ptb = ptb.iloc[ids].reset_index(drop=True)
        ptb_x = ptb_x[ids]
        ptb["_row_pos"] = np.arange(len(ptb), dtype=int)
    print(f"{v114.now()} building target-param pool", flush=True)
    pool0, poolx0 = build_pool(but_tv, but_x, ptb, ptb_x, int(args.pool_per_subtype), rng)
    print(f"{v114.now()} recomputing target-param pool features rows={len(pool0)}", flush=True)
    pool, pool_x = v114.normalize_frame(pool0, poolx0, "clean-only target-param pool")
    print(f"{v114.now()} selecting target-param nearest subset", flush=True)
    selected, selected_x, audit = select_nearest(but_tv, pool, pool_x, int(args.final_per_subtype), rng)
    name = f"ptb_v114_cleanonly_target_param_s{args.seed}"
    selected["split"] = v114.assign_protocol_splits(selected, int(args.seed) + 55)
    out_path = v114.PROTOCOL_ROOT / name
    v114.save_protocol(
        out_path,
        selected,
        selected_x,
        {
            "protocol": name,
            "seed": int(args.seed),
            "rows": int(len(selected)),
            "pool_per_subtype": int(args.pool_per_subtype),
            "final_per_subtype": int(args.final_per_subtype),
            "contract": "Formal clean-only line: BUT clean anchors/style and BUT feature targets only; no BUT medium/bad waveform donor.",
        },
    )
    report_dir = v114.REPORT_ROOT / "v114_cleanonly_target_param" / f"s{args.seed}" / name
    report_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(lp(report_dir / f"{name}_selection_audit.csv"), index=False)
    if not args.skip_audit:
        anchors = v114.select_clean_anchors(but_tv, min_rows=120)
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
    print(json.dumps({"protocol": str(out_path), "report": str(report_dir), "rows": int(len(selected))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
