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


def apply_medium_mechanism(x: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    _, scale = v114.robust_stats(y)
    scale = max(scale, 1e-5)
    st = subtype.lower()
    if "baseline" in st or "outlier" in st or "boundary" in st:
        t = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
        drift = np.sin(2.0 * math.pi * rng.uniform(0.08, 0.35) * t + rng.uniform(0.0, 2.0 * math.pi))
        y = y + rng.uniform(0.05, 0.18) * scale * drift
    if "visible" in st or "detail" in st or "overlap" in st:
        y = y + rng.uniform(0.012, 0.040) * scale * v114.fft_band_noise(len(y), rng, 18.0, 42.0)
        y = y + rng.uniform(0.010, 0.032) * scale * v114.fft_band_noise(len(y), rng, 0.5, 6.0)
    if "lowqrs" in st or "isolated" in st:
        y = v114.attenuate_peak_windows(y, rng, factor=rng.uniform(0.48, 0.78), q=91.0, radius=8)
    if "clean_core" in st:
        y = y + rng.uniform(0.004, 0.014) * scale * rng.normal(size=len(y)).astype(np.float32)
    return v114.soft_clip_like_reference(y, x, widen=1.35)


def apply_good_mechanism(x: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    _, scale = v114.robust_stats(y)
    scale = max(scale, 1e-5)
    st = subtype.lower()
    if "core" in st:
        y = y + rng.uniform(0.001, 0.006) * scale * rng.normal(size=len(y)).astype(np.float32)
    if "overlap" in st or "mild" in st:
        y = y + rng.uniform(0.004, 0.016) * scale * v114.fft_band_noise(len(y), rng, 18.0, 38.0)
    if "baseline" in st or "hard" in st:
        t = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
        y = y + rng.uniform(0.015, 0.055) * scale * np.sin(
            2.0 * math.pi * rng.uniform(0.06, 0.18) * t + rng.uniform(0.0, 2.0 * math.pi)
        ).astype(np.float32)
    if "isolated" in st:
        y = v114.attenuate_peak_windows(y, rng, factor=rng.uniform(0.82, 0.94), q=93.0, radius=5)
    return v114.soft_clip_like_reference(y, x, widen=1.25)


def cleanonly_synthesize(
    *,
    cls: str,
    subtype: str,
    ptb_signal: np.ndarray,
    style_ref: np.ndarray,
    ptb_meta: pd.Series,
    style_row: pd.Series,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    base = v114.style_ptb_to_but(ptb_signal, style_ref, rng)
    if cls == "good":
        x = apply_good_mechanism(base, subtype, rng)
    elif cls == "medium":
        x = apply_medium_mechanism(base, subtype, rng)
    else:
        x = v114.apply_bad_mechanism(base, subtype, rng)
    rec = {
        "source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "split": "train",
        "y": v114.CLASS_TO_INT[cls],
        "class_name": cls,
        "record_id": f"v114_cleanonly_{v114.clean_str(ptb_meta.get('record_id', 'ptb'), 'ptb')}",
        "subject_id": v114.clean_str(ptb_meta.get("subject_id", "ptb"), "ptb"),
        "transport_subtype": subtype,
        "display_subtype": subtype,
        "original_region": subtype,
        "clean_policy": "v114_cleanonly_style_generator",
        "v114_source_line": "D2_cleanonly_ptb_style_mechanism",
        "v114_style_subject_id": v114.clean_str(style_row.get("subject_id", ""), ""),
        "v114_style_source_idx": v114.clean_str(style_row.get("source_idx", ""), ""),
        "v114_native_replay": 0,
        "v114_subtype": subtype,
        "ptbxl_source_idx": v114.clean_str(ptb_meta.get("source_idx", ""), ""),
        "ptbxl_record_id": v114.clean_str(ptb_meta.get("record_id", ""), ""),
    }
    return x.astype(np.float32), rec


def build_pool(
    *,
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
            for _ in range(int(pool_per_subtype)):
                ppos = int(ptb_pos[int(rng.integers(0, len(ptb_pos)))])
                apos = int(anchor_pos[int(rng.integers(0, len(anchor_pos)))])
                ptb_meta = ptb.loc[ptb["_row_pos"].eq(ppos)].iloc[0]
                style_row = but_tv.loc[but_tv["_row_pos"].eq(apos)].iloc[0]
                sig, rec = cleanonly_synthesize(
                    cls=cls,
                    subtype=str(subtype),
                    ptb_signal=ptb_x[ppos],
                    style_ref=but_x[apos],
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


def robust_nearest_select(
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
            c = pool.loc[pool["class_name"].astype(str).eq(cls) & sub_p.eq(subtype)]
            n = min(int(final_per_subtype), len(c))
            if len(t) == 0 or n <= 0:
                continue
            tm = v114.safe_feature_matrix(t, features)
            cm = v114.safe_feature_matrix(c, features)
            med = np.nanmedian(tm, axis=0)
            q25 = np.nanpercentile(tm, 25, axis=0)
            q75 = np.nanpercentile(tm, 75, axis=0)
            scale = np.maximum(q75 - q25, 1e-5)
            tm = np.nan_to_num((tm - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
            cm = np.nan_to_num((cm - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
            target_idx = rng.choice(np.arange(len(tm)), size=n, replace=len(tm) < n)
            remaining = set(range(len(c)))
            c_index = c.index.to_numpy(dtype=int)
            dists: list[float] = []
            for ti in target_idx:
                rem = np.fromiter(remaining, dtype=int)
                dz = cm[rem] - tm[int(ti)]
                dist = np.sum(dz * dz, axis=1)
                j = int(rem[int(np.argmin(dist))])
                remaining.remove(j)
                selected.append(int(c_index[j]))
                dists.append(float(np.sqrt(np.min(dist))))
                if not remaining:
                    break
            audit.append(
                {
                    "class_name": cls,
                    "subtype": subtype,
                    "target_n": int(len(t)),
                    "pool_n": int(len(c)),
                    "selected_n": int(min(n, len(dists))),
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
    parser.add_argument("--seed", type=int, default=20260800)
    parser.add_argument("--pool-per-subtype", type=int, default=1200)
    parser.add_argument("--final-per-subtype", type=int, default=400)
    parser.add_argument("--max-ptb-carriers", type=int, default=9000)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    print(f"{v114.now()} loading BUT clean style target", flush=True)
    but0, butx0 = v114.V81.load_protocol(v114.DEFAULT_BUT_PROTOCOL)
    but, but_x = v114.normalize_frame(but0, butx0, "BUT reference")
    but_tv = v114.split_train_val(but)
    print(f"{v114.now()} loading PTB carrier", flush=True)
    ptb0, ptbx0 = v114.V81.load_protocol(v114.DEFAULT_PTB_CARRIER_PROTOCOL)
    ptb, ptb_x = v114.normalize_frame(ptb0, ptbx0, "PTB raw carrier")
    if len(ptb) > int(args.max_ptb_carriers):
        ids = rng.choice(np.arange(len(ptb)), size=int(args.max_ptb_carriers), replace=False)
        ptb = ptb.iloc[ids].reset_index(drop=True)
        ptb_x = ptb_x[ids]
        ptb["_row_pos"] = np.arange(len(ptb), dtype=int)
    print(f"{v114.now()} building clean-only pool", flush=True)
    pool0, poolx0 = build_pool(
        but_tv=but_tv,
        but_x=but_x,
        ptb=ptb,
        ptb_x=ptb_x,
        pool_per_subtype=int(args.pool_per_subtype),
        rng=rng,
    )
    print(f"{v114.now()} recomputing pool features rows={len(pool0)}", flush=True)
    pool, pool_x = v114.normalize_frame(pool0, poolx0, "clean-only PTB style mechanism pool")
    print(f"{v114.now()} selecting nearest clean-only subset", flush=True)
    selected, selected_x, audit = robust_nearest_select(
        but_tv,
        pool,
        pool_x,
        final_per_subtype=int(args.final_per_subtype),
        rng=rng,
    )
    name = f"ptb_v114_cleanonly_style_mechanism_s{args.seed}"
    selected["split"] = v114.assign_protocol_splits(selected, int(args.seed) + 77)
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
            "contract": "Formal clean-only line: BUT clean anchors/style only; medium/bad BUT rows are target distribution only, never waveform donors.",
        },
    )
    report_dir = v114.REPORT_ROOT / "v114_cleanonly_style_generator" / f"s{args.seed}" / name
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
    print(
        json.dumps(
            {
                "protocol": str(out_path),
                "report": str(report_dir),
                "rows": int(len(selected)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
