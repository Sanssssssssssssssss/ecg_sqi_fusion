from __future__ import annotations

import argparse
import importlib.util
import json
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


def choose_generated_nearest(
    target: pd.DataFrame,
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    features: list[str],
) -> np.ndarray:
    if n <= 0 or len(pool) == 0:
        return np.zeros(0, dtype=int)
    n = min(int(n), int(len(pool)))
    t = v114.safe_feature_matrix(target, features)
    c = v114.safe_feature_matrix(pool, features)
    med = np.nanmedian(t, axis=0)
    q25 = np.nanpercentile(t, 25, axis=0)
    q75 = np.nanpercentile(t, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-5)
    t = np.nan_to_num((t - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.nan_to_num((c - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    # Greedy nearest-to-random-target bootstrap without sklearn overhead.
    target_ids = rng.choice(np.arange(len(t)), size=n, replace=len(t) < n)
    order = np.arange(len(target_ids))
    rng.shuffle(order)
    remaining = set(range(len(pool)))
    chosen_local: list[int] = []
    for oi in order:
        if not remaining:
            break
        dz = c[list(remaining)] - t[int(target_ids[int(oi)])]
        dist = np.sum(dz * dz, axis=1)
        rem_list = list(remaining)
        pick = int(rem_list[int(np.argmin(dist))])
        remaining.remove(pick)
        chosen_local.append(pick)
    if len(chosen_local) < n and remaining:
        fill = rng.choice(np.asarray(list(remaining), dtype=int), size=n - len(chosen_local), replace=False)
        chosen_local.extend(int(x) for x in fill)
    return pool.index.to_numpy(dtype=int)[np.asarray(chosen_local[:n], dtype=int)]


def build_mix(
    *,
    but_tv: pd.DataFrame,
    native: pd.DataFrame,
    native_x: np.ndarray,
    d2: pd.DataFrame,
    d2_x: np.ndarray,
    frac: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    features = v114.available_features(but_tv, d2, v114.V81.MATCH_FEATURES)
    sub_t = v114.subtype_col(but_tv)
    sub_n = v114.subtype_col(native)
    sub_d = v114.subtype_col(d2)
    rows: list[pd.DataFrame] = []
    arrays: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    for cls in v114.CLASS_ORDER:
        for subtype in sorted(sub_t.loc[but_tv["class_name"].astype(str).eq(cls)].unique()):
            target = but_tv.loc[but_tv["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)].copy()
            quota = int(len(target))
            if quota <= 0:
                continue
            d2_n = int(round(quota * float(frac)))
            native_n = max(0, quota - d2_n)
            native_pool = native.loc[native["class_name"].astype(str).eq(cls) & sub_n.eq(subtype)]
            d2_pool = d2.loc[d2["class_name"].astype(str).eq(cls) & sub_d.eq(subtype)]
            if len(native_pool) == 0:
                d2_n = quota
                native_n = 0
            if len(d2_pool) == 0:
                native_n = quota
                d2_n = 0
            native_idx = np.zeros(0, dtype=int)
            if native_n > 0:
                native_idx = rng.choice(
                    native_pool.index.to_numpy(dtype=int),
                    size=native_n,
                    replace=len(native_pool) < native_n,
                )
                rows.append(native.loc[native_idx])
                arrays.append(native_x[native_idx])
            d2_idx = choose_generated_nearest(target, d2_pool, d2_n, rng, features)
            if len(d2_idx) > 0:
                rows.append(d2.loc[d2_idx])
                arrays.append(d2_x[d2_idx])
            audit.append(
                {
                    "class_name": cls,
                    "subtype": subtype,
                    "quota": quota,
                    "native_n": int(len(native_idx)),
                    "d2_n": int(len(d2_idx)),
                    "d2_frac_actual": float(len(d2_idx) / max(quota, 1)),
                }
            )
    out = pd.concat(rows, ignore_index=True)
    out_x = np.vstack(arrays).astype(np.float32)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    out["v114_mix_requested_d2_frac"] = float(frac)
    out["v114_selection_method"] = "native_plus_d2_ratio_nearest"
    return out, out_x, pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260794)
    parser.add_argument("--base-seed", type=int, default=20260793)
    parser.add_argument("--fractions", default="0.10,0.20,0.30,0.35")
    args = parser.parse_args()
    fractions = [float(x.strip()) for x in str(args.fractions).split(",") if x.strip()]
    print(f"{v114.now()} loading BUT/native/D2 protocols", flush=True)
    but0, butx0 = v114.V81.load_protocol(v114.DEFAULT_BUT_PROTOCOL)
    but, but_x = v114.normalize_frame(but0, butx0, "BUT reference")
    but_tv = v114.split_train_val(but)
    native_name = f"hybrid_v114_bayesian_match_natural_prior_eval_s{args.base_seed}"
    d2_name = f"ptb_v114_but_style_residual_s{args.base_seed}"
    native0, native_x0 = v114.V81.load_protocol(v114.PROTOCOL_ROOT / native_name)
    d20, d2_x0 = v114.V81.load_protocol(v114.PROTOCOL_ROOT / d2_name)
    native = native0.copy()
    d2 = d20.copy()
    native["_row_pos"] = np.arange(len(native), dtype=int)
    d2["_row_pos"] = np.arange(len(d2), dtype=int)
    anchors = v114.select_clean_anchors(but_tv, min_rows=120)
    q_features = v114.available_features(but, but, v114.QUALITY_DELTA_FEATURES)
    baselines, global_base = v114.subject_baselines(anchors, q_features)
    results: list[dict[str, object]] = []
    for i, frac in enumerate(fractions):
        seed = int(args.seed) + i
        name = f"hybrid_v114_ptbfrac{int(round(frac*100)):02d}_natural_prior_s{seed}"
        print(f"{v114.now()} building {name}", flush=True)
        out, out_x, audit = build_mix(
            but_tv=but_tv,
            native=native,
            native_x=native_x0,
            d2=d2,
            d2_x=d2_x0,
            frac=frac,
            seed=seed,
        )
        out["split"] = v114.assign_protocol_splits(out, seed + 99)
        out_path = v114.PROTOCOL_ROOT / name
        v114.save_protocol(
            out_path,
            out,
            out_x,
            {
                "protocol": name,
                "seed": seed,
                "base_seed": int(args.base_seed),
                "requested_d2_frac": frac,
                "rows": int(len(out)),
                "contract": "BUT-native natural-prior support mixed with nearest D2 generated rows to target a controlled MMD around 0.06.",
            },
        )
        report_dir = v114.REPORT_ROOT / "v114_but_style_residual_hybrid" / f"s{seed}" / name
        report_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(lp(report_dir / f"{name}_mix_audit.csv"), index=False)
        v114.audit_protocol(
            but=but,
            but_x=but_x,
            synth=out,
            synth_x=out_x,
            tag=name,
            report_dir=report_dir,
            seed=seed + 113,
            baselines=baselines,
            global_base=global_base,
        )
        glob = pd.read_csv(lp(report_dir / f"{name}_global_distribution_metrics.csv"))
        metric = pd.read_csv(lp(report_dir / f"{name}_distribution_metrics.csv"))
        all_mmd = float(glob.loc[glob["scope"].eq("all_labels"), "rbf_mmd"].iloc[0])
        results.append(
            {
                "name": name,
                "requested_d2_frac": frac,
                "all_labels_mmd": all_mmd,
                "max_subtype_mmd": float(metric["rbf_mmd"].max()),
                "count_subtype_gt_0p01": int((metric["rbf_mmd"] > 0.01).sum()),
                "protocol_path": str(out_path),
                "report_dir": str(report_dir),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    summary = pd.DataFrame(results)
    out_summary = v114.REPORT_ROOT / "v114_but_style_residual_hybrid" / f"s{args.seed}" / "ptb_ratio_sweep_summary.csv"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(lp(out_summary), index=False)
    print(json.dumps({"summary": str(out_summary), "results": results}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
