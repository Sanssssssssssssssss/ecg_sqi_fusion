from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
V114_PATH = HERE / "build_v114_but_style_residual_hybrid.py"
spec = importlib.util.spec_from_file_location("v114_hybrid", V114_PATH)
v114 = importlib.util.module_from_spec(spec)
sys.modules["v114_hybrid"] = v114
assert spec.loader is not None
spec.loader.exec_module(v114)


def lp(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def zscore_by_target(target: pd.DataFrame, cand: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    t = v114.safe_feature_matrix(target, features)
    c = v114.safe_feature_matrix(cand, features)
    med = np.nanmedian(t, axis=0)
    q25 = np.nanpercentile(t, 25, axis=0)
    q75 = np.nanpercentile(t, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-5)
    return (t - med) / scale, (c - med) / scale


def nearest_bootstrap_select(
    *,
    but_tv: pd.DataFrame,
    cand: pd.DataFrame,
    cand_x: np.ndarray,
    final_per_subtype: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    features = v114.available_features(but_tv, cand, v114.V81.MATCH_FEATURES)
    sub_t = v114.subtype_col(but_tv)
    sub_c = v114.subtype_col(cand)
    selected: list[int] = []
    audit_rows: list[dict[str, object]] = []

    for cls in v114.CLASS_ORDER:
        subtypes = sorted(sub_t.loc[but_tv["class_name"].astype(str).eq(cls)].unique())
        for subtype in subtypes:
            target = but_tv.loc[but_tv["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)].copy()
            pool = cand.loc[cand["class_name"].astype(str).eq(cls) & sub_c.eq(subtype)].copy()
            if len(target) == 0 or len(pool) == 0:
                audit_rows.append(
                    {
                        "class_name": cls,
                        "subtype": subtype,
                        "target_n": int(len(target)),
                        "candidate_n": int(len(pool)),
                        "selected_n": 0,
                        "status": "missing_target_or_pool",
                    }
                )
                continue
            n = min(int(final_per_subtype), int(len(pool)))
            target_idx = rng.choice(
                target.index.to_numpy(dtype=int),
                size=n,
                replace=len(target) < n,
            )
            target_sample = target.loc[target_idx]
            t_z, c_z = zscore_by_target(target_sample, pool, features)
            c_z = np.nan_to_num(c_z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t_z = np.nan_to_num(t_z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            k = min(max(16, int(np.ceil(n * 0.08))), min(128, len(pool)))
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
            nn.fit(c_z)
            dist, neigh = nn.kneighbors(t_z, return_distance=True)
            pool_indices = pool.index.to_numpy(dtype=int)
            used: set[int] = set()
            chosen: list[int] = []
            chosen_dist: list[float] = []
            order = np.arange(len(target_sample))
            rng.shuffle(order)
            for row_i in order:
                picked = None
                picked_d = None
                for d, local_j in zip(dist[row_i], neigh[row_i]):
                    cand_global = int(pool_indices[int(local_j)])
                    if cand_global not in used:
                        picked = cand_global
                        picked_d = float(d)
                        break
                if picked is not None:
                    used.add(picked)
                    chosen.append(picked)
                    chosen_dist.append(float(picked_d))
                if len(chosen) >= n:
                    break
            if len(chosen) < n:
                remaining = [int(i) for i in pool_indices if int(i) not in used]
                fill = rng.choice(np.asarray(remaining, dtype=int), size=n - len(chosen), replace=False)
                chosen.extend(int(x) for x in fill)
            selected.extend(chosen[:n])
            audit_rows.append(
                {
                    "class_name": cls,
                    "subtype": subtype,
                    "target_n": int(len(target)),
                    "candidate_n": int(len(pool)),
                    "selected_n": int(len(chosen[:n])),
                    "median_nn_dist": float(np.median(chosen_dist)) if chosen_dist else np.nan,
                    "p90_nn_dist": float(np.percentile(chosen_dist, 90)) if chosen_dist else np.nan,
                    "features": len(features),
                    "status": "ok",
                }
            )

    out = cand.loc[np.asarray(selected, dtype=int)].copy().reset_index(drop=True)
    out_x = cand_x[np.asarray(selected, dtype=int)].astype(np.float32)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    out["v114_selection_method"] = "d2_generated_nearest_bootstrap"
    return out, out_x, pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260792)
    parser.add_argument("--pool-per-subtype", type=int, default=4000)
    parser.add_argument("--final-per-subtype", type=int, default=1000)
    parser.add_argument("--max-donors-per-subtype", type=int, default=6000)
    parser.add_argument("--max-ptb-carriers", type=int, default=9000)
    parser.add_argument("--d2-ptb-mix-min", type=float, default=0.0)
    parser.add_argument("--d2-ptb-mix-max", type=float, default=0.002)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    print(f"{v114.now()} loading BUT reference", flush=True)
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

    anchors = v114.select_clean_anchors(but_tv, min_rows=120)
    q_features = v114.available_features(but, but, v114.QUALITY_DELTA_FEATURES)
    baselines, global_base = v114.subject_baselines(anchors, q_features)
    bank = v114.build_residual_bank(
        but_tv,
        but_x,
        anchors,
        rng,
        max_donors_per_subtype=int(args.max_donors_per_subtype),
    )
    counts = v114.target_counts(but_tv, int(args.pool_per_subtype), mode="balanced")
    print(f"{v114.now()} building D2-only generated pool rows~{sum(counts.values())}", flush=True)
    d2_frame0, d2_x0 = v114.build_candidate_line(
        source_line="D2_ptb_but_style_residual",
        counts=counts,
        but_x=but_x,
        ptb_frame=ptb,
        ptb_x=ptb_x,
        anchors=anchors,
        bank=bank,
        rng=rng,
        d2_target_dominant=True,
        d2_ptb_mix_min=float(args.d2_ptb_mix_min),
        d2_ptb_mix_max=float(args.d2_ptb_mix_max),
    )
    print(f"{v114.now()} recomputing D2 pool features rows={len(d2_frame0)}", flush=True)
    d2, d2_x = v114.normalize_frame(d2_frame0, d2_x0, "D2 generated nearest-match pool")
    print(f"{v114.now()} selecting generated nearest/bootstrap subset", flush=True)
    selected, selected_x, selection_audit = nearest_bootstrap_select(
        but_tv=but_tv,
        cand=d2,
        cand_x=d2_x,
        final_per_subtype=int(args.final_per_subtype),
        rng=rng,
    )
    name = f"ptb_v114_d2_generated_nearestmatch_s{args.seed}"
    selected["split"] = v114.assign_protocol_splits(selected, int(args.seed) + 19)
    protocol_path = v114.PROTOCOL_ROOT / name
    summary = {
        "protocol": name,
        "line": "D2_generated_nearest_bootstrap",
        "seed": int(args.seed),
        "rows": int(len(selected)),
        "pool_per_subtype": int(args.pool_per_subtype),
        "final_per_subtype": int(args.final_per_subtype),
        "d2_ptb_mix_min": float(args.d2_ptb_mix_min),
        "d2_ptb_mix_max": float(args.d2_ptb_mix_max),
        "contract": "Generated-only D2 target-dominant pool, selected by nearest/bootstrap matching to BUT train+val subtype distributions; BUT test excluded.",
    }
    v114.save_protocol(protocol_path, selected, selected_x, summary)
    report_dir = v114.REPORT_ROOT / "v114_but_style_residual_hybrid" / f"s{args.seed}" / name
    report_dir.mkdir(parents=True, exist_ok=True)
    selection_audit.to_csv(lp(report_dir / f"{name}_selection_audit.csv"), index=False)
    if not args.skip_audit:
        v114.audit_protocol(
            but=but,
            but_x=but_x,
            synth=selected,
            synth_x=selected_x,
            tag=name,
            report_dir=report_dir,
            seed=int(args.seed) + 31,
            baselines=baselines,
            global_base=global_base,
        )
    out = {
        "protocol_path": str(protocol_path),
        "report_dir": str(report_dir),
        "selection_audit": str(report_dir / f"{name}_selection_audit.csv"),
        "rows": int(len(selected)),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
