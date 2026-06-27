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

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def lp(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def resolve_device(requested: str) -> str:
    if requested == "cpu" or torch is None:
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rff(x: np.ndarray, ref: np.ndarray, dim: int, seed: int, device: str):
    if torch is None:
        raise RuntimeError("torch is required")
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    ref_t = torch.as_tensor(np.asarray(ref, dtype=np.float32), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if ref_t.shape[0] > 1200:
        idx = torch.randperm(ref_t.shape[0], device=device, generator=gen)[:1200]
        ref_s = ref_t[idx]
    else:
        ref_s = ref_t
    if ref_s.shape[0] > 2 and ref_s.shape[1] > 0:
        d = torch.cdist(ref_s, ref_s).pow(2)
        vals = d[d > 0]
        med = torch.median(vals) if vals.numel() else torch.tensor(1.0, device=device)
    else:
        med = torch.tensor(1.0, device=device)
    sigma = torch.sqrt(torch.clamp(med, min=1e-6))
    bands = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device)
    per = max(8, int(math.ceil(dim / len(bands))))
    parts = []
    for band in bands:
        w = torch.randn((x_t.shape[1], per), device=device, generator=gen) / torch.clamp(sigma * band, min=1e-5)
        b = torch.rand((per,), device=device, generator=gen) * (2.0 * math.pi)
        parts.append(math.sqrt(2.0 / (per * len(bands))) * torch.cos(x_t @ w + b))
    return torch.cat(parts, dim=1)[:, :dim].contiguous()


def robust_z(target: pd.DataFrame, frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    t = v114.safe_feature_matrix(target, features)
    x = v114.safe_feature_matrix(frame, features)
    center, scale = v114.V81.robust_fit(t)
    return v114.V81.robust_z(t, center, scale).astype(np.float32), v114.V81.robust_z(x, center, scale).astype(np.float32)


def initial_pairing(target_z: np.ndarray, native_z: np.ndarray, d2_z: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    # Pick D2 candidates close to the target support, then remove the most similar
    # native rows so the selected mean moves as little as possible.
    t_idx = rng.choice(np.arange(len(target_z)), size=min(max(k * 2, k), len(target_z)), replace=len(target_z) < max(k * 2, k))
    target_sample = target_z[t_idx]
    d2_batch = 512
    min_dist = np.full(len(d2_z), np.inf, dtype=np.float32)
    for start in range(0, len(d2_z), d2_batch):
        dz = d2_z[start : start + d2_batch, None, :] - target_sample[None, :, :]
        min_dist[start : start + d2_batch] = np.sqrt(np.min(np.sum(dz * dz, axis=2), axis=1))
    d2_sel = np.argsort(min_dist)[:k].astype(int)
    removed: list[int] = []
    used: set[int] = set()
    for j in d2_sel:
        dz = native_z - d2_z[int(j)]
        dist = np.sum(dz * dz, axis=1)
        for cand in np.argsort(dist)[: min(64, len(dist))]:
            ci = int(cand)
            if ci not in used:
                used.add(ci)
                removed.append(ci)
                break
    if len(removed) < k:
        rest = [i for i in range(len(native_z)) if i not in used]
        fill = rng.choice(np.asarray(rest, dtype=int), size=k - len(removed), replace=False)
        removed.extend(int(x) for x in fill)
    return d2_sel[:k].astype(int), np.asarray(removed[:k], dtype=int)


def optimize_subtype(
    *,
    target: pd.DataFrame,
    native: pd.DataFrame,
    d2: pd.DataFrame,
    native_x: np.ndarray,
    d2_x: np.ndarray,
    ptb_min_frac: float,
    features: list[str],
    steps: int,
    batch: int,
    rff_dim: int,
    device: str,
    seed: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object], pd.DataFrame]:
    quota = int(len(target))
    if quota == 0:
        return pd.DataFrame(), np.zeros((0, native_x.shape[-1]), dtype=np.float32), {}, pd.DataFrame()
    native = native.copy().reset_index(drop=True)
    d2 = d2.copy().reset_index(drop=True)
    native_x = np.asarray(native_x, dtype=np.float32)
    d2_x = np.asarray(d2_x, dtype=np.float32)
    if len(native) < quota:
        take = rng.choice(np.arange(len(native)), size=quota, replace=True)
        native = native.iloc[take].reset_index(drop=True)
        native_x = native_x[take]
    else:
        native = native.iloc[:quota].reset_index(drop=True)
        native_x = native_x[:quota]
    k = int(math.ceil(float(ptb_min_frac) * quota))
    k = min(k, quota, len(d2))
    if k <= 0:
        native["v114_constrained_role"] = "native"
        return native, native_x, {"quota": quota, "ptb_selected": 0, "ptb_frac_actual": 0.0, "energy": 0.0}, pd.DataFrame()
    target_z, native_z = robust_z(target, native, features)
    _, d2_z = robust_z(target, d2, features)
    all_z = np.vstack([target_z, native_z, d2_z])
    phi = rff(all_z, target_z, int(rff_dim), int(seed), device)
    target_phi = phi[: len(target_z)]
    native_phi = phi[len(target_z) : len(target_z) + len(native_z)]
    d2_phi = phi[len(target_z) + len(native_z) :]
    target_mu = target_phi.mean(dim=0)
    native_sum = native_phi.sum(dim=0)
    d2_sel, removed = initial_pairing(target_z, native_z, d2_z, k, rng)
    d2_sel_t = torch.as_tensor(d2_sel, device=device, dtype=torch.long)
    removed_t = torch.as_tensor(removed, device=device, dtype=torch.long)
    selected_mu = (native_sum - native_phi[removed_t].sum(dim=0) + d2_phi[d2_sel_t].sum(dim=0)) / float(quota)
    energy_t = (selected_mu - target_mu).pow(2).mean()
    energy = float(energy_t.detach().cpu())
    best_energy = energy
    best_d2 = d2_sel.copy()
    best_removed = removed.copy()
    trace: list[dict[str, object]] = []
    selected_d2_mask = np.zeros(len(d2), dtype=bool)
    selected_d2_mask[d2_sel] = True
    removed_mask = np.zeros(len(native), dtype=bool)
    removed_mask[removed] = True
    batch = max(1, int(batch))
    for step in range(0, int(steps), batch):
        bsz = min(batch, int(steps) - step)
        available_d2 = np.where(~selected_d2_mask)[0]
        available_native = np.where(~removed_mask)[0]
        if len(available_d2) == 0 and len(available_native) == 0:
            break
        proposals = []
        # D2 replacement proposals.
        if len(available_d2) > 0:
            out_pos = rng.integers(0, k, size=bsz)
            in_d2 = rng.choice(available_d2, size=bsz, replace=True)
            old_d2 = d2_sel[out_pos]
            mu_prop = selected_mu[None, :] + (d2_phi[torch.as_tensor(in_d2, device=device)] - d2_phi[torch.as_tensor(old_d2, device=device)]) / float(quota)
            e = (mu_prop - target_mu[None, :]).pow(2).mean(dim=1).detach().cpu().numpy()
            for ii in np.argsort(e)[: min(8, len(e))]:
                proposals.append(("d2", float(e[int(ii)]), int(out_pos[int(ii)]), int(in_d2[int(ii)]), int(old_d2[int(ii)])))
        # Native removed-row replacement proposals.
        if len(available_native) > 0:
            out_pos = rng.integers(0, k, size=bsz)
            in_nat = rng.choice(available_native, size=bsz, replace=True)
            old_nat = removed[out_pos]
            # Removing a new native row and restoring the old removed row.
            mu_prop = selected_mu[None, :] + (native_phi[torch.as_tensor(old_nat, device=device)] - native_phi[torch.as_tensor(in_nat, device=device)]) / float(quota)
            e = (mu_prop - target_mu[None, :]).pow(2).mean(dim=1).detach().cpu().numpy()
            for ii in np.argsort(e)[: min(8, len(e))]:
                proposals.append(("native", float(e[int(ii)]), int(out_pos[int(ii)]), int(in_nat[int(ii)]), int(old_nat[int(ii)])))
        proposals.sort(key=lambda x: x[1])
        accepted = 0
        for kind, e_val, pos, new_i, old_i in proposals:
            if e_val >= energy:
                continue
            if kind == "d2":
                if selected_d2_mask[new_i] or not selected_d2_mask[old_i]:
                    continue
                selected_d2_mask[old_i] = False
                selected_d2_mask[new_i] = True
                d2_sel[pos] = new_i
            else:
                if removed_mask[new_i] or not removed_mask[old_i]:
                    continue
                removed_mask[old_i] = False
                removed_mask[new_i] = True
                removed[pos] = new_i
            d2_sel_t = torch.as_tensor(d2_sel, device=device, dtype=torch.long)
            removed_t = torch.as_tensor(removed, device=device, dtype=torch.long)
            selected_mu = (native_sum - native_phi[removed_t].sum(dim=0) + d2_phi[d2_sel_t].sum(dim=0)) / float(quota)
            energy = float((selected_mu - target_mu).pow(2).mean().detach().cpu())
            accepted += 1
            if energy < best_energy:
                best_energy = energy
                best_d2 = d2_sel.copy()
                best_removed = removed.copy()
        if step in {0, int(steps) // 4, int(steps) // 2, (3 * int(steps)) // 4} or step + batch >= int(steps):
            trace.append({"step": int(step), "energy": float(energy), "best_energy": float(best_energy), "accepted": int(accepted)})
    keep_native = np.ones(len(native), dtype=bool)
    keep_native[best_removed] = False
    out_native = native.loc[keep_native].copy()
    out_d2 = d2.iloc[best_d2].copy()
    out_native["v114_constrained_role"] = "native"
    out_d2["v114_constrained_role"] = "ptb_generated"
    out = pd.concat([out_native, out_d2], ignore_index=True)
    out_x = np.vstack([native_x[keep_native], d2_x[best_d2]]).astype(np.float32)
    info = {
        "quota": quota,
        "ptb_selected": int(len(best_d2)),
        "ptb_frac_actual": float(len(best_d2) / max(quota, 1)),
        "initial_energy": float(trace[0]["energy"]) if trace else float("nan"),
        "best_energy": float(best_energy),
        "ptb_cap_frac": float(min(len(d2), quota) / max(quota, 1)),
    }
    return out, out_x, info, pd.DataFrame(trace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--base-seed", type=int, default=20260793)
    parser.add_argument("--ptb-min-frac", type=float, default=0.20)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--rff-dim", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--native-protocol", default="")
    parser.add_argument("--d2-protocol", default="")
    args = parser.parse_args()
    rng = np.random.default_rng(int(args.seed))
    device = resolve_device(str(args.device))
    print(f"{v114.now()} constrained MMD KMM device={device} ptb_min={args.ptb_min_frac}", flush=True)
    but0, butx0 = v114.V81.load_protocol(v114.DEFAULT_BUT_PROTOCOL)
    but, but_x = v114.normalize_frame(but0, butx0, "BUT reference")
    but_tv = v114.split_train_val(but)
    native_name = args.native_protocol or f"hybrid_v114_bayesian_match_natural_prior_eval_s{args.base_seed}"
    d2_name = args.d2_protocol or f"ptb_v114_but_style_residual_s{args.base_seed}"
    native0, native_x0 = v114.V81.load_protocol(v114.PROTOCOL_ROOT / native_name)
    d20, d2_x0 = v114.V81.load_protocol(v114.PROTOCOL_ROOT / d2_name)
    native = native0.copy()
    d2 = d20.copy()
    native["_row_pos"] = np.arange(len(native), dtype=int)
    d2["_row_pos"] = np.arange(len(d2), dtype=int)
    features = v114.available_features(but_tv, pd.concat([native, d2], ignore_index=True), v114.V81.MATCH_FEATURES)
    sub_t = v114.subtype_col(but_tv)
    sub_n = v114.subtype_col(native)
    sub_d = v114.subtype_col(d2)
    frames: list[pd.DataFrame] = []
    arrays: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    trace_frames: list[pd.DataFrame] = []
    for si, cls in enumerate(v114.CLASS_ORDER):
        for subtype in sorted(sub_t.loc[but_tv["class_name"].astype(str).eq(cls)].unique()):
            target = but_tv.loc[but_tv["class_name"].astype(str).eq(cls) & sub_t.eq(subtype)].copy()
            native_pool = native.loc[native["class_name"].astype(str).eq(cls) & sub_n.eq(subtype)].copy()
            d2_pool = d2.loc[d2["class_name"].astype(str).eq(cls) & sub_d.eq(subtype)].copy()
            if len(target) == 0 or len(native_pool) == 0:
                continue
            print(f"{v114.now()} optimize {cls}/{subtype}: target={len(target)} native={len(native_pool)} d2={len(d2_pool)}", flush=True)
            out, out_x, info, trace = optimize_subtype(
                target=target,
                native=native_pool,
                d2=d2_pool,
                native_x=native_x0[native_pool.index.to_numpy(dtype=int)],
                d2_x=d2_x0[d2_pool.index.to_numpy(dtype=int)],
                ptb_min_frac=float(args.ptb_min_frac),
                features=features,
                steps=int(args.steps),
                batch=int(args.batch),
                rff_dim=int(args.rff_dim),
                device=device,
                seed=int(args.seed) + si * 1000 + len(audit_rows),
                rng=rng,
            )
            info.update({"class_name": cls, "subtype": subtype, "target_n": int(len(target)), "native_pool_n": int(len(native_pool)), "d2_pool_n": int(len(d2_pool))})
            audit_rows.append(info)
            if not trace.empty:
                trace["class_name"] = cls
                trace["subtype"] = subtype
                trace_frames.append(trace)
            frames.append(out)
            arrays.append(out_x)
    result = pd.concat(frames, ignore_index=True)
    result_x = np.vstack(arrays).astype(np.float32)
    result["idx"] = np.arange(len(result), dtype=int)
    result["_row_pos"] = np.arange(len(result), dtype=int)
    result["split"] = v114.assign_protocol_splits(result, int(args.seed) + 77)
    frac_tag = int(round(float(args.ptb_min_frac) * 100))
    name = f"hybrid_v114_constrained_mmd_ptbmin{frac_tag:02d}_s{args.seed}"
    out_path = v114.PROTOCOL_ROOT / name
    v114.save_protocol(
        out_path,
        result,
        result_x,
        {
            "protocol": name,
            "seed": int(args.seed),
            "base_seed": int(args.base_seed),
            "ptb_min_frac": float(args.ptb_min_frac),
            "rows": int(len(result)),
            "objective": "minimize empirical v110 RFF-MMD subject to PTB/generated minimum fraction per subtype where candidate support allows",
            "native_protocol": native_name,
            "d2_protocol": d2_name,
        },
    )
    report_dir = v114.REPORT_ROOT / "v114_constrained_mmd_kmm" / f"s{args.seed}" / name
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(lp(report_dir / f"{name}_constraint_audit.csv"), index=False)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(lp(report_dir / f"{name}_mmd_trace.csv"), index=False)
    anchors = v114.select_clean_anchors(but_tv, min_rows=120)
    q_features = v114.available_features(but, but, v114.QUALITY_DELTA_FEATURES)
    baselines, global_base = v114.subject_baselines(anchors, q_features)
    v114.audit_protocol(
        but=but,
        but_x=but_x,
        synth=result,
        synth_x=result_x,
        tag=name,
        report_dir=report_dir,
        seed=int(args.seed) + 99,
        baselines=baselines,
        global_base=global_base,
    )
    glob = pd.read_csv(lp(report_dir / f"{name}_global_distribution_metrics.csv"))
    metric = pd.read_csv(lp(report_dir / f"{name}_distribution_metrics.csv"))
    print(
        json.dumps(
            {
                "protocol": str(out_path),
                "report": str(report_dir),
                "ptb_frac_actual": float(audit["ptb_selected"].sum() / audit["quota"].sum()),
                "all_labels_mmd": float(glob.loc[glob["scope"].eq("all_labels"), "rbf_mmd"].iloc[0]),
                "max_subtype_mmd": float(metric["rbf_mmd"].max()),
                "subtype_gt_0p01": int((metric["rbf_mmd"] > 0.01).sum()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
