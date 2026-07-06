#!/usr/bin/env python
"""V116 native-budgeted support-aware semi-synthetic repair.

This runner makes the v115 conclusion explicit:

* clean-only is a negative control, not the main line;
* BUT train-only native/residual support may be used, but is budgeted;
* PTB carriers provide physiology diversity;
* selection targets class-conditional distribution preservation under class
  balancing, without using BUT test rows for generation or selection.

Implementation intentionally lives in the external experiment tree and reuses
only v110/v114/v115 audit/generation utilities.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent


def import_local(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v114 = import_local("v114_hybrid", HERE / "build_v114_but_style_residual_hybrid.py")
v115 = import_local("v115_support", HERE / "run_v115_support_aware_distribution_repair.py")


REPORT_ROOT = v114.REPORT_ROOT / "v116_native_budget_repair"
PROTOCOL_ROOT = v114.PROTOCOL_ROOT

RESIDUAL_FAMILIES = [
    "beat_template_residual",
    "band_limited_residual",
    "detector_instability_residual",
    "flatline_clipping_reset_residual",
    "pseudo_qrs_periodic_residual",
]


def lp(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def now() -> str:
    return v114.now()


def write_md(path: Path, title: str, sections: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")


def sample_frame(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= int(n):
        return frame.copy()
    return frame.iloc[rng.choice(np.arange(len(frame)), size=int(n), replace=False)].copy()


def but_train_frame(but: pd.DataFrame) -> pd.DataFrame:
    out = but.loc[but["split"].astype(str).eq("train")].copy()
    if len(out) == 0:
        out = but.loc[but["split"].astype(str).isin(["train", "val"])].copy()
    return out


def class_metric(tag: str, scope: str, a: pd.DataFrame, b: pd.DataFrame, features: list[str], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    row = v115.metric_row(tag, scope, a, b, features, rng)
    row["class_name"] = scope.replace("class_", "") if scope.startswith("class_") else scope
    return row


def native_split_floor(
    but_train: pd.DataFrame,
    *,
    features: list[str],
    draws: int,
    seed: int,
    max_rows_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for draw in range(int(draws)):
        a_parts: list[pd.DataFrame] = []
        b_parts: list[pd.DataFrame] = []
        for cls in v114.CLASS_ORDER:
            cls_rows = but_train.loc[but_train["class_name"].astype(str).eq(cls)].copy()
            cls_rows = sample_frame(cls_rows, int(max_rows_per_class) * 2, rng)
            idx = np.arange(len(cls_rows))
            rng.shuffle(idx)
            mid = max(2, len(idx) // 2)
            a = cls_rows.iloc[idx[:mid]].copy()
            b = cls_rows.iloc[idx[mid:]].copy()
            if len(a) >= 5 and len(b) >= 5:
                r = class_metric("native_split_floor", f"class_{cls}", a, b, features, int(seed) + draw * 17)
                r["draw"] = int(draw)
                rows.append(r)
                a_parts.append(a)
                b_parts.append(b)
        if a_parts and b_parts:
            aa = pd.concat(a_parts, ignore_index=True)
            bb = pd.concat(b_parts, ignore_index=True)
            r = class_metric("native_split_floor", "all_labels", aa, bb, features, int(seed) + draw * 17 + 3)
            r["draw"] = int(draw)
            rows.append(r)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby("scope", as_index=False)
        .agg(
            draws=("draw", "count"),
            rbf_mmd_median=("rbf_mmd", "median"),
            rbf_mmd_q90=("rbf_mmd", lambda s: float(np.percentile(s, 90))),
            rbf_mmd_q95=("rbf_mmd", lambda s: float(np.percentile(s, 95))),
            pca_overlap_median=("pca_density_overlap", "median"),
            pca_overlap_q05=("pca_density_overlap", lambda s: float(np.percentile(s, 5))),
            sym_domain_auc_median=("sym_domain_auc", "median"),
            sym_domain_auc_q95=("sym_domain_auc", lambda s: float(np.percentile(s, 95))),
        )
        .sort_values("scope")
    )
    return detail, summary


def quota_counts_by_subtype(but_train: pd.DataFrame, per_subtype: int) -> dict[tuple[str, str], int]:
    sub = v114.subtype_col(but_train)
    counts: dict[tuple[str, str], int] = {}
    for cls in v114.CLASS_ORDER:
        vals = sorted(sub.loc[but_train["class_name"].astype(str).eq(cls)].dropna().astype(str).unique())
        for subtype in vals:
            if subtype:
                counts[(str(cls), str(subtype))] = int(per_subtype)
    return counts


def build_native_pool(but_train: pd.DataFrame, but_x: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    frame = but_train.copy().reset_index(drop=True)
    pos = frame["_row_pos"].astype(int).to_numpy()
    x = but_x[pos].astype(np.float32)
    subtype = v114.subtype_col(frame)
    frame["transport_subtype"] = subtype.astype(str)
    frame["display_subtype"] = subtype.astype(str)
    frame["v116_candidate_type"] = "but_train_native_anchor"
    frame["v116_native_replay"] = 1
    frame["v116_residual_transfer"] = 0
    frame["v116_ptb_generated"] = 0
    frame["v116_native_donor_id"] = frame.get("source_idx", pd.Series(frame.index, index=frame.index)).astype(str)
    frame["v116_residual_donor_id"] = ""
    frame["v116_style_donor_id"] = frame.get("record_id", pd.Series("", index=frame.index)).astype(str)
    frame["v116_seed"] = int(seed)
    frame["idx"] = np.arange(len(frame), dtype=int)
    frame["_row_pos"] = np.arange(len(frame), dtype=int)
    frame["split"] = "train"
    return frame, x


def morph_native_signal(
    x: np.ndarray,
    cls: str,
    subtype: str,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    strength = max(0.0, float(strength))
    ref = np.asarray(x, dtype=np.float32).reshape(-1)
    y = ref.copy()
    med, scale = v114.robust_stats(ref)
    scale = max(float(scale), 1e-5)
    shift_radius = max(1, int(round(5 * strength)))
    shift = int(rng.integers(-shift_radius, shift_radius + 1))
    gain = float(rng.uniform(1.0 - 0.045 * strength, 1.0 + 0.045 * strength))
    baseline = float(rng.uniform(-0.025 * strength, 0.025 * strength))
    noise = float(rng.uniform(0.001 * strength, 0.012 * strength))
    if shift:
        y = v114.shift_signal(y, shift)
    y = (med + gain * (y - med)).astype(np.float32)
    t = np.linspace(-0.5, 0.5, len(y), dtype=np.float32)
    y = y + baseline * scale * t
    if str(cls) == "medium" and rng.random() < 0.35 * strength:
        y = y + float(rng.uniform(0.004, 0.020)) * strength * scale * v114.fft_band_noise(len(y), rng, 0.35, 8.0)
    y = y + rng.normal(0.0, noise * scale, size=len(y)).astype(np.float32)
    y = v114.soft_clip_like_reference(y, ref, widen=1.28)
    return y.astype(np.float32), {
        "v116_native_morph_shift": shift,
        "v116_native_morph_gain": gain,
        "v116_native_morph_baseline": baseline,
        "v116_native_morph_noise": noise,
        "v116_native_morph_strength": strength,
    }


def build_native_morph_pool(
    but_train: pd.DataFrame,
    but_x: np.ndarray,
    seed: int,
    copies_per_row: int,
    strength: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    rng = np.random.default_rng(int(seed) + 616)
    src = but_train.loc[but_train["class_name"].astype(str).isin(["medium", "bad"])].copy().reset_index(drop=True)
    subtype = v114.subtype_col(src).astype(str)
    for i, row in src.iterrows():
        cls = str(row["class_name"])
        st = str(subtype.iloc[i])
        pos = int(row["_row_pos"])
        donor_id = str(row.get("source_idx", row.get("idx", i)))
        for copy_i in range(int(copies_per_row)):
            sig, meta = morph_native_signal(but_x[pos], cls, st, rng, strength=float(strength))
            rec = row.to_dict()
            rec["source_idx"] = f"v116_native_morph_{donor_id}_{copy_i}"
            rec["record_id"] = f"v116_native_morph_{row.get('record_id', donor_id)}_{copy_i}"
            rec["transport_subtype"] = st
            rec["display_subtype"] = st
            rec["v116_candidate_type"] = "but_native_morph"
            rec["v116_native_replay"] = 0
            rec["v116_native_morph"] = 1
            rec["v116_generated"] = 1
            rec["v116_residual_transfer"] = 0
            rec["v116_ptb_generated"] = 0
            rec["v116_native_donor_id"] = donor_id
            rec["v116_native_morph_copy"] = int(copy_i)
            rec["v116_seed"] = int(seed)
            rec.update(meta)
            rows.append(rec)
            signals.append(sig)
    if not rows:
        return pd.DataFrame(), np.zeros((0, but_x.shape[-1]), dtype=np.float32)
    frame = pd.DataFrame(rows)
    x = np.vstack(signals).astype(np.float32)
    frame, x = v114.normalize_frame(frame, x, "v116 BUT native-morph candidates")
    frame["v116_candidate_type"] = "but_native_morph"
    frame["v116_native_replay"] = 0
    frame["v116_native_morph"] = 1
    frame["v116_generated"] = 1
    frame["v116_residual_transfer"] = 0
    frame["v116_ptb_generated"] = 0
    frame["v116_seed"] = int(seed)
    return frame, x


def residual_family(cls: str, subtype: str) -> str:
    st = str(subtype).lower()
    if "contact" in st or "flatline" in st or "reset" in st:
        return "flatline_clipping_reset_residual"
    if "detector" in st or "template" in st or "lowqrs" in st or "low_qrs" in st:
        return "detector_instability_residual"
    if "baseline" in st or "lowfreq" in st:
        return "band_limited_residual"
    if "dense" in st or "periodic" in st or "right" in st:
        return "pseudo_qrs_periodic_residual"
    if cls == "bad":
        return "pseudo_qrs_periodic_residual"
    return "beat_template_residual"


def mark_residual_pool(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = frame.copy()
    subtype = v114.subtype_col(out)
    out["v116_candidate_type"] = "ptb_carrier_but_residual_transfer"
    out["v116_native_replay"] = 0
    out["v116_residual_transfer"] = 1
    out["v116_ptb_generated"] = 0
    out["v116_residual_family"] = [
        residual_family(str(c), str(s)) for c, s in zip(out["class_name"].astype(str), subtype.astype(str))
    ]
    out["v116_residual_donor_id"] = out.get("v114_donor_record_id", out.get("source_idx", pd.Series("", index=out.index))).astype(str)
    out["v116_style_donor_id"] = out.get("v114_style_subject_id", pd.Series("", index=out.index)).astype(str)
    out["v116_ptb_carrier_id"] = out.get("ptbxl_source_idx", out.get("source_idx", pd.Series("", index=out.index))).astype(str)
    out["v116_seed"] = int(seed)
    return out


def mark_clean_pool(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = frame.copy()
    out["v116_candidate_type"] = "ptb_clean_style_generator"
    out["v116_native_replay"] = 0
    out["v116_residual_transfer"] = 0
    out["v116_ptb_generated"] = 1
    out["v116_residual_family"] = "clean_style_generator"
    out["v116_residual_donor_id"] = ""
    out["v116_style_donor_id"] = out.get("v115_style_source_idx", out.get("record_id", pd.Series("", index=out.index))).astype(str)
    out["v116_ptb_carrier_id"] = out.get("ptbxl_source_idx", out.get("source_idx", pd.Series("", index=out.index))).astype(str)
    out["v116_seed"] = int(seed)
    return out


def build_candidate_pools(
    *,
    but_train: pd.DataFrame,
    but_x: np.ndarray,
    ptb: pd.DataFrame,
    ptb_x: np.ndarray,
    seed: int,
    clean_candidates_per_class: int,
    residual_per_subtype: int,
    max_donors_per_subtype: int,
    native_morph_copies: int,
    native_morph_strength: float,
    d2_target_dominant: bool,
    d2_ptb_mix_min: float,
    d2_ptb_mix_max: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    anchors = v114.select_clean_anchors(but_train, min_rows=80)
    bank = v114.build_residual_bank(
        but_train,
        but_x,
        anchors,
        rng,
        max_donors_per_subtype=int(max_donors_per_subtype),
    )
    native, native_x = build_native_pool(but_train, but_x, int(seed))
    if int(native_morph_copies) > 0:
        native_morph, native_morph_x = build_native_morph_pool(
            but_train,
            but_x,
            int(seed),
            int(native_morph_copies),
            float(native_morph_strength),
        )
    else:
        native_morph = pd.DataFrame()
        native_morph_x = np.zeros((0, but_x.shape[-1]), dtype=np.float32)
    clean, clean_x = v115.build_cleanonly_regime_bank(
        but_tv=but_train,
        but_x=but_x,
        ptb=ptb,
        ptb_x=ptb_x,
        rng=rng,
        max_candidates_per_class=int(clean_candidates_per_class),
        bad_regimes=v115.REGIMES,
    )
    clean = mark_clean_pool(clean, int(seed))
    counts = quota_counts_by_subtype(but_train, int(residual_per_subtype))
    residual0, residual_x0 = v114.build_candidate_line(
        source_line="D2_ptb_but_style_residual",
        counts=counts,
        but_x=but_x,
        ptb_frame=ptb,
        ptb_x=ptb_x,
        anchors=anchors,
        bank=bank,
        rng=rng,
        d2_target_dominant=bool(d2_target_dominant),
        d2_ptb_mix_min=float(d2_ptb_mix_min),
        d2_ptb_mix_max=float(d2_ptb_mix_max),
    )
    residual, residual_x = v114.normalize_frame(residual0, residual_x0, "v116 PTB residual-transfer candidates")
    residual = mark_residual_pool(residual, int(seed))
    donor_rows = []
    for (cls, subtype), items in bank.items():
        for item in items:
            donor_rows.append(
                {
                    "class_name": cls,
                    "subtype": subtype,
                    "donor_subject_id": str(item.row.get("subject_id", "")),
                    "donor_record_id": str(item.row.get("record_id", "")),
                    "donor_source_idx": str(item.row.get("source_idx", "")),
                    "residual_family": residual_family(cls, subtype),
                    "severity": float(item.severity),
                }
            )
    return (
        {"native": native, "native_morph": native_morph, "residual": residual, "clean": clean},
        {"native": native_x, "native_morph": native_morph_x, "residual": residual_x, "clean": clean_x},
        pd.DataFrame(donor_rows),
    )


def concat_pool(frames: list[pd.DataFrame], signals: list[np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    x = np.vstack(signals).astype(np.float32) if signals else np.zeros((0, 0), dtype=np.float32)
    frame["idx"] = np.arange(len(frame), dtype=int)
    frame["_row_pos"] = np.arange(len(frame), dtype=int)
    return frame, x


def parse_class_quota_spec(spec: str) -> tuple[str, dict[str, float]]:
    """Parse good=0.45,medium=0.35,bad=0.40 into a stable tag and map."""
    mapping: dict[str, float] = {}
    aliases = {"g": "good", "good": "good", "m": "medium", "med": "medium", "medium": "medium", "b": "bad", "bad": "bad"}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid class quota part {part!r}; expected class=value")
        key, val = part.split("=", 1)
        cls = aliases.get(key.strip().lower())
        if cls is None:
            raise ValueError(f"Unknown class quota key {key!r}; expected good/medium/bad")
        q = float(val)
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"Class quota {part!r} outside [0,1]")
        mapping[cls] = q
    missing = [c for c in v114.CLASS_ORDER if c not in mapping]
    if missing:
        raise ValueError(f"Class quota spec {spec!r} missing {missing}")
    tag = "g{good:02d}_m{medium:02d}_b{bad:02d}".format(
        good=int(round(mapping["good"] * 100)),
        medium=int(round(mapping["medium"] * 100)),
        bad=int(round(mapping["bad"] * 100)),
    )
    return tag, mapping


def quota_specs(native_grid: str, native_class_grid: str) -> list[tuple[str, float, dict[str, float]]]:
    out: list[tuple[str, float, dict[str, float]]] = []
    if str(native_grid).strip().lower() not in {"", "none", "null", "false"}:
        grid_items = str(native_grid).split(",")
    else:
        grid_items = []
    for x in grid_items:
        if not x.strip():
            continue
        q = float(x.strip())
        out.append((f"q{int(round(q * 100)):02d}", q, {c: q for c in v114.CLASS_ORDER}))
    for spec in str(native_class_grid).split(";"):
        spec = spec.strip()
        if not spec:
            continue
        tag, mapping = parse_class_quota_spec(spec)
        avg = float(np.mean([mapping[c] for c in v114.CLASS_ORDER]))
        out.append((tag, avg, mapping))
    return out


def select_quota_class(
    *,
    cls: str,
    target_cls: pd.DataFrame,
    native_pool: pd.DataFrame,
    native_x: np.ndarray,
    nonnative_pool: pd.DataFrame,
    nonnative_x: np.ndarray,
    native_quota: float,
    final_n: int,
    features: list[str],
    rng: np.random.Generator,
    swaps: int,
    support_rows: int,
    device: str,
    rff_dim: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    n_native = min(int(round(int(final_n) * float(native_quota))), len(native_pool))
    n_other = max(0, min(int(final_n) - n_native, len(nonnative_pool)))
    out_frames: list[pd.DataFrame] = []
    out_xs: list[np.ndarray] = []
    traces: list[pd.DataFrame] = []
    if n_native > 0:
        sel, sx, tr = v115.greedy_set_select(
            target_cls=target_cls,
            pool_cls=native_pool.reset_index(drop=True),
            pool_x=native_x,
            final_n=n_native,
            features=features,
            rng=rng,
            swaps=max(50, int(swaps) // 3),
            class_name=f"{cls}_native_q{native_quota:.2f}",
            support_max_target_rows=int(support_rows),
            device_request=str(device),
            rff_dim=int(rff_dim),
            seed=int(seed) + 31,
        )
        out_frames.append(sel)
        out_xs.append(sx)
        traces.append(tr)
    if n_other > 0:
        sel, sx, tr = v115.greedy_set_select(
            target_cls=target_cls,
            pool_cls=nonnative_pool.reset_index(drop=True),
            pool_x=nonnative_x,
            final_n=n_other,
            features=features,
            rng=rng,
            swaps=max(50, int(swaps)),
            class_name=f"{cls}_nonnative_q{native_quota:.2f}",
            support_max_target_rows=int(support_rows),
            device_request=str(device),
            rff_dim=int(rff_dim),
            seed=int(seed) + 71,
        )
        out_frames.append(sel)
        out_xs.append(sx)
        traces.append(tr)
    out, out_x = concat_pool(out_frames, out_xs)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    trace["class_name_quota"] = cls
    trace["native_quota"] = float(native_quota)
    return out, out_x, trace


def natural_subtype_counts(target_cls: pd.DataFrame, final_n: int) -> dict[str, int]:
    subtype = v114.subtype_col(target_cls).astype(str)
    counts = subtype.value_counts().sort_index()
    if counts.empty:
        return {"": int(final_n)}
    raw = counts.astype(float) / float(counts.sum()) * int(final_n)
    base = np.floor(raw).astype(int)
    remainder = int(final_n) - int(base.sum())
    if remainder > 0:
        frac = (raw - base).sort_values(ascending=False)
        for key in frac.index[:remainder]:
            base.loc[key] += 1
    elif remainder < 0:
        frac = (raw - base).sort_values(ascending=True)
        for key in frac.index[: abs(remainder)]:
            if base.loc[key] > 0:
                base.loc[key] -= 1
    return {str(k): int(v) for k, v in base.items() if int(v) > 0}


def select_quota_class_natural_subtypes(
    *,
    cls: str,
    target_cls: pd.DataFrame,
    native_pool: pd.DataFrame,
    native_x: np.ndarray,
    nonnative_pool: pd.DataFrame,
    nonnative_x: np.ndarray,
    native_quota: float,
    final_n: int,
    features: list[str],
    rng: np.random.Generator,
    swaps: int,
    support_rows: int,
    device: str,
    rff_dim: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    subtype_counts = natural_subtype_counts(target_cls, int(final_n))
    native = native_pool.reset_index(drop=True).copy()
    nonnative = nonnative_pool.reset_index(drop=True).copy()
    native["_local_pool_pos"] = np.arange(len(native), dtype=int)
    nonnative["_local_pool_pos"] = np.arange(len(nonnative), dtype=int)
    native_sub = v114.subtype_col(native).astype(str) if len(native) else pd.Series([], dtype=str)
    nonnative_sub = v114.subtype_col(nonnative).astype(str) if len(nonnative) else pd.Series([], dtype=str)
    target_sub_all = v114.subtype_col(target_cls).astype(str)
    frames: list[pd.DataFrame] = []
    xs: list[np.ndarray] = []
    traces: list[pd.DataFrame] = []
    for j, (subtype, sub_n) in enumerate(subtype_counts.items()):
        target_sub = target_cls.loc[target_sub_all.eq(subtype)].copy()
        native_mask = native_sub.eq(subtype).to_numpy() if len(native) else np.zeros(0, dtype=bool)
        nonnative_mask = nonnative_sub.eq(subtype).to_numpy() if len(nonnative) else np.zeros(0, dtype=bool)
        n_frame = native.loc[native_mask].copy()
        nn_frame = nonnative.loc[nonnative_mask].copy()
        n_x = native_x[n_frame["_local_pool_pos"].astype(int).to_numpy()] if len(n_frame) else np.zeros((0,) + native_x.shape[1:], dtype=native_x.dtype)
        nn_x = nonnative_x[nn_frame["_local_pool_pos"].astype(int).to_numpy()] if len(nn_frame) else np.zeros((0,) + nonnative_x.shape[1:], dtype=nonnative_x.dtype)
        for f in (n_frame, nn_frame):
            if "_local_pool_pos" in f.columns:
                f.drop(columns=["_local_pool_pos"], inplace=True)
        if len(target_sub) == 0 or (len(n_frame) + len(nn_frame)) == 0:
            continue
        sel, sel_x, tr = select_quota_class(
            cls=f"{cls}/{subtype}",
            target_cls=target_sub,
            native_pool=n_frame,
            native_x=n_x,
            nonnative_pool=nn_frame,
            nonnative_x=nn_x,
            native_quota=float(native_quota),
            final_n=int(sub_n),
            features=features,
            rng=rng,
            swaps=max(50, int(swaps) // max(1, len(subtype_counts))),
            support_rows=max(50, min(int(support_rows), len(target_sub))),
            device=str(device),
            rff_dim=int(rff_dim),
            seed=int(seed) + j * 503,
        )
        sel["v116_natural_subtype_prior"] = 1
        sel["v116_natural_subtype_target_n"] = int(sub_n)
        frames.append(sel)
        xs.append(sel_x)
        traces.append(tr)
    out, out_x = concat_pool(frames, xs)
    if len(out) < int(final_n):
        # Fill any support gaps from the full class pool with the original class selector.
        missing = int(final_n) - len(out)
        fill, fill_x, tr = select_quota_class(
            cls=f"{cls}/natural_fill",
            target_cls=target_cls,
            native_pool=native.drop(columns=["_local_pool_pos"], errors="ignore"),
            native_x=native_x,
            nonnative_pool=nonnative.drop(columns=["_local_pool_pos"], errors="ignore"),
            nonnative_x=nonnative_x,
            native_quota=float(native_quota),
            final_n=missing,
            features=features,
            rng=rng,
            swaps=max(50, int(swaps) // 3),
            support_rows=int(support_rows),
            device=str(device),
            rff_dim=int(rff_dim),
            seed=int(seed) + 991,
        )
        fill["v116_natural_subtype_prior"] = 0
        out, out_x = concat_pool([out, fill], [out_x, fill_x])
        traces.append(tr)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    trace["class_name_quota"] = cls
    trace["native_quota"] = float(native_quota)
    trace["natural_subtype_prior"] = 1
    return out, out_x, trace


def select_native_budget(
    *,
    but_train: pd.DataFrame,
    pools: dict[str, pd.DataFrame],
    pool_xs: dict[str, np.ndarray],
    native_quota: float,
    native_quota_by_class: dict[str, float] | None,
    final_per_class: int,
    features: list[str],
    seed: int,
    swaps: int,
    support_rows: int,
    device: str,
    rff_dim: int,
    natural_subtype_prior: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    frames: list[pd.DataFrame] = []
    xs: list[np.ndarray] = []
    traces: list[pd.DataFrame] = []
    for i, cls in enumerate(v114.CLASS_ORDER):
        class_quota = float((native_quota_by_class or {}).get(cls, native_quota))
        target_cls = but_train.loc[but_train["class_name"].astype(str).eq(cls)].copy()
        native_cls = pools["native"].loc[pools["native"]["class_name"].astype(str).eq(cls)].copy()
        native_idx = native_cls.index.to_numpy(dtype=int)
        non_frames = []
        non_xs = []
        for name in ["residual", "clean"]:
            f = pools[name].loc[pools[name]["class_name"].astype(str).eq(cls)].copy()
            idx = f.index.to_numpy(dtype=int)
            non_frames.append(f)
            non_xs.append(pool_xs[name][idx])
        nonnative_cls, nonnative_x = concat_pool(non_frames, non_xs)
        if natural_subtype_prior:
            sel, sel_x, tr = select_quota_class_natural_subtypes(
                cls=cls,
                target_cls=target_cls,
                native_pool=native_cls,
                native_x=pool_xs["native"][native_idx],
                nonnative_pool=nonnative_cls,
                nonnative_x=nonnative_x,
                native_quota=class_quota,
                final_n=int(final_per_class),
                features=features,
                rng=rng,
                swaps=int(swaps),
                support_rows=int(support_rows),
                device=str(device),
                rff_dim=int(rff_dim),
                seed=int(seed) + i * 101 + int(round(class_quota * 1000)),
            )
        else:
            sel, sel_x, tr = select_quota_class(
                cls=cls,
                target_cls=target_cls,
                native_pool=native_cls,
                native_x=pool_xs["native"][native_idx],
                nonnative_pool=nonnative_cls,
                nonnative_x=nonnative_x,
                native_quota=class_quota,
                final_n=int(final_per_class),
                features=features,
                rng=rng,
                swaps=int(swaps),
                support_rows=int(support_rows),
                device=str(device),
                rff_dim=int(rff_dim),
                seed=int(seed) + i * 101 + int(round(class_quota * 1000)),
            )
        sel["v116_requested_native_quota_class"] = class_quota
        frames.append(sel)
        xs.append(sel_x)
        traces.append(tr)
    selected, selected_x = concat_pool(frames, xs)
    selected["split"] = v114.assign_protocol_splits(selected, int(seed) + int(round(native_quota * 1000)))
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    return selected, selected_x, trace


def relabel_for_gap_fill(frame: pd.DataFrame, candidate_type: str, generated: int) -> pd.DataFrame:
    out = frame.copy()
    out["v116_candidate_type"] = str(candidate_type)
    out["v116_generated"] = int(generated)
    if candidate_type == "original_but":
        out["v116_native_replay"] = 1
        out["v116_native_morph"] = 0
        out["v116_residual_transfer"] = 0
        out["v116_ptb_generated"] = 0
    elif candidate_type == "ptb_morph":
        out["v116_native_replay"] = 0
        out["v116_native_morph"] = 0
        out["v116_residual_transfer"] = 1
        out["v116_ptb_generated"] = 1
    elif candidate_type == "clean_style":
        out["v116_native_replay"] = 0
        out["v116_native_morph"] = 0
        out["v116_residual_transfer"] = 0
        out["v116_ptb_generated"] = 1
    return out


def _smc_initial_particle(
    target_z: np.ndarray,
    pool_z: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(pool_z)
    _, idx = nn.kneighbors(target_z, return_distance=True)
    selected = list(dict.fromkeys(idx[:, 0].astype(int).tolist()))
    if len(selected) < n:
        center = np.nanmedian(target_z, axis=0)
        order = np.argsort(np.linalg.norm(pool_z - center[None, :], axis=1))
        used = set(selected)
        for item in order:
            j = int(item)
            if j not in used:
                selected.append(j)
                used.add(j)
            if len(selected) >= n:
                break
    if len(selected) < n:
        used = set(selected)
        rest = [int(j) for j in rng.permutation(len(pool_z)) if int(j) not in used]
        selected.extend(rest[: n - len(selected)])
    return np.asarray(selected[:n], dtype=np.int64)


def _mutate_particle(
    particle: np.ndarray,
    pool_n: int,
    swaps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = particle.copy()
    swaps = min(max(0, int(swaps)), len(out), max(0, int(pool_n) - len(out)))
    if swaps <= 0:
        return out
    drop = rng.choice(np.arange(len(out)), size=swaps, replace=False)
    mask = np.ones(int(pool_n), dtype=bool)
    mask[out] = False
    add = rng.choice(np.flatnonzero(mask), size=swaps, replace=False)
    out[drop] = add
    return out


def strict_smc_set_select(
    *,
    target_cls: pd.DataFrame,
    pool_cls: pd.DataFrame,
    pool_x: np.ndarray,
    final_n: int,
    features: list[str],
    rng: np.random.Generator,
    swaps: int,
    class_name: str,
    support_max_target_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    n = min(int(final_n), len(pool_cls))
    if n <= 0:
        return pool_cls.iloc[[]].copy(), pool_x[:0], pd.DataFrame()
    target_eval = sample_frame(target_cls, max(150, int(support_max_target_rows)), rng).reset_index(drop=True)
    target_z, pool_z = v115.robust_z_against(target_eval, pool_cls, features)
    target_mean = np.nanmean(target_z, axis=0)
    target_std = np.nanstd(target_z, axis=0)
    target_q = np.nanpercentile(target_z, [10, 50, 90], axis=0)

    def score(indices: np.ndarray) -> float:
        # ponytail: fast particle score; exact support/MMD objective is computed once for the selected set.
        z = pool_z[indices]
        mean_gap = float(np.nanmean(np.abs(np.nanmean(z, axis=0) - target_mean)))
        std_gap = float(np.nanmean(np.abs(np.nanstd(z, axis=0) - target_std)))
        q_gap = float(np.nanmean(np.abs(np.nanpercentile(z, [10, 50, 90], axis=0) - target_q)))
        return 0.42 * mean_gap + 0.28 * std_gap + 0.30 * q_gap

    pool_n = len(pool_cls)
    n_particles = max(12, min(32, int(swaps) // 50))
    generations = max(5, min(10, int(swaps) // 150 + 4))
    base = _smc_initial_particle(target_z, pool_z, n, rng)
    particles = [base]
    for _ in range(n_particles - 1):
        if rng.random() < 0.25:
            particles.append(rng.choice(np.arange(pool_n), size=n, replace=False).astype(np.int64))
        else:
            particles.append(_mutate_particle(base, pool_n, max(1, int(round(n * rng.uniform(0.05, 0.35)))), rng))

    trace: list[dict[str, Any]] = []
    best_particle = particles[0].copy()
    best_score = float("inf")
    for generation in range(generations):
        scores = np.asarray([score(p) for p in particles], dtype=np.float64)
        if float(np.min(scores)) < best_score:
            best_score = float(np.min(scores))
            best_particle = particles[int(np.argmin(scores))].copy()
        q = max(0.25, 0.80 - 0.55 * generation / max(generations - 1, 1))
        epsilon = float(np.quantile(scores, q))
        accepted = scores <= epsilon
        if not np.any(accepted):
            accepted[np.argsort(scores)[: max(1, n_particles // 2)]] = True
        scale = max(float(epsilon - np.min(scores)), 1e-6)
        weights = np.zeros(n_particles, dtype=np.float64)
        weights[accepted] = np.exp(-(scores[accepted] - float(np.min(scores))) / scale)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            weights[accepted] = 1.0 / max(int(np.sum(accepted)), 1)
        else:
            weights = weights / total_weight
        for particle_id, (s, w, ok) in enumerate(zip(scores, weights, accepted)):
            trace.append(
                {
                    "selector": "strict_smc",
                    "generation": int(generation),
                    "particle": int(particle_id),
                    "class_name": class_name,
                    "smc_score": float(s),
                    "epsilon": float(epsilon),
                    "weight": float(w),
                    "accepted": int(ok),
                    "selected_n": int(n),
                    "pool_n": int(pool_n),
                }
            )
        if generation == generations - 1:
            break
        parents = rng.choice(np.arange(n_particles), size=n_particles - 1, replace=True, p=weights)
        frac = 1.0 - generation / max(generations - 1, 1)
        swap_n = max(1, int(round(n * (0.015 + 0.09 * frac))))
        particles = [best_particle.copy()] + [_mutate_particle(particles[int(parent)], pool_n, swap_n, rng) for parent in parents]

    out = pool_cls.iloc[best_particle].copy().reset_index(drop=True)
    out_x = pool_x[best_particle].astype(np.float32)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    support, _ = v115.support_floor_and_coverage(
        target_cls.assign(split="train"),
        out,
        features=features,
        rng=rng,
        max_target_rows=max(150, int(support_max_target_rows)),
    )
    cov = float(support.loc[support["subtype"].eq("__class__"), "coverage_q95"].iloc[0]) if not support.empty else 0.0
    exact = v115.set_objective(target_cls, out, cov, features, rng, {"mmd": 1.0, "swd": 1.0, "cdf": 1.0})
    trace.append(
        {
            "selector": "strict_smc",
            "generation": int(generations),
            "particle": -1,
            "class_name": class_name,
            "smc_score": float(best_score),
            "epsilon": float("nan"),
            "weight": 1.0,
            "accepted": 1,
            "selected_n": int(n),
            "pool_n": int(pool_n),
            "coverage_q95": cov,
            **exact,
        }
    )
    return out, out_x, pd.DataFrame(trace)


def select_component(
    *,
    cls: str,
    label: str,
    target_cls: pd.DataFrame,
    pool_cls: pd.DataFrame,
    pool_x: np.ndarray,
    final_n: int,
    features: list[str],
    rng: np.random.Generator,
    swaps: int,
    support_rows: int,
    device: str,
    rff_dim: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if int(final_n) <= 0:
        return pd.DataFrame(), np.zeros((0,) + pool_x.shape[1:], dtype=np.float32), pd.DataFrame()
    if len(pool_cls) < int(final_n):
        raise RuntimeError(f"{cls}/{label} pool too small: need {final_n}, have {len(pool_cls)}")
    sel, sx, tr = strict_smc_set_select(
        target_cls=target_cls,
        pool_cls=pool_cls.reset_index(drop=True),
        pool_x=pool_x,
        final_n=int(final_n),
        features=features,
        rng=rng,
        swaps=max(50, int(swaps)),
        class_name=f"{cls}_{label}",
        support_max_target_rows=int(support_rows),
    )
    tr["gap_fill_component"] = str(label)
    tr["device"] = str(device)
    tr["rff_dim"] = int(rff_dim)
    tr["selector_seed"] = int(seed)
    return sel, sx, tr


def sample_component(
    *,
    cls: str,
    label: str,
    pool_cls: pd.DataFrame,
    pool_x: np.ndarray,
    final_n: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if int(final_n) <= 0:
        return pd.DataFrame(), np.zeros((0,) + pool_x.shape[1:], dtype=np.float32), pd.DataFrame()
    if len(pool_cls) < int(final_n):
        raise RuntimeError(f"{cls}/{label} pool too small: need {final_n}, have {len(pool_cls)}")
    take = rng.choice(np.arange(len(pool_cls)), size=int(final_n), replace=False)
    out = pool_cls.iloc[take].copy().reset_index(drop=True)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    trace = pd.DataFrame([{"class_name_quota": cls, "gap_fill_component": label, "selector": "random", "selected_n": int(final_n), "pool_n": int(len(pool_cls))}])
    return out, pool_x[take].astype(np.float32), trace


def select_gap_fill(
    *,
    but_train: pd.DataFrame,
    pools: dict[str, pd.DataFrame],
    pool_xs: dict[str, np.ndarray],
    final_per_class: int,
    clean_cap: float,
    native_morph_min_frac: float,
    native_morph_selection: str,
    features: list[str],
    seed: int,
    swaps: int,
    support_rows: int,
    device: str,
    rff_dim: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    frames: list[pd.DataFrame] = []
    xs: list[np.ndarray] = []
    traces: list[pd.DataFrame] = []
    for i, cls in enumerate(v114.CLASS_ORDER):
        target_cls = but_train.loc[but_train["class_name"].astype(str).eq(cls)].copy()
        native_cls = pools["native"].loc[pools["native"]["class_name"].astype(str).eq(cls)].copy()
        if len(native_cls) > int(final_per_class):
            native_cls = sample_frame(native_cls, int(final_per_class), rng)
            traces.append(
                pd.DataFrame(
                    [
                        {
                            "class_name_quota": cls,
                            "gap_fill_component": "original_but_downsample",
                            "selector": "seeded_random",
                            "selected_n": int(final_per_class),
                            "pool_n": int(len(pools["native"].loc[pools["native"]["class_name"].astype(str).eq(cls)])),
                            "note": "public fallback native pool exceeded frozen final_per_class",
                        }
                    ]
                )
            )
        native_idx = native_cls.index.to_numpy(dtype=int)
        if cls == "good":
            if len(native_cls) != int(final_per_class):
                raise RuntimeError(f"gap_fill expects good original count {final_per_class}, got {len(native_cls)}")
            orig = relabel_for_gap_fill(native_cls, "original_but", 0)
            frames.append(orig)
            xs.append(pool_xs["native"][native_idx])
            continue

        orig = relabel_for_gap_fill(native_cls, "original_but", 0)
        frames.append(orig)
        xs.append(pool_xs["native"][native_idx])
        gap_n = int(final_per_class) - len(native_cls)
        clean_cls = pools["clean"].loc[pools["clean"]["class_name"].astype(str).eq(cls)].copy()
        clean_idx = clean_cls.index.to_numpy(dtype=int)
        morph_cls = pools["native_morph"].loc[pools["native_morph"]["class_name"].astype(str).eq(cls)].copy()
        morph_idx = morph_cls.index.to_numpy(dtype=int)
        residual_cls = pools["residual"].loc[pools["residual"]["class_name"].astype(str).eq(cls)].copy()
        residual_idx = residual_cls.index.to_numpy(dtype=int)

        base_clean_n = min(int(round(gap_n * max(0.0, float(clean_cap)))), gap_n)
        rest_n = gap_n - base_clean_n
        base_morph_n = min(rest_n, len(morph_cls), int(round(rest_n * max(0.0, float(native_morph_min_frac)))))
        base_ptb_n = rest_n - base_morph_n
        clean_n = base_clean_n
        morph_n = base_morph_n
        ptb_n = min(base_ptb_n, len(residual_cls))
        short_n = gap_n - clean_n - morph_n - ptb_n
        if short_n > 0:
            add_morph = min(short_n, max(0, len(morph_cls) - morph_n))
            morph_n += add_morph
            short_n -= add_morph
        if short_n > 0:
            add_clean = min(short_n, max(0, len(clean_cls) - clean_n))
            clean_n += add_clean
            short_n -= add_clean
        if short_n > 0:
            raise RuntimeError(
                f"{cls}/gap_fill pools too small: need {gap_n} generated, "
                f"clean={len(clean_cls)}, native_morph={len(morph_cls)}, residual={len(residual_cls)}, short={short_n}"
            )
        if (clean_n, morph_n, ptb_n) != (base_clean_n, base_morph_n, base_ptb_n):
            traces.append(
                pd.DataFrame(
                    [
                        {
                            "class_name_quota": cls,
                            "gap_fill_component": "dynamic_shortage_reallocation",
                            "selector": "capacity_aware",
                            "selected_n": int(gap_n),
                            "pool_n": int(len(clean_cls) + len(morph_cls) + len(residual_cls)),
                            "base_clean_n": int(base_clean_n),
                            "base_native_morph_n": int(base_morph_n),
                            "base_ptb_morph_n": int(base_ptb_n),
                            "clean_n": int(clean_n),
                            "native_morph_n": int(morph_n),
                            "ptb_morph_n": int(ptb_n),
                            "note": "public fallback candidate pools differed from frozen capacity",
                        }
                    ]
                )
            )
        if clean_n > 0:
            clean_sel, clean_x, tr = select_component(
                cls=cls,
                label="clean_style",
                target_cls=target_cls,
                pool_cls=relabel_for_gap_fill(clean_cls, "clean_style", 1),
                pool_x=pool_xs["clean"][clean_idx],
                final_n=clean_n,
                features=features,
                rng=rng,
                swaps=max(50, int(swaps) // 4),
                support_rows=int(support_rows),
                device=str(device),
                rff_dim=int(rff_dim),
                seed=int(seed) + i * 1009 + 17,
            )
            frames.append(clean_sel)
            xs.append(clean_x)
            traces.append(tr)
        if morph_n > 0:
            morph_pool = relabel_for_gap_fill(morph_cls, "but_native_morph", 1)
            morph_x = pool_xs["native_morph"][morph_idx]
            if str(native_morph_selection) == "random":
                morph_sel, morph_sel_x, tr = sample_component(
                    cls=cls,
                    label="but_native_morph",
                    pool_cls=morph_pool,
                    pool_x=morph_x,
                    final_n=morph_n,
                    rng=rng,
                )
            else:
                morph_sel, morph_sel_x, tr = select_component(
                    cls=cls,
                    label="but_native_morph",
                    target_cls=target_cls,
                    pool_cls=morph_pool,
                    pool_x=morph_x,
                    final_n=morph_n,
                    features=features,
                    rng=rng,
                    swaps=max(50, int(swaps) // 2),
                    support_rows=int(support_rows),
                    device=str(device),
                    rff_dim=int(rff_dim),
                    seed=int(seed) + i * 1009 + 53,
                )
            frames.append(morph_sel)
            xs.append(morph_sel_x)
            traces.append(tr)
        ptb_sel, ptb_sel_x, tr = select_component(
            cls=cls,
            label="ptb_morph",
            target_cls=target_cls,
            pool_cls=relabel_for_gap_fill(residual_cls, "ptb_morph", 1),
            pool_x=pool_xs["residual"][residual_idx],
            final_n=ptb_n,
            features=features,
            rng=rng,
            swaps=int(swaps),
            support_rows=int(support_rows),
            device=str(device),
            rff_dim=int(rff_dim),
            seed=int(seed) + i * 1009 + 71,
        )
        frames.append(ptb_sel)
        xs.append(ptb_sel_x)
        traces.append(tr)
    selected, selected_x = concat_pool(frames, xs)
    selected["split"] = v114.assign_protocol_splits(selected, int(seed) + 404)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    return selected, selected_x, trace


def distribution_metrics_with_floor(
    but_train: pd.DataFrame,
    selected: pd.DataFrame,
    tag: str,
    seed: int,
    floor: pd.DataFrame,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # v115's compute routine targets train+val by split. For v116 the target is
    # explicit BUT train, so mark the target as train.
    target = but_train.copy()
    target["split"] = "train"
    metric, summary, global_metric = v115.compute_v110_metrics(target, selected, tag, int(seed))
    # v116 intentionally changes P(Y) to a balanced training prior.  Therefore
    # the all-label audit must compare against a class-balanced BUT target,
    # otherwise the all-label MMD mostly measures the intended prior shift
    # rather than class-conditional distribution preservation.
    rng = np.random.default_rng(int(seed) + 909)
    balanced_parts: list[pd.DataFrame] = []
    for cls in v114.CLASS_ORDER:
        t = target.loc[target["class_name"].astype(str).eq(cls)].copy()
        s_n = int(selected["class_name"].astype(str).eq(cls).sum())
        if len(t) == 0 or s_n <= 0:
            continue
        take = min(len(t), s_n)
        balanced_parts.append(sample_frame(t, take, rng))
    if balanced_parts:
        balanced_target = pd.concat(balanced_parts, ignore_index=True)
        features = v114.available_features(balanced_target, selected, v114.V81.MATCH_FEATURES)
        balanced_row = v115.metric_row(tag, "all_labels", balanced_target, selected, features, rng)
        global_metric = global_metric.loc[global_metric["scope"].astype(str).ne("all_labels")].copy()
        global_metric = pd.concat([pd.DataFrame([balanced_row]), global_metric], ignore_index=True)
    floor_map = floor.set_index("scope")["rbf_mmd_q95"].to_dict() if not floor.empty else {}
    rows = []
    for _, r in global_metric.iterrows():
        scope = str(r["scope"])
        lim = float(floor_map.get(scope, np.nan))
        mmd = float(r["rbf_mmd"])
        rows.append(
            {
                "tag": tag,
                "scope": scope,
                "rbf_mmd": mmd,
                "floor_q95": lim,
                "epsilon": float(epsilon),
                "floor_pass": bool(math.isfinite(lim) and mmd <= lim + float(epsilon)),
                "practical_pass": bool(
                    (scope == "all_labels" and mmd <= 0.10)
                    or (scope == "class_bad" and mmd <= 0.14 and float(r["pca_density_overlap"]) >= 0.45)
                    or (scope in {"class_good", "class_medium"} and mmd <= 0.10)
                ),
                "pca_density_overlap": float(r["pca_density_overlap"]),
                "sym_domain_auc": float(r["sym_domain_auc"]),
            }
        )
    return metric, summary, global_metric, pd.DataFrame(rows)


def donor_reuse_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in ["v116_native_donor_id", "v116_residual_donor_id", "v116_style_donor_id", "v116_ptb_carrier_id"]:
        if col not in frame.columns:
            continue
        vals = frame[col].fillna("").astype(str)
        vals = vals.loc[vals.ne("")]
        if vals.empty:
            continue
        counts = vals.value_counts()
        rows.append(
            {
                "id_column": col,
                "unique_ids": int(counts.size),
                "max_reuse": int(counts.iloc[0]),
                "max_reuse_share": float(counts.iloc[0] / max(len(frame), 1)),
                "top_id": str(counts.index[0]),
            }
        )
    if "v116_candidate_type" in frame.columns:
        for name, n in frame["v116_candidate_type"].fillna("").astype(str).value_counts().items():
            rows.append({"id_column": "candidate_type", "top_id": name, "max_reuse": int(n), "max_reuse_share": float(n / max(len(frame), 1)), "unique_ids": np.nan})
    return pd.DataFrame(rows)


def nearest_neighbor_audit(
    selected: pd.DataFrame,
    selected_x: np.ndarray,
    but_test: pd.DataFrame,
    but_x: np.ndarray,
    features: list[str],
    seed: int,
    max_rows: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    if len(but_test) < 5 or len(selected) < 5:
        return pd.DataFrame()
    sel = sample_frame(selected, int(max_rows), rng).reset_index(drop=True)
    test = sample_frame(but_test, int(max_rows), rng).reset_index(drop=True)
    sel_x = selected_x[sel["_row_pos"].astype(int).to_numpy()]
    test_x = but_x[test["_row_pos"].astype(int).to_numpy()]
    for cls in ["all"] + v114.CLASS_ORDER:
        if cls == "all":
            s_mask = np.ones(len(sel), dtype=bool)
            t_mask = np.ones(len(test), dtype=bool)
        else:
            s_mask = sel["class_name"].astype(str).eq(cls).to_numpy()
            t_mask = test["class_name"].astype(str).eq(cls).to_numpy()
        if s_mask.sum() < 3 or t_mask.sum() < 3:
            continue
        s_frame = sel.loc[s_mask].copy()
        t_frame = test.loc[t_mask].copy()
        tz, sz = v115.robust_z_against(t_frame, s_frame, features)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(tz)
        d, _ = nn.kneighbors(sz, return_distance=True)
        sx = sel_x[s_mask]
        tx = test_x[t_mask]
        # Normalize raw waveform distance to robust test scale.
        scale = np.nanmedian(np.nanstd(tx, axis=1))
        scale = max(float(scale), 1e-5)
        raw_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(tx / scale)
        rd, _ = raw_nn.kneighbors(sx / scale, return_distance=True)
        rows.append(
            {
                "scope": cls,
                "selected_n": int(s_mask.sum()),
                "test_n": int(t_mask.sum()),
                "feature_nn_min": float(np.min(d)),
                "feature_nn_p01": float(np.percentile(d, 1)),
                "feature_nn_median": float(np.median(d)),
                "raw_nn_min": float(np.min(rd)),
                "raw_nn_p01": float(np.percentile(rd, 1)),
                "raw_nn_median": float(np.median(rd)),
                "near_duplicate_feature_count": int(np.sum(d[:, 0] < 1e-6)),
                "near_duplicate_raw_count": int(np.sum(rd[:, 0] < 1e-6)),
            }
        )
    return pd.DataFrame(rows)


def save_protocol(name: str, frame: pd.DataFrame, x: np.ndarray, seed: int, summary: dict[str, Any]) -> Path:
    out = frame.copy().reset_index(drop=True)
    out["idx"] = np.arange(len(out), dtype=int)
    out["_row_pos"] = np.arange(len(out), dtype=int)
    if "split" not in out.columns:
        out["split"] = v114.assign_protocol_splits(out, int(seed))
    path = PROTOCOL_ROOT / name
    v114.save_protocol(path, out, x.astype(np.float32), summary)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "smoke", "floor", "pareto", "model_validation"], default="all")
    parser.add_argument("--balance-policy", choices=["native_budget", "gap_fill"], default="native_budget")
    parser.add_argument("--seed", type=int, default=20260860)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--native-grid", default="0,0.25,0.35,0.40,0.45,0.55")
    parser.add_argument(
        "--native-class-grid",
        default="",
        help="Optional semicolon-separated class-wise quota specs, e.g. good=0.45,medium=0.35,bad=0.40;good=0.45,medium=0.30,bad=0.35",
    )
    parser.add_argument("--final-per-class", type=int, default=900)
    parser.add_argument("--clean-candidates-per-class", type=int, default=1600)
    parser.add_argument("--residual-per-subtype", type=int, default=220)
    parser.add_argument("--max-donors-per-subtype", type=int, default=180)
    parser.add_argument("--native-morph-copies", type=int, default=0)
    parser.add_argument("--native-morph-strength", type=float, default=1.0)
    parser.add_argument("--gap-clean-cap", type=float, default=0.05)
    parser.add_argument("--gap-native-morph-min-frac", type=float, default=0.0)
    parser.add_argument("--gap-native-morph-selection", choices=["smc", "random"], default="smc")
    parser.add_argument("--max-ptb-carriers", type=int, default=6000)
    parser.add_argument("--floor-draws", type=int, default=96)
    parser.add_argument("--floor-max-rows-per-class", type=int, default=1200)
    parser.add_argument("--selector-swaps", type=int, default=1500)
    parser.add_argument("--support-max-target-rows", type=int, default=1200)
    parser.add_argument("--rff-dim", type=int, default=1024)
    parser.add_argument("--floor-epsilon", type=float, default=0.02)
    parser.add_argument("--d2-ptb-mix-min", type=float, default=0.005)
    parser.add_argument("--d2-ptb-mix-max", type=float, default=0.055)
    parser.add_argument("--d2-target-dominant", dest="d2_target_dominant", action="store_true", default=True)
    parser.add_argument("--no-d2-target-dominant", dest="d2_target_dominant", action="store_false")
    parser.add_argument("--natural-subtype-prior", action="store_true", help="Match BUT train subtype shares inside each class instead of letting set selection drift subtype quotas.")
    parser.add_argument("--skip-protocol-write", action="store_true")
    parser.add_argument("--run-model", action="store_true")
    args = parser.parse_args()
    if args.stage == "smoke":
        args.native_grid = "0,0.55"
        args.native_class_grid = ""
        args.final_per_class = min(args.final_per_class, 60)
        args.clean_candidates_per_class = min(args.clean_candidates_per_class, 120)
        args.residual_per_subtype = min(args.residual_per_subtype, 12)
        args.max_donors_per_subtype = min(args.max_donors_per_subtype, 20)
        args.native_morph_copies = min(args.native_morph_copies, 1)
        args.max_ptb_carriers = min(args.max_ptb_carriers, 500)
        args.floor_draws = min(args.floor_draws, 8)
        args.floor_max_rows_per_class = min(args.floor_max_rows_per_class, 120)
        args.selector_swaps = min(args.selector_swaps, 80)
        args.support_max_target_rows = min(args.support_max_target_rows, 150)
        args.rff_dim = min(args.rff_dim, 512)

    rng = np.random.default_rng(int(args.seed))
    tag = f"s{args.seed}"
    report_root = REPORT_ROOT / tag
    report_root.mkdir(parents=True, exist_ok=True)

    print(f"{now()} loading BUT reference", flush=True)
    but0, butx0 = v114.V81.load_protocol(v114.DEFAULT_BUT_PROTOCOL)
    but, but_x = v114.normalize_frame(but0, butx0, "v116 BUT reference")
    but_train = but_train_frame(but)
    but_test = but.loc[but["split"].astype(str).eq("test")].copy()
    features = v114.available_features(but_train, but_train, v114.V81.MATCH_FEATURES)

    print(f"{now()} native floor calibration", flush=True)
    floor_detail, floor_summary = native_split_floor(
        but_train,
        features=features,
        draws=int(args.floor_draws),
        seed=int(args.seed),
        max_rows_per_class=int(args.floor_max_rows_per_class),
    )
    floor_detail.to_csv(lp(report_root / "native_split_floor_detail.csv"), index=False)
    floor_summary.to_csv(lp(report_root / "native_split_floor.csv"), index=False)
    if args.stage == "floor":
        print(json.dumps({"report": str(report_root), "floor": str(report_root / "native_split_floor.csv")}, indent=2), flush=True)
        return

    print(f"{now()} loading PTB clean carriers", flush=True)
    ptb0, ptbx0 = v114.V81.load_protocol(v114.DEFAULT_PTB_CARRIER_PROTOCOL)
    ptb, ptb_x = v114.normalize_frame(ptb0, ptbx0, "v116 PTB clean carrier")
    if len(ptb) > int(args.max_ptb_carriers):
        take = rng.choice(np.arange(len(ptb)), size=int(args.max_ptb_carriers), replace=False)
        ptb = ptb.iloc[take].reset_index(drop=True)
        ptb_x = ptb_x[take]
        ptb["_row_pos"] = np.arange(len(ptb), dtype=int)

    print(f"{now()} building candidate pools", flush=True)
    pools, pool_xs, residual_bank_audit = build_candidate_pools(
        but_train=but_train,
        but_x=but_x,
        ptb=ptb,
        ptb_x=ptb_x,
        seed=int(args.seed),
        clean_candidates_per_class=int(args.clean_candidates_per_class),
        residual_per_subtype=int(args.residual_per_subtype),
        max_donors_per_subtype=int(args.max_donors_per_subtype),
        native_morph_copies=int(args.native_morph_copies),
        native_morph_strength=float(args.native_morph_strength),
        d2_target_dominant=bool(args.d2_target_dominant),
        d2_ptb_mix_min=float(args.d2_ptb_mix_min),
        d2_ptb_mix_max=float(args.d2_ptb_mix_max),
    )
    residual_bank_audit.to_csv(lp(report_root / "residual_bank_audit.csv"), index=False)
    all_pool = pd.concat(list(pools.values()), ignore_index=True)
    features = v114.available_features(but_train, all_pool, v114.V81.MATCH_FEATURES)

    if str(args.balance_policy) == "gap_fill":
        nm_tag = int(round(100.0 * max(0.0, float(args.gap_native_morph_min_frac))))
        ms_tag = int(round(100.0 * max(0.0, float(args.native_morph_strength))))
        sel_tag = "rnd" if str(args.gap_native_morph_selection) == "random" else "smc"
        name = f"v116_gapfill_dual_goodorig_nm{nm_tag:02d}_ms{ms_tag:02d}_{sel_tag}_s{args.seed}"
        print(f"{now()} selecting gap-fill protocol {name}", flush=True)
        selected, selected_x, trace = select_gap_fill(
            but_train=but_train,
            pools=pools,
            pool_xs=pool_xs,
            final_per_class=int(args.final_per_class),
            clean_cap=float(args.gap_clean_cap),
            native_morph_min_frac=float(args.gap_native_morph_min_frac),
            native_morph_selection=str(args.gap_native_morph_selection),
            features=features,
            seed=int(args.seed),
            swaps=int(args.selector_swaps),
            support_rows=int(args.support_max_target_rows),
            device=str(args.device),
            rff_dim=int(args.rff_dim),
        )
        selected["v116_balance_policy"] = "gap_fill"
        selected["idx"] = np.arange(len(selected), dtype=int)
        selected["_row_pos"] = np.arange(len(selected), dtype=int)
        trace.to_csv(lp(report_root / f"{name}_selector_trace.csv"), index=False)
        metric, summary, global_metric, floor_eval = distribution_metrics_with_floor(
            but_train,
            selected,
            name,
            int(args.seed) + 421,
            floor_summary,
            float(args.floor_epsilon),
        )
        metric.to_csv(lp(report_root / f"{name}_distribution_metrics_v110.csv"), index=False)
        summary.to_csv(lp(report_root / f"{name}_class_subtype_median_summary.csv"), index=False)
        global_metric.to_csv(lp(report_root / f"{name}_global_distribution_metrics.csv"), index=False)
        donor_reuse_audit(selected).to_csv(lp(report_root / f"{name}_donor_reuse_audit.csv"), index=False)
        nearest_neighbor_audit(
            selected,
            selected_x,
            but_test,
            but_x,
            features,
            int(args.seed) + 503,
            max_rows=max(100, min(1200, int(args.support_max_target_rows))),
        ).to_csv(lp(report_root / f"{name}_nearest_neighbor_leakage_audit.csv"), index=False)
        v114.V81.plot_shared_pca(but_train.assign(split="train"), selected, report_root, name)
        v114.V81.plot_metric_heatmap(metric, report_root, name, "rbf_mmd")
        counts = selected.groupby(["class_name", "v116_candidate_type"], as_index=False).size()
        counts.to_csv(lp(report_root / f"{name}_candidate_type_counts.csv"), index=False)
        if not args.skip_protocol_write:
            save_protocol(
                name,
                selected,
                selected_x,
                int(args.seed) + 404,
                {
                    "protocol": name,
                    "line": "v116 gap-fill dual-AUC repair",
                    "seed": int(args.seed),
                    "balance_policy": "gap_fill",
                    "final_per_class": int(args.final_per_class),
                    "gap_clean_cap": float(args.gap_clean_cap),
                    "gap_native_morph_min_frac": float(args.gap_native_morph_min_frac),
                    "native_morph_copies": int(args.native_morph_copies),
                    "native_morph_strength": float(args.native_morph_strength),
                    "gap_native_morph_selection": str(args.gap_native_morph_selection),
                    "but_test_exclusion": "BUT test excluded from donors, targets, selection, thresholds, and early stopping.",
                    "model_contract": "Final model uses waveform-derived dual-view inputs; SQI columns are not model inputs.",
                },
            )
        pd.DataFrame(
            [
                {
                    "protocol": name,
                    "balance_policy": "gap_fill",
                    "final_per_class": int(args.final_per_class),
                    "gap_native_morph_min_frac": float(args.gap_native_morph_min_frac),
                    "native_morph_strength": float(args.native_morph_strength),
                    "gap_native_morph_selection": str(args.gap_native_morph_selection),
                    "rows": int(len(selected)),
                    "original_but_rows": int((selected["v116_candidate_type"].astype(str) == "original_but").sum()),
                    "generated_rows": int(pd.to_numeric(selected["v116_generated"], errors="coerce").fillna(0).sum()),
                }
            ]
        ).to_csv(lp(report_root / "protocol_manifest.csv"), index=False)
        floor_eval["protocol"] = name
        floor_eval.to_csv(lp(report_root / "gap_fill_floor_eval.csv"), index=False)
        write_md(
            report_root / "v116_gapfill_dual_goodorig_report.md",
            "v116 Gap-Fill Good-Original Repair",
            [
                "Good is kept as original BUT only. Medium and bad keep all original BUT rows and fill only the class gaps.",
                "Generated gap candidates are BUT native-morph, PTB morph, and clean-style capped by `gap_clean_cap`.",
                "Dual-view generated-vs-original AUC is audited separately by `diagnose_dual_generated_auc.py`; SQI AUC is not a reporting gate for this branch.",
                "Candidate counts:\n\n```text\n" + counts.to_string(index=False) + "\n```",
            ],
        )
        if args.run_model or args.stage == "model_validation":
            cmd = (
                f"python -m src.transformer_pipeline.cli train --model E31 --run # policy={name}, seed={int(args.seed)}, no-record-balanced-sampler"
            )
            (report_root / "model_validation_commands.txt").write_text(cmd + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_root), "protocol": str(PROTOCOL_ROOT / name)}, indent=2), flush=True)
        return

    specs = quota_specs(str(args.native_grid), str(args.native_class_grid))
    pareto_rows: list[pd.DataFrame] = []
    protocol_rows: list[dict[str, Any]] = []
    for quota_tag, quota, quota_by_class in specs:
        quota_desc = ",".join(f"{c}={quota_by_class[c]:.2f}" for c in v114.CLASS_ORDER)
        print(f"{now()} selecting native quota {quota_tag} ({quota_desc})", flush=True)
        selected, selected_x, trace = select_native_budget(
            but_train=but_train,
            pools=pools,
            pool_xs=pool_xs,
            native_quota=float(quota),
            native_quota_by_class=quota_by_class,
            final_per_class=int(args.final_per_class),
            features=features,
            seed=int(args.seed) + int(round(quota * 1000)) + sum(int(round(quota_by_class[c] * 1000)) for c in v114.CLASS_ORDER),
            swaps=int(args.selector_swaps),
            support_rows=int(args.support_max_target_rows),
            device=str(args.device),
            rff_dim=int(args.rff_dim),
            natural_subtype_prior=bool(args.natural_subtype_prior),
        )
        selected["v116_requested_native_quota"] = float(quota)
        for cls in v114.CLASS_ORDER:
            selected[f"v116_requested_native_quota_{cls}"] = float(quota_by_class[cls])
        selected["idx"] = np.arange(len(selected), dtype=int)
        selected["_row_pos"] = np.arange(len(selected), dtype=int)
        actual_native = float(pd.to_numeric(selected.get("v116_native_replay", pd.Series(0, index=selected.index)), errors="coerce").fillna(0).mean())
        name = f"v116_native_budget_{quota_tag}_s{args.seed}"
        trace.to_csv(lp(report_root / f"{name}_selector_trace.csv"), index=False)
        metric, summary, global_metric, floor_eval = distribution_metrics_with_floor(
            but_train,
            selected,
            name,
            int(args.seed) + int(round(quota * 1000)) + 19 + sum(int(round(quota_by_class[c] * 1000)) for c in v114.CLASS_ORDER),
            floor_summary,
            float(args.floor_epsilon),
        )
        metric.to_csv(lp(report_root / f"{name}_distribution_metrics_v110.csv"), index=False)
        summary.to_csv(lp(report_root / f"{name}_class_subtype_median_summary.csv"), index=False)
        global_metric.to_csv(lp(report_root / f"{name}_global_distribution_metrics.csv"), index=False)
        floor_eval["native_quota_requested"] = float(quota)
        floor_eval["native_quota_good"] = float(quota_by_class["good"])
        floor_eval["native_quota_medium"] = float(quota_by_class["medium"])
        floor_eval["native_quota_bad"] = float(quota_by_class["bad"])
        floor_eval["actual_native_fraction"] = actual_native
        floor_eval["protocol"] = name
        pareto_rows.append(floor_eval)
        donor_reuse_audit(selected).to_csv(lp(report_root / f"{name}_donor_reuse_audit.csv"), index=False)
        nearest_neighbor_audit(
            selected,
            selected_x,
            but_test,
            but_x,
            features,
            int(args.seed) + 503,
            max_rows=max(100, min(1200, int(args.support_max_target_rows))),
        ).to_csv(lp(report_root / f"{name}_nearest_neighbor_leakage_audit.csv"), index=False)
        v114.V81.plot_shared_pca(but_train.assign(split="train"), selected, report_root, name)
        v114.V81.plot_metric_heatmap(metric, report_root, name, "rbf_mmd")
        if not args.skip_protocol_write:
            save_protocol(
                name,
                selected,
                selected_x,
                int(args.seed) + int(round(quota * 1000)),
                {
                    "protocol": name,
                    "line": "v116 leakage-safe support-aware semi-synthetic repair",
                    "seed": int(args.seed),
                    "native_quota_requested": float(quota),
                    "native_quota_by_class": quota_by_class,
                    "actual_native_fraction": actual_native,
                    "natural_subtype_prior": bool(args.natural_subtype_prior),
                    "but_test_exclusion": "BUT test excluded from donors, targets, selection, thresholds, and early stopping.",
                    "model_contract": "Final model must use waveform-derived inputs only.",
                },
            )
        protocol_rows.append(
            {
                "protocol": name,
                "native_quota_requested": float(quota),
                "native_quota_good": float(quota_by_class["good"]),
                "native_quota_medium": float(quota_by_class["medium"]),
                "native_quota_bad": float(quota_by_class["bad"]),
                "actual_native_fraction": actual_native,
            }
        )

    pareto = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()
    pareto.to_csv(lp(report_root / "native_budget_pareto.csv"), index=False)
    pd.DataFrame(protocol_rows).to_csv(lp(report_root / "protocol_manifest.csv"), index=False)

    if not pareto.empty:
        pivot = pareto.pivot_table(index=["protocol", "native_quota_requested", "actual_native_fraction"], columns="scope", values=["rbf_mmd", "pca_density_overlap", "floor_pass"], aggfunc="first")
        summary_text = pivot.reset_index().to_string(index=False)
    else:
        summary_text = "No Pareto rows generated."
    write_md(
        report_root / "v116_native_budget_repair_report.md",
        "v116 Native-Budgeted Support-Aware Repair",
        [
            "This line treats clean-only as a negative control and optimizes the minimum BUT train-only native/residual support needed to preserve class-conditional distributions.",
            "BUT test rows are excluded from donor pools, feature targets, selector objectives, thresholds, and model selection.",
            "Native floor calibration: see `native_split_floor.csv`.",
            "Native budget Pareto table: see `native_budget_pareto.csv`.",
            "Candidate pools: BUT train native anchors, PTB+BUT residual transfer, and PTB clean-style generator.",
            "Pareto snapshot:\n\n```text\n" + summary_text + "\n```",
        ],
    )

    if args.run_model or args.stage == "model_validation":
        commands = []
        for row in protocol_rows:
            commands.append(
                f"python -m src.transformer_pipeline.cli train --model E31 --run # policy={row['protocol']}, seed={int(args.seed)}, no-record-balanced-sampler"
            )
        (report_root / "model_validation_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_root), "pareto": str(report_root / "native_budget_pareto.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
