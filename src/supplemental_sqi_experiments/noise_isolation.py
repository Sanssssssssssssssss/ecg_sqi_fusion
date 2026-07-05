from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from src.sqi_pipeline.noise.make_paper_aligned_balanced_cases import (
    NOISE_TYPES,
    SNR_DB,
    _assign_group_types,
    _cache_clean_cases,
    _make_noisy_case_paper,
    _plot_label_counts,
    _resolve_path,
    _stratified_group_split,
    _validate_split,
)
from src.sqi_pipeline.noise.make_balanced_noisy_cases import (
    CASE_SEC,
    FS_NOISE_IN,
    N_SAMPLES,
    load_clean_500_from_wfdb,
    read_nstdb,
    write_case_500_npz,
)
from src.utils.paths import project_root


REGION_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass
class RegionNoiseSampler:
    nstdb_signals: dict[str, np.ndarray]
    rng: np.random.Generator
    stride_s: float
    region_by_split: dict[str, tuple[float, float]]

    def __post_init__(self) -> None:
        self.seg_len_in = int(round(CASE_SEC * FS_NOISE_IN))
        stride = max(1, int(round(float(self.stride_s) * FS_NOISE_IN)))
        self.starts: dict[tuple[str, str], list[int]] = {}
        self.region_bounds_samples: dict[tuple[str, str], tuple[int, int]] = {}
        for noise_type, sig in self.nstdb_signals.items():
            n = int(sig.shape[0])
            if n <= self.seg_len_in:
                raise ValueError(f"NSTDB {noise_type} too short for {CASE_SEC}s windows")
            for split, (lo_frac, hi_frac) in self.region_by_split.items():
                region_lo = int(np.floor(lo_frac * n))
                region_hi = int(np.floor(hi_frac * n))
                max_start = region_hi - self.seg_len_in
                if max_start < region_lo:
                    raise ValueError(f"NSTDB region {split} for {noise_type} is too short for {CASE_SEC}s windows")
                starts = np.arange(region_lo, max_start + 1, stride, dtype=int)
                self.rng.shuffle(starts)
                self.starts[(noise_type, split)] = starts.tolist()
                self.region_bounds_samples[(noise_type, split)] = (int(region_lo), int(region_hi))

    def draw_500(self, noise_type: str, split: str) -> tuple[np.ndarray, int]:
        key = (noise_type, split)
        starts = self.starts[key]
        if not starts:
            raise RuntimeError(f"No NSTDB segment starts left for noise_type={noise_type}, split={split}")
        start = int(starts.pop())
        sig360 = self.nstdb_signals[noise_type]
        seg360 = sig360[start : start + self.seg_len_in, :]
        seg500 = resample_poly(seg360, up=25, down=18, axis=0).astype(np.float64)
        if seg500.shape[0] < N_SAMPLES:
            seg500 = np.pad(seg500, ((0, N_SAMPLES - seg500.shape[0]), (0, 0)), mode="edge")
        return seg500[:N_SAMPLES, :], start


def _region_by_split() -> dict[str, tuple[float, float]]:
    train = REGION_RATIOS["train"]
    val = REGION_RATIOS["val"]
    return {"train": (0.0, train), "val": (train, train + val), "test": (train + val, 1.0)}


def audit_noise_overlap(audit_csv: str | Path, out_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(audit_csv)
    if df.empty:
        out = pd.DataFrame(columns=["split_a", "split_b", "noise_type", "n_overlaps"])
        out.to_csv(out_csv, index=False)
        return out
    df["noise_start_360"] = df["noise_start_360"].astype(int)
    if "noise_end_360" not in df.columns:
        df["noise_end_360"] = df["noise_start_360"] + int(round(CASE_SEC * FS_NOISE_IN))
    rows = []
    for noise_type, g in df.groupby("noise_type"):
        by_split = {s: gg.sort_values("noise_start_360") for s, gg in g.groupby("split")}
        splits = sorted(by_split)
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                ga = by_split[a]
                gb = by_split[b]
                count = 0
                for ra in ga.itertuples(index=False):
                    overlap = (gb["noise_start_360"].to_numpy() < int(ra.noise_end_360)) & (
                        gb["noise_end_360"].to_numpy() > int(ra.noise_start_360)
                    )
                    count += int(overlap.sum())
                rows.append({"split_a": a, "split_b": b, "noise_type": noise_type, "n_overlaps": int(count)})
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def build_isolated_dataset(params: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    seed = int(params.get("seed", 0))
    snr_db = float(params.get("snr_db", SNR_DB))
    noise_stride_s = float(params.get("noise_start_stride_s", 10.0))
    force = bool(params.get("force", False))

    manifest_csv = _resolve_path(root, params["manifest_csv"])
    out_split = _resolve_path(root, params["out_split_csv"])
    audit_csv = _resolve_path(root, params.get("audit_csv", out_split.with_suffix(".audit.csv")))
    overlap_csv = _resolve_path(root, params.get("overlap_csv", out_split.with_suffix(".noise_overlap_audit.csv")))
    qc_png = _resolve_path(root, params.get("qc_png", out_split.with_suffix(".label_counts.png")))
    set_a_dir = _resolve_path(root, params["set_a_dir"])
    nstdb_dir = _resolve_path(root, params["nstdb_dir"])
    cases_dir = _resolve_path(root, params["cases_500_dir"])

    if out_split.exists() and audit_csv.exists() and overlap_csv.exists() and not force:
        return {"step": "isolated_paper_balanced_seta", "skipped": True, "outputs": [str(out_split), str(audit_csv), str(overlap_csv)]}

    df_manifest = pd.read_csv(manifest_csv)
    df_labeled = df_manifest[df_manifest["quality_record"].isin(["acceptable", "unacceptable"])].copy()
    df_labeled["record_id"] = df_labeled["record_id"].astype(str)
    df_labeled["y"] = df_labeled["quality_record"].map({"acceptable": 1, "unacceptable": -1}).astype(int)
    df_labeled = df_labeled.sort_values("record_id").reset_index(drop=True)

    n_good = int((df_labeled["y"] == 1).sum())
    n_bad = int((df_labeled["y"] == -1).sum())
    need = n_good - n_bad
    n_sources = int(np.ceil(need / len(NOISE_TYPES))) if need else 0
    rng = np.random.default_rng(seed)
    acceptable_ids = df_labeled.loc[df_labeled["y"] == 1, "record_id"].astype(str).to_numpy()
    chosen_sources = rng.choice(acceptable_ids, size=n_sources, replace=False).astype(str).tolist() if n_sources else []

    _cache_clean_cases(df_labeled["record_id"].astype(str).tolist(), set_a_dir, cases_dir, force=force)

    rows: list[dict[str, Any]] = []
    for _, r in df_labeled.iterrows():
        rid = str(r["record_id"])
        rows.append(
            {
                "record_id": rid,
                "y": int(r["y"]),
                "split": "",
                "seed": seed,
                "quality_record": str(r["quality_record"]),
                "is_augmented": 0,
                "source_record_id": rid,
                "noise_type": "",
                "snr_db": np.nan,
            }
        )
    provisional = pd.DataFrame(rows)

    # Add synthetic rows before group split so clean sources and derivatives stay together.
    made = 0
    for source_id in chosen_sources:
        for noise_type in NOISE_TYPES:
            if made >= need:
                break
            provisional.loc[len(provisional)] = {
                "record_id": f"{source_id}__paper_{noise_type}__snr{int(snr_db)}__seed{seed}__{made:05d}",
                "y": -1,
                "split": "",
                "seed": seed,
                "quality_record": "paper_noisy_unacceptable",
                "is_augmented": 1,
                "source_record_id": source_id,
                "noise_type": noise_type,
                "snr_db": float(snr_db),
            }
            made += 1
        if made >= need:
            break
    group_df = _assign_group_types(provisional)
    split_map = _stratified_group_split(group_df, seed)
    provisional["split"] = provisional["source_record_id"].map(split_map).astype(str)
    _validate_split(provisional)

    nstdb_signals = {noise_type: read_nstdb(noise_type, nstdb_dir) for noise_type in NOISE_TYPES}
    sampler = RegionNoiseSampler(nstdb_signals, rng=rng, stride_s=noise_stride_s, region_by_split=_region_by_split())
    audit_rows: list[dict[str, Any]] = []
    for r in provisional[provisional["is_augmented"].astype(int).eq(1)].itertuples(index=False):
        source_id = str(r.source_record_id)
        noise_type = str(r.noise_type)
        split = str(r.split)
        clean = load_clean_500_from_wfdb(source_id, set_a_dir)
        noise2_500, start360 = sampler.draw_500(noise_type, split)
        noisy, meta_extra = _make_noisy_case_paper(clean, noise2_500, rng, snr_db)
        out_npz = cases_dir / f"{r.record_id}.npz"
        meta = {
            "new_record_id": str(r.record_id),
            "source_record_id": source_id,
            "kind": "paper_aligned_noisy_noise_time_isolated",
            "noise_type": noise_type,
            "noise_start_360": int(start360),
            "noise_end_360": int(start360 + round(CASE_SEC * FS_NOISE_IN)),
            "noise_region": split,
            "noise_start_stride_s": float(noise_stride_s),
            "fs_noise_in": FS_NOISE_IN,
            **meta_extra,
        }
        write_case_500_npz(out_npz, noisy, meta)
        audit_rows.append(
            {
                "record_id": str(r.record_id),
                "source_record_id": source_id,
                "split": split,
                "noise_type": noise_type,
                "snr_db": float(snr_db),
                "noise_record": noise_type,
                "noise_start_360": int(start360),
                "noise_end_360": int(start360 + round(CASE_SEC * FS_NOISE_IN)),
                "noise_region": split,
                "noise_start_stride_s": float(noise_stride_s),
                "pca_mode": "paper_pca_deterministic_third_axis",
                "dower_mode": "inverse_dower_12x3",
                "out_npz_500": str(out_npz),
            }
        )

    out_split.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    provisional.to_csv(out_split, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
    _plot_label_counts(provisional, qc_png)
    overlap = audit_noise_overlap(audit_csv, overlap_csv)
    if not overlap.empty and int(overlap["n_overlaps"].sum()) != 0:
        raise AssertionError(f"Noise intervals overlap across splits; see {overlap_csv}")
    return {
        "step": "isolated_paper_balanced_seta",
        "skipped": False,
        "outputs": [str(out_split), str(audit_csv), str(overlap_csv), str(qc_png)],
    }
