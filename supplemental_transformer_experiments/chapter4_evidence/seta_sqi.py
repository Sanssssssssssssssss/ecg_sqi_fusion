from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.sqi_pipeline.features.make_record84 import run as record84_run
from src.sqi_pipeline.features.norm_record84_ks import run as norm_run
from src.sqi_pipeline.qrs.run_qrs_cache import run as qrs_run
from supplemental_transformer_experiments.sqi12_gapfill import run as sqi12

from .common import LEADS_12, ROOT, SYNTH_QUOTA, Paths, dry, ensure_dirs, stable_record_ids, write_json


ARMS = ("native_imbalanced", "fixed_synthetic", "quota_draw", "smc_gapfill")


def arm_dir(paths: Paths, arm: str) -> Path:
    return paths.seta_arms / arm


def _features_ready(root: Path) -> bool:
    return (
        (root / "features" / "record84_norm.parquet").exists()
        and (root / "features" / "record84.parquet").exists()
        and (root / "splits" / "split.csv").exists()
    )


def _load_smc(paths: Paths) -> tuple[pd.DataFrame, np.ndarray]:
    protocol = pd.read_csv(paths.seta / "data" / "protocol_gapfill.csv")
    X = np.load(paths.seta / "data" / "signals.npz", allow_pickle=True)["X"].astype(np.float32)
    if len(protocol) != X.shape[0]:
        raise ValueError("Set-A protocol/signals row mismatch")
    return protocol, X


def _original_only(protocol: pd.DataFrame, X: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    mask = protocol["generated"].astype(int).eq(0).to_numpy()
    return protocol.loc[mask].reset_index(drop=True), X[mask].astype(np.float32)


def _paper_synthetic_sources() -> tuple[pd.DataFrame, Path]:
    paper_root = ROOT / "outputs" / "sqi_paper_aligned"
    split_csv = paper_root / "splits" / "split_seta_seed0_paper_balanced.csv"
    signal_dir = paper_root / "resampled_125"
    if not split_csv.exists():
        raise FileNotFoundError(f"missing paper synthetic split: {split_csv}")
    if not signal_dir.exists():
        raise FileNotFoundError(f"missing paper synthetic signals: {signal_dir}")
    return pd.read_csv(split_csv), signal_dir


def _load_paper_sig125(signal_dir: Path, record_id: str) -> np.ndarray:
    path = signal_dir / f"{record_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing paper synthetic signal: {path}")
    z = np.load(path, allow_pickle=True)
    sig = z["sig_125"].astype(np.float32)
    if sig.shape != (1250, 12):
        raise ValueError(f"paper synthetic signal shape mismatch for {record_id}: {sig.shape}")
    return sig


def _fixed_synthetic(protocol: pd.DataFrame, X: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    original, Xo = _original_only(protocol, X)
    rng = np.random.default_rng(seed + 4401)
    train_bad = int((original["split"].eq("train") & original["y"].eq(0)).sum())
    train_acc = int((original["split"].eq("train") & original["y"].eq(1)).sum())
    gap = train_acc - train_bad
    if gap != 383:
        raise ValueError(f"unexpected Set-A gap for fixed_synthetic: {gap}")
    paper_split, paper_signal_dir = _paper_synthetic_sources()
    paper_train = paper_split.loc[
        paper_split["split"].eq("train")
        & paper_split["is_augmented"].astype(int).eq(1)
        & paper_split["noise_type"].isin(["em", "ma"])
    ].copy()
    if len(paper_train) < gap:
        raise ValueError(f"paper synthetic train pool too small: {len(paper_train)} < {gap}")
    chosen = paper_train.iloc[rng.choice(len(paper_train), size=gap, replace=False)].reset_index(drop=True)
    rows = [original]
    xs = [Xo]
    gen_rows = []
    gen_x = []
    for j, src in chosen.iterrows():
        record_id = str(src.record_id)
        source_record_id = str(src.source_record_id)
        noise_type = str(src.noise_type)
        gen_rows.append(
            {
                **{c: "" for c in original.columns},
                "row_idx": len(original) + j,
                "record_id": record_id,
                "source_record_id": source_record_id,
                "split": "train",
                "quality_record": "paper_noisy_unacceptable",
                "label": "unacceptable",
                "y": 0,
                "generated": 1,
                "candidate_type": f"paper_{noise_type}",
                "donor_split": "paper_train",
                "style_anchor_record_id": source_record_id,
                "noise_type": noise_type,
                "snr_db": float(src.snr_db),
            }
        )
        gen_x.append(_load_paper_sig125(paper_signal_dir, record_id))
    rows.append(pd.DataFrame(gen_rows))
    xs.append(np.stack(gen_x, axis=0).astype(np.float32))
    out = pd.concat(rows, ignore_index=True)
    out["row_idx"] = np.arange(len(out), dtype=int)
    return out, np.concatenate(xs, axis=0).astype(np.float32)


def _candidate_pool_signals(paths: Paths, seed: int, max_ptb: int) -> np.ndarray:
    pool = pd.read_csv(paths.seta / "data" / "candidate_pool.csv")
    pool_npz = paths.seta / "data" / "candidate_pool_signals.npz"
    if not pool_npz.exists():
        raise FileNotFoundError(
            f"missing {pool_npz}; rerun seta-build after the candidate-pool signal export fix"
        )
    X = np.load(pool_npz, allow_pickle=True)["X"].astype(np.float32)
    if X.shape[0] != len(pool):
        raise ValueError(f"candidate_pool_signals row mismatch: {X.shape[0]} != {len(pool)}")
    return X


def _quota_draw(paths: Paths, protocol: pd.DataFrame, X: np.ndarray, seed: int, max_ptb: int) -> tuple[pd.DataFrame, np.ndarray]:
    original, Xo = _original_only(protocol, X)
    pool = pd.read_csv(paths.seta / "data" / "candidate_pool.csv")
    pool_x = _candidate_pool_signals(paths, seed=seed, max_ptb=max_ptb)
    if len(pool) != pool_x.shape[0]:
        raise ValueError("candidate pool/signals row mismatch")
    rng = np.random.default_rng(seed + 5107)
    selected_idx: list[int] = []
    for label, n in SYNTH_QUOTA.items():
        idx = pool.index[pool["candidate_type"].eq(label)].to_numpy()
        if len(idx) < n:
            raise ValueError(f"quota_draw pool too small for {label}: {len(idx)} < {n}")
        selected_idx.extend(rng.choice(idx, size=n, replace=False).astype(int).tolist())
    selected = pool.iloc[selected_idx].reset_index(drop=True)
    selected_x = pool_x[np.asarray(selected_idx, dtype=int)]
    gen_rows = []
    for j, row in selected.iterrows():
        gen_rows.append(
            {
                **{c: "" for c in original.columns},
                "row_idx": len(original) + j,
                "record_id": str(row.record_id),
                "source_record_id": str(row.source_record_id),
                "split": "train",
                "quality_record": "generated_unacceptable",
                "label": "unacceptable",
                "y": 0,
                "generated": 1,
                "candidate_type": str(row.candidate_type),
                "donor_split": "ptb" if str(row.candidate_type) == "ptb12_morph" else "train",
                "ptb_ecg_id": row.get("ptb_ecg_id", ""),
                "ptb_patient_id": row.get("ptb_patient_id", ""),
                "style_anchor_record_id": row.get("style_anchor_record_id", ""),
            }
        )
    out = pd.concat([original, pd.DataFrame(gen_rows)], ignore_index=True)
    out["row_idx"] = np.arange(len(out), dtype=int)
    return out, np.concatenate([Xo, selected_x], axis=0).astype(np.float32)


def _arm_protocol(paths: Paths, arm: str, seed: int, max_ptb: int) -> tuple[pd.DataFrame, np.ndarray]:
    protocol, X = _load_smc(paths)
    if arm == "smc_gapfill":
        return protocol.copy().reset_index(drop=True), X
    if arm == "native_imbalanced":
        return _original_only(protocol, X)
    if arm == "fixed_synthetic":
        return _fixed_synthetic(protocol, X, seed)
    if arm == "quota_draw":
        return _quota_draw(paths, protocol, X, seed, max_ptb)
    raise ValueError(f"unknown arm: {arm}")


def _write_workspace(root: Path, protocol: pd.DataFrame, X: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ["splits", "resampled_125", "qrs", "features"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    rid = stable_record_ids(len(protocol))
    work = protocol.copy().reset_index(drop=True)
    work["orig_record_id"] = work["record_id"].astype(str)
    work["record_id"] = rid
    y_pm1 = np.where(work["y"].astype(int).eq(1), 1, -1).astype(int)
    split = pd.DataFrame(
        {
            "record_id": rid,
            "y": y_pm1,
            "split": work["split"].astype(str),
            "seed": 0,
            "quality_record": work["quality_record"].astype(str),
            "is_augmented": work["generated"].astype(int),
            "source_record_id": work["source_record_id"].astype(str),
            "candidate_type": work["candidate_type"].astype(str),
            "orig_record_id": work["orig_record_id"].astype(str),
        }
    )
    work.to_csv(root / "protocol.csv", index=False)
    split.to_csv(root / "splits" / "split.csv", index=False)
    construction = pd.concat(
        [
            work[["record_id", "orig_record_id", "source_record_id", "split", "quality_record", "y", "generated", "candidate_type"]].reset_index(drop=True),
            sqi12.feature_frame(X).reset_index(drop=True),
        ],
        axis=1,
    )
    construction.to_csv(root / "construction_features.csv", index=False)
    np.savez_compressed(root / "signals.npz", X=X.astype(np.float32), leads=np.asarray(LEADS_12, dtype=object), fs=np.int32(125))
    for i, record_id in enumerate(rid):
        np.savez_compressed(
            root / "resampled_125" / f"{record_id}.npz",
            sig_125=X[i].astype(np.float32),
            fs=np.int32(125),
            leads=np.asarray(LEADS_12, dtype=object),
        )


def _paper_detector_paths() -> tuple[Path, Path]:
    tool_dir = ROOT / "outputs" / "sqi_paper_aligned" / "qrs" / "tools" / "bin"
    wqrs = tool_dir / "wqrs.exe"
    eplimited = tool_dir / "eplimited.exe"
    missing = [str(p) for p in (wqrs, eplimited) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Chapter 4 paper-SQI rerun needs the checked paper QRS tools. "
            f"Missing: {missing}"
        )
    return wqrs, eplimited


def _run_sqi(root: Path, *, force: bool) -> None:
    split_csv = root / "splits" / "split.csv"
    wqrs, eplimited = _paper_detector_paths()
    qrs_run(
        {
            "force": force,
            "split_csv": str(split_csv),
            "resampled_dir": str(root / "resampled_125"),
            "out_dir": str(root / "qrs"),
            "detector_profile": "paper",
            "wqrs_exe": str(wqrs),
            "eplimited_exe": str(eplimited),
            "paper_qrs_work_dir": str(root / "qrs" / "_wfdb_tmp"),
            "qrs_summary_csv": str(root / "qrs" / "qrs_summary.csv"),
        }
    )
    qrs_summary = root / "qrs" / "qrs_summary.csv"
    if not qrs_summary.exists():
        raise RuntimeError(f"missing QRS summary: {qrs_summary}")
    qrs = pd.read_csv(qrs_summary)
    if qrs[["n_detector1", "n_detector2"]].isna().any().any():
        raise RuntimeError(f"QRS detector count contains NaN: {qrs_summary}")
    record84_run(
        {
            "force": force,
            "split_csv": str(split_csv),
            "resampled_dir": str(root / "resampled_125"),
            "qrs_dir": str(root / "qrs"),
            "out_dir": str(root / "features"),
            "sqi_mode": "paper",
            "isqi_use": "r1",
        }
    )
    norm_run(
        {
            "force": force,
            "split_csv": str(split_csv),
            "in_parquet": str(root / "features" / "record84.parquet"),
            "out_dir": str(root / "features"),
            "out_stats": str(root / "features" / "norm_stats_seed0.json"),
            "out_parquet": str(root / "features" / "record84_norm.parquet"),
        }
    )


def _seed_original_qrs(root: Path, native_root: Path) -> int:
    if root.resolve() == native_root.resolve():
        return 0
    src_dir = native_root / "qrs"
    dst_dir = root / "qrs"
    if not src_dir.exists():
        return 0
    split = pd.read_csv(root / "splits" / "split.csv")
    original_ids = split.loc[split["is_augmented"].astype(int).eq(0), "record_id"].astype(str).tolist()
    copied = 0
    for rid in original_ids:
        src = src_dir / f"{rid}.npz"
        dst = dst_dir / f"{rid}.npz"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def run(paths: Paths, *, execute: bool, force: bool, seed: int = 0, max_ptb: int = 2400) -> dict[str, Any]:
    if not execute:
        dry("seta-sqi", paths)
        return {"step": "seta-sqi", "skipped": True}
    ensure_dirs(paths)
    out: dict[str, Any] = {"arms": {}}
    native_root = arm_dir(paths, "native_imbalanced")
    for arm in ARMS:
        root = arm_dir(paths, arm)
        if _features_ready(root) and not force:
            print(f"[seta-sqi] exists: {arm}")
            qrs_seeded = 0
        else:
            protocol, X = _arm_protocol(paths, arm, seed, max_ptb)
            _write_workspace(root, protocol, X)
            qrs_seeded = _seed_original_qrs(root, native_root)
            _run_sqi(root, force=(qrs_seeded == 0))
        split = pd.read_csv(root / "splits" / "split.csv")
        qrs = pd.read_csv(root / "qrs" / "qrs_summary.csv")
        out["arms"][arm] = {
            "rows": int(len(split)),
            "split_counts": {f"{a}_{b}": int(v) for (a, b), v in split.groupby(["split", "quality_record"]).size().items()},
            "candidate_type_counts": {str(k): int(v) for k, v in split["candidate_type"].value_counts().items()},
            "qrs_rows": int(len(qrs)),
            "qrs_seeded_from_native": int(qrs_seeded),
            "qrs_detector_profile": sorted(qrs["detector_profile"].astype(str).unique().tolist()),
            "feature_rows": int(len(pd.read_parquet(root / "features" / "record84_norm.parquet", columns=["record_id"]))),
        }
    write_json(paths.reports / "seta_sqi_audit.json", out)
    print(json.dumps(out, indent=2))
    return {"step": "seta-sqi", "skipped": False, "outputs": out}
