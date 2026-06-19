"""Combine PTB-only boundary replay banks for cross-dataset training.

The resulting bank contains only PTB synthetic train waveforms.  It concatenates
the bad-distribution matched bank and the good/medium boundary bank so the
existing distribution-bank trainer can replay a single NPZ/manifest pair.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
ANALYSIS_DIR = ROOT / "outputs" / "external_benchmarks" / RUN_TAG / "analysis" / "good_medium_geometry_repair"
OUT_DIR = ANALYSIS_DIR / "ptb_combo_boundary_bank"
RUNNER = ANALYSIS_DIR / "run_ptb_bad_alignment_cross_dataset.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("combo_bank_runner_ref", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_bank(signals_path: Path, role_prefix: str, labels_by_idx: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    if not signals_path.exists():
        raise FileNotFoundError(signals_path)
    manifest_path = signals_path.with_name(signals_path.name.replace("_signals.npz", "_manifest.csv"))
    if not manifest_path.exists():
        # bad distribution bank keeps the shorter canonical manifest name.
        alt = signals_path.with_name("ptb_bad_distribution_matched_manifest.csv")
        if alt.exists():
            manifest_path = alt
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest for {signals_path}")
    x = np.load(signals_path)["X"].astype(np.float32)
    m = pd.read_csv(manifest_path)
    if len(m) != len(x):
        raise ValueError(f"bank length mismatch: {signals_path} has {len(x)} signals vs {len(m)} manifest rows")
    if "source_idx" not in m.columns:
        raise ValueError(f"manifest missing source_idx: {manifest_path}")
    m = m.copy()
    if "y_class" not in m.columns:
        m["y_class"] = [labels_by_idx.loc[int(idx)] for idx in m["source_idx"].to_numpy(dtype=np.int64)]
    m["combo_source_bank"] = role_prefix
    if "bank_role" not in m.columns:
        m["bank_role"] = role_prefix
    else:
        m["bank_role"] = role_prefix + "|" + m["bank_role"].astype(str)
    if "stress" not in m.columns:
        m["stress"] = (m["y_class"].astype(str) == "bad").astype(float)
    return x, m


def block_key(frame: pd.DataFrame) -> pd.Series:
    keys = frame["bank_role"].fillna("").astype(str)
    if "profile_block" in frame.columns:
        profile = frame["profile_block"].fillna("").astype(str)
        keys = np.where(profile.str.len().to_numpy() > 0, keys + "|" + profile, keys)
    return pd.Series(keys, index=frame.index).replace("", "unknown")


def balanced_replay_sample(
    x: np.ndarray,
    manifest: pd.DataFrame,
    rng: np.random.Generator,
    target_per_class: int,
    balance_blocks: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Resample the replay bank so rare classes/blocks are actually visible.

    This is intentionally a data-level replay balance, not a class-weight trick:
    sparse boundary blocks may be repeated with replacement, and the manifest
    records the source row and repeat id for auditability.
    """
    y = manifest["y_class"].astype(str).to_numpy()
    classes = sorted(pd.unique(y))
    counts = {cls: int(np.sum(y == cls)) for cls in classes}
    target = int(target_per_class) if int(target_per_class) > 0 else max(counts.values())
    picks: list[int] = []
    repeat_ids: list[int] = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        if cls_idx.size == 0:
            continue
        if not balance_blocks:
            chosen = rng.choice(cls_idx, size=target, replace=cls_idx.size < target)
            picks.extend(chosen.tolist())
            repeat_ids.extend(list(range(len(chosen))))
            continue
        cls_keys = block_key(manifest.iloc[cls_idx].reset_index(drop=True)).to_numpy()
        unique_blocks = sorted(pd.unique(cls_keys))
        base_quota = target // max(1, len(unique_blocks))
        remainder = target - base_quota * len(unique_blocks)
        for b_i, block in enumerate(unique_blocks):
            block_pos = cls_idx[np.flatnonzero(cls_keys == block)]
            quota = base_quota + (1 if b_i < remainder else 0)
            if quota <= 0:
                continue
            chosen = rng.choice(block_pos, size=quota, replace=block_pos.size < quota)
            picks.extend(chosen.tolist())
            repeat_ids.extend(list(range(len(chosen))))
    picks_arr = np.asarray(picks, dtype=np.int64)
    order = rng.permutation(len(picks_arr))
    picks_arr = picks_arr[order]
    repeat_arr = np.asarray(repeat_ids, dtype=np.int64)[order]
    out_x = x[picks_arr].astype(np.float32)
    out_m = manifest.iloc[picks_arr].reset_index(drop=True).copy()
    out_m["replay_source_row"] = picks_arr
    out_m["replay_repeat_id"] = repeat_arr
    out_m["replay_balanced"] = True
    return out_x, out_m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bad-bank",
        type=Path,
        default=ANALYSIS_DIR / "ptb_bad_distribution_match" / "ptb_bad_distribution_matched_signals.npz",
    )
    parser.add_argument(
        "--gm-bank",
        type=Path,
        default=ANALYSIS_DIR / "ptb_gm_boundary_replay_bank" / "ptb_gm_boundary_replay_clean_drop_train_signals.npz",
    )
    parser.add_argument("--bad-frac", type=float, default=1.0)
    parser.add_argument("--extra-bad-bank", type=Path, default=Path(""))
    parser.add_argument("--extra-bad-frac", type=float, default=0.0)
    parser.add_argument("--gm-frac", type=float, default=1.0)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--target-per-class", type=int, default=0)
    parser.add_argument("--balance-blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=2026061907)
    parser.add_argument("--stem", type=str, default="ptb_combo_bad_gm_boundary_clean_drop_train")
    args = parser.parse_args()

    runner = load_runner()
    labels = pd.read_csv(runner.ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx")
    labels_by_idx = labels.set_index("idx")["y_class"].astype(str)

    rng = np.random.default_rng(int(args.seed))
    parts = []
    manifests = []
    bank_specs = [(args.bad_bank, "bad_distribution", args.bad_frac)]
    if str(args.extra_bad_bank) and float(args.extra_bad_frac) > 0:
        bank_specs.append((args.extra_bad_bank, "bad_controlled_distribution", args.extra_bad_frac))
    bank_specs.append((args.gm_bank, "gm_boundary", args.gm_frac))
    for path, role, frac in bank_specs:
        x, m = load_bank(path, role, labels_by_idx)
        if float(frac) <= 0:
            continue
        if float(frac) < 1.0:
            n = max(1, int(round(len(m) * float(frac))))
            take = rng.choice(np.arange(len(m)), size=n, replace=False)
            x = x[take]
            m = m.iloc[take].reset_index(drop=True)
        parts.append(x)
        manifests.append(m)

    if not parts:
        raise RuntimeError("no bank rows selected")
    out_x = np.concatenate(parts, axis=0).astype(np.float32)
    out_m = pd.concat(manifests, ignore_index=True)
    pre_balance_class_counts = out_m["y_class"].value_counts().to_dict()
    pre_balance_block_counts = out_m["bank_role"].value_counts().to_dict()
    if bool(args.balance_classes):
        out_x, out_m = balanced_replay_sample(
            out_x,
            out_m,
            rng,
            int(args.target_per_class),
            bool(args.balance_blocks),
        )
    order = rng.permutation(len(out_m))
    out_x = out_x[order]
    out_m = out_m.iloc[order].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals_path = OUT_DIR / f"{args.stem}_signals.npz"
    manifest_path = OUT_DIR / f"{args.stem}_manifest.csv"
    np.savez_compressed(signals_path, X=out_x)
    out_m.to_csv(manifest_path, index=False)
    summary = {
        "signals": str(signals_path),
        "manifest": str(manifest_path),
        "rows": int(len(out_m)),
        "class_counts": out_m["y_class"].value_counts().to_dict(),
        "role_counts": out_m["combo_source_bank"].value_counts().to_dict(),
        "bank_role_counts": out_m["bank_role"].value_counts().to_dict(),
        "pre_balance_class_counts": pre_balance_class_counts,
        "pre_balance_bank_role_counts": pre_balance_block_counts,
        "balance_classes": bool(args.balance_classes),
        "target_per_class": int(args.target_per_class),
        "balance_blocks": bool(args.balance_blocks),
        "bad_frac": float(args.bad_frac),
        "gm_frac": float(args.gm_frac),
        "seed": int(args.seed),
    }
    (OUT_DIR / f"{args.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
