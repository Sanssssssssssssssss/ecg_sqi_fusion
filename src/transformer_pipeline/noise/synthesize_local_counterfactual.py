from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.utils.paths import project_root
    from src.transformer_pipeline.noise.make_rr_noise_level import detect_r_peaks
    from src.transformer_pipeline.noise.synthesize_snr_dataset import (
        FS,
        N,
        WIN_SEC,
        add_noise_at_snr,
        load_noise_125,
        make_noise_segment,
        parse_noise_kinds,
        split_noise_ranges,
    )
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root
    from src.transformer_pipeline.noise.make_rr_noise_level import detect_r_peaks
    from src.transformer_pipeline.noise.synthesize_snr_dataset import (
        FS,
        N,
        WIN_SEC,
        add_noise_at_snr,
        load_noise_125,
        make_noise_segment,
        parse_noise_kinds,
        split_noise_ranges,
    )

logger = logging.getLogger(__name__)

PLACEMENTS = ("qrs_overlap", "tst_overlap", "noncritical", "uniform")
SNR_PROFILES = (1.5, 4.0, 8.0)
LABEL_TO_INT = {"good": 0, "medium": 1, "bad": 2}


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer_e6_local_counterfactual")
    source_artifact_dir = _path(params.get("source_artifact_dir"), root / "outputs/transformer")
    nstdb = _path(params.get("nstdb_root"), root / "data" / "physionet" / "nstdb")
    seed = int(params.get("seed", 0))
    force = bool(params.get("force", False))
    max_train_clean = _none_or_int(params.get("max_train_clean"), 4000)
    max_val_clean = _none_or_int(params.get("max_val_clean"), 800)
    max_test_clean = _none_or_int(params.get("max_test_clean"), 800)
    noise_kinds = parse_noise_kinds(params.get("noise_kinds", "em,ma,bw,mix"))

    x_npz = source_artifact_dir / "segments" / "ptbxl_leadI_x_10s_125hz.npz"
    split_csv = source_artifact_dir / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
    out_dir = artifact_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_clean = out_dir / "synth_10s_125hz_clean.npz"
    out_noisy = out_dir / "synth_10s_125hz_noisy.npz"
    out_mask = out_dir / "synth_10s_125hz_local_mask.npz"
    out_lbl = out_dir / "synth_10s_125hz_labels.csv"
    out_summary = out_dir / "local_counterfactual_summary.json"
    outputs = [out_clean, out_noisy, out_mask, out_lbl, out_summary]
    if all(path.exists() for path in outputs) and not force:
        logger.info("local_counterfactual: outputs exist -> skip (set --force to rerun)")
        return {"step": "local_counterfactual", "skipped": True, "outputs": [str(p) for p in outputs]}

    X_all = np.load(x_npz)["X"].astype(np.float32)
    df_split = pd.read_csv(split_csv)
    need_cols = {"seg_id", "ecg_id", "npz_index", "split"}
    if not need_cols.issubset(df_split.columns):
        raise ValueError(f"split csv missing columns: {sorted(need_cols - set(df_split.columns))}")

    rng = np.random.default_rng(seed)
    tracks = {"em": load_noise_125(nstdb / "em"), "ma": load_noise_125(nstdb / "ma")}
    bw_path = nstdb / "bw"
    tracks["bw"] = load_noise_125(bw_path) if bw_path.with_suffix(".hea").exists() else tracks["ma"]
    ranges = {name: split_noise_ranges(len(track)) for name, track in tracks.items()}

    x_clean_rows: list[np.ndarray] = []
    x_noisy_rows: list[np.ndarray] = []
    local_masks: list[np.ndarray] = []
    critical_masks: list[np.ndarray] = []
    label_rows: list[dict[str, object]] = []

    split_limits = {"train": max_train_clean, "val": max_val_clean, "test": max_test_clean}
    group_id = 0
    for split_name in ("train", "val", "test"):
        split_rows = df_split[df_split["split"].astype(str) == split_name].reset_index(drop=True)
        split_rows = _sample_rows(split_rows, split_limits[split_name], rng)
        for row_i, row in enumerate(split_rows.itertuples(index=False), start=1):
            if verbose and row_i % 500 == 0:
                logger.info("E6 synth %s progress: %d/%d", split_name, row_i, len(split_rows))
            clean = X_all[int(row.npz_index)].astype(np.float32, copy=False)
            qrs_mask, tst_mask, critical_mask, peaks = critical_region_masks(clean)
            noise_kind = noise_kinds[group_id % len(noise_kinds)]
            snr_db = float(SNR_PROFILES[group_id % len(SNR_PROFILES)])
            raw_noise = make_noise_segment(noise_kind, split_name, tracks, ranges, rng)
            for placement in PLACEMENTS:
                envelope = placement_envelope(
                    placement=placement,
                    qrs_mask=qrs_mask,
                    tst_mask=tst_mask,
                    critical_mask=critical_mask,
                    rng=rng,
                )
                placed_noise = raw_noise * envelope
                if float(np.mean(placed_noise * placed_noise)) < 1e-12:
                    placed_noise = raw_noise.copy()
                    envelope = np.ones(N, dtype=np.float32)
                noisy = add_noise_at_snr(clean, placed_noise, snr_db)
                injected_noise = (noisy - clean).astype(np.float32)
                crit_ratio = critical_noise_energy_ratio(injected_noise, critical_mask)
                beat_frac = contaminated_beat_fraction(envelope, peaks)
                y_class = local_quality_label(snr_db, crit_ratio, beat_frac)
                old_oracle = old_snr_label(snr_db)

                x_clean_rows.append(clean.astype(np.float32, copy=False))
                x_noisy_rows.append(noisy.astype(np.float32, copy=False))
                local_masks.append(envelope.astype(np.float32, copy=False))
                critical_masks.append(critical_mask.astype(np.float32, copy=False))
                label_rows.append(
                    {
                        "idx": len(label_rows),
                        "seg_id": int(row.seg_id),
                        "ecg_id": int(row.ecg_id),
                        "split": split_name,
                        "y_class": y_class,
                        "snr_db": snr_db,
                        "noise_kind": noise_kind,
                        "placement": placement,
                        "counterfactual_group": int(group_id),
                        "critical_noise_energy_ratio": float(crit_ratio),
                        "contaminated_beat_fraction": float(beat_frac),
                        "old_snr_oracle_class": old_oracle,
                        "fs": FS,
                        "window_sec": WIN_SEC,
                        "source_npz_index": int(row.npz_index),
                        "sample_source": "e6_local_counterfactual",
                    }
                )
            group_id += 1

    X_clean = np.stack(x_clean_rows, axis=0).astype(np.float32)
    X_noisy = np.stack(x_noisy_rows, axis=0).astype(np.float32)
    M = np.stack(local_masks, axis=0).astype(np.float32)
    C = np.stack(critical_masks, axis=0).astype(np.float32)
    labels = pd.DataFrame(label_rows)
    labels["idx"] = np.arange(len(labels), dtype=int)

    np.savez(out_clean, X_clean=X_clean)
    np.savez(out_noisy, X_noisy=X_noisy)
    np.savez(out_mask, M=M, critical_mask=C)
    labels.to_csv(out_lbl, index=False)

    summary = build_summary(labels, X_clean, X_noisy)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("E6 rows=%d groups=%d", len(labels), group_id)
    for split_name in ("train", "val", "test"):
        d = labels[labels["split"] == split_name]
        logger.info("[%s] y_class counts: %s", split_name, d["y_class"].value_counts().sort_index().to_dict())
        logger.info("[%s] placement counts: %s", split_name, d["placement"].value_counts().sort_index().to_dict())
    logger.info("measured SNR oracle accuracy: %.6f", summary["measured_snr_oracle_acc"])
    logger.info("same-SNR mixed-label groups: %d/%d", summary["mixed_label_counterfactual_groups"], summary["counterfactual_groups"])
    for path in outputs:
        logger.info("Saved: %s", _display(path, root))
    return {
        "step": "local_counterfactual",
        "skipped": False,
        "outputs": [str(p) for p in outputs],
        "rows": int(len(labels)),
        "measured_snr_oracle_acc": summary["measured_snr_oracle_acc"],
        "mixed_label_counterfactual_groups": summary["mixed_label_counterfactual_groups"],
    }


def critical_region_masks(clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        peaks = detect_r_peaks(clean, FS)
    except Exception:
        peaks = np.array([], dtype=int)
    if len(peaks) < 3:
        peaks = np.arange(FS, N - FS, FS, dtype=int)

    qrs = np.zeros(N, dtype=np.float32)
    tst = np.zeros(N, dtype=np.float32)
    for r in peaks:
        _mark(qrs, int(r) - int(0.06 * FS), int(r) + int(0.10 * FS))
        _mark(tst, int(r) + int(0.12 * FS), int(r) + int(0.42 * FS))
    critical = np.maximum(qrs, tst)
    return qrs, tst, critical.astype(np.float32), peaks.astype(int)


def placement_envelope(
    *,
    placement: str,
    qrs_mask: np.ndarray,
    tst_mask: np.ndarray,
    critical_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if placement == "qrs_overlap":
        env = qrs_mask.copy()
    elif placement == "tst_overlap":
        env = tst_mask.copy()
    elif placement == "noncritical":
        env = 1.0 - critical_mask
        env = dropout_noncritical_regions(env, rng)
    elif placement == "uniform":
        env = np.ones(N, dtype=np.float32)
    else:
        raise ValueError(f"unknown placement: {placement}")
    return smooth_envelope(np.clip(env, 0.0, 1.0).astype(np.float32))


def dropout_noncritical_regions(env: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Keep the noncritical placement broad but not perfectly periodic.
    keep = np.zeros_like(env, dtype=np.float32)
    nonzero = np.flatnonzero(env > 0.5)
    if len(nonzero) == 0:
        return env.astype(np.float32)
    target_width = int(rng.integers(int(0.6 * FS), int(1.4 * FS)))
    for _ in range(4):
        c = int(rng.choice(nonzero))
        _mark(keep, c - target_width // 2, c + target_width // 2)
    return np.minimum(env, keep).astype(np.float32)


def smooth_envelope(env: np.ndarray) -> np.ndarray:
    if float(env.max()) <= 0.0:
        return env.astype(np.float32)
    k = max(5, int(0.08 * FS))
    if k % 2 == 0:
        k += 1
    win = np.hanning(k).astype(np.float32)
    win = win / max(1e-6, float(win.sum()))
    out = np.convolve(env.astype(np.float32), win, mode="same")
    out = out / max(1e-6, float(out.max()))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def critical_noise_energy_ratio(noise: np.ndarray, critical_mask: np.ndarray) -> float:
    e = noise.astype(np.float64) ** 2
    total = float(e.sum()) + 1e-12
    return float((e * critical_mask.astype(np.float64)).sum() / total)


def contaminated_beat_fraction(envelope: np.ndarray, peaks: np.ndarray) -> float:
    if len(peaks) == 0:
        return 0.0
    contaminated = 0
    for r in peaks:
        left = max(0, int(r) - int(0.08 * FS))
        right = min(N, int(r) + int(0.42 * FS))
        if right <= left:
            continue
        if float(envelope[left:right].mean()) >= 0.20:
            contaminated += 1
    return float(contaminated) / float(max(1, len(peaks)))


def local_quality_label(snr_db: float, critical_ratio: float, beat_frac: float) -> str:
    if snr_db < 2.5:
        score = 1.3
    elif snr_db < 6.5:
        score = 0.9
    elif snr_db < 10.0:
        score = 0.25
    else:
        score = 0.0
    if critical_ratio >= 0.55:
        score += 1.1
    elif critical_ratio >= 0.20:
        score += 0.5
    if beat_frac >= 0.60:
        score += 0.6
    elif beat_frac >= 0.25:
        score += 0.3
    if snr_db >= 2.5 and critical_ratio < 0.12 and beat_frac < 0.20:
        score -= 0.9
    if score < 0.75:
        return "good"
    if score < 2.20:
        return "medium"
    return "bad"


def old_snr_label(snr_db: float) -> str:
    if snr_db >= 10.0:
        return "good"
    if snr_db >= 0.0:
        return "medium"
    return "bad"


def build_summary(labels: pd.DataFrame, X_clean: np.ndarray, X_noisy: np.ndarray) -> dict[str, Any]:
    snr_measured = 10.0 * np.log10(
        (np.mean(X_clean * X_clean, axis=1) + 1e-12)
        / (np.mean((X_noisy - X_clean) ** 2, axis=1) + 1e-12)
    )
    oracle = np.array([old_snr_label(float(v)) for v in snr_measured], dtype=object)
    y = labels["y_class"].astype(str).to_numpy()
    group_labels = labels.groupby("counterfactual_group")["y_class"].nunique()
    same_snr_mixed = labels.groupby("counterfactual_group")["snr_db"].nunique().eq(1) & group_labels.gt(1)
    return {
        "rows": int(len(labels)),
        "counterfactual_groups": int(labels["counterfactual_group"].nunique()),
        "mixed_label_counterfactual_groups": int(same_snr_mixed.sum()),
        "measured_snr_oracle_acc": float(np.mean(oracle == y)),
        "measured_snr_db": {
            "min": float(np.min(snr_measured)),
            "mean": float(np.mean(snr_measured)),
            "max": float(np.max(snr_measured)),
        },
        "split_y_class_counts": {
            sp: {str(k): int(v) for k, v in d["y_class"].value_counts().sort_index().to_dict().items()}
            for sp, d in labels.groupby("split")
        },
        "split_placement_counts": {
            sp: {str(k): int(v) for k, v in d["placement"].value_counts().sort_index().to_dict().items()}
            for sp, d in labels.groupby("split")
        },
        "label_by_placement": {
            str(p): {str(k): int(v) for k, v in d["y_class"].value_counts().sort_index().to_dict().items()}
            for p, d in labels.groupby("placement")
        },
    }


def _mark(mask: np.ndarray, left: int, right: int) -> None:
    left = max(0, int(left))
    right = min(len(mask), int(right))
    if right > left:
        mask[left:right] = 1.0


def _sample_rows(df: pd.DataFrame, limit: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    idx = rng.choice(np.arange(len(df)), size=limit, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def _none_or_int(value: object, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    n = int(value)
    return None if n <= 0 else n


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the E6 local-counterfactual transformer benchmark.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_e6_local_counterfactual")
    parser.add_argument("--source_artifact_dir", default="outputs/transformer")
    parser.add_argument("--nstdb_root", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_clean", type=int, default=4000)
    parser.add_argument("--max_val_clean", type=int, default=800)
    parser.add_argument("--max_test_clean", type=int, default=800)
    parser.add_argument("--noise_kinds", default="em,ma,bw,mix")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
