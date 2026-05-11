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
    from src.transformer_pipeline.noise.synthesize_local_counterfactual import (
        PLACEMENTS,
        critical_region_masks,
        old_snr_label,
        placement_envelope,
    )
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
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.transformer_pipeline.noise.synthesize_local_counterfactual import (
        PLACEMENTS,
        critical_region_masks,
        old_snr_label,
        placement_envelope,
    )
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
    from src.utils.paths import project_root


logger = logging.getLogger(__name__)

DEFAULT_SNR_PROFILES = (-2.0, 4.0, 12.0, 16.0, 20.0)
CLASS_ORDER = ("good", "medium", "bad")


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)

    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer_e6b_balanced_local")
    source_artifact_dir = _path(params.get("source_artifact_dir"), root / "outputs/transformer")
    nstdb = _path(params.get("nstdb_root"), root / "data" / "physionet" / "nstdb")
    seed = int(params.get("seed", 0))
    force = bool(params.get("force", False))
    max_train_clean = _none_or_int(params.get("max_train_clean"), 1000)
    max_val_clean = _none_or_int(params.get("max_val_clean"), 250)
    max_test_clean = _none_or_int(params.get("max_test_clean"), 250)
    noise_kinds = parse_noise_kinds(params.get("noise_kinds", "em,ma,bw,mix"))
    snr_profiles = parse_float_tuple(params.get("snr_profiles"), DEFAULT_SNR_PROFILES)

    x_npz = source_artifact_dir / "segments" / "ptbxl_leadI_x_10s_125hz.npz"
    split_csv = source_artifact_dir / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
    out_dir = artifact_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_clean = out_dir / "synth_10s_125hz_clean.npz"
    out_noisy = out_dir / "synth_10s_125hz_noisy.npz"
    out_mask = out_dir / "synth_10s_125hz_local_mask.npz"
    out_lbl = out_dir / "synth_10s_125hz_labels.csv"
    out_summary = out_dir / "local_counterfactual_balanced_summary.json"
    outputs = [out_clean, out_noisy, out_mask, out_lbl, out_summary]
    if all(path.exists() for path in outputs) and not force:
        logger.info("E6b outputs exist -> skip (set --force to rerun)")
        return {"step": "local_counterfactual_balanced", "skipped": True, "outputs": [str(p) for p in outputs]}

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
    qrs_masks: list[np.ndarray] = []
    tst_masks: list[np.ndarray] = []
    label_rows: list[dict[str, object]] = []

    split_limits = {"train": max_train_clean, "val": max_val_clean, "test": max_test_clean}
    group_id = 0
    for split_name in ("train", "val", "test"):
        split_rows = df_split[df_split["split"].astype(str) == split_name].reset_index(drop=True)
        split_rows = sample_rows(split_rows, split_limits[split_name], rng)
        for row_i, row in enumerate(split_rows.itertuples(index=False), start=1):
            if verbose and row_i % 250 == 0:
                logger.info("E6b synth %s progress: %d/%d", split_name, row_i, len(split_rows))

            clean = X_all[int(row.npz_index)].astype(np.float32, copy=False)
            qrs_mask, tst_mask, critical_mask, peaks = critical_region_masks(clean)
            noise_kind = noise_kinds[group_id % len(noise_kinds)]
            raw_noise = make_noise_segment(noise_kind, split_name, tracks, ranges, rng)
            for snr_db in snr_profiles:
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

                    noisy = add_noise_at_snr(clean, placed_noise, float(snr_db))
                    injected_noise = (noisy - clean).astype(np.float32)
                    quality = absolute_quality_metrics(
                        clean=clean,
                        noise=injected_noise,
                        qrs_mask=qrs_mask,
                        tst_mask=tst_mask,
                        critical_mask=critical_mask,
                        peaks=peaks,
                    )
                    y_class = absolute_quality_label(quality)

                    x_clean_rows.append(clean.astype(np.float32, copy=False))
                    x_noisy_rows.append(noisy.astype(np.float32, copy=False))
                    local_masks.append(envelope.astype(np.float32, copy=False))
                    critical_masks.append(critical_mask.astype(np.float32, copy=False))
                    qrs_masks.append(qrs_mask.astype(np.float32, copy=False))
                    tst_masks.append(tst_mask.astype(np.float32, copy=False))
                    label_rows.append(
                        {
                            "idx": len(label_rows),
                            "seg_id": int(row.seg_id),
                            "ecg_id": int(row.ecg_id),
                            "split": split_name,
                            "y_class": y_class,
                            "snr_db": float(snr_db),
                            "snr_profile": format_snr_profile(float(snr_db)),
                            "noise_kind": noise_kind,
                            "placement": placement,
                            "counterfactual_group": int(group_id),
                            "global_snr_db": quality["global_snr_db"],
                            "critical_snr_db": quality["critical_snr_db"],
                            "min_beat_critical_snr_db": quality["min_beat_critical_snr_db"],
                            "qrs_snr_db": quality["qrs_snr_db"],
                            "tst_snr_db": quality["tst_snr_db"],
                            "contaminated_beat_fraction": quality["contaminated_beat_fraction"],
                            "max_consecutive_contaminated_beats": quality["max_consecutive_contaminated_beats"],
                            "old_snr_oracle_class": old_snr_label(float(snr_db)),
                            "fs": FS,
                            "window_sec": WIN_SEC,
                            "source_npz_index": int(row.npz_index),
                            "sample_source": "e6b_balanced_local_counterfactual",
                        }
                    )
            group_id += 1

    X_clean = np.stack(x_clean_rows, axis=0).astype(np.float32)
    X_noisy = np.stack(x_noisy_rows, axis=0).astype(np.float32)
    labels = pd.DataFrame(label_rows)
    labels["idx"] = np.arange(len(labels), dtype=int)
    labels = stable_split_order(labels)
    order = labels["idx"].to_numpy(dtype=np.int64)

    # Restore arrays to the shuffled label order and then assign contiguous indices.
    X_clean = X_clean[order]
    X_noisy = X_noisy[order]
    M = np.stack(local_masks, axis=0).astype(np.float32)[order]
    C = np.stack(critical_masks, axis=0).astype(np.float32)[order]
    Q = np.stack(qrs_masks, axis=0).astype(np.float32)[order]
    T = np.stack(tst_masks, axis=0).astype(np.float32)[order]
    labels["source_idx_before_shuffle"] = labels["idx"].astype(int)
    labels["idx"] = np.arange(len(labels), dtype=int)

    np.savez(out_clean, X_clean=X_clean)
    np.savez(out_noisy, X_noisy=X_noisy)
    np.savez(out_mask, M=M, critical_mask=C, qrs_mask=Q, tst_mask=T)
    labels.to_csv(out_lbl, index=False)

    summary = build_summary(labels, X_clean, X_noisy, snr_profiles=snr_profiles, noise_kinds=noise_kinds)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("E6b rows=%d groups=%d variants_per_group=%d", len(labels), group_id, len(snr_profiles) * len(PLACEMENTS))
    for split_name in ("train", "val", "test"):
        d = labels[labels["split"] == split_name]
        logger.info("[%s] y_class counts: %s", split_name, d["y_class"].value_counts().sort_index().to_dict())
        logger.info("[%s] placement counts: %s", split_name, d["placement"].value_counts().sort_index().to_dict())
    logger.info("measured SNR oracle accuracy: %.6f", summary["measured_snr_oracle_acc"])
    logger.info("complete variant groups: %d/%d", summary["complete_variant_groups"], summary["counterfactual_groups"])
    for path in outputs:
        logger.info("Saved: %s", display(path, root))

    return {
        "step": "local_counterfactual_balanced",
        "skipped": False,
        "outputs": [str(p) for p in outputs],
        "rows": int(len(labels)),
        "measured_snr_oracle_acc": summary["measured_snr_oracle_acc"],
        "complete_variant_groups": summary["complete_variant_groups"],
    }


def absolute_quality_metrics(
    *,
    clean: np.ndarray,
    noise: np.ndarray,
    qrs_mask: np.ndarray,
    tst_mask: np.ndarray,
    critical_mask: np.ndarray,
    peaks: np.ndarray,
) -> dict[str, float | int]:
    global_snr = region_snr_db(clean, noise, np.ones(N, dtype=np.float32))
    critical_snr = region_snr_db(clean, noise, critical_mask)
    qrs_snr = region_snr_db(clean, noise, qrs_mask)
    tst_snr = region_snr_db(clean, noise, tst_mask)
    beat_snrs = beat_critical_snrs(clean, noise, peaks)
    contaminated = [v < 10.0 for v in beat_snrs]
    return {
        "global_snr_db": global_snr,
        "critical_snr_db": critical_snr,
        "min_beat_critical_snr_db": float(np.min(beat_snrs)) if len(beat_snrs) else 60.0,
        "qrs_snr_db": qrs_snr,
        "tst_snr_db": tst_snr,
        "contaminated_beat_fraction": float(np.mean(contaminated)) if contaminated else 0.0,
        "max_consecutive_contaminated_beats": int(max_consecutive(contaminated)),
    }


def absolute_quality_label(metrics: dict[str, float | int]) -> str:
    global_snr = float(metrics["global_snr_db"])
    critical_snr = float(metrics["critical_snr_db"])
    min_beat = float(metrics["min_beat_critical_snr_db"])
    qrs_snr = float(metrics["qrs_snr_db"])
    tst_snr = float(metrics["tst_snr_db"])
    beat_frac = float(metrics["contaminated_beat_fraction"])
    max_run = int(metrics["max_consecutive_contaminated_beats"])

    if min_beat < 3.0 or beat_frac >= 0.60 or qrs_snr < 4.0:
        return "bad"
    if (
        min_beat < 10.0
        or 0.25 <= beat_frac < 0.60
        or max_run >= 2
        or qrs_snr < 10.0
        or tst_snr < 8.0
        or critical_snr < 10.0
        or global_snr < 8.0
    ):
        return "medium"
    return "good"


def region_snr_db(clean: np.ndarray, noise: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if int(m.sum()) <= 0:
        return 60.0
    sig = clean[m].astype(np.float64)
    err = noise[m].astype(np.float64)
    sig_power = float(np.mean(sig * sig)) + 1e-12
    noise_power = float(np.mean(err * err)) + 1e-12
    value = 10.0 * np.log10(sig_power / noise_power)
    return float(np.clip(value, -30.0, 60.0))


def beat_critical_snrs(clean: np.ndarray, noise: np.ndarray, peaks: np.ndarray) -> list[float]:
    values: list[float] = []
    for r in peaks:
        left = max(0, int(r) - int(0.08 * FS))
        right = min(N, int(r) + int(0.42 * FS))
        if right <= left:
            continue
        mask = np.zeros(N, dtype=np.float32)
        mask[left:right] = 1.0
        values.append(region_snr_db(clean, noise, mask))
    return values


def max_consecutive(items: list[bool]) -> int:
    best = 0
    cur = 0
    for item in items:
        if item:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def stable_split_order(labels: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split_name, d in labels.groupby("split", sort=False):
        parts.append(
            d.sort_values(
                ["counterfactual_group", "noise_kind", "snr_db", "placement"],
                kind="mergesort",
            )
        )
    return pd.concat(parts, ignore_index=True)


def build_summary(
    labels: pd.DataFrame,
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    *,
    snr_profiles: tuple[float, ...],
    noise_kinds: tuple[str, ...],
) -> dict[str, Any]:
    snr_measured = 10.0 * np.log10(
        (np.mean(X_clean * X_clean, axis=1) + 1e-12)
        / (np.mean((X_noisy - X_clean) ** 2, axis=1) + 1e-12)
    )
    oracle = np.array([old_snr_label(float(v)) for v in snr_measured], dtype=object)
    y = labels["y_class"].astype(str).to_numpy()
    expected_variants = len(snr_profiles) * len(PLACEMENTS)
    group_sizes = labels.groupby("counterfactual_group").size()
    complete_groups = int(group_sizes.eq(expected_variants).sum())
    split_y = split_counts(labels, "y_class")
    return {
        "rows": int(len(labels)),
        "counterfactual_groups": int(labels["counterfactual_group"].nunique()),
        "expected_variants_per_group": int(expected_variants),
        "complete_variant_groups": complete_groups,
        "snr_profiles": [float(v) for v in snr_profiles],
        "placements": list(PLACEMENTS),
        "noise_kinds": list(noise_kinds),
        "measured_snr_oracle_acc": float(np.mean(oracle == y)),
        "old_snr_profile_oracle_acc": float(np.mean(labels["old_snr_oracle_class"].astype(str).to_numpy() == y)),
        "measured_snr_db": {
            "min": float(np.min(snr_measured)),
            "mean": float(np.mean(snr_measured)),
            "max": float(np.max(snr_measured)),
        },
        "split_y_class_counts": split_y,
        "split_class_balance_ratio": {
            sp: class_balance_ratio(counts)
            for sp, counts in split_y.items()
        },
        "split_noise_kind_counts": split_counts(labels, "noise_kind"),
        "split_placement_counts": split_counts(labels, "placement"),
        "split_snr_profile_counts": split_counts(labels, "snr_profile"),
        "label_by_placement": nested_counts(labels, "placement", "y_class"),
        "label_by_snr_profile": nested_counts(labels, "snr_profile", "y_class"),
        "label_by_noise_kind": nested_counts(labels, "noise_kind", "y_class"),
        "label_by_noise_placement_snr": stratum_counts(labels),
    }


def stratum_counts(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for keys, d in labels.groupby(["noise_kind", "placement", "snr_profile"]):
        key = "|".join(str(v) for v in keys)
        out[key] = {str(k): int(v) for k, v in d["y_class"].value_counts().sort_index().to_dict().items()}
    return out


def split_counts(labels: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    return {
        str(sp): {str(k): int(v) for k, v in d[column].value_counts().sort_index().to_dict().items()}
        for sp, d in labels.groupby("split")
    }


def nested_counts(labels: pd.DataFrame, by: str, value: str) -> dict[str, dict[str, int]]:
    return {
        str(name): {str(k): int(v) for k, v in d[value].value_counts().sort_index().to_dict().items()}
        for name, d in labels.groupby(by)
    }


def class_balance_ratio(counts: dict[str, int]) -> float:
    vals = [int(counts.get(name, 0)) for name in CLASS_ORDER]
    lo = max(1, min(vals))
    hi = max(vals)
    return float(hi / lo)


def parse_float_tuple(value: object, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None or value == "":
        return default
    vals = tuple(float(v.strip()) for v in str(value).split(",") if v.strip())
    if len(vals) < 3:
        raise ValueError("snr_profiles must contain at least three values")
    return vals


def format_snr_profile(value: float) -> str:
    return f"{value:g}dB"


def sample_rows(df: pd.DataFrame, limit: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    idx = rng.choice(np.arange(len(df)), size=limit, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def _none_or_int(value: object, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def display(path: Path, root: Path) -> str:
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
    args = parse_args()
    run(vars(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the E6b balanced local-counterfactual benchmark.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_e6b_balanced_local")
    parser.add_argument("--source_artifact_dir", default="outputs/transformer")
    parser.add_argument("--nstdb_root", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_clean", type=int, default=1000)
    parser.add_argument("--max_val_clean", type=int, default=250)
    parser.add_argument("--max_test_clean", type=int, default=250)
    parser.add_argument("--noise_kinds", default="em,ma,bw,mix")
    parser.add_argument("--snr_profiles", default="-2,4,12,16,20")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
