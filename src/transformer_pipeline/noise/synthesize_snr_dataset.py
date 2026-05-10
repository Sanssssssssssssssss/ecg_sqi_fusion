from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample_poly

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root

logger = logging.getLogger(__name__)

GOOD_SNR = (16, 20)
MEDIUM_SNR = (2, 6)
BAD_SNR = (-6, -2)
SNR_BINS = {
    "good": ((16.0, 17.0), (17.0, 18.0), (18.0, 19.0), (19.0, 20.0)),
    "medium": ((2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)),
    "bad": ((-6.0, -5.0), (-5.0, -4.0), (-4.0, -3.0), (-3.0, -2.0)),
}
DEFAULT_NOISE_KINDS = ("em", "ma", "mix")

FS = 125
WIN_SEC = 10
N = FS * WIN_SEC


def load_noise_125(path_no_ext: Path) -> np.ndarray:
    sig, fields = wfdb.rdsamp(str(path_no_ext))
    fs_raw = int(round(float(fields["fs"])))
    v = sig.mean(axis=1).astype(np.float64)
    v = v - np.mean(v)
    if fs_raw != FS:
        g = math.gcd(FS, fs_raw)
        v = resample_poly(v, up=FS // g, down=fs_raw // g)
    return v.astype(np.float32)


def split_noise_ranges(length: int) -> dict[str, tuple[int, int]]:
    a = length // 3
    return {"train": (0, a), "val": (a, 2 * a), "test": (2 * a, length)}


def sample_noise_segment(
    track: np.ndarray,
    split_name: str,
    ranges: dict[str, tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    lo, hi = ranges[split_name]
    if hi - lo <= N:
        raise ValueError(f"Noise range too short for split={split_name}")
    s = int(rng.integers(lo, hi - N))
    return track[s : s + N]


def assign_balanced_classes(n: int, rng: np.random.Generator) -> np.ndarray:
    base = n // 3
    rem = n % 3
    counts = {"good": base, "medium": base, "bad": base}
    for k in ["good", "medium", "bad"][:rem]:
        counts[k] += 1
    arr = np.array(["good"] * counts["good"] + ["medium"] * counts["medium"] + ["bad"] * counts["bad"], dtype=object)
    rng.shuffle(arr)
    return arr


def sample_snr_db(y_class: str, rng: np.random.Generator) -> float:
    if y_class == "good":
        return float(rng.uniform(*GOOD_SNR))
    if y_class == "medium":
        return float(rng.uniform(*MEDIUM_SNR))
    return float(rng.uniform(*BAD_SNR))


def sample_snr_db_stratified(y_class: str, index: int, rng: np.random.Generator) -> float:
    bins = SNR_BINS[y_class]
    lo, hi = bins[index % len(bins)]
    return float(rng.uniform(lo, hi))


def add_noise_at_snr(x: np.ndarray, v: np.ndarray, snr_db: float) -> np.ndarray:
    px = float(np.mean(x * x)) + 1e-12
    pv = float(np.mean(v * v)) + 1e-12
    a = np.sqrt(px / (pv * (10.0 ** (snr_db / 10.0))))
    return (x + a * v).astype(np.float32)


def parse_noise_kinds(value: object) -> tuple[str, ...]:
    if not value:
        return DEFAULT_NOISE_KINDS
    kinds = tuple(k.strip().lower() for k in str(value).split(",") if k.strip())
    allowed = {"em", "ma", "bw", "mix", "pl", "lp", "hp", "burst", "ampmod"}
    unknown = sorted(set(kinds) - allowed)
    if unknown:
        raise ValueError(f"unknown train noise kind(s): {unknown}")
    return kinds or DEFAULT_NOISE_KINDS


def choose_balanced(items: tuple[str, ...], index: int, rng: np.random.Generator) -> str:
    offset = int(rng.integers(0, len(items))) if index == 0 and len(items) > 1 else 0
    return items[(index + offset) % len(items)]


def make_noise_segment(
    kind: str,
    split_name: str,
    tracks: dict[str, np.ndarray],
    ranges: dict[str, dict[str, tuple[int, int]]],
    rng: np.random.Generator,
) -> np.ndarray:
    if kind in {"em", "ma", "bw"}:
        return sample_noise_segment(tracks[kind], split_name, ranges[kind], rng).astype(np.float32)
    if kind == "mix":
        return (
            sample_noise_segment(tracks["em"], split_name, ranges["em"], rng)
            + sample_noise_segment(tracks["ma"], split_name, ranges["ma"], rng)
        ).astype(np.float32)
    if kind == "pl":
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        t = np.arange(N, dtype=np.float32) / float(FS)
        return np.sin(2.0 * np.pi * 50.0 * t + phase).astype(np.float32)
    if kind == "lp":
        white = rng.normal(0.0, 1.0, size=N).astype(np.float32)
        b, a = butter(2, 8.0, btype="lowpass", fs=FS)
        return filtfilt(b, a, white).astype(np.float32)
    if kind == "hp":
        white = rng.normal(0.0, 1.0, size=N).astype(np.float32)
        b, a = butter(2, 20.0, btype="highpass", fs=FS)
        return filtfilt(b, a, white).astype(np.float32)
    if kind == "burst":
        white = rng.normal(0.0, 1.0, size=N).astype(np.float32)
        mask = np.zeros(N, dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            width = int(rng.integers(FS // 5, FS * 2))
            start = int(rng.integers(0, max(1, N - width)))
            mask[start : start + width] = 1.0
        return (white * mask).astype(np.float32)
    if kind == "ampmod":
        base = sample_noise_segment(tracks["ma"], split_name, ranges["ma"], rng).astype(np.float32)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        t = np.arange(N, dtype=np.float32) / float(FS)
        mod = 0.5 + 0.5 * np.sin(2.0 * np.pi * float(rng.uniform(0.1, 0.5)) * t + phase)
        return (base * mod).astype(np.float32)
    raise ValueError(f"unknown noise kind: {kind}")


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer")
    source_artifact_dir = _path(params.get("source_artifact_dir"), artifact_dir)
    preserve_eval_from = params.get("preserve_eval_from")
    preserve_eval_dir = _path(preserve_eval_from, Path("")) if preserve_eval_from else None
    nstdb = _path(params.get("nstdb_root"), root / "data" / "physionet" / "nstdb")
    seed = int(params.get("seed", 0))
    force = bool(params.get("force", False))
    train_aug_mode = str(params.get("train_aug_mode", "single")).lower()
    train_aug_k = int(params.get("train_aug_k", 1))
    train_noise_kinds = parse_noise_kinds(params.get("train_noise_kinds"))
    stratify_noise_snr = bool(params.get("stratify_noise_snr", False))
    if train_aug_mode not in {"single", "multiview", "triplet"}:
        raise ValueError("train_aug_mode must be one of: single, multiview, triplet")
    if train_aug_k < 1:
        raise ValueError("train_aug_k must be >= 1")

    x_npz = source_artifact_dir / "segments" / "ptbxl_leadI_x_10s_125hz.npz"
    seg_csv = source_artifact_dir / "segments" / "ptbxl_leadI_segments_10s_125hz.csv"
    split_csv = source_artifact_dir / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
    out_dir = artifact_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_clean = out_dir / "synth_10s_125hz_clean.npz"
    out_noisy = out_dir / "synth_10s_125hz_noisy.npz"
    out_lbl = out_dir / "synth_10s_125hz_labels.csv"

    outputs = [out_clean, out_noisy, out_lbl]
    if all(path.exists() for path in outputs) and not force:
        logger.info("synthesize_noise: outputs exist -> skip (set --force to rerun)")
        return {"step": "synthesize_noise", "skipped": True, "outputs": [str(p) for p in outputs]}

    X_all = np.load(x_npz)["X"].astype(np.float32)
    df_seg = pd.read_csv(seg_csv)
    df_split = pd.read_csv(split_csv)

    need_cols = {"seg_id", "ecg_id", "npz_index", "split"}
    if not need_cols.issubset(df_split.columns):
        raise ValueError(f"split csv missing columns: {sorted(need_cols - set(df_split.columns))}")
    if "seg_id" not in df_seg.columns:
        raise ValueError("segments csv missing seg_id")

    rng = np.random.default_rng(seed)

    tracks = {"em": load_noise_125(nstdb / "em"), "ma": load_noise_125(nstdb / "ma")}
    bw_path = nstdb / "bw"
    if bw_path.with_suffix(".hea").exists() or (nstdb / "bw.hea").exists():
        tracks["bw"] = load_noise_125(bw_path)
    else:
        tracks["bw"] = tracks["ma"]
    ranges = {name: split_noise_ranges(len(track)) for name, track in tracks.items()}

    x_clean_rows: list[np.ndarray] = []
    x_noisy_rows: list[np.ndarray] = []
    label_rows: list[dict[str, object]] = []
    df_base = df_split[["seg_id", "ecg_id", "npz_index", "split"]].copy()
    train_df = df_base[df_base["split"].astype(str) == "train"].reset_index(drop=True)

    def append_sample(
        *,
        base_row: Any,
        y_class: str,
        noise_kind: str,
        snr_db: float,
        clean: np.ndarray,
        noisy: np.ndarray,
        variant_id: int,
        source: str,
    ) -> None:
        x_clean_rows.append(clean.astype(np.float32, copy=False))
        x_noisy_rows.append(noisy.astype(np.float32, copy=False))
        label_rows.append(
            {
                "idx": len(label_rows),
                "seg_id": int(base_row.seg_id),
                "ecg_id": int(base_row.ecg_id),
                "split": str(base_row.split),
                "y_class": y_class,
                "snr_db": float(snr_db),
                "noise_kind": noise_kind,
                "fs": FS,
                "window_sec": WIN_SEC,
                "source_npz_index": int(getattr(base_row, "npz_index", getattr(base_row, "source_npz_index", -1))),
                "variant_id": int(variant_id),
                "sample_source": source,
            }
        )

    if train_aug_mode == "triplet":
        group_count = len(train_df) * train_aug_k
        kind_counter = 0
        snr_counter = {"good": 0, "medium": 0, "bad": 0}
        for row_i, row in enumerate(train_df.itertuples(index=False), start=1):
            if verbose and row_i % 5000 == 0:
                logger.info("synthesize train triplet progress: %d/%d", row_i, len(train_df))
            clean = X_all[int(row.npz_index)].astype(np.float32, copy=False)
            for rep in range(train_aug_k):
                if stratify_noise_snr:
                    noise_kind = choose_balanced(train_noise_kinds, kind_counter, rng)
                    kind_counter += 1
                else:
                    noise_kind = str(rng.choice(np.array(train_noise_kinds, dtype=object)))
                v = make_noise_segment(noise_kind, "train", tracks, ranges, rng)
                for y_class_name in ("good", "medium", "bad"):
                    if stratify_noise_snr:
                        snr_db = sample_snr_db_stratified(y_class_name, snr_counter[y_class_name], rng)
                        snr_counter[y_class_name] += 1
                    else:
                        snr_db = sample_snr_db(y_class_name, rng)
                    append_sample(
                        base_row=row,
                        y_class=y_class_name,
                        noise_kind=noise_kind,
                        snr_db=snr_db,
                        clean=clean,
                        noisy=add_noise_at_snr(clean, v, snr_db),
                        variant_id=rep,
                        source=f"train_{train_aug_mode}",
                    )
        logger.info("train triplet groups: %d", group_count)
    else:
        train_views: list[Any] = []
        for row in train_df.itertuples(index=False):
            for rep in range(train_aug_k if train_aug_mode == "multiview" else 1):
                train_views.append((row, rep))
        y_classes = assign_balanced_classes(len(train_views), rng)
        snr_counter = {"good": 0, "medium": 0, "bad": 0}
        for i, ((row, rep), y_class_name) in enumerate(zip(train_views, y_classes), start=1):
            if verbose and i % 10000 == 0:
                logger.info("synthesize train %s progress: %d/%d", train_aug_mode, i, len(train_views))
            clean = X_all[int(row.npz_index)].astype(np.float32, copy=False)
            if stratify_noise_snr:
                noise_kind = choose_balanced(train_noise_kinds, i - 1, rng)
                snr_db = sample_snr_db_stratified(str(y_class_name), snr_counter[str(y_class_name)], rng)
                snr_counter[str(y_class_name)] += 1
            else:
                noise_kind = str(rng.choice(np.array(train_noise_kinds, dtype=object)))
                snr_db = sample_snr_db(str(y_class_name), rng)
            v = make_noise_segment(noise_kind, "train", tracks, ranges, rng)
            append_sample(
                base_row=row,
                y_class=str(y_class_name),
                noise_kind=noise_kind,
                snr_db=snr_db,
                clean=clean,
                noisy=add_noise_at_snr(clean, v, snr_db),
                variant_id=int(rep),
                source=f"train_{train_aug_mode}",
            )

    if preserve_eval_dir is not None:
        base_clean = np.load(preserve_eval_dir / "datasets" / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
        base_noisy = np.load(preserve_eval_dir / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        base_labels = pd.read_csv(preserve_eval_dir / "datasets" / "synth_10s_125hz_labels.csv")
        for row in base_labels[base_labels["split"].astype(str).isin(["val", "test"])].itertuples(index=False):
            idx = int(row.idx)
            append_sample(
                base_row=row,
                y_class=str(row.y_class),
                noise_kind=str(row.noise_kind),
                snr_db=float(row.snr_db),
                clean=base_clean[idx],
                noisy=base_noisy[idx],
                variant_id=int(getattr(row, "variant_id", 0)),
                source="preserved_eval",
            )
    else:
        for sp in ["val", "test"]:
            eval_df = df_base[df_base["split"].astype(str) == sp].reset_index(drop=True)
            y_classes = assign_balanced_classes(len(eval_df), rng)
            snr_counter = {"good": 0, "medium": 0, "bad": 0}
            for i, (row, y_class_name) in enumerate(zip(eval_df.itertuples(index=False), y_classes), start=1):
                clean = X_all[int(row.npz_index)].astype(np.float32, copy=False)
                noise_kind = str(rng.choice(np.array(DEFAULT_NOISE_KINDS, dtype=object)))
                if stratify_noise_snr:
                    snr_db = sample_snr_db_stratified(str(y_class_name), snr_counter[str(y_class_name)], rng)
                    snr_counter[str(y_class_name)] += 1
                else:
                    snr_db = sample_snr_db(str(y_class_name), rng)
                v = make_noise_segment(noise_kind, sp, tracks, ranges, rng)
                append_sample(
                    base_row=row,
                    y_class=str(y_class_name),
                    noise_kind=noise_kind,
                    snr_db=snr_db,
                    clean=clean,
                    noisy=add_noise_at_snr(clean, v, snr_db),
                    variant_id=0,
                    source="eval_single",
                )

    X_clean = np.stack(x_clean_rows, axis=0).astype(np.float32)
    X_noisy = np.stack(x_noisy_rows, axis=0).astype(np.float32)
    labels = pd.DataFrame(label_rows)
    labels["idx"] = np.arange(len(labels), dtype=int)

    np.savez(out_clean, X_clean=X_clean.astype(np.float32))
    np.savez(out_noisy, X_noisy=X_noisy.astype(np.float32))
    labels.to_csv(out_lbl, index=False)

    split_counts: dict[str, dict[str, int]] = {}
    for sp in ["train", "val", "test"]:
        d = labels[labels["split"] == sp]
        y_counts = {str(k): int(v) for k, v in d["y_class"].value_counts().sort_index().to_dict().items()}
        nk_counts = {str(k): int(v) for k, v in d["noise_kind"].value_counts().sort_index().to_dict().items()}
        split_counts[sp] = y_counts
        logger.info("[%s] y_class counts: %s", sp, y_counts)
        logger.info("[%s] noise_kind counts: %s", sp, nk_counts)
    logger.info("train_aug_mode=%s train_aug_k=%d stratify_noise_snr=%s", train_aug_mode, train_aug_k, stratify_noise_snr)
    logger.info("Saved clean: %s", _display(out_clean, root))
    logger.info("Saved noisy: %s", _display(out_noisy, root))
    logger.info("Saved labels: %s", _display(out_lbl, root))
    return {
        "step": "synthesize_noise",
        "skipped": False,
        "outputs": [str(p) for p in outputs],
        "rows": int(len(labels)),
        "split_y_class_counts": split_counts,
    }


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize noisy PTB-XL Lead I segments at balanced SNR classes.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--source_artifact_dir", default="")
    parser.add_argument("--preserve_eval_from", default="")
    parser.add_argument("--nstdb_root", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_aug_mode", choices=("single", "multiview", "triplet"), default="single")
    parser.add_argument("--train_aug_k", type=int, default=1)
    parser.add_argument("--train_noise_kinds", default="em,ma,mix")
    parser.add_argument("--stratify_noise_snr", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
