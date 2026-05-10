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
from scipy.signal import resample_poly

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


def add_noise_at_snr(x: np.ndarray, v: np.ndarray, snr_db: float) -> np.ndarray:
    px = float(np.mean(x * x)) + 1e-12
    pv = float(np.mean(v * v)) + 1e-12
    a = np.sqrt(px / (pv * (10.0 ** (snr_db / 10.0))))
    return (x + a * v).astype(np.float32)


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer")
    nstdb = _path(params.get("nstdb_root"), root / "data" / "physionet" / "nstdb")
    seed = int(params.get("seed", 0))
    force = bool(params.get("force", False))

    x_npz = artifact_dir / "segments" / "ptbxl_leadI_x_10s_125hz.npz"
    seg_csv = artifact_dir / "segments" / "ptbxl_leadI_segments_10s_125hz.csv"
    split_csv = artifact_dir / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
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

    em = load_noise_125(nstdb / "em")
    ma = load_noise_125(nstdb / "ma")
    em_ranges = split_noise_ranges(len(em))
    ma_ranges = split_noise_ranges(len(ma))

    df = df_split[["seg_id", "ecg_id", "npz_index", "split"]].copy()
    y_class = np.empty(len(df), dtype=object)
    for sp in ["train", "val", "test"]:
        idx = np.where(df["split"].to_numpy() == sp)[0]
        y_class[idx] = assign_balanced_classes(len(idx), rng)
    df["y_class"] = y_class
    df["noise_kind"] = rng.choice(np.array(["em", "ma", "mix"], dtype=object), size=len(df), replace=True)
    df["snr_db"] = [sample_snr_db(c, rng) for c in df["y_class"].tolist()]

    npz_idx = df["npz_index"].astype(int).to_numpy()
    X_clean = X_all[npz_idx].astype(np.float32, copy=False)
    X_noisy = np.empty_like(X_clean, dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False), start=1):
        if verbose and i % 10000 == 0:
            logger.info("synthesize_noise progress: %d/%d", i, len(df))
        x = X_clean[i - 1]
        if row.noise_kind == "em":
            v = sample_noise_segment(em, row.split, em_ranges, rng)
        elif row.noise_kind == "ma":
            v = sample_noise_segment(ma, row.split, ma_ranges, rng)
        else:
            v = sample_noise_segment(em, row.split, em_ranges, rng) + sample_noise_segment(ma, row.split, ma_ranges, rng)
        X_noisy[i - 1] = add_noise_at_snr(x, v.astype(np.float32), float(row.snr_db))

    labels = pd.DataFrame(
        {
            "idx": np.arange(len(df), dtype=int),
            "seg_id": df["seg_id"].to_numpy(),
            "ecg_id": df["ecg_id"].to_numpy(),
            "split": df["split"].to_numpy(),
            "y_class": df["y_class"].to_numpy(),
            "snr_db": df["snr_db"].to_numpy(),
            "noise_kind": df["noise_kind"].to_numpy(),
            "fs": FS,
            "window_sec": WIN_SEC,
        }
    )

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
    parser.add_argument("--nstdb_root", default="")
    parser.add_argument("--seed", type=int, default=0)
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
