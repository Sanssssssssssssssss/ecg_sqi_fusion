# src/augment/noise/ptbxl_step4_synthesize_noise_snr_balanced.py
from __future__ import annotations

import math
import sys
from pathlib import Path

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


SEED = 0
FS = 125
WIN_SEC = 10
N = FS * WIN_SEC  # 1250


def load_noise_125(path_no_ext: Path) -> np.ndarray:
    sig, fields = wfdb.rdsamp(str(path_no_ext))
    fs_raw = int(round(float(fields["fs"])))
    v = sig.mean(axis=1).astype(np.float64) # type: ignore
    v = v - np.mean(v)
    if fs_raw != FS:
        g = math.gcd(FS, fs_raw)
        v = resample_poly(v, up=FS // g, down=fs_raw // g)
    return v.astype(np.float32)


def split_noise_ranges(length: int) -> dict[str, tuple[int, int]]:
    a = length // 3
    return {"train": (0, a), "val": (a, 2 * a), "test": (2 * a, length)}


def sample_noise_segment(track: np.ndarray, split_name: str, ranges: dict[str, tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
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
        return float(rng.uniform(16, 20))
    if y_class == "medium":
        return float(rng.uniform(4, 8))
    return float(rng.uniform(-6, -2))


def add_noise_at_snr(x: np.ndarray, v: np.ndarray, snr_db: float) -> np.ndarray:
    px = float(np.mean(x * x)) + 1e-12
    pv = float(np.mean(v * v)) + 1e-12
    a = np.sqrt(px / (pv * (10.0 ** (snr_db / 10.0))))
    return (x + a * v).astype(np.float32)


def main() -> None:
    root = project_root()
    x_npz = root / "artifact1" / "segments" / "ptbxl_leadI_x_10s_125hz.npz"
    seg_csv = root / "artifact1" / "segments" / "ptbxl_leadI_segments_10s_125hz.csv"
    split_csv = root / "artifact1" / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
    nstdb = root / "data" / "physionet" / "nstdb"

    out_dir = root / "artifact1" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_clean = out_dir / "synth_10s_125hz_clean.npz"
    out_noisy = out_dir / "synth_10s_125hz_noisy.npz"
    out_lbl = out_dir / "synth_10s_125hz_labels.csv"

    X_all = np.load(x_npz)["X"].astype(np.float32)
    df_seg = pd.read_csv(seg_csv)
    df_split = pd.read_csv(split_csv)

    need_cols = {"seg_id", "ecg_id", "npz_index", "split"}
    if not need_cols.issubset(df_split.columns):
        raise ValueError(f"split csv missing columns: {sorted(need_cols - set(df_split.columns))}")

    df = df_split[["seg_id", "ecg_id", "npz_index", "split"]].copy()
    if "seg_id" not in df_seg.columns:
        raise ValueError("segments csv missing seg_id")

    rng = np.random.default_rng(SEED)

    em = load_noise_125(nstdb / "em")
    ma = load_noise_125(nstdb / "ma")
    em_ranges = split_noise_ranges(len(em))
    ma_ranges = split_noise_ranges(len(ma))

    # Assign balanced y_class inside each split.
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

    for i, row in enumerate(df.itertuples(index=False)):
        x = X_clean[i]
        if row.noise_kind == "em":
            v = sample_noise_segment(em, row.split, em_ranges, rng)
        elif row.noise_kind == "ma":
            v = sample_noise_segment(ma, row.split, ma_ranges, rng)
        else:
            v = sample_noise_segment(em, row.split, em_ranges, rng) + sample_noise_segment(ma, row.split, ma_ranges, rng)
        X_noisy[i] = add_noise_at_snr(x, v.astype(np.float32), float(row.snr_db))

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

    np.savez_compressed(out_clean, X_clean=X_clean.astype(np.float32))
    np.savez_compressed(out_noisy, X_noisy=X_noisy.astype(np.float32))
    labels.to_csv(out_lbl, index=False)

    for sp in ["train", "val", "test"]:
        d = labels[labels["split"] == sp]
        print(f"[{sp}] y_class counts: {d['y_class'].value_counts().to_dict()}")
        print(f"[{sp}] noise_kind counts: {d['noise_kind'].value_counts().to_dict()}")
    print(f"Saved clean: {out_clean}")
    print(f"Saved noisy: {out_noisy}")
    print(f"Saved labels: {out_lbl}")
    print("NEXT INPUT: artifact1/datasets/synth_10s_125hz_labels.csv + npz files")


if __name__ == "__main__":
    main()
