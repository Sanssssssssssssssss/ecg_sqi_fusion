# src/preprocess/resample_125.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # keep safe in headless env

from scipy.signal import decimate

from src.utils.paths import project_root


# ================== CONFIG (edit here) ==================
SEED = 0  # (not used, kept for convention)

# input/output fs
FS_IN = 500
FS_OUT = 125
DECIM_Q = FS_IN // FS_OUT  # 4
assert FS_IN % FS_OUT == 0 and DECIM_Q == 4

CASE_SEC = 10.0
N_SAMPLES_IN = int(FS_IN * CASE_SEC)   # 5000

# which split csv to read (switch here only)
# - clean only:  split_seta_seed0.csv
# - balanced:    split_seta_seed0_balanced.csv
SPLIT_CSV = "artifacts/splits/split_seta_seed0_balanced.csv"

# 500Hz cases folder: contains both clean + noisy *.npz with key 'sig_500'
CASES_500_DIR = "artifacts/cases_500"

# output cache (same as your old pipeline)
OUT_DIR = "artifacts/resampled_125"

# overwrite behavior
OVERWRITE = True  # True => always overwrite; False => skip if exists

# decimate config (keep consistent with your previous 1.3)
DECIMATE_FTYPE = "fir"
DECIMATE_ZERO_PHASE = True

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# ========================================================


def _rid_str(x) -> str:
    """
    Robust record_id to string (handles 1746739.0 etc.).
    """
    if x is None:
        return ""
    if isinstance(x, (np.integer,)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        if np.isfinite(x):
            return str(int(x))
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        return s[:-2]
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def load_case_500(record_id: str, cases_dir: Path) -> np.ndarray:
    """
    Load a 500Hz 12-lead case from cases_500/{record_id}.npz with key 'sig_500'.
    Enforces:
      - fs == 500
      - leads == LEADS_12
      - length >= 10s (will crop to 10s)
    Returns shape (5000, 12) float64.
    """
    p = cases_dir / f"{record_id}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing 500Hz case npz: {p}")

    z = np.load(p, allow_pickle=True)

    if "sig_500" not in z:
        raise ValueError(f"{p}: missing key 'sig_500' (this resample expects 500Hz cache)")

    sig = z["sig_500"].astype(np.float64)
    fs = int(z["fs"])
    leads = list(z["leads"].tolist())

    if fs != FS_IN:
        raise ValueError(f"{record_id}: fs mismatch {fs}, expected {FS_IN}")
    if leads != LEADS_12:
        raise ValueError(f"{record_id}: lead order mismatch")

    if sig.ndim != 2 or sig.shape[1] != 12:
        raise ValueError(f"{record_id}: expected (N,12), got {sig.shape}")
    if sig.shape[0] < N_SAMPLES_IN:
        raise ValueError(f"{record_id}: too short {sig.shape[0]} < {N_SAMPLES_IN}")

    sig = sig[:N_SAMPLES_IN, :]
    return sig


def resample_500_to_125(sig500_12: np.ndarray) -> np.ndarray:
    """
    Anti-alias FIR + zero-phase decimation by 4.
    Input:  (5000,12)
    Output: (~1250,12)
    """
    y = decimate(
        sig500_12,
        q=DECIM_Q,
        axis=0,
        ftype=DECIMATE_FTYPE,
        zero_phase=DECIMATE_ZERO_PHASE,
    )
    return y.astype(np.float32)


def main() -> None:
    root = project_root()

    split_csv = root / SPLIT_CSV
    cases_dir = root / CASES_500_DIR
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[resample_125] split_csv: {split_csv}")
    print(f"[resample_125] cases_500: {cases_dir}")
    print(f"[resample_125] out_dir:   {out_dir}")
    print(f"[resample_125] {FS_IN}Hz -> {FS_OUT}Hz | q={DECIM_Q} | decimate(ftype={DECIMATE_FTYPE}, zero_phase={DECIMATE_ZERO_PHASE})")
    print(f"[resample_125] overwrite={OVERWRITE} | crop={CASE_SEC}s ({N_SAMPLES_IN} samples at {FS_IN}Hz)")

    df = pd.read_csv(split_csv)
    if "record_id" not in df.columns:
        raise ValueError("split csv must contain record_id column")

    df["record_id"] = df["record_id"].apply(_rid_str)
    record_ids = df["record_id"].astype(str).tolist()

    print(f"[resample_125] total record_ids: {len(record_ids)}")

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, rid in enumerate(record_ids, start=1):
        out_path = out_dir / f"{rid}.npz"

        if out_path.exists() and not OVERWRITE:
            n_skip += 1
            if (i <= 5) or (i % 200 == 0):
                print(f"[skip] {rid} exists")
            continue

        try:
            sig500 = load_case_500(rid, cases_dir)
            sig125 = resample_500_to_125(sig500)

            np.savez_compressed(
                out_path,
                record_id=rid,
                fs=np.int32(FS_OUT),
                leads=np.array(LEADS_12, dtype=object),
                sig_125=sig125,
            )

            n_ok += 1
            if (i <= 5) or (i % 200 == 0):
                print(f"[ok] {rid}: {sig500.shape} -> {sig125.shape} saved {out_path.name}")

        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {rid}: {type(e).__name__}: {e}")

    print("\n=== DONE resample_125 (cases_500 -> resampled_125) ===")
    print(f"saved/overwritten: {n_ok}")
    print(f"skipped:           {n_skip}")
    print(f"failed:            {n_fail}")
    print(f"output dir:        {out_dir}")


if __name__ == "__main__":
    main()
