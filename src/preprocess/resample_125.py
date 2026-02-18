# src/preprocess/resample_125.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # keep safe in headless env

from scipy.signal import decimate

from src.utils.paths import project_root

logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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

def _outputs_exist(out_dir: Path, split_csv: Path) -> bool:
    """
    Minimal skip check:
      - output directory exists and contains at least one .npz
      - split_csv exists
    (We keep it simple for stability.)
    """
    if not (split_csv.exists() and split_csv.is_file() and split_csv.stat().st_size > 0):
        return False
    if not out_dir.exists():
        return False
    # if already produced anything, we assume step done unless force=True
    return any(out_dir.glob("*.npz"))


def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - overwrite: bool   (default uses OVERWRITE constant)
      - split_csv: str    (override SPLIT_CSV)
      - cases_500_dir: str (override CASES_500_DIR)
      - out_dir: str      (override OUT_DIR)

    Returns dict with outputs list and skipped bool.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()

    # allow lightweight override, but keep defaults identical
    split_csv = params.get("split_csv") or SPLIT_CSV
    cases_500_dir = params.get("cases_500_dir") or CASES_500_DIR
    out_dir_s = params.get("out_dir") or OUT_DIR
    overwrite = params.get("overwrite")
    if overwrite is None:
        overwrite = OVERWRITE
    else:
        overwrite = bool(overwrite)

    split_csv_p = root / split_csv
    cases_dir = root / cases_500_dir
    out_dir = root / out_dir_s
    out_dir.mkdir(parents=True, exist_ok=True)

    if (not force) and _outputs_exist(out_dir, split_csv_p):
        logger.info("resample_125: outputs exist -> skip (set force=True to rerun)")
        return {"step": "resample_125", "skipped": True, "outputs": [str(out_dir)]}

    logger.info("[resample_125] split_csv: %s", split_csv_p)
    logger.info("[resample_125] cases_500: %s", cases_dir)
    logger.info("[resample_125] out_dir:   %s", out_dir)
    logger.info(
        "[resample_125] %dHz -> %dHz | q=%d | decimate(ftype=%s, zero_phase=%s)",
        FS_IN, FS_OUT, DECIM_Q, DECIMATE_FTYPE, DECIMATE_ZERO_PHASE
    )
    logger.info(
        "[resample_125] overwrite=%s | crop=%.1fs (%d samples at %dHz)",
        overwrite, CASE_SEC, N_SAMPLES_IN, FS_IN
    )

    df = pd.read_csv(split_csv_p)
    if "record_id" not in df.columns:
        raise ValueError("split csv must contain record_id column")

    df["record_id"] = df["record_id"].apply(_rid_str)
    record_ids = df["record_id"].astype(str).tolist()

    logger.info("[resample_125] total record_ids: %d", len(record_ids))

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, rid in enumerate(record_ids, start=1):
        out_path = out_dir / f"{rid}.npz"

        if out_path.exists() and (not overwrite):
            n_skip += 1
            if (i <= 5) or (i % 200 == 0):
                logger.info("[skip] %s exists", rid)
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
                logger.info("[ok] %s: %s -> %s saved %s", rid, sig500.shape, sig125.shape, out_path.name)

        except Exception as e:
            n_fail += 1
            logger.warning("[FAIL] %s: %s: %s", rid, type(e).__name__, e)

    logger.info("=== DONE resample_125 (cases_500 -> %s) ===", out_dir_s)
    logger.info("saved/overwritten: %d", n_ok)
    logger.info("skipped:           %d", n_skip)
    logger.info("failed:            %d", n_fail)
    logger.info("output dir:        %s", out_dir)

    return {"step": "resample_125", "skipped": False, "outputs": [str(out_dir)]}


def main() -> None:
    params = {
        "verbose": False,
        "force": False,
        # "overwrite": OVERWRITE,   # 可不写，默认用常量
    }
    run(params)


if __name__ == "__main__":
    main()
