# src/qrs/run_qrs_cache.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd


from src.utils.paths import project_root
from src.qrs.detectors import run_xqrs, run_gqrs

logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# --------- configuration ---------
SEED = 0
FS = 125
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

DETECTOR1 = "xqrs"
DETECTOR2 = "gqrs"

# The tolerance is write in npz, it can be changed in sqi
BEAT_MATCH_TOL_MS = 150
# --------------------------------------------------


def load_sig125(record_id: str, resampled_dir: Path) -> tuple[np.ndarray, int, list[str]]:
    z = np.load(resampled_dir / f"{record_id}.npz", allow_pickle=True)
    sig = z["sig_125"].astype(np.float32)
    fs = int(z["fs"])
    leads = list(z["leads"].tolist())
    return sig, fs, leads

def _outputs_exist(out_dir: Path) -> bool:
    """
    Minimal skip check: output dir exists and has at least one .npz.
    """
    if not out_dir.exists():
        return False
    return any(out_dir.glob("*.npz"))

def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - split_csv: str (default artifacts/splits/split_seta_seed0_balanced.csv)
      - resampled_dir: str (default artifacts/resampled_125)
      - out_dir: str (default artifacts/qrs)
      - beat_match_tol_ms: int (default BEAT_MATCH_TOL_MS)

    Returns dict with outputs list and skipped bool.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()

    split_csv = params.get("split_csv") or (root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv")
    resampled_dir = params.get("resampled_dir") or (root / "artifacts" / "resampled_125")
    out_dir = params.get("out_dir") or (root / "artifacts" / "qrs")

    split_csv = Path(str(split_csv))
    resampled_dir = Path(str(resampled_dir))
    out_dir = Path(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional override (default keeps your constant)
    global BEAT_MATCH_TOL_MS
    if "beat_match_tol_ms" in params and params["beat_match_tol_ms"] is not None:
        BEAT_MATCH_TOL_MS = int(params["beat_match_tol_ms"])

    if (not force) and _outputs_exist(out_dir):
        logger.info("qrs_cache: outputs exist -> skip (set force=True to rerun)")
        return {"step": "qrs_cache", "skipped": True, "outputs": [str(out_dir)]}

    df = pd.read_csv(split_csv)
    record_ids = df["record_id"].astype(str).tolist()

    logger.info("split_csv: %s", split_csv)
    logger.info("resampled_125: %s", resampled_dir)
    logger.info("out qrs dir: %s", out_dir)
    logger.info("fs=%d, detectors=(%s,%s), beat_match_tol_ms=%d", FS, DETECTOR1, DETECTOR2, BEAT_MATCH_TOL_MS)
    logger.info("records=%d", len(record_ids))

    n_ok, n_skip, n_fail = 0, 0, 0

    for i, rid in enumerate(record_ids, start=1):
        out_npz = out_dir / f"{rid}.npz"
        if out_npz.exists():
            n_skip += 1
            if i <= 5 or i % 200 == 0:
                logger.info("[skip] %s", rid)
            continue

        try:
            sig12, fs, leads = load_sig125(rid, resampled_dir)
            if fs != FS:
                raise ValueError(f"{rid}: expected fs={FS}, got {fs}")
            if leads != LEADS_12:
                raise ValueError(f"{rid}: lead order mismatch")

            r1_list, r2_list = [], []
            for li in range(sig12.shape[1]):
                x = sig12[:, li]
                r1_list.append(run_xqrs(x, fs=FS))
                r2_list.append(run_gqrs(x, fs=FS))

            np.savez_compressed(
                out_npz,
                record_id=rid,
                fs=np.array(FS, dtype=np.int32),
                leads=np.array(LEADS_12, dtype=object),
                detector1=np.array(DETECTOR1, dtype=object),
                detector2=np.array(DETECTOR2, dtype=object),
                beat_match_tol_ms=np.array(BEAT_MATCH_TOL_MS, dtype=np.int32),
                rpeaks_1=np.array(r1_list, dtype=object),
                rpeaks_2=np.array(r2_list, dtype=object),
            )

            n_ok += 1
            if i <= 5 or i % 200 == 0:
                logger.info("[ok] %s -> %s", rid, out_npz.name)

        except Exception as e:
            n_fail += 1
            logger.warning("[FAIL] %s: %s: %s", rid, type(e).__name__, e)

    logger.info("=== DONE cache qrs ===")
    logger.info("saved : %d", n_ok)
    logger.info("skipped: %d", n_skip)
    logger.info("failed : %d", n_fail)

    return {"step": "qrs_cache", "skipped": False, "outputs": [str(out_dir)]}

def main() -> None:
    params = {
        "verbose": False,
        "force": False,
        # default balanced split + resampled_125 + artifacts/qrs
        # "beat_match_tol_ms": 150,
    }
    run(params)

if __name__ == "__main__":
    main()
