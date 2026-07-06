# src/qrs/run_qrs_cache.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import os
import numpy as np
import pandas as pd


from src.utils.paths import project_root
from src.sqi_pipeline.qrs.detectors import run_xqrs, run_gqrs
from src.sqi_pipeline.qrs.paper_detectors import (
    PaperQRSUnavailable,
    resolve_paper_qrs_executables,
    run_paper_qrs_12lead,
)

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

def _outputs_exist(out_dir: Path, split_csv: Path) -> bool:
    """
    Skip only when every split row has a matching QRS .npz.
    """
    if not out_dir.exists():
        return False
    if not (split_csv.exists() and split_csv.is_file() and split_csv.stat().st_size > 0):
        return False
    try:
        df = pd.read_csv(split_csv, usecols=["record_id"])
    except Exception:
        return False
    expected = {str(x) for x in df["record_id"].astype(str)}
    if not expected:
        return False
    have = {p.stem for p in out_dir.glob("*.npz")}
    return expected.issubset(have)


def _scalar_to_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _summary_rows_from_cache(path: Path, *, fallback_profile: str, fallback_warmup_sec: float) -> list[dict[str, Any]]:
    z = np.load(path, allow_pickle=True)
    rid = _scalar_to_str(z["record_id"])
    leads = [str(x) for x in z["leads"].tolist()]
    r1_list = z["rpeaks_1"]
    r2_list = z["rpeaks_2"]
    detector1 = _scalar_to_str(z["detector1"]) if "detector1" in z.files else DETECTOR1
    detector2 = _scalar_to_str(z["detector2"]) if "detector2" in z.files else DETECTOR2
    detector_profile = _scalar_to_str(z["detector_profile"]) if "detector_profile" in z.files else fallback_profile
    warmup = float(np.asarray(z["eplimited_warmup_sec"]).item()) if "eplimited_warmup_sec" in z.files else fallback_warmup_sec
    return [
        {
            "record_id": rid,
            "lead": lead,
            "detector1": detector1,
            "detector2": detector2,
            "n_detector1": int(len(r1)),
            "n_detector2": int(len(r2)),
            "detector_profile": detector_profile,
            "eplimited_warmup_sec": warmup if detector_profile == "paper" else 0.0,
        }
        for lead, r1, r2 in zip(leads, r1_list, r2_list)
    ]


def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - split_csv: str (default outputs/sqi/splits/split_seta_seed0_balanced.csv)
      - resampled_dir: str (default outputs/sqi/resampled_125)
      - out_dir: str (default outputs/sqi/qrs)
      - beat_match_tol_ms: int (default BEAT_MATCH_TOL_MS)
      - detector_profile: "wfdb"|"paper" (default "wfdb")
      - eplimited_warmup_sec: float (paper profile, default 8.0)
      - qrs_summary_csv: optional CSV with per-record/per-lead detector counts

    Returns dict with outputs list and skipped bool.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()

    split_csv = params.get("split_csv") or (root / "outputs/sqi" / "splits" / "split_seta_seed0_balanced.csv")
    resampled_dir = params.get("resampled_dir") or (root / "outputs/sqi" / "resampled_125")
    out_dir = params.get("out_dir") or (root / "outputs/sqi" / "qrs")
    detector_profile = str(params.get("detector_profile", "wfdb"))
    if detector_profile not in {"wfdb", "paper"}:
        raise ValueError("detector_profile must be 'wfdb' or 'paper'")
    eplimited_warmup_sec = float(params.get("eplimited_warmup_sec", 8.0))

    split_csv = Path(str(split_csv))
    resampled_dir = Path(str(resampled_dir))
    out_dir = Path(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(str(params.get("qrs_summary_csv") or (out_dir / "qrs_summary.csv")))

    # optional override (default keeps your constant)
    global BEAT_MATCH_TOL_MS
    if "beat_match_tol_ms" in params and params["beat_match_tol_ms"] is not None:
        BEAT_MATCH_TOL_MS = int(params["beat_match_tol_ms"])

    if (not force) and _outputs_exist(out_dir, split_csv):
        logger.info("qrs_cache: outputs exist -> skip (set force=True to rerun)")
        outputs = [str(out_dir)]
        if summary_csv.exists():
            outputs.append(str(summary_csv))
        return {"step": "qrs_cache", "skipped": True, "outputs": outputs}

    df = pd.read_csv(split_csv)
    record_ids = df["record_id"].astype(str).tolist()

    paper_executables = None
    runtime_detector_profile = detector_profile
    paper_work_dir = Path(str(params.get("paper_qrs_work_dir") or (out_dir / "_wfdb_tmp")))
    if detector_profile == "paper":
        try:
            paper_executables = resolve_paper_qrs_executables(params, out_dir)
        except PaperQRSUnavailable:
            if _truthy(params.get("allow_wfdb_qrs_fallback")) or _truthy(os.environ.get("SQI_ALLOW_WFDB_QRS_FALLBACK")):
                runtime_detector_profile = "paper_fallback_wfdb"
                logger.warning(
                    "paper QRS executables are unavailable; using WFDB Python xqrs/gqrs fallback "
                    "because SQI_ALLOW_WFDB_QRS_FALLBACK is enabled. Results are runnable but not paper-exact."
                )
            else:
                raise
        if paper_executables is not None:
            logger.info("paper QRS executables: wqrs=%s eplimited=%s", paper_executables.wqrs, paper_executables.eplimited)

    logger.info("split_csv: %s", split_csv)
    logger.info("resampled_125: %s", resampled_dir)
    logger.info("out qrs dir: %s", out_dir)
    if runtime_detector_profile == "paper":
        logger.info(
            "fs=%d, detectors=(wqrs,eplimited), beat_match_tol_ms=%d, eplimited_warmup_sec=%.3f",
            FS,
            BEAT_MATCH_TOL_MS,
            eplimited_warmup_sec,
        )
    elif runtime_detector_profile == "paper_fallback_wfdb":
        logger.info("fs=%d, detectors=(xqrs,gqrs), beat_match_tol_ms=%d, profile=paper_fallback_wfdb", FS, BEAT_MATCH_TOL_MS)
    else:
        logger.info("fs=%d, detectors=(%s,%s), beat_match_tol_ms=%d", FS, DETECTOR1, DETECTOR2, BEAT_MATCH_TOL_MS)
    logger.info("records=%d", len(record_ids))

    n_ok, n_skip, n_fail = 0, 0, 0
    summary_rows: list[dict[str, Any]] = []

    for i, rid in enumerate(record_ids, start=1):
        out_npz = out_dir / f"{rid}.npz"
        if out_npz.exists() and not force:
            n_skip += 1
            summary_rows.extend(
                _summary_rows_from_cache(
                    out_npz,
                    fallback_profile=runtime_detector_profile,
                    fallback_warmup_sec=eplimited_warmup_sec,
                )
            )
            if i <= 5 or i % 200 == 0:
                logger.info("[skip] %s", rid)
            continue

        try:
            sig12, fs, leads = load_sig125(rid, resampled_dir)
            if fs != FS:
                raise ValueError(f"{rid}: expected fs={FS}, got {fs}")
            if leads != LEADS_12:
                raise ValueError(f"{rid}: lead order mismatch")

            if runtime_detector_profile == "paper":
                if paper_executables is None:
                    raise RuntimeError("paper QRS executables were not resolved")
                r1_list, r2_list = run_paper_qrs_12lead(
                    record_id=rid,
                    sig12=sig12,
                    fs=FS,
                    leads=leads,
                    executables=paper_executables,
                    work_dir=paper_work_dir,
                    eplimited_warmup_sec=eplimited_warmup_sec,
                )
                detector1 = "wqrs"
                detector2 = "eplimited"
            else:
                r1_list, r2_list = [], []
                for li in range(sig12.shape[1]):
                    x = sig12[:, li]
                    r1_list.append(run_xqrs(x, fs=FS))
                    r2_list.append(run_gqrs(x, fs=FS))
                detector1 = DETECTOR1
                detector2 = DETECTOR2

            for lead, r1, r2 in zip(leads, r1_list, r2_list):
                summary_rows.append({
                    "record_id": rid,
                    "lead": lead,
                    "detector1": detector1,
                    "detector2": detector2,
                    "n_detector1": int(len(r1)),
                    "n_detector2": int(len(r2)),
                    "detector_profile": runtime_detector_profile,
                    "eplimited_warmup_sec": eplimited_warmup_sec if runtime_detector_profile == "paper" else 0.0,
                })

            np.savez(
                out_npz,
                record_id=rid,
                fs=np.array(FS, dtype=np.int32),
                leads=np.array(LEADS_12, dtype=object),
                detector1=np.array(detector1, dtype=object),
                detector2=np.array(detector2, dtype=object),
                detector_profile=np.array(runtime_detector_profile, dtype=object),
                eplimited_warmup_sec=np.array(eplimited_warmup_sec if runtime_detector_profile == "paper" else 0.0),
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

    outputs = [str(out_dir)]
    if summary_rows:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        logger.info("summary: %s rows=%d", summary_csv, len(summary_rows))
        outputs.append(str(summary_csv))

    if runtime_detector_profile == "paper" and n_fail > 0:
        raise RuntimeError(f"paper QRS failed for {n_fail} records; see logs above")

    return {"step": "qrs_cache", "skipped": False, "outputs": outputs}

def main() -> None:
    params = {
        "verbose": False,
        "force": False,
        # default balanced split + resampled_125 + outputs/sqi/qrs
        # "beat_match_tol_ms": 150,
    }
    run(params)

if __name__ == "__main__":
    main()
