import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sqi_pipeline.qrs import run_qrs_cache


def _write_qrs_cache(path: Path, *, record_id: str, profile: str, detector1: str, detector2: str) -> None:
    rpeaks = np.array([np.array([10, 20], dtype=int) for _ in run_qrs_cache.LEADS_12], dtype=object)
    np.savez(
        path,
        record_id=np.array(record_id, dtype=object),
        fs=np.array(run_qrs_cache.FS, dtype=np.int32),
        leads=np.array(run_qrs_cache.LEADS_12, dtype=object),
        detector1=np.array(detector1, dtype=object),
        detector2=np.array(detector2, dtype=object),
        detector_profile=np.array(profile, dtype=object),
        eplimited_warmup_sec=np.array(8.0 if profile == "paper" else 0.0),
        beat_match_tol_ms=np.array(run_qrs_cache.BEAT_MATCH_TOL_MS, dtype=np.int32),
        rpeaks_1=rpeaks,
        rpeaks_2=rpeaks,
    )


def test_paper_qrs_cache_requires_paper_detector_metadata(tmp_path):
    paper_path = tmp_path / "paper.npz"
    baseline_path = tmp_path / "baseline.npz"
    _write_qrs_cache(paper_path, record_id="1001", profile="paper", detector1="wqrs", detector2="eplimited")
    _write_qrs_cache(baseline_path, record_id="1001", profile="wfdb", detector1="xqrs", detector2="gqrs")

    assert run_qrs_cache._cache_matches_metadata(
        paper_path,
        expected_record_id="1001",
        detector_profile="paper",
        eplimited_warmup_sec=8.0,
    )
    assert not run_qrs_cache._cache_matches_metadata(
        baseline_path,
        expected_record_id="1001",
        detector_profile="paper",
        eplimited_warmup_sec=8.0,
    )


def test_outputs_exist_rejects_mixed_detector_cache(tmp_path):
    split_csv = tmp_path / "split.csv"
    pd.DataFrame({"record_id": ["1001", "1002"]}).to_csv(split_csv, index=False)
    _write_qrs_cache(tmp_path / "1001.npz", record_id="1001", profile="paper", detector1="wqrs", detector2="eplimited")
    _write_qrs_cache(tmp_path / "1002.npz", record_id="1002", profile="wfdb", detector1="xqrs", detector2="gqrs")

    assert not run_qrs_cache._outputs_exist(
        tmp_path,
        split_csv,
        detector_profile="paper",
        eplimited_warmup_sec=8.0,
    )
