import pytest

from wfdb_qrs_kit.exceptions import DetectorNotFoundError

from src.sqi_pipeline.qrs.paper_detectors import PaperQRSUnavailable, _run_kit


def test_run_kit_retries_transient_detector_failure():
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("eplimited.exe failed with code 1")
        return "ok"

    assert _run_kit("detect_both", flaky) == "ok"
    assert calls == 3


def test_run_kit_does_not_retry_missing_detector():
    calls = 0

    def missing():
        nonlocal calls
        calls += 1
        raise DetectorNotFoundError("missing wqrs")

    with pytest.raises(PaperQRSUnavailable, match="missing wqrs"):
        _run_kit("detect_both", missing)

    assert calls == 1
