# src/qrs/detectors.py
from __future__ import annotations

import numpy as np
import wfdb.processing as wproc
import contextlib
import io
from typing import Iterator

@contextlib.contextmanager
def _silence_wfdb_prints(enabled: bool = True) -> Iterator[None]:
    """
    wfdb's XQRS/GQRS outputs are printed directly via print().
    Here, we redirect stdout/stderr to silence them.
    This does not alter the algorithm's logic; it merely reduces console noise.
    """
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def run_xqrs(sig_1d: np.ndarray, fs: int, quiet: bool = True) -> np.ndarray:
    """WFDB XQRS detector -> rpeak sample indices."""
    with _silence_wfdb_prints(quiet):
        xq = wproc.XQRS(sig=sig_1d, fs=fs)
        xq.detect()
        r = np.asarray(xq.qrs_inds, dtype=int)

    r = r[(r >= 0) & (r < len(sig_1d))]
    return r


def run_gqrs(sig_1d: np.ndarray, fs: int, quiet: bool = True) -> np.ndarray:
    """WFDB gqrs_detect -> rpeak sample indices."""
    with _silence_wfdb_prints(quiet):
        r = np.asarray(wproc.gqrs_detect(sig=sig_1d, fs=fs), dtype=int)

    r = r[(r >= 0) & (r < len(sig_1d))]
    return r

