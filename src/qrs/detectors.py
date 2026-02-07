# src/qrs/detectors.py
from __future__ import annotations

import numpy as np
import wfdb.processing as wproc


def run_xqrs(sig_1d: np.ndarray, fs: int) -> np.ndarray:
    """WFDB XQRS detector -> rpeak sample indices."""
    xq = wproc.XQRS(sig=sig_1d, fs=fs)
    xq.detect()
    r = np.asarray(xq.qrs_inds, dtype=int)
    r = r[(r >= 0) & (r < len(sig_1d))]
    return r


def run_gqrs(sig_1d: np.ndarray, fs: int) -> np.ndarray:
    """WFDB gqrs_detect -> rpeak sample indices."""
    r = np.asarray(wproc.gqrs_detect(sig=sig_1d, fs=fs), dtype=int)
    r = r[(r >= 0) & (r < len(sig_1d))]
    return r
