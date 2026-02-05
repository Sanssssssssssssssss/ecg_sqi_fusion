from __future__ import annotations

import numpy as np
import wfdb


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    x = np.nan_to_num(x)
    return x


def detect_xqrs(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    XQRS detector (Python). Returns sample indices (int) at current fs.
    """
    sig = _ensure_1d(sig)
    # xqrs_detect expects units similar to mV-ish; for safety normalize
    s = sig - np.median(sig)
    mad = np.median(np.abs(s)) + 1e-12
    s = s / mad
    try:
        beats = wfdb.processing.xqrs_detect(sig=s, fs=fs)
    except Exception:
        beats = np.array([], dtype=int)
    return np.asarray(beats, dtype=int)


def detect_gqrs(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    GQRS detector (Python). Returns sample indices (int) at current fs.
    """
    sig = _ensure_1d(sig)
    s = sig - np.median(sig)
    mad = np.median(np.abs(s)) + 1e-12
    s = s / mad
    try:
        beats = wfdb.processing.gqrs_detect(sig=s, fs=fs)
    except Exception:
        beats = np.array([], dtype=int)
    return np.asarray(beats, dtype=int)
