# src/preprocess/resample.py
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction


def downsample_to(sig: np.ndarray, fs_in: float, fs_out: float = 125.0) -> tuple[np.ndarray, float]:
    """
    Downsample with anti-aliasing using resample_poly.
    Returns (sig_resampled, fs_out)
    """
    sig = np.asarray(sig, dtype=float)
    if abs(fs_in - fs_out) < 1e-9:
        return sig, fs_in

    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y = resample_poly(sig, up=up, down=down, axis=0)
    return y.astype(float), fs_out
