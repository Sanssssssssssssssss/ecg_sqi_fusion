# src/sqi/sqi_7.py
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis


def _match_beats(a: np.ndarray, b: np.ndarray, tol: int) -> int:
    """
    Count matched beats between sorted integer arrays a and b within +/- tol samples.
    Greedy two-pointer.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    i = j = 0
    m = 0
    while i < len(a) and j < len(b):
        da = a[i] - b[j]
        if abs(da) <= tol:
            m += 1
            i += 1
            j += 1
        elif da < -tol:
            i += 1
        else:
            j += 1
    return m


def bsqi(beats1: np.ndarray, beats2: np.ndarray, tol: int) -> float:
    """
    bSQI: agreement between two detectors.
    Common definition: 2M/(N1+N2)
    """
    n1 = len(beats1)
    n2 = len(beats2)
    if n1 + n2 == 0:
        return 0.0
    m = _match_beats(beats1, beats2, tol=tol)
    return float(2.0 * m / (n1 + n2))


def isqi_pairwise(beats_list: list[np.ndarray], lead_idx: int, tol: int) -> float:
    """
    iSQI proxy: average beat agreement between this lead and all other leads
    using ONE detector's beat lists (e.g., XQRS).
    """
    b_i = beats_list[lead_idx]
    if len(beats_list) <= 1:
        return 0.0
    scores = []
    for j, b_j in enumerate(beats_list):
        if j == lead_idx:
            continue
        n = len(b_i) + len(b_j)
        if n == 0:
            scores.append(0.0)
            continue
        m = _match_beats(b_i, b_j, tol=tol)
        scores.append(2.0 * m / n)
    return float(np.mean(scores)) if scores else 0.0


def _bandpower(f: np.ndarray, pxx: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (f >= f_lo) & (f < f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(pxx[mask], f[mask]))


def psqi(sig: np.ndarray, fs: float) -> float:
    """
    pSQI: power(5-15Hz) / power(5-40Hz)
    """
    f, pxx = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
    p_5_15 = _bandpower(f, pxx, 5.0, 15.0)
    p_5_40 = _bandpower(f, pxx, 5.0, 40.0)
    return float(p_5_15 / (p_5_40 + 1e-12))


def bassqi(sig: np.ndarray, fs: float) -> float:
    """
    basSQI: 1 - power(0-1Hz)/power(0-40Hz)
    """
    f, pxx = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
    p_0_1 = _bandpower(f, pxx, 0.0, 1.0)
    p_0_40 = _bandpower(f, pxx, 0.0, 40.0)
    return float(1.0 - p_0_1 / (p_0_40 + 1e-12))


def fsqi(sig: np.ndarray, eps: float = 1e-8) -> float:
    """
    fSQI: flatline ratio proxy = fraction of near-zero first differences
    """
    d = np.diff(sig)
    return float(np.mean(np.abs(d) < eps)) if len(d) else 0.0


def ssqi(sig: np.ndarray) -> float:
    x = np.asarray(sig, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    # near-constant -> skew undefined; return 0
    if np.std(x) < 1e-12:
        return 0.0
    v = float(skew(x, bias=False))
    return 0.0 if (not np.isfinite(v)) else v


def ksqi(sig: np.ndarray) -> float:
    x = np.asarray(sig, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    # near-constant -> kurtosis undefined; return 3 (normal) or 0
    if np.std(x) < 1e-12:
        return 3.0
    v = float(kurtosis(x, fisher=False, bias=False))
    return 3.0 if (not np.isfinite(v)) else v

