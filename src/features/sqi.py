# src/features/sqi.py
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis


MOMENT_STD_EPS = 1e-8  # mV level
# ---- Li 2008 style bSQI/iSQI config (fixed) ----
BSQI_WIN_SEC = 10.0         # w = 10s
BSQI_HALF_SEC = BSQI_WIN_SEC / 2.0  # ±5s

# tolerance is passed in as tol_samp, but the intended default is 150ms in paper
# ------------------------------------------------

def _clean_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    return x


def welch_psd(
    x: np.ndarray,
    fs: int,
    fmax: float = 40.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int = 128,
    detrend: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    nper = min(nperseg, len(x))
    nov = min(noverlap, max(0, nper // 2))
    f, p = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nper,
        noverlap=nov,
        detrend=detrend,
        scaling="density",
    )
    m = (f >= 0) & (f <= fmax)
    return f[m], p[m]


def band_power(f: np.ndarray, p: np.ndarray, f_lo: float, f_hi: float) -> float:
    m = (f >= f_lo) & (f <= f_hi)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(p[m], f[m]))


def sqi_pSQI(x: np.ndarray, fs: int, welch_kwargs: dict) -> float:
    # pSQI = ∫5-15 P / ∫5-40 P
    f, p = welch_psd(x, fs=fs, **welch_kwargs)
    num = band_power(f, p, 5.0, 15.0)
    den = band_power(f, p, 5.0, 40.0) + 1e-12
    return float(num / den)


def sqi_basSQI(x: np.ndarray, fs: int, welch_kwargs: dict) -> float:
    # basSQI = 1 - ∫0-1 P / ∫0-40 P
    f, p = welch_psd(x, fs=fs, **welch_kwargs)
    num = band_power(f, p, 0.0, 1.0)
    den = band_power(f, p, 0.0, 40.0) + 1e-12
    return float(1.0 - (num / den))


def sqi_sSQI(x: np.ndarray, std_eps: float = MOMENT_STD_EPS) -> float:
    """
    sSQI = skewness. protect std
    0.0 return represent“no skewness”。
    """
    x = _clean_1d(x)
    if x.size < 3:
        return 0.0
    s = np.std(x, ddof=1)
    if not np.isfinite(s) or s < std_eps:
        return 0.0
    v = skew(x, bias=False)
    return float(v) if np.isfinite(v) else 0.0

def sqi_kSQI(x: np.ndarray, std_eps: float = MOMENT_STD_EPS) -> float:
    """
    kSQI = kurtosis (Pearson, fisher=False). protect std
    0.0 return represent“no kurtosis”.
    """
    x = _clean_1d(x)
    if x.size < 4:
        return 0.0
    s = np.std(x, ddof=1)
    if not np.isfinite(s) or s < std_eps:
        return 0.0
    v = kurtosis(x, fisher=False, bias=False)
    return float(v) if np.isfinite(v) else 0.0


def sqi_fSQI(x: np.ndarray, flatline_eps: float = 1e-4) -> float:
    # flatline proxy = fraction of |diff| < eps
    dx = np.abs(np.diff(x))
    if dx.size == 0:
        return 1.0
    return float(np.mean(dx < flatline_eps))


# ---------------- beat matching utils ----------------

def match_count(ref: np.ndarray, other: np.ndarray, tol_samp: int) -> int:
    """
    Count how many ref beats have a match in other within ±tol.
    Greedy two-pointer matching, both arrays sorted.
    """
    if len(ref) == 0 or len(other) == 0:
        return 0
    i = j = 0
    c = 0
    while i < len(ref) and j < len(other):
        d = other[j] - ref[i]
        if abs(d) <= tol_samp:
            c += 1
            i += 1
            j += 1
        elif d < -tol_samp:
            j += 1
        else:
            i += 1
    return c

def _window_slice(peaks: np.ndarray, center: int, half_w_samp: int) -> np.ndarray:
    """Return peaks within [center-half_w, center+half_w]. peaks sorted."""
    if peaks.size == 0:
        return peaks
    lo = center - half_w_samp
    hi = center + half_w_samp
    i0 = int(np.searchsorted(peaks, lo, side="left"))
    i1 = int(np.searchsorted(peaks, hi, side="right"))
    return peaks[i0:i1]


def _union_count(n1: int, n2: int, n_matched: int) -> int:
    """N_all = N1 + N2 - N_matched (no double counting)."""
    return int(n1 + n2 - n_matched)

#
# def sqi_bSQI(r1: np.ndarray, r2: np.ndarray, tol_samp: int, denom: str = "r2") -> float:
#     """
#     Agreement between two detectors.
#     denom='r2' => fraction of r2 beats matched by r1
#     denom='r1' => fraction of r1 beats matched by r2
#     note this version is not symmetric
#     """
#     if denom == "r2":
#         base, other = r2, r1
#     else:
#         base, other = r1, r2
#     if len(base) == 0:
#         return 0.0
#     m = match_count(base, other, tol_samp)
#     return float(m / (len(base) + 1e-12))
#
#
# def isqi_all_leads_intersection(rpeaks_by_lead: list[np.ndarray], tol_samp: int, anchor_lead: int = 1) -> int:
#     """
#     Compute intersection count: beats present in ALL leads (within tol),
#     using an anchor lead to enumerate candidate beats (default lead II index=1).
#     """
#     anchor = rpeaks_by_lead[anchor_lead]
#     if len(anchor) == 0:
#         return 0
#
#     n_inter = 0
#     for t in anchor:
#         ok = True
#         for li, r in enumerate(rpeaks_by_lead):
#             if li == anchor_lead:
#                 continue
#             if len(r) == 0:
#                 ok = False
#                 break
#             k = np.searchsorted(r, t)
#             candidates = []
#             if k < len(r):
#                 candidates.append(abs(r[k] - t))
#             if k > 0:
#                 candidates.append(abs(r[k - 1] - t))
#             if (min(candidates) if candidates else 10**9) > tol_samp:
#                 ok = False
#                 break
#         if ok:
#             n_inter += 1
#     return n_inter
#
#
# def sqi_iSQI_per_lead(
#     rpeaks_by_lead: list[np.ndarray],
#     tol_samp: int,
#     anchor_lead: int = 1
# ) -> list[float]:
#     """
#     iSQI per lead = (# beats in all-leads intersection) / (# beats in this lead).
#     Paper uses all-leads intersection idea.
#     """
#     n_inter = isqi_all_leads_intersection(rpeaks_by_lead, tol_samp, anchor_lead=anchor_lead)
#     out = []
#     for r in rpeaks_by_lead:
#         out.append(float(n_inter / (len(r) + 1e-12)))
#     return out


#

def sqi_bSQI_li2008_global(r1: np.ndarray, r2: np.ndarray, tol_samp: int) -> float:
    """
    Li2008-style bSQI but computed over the full segment (no sliding window):
      bSQI = Nmatched / Nall
      Nall = N1 + N2 - Nmatched
    Symmetric by construction.
    """
    r1 = np.asarray(r1, dtype=int)
    r2 = np.asarray(r2, dtype=int)
    if r1.size == 0 and r2.size == 0:
        return 0.0
    if r1.size == 0 or r2.size == 0:
        return 0.0

    n_matched = match_count(r1, r2, tol_samp)
    n_all = int(len(r1) + len(r2) - n_matched)
    if n_all <= 0:
        return 0.0
    return float(n_matched / (n_all + 1e-12))


def sqi_iSQI_li2008_global_per_lead(
    rpeaks_by_lead: list[np.ndarray],
    tol_samp: int,
) -> list[float]:
    """
    Li2008-style iSQI but computed over the full segment (no sliding window):
      iSQI_i = max_{j!=i} ( Nmatched(i,j) / Nall(i,j) )
      Nall(i,j) = Ni + Nj - Nmatched(i,j)
    Returns a list length=12.
    """
    L = len(rpeaks_by_lead)
    r = [np.asarray(x, dtype=int) for x in rpeaks_by_lead]
    if L == 0:
        return []
    out: list[float] = []

    for i in range(L):
        if r[i].size == 0:
            out.append(0.0)
            continue

        best = 0.0
        for j in range(L):
            if j == i or r[j].size == 0:
                continue
            n_matched = match_count(r[i], r[j], tol_samp)
            n_all = int(len(r[i]) + len(r[j]) - n_matched)
            if n_all > 0:
                best = max(best, float(n_matched / (n_all + 1e-12)))

        out.append(best)

    return out

