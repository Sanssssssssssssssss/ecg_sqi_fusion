# src/noise/make_12lead_noise.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import wfdb
from scipy.signal import resample_poly

from src.utils.paths import project_root
from src.noise.pca_orthogonal import twolead_to_three_orthogonal
from src.noise.dower import vcg_to_12lead


@dataclass(frozen=True)
class NoiseWindow:
    kind: str          # "em" or "ma"
    split: str         # "train" or "test"
    start_s: float     # start time (seconds) within the FULL record
    duration_s: float  # duration (seconds)
    fs_src: float      # NSTDB sampling rate
    fs_target: float   # target sampling rate
    start_idx: int     # sample index in source fs
    end_idx: int       # sample index in source fs (exclusive)


def _nstdb_record_path(kind: str) -> str:
    root = project_root()
    rec = root / "data" / "physionet" / "nstdb" / kind
    return str(rec)


def _read_nstdb_header(kind: str) -> wfdb.Record:
    rec = _nstdb_record_path(kind)
    return wfdb.rdheader(rec)


def _half_bounds(sig_len: int, split: str) -> tuple[int, int]:
    mid = sig_len // 2
    if split == "train":
        return 0, mid
    if split == "test":
        return mid, sig_len
    raise ValueError("split must be 'train' or 'test'")


def pick_noise_window(
    kind: str,
    split: str,
    duration_s: float,
    seed: int = 0,
) -> NoiseWindow:
    """
    Choose a random window fully inside the requested half.
    The window is defined in SOURCE sampling rate coordinates.
    """
    if kind not in ("em", "ma"):
        raise ValueError("kind must be 'em' or 'ma'")

    h = _read_nstdb_header(kind)
    fs_src = float(h.fs)
    sig_len = int(h.sig_len)

    lo, hi = _half_bounds(sig_len, split)

    raw_len = int(round(duration_s * fs_src))
    if raw_len <= 0:
        raise ValueError("duration_s must be > 0")

    # last possible start so that [start, start+raw_len) stays within [lo,hi)
    max_start = hi - raw_len
    if max_start <= lo:
        raise ValueError(
            f"Requested duration {duration_s}s too long for {kind} {split} half. "
            f"half_len_samples={hi-lo}, raw_len={raw_len}, fs_src={fs_src}"
        )

    rng = np.random.default_rng(seed)
    start_idx = int(rng.integers(lo, max_start))
    end_idx = start_idx + raw_len

    start_s = start_idx / fs_src
    return NoiseWindow(
        kind=kind,
        split=split,
        start_s=float(start_s),
        duration_s=float(duration_s),
        fs_src=fs_src,
        fs_target=np.nan,  # will be filled later
        start_idx=start_idx,
        end_idx=end_idx,
    )


def _resample_to_target(x: np.ndarray, fs_src: float, fs_target: float, target_len: int) -> np.ndarray:
    """
    Resample multi-channel signal x (N, C) from fs_src to fs_target using resample_poly.
    Ensures output length == target_len by crop/pad.
    """
    if abs(fs_src - fs_target) < 1e-9:
        y = x
    else:
        # Use a rational approximation with limited denominator for stability.
        # (Good enough for 360->500 etc.)
        from fractions import Fraction
        frac = Fraction(fs_target / fs_src).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        y = resample_poly(x, up=up, down=down, axis=0)

    # force exact length
    if y.shape[0] < target_len:
        y = np.pad(y, ((0, target_len - y.shape[0]), (0, 0)))
    elif y.shape[0] > target_len:
        y = y[:target_len, :]
    return y


def make_12lead_noise(
    kind: str,
    split: str,
    duration_s: float = 10.0,
    fs_target: float = 500.0,
    seed: int = 0,
    start_idx: int | None = None,
) -> tuple[np.ndarray, NoiseWindow]:
    """
    Generate correlated 12-lead noise (duration_s seconds) from NSTDB em/ma.

    Returns:
        noise12: (T, 12) where T = round(duration_s * fs_target)
        meta: NoiseWindow describing what source window was used

    Key properties (matches paper intent):
      - uses only the requested half of NSTDB record to avoid leakage
      - window length is defined in seconds (not samples)
      - output is always exactly duration_s*fs_target samples
      - reads only a slice from disk (fast)
      - PCA->3 orthogonal components -> Dower -> 12-lead correlation structure
    """
    if kind not in ("em", "ma"):
        raise ValueError("kind must be 'em' or 'ma'")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    # header for bounds + fs
    h = _read_nstdb_header(kind)
    fs_src = float(h.fs)
    sig_len = int(h.sig_len)
    lo, hi = _half_bounds(sig_len, split)

    raw_len = int(round(duration_s * fs_src))
    target_len = int(round(duration_s * fs_target))

    # choose start
    if start_idx is None:
        max_start = hi - raw_len
        if max_start <= lo:
            raise ValueError(
                f"Requested duration {duration_s}s too long for {kind} {split} half. "
                f"half_len_samples={hi-lo}, raw_len={raw_len}, fs_src={fs_src}"
            )
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(lo, max_start))
    else:
        # validate
        if start_idx < lo or (start_idx + raw_len) > hi:
            raise ValueError(
                f"start_idx window [{start_idx},{start_idx+raw_len}) outside {split} half [{lo},{hi})"
            )

    end_idx = start_idx + raw_len

    # read only the needed slice
    rec = _nstdb_record_path(kind)
    seg2, fields = wfdb.rdsamp(rec, sampfrom=start_idx, sampto=end_idx)  # (raw_len, 2)
    seg2 = np.asarray(seg2, dtype=float)

    # resample to target
    seg2 = _resample_to_target(seg2, fs_src=fs_src, fs_target=fs_target, target_len=target_len)

    # build 3 orthogonal components (paper says PCA creates 3rd orthogonal lead;
    # paper doesn't specify exact method for 3rd, ours uses phase randomization + GS)
    vcg3 = twolead_to_three_orthogonal(seg2, seed=seed)  # (target_len, 3)

    # Dower mapping to 12-lead
    noise12 = vcg_to_12lead(vcg3)  # (target_len, 12)

    meta = NoiseWindow(
        kind=kind,
        split=split,
        start_s=float(start_idx / fs_src),
        duration_s=float(duration_s),
        fs_src=fs_src,
        fs_target=float(fs_target),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
    )
    return noise12, meta
