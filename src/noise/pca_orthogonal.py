# src/noise/pca_orthogonal.py
from __future__ import annotations
import numpy as np

def _phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep magnitude spectrum, randomize phases -> similar PSD, different realization."""
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = rng.uniform(0, 2*np.pi, size=mag.shape)
    # keep DC & Nyquist phase = 0 for real signal stability
    phase[0] = 0.0
    if len(phase) > 1 and (x.size % 2 == 0):
        phase[-1] = 0.0
    Y = mag * np.exp(1j * phase)
    y = np.fft.irfft(Y, n=x.size)
    return y

def _gs_orthogonalize(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Make v orthogonal to all vectors in basis (in sample inner-product sense)."""
    u = v.astype(float).copy()
    for b in basis:
        denom = float(np.dot(b, b))
        if denom > 1e-12:
            u = u - (np.dot(u, b) / denom) * b
    return u

def twolead_to_three_orthogonal(x2: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    x2: (N,2) -> (N,3) [pc1, pc2, pc3]
    pc1,pc2 from PCA; pc3 from phase-rand(pc1) then GS-orthogonalized to pc1,pc2.
    """
    assert x2.ndim == 2 and x2.shape[1] == 2
    rng = np.random.default_rng(seed)

    X = x2 - x2.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U * S  # (N,2): pc1, pc2
    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]

    cand = _phase_randomize(pc1, rng)
    pc3 = _gs_orthogonalize(cand, [pc1, pc2])

    # scale pc3 to similar RMS as pc1 (avoid tiny energy after orthogonalization)
    rms1 = np.sqrt(np.mean(pc1**2)) + 1e-12
    rms3 = np.sqrt(np.mean(pc3**2)) + 1e-12
    pc3 = pc3 * (rms1 / rms3)

    return np.column_stack([pc1, pc2, pc3])
