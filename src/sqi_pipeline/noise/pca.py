# src/augment/noise/pca.py
from __future__ import annotations

import numpy as np


def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    sd = np.std(x) + eps
    return (x - mu) / sd


def _gram_schmidt_orthonormal(vecs: list[np.ndarray], eps: float = 1e-12) -> list[np.ndarray]:
    """
    Orthonormalize a list of 1D vectors (length N) using Gram-Schmidt.
    Returns list of orthonormal vectors (same length).
    """
    out: list[np.ndarray] = []
    for v in vecs:
        u = np.asarray(v, dtype=np.float64).copy()
        for b in out:
            u = u - np.dot(u, b) * b
        n = np.linalg.norm(u)
        if n <= eps:
            continue
        out.append(u / n)
    return out


def make_3_orthogonal_from_2(
    noise2: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Input:
      noise2: (N, 2) noise leads from NSTDB (after resampling to target fs).

    Output:
      noise3: (N, 3) orthogonal (approximately) noise leads.

    Rationale (matches paper intent):
      - Use PCA to find dominant component direction in the 2-lead noise.
      - Produce two orthogonal components (PC1, PC2) spanning the 2-lead space.
      - Create a third lead with "arbitrary orientation" but orthogonal to PC1/PC2
        in the N-dimensional sample space (time-series vectors), via random vector
        + Gram-Schmidt.

    NOTE:
      In strict 2D lead-space you cannot create a third independent axis, so this
      step is necessarily a pragmatic construction in sample-space, consistent
      with the paper's "assume arbitrary orientation" statement.
    """
    x = np.asarray(noise2, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"noise2 must be (N,2), got {x.shape}")

    # Center
    x0 = x - np.mean(x, axis=0, keepdims=True)

    # PCA on 2D (cov 2x2)
    cov = (x0.T @ x0) / max(1, x0.shape[0] - 1)
    w, v = np.linalg.eigh(cov)  # ascending
    v = v[:, np.argsort(w)[::-1]]  # columns: [pc1_dir, pc2_dir]

    pc1 = x0 @ v[:, 0]  # (N,)
    pc2 = x0 @ v[:, 1]  # (N,)

    # Normalize pc1/pc2 in time-series space and orthonormalize
    b1 = _zscore(pc1)
    b2 = _zscore(pc2)
    basis = _gram_schmidt_orthonormal([b1, b2], eps=eps)
    if len(basis) < 2:
        # fallback: use original channels
        c1 = _zscore(x0[:, 0])
        c2 = _zscore(x0[:, 1])
        basis = _gram_schmidt_orthonormal([c1, c2], eps=eps)

    # Create third by random vector, orthogonalize vs basis
    r = rng.standard_normal(size=x0.shape[0])
    basis3 = _gram_schmidt_orthonormal([basis[0], basis[1], r], eps=eps)
    if len(basis3) < 3:
        # fallback: use a shifted random vector
        r2 = np.roll(r, 1)
        basis3 = _gram_schmidt_orthonormal([basis[0], basis[1], r2], eps=eps)
        if len(basis3) < 3:
            raise RuntimeError("Failed to construct 3 orthogonal noise leads.")

    o1, o2, o3 = basis3[0], basis3[1], basis3[2]

    # Scale back to roughly match original noise RMS (per lead)
    target_rms = float(np.sqrt(np.mean(x0[:, 0] ** 2) + np.mean(x0[:, 1] ** 2)) / np.sqrt(2.0) + eps)
    def scale_to_rms(u: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(u**2)) + eps)
        return u * (target_rms / rms)

    y1 = scale_to_rms(o1)
    y2 = scale_to_rms(o2)
    y3 = scale_to_rms(o3)

    noise3 = np.stack([y1, y2, y3], axis=1)  # (N,3)
    return noise3
