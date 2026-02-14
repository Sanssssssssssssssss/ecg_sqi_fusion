# src/augment/noise/dower.py
from __future__ import annotations

import numpy as np


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# A commonly used inverse Dower matrix mapping orthogonal XYZ -> 12-lead ECG.
# Shape: (12, 3)
# Reference: widely used "inverse Dower transform" coefficients (approx).
_IDOWER_12x3 = np.array(
    [
        [ 0.156, -0.010, -0.172],  # I
        [ 0.057, -0.019, -0.106],  # II
        [-0.099, -0.009,  0.066],  # III
        [-0.231,  0.019,  0.278],  # aVR
        [ 0.152,  0.010, -0.112],  # aVL
        [ 0.080, -0.029, -0.178],  # aVF
        [-0.239, -0.310,  0.246],  # V1
        [-0.122, -0.231,  0.174],  # V2
        [ 0.015, -0.151,  0.106],  # V3
        [ 0.181, -0.071,  0.041],  # V4
        [ 0.290, -0.010, -0.010],  # V5
        [ 0.243,  0.044, -0.022],  # V6
    ],
    dtype=np.float64
)


def dower_3_to_12(noise3: np.ndarray) -> np.ndarray:
    """
    Map (N,3) orthogonal leads -> (N,12) leads via inverse Dower matrix.
    """
    x = np.asarray(noise3, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"noise3 must be (N,3), got {x.shape}")
    y12 = x @ _IDOWER_12x3.T  # (N,12)
    return y12