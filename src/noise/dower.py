# src/noise/dower.py
from __future__ import annotations
import numpy as np

# Dower transform matrix: s = D v
# s = [V1 V2 V3 V4 V5 V6 I II]^T , v = [X Y Z]^T
DOWER_D = np.array([
    [-0.515,  0.157, -0.917],  # V1
    [ 0.044,  0.164, -1.387],  # V2
    [ 0.882,  0.098, -1.277],  # V3
    [ 1.213,  0.127, -0.601],  # V4
    [ 1.125,  0.127, -0.086],  # V5
    [ 0.831,  0.076,  0.230],  # V6
    [ 0.632, -0.235,  0.059],  # I
    [ 0.235,  1.066, -0.132],  # II
], dtype=float)

def vcg_to_8lead(vcg_xyz: np.ndarray) -> np.ndarray:
    """
    vcg_xyz: (N, 3) -> returns (N, 8) in order [V1..V6, I, II]
    """
    assert vcg_xyz.ndim == 2 and vcg_xyz.shape[1] == 3
    return vcg_xyz @ DOWER_D.T

def lead8_to_lead12(v8: np.ndarray) -> np.ndarray:
    """
    v8: (N, 8) in order [V1..V6, I, II]
    returns (N, 12) in standard order:
    [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
    """
    assert v8.ndim == 2 and v8.shape[1] == 8
    V1,V2,V3,V4,V5,V6,I,II = [v8[:,i] for i in range(8)]
    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0
    return np.column_stack([I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6])

def vcg_to_12lead(vcg_xyz: np.ndarray) -> np.ndarray:
    return lead8_to_lead12(vcg_to_8lead(vcg_xyz))
