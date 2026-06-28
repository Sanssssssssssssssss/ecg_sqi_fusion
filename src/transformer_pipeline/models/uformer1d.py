"""Legacy compatibility shim.

The old Uformer mainline was removed. Some archived experiment helpers still
import these names at module import time while building v116 data; fail only if
that old model is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Uformer1DConfig:
    noise_scale: float = 0.9
    dropout: float = 0.05
    feature_set: str = "full_tokens"
    detach_encoder_features: bool = True
    head_hidden_dim: int = 128
    denoiser_width: float = 1.0


class UformerDenoiseSQIModel:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("legacy Uformer mainline was removed; use data_v1_gapfill E31")


def feature_vector(*_args, **_kwargs):
    raise RuntimeError("legacy Uformer feature extraction was removed; use data_v1_gapfill E31")
