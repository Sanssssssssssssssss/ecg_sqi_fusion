"""Transformer model definitions.

`mtl_transformer.py` keeps the legacy E3.11 multi-task Transformer baseline.
`uformer1d.py` is the current E3.11f Uformer denoise + detached SQI mainline.
"""

from src.transformer_pipeline.models.uformer1d import (
    SQIMLPHead,
    Uformer1DConfig,
    Uformer1DResidualDenoiser,
    UformerDenoiseSQIModel,
)

__all__ = [
    "SQIMLPHead",
    "Uformer1DConfig",
    "Uformer1DResidualDenoiser",
    "UformerDenoiseSQIModel",
]
