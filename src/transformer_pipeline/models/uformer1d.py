"""Uformer-style 1D residual denoiser and detached SQI classifier head.

This is the E3.11f thesis/mainline model family.  The denoiser predicts
residual noise, then the SQI classifier reads detached multi-scale denoiser
features plus signal summaries.  The detach is intentional: CE should learn
from the mature denoising representation without corrupting waveform recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Uformer1DConfig:
    noise_scale: float = 0.9
    dropout: float = 0.05
    feature_set: str = "full_tokens"
    detach_encoder_features: bool = True
    head_hidden_dim: int = 128
    denoiser_width: float = 1.0


def scale_channels(base: int, width: float) -> int:
    if abs(float(width) - 1.0) < 1e-9:
        return int(base)
    return max(8, int(round(base * float(width) / 8.0) * 8))


def compatible_heads(dim: int, preferred: int) -> int:
    for heads in [preferred, 8, 6, 5, 4, 3, 2, 1]:
        if dim % heads == 0:
            return heads
    return 1


def tensor_summary(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    return torch.cat(
        [
            flat.mean(1, keepdim=True),
            flat.std(1, unbiased=False, keepdim=True),
            flat.abs().amax(1, keepdim=True),
            torch.sqrt(torch.mean(flat * flat, dim=1, keepdim=True) + 1e-8),
        ],
        dim=1,
    )


def pooled_channels(x: torch.Tensor) -> torch.Tensor:
    avg = F.adaptive_avg_pool1d(x, 1).flatten(1)
    mx = F.adaptive_max_pool1d(x.abs(), 1).flatten(1)
    return torch.cat([avg, mx], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        groups = max(1, min(8, out_ch // 4))
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock1D(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        h = self.ln1(tokens)
        attn, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + self.drop(attn)
        tokens = tokens + self.mlp(self.ln2(tokens))
        return tokens.transpose(1, 2)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class Uformer1DResidualDenoiser(nn.Module):
    """Conv local stem + hierarchical Transformer encoder + U-shaped decoder."""

    def __init__(self, dropout: float = 0.05, width: float = 1.0):
        super().__init__()
        c1 = scale_channels(32, width)
        c2 = scale_channels(64, width)
        c3 = scale_channels(128, width)
        cb = scale_channels(160, width)
        u3 = scale_channels(96, width)
        u2 = scale_channels(64, width)
        u1 = scale_channels(32, width)
        self.e1 = ConvBlock(1, c1, kernel=9)
        self.d1 = nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(ConvBlock(c2, c2), TransformerBlock1D(c2, compatible_heads(c2, 4), dropout=dropout))
        self.d2 = nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.e3 = nn.Sequential(ConvBlock(c3, c3), TransformerBlock1D(c3, compatible_heads(c3, 4), dropout=dropout))
        self.d3 = nn.Conv1d(c3, cb, kernel_size=4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ConvBlock(cb, cb),
            TransformerBlock1D(cb, compatible_heads(cb, 5), dropout=dropout),
            TransformerBlock1D(cb, compatible_heads(cb, 5), dropout=dropout),
        )
        self.u3 = UpBlock(cb, c3, u3)
        self.u2 = UpBlock(u3, c2, u2)
        self.u1 = UpBlock(u2, c1, u1)
        self.out = nn.Conv1d(u1, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        s1 = self.e1(x)
        s2 = self.e2(F.gelu(self.d1(s1)))
        s3 = self.e3(F.gelu(self.d2(s2)))
        b = self.bottleneck(F.gelu(self.d3(s3)))
        y = self.u3(b, s3)
        y = self.u2(y, s2)
        y = self.u1(y, s1)
        noise_hat = self.out(y)
        return {"noise_hat": noise_hat, "features": [s1, s2, s3, b], "bottleneck": b}


def feature_vector(
    feature_set: str,
    noisy: torch.Tensor,
    denoise: torch.Tensor,
    features: list[torch.Tensor],
    bottleneck: torch.Tensor,
) -> torch.Tensor:
    residual = torch.abs(noisy - denoise)
    summaries = {
        "summary_only": torch.cat([tensor_summary(noisy), tensor_summary(denoise), tensor_summary(residual)], dim=1),
        "residual_summary_only": tensor_summary(residual),
        "bottleneck_only": pooled_channels(bottleneck),
    }
    if feature_set in summaries:
        return summaries[feature_set]
    if feature_set != "full_tokens":
        raise ValueError(f"unknown feature_set={feature_set}")
    pooled = [pooled_channels(x) for x in features]
    return torch.cat([*pooled, summaries["summary_only"]], dim=1)


class SQIMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UformerDenoiseSQIModel(nn.Module):
    def __init__(self, cfg: Uformer1DConfig, head: SQIMLPHead | None = None):
        super().__init__()
        self.cfg = cfg
        self.denoiser = Uformer1DResidualDenoiser(dropout=cfg.dropout, width=cfg.denoiser_width)
        self.head = head

    def infer_feature_dim(self, seq_len: int = 1250, device: torch.device | None = None) -> int:
        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            sample = torch.zeros(2, 1, seq_len, device=device)
            dim = int(self(sample)["features"].shape[1])
        self.train(was_training)
        return dim

    def attach_head(self, in_dim: int | None = None) -> None:
        if in_dim is None:
            in_dim = self.infer_feature_dim()
        self.head = SQIMLPHead(in_dim, self.cfg.head_hidden_dim, self.cfg.dropout)

    def forward(self, noisy: torch.Tensor) -> dict[str, torch.Tensor | None]:
        denoiser_out = self.denoiser(noisy)
        noise_hat = denoiser_out["noise_hat"]
        assert isinstance(noise_hat, torch.Tensor)
        features = denoiser_out["features"]
        bottleneck = denoiser_out["bottleneck"]
        assert isinstance(features, list)
        assert isinstance(bottleneck, torch.Tensor)
        denoise = noisy - float(self.cfg.noise_scale) * noise_hat
        feat = feature_vector(self.cfg.feature_set, noisy, denoise, features, bottleneck)
        if self.cfg.detach_encoder_features:
            feat = feat.detach()
        logits = self.head(feat) if self.head is not None else None
        return {"noise_hat": noise_hat, "denoise": denoise, "features": feat, "logits": logits}
