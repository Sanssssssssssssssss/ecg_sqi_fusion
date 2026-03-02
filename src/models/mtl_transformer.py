# src/models/mtl_transformer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class MTLTransformerConfig:
    # input spec
    fs: int = 125
    window_sec: int = 10
    in_ch: int = 1
    T: int = 1250  # fixed: 10s@125Hz

    # patch spec (paper)
    patch_size: int = 20
    stride: int = 10  # 50% overlap
    L: int = 124  # derived from T=(L-1)*stride + patch_size

    # patch embedding conv stack (paper)
    conv1_k: int = 15
    conv1_out: int = 32
    conv2_k: int = 7
    conv2_out: int = 64

    conv3_k: int = 20
    conv3_stride: int = 10
    conv3_dilation: int = 2
    conv3_out: int = 128

    # [ASSUMPTION] padding rules:
    # conv1/conv2 not specified -> use "same" padding to keep length stable.
    # conv3 padding chosen to make output token length exactly L=124 under T=1250.
    conv1_padding: Optional[int] = None  # if None => same
    conv2_padding: Optional[int] = None  # if None => same
    conv3_padding: int = 10

    # encoder (paper)
    enc_layers: int = 6
    enc_d_model: int = 128
    enc_heads: int = 8
    enc_mlp_ratio: int = 3  # d_ff = 384

    # decoder (paper)
    dec_in: int = 128
    dec_d_model: int = 64
    dec_layers: int = 3
    dec_heads: int = 8
    dec_mlp_ratio: int = 3  # d_ff = 192

    # heads (paper)
    head_patch_out: int = 20  # per token predicts 20 points

    # [ASSUMPTION] dropout not specified -> default 0 for strict reproducible baseline.
    dropout: float = 0.1

    # smooth (paper says only for denoise head)
    smooth_k: int = 5


# ============================================================
# Utilities
# ============================================================
def _same_padding_1d(kernel: int, dilation: int = 1, stride: int = 1) -> int:
    # For stride=1 only; for stride>1 "same" is ambiguous, so we only use this for conv1/conv2.
    eff = dilation * (kernel - 1) + 1
    return (eff - 1) // 2


def unpatchify_overlap_add(
    patches: torch.Tensor,  # (B, L, P)
    T: int,
    stride: int,
) -> torch.Tensor:
    """
    Overlap-add reconstruction with average in overlaps.
    patches: (B, L, patch_size)
    returns: (B, T)
    """
    B, L, P = patches.shape
    device = patches.device
    out = torch.zeros((B, T), device=device, dtype=patches.dtype)
    cnt = torch.zeros((B, T), device=device, dtype=patches.dtype)

    for i in range(L):
        s = i * stride
        e = s + P
        out[:, s:e] += patches[:, i, :]
        cnt[:, s:e] += 1.0

    out = out / torch.clamp(cnt, min=1.0)
    return out


# ============================================================
# Transformer Blocks (Encoder-style, no cross-attn)
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        d_ff = d_model * mlp_ratio

        # [ASSUMPTION] pre-LN vs post-LN not specified -> use pre-LN (common, stable).
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out) # residual
        x = x + self.mlp(self.ln2(x)) # second residual
        return x


# ============================================================
# Model
# ============================================================
class MTLTransformerPTBXL(nn.Module):
    """
    Strict paper-style backbone:
      input -> conv-first patch embedding -> encoder -> linear -> decoder (self-attn only)
      -> 3 heads: denoise (patch->unpatchify->smooth), noise-level (patch->unpatchify), cls (GAP->FC)
    """
    def __init__(self, cfg: MTLTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # --------- Patch Embedding (conv-first) ---------
        p1 = _same_padding_1d(cfg.conv1_k) if cfg.conv1_padding is None else cfg.conv1_padding
        p2 = _same_padding_1d(cfg.conv2_k) if cfg.conv2_padding is None else cfg.conv2_padding

        self.conv1 = nn.Conv1d(cfg.in_ch, cfg.conv1_out, kernel_size=cfg.conv1_k, stride=1, padding=p1)
        self.bn1 = nn.BatchNorm1d(cfg.conv1_out)
        self.conv2 = nn.Conv1d(cfg.conv1_out, cfg.conv2_out, kernel_size=cfg.conv2_k, stride=1, padding=p2)
        self.bn2 = nn.BatchNorm1d(cfg.conv2_out)

        # Paper: conv3 produces tokens; LN+ReLU after it
        # [ASSUMPTION] conv3 padding not specified -> set to cfg.conv3_padding (chosen to hit L=124).
        self.conv3 = nn.Conv1d(
            cfg.conv2_out,
            cfg.conv3_out,
            kernel_size=cfg.conv3_k,
            stride=cfg.conv3_stride,
            dilation=cfg.conv3_dilation,
            padding=cfg.conv3_padding,
        )
        self.ln3 = nn.LayerNorm(cfg.conv3_out)

        # --------- Encoder ---------
        self.enc = nn.ModuleList(
            [TransformerBlock(cfg.enc_d_model, cfg.enc_heads, cfg.enc_mlp_ratio, cfg.dropout) for _ in range(cfg.enc_layers)]
        )

        # --------- Decoder (linear downproj + encoder-style blocks; no cross-attn) ---------
        self.dec_in = nn.Linear(cfg.dec_in, cfg.dec_d_model)
        self.dec = nn.ModuleList(
            [TransformerBlock(cfg.dec_d_model, cfg.dec_heads, cfg.dec_mlp_ratio, cfg.dropout) for _ in range(cfg.dec_layers)]
        )

        # --------- Heads ---------
        # Head1 denoise: token -> 20 points
        self.head_denoise = nn.Linear(cfg.dec_d_model, cfg.head_patch_out)

        # Smooth layer (only for denoise)
        # [ASSUMPTION] padding uses "same" behavior for stride=1 so output length preserved.
        smooth_pad = (cfg.smooth_k - 1) // 2
        self.smooth = nn.Conv1d(1, 1, kernel_size=cfg.smooth_k, stride=1, padding=smooth_pad, bias=False)

        # Head2 noise-level: token -> 20 points
        self.head_level = nn.Linear(cfg.dec_d_model, cfg.head_patch_out)

        # Head3 classification: GAP -> FC
        # [ASSUMPTION] dropout prob not specified -> cfg.dropout (default 0).
        self.cls_drop = nn.Dropout(cfg.dropout)
        self.cls_fc = nn.Linear(cfg.dec_d_model, 3)

    # --------- Internal: patch embedding forward ---------
    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, T)
        returns tokens: (B, L, 128)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # (B, 128, L_conv)
        # to (B, L, 128) for transformer
        x = x.transpose(1, 2)
        x = F.relu(self.ln3(x))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
          y_denoise: (B, 1, T)
          y_level:   (B, 1, T)
          logits:    (B, 3)
        """
        cfg = self.cfg
        assert x.ndim == 3 and x.shape[1] == cfg.in_ch, f"expect (B,1,T), got {tuple(x.shape)}"
        assert x.shape[2] == cfg.T, f"expect T={cfg.T}, got {x.shape[2]}"

        # --------- Patch Embedding ---------
        tok = self._patch_embed(x)  # (B, L, 128)
        # [ASSUMPTION CHECK] must match derived L=124 for strict unpatchify closure.
        if tok.shape[1] != cfg.L:
            raise RuntimeError(
                f"Token length mismatch: got L={tok.shape[1]}, expect L={cfg.L}. "
                f"Adjust conv3 padding/params to enforce L."
            )

        # --------- Encoder ---------
        h = tok
        for blk in self.enc:
            h = blk(h)  # (B, L, 128)

        # --------- Decoder ---------
        z = self.dec_in(h)  # (B, L, 64)
        for blk in self.dec:
            z = blk(z)  # (B, L, 64)

        # --------- Head 1: Denoise (patch -> unpatchify -> smooth) ---------
        p1 = self.head_denoise(z)  # (B, L, 20)
        y1 = unpatchify_overlap_add(p1, T=cfg.T, stride=cfg.stride)  # (B, T)
        y1 = y1.unsqueeze(1)  # (B, 1, T)
        y1 = self.smooth(y1)  # (B, 1, T)

        # --------- Head 2: Noise Level (patch -> unpatchify) ---------
        p2 = self.head_level(z)  # (B, L, 20)
        y2 = unpatchify_overlap_add(p2, T=cfg.T, stride=cfg.stride).unsqueeze(1)  # (B, 1, T)

        # --------- Head 3: Classification (GAP -> FC) ---------
        g = z.mean(dim=1)  # (B, 64)
        logits = self.cls_fc(self.cls_drop(g))  # (B, 3)

        return y1, y2, logits