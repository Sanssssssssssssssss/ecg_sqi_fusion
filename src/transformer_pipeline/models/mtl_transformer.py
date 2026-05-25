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
    cls_pool: str = "decoder"
    use_positional_embedding: bool = False
    use_ordinal_head: bool = False
    use_snr_head: bool = False
    use_local_mask_head: bool = False
    use_noise_type_head: bool = False
    noise_type_classes: int = 4
    use_sqi_head: bool = False
    sqi_dim: int = 7

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
        if cfg.cls_pool in {"cls", "cls_mean"}:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.enc_d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if cfg.use_positional_embedding:
            token_count = cfg.L + (1 if cfg.cls_pool in {"cls", "cls_mean"} else 0)
            self.pos_embed = nn.Parameter(torch.zeros(1, token_count, cfg.enc_d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        if cfg.cls_pool in {"decoder", "cls"}:
            cls_in = cfg.dec_d_model
        elif cfg.cls_pool == "cls_mean":
            cls_in = cfg.dec_d_model * 2
        elif cfg.cls_pool == "encoder":
            cls_in = cfg.enc_d_model
        elif cfg.cls_pool == "both":
            cls_in = cfg.enc_d_model + cfg.dec_d_model
        else:
            raise ValueError("cls_pool must be 'decoder', 'encoder', 'both', 'cls', or 'cls_mean'")

        # [ASSUMPTION] dropout prob not specified -> cfg.dropout.
        self.cls_drop = nn.Dropout(cfg.dropout)
        self.cls_fc = nn.Linear(cls_in, 3)
        if cfg.use_ordinal_head:
            self.ordinal_fc = nn.Linear(cls_in, 2)
        if cfg.use_snr_head:
            self.snr_fc = nn.Linear(cls_in, 1)
        if cfg.use_local_mask_head:
            self.head_local_mask = nn.Linear(cfg.dec_d_model, cfg.head_patch_out)
        if cfg.use_noise_type_head:
            self.noise_type_fc = nn.Linear(cls_in, cfg.noise_type_classes)
        if cfg.use_sqi_head:
            self.sqi_fc = nn.Linear(cls_in, cfg.sqi_dim)

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

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Return encoder/decoder token states and the pooled classification feature.
        This is used by isolated pretraining experiments; it does not add parameters
        or change the public forward output.
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
        if cfg.cls_pool in {"cls", "cls_mean"}:
            cls = self.cls_token.expand(tok.shape[0], -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        if cfg.use_positional_embedding:
            tok = tok + self.pos_embed[:, : tok.shape[1], :]

        # --------- Encoder ---------
        h = tok
        for blk in self.enc:
            h = blk(h)  # (B, L, 128)

        # --------- Decoder ---------
        z = self.dec_in(h)  # (B, L, 64)
        for blk in self.dec:
            z = blk(z)  # (B, L, 64)

        # --------- Head 1: Denoise (patch -> unpatchify -> smooth) ---------
        z_patch = z[:, 1:, :] if cfg.cls_pool in {"cls", "cls_mean"} else z
        h_patch = h[:, 1:, :] if cfg.cls_pool in {"cls", "cls_mean"} else h

        p1 = self.head_denoise(z_patch)  # (B, L, 20)
        y1 = unpatchify_overlap_add(p1, T=cfg.T, stride=cfg.stride)  # (B, T)
        y1 = y1.unsqueeze(1)  # (B, 1, T)
        y1 = self.smooth(y1)  # (B, 1, T)

        # --------- Head 2: Noise Level (patch -> unpatchify) ---------
        p2 = self.head_level(z_patch)  # (B, L, 20)
        y2 = unpatchify_overlap_add(p2, T=cfg.T, stride=cfg.stride).unsqueeze(1)  # (B, 1, T)

        # --------- Head 3: Classification (GAP -> FC) ---------
        if cfg.cls_pool == "decoder":
            g = z.mean(dim=1)  # (B, 64)
        elif cfg.cls_pool == "encoder":
            g = h.mean(dim=1)  # (B, 128)
        elif cfg.cls_pool == "both":
            g = torch.cat([h.mean(dim=1), z.mean(dim=1)], dim=1)  # (B, 192)
        elif cfg.cls_pool == "cls":
            g = z[:, 0, :]  # (B, 64)
        elif cfg.cls_pool == "cls_mean":
            g = torch.cat([z[:, 0, :], z_patch.mean(dim=1)], dim=1)  # (B, 128)
        else:
            raise ValueError("unknown cls_pool")
        g = self.cls_drop(g)
        return {"encoder": h_patch, "decoder": z_patch, "pooled": g, "denoise": y1, "level": y2}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        returns:
          y_denoise: (B, 1, T)
          y_level:   (B, 1, T)
          logits:    (B, 3)
        """
        cfg = self.cfg
        features = self.forward_features(x)
        z = features["decoder"]
        g = features["pooled"]
        y1 = features["denoise"]
        y2 = features["level"]
        logits = self.cls_fc(g)  # (B, 3)

        out: tuple[torch.Tensor, ...] = (y1, y2, logits)
        if cfg.use_ordinal_head:
            out = out + (self.ordinal_fc(g),)
        if cfg.use_snr_head:
            out = out + (self.snr_fc(g).squeeze(1),)
        if cfg.use_local_mask_head:
            p3 = self.head_local_mask(z)  # (B, L, 20)
            y3 = unpatchify_overlap_add(p3, T=cfg.T, stride=cfg.stride).unsqueeze(1)
            out = out + (y3,)
        if cfg.use_noise_type_head:
            out = out + (self.noise_type_fc(g),)
        if cfg.use_sqi_head:
            out = out + (self.sqi_fc(g),)
        return out
