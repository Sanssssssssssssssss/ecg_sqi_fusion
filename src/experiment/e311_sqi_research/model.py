from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding_1d(kernel: int, dilation: int = 1) -> int:
    eff = dilation * (kernel - 1) + 1
    return (eff - 1) // 2


def unpatchify_overlap_add(patches: torch.Tensor, *, T: int, stride: int) -> torch.Tensor:
    bsz, n_tok, patch = patches.shape
    out = patches.new_zeros((bsz, T))
    cnt = patches.new_zeros((bsz, T))
    for i in range(n_tok):
        s = i * stride
        e = min(T, s + patch)
        w = e - s
        if w > 0:
            out[:, s:e] += patches[:, i, :w]
            cnt[:, s:e] += 1.0
    return out / cnt.clamp_min(1.0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        d_ff = d_model * mlp_ratio
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
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(y)
        return x + self.mlp(self.ln2(x))


class AttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class ScaleTokenizer(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch: int, stride: int) -> None:
        super().__init__()
        self.patch = int(patch)
        self.stride = int(stride)
        self.proj = nn.Linear(in_ch * patch, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.patch:
            raise RuntimeError(f"sequence too short for patch={self.patch}")
        patches = x.unfold(-1, self.patch, self.stride)  # B, C, N, patch
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        return F.relu(self.ln(self.proj(patches)))


INPUT_SQI_STAT_DIM = 23
PRED_SQI_STAT_DIM = 7
LOCAL_SQI_STAT_DIM = 10
SQI_STAT_PROJ_DIM = 32


@dataclass(frozen=True)
class ResearchConfig:
    in_ch: int = 1
    T: int = 1250
    patch_size: int = 20
    stride: int = 10
    L: int = 124
    conv1_k: int = 15
    conv1_out: int = 32
    conv2_k: int = 7
    conv2_out: int = 64
    conv3_k: int = 20
    conv3_stride: int = 10
    conv3_dilation: int = 2
    conv3_out: int = 128
    conv3_padding: int = 10
    enc_layers: int = 6
    enc_d_model: int = 128
    enc_heads: int = 8
    enc_mlp_ratio: int = 3
    dec_d_model: int = 64
    dec_layers: int = 3
    dec_heads: int = 8
    dec_mlp_ratio: int = 3
    head_patch_out: int = 20
    smooth_k: int = 5
    dropout: float = 0.1
    use_positional_embedding: bool = True
    cls_pool: str = "cls"
    head_type: str = "baseline"
    multiscale: str = "none"
    use_snr_head: bool = True
    use_ordinal_head: bool = False
    cls_hidden: int = 0
    sqi_delta_scale: float = 1.0


class ResearchSQITransformer(nn.Module):
    """
    Isolated E3.11 research model.

    The backbone module names intentionally mirror the production transformer so
    D1 checkpoints can be partially loaded without importing or modifying the
    main pipeline.
    """

    def __init__(self, cfg: ResearchConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv1d(cfg.in_ch, cfg.conv1_out, kernel_size=cfg.conv1_k, padding=_same_padding_1d(cfg.conv1_k))
        self.bn1 = nn.BatchNorm1d(cfg.conv1_out)
        self.conv2 = nn.Conv1d(cfg.conv1_out, cfg.conv2_out, kernel_size=cfg.conv2_k, padding=_same_padding_1d(cfg.conv2_k))
        self.bn2 = nn.BatchNorm1d(cfg.conv2_out)
        self.conv3 = nn.Conv1d(
            cfg.conv2_out,
            cfg.conv3_out,
            kernel_size=cfg.conv3_k,
            stride=cfg.conv3_stride,
            dilation=cfg.conv3_dilation,
            padding=cfg.conv3_padding,
        )
        self.ln3 = nn.LayerNorm(cfg.conv3_out)

        self.extra_tokenizers = nn.ModuleList()
        self.scale_embed: nn.Parameter | None = None
        if cfg.multiscale == "10_20_40":
            scales = ((10, 5), (40, 20))
        elif cfg.multiscale == "20_50_100":
            scales = ((50, 25), (100, 50))
        elif cfg.multiscale == "none":
            scales = ()
        else:
            raise ValueError(f"unknown multiscale={cfg.multiscale!r}")
        for patch, stride in scales:
            self.extra_tokenizers.append(ScaleTokenizer(cfg.in_ch, cfg.enc_d_model, patch, stride))
        if scales:
            self.scale_embed = nn.Parameter(torch.zeros(1, len(scales), cfg.enc_d_model))
            nn.init.trunc_normal_(self.scale_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.enc_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if cfg.use_positional_embedding:
            pos_len = 512 if scales else cfg.L + 1
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, cfg.enc_d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.enc = nn.ModuleList(
            [TransformerBlock(cfg.enc_d_model, cfg.enc_heads, cfg.enc_mlp_ratio, cfg.dropout) for _ in range(cfg.enc_layers)]
        )
        self.dec_in = nn.Linear(cfg.enc_d_model, cfg.dec_d_model)
        self.dec = nn.ModuleList(
            [TransformerBlock(cfg.dec_d_model, cfg.dec_heads, cfg.dec_mlp_ratio, cfg.dropout) for _ in range(cfg.dec_layers)]
        )

        self.head_denoise = nn.Linear(cfg.dec_d_model, cfg.head_patch_out)
        smooth_pad = (cfg.smooth_k - 1) // 2
        self.smooth = nn.Conv1d(1, 1, kernel_size=cfg.smooth_k, padding=smooth_pad, bias=False)
        self.head_level = nn.Linear(cfg.dec_d_model, cfg.head_patch_out)

        self.enc_pool = AttentionPool(cfg.enc_d_model)
        self.dec_pool = AttentionPool(cfg.dec_d_model)
        self.cls_drop = nn.Dropout(cfg.dropout)
        self.sqi_stat_dim = self._sqi_stat_dim()
        self.sqi_stat_proj = (
            nn.Sequential(
                nn.LayerNorm(self.sqi_stat_dim),
                nn.Linear(self.sqi_stat_dim, SQI_STAT_PROJ_DIM),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            if self.sqi_stat_dim > 0
            else None
        )
        cls_in = self._classification_dim()
        self.cls_fc = nn.Linear(cls_in, 3)
        self.cls_mlp = (
            nn.Sequential(
                nn.LayerNorm(cls_in),
                nn.Linear(cls_in, cfg.cls_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.cls_hidden, 3),
            )
            if cfg.cls_hidden > 0
            else None
        )
        self.sqi_delta = None
        if self._is_sqi_residual_head():
            self.sqi_delta = nn.Sequential(
                nn.LayerNorm(self.sqi_stat_dim),
                nn.Linear(self.sqi_stat_dim, 32),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(32, 3),
            )
            nn.init.zeros_(self.sqi_delta[-1].weight)
            nn.init.zeros_(self.sqi_delta[-1].bias)
        if cfg.use_snr_head:
            self.snr_fc = nn.Linear(cls_in, 1)
        if cfg.use_ordinal_head:
            self.ordinal_fc = nn.Linear(cls_in, 2)

    def _is_sqi_residual_head(self) -> bool:
        return self.cfg.head_type in {
            "sqi_resid_input",
            "sqi_resid_pred_detach",
            "sqi_resid_mil_detach",
        }

    def _sqi_stat_dim(self) -> int:
        if self.cfg.head_type in {"sqi_input", "sqi_input_attn", "sqi_resid_input"}:
            return INPUT_SQI_STAT_DIM
        if self.cfg.head_type in {"sqi_pred", "sqi_pred_detach", "sqi_resid_pred_detach"}:
            return INPUT_SQI_STAT_DIM + PRED_SQI_STAT_DIM
        if self.cfg.head_type in {"sqi_mil", "sqi_mil_detach", "sqi_resid_mil_detach"}:
            return INPUT_SQI_STAT_DIM + PRED_SQI_STAT_DIM + LOCAL_SQI_STAT_DIM
        return 0

    def _classification_dim(self) -> int:
        if self.cfg.head_type == "baseline":
            return self.cfg.dec_d_model
        if self.cfg.head_type == "sqi_interpretable":
            return self.cfg.enc_d_model + self.cfg.dec_d_model + 10
        if self.cfg.head_type == "local_quality_v2":
            return self.cfg.dec_d_model + 10
        if self.cfg.head_type == "sqi_local":
            return self.cfg.enc_d_model + self.cfg.dec_d_model + 20
        if self.cfg.head_type == "sqi_input":
            return self.cfg.dec_d_model + SQI_STAT_PROJ_DIM
        if self.cfg.head_type == "sqi_input_attn":
            return self.cfg.enc_d_model + self.cfg.dec_d_model + SQI_STAT_PROJ_DIM
        if self.cfg.head_type in {"sqi_pred", "sqi_pred_detach", "sqi_mil", "sqi_mil_detach"}:
            return self.cfg.dec_d_model + SQI_STAT_PROJ_DIM
        if self._is_sqi_residual_head():
            return self.cfg.dec_d_model
        raise ValueError(f"unknown head_type={self.cfg.head_type!r}")

    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x).transpose(1, 2)
        return F.relu(self.ln3(x))

    def _tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        primary = self._patch_embed(x)
        if primary.shape[1] != self.cfg.L:
            raise RuntimeError(f"token length mismatch got {primary.shape[1]} expected {self.cfg.L}")

        extras: list[torch.Tensor] = []
        for i, tok in enumerate(self.extra_tokenizers):
            t = tok(x)
            if self.scale_embed is not None:
                t = t + self.scale_embed[:, i : i + 1, :]
            extras.append(t)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        all_tok = torch.cat([cls, primary, *extras], dim=1)
        if self.cfg.use_positional_embedding:
            if all_tok.shape[1] > self.pos_embed.shape[1]:
                raise RuntimeError(f"pos_embed too short for {all_tok.shape[1]} tokens")
            all_tok = all_tok + self.pos_embed[:, : all_tok.shape[1], :]
        return all_tok, primary.shape[1]

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        all_tok, n_primary = self._tokens(x)
        h = all_tok
        for blk in self.enc:
            h = blk(h)

        h_cls_primary = h[:, : n_primary + 1, :]
        z = self.dec_in(h_cls_primary)
        for blk in self.dec:
            z = blk(z)
        z_patch = z[:, 1:, :]
        h_patch = h[:, 1 : n_primary + 1, :]

        den_patch = self.head_denoise(z_patch)
        y_den = unpatchify_overlap_add(den_patch, T=self.cfg.T, stride=self.cfg.stride).unsqueeze(1)
        y_den = self.smooth(y_den)
        lvl_patch = self.head_level(z_patch)
        y_lvl = unpatchify_overlap_add(lvl_patch, T=self.cfg.T, stride=self.cfg.stride).unsqueeze(1)

        return {
            "encoder_all": h,
            "encoder": h_patch,
            "decoder_all": z,
            "decoder": z_patch,
            "decoder_cls": z[:, 0, :],
            "denoise": y_den,
            "level": y_lvl,
        }

    def _summary_stats(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return torch.cat([flat.mean(1, keepdim=True), flat.std(1, keepdim=True), flat.amax(1, keepdim=True)], dim=1)

    def _abs_summary_stats(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return torch.cat(
            [
                flat.mean(1, keepdim=True),
                flat.std(1, unbiased=False, keepdim=True),
                flat.abs().amax(1, keepdim=True),
            ],
            dim=1,
        )

    def _patch_stats(self, x: torch.Tensor) -> torch.Tensor:
        patch = min(self.cfg.patch_size, int(x.shape[-1]))
        stride = min(self.cfg.stride, patch)
        p = x.abs().unfold(-1, patch, stride).mean(dim=-1).squeeze(1)
        k = min(8, p.shape[1])
        return torch.cat(
            [
                p.mean(1, keepdim=True),
                p.std(1, unbiased=False, keepdim=True),
                p.amax(1, keepdim=True),
                p.topk(k, dim=1).values.mean(1, keepdim=True),
            ],
            dim=1,
        )

    def _input_sqi_stats(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, :1, :]
        dx = x0[:, :, 1:] - x0[:, :, :-1]
        ddx = dx[:, :, 1:] - dx[:, :, :-1]
        trend = F.avg_pool1d(F.pad(x0, (31, 31), mode="replicate"), kernel_size=63, stride=1)
        hf = x0 - trend
        return torch.cat(
            [
                self._abs_summary_stats(x0),
                self._abs_summary_stats(dx),
                self._abs_summary_stats(ddx),
                self._abs_summary_stats(trend),
                self._abs_summary_stats(hf),
                self._patch_stats(x0),
                self._patch_stats(dx),
            ],
            dim=1,
        )

    def _pred_sqi_stats(self, x: torch.Tensor, residual: torch.Tensor, level_prob: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self._abs_summary_stats(residual),
                self._abs_summary_stats(level_prob),
                self._estimated_snr_feature(x, residual),
            ],
            dim=1,
        )

    def _project_sqi_stats(self, stats: torch.Tensor) -> torch.Tensor:
        if self.sqi_stat_proj is None:
            raise RuntimeError(f"head_type={self.cfg.head_type!r} has no SQI stat projection")
        return self.sqi_stat_proj(stats)

    def _estimated_snr_feature(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        signal_power = x[:, :1, :].pow(2).mean(dim=(1, 2), keepdim=True)
        residual_power = residual.pow(2).mean(dim=(1, 2), keepdim=True)
        return torch.log10((signal_power + 1e-6) / (residual_power + 1e-6)).view(-1, 1)

    def _local_stats(self, residual: torch.Tensor, level_prob: torch.Tensor) -> torch.Tensor:
        r_patch = residual.unfold(-1, self.cfg.patch_size, self.cfg.stride).mean(dim=-1).squeeze(1)
        l_patch = level_prob.unfold(-1, self.cfg.patch_size, self.cfg.stride).mean(dim=-1).squeeze(1)
        k = min(8, r_patch.shape[1])
        r_top = r_patch.topk(k, dim=1).values.mean(1, keepdim=True)
        l_top = l_patch.topk(k, dim=1).values.mean(1, keepdim=True)
        return torch.cat(
            [
                r_patch.mean(1, keepdim=True),
                r_patch.std(1, keepdim=True),
                r_patch.amax(1, keepdim=True),
                r_top,
                l_patch.mean(1, keepdim=True),
                l_patch.std(1, keepdim=True),
                l_patch.amax(1, keepdim=True),
                l_top,
                (r_patch * l_patch).mean(1, keepdim=True),
                (r_patch * l_patch).amax(1, keepdim=True),
            ],
            dim=1,
        )

    def _sqi_stat_parts(self, x: torch.Tensor, residual: torch.Tensor, level_prob: torch.Tensor) -> torch.Tensor:
        stat_parts = [self._input_sqi_stats(x)]
        r = residual.detach() if "detach" in self.cfg.head_type else residual
        l = level_prob.detach() if "detach" in self.cfg.head_type else level_prob
        if self.cfg.head_type in {"sqi_pred", "sqi_pred_detach", "sqi_mil", "sqi_mil_detach", "sqi_resid_pred_detach", "sqi_resid_mil_detach"}:
            stat_parts.append(self._pred_sqi_stats(x, r, l))
        if self.cfg.head_type in {"sqi_mil", "sqi_mil_detach", "sqi_resid_mil_detach"}:
            stat_parts.append(self._local_stats(r, l))
        return torch.cat(stat_parts, dim=1)

    def classification_feature(self, x: torch.Tensor, f: dict[str, torch.Tensor]) -> torch.Tensor:
        residual = torch.abs(x[:, :1, :] - f["denoise"])
        level_prob = torch.sigmoid(f["level"])
        if self.cfg.head_type == "baseline":
            g = f["decoder_cls"]
        elif self.cfg.head_type == "sqi_interpretable":
            stats = torch.cat(
                [
                    self._summary_stats(residual),
                    self._summary_stats(level_prob),
                    self._summary_stats(x[:, :1, :]),
                    self._estimated_snr_feature(x, residual),
                ],
                dim=1,
            )
            g = torch.cat([self.enc_pool(f["encoder_all"]), self.dec_pool(f["decoder"]), stats], dim=1)
        elif self.cfg.head_type == "local_quality_v2":
            g = torch.cat([f["decoder_cls"], self._local_stats(residual, level_prob)], dim=1)
        elif self.cfg.head_type == "sqi_local":
            stats = torch.cat(
                [
                    self._summary_stats(residual),
                    self._summary_stats(level_prob),
                    self._summary_stats(x[:, :1, :]),
                    self._estimated_snr_feature(x, residual),
                ],
                dim=1,
            )
            g = torch.cat(
                [self.enc_pool(f["encoder_all"]), self.dec_pool(f["decoder"]), stats, self._local_stats(residual, level_prob)],
                dim=1,
            )
        elif self.cfg.head_type in {"sqi_input", "sqi_input_attn", "sqi_pred", "sqi_pred_detach", "sqi_mil", "sqi_mil_detach"}:
            sqi_feat = self._project_sqi_stats(self._sqi_stat_parts(x, residual, level_prob))
            if self.cfg.head_type == "sqi_input_attn":
                g = torch.cat([self.enc_pool(f["encoder_all"]), f["decoder_cls"], sqi_feat], dim=1)
            else:
                g = torch.cat([f["decoder_cls"], sqi_feat], dim=1)
        elif self._is_sqi_residual_head():
            g = f["decoder_cls"]
        else:
            raise ValueError(f"unknown head_type={self.cfg.head_type!r}")
        return self.cls_drop(g)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.forward_features(x)
        g = self.classification_feature(x, f)
        logits = self.cls_mlp(g) if self.cls_mlp is not None else self.cls_fc(g)
        if self.sqi_delta is not None:
            residual = torch.abs(x[:, :1, :] - f["denoise"])
            level_prob = torch.sigmoid(f["level"])
            logits = logits + float(self.cfg.sqi_delta_scale) * self.sqi_delta(self._sqi_stat_parts(x, residual, level_prob))
        out = {
            "denoise": f["denoise"],
            "level": f["level"],
            "logits": logits,
        }
        if self.cfg.use_snr_head:
            out["snr_hat"] = self.snr_fc(g).squeeze(1)
        if self.cfg.use_ordinal_head:
            out["ordinal_logits"] = self.ordinal_fc(g)
        return out

    def shared_parameters(self) -> list[nn.Parameter]:
        modules: list[nn.Module] = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.ln3, self.enc, self.dec_in, self.dec]
        return [p for module in modules for p in module.parameters() if p.requires_grad]
