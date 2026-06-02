"""Morph-aware residual denoise losses for E3.11f Uformer mainline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_weights(name: str) -> dict[str, float]:
    variants = {
        "morph_soft_guard": {
            "clean": 0.90,
            "residual": 0.25,
            "gdm": 0.40,
            "max": 0.10,
            "fft": 0.45,
            "local": 0.75,
            "identity": 0.16,
        },
        "morph_soft_identity": {
            "clean": 0.80,
            "residual": 0.20,
            "gdm": 0.25,
            "max": 0.06,
            "fft": 0.35,
            "local": 0.55,
            "identity": 0.26,
        },
    }
    if name not in variants:
        raise ValueError(f"loss_variant must be one of {sorted(variants)}")
    return variants[name]


def per_sample_smooth_l1(a: torch.Tensor, b: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    return F.smooth_l1_loss(a, b, beta=beta, reduction="none").flatten(1).mean(dim=1)


def weighted_point_smooth_l1(a: torch.Tensor, b: torch.Tensor, point_weight: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    loss = F.smooth_l1_loss(a.squeeze(1), b.squeeze(1), beta=beta, reduction="none")
    return (loss * point_weight).sum(dim=1) / (point_weight.sum(dim=1) + 1e-8)


def gradient_difference_loss(pred: torch.Tensor, clean: torch.Tensor, point_weight: torch.Tensor) -> torch.Tensor:
    dp = pred[:, :, 1:] - pred[:, :, :-1]
    dc = clean[:, :, 1:] - clean[:, :, :-1]
    w = 0.5 * (point_weight[:, 1:] + point_weight[:, :-1])
    loss = F.smooth_l1_loss(dp.abs(), dc.abs(), beta=0.03, reduction="none").squeeze(1)
    return (loss * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)


def max_abs_loss(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return torch.amax(torch.abs(pred - clean).flatten(1), dim=1)


def multi_res_fft_loss(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    vals: list[torch.Tensor] = []
    for scale in (1, 2, 4):
        p = pred
        c = clean
        if scale > 1:
            p = F.avg_pool1d(p, kernel_size=scale, stride=scale)
            c = F.avg_pool1d(c, kernel_size=scale, stride=scale)
        pf = torch.fft.rfft(p.squeeze(1), dim=1, norm="ortho")
        cf = torch.fft.rfft(c.squeeze(1), dim=1, norm="ortho")
        pm = torch.log1p(torch.abs(pf))
        cm = torch.log1p(torch.abs(cf))
        vals.append(F.smooth_l1_loss(pm, cm, beta=0.03, reduction="none").mean(dim=1))
    return torch.stack(vals, dim=0).mean(dim=0)


def morph_loss(
    pred: torch.Tensor,
    noise_hat: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    y: torch.Tensor,
    point_weight: torch.Tensor,
    sample_weight: torch.Tensor,
    variant: str = "morph_soft_guard",
) -> tuple[torch.Tensor, dict[str, float]]:
    w = loss_weights(variant)
    noise_target = noisy - clean
    per_clean = per_sample_smooth_l1(pred, clean)
    per_residual = per_sample_smooth_l1(noise_hat, noise_target)
    per_gdm = gradient_difference_loss(pred, clean, point_weight)
    per_max = max_abs_loss(pred, clean)
    per_fft = multi_res_fft_loss(pred, clean) if w["fft"] > 0.0 else torch.zeros_like(per_clean)
    per_local = weighted_point_smooth_l1(pred, clean, point_weight)
    noise_power = torch.mean(noise_target.flatten(1) ** 2, dim=1)
    low_noise = noise_power <= torch.quantile(noise_power.detach(), 0.35)
    identity_mask = ((y == 0) | low_noise).float()
    per_identity = per_sample_smooth_l1(pred, noisy)
    per_total = (
        w["clean"] * per_clean
        + w["residual"] * per_residual
        + w["gdm"] * per_gdm
        + w["max"] * per_max
        + w["fft"] * per_fft
        + w["local"] * per_local
        + w["identity"] * per_identity * identity_mask
    )
    total = torch.mean(per_total * sample_weight)
    dbg = {
        "loss_clean": float(per_clean.mean().detach().cpu().item()),
        "loss_residual": float(per_residual.mean().detach().cpu().item()),
        "loss_gdm": float(per_gdm.mean().detach().cpu().item()),
        "loss_max": float(per_max.mean().detach().cpu().item()),
        "loss_fft": float(per_fft.mean().detach().cpu().item()),
        "loss_local": float(per_local.mean().detach().cpu().item()),
        "loss_identity": float((per_identity * identity_mask).mean().detach().cpu().item()),
    }
    return total, dbg
