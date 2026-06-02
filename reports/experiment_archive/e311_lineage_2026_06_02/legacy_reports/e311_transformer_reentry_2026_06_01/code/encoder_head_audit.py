from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).absolute()
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else next(p for p in THIS_FILE.parents if (p / "src").exists())
EXP_ROOT = ROOT / "outputs" / "experiment" / "e311_sqi_denoise_classifier_grid"
FULL_JOINT = EXP_ROOT / "full_joint_train.py"
MORPH_EXP = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid"
MORPH_TRAIN = MORPH_EXP / "morph_train.py"
DEFAULT_SOURCE = MORPH_EXP / "data" / "med6p25_badgap7_badcm0p75"
DEFAULT_DENOISER = MORPH_EXP / "baseline" / "baseline_best_med6p25_badgap7_scale0p955" / "checkpoints" / "ckpt_best_denoise.pt"
DEFAULT_CLASSIFIER = (
    ROOT
    / "outputs"
    / "experiment"
    / "e311_denoise_strong_grid"
    / "artifact"
    / "models"
    / "promote_strong_noise_clean_multiscale_deriv_multiscale_patch_ld5p0_ep8_ld12p5_ep14"
    / "ckpt_best_val.pt"
)

CLASS_NAMES = ["good", "medium", "bad"]
FEATURE_SETS = (
    "full",
    "encoder_only",
    "bottleneck_only",
    "summary_only",
    "residual_summary_only",
    "noisy_summary_only",
    "denoised_summary_only",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cls_report(y: np.ndarray, logits: np.ndarray, prefix: str = "") -> dict[str, Any]:
    pred = np.argmax(logits, axis=1)
    mat = np.zeros((3, 3), dtype=int)
    rec: list[float] = []
    prec: list[float] = []
    f1: list[float] = []
    for a, b in zip(y, pred):
        mat[int(a), int(b)] += 1
    for c in range(3):
        tp = float(np.sum((y == c) & (pred == c)))
        fn = float(np.sum((y == c) & (pred != c)))
        fp = float(np.sum((y != c) & (pred == c)))
        r = tp / max(1.0, tp + fn)
        p = tp / max(1.0, tp + fp)
        rec.append(r)
        prec.append(p)
        f1.append(2.0 * p * r / max(1e-12, p + r))
    return {
        f"{prefix}acc": float(np.mean(pred == y)),
        f"{prefix}confusion_3x3": mat.tolist(),
        f"{prefix}recall_good_medium_bad": rec,
        f"{prefix}precision_good_medium_bad": prec,
        f"{prefix}f1_good_medium_bad": f1,
    }


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


def make_feature_vector(
    feature_set: str,
    noisy: torch.Tensor,
    denoise: torch.Tensor,
    residual: torch.Tensor,
    s1: torch.Tensor | None = None,
    s2: torch.Tensor | None = None,
    s3: torch.Tensor | None = None,
    bottleneck: torch.Tensor | None = None,
) -> torch.Tensor:
    summaries = {
        "noisy_summary_only": tensor_summary(noisy),
        "denoised_summary_only": tensor_summary(denoise),
        "residual_summary_only": tensor_summary(residual),
    }
    summary_all = torch.cat([summaries["noisy_summary_only"], summaries["denoised_summary_only"], summaries["residual_summary_only"]], dim=1)
    if feature_set == "summary_only":
        return summary_all
    if feature_set in summaries:
        return summaries[feature_set]
    if bottleneck is None:
        if feature_set == "full":
            return summary_all
        raise ValueError(f"feature_set={feature_set} requires residual_unet encoder features")
    bottleneck_feat = pooled_channels(bottleneck)
    if feature_set == "bottleneck_only":
        return bottleneck_feat
    encoder_feat = torch.cat([pooled_channels(s1), pooled_channels(s2), pooled_channels(s3), bottleneck_feat], dim=1)
    if feature_set == "encoder_only":
        return encoder_feat
    if feature_set == "full":
        return torch.cat([encoder_feat, summary_all], dim=1)
    raise ValueError(f"unknown feature_set: {feature_set}")


def residual_unet_forward_with_features(
    denoiser: nn.Module,
    noisy: torch.Tensor,
    feature_set: str = "full",
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return denoise, noise_hat, and encoder feature vector for the residual_unet denoiser."""
    net = getattr(denoiser, "net", denoiser)
    if not all(hasattr(net, name) for name in ("e1", "d1", "e2", "d2", "e3", "d3", "mid", "u3", "u2", "u1", "out")):
        _denoise, noise_hat = denoiser(noisy)
        denoise = noisy - float(noise_scale) * noise_hat
        residual = torch.abs(noisy - denoise)
        feats = make_feature_vector(feature_set, noisy, denoise, residual)
        return denoise, noise_hat, feats

    s1 = net.e1(noisy)
    s2 = net.e2(F.gelu(net.d1(s1)))
    s3 = net.e3(F.gelu(net.d2(s2)))
    b = net.mid(F.gelu(net.d3(s3)))
    x = net._upsample(b, s3)
    x = net.u3(torch.cat([x, s3], dim=1))
    x = net._upsample(x, s2)
    x = net.u2(torch.cat([x, s2], dim=1))
    x = net._upsample(x, s1)
    x = net.u1(torch.cat([x, s1], dim=1))
    noise_hat = net.out(x)
    denoise = noisy - float(noise_scale) * noise_hat
    residual = torch.abs(noisy - denoise)
    feats = make_feature_vector(feature_set, noisy, denoise, residual, s1=s1, s2=s2, s3=s3, bottleneck=b)
    return denoise, noise_hat, feats


class EncoderClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
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


class EncoderDeltaHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, gated: bool):
        super().__init__()
        self.gated = gated
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.gate = nn.Linear(hidden_dim, 1) if gated else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.net(x)
        delta = self.delta(h)
        gate = torch.sigmoid(self.gate(h)) if self.gate is not None else None
        if gate is not None:
            delta = delta * gate
        return delta, gate


class TransformerLatentAdapter(nn.Module):
    def __init__(self, feat_dim: int, pooled_dim: int, enc_dim: int, hidden_dim: int, dropout: float, mode: str):
        super().__init__()
        self.mode = mode
        self.shared = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if mode == "film":
            self.to_film = nn.Linear(hidden_dim, pooled_dim * 2)
            nn.init.zeros_(self.to_film.weight)
            nn.init.zeros_(self.to_film.bias)
        elif mode == "cross":
            self.to_kv = nn.Linear(hidden_dim, pooled_dim)
            self.cross_attn = nn.MultiheadAttention(pooled_dim, num_heads=max(1, min(4, pooled_dim // 16)), dropout=dropout, batch_first=True)
            self.out = nn.Linear(pooled_dim, pooled_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif mode == "token":
            self.to_token = nn.Linear(hidden_dim, enc_dim)
            nn.init.zeros_(self.to_token.weight)
            nn.init.zeros_(self.to_token.bias)
        else:
            raise ValueError(f"unknown Transformer adapter mode: {mode}")

    def film(self, pooled: torch.Tensor, feat: torch.Tensor, scale: float) -> torch.Tensor:
        h = self.shared(feat)
        gamma, beta = self.to_film(h).chunk(2, dim=1)
        return pooled + float(scale) * (gamma * pooled + beta)

    def cross(self, pooled: torch.Tensor, feat: torch.Tensor, scale: float) -> torch.Tensor:
        h = self.shared(feat)
        kv = self.to_kv(h).unsqueeze(1)
        q = pooled.unsqueeze(1)
        attn, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return pooled + float(scale) * self.out(attn.squeeze(1))

    def token(self, feat: torch.Tensor, scale: float) -> torch.Tensor:
        h = self.shared(feat)
        return float(scale) * self.to_token(h).unsqueeze(1)


class WaveformCNNHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(196),
            nn.Linear(196, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        avg = F.adaptive_avg_pool1d(h, 1).flatten(1)
        mx = F.adaptive_max_pool1d(h.abs(), 1).flatten(1)
        feat = torch.cat([avg, mx, tensor_summary(x)], dim=1)
        return self.head(feat)


def derivative_summary(x: torch.Tensor) -> torch.Tensor:
    return tensor_summary(x[..., 1:] - x[..., :-1])


def explicit_sqi_features(noisy: torch.Tensor, denoise: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    residual = torch.abs(noisy - denoise)
    resid_signed = noisy - denoise
    smooth = F.avg_pool1d(resid_signed, kernel_size=63, stride=1, padding=31)
    hf = resid_signed - smooth
    rms_signal = torch.sqrt(torch.mean(denoise.flatten(1) ** 2, dim=1, keepdim=True) + 1e-8)
    rms_resid = torch.sqrt(torch.mean(resid_signed.flatten(1) ** 2, dim=1, keepdim=True) + 1e-8)
    est_snr = 20.0 * torch.log10((rms_signal + 1e-6) / (rms_resid + 1e-6))
    est_snr_norm = torch.clamp((est_snr + 5.0) / 25.0, 0.0, 1.0)
    probs = torch.softmax(base_logits, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = top2[:, :1] - top2[:, 1:2]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
    return torch.cat(
        [
            tensor_summary(noisy),
            tensor_summary(denoise),
            tensor_summary(residual),
            derivative_summary(noisy),
            derivative_summary(denoise),
            derivative_summary(resid_signed),
            tensor_summary(smooth),
            tensor_summary(hf),
            est_snr_norm,
            rms_resid,
            probs.amax(dim=1, keepdim=True),
            margin,
            entropy,
            probs[:, 2:3],
        ],
        dim=1,
    )


class EncoderAuditModel(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        encoder_head: EncoderClassifierHead | None,
        waveform_head: WaveformCNNHead | None,
        transformer_classifier: nn.Module | None,
        encoder_delta: EncoderDeltaHead | None,
        explicit_sqi_delta: EncoderDeltaHead | None,
        transformer_adapter: TransformerLatentAdapter | None,
        variant: str,
        noise_scale: float,
        delta_scale: float,
        detach_encoder_features: bool,
        feature_set: str,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.encoder_head = encoder_head
        self.waveform_head = waveform_head
        self.transformer_classifier = transformer_classifier
        self.encoder_delta = encoder_delta
        self.explicit_sqi_delta = explicit_sqi_delta
        self.transformer_adapter = transformer_adapter
        self.variant = variant
        self.noise_scale = float(noise_scale)
        self.delta_scale = float(delta_scale)
        self.detach_encoder_features = bool(detach_encoder_features)
        self.feature_set = str(feature_set)

    def forward(self, noisy: torch.Tensor) -> dict[str, torch.Tensor | None]:
        denoise, noise_hat, enc_feat = residual_unet_forward_with_features(
            self.denoiser,
            noisy,
            feature_set=self.feature_set,
            noise_scale=self.noise_scale,
        )
        feat = enc_feat.detach() if self.detach_encoder_features else enc_feat
        gate = None
        delta = None
        snr_pred = None
        local_pred = None
        if self.variant == "encoder_head_only":
            assert self.encoder_head is not None
            logits = self.encoder_head(feat)
            base_logits = logits
        elif self.variant == "waveform_only_no_sqi":
            assert self.waveform_head is not None
            waveform = denoise.detach() if self.detach_encoder_features else denoise
            logits = self.waveform_head(waveform)
            base_logits = logits
        elif self.variant == "encoder_latent_plus_explicit_sqi":
            assert self.encoder_head is not None and self.explicit_sqi_delta is not None
            base_logits = self.encoder_head(feat)
            sqi_feat = explicit_sqi_features(noisy, denoise, base_logits)
            if self.detach_encoder_features:
                sqi_feat = sqi_feat.detach()
            delta, gate = self.explicit_sqi_delta(sqi_feat)
            logits = base_logits + self.delta_scale * delta
        elif self.variant == "dual_branch":
            assert self.transformer_classifier is not None and self.encoder_delta is not None
            out = self.transformer_classifier(denoise)
            base_logits = out[2]
            snr_pred = out[3] if len(out) > 3 else None
            local_pred = out[4] if len(out) > 4 else None
            delta, gate = self.encoder_delta(feat)
            logits = base_logits + self.delta_scale * delta
        elif self.variant in {"transformer_film_sqi", "transformer_cross_sqi", "transformer_sqi_token_prefix"}:
            assert self.transformer_classifier is not None and self.transformer_adapter is not None
            base_out = self.transformer_classifier(denoise)
            base_logits = base_out[2]
            snr_pred = base_out[3] if len(base_out) > 3 else None
            local_pred = base_out[4] if len(base_out) > 4 else None
            if self.variant == "transformer_film_sqi":
                f = self.transformer_classifier.forward_features(denoise)
                pooled = self.transformer_adapter.film(f["pooled"], feat, self.delta_scale)
                logits = self.transformer_classifier.cls_fc(pooled)
            elif self.variant == "transformer_cross_sqi":
                f = self.transformer_classifier.forward_features(denoise)
                pooled = self.transformer_adapter.cross(f["pooled"], feat, self.delta_scale)
                logits = self.transformer_classifier.cls_fc(pooled)
            else:
                logits = self._forward_transformer_with_sqi_token(denoise, feat)
        else:
            raise ValueError(f"unknown variant: {self.variant}")
        return {
            "denoise": denoise,
            "noise_hat": noise_hat,
            "base_logits": base_logits,
            "logits": logits,
            "encoder_features": enc_feat,
            "delta": delta,
            "gate": gate,
            "snr_pred": snr_pred,
            "local_pred": local_pred,
        }

    def _forward_transformer_with_sqi_token(self, denoise: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        model = self.transformer_classifier
        adapter = self.transformer_adapter
        assert model is not None and adapter is not None
        cfg = model.cfg
        if cfg.cls_pool not in {"cls", "cls_mean"}:
            raise ValueError("transformer_sqi_token_prefix requires cls or cls_mean pooling")
        tok = model._patch_embed(denoise)
        if tok.shape[1] != cfg.L:
            raise RuntimeError(f"Token length mismatch: got {tok.shape[1]}, expect {cfg.L}")
        cls = model.cls_token.expand(tok.shape[0], -1, -1)
        sqi = adapter.token(feat, self.delta_scale)
        if cfg.use_positional_embedding:
            cls = cls + model.pos_embed[:, :1, :]
            tok = tok + model.pos_embed[:, 1 : 1 + tok.shape[1], :]
        h = torch.cat([cls, sqi, tok], dim=1)
        for blk in model.enc:
            h = blk(h)
        z = model.dec_in(h)
        for blk in model.dec:
            z = blk(z)
        z_patch = z[:, 2:, :]
        if cfg.cls_pool == "cls":
            pooled = z[:, 0, :]
        else:
            pooled = torch.cat([z[:, 0, :], z_patch.mean(dim=1)], dim=1)
        return model.cls_fc(model.cls_drop(pooled))


def severity_from_snr(snr_db: torch.Tensor) -> torch.Tensor:
    return torch.clamp((13.0 - snr_db.float()) / 14.5, 0.0, 1.0)


@torch.no_grad()
def collect_outputs(model: EncoderAuditModel, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in [
        "noisy",
        "clean",
        "denoise_pred",
        "y",
        "base_logits",
        "logits",
        "prob_denoise",
        "prob_base",
        "idx",
        "ecg_id",
        "snr_db",
        "original_snr_db",
        "local_mask",
        "critical_mask",
        "qrs_mask",
        "tst_mask",
        "encoder_delta",
        "encoder_gate",
    ]}
    noise_kind_all: list[np.ndarray] = []
    placement_all: list[np.ndarray] = []
    for batch in loader:
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        out = model(noisy)
        logits = out["logits"]
        base_logits = out["base_logits"]
        denoise = out["denoise"]
        chunks["noisy"].append(noisy.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["clean"].append(batch["clean"].squeeze(1).numpy().astype(np.float32))
        chunks["denoise_pred"].append(denoise.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["y"].append(batch["y"].numpy().astype(np.int64))
        chunks["base_logits"].append(base_logits.cpu().numpy().astype(np.float32))
        chunks["logits"].append(logits.cpu().numpy().astype(np.float32))
        chunks["prob_base"].append(torch.softmax(base_logits, dim=1).cpu().numpy().astype(np.float32))
        chunks["prob_denoise"].append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
        bs = int(noisy.shape[0])
        if out["delta"] is not None:
            chunks["encoder_delta"].append(out["delta"].cpu().numpy().astype(np.float32))
        else:
            chunks["encoder_delta"].append(np.zeros((bs, 3), dtype=np.float32))
        if out["gate"] is not None:
            chunks["encoder_gate"].append(out["gate"].cpu().numpy().astype(np.float32))
        else:
            chunks["encoder_gate"].append(np.full((bs, 1), np.nan, dtype=np.float32))
        for key in ["idx", "ecg_id"]:
            chunks[key].append(batch[key].numpy().astype(np.int64))
        for key in ["snr_db", "original_snr_db", "local_mask", "critical_mask", "qrs_mask", "tst_mask"]:
            chunks[key].append(batch[key].numpy().astype(np.float32))
        noise_kind_all.append(np.asarray(batch["noise_kind"], dtype=str))
        placement_all.append(np.asarray(batch["placement"], dtype=str))
    arrays = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    arrays["noise_kind"] = np.concatenate(noise_kind_all, axis=0)
    arrays["placement"] = np.concatenate(placement_all, axis=0)
    arrays["y_pred_denoise"] = np.argmax(arrays["logits"], axis=1).astype(np.int64)
    arrays["y_pred_base"] = np.argmax(arrays["base_logits"], axis=1).astype(np.int64)
    arrays["y_pred_noisy"] = arrays["y_pred_base"]
    return arrays


def _safe_metric(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def select_gallery_positions(arrays: dict[str, np.ndarray], mode: str, max_rows: int = 12) -> list[int]:
    y = arrays["y"].astype(np.int64)
    noisy = arrays["noisy"]
    clean = arrays["clean"]
    denoise = arrays["denoise_pred"]
    noise_mse = np.mean((noisy - clean) ** 2, axis=1)
    pred_mse = np.mean((denoise - clean) ** 2, axis=1)
    critical = arrays.get("critical_mask", np.zeros_like(clean))
    qrs = arrays.get("qrs_mask", np.zeros_like(clean))
    tst = arrays.get("tst_mask", np.zeros_like(clean))
    local_mass = np.mean(critical + qrs + tst, axis=1)
    rng = np.random.default_rng(20260601)
    chosen: list[int] = []

    def add(items: np.ndarray | list[int], limit: int | None = None) -> None:
        for item in list(items):
            pos = int(item)
            if pos not in chosen:
                chosen.append(pos)
            if limit is not None and len(chosen) >= limit:
                break

    if mode == "balanced":
        for cls in [0, 1, 2]:
            idx = np.where(y == cls)[0]
            if idx.size == 0:
                continue
            high = idx[np.argsort(noise_mse[idx])[::-1]][:2]
            mid = idx[np.argsort(np.abs(noise_mse[idx] - np.median(noise_mse[idx])))][:2]
            sample = rng.choice(idx, size=min(2, idx.size), replace=False)
            add(np.concatenate([high, mid, sample]))
    elif mode == "hard_bad":
        idx = np.where(y >= 1)[0]
        add(idx[np.argsort(noise_mse[idx])[::-1]][:max_rows])
    elif mode == "good_safety":
        idx = np.where(y == 0)[0]
        score = pred_mse - noise_mse
        add(idx[np.argsort(score[idx])[::-1]][:max_rows])
    elif mode == "worst_residual":
        add(np.argsort(pred_mse)[::-1][:max_rows])
    elif mode == "qrs_tst_focus":
        add(np.argsort(local_mass)[::-1][:max_rows])
    else:
        raise ValueError(f"unknown gallery mode: {mode}")
    return chosen[:max_rows]


def export_fixed_gallery(arrays: dict[str, np.ndarray], positions: list[int], out_png: Path, title: str) -> None:
    if not positions:
        return
    t = np.arange(arrays["clean"].shape[1], dtype=np.float32) / 125.0
    cols = [("clean", arrays["clean"]), ("before/noisy", arrays["noisy"]), ("after/denoise", arrays["denoise_pred"])]
    fig, axes = plt.subplots(len(positions), 3, figsize=(12.2, max(5.8, 1.25 * len(positions))), sharex=True, sharey=False)
    if len(positions) == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, pos in enumerate(positions):
        row_min = min(float(arr[pos].min()) for _, arr in cols)
        row_max = max(float(arr[pos].max()) for _, arr in cols)
        pad = 0.06 * (row_max - row_min + 1e-8)
        y = int(arrays["y"][pos])
        pred = int(arrays["y_pred_denoise"][pos])
        base = int(arrays["y_pred_base"][pos])
        snr = float(arrays["snr_db"][pos])
        orig = float(arrays["original_snr_db"][pos])
        for c, (name, arr) in enumerate(cols):
            ax = axes[r, c]
            ax.plot(t, arr[pos], linewidth=0.68)
            ax.set_ylim(row_min - pad, row_max + pad)
            ax.grid(True, alpha=0.15)
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(
                    f"{CLASS_NAMES[y]} idx={int(arrays['idx'][pos])}\n"
                    f"ecg={int(arrays['ecg_id'][pos])} snr={snr:.2f}/{orig:.2f}\n"
                    f"{arrays['noise_kind'][pos]} {arrays['placement'][pos]}\n"
                    f"pred={CLASS_NAMES[pred]} base={CLASS_NAMES[base]}",
                    fontsize=7.2,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=60,
                )
            else:
                ax.set_yticklabels([])
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190)
    plt.close(fig)


def write_visual_review(out_dir: Path, report: dict[str, Any], metrics: dict[str, Any]) -> None:
    overall = metrics.get("overall", {})
    good = metrics.get("by_class", {}).get("good", {})
    tags: list[str] = []
    if _safe_metric(good, "mse_ratio_pred_over_noisy", 0.0) > 0.85 or _safe_metric(good, "qrs_amp_rel_abs_err_delta", 0.0) > 0.018:
        tags.append("good_overfiltered")
    if _safe_metric(overall, "qrs_amp_rel_abs_err_delta", 0.0) > 0.012:
        tags.append("qrs_amplitude_distorted")
    if _safe_metric(overall, "qrs_timing_abs_ms_delta", 0.0) > 1.5:
        tags.append("qrs_timing_shift")
    if _safe_metric(overall, "tst_mae_reduction", 1.0) < 0.0:
        tags.append("t_wave_flattened")
    if _safe_metric(overall, "drift_error_energy_reduction", 0.0) < 0.20:
        tags.append("baseline_drift_left")
    if _safe_metric(overall, "hf_error_energy_reduction", 0.0) < 0.20:
        tags.append("hf_noise_left")
    if _safe_metric(overall, "critical_mae_reduction", 0.0) < 0.15:
        tags.append("morph_damage_not_repaired")
    if _safe_metric(overall, "mse_ratio_pred_over_noisy", 1.0) < 0.75 and _safe_metric(overall, "snr_improve_db_mean", 0.0) > 1.0:
        tags.append("looks_better_than_noisy")
    if _safe_metric(overall, "mse_ratio_pred_over_noisy", 1.0) < 0.18 and _safe_metric(overall, "ssim_delta_mean", 0.0) > 0.05:
        tags.append("looks_close_to_clean")
    recalls = report.get("final_recall_good_medium_bad", [])
    text = [
        "# Visual Review",
        "",
        "This file is an automatic first-pass visual triage. The heartbeat supervisor should inspect the PNG galleries and amend this section when the metric tags disagree with the plots.",
        "",
        f"- tags: {', '.join(tags) if tags else 'none'}",
        f"- final_acc: {report.get('final_acc')}",
        f"- final_recall_good_medium_bad: {recalls}",
        f"- denoise_score: {overall.get('denoise_score')}",
        f"- mse_ratio_pred_over_noisy: {overall.get('mse_ratio_pred_over_noisy')}",
        f"- snr_improve_db_mean: {overall.get('snr_improve_db_mean')}",
        f"- qrs_amp_rel_abs_err_delta: {overall.get('qrs_amp_rel_abs_err_delta')}",
        f"- qrs_timing_abs_ms_delta: {overall.get('qrs_timing_abs_ms_delta')}",
        f"- critical_mae_reduction: {overall.get('critical_mae_reduction')}",
        "",
        "## Galleries",
        "",
        "- debug/balanced_gallery.png",
        "- debug/hard_bad_gallery.png",
        "- debug/good_safety_gallery.png",
        "- debug/worst_residual_gallery.png",
        "- debug/qrs_tst_focus_gallery.png",
    ]
    (out_dir / "visual_review.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def evaluate(model: EncoderAuditModel, loader: DataLoader, device: torch.device, morph: Any, out_dir: Path | None = None) -> dict[str, Any]:
    arrays = collect_outputs(model, loader, device)
    y = arrays["y"].astype(np.int64)
    report = {
        **cls_report(y, arrays["base_logits"], "base_"),
        **cls_report(y, arrays["logits"], "final_"),
    }
    report["corrected_by_encoder_delta"] = int(np.sum((arrays["y_pred_base"] != y) & (arrays["y_pred_denoise"] == y)))
    report["harmed_by_encoder_delta"] = int(np.sum((arrays["y_pred_base"] == y) & (arrays["y_pred_denoise"] != y)))
    report["encoder_delta_abs_mean"] = float(np.mean(np.abs(arrays["encoder_delta"])))
    report["encoder_delta_abs_p95"] = float(np.percentile(np.abs(arrays["encoder_delta"]), 95))
    if np.isfinite(arrays["encoder_gate"]).any():
        report["encoder_gate_mean"] = float(np.nanmean(arrays["encoder_gate"]))
        report["encoder_gate_p95"] = float(np.nanpercentile(arrays["encoder_gate"], 95))
    masks = {
        "local": arrays["local_mask"],
        "critical": arrays["critical_mask"],
        "qrs": arrays["qrs_mask"],
        "tst": arrays["tst_mask"],
    }
    metrics = morph.build_metrics(arrays["clean"], arrays["noisy"], arrays["denoise_pred"], y, arrays["noise_kind"], arrays["placement"], masks)
    if out_dir is not None:
        (out_dir / "denoise_eval").mkdir(parents=True, exist_ok=True)
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "denoise_eval" / "test_denoise_outputs.npz",
            noisy=arrays["noisy"],
            clean=arrays["clean"],
            denoise_pred=arrays["denoise_pred"],
            y=y,
            y_pred=arrays["y_pred_denoise"],
            y_pred_base=arrays["y_pred_base"],
            prob=arrays["prob_denoise"],
            prob_base=arrays["prob_base"],
            encoder_delta=arrays["encoder_delta"],
            encoder_gate=arrays["encoder_gate"],
            idx=arrays["idx"],
            ecg_id=arrays["ecg_id"],
            snr_db=arrays["snr_db"],
            original_snr_db=arrays["original_snr_db"],
            noise_kind=arrays["noise_kind"],
            placement=arrays["placement"],
        )
        write_json(out_dir / "test_report.json", report)
        write_json(out_dir / "denoise_eval" / "denoise_metrics.json", metrics)
        morph.export_gallery(arrays, out_dir / "debug" / "denoise_compact_test.png", hard_only=False, title="encoder-head audit: compact SNR-marked gallery")
        morph.export_gallery(arrays, out_dir / "debug" / "denoise_hard_test.png", hard_only=True, title="encoder-head audit: hard SNR-marked gallery")
        for mode, title in [
            ("balanced", "balanced fixed-sample denoise gallery"),
            ("hard_bad", "hard bad/medium high-noise denoise gallery"),
            ("good_safety", "good-class overfiltering safety gallery"),
            ("worst_residual", "worst residual-to-clean denoise gallery"),
            ("qrs_tst_focus", "QRS/TST critical-region focus gallery"),
        ]:
            export_fixed_gallery(arrays, select_gallery_positions(arrays, mode), out_dir / "debug" / f"{mode}_gallery.png", title)
        write_visual_review(out_dir, report, metrics)
    return {"report": report, "metrics": metrics}


def path_or_none(value: str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text)


def make_model(args: argparse.Namespace, morph: Any, full_joint: Any, train_loader: DataLoader, device: torch.device) -> EncoderAuditModel:
    denoiser_ckpt = None if str(args.init_mode) == "he_scratch" else path_or_none(args.init_denoiser_checkpoint)
    denoiser = full_joint.load_denoiser(morph, denoiser_ckpt, device)
    if str(args.init_mode) == "he_scratch":
        full_joint.init_he_scratch(denoiser)
        print("init_mode=he_scratch | denoiser reinitialized with Kaiming/He", flush=True)
    else:
        print(f"init_mode={args.init_mode} | warm denoiser kept", flush=True)
    if bool(args.freeze_denoiser):
        for param in denoiser.parameters():
            param.requires_grad_(False)
        denoiser.eval()
        print("freeze_denoiser=True | denoiser params excluded from optimizer", flush=True)
    transformer = None
    encoder_head = None
    encoder_delta = None
    waveform_head = None
    explicit_sqi_delta = None
    transformer_adapter = None
    with torch.no_grad():
        sample = next(iter(train_loader))
        noisy = sample["noisy"].to(device=device, dtype=torch.float32)
        _, _, feat = residual_unet_forward_with_features(
            denoiser,
            noisy,
            feature_set=str(args.feature_set),
            noise_scale=float(args.noise_scale),
        )
        feat_dim = int(feat.shape[1])
    if args.variant == "encoder_head_only":
        encoder_head = EncoderClassifierHead(feat_dim, int(args.hidden_dim), float(args.dropout)).to(device)
    elif args.variant == "waveform_only_no_sqi":
        waveform_head = WaveformCNNHead(int(args.hidden_dim), float(args.dropout)).to(device)
    elif args.variant == "encoder_latent_plus_explicit_sqi":
        encoder_head = EncoderClassifierHead(feat_dim, int(args.hidden_dim), float(args.dropout)).to(device)
        with torch.no_grad():
            denoise, _, _ = residual_unet_forward_with_features(
                denoiser,
                noisy,
                feature_set=str(args.feature_set),
                noise_scale=float(args.noise_scale),
            )
            dummy_logits = torch.zeros((int(noisy.shape[0]), 3), dtype=torch.float32, device=device)
            sqi_dim = int(explicit_sqi_features(noisy, denoise, dummy_logits).shape[1])
        explicit_sqi_delta = EncoderDeltaHead(sqi_dim, int(args.sqi_hidden_dim), float(args.dropout), bool(args.gated_delta)).to(device)
    elif args.variant in {"dual_branch", "transformer_film_sqi", "transformer_cross_sqi", "transformer_sqi_token_prefix"}:
        classifier_ckpt = None if str(args.init_mode) in {"he_scratch", "denoiser_warm_classifier_he"} else path_or_none(args.init_classifier_checkpoint)
        transformer = full_joint.build_trainable_classifier(morph, classifier_ckpt, device, dropout=float(args.transformer_dropout))
        if str(args.init_mode) in {"he_scratch", "denoiser_warm_classifier_he"}:
            full_joint.init_he_scratch(transformer)
            print(f"init_mode={args.init_mode} | transformer classifier reinitialized with Kaiming/He", flush=True)
        if args.variant == "dual_branch":
            encoder_delta = EncoderDeltaHead(feat_dim, int(args.hidden_dim), float(args.dropout), bool(args.gated_delta)).to(device)
        else:
            with torch.no_grad():
                denoise, _, _ = residual_unet_forward_with_features(
                    denoiser,
                    noisy,
                    feature_set=str(args.feature_set),
                    noise_scale=float(args.noise_scale),
                )
                pooled_dim = int(transformer.forward_features(denoise)["pooled"].shape[1])
            adapter_mode = {
                "transformer_film_sqi": "film",
                "transformer_cross_sqi": "cross",
                "transformer_sqi_token_prefix": "token",
            }[str(args.variant)]
            transformer_adapter = TransformerLatentAdapter(
                feat_dim,
                pooled_dim,
                int(transformer.cfg.enc_d_model),
                int(args.hidden_dim),
                float(args.dropout),
                adapter_mode,
            ).to(device)
    else:
        raise ValueError(f"unknown variant: {args.variant}")
    print(
        f"variant={args.variant} feature_set={args.feature_set} feat_dim={feat_dim} "
        f"detach_encoder_features={bool(args.detach_encoder_features)} freeze_denoiser={bool(args.freeze_denoiser)}",
        flush=True,
    )
    return EncoderAuditModel(
        denoiser=denoiser,
        encoder_head=encoder_head,
        waveform_head=waveform_head,
        transformer_classifier=transformer,
        encoder_delta=encoder_delta,
        explicit_sqi_delta=explicit_sqi_delta,
        transformer_adapter=transformer_adapter,
        variant=args.variant,
        noise_scale=float(args.noise_scale),
        delta_scale=float(args.delta_scale),
        detach_encoder_features=bool(args.detach_encoder_features),
        feature_set=str(args.feature_set),
    ).to(device)


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    morph = import_module(MORPH_TRAIN, "e311_morph_train_encoder_head_audit")
    full_joint = import_module(FULL_JOINT, "e311_full_joint_for_encoder_head_audit")

    source = Path(args.source_artifact_dir)
    train_ds = morph.ECGDenoiseDataset(source, "train")
    val_ds = morph.ECGDenoiseDataset(source, "val")
    test_ds = morph.ECGDenoiseDataset(source, "test")
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"splits train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = make_model(args, morph, full_joint, train_loader, device)
    if path_or_none(args.eval_checkpoint) is not None:
        ckpt = torch.load(path_or_none(args.eval_checkpoint), map_location=device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"eval_checkpoint={args.eval_checkpoint} | loaded model_state missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    class_weight = torch.tensor([float(x) for x in str(args.class_weight).split(",")], dtype=torch.float32, device=device)
    params: list[dict[str, Any]] = []
    denoiser_params = [p for p in model.denoiser.parameters() if p.requires_grad]
    if denoiser_params:
        params.append({"params": denoiser_params, "lr": float(args.lr_denoiser)})
    if model.encoder_head is not None:
        params.append({"params": [p for p in model.encoder_head.parameters() if p.requires_grad], "lr": float(args.lr_head)})
    if model.waveform_head is not None:
        params.append({"params": [p for p in model.waveform_head.parameters() if p.requires_grad], "lr": float(args.lr_head)})
    if model.transformer_classifier is not None:
        params.append({"params": [p for p in model.transformer_classifier.parameters() if p.requires_grad], "lr": float(args.lr_classifier)})
    if model.encoder_delta is not None:
        params.append({"params": [p for p in model.encoder_delta.parameters() if p.requires_grad], "lr": float(args.lr_head)})
    if model.explicit_sqi_delta is not None:
        params.append({"params": [p for p in model.explicit_sqi_delta.parameters() if p.requires_grad], "lr": float(args.lr_sqi)})
    if model.transformer_adapter is not None:
        params.append({"params": [p for p in model.transformer_adapter.parameters() if p.requires_grad], "lr": float(args.lr_head)})
    params = [group for group in params if group["params"]]
    optimizer = torch.optim.AdamW(params, weight_decay=float(args.weight_decay)) if params else None

    log: list[dict[str, Any]] = []
    best_key: tuple[float, ...] = (-1e9,)
    best_state: dict[str, Any] | None = None
    for epoch in range(int(args.epochs)):
        model.train()
        if bool(args.freeze_denoiser):
            model.denoiser.eval()
        sums = {"total": 0.0, "cls": 0.0, "den": 0.0, "snr": 0.0, "local": 0.0}
        n = 0
        for batch_idx, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and batch_idx >= int(args.max_train_batches):
                break
            noisy = batch["noisy"].to(device=device, dtype=torch.float32)
            clean = batch["clean"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.long)
            point_weight = batch["point_weight"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            out = model(noisy)
            l_cls = F.cross_entropy(out["logits"], y, weight=class_weight)
            l_den, _dbg = morph.morph_loss(out["denoise"], out["noise_hat"], noisy, clean, y, point_weight, sample_weight, str(args.loss_variant))
            l_snr = torch.zeros((), device=device)
            l_local = torch.zeros((), device=device)
            if out["snr_pred"] is not None and float(args.lambda_snr) > 0.0:
                severity = severity_from_snr(batch["snr_db"].to(device=device, dtype=torch.float32))
                l_snr = F.smooth_l1_loss(torch.sigmoid(out["snr_pred"]), severity)
            if out["local_pred"] is not None and float(args.lambda_local) > 0.0:
                local_mask = batch["local_mask"].to(device=device, dtype=torch.float32).unsqueeze(1)
                l_local = F.binary_cross_entropy_with_logits(out["local_pred"], local_mask)
            loss = (
                float(args.lambda_cls) * l_cls
                + float(args.lambda_den) * l_den
                + float(args.lambda_snr) * l_snr
                + float(args.lambda_local) * l_local
            )
            if optimizer is None:
                raise RuntimeError("No trainable parameters available for this run")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            bs = int(noisy.shape[0])
            n += bs
            sums["total"] += float(loss.detach().cpu().item()) * bs
            sums["cls"] += float(l_cls.detach().cpu().item()) * bs
            sums["den"] += float(l_den.detach().cpu().item()) * bs
            sums["snr"] += float(l_snr.detach().cpu().item()) * bs
            sums["local"] += float(l_local.detach().cpu().item()) * bs
        val_eval = evaluate(model, val_loader, device, morph, None)
        val_report = val_eval["report"]
        val_metrics = val_eval["metrics"]["overall"]
        acc = float(val_report["final_acc"])
        good, medium, bad = [float(x) for x in val_report["final_recall_good_medium_bad"]]
        denoise_score = float(val_metrics["denoise_score"])
        key = (
            acc - 0.02 * max(0.0, 2.75 - denoise_score) - 0.5 * max(0.0, 0.94 - good) - 0.5 * max(0.0, 0.94 - medium),
            acc,
            bad,
            medium,
            denoise_score,
        )
        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": v / max(1, n) for k, v in sums.items()},
            "val_base_acc": val_report["base_acc"],
            "val_final_acc": acc,
            "val_good_recall": good,
            "val_medium_recall": medium,
            "val_bad_recall": bad,
            "val_denoise_score": denoise_score,
            "val_mse_ratio": val_metrics["mse_ratio_pred_over_noisy"],
            "val_snr_gain": val_metrics["snr_improve_db_mean"],
        }
        log.append(row)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={row['train_total']:.4f} "
            f"val_acc={acc:.5f} good={good:.5f} med={medium:.5f} bad={bad:.5f} "
            f"den={denoise_score:.3f} snr+={row['val_snr_gain']:.3f}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_state = {
                "epoch": epoch + 1,
                "key": key,
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    test_eval = evaluate(model, test_loader, device, morph, out_dir)
    write_json(out_dir / "train_log.json", log)
    torch.save({"model_state": model.state_dict(), "args": vars(args), "best_state_meta": best_state}, out_dir / "ckpt_best_encoder_audit.pt")
    summary = {
        "status": "completed",
        "variant": args.variant,
        "args": vars(args),
        "split_counts": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "best_epoch": None if best_state is None else best_state["epoch"],
        "test_report": test_eval["report"],
        "denoise_metrics": test_eval["metrics"]["overall"],
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False)[:2500], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment-only denoiser encoder classifier audit.")
    parser.add_argument(
        "--variant",
        choices=(
            "encoder_head_only",
            "waveform_only_no_sqi",
            "encoder_latent_plus_explicit_sqi",
            "dual_branch",
            "transformer_film_sqi",
            "transformer_cross_sqi",
            "transformer_sqi_token_prefix",
        ),
        required=True,
    )
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_denoiser_checkpoint", default=str(DEFAULT_DENOISER))
    parser.add_argument("--init_classifier_checkpoint", default=str(DEFAULT_CLASSIFIER))
    parser.add_argument("--eval_checkpoint", default="none")
    parser.add_argument("--init_mode", choices=("warm", "he_scratch", "denoiser_warm_classifier_he"), default="warm")
    parser.add_argument("--feature_set", choices=FEATURE_SETS, default="full")
    parser.add_argument("--freeze_denoiser", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--transformer_dropout", type=float, default=0.10)
    parser.add_argument("--gated_delta", action="store_true")
    parser.add_argument("--detach_encoder_features", action="store_true")
    parser.add_argument("--loss_variant", default="morph_soft_guard")
    parser.add_argument("--noise_scale", type=float, default=0.955)
    parser.add_argument("--delta_scale", type=float, default=0.04)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--lambda_den", type=float, default=10.0)
    parser.add_argument("--lambda_snr", type=float, default=0.03)
    parser.add_argument("--lambda_local", type=float, default=0.005)
    parser.add_argument("--class_weight", default="1,1.40,1.70")
    parser.add_argument("--lr_denoiser", type=float, default=4e-5)
    parser.add_argument("--lr_classifier", type=float, default=4e-6)
    parser.add_argument("--lr_head", type=float, default=6e-4)
    parser.add_argument("--lr_sqi", type=float, default=6e-4)
    parser.add_argument("--sqi_hidden_dim", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
