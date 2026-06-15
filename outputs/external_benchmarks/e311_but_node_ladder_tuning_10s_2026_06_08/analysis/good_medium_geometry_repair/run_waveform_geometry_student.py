"""Waveform Geometry Student research runner.

External-only experiment.  The classifier sees waveform channels only.  The
47-column SQI/geometry table is used as a training-time teacher target, never as
an inference input.  Original BUT rows are report-only and are not used for
training, validation, threshold selection, or candidate selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ARCH_SCRIPT = ANALYSIS_DIR / "run_waveform_architecture_search.py"
NODE_ID = "N17043_gm_probe"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

REFERENCE_ROWS = [
    {
        "candidate": "47_feature_tabular_upper_bound",
        "bucket": "original_test_all_10s+",
        "acc": 0.963548,
        "macro_f1": 0.930683,
        "good_recall": 0.956319,
        "medium_recall": 0.972887,
        "bad_recall": 0.927007,
        "notes": "tabular-only upper bound; geometry features are inference inputs",
    },
    {
        "candidate": "current_best_waveform_transformer",
        "bucket": "original_test_all_10s+",
        "acc": 0.811018,
        "macro_f1": 0.730436,
        "good_recall": 0.960714,
        "medium_recall": 0.725938,
        "bad_recall": 0.401460,
        "notes": "aug_convtx_balanced_focal raw",
    },
    {
        "candidate": "stattoken_baseline",
        "bucket": "original_test_all_10s+",
        "acc": 0.813141,
        "macro_f1": 0.733720,
        "good_recall": 0.962088,
        "medium_recall": 0.725938,
        "bad_recall": 0.433090,
        "notes": "stattx_medium_guard raw",
    },
    {
        "candidate": "reduced_feature_auto_topk20",
        "bucket": "original_test_all_10s+",
        "acc": 0.913531,
        "macro_f1": np.nan,
        "good_recall": 0.973352,
        "medium_recall": 0.884546,
        "bad_recall": 0.695864,
        "notes": "reduced tabular model; not waveform-only",
    },
]

TEACHER_COLUMNS = [
    "pc1",
    "pc2",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "qrs_visibility",
    "qrs_prom_p90",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "diff_abs_p95",
    "template_corr",
    "amplitude_entropy",
    "low_amp_ratio",
    "sqi_bSQI",
    "sqi_sSQI",
    "sqi_kSQI",
]

CORE_COLUMNS = [
    "pc1",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "qrs_visibility",
    "baseline_step",
    "flatline_ratio",
    "non_qrs_diff_p95",
]

CANDIDATES: dict[str, dict[str, Any]] = {
    "mapstudent_patchtx": {
        "arch": "mapstudent",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "lr": 5.0e-4,
        "class_weight": [1.00, 1.22, 2.00],
        "focal_gamma": 0.8,
        "aux_weight": 0.75,
        "core_aux_weight": 0.55,
        "supcon_weight": 0.10,
        "rank_weight": 0.18,
        "recon_weight": 0.0,
        "seed": 20260801,
    },
    "masked_geometry_mae": {
        "arch": "masked_mae",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "patch": 25,
        "mask_ratio": 0.35,
        "lr": 4.5e-4,
        "class_weight": [1.00, 1.22, 2.00],
        "focal_gamma": 0.8,
        "aux_weight": 0.70,
        "core_aux_weight": 0.50,
        "supcon_weight": 0.08,
        "rank_weight": 0.18,
        "recon_weight": 0.20,
        "seed": 20260802,
    },
    "stattoken_v2": {
        "arch": "stattoken_v2",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.5e-4,
        "class_weight": [1.00, 1.25, 2.15],
        "focal_gamma": 1.0,
        "aux_weight": 0.65,
        "core_aux_weight": 0.50,
        "supcon_weight": 0.10,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "seed": 20260803,
    },
    "morphology_token_tx": {
        "arch": "morphology_token",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "lr": 5.0e-4,
        "class_weight": [1.00, 1.25, 2.05],
        "focal_gamma": 0.9,
        "aux_weight": 0.70,
        "core_aux_weight": 0.55,
        "supcon_weight": 0.12,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "seed": 20260804,
    },
    "stattoken_v2_badstress": {
        "arch": "stattoken_v2",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.18, 2.75],
        "focal_gamma": 1.1,
        "aux_weight": 0.62,
        "core_aux_weight": 0.48,
        "supcon_weight": 0.10,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "augment_strength": 0.35,
        "bad_outlier_aug_strength": 1.65,
        "seed": 20260805,
    },
    "morphology_token_badstress": {
        "arch": "morphology_token",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "lr": 4.5e-4,
        "class_weight": [1.00, 1.18, 2.55],
        "focal_gamma": 1.0,
        "aux_weight": 0.65,
        "core_aux_weight": 0.50,
        "supcon_weight": 0.12,
        "rank_weight": 0.20,
        "recon_weight": 0.0,
        "augment_strength": 0.35,
        "bad_outlier_aug_strength": 1.55,
        "seed": 20260806,
    },
    "qrs_detail_token_tx": {
        "arch": "qrs_detail_token",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.8e-4,
        "class_weight": [1.00, 1.24, 2.05],
        "focal_gamma": 0.9,
        "aux_weight": 0.78,
        "core_aux_weight": 0.70,
        "supcon_weight": 0.12,
        "rank_weight": 0.25,
        "recon_weight": 0.0,
        "seed": 20260807,
    },
    "atlas_memory_tx": {
        "arch": "atlas_memory",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "lr": 4.4e-4,
        "class_weight": [1.00, 1.24, 2.10],
        "focal_gamma": 0.9,
        "aux_weight": 0.72,
        "core_aux_weight": 0.70,
        "supcon_weight": 0.18,
        "rank_weight": 0.30,
        "recon_weight": 0.0,
        "seed": 20260808,
    },
    "qrs_atlas_memory_tx": {
        "arch": "qrs_atlas_memory",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.24, 2.10],
        "focal_gamma": 0.9,
        "aux_weight": 0.76,
        "core_aux_weight": 0.76,
        "supcon_weight": 0.18,
        "rank_weight": 0.32,
        "recon_weight": 0.0,
        "seed": 20260809,
    },
    "teacher_atlas_student": {
        "arch": "teacher_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "teacher_prototypes_per_class": 10,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.24, 2.15],
        "focal_gamma": 0.9,
        "aux_weight": 0.72,
        "core_aux_weight": 0.80,
        "atlas_distill_weight": 0.55,
        "supcon_weight": 0.16,
        "rank_weight": 0.34,
        "recon_weight": 0.0,
        "seed": 20260810,
    },
    "qrs_teacher_atlas_student": {
        "arch": "qrs_teacher_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "teacher_prototypes_per_class": 10,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.24, 2.15],
        "focal_gamma": 0.9,
        "aux_weight": 0.76,
        "core_aux_weight": 0.84,
        "atlas_distill_weight": 0.58,
        "supcon_weight": 0.18,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "seed": 20260811,
    },
    "hybrid_atlas_student": {
        "arch": "hybrid_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "teacher_prototypes_per_class": 10,
        "lr": 4.1e-4,
        "class_weight": [1.00, 1.24, 2.20],
        "focal_gamma": 0.9,
        "aux_weight": 0.76,
        "core_aux_weight": 0.84,
        "atlas_distill_weight": 0.58,
        "supcon_weight": 0.18,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "seed": 20260812,
    },
    "neighbor_stattoken_student": {
        "arch": "stattoken_v2",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 160,
        "lr": 4.2e-4,
        "class_weight": [1.00, 1.24, 2.35],
        "focal_gamma": 0.9,
        "aux_weight": 0.70,
        "core_aux_weight": 0.82,
        "neighbor_weight": 0.75,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.14,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "augment_strength": 0.30,
        "bad_outlier_aug_strength": 1.25,
        "seed": 20260813,
    },
    "neighbor_hybrid_atlas_student": {
        "arch": "hybrid_atlas_student",
        "channels": "robust3",
        "width": 96,
        "layers": 4,
        "heads": 4,
        "stat_hidden": 128,
        "prototypes_per_class": 8,
        "teacher_prototypes_per_class": 10,
        "lr": 4.0e-4,
        "class_weight": [1.00, 1.24, 2.25],
        "focal_gamma": 0.9,
        "aux_weight": 0.74,
        "core_aux_weight": 0.84,
        "atlas_distill_weight": 0.52,
        "neighbor_weight": 0.70,
        "neighbor_tau": 0.42,
        "supcon_weight": 0.16,
        "rank_weight": 0.36,
        "recon_weight": 0.0,
        "seed": 20260814,
    },
}


def load_arch_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_architecture_search", ARCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ARCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCH = load_arch_module()
GEOM = ARCH.GEOM
FEATURE_COLUMNS = list(ARCH.FEATURE_COLUMNS)
TEACHER_IDX = [FEATURE_COLUMNS.index(c) for c in TEACHER_COLUMNS if c in FEATURE_COLUMNS]
CORE_IDX = [FEATURE_COLUMNS.index(c) for c in CORE_COLUMNS if c in FEATURE_COLUMNS]


@dataclass
class EvalPayload:
    report: dict[str, Any]
    probs: np.ndarray
    pred: np.ndarray
    aux_pred: np.ndarray
    embeddings: np.ndarray
    y: np.ndarray


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(tokens)
        score = torch.einsum("btd,d->bt", h, self.query) / math.sqrt(max(1, h.shape[-1]))
        weight = torch.softmax(score, dim=1)
        return torch.einsum("bt,btd->bd", weight, tokens)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


def make_encoder_layer(width: int, heads: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * 4,
        dropout=0.08,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class PatchStudentEncoder(nn.Module):
    """PatchTST-style channel-aware patch Transformer."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int):
        super().__init__()
        self.in_ch = in_ch
        self.patch = patch
        self.patch_proj = nn.Linear(patch, width)
        self.channel_embed = nn.Parameter(torch.randn(in_ch, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pos = PositionalEncoding(width, max_len=768)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 3

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        keep = (n // self.patch) * self.patch
        x = x[:, :, :keep].reshape(b, c, keep // self.patch, self.patch)
        tokens = self.patch_proj(x).reshape(b, c * (keep // self.patch), -1)
        ch = self.channel_embed[:, None, :].expand(c, keep // self.patch, -1).reshape(1, c * (keep // self.patch), -1)
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls, tokens + ch], dim=1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenize(x)
        return self.encoder(self.pos(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        cls = tokens[:, 0, :]
        body = tokens[:, 1:, :]
        return torch.cat([cls, self.pool(body), body.mean(dim=1)], dim=1)


class MaskedGeometryMAEEncoder(PatchStudentEncoder):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int, patch: int):
        super().__init__(in_ch, width, layers, heads, patch)
        self.mask_token = nn.Parameter(torch.zeros(width))
        self.decoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, patch))

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        b, c, n = x.shape
        keep = (n // self.patch) * self.patch
        target = x[:, :, :keep].reshape(b, c * (keep // self.patch), self.patch)
        tokens = self.tokenize(x)
        patch_tokens = tokens[:, 1:, :]
        mask = None
        if self.training and mask_ratio > 0:
            mask = torch.rand(patch_tokens.shape[:2], device=x.device) < float(mask_ratio)
            patch_tokens = torch.where(mask[:, :, None], self.mask_token[None, None, :], patch_tokens)
            tokens = torch.cat([tokens[:, :1, :], patch_tokens], dim=1)
        encoded = self.encoder(self.pos(tokens))
        cls = encoded[:, 0, :]
        body = encoded[:, 1:, :]
        feat = torch.cat([cls, self.pool(body), body.mean(dim=1)], dim=1)
        recon = self.decoder(body)
        return feat, recon, mask if mask is not None else torch.ones(body.shape[:2], dtype=torch.bool, device=x.device)


class ConvSubsampleEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=256)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).transpose(1, 2)
        tokens = self.encoder(self.pos(tokens))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1)], dim=1)


class StatTokenTransformerV2(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int, stat_hidden: int):
        super().__init__()
        self.encoder = ConvSubsampleEncoder(in_ch, width, layers, heads)
        self.stat_dim = in_ch * 16
        self.stat_branch = nn.Sequential(
            nn.LayerNorm(self.stat_dim),
            nn.Linear(self.stat_dim, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, stat_hidden),
            nn.GELU(),
        )
        self.out_dim = self.encoder.out_dim + stat_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder(x), self.stat_branch(waveform_stats_v2(x))], dim=1)


class MorphologyTokenTransformer(nn.Module):
    def __init__(self, in_ch: int, width: int, layers: int, heads: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_ch, width, kernel_size=9, stride=4, padding=4),
                nn.Conv1d(in_ch, width, kernel_size=31, stride=8, padding=15),
                nn.Conv1d(in_ch, width, kernel_size=91, stride=16, padding=45),
                nn.Conv1d(in_ch, width, kernel_size=151, stride=25, padding=75),
            ]
        )
        self.branch_embed = nn.Parameter(torch.randn(len(self.branches), width) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.branches])
        self.pos = PositionalEncoding(width, max_len=768)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.out_dim = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed morphology channels are computed inside the network from the
        # waveform input. They are not precomputed external features.
        diff = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        detail = torch.abs(diff)
        smooth = F.avg_pool1d(x, kernel_size=63, stride=1, padding=31)
        morph = torch.cat([x, diff, detail, smooth], dim=1)
        pieces = []
        for i, conv in enumerate(self.branches):
            # The conv expects the original channel count, so fold morphology
            # views back by averaging groups. This keeps parameter count stable.
            view = morph.reshape(morph.shape[0], 4, x.shape[1], morph.shape[2]).mean(dim=1)
            t = conv(view).transpose(1, 2)
            pieces.append(self.norms[i](t) + self.branch_embed[i][None, None, :])
        tokens = self.encoder(self.pos(torch.cat(pieces, dim=1)))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1)], dim=1)


class QRSDetailTokenTransformer(nn.Module):
    """High-resolution local token encoder for QRS visibility/prominence."""

    def __init__(self, in_ch: int, width: int, layers: int, heads: int, stat_hidden: int):
        super().__init__()
        self.highres = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.detail = nn.Sequential(
            nn.Conv1d(in_ch * 3, width, kernel_size=11, stride=4, padding=5),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=768)
        self.encoder = nn.TransformerEncoder(make_encoder_layer(width, heads), num_layers=layers)
        self.pool = AttentionPool(width)
        self.stat_branch = nn.Sequential(
            nn.LayerNorm(in_ch * 15),
            nn.Linear(in_ch * 15, stat_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(stat_hidden, stat_hidden),
            nn.GELU(),
        )
        self.out_dim = width * 2 + stat_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        smooth = F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
        highpass = x - smooth
        detail_view = torch.cat([x, diff, highpass], dim=1)
        tokens = torch.cat([self.highres(x).transpose(1, 2), self.detail(detail_view).transpose(1, 2)], dim=1)
        tokens = self.encoder(self.pos(tokens))
        return torch.cat([self.pool(tokens), tokens.mean(dim=1), self.stat_branch(qrs_detail_stats(x))], dim=1)


class AtlasMemoryTransformer(nn.Module):
    """Prototype-distance memory for boundary confidence / KNN-purity targets."""

    def __init__(self, base: nn.Module, base_dim: int, prototypes_per_class: int):
        super().__init__()
        self.base = base
        self.base_dim = int(base_dim)
        self.prototypes_per_class = int(prototypes_per_class)
        self.prototype_proj = nn.Sequential(nn.LayerNorm(base_dim), nn.Linear(base_dim, base_dim), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(3, self.prototypes_per_class, base_dim) * 0.04)
        self.out_dim = base_dim + 9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base(x)
        z = F.normalize(self.prototype_proj(feat), dim=1)
        p = F.normalize(self.prototypes, dim=2)
        dist = torch.sum((z[:, None, None, :] - p[None, :, :, :]) ** 2, dim=3)
        min_dist = dist.min(dim=2).values
        mean_dist = dist.mean(dim=2)
        purity_like = torch.softmax(-min_dist, dim=1)
        return torch.cat([feat, min_dist, mean_dist, purity_like], dim=1)


def qrs_detail_stats(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    diff = x[:, :, 1:] - x[:, :, :-1]
    smooth31 = F.avg_pool1d(x, kernel_size=31, stride=1, padding=15)
    smooth101 = F.avg_pool1d(x, kernel_size=101, stride=1, padding=50)
    highpass = x - smooth31
    local_peak = highpass.abs()
    k01 = max(1, int(x.shape[2] * 0.01))
    k03 = max(1, int(x.shape[2] * 0.03))
    k10 = max(1, int(x.shape[2] * 0.10))
    top01 = torch.topk(local_peak, k=k01, dim=2).values.mean(dim=2)
    top03 = torch.topk(local_peak, k=k03, dim=2).values.mean(dim=2)
    top10 = torch.topk(local_peak, k=k10, dim=2).values.mean(dim=2)
    prom_ratio = top03 / (top10 + eps)
    detail_rms = torch.sqrt((highpass * highpass).mean(dim=2) + eps)
    diff_top = torch.topk(diff.abs(), k=k03, dim=2).values.mean(dim=2)
    diff_bg = torch.topk(diff.abs(), k=k10, dim=2).values.mean(dim=2)
    diff_ratio = diff_top / (diff_bg + eps)
    baseline_span = smooth101.amax(dim=2) - smooth101.amin(dim=2)
    baseline_slope = (smooth101[:, :, -1] - smooth101[:, :, 0]).abs()
    flat = (diff.abs() < 0.015).float().mean(dim=2)
    low_amp = (x.abs() < 0.050).float().mean(dim=2)
    mean_abs = x.abs().mean(dim=2)
    rms = torch.sqrt((x * x).mean(dim=2) + eps)
    ptp = x.amax(dim=2) - x.amin(dim=2)
    zcr = ((x[:, :, 1:] * x[:, :, :-1]) < 0).float().mean(dim=2)
    return torch.cat([top01, top03, top10, prom_ratio, detail_rms, diff_top, diff_ratio, baseline_span, baseline_slope, flat, low_amp, mean_abs, rms, ptp, zcr], dim=1)


class GeometryStudent(nn.Module):
    def __init__(self, cfg: dict[str, Any], in_ch: int):
        super().__init__()
        arch = str(cfg["arch"])
        width = int(cfg["width"])
        layers = int(cfg["layers"])
        heads = int(cfg["heads"])
        self.arch = arch
        self.use_teacher_atlas = arch in {"teacher_atlas_student", "qrs_teacher_atlas_student", "hybrid_atlas_student"}
        if arch == "mapstudent":
            self.encoder = PatchStudentEncoder(in_ch, width, layers, heads, int(cfg["patch"]))
            out_dim = self.encoder.out_dim
        elif arch == "masked_mae":
            self.encoder = MaskedGeometryMAEEncoder(in_ch, width, layers, heads, int(cfg["patch"]))
            out_dim = self.encoder.out_dim
        elif arch == "stattoken_v2":
            self.encoder = StatTokenTransformerV2(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "morphology_token":
            self.encoder = MorphologyTokenTransformer(in_ch, width, layers, heads)
            out_dim = self.encoder.out_dim
        elif arch == "qrs_detail_token":
            self.encoder = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "atlas_memory":
            base = StatTokenTransformerV2(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        elif arch == "qrs_atlas_memory":
            base = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        elif arch == "teacher_atlas_student":
            self.encoder = StatTokenTransformerV2(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "qrs_teacher_atlas_student":
            self.encoder = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            out_dim = self.encoder.out_dim
        elif arch == "hybrid_atlas_student":
            base = QRSDetailTokenTransformer(in_ch, width, layers, heads, int(cfg["stat_hidden"]))
            self.encoder = AtlasMemoryTransformer(base, base.out_dim, int(cfg["prototypes_per_class"]))
            out_dim = self.encoder.out_dim
        else:
            raise ValueError(f"unknown architecture {arch}")
        if self.use_teacher_atlas:
            proto_count = int(cfg["teacher_prototypes_per_class"])
            self.register_buffer("teacher_prototypes", torch.zeros(3, proto_count, len(CORE_IDX)), persistent=True)
            class_head_in = 192 + 9
        else:
            class_head_in = 192
        self.project = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, 192), nn.GELU(), nn.Dropout(0.08))
        self.class_head = nn.Sequential(nn.LayerNorm(class_head_in), nn.Linear(class_head_in, 96), nn.GELU(), nn.Dropout(0.08), nn.Linear(96, 3))
        self.aux_head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 128), nn.GELU(), nn.Linear(128, len(FEATURE_COLUMNS)))

    def set_teacher_prototypes(self, prototypes: np.ndarray | torch.Tensor) -> None:
        if not self.use_teacher_atlas:
            return
        tensor = torch.as_tensor(prototypes, dtype=torch.float32, device=self.teacher_prototypes.device)
        if tuple(tensor.shape) != tuple(self.teacher_prototypes.shape):
            raise ValueError(f"prototype shape mismatch: got {tuple(tensor.shape)} expected {tuple(self.teacher_prototypes.shape)}")
        self.teacher_prototypes.copy_(tensor)

    def teacher_distance_features(self, aux_values: torch.Tensor) -> torch.Tensor:
        core = aux_values[:, CORE_IDX]
        dist = torch.mean((core[:, None, None, :] - self.teacher_prototypes[None, :, :, :]) ** 2, dim=3)
        min_dist = dist.min(dim=2).values
        mean_dist = dist.mean(dim=2)
        purity_like = torch.softmax(-min_dist, dim=1)
        return torch.cat([min_dist, mean_dist, purity_like], dim=1)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> dict[str, torch.Tensor | None]:
        recon = None
        recon_mask = None
        if self.arch == "masked_mae":
            feat, recon, recon_mask = self.encoder(x, mask_ratio)
        else:
            feat = self.encoder(x)
        emb = self.project(feat)
        aux_pred = self.aux_head(emb)
        if self.use_teacher_atlas:
            atlas_pred = self.teacher_distance_features(aux_pred)
            head_input = torch.cat([emb, atlas_pred], dim=1)
        else:
            atlas_pred = None
            head_input = emb
        return {
            "logits": self.class_head(head_input),
            "aux_pred": aux_pred,
            "atlas_pred": atlas_pred,
            "embedding": emb,
            "recon": recon,
            "recon_mask": recon_mask,
        }


class AugmentedSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, base: Any, strength: float, seed: int, bad_strength: float):
        self.base = base
        self.strength = float(strength)
        self.bad_strength = float(bad_strength)
        self.rng = np.random.default_rng(seed)
        self.x = base.x
        self.y = base.y
        self.aux = base.aux
        self.mean = base.mean
        self.std = base.std
        self.frame = base.frame

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self.base[i]
        x = item["x"].detach().cpu().numpy().astype(np.float32, copy=True)
        y = int(item["y"])
        x = augment_synthetic_waveform(x, self.rng, self.strength, y, self.bad_strength)
        return {"x": torch.from_numpy(x), "aux": item["aux"], "y": item["y"]}


def waveform_stats_v2(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    diff = x[:, :, 1:] - x[:, :, :-1]
    mean = x.mean(dim=2)
    std = x.std(dim=2)
    mean_abs = x.abs().mean(dim=2)
    rms = torch.sqrt((x * x).mean(dim=2) + eps)
    peak_to_peak = x.amax(dim=2) - x.amin(dim=2)
    diff_abs = diff.abs().mean(dim=2)
    diff_rms = torch.sqrt((diff * diff).mean(dim=2) + eps)
    flat = (diff.abs() < 0.015).float().mean(dim=2)
    low_amp = (x.abs() < 0.050).float().mean(dim=2)
    zcr = ((x[:, :, 1:] * x[:, :, :-1]) < 0).float().mean(dim=2)
    k10 = max(1, int(diff.shape[2] * 0.10))
    k02 = max(1, int(diff.shape[2] * 0.02))
    diff_tail10 = torch.topk(diff.abs(), k=k10, dim=2).values.mean(dim=2)
    diff_tail02 = torch.topk(diff.abs(), k=k02, dim=2).values.mean(dim=2)
    smooth_short = F.avg_pool1d(x, kernel_size=15, stride=1, padding=7)
    smooth_long = F.avg_pool1d(x, kernel_size=101, stride=1, padding=50)
    detail_short = (x - smooth_short).abs().mean(dim=2)
    detail_long = (x - smooth_long).abs().mean(dim=2)
    baseline_slope = (smooth_long[:, :, -1] - smooth_long[:, :, 0]).abs()
    contact = (x.abs() < 0.015).float().mean(dim=2)
    return torch.cat(
        [
            mean,
            std,
            mean_abs,
            rms,
            peak_to_peak,
            diff_abs,
            diff_rms,
            flat,
            low_amp,
            zcr,
            diff_tail10,
            diff_tail02,
            detail_short,
            detail_long,
            baseline_slope,
            contact,
        ],
        dim=1,
    )


def augment_synthetic_waveform(x: np.ndarray, rng: np.random.Generator, strength: float, y: int, bad_strength: float) -> np.ndarray:
    if strength > 0:
        if rng.random() < 0.65:
            x = np.roll(x, int(rng.integers(-32, 33)), axis=1)
        if rng.random() < 0.75:
            x *= float(rng.uniform(0.88, 1.14))
        if rng.random() < 0.55:
            x += rng.normal(0.0, float(rng.uniform(0.005, 0.025) * strength), size=x.shape).astype(np.float32)
        if rng.random() < 0.35:
            width = int(rng.integers(18, 90))
            start = int(rng.integers(0, max(1, x.shape[1] - width)))
            x[:, start : start + width] *= float(rng.uniform(0.15, 0.70))
    if y == CLASS_TO_INT["bad"] and bad_strength > 0:
        x = augment_controlled_bad_outlier(x, rng, bad_strength)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_controlled_bad_outlier(x: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    # Controlled bad expansion: broad but not absurd contact/baseline/detail
    # perturbations.  This is synthetic PTB augmentation only.
    n = x.shape[1]
    if rng.random() < 0.80:
        width = int(rng.integers(120, 360))
        start = int(rng.integers(0, max(1, n - width)))
        x[:, start : start + width] *= float(rng.uniform(0.05, 0.45))
    if rng.random() < 0.75:
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        drift = float(rng.uniform(0.08, 0.24) * strength)
        x += drift * np.sin(2 * np.pi * float(rng.uniform(0.08, 0.35)) * t + float(rng.uniform(0, 2 * np.pi)))[None, :]
    if rng.random() < 0.65:
        x += rng.normal(0.0, float(rng.uniform(0.035, 0.110) * strength), size=x.shape).astype(np.float32)
    if rng.random() < 0.45:
        clip = float(rng.uniform(0.45, 1.05))
        x = np.clip(x, -clip, clip)
    return x


def build_teacher_prototypes(aux: np.ndarray, y: np.ndarray, prototypes_per_class: int) -> np.ndarray:
    core = np.asarray(aux[:, CORE_IDX], dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    out = np.zeros((3, int(prototypes_per_class), core.shape[1]), dtype=np.float32)
    pc1_col = 0
    if "pc1" in CORE_COLUMNS:
        pc1_col = CORE_COLUMNS.index("pc1")
    for cls in range(3):
        rows = core[labels == cls]
        if rows.size == 0:
            continue
        order = np.argsort(rows[:, pc1_col])
        chunks = np.array_split(rows[order], int(prototypes_per_class))
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0:
                out[cls, i] = rows.mean(axis=0)
            else:
                out[cls, i] = chunk.mean(axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "search", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES.keys()))
    parser.add_argument("--top-k-search", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--but-report-mode", choices=["passed", "always", "none"], default="passed")
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}; expected {sorted(CANDIDATES)}")
    if args.stage == "smoke":
        epochs = args.epochs or 2
        run_set(names, epochs, args, run_label="smoke", but_report_mode="none")
        return
    if args.stage == "search":
        epochs = args.epochs or 12
        run_set(names, epochs, args, run_label="search", but_report_mode=args.but_report_mode)
        return
    smoke_payload = run_set(names, args.epochs or 2, args, run_label="smoke", but_report_mode="none")
    ranked = rank_candidates(smoke_payload["metrics"])
    selected = [x for x in ranked[: int(args.top_k_search)] if x in CANDIDATES]
    if not selected:
        selected = names[: int(args.top_k_search)]
    print(f"selected for long search: {selected}", flush=True)
    search_args = argparse.Namespace(**vars(args))
    run_set(selected, max(10, args.epochs or 12), search_args, run_label="search", but_report_mode=args.but_report_mode)


def run_set(names: list[str], epochs: int, args: argparse.Namespace, run_label: str, but_report_mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries = []
    for name in names:
        summary = train_candidate(name, CANDIDATES[name], epochs, args, run_label, but_report_mode)
        summaries.append(summary)
        rows.extend(summary["metric_rows"])
    metrics = pd.DataFrame(rows)
    prefix = f"waveform_geometry_student_{run_label}"
    metrics_path = ANALYSIS_DIR / f"{prefix}_metrics.csv"
    summary_path = ANALYSIS_DIR / f"{prefix}_summary.json"
    report_path = ANALYSIS_DIR / f"{prefix}_report.md"
    report_copy = REPORT_DIR / report_path.name
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "created_at": now(),
        "run_label": run_label,
        "variant_id": ARCH.VARIANT_ID,
        "dirty_worktree_warning": git_status_short(),
        "input_contract": "waveform only; SQI/geometry columns are training teacher targets, never classifier inputs",
        "training_input": "PTB-generated synthetic waveform only",
        "original_used_for_training": False,
        "but_usage": "report-only held-out buckets",
        "teacher_columns": [FEATURE_COLUMNS[i] for i in TEACHER_IDX],
        "core_teacher_columns": [FEATURE_COLUMNS[i] for i in CORE_IDX],
        "reference_rows": REFERENCE_ROWS,
        "metrics_csv": str(metrics_path),
        "report": str(report_copy),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(metrics, payload)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report, flush=True)
    return {"metrics": metrics, "payload": payload}


def train_candidate(
    name: str,
    cfg: dict[str, Any],
    epochs: int,
    args: argparse.Namespace,
    run_label: str,
    but_report_mode: str,
) -> dict[str, Any]:
    print(f"training {run_label}:{name}: {cfg}", flush=True)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_train_ds = ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    if float(cfg.get("augment_strength", 0.0)) > 0 or float(cfg.get("bad_outlier_aug_strength", 0.0)) > 0:
        train_ds = AugmentedSyntheticDataset(
            base_train_ds,
            float(cfg.get("augment_strength", 0.0)),
            int(cfg["seed"]),
            float(cfg.get("bad_outlier_aug_strength", 0.0)),
        )
    else:
        train_ds = base_train_ds
    val_ds = ARCH.SyntheticWaveDataset("val", str(cfg["channels"]))
    test_ds = ARCH.SyntheticWaveDataset("test", str(cfg["channels"]))
    norm = ARCH.FeatureNorm(mean=base_train_ds.mean, std=base_train_ds.std)
    original_ds = ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    original_loader = DataLoader(original_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    model = GeometryStudent(cfg, int(train_ds.x.shape[1])).to(device)
    if getattr(model, "use_teacher_atlas", False):
        prototypes = build_teacher_prototypes(base_train_ds.aux, base_train_ds.y, int(cfg["teacher_prototypes_per_class"]))
        model.set_teacher_prototypes(prototypes)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    class_weight = torch.tensor(cfg["class_weight"], dtype=torch.float32, device=device)
    best_state = None
    best_epoch = 0
    best_score = -1e9
    logs = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(x, mask_ratio=float(cfg.get("mask_ratio", 0.0)))
            cls_loss = classification_loss(out["logits"], y, class_weight, float(cfg.get("focal_gamma", 0.0)))
            aux_loss = F.smooth_l1_loss(out["aux_pred"][:, TEACHER_IDX], aux[:, TEACHER_IDX])
            core_loss = F.smooth_l1_loss(out["aux_pred"][:, CORE_IDX], aux[:, CORE_IDX]) if CORE_IDX else torch.zeros((), device=device)
            if out.get("atlas_pred") is not None:
                atlas_target = model.teacher_distance_features(aux)
                atlas_loss = F.smooth_l1_loss(out["atlas_pred"], atlas_target)
            else:
                atlas_loss = torch.zeros((), device=device)
            con_loss = supervised_contrastive_loss(out["embedding"], y)
            rank_loss = feature_rank_loss(out["aux_pred"], aux)
            neighbor_loss = teacher_neighborhood_loss(out["embedding"], aux, float(cfg.get("neighbor_tau", 0.45)))
            rec_loss = reconstruction_loss(out, x, int(cfg.get("patch", 25)))
            loss = (
                cls_loss
                + float(cfg["aux_weight"]) * aux_loss
                + float(cfg["core_aux_weight"]) * core_loss
                + float(cfg.get("atlas_distill_weight", 0.0)) * atlas_loss
                + float(cfg.get("neighbor_weight", 0.0)) * neighbor_loss
                + float(cfg["supcon_weight"]) * con_loss
                + float(cfg["rank_weight"]) * rank_loss
                + float(cfg["recon_weight"]) * rec_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "cls": float(cls_loss.detach().cpu()),
                    "aux": float(aux_loss.detach().cpu()),
                    "core": float(core_loss.detach().cpu()),
                    "atlas": float(atlas_loss.detach().cpu()),
                    "neighbor": float(neighbor_loss.detach().cpu()),
                    "supcon": float(con_loss.detach().cpu()),
                    "rank": float(rank_loss.detach().cpu()),
                    "recon": float(rec_loss.detach().cpu()),
                }
            )
        val_payload = eval_loader(model, val_loader, device)
        val_rep = val_payload.report
        score = (
            val_rep["acc"]
            + 0.25 * val_rep["macro_f1"]
            + 0.25 * min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])
            - 0.02 * val_rep["teacher_core_mae"]
        )
        loss_mean = pd.DataFrame(losses).mean(numeric_only=True).to_dict()
        logs.append({"epoch": epoch, **{f"train_{k}": v for k, v in loss_mean.items()}, **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}})
        print(
            f"{run_label}:{name} epoch={epoch}/{epochs} loss={loss_mean['loss']:.4f} "
            f"val_acc={val_rep['acc']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} "
            f"teacher_core_mae={val_rep['teacher_core_mae']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_payload = eval_loader(model, val_loader, device)
    test_payload = eval_loader(model, test_loader, device)
    threshold = ARCH.calibrate_bad_threshold(val_ds.y, val_payload.probs)
    synthetic_badcal = GEOM.metric_report(test_ds.y, ARCH.apply_bad_threshold(test_payload.probs, threshold), test_payload.probs)
    metric_rows = [
        metric_row(name, run_label, "synthetic_val", val_payload.report),
        metric_row(name, run_label, "synthetic_test", test_payload.report),
        metric_row(f"{name}_badcal", run_label, "synthetic_test", synthetic_badcal),
    ]
    useful = (
        test_payload.report["acc"] >= 0.94
        and test_payload.report["good_recall"] >= 0.92
        and test_payload.report["medium_recall"] >= 0.90
        and test_payload.report["bad_recall"] >= 0.90
    )
    original_payload = None
    if but_report_mode == "always" or (but_report_mode == "passed" and useful):
        original_payload = eval_loader(model, original_loader, device)
        for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
            metric_rows.append(metric_row(name, run_label, bucket, ARCH.bucket_report(original_ds, original_payload.probs, bucket)))
            metric_rows.append(metric_row(f"{name}_badcal", run_label, bucket, ARCH.bucket_report(original_ds, original_payload.probs, bucket, threshold)))
    run_dir = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / run_label / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "model_kind": "waveform_geometry_student",
            "input_contract": "waveform only; geometry/SQI teacher targets used only in training loss",
            "teacher_columns": [FEATURE_COLUMNS[i] for i in TEACHER_IDX],
            "core_teacher_columns": [FEATURE_COLUMNS[i] for i in CORE_IDX],
            "feature_columns": FEATURE_COLUMNS,
            "feature_train_mean": base_train_ds.mean,
            "feature_train_std": base_train_ds.std,
            "teacher_prototypes": model.teacher_prototypes.detach().cpu().numpy() if getattr(model, "use_teacher_atlas", False) else None,
            "best_epoch": best_epoch,
            "bad_threshold_trainval": threshold,
            "original_rows_used_for_training": False,
            "but_usage": "report-only",
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    write_diagnostics(name, run_label, run_dir, test_ds, test_payload, original_ds if original_payload else None, original_payload)
    return {
        "candidate": name,
        "run_label": run_label,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "useful_gate_passed": useful,
        "bad_threshold_trainval": threshold,
        "synthetic_test": test_payload.report,
        "original_report_ran": original_payload is not None,
        "metric_rows": metric_rows,
    }


def classification_loss(logits: torch.Tensor, y: torch.Tensor, class_weight: torch.Tensor, focal_gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, weight=class_weight, reduction="none")
    if focal_gamma <= 0:
        return ce.mean()
    pt = torch.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1).clamp(1e-5, 1.0)
    return (((1.0 - pt) ** float(focal_gamma)) * ce).mean()


def supervised_contrastive_loss(emb: torch.Tensor, y: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    if emb.shape[0] < 4:
        return torch.zeros((), device=emb.device)
    z = F.normalize(emb, dim=1)
    logits = (z @ z.T) / temperature
    logits = logits - torch.eye(z.shape[0], device=z.device) * 1e9
    same = (y[:, None] == y[None, :]) & (~torch.eye(z.shape[0], dtype=torch.bool, device=z.device))
    if same.sum() == 0:
        return torch.zeros((), device=emb.device)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per = -(log_prob * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
    return per[same.any(dim=1)].mean()


def feature_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 8 or not CORE_IDX:
        return torch.zeros((), device=pred.device)
    idx = torch.arange(pred.shape[0], device=pred.device)
    shuffled = idx[torch.randperm(pred.shape[0], device=pred.device)]
    losses = []
    for col in CORE_IDX:
        td = target[:, col] - target[shuffled, col]
        valid = td.abs() > 0.35
        if valid.any():
            sign = torch.sign(td[valid])
            pdiff = pred[:, col][valid] - pred[shuffled, col][valid]
            losses.append(F.relu(0.05 - sign * pdiff).mean())
    if not losses:
        return torch.zeros((), device=pred.device)
    return torch.stack(losses).mean()


def teacher_neighborhood_loss(embedding: torch.Tensor, aux: torch.Tensor, tau: float) -> torch.Tensor:
    if embedding.shape[0] < 8 or not CORE_IDX:
        return torch.zeros((), device=embedding.device)
    teacher = F.normalize(aux[:, CORE_IDX], dim=1)
    student = F.normalize(embedding, dim=1)
    teacher_dist = torch.cdist(teacher, teacher, p=2)
    student_dist = torch.cdist(student, student, p=2)
    eye = torch.eye(teacher_dist.shape[0], dtype=torch.bool, device=teacher_dist.device)
    scale = max(float(tau), 1e-3)
    teacher_logits = (-teacher_dist / scale).masked_fill(eye, -1e4)
    student_logits = (-student_dist / scale).masked_fill(eye, -1e4)
    target = torch.softmax(teacher_logits, dim=1).detach()
    log_pred = torch.log_softmax(student_logits, dim=1)
    return F.kl_div(log_pred, target, reduction="batchmean")


def reconstruction_loss(out: dict[str, torch.Tensor | None], x: torch.Tensor, patch: int) -> torch.Tensor:
    recon = out.get("recon")
    mask = out.get("recon_mask")
    if recon is None or mask is None:
        return torch.zeros((), device=x.device)
    b, c, n = x.shape
    keep = (n // patch) * patch
    target = x[:, :, :keep].reshape(b, c * (keep // patch), patch)
    if mask.sum() == 0:
        return torch.zeros((), device=x.device)
    return F.smooth_l1_loss(recon[mask], target[mask])


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalPayload:
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    aux_pred: list[np.ndarray] = []
    aux_abs: list[np.ndarray] = []
    emb: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        aux = batch["aux"].to(device=device, dtype=torch.float32)
        out = model(x, mask_ratio=0.0)
        p = F.softmax(out["logits"], dim=1).detach().cpu().numpy()
        probs.append(p)
        pred_aux = out["aux_pred"].detach().cpu().numpy()
        aux_pred.append(pred_aux)
        aux_abs.append(torch.mean(torch.abs(out["aux_pred"] - aux), dim=0).detach().cpu().numpy())
        emb.append(out["embedding"].detach().cpu().numpy())
        ys.append(batch["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    pred = p.argmax(axis=1).astype(np.int64)
    rep = GEOM.metric_report(y, pred, p)
    aux_mae = np.stack(aux_abs).mean(axis=0)
    rep["teacher_mae"] = float(aux_mae.mean())
    rep["teacher_core_mae"] = float(aux_mae[CORE_IDX].mean()) if CORE_IDX else rep["teacher_mae"]
    return EvalPayload(rep, p, pred, np.concatenate(aux_pred), np.concatenate(emb), y)


def metric_row(candidate: str, run_label: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "run_label": run_label, "bucket": bucket}
    row.update(rep)
    row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
    return row


def write_diagnostics(
    name: str,
    run_label: str,
    run_dir: Path,
    synthetic_ds: Any,
    synthetic: EvalPayload,
    original_ds: Any | None,
    original: EvalPayload | None,
) -> None:
    feature_rows = feature_recovery_rows(name, run_label, "synthetic_test", synthetic, synthetic_ds.aux)
    pd.DataFrame(feature_rows).to_csv(run_dir / "feature_teacher_recovery_synthetic_test.csv", index=False)
    if original is not None and original_ds is not None:
        pd.DataFrame(feature_recovery_rows(name, run_label, "original_all_10s+", original, original_ds.aux)).to_csv(run_dir / "feature_teacher_recovery_original.csv", index=False)
    write_error_rows(run_dir / "synthetic_test_error_rows.csv", synthetic_ds, synthetic)
    plot_embedding_pca(run_dir / "synthetic_test_embedding_pca.png", synthetic.embeddings, synthetic.y, synthetic.pred, f"{name} synthetic embedding")
    plot_waveform_errors(run_dir / "synthetic_test_error_waveforms.png", synthetic_ds, synthetic, f"{name} synthetic mistakes")


def feature_recovery_rows(candidate: str, run_label: str, split: str, payload: EvalPayload, target_aux: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i, col in enumerate(FEATURE_COLUMNS):
        t = target_aux[:, i].astype(float)
        p = payload.aux_pred[:, i].astype(float)
        if np.std(t) > 1e-8 and np.std(p) > 1e-8:
            corr = float(np.corrcoef(t, p)[0, 1])
        else:
            corr = float("nan")
        rows.append(
            {
                "candidate": candidate,
                "run_label": run_label,
                "split": split,
                "feature": col,
                "mae_z": float(np.mean(np.abs(p - t))),
                "corr": corr,
                "is_teacher_core": col in CORE_COLUMNS,
                "is_teacher": col in TEACHER_COLUMNS,
            }
        )
    return rows


def write_error_rows(path: Path, ds: Any, payload: EvalPayload) -> None:
    rows = []
    frame = getattr(ds, "frame", None)
    for i, (y, pred) in enumerate(zip(payload.y, payload.pred)):
        if int(y) == int(pred):
            continue
        base = {
            "row": i,
            "true": INT_TO_CLASS[int(y)],
            "pred": INT_TO_CLASS[int(pred)],
            "prob_good": float(payload.probs[i, 0]),
            "prob_medium": float(payload.probs[i, 1]),
            "prob_bad": float(payload.probs[i, 2]),
        }
        if frame is not None:
            for col in ["idx", "split", "y_class", "boundary_block", "source_variant"]:
                if col in frame.columns:
                    base[col] = frame.iloc[i][col]
        rows.append(base)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_embedding_pca(path: Path, emb: np.ndarray, y: np.ndarray, pred: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        z = emb - emb.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        xy = z @ vt[:2].T
        rng = np.random.default_rng(202608)
        take = np.arange(len(y))
        if len(take) > 3000:
            take = rng.choice(take, size=3000, replace=False)
        colors = np.array(["#2ca25f", "#3182bd", "#de2d26"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
        axes[0].scatter(xy[take, 0], xy[take, 1], s=6, c=colors[y[take]], alpha=0.55, linewidths=0)
        axes[0].set_title("true label")
        axes[1].scatter(xy[take, 0], xy[take, 1], s=6, c=colors[pred[take]], alpha=0.55, linewidths=0)
        axes[1].set_title("prediction")
        for ax in axes:
            ax.set_xlabel("embedding PC1")
            ax.set_ylabel("embedding PC2")
            ax.grid(alpha=0.15)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"embedding PCA plot failed: {exc}", encoding="utf-8")


def plot_waveform_errors(path: Path, ds: Any, payload: EvalPayload, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        mistake_idx = np.where(payload.y != payload.pred)[0]
        if len(mistake_idx) == 0:
            path.with_suffix(".txt").write_text("no mistakes to plot", encoding="utf-8")
            return
        selected = []
        for true_label, pred_label in [(0, 1), (1, 0), (2, 1), (1, 2)]:
            idx = mistake_idx[(payload.y[mistake_idx] == true_label) & (payload.pred[mistake_idx] == pred_label)]
            selected.extend(idx[:3].tolist())
        selected = selected[:12] or mistake_idx[:12].tolist()
        n = len(selected)
        cols = 3
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), dpi=160, squeeze=False)
        for ax, i in zip(axes.ravel(), selected):
            x = ds.x[i, 0]
            ax.plot(x, color="#222222", lw=0.9)
            ax.set_title(f"{INT_TO_CLASS[int(payload.y[i])]} -> {INT_TO_CLASS[int(payload.pred[i])]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(selected) :]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"waveform plot failed: {exc}", encoding="utf-8")


def rank_candidates(metrics: pd.DataFrame) -> list[str]:
    view = metrics[(metrics["bucket"] == "synthetic_test") & (~metrics["candidate"].astype(str).str.endswith("_badcal"))].copy()
    if view.empty:
        return []
    view["score"] = view["acc"].astype(float) + 0.25 * view["macro_f1"].astype(float) + 0.25 * view[["good_recall", "medium_recall", "bad_recall"]].astype(float).min(axis=1)
    return view.sort_values("score", ascending=False)["candidate"].tolist()


def render_report(metrics: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# Waveform Geometry Student",
        "",
        "Input is waveform only. SQI/geometry features are teacher targets in the loss, not classifier inputs. Original BUT is report-only.",
        "",
        "## Dirty Worktree Warning",
        "",
        "Existing worktree changes were present before this runner. This experiment writes only external analysis/report/run outputs.",
        "",
        "```text",
        (payload.get("dirty_worktree_warning") or "(clean)").strip()[:4000],
        "```",
        "",
        "## Metrics",
        "",
        "| Candidate | Run | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher MAE | Core MAE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    show = metrics.copy()
    for _, row in show.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['run_label']} | {row['bucket']} | {float(row['acc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} | "
            f"{float(row.get('teacher_mae', np.nan)):.4f} | {float(row.get('teacher_core_mae', np.nan)):.4f} |"
        )
    lines.extend(["", "## Baselines", ""])
    for row in payload["reference_rows"]:
        lines.append(
            f"- `{row['candidate']}` {row['bucket']}: acc `{row['acc']:.6f}`, recalls "
            f"{row['good_recall']:.3f}/{row['medium_recall']:.3f}/{row['bad_recall']:.3f}`. {row['notes']}."
        )
    lines.extend(["", "## Candidates", ""])
    for item in payload["summaries"]:
        gate = "pass" if item.get("useful_gate_passed") else "fail"
        lines.append(
            f"- `{item['candidate']}` ({item['run_label']}): best_epoch={item['best_epoch']}, useful_gate={gate}, "
            f"original_report_ran={item['original_report_ran']}, checkpoint=`{item['checkpoint']}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Selection uses synthetic/node diagnostic only.",
            "- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.",
            "- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.",
            f"- Metrics CSV: `{payload['metrics_csv']}`",
            "",
        ]
    )
    return "\n".join(lines)


def git_status_short() -> str:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=str(ROOT), text=True, stderr=subprocess.STDOUT, timeout=15)
        return out
    except Exception as exc:
        return f"git status failed: {exc}"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    main()
