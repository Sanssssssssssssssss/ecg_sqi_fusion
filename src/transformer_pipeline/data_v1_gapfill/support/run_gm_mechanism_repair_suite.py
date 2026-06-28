from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
BASE_SCRIPT = SCRIPT_PATH.with_name("run_event_factorized_sqi_conformer.py")
spec = importlib.util.spec_from_file_location("event_factorized_sqi_conformer_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load base runner: {BASE_SCRIPT}")
EVT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = EVT
spec.loader.exec_module(EVT)


SUITE_NAME = "gm_mechanism_repair_suite"
SUITE_OUT_DIR = EVT.ANALYSIS_DIR / SUITE_NAME
SUITE_REPORT_DIR = EVT.REPORT_DIR / SUITE_NAME
SUITE_RUN_DIR = EVT.OUT_ROOT / "runs" / SUITE_NAME

BaseModel = EVT.EventFactorizedSQIConformer
LegacyFactorTransform = EVT.FactorTransform
ORIG_FACTOR_LOSS = EVT.factor_loss
ORIG_FACTOR_RECOVERY_ROWS = EVT.factor_recovery_rows
ORIG_TRAIN_MODEL = EVT.train_model
ORIG_FIT_FACTOR_TRANSFORM = EVT.fit_factor_transform
ORIG_UNIFIED_SUBTYPE_AUX_LOSS = EVT.unified_subtype_aux_loss

ACTIVE_CFG: dict[str, Any] = {}
ACTIVE_FACTOR_TRANSFORM: Any | None = None


@dataclass(frozen=True)
class FactorSpec:
    name: str
    kind: str
    max_value: float | None = None
    boundary_role: str | None = None


FACTOR_SPECS: dict[str, FactorSpec] = {
    "qrs_visibility": FactorSpec("qrs_visibility", "scale", 2.0, "good"),
    "detector_agreement": FactorSpec("detector_agreement", "bounded", None, "good"),
    "baseline_step": FactorSpec("baseline_step", "robust_z", None, "medium"),
    "flatline_ratio": FactorSpec("flatline_ratio", "bounded", None, "medium"),
    "sqi_basSQI": FactorSpec("sqi_basSQI", "bounded", None, "good"),
    "non_qrs_diff_p95": FactorSpec("non_qrs_diff_p95", "robust_z", None, "medium"),
    "non_qrs_rms_ratio": FactorSpec("non_qrs_rms_ratio", "bounded", None, "medium"),
    "qrs_band_ratio": FactorSpec("qrs_band_ratio", "log_scale", 10.0, "good"),
    "template_corr": FactorSpec("template_corr", "bounded", None, "good"),
    "amplitude_entropy": FactorSpec("amplitude_entropy", "bounded", None, "medium"),
    "contact_loss_win_ratio": FactorSpec("contact_loss_win_ratio", "bounded", None, "medium"),
    "band_0p3_1": FactorSpec("band_0p3_1", "bounded", None, "medium"),
}

HARD_GM_SUBTYPE_NAMES = {
    "good_overlap_boundary",
    "good_isolated_low_purity",
    "good_mild_artifact_outlier",
    "good_hard_baseline_lowqrs",
    "medium_overlap_boundary",
    "medium_isolated_lowqrs",
    "medium_visible_qrs_detail",
    "medium_outlier_or_bad_boundary",
    "medium_hard_baseline_lowqrs",
}
RELIABLE_GM_FACTOR_NAMES = {
    "baseline_step",
    "non_qrs_rms_ratio",
    "qrs_visibility",
    "sqi_basSQI",
    "qrs_band_ratio",
    "non_qrs_diff_p95",
}
RELIABLE_GM_LOCAL_STATS = {0, 5, 6, 7, 8}
FACTOR_ABLATION_FEATURES = [
    "template_corr",
    "detector_agreement",
    "contact_loss_win_ratio",
    "amplitude_entropy",
    "qrs_visibility",
    "baseline_step",
]


def _safe_logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(p) - torch.log1p(-p)


class MechanismFactorTransform:
    def __init__(self, columns: list[str], mean: np.ndarray, std: np.ndarray, train_encoded: np.ndarray):
        self.columns = list(columns)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.specs = [FACTOR_SPECS.get(c, FactorSpec(c, "robust_z")) for c in self.columns]
        var = np.nanvar(train_encoded, axis=0)
        self.boundary_active = {
            c: bool(np.isfinite(v) and v > 1.0e-6) for c, v in zip(self.columns, var)
        }

    def _encode_col(self, name: str, col: np.ndarray, j: int) -> np.ndarray:
        spec = FACTOR_SPECS.get(name, FactorSpec(name, "robust_z"))
        col = np.asarray(col, dtype=np.float32)
        if spec.kind == "bounded":
            return np.clip(col, 0.0, 1.0)
        if spec.kind == "scale":
            denom = float(spec.max_value or 1.0)
            return np.clip(col / denom, 0.0, 1.0)
        if spec.kind == "log_scale":
            max_value = float(spec.max_value or 10.0)
            return np.clip(np.log1p(np.maximum(col, 0.0)) / np.log1p(max_value), 0.0, 1.0)
        return (col - self.mean[j]) / self.std[j]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        out = np.zeros_like(raw, dtype=np.float32)
        for j, name in enumerate(self.columns):
            out[:, j] = self._encode_col(name, raw[:, j], j)
        return out

    def decode_target(self, encoded: np.ndarray) -> np.ndarray:
        encoded = np.asarray(encoded, dtype=np.float32)
        out = np.zeros_like(encoded, dtype=np.float32)
        for j, spec in enumerate(self.specs):
            x = encoded[:, j]
            if spec.kind == "bounded":
                out[:, j] = np.clip(x, 0.0, 1.0)
            elif spec.kind == "scale":
                out[:, j] = np.clip(x, 0.0, 1.0) * float(spec.max_value or 1.0)
            elif spec.kind == "log_scale":
                max_value = float(spec.max_value or 10.0)
                out[:, j] = np.expm1(np.clip(x, 0.0, 1.0) * np.log1p(max_value))
            else:
                out[:, j] = x * self.std[j] + self.mean[j]
        return out

    def pred_to_training_space(self, pred: torch.Tensor) -> torch.Tensor:
        cols: list[torch.Tensor] = []
        for j, spec in enumerate(self.specs):
            x = pred[:, j]
            if spec.kind in {"bounded", "scale", "log_scale"}:
                if spec.name == "detector_agreement":
                    cols.append(x.clamp(0.0, 1.0))
                else:
                    cols.append(torch.sigmoid(x))
            else:
                cols.append(x)
        return torch.stack(cols, dim=1)

    def pred_to_raw_tensor(self, pred: torch.Tensor) -> torch.Tensor:
        cols: list[torch.Tensor] = []
        for j, spec in enumerate(self.specs):
            if spec.kind in {"bounded", "scale", "log_scale"}:
                if spec.name == "detector_agreement":
                    p = pred[:, j].clamp(0.0, 1.0)
                else:
                    p = torch.sigmoid(pred[:, j])
                if spec.kind == "scale":
                    cols.append(p * float(spec.max_value or 1.0))
                elif spec.kind == "log_scale":
                    cols.append(torch.expm1(p * math.log1p(float(spec.max_value or 10.0))))
                else:
                    cols.append(p)
            else:
                mean = torch.as_tensor(self.mean[j], device=pred.device, dtype=pred.dtype)
                std = torch.as_tensor(self.std[j], device=pred.device, dtype=pred.dtype)
                cols.append(pred[:, j] * std + mean)
        return torch.stack(cols, dim=1)

    def target_to_raw_tensor(self, encoded: torch.Tensor) -> torch.Tensor:
        cols: list[torch.Tensor] = []
        for j, spec in enumerate(self.specs):
            x = encoded[:, j]
            if spec.kind == "bounded":
                cols.append(x.clamp(0.0, 1.0))
            elif spec.kind == "scale":
                cols.append(x.clamp(0.0, 1.0) * float(spec.max_value or 1.0))
            elif spec.kind == "log_scale":
                cols.append(torch.expm1(x.clamp(0.0, 1.0) * math.log1p(float(spec.max_value or 10.0))))
            else:
                mean = torch.as_tensor(self.mean[j], device=encoded.device, dtype=encoded.dtype)
                std = torch.as_tensor(self.std[j], device=encoded.device, dtype=encoded.dtype)
                cols.append(x * std + mean)
        return torch.stack(cols, dim=1)


def fit_factor_transform_suite(rows: pd.DataFrame, columns: list[str] | None = None):
    if ACTIVE_CFG.get("factor_contract") == "legacy":
        legacy_raw = EVT.numeric_frame(rows, EVT.FACTOR_COLUMNS)
        legacy_mean = legacy_raw.mean(axis=0).astype(np.float32)
        legacy_std = legacy_raw.std(axis=0).astype(np.float32)
        legacy_std = np.where(legacy_std > 1.0e-6, legacy_std, 1.0).astype(np.float32)
        return LegacyFactorTransform(columns=list(EVT.FACTOR_COLUMNS), mean=legacy_mean, std=legacy_std)
    columns = list(columns or EVT.FACTOR_COLUMNS)
    raw = EVT.numeric_frame(rows, columns).astype(np.float32)
    mean = np.nanmean(raw, axis=0).astype(np.float32)
    std = np.nanstd(raw, axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    tmp = MechanismFactorTransform(columns, mean, std, np.zeros_like(raw, dtype=np.float32))
    encoded = tmp.transform(raw)
    return MechanismFactorTransform(columns, mean, std, encoded)


def factor_loss_suite(pred: torch.Tensor, target: torch.Tensor, columns: list[str]) -> torch.Tensor:
    if not isinstance(ACTIVE_FACTOR_TRANSFORM, MechanismFactorTransform):
        losses = []
        for j, name in enumerate(columns):
            if name in EVT.RATIO_LIKE:
                if name == "detector_agreement":
                    val = F.binary_cross_entropy(
                        pred[:, j].clamp(1.0e-5, 1.0 - 1.0e-5),
                        target[:, j].clamp(0.0, 1.0),
                    )
                else:
                    val = F.binary_cross_entropy(
                        torch.sigmoid(pred[:, j]).clamp(1.0e-5, 1.0 - 1.0e-5),
                        target[:, j].clamp(0.0, 1.0),
                    )
            else:
                val = F.smooth_l1_loss(pred[:, j], target[:, j])
            losses.append(val)
        return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    for j, spec in enumerate(ACTIVE_FACTOR_TRANSFORM.specs):
        pj = pred[:, j]
        tj = target[:, j]
        if spec.kind in {"bounded", "scale", "log_scale"}:
            if spec.name == "detector_agreement":
                losses.append(F.smooth_l1_loss(pj.clamp(0.0, 1.0), tj, beta=0.05))
            else:
                losses.append(F.binary_cross_entropy_with_logits(pj, tj))
        else:
            losses.append(F.smooth_l1_loss(pj, tj, beta=0.35))
    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def factor_recovery_rows_suite(
    candidate: str, bucket: str, pred: np.ndarray, target: np.ndarray, y: np.ndarray
) -> list[dict[str, Any]]:
    if not isinstance(ACTIVE_FACTOR_TRANSFORM, MechanismFactorTransform):
        return ORIG_FACTOR_RECOVERY_ROWS(candidate, bucket, pred, target, y)
    pred_t = torch.as_tensor(pred, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    pred_raw = ACTIVE_FACTOR_TRANSFORM.pred_to_raw_tensor(pred_t).cpu().numpy()
    target_raw = ACTIVE_FACTOR_TRANSFORM.target_to_raw_tensor(target_t).cpu().numpy()
    rows: list[dict[str, Any]] = []
    for j, col in enumerate(EVT.FACTOR_COLUMNS):
        p = pred_raw[:, j]
        t = target_raw[:, j]
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() < 3:
            corr = np.nan
        else:
            corr = float(np.corrcoef(p[mask], t[mask])[0, 1])
        row = {
            "candidate": candidate,
            "bucket": bucket,
            "feature": col,
            "corr_all": corr,
            "mae": float(np.mean(np.abs(p[mask] - t[mask]))) if mask.any() else np.nan,
            "target_std": float(np.std(t[mask])) if mask.any() else np.nan,
            "contract": "decoded_mechanism",
        }
        vals = []
        for cls, idx in EVT.CLASS_TO_INT.items():
            cmask = mask & (y == idx)
            if cmask.sum() >= 3 and np.std(p[cmask]) > 1.0e-8 and np.std(t[cmask]) > 1.0e-8:
                ccorr = float(np.corrcoef(p[cmask], t[cmask])[0, 1])
            else:
                ccorr = np.nan
            row[f"corr_{cls}"] = ccorr
            if np.isfinite(ccorr):
                vals.append(ccorr)
        row["corr_min_supported_class"] = float(min(vals)) if vals else np.nan
        rows.append(row)
    return rows


def improved_detector_agreement(local_logits: torch.Tensor) -> torch.Tensor:
    a = torch.sigmoid(local_logits[:, 0])
    b = torch.sigmoid(local_logits[:, 1])
    inter = (a * b).sum(dim=1)
    mass_a = a.sum(dim=1)
    mass_b = b.sum(dim=1)
    dice = (2.0 * inter + 1.0e-6) / (mass_a + mass_b + 1.0e-6)
    min_mass = torch.minimum(mass_a, mass_b)
    activity = torch.sigmoid((min_mass - 4.0) / 3.0)
    return (dice * activity).clamp(0.0, 1.0)


def _topk_mean(x: torch.Tensor, k: int = 32) -> torch.Tensor:
    if x.shape[1] <= k:
        return x.mean(dim=1)
    return torch.topk(x, k=k, dim=1).values.mean(dim=1)


def _soft_longest_run(x: torch.Tensor, width: int = 64) -> torch.Tensor:
    if x.shape[1] < 2:
        return x.squeeze(1)
    width = max(2, min(width, x.shape[1]))
    pooled = F.max_pool1d(x.unsqueeze(1), kernel_size=width, stride=1).squeeze(1)
    return pooled.max(dim=1).values


def local_stats_from_logits(local_logits: torch.Tensor) -> torch.Tensor:
    qrs = torch.maximum(torch.sigmoid(local_logits[:, 0]), torch.sigmoid(local_logits[:, 1]))
    baseline = local_logits[:, 2].abs()
    contact = torch.sigmoid(local_logits[:, 3])
    reset = torch.sigmoid(local_logits[:, 4])
    flatline = torch.sigmoid(local_logits[:, 5])
    detail = torch.sigmoid(local_logits[:, 6])
    return torch.stack(
        [
            _topk_mean(qrs, 24),
            qrs.mean(dim=1),
            _topk_mean(contact, 24),
            _soft_longest_run(contact, 64),
            _topk_mean(flatline, 24),
            _soft_longest_run(flatline, 64),
            baseline.mean(dim=1),
            _topk_mean(baseline, 32),
            _topk_mean(detail, 32),
            _topk_mean(reset, 16),
        ],
        dim=1,
    )


def _cfg_factor_median(device: torch.device, dtype: torch.dtype, n: int) -> torch.Tensor:
    vals = ACTIVE_CFG.get("factor_train_median", None)
    if vals is None:
        return torch.zeros(n, device=device, dtype=dtype)
    arr = torch.as_tensor(vals, device=device, dtype=dtype)
    if arr.numel() != n:
        return torch.zeros(n, device=device, dtype=dtype)
    return arr


def final_gm_logit_from_out(out: dict[str, torch.Tensor]) -> torch.Tensor:
    logp = out["logits"].clamp_min(math.log(1.0e-6))
    return logp[:, 1] - logp[:, 0]


def final_bad_logit_from_out(out: dict[str, torch.Tensor]) -> torch.Tensor:
    logp = out["logits"].clamp_min(math.log(1.0e-6))
    nonbad = torch.logsumexp(logp[:, :2], dim=1)
    return logp[:, 2] - nonbad


def final_error_route_counts(rep: dict[str, Any]) -> dict[str, float]:
    raw = rep.get("confusion_3x3", np.zeros((3, 3)))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = np.zeros((3, 3))
    conf = np.asarray(raw, dtype=np.float64)
    if conf.shape != (3, 3):
        conf = np.zeros((3, 3), dtype=np.float64)
    med_total = float(conf[1].sum())
    return {
        "good_to_medium": float(conf[0, 1]),
        "good_to_bad": float(conf[0, 2]),
        "medium_to_good": float(conf[1, 0]),
        "medium_to_bad": float(conf[1, 2]),
        "bad_to_good": float(conf[2, 0]),
        "bad_to_medium": float(conf[2, 1]),
        "medium_to_bad_rate": float(conf[1, 2] / med_total) if med_total > 0 else np.nan,
    }


class GMMechanismConformer(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = int(self.query.shape[1])
        factor_dim = len(EVT.FACTOR_COLUMNS)
        stats_dim = 10
        fused_dim = width + factor_dim + stats_dim
        self.gm_direct_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.gm_factor_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, width),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(width, 1),
        )
        self.gm_expert_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, width),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(width, 3),
        )
        self.gm_fusion_gate = nn.Parameter(torch.tensor(-1.5))
        self.good_quality_indices = [
            i for i, cls in enumerate(EVT.QUALITY_SUBTYPE_CLASS) if int(cls) == EVT.CLASS_TO_INT["good"]
        ]
        self.medium_quality_indices = [
            i for i, cls in enumerate(EVT.QUALITY_SUBTYPE_CLASS) if int(cls) == EVT.CLASS_TO_INT["medium"]
        ]
        self.bad_quality_indices = [
            i for i, cls in enumerate(EVT.QUALITY_SUBTYPE_CLASS) if int(cls) == EVT.CLASS_TO_INT["bad"]
        ]
        self.good_cond_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, len(self.good_quality_indices))
        )
        self.medium_cond_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, len(self.medium_quality_indices))
        )
        self.bad_cond_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, len(self.bad_quality_indices))
        )
        self.boundary_label_heads = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1)) for _ in EVT.BOUNDARY_FAMILY_NAMES]
        )

    def _factor_view(self, factor_pred: torch.Tensor) -> torch.Tensor:
        if isinstance(ACTIVE_FACTOR_TRANSFORM, MechanismFactorTransform):
            return ACTIVE_FACTOR_TRANSFORM.pred_to_training_space(factor_pred)
        cols: list[torch.Tensor] = []
        for j, col in enumerate(EVT.FACTOR_COLUMNS):
            if col in EVT.RATIO_LIKE:
                if col == "detector_agreement":
                    cols.append(factor_pred[:, j].clamp(0.0, 1.0))
                else:
                    cols.append(torch.sigmoid(factor_pred[:, j]))
            else:
                cols.append(factor_pred[:, j])
        return torch.stack(cols, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = ACTIVE_CFG
        gm_mode = str(cfg.get("gm_mode", "legacy"))
        if gm_mode == "legacy":
            return super().forward(x)

        tok = self.forward_tokens(x)
        hi_ch = tok["hi_ch"]
        hi_tokens = tok["hi_tokens"]
        ctx = tok["context_tokens"]
        query = tok["query_tokens"]
        pooled = ctx.mean(dim=1)
        local_logits = self.local_heads(hi_ch)
        local_pool = hi_tokens.mean(dim=1)
        qrs_repr = query[:, 0, :]
        baseline_repr = query[:, 2, :]
        contact_repr = query[:, 3, :]
        detail_repr = query[:, 4, :]
        gm_repr = query[:, 6, :]
        bad_repr = query[:, 7, :]
        factor_pred = self.factor_head(query[:, :6, :].reshape(x.shape[0], -1))
        if "detector_agreement" in EVT.FACTOR_COLUMNS:
            det_idx = EVT.FACTOR_COLUMNS.index("detector_agreement")
            factor_pred = factor_pred.clone()
            factor_pred[:, det_idx] = improved_detector_agreement(local_logits)
        factor_view = self._factor_view(factor_pred)
        local_stats = local_stats_from_logits(local_logits)
        factor_view_for_gm = factor_view
        local_stats_for_gm = local_stats

        ablate_feature = str(cfg.get("factor_ablation_feature", "") or "")
        med = _cfg_factor_median(factor_view.device, factor_view.dtype, factor_view.shape[1])
        if ablate_feature in EVT.FACTOR_COLUMNS:
            j = EVT.FACTOR_COLUMNS.index(ablate_feature)
            factor_view_for_gm = factor_view_for_gm.clone()
            factor_view_for_gm[:, j] = med[j]

        if bool(cfg.get("reliable_factor_fusion", False)):
            reliable_idx = [
                j for j, name in enumerate(EVT.FACTOR_COLUMNS) if name in RELIABLE_GM_FACTOR_NAMES
            ]
            keep = torch.zeros(factor_view.shape[1], device=factor_view.device, dtype=torch.bool)
            if reliable_idx:
                keep[torch.as_tensor(reliable_idx, device=factor_view.device, dtype=torch.long)] = True
            factor_view_for_gm = factor_view_for_gm.clone()
            factor_view_for_gm[:, ~keep] = med[~keep].unsqueeze(0)

            local_keep = torch.zeros(local_stats.shape[1], device=local_stats.device, dtype=torch.bool)
            local_idx = [i for i in RELIABLE_GM_LOCAL_STATS if i < local_stats.shape[1]]
            if local_idx:
                local_keep[torch.as_tensor(local_idx, device=local_stats.device, dtype=torch.long)] = True
            local_stats_for_gm = local_stats_for_gm.clone()
            local_stats_for_gm[:, ~local_keep] = 0.0

        if bool(cfg.get("detach_gm_evidence", False)):
            factor_view_for_gm = factor_view_for_gm.detach()
            local_stats_for_gm = local_stats_for_gm.detach()

        bad_logit = self.bad_head(bad_repr).squeeze(1)
        gm_direct = self.medium_head(gm_repr).squeeze(1)
        gm_input = torch.cat([gm_repr, factor_view_for_gm, local_stats_for_gm], dim=1)
        gm_factor_logit = self.gm_factor_head(gm_input).squeeze(1)

        gm_expert_logits = None
        if gm_mode in {"family_moe", "pairrank", "beat_background"}:
            expert_logits = self.gm_expert_head(gm_input) + 0.15 * gm_direct.unsqueeze(1)
            expert_probs = torch.sigmoid(expert_logits)
            medium_prob = 1.0 - torch.prod((1.0 - expert_probs).clamp(1.0e-5, 1.0), dim=1)
            medium_logit = _safe_logit(medium_prob)
            gm_expert_logits = expert_logits
        elif gm_mode == "factor_fused":
            if bool(cfg.get("use_gated_factor_residual", False)):
                medium_logit = gm_direct + torch.sigmoid(self.gm_fusion_gate) * gm_factor_logit
            else:
                medium_logit = gm_direct + float(cfg.get("gm_factor_gain", 1.0)) * gm_factor_logit
        else:
            medium_logit = gm_direct

        bad_prob = torch.sigmoid(bad_logit)
        medium_given_nonbad = torch.sigmoid(medium_logit)
        probs = torch.stack(
            [
                (1.0 - bad_prob) * (1.0 - medium_given_nonbad),
                (1.0 - bad_prob) * medium_given_nonbad,
                bad_prob,
            ],
            dim=1,
        ).clamp(1.0e-6, 1.0 - 1.0e-6)
        logits = torch.log(probs)

        quality_repr = torch.cat([gm_repr, bad_repr], dim=1)
        conditional = bool(cfg.get("conditional_subtype", False))
        if conditional:
            good_log = torch.log(probs[:, 0].clamp_min(1.0e-6)).unsqueeze(1)
            med_log = torch.log(probs[:, 1].clamp_min(1.0e-6)).unsqueeze(1)
            bad_log = torch.log(probs[:, 2].clamp_min(1.0e-6)).unsqueeze(1)
            quality_subtype_logits = factor_pred.new_full(
                (x.shape[0], len(EVT.QUALITY_SUBTYPE_NAMES)), -30.0
            )
            good_idx = torch.as_tensor(self.good_quality_indices, device=x.device, dtype=torch.long)
            med_idx = torch.as_tensor(self.medium_quality_indices, device=x.device, dtype=torch.long)
            bad_idx = torch.as_tensor(self.bad_quality_indices, device=x.device, dtype=torch.long)
            quality_subtype_logits[:, good_idx] = good_log + F.log_softmax(self.good_cond_head(gm_repr), dim=1)
            quality_subtype_logits[:, med_idx] = med_log + F.log_softmax(self.medium_cond_head(gm_repr), dim=1)
            quality_subtype_logits[:, bad_idx] = bad_log + F.log_softmax(self.bad_cond_head(bad_repr), dim=1)
        else:
            quality_subtype_logits = self.quality_subtype_head(quality_repr)

        subtype_alpha = float(cfg.get("subtype_class_fusion_alpha", 0.0))
        if subtype_alpha > 0.0:
            subtype_probs = subtype_class_probs(quality_subtype_logits)
            alpha = max(0.0, min(subtype_alpha, 0.85))
            fused_log = (1.0 - alpha) * torch.log(probs.clamp_min(1.0e-6)) + alpha * torch.log(
                subtype_probs.clamp_min(1.0e-6)
            )
            probs = F.softmax(fused_log, dim=1).clamp(1.0e-6, 1.0 - 1.0e-6)
            logits = torch.log(probs)

        boundary_label_logits = torch.cat([h(gm_repr) for h in self.boundary_label_heads], dim=1)
        artifact_presence_logit = self.artifact_head(bad_repr).squeeze(1)
        artifact_type_logits = self.artifact_type_head(bad_repr)
        artifact_severity = torch.sigmoid(self.severity_head(bad_repr).squeeze(1))

        out = {
            "logits": logits,
            "probs": probs,
            "bad_logit": bad_logit,
            "medium_logit": medium_logit,
            "gm_final_logit": final_gm_logit_from_out({"logits": logits}),
            "bad_final_logit": final_bad_logit_from_out({"logits": logits}),
            "local_logits": local_logits,
            "factor_pred": factor_pred,
            "decoded_factor_pred": factor_view,
            "gm_factor_view": factor_view_for_gm,
            "local_stats": local_stats,
            "detector_agreement": factor_pred[:, EVT.DETECTOR_FACTOR_IDX],
            "artifact_presence_logit": artifact_presence_logit,
            "artifact_logit": artifact_presence_logit,
            "artifact_type_logits": artifact_type_logits,
            "artifact_severity": artifact_severity,
            "quality_subtype_logits": quality_subtype_logits,
            "gm_subtype_logits": self.gm_subtype_head(gm_repr),
            "bad_subtype_logits": self.bad_subtype_head(bad_repr),
            "boundary_family_logits": self.boundary_family_head(gm_repr),
            "boundary_label_logit": self.boundary_label_head(gm_repr).squeeze(1),
            "boundary_label_logits_by_family": boundary_label_logits,
        }
        if gm_expert_logits is not None:
            out["gm_expert_logits"] = gm_expert_logits
        return out


def conditional_subtype_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    device = out["logits"].device
    target = batch["quality_subtype"].to(device)
    mask = target >= 0
    if not mask.any():
        leaf = out["logits"].new_tensor(0.0)
    else:
        qlog = out["quality_subtype_logits"]
        if ACTIVE_CFG.get("conditional_subtype", False):
            leaf = F.nll_loss(qlog[mask], target[mask])
        else:
            leaf = F.cross_entropy(qlog[mask], target[mask])

    fam = batch["boundary_family"].to(device)
    lab = batch["boundary_label"].float().to(device)
    fam_mask = fam >= 0
    if fam_mask.any():
        fam_loss = F.cross_entropy(out["boundary_family_logits"][fam_mask], fam[fam_mask])
        if "boundary_label_logits_by_family" in out:
            logits_by_family = out["boundary_label_logits_by_family"][fam_mask]
            fam_idx = fam[fam_mask].clamp_min(0)
            label_logit = logits_by_family.gather(1, fam_idx.unsqueeze(1)).squeeze(1)
        else:
            label_logit = out["boundary_label_logit"][fam_mask]
        label_loss = F.binary_cross_entropy_with_logits(label_logit, lab[fam_mask])
    else:
        fam_loss = out["logits"].new_tensor(0.0)
        label_loss = out["logits"].new_tensor(0.0)
    return leaf + 0.35 * fam_loss + 0.55 * label_loss


def pairrank_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if not ACTIVE_CFG.get("use_pairrank", False):
        return out["logits"].new_tensor(0.0)
    y = batch["y"].to(out["logits"].device)
    pair = batch.get("pair_id")
    if pair is None:
        return out["logits"].new_tensor(0.0)
    pair_np = pair.detach().cpu().numpy()
    losses: list[torch.Tensor] = []
    scores = final_gm_logit_from_out(out) if ACTIVE_CFG.get("pairrank_use_final", False) else out["medium_logit"]
    for pid in np.unique(pair_np):
        idx_np = np.where(pair_np == pid)[0]
        if len(idx_np) < 2:
            continue
        idx = torch.as_tensor(idx_np, device=scores.device, dtype=torch.long)
        yy = y[idx]
        good_idx = idx[yy == 0]
        med_idx = idx[yy == 1]
        if good_idx.numel() and med_idx.numel():
            diff = scores[med_idx].mean() - scores[good_idx].mean()
            losses.append(F.softplus(torch.as_tensor(0.25, device=scores.device) - diff))
    if not losses:
        return out["logits"].new_tensor(0.0)
    return torch.stack(losses).mean()


def hard_gm_mask(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    y = batch["y"].to(device)
    mask = torch.zeros_like(y, dtype=torch.bool)
    fam = batch.get("boundary_family")
    if fam is not None:
        mask |= fam.to(device).long().ge(0)
    sub = batch.get("quality_subtype")
    if sub is not None:
        hard_idx = [
            i for i, name in enumerate(EVT.QUALITY_SUBTYPE_NAMES) if name in HARD_GM_SUBTYPE_NAMES
        ]
        if hard_idx:
            hard = torch.as_tensor(hard_idx, device=device, dtype=torch.long)
            sub_t = sub.to(device).long()
            mask |= (sub_t.unsqueeze(1) == hard.unsqueeze(0)).any(dim=1)
    return mask & (y != EVT.CLASS_TO_INT["bad"])


def hard_gm_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if float(ACTIVE_CFG.get("hard_gm_weight", 0.0)) <= 0.0:
        return out["logits"].new_tensor(0.0)
    y = batch["y"].to(out["logits"].device)
    mask = hard_gm_mask(batch, out["logits"].device)
    if not bool(mask.any()):
        return out["logits"].new_tensor(0.0)
    target = y[mask].eq(EVT.CLASS_TO_INT["medium"]).float()
    return F.binary_cross_entropy_with_logits(final_gm_logit_from_out(out)[mask], target)


def medium_bad_guard_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if float(ACTIVE_CFG.get("medium_bad_guard_weight", 0.0)) <= 0.0:
        return out["logits"].new_tensor(0.0)
    y = batch["y"].to(out["logits"].device)
    mask = y.eq(EVT.CLASS_TO_INT["medium"])
    if not bool(mask.any()):
        return out["logits"].new_tensor(0.0)
    bad_margin = float(ACTIVE_CFG.get("bad_margin", -0.5))
    return F.softplus(final_bad_logit_from_out(out)[mask] + bad_margin).mean()


def subtype_class_probs(qlogits: torch.Tensor) -> torch.Tensor:
    leaf_prob = F.softmax(qlogits, dim=1)
    class_ids = torch.as_tensor(
        EVT.QUALITY_SUBTYPE_CLASS, device=qlogits.device, dtype=torch.long
    )
    n_class = int(max(EVT.CLASS_TO_INT.values())) + 1
    out = qlogits.new_zeros((qlogits.shape[0], n_class))
    out.scatter_add_(1, class_ids.unsqueeze(0).expand(qlogits.shape[0], -1), leaf_prob)
    return out.clamp(1.0e-6, 1.0)


def subtype_class_consistency_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    weight = float(ACTIVE_CFG.get("subtype_class_consistency_weight", 0.0))
    if weight <= 0.0 or "quality_subtype_logits" not in out:
        return out["logits"].new_tensor(0.0)
    main = out["probs"].clamp(1.0e-6, 1.0)
    sub = subtype_class_probs(out["quality_subtype_logits"])
    # Symmetric enough for calibration, but numerically mild: subtype should not
    # become a second unconstrained classifier that disagrees with hierarchy.
    return 0.5 * (
        F.kl_div(main.log(), sub.detach(), reduction="batchmean")
        + F.kl_div(sub.log(), main.detach(), reduction="batchmean")
    )


def class_loss_suite(
    out: dict[str, torch.Tensor],
    y: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    mode = str(cfg.get("class_loss_mode", "nll"))
    logp = out["logits"]
    weights = EVT.class_weight_tensor(cfg, device)
    if mode == "focal":
        nll = F.nll_loss(logp, y, reduction="none")
        pt = logp.gather(1, y.view(-1, 1)).squeeze(1).exp().clamp(1.0e-6, 1.0)
        gamma = float(cfg.get("focal_gamma", 1.5))
        loss = ((1.0 - pt) ** gamma) * nll
        if weights is not None:
            loss = loss * weights[y]
        return loss.mean()
    smoothing = float(cfg.get("label_smoothing", 0.0))
    if mode == "smooth" or smoothing > 0.0:
        smoothing = min(max(smoothing, 0.0), 0.2)
        n_class = int(logp.shape[1])
        target = torch.full_like(logp, smoothing / max(n_class - 1, 1))
        target.scatter_(1, y.view(-1, 1), 1.0 - smoothing)
        loss = -(target * logp).sum(dim=1)
        if weights is not None:
            loss = loss * weights[y]
        return loss.mean()
    return EVT.class_nll(out, y, cfg, device)


def model_loss_suite(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    out = model(x)
    y = batch["y"].to(device)
    loss_cls = class_loss_suite(out, y, cfg, device)
    loss_factor = factor_loss_suite(out["factor_pred"], batch["factor"].to(device), EVT.FACTOR_COLUMNS)
    loss_local, local_parts = EVT.local_map_loss(out, x)
    loss_art, art_parts = EVT.artifact_loss(out, batch, device)
    loss_sub = conditional_subtype_loss(out, batch)
    loss_pair = pairrank_loss(out, batch)
    loss_subcls = subtype_class_consistency_loss(out)
    loss_hard_gm = hard_gm_loss(out, batch)
    loss_medium_bad_guard = medium_bad_guard_loss(out, batch)

    total = (
        float(cfg.get("cls_weight", 1.0)) * loss_cls
        + float(cfg.get("factor_weight", 0.18)) * loss_factor
        + float(cfg.get("local_weight", 0.14)) * loss_local
        + float(cfg.get("artifact_weight", 0.18)) * loss_art
        + float(cfg.get("subtype_weight", 0.06)) * loss_sub
        + float(cfg.get("pairrank_weight", 0.0)) * loss_pair
        + float(cfg.get("subtype_class_consistency_weight", 0.0)) * loss_subcls
        + float(cfg.get("hard_gm_weight", 0.0)) * loss_hard_gm
        + float(cfg.get("medium_bad_guard_weight", 0.0)) * loss_medium_bad_guard
    )
    parts = {
        "loss_cls": float(loss_cls.detach().cpu()),
        "loss_factor": float(loss_factor.detach().cpu()),
        "loss_local": float(loss_local.detach().cpu()),
        "loss_artifact": float(loss_art.detach().cpu()),
        "loss_subtype": float(loss_sub.detach().cpu()),
        "loss_pairrank": float(loss_pair.detach().cpu()),
        "loss_subtype_class_consistency": float(loss_subcls.detach().cpu()),
        "loss_hard_gm": float(loss_hard_gm.detach().cpu()),
        "loss_medium_bad_guard": float(loss_medium_bad_guard.detach().cpu()),
        "class_loss_mode": str(cfg.get("class_loss_mode", "nll")),
    }
    parts.update(local_parts)
    parts.update(art_parts)
    return total, parts, out


def suite_checkpoint_score(rep: dict[str, Any], cfg: dict[str, Any]) -> float:
    if str(cfg.get("suite_checkpoint_score", "")) != "gm_guarded":
        return float(rep["record_macro_supported_f1"]) + 0.20 * min(
            float(rep["good_recall"]), float(rep["medium_recall"]), float(rep["bad_recall"])
        ) - 0.05 * float(rep["bad_fpr_nonbad"])
    routes = final_error_route_counts(rep)
    bad_recall = float(rep["bad_recall"])
    if bad_recall < float(cfg.get("checkpoint_bad_guard", 0.965)):
        return -1.0e6 + bad_recall
    good = float(rep["good_recall"])
    medium = float(rep["medium_recall"])
    macro_f1 = float(rep.get("macro_f1", rep.get("macro_f1_sklearn", 0.0)))
    gm_balanced = 0.5 * (good + medium)
    medium_to_bad_rate = float(routes.get("medium_to_bad_rate", 0.0))
    return (
        0.45 * macro_f1
        + 0.35 * gm_balanced
        + 0.20 * min(good, medium)
        - 0.10 * abs(good - medium)
        - 0.05 * medium_to_bad_rate
    )


def decode_factor_arrays(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(ACTIVE_FACTOR_TRANSFORM, MechanismFactorTransform):
        return pred, target
    pred_t = torch.as_tensor(pred, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    pred_raw = ACTIVE_FACTOR_TRANSFORM.pred_to_raw_tensor(pred_t).cpu().numpy()
    target_raw = ACTIVE_FACTOR_TRANSFORM.target_to_raw_tensor(target_t).cpu().numpy()
    return pred_raw, target_raw


def error_conditioned_factor_residual_rows(
    candidate: str,
    bucket: str,
    probs: np.ndarray,
    pred_factor: np.ndarray,
    true_factor: np.ndarray,
    y: np.ndarray,
) -> list[dict[str, Any]]:
    pred_cls = np.argmax(probs, axis=1)
    pred_raw, true_raw = decode_factor_arrays(pred_factor, true_factor)
    groups = {
        "correct_medium": (y == EVT.CLASS_TO_INT["medium"]) & (pred_cls == EVT.CLASS_TO_INT["medium"]),
        "medium_to_good": (y == EVT.CLASS_TO_INT["medium"]) & (pred_cls == EVT.CLASS_TO_INT["good"]),
        "medium_to_bad": (y == EVT.CLASS_TO_INT["medium"]) & (pred_cls == EVT.CLASS_TO_INT["bad"]),
    }
    rows: list[dict[str, Any]] = []
    for group, mask in groups.items():
        for j, feature in enumerate(EVT.FACTOR_COLUMNS):
            if not bool(np.any(mask)):
                rows.append(
                    {
                        "candidate": candidate,
                        "bucket": bucket,
                        "group": group,
                        "feature": feature,
                        "n": 0,
                        "pred_mean": np.nan,
                        "true_mean": np.nan,
                        "signed_error_mean": np.nan,
                        "abs_error_mean": np.nan,
                    }
                )
                continue
            err = pred_raw[mask, j] - true_raw[mask, j]
            rows.append(
                {
                    "candidate": candidate,
                    "bucket": bucket,
                    "group": group,
                    "feature": feature,
                    "n": int(np.sum(mask)),
                    "pred_mean": float(np.nanmean(pred_raw[mask, j])),
                    "true_mean": float(np.nanmean(true_raw[mask, j])),
                    "signed_error_mean": float(np.nanmean(err)),
                    "abs_error_mean": float(np.nanmean(np.abs(err))),
                }
            )
    return rows


def error_route_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    row.update(final_error_route_counts(rep))
    return row


def train_model_suite(
    candidate: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    split_root: Path | None = None,
    fold: int = 0,
    seed_index: int = 0,
):
    global ACTIVE_CFG
    ACTIVE_CFG = dict(cfg)
    if str(cfg.get("suite_checkpoint_score", "")) != "gm_guarded":
        return ORIG_TRAIN_MODEL(candidate, cfg, args, split_root, fold, seed_index)

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, val_loader, test_loader, train_ds = EVT.make_loaders(args, split_root, fold)
    train_eval_loader = EVT.DataLoader(
        train_ds,
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )
    model = EVT.EventFactorizedSQIConformer(
        in_ch=int(train_ds.x.shape[1]),
        factor_dim=len(EVT.FACTOR_COLUMNS),
        width=int(cfg.get("width", 96)),
        layers=int(cfg.get("layers", 3)),
        heads=int(cfg.get("heads", 4)),
        use_queries=bool(cfg.get("use_queries", True)),
        use_highres_fusion=bool(cfg.get("use_highres_fusion", True)),
        use_local_supervision=bool(cfg.get("use_local_supervision", True)),
        use_artifact_aux=bool(cfg.get("use_artifact_aux", True)),
        use_hierarchical_head=bool(cfg.get("use_hierarchical_head", True)),
        branch_upper_block=bool(cfg.get("branch_upper_block", False)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    pretrain_logs = EVT.pretrain_model(model, train_loader, cfg, device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0e18
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_parts: list[dict[str, float]] = []
        losses: list[float] = []
        for batch in train_loader:
            total_loss, parts, _ = EVT.model_loss(model, batch, cfg, device)
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(total_loss.detach().cpu()))
            epoch_parts.append(parts)
        train_rep, _, _, _, _, _, _, _, _ = EVT.eval_loader(model, train_eval_loader, device, candidate)
        val_rep, _, _, _, _, _, _, _, _ = EVT.eval_loader(model, val_loader, device, candidate)
        score = suite_checkpoint_score(val_rep, cfg)
        routes = final_error_route_counts(val_rep)
        mean_parts = pd.DataFrame(epoch_parts).mean(numeric_only=True).to_dict() if epoch_parts else {}
        log_row = {
            "phase": "finetune",
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            "suite_checkpoint_score": float(score),
            **{f"train_loss_part_{k}": v for k, v in mean_parts.items()},
            **{f"train_{k}": v for k, v in train_rep.items() if k != "confusion_3x3"},
            **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"},
            **{f"val_{k}": v for k, v in routes.items()},
        }
        logs.append(log_row)
        print(
            f"{candidate} fold={fold} seed={seed_index} epoch={epoch}/{args.epochs} "
            f"loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} score={score:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} "
            f"m2g={routes['medium_to_good']:.0f} m2b={routes['medium_to_bad']:.0f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    train_rep, train_probs, train_factor, train_true_factor, train_y, train_artifact, train_rec_rows, _, train_extra = EVT.eval_loader(
        model, train_eval_loader, device, candidate
    )
    val_rep, val_probs, val_factor, val_true_factor, val_y, val_artifact, val_rec_rows, _, val_extra = EVT.eval_loader(
        model, val_loader, device, candidate
    )
    test_rep, test_probs, test_factor, test_true_factor, test_y, test_artifact, test_rec_rows, _, test_extra = EVT.eval_loader(
        model, test_loader, device, candidate
    )
    run_name = f"{candidate}_fold{fold}_seed{seed_index}"
    run_path = EVT.RUN_DIR / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "factor_columns": EVT.FACTOR_COLUMNS,
            "local_map_names": EVT.LOCAL_MAP_NAMES,
            "quality_subtype_names": EVT.QUALITY_SUBTYPE_NAMES,
            "quality_subtype_class": EVT.QUALITY_SUBTYPE_CLASS,
            "boundary_family_names": EVT.BOUNDARY_FAMILY_NAMES,
            "input_contract": "waveform-derived channels only; factors are teacher targets only",
            "model_kind": "GMMechanismConformer",
            "no_warm_start": True,
            "best_score": float(best_score),
            "checkpoint_score": str(cfg.get("suite_checkpoint_score")),
        },
        run_path / "ckpt_best.pt",
    )
    pd.DataFrame(pretrain_logs + logs).to_csv(run_path / "train_log.csv", index=False)
    np.savez(
        run_path / "val_predictions.npz",
        probs=val_probs,
        factor_pred=val_factor,
        factor_true=val_true_factor,
        y=val_y,
        artifact_presence=val_artifact,
        **{f"val_{k}": v for k, v in val_extra.items()},
    )
    np.savez(
        run_path / "test_predictions.npz",
        probs=test_probs,
        factor_pred=test_factor,
        factor_true=test_true_factor,
        y=test_y,
        artifact_presence=test_artifact,
        **{f"test_{k}": v for k, v in test_extra.items()},
    )
    metrics = [
        EVT.metric_row(candidate, "clean_train", train_rep),
        EVT.metric_row(candidate, "clean_val", val_rep),
        EVT.metric_row(candidate, "clean_test", test_rep),
    ]
    recovery = EVT.factor_recovery_rows(candidate, "clean_train", train_factor, train_true_factor, train_y)
    recovery += EVT.factor_recovery_rows(candidate, "clean_val", val_factor, val_true_factor, val_y)
    recovery += EVT.factor_recovery_rows(candidate, "clean_test", test_factor, test_true_factor, test_y)
    error_routes = [
        error_route_row(candidate, "clean_train", train_rep),
        error_route_row(candidate, "clean_val", val_rep),
        error_route_row(candidate, "clean_test", test_rep),
    ]
    error_residual = error_conditioned_factor_residual_rows(
        candidate, "clean_train", train_probs, train_factor, train_true_factor, train_y
    )
    error_residual += error_conditioned_factor_residual_rows(
        candidate, "clean_val", val_probs, val_factor, val_true_factor, val_y
    )
    error_residual += error_conditioned_factor_residual_rows(
        candidate, "clean_test", test_probs, test_factor, test_true_factor, test_y
    )
    return {
        "candidate": candidate,
        "fold": fold,
        "seed": seed,
        "seed_index": seed_index,
        "run_path": str(run_path),
        "metrics": metrics,
        "record_metrics": train_rec_rows + val_rec_rows + test_rec_rows,
        "recovery": recovery,
        "error_routes": error_routes,
        "error_residual": error_residual,
    }


def make_loaders_suite(*args, **kwargs):
    global ACTIVE_FACTOR_TRANSFORM
    loaders = ORIG_MAKE_LOADERS(*args, **kwargs)
    ACTIVE_FACTOR_TRANSFORM = loaders[0].dataset.factor_transform
    train_factor = getattr(loaders[0].dataset, "factor", None)
    if train_factor is not None:
        try:
            ACTIVE_CFG["factor_train_median"] = np.nanmedian(
                np.asarray(train_factor, dtype=np.float32), axis=0
            ).astype(float).tolist()
        except Exception:
            ACTIVE_CFG.pop("factor_train_median", None)
    return loaders


ORIG_MAKE_LOADERS = EVT.make_loaders


def install_patches() -> None:
    EVT.OUT_DIR = SUITE_OUT_DIR
    EVT.REPORT_OUT_DIR = SUITE_REPORT_DIR
    EVT.RUN_DIR = SUITE_RUN_DIR
    EVT.ensure_dirs()
    EVT.FactorTransform = MechanismFactorTransform
    EVT.fit_factor_transform = fit_factor_transform_suite
    EVT.factor_loss = factor_loss_suite
    EVT.factor_recovery_rows = factor_recovery_rows_suite
    EVT.EventFactorizedSQIConformer = GMMechanismConformer
    EVT.model_loss = model_loss_suite
    EVT.train_model = train_model_suite
    EVT.make_loaders = make_loaders_suite


def suite_candidates() -> list[tuple[str, dict[str, Any]]]:
    common = {
        "width": 96,
        "layers": 3,
        "heads": 4,
        "dropout": 0.12,
        "lr": 1.5e-4,
        "weight_decay": 2.0e-4,
        "cls_weight": 1.0,
        "factor_weight": 0.18,
        "local_weight": 0.14,
        "artifact_weight": 0.18,
        "subtype_weight": 0.055,
        "class_weights": [1.0, 1.08, 1.08],
    }
    return [
        (
            "E4_v112_lowaux_lr15e4",
            {
                **common,
                "factor_contract": "legacy",
                "gm_mode": "legacy",
                "factor_weight": 0.14,
                "local_weight": 0.10,
                "artifact_weight": 0.12,
                "subtype_weight": 0.04,
            },
        ),
        (
            "E5_factor_contract_only",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "direct",
                "subtype_weight": 0.04,
            },
        ),
        (
            "E6_factor_fused_gm",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "factor_weight": 0.22,
                "subtype_weight": 0.05,
            },
        ),
        (
            "E7_family_moe_condsubtype",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "family_moe",
                "conditional_subtype": True,
                "factor_weight": 0.22,
                "subtype_weight": 0.085,
            },
        ),
        (
            "E8_pairrank_hardsampler",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "pairrank",
                "conditional_subtype": True,
                "use_pairrank": True,
                "pairrank_weight": 0.16,
                "factor_weight": 0.22,
                "subtype_weight": 0.085,
            },
        ),
        (
            "E9_beat_background_tokens",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "beat_background",
                "conditional_subtype": True,
                "use_pairrank": True,
                "pairrank_weight": 0.16,
                "factor_weight": 0.24,
                "subtype_weight": 0.09,
            },
        ),
        (
            "E10_e6_medguard",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "factor_weight": 0.22,
                "subtype_weight": 0.05,
                "class_weights": [1.0, 1.18, 1.08],
            },
        ),
        (
            "E11_e6_lowgain",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "gm_factor_gain": 0.55,
                "factor_weight": 0.22,
                "subtype_weight": 0.05,
            },
        ),
        (
            "E12_e6_pairrank_only",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_weight": 0.08,
                "factor_weight": 0.22,
                "subtype_weight": 0.05,
            },
        ),
        (
            "E13_e6_boundary_aux_stronger",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "factor_weight": 0.22,
                "subtype_weight": 0.08,
            },
        ),
        (
            "E14_e6_medguard_lowgain",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "gm_factor_gain": 0.65,
                "factor_weight": 0.22,
                "subtype_weight": 0.05,
                "class_weights": [1.0, 1.16, 1.08],
            },
        ),
        (
            "E15_e6_classheavy_lowaux",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "cls_weight": 1.40,
                "factor_weight": 0.14,
                "local_weight": 0.05,
                "artifact_weight": 0.10,
                "subtype_weight": 0.02,
            },
        ),
        (
            "E16_e14_classheavy_lowaux",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "gm_factor_gain": 0.65,
                "cls_weight": 1.40,
                "factor_weight": 0.14,
                "local_weight": 0.05,
                "artifact_weight": 0.10,
                "subtype_weight": 0.02,
                "class_weights": [1.0, 1.16, 1.08],
            },
        ),
        (
            "E17_e6_focal_gm",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "class_loss_mode": "focal",
                "focal_gamma": 1.5,
                "cls_weight": 1.20,
                "factor_weight": 0.18,
                "local_weight": 0.08,
                "artifact_weight": 0.12,
                "subtype_weight": 0.035,
                "class_weights": [1.0, 1.12, 1.06],
            },
        ),
        (
            "E18_e6_smooth_ce",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "class_loss_mode": "smooth",
                "label_smoothing": 0.04,
                "cls_weight": 1.30,
                "factor_weight": 0.16,
                "local_weight": 0.08,
                "artifact_weight": 0.12,
                "subtype_weight": 0.035,
            },
        ),
        (
            "E19_e6_nolocal_classfocus",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "cls_weight": 1.55,
                "factor_weight": 0.10,
                "local_weight": 0.0,
                "artifact_weight": 0.08,
                "subtype_weight": 0.02,
            },
        ),
        (
            "E20_e6_lowlr_balanced",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "lr": 1.0e-4,
                "dropout": 0.10,
                "cls_weight": 1.20,
                "factor_weight": 0.18,
                "local_weight": 0.08,
                "artifact_weight": 0.12,
                "subtype_weight": 0.04,
                "class_weights": [1.0, 1.12, 1.08],
            },
        ),
        (
            "E21_e6_subcls_consistency",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.08,
            },
        ),
        (
            "E22_e6_subtype_class_fusion",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.25,
            },
        ),
        (
            "E23_e14_subtype_class_fusion",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "gm_factor_gain": 0.65,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.25,
                "class_weights": [1.0, 1.16, 1.08],
            },
        ),
        (
            "E24_e6_subtype_fusion_pairrank",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_weight": 0.08,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.20,
            },
        ),
        (
            "E31_wave_mechanism_conformer",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "direct",
                "factor_weight": 0.16,
                "local_weight": 0.14,
                "artifact_weight": 0.14,
                "subtype_weight": 0.02,
                "subtype_class_consistency_weight": 0.0,
                "subtype_class_fusion_alpha": 0.0,
                "use_pairrank": False,
                "pairrank_weight": 0.0,
                "hard_gm_weight": 0.0,
                "medium_bad_guard_weight": 0.0,
                "lr": 1.5e-4,
                "class_weights": [1.0, 1.08, 1.08],
            },
        ),
        (
            "E25_e24_lowalpha",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_weight": 0.08,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.14,
            },
        ),
        (
            "E24_e29e30_baseline",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_weight": 0.08,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.20,
            },
        ),
        (
            "E26_e24_lowlr",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_weight": 0.08,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.20,
                "lr": 1.0e-4,
            },
        ),
        (
            "E27_e24_lowalpha_lowlr",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_weight": 0.08,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.14,
                "lr": 1.0e-4,
            },
        ),
        (
            "E28_e24_softpair_lowalpha",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_weight": 0.04,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.16,
            },
        ),
        (
            "E29_e24_finalscore_pairrank_mediumguard",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_use_final": True,
                "pairrank_weight": 0.08,
                "medium_bad_guard_weight": 0.05,
                "bad_margin": -0.5,
                "hard_gm_weight": 0.10,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.20,
                "suite_checkpoint_score": "gm_guarded",
                "checkpoint_bad_guard": 0.965,
            },
        ),
        (
            "E30_e29_reliable_detached_factor_fusion",
            {
                **common,
                "factor_contract": "mechanism",
                "gm_mode": "factor_fused",
                "use_pairrank": True,
                "pairrank_use_final": True,
                "pairrank_weight": 0.08,
                "medium_bad_guard_weight": 0.05,
                "bad_margin": -0.5,
                "hard_gm_weight": 0.10,
                "factor_weight": 0.22,
                "subtype_weight": 0.075,
                "subtype_class_consistency_weight": 0.06,
                "subtype_class_fusion_alpha": 0.20,
                "suite_checkpoint_score": "gm_guarded",
                "checkpoint_bad_guard": 0.965,
                "reliable_factor_fusion": True,
                "detach_gm_evidence": True,
                "use_gated_factor_residual": True,
            },
        ),
    ]


def write_factor_contract_report(split_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for fold in range(args.folds):
        _, atlas = EVT.load_recordheldout_frame(split_root, fold)
        train_rows = atlas.loc[atlas["split"].astype(str).eq("train")].copy()
        cfg = {"factor_contract": "mechanism", "gm_mode": "direct"}
        global ACTIVE_CFG
        ACTIVE_CFG = cfg
        transform = fit_factor_transform_suite(train_rows, EVT.FACTOR_COLUMNS)
        encoded = transform.transform(train_rows[EVT.FACTOR_COLUMNS].to_numpy(dtype=np.float32))
        for j, col in enumerate(EVT.FACTOR_COLUMNS):
            rows.append(
                {
                    "fold": fold,
                    "feature": col,
                    "encoded_std": float(np.std(encoded[:, j])),
                    "encoded_var": float(np.var(encoded[:, j])),
                    "boundary_active": bool(transform.boundary_active.get(col, True)),
                    "contract": FACTOR_SPECS.get(col, FactorSpec(col, "robust_z")).kind,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(SUITE_OUT_DIR / "factor_target_variance.csv", index=False)
    return df


def pairrank_audit(split_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for fold in range(args.folds):
        _, atlas = EVT.load_recordheldout_frame(split_root, fold)
        train_rows = atlas.loc[atlas["split"].astype(str).eq("train")].copy()
        if "source_idx" in train_rows.columns:
            pair = train_rows["source_idx"].to_numpy()
        elif "source_row" in train_rows.columns:
            pair = train_rows["source_row"].to_numpy()
        else:
            pair = np.arange(len(train_rows))
        class_col = "class_name" if "class_name" in train_rows.columns else "label"
        y = train_rows[class_col].astype(str).map(EVT.CLASS_TO_INT).fillna(-1).astype(int).to_numpy()
        subtype = EVT.infer_display_subtype(train_rows).astype(str).to_numpy()
        fam = np.asarray(
            [EVT.BOUNDARY_SUBTYPE_TO_FAMILY.get(str(name), -100) for name in subtype],
            dtype=np.int64,
        )
        for pid, idx in pd.Series(np.arange(len(train_rows))).groupby(pair).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            labels = sorted(set(int(v) for v in y[idx_arr] if v >= 0))
            rows.append(
                {
                    "fold": fold,
                    "pair_id": str(pid),
                    "n": int(len(idx_arr)),
                    "has_good": int(0 in labels),
                    "has_medium": int(1 in labels),
                    "has_bad": int(2 in labels),
                    "gm_pair": int((0 in labels) and (1 in labels)),
                    "families": ",".join(sorted(set(str(v) for v in fam[idx_arr] if int(v) >= 0))),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(SUITE_OUT_DIR / "pairrank_audit.csv", index=False)
    return df


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def _class_metrics_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    main = metrics[(metrics["bucket"] == "clean_test") & (metrics["fold"] >= 0)].copy()
    if main.empty:
        return pd.DataFrame()
    cols = ["acc", "macro_f1", "good_recall", "medium_recall", "bad_recall"]
    return main.groupby("candidate")[cols].agg(["mean", "std", "max"]).reset_index()


def write_waveform_panel(candidate: str, result: dict[str, Any], split_root: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    try:
        global ACTIVE_CFG
        ACTIVE_CFG = dict(cfg)
        fold = int(result["fold"])
        _, _, test_loader, _ = EVT.make_loaders(args, split_root, fold)
        ds = test_loader.dataset
        pred = np.argmax(np.asarray(result["test_probs"], dtype=np.float32), axis=1)
        y = np.asarray(result["test_y"], dtype=int)
        categories = [
            ("good_to_medium", np.where((y == 0) & (pred == 1))[0]),
            ("medium_to_good", np.where((y == 1) & (pred == 0))[0]),
            ("bad_to_nonbad", np.where((y == 2) & (pred != 2))[0]),
            ("nonbad_to_bad", np.where((y != 2) & (pred == 2))[0]),
        ]
        fig, axes = plt.subplots(len(categories), 4, figsize=(12, 7), sharex=True)
        for r, (name, idxs) in enumerate(categories):
            idxs = idxs[:4]
            for c in range(4):
                ax = axes[r, c]
                ax.axis("off")
                if c >= len(idxs):
                    continue
                row = int(idxs[c])
                x = ds.x[row, 0]
                ax.plot(x, lw=0.8, color="#1f4e79")
                ax.set_title(
                    f"{name}\ny={EVT.INT_TO_CLASS[int(y[row])]} p={EVT.INT_TO_CLASS[int(pred[row])]}",
                    fontsize=8,
                )
        fig.suptitle(candidate)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(SUITE_REPORT_DIR / f"{candidate}_fold{result['fold']}_waveform_panel.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        (SUITE_OUT_DIR / "waveform_panel_errors.log").open("a", encoding="utf-8").write(
            f"{candidate} fold={result.get('fold')} error={exc}\n"
        )


def posthoc_model_diagnostics(candidate: str, result: dict[str, Any], split_root: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    try:
        global ACTIVE_CFG
        ACTIVE_CFG = dict(cfg)
        fold = int(result["fold"])
        _, _, test_loader, _ = EVT.make_loaders(args, split_root, fold)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model = EVT.EventFactorizedSQIConformer(
            in_ch=test_loader.dataset.x.shape[1],
            factor_dim=len(EVT.FACTOR_COLUMNS),
            width=int(cfg.get("width", 96)),
            layers=int(cfg.get("layers", 3)),
            heads=int(cfg.get("heads", 4)),
        ).to(device)
        ckpt = torch.load(Path(result["run_path"]) / "ckpt_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()
        dice_vals: list[float] = []
        peak_errs: list[float] = []
        expert_probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                out = model(x)
                local_target = EVT.resize_targets(EVT.pseudo_local_targets(x), out["local_logits"].shape[-1])
                qrs_p = torch.maximum(torch.sigmoid(out["local_logits"][:, 0]), torch.sigmoid(out["local_logits"][:, 1]))
                qrs_t = torch.maximum(local_target[:, 0], local_target[:, 1])
                inter = (qrs_p * qrs_t).sum(dim=1)
                dice = (2.0 * inter + 1.0e-6) / (qrs_p.sum(dim=1) + qrs_t.sum(dim=1) + 1.0e-6)
                dice_vals.extend(dice.detach().cpu().numpy().tolist())
                peak_err = (qrs_p.gt(0.5).float().sum(dim=1) - qrs_t.gt(0.5).float().sum(dim=1)).abs()
                peak_errs.extend(peak_err.detach().cpu().numpy().tolist())
                if "gm_expert_logits" in out:
                    expert_probs.append(torch.sigmoid(out["gm_expert_logits"]).detach().cpu().numpy())
        local_rows.append(
            {
                "candidate": candidate,
                "fold": fold,
                "seed_index": result.get("seed_index", 0),
                "qrs_event_dice": float(np.mean(dice_vals)) if dice_vals else np.nan,
                "peak_count_error": float(np.mean(peak_errs)) if peak_errs else np.nan,
            }
        )
        if expert_probs:
            probs = np.concatenate(expert_probs, axis=0)
            names = ["lowqrs_template", "baseline_contact_detail", "generic_overlap"]
            for j, name in enumerate(names):
                expert_rows.append(
                    {
                        "candidate": candidate,
                        "fold": fold,
                        "seed_index": result.get("seed_index", 0),
                        "expert": name,
                        "mean_prob": float(probs[:, j].mean()),
                        "usage_gt_0p5": float((probs[:, j] > 0.5).mean()),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        local_rows.append(
            {
                "candidate": candidate,
                "fold": result.get("fold"),
                "seed_index": result.get("seed_index"),
                "qrs_event_dice": np.nan,
                "peak_count_error": np.nan,
                "error": str(exc),
            }
        )
    return local_rows, expert_rows


def write_report(
    metrics: pd.DataFrame,
    recovery: pd.DataFrame,
    pair_df: pd.DataFrame,
    factor_var: pd.DataFrame,
    local_df: pd.DataFrame,
    expert_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    summary = _class_metrics_summary(metrics)
    report = SUITE_REPORT_DIR / "gm_mechanism_repair_suite_report.md"
    lines = [
        "# GM Mechanism Repair Suite",
        "",
        f"- Policy: `{args.policy}`",
        f"- Folds: `{args.folds}`",
        f"- Epochs: `{args.epochs}`",
        f"- Seed: `{args.seed}`",
        f"- Suite: `{args.suite}`",
        "",
        "## Clean Test Summary",
        "",
    ]
    if summary.empty:
        lines.append("No clean-test metrics were produced.")
    else:
        lines.append(EVT.df_to_markdown(summary))
    lines += [
        "",
        "## Factor Contract Variance",
        "",
        EVT.df_to_markdown(factor_var.groupby("feature")[["encoded_std", "encoded_var"]].mean().reset_index()),
        "",
        "## Pairrank Audit",
        "",
        f"- candidate pair groups: `{int(pair_df['gm_pair'].sum()) if not pair_df.empty else 0}`",
        f"- total groups: `{len(pair_df)}`",
        "",
        "## Local Event Diagnostics",
        "",
        EVT.df_to_markdown(local_df) if not local_df.empty else "No local diagnostics.",
        "",
        "## Expert Usage",
        "",
        EVT.df_to_markdown(expert_df) if not expert_df.empty else "No expert usage rows.",
        "",
        "## Decoded Factor Recovery",
        "",
    ]
    if not recovery.empty:
        rec = (
            recovery[recovery["bucket"] == "clean_test"]
            .groupby(["candidate", "feature"])[["corr_all", "mae"]]
            .mean()
            .reset_index()
        )
        lines.append(EVT.df_to_markdown(rec))
    else:
        lines.append("No recovery rows.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_posthoc(args: argparse.Namespace, split_root: Path) -> None:
    candidates = suite_candidates()
    if args.candidates:
        keep = {c.strip() for c in args.candidates.split(",") if c.strip()}
        candidates = [(name, cfg) for name, cfg in candidates if name in keep]
    local_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    for name, default_cfg in candidates:
        for fold in range(int(args.folds)):
            run_path = SUITE_RUN_DIR / f"{name}_fold{fold}_seed0"
            if not (run_path / "ckpt_best.pt").exists():
                continue
            try:
                ckpt = torch.load(run_path / "ckpt_best.pt", map_location="cpu")
                cfg = dict(default_cfg)
                cfg.update(dict(ckpt.get("candidate_config", {})))
            except Exception:
                cfg = dict(default_cfg)
            cfg["seed"] = int(args.seed) + int(fold)
            result = {"candidate": name, "fold": fold, "seed_index": 0, "run_path": str(run_path)}
            loc, exp = posthoc_model_diagnostics(name, result, split_root, args, cfg)
            local_rows.extend(loc)
            expert_rows.extend(exp)
    local_df = pd.DataFrame(local_rows)
    expert_df = pd.DataFrame(expert_rows)
    local_df.to_csv(SUITE_OUT_DIR / "local_event_metrics.csv", index=False)
    expert_df.to_csv(SUITE_OUT_DIR / "expert_usage.csv", index=False)

    metrics_df = pd.read_csv(SUITE_OUT_DIR / "candidate_metrics.csv") if (SUITE_OUT_DIR / "candidate_metrics.csv").exists() else pd.DataFrame()
    recovery_df = pd.read_csv(SUITE_OUT_DIR / "factor_recovery_decoded.csv") if (SUITE_OUT_DIR / "factor_recovery_decoded.csv").exists() else pd.DataFrame()
    pair_df = pd.read_csv(SUITE_OUT_DIR / "pairrank_audit.csv") if (SUITE_OUT_DIR / "pairrank_audit.csv").exists() else pd.DataFrame()
    factor_var = pd.read_csv(SUITE_OUT_DIR / "factor_target_variance.csv") if (SUITE_OUT_DIR / "factor_target_variance.csv").exists() else pd.DataFrame()
    if not metrics_df.empty and not recovery_df.empty and not factor_var.empty:
        write_report(metrics_df, recovery_df, pair_df, factor_var, local_df, expert_df, args)
    print(f"[suite] posthoc local rows={len(local_df)} expert rows={len(expert_df)}", flush=True)


def run_factor_ablation_posthoc(args: argparse.Namespace, split_root: Path) -> None:
    candidates = suite_candidates()
    if args.candidates:
        keep = {c.strip() for c in args.candidates.split(",") if c.strip()}
    else:
        keep = {"E24_e6_subtype_fusion_pairrank"}
    candidates = [(name, cfg) for name, cfg in candidates if name in keep]
    rows: list[dict[str, Any]] = []
    for name, default_cfg in candidates:
        for fold in range(int(args.folds)):
            run_path = SUITE_RUN_DIR / f"{name}_fold{fold}_seed0"
            if not (run_path / "ckpt_best.pt").exists():
                continue
            try:
                ckpt = torch.load(run_path / "ckpt_best.pt", map_location="cpu")
                cfg = dict(default_cfg)
                cfg.update(dict(ckpt.get("candidate_config", {})))
            except Exception:
                cfg = dict(default_cfg)
                ckpt = {}
            cfg["seed"] = int(args.seed) + int(fold)
            global ACTIVE_CFG
            ACTIVE_CFG = dict(cfg)
            _, val_loader, test_loader, _ = EVT.make_loaders(args, split_root, fold)
            factor_median = list(ACTIVE_CFG.get("factor_train_median", []))
            device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
            model = EVT.EventFactorizedSQIConformer(
                in_ch=test_loader.dataset.x.shape[1],
                factor_dim=len(EVT.FACTOR_COLUMNS),
                width=int(cfg.get("width", 96)),
                layers=int(cfg.get("layers", 3)),
                heads=int(cfg.get("heads", 4)),
            ).to(device)
            state = ckpt.get("model_state", {})
            model.load_state_dict(state, strict=False)
            model.eval()
            for feature in ["none"] + FACTOR_ABLATION_FEATURES:
                cfg_eval = dict(cfg)
                cfg_eval["factor_train_median"] = factor_median
                if feature != "none":
                    cfg_eval["factor_ablation_feature"] = feature
                ACTIVE_CFG = cfg_eval
                for bucket, loader in [("clean_val", val_loader), ("clean_test", test_loader)]:
                    rep, _, _, _, _, _, _, _, _ = EVT.eval_loader(model, loader, device, name)
                    row = {
                        "candidate": name,
                        "fold": fold,
                        "seed_index": 0,
                        "bucket": bucket,
                        "masked_feature": feature,
                        "acc": float(rep["acc"]),
                        "macro_f1": float(rep["macro_f1"]),
                        "good_recall": float(rep["good_recall"]),
                        "medium_recall": float(rep["medium_recall"]),
                        "bad_recall": float(rep["bad_recall"]),
                        "gm_balanced": 0.5 * (float(rep["good_recall"]) + float(rep["medium_recall"])),
                    }
                    row.update(final_error_route_counts(rep))
                    rows.append(row)
    df = pd.DataFrame(rows)
    out = SUITE_OUT_DIR / "factor_ablation_posthoc.csv"
    df.to_csv(out, index=False)
    if not df.empty:
        base = df[df["masked_feature"].eq("none")].copy()
        deltas = []
        keys = ["acc", "gm_balanced", "good_recall", "medium_recall", "medium_to_good", "medium_to_bad"]
        for _, row in df[~df["masked_feature"].eq("none")].iterrows():
            match = base[
                base["candidate"].eq(row["candidate"])
                & base["fold"].eq(row["fold"])
                & base["bucket"].eq(row["bucket"])
            ]
            if match.empty:
                continue
            brow = match.iloc[0]
            item = {
                "candidate": row["candidate"],
                "fold": int(row["fold"]),
                "bucket": row["bucket"],
                "masked_feature": row["masked_feature"],
            }
            for key in keys:
                item[f"delta_{key}"] = float(row[key] - brow[key])
            deltas.append(item)
        pd.DataFrame(deltas).to_csv(SUITE_OUT_DIR / "factor_ablation_posthoc_deltas.csv", index=False)
    print(f"[suite] factor ablation rows={len(df)} out={out}", flush=True)


def run_suite(args: argparse.Namespace) -> None:
    install_patches()
    split_seed = int(getattr(args, "split_seed", args.seed))
    split_root = SUITE_OUT_DIR / "rh_splits" / f"{EVT.policy_alias(str(args.policy))}_k{args.folds}_s{split_seed}"
    required_split_file = split_root / "fold0" / "source_protocol.json"
    if not split_root.exists() or not required_split_file.exists():
        split_root = EVT.build_recordheldout_splits(str(args.policy), int(args.folds), split_seed)
    if args.suite == "posthoc":
        run_posthoc(args, split_root)
        return
    if args.suite == "factor_ablation_posthoc":
        run_factor_ablation_posthoc(args, split_root)
        return
    factor_var = write_factor_contract_report(split_root, args)
    pair_df = pairrank_audit(split_root, args)

    all_metrics: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_recovery: list[dict[str, Any]] = []
    all_error_routes: list[dict[str, Any]] = []
    all_error_residual: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []

    candidates = suite_candidates()
    if args.candidates:
        keep = {c.strip() for c in args.candidates.split(",") if c.strip()}
        candidates = [(name, cfg) for name, cfg in candidates if name in keep]
    elif args.suite == "smoke":
        candidates = candidates[:3]

    baseline_acc: float | None = None
    e7e8_improved = False
    for name, cfg in candidates:
        run_epochs = int(args.epochs)
        if name == "E9_beat_background_tokens" and not e7e8_improved and args.suite == "all":
            run_epochs = min(2, run_epochs)
        run_args = clone_args(args, epochs=run_epochs)
        print(f"[suite] running {name} epochs={run_epochs}", flush=True)
        candidate_results: list[dict[str, Any]] = []
        for fold in range(args.folds):
            cfg_run = dict(cfg)
            cfg_run["seed"] = int(args.seed) + int(fold)
            res = EVT.train_model(name, cfg_run, run_args, split_root, fold, 0)
            candidate_results.append(res)
            for row in res["metrics"]:
                item = dict(row)
                item["fold"] = int(res["fold"])
                item["seed_index"] = int(res.get("seed_index", 0))
                all_metrics.append(item)
            for row in res.get("record_metrics", []):
                item = dict(row)
                item["fold"] = int(res["fold"])
                item["seed_index"] = int(res.get("seed_index", 0))
                all_records.append(item)
            for row in res.get("recovery", []):
                item = dict(row)
                item["fold"] = int(res["fold"])
                item["seed_index"] = int(res.get("seed_index", 0))
                all_recovery.append(item)
            for row in res.get("error_routes", []):
                item = dict(row)
                item["fold"] = int(res["fold"])
                item["seed_index"] = int(res.get("seed_index", 0))
                all_error_routes.append(item)
            for row in res.get("error_residual", []):
                item = dict(row)
                item["fold"] = int(res["fold"])
                item["seed_index"] = int(res.get("seed_index", 0))
                all_error_residual.append(item)
            write_waveform_panel(name, res, split_root, run_args, cfg_run)
            loc, exp = posthoc_model_diagnostics(name, res, split_root, run_args, cfg_run)
            local_rows.extend(loc)
            expert_rows.extend(exp)

        c_rows = []
        for r in candidate_results:
            for m in r["metrics"]:
                item = dict(m)
                item["fold"] = int(r["fold"])
                item["seed_index"] = int(r.get("seed_index", 0))
                c_rows.append(item)
        c_metrics = pd.DataFrame(c_rows)
        clean = c_metrics[(c_metrics["bucket"] == "clean_test") & (c_metrics["fold"] >= 0)]
        mean_acc = float(clean["acc"].mean()) if not clean.empty else float("nan")
        gm_bal = float(((clean["good_recall"] + clean["medium_recall"]) / 2.0).mean()) if not clean.empty else float("nan")
        print(f"[suite] {name} mean_acc={mean_acc:.4f} gm_bal={gm_bal:.4f}", flush=True)
        if name.startswith("E4_"):
            baseline_acc = mean_acc
        if name.startswith(("E7_", "E8_")) and baseline_acc is not None and mean_acc >= baseline_acc + 0.008:
            e7e8_improved = True

    metrics_df = pd.DataFrame(all_metrics)
    records_df = pd.DataFrame(all_records)
    recovery_df = pd.DataFrame(all_recovery)
    metric_error_routes = []
    if not metrics_df.empty:
        for _, mrow in metrics_df.iterrows():
            item = {
                "candidate": mrow.get("candidate"),
                "bucket": mrow.get("bucket"),
                "fold": mrow.get("fold"),
                "seed_index": mrow.get("seed_index"),
            }
            item.update(final_error_route_counts(mrow.to_dict()))
            metric_error_routes.append(item)
    error_routes_df = pd.DataFrame(metric_error_routes or all_error_routes)
    error_residual_df = pd.DataFrame(all_error_residual)
    local_df = pd.DataFrame(local_rows)
    expert_df = pd.DataFrame(expert_rows)

    metrics_df.to_csv(SUITE_OUT_DIR / "candidate_metrics.csv", index=False)
    records_df.to_csv(SUITE_OUT_DIR / "record_metrics.csv", index=False)
    recovery_df.to_csv(SUITE_OUT_DIR / "factor_recovery_decoded.csv", index=False)
    error_routes_df.to_csv(SUITE_OUT_DIR / "error_route_metrics.csv", index=False)
    error_residual_df.to_csv(SUITE_OUT_DIR / "error_conditioned_factor_residual.csv", index=False)
    local_df.to_csv(SUITE_OUT_DIR / "local_event_metrics.csv", index=False)
    expert_df.to_csv(SUITE_OUT_DIR / "expert_usage.csv", index=False)

    if not metrics_df.empty:
        conf = metrics_df[metrics_df["bucket"] == "clean_test"].copy()
        conf.to_csv(SUITE_OUT_DIR / "confusion_by_candidate.csv", index=False)
    if not metrics_df.empty:
        family_cols = [
            c
            for c in metrics_df.columns
            if c.startswith("boundary_") or c in {"candidate", "fold", "seed", "bucket"}
        ]
        metrics_df[family_cols].to_csv(SUITE_OUT_DIR / "boundary_family_metrics.csv", index=False)

    write_report(metrics_df, recovery_df, pair_df, factor_var, local_df, expert_df, args)
    print(f"[suite] report: {SUITE_REPORT_DIR / 'gm_mechanism_repair_suite_report.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="ptb_v112_gm_buffered_large_hybrid_s20260741")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--split-seed", type=int, default=20260901)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--suite", choices=["smoke", "all", "posthoc", "factor_ablation_posthoc"], default="all")
    p.add_argument("--candidates", default="")
    p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--max-val-rows", type=int, default=0)
    p.add_argument("--max-test-rows", type=int, default=0)
    p.add_argument("--record-balanced-sampler", dest="record_balanced_sampler", action="store_true", default=True)
    p.add_argument("--no-record-balanced-sampler", dest="record_balanced_sampler", action="store_false")
    p.add_argument("--audit-gradients", action="store_true", default=False)
    p.add_argument("--gradient-batches", type=int, default=30)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())
