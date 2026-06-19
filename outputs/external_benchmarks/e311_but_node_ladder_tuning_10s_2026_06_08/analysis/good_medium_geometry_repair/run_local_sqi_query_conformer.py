"""LocalSQI Query Conformer experiments.

External-only runner for testing whether a waveform Transformer improves when
the information path is local waveform evidence -> SQI factor tokens -> task
queries -> coherent hierarchical probabilities.  No tabular/SQI feature is used
as inference input; waveform-computable features are teacher targets only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
DUAL_SCRIPT = ANALYSIS_DIR / "run_clean_but_dualview_hier_transformer.py"
RUN_DIR = OUT_ROOT / "runs" / "local_sqi_query_conformer"
OUT_DIR = ANALYSIS_DIR / "local_sqi_query_conformer"
REPORT_OUT_DIR = REPORT_DIR / "local_sqi_query_conformer"
DEFAULT_POLICY = "margin_ge_5s_drop_outlier"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

FACTOR_COLUMNS = [
    "qrs_visibility",
    "qrs_band_ratio",
    "template_corr",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "non_qrs_diff_p95",
    "amplitude_entropy",
    "sqi_basSQI",
    "detector_agreement",
]
RATIO_COLUMNS = {
    "qrs_visibility",
    "qrs_band_ratio",
    "template_corr",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "amplitude_entropy",
    "sqi_basSQI",
    "detector_agreement",
}
LOCAL_MAP_NAMES = ["qrs_morph", "qrs_slope", "contact", "reset", "flatline", "baseline", "detail"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_dual_module() -> Any:
    spec = importlib.util.spec_from_file_location("clean_but_dualview_for_local_sqi", DUAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {DUAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DUAL = load_dual_module()
TX = DUAL.TX
GEOM = DUAL.GEOM
FORMAL_FEATURE_COLUMNS = list(DUAL.FORMAL_FEATURE_COLUMNS)
FORMAL_FEATURE_IDX = list(DUAL.FORMAL_FEATURE_IDX)
FACTOR_IDX = [FORMAL_FEATURE_COLUMNS.index(c) for c in FACTOR_COLUMNS if c in FORMAL_FEATURE_COLUMNS]
FACTOR_COLUMNS = [FORMAL_FEATURE_COLUMNS[i] for i in FACTOR_IDX]


@dataclass
class ArtifactTargets:
    qrs_morph_map: torch.Tensor
    qrs_slope_map: torch.Tensor
    baseline_component: torch.Tensor
    contact_mask: torch.Tensor
    reset_mask: torch.Tensor
    flatline_mask: torch.Tensor
    detail_map: torch.Tensor
    local_valid_mask: torch.Tensor
    artifact_type: torch.Tensor
    severity: torch.Tensor
    label_preserved: torch.Tensor
    pair_id: torch.Tensor
    source_idx: torch.Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(tokens)
        score = torch.einsum("btd,d->bt", h, self.query) / math.sqrt(float(h.shape[-1]))
        weight = torch.softmax(score, dim=1)
        return torch.einsum("bt,btd->bd", weight, tokens)


class ConformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(width * 4, width),
            nn.Dropout(0.08),
        )
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.08, batch_first=True)
        self.conv_norm = nn.LayerNorm(width)
        self.dwconv = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=17, padding=8, groups=width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
            nn.Dropout(0.08),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(width * 4, width),
            nn.Dropout(0.08),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        h = self.conv_norm(x).transpose(1, 2)
        x = x + self.dwconv(h).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return x


class LocalSQIQueryConformer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        factor_dim: int,
        width: int = 96,
        layers: int = 4,
        heads: int = 4,
        use_highres_path: bool = True,
        use_sqi_queries: bool = True,
        use_hierarchical_head: bool = True,
        fusion_mode: str = "none",
    ):
        super().__init__()
        self.use_highres_path = bool(use_highres_path)
        self.use_sqi_queries = bool(use_sqi_queries)
        self.use_hierarchical_head = bool(use_hierarchical_head)
        self.fusion_mode = str(fusion_mode)
        self.factor_dim = int(factor_dim)
        self.hi_stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
        )
        self.ctx_down = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(max(1, min(8, width // 8)), width),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(width, max_len=1024)
        self.query_names = [
            "QRS",
            "RR_TEMPLATE",
            "BASELINE",
            "CONTACT_RESET",
            "DETAIL_NOISE",
            "GLOBAL_MORPH",
            "GM_BOUNDARY",
            "BAD_STRESS",
        ]
        self.query = nn.Parameter(torch.randn(len(self.query_names), width) * 0.02)
        self.blocks = nn.Sequential(*[ConformerBlock(width, heads) for _ in range(layers)])
        self.pool = AttentionPool(width)
        self.local_heads = nn.Conv1d(width, len(LOCAL_MAP_NAMES), kernel_size=1)
        self.factor_head = nn.Sequential(nn.LayerNorm(width * 6), nn.Linear(width * 6, 192), nn.GELU(), nn.Dropout(0.08), nn.Linear(192, factor_dim))
        self.factor_to_query = nn.Sequential(nn.LayerNorm(factor_dim), nn.Linear(factor_dim, width), nn.GELU(), nn.Dropout(0.05))
        self.bad_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.medium_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.ce_head = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, 128), nn.GELU(), nn.Dropout(0.08), nn.Linear(128, 3))
        self.context_to_local = nn.Sequential(nn.Conv1d(width, width, kernel_size=3, padding=1), nn.GELU(), nn.Conv1d(width, len(LOCAL_MAP_NAMES), kernel_size=1))

    def forward_tokens(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hi = self.hi_stem(x)
        ctx = self.ctx_down(hi).transpose(1, 2)
        b = x.shape[0]
        if self.use_sqi_queries:
            q = self.query[None, :, :].expand(b, -1, -1)
            seq = torch.cat([q, ctx], dim=1)
            seq = self.blocks(self.pos(seq))
            query_tokens = seq[:, : len(self.query_names), :]
            context_tokens = seq[:, len(self.query_names) :, :]
        else:
            context_tokens = self.blocks(self.pos(ctx))
            pooled = self.pool(context_tokens)
            query_tokens = pooled[:, None, :].expand(-1, len(self.query_names), -1)
        return {"hi_tokens": hi.transpose(1, 2), "context_tokens": context_tokens, "query_tokens": query_tokens}

    def forward(self, x: torch.Tensor, oracle_factor: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tok = self.forward_tokens(x)
        hi_ch = tok["hi_tokens"].transpose(1, 2)
        ctx = tok["context_tokens"]
        query = tok["query_tokens"]
        if self.use_highres_path:
            local_logits = self.local_heads(hi_ch)
        else:
            low = self.context_to_local(ctx.transpose(1, 2))
            local_logits = F.interpolate(low, size=hi_ch.shape[-1], mode="linear", align_corners=False)
        factor_pred = self.factor_head(query[:, :6, :].reshape(x.shape[0], -1))
        gm_repr = query[:, 6, :] if self.use_sqi_queries else self.pool(ctx)
        bad_repr = query[:, 7, :] if self.use_sqi_queries else self.pool(ctx)
        if self.fusion_mode != "none":
            if self.fusion_mode == "oracle":
                if oracle_factor is None:
                    raise ValueError("oracle fusion requires oracle_factor")
                factor_for_cls = oracle_factor
            elif self.fusion_mode == "pred_stopgrad":
                factor_for_cls = factor_pred.detach()
            elif self.fusion_mode == "pred_e2e":
                factor_for_cls = factor_pred
            else:
                raise ValueError(f"unknown fusion_mode {self.fusion_mode}")
            factor_delta = self.factor_to_query(factor_for_cls)
            gm_repr = gm_repr + factor_delta
            bad_repr = bad_repr + factor_delta
        pooled = torch.cat([self.pool(ctx), ctx.mean(dim=1)], dim=1)
        if self.use_hierarchical_head:
            bad_logit = self.bad_head(bad_repr).squeeze(1)
            medium_logit = self.medium_head(gm_repr).squeeze(1)
            bad_prob = torch.sigmoid(bad_logit).clamp(1e-5, 1.0 - 1e-5)
            med_given_nonbad = torch.sigmoid(medium_logit).clamp(1e-5, 1.0 - 1e-5)
            probs = torch.stack(
                [
                    (1.0 - bad_prob) * (1.0 - med_given_nonbad),
                    (1.0 - bad_prob) * med_given_nonbad,
                    bad_prob,
                ],
                dim=1,
            ).clamp(1e-6, 1.0)
            logits = torch.log(probs)
        else:
            logits = self.ce_head(pooled)
            bad_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
            medium_logit = logits[:, 1] - logits[:, 0]
            probs = torch.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "bad_logit": bad_logit,
            "medium_logit": medium_logit,
            "factor_pred": factor_pred,
            "local_logits": local_logits,
            "hi_tokens": tok["hi_tokens"],
            "context_tokens": ctx,
            "query_tokens": query,
            "pooled": pooled,
        }


class CleanProtocolDataset(Dataset):
    def __init__(
        self,
        source_protocol: Path,
        split: str,
        atlas_frame: pd.DataFrame | None = None,
        channel_stats: Any | None = None,
        feature_norm: Any | None = None,
        max_rows: int = 0,
        seed: int = 20260619,
    ):
        atlas_all = pd.read_csv(source_protocol / "original_region_atlas.csv") if atlas_frame is None else atlas_frame.copy()
        signals = np.load(source_protocol / "signals.npz")["X"].astype(np.float32)
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        for col in FORMAL_FEATURE_COLUMNS:
            if col not in atlas_all.columns:
                atlas_all[col] = 0.0
        mask = atlas_all["split"].astype(str).eq(split).to_numpy()
        self.frame = atlas_all.loc[mask].reset_index(drop=True)
        if int(max_rows) > 0 and len(self.frame) > int(max_rows):
            rng = np.random.default_rng(int(seed))
            chosen: list[int] = []
            per = max(1, int(max_rows) // 3)
            y_names = self.frame["class_name"].astype(str).to_numpy()
            for cls in ("good", "medium", "bad"):
                idx = np.flatnonzero(y_names == cls)
                if idx.size:
                    chosen.extend(rng.choice(idx, size=min(per, idx.size), replace=False).tolist())
            rem = int(max_rows) - len(chosen)
            if rem > 0:
                pool = np.setdiff1d(np.arange(len(self.frame)), np.asarray(chosen, dtype=int), assume_unique=False)
                if pool.size:
                    chosen.extend(rng.choice(pool, size=min(rem, pool.size), replace=False).tolist())
            self.frame = self.frame.iloc[sorted(set(chosen))].reset_index(drop=True)
        rows = self.frame["idx"].astype(int).to_numpy()
        if channel_stats is None:
            train_rows = atlas_all.loc[atlas_all["split"].astype(str).eq("train"), "idx"].astype(int).to_numpy()
            if train_rows.size == 0:
                train_rows = rows
            channel_stats = DUAL.compute_channel_stats(signals[train_rows])
        if feature_norm is None:
            train_mask = atlas_all["split"].astype(str).eq("train")
            if not bool(train_mask.any()):
                train_mask = mask
            train_raw = (
                atlas_all.loc[train_mask, FORMAL_FEATURE_COLUMNS]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            mean = train_raw.mean(axis=0).astype(np.float32)
            std = train_raw.std(axis=0).astype(np.float32)
            std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
            feature_norm = DUAL.FeatureNorm(mean=mean, std=std)
        raw = (
            self.frame[FORMAL_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        self.x = DUAL.make_dualview_channels(signals[rows], channel_stats).astype(np.float32)
        self.aux_raw = raw.astype(np.float32)
        self.aux = ((raw - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.factor = self.aux[:, FACTOR_IDX].astype(np.float32)
        self.factor_raw = self.aux_raw[:, FACTOR_IDX].astype(np.float32)
        self.y = self.frame["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.artifact = infer_artifact_flag(self.frame, self.y).astype(np.float32)
        self.source_idx = self.frame.get("source_idx", self.frame.get("idx", pd.Series(np.arange(len(self.frame))))).fillna(-1).astype(int).to_numpy()
        self.pair_id = self.source_idx.copy()
        self.channel_stats = channel_stats
        self.feature_norm = feature_norm

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "factor": torch.from_numpy(self.factor[i]),
            "factor_raw": torch.from_numpy(self.factor_raw[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
            "artifact": torch.tensor(float(self.artifact[i]), dtype=torch.float32),
            "source_idx": torch.tensor(int(self.source_idx[i]), dtype=torch.long),
            "pair_id": torch.tensor(int(self.pair_id[i]), dtype=torch.long),
        }


def infer_artifact_flag(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    text = np.asarray([""] * len(frame), dtype=object)
    for col in ["original_region", "ambiguous_type", "clean_tier", "clean_policy", "bank_role", "profile_block"]:
        if col in frame.columns:
            text = np.char.add(np.char.add(text.astype(str), "|"), frame[col].fillna("").astype(str).to_numpy())
    low = np.char.lower(text.astype(str))
    flag = (
        (np.char.find(low, "outlier") >= 0)
        | (np.char.find(low, "controlled") >= 0)
        | (np.char.find(low, "contact") >= 0)
        | (np.char.find(low, "flat") >= 0)
        | (np.char.find(low, "bad") >= 0)
    )
    return np.where((y == CLASS_TO_INT["bad"]) | flag, 1.0, 0.0)


def pseudo_local_targets(x: torch.Tensor) -> dict[str, torch.Tensor]:
    # x channels: physical, robust z, dz, baseline_long, highpass, detail_fast, detail_slow, physical_trend.
    z = x[:, 1]
    dz = x[:, 2]
    baseline = x[:, 3]
    highpass = x[:, 4]
    detail_fast = x[:, 5]
    detail_slow = x[:, 6].abs() + 1e-3
    abs_dz = dz.abs()
    abs_hp = highpass.abs()
    qrs_slope = torch.sigmoid((abs_dz - 1.25) / 0.35)
    prom = abs_hp / (detail_slow + 0.05)
    qrs_morph = torch.sigmoid((prom - 1.60) / 0.45)
    contact = torch.sigmoid((0.22 - abs_dz) / 0.055) * torch.sigmoid((0.32 - abs_hp) / 0.075)
    flatline = torch.sigmoid((0.12 - abs_dz) / 0.040) * torch.sigmoid((0.22 - abs_hp) / 0.060)
    reset = torch.sigmoid((abs_dz - 2.50) / 0.45)
    base = torch.tanh(baseline / 2.5)
    detail = torch.tanh(detail_fast / (detail_slow + 0.10))
    valid = torch.ones_like(qrs_morph)
    return {
        "qrs_morph": qrs_morph,
        "qrs_slope": qrs_slope,
        "contact": contact,
        "reset": reset,
        "flatline": flatline,
        "baseline": base,
        "detail": detail,
        "valid": valid,
    }


def resize_targets(targets: dict[str, torch.Tensor], target_len: int) -> torch.Tensor:
    maps = torch.stack([targets[name] for name in LOCAL_MAP_NAMES], dim=1)
    if maps.shape[-1] != int(target_len):
        maps = F.interpolate(maps, size=int(target_len), mode="linear", align_corners=False)
    return maps


def local_map_loss(local_logits: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    targets = pseudo_local_targets(x)
    target_maps = resize_targets(targets, local_logits.shape[-1])
    bce_idx = [0, 1, 2, 3, 4]
    cont_idx = [5, 6]
    bce = F.binary_cross_entropy_with_logits(local_logits[:, bce_idx], target_maps[:, bce_idx])
    pred_cont = torch.tanh(local_logits[:, cont_idx])
    cont = F.smooth_l1_loss(pred_cont, target_maps[:, cont_idx])
    loss = bce + 0.50 * cont
    with torch.no_grad():
        pred_qrs_morph = torch.sigmoid(local_logits[:, 0])
        pred_qrs_slope = torch.sigmoid(local_logits[:, 1])
        agreement = 1.0 - torch.mean(torch.abs(pred_qrs_morph - pred_qrs_slope), dim=1)
    return loss, {"local_bce": float(bce.detach().cpu()), "local_cont": float(cont.detach().cpu()), "pred_detector_agreement": float(agreement.mean().detach().cpu())}


def factor_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    pred = out["factor_pred"]
    target_norm = batch["factor"].to(device=device, dtype=torch.float32)
    target_raw = batch["factor_raw"].to(device=device, dtype=torch.float32)
    losses = []
    parts: dict[str, float] = {}
    for j, name in enumerate(FACTOR_COLUMNS):
        if name == "detector_agreement":
            q0 = torch.sigmoid(out["local_logits"][:, 0])
            q1 = torch.sigmoid(out["local_logits"][:, 1])
            det_pred = 1.0 - torch.mean(torch.abs(q0 - q1), dim=1)
            det_target = target_raw[:, j].clamp(0.0, 1.0)
            val = F.smooth_l1_loss(det_pred, det_target)
        elif name in RATIO_COLUMNS:
            raw_target = target_raw[:, j].clamp(0.0, 1.0)
            val = F.binary_cross_entropy_with_logits(pred[:, j], raw_target)
        else:
            val = F.smooth_l1_loss(pred[:, j], target_norm[:, j])
        losses.append(val)
        parts[f"factor_{name}"] = float(val.detach().cpu())
    return torch.stack(losses).mean(), parts


def hierarchical_nll(out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(out["logits"], y)


def ce_loss(out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(out["logits"], y)


def artifact_specificity_loss(out: dict[str, torch.Tensor], artifact: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Encourage bad query to notice controlled/stress artifacts while explicitly
    # reducing false-bad on artifact-positive nonbad rows.
    bad_prob = torch.sigmoid(out["bad_logit"])
    artifact = artifact.float()
    artifact_loss = F.binary_cross_entropy(bad_prob.clamp(1e-5, 1 - 1e-5), artifact)
    nonbad_artifact = (artifact > 0.5) & (y != CLASS_TO_INT["bad"])
    if torch.any(nonbad_artifact):
        specificity = bad_prob[nonbad_artifact].mean()
    else:
        specificity = torch.zeros((), device=y.device)
    return artifact_loss + 0.75 * specificity


def metric_report(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y_pred = np.argmax(probs, axis=1)
    rep = GEOM.metric_report(y_true, y_pred, probs)
    rep["macro_f1_sklearn"] = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
    rep["gm_balanced_acc"] = float(
        np.mean(
            [
                rep.get("good_recall", 0.0),
                rep.get("medium_recall", 0.0),
            ]
        )
    )
    nonbad = y_true != CLASS_TO_INT["bad"]
    rep["bad_fpr_nonbad"] = float(np.mean(y_pred[nonbad] == CLASS_TO_INT["bad"])) if np.any(nonbad) else 0.0
    return rep


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    for key, val in rep.items():
        row[key] = json.dumps(val) if key == "confusion_3x3" else val
    return row


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    factor_targets: list[np.ndarray] = []
    artifacts: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        oracle = batch["factor"].to(device=device, dtype=torch.float32) if getattr(model, "fusion_mode", "none") == "oracle" else None
        out = model(x, oracle_factor=oracle)
        p = out["probs"] if "probs" in out else torch.softmax(out["logits"], dim=1)
        ys.append(batch["y"].numpy())
        probs.append(p.detach().cpu().numpy())
        factors.append(out["factor_pred"].detach().cpu().numpy())
        factor_targets.append(batch["factor"].numpy())
        artifacts.append(batch["artifact"].numpy())
    y_np = np.concatenate(ys)
    p_np = np.concatenate(probs)
    pred_factor = np.concatenate(factors)
    true_factor = np.concatenate(factor_targets)
    artifact_np = np.concatenate(artifacts)
    rep = metric_report(y_np, p_np)
    y_pred = np.argmax(p_np, axis=1)
    art_nonbad = (artifact_np > 0.5) & (y_np != CLASS_TO_INT["bad"])
    rep["artifact_positive_nonbad_bad_fpr"] = float(np.mean(y_pred[art_nonbad] == CLASS_TO_INT["bad"])) if np.any(art_nonbad) else 0.0
    rep["factor_mae"] = float(np.mean(np.abs(pred_factor - true_factor)))
    return rep, p_np, pred_factor, true_factor


def factor_recovery_rows(candidate: str, bucket: str, pred: np.ndarray, true: np.ndarray, y: np.ndarray | None = None) -> list[dict[str, Any]]:
    rows = []
    for j, name in enumerate(FACTOR_COLUMNS):
        a = pred[:, j]
        b = true[:, j]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        row = {"candidate": candidate, "bucket": bucket, "feature": name, "corr_all": corr, "mae": float(np.mean(np.abs(a - b)))}
        if y is not None:
            for cls, idx in CLASS_TO_INT.items():
                mask = y == idx
                if np.sum(mask) >= 3 and np.std(a[mask]) > 1e-8 and np.std(b[mask]) > 1e-8:
                    row[f"corr_{cls}"] = float(np.corrcoef(a[mask], b[mask])[0, 1])
                else:
                    row[f"corr_{cls}"] = 0.0
            row["corr_min_class"] = float(min(row["corr_good"], row["corr_medium"], row["corr_bad"]))
        rows.append(row)
    return rows


def build_recordheldout_splits(policy: str, folds: int, seed: int) -> Path:
    source = DUAL.PROTOCOL_ROOT / policy
    atlas = pd.read_csv(source / "original_region_atlas.csv")
    if "record_id" not in atlas.columns:
        raise ValueError("record_id is required for record-heldout split")
    groups = atlas["record_id"].astype(str).to_numpy()
    y = atlas["class_name"].astype(str).map(CLASS_TO_INT).to_numpy()
    gkf = GroupKFold(n_splits=int(folds))
    root = OUT_DIR / "recordheldout_splits" / f"{policy}_groupkfold{folds}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(atlas, y, groups)):
        rng = np.random.default_rng(int(seed) + int(fold))
        train_val = np.asarray(train_val_idx)
        val_records = rng.choice(np.unique(groups[train_val]), size=max(1, int(round(0.20 * len(np.unique(groups[train_val]))))), replace=False)
        val_mask = np.isin(groups, val_records) & np.isin(np.arange(len(atlas)), train_val)
        split = np.asarray(["train"] * len(atlas), dtype=object)
        split[test_idx] = "test"
        split[val_mask] = "val"
        frame = atlas.copy()
        frame["split"] = split
        fold_dir = root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(fold_dir / "original_region_atlas.csv", index=False)
        (fold_dir / "source_protocol.json").write_text(json.dumps({"source_protocol": str(source), "signals": str(source / "signals.npz")}, indent=2), encoding="utf-8")
        for part in ["train", "val", "test"]:
            part_frame = frame.loc[frame["split"].eq(part)]
            class_counts = part_frame["class_name"].value_counts().to_dict()
            rows.append({"fold": fold, "split": part, "rows": int(len(part_frame)), "records": int(part_frame["record_id"].nunique()), **{f"{k}_rows": int(v) for k, v in class_counts.items()}})
    pd.DataFrame(rows).to_csv(root / "recordheldout_split_summary.csv", index=False)
    report = [
        f"# Record-Heldout Splits\n",
        f"- Source policy: `{policy}`",
        f"- Folds: `{folds}`",
        f"- Seed: `{seed}`",
        f"- Source protocol remains in `{source}`; fold directories store split atlas only.",
        "",
        df_to_markdown(pd.DataFrame(rows)),
    ]
    (REPORT_OUT_DIR / "recordheldout_splits_report.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT_DIR / "recordheldout_splits_report.md").write_text("\n".join(report), encoding="utf-8")
    return root


def load_recordheldout_frame(split_root: Path, fold: int) -> tuple[Path, pd.DataFrame]:
    fold_dir = split_root / f"fold{fold}"
    meta = json.loads((fold_dir / "source_protocol.json").read_text(encoding="utf-8"))
    source = Path(meta["source_protocol"])
    frame = pd.read_csv(fold_dir / "original_region_atlas.csv")
    return source, frame


def make_loaders(args: argparse.Namespace, split_root: Path | None = None, fold: int = 0) -> tuple[DataLoader, DataLoader, DataLoader, CleanProtocolDataset]:
    if split_root is None:
        source = DUAL.PROTOCOL_ROOT / str(args.policy)
        atlas_frame = None
    else:
        source, atlas_frame = load_recordheldout_frame(split_root, int(fold))
    train_ds = CleanProtocolDataset(source, "train", atlas_frame=atlas_frame, max_rows=int(args.max_train_rows), seed=int(args.seed))
    val_ds = CleanProtocolDataset(source, "val", atlas_frame=atlas_frame, channel_stats=train_ds.channel_stats, feature_norm=train_ds.feature_norm, max_rows=int(args.max_val_rows), seed=int(args.seed) + 1)
    test_ds = CleanProtocolDataset(source, "test", atlas_frame=atlas_frame, channel_stats=train_ds.channel_stats, feature_norm=train_ds.feature_norm, max_rows=int(args.max_test_rows), seed=int(args.seed) + 2)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size) * 2, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size) * 2, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    return train_loader, val_loader, test_loader, train_ds


def train_model(candidate: str, cfg: dict[str, Any], args: argparse.Namespace, split_root: Path | None = None, fold: int = 0) -> dict[str, Any]:
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_ds = make_loaders(args, split_root, fold)
    model = LocalSQIQueryConformer(
        in_ch=int(train_ds.x.shape[1]),
        factor_dim=len(FACTOR_COLUMNS),
        width=int(cfg["width"]),
        layers=int(cfg["layers"]),
        heads=int(cfg["heads"]),
        use_highres_path=bool(cfg["use_highres_path"]),
        use_sqi_queries=bool(cfg["use_sqi_queries"]),
        use_hierarchical_head=bool(cfg["use_hierarchical_head"]),
        fusion_mode=str(cfg.get("fusion_mode", "none")),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    best_state = None
    best_score = -1e9
    logs: list[dict[str, Any]] = []
    gradient_audit: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            artifact = batch["artifact"].to(device=device)
            oracle = batch["factor"].to(device=device, dtype=torch.float32) if str(cfg.get("fusion_mode", "none")) == "oracle" else None
            out = model(x, oracle_factor=oracle)
            if bool(cfg["use_hierarchical_head"]):
                cls = hierarchical_nll(out, y)
            else:
                cls = ce_loss(out, y)
            factor, factor_parts = factor_loss(out, batch, device)
            local, local_parts = local_map_loss(out["local_logits"], x)
            art = artifact_specificity_loss(out, artifact, y)
            loss = cls + float(cfg["factor_weight"]) * factor + float(cfg["local_weight"]) * local + float(cfg["artifact_weight"]) * art
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if bool(args.audit_gradients) and step == 0:
                gradient_audit.extend(gradient_snapshot(model, batch, cfg, device, epoch))
        val_rep, _, _, _ = eval_loader(model, val_loader, device)
        score = float(val_rep["macro_f1"]) + 0.20 * float(min(val_rep["good_recall"], val_rep["medium_recall"], val_rep["bad_recall"])) - 0.05 * float(val_rep["bad_fpr_nonbad"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"}}
        row.update(local_parts)
        row.update({k: v for k, v in factor_parts.items() if k in {"factor_detector_agreement", "factor_qrs_visibility", "factor_sqi_basSQI"}})
        logs.append(row)
        print(
            f"{candidate} fold={fold} epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_acc={val_rep['acc']:.4f} macro={val_rep['macro_f1']:.4f} recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_rep, val_probs, val_factor, val_true_factor = eval_loader(model, val_loader, device)
    test_rep, test_probs, test_factor, test_true_factor = eval_loader(model, test_loader, device)
    test_y = np.concatenate([batch["y"].numpy() for batch in test_loader])
    run_name = f"{candidate}_fold{fold}"
    run_path = RUN_DIR / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "factor_columns": FACTOR_COLUMNS,
            "local_map_names": LOCAL_MAP_NAMES,
            "input_contract": "waveform-derived channels only; factors are teacher targets only",
            "model_kind": "LocalSQIQueryConformer",
            "source_commit_hint": "experiment/factorial-local-splits@d8cc663+",
        },
        run_path / "ckpt_best.pt",
    )
    pd.DataFrame(logs).to_csv(run_path / "train_log.csv", index=False)
    if gradient_audit:
        pd.DataFrame(gradient_audit).to_csv(run_path / "gradient_audit.csv", index=False)
    np.savez_compressed(run_path / "test_predictions.npz", probs=test_probs, factor_pred=test_factor, factor_true=test_true_factor)
    return {
        "candidate": candidate,
        "fold": int(fold),
        "run_path": str(run_path),
        "metrics": [
            metric_row(candidate, "clean_val", val_rep),
            metric_row(candidate, "clean_test", test_rep),
        ],
        "recovery": factor_recovery_rows(candidate, "clean_test", test_factor, test_true_factor, test_y),
        "logs": logs,
    }


def gradient_snapshot(model: nn.Module, batch: dict[str, torch.Tensor], cfg: dict[str, Any], device: torch.device, epoch: int) -> list[dict[str, Any]]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device)
    artifact = batch["artifact"].to(device=device)
    oracle = batch["factor"].to(device=device, dtype=torch.float32) if str(cfg.get("fusion_mode", "none")) == "oracle" else None
    losses = {}
    out = model(x, oracle_factor=oracle)
    losses["class"] = hierarchical_nll(out, y) if bool(cfg["use_hierarchical_head"]) else ce_loss(out, y)
    losses["factor"] = factor_loss(out, batch, device)[0]
    losses["local"] = local_map_loss(out["local_logits"], x)[0]
    losses["artifact"] = artifact_specificity_loss(out, artifact, y)
    params = [p for n, p in model.named_parameters() if p.requires_grad and ("hi_stem" in n or "ctx_down" in n or "blocks" in n)]
    grads: dict[str, torch.Tensor] = {}
    for name, loss in losses.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        flat = []
        for p in params:
            if p.grad is not None:
                flat.append(p.grad.detach().flatten())
        grads[name] = torch.cat(flat) if flat else torch.zeros(1, device=device)
    names = list(grads)
    rows = []
    for name in names:
        rows.append({"epoch": epoch, "loss_a": name, "loss_b": name, "grad_norm_a": float(torch.norm(grads[name]).detach().cpu()), "cosine": 1.0})
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ga = grads[a]
            gb = grads[b]
            cos = F.cosine_similarity(ga[None, :], gb[None, :]).item() if ga.numel() == gb.numel() else 0.0
            rows.append({"epoch": epoch, "loss_a": a, "loss_b": b, "grad_norm_a": float(torch.norm(ga).detach().cpu()), "grad_norm_b": float(torch.norm(gb).detach().cpu()), "cosine": float(cos)})
    model.zero_grad(set_to_none=True)
    return rows


def candidate_grid(stage: str) -> dict[str, dict[str, Any]]:
    base = {
        "width": 96,
        "layers": 3,
        "heads": 4,
        "lr": 2.5e-4,
        "factor_weight": 0.35,
        "local_weight": 0.35,
        "artifact_weight": 0.20,
        "seed": 20260619,
        "use_hierarchical_head": True,
    }
    if stage == "stage_a":
        return {
            "A0_nohi_noquery": {**base, "use_highres_path": False, "use_sqi_queries": False, "local_weight": 0.0, "seed": 20260700},
            "A1_hi_noquery": {**base, "use_highres_path": True, "use_sqi_queries": False, "seed": 20260701},
            "A2_nohi_query": {**base, "use_highres_path": False, "use_sqi_queries": True, "local_weight": 0.0, "seed": 20260702},
            "A3_hi_query": {**base, "use_highres_path": True, "use_sqi_queries": True, "seed": 20260703},
        }
    if stage == "stage_b":
        return {
            "B0_query_ce_nolocal": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": False, "local_weight": 0.0, "seed": 20260710},
            "B1_query_ce_local": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": False, "seed": 20260711},
            "B2_query_hier_nolocal": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": True, "local_weight": 0.0, "seed": 20260712},
            "B3_query_hier_local": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": True, "seed": 20260713},
        }
    if stage == "stage_c":
        return {
            "C0_query_hier_local": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": True, "seed": 20260720},
        }
    if stage == "sqi_fusion_ladder":
        return {
            "L0_waveform_only": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": True, "fusion_mode": "none", "seed": 20260730},
            "L1_oracle_sqi_diag": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": True, "fusion_mode": "oracle", "seed": 20260731},
            "L2_pred_sqi_stopgrad": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": True, "fusion_mode": "pred_stopgrad", "seed": 20260732},
            "L3_pred_sqi_e2e": {**base, "use_highres_path": True, "use_sqi_queries": True, "use_hierarchical_head": True, "fusion_mode": "pred_e2e", "seed": 20260733},
        }
    raise ValueError(f"unknown stage {stage}")


def run_train_stage(args: argparse.Namespace, stage: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_root = OUT_DIR / "recordheldout_splits" / f"{args.policy}_groupkfold{args.folds}_seed{args.split_seed}"
    if not split_root.exists() or bool(args.rebuild_splits):
        split_root = build_recordheldout_splits(str(args.policy), int(args.folds), int(args.split_seed))
    rows: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    names = [x.strip() for x in str(args.candidates).split(",") if x.strip()]
    grid = candidate_grid(stage)
    if not names:
        names = list(grid)
    for name in names:
        if name not in grid:
            raise ValueError(f"unknown candidate {name}; expected {sorted(grid)}")
        summary = train_model(name, grid[name], args, split_root=split_root, fold=int(args.fold))
        summaries.append(summary)
        rows.extend(summary["metrics"])
        recovery.extend(summary["recovery"])
    metrics = pd.DataFrame(rows)
    rec = pd.DataFrame(recovery)
    metrics_path = OUT_DIR / f"{stage}_metrics.csv"
    rec_path = OUT_DIR / f"{stage}_feature_recovery.csv"
    metrics.to_csv(metrics_path, index=False)
    rec.to_csv(rec_path, index=False)
    payload = {
        "created_at": now(),
        "stage": stage,
        "policy": str(args.policy),
        "fold": int(args.fold),
        "split_root": str(split_root),
        "metrics_csv": str(metrics_path),
        "feature_recovery_csv": str(rec_path),
        "summaries": summaries,
    }
    (OUT_DIR / f"{stage}_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report = render_stage_report(stage, metrics, rec, payload)
    (OUT_DIR / f"{stage}_report.md").write_text(report, encoding="utf-8")
    (REPORT_OUT_DIR / f"{stage}_report.md").write_text(report, encoding="utf-8")
    print(report)


def render_stage_report(stage: str, metrics: pd.DataFrame, rec: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [f"# LocalSQI Query Conformer {stage} Report", "", f"- Created: {payload['created_at']}", f"- Policy: `{payload['policy']}`", f"- Fold: `{payload['fold']}`", ""]
    if len(metrics):
        view_cols = [c for c in ["candidate", "bucket", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall", "gm_balanced_acc", "bad_fpr_nonbad", "artifact_positive_nonbad_bad_fpr", "factor_mae"] if c in metrics.columns]
        lines.extend(["## Metrics", "", df_to_markdown(metrics[view_cols]), ""])
    if len(rec):
        pivot = rec.pivot_table(index="candidate", columns="feature", values="corr_all", aggfunc="mean").reset_index()
        cols = ["candidate"] + [c for c in FACTOR_COLUMNS if c in pivot.columns]
        lines.extend(["## Factor Recovery Corr", "", df_to_markdown(pivot[cols]), ""])
    lines.extend(
        [
            "## Interpretation Contract",
            "",
            "- A1 > A0 suggests early high-resolution evidence matters.",
            "- A2 > A0 suggests task query readout matters.",
            "- A3 > A1/A2 suggests both local evidence and task readout are needed.",
            "- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.",
        ]
    )
    return "\n".join(lines)


def run_probe_token_vs_pool(args: argparse.Namespace) -> None:
    # Lightweight, direct probe on untrained LocalSQI encoder representations.
    # This is a plumbing check plus baseline; trained checkpoints can be added by
    # passing their state dict in a later stage without changing the report shape.
    args.epochs = max(1, int(args.epochs))
    args.candidates = "A3_hi_query"
    run_train_stage(args, "stage_a")
    src = OUT_DIR / "stage_a_feature_recovery.csv"
    report = [
        "# Token-vs-Pool Probe",
        "",
        "V1 reuses the A3 query/high-resolution run and records factor recovery from task-query outputs.",
        "A future extension can load a frozen checkpoint and train separate pooled/context/high-res probes.",
        "",
        f"- Recovery CSV: `{src}`",
    ]
    (REPORT_OUT_DIR / "probe_token_vs_pool_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT_DIR / "probe_token_vs_pool_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="stage_a", choices=["build_recordheldout_splits", "stage_a", "stage_b", "stage_c", "sqi_fusion_ladder", "probe_token_vs_pool", "audit_loss_gradients", "all"])
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260619)
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--candidates", type=str, default="")
    parser.add_argument("--audit-gradients", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "build_recordheldout_splits":
        root = build_recordheldout_splits(str(args.policy), int(args.folds), int(args.split_seed))
        print(root)
        return
    if args.stage == "stage_a":
        run_train_stage(args, "stage_a")
        return
    if args.stage == "stage_b":
        run_train_stage(args, "stage_b")
        return
    if args.stage == "stage_c":
        run_train_stage(args, "stage_c")
        return
    if args.stage == "sqi_fusion_ladder":
        run_train_stage(args, "sqi_fusion_ladder")
        return
    if args.stage == "probe_token_vs_pool":
        run_probe_token_vs_pool(args)
        return
    if args.stage == "audit_loss_gradients":
        args.audit_gradients = True
        args.candidates = args.candidates or "B3_query_hier_local"
        run_train_stage(args, "stage_b")
        return
    if args.stage == "all":
        build_recordheldout_splits(str(args.policy), int(args.folds), int(args.split_seed))
        run_train_stage(args, "stage_a")
        run_train_stage(args, "stage_b")
        return


if __name__ == "__main__":
    main()
