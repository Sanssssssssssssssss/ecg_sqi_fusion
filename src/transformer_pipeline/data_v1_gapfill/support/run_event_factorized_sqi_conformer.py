"""Event-factorized SQI Conformer experiments.

External-only runner for the corrected LocalSQI follow-up:

waveform-derived channels -> physiology/artifact/detail factor tokens ->
GM/BAD task queries -> coherent hierarchical probabilities.

This runner intentionally does not import or patch the old LocalSQI runner.
It reuses only the clean-BUT protocol loader utilities and 8-channel waveform
view from ``run_clean_but_dualview_hier_transformer.py``.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DUAL_SCRIPT = SCRIPT_DIR / "run_clean_but_dualview_hier_transformer.py"
RUN_DIR = OUT_ROOT / "runs" / "event_factorized_sqi_conformer"
OUT_DIR = ANALYSIS_DIR / "event_factorized_sqi_conformer"
REPORT_OUT_DIR = REPORT_DIR / "event_factorized_sqi_conformer"
DEFAULT_POLICY = "margin_ge_5s_drop_outlier"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
V116_ORIGINAL_SPLIT_COUNTS = {
    "good": {"train": 8424, "val": 1053, "test": 1053},
    "medium": {"train": 5199, "val": 618, "test": 632},
    "bad": {"train": 1314, "val": 178, "test": 164},
}
V116_TRAIN_TARGET_PER_CLASS = 8310

FACTOR_COLUMNS_REQUESTED = [
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "flatline_ratio",
    "sqi_basSQI",
    "non_qrs_diff_p95",
    "non_qrs_rms_ratio",
    "qrs_band_ratio",
    "template_corr",
    "amplitude_entropy",
    "contact_loss_win_ratio",
    "band_0p3_1",
]
RATIO_LIKE = {
    "qrs_visibility",
    "detector_agreement",
    "qrs_band_ratio",
    "template_corr",
    "flatline_ratio",
    "sqi_basSQI",
    "amplitude_entropy",
    "contact_loss_win_ratio",
    "non_qrs_rms_ratio",
    "band_0p3_1",
}
LOCAL_MAP_NAMES = [
    "qrs_event_a",
    "qrs_event_b",
    "baseline",
    "contact",
    "reset",
    "flatline",
    "detail",
]
GM_SUBTYPE_NAMES = [
    "good_clean_core",
    "good_overlap_boundary",
    "good_isolated_low_purity",
    "good_mild_artifact_outlier",
    "good_hard_baseline_lowqrs",
    "medium_clean_core",
    "medium_overlap_boundary",
    "medium_isolated_lowqrs",
    "medium_visible_qrs_detail",
    "medium_outlier_or_bad_boundary",
    "medium_hard_baseline_lowqrs",
]
BAD_SUBTYPE_NAMES = [
    "bad_dense_right_island",
    "bad_detector_template_disagree",
    "bad_baseline_wander_lowfreq",
    "bad_contact_reset_flatline",
    "bad_low_qrs_visibility",
    "bad_highfreq_detail_noise",
    "bad_other_boundary",
]
QUALITY_SUBTYPE_NAMES = GM_SUBTYPE_NAMES + BAD_SUBTYPE_NAMES
QUALITY_SUBTYPE_TO_INT = {name: i for i, name in enumerate(QUALITY_SUBTYPE_NAMES)}
QUALITY_SUBTYPE_CLASS = [
    CLASS_TO_INT["good"] if name.startswith("good_") else CLASS_TO_INT["medium"] if name.startswith("medium_") else CLASS_TO_INT["bad"]
    for name in QUALITY_SUBTYPE_NAMES
]
BOUNDARY_FAMILY_NAMES = ["isolated_lowqrs", "mildartifact_hardbaseline"]
BOUNDARY_SUBTYPE_TO_FAMILY = {
    "good_isolated_low_purity": 0,
    "medium_isolated_lowqrs": 0,
    "good_mild_artifact_outlier": 1,
    "medium_hard_baseline_lowqrs": 1,
}
BOUNDARY_SUBTYPE_TO_LABEL = {
    "good_isolated_low_purity": 0,
    "good_mild_artifact_outlier": 0,
    "medium_isolated_lowqrs": 1,
    "medium_hard_baseline_lowqrs": 1,
}
BOUNDARY_MEDIUM_EVIDENCE_FEATURES = ["baseline_step", "band_0p3_1", "non_qrs_rms_ratio"]
BOUNDARY_GOOD_EVIDENCE_FEATURES = ["sqi_basSQI", "qrs_band_ratio", "template_corr"]
ARTIFACT_TYPE_NAMES = [
    "none",
    "explicit",
    "baseline",
    "flatline",
    "contact",
    "detail",
    "bad_region",
]
GM_SUBTYPE_TO_INT = {name: i for i, name in enumerate(GM_SUBTYPE_NAMES)}
BAD_SUBTYPE_TO_INT = {name: i for i, name in enumerate(BAD_SUBTYPE_NAMES)}
ARTIFACT_TYPE_TO_INT = {name: i for i, name in enumerate(ARTIFACT_TYPE_NAMES)}


def load_dual_module() -> Any:
    spec = importlib.util.spec_from_file_location("clean_but_dualview_for_event_factorized", DUAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {DUAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DUAL = load_dual_module()
TX = DUAL.TX
FEATURES = DUAL.FEATURES
FORMAL_FEATURE_COLUMNS = list(DUAL.FORMAL_FEATURE_COLUMNS)
FACTOR_COLUMNS = [c for c in FACTOR_COLUMNS_REQUESTED if c in FORMAL_FEATURE_COLUMNS]
if "detector_agreement" not in FACTOR_COLUMNS:
    raise RuntimeError("detector_agreement is required in FORMAL_FEATURE_COLUMNS")
FACTOR_IDX = [FORMAL_FEATURE_COLUMNS.index(c) for c in FACTOR_COLUMNS]
DETECTOR_FACTOR_IDX = FACTOR_COLUMNS.index("detector_agreement")


def policy_alias(policy: str) -> str:
    text = str(policy)
    if text.startswith("v116_gapfill_dual_goodorig_nm40_ms10_smc"):
        return "v116gap_smc"
    aliases = {
        "margin_ge_5s_drop_outlier": "mg5_drop",
        "margin_ge_2s_drop_outlier": "mg2_drop",
        "margin_ge_8s_drop_outlier": "mg8_drop",
        "margin_ge_10s_drop_outlier": "mg10_drop",
    }
    return aliases.get(text, text.replace("margin_ge_", "mg").replace("_drop_outlier", "_drop")[:32])


def v116_gapfill_policy(policy: str) -> bool:
    return str(policy).startswith("v116_gapfill_dual_goodorig_nm")


def v116_original_split(original: pd.DataFrame, seed: int) -> pd.Series:
    split = pd.Series("train", index=original.index, dtype=object)
    for cls, counts in V116_ORIGINAL_SPLIT_COUNTS.items():
        idx = original.index[original["class_name"].astype(str).eq(cls)].to_numpy()
        min_needed = int(counts["val"]) + int(counts["test"])
        if len(idx) < min_needed:
            raise ValueError(f"v116 original split needs at least {min_needed} {cls} rows, got {len(idx)}")
        rng = np.random.default_rng(int(seed) + CLASS_TO_INT[cls] * 1009)
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = int(counts["val"])
        n_test = int(counts["test"])
        split.loc[idx[:n_val]] = "val"
        split.loc[idx[n_val : n_val + n_test]] = "test"
    return split


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def df_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(int(max_rows)).copy()
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for c in view.columns:
            val = row[c]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > len(view):
        lines.append(f"\n_Showing {len(view)} of {len(df)} rows._")
    return "\n".join(lines)


def numeric_frame(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    work = frame.copy()
    for col in cols:
        if col not in work.columns:
            work[col] = 0.0
    return (
        work[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


@dataclass
class FactorTransform:
    columns: list[str]
    mean: np.ndarray
    std: np.ndarray

    def transform(self, raw: np.ndarray) -> np.ndarray:
        out = np.zeros((raw.shape[0], len(self.columns)), dtype=np.float32)
        for j, name in enumerate(self.columns):
            col = raw[:, j].astype(np.float32)
            if name in RATIO_LIKE:
                out[:, j] = np.clip(col, 0.0, 1.0)
            else:
                out[:, j] = (col - self.mean[j]) / max(float(self.std[j]), 1e-6)
        return out.astype(np.float32)


def fit_factor_transform(train_frame: pd.DataFrame) -> FactorTransform:
    raw = numeric_frame(train_frame, FACTOR_COLUMNS)
    mean = raw.mean(axis=0).astype(np.float32)
    std = raw.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return FactorTransform(columns=list(FACTOR_COLUMNS), mean=mean, std=std)


@dataclass
class ArtifactThresholds:
    baseline_step: float
    flatline_ratio: float
    contact_loss_win_ratio: float
    non_qrs_diff_p95: float


def fit_artifact_thresholds(train_frame: pd.DataFrame) -> ArtifactThresholds:
    def q(col: str, default: float) -> float:
        if col not in train_frame.columns:
            return default
        vals = pd.to_numeric(train_frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return default
        return float(vals.quantile(0.95))

    return ArtifactThresholds(
        baseline_step=q("baseline_step", 0.0),
        flatline_ratio=q("flatline_ratio", 0.0),
        contact_loss_win_ratio=q("contact_loss_win_ratio", 0.0),
        non_qrs_diff_p95=q("non_qrs_diff_p95", 0.0),
    )


def infer_artifact_targets(frame: pd.DataFrame, y: np.ndarray, thresholds: ArtifactThresholds) -> pd.DataFrame:
    """Infer artifact targets without using clean_policy text.

    artifact_presence means "some artifact evidence is present"; it is not a
    synonym for the bad class. A non-bad row can be artifact-positive while the
    diagnostic bad target remains negative.
    """

    region = frame.get("original_region", pd.Series([""] * len(frame))).fillna("").astype(str).str.lower()
    explicit_cols = [c for c in ["ambiguous_type", "bank_role", "profile_block", "artifact_type", "artifact_source"] if c in frame.columns]
    explicit_text = pd.Series([""] * len(frame), index=frame.index, dtype=object)
    for col in explicit_cols:
        explicit_text = explicit_text + "|" + frame[col].fillna("").astype(str).str.lower()
    explicit_artifact = explicit_text.str.contains("outlier|controlled|contact|flat|reset|artifact|stress|motion|drift", regex=True)

    feature_artifact = pd.Series(False, index=frame.index)
    feature_type = np.asarray(["none"] * len(frame), dtype=object)
    for col, cutoff, label in [
        ("baseline_step", thresholds.baseline_step, "baseline"),
        ("flatline_ratio", thresholds.flatline_ratio, "flatline"),
        ("contact_loss_win_ratio", thresholds.contact_loss_win_ratio, "contact"),
        ("non_qrs_diff_p95", thresholds.non_qrs_diff_p95, "detail"),
    ]:
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            mask = vals.ge(float(cutoff)) & vals.gt(0.0)
            feature_artifact = feature_artifact | mask
            feature_type = np.where((feature_type == "none") & mask.to_numpy(), label, feature_type)

    region_artifact = region.str.contains("outlier|bad_island|bad_boundary|medium_bad", regex=True)
    artifact_presence = explicit_artifact | feature_artifact | region_artifact | (y == CLASS_TO_INT["bad"])
    diagnostic_bad = y == CLASS_TO_INT["bad"]
    severity = (
        0.25 * feature_artifact.astype(float)
        + 0.50 * region_artifact.astype(float)
        + 1.00 * diagnostic_bad.astype(float)
    ).clip(0.0, 1.0)
    artifact_type = np.where(region.str.contains("bad", regex=False).to_numpy(), "bad_region", feature_type)
    artifact_type = np.where(explicit_artifact.to_numpy() & (artifact_type == "none"), "explicit", artifact_type)
    return pd.DataFrame(
        {
            "artifact_presence": artifact_presence.astype(bool).to_numpy(),
            "diagnostic_bad": diagnostic_bad.astype(bool),
            "artifact_type": artifact_type,
            "artifact_severity": np.asarray(severity, dtype=np.float32),
            "artifact_source_explicit": explicit_artifact.astype(bool).to_numpy(),
            "artifact_source_feature": feature_artifact.astype(bool).to_numpy(),
            "artifact_source_region": region_artifact.astype(bool).to_numpy(),
        }
    )


def infer_display_subtype(frame: pd.DataFrame) -> pd.Series:
    """Return a stable subtype label when an experiment protocol provides one."""

    for col in ["display_subtype", "transport_subtype", "subtype_id", "target_subtype", "target_bad_subtype", "computed_subtype", "subtype_for_split"]:
        if col in frame.columns:
            vals = frame[col].fillna("").astype(str)
            mask = vals.ne("") & vals.ne("nan")
            if mask.any():
                out = pd.Series("unknown", index=frame.index, dtype=object)
                out.loc[mask] = vals.loc[mask]
                return out
    return frame.get("original_region", pd.Series(["unknown"] * len(frame))).fillna("unknown").astype(str)


def pseudo_local_targets(x: torch.Tensor) -> dict[str, torch.Tensor]:
    # x channels from DUAL.make_dualview_channels:
    # physical, robust z, dz, baseline_long, highpass, detail_fast, detail_slow, physical_trend.
    dz = x[:, 2]
    baseline = x[:, 3]
    highpass = x[:, 4]
    detail_fast = x[:, 5]
    detail_slow = x[:, 6].abs() + 1e-3
    abs_dz = dz.abs()
    abs_hp = highpass.abs()
    qrs_event_a = torch.sigmoid((abs_dz - 1.20) / 0.30)
    prom = abs_hp / (detail_slow + 0.05)
    qrs_event_b = torch.sigmoid((prom - 1.55) / 0.40)
    contact = torch.sigmoid((0.22 - abs_dz) / 0.055) * torch.sigmoid((0.32 - abs_hp) / 0.075)
    flatline = torch.sigmoid((0.12 - abs_dz) / 0.040) * torch.sigmoid((0.22 - abs_hp) / 0.060)
    reset = torch.sigmoid((abs_dz - 2.50) / 0.45)
    base = torch.tanh(baseline / 2.5)
    detail = torch.tanh(detail_fast / (detail_slow + 0.10))
    valid = torch.ones_like(qrs_event_a)
    return {
        "qrs_event_a": qrs_event_a,
        "qrs_event_b": qrs_event_b,
        "baseline": base,
        "contact": contact,
        "reset": reset,
        "flatline": flatline,
        "detail": detail,
        "valid": valid,
    }


def soft_event_detector_agreement(local_logits: torch.Tensor) -> torch.Tensor:
    a = torch.sigmoid(local_logits[:, 0])
    b = torch.sigmoid(local_logits[:, 1])
    return (1.0 - torch.mean(torch.abs(a - b), dim=1)).clamp(0.0, 1.0)


class CleanProtocolDataset(Dataset):
    def __init__(
        self,
        source_protocol: Path,
        split: str,
        atlas_frame: pd.DataFrame | None = None,
        channel_stats: Any | None = None,
        feature_norm: Any | None = None,
        factor_transform: FactorTransform | None = None,
        artifact_thresholds: ArtifactThresholds | None = None,
        max_rows: int = 0,
        seed: int = 20260620,
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
        train_frame = atlas_all.loc[atlas_all["split"].astype(str).eq("train")].copy()
        if train_frame.empty:
            train_frame = atlas_all.copy()
        if feature_norm is None:
            train_raw = numeric_frame(train_frame, FORMAL_FEATURE_COLUMNS)
            mean = train_raw.mean(axis=0).astype(np.float32)
            std = train_raw.std(axis=0).astype(np.float32)
            std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
            feature_norm = DUAL.FeatureNorm(mean=mean, std=std)
        if factor_transform is None:
            factor_transform = fit_factor_transform(train_frame)
        if artifact_thresholds is None:
            artifact_thresholds = fit_artifact_thresholds(train_frame)
        raw = numeric_frame(self.frame, FORMAL_FEATURE_COLUMNS)
        factor_raw = numeric_frame(self.frame, FACTOR_COLUMNS)
        self.x = DUAL.make_dualview_channels(signals[rows], channel_stats).astype(np.float32)
        self.aux_raw = raw.astype(np.float32)
        self.aux = ((raw - feature_norm.mean[None, :]) / feature_norm.std[None, :]).astype(np.float32)
        self.factor_raw = factor_raw.astype(np.float32)
        self.factor = factor_transform.transform(factor_raw)
        self.y = self.frame["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        targets = infer_artifact_targets(self.frame, self.y, artifact_thresholds)
        self.artifact_presence = targets["artifact_presence"].astype(float).to_numpy(dtype=np.float32)
        self.diagnostic_bad = targets["diagnostic_bad"].astype(float).to_numpy(dtype=np.float32)
        self.artifact_severity = targets["artifact_severity"].astype(float).to_numpy(dtype=np.float32)
        self.artifact_type = targets["artifact_type"].astype(str).to_numpy()
        self.display_subtype = infer_display_subtype(self.frame).astype(str).to_numpy()
        self.gm_subtype = np.full(len(self.frame), -100, dtype=np.int64)
        self.bad_subtype = np.full(len(self.frame), -100, dtype=np.int64)
        self.quality_subtype = np.full(len(self.frame), -100, dtype=np.int64)
        self.boundary_family = np.full(len(self.frame), -100, dtype=np.int64)
        self.boundary_label = np.full(len(self.frame), -100, dtype=np.int64)
        self.artifact_type_target = np.zeros(len(self.frame), dtype=np.int64)
        for row_i, name in enumerate(self.display_subtype):
            if int(self.y[row_i]) in {CLASS_TO_INT["good"], CLASS_TO_INT["medium"]} and name in GM_SUBTYPE_TO_INT:
                self.gm_subtype[row_i] = int(GM_SUBTYPE_TO_INT[name])
            if int(self.y[row_i]) == CLASS_TO_INT["bad"] and name in BAD_SUBTYPE_TO_INT:
                self.bad_subtype[row_i] = int(BAD_SUBTYPE_TO_INT[name])
            if name in QUALITY_SUBTYPE_TO_INT:
                expected_class = int(QUALITY_SUBTYPE_CLASS[QUALITY_SUBTYPE_TO_INT[name]])
                if int(self.y[row_i]) == expected_class:
                    self.quality_subtype[row_i] = int(QUALITY_SUBTYPE_TO_INT[name])
            if name in BOUNDARY_SUBTYPE_TO_FAMILY:
                self.boundary_family[row_i] = int(BOUNDARY_SUBTYPE_TO_FAMILY[name])
                self.boundary_label[row_i] = int(BOUNDARY_SUBTYPE_TO_LABEL[name])
            art_name = str(self.artifact_type[row_i])
            self.artifact_type_target[row_i] = int(ARTIFACT_TYPE_TO_INT.get(art_name, 0))
        self.record_id = self.frame.get("record_id", pd.Series(["unknown"] * len(self.frame))).fillna("unknown").astype(str).to_numpy()
        raw_source_idx = self.frame.get("source_idx", self.frame.get("idx", pd.Series(np.arange(len(self.frame)), index=self.frame.index)))
        fallback_idx = pd.to_numeric(self.frame.get("idx", pd.Series(np.arange(len(self.frame)), index=self.frame.index)), errors="coerce")
        self.source_idx = pd.to_numeric(raw_source_idx, errors="coerce").fillna(fallback_idx).fillna(-1).astype(int).to_numpy()
        self.pair_id = self.source_idx.copy()
        self.channel_stats = channel_stats
        self.feature_norm = feature_norm
        self.factor_transform = factor_transform
        self.artifact_thresholds = artifact_thresholds

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "aux": torch.from_numpy(self.aux[i]),
            "factor": torch.from_numpy(self.factor[i]),
            "factor_raw": torch.from_numpy(self.factor_raw[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
            "artifact_presence": torch.tensor(float(self.artifact_presence[i]), dtype=torch.float32),
            "diagnostic_bad": torch.tensor(float(self.diagnostic_bad[i]), dtype=torch.float32),
            "artifact_severity": torch.tensor(float(self.artifact_severity[i]), dtype=torch.float32),
            "gm_subtype": torch.tensor(int(self.gm_subtype[i]), dtype=torch.long),
            "bad_subtype": torch.tensor(int(self.bad_subtype[i]), dtype=torch.long),
            "quality_subtype": torch.tensor(int(self.quality_subtype[i]), dtype=torch.long),
            "boundary_family": torch.tensor(int(self.boundary_family[i]), dtype=torch.long),
            "boundary_label": torch.tensor(int(self.boundary_label[i]), dtype=torch.long),
            "artifact_type_target": torch.tensor(int(self.artifact_type_target[i]), dtype=torch.long),
            "source_idx": torch.tensor(int(self.source_idx[i]), dtype=torch.long),
            "pair_id": torch.tensor(int(self.pair_id[i]), dtype=torch.long),
            "record_id": self.record_id[i],
        }


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class ConformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.08):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(width)
        self.dwconv = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=17, padding=8, groups=width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
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


class EventFactorizedSQIConformer(nn.Module):
    query_names = [
        "QRS",
        "RR_TEMPLATE",
        "BASELINE",
        "CONTACT_RESET",
        "DETAIL_NOISE",
        "GLOBAL_MORPH",
        "GM_BOUNDARY",
        "BAD_STRESS",
    ]

    def __init__(
        self,
        in_ch: int,
        factor_dim: int,
        width: int = 96,
        layers: int = 4,
        heads: int = 4,
        use_queries: bool = True,
        use_highres_fusion: bool = True,
        use_local_supervision: bool = True,
        use_artifact_aux: bool = True,
        use_hierarchical_head: bool = True,
        branch_upper_block: bool = False,
        dropout: float = 0.08,
    ):
        super().__init__()
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.factor_dim = int(factor_dim)
        self.use_queries = bool(use_queries)
        self.use_highres_fusion = bool(use_highres_fusion)
        self.use_local_supervision = bool(use_local_supervision)
        self.use_artifact_aux = bool(use_artifact_aux)
        self.use_hierarchical_head = bool(use_hierarchical_head)
        self.branch_upper_block = bool(branch_upper_block)
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
        self.pos = PositionalEncoding(width)
        self.query = nn.Parameter(torch.randn(len(self.query_names), width) * 0.02)
        self.blocks = nn.Sequential(*[ConformerBlock(width, heads, dropout) for _ in range(layers)])
        self.physiology_blocks = nn.Sequential(*[ConformerBlock(width, heads, dropout) for _ in range(1 if branch_upper_block else 0)])
        self.artifact_blocks = nn.Sequential(*[ConformerBlock(width, heads, dropout) for _ in range(1 if branch_upper_block else 0)])
        self.pool = AttentionPool(width)
        self.hi_cross_attn = nn.MultiheadAttention(width, heads, dropout=0.05, batch_first=True)
        self.local_heads = nn.Conv1d(width, len(LOCAL_MAP_NAMES), kernel_size=1)
        self.factor_head = nn.Sequential(nn.LayerNorm(width * 6), nn.Linear(width * 6, 192), nn.GELU(), nn.Dropout(dropout), nn.Linear(192, factor_dim))
        self.factor_override = nn.Parameter(torch.zeros(factor_dim), requires_grad=False)
        self.artifact_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.severity_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.bad_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.medium_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.gm_subtype_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(GM_SUBTYPE_NAMES)))
        self.bad_subtype_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(BAD_SUBTYPE_NAMES)))
        self.quality_subtype_head = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, 128), nn.GELU(), nn.Linear(128, len(QUALITY_SUBTYPE_NAMES)))
        self.boundary_family_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(BOUNDARY_FAMILY_NAMES)))
        self.boundary_label_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.artifact_type_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(ARTIFACT_TYPE_NAMES)))
        self.ce_head = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))
        self.recon_head = nn.Conv1d(width, in_ch, kernel_size=1)

    def _maybe_patch_query(self, query_tokens: torch.Tensor, location: str) -> torch.Tensor:
        patch = getattr(self, "_query_patch", None)
        if not patch or str(patch.get("location")) != str(location):
            return query_tokens
        repl = patch["values"].to(device=query_tokens.device, dtype=query_tokens.dtype)
        if repl.ndim == 1:
            repl = repl.unsqueeze(0)
        if repl.shape[0] == 1 and query_tokens.shape[0] != 1:
            repl = repl.expand(query_tokens.shape[0], -1)
        out = query_tokens.clone()
        out[:, int(patch["query_index"]), :] = repl
        return out

    def forward_tokens(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hi_ch = self.hi_stem(x)
        hi_tokens = hi_ch.transpose(1, 2)
        ctx = self.ctx_down(hi_ch).transpose(1, 2)
        b = x.shape[0]
        if self.use_queries:
            q = self.query[None, :, :].expand(b, -1, -1)
            seq = torch.cat([q, ctx], dim=1)
            seq = self.blocks(self.pos(seq))
            query_tokens = seq[:, : len(self.query_names), :]
            context_tokens = seq[:, len(self.query_names) :, :]
        else:
            context_tokens = self.blocks(self.pos(ctx))
            pooled = self.pool(context_tokens)
            query_tokens = pooled[:, None, :].expand(-1, len(self.query_names), -1)
        query_tokens = self._maybe_patch_query(query_tokens, "Z_Q")
        query_tokens_pre_hires = query_tokens
        if self.use_highres_fusion:
            query_delta, _ = self.hi_cross_attn(query_tokens, hi_tokens, hi_tokens, need_weights=False)
            query_tokens = query_tokens + query_delta
        query_tokens = self._maybe_patch_query(query_tokens, "Q_e")
        if self.branch_upper_block:
            phys = self.physiology_blocks(torch.cat([query_tokens[:, :6, :], context_tokens], dim=1))[:, :6, :]
            art = self.artifact_blocks(torch.cat([query_tokens[:, 6:, :], context_tokens], dim=1))[:, :2, :]
            query_tokens = torch.cat([phys, art], dim=1)
        return {
            "hi_ch": hi_ch,
            "hi_tokens": hi_tokens,
            "context_tokens": context_tokens,
            "query_tokens_pre_hires": query_tokens_pre_hires,
            "query_tokens": query_tokens,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tok = self.forward_tokens(x)
        hi_ch = tok["hi_ch"]
        ctx = tok["context_tokens"]
        query = tok["query_tokens"]
        local_logits = self.local_heads(hi_ch)
        detector_agreement = soft_event_detector_agreement(local_logits)
        factor_raw = self.factor_head(query[:, :6, :].reshape(x.shape[0], -1))
        factor_pred = factor_raw.clone()
        # The detector feature used for loss/recovery/report is the same soft
        # event-agreement statistic supervised through local QRS maps.
        factor_pred[:, DETECTOR_FACTOR_IDX] = detector_agreement
        gm_repr = query[:, 6, :]
        bad_repr = query[:, 7, :]
        artifact_logit = self.artifact_head(bad_repr).squeeze(1)
        severity_pred = torch.sigmoid(self.severity_head(bad_repr).squeeze(1))
        gm_subtype_logits = self.gm_subtype_head(gm_repr)
        bad_subtype_logits = self.bad_subtype_head(bad_repr)
        quality_repr = torch.cat([gm_repr, bad_repr], dim=1)
        quality_subtype_logits = self.quality_subtype_head(quality_repr)
        boundary_family_logits = self.boundary_family_head(gm_repr)
        boundary_label_logit = self.boundary_label_head(gm_repr).squeeze(1)
        artifact_type_logits = self.artifact_type_head(bad_repr)
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
            pooled = torch.cat([self.pool(ctx), ctx.mean(dim=1)], dim=1)
            logits = self.ce_head(pooled)
            probs = torch.softmax(logits, dim=1)
            bad_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
            medium_logit = logits[:, 1] - logits[:, 0]
        recon = F.interpolate(self.recon_head(hi_ch), size=x.shape[-1], mode="linear", align_corners=False)
        return {
            "logits": logits,
            "probs": probs,
            "bad_logit": bad_logit,
            "medium_logit": medium_logit,
            "artifact_logit": artifact_logit,
            "artifact_severity": severity_pred,
            "gm_subtype_logits": gm_subtype_logits,
            "bad_subtype_logits": bad_subtype_logits,
            "quality_subtype_logits": quality_subtype_logits,
            "boundary_family_logits": boundary_family_logits,
            "boundary_label_logit": boundary_label_logit,
            "artifact_type_logits": artifact_type_logits,
            "factor_pred": factor_pred,
            "detector_agreement": detector_agreement,
            "local_logits": local_logits,
            "recon": recon,
            "hi_tokens": tok["hi_tokens"],
            "context_tokens": ctx,
            "query_tokens_pre_hires": tok["query_tokens_pre_hires"],
            "query_tokens": query,
        }


def resize_targets(targets: dict[str, torch.Tensor], target_len: int) -> torch.Tensor:
    maps = torch.stack([targets[name] for name in LOCAL_MAP_NAMES], dim=1)
    if maps.shape[-1] != int(target_len):
        maps = F.interpolate(maps, size=int(target_len), mode="linear", align_corners=False)
    return maps


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 1.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    return (bce * ((1.0 - pt).clamp(0.0, 1.0) ** gamma)).mean()


def local_map_loss(out: dict[str, torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    logits = out["local_logits"]
    target_maps = resize_targets(pseudo_local_targets(x), logits.shape[-1])
    qrs = focal_bce_with_logits(logits[:, 0:2], target_maps[:, 0:2])
    contact_reset_flat = focal_bce_with_logits(logits[:, 3:6], target_maps[:, 3:6])
    cont = F.smooth_l1_loss(torch.tanh(logits[:, [2, 6]]), target_maps[:, [2, 6]])
    loss = 0.55 * qrs + 0.25 * contact_reset_flat + 0.20 * cont
    return loss, {
        "local_qrs": float(qrs.detach().cpu()),
        "local_artifact": float(contact_reset_flat.detach().cpu()),
        "local_cont": float(cont.detach().cpu()),
        "detector_agreement_batch": float(out["detector_agreement"].mean().detach().cpu()),
    }


def factor_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    pred = out["factor_pred"]
    target = batch["factor"].to(device=device, dtype=torch.float32)
    losses = []
    parts: dict[str, float] = {}
    for j, name in enumerate(FACTOR_COLUMNS):
        if name in RATIO_LIKE:
            # Probability-form BCE is unsafe under CUDA autocast. Keep only
            # this scalar auxiliary term in float32 so the Conformer forward
            # and the remaining losses can still use AMP.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                prob = pred[:, j].float() if name == "detector_agreement" else torch.sigmoid(pred[:, j].float())
                val = F.binary_cross_entropy(prob.clamp(1e-5, 1.0 - 1e-5), target[:, j].float().clamp(0.0, 1.0))
        else:
            val = F.smooth_l1_loss(pred[:, j], target[:, j])
        losses.append(val)
        parts[f"factor_{name}"] = float(val.detach().cpu())
    return torch.stack(losses).mean(), parts


def class_weight_tensor(cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    weights = cfg.get("class_weights")
    if weights is None:
        return None
    if len(weights) != len(CLASS_TO_INT):
        raise ValueError(f"class_weights must have {len(CLASS_TO_INT)} entries")
    return torch.tensor([float(v) for v in weights], dtype=torch.float32, device=device)


def class_nll(out: dict[str, torch.Tensor], y: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    weights = class_weight_tensor(cfg, device)
    if bool(cfg.get("use_hierarchical_head", True)):
        return F.nll_loss(out["logits"], y, weight=weights)
    return F.cross_entropy(out["logits"], y, weight=weights)


def hierarchical_nll(out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(out["logits"], y)


def artifact_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    artifact_presence = batch["artifact_presence"].to(device=device, dtype=torch.float32)
    diagnostic_bad = batch["diagnostic_bad"].to(device=device, dtype=torch.float32)
    severity = batch["artifact_severity"].to(device=device, dtype=torch.float32)
    artifact_bce = focal_bce_with_logits(out["artifact_logit"], artifact_presence)
    severity_loss = F.smooth_l1_loss(out["artifact_severity"], severity)
    bad_target = diagnostic_bad
    bad_loss = focal_bce_with_logits(out["bad_logit"], bad_target)
    bad_prob = torch.sigmoid(out["bad_logit"])
    artifact_nonbad = (artifact_presence > 0.5) & (diagnostic_bad < 0.5)
    specificity = bad_prob[artifact_nonbad].mean() if torch.any(artifact_nonbad) else torch.zeros((), device=device)
    total = artifact_bce + 0.35 * severity_loss + 0.65 * bad_loss + 0.75 * specificity
    return total, {
        "artifact_bce": float(artifact_bce.detach().cpu()),
        "artifact_severity_loss": float(severity_loss.detach().cpu()),
        "bad_specific_loss": float(bad_loss.detach().cpu()),
        "artifact_nonbad_bad_prob": float(specificity.detach().cpu()),
    }


def subtype_class_probs_from_leaf(quality_logits: torch.Tensor) -> torch.Tensor:
    leaf_probs = torch.softmax(quality_logits, dim=1)
    class_ids = torch.tensor(QUALITY_SUBTYPE_CLASS, dtype=torch.long, device=quality_logits.device)
    class_probs = []
    for cls_idx in range(len(CLASS_TO_INT)):
        class_probs.append(leaf_probs[:, class_ids == cls_idx].sum(dim=1))
    return torch.stack(class_probs, dim=1).clamp(1e-6, 1.0)


def boundary_feature_direction_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    family = batch.get("boundary_family")
    label = batch.get("boundary_label")
    if family is None or label is None:
        return out["logits"].new_tensor(0.0), {}
    family = family.to(device=device).long()
    label = label.to(device=device).long()
    mask = (family >= 0) & (label >= 0)
    if not torch.any(mask):
        return out["logits"].new_tensor(0.0), {}

    losses: list[torch.Tensor] = []
    parts: dict[str, float] = {}
    family_loss = F.cross_entropy(out["boundary_family_logits"][mask], family[mask])
    label_target = label[mask].float()
    label_loss = F.binary_cross_entropy_with_logits(out["boundary_label_logit"][mask], label_target)
    losses.extend([family_loss, label_loss])
    parts["boundary_family_loss"] = float(family_loss.detach().cpu())
    parts["boundary_label_loss"] = float(label_loss.detach().cpu())

    pred = out["factor_pred"]
    medium_terms = [pred[:, FACTOR_COLUMNS.index(name)] for name in BOUNDARY_MEDIUM_EVIDENCE_FEATURES if name in FACTOR_COLUMNS]
    good_terms = [pred[:, FACTOR_COLUMNS.index(name)] for name in BOUNDARY_GOOD_EVIDENCE_FEATURES if name in FACTOR_COLUMNS]
    if medium_terms and good_terms:
        medium_evidence = torch.stack(medium_terms, dim=0).mean(dim=0)
        good_evidence = torch.stack(good_terms, dim=0).mean(dim=0)
        feature_logit = medium_evidence - good_evidence
        feature_loss = F.binary_cross_entropy_with_logits(feature_logit[mask], label_target)
        losses.append(feature_loss)
        parts["boundary_feature_direction_loss"] = float(feature_loss.detach().cpu())

    total = torch.stack(losses).mean()
    parts["boundary_aux_loss"] = float(total.detach().cpu())
    parts["boundary_rows"] = float(mask.sum().detach().cpu())
    return total, parts


def unified_subtype_aux_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch.get("quality_subtype")
    if target is None:
        return out["logits"].new_tensor(0.0), {}
    target = target.to(device=device).long()
    mask = target >= 0
    if not torch.any(mask):
        return out["logits"].new_tensor(0.0), {}

    leaf_loss = F.cross_entropy(out["quality_subtype_logits"], target, ignore_index=-100)
    subtype_class_probs = subtype_class_probs_from_leaf(out["quality_subtype_logits"])
    y = batch["y"].to(device=device).long()
    subtype_class_loss = F.nll_loss(torch.log(subtype_class_probs), y)
    main_probs = out["probs"].clamp(1e-6, 1.0)
    consistency = 0.5 * (
        F.kl_div(torch.log(subtype_class_probs), main_probs.detach(), reduction="batchmean")
        + F.kl_div(torch.log(main_probs), subtype_class_probs.detach(), reduction="batchmean")
    )
    boundary, boundary_parts = boundary_feature_direction_loss(out, batch, device)
    total = leaf_loss + 0.55 * subtype_class_loss + 0.35 * consistency + 0.70 * boundary
    parts = {
        "quality_subtype_leaf_loss": float(leaf_loss.detach().cpu()),
        "quality_subtype_class_loss": float(subtype_class_loss.detach().cpu()),
        "quality_subtype_consistency_loss": float(consistency.detach().cpu()),
        "quality_subtype_aux_loss": float(total.detach().cpu()),
    }
    parts.update(boundary_parts)
    return total, parts


def subtype_aux_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    parts: dict[str, float] = {}
    gm_target = batch.get("gm_subtype")
    if gm_target is not None:
        gm_target = gm_target.to(device=device).long()
        if (gm_target >= 0).any():
            gm_loss = F.cross_entropy(out["gm_subtype_logits"], gm_target, ignore_index=-100)
            losses.append(gm_loss)
            parts["gm_subtype_loss"] = float(gm_loss.detach().cpu())
    bad_target = batch.get("bad_subtype")
    if bad_target is not None:
        bad_target = bad_target.to(device=device).long()
        if (bad_target >= 0).any():
            bad_loss = F.cross_entropy(out["bad_subtype_logits"], bad_target, ignore_index=-100)
            losses.append(bad_loss)
            parts["bad_subtype_loss"] = float(bad_loss.detach().cpu())
    artifact_target = batch.get("artifact_type_target")
    if artifact_target is not None:
        artifact_target = artifact_target.to(device=device).long()
        artifact_type = F.cross_entropy(out["artifact_type_logits"], artifact_target)
        losses.append(artifact_type)
        parts["artifact_type_loss"] = float(artifact_type.detach().cpu())
    if not losses:
        return out["logits"].new_tensor(0.0), parts
    total = torch.stack(losses).mean()
    parts["subtype_aux_loss"] = float(total.detach().cpu())
    return total, parts


def model_loss(model: nn.Module, batch: dict[str, torch.Tensor], cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device)
    out = model(x)
    cls = class_nll(out, y, cfg, device)
    factor, factor_parts = factor_loss(out, batch, device)
    local = torch.zeros((), device=device)
    local_parts: dict[str, float] = {}
    if bool(cfg.get("use_local_supervision", True)):
        local, local_parts = local_map_loss(out, x)
    art = torch.zeros((), device=device)
    art_parts: dict[str, float] = {}
    if bool(cfg.get("use_artifact_aux", True)):
        art, art_parts = artifact_loss(out, batch, device)
    subtype = torch.zeros((), device=device)
    subtype_parts: dict[str, float] = {}
    if float(cfg.get("subtype_weight", 0.0)) > 0.0:
        if bool(cfg.get("use_unified_subtype_head", False)):
            subtype, subtype_parts = unified_subtype_aux_loss(out, batch, device)
        else:
            subtype, subtype_parts = subtype_aux_loss(out, batch, device)
    total = (
        cls
        + float(cfg.get("factor_weight", 0.30)) * factor
        + float(cfg.get("local_weight", 0.25)) * local
        + float(cfg.get("artifact_weight", 0.25)) * art
        + float(cfg.get("subtype_weight", 0.0)) * subtype
    )
    parts = {"loss_total": float(total.detach().cpu()), "loss_cls": float(cls.detach().cpu()), "loss_factor": float(factor.detach().cpu()), "loss_local": float(local.detach().cpu()), "loss_artifact": float(art.detach().cpu()), "loss_subtype": float(subtype.detach().cpu())}
    parts.update({k: v for k, v in factor_parts.items() if k in {"factor_detector_agreement", "factor_qrs_visibility", "factor_sqi_basSQI"}})
    parts.update(local_parts)
    parts.update(art_parts)
    parts.update(subtype_parts)
    return total, parts, out


def load_recordheldout_frame(split_root: Path, fold: int) -> tuple[Path, pd.DataFrame]:
    fold_dir = split_root / f"fold{fold}"
    meta = json.loads((fold_dir / "source_protocol.json").read_text(encoding="utf-8"))
    return Path(meta["source_protocol"]), pd.read_csv(fold_dir / "original_region_atlas.csv")


def build_recordheldout_splits(policy: str, folds: int, seed: int) -> Path:
    source = DUAL.PROTOCOL_ROOT / policy
    atlas = pd.read_csv(source / "original_region_atlas.csv")
    if "record_id" not in atlas.columns:
        raise ValueError("record_id is required for record-heldout split")
    candidate_type = atlas.get("v116_candidate_type", pd.Series(["original_but"] * len(atlas), index=atlas.index)).fillna("original_but").astype(str)
    original_mask = candidate_type.eq("original_but").to_numpy()
    original = atlas.loc[original_mask].copy()
    if original.empty:
        raise ValueError("record-heldout split requires original_but rows")
    source_to_record = dict(zip(original["source_idx"].astype(str), original["record_id"].astype(str))) if "source_idx" in original.columns else {}
    idx_to_record = dict(zip(original["idx"].astype(str), original["record_id"].astype(str))) if "idx" in original.columns else {}
    link_record = atlas["record_id"].astype(str).copy()
    for col in ["v116_native_donor_id", "v116_style_donor_id", "v114_donor_source_idx"]:
        if col not in atlas.columns:
            continue
        donor = pd.to_numeric(atlas[col], errors="coerce")
        mapped = donor.map(lambda v: source_to_record.get(str(int(v)), idx_to_record.get(str(int(v)))) if np.isfinite(v) else None)
        link_record = link_record.where(mapped.isna(), mapped.astype(str))
    if "v116_residual_donor_id" in atlas.columns:
        donor_record = pd.to_numeric(atlas["v116_residual_donor_id"], errors="coerce")
        mapped = donor_record.map(lambda v: str(int(v)) if np.isfinite(v) else None)
        link_record = link_record.where(mapped.isna(), mapped.astype(str))
    groups = original["record_id"].astype(str).to_numpy()
    y = original["class_name"].astype(str).map(CLASS_TO_INT).to_numpy()
    root = OUT_DIR / "rh_splits" / f"{policy_alias(policy)}_k{folds}_s{seed}"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    if int(folds) <= 1:
        original_split = v116_original_split(original, int(seed)) if v116_gapfill_policy(policy) else original["split"].astype(str)
        source_to_split = dict(zip(original["source_idx"].astype(str), original_split.astype(str))) if "source_idx" in original.columns else {}
        idx_to_split = dict(zip(original["idx"].astype(str), original_split.astype(str))) if "idx" in original.columns else {}
        record_to_split = (
            pd.DataFrame({"record_id": original["record_id"].astype(str), "split": original_split.astype(str)})
            .groupby("record_id")["split"]
            .agg(lambda s: str(s.mode().iloc[0]))
            .to_dict()
        )

        def map_to_original_split(values: pd.Series) -> pd.Series:
            keys = values.fillna("").astype(str)
            mapped = keys.map(source_to_split).combine_first(keys.map(idx_to_split)).combine_first(keys.map(record_to_split))
            numeric = pd.to_numeric(keys, errors="coerce")
            numeric_keys = numeric.map(lambda v: str(int(v)) if np.isfinite(v) else None)
            return mapped.combine_first(numeric_keys.map(source_to_split)).combine_first(numeric_keys.map(idx_to_split)).combine_first(numeric_keys.map(record_to_split))

        link_split = pd.Series("unused", index=atlas.index, dtype=object)
        link_split.loc[original.index] = original_split.astype(str)
        for col in ["v116_native_donor_id", "v116_style_donor_id", "v114_donor_source_idx", "v116_residual_donor_id"]:
            if col not in atlas.columns:
                continue
            mapped = map_to_original_split(atlas[col])
            mapped = mapped.mask(original_mask)
            link_split = link_split.where(mapped.isna(), mapped.astype(str))
        split = np.asarray(["unused"] * len(atlas), dtype=object)
        source_split = link_split.astype(str).to_numpy()
        split[original_mask & np.isin(source_split, ["train", "val", "test"])] = source_split[original_mask & np.isin(source_split, ["train", "val", "test"])]
        split[~original_mask & link_split.astype(str).eq("train").to_numpy()] = "train"
        train_counts = atlas.loc[split == "train", "class_name"].astype(str).value_counts()
        target_n = V116_TRAIN_TARGET_PER_CLASS if v116_gapfill_policy(policy) else int(train_counts.min())
        if int(train_counts.min()) < int(target_n):
            raise ValueError(f"Cannot build {policy} split: smallest train class has {int(train_counts.min())}, target is {target_n}")

        def referenced_train_generated_keys() -> set[str]:
            refs: set[str] = set()
            gen = atlas.loc[(split == "train") & ~original_mask]
            for col in ["v116_native_donor_id", "v116_style_donor_id", "v114_donor_source_idx", "v116_residual_donor_id"]:
                if col not in gen.columns:
                    continue
                for raw in gen[col].dropna().astype(str):
                    if not raw or raw.lower() == "nan":
                        continue
                    refs.add(raw)
                    num = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
                    if np.isfinite(num):
                        refs.add(str(int(num)))
            return refs

        def unreferenced_original_first(indices: np.ndarray, refs: set[str]) -> np.ndarray:
            safe: list[int] = []
            rest: list[int] = []
            for j in indices:
                row = atlas.iloc[int(j)]
                keys = {str(row.get("source_idx", "")), str(row.get("idx", "")), str(row.get("record_id", ""))}
                (rest if keys & refs else safe).append(int(j))
            return np.asarray(safe + rest, dtype=int)

        for cls, count in train_counts.items():
            surplus = int(count) - target_n
            if surplus <= 0:
                continue
            cls_idx = np.flatnonzero((split == "train") & atlas["class_name"].astype(str).eq(str(cls)).to_numpy())
            gen_idx = cls_idx[~original_mask[cls_idx]]
            orig_idx = cls_idx[original_mask[cls_idx]]
            rng = np.random.default_rng(int(seed) + CLASS_TO_INT.get(str(cls), 9) * 1009)
            drop_parts = []
            if len(gen_idx):
                take = min(surplus, len(gen_idx))
                drop_parts.append(rng.choice(gen_idx, size=take, replace=False))
                surplus -= take
            if surplus > 0:
                orig_idx = unreferenced_original_first(orig_idx, referenced_train_generated_keys())
                drop_parts.append(orig_idx[:surplus])
            if drop_parts:
                split[np.concatenate(drop_parts)] = "unused"
        frame = atlas.copy()
        frame["split"] = split
        fold_dir = root / "fold0"
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(fold_dir / "original_region_atlas.csv", index=False)
        (fold_dir / "source_protocol.json").write_text(json.dumps({"source_protocol": str(source), "signals": str(source / "signals.npz")}, indent=2), encoding="utf-8")
        for part in ["train", "val", "test", "unused"]:
            part_frame = frame.loc[frame["split"].eq(part)]
            class_counts = part_frame["class_name"].value_counts().to_dict()
            rows.append({"fold": 0, "split": part, "rows": int(len(part_frame)), "records": int(part_frame["record_id"].nunique()), **{f"{k}_rows": int(v) for k, v in class_counts.items()}})
        pd.DataFrame(rows).to_csv(root / "recordheldout_split_summary.csv", index=False)
        return root

    gkf = StratifiedGroupKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    original_pos = np.flatnonzero(original_mask)
    linked = link_record.astype(str).to_numpy()
    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(original, y, groups)):
        rng = np.random.default_rng(int(seed) + int(fold))
        train_val_idx = np.asarray(train_val_idx)
        train_val_records = np.unique(groups[train_val_idx])
        inner_splits = max(2, min(5, len(train_val_records)))
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=int(seed) + int(fold))
        _, val_rel = next(inner.split(original.iloc[train_val_idx], y[train_val_idx], groups[train_val_idx]))
        val_records = np.unique(groups[train_val_idx[np.asarray(val_rel)]])
        test_records = np.unique(groups[np.asarray(test_idx)])
        split = np.asarray(["train"] * len(atlas), dtype=object)
        split[~original_mask & (np.isin(linked, test_records) | np.isin(linked, val_records))] = "unused"
        split[original_pos[np.asarray(test_idx)]] = "test"
        split[original_mask & np.isin(linked, val_records)] = "val"
        frame = atlas.copy()
        frame["split"] = split
        fold_dir = root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(fold_dir / "original_region_atlas.csv", index=False)
        (fold_dir / "source_protocol.json").write_text(json.dumps({"source_protocol": str(source), "signals": str(source / "signals.npz")}, indent=2), encoding="utf-8")
        for part in ["train", "val", "test", "unused"]:
            part_frame = frame.loc[frame["split"].eq(part)]
            class_counts = part_frame["class_name"].value_counts().to_dict()
            rows.append({"fold": fold, "split": part, "rows": int(len(part_frame)), "records": int(part_frame["record_id"].nunique()), **{f"{k}_rows": int(v) for k, v in class_counts.items()}})
    pd.DataFrame(rows).to_csv(root / "recordheldout_split_summary.csv", index=False)
    return root


def record_balanced_sampler(ds: CleanProtocolDataset, seed: int) -> WeightedRandomSampler:
    rec = pd.Series(ds.record_id).astype(str)
    y = pd.Series(ds.y).astype(int)
    counts = pd.DataFrame({"record": rec, "y": y}).groupby(["record", "y"]).size()
    weights = []
    for r, cls in zip(rec, y):
        weights.append(1.0 / float(max(1, counts.loc[(r, cls)])))
    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = weights_np / weights_np.mean()
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(weights_np, dtype=torch.double), num_samples=len(weights_np), replacement=True, generator=gen)


def make_loaders(args: argparse.Namespace, split_root: Path | None = None, fold: int = 0) -> tuple[DataLoader, DataLoader, DataLoader | None, CleanProtocolDataset]:
    if split_root is None:
        source = DUAL.PROTOCOL_ROOT / str(args.policy)
        atlas_frame = None
    else:
        source, atlas_frame = load_recordheldout_frame(split_root, int(fold))
    train_ds = CleanProtocolDataset(source, "train", atlas_frame=atlas_frame, max_rows=int(args.max_train_rows), seed=int(args.seed))
    val_ds = CleanProtocolDataset(
        source,
        "val",
        atlas_frame=atlas_frame,
        channel_stats=train_ds.channel_stats,
        feature_norm=train_ds.feature_norm,
        factor_transform=train_ds.factor_transform,
        artifact_thresholds=train_ds.artifact_thresholds,
        max_rows=int(args.max_val_rows),
        seed=int(args.seed) + 1,
    )
    test_ds = None
    if not bool(getattr(args, "skip_test", False)):
        test_ds = CleanProtocolDataset(
            source,
            "test",
            atlas_frame=atlas_frame,
            channel_stats=train_ds.channel_stats,
            feature_norm=train_ds.feature_norm,
            factor_transform=train_ds.factor_transform,
            artifact_thresholds=train_ds.artifact_thresholds,
            max_rows=int(args.max_test_rows),
            seed=int(args.seed) + 2,
        )
    pin = torch.cuda.is_available()
    sampler = record_balanced_sampler(train_ds, int(args.seed)) if bool(args.record_balanced_sampler) else None
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=(sampler is None), sampler=sampler, num_workers=int(args.num_workers), pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size) * 2, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    test_loader = None if test_ds is None else DataLoader(test_ds, batch_size=int(args.batch_size) * 2, shuffle=False, num_workers=int(args.num_workers), pin_memory=pin)
    return train_loader, val_loader, test_loader, train_ds


def basic_metric(y_true: np.ndarray, probs: np.ndarray, supported_only: bool = False) -> dict[str, Any]:
    y_pred = np.argmax(probs, axis=1)
    labels = [0, 1, 2] if not supported_only else [c for c in [0, 1, 2] if np.any(y_true == c)]
    labels = labels or [0, 1, 2]
    rep = FEATURES.metric_report(y_true, y_pred, probs)
    rep["macro_f1_sklearn"] = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    rep["supported_labels"] = ",".join(INT_TO_CLASS[c] for c in labels)
    nonbad = y_true != CLASS_TO_INT["bad"]
    rep["bad_fpr_nonbad"] = float(np.mean(y_pred[nonbad] == CLASS_TO_INT["bad"])) if np.any(nonbad) else np.nan
    return rep


def metric_row(candidate: str, bucket: str, rep: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "bucket": bucket}
    for key, val in rep.items():
        row[key] = json.dumps(val) if key == "confusion_3x3" else val
    return row


def record_metric_rows(candidate: str, y: np.ndarray, probs: np.ndarray, record_ids: np.ndarray, artifact_presence: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for record in sorted(pd.unique(pd.Series(record_ids).astype(str))):
        mask = record_ids.astype(str) == str(record)
        full = basic_metric(y[mask], probs[mask], supported_only=False)
        supported = basic_metric(y[mask], probs[mask], supported_only=True)
        y_pred = np.argmax(probs[mask], axis=1)
        art_nonbad = (artifact_presence[mask] > 0.5) & (y[mask] != CLASS_TO_INT["bad"])
        row = {
            "candidate": candidate,
            "record_id": str(record),
            "rows": int(np.sum(mask)),
            "good_rows": int(np.sum(y[mask] == 0)),
            "medium_rows": int(np.sum(y[mask] == 1)),
            "bad_rows": int(np.sum(y[mask] == 2)),
            "full_macro_f1": float(full["macro_f1_sklearn"]),
            "supported_macro_f1": float(supported["macro_f1_sklearn"]),
            "acc": float(full["acc"]),
            "good_recall": float(full["good_recall"]),
            "medium_recall": float(full["medium_recall"]),
            "bad_recall": float(full["bad_recall"]),
            "artifact_positive_nonbad": int(np.sum(art_nonbad)),
            "artifact_positive_nonbad_bad_fpr": float(np.mean(y_pred[art_nonbad] == 2)) if np.any(art_nonbad) else np.nan,
        }
        rows.append(row)
    rec_df = pd.DataFrame(rows)
    bad_rec = rec_df.loc[rec_df["bad_rows"].gt(0)]
    summary = {
        "record_macro_acc": float(rec_df["acc"].mean()) if not rec_df.empty else np.nan,
        "record_macro_supported_f1": float(rec_df["supported_macro_f1"].mean()) if not rec_df.empty else np.nan,
        "record_macro_full_f1": float(rec_df["full_macro_f1"].mean()) if not rec_df.empty else np.nan,
        "bad_record_count": int(len(bad_rec)),
        "bad_containing_record_bad_recall_mean": float(bad_rec["bad_recall"].mean()) if not bad_rec.empty else np.nan,
        "bad_containing_record_acc_mean": float(bad_rec["acc"].mean()) if not bad_rec.empty else np.nan,
    }
    return rows, summary


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device, candidate: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    factor_targets: list[np.ndarray] = []
    artifacts: list[np.ndarray] = []
    records: list[np.ndarray] = []
    quality_targets: list[np.ndarray] = []
    quality_probs: list[np.ndarray] = []
    boundary_families: list[np.ndarray] = []
    boundary_family_preds: list[np.ndarray] = []
    boundary_labels: list[np.ndarray] = []
    boundary_label_probs: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x)
        p = out["probs"] if "probs" in out else torch.softmax(out["logits"], dim=1)
        ys.append(batch["y"].numpy())
        probs.append(p.detach().cpu().numpy())
        factors.append(out["factor_pred"].detach().cpu().numpy())
        factor_targets.append(batch["factor"].numpy())
        artifacts.append(batch["artifact_presence"].numpy())
        records.append(np.asarray(batch["record_id"], dtype=object))
        quality_targets.append(batch["quality_subtype"].numpy())
        quality_probs.append(torch.softmax(out["quality_subtype_logits"], dim=1).detach().cpu().numpy())
        boundary_families.append(batch["boundary_family"].numpy())
        boundary_family_preds.append(torch.argmax(out["boundary_family_logits"], dim=1).detach().cpu().numpy())
        boundary_labels.append(batch["boundary_label"].numpy())
        boundary_label_probs.append(torch.sigmoid(out["boundary_label_logit"]).detach().cpu().numpy())
    y_np = np.concatenate(ys)
    p_np = np.concatenate(probs)
    pred_factor = np.concatenate(factors)
    true_factor = np.concatenate(factor_targets)
    artifact_np = np.concatenate(artifacts)
    record_np = np.concatenate(records).astype(str)
    quality_target_np = np.concatenate(quality_targets)
    quality_prob_np = np.concatenate(quality_probs)
    boundary_family_np = np.concatenate(boundary_families)
    boundary_family_pred_np = np.concatenate(boundary_family_preds)
    boundary_label_np = np.concatenate(boundary_labels)
    boundary_label_prob_np = np.concatenate(boundary_label_probs)
    rep = basic_metric(y_np, p_np, supported_only=False)
    y_pred = np.argmax(p_np, axis=1)
    art_nonbad = (artifact_np > 0.5) & (y_np != CLASS_TO_INT["bad"])
    rep["artifact_positive_nonbad_count"] = int(np.sum(art_nonbad))
    rep["artifact_positive_nonbad_bad_fpr"] = float(np.mean(y_pred[art_nonbad] == CLASS_TO_INT["bad"])) if np.any(art_nonbad) else np.nan
    rep["factor_mae"] = float(np.mean(np.abs(pred_factor - true_factor)))
    q_mask = quality_target_np >= 0
    q_pred = np.argmax(quality_prob_np, axis=1)
    rep["quality_subtype_rows"] = int(np.sum(q_mask))
    rep["quality_subtype_acc"] = float(np.mean(q_pred[q_mask] == quality_target_np[q_mask])) if np.any(q_mask) else np.nan
    subtype_class_probs = np.zeros((quality_prob_np.shape[0], 3), dtype=np.float32)
    class_ids = np.asarray(QUALITY_SUBTYPE_CLASS, dtype=np.int64)
    for cls_idx in range(3):
        subtype_class_probs[:, cls_idx] = quality_prob_np[:, class_ids == cls_idx].sum(axis=1)
    rep["quality_subtype_class_acc"] = float(np.mean(np.argmax(subtype_class_probs, axis=1) == y_np)) if len(y_np) else np.nan
    b_mask = (boundary_family_np >= 0) & (boundary_label_np >= 0)
    b_pred = (boundary_label_prob_np >= 0.5).astype(np.int64)
    rep["boundary_four_rows"] = int(np.sum(b_mask))
    rep["boundary_label_acc"] = float(np.mean(b_pred[b_mask] == boundary_label_np[b_mask])) if np.any(b_mask) else np.nan
    rep["boundary_family_acc"] = float(np.mean(boundary_family_pred_np[b_mask] == boundary_family_np[b_mask])) if np.any(b_mask) else np.nan
    if np.any(b_mask):
        recalls = []
        for lbl in [0, 1]:
            lm = b_mask & (boundary_label_np == lbl)
            if np.any(lm):
                recalls.append(float(np.mean(b_pred[lm] == lbl)))
        rep["boundary_label_balanced_acc"] = float(np.mean(recalls)) if recalls else np.nan
    else:
        rep["boundary_label_balanced_acc"] = np.nan
    rec_rows, rec_summary = record_metric_rows(candidate, y_np, p_np, record_np, artifact_np)
    rep.update(rec_summary)
    extra = {
        "quality_subtype_target": quality_target_np,
        "quality_subtype_probs": quality_prob_np,
        "boundary_family": boundary_family_np,
        "boundary_family_pred": boundary_family_pred_np,
        "boundary_label": boundary_label_np,
        "boundary_label_prob": boundary_label_prob_np,
        "subtype_class_probs": subtype_class_probs,
    }
    return rep, p_np, pred_factor, true_factor, y_np, artifact_np, rec_rows, rec_summary, extra


def factor_recovery_rows(candidate: str, bucket: str, pred: np.ndarray, true: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for j, name in enumerate(FACTOR_COLUMNS):
        a = pred[:, j]
        b = true[:, j]
        corr_all = 0.0 if np.std(a) < 1e-8 or np.std(b) < 1e-8 else float(np.corrcoef(a, b)[0, 1])
        row = {"candidate": candidate, "bucket": bucket, "feature": name, "corr_all": corr_all, "mae": float(np.mean(np.abs(a - b)))}
        vals = []
        for cls, idx in CLASS_TO_INT.items():
            mask = y == idx
            if np.sum(mask) >= 3 and np.std(a[mask]) > 1e-8 and np.std(b[mask]) > 1e-8:
                corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
            else:
                corr = np.nan
            row[f"corr_{cls}"] = corr
            if not np.isnan(corr):
                vals.append(corr)
        row["corr_min_supported_class"] = float(min(vals)) if vals else np.nan
        rows.append(row)
    return rows


def gradient_snapshot(model: nn.Module, batch: dict[str, torch.Tensor], cfg: dict[str, Any], device: torch.device, epoch: int, max_batches_seen: int) -> list[dict[str, Any]]:
    losses: dict[str, torch.Tensor] = {}
    total, parts, out = model_loss(model, batch, cfg, device)
    x = batch["x"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device)
    losses["class"] = class_nll(out, y, cfg, device)
    losses["factor"] = factor_loss(out, batch, device)[0]
    if bool(cfg.get("use_local_supervision", True)):
        losses["local"] = local_map_loss(out, x)[0]
    if bool(cfg.get("use_artifact_aux", True)):
        losses["artifact"] = artifact_loss(out, batch, device)[0]
    if float(cfg.get("subtype_weight", 0.0)) > 0.0:
        losses["subtype"] = unified_subtype_aux_loss(out, batch, device)[0] if bool(cfg.get("use_unified_subtype_head", False)) else subtype_aux_loss(out, batch, device)[0]
    layers = {
        "hi_stem": "hi_stem",
        "ctx_down": "ctx_down",
        "lower_conformer": "blocks",
        "heads": "head",
    }
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rows: list[dict[str, Any]] = []
    grads_by_loss: dict[str, list[torch.Tensor]] = {}
    for loss_name, loss in losses.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        flat_all = []
        for param_name, param in named_params:
            grad = torch.zeros_like(param).flatten() if param.grad is None else param.grad.detach().flatten()
            flat_all.append(grad)
            group = "heads"
            if param_name.startswith("hi_stem"):
                group = "hi_stem"
            elif param_name.startswith("ctx_down"):
                group = "ctx_down"
            elif param_name.startswith("blocks") or param_name.startswith("physiology_blocks") or param_name.startswith("artifact_blocks"):
                group = "lower_conformer"
            rows.append(
                {
                    "epoch": epoch,
                    "batch_index": max_batches_seen,
                    "loss": loss_name,
                    "layer_group": group,
                    "grad_norm": float(torch.norm(grad).detach().cpu()),
                    "raw_loss": float(loss.detach().cpu()),
                }
            )
        grads_by_loss[loss_name] = flat_all
    names = list(grads_by_loss)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ga = torch.cat(grads_by_loss[a])
            gb = torch.cat(grads_by_loss[b])
            rows.append(
                {
                    "epoch": epoch,
                    "batch_index": max_batches_seen,
                    "loss": f"{a}__vs__{b}",
                    "layer_group": "all_shared_effective",
                    "grad_norm": float(torch.norm(ga).detach().cpu()),
                    "grad_norm_b": float(torch.norm(gb).detach().cpu()),
                    "cosine": float(F.cosine_similarity(ga[None, :], gb[None, :]).item()),
                }
            )
    model.zero_grad(set_to_none=True)
    return rows


def pcgrad_backward(losses: dict[str, torch.Tensor], model: nn.Module) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    for loss in losses.values():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grads.append([torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params])
    for i in range(len(grads)):
        gi_flat = torch.cat([g.flatten() for g in grads[i]])
        for j in range(len(grads)):
            if i == j:
                continue
            gj_flat = torch.cat([g.flatten() for g in grads[j]])
            dot = torch.dot(gi_flat, gj_flat)
            if dot < 0:
                denom = torch.dot(gj_flat, gj_flat).clamp_min(1e-12)
                coeff = dot / denom
                for k in range(len(params)):
                    grads[i][k] = grads[i][k] - coeff * grads[j][k]
    model.zero_grad(set_to_none=True)
    for p_idx, p in enumerate(params):
        p.grad = torch.stack([g[p_idx] for g in grads], dim=0).mean(dim=0)


def cagrad_light_backward(losses: dict[str, torch.Tensor], model: nn.Module, conflict_scale: float = 0.50) -> None:
    """Conflict-averse gradient composition for the phase-2 optimizer ablation.

    This is a compact CAGrad-style variant: task gradients are measured against
    the mean gradient, tasks that conflict with the mean receive lower simplex
    weight, and the weighted gradient is blended back with the mean. It is meant
    as an honest conflict-averse comparator, not a full optimizer replacement.
    """

    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    flats = []
    for loss in losses.values():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        task_grad = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
        grads.append(task_grad)
        flats.append(torch.cat([g.flatten() for g in task_grad]))
    mat = torch.stack(flats, dim=0)
    mean = mat.mean(dim=0)
    denom = (torch.norm(mat, dim=1) * torch.norm(mean)).clamp_min(1e-12)
    cosine = torch.mv(mat, mean) / denom
    weights = torch.softmax(float(conflict_scale) * cosine, dim=0)
    weighted_flat = torch.sum(mat * weights[:, None], dim=0)
    final_flat = 0.50 * mean + 0.50 * weighted_flat
    model.zero_grad(set_to_none=True)
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = final_flat[offset : offset + n].view_as(p).clone()
        offset += n


def candidate_grid(stage: str, seed: int) -> dict[str, dict[str, Any]]:
    base = {
        "width": 96,
        "layers": 3,
        "heads": 4,
        "lr": 2.0e-4,
        "factor_weight": 0.30,
        "local_weight": 0.25,
        "artifact_weight": 0.25,
        "seed": int(seed),
        "use_hierarchical_head": True,
        "use_unified_subtype_head": False,
        "optimizer_mode": "ordinary",
        "pretrain_mode": "none",
        "pretrain_epochs": 0,
    }
    if stage == "phase1":
        return {
            "E0_noquery_nohi_nolocal_noart": {**base, "use_queries": False, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False},
            "E1_query_only": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False},
            "E1_query_only_subtype_aux": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "subtype_weight": 0.18},
            "E1_query_only_unified_subtype_tiny": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "use_unified_subtype_head": True, "subtype_weight": 0.05},
            "E1_query_only_unified_subtype_low": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "use_unified_subtype_head": True, "subtype_weight": 0.10},
            "E1_query_only_unified_subtype_low_lr1e4": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "use_unified_subtype_head": True, "subtype_weight": 0.10, "lr": 1.0e-4},
            "E1_query_only_unified_subtype_low_lr15e4": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "use_unified_subtype_head": True, "subtype_weight": 0.10, "lr": 1.5e-4},
            "E1_query_only_unified_subtype_low_lr75e5": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "use_unified_subtype_head": True, "subtype_weight": 0.10, "lr": 7.5e-5},
            "E1_query_only_unified_subtype_mid": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "use_unified_subtype_head": True, "subtype_weight": 0.18},
            "E1_query_only_unified_subtype": {**base, "use_queries": True, "use_highres_fusion": False, "use_local_supervision": False, "use_artifact_aux": False, "use_unified_subtype_head": True, "subtype_weight": 0.28},
            "E2_query_highres": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": False, "use_artifact_aux": False},
            "E3_query_highres_local": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": False},
            "E4_query_highres_local_art": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True},
            "E4_query_highres_local_art_subtype_aux": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "subtype_weight": 0.18},
            "E4_query_highres_local_art_unified_subtype": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "use_unified_subtype_head": True, "subtype_weight": 0.28},
            "E4_query_highres_local_art_unified_lowaux_lr15e4": {
                **base,
                "use_queries": True,
                "use_highres_fusion": True,
                "use_local_supervision": True,
                "use_artifact_aux": True,
                "use_unified_subtype_head": True,
                "subtype_weight": 0.16,
                "lr": 1.5e-4,
            },
            "E4_query_highres_local_art_unified_factorprotect_lr15e4": {
                **base,
                "use_queries": True,
                "use_highres_fusion": True,
                "use_local_supervision": True,
                "use_artifact_aux": True,
                "use_unified_subtype_head": True,
                "factor_weight": 0.38,
                "local_weight": 0.22,
                "artifact_weight": 0.18,
                "subtype_weight": 0.16,
                "lr": 1.5e-4,
            },
            "E4_query_highres_local_art_unified_gmweighted_lr15e4": {
                **base,
                "use_queries": True,
                "use_highres_fusion": True,
                "use_local_supervision": True,
                "use_artifact_aux": True,
                "use_unified_subtype_head": True,
                "factor_weight": 0.34,
                "local_weight": 0.22,
                "artifact_weight": 0.18,
                "subtype_weight": 0.16,
                "class_weights": [1.30, 1.12, 0.78],
                "lr": 1.5e-4,
            },
            "E4_query_highres_local_art_subtype_aux_lr1e4": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "subtype_weight": 0.18, "lr": 1.0e-4},
            "E4_query_highres_local_art_subtype_aux_lr15e4": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "subtype_weight": 0.18, "lr": 1.5e-4},
            "E4_query_highres_local_art_subtype_aux_lr75e5": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "subtype_weight": 0.18, "lr": 7.5e-5},
            "E4_query_highres_local_art_subtype_aux_lowaux_lr15e4": {
                **base,
                "use_queries": True,
                "use_highres_fusion": True,
                "use_local_supervision": True,
                "use_artifact_aux": True,
                "factor_weight": 0.18,
                "local_weight": 0.18,
                "artifact_weight": 0.20,
                "subtype_weight": 0.12,
                "lr": 1.5e-4,
            },
        }
    if stage == "phase2":
        return {
            "O0_e4_ordinary": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "optimizer_mode": "ordinary"},
            "O1_e4_pcgrad_class_artifact": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "optimizer_mode": "pcgrad_class_artifact"},
            "O2_e4_cagrad_light": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "optimizer_mode": "cagrad_light"},
            "O3_e4_upperblock_branch": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "branch_upper_block": True},
        }
    if stage == "phase3":
        return {
            "P0_no_pretrain": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True},
            "P1_generic_mask": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "pretrain_mode": "generic_mask", "pretrain_epochs": 2},
            "P2_ecg_beat_rhythm_mask": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "pretrain_mode": "ecg_mask", "pretrain_epochs": 2},
            "P2_ecg_beat_rhythm_mask_subtype_aux": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "pretrain_mode": "ecg_mask", "pretrain_epochs": 2, "subtype_weight": 0.18},
            "P2_ecg_beat_rhythm_mask_lowfactor_subtype": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "pretrain_mode": "ecg_mask", "pretrain_epochs": 2, "subtype_weight": 0.18, "factor_weight": 0.12},
            "P3_clean_noisy_physio_distill": {**base, "use_queries": True, "use_highres_fusion": True, "use_local_supervision": True, "use_artifact_aux": True, "pretrain_mode": "factorized_proxy", "pretrain_epochs": 2},
            "P2_ecg_mask_unified_lowaux_v112": {
                **base,
                "use_queries": True,
                "use_highres_fusion": True,
                "use_local_supervision": True,
                "use_artifact_aux": True,
                "use_unified_subtype_head": True,
                "pretrain_mode": "ecg_mask",
                "pretrain_epochs": 2,
                "factor_weight": 0.34,
                "local_weight": 0.22,
                "artifact_weight": 0.18,
                "subtype_weight": 0.16,
                "lr": 1.5e-4,
            },
            "P3_factorized_proxy_unified_gmweighted_v112": {
                **base,
                "use_queries": True,
                "use_highres_fusion": True,
                "use_local_supervision": True,
                "use_artifact_aux": True,
                "use_unified_subtype_head": True,
                "pretrain_mode": "factorized_proxy",
                "pretrain_epochs": 2,
                "factor_weight": 0.34,
                "local_weight": 0.22,
                "artifact_weight": 0.18,
                "subtype_weight": 0.16,
                "class_weights": [1.24, 1.10, 0.82],
                "lr": 1.5e-4,
            },
        }
    raise ValueError(f"unknown stage {stage}")


def make_time_mask(x: torch.Tensor, mask_frac: float = 0.18, block: int = 48) -> torch.Tensor:
    b, _, n = x.shape
    mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
    spans = max(1, int((float(mask_frac) * n) // max(1, int(block))))
    for i in range(b):
        for _ in range(spans):
            start = int(torch.randint(0, max(1, n - int(block)), (1,), device=x.device).item())
            mask[i, 0, start : start + int(block)] = True
    return mask


def pretrain_model(model: nn.Module, loader: DataLoader, cfg: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    mode = str(cfg.get("pretrain_mode", "none"))
    epochs = int(cfg.get("pretrain_epochs", 0))
    if mode == "none" or epochs <= 0:
        return []
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]) * 0.75, weight_decay=1e-4)
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            mask = make_time_mask(x, block=32 if mode == "ecg_mask" else 64)
            x_masked = x.masked_fill(mask, 0.0)
            out = model(x_masked)
            recon = F.smooth_l1_loss(out["recon"].masked_select(mask.expand_as(x)), x.masked_select(mask.expand_as(x)))
            loss = recon
            if mode in {"ecg_mask", "factorized_proxy"}:
                factor, _ = factor_loss(out, batch, device)
                local, _ = local_map_loss(out, x)
                loss = loss + 0.35 * factor + 0.25 * local
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        logs.append({"phase": f"pretrain_{mode}", "epoch": epoch, "loss": float(np.mean(losses))})
        print(f"pretrain {mode} epoch={epoch}/{epochs} loss={np.mean(losses):.4f}", flush=True)
    return logs


def train_model(candidate: str, cfg: dict[str, Any], args: argparse.Namespace, split_root: Path | None = None, fold: int = 0, seed_index: int = 0) -> dict[str, Any]:
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_ds = make_loaders(args, split_root, fold)
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    model = EventFactorizedSQIConformer(
        in_ch=int(train_ds.x.shape[1]),
        factor_dim=len(FACTOR_COLUMNS),
        width=int(cfg["width"]),
        layers=int(cfg["layers"]),
        heads=int(cfg["heads"]),
        use_queries=bool(cfg.get("use_queries", True)),
        use_highres_fusion=bool(cfg.get("use_highres_fusion", True)),
        use_local_supervision=bool(cfg.get("use_local_supervision", True)),
        use_artifact_aux=bool(cfg.get("use_artifact_aux", True)),
        use_hierarchical_head=bool(cfg.get("use_hierarchical_head", True)),
        branch_upper_block=bool(cfg.get("branch_upper_block", False)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    pretrain_logs = pretrain_model(model, train_loader, cfg, device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    logs: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    grad_seen = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_parts: list[dict[str, float]] = []
        losses = []
        for step, batch in enumerate(train_loader):
            if str(cfg.get("optimizer_mode", "ordinary")) in {"pcgrad_class_artifact", "cagrad_light"}:
                x = batch["x"].to(device=device, dtype=torch.float32)
                y = batch["y"].to(device=device)
                out = model(x)
                losses_dict = {
                    "class": class_nll(out, y, cfg, device),
                    "artifact": artifact_loss(out, batch, device)[0],
                }
                if str(cfg.get("optimizer_mode", "ordinary")) == "pcgrad_class_artifact":
                    pcgrad_backward(losses_dict, model)
                else:
                    factor_task = factor_loss(out, batch, device)[0]
                    local_task = local_map_loss(out, x)[0] if bool(cfg.get("use_local_supervision", True)) else torch.zeros((), device=device)
                    losses_dict["factor"] = factor_task
                    losses_dict["local"] = local_task
                    cagrad_light_backward(losses_dict, model)
                opt.step()
                opt.zero_grad(set_to_none=True)
                total_loss = sum(losses_dict.values())
                parts = {"loss_total": float(total_loss.detach().cpu()), "loss_cls": float(losses_dict["class"].detach().cpu()), "loss_artifact": float(losses_dict["artifact"].detach().cpu())}
            else:
                total_loss, parts, _ = model_loss(model, batch, cfg, device)
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            losses.append(float(total_loss.detach().cpu()))
            epoch_parts.append(parts)
            if bool(args.audit_gradients) and grad_seen < int(args.gradient_batches):
                gradient_rows.extend(gradient_snapshot(model, batch, cfg, device, epoch, grad_seen))
                grad_seen += 1
        train_rep, _, _, _, _, _, _, _, _ = eval_loader(model, train_eval_loader, device, candidate)
        val_rep, _, _, _, _, _, _, _, _ = eval_loader(model, val_loader, device, candidate)
        score = float(val_rep["record_macro_supported_f1"]) + 0.20 * min(float(val_rep["good_recall"]), float(val_rep["medium_recall"]), float(val_rep["bad_recall"])) - 0.05 * float(val_rep["bad_fpr_nonbad"])
        mean_parts = pd.DataFrame(epoch_parts).mean(numeric_only=True).to_dict() if epoch_parts else {}
        log_row = {
            "phase": "finetune",
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"train_loss_part_{k}": v for k, v in mean_parts.items()},
            **{f"train_{k}": v for k, v in train_rep.items() if k != "confusion_3x3"},
            **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"},
        }
        logs.append(log_row)
        print(
            f"{candidate} fold={fold} seed={seed_index} epoch={epoch}/{args.epochs} "
            f"loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} val_rec_macro={val_rep['record_macro_supported_f1']:.4f} "
            f"recalls={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    train_rep, train_probs, train_factor, train_true_factor, train_y, train_artifact, train_rec_rows, _, train_extra = eval_loader(model, train_eval_loader, device, candidate)
    val_rep, val_probs, val_factor, val_true_factor, val_y, val_artifact, val_rec_rows, _, val_extra = eval_loader(model, val_loader, device, candidate)
    test_rep, test_probs, test_factor, test_true_factor, test_y, test_artifact, test_rec_rows, _, test_extra = eval_loader(model, test_loader, device, candidate)
    run_name = f"{candidate}_fold{fold}_seed{seed_index}"
    run_path = RUN_DIR / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "factor_columns": FACTOR_COLUMNS,
            "local_map_names": LOCAL_MAP_NAMES,
            "quality_subtype_names": QUALITY_SUBTYPE_NAMES,
            "quality_subtype_class": QUALITY_SUBTYPE_CLASS,
            "boundary_family_names": BOUNDARY_FAMILY_NAMES,
            "input_contract": "waveform-derived channels only; factors are teacher targets only",
            "model_kind": "EventFactorizedSQIConformer",
            "no_warm_start": True,
        },
        run_path / "ckpt_best.pt",
    )
    pd.DataFrame(pretrain_logs + logs).to_csv(run_path / "train_log.csv", index=False)
    if gradient_rows:
        pd.DataFrame(gradient_rows).to_csv(run_path / "gradient_audit.csv", index=False)
    np.savez_compressed(
        run_path / "test_predictions.npz",
        probs=test_probs,
        factor_pred=test_factor,
        factor_true=test_true_factor,
        y=test_y,
        artifact_presence=test_artifact,
        **{f"test_{k}": v for k, v in test_extra.items()},
    )
    np.savez_compressed(
        run_path / "val_predictions.npz",
        probs=val_probs,
        factor_pred=val_factor,
        factor_true=val_true_factor,
        y=val_y,
        artifact_presence=val_artifact,
        **{f"val_{k}": v for k, v in val_extra.items()},
    )
    return {
        "candidate": candidate,
        "fold": int(fold),
        "seed_index": int(seed_index),
        "run_path": str(run_path),
        "metrics": [
            metric_row(candidate, "clean_train", train_rep),
            metric_row(candidate, "clean_val", val_rep),
            metric_row(candidate, "clean_test", test_rep),
        ],
        "record_metrics": train_rec_rows + val_rec_rows + test_rec_rows,
        "recovery": factor_recovery_rows(candidate, "clean_test", test_factor, test_true_factor, test_y),
        "logs": logs,
        "test_y": test_y,
        "test_probs": test_probs,
        "test_artifact": test_artifact,
    }


def phase0_correctness(args: argparse.Namespace) -> Path:
    ensure_dirs()
    source = DUAL.PROTOCOL_ROOT / str(args.policy)
    atlas = pd.read_csv(source / "original_region_atlas.csv")
    train_loader, val_loader, test_loader, train_ds = make_loaders(args)
    rows = []
    for split_name, ds in [("train", train_ds), ("val", val_loader.dataset), ("test", test_loader.dataset)]:
        frame = ds.frame.copy()
        target_df = pd.DataFrame(
            {
                "split": split_name,
                "class_name": frame["class_name"].astype(str),
                "record_id": frame.get("record_id", pd.Series(["unknown"] * len(frame))).astype(str),
                "original_region": frame.get("original_region", pd.Series(["unknown"] * len(frame))).astype(str),
                "artifact_presence": ds.artifact_presence.astype(bool),
                "diagnostic_bad": ds.diagnostic_bad.astype(bool),
                "artifact_type": ds.artifact_type,
                "clean_policy": frame.get("clean_policy", pd.Series([""] * len(frame))).astype(str),
            }
        )
        rows.append(target_df)
    audit = pd.concat(rows, ignore_index=True)
    artifact_nonbad = audit["artifact_presence"] & ~audit["diagnostic_bad"]
    all_nonbad = ~audit["diagnostic_bad"]
    crosstab = pd.crosstab([audit["split"], audit["class_name"]], audit["artifact_presence"], margins=True)
    region_tab = pd.crosstab(audit["original_region"], [audit["class_name"], audit["artifact_presence"]]).reset_index()
    record_support = audit.groupby(["split", "record_id", "class_name"]).size().unstack(fill_value=0).reset_index()
    artifact_audit = {
        "rows": int(len(audit)),
        "nonbad_rows": int(all_nonbad.sum()),
        "artifact_positive_nonbad_count": int(artifact_nonbad.sum()),
        "artifact_positive_nonbad_equals_all_nonbad": bool(int(artifact_nonbad.sum()) == int(all_nonbad.sum())),
        "clean_policy_used_as_artifact_evidence": False,
        "artifact_thresholds": train_ds.artifact_thresholds.__dict__,
        "factor_columns": FACTOR_COLUMNS,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = candidate_grid("phase1", int(args.seed))["E4_query_highres_local_art"]
    model = EventFactorizedSQIConformer(
        in_ch=int(train_ds.x.shape[1]),
        factor_dim=len(FACTOR_COLUMNS),
        width=int(cfg["width"]),
        layers=1,
        heads=int(cfg["heads"]),
        use_queries=True,
        use_highres_fusion=True,
        use_local_supervision=True,
        use_artifact_aux=True,
    ).to(device)
    batch = next(iter(train_loader))
    total, parts, out = model_loss(model, batch, cfg, device)
    model.zero_grad(set_to_none=True)
    factor = factor_loss(out, batch, device)[0]
    factor.backward(retain_graph=True)
    detector_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None and ("local_heads" in name or "factor_head" in name):
            detector_grad_norm += float(torch.norm(param.grad).detach().cpu())
    grad_rows = gradient_snapshot(model, batch, cfg, device, epoch=0, max_batches_seen=0)
    detector_same_value = True
    with torch.no_grad():
        det_from_factor = out["factor_pred"][:, DETECTOR_FACTOR_IDX].detach().cpu().numpy()
        det_from_event = out["detector_agreement"].detach().cpu().numpy()
        detector_same_value = bool(np.allclose(det_from_factor, det_from_event, atol=1e-6))
        det_range = {"min": float(det_from_factor.min()), "max": float(det_from_factor.max()), "mean": float(det_from_factor.mean())}

    phase0_dir = OUT_DIR / "phase0_correctness"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(phase0_dir / "artifact_targets_audit.csv", index=False)
    crosstab.to_csv(phase0_dir / "artifact_class_crosstab.csv")
    region_tab.to_csv(phase0_dir / "artifact_region_crosstab.csv", index=False)
    record_support.to_csv(phase0_dir / "record_class_support.csv", index=False)
    pd.DataFrame(grad_rows).to_csv(phase0_dir / "gradient_audit_onebatch.csv", index=False)
    (phase0_dir / "phase0_correctness_summary.json").write_text(
        json.dumps({**artifact_audit, "detector_grad_norm": detector_grad_norm, "detector_same_value_in_factor_and_report": detector_same_value, "detector_range": det_range, "one_batch_loss_parts": parts}, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Phase 0 Correctness Report",
        "",
        f"- Generated: {now()}",
        f"- Policy: `{args.policy}`",
        "- Scope: external experiment runner only; no `src/sqi_pipeline` changes.",
        "- Artifact target rule excludes `clean_policy` text. `clean_policy=margin_ge_5s_drop_outlier` is no longer treated as artifact evidence.",
        "",
        "## Acceptance Checks",
        "",
        f"- `artifact_positive_nonbad_count`: `{artifact_audit['artifact_positive_nonbad_count']}`",
        f"- `artifact_positive_nonbad_equals_all_nonbad`: `{artifact_audit['artifact_positive_nonbad_equals_all_nonbad']}`",
        f"- `clean_policy_used_as_artifact_evidence`: `{artifact_audit['clean_policy_used_as_artifact_evidence']}`",
        f"- `detector_grad_norm`: `{detector_grad_norm:.6g}`",
        f"- `detector_same_value_in_factor_and_report`: `{detector_same_value}`",
        f"- `detector_range`: `{det_range}`",
        "",
        "## Artifact x Class Crosstab",
        "",
        df_to_markdown(crosstab.reset_index()),
        "",
        "## Record Support Preview",
        "",
        df_to_markdown(record_support.head(30)),
        "",
        "## Output Files",
        "",
        f"- Artifact audit CSV: `{phase0_dir / 'artifact_targets_audit.csv'}`",
        f"- Gradient audit CSV: `{phase0_dir / 'gradient_audit_onebatch.csv'}`",
        f"- Summary JSON: `{phase0_dir / 'phase0_correctness_summary.json'}`",
    ]
    report_path = REPORT_OUT_DIR / "phase0_correctness_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def write_stage_report(stage: str, results: list[dict[str, Any]]) -> None:
    ensure_dirs()
    metrics = []
    recovery = []
    records = []
    for res in results:
        metrics.extend(res["metrics"])
        recovery.extend(res["recovery"])
        records.extend(res["record_metrics"])
    metrics_df = pd.DataFrame(metrics)
    recovery_df = pd.DataFrame(recovery)
    records_df = pd.DataFrame(records)
    metrics_path = OUT_DIR / f"{stage}_metrics.csv"
    recovery_path = OUT_DIR / f"{stage}_feature_recovery.csv"
    record_path = OUT_DIR / f"{stage}_record_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    recovery_df.to_csv(recovery_path, index=False)
    records_df.to_csv(record_path, index=False)
    report = [
        f"# Event-Factorized SQI Conformer {stage} Report",
        "",
        f"- Generated: {now()}",
        "- Formal input contract: waveform-derived channels only.",
        "- SQI/factor targets are training teacher/diagnostic targets only.",
        "- All checkpoints in this stage are trained from scratch.",
        "",
        "## Metrics",
        "",
        df_to_markdown(metrics_df.sort_values(["bucket", "record_macro_supported_f1"], ascending=[True, False]).head(80)),
        "",
        "## Feature Recovery",
        "",
        df_to_markdown(recovery_df.head(80)),
        "",
        "## Record Metrics Preview",
        "",
        df_to_markdown(records_df.head(80)),
        "",
        "## Files",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Feature recovery: `{recovery_path}`",
        f"- Record metrics: `{record_path}`",
        f"- Checkpoints/logs: `{RUN_DIR}`",
    ]
    (REPORT_OUT_DIR / f"event_factorized_sqi_conformer_{stage}_report.md").write_text("\n".join(report), encoding="utf-8")


def run_stage(stage: str, args: argparse.Namespace) -> None:
    ensure_dirs()
    split_root = OUT_DIR / "rh_splits" / f"{policy_alias(str(args.policy))}_k{args.folds}_s{args.split_seed}"
    required_split_file = split_root / "fold0" / "source_protocol.json"
    if not split_root.exists() or not required_split_file.exists():
        split_root = build_recordheldout_splits(str(args.policy), int(args.folds), int(args.split_seed))
    all_results: list[dict[str, Any]] = []
    base_seeds = [int(args.seed) + i * 1000 for i in range(int(args.seeds))]
    grids_per_seed = [candidate_grid(stage, seed) for seed in base_seeds]
    candidates = list(grids_per_seed[0].keys())
    if args.candidates:
        allow = set(args.candidates.split(","))
        candidates = [c for c in candidates if c in allow]
    for candidate in candidates:
        for fold in range(int(args.folds)):
            for seed_index, grid in enumerate(grids_per_seed):
                cfg = dict(grid[candidate])
                cfg["seed"] = int(base_seeds[seed_index] + fold)
                res = train_model(candidate, cfg, args, split_root=split_root, fold=fold, seed_index=seed_index)
                all_results.append(res)
                write_stage_report(stage, all_results)
    write_stage_report(stage, all_results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="phase0", choices=["build_recordheldout_splits", "phase0", "phase1", "phase2", "phase3", "all"])
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260620)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--record-balanced-sampler", dest="record_balanced_sampler", action="store_true", default=True)
    parser.add_argument("--no-record-balanced-sampler", dest="record_balanced_sampler", action="store_false")
    parser.add_argument("--audit-gradients", action="store_true", default=False)
    parser.add_argument("--gradient-batches", type=int, default=30)
    parser.add_argument("--candidates", type=str, default="")
    args = parser.parse_args()
    ensure_dirs()
    if args.stage == "build_recordheldout_splits":
        root = build_recordheldout_splits(str(args.policy), int(args.folds), int(args.split_seed))
        print(root)
    elif args.stage == "phase0":
        report = phase0_correctness(args)
        print(report)
    elif args.stage == "all":
        phase0_correctness(args)
        for stage in ["phase1", "phase2", "phase3"]:
            run_stage(stage, args)
    else:
        run_stage(str(args.stage), args)


if __name__ == "__main__":
    main()
