"""E3.11f dataset loader for the Uformer mainline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


@dataclass(frozen=True)
class SplitAudit:
    train: int
    val: int
    test: int
    class_counts: dict[str, dict[str, int]]
    noise_kind_counts: dict[str, dict[str, int]]
    placement_counts: dict[str, dict[str, int]]
    snr_mean_by_split_class: dict[str, dict[str, float]]


class E311DenoiseDataset(Dataset):
    """Loads the med6.25/bad-gap7 artifact without regenerating data."""

    def __init__(self, source_artifact_dir: str | Path, split: str):
        self.source_artifact_dir = Path(source_artifact_dir)
        data_dir = self.source_artifact_dir / "datasets"
        labels_path = data_dir / "synth_10s_125hz_labels_with_level.csv"
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        labels = pd.read_csv(labels_path).sort_values("idx").reset_index(drop=True)
        keep = labels["split"].astype(str).to_numpy() == split
        self.labels = labels.loc[keep].reset_index(drop=True)
        rows = self.labels["idx"].to_numpy(dtype=np.int64)

        clean_all = np.load(data_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
        noisy_all = np.load(data_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
        masks = np.load(data_dir / "synth_10s_125hz_local_mask.npz")
        self.clean = clean_all[rows]
        self.noisy = noisy_all[rows]
        self.local_mask = masks["M"].astype(np.float32)[rows]
        self.critical_mask = masks["critical_mask"].astype(np.float32)[rows]
        self.qrs_mask = masks["qrs_mask"].astype(np.float32)[rows]
        self.tst_mask = masks["tst_mask"].astype(np.float32)[rows]
        self.y = self.labels["y_class"].map(CLASS_TO_INT).to_numpy(dtype=np.int64)
        self.idx = self.labels["idx"].to_numpy(dtype=np.int64)
        self.ecg_id = self.labels["ecg_id"].to_numpy(dtype=np.int64)
        self.snr_db = self.labels["measured_snr_db"].to_numpy(dtype=np.float32)
        if "original_measured_snr_db" in self.labels:
            self.original_snr_db = self.labels["original_measured_snr_db"].to_numpy(dtype=np.float32)
        else:
            self.original_snr_db = self.snr_db.copy()
        self.noise_kind = self.labels["noise_kind"].astype(str).to_numpy()
        self.placement = self.labels["placement"].astype(str).to_numpy()

        morph_cols = [
            "qrs_nprd",
            "tst_nprd",
            "critical_damage_score",
            "diagnostic_damage_score",
            "smooth_morph_score",
        ]
        morph = self.labels[morph_cols].fillna(0.0).to_numpy(dtype=np.float32)
        morph = np.clip(morph, 0.0, np.percentile(morph, 98, axis=0, keepdims=True) + 1e-6)
        morph = morph / (np.max(morph, axis=0, keepdims=True) + 1e-6)
        self.sample_weight = (1.0 + 0.6 * morph[:, 2] + 0.35 * morph[:, 3] + 0.35 * morph[:, 4]).astype(np.float32)
        if "sample_weight" in self.labels:
            external_weight = self.labels["sample_weight"].fillna(1.0).to_numpy(dtype=np.float32)
            self.cls_sample_weight = np.clip(external_weight, 0.05, 5.0).astype(np.float32)
        else:
            self.cls_sample_weight = np.ones_like(self.sample_weight, dtype=np.float32)
        if {"soft_good", "soft_medium", "soft_bad"}.issubset(self.labels.columns):
            soft = self.labels[["soft_good", "soft_medium", "soft_bad"]].fillna(0.0).to_numpy(dtype=np.float32)
            denom = np.maximum(soft.sum(axis=1, keepdims=True), 1e-6)
            self.soft_y = soft / denom
        else:
            self.soft_y = np.eye(3, dtype=np.float32)[self.y]

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int | float | str]:
        point_weight = (
            1.0
            + 1.6 * self.local_mask[i]
            + 2.2 * self.critical_mask[i]
            + 1.7 * self.qrs_mask[i]
            + 1.1 * self.tst_mask[i]
        ).astype(np.float32)
        return {
            "noisy": torch.from_numpy(self.noisy[i]).unsqueeze(0),
            "clean": torch.from_numpy(self.clean[i]).unsqueeze(0),
            "point_weight": torch.from_numpy(point_weight),
            "local_mask": torch.from_numpy(self.local_mask[i]),
            "critical_mask": torch.from_numpy(self.critical_mask[i]),
            "qrs_mask": torch.from_numpy(self.qrs_mask[i]),
            "tst_mask": torch.from_numpy(self.tst_mask[i]),
            "sample_weight": torch.tensor(float(self.sample_weight[i]), dtype=torch.float32),
            "cls_sample_weight": torch.tensor(float(self.cls_sample_weight[i]), dtype=torch.float32),
            "soft_y": torch.from_numpy(self.soft_y[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
            "idx": torch.tensor(int(self.idx[i]), dtype=torch.long),
            "ecg_id": torch.tensor(int(self.ecg_id[i]), dtype=torch.long),
            "snr_db": torch.tensor(float(self.snr_db[i]), dtype=torch.float32),
            "original_snr_db": torch.tensor(float(self.original_snr_db[i]), dtype=torch.float32),
            "noise_kind": self.noise_kind[i],
            "placement": self.placement[i],
        }


def split_audit(source_artifact_dir: str | Path) -> SplitAudit:
    data_dir = Path(source_artifact_dir) / "datasets"
    labels = pd.read_csv(data_dir / "synth_10s_125hz_labels_with_level.csv")
    split_counts = labels["split"].value_counts().to_dict()

    def count(col: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for split, group in labels.groupby("split"):
            out[str(split)] = {str(k): int(v) for k, v in group[col].value_counts().sort_index().to_dict().items()}
        return out

    snr: dict[str, dict[str, float]] = {}
    for (split, cls), group in labels.groupby(["split", "y_class"]):
        snr.setdefault(str(split), {})[str(cls)] = float(group["measured_snr_db"].mean())
    return SplitAudit(
        train=int(split_counts.get("train", 0)),
        val=int(split_counts.get("val", 0)),
        test=int(split_counts.get("test", 0)),
        class_counts=count("y_class"),
        noise_kind_counts=count("noise_kind"),
        placement_counts=count("placement"),
        snr_mean_by_split_class=snr,
    )


def audit_to_json(audit: SplitAudit) -> dict[str, Any]:
    return {
        "train": audit.train,
        "val": audit.val,
        "test": audit.test,
        "class_counts": audit.class_counts,
        "noise_kind_counts": audit.noise_kind_counts,
        "placement_counts": audit.placement_counts,
        "snr_mean_by_split_class": audit.snr_mean_by_split_class,
    }
