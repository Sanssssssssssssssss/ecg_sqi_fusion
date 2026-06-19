"""Waveform-only hard SQI feature recovery probe.

This is an external-only diagnostic runner.  It does not train a classifier and
does not use BUT for selection.  The goal is to isolate whether a waveform
tokenizer/encoder can recover the ECG-computable hard SQI targets that the
classification models keep missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
STUDENT_PATH = ANALYSIS_DIR / "run_waveform_geometry_student.py"


def load_student_module() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_geometry_student", STUDENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {STUDENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = load_student_module()

WAVEFORM_FACT_FEATURES = [
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "sqi_basSQI",
    "flatline_ratio",
    "low_amp_ratio",
    "non_qrs_diff_p95",
    "contact_loss_win_ratio",
    "fatal_or_score",
]

GEOMETRY_PROXY_FEATURES = [
    "pc2",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
]

ALL_EVAL_FEATURES = WAVEFORM_FACT_FEATURES + GEOMETRY_PROXY_FEATURES


class GroupSpecificPhysioProbe(torch.nn.Module):
    """Transformer probe with explicit per-physiology feature heads.

    The previous aux-head probes put all waveform facts into one shared vector.
    This module keeps inference waveform-only but assigns feature supervision to
    the token group that can physically support it: QRS visibility from QRS-bank
    tokens, detector agreement from detector/RR tokens, baseline features from
    baseline-frequency tokens, and contact/fatal/detail facts from stress and
    artifact tokens.
    """

    GROUPS = [
        ("wave", 16),
        ("qrs_detail", 15),
        ("qrsbank", 19),
        ("detector", 16),
        ("rr", 24),
        ("baseline", 24),
        ("stress", 24),
        ("artifact", 51),
    ]

    def __init__(self, in_ch: int, width: int = 96, layers: int = 3, heads: int = 4, hidden: int = 192):
        super().__init__()
        self.in_ch = int(in_ch)
        self.width = int(width)
        self.group_names = [g for g, _ in self.GROUPS]
        self.proj = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, width), torch.nn.GELU())
                for name, dim in self.GROUPS
            }
        )
        self.channel_embed = torch.nn.Parameter(torch.randn(self.in_ch, width) * 0.02)
        self.group_embed = torch.nn.Parameter(torch.randn(len(self.GROUPS), width) * 0.02)
        self.cls = torch.nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pos = G.PositionalEncoding(width, max_len=128)
        self.encoder = torch.nn.TransformerEncoder(G.make_encoder_layer(width, heads), num_layers=layers)
        self.global_head = torch.nn.Sequential(
            torch.nn.LayerNorm(width * (len(self.GROUPS) + 1)),
            torch.nn.Linear(width * (len(self.GROUPS) + 1), hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.08),
            torch.nn.Linear(hidden, 5),
        )
        self.qrs_head = self._head(width * 2, 1)
        self.det_head = self._head(width * 2, 1)
        self.base_head = self._head(width, 3)
        self.stress_head = self._head(width * 2, 4)
        self.full_aux_bias = torch.nn.Parameter(torch.zeros(len(G.FEATURE_COLUMNS)))
        self.feature_map = {
            "qrs_visibility": ("qrs", 0),
            "detector_agreement": ("det", 0),
            "baseline_step": ("base", 0),
            "sqi_basSQI": ("base", 1),
            "flatline_ratio": ("base", 2),
            "low_amp_ratio": ("stress", 0),
            "non_qrs_diff_p95": ("stress", 1),
            "contact_loss_win_ratio": ("stress", 2),
            "fatal_or_score": ("stress", 3),
            "pc2": ("global", 0),
            "pc3": ("global", 1),
            "pca_margin": ("global", 2),
            "boundary_confidence": ("global", 3),
            "knn_label_purity": ("global", 4),
        }

    @staticmethod
    def _head(dim: int, out: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, max(64, dim // 2)),
            torch.nn.GELU(),
            torch.nn.Dropout(0.06),
            torch.nn.Linear(max(64, dim // 2), out),
        )

    def artifact_summary(self, x: torch.Tensor) -> torch.Tensor:
        _, scalar = G.artifact_event_raw_windows_and_features(x, 16, 151)
        score = scalar[..., 13]
        k = max(1, min(4, scalar.shape[2]))
        idx = torch.topk(score, k=k, dim=2).indices.unsqueeze(-1).expand(-1, -1, -1, scalar.shape[-1])
        top_mean = torch.gather(scalar, 2, idx).mean(dim=2)
        return torch.cat([scalar.mean(dim=2), scalar.amax(dim=2), top_mean], dim=2)

    def parts(self, x: torch.Tensor) -> list[torch.Tensor]:
        b = x.shape[0]
        return [
            G.waveform_stats_v2(x).reshape(b, self.in_ch, 16),
            G.qrs_detail_stats(x).reshape(b, self.in_ch, 15),
            G.qrs_visibility_bank_stats(x).reshape(b, self.in_ch, 19),
            G.qrs_detector_agreement_stats(x).reshape(b, self.in_ch, 16),
            G.rr_consistency_stats(x).reshape(b, self.in_ch, 24),
            G.baseline_frequency_stats(x).reshape(b, self.in_ch, 24),
            G.waveform_stress_stats(x).reshape(b, self.in_ch, 24),
            self.artifact_summary(x).reshape(b, self.in_ch, 51),
        ]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        ch = self.channel_embed.view(1, self.in_ch, self.width)
        tokens = []
        for i, (name, _dim) in enumerate(self.GROUPS):
            tokens.append(self.proj[name](self.parts(x)[i]) + ch + self.group_embed[i].view(1, 1, -1))
        body_in = torch.cat(tokens, dim=1)
        encoded = self.encoder(self.pos(torch.cat([self.cls.expand(b, -1, -1), body_in], dim=1)))
        body = encoded[:, 1:, :].reshape(b, len(self.GROUPS), self.in_ch, self.width).mean(dim=2)
        pooled = torch.cat([encoded[:, 0, :], body.flatten(start_dim=1)], dim=1)
        qrs_vec = torch.cat([body[:, self.group_names.index("qrs_detail")], body[:, self.group_names.index("qrsbank")]], dim=1)
        det_vec = torch.cat([body[:, self.group_names.index("detector")], body[:, self.group_names.index("rr")]], dim=1)
        base_vec = body[:, self.group_names.index("baseline")]
        stress_vec = torch.cat([body[:, self.group_names.index("stress")], body[:, self.group_names.index("artifact")]], dim=1)
        outs = {
            "qrs": self.qrs_head(qrs_vec),
            "det": self.det_head(det_vec),
            "base": self.base_head(base_vec),
            "stress": self.stress_head(stress_vec),
            "global": self.global_head(pooled),
        }
        aux = self.full_aux_bias.view(1, -1).expand(b, -1).clone()
        for feat, (head_name, col) in self.feature_map.items():
            if feat in G.FEATURE_COLUMNS:
                aux[:, G.FEATURE_COLUMNS.index(feat)] = outs[head_name][:, col]
        return {"aux_pred": aux}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_cfg(name: str) -> dict[str, Any]:
    if name == "qrsbase":
        cfg = deepcopy(G.CANDIDATES["featurefirst_top20_hardrec_qfeatbin_qrsbase_a050"])
    elif name == "auxphysio":
        cfg = deepcopy(G.CANDIDATES["featurefirst_top20_qrsbase_auxphysio_conservative_a050"])
        cfg.pop("init_from_candidate", None)
        cfg["init_skip_mismatch"] = False
    elif name == "physio_token":
        cfg = deepcopy(G.CANDIDATES["physio_sqi_token_balanced_a050"])
    elif name == "rawbeat_physioctx":
        cfg = deepcopy(G.CANDIDATES["featurefirst_top20_rawbeat_artifact_auxctx_physioctx_balanced_a050"])
        cfg.pop("init_from_candidate", None)
        cfg["init_skip_mismatch"] = False
    elif name == "detbase":
        cfg = deepcopy(G.CANDIDATES["detbase_tokens_qfeatbin_balanced"])
    elif name == "detbase_badguard":
        cfg = deepcopy(G.CANDIDATES["detbase_tokens_qfeatbin_badguard"])
    elif name == "detbase_artifact":
        cfg = deepcopy(G.CANDIDATES["detbase_tokens_artifact_event_mediumguard"])
    elif name == "groupheads_physio":
        cfg = {
            "probe_arch": "groupheads_physio",
            "channels": "robust3",
            "width": 96,
            "layers": 3,
            "heads": 4,
            "hidden": 192,
            "seed": 20261510,
        }
    elif name == "groupheads_physio_wide":
        cfg = {
            "probe_arch": "groupheads_physio",
            "channels": "robust3",
            "width": 128,
            "layers": 4,
            "heads": 4,
            "hidden": 256,
            "seed": 20261511,
        }
    else:
        raise ValueError(f"unknown probe candidate {name}")
    cfg["seed"] = int(cfg.get("seed", 20261500)) + 9000
    cfg["channels"] = str(cfg.get("channels", "robust3"))
    return cfg


def set_optional_prototypes(model: torch.nn.Module, cfg: dict[str, Any], train_ds: Any) -> None:
    if getattr(model, "use_teacher_atlas", False):
        prototypes = G.build_teacher_prototypes(
            train_ds.aux,
            train_ds.y,
            int(cfg.get("teacher_prototypes_per_class", 14)),
            str(cfg.get("teacher_prototype_mode", "farthest")),
        )
        model.set_teacher_prototypes(prototypes)
    if getattr(model, "use_waveform_atlas", False):
        wave_proto, wave_mean, wave_std = G.build_waveform_atlas(
            train_ds,
            int(cfg.get("waveform_atlas_prototypes_per_class", 24)),
            str(cfg.get("waveform_atlas_mode", "farthest")),
        )
        model.set_waveform_atlas(wave_proto, wave_mean, wave_std)


def get_loaders(cfg: dict[str, Any], batch_size: int, num_workers: int) -> tuple[Any, dict[str, DataLoader]]:
    train_ds = G.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    val_ds = G.ARCH.SyntheticWaveDataset("val", str(cfg["channels"]))
    test_ds = G.ARCH.SyntheticWaveDataset("test", str(cfg["channels"]))
    norm = G.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = G.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    loaders = {
        "synthetic_train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "synthetic_val": DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers),
        "synthetic_test": DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers),
        "original_test_all_10s+": DataLoader(original_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers),
    }
    return train_ds, loaders


def feature_indices(features: list[str]) -> list[int]:
    missing = [x for x in features if x not in G.FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"missing feature columns in schema: {missing}")
    return [G.FEATURE_COLUMNS.index(x) for x in features]


def weighted_feature_loss(pred: torch.Tensor, target: torch.Tensor, idx: list[int]) -> torch.Tensor:
    pred_sel = pred[:, idx]
    target_sel = torch.nan_to_num(target[:, idx], nan=0.0, posinf=0.0, neginf=0.0)
    loss = F.smooth_l1_loss(pred_sel, target_sel, reduction="none")
    # Give the historically weak waveform facts extra influence.
    weights = torch.ones(len(idx), device=pred.device)
    for j, col in enumerate(idx):
        name = G.FEATURE_COLUMNS[col]
        if name in {"qrs_visibility", "detector_agreement", "baseline_step", "sqi_basSQI"}:
            weights[j] = 2.5
        elif name in {"contact_loss_win_ratio", "fatal_or_score"}:
            weights[j] = 2.0
        elif name in {"pc2", "pc3"}:
            weights[j] = 0.65
    return (loss * weights[None, :]).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, features: list[str]) -> list[dict[str, Any]]:
    model.eval()
    idx = feature_indices(features)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x)
        preds.append(out["aux_pred"][:, idx].detach().cpu().numpy())
        targets.append(batch["aux"][:, idx].detach().cpu().numpy())
        labels.append(batch["y"].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.nan_to_num(np.concatenate(targets, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate(labels, axis=0)
    rows = []
    for j, name in enumerate(features):
        pj = pred[:, j].astype(float)
        tj = target[:, j].astype(float)
        if np.std(pj) < 1e-8 or np.std(tj) < 1e-8:
            corr = np.nan
        else:
            corr = float(np.corrcoef(pj, tj)[0, 1])
        rows.append(
            {
                "feature": name,
                "corr": corr,
                "mae_z": float(np.mean(np.abs(pj - tj))),
                "label_good_corr": class_corr(pj, tj, y, 0),
                "label_medium_corr": class_corr(pj, tj, y, 1),
                "label_bad_corr": class_corr(pj, tj, y, 2),
            }
        )
    return rows


def class_corr(pred: np.ndarray, target: np.ndarray, y: np.ndarray, cls: int) -> float:
    mask = y == cls
    if mask.sum() < 8:
        return float("nan")
    pj = pred[mask].astype(float)
    tj = target[mask].astype(float)
    if np.std(pj) < 1e-8 or np.std(tj) < 1e-8:
        return float("nan")
    return float(np.corrcoef(pj, tj)[0, 1])


def run_probe(name: str, epochs: int, batch_size: int, num_workers: int, lr: float) -> dict[str, Any]:
    cfg = make_cfg(name)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, loaders = get_loaders(cfg, batch_size, num_workers)
    if str(cfg.get("probe_arch", "")) == "groupheads_physio":
        model = GroupSpecificPhysioProbe(
            int(train_ds.x.shape[1]),
            int(cfg.get("width", 96)),
            int(cfg.get("layers", 3)),
            int(cfg.get("heads", 4)),
            int(cfg.get("hidden", 192)),
        ).to(device)
    else:
        model = G.GeometryStudent(cfg, int(train_ds.x.shape[1])).to(device)
        set_optional_prototypes(model, cfg, train_ds)
    target_features = WAVEFORM_FACT_FEATURES
    target_idx = feature_indices(target_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    best = {"val_mean_corr": -999.0, "state": None, "epoch": 0}
    epoch_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in loaders["synthetic_train"]:
            x = batch["x"].to(device=device, dtype=torch.float32)
            aux = batch["aux"].to(device=device, dtype=torch.float32)
            out = model(x)
            loss = weighted_feature_loss(out["aux_pred"], aux, target_idx)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(x.shape[0])
            total_n += int(x.shape[0])
        val_rows = evaluate(model, loaders["synthetic_val"], device, target_features)
        val_mean = float(np.nanmean([r["corr"] for r in val_rows]))
        epoch_rows.append({"candidate": name, "epoch": epoch, "loss": total_loss / max(total_n, 1), "val_mean_corr": val_mean})
        print(f"{name} epoch={epoch}/{epochs} loss={total_loss/max(total_n,1):.4f} val_fact_corr={val_mean:.4f}", flush=True)
        if val_mean > best["val_mean_corr"]:
            best = {
                "val_mean_corr": val_mean,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    split_rows = []
    for split, loader in loaders.items():
        if split == "synthetic_train":
            continue
        for row in evaluate(model, loader, device, ALL_EVAL_FEATURES):
            row.update({"candidate": name, "split": split, "best_epoch": int(best["epoch"])})
            split_rows.append(row)
    return {"candidate": name, "best_epoch": int(best["epoch"]), "epoch_rows": epoch_rows, "feature_rows": split_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="qrsbase,auxphysio,physio_token,rawbeat_physioctx,detbase,detbase_badguard,detbase_artifact,groupheads_physio,groupheads_physio_wide")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    args = parser.parse_args()
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_features: list[dict[str, Any]] = []
    all_epochs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in names:
        result = run_probe(name, args.epochs, args.batch_size, args.num_workers, args.lr)
        all_features.extend(result["feature_rows"])
        all_epochs.extend(result["epoch_rows"])
        summaries.append({"candidate": name, "best_epoch": result["best_epoch"]})
    feature_df = pd.DataFrame(all_features)
    epoch_df = pd.DataFrame(all_epochs)
    feature_path = ANALYSIS_DIR / "waveform_feature_probe_recovery.csv"
    epoch_path = ANALYSIS_DIR / "waveform_feature_probe_epochs.csv"
    summary_path = ANALYSIS_DIR / "waveform_feature_probe_summary.csv"
    feature_df.to_csv(feature_path, index=False)
    epoch_df.to_csv(epoch_path, index=False)
    summary_rows = []
    for (candidate, split), chunk in feature_df.groupby(["candidate", "split"]):
        facts = chunk[chunk["feature"].isin(WAVEFORM_FACT_FEATURES)]
        summary_rows.append(
            {
                "candidate": candidate,
                "split": split,
                "best_epoch": int(chunk["best_epoch"].iloc[0]),
                "fact_corr_mean": float(np.nanmean(facts["corr"])),
                "fact_corr_min": float(np.nanmin(facts["corr"])),
                "qrs_visibility_corr": get_feature_corr(chunk, "qrs_visibility"),
                "detector_agreement_corr": get_feature_corr(chunk, "detector_agreement"),
                "baseline_step_corr": get_feature_corr(chunk, "baseline_step"),
                "sqi_basSQI_corr": get_feature_corr(chunk, "sqi_basSQI"),
                "contact_loss_corr": get_feature_corr(chunk, "contact_loss_win_ratio"),
                "fatal_or_corr": get_feature_corr(chunk, "fatal_or_score"),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["split", "fact_corr_mean"], ascending=[True, False])
    summary_df.to_csv(summary_path, index=False)
    payload = {
        "created_at": now(),
        "contract": "waveform-only feature recovery probe; BUT is evaluation only",
        "target_features": WAVEFORM_FACT_FEATURES,
        "eval_features": ALL_EVAL_FEATURES,
        "feature_csv": str(feature_path),
        "epoch_csv": str(epoch_path),
        "summary_csv": str(summary_path),
        "summaries": summaries,
    }
    (ANALYSIS_DIR / "waveform_feature_probe_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False), flush=True)
    print(f"saved {summary_path}", flush=True)


def get_feature_corr(chunk: pd.DataFrame, feature: str) -> float:
    hit = chunk[chunk["feature"].eq(feature)]
    if hit.empty:
        return float("nan")
    return float(hit["corr"].iloc[0])


if __name__ == "__main__":
    main()
