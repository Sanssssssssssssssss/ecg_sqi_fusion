"""PTB-trained seven-SQI fusion heads evaluated on BUT 10s P1.

This experiment asks a narrow question: do the traditional inference-time
single-lead SQI features help the Uformer representation transfer to BUT QDB?

The denoiser is frozen.  We train only lightweight PTB heads on:

* Uformer full-token features only.
* The seven traditional SQI features only.
* Uformer features concatenated with the seven SQI features.
* A zero-init SQI residual/gated-delta head on top of Uformer logits.

Calibration for BUT uses only the BUT validation split.  Test is reported once
per trained head and calibration policy.  ``src/sqi_pipeline`` is imported for
the existing SQI formulas and QRS detectors, but never modified.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.sqi_pipeline.features.sqi import (
    sqi_basSQI,
    sqi_bSQI_li2008_global,
    sqi_fSQI,
    sqi_iSQI_li2008_global_per_lead,
    sqi_kSQI,
    sqi_pSQI,
    sqi_sSQI,
)
from src.sqi_pipeline.qrs.detectors import run_gqrs, run_xqrs
from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_10s_tuning import (
    apply_calibration,
    bias_grid_many,
    threshold_grid_many,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_MAINLINE_CKPT,
    DEFAULT_OUT_ROOT as BUT_PROTOCOL_OUT_ROOT,
    DEFAULT_SOURCE_ARTIFACT,
    load_protocol_signals,
)
from src.transformer_pipeline.external_benchmarks.run import (
    FS_TARGET,
    INT_TO_CLASS,
    N_TARGET,
    extract_uformer_features,
    multiclass_report,
    plot_wave_gallery,
)


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[3]
RUN_TAG = "e311_but_sqi_fusion_ptb_train_10s_2026_06_04"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_PTB_SQI_CONTROL = ROOT / "outputs" / "controls" / "e311f_ptb_sqi_three_class"
DEFAULT_BUT_PROTOCOL_DIR = BUT_PROTOCOL_OUT_ROOT / "protocols" / "p1_current_10s_center"

SQI_COLUMNS = ["I__iSQI", "I__bSQI", "I__pSQI", "I__sSQI", "I__kSQI", "I__fSQI", "I__basSQI"]
CLASS_NAMES = ("good", "medium", "bad")
STATE_NAME = "sqi_fusion_state.json"
SUMMARY_NAME = "sqi_fusion_summary.jsonl"
EPS = 1e-8
WELCH_KW = dict(fmax=40.0, window="hann", nperseg=256, noverlap=128, detrend="constant")
FLATLINE_EPS = 1e-4
BEAT_MATCH_TOL_MS = 150


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_state(path: Path, **kwargs: Any) -> None:
    state = read_json(path) if path.exists() else {"started_at": now_iso()}
    state.update(kwargs)
    write_json(path, state)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_ptb_split_and_signals(source_artifact: Path, ptb_sqi_control: Path) -> tuple[np.ndarray, pd.DataFrame]:
    split_csv = ptb_sqi_control / "splits" / "transformer_three_class_seed0.csv"
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)
    split = pd.read_csv(split_csv)
    split["record_id"] = split["record_id"].astype(str)
    split["y"] = split["y"].astype(int)
    split["split"] = split["split"].astype(str)
    noisy = np.load(source_artifact / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    X = noisy[split["source_idx"].to_numpy(dtype=np.int64)][:, None, :]
    if X.shape[1:] != (1, N_TARGET):
        raise ValueError(f"Expected PTB X shape (N,1,{N_TARGET}), got {X.shape}")
    return X.astype(np.float32), split


def load_ptb_sqi_features(ptb_sqi_control: Path, split: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    feat_path = ptb_sqi_control / "features" / "record7_norm.parquet"
    stats_path = ptb_sqi_control / "features" / "norm_stats_seed0.json"
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    feat = pd.read_parquet(feat_path)
    feat["record_id"] = feat["record_id"].astype(str)
    merged = split[["record_id"]].merge(feat[["record_id", *SQI_COLUMNS]], on="record_id", how="left")
    if merged[SQI_COLUMNS].isna().any().any():
        raise ValueError("Missing PTB SQI rows after record_id alignment.")
    stats = read_json(stats_path) if stats_path.exists() else {}
    return merged[SQI_COLUMNS].to_numpy(dtype=np.float32), stats


def sqi_for_signal(x: np.ndarray) -> list[float]:
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    tol_samp = int(round(BEAT_MATCH_TOL_MS * FS_TARGET / 1000.0))
    r1 = run_xqrs(y.astype(np.float32), fs=FS_TARGET)
    r2 = run_gqrs(y.astype(np.float32), fs=FS_TARGET)
    i_sqi = float(sqi_iSQI_li2008_global_per_lead([np.asarray(r1, dtype=int)], tol_samp)[0])
    b_sqi = float(sqi_bSQI_li2008_global(np.asarray(r1, dtype=int), np.asarray(r2, dtype=int), tol_samp))
    p_sqi = float(sqi_pSQI(y, fs=FS_TARGET, welch_kwargs=WELCH_KW))
    s_sqi = float(sqi_sSQI(y))
    k_sqi = float(sqi_kSQI(y))
    f_sqi = float(sqi_fSQI(y, flatline_eps=FLATLINE_EPS))
    bas_sqi = float(sqi_basSQI(y, fs=FS_TARGET, welch_kwargs=WELCH_KW))
    return [i_sqi, b_sqi, p_sqi, s_sqi, k_sqi, f_sqi, bas_sqi]


def compute_but_sqi_features(
    X: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    norm_stats: dict[str, Any],
    force: bool = False,
    max_rows: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_npy = out_dir / "but_record7_raw.npy"
    norm_npy = out_dir / "but_record7_norm.npy"
    raw_csv = out_dir / "but_record7_raw.csv"
    partial_npy = out_dir / "but_record7_raw_partial.npy"
    progress_path = out_dir / "but_sqi_progress.json"
    n = int(len(X) if max_rows <= 0 else min(len(X), max_rows))
    if raw_npy.exists() and norm_npy.exists() and not force and max_rows <= 0:
        audit = read_json(out_dir / "but_sqi_feature_audit.json")
        return np.load(norm_npy).astype(np.float32), audit

    if partial_npy.exists() and not force:
        raw = np.load(partial_npy).astype(np.float32)
        if raw.shape[0] != n:
            raw = np.full((n, len(SQI_COLUMNS)), np.nan, dtype=np.float32)
    else:
        raw = np.full((n, len(SQI_COLUMNS)), np.nan, dtype=np.float32)

    start = int(np.where(np.isfinite(raw).any(axis=1))[0][-1] + 1) if np.isfinite(raw).any() else 0
    failures: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i in range(start, n):
        try:
            raw[i] = np.asarray(sqi_for_signal(X[i, 0]), dtype=np.float32)
        except Exception as exc:
            failures.append({"i": int(i), "window_id": str(meta.iloc[i].get("window_id", i)), "error": f"{type(exc).__name__}: {exc}"})
            raw[i] = np.zeros(len(SQI_COLUMNS), dtype=np.float32)
        if (i + 1) % 100 == 0 or i + 1 == n:
            np.save(partial_npy, raw)
            update_state(
                progress_path,
                status="running",
                computed=int(i + 1),
                total=n,
                failures=len(failures),
                updated_at=now_iso(),
            )

    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm = apply_legacy_sqi_norm(raw, norm_stats)
    if max_rows <= 0:
        np.save(raw_npy, raw)
        np.save(norm_npy, norm)
        pd.DataFrame(raw, columns=SQI_COLUMNS).assign(
            window_id=meta["window_id"].astype(str).to_numpy(),
            split=meta["split"].astype(str).to_numpy(),
            y=meta["y"].to_numpy(dtype=int),
            y_class=meta["y_class"].astype(str).to_numpy(),
        ).to_csv(raw_csv, index=False)
    audit = {
        "feature_policy": "true seven SQI from noisy ECG only: iSQI,bSQI,pSQI,sSQI,kSQI,fSQI,basSQI",
        "columns": SQI_COLUMNS,
        "n": int(n),
        "failures": failures[:50],
        "failure_count": int(len(failures)),
        "duration_sec": float(time.perf_counter() - t0),
        "norm_policy": "reuse PTB SQI adapter normalization for sSQI/kSQI; downstream heads also train-only-standardize selected features",
        "norm_stats_source": norm_stats,
    }
    write_json(out_dir / "but_sqi_feature_audit.json", audit)
    update_state(progress_path, status="complete", computed=n, total=n, failures=len(failures), updated_at=now_iso())
    return norm, audit


def apply_legacy_sqi_norm(raw: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
    out = raw.astype(np.float32).copy()
    med = norm_stats.get("median_train", {}) or {}
    std = norm_stats.get("std_train", {}) or {}
    for name in ("I__sSQI", "I__kSQI"):
        if name not in SQI_COLUMNS:
            continue
        j = SQI_COLUMNS.index(name)
        scale = float(std.get(name, 1.0))
        if abs(scale) < EPS:
            scale = 1.0
        out[:, j] = (out[:, j] - float(med.get(name, 0.0))) / scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def cache_uformer_features(
    cache_path: Path,
    checkpoint: Path,
    X: np.ndarray,
    feature_set: str,
    batch_size: int,
    device: torch.device,
    force: bool = False,
) -> np.ndarray:
    if cache_path.exists() and not force:
        return np.load(cache_path).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feats = extract_uformer_features(checkpoint, X, feature_set, batch_size, device)
    np.save(cache_path, feats.astype(np.float32))
    return feats.astype(np.float32)


def fit_standardizer(X: np.ndarray, train_mask: np.ndarray) -> dict[str, np.ndarray]:
    mean = X[train_mask].mean(axis=0, keepdims=True)
    std = X[train_mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_standardizer(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((X - stats["mean"]) / stats["std"]).astype(np.float32)


class PlainMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.08):
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


class GatedSQIDeltaHead(nn.Module):
    def __init__(self, u_dim: int, sqi_dim: int = 7, hidden_dim: int = 128, delta_scale: float = 0.20, dropout: float = 0.08):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.base = PlainMLP(u_dim, hidden_dim, dropout)
        self.delta = nn.Sequential(
            nn.LayerNorm(sqi_dim),
            nn.Linear(sqi_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 3),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(u_dim + sqi_dim),
            nn.Linear(u_dim + sqi_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden_dim // 2), 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x[:, :-7]
        sqi = x[:, -7:]
        base_logits = self.base(u)
        gate = self.gate(x)
        return base_logits + self.delta_scale * gate * self.delta(sqi)


class BranchSQILogitFusionHead(nn.Module):
    """Two-branch head that gives SQI a fixed logit-level voice.

    The earlier gated-delta variants deliberately kept SQI small.  This branch
    makes the SQI contribution explicit: Uformer logits and SQI logits are
    learned separately, then combined with a fixed ratio.  That gives us a
    clean high-SQI ablation without changing the frozen Uformer features.
    """

    def __init__(
        self,
        u_dim: int,
        sqi_dim: int = 7,
        hidden_dim: int = 128,
        sqi_ratio: float = 1.0,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.sqi_ratio = float(sqi_ratio)
        self.u_branch = PlainMLP(u_dim, hidden_dim, dropout)
        self.sqi_branch = PlainMLP(sqi_dim, max(64, hidden_dim // 2), dropout)
        self.joint_bias = nn.Sequential(
            nn.LayerNorm(u_dim + sqi_dim),
            nn.Linear(u_dim + sqi_dim, max(64, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden_dim // 2), 3),
        )
        nn.init.zeros_(self.joint_bias[-1].weight)
        nn.init.zeros_(self.joint_bias[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x[:, :-7]
        sqi = x[:, -7:]
        return self.u_branch(u) + self.sqi_ratio * self.sqi_branch(sqi) + 0.10 * self.joint_bias(x)


def train_torch_head(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    train = split == "train"
    val = split == "val"
    cls_weight = torch.tensor([float(v) for v in str(args.class_weight).split(",")], dtype=torch.float32, device=device)
    ds = TensorDataset(
        torch.from_numpy(X[train]).float(),
        torch.from_numpy(y[train].astype(np.int64)).long(),
    )
    loader = DataLoader(ds, batch_size=int(args.head_batch_size), shuffle=True, num_workers=0)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr_head), weight_decay=float(args.weight_decay))
    best: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    log: list[dict[str, Any]] = []
    Xv = torch.from_numpy(X[val]).float().to(device)
    yv = y[val]
    for epoch in range(int(args.epochs_head)):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=cls_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * int(xb.shape[0])
            n += int(xb.shape[0])
        probs_val = predict_torch_probs(model, Xv, device=device, tensor_ready=True)
        pred_val = probs_val.argmax(axis=1)
        rep = multiclass_report(yv, pred_val, probs_val)
        rec = rep["recall_good_medium_bad"]
        key = (float(rep["acc"]), float(rep["macro_f1"]), float(rec[2]))
        row = {"epoch": epoch + 1, "train_loss": total / max(1, n), "val_report": rep}
        log.append(row)
        if best is None or key > best:
            best = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), {"variant": name, "train_log": log, "best_key": list(best or ())}


@torch.no_grad()
def predict_torch_probs(model: nn.Module, X: np.ndarray | torch.Tensor, device: torch.device, tensor_ready: bool = False) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    if tensor_ready:
        logits = model(X)
        return F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
    for s in range(0, len(X), 2048):
        xb = torch.from_numpy(np.asarray(X[s : s + 2048], dtype=np.float32)).to(device)
        logits = model(xb)
        probs.append(F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(probs, axis=0)


def classifier_scores_from_probs(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32)
    return probs.astype(np.float32), logits


def but_calibrations(probs_val: np.ndarray, logits_val: np.ndarray, y_val: np.ndarray) -> list[dict[str, Any]]:
    cals = [{"policy": "raw_argmax", "val_report": multiclass_report(y_val, probs_val.argmax(axis=1), probs_val)}]
    cals.extend(threshold_grid_many(probs_val, y_val, ["acc_primary", "balanced_primary", "macro_primary", "bad70_acc", "bad80_acc", "bad85_acc"]))
    cals.extend(bias_grid_many(logits_val, y_val, ["bias_acc", "bias_macro", "bias_bad70_acc"]))
    return cals


def eval_model_on_ptb_but(
    variant: str,
    probs_ptb: np.ndarray,
    probs_but: np.ndarray,
    y_ptb: np.ndarray,
    split_ptb: np.ndarray,
    y_but: np.ndarray,
    split_but: np.ndarray,
) -> dict[str, Any]:
    _, logits_but = classifier_scores_from_probs(probs_but)
    ptb_reports = {}
    for sp in ("train", "val", "test"):
        mask = split_ptb == sp
        ptb_reports[sp] = multiclass_report(y_ptb[mask], probs_ptb[mask].argmax(axis=1), probs_ptb[mask])
    val = split_but == "val"
    test = split_but == "test"
    raw_test = multiclass_report(y_but[test], probs_but[test].argmax(axis=1), probs_but[test])
    cals = but_calibrations(probs_but[val], logits_but[val], y_but[val])
    cal_rows: list[dict[str, Any]] = []
    for cal in cals:
        if cal["policy"] == "raw_argmax":
            pred = probs_but[test].argmax(axis=1)
        else:
            pred = apply_calibration(probs_but[test], logits_but[test], cal)
        cal_rows.append({"policy": cal["policy"], "calibration": cal, "test_report": multiclass_report(y_but[test], pred, probs_but[test])})

    def val_report(row: dict[str, Any]) -> dict[str, Any]:
        return row["calibration"]["val_report"]

    best_macro = max(cal_rows, key=lambda r: (val_report(r)["macro_f1"], val_report(r)["balanced_acc"], val_report(r)["acc"]))
    best_balanced = max(cal_rows, key=lambda r: (val_report(r)["balanced_acc"], val_report(r)["macro_f1"], val_report(r)["acc"]))
    best_acc = max(cal_rows, key=lambda r: (val_report(r)["acc"], val_report(r)["macro_f1"], val_report(r)["balanced_acc"]))
    return {
        "variant": variant,
        "ptb_reports": ptb_reports,
        "but_raw_test": raw_test,
        "but_calibrated_rows": cal_rows,
        "but_best_macro": best_macro,
        "but_best_balanced": best_balanced,
        "but_best_acc": best_acc,
        "calibration_source": "BUT validation split only",
    }


def fit_sklearn_logreg(X: np.ndarray, y: np.ndarray, split: np.ndarray, C: float = 1.0) -> Pipeline:
    train = split == "train"
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs")),
        ]
    )
    clf.fit(X[train], y[train])
    return clf


def export_predictions(
    out_csv: Path,
    meta: pd.DataFrame,
    probs: np.ndarray,
    calibrated: dict[str, Any],
    split: np.ndarray,
) -> None:
    _, logits = classifier_scores_from_probs(probs)
    if calibrated["policy"] == "raw_argmax":
        pred = probs.argmax(axis=1)
    else:
        pred = apply_calibration(probs, logits, calibrated["calibration"])
    out = meta.copy()
    out["pred_int"] = pred.astype(int)
    out["pred_class"] = [INT_TO_CLASS[int(v)] for v in pred]
    for i, cls in enumerate(CLASS_NAMES):
        out[f"prob_{cls}"] = probs[:, i]
    out["eval_split"] = split
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def export_but_galleries(X: np.ndarray, meta: pd.DataFrame, probs: np.ndarray, best: dict[str, Any], out_dir: Path) -> None:
    _, logits = classifier_scores_from_probs(probs)
    if best["policy"] == "raw_argmax":
        pred = probs.argmax(axis=1)
    else:
        pred = apply_calibration(probs, logits, best["calibration"])
    y = meta["y"].to_numpy(dtype=np.int64)
    split = meta["split"].astype(str).to_numpy()
    test = split == "test"
    masks = {
        "bad_missed": test & (y == 2) & (pred != 2),
        "correct_bad": test & (y == 2) & (pred == 2),
        "medium_confused": test & (y == 1) & (pred != 1),
        "good_false_bad": test & (y == 0) & (pred == 2),
    }
    for name, mask in masks.items():
        idx = np.flatnonzero(mask)[:24].tolist()
        if idx:
            plot_wave_gallery(X, meta, [int(i) for i in idx], out_dir / f"{name}.png", f"SQI fusion {name}", probs=probs)


def plot_sqi_distributions(ptb_meta: pd.DataFrame, ptb_sqi: np.ndarray, but_meta: pd.DataFrame, but_sqi: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    axes = axes.ravel()
    for j, col in enumerate(SQI_COLUMNS):
        ax = axes[j]
        for label, arr, meta, prefix in [
            ("PTB", ptb_sqi, ptb_meta, "solid"),
            ("BUT", but_sqi, but_meta, "dashed"),
        ]:
            for cls_i, cls in enumerate(CLASS_NAMES):
                mask = meta["y"].to_numpy(dtype=int) == cls_i
                vals = arr[mask, j]
                if len(vals):
                    ax.hist(vals, bins=35, alpha=0.25 if label == "PTB" else 0.18, density=True, label=f"{label} {cls}" if j == 0 else None)
        ax.set_title(col.replace("I__", ""))
        ax.grid(True, alpha=0.18)
    axes[-1].axis("off")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def prepare_features(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    cache = out_root / "feature_cache"
    state_path = out_root / STATE_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    update_state(state_path, status="preparing_features", updated_at=now_iso())
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    X_ptb, ptb_meta = load_ptb_split_and_signals(Path(args.source_artifact_dir), Path(args.ptb_sqi_control_dir))
    ptb_sqi, norm_stats = load_ptb_sqi_features(Path(args.ptb_sqi_control_dir), ptb_meta)
    X_but_obj, but_meta, is_ensemble = load_protocol_signals(Path(args.but_protocol_dir))
    if is_ensemble:
        raise ValueError("This runner expects formal 10s P1 protocol, not ensemble protocol.")
    X_but = X_but_obj.astype(np.float32)
    but_sqi, but_sqi_audit = compute_but_sqi_features(
        X_but,
        but_meta,
        cache / "but_sqi7",
        norm_stats,
        force=bool(args.force_sqi),
        max_rows=int(args.max_sqi_rows),
    )
    if int(args.max_sqi_rows) > 0:
        X_but = X_but[: int(args.max_sqi_rows)]
        but_meta = but_meta.iloc[: int(args.max_sqi_rows)].reset_index(drop=True)
    ptb_u = cache_uformer_features(
        cache / "ptb_uformer_full_tokens.npy",
        Path(args.checkpoint),
        X_ptb,
        "full_tokens",
        int(args.batch_size_eval),
        device,
        force=bool(args.force_uformer),
    )
    but_u = cache_uformer_features(
        cache / ("but_uformer_full_tokens_smoke.npy" if int(args.max_sqi_rows) > 0 else "but_uformer_full_tokens.npy"),
        Path(args.checkpoint),
        X_but,
        "full_tokens",
        int(args.batch_size_eval),
        device,
        force=bool(args.force_uformer),
    )
    audit = {
        "status": "features_ready",
        "device": str(device),
        "ptb": {"X_shape": list(X_ptb.shape), "uformer_shape": list(ptb_u.shape), "sqi_shape": list(ptb_sqi.shape), "split_counts": ptb_meta["split"].value_counts().to_dict()},
        "but": {"X_shape": list(X_but.shape), "uformer_shape": list(but_u.shape), "sqi_shape": list(but_sqi.shape), "split_counts": but_meta["split"].value_counts().to_dict()},
        "but_sqi_audit": but_sqi_audit,
        "checkpoint": str(args.checkpoint),
    }
    write_json(out_root / "feature_audit.json", audit)
    plot_sqi_distributions(ptb_meta, ptb_sqi, but_meta, but_sqi, out_root / "visuals" / "ptb_vs_but_sqi7_distributions.png")
    update_state(state_path, status="features_ready", updated_at=now_iso(), feature_audit=audit)
    return audit


def load_prepared(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    cache = out_root / "feature_cache"
    X_ptb, ptb_meta = load_ptb_split_and_signals(Path(args.source_artifact_dir), Path(args.ptb_sqi_control_dir))
    ptb_sqi, _ = load_ptb_sqi_features(Path(args.ptb_sqi_control_dir), ptb_meta)
    X_but_obj, but_meta, is_ensemble = load_protocol_signals(Path(args.but_protocol_dir))
    if is_ensemble:
        raise ValueError("This runner expects formal 10s P1 protocol, not ensemble protocol.")
    X_but = X_but_obj.astype(np.float32)
    but_sqi_path = cache / "but_sqi7" / "but_record7_norm.npy"
    but_u_path = cache / "but_uformer_full_tokens.npy"
    if not but_sqi_path.exists() or not but_u_path.exists():
        raise FileNotFoundError("Prepared BUT SQI/Uformer features are missing; run --stage prepare_features first.")
    return {
        "ptb_meta": ptb_meta,
        "but_meta": but_meta,
        "X_but": X_but,
        "ptb_u": np.load(cache / "ptb_uformer_full_tokens.npy").astype(np.float32),
        "but_u": np.load(but_u_path).astype(np.float32),
        "ptb_sqi": ptb_sqi.astype(np.float32),
        "but_sqi": np.load(but_sqi_path).astype(np.float32),
    }


def uses_uformer_sqi_features(variant: str) -> bool:
    return (
        variant == "uformer_plus_sqi_concat"
        or variant.startswith("uformer_plus_sqi_gated_delta_")
        or variant.startswith("uformer_plus_sqi_branch_ratio")
    )


def variant_features(data: dict[str, Any], variant: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if variant == "uformer_no_sqi":
        return data["ptb_u"], data["but_u"], {"u_dim": int(data["ptb_u"].shape[1]), "sqi_dim": 0}
    if variant == "sqi_only_7feat":
        return data["ptb_sqi"], data["but_sqi"], {"u_dim": 0, "sqi_dim": 7}
    if uses_uformer_sqi_features(variant):
        return (
            np.concatenate([data["ptb_u"], data["ptb_sqi"]], axis=1).astype(np.float32),
            np.concatenate([data["but_u"], data["but_sqi"]], axis=1).astype(np.float32),
            {"u_dim": int(data["ptb_u"].shape[1]), "sqi_dim": 7},
        )
    raise ValueError(variant)


def make_torch_model(variant: str, dims: dict[str, Any], args: argparse.Namespace) -> nn.Module:
    hidden = int(args.hidden_dim)
    if variant.startswith("uformer_plus_sqi_gated_delta_"):
        raw = variant.rsplit("_", 1)[-1].replace("p", ".")
        return GatedSQIDeltaHead(int(dims["u_dim"]), 7, hidden, delta_scale=float(raw), dropout=float(args.dropout))
    if variant.startswith("uformer_plus_sqi_branch_ratio"):
        raw = variant.replace("uformer_plus_sqi_branch_ratio", "").replace("p", ".")
        return BranchSQILogitFusionHead(int(dims["u_dim"]), 7, hidden, sqi_ratio=float(raw), dropout=float(args.dropout))
    in_dim = int(dims["u_dim"]) + int(dims["sqi_dim"])
    return PlainMLP(in_dim, hidden, float(args.dropout))


def torch_variants(args: argparse.Namespace) -> list[str]:
    base = [
        "uformer_no_sqi",
        "sqi_only_7feat",
        "uformer_plus_sqi_concat",
        "uformer_plus_sqi_gated_delta_0p10",
        "uformer_plus_sqi_gated_delta_0p25",
    ]
    heavy = [
        "uformer_no_sqi",
        "sqi_only_7feat",
        "uformer_plus_sqi_concat",
        "uformer_plus_sqi_gated_delta_0p25",
        "uformer_plus_sqi_gated_delta_0p50",
        "uformer_plus_sqi_gated_delta_1p00",
        "uformer_plus_sqi_gated_delta_2p00",
        "uformer_plus_sqi_branch_ratio0p50",
        "uformer_plus_sqi_branch_ratio1p00",
        "uformer_plus_sqi_branch_ratio2p00",
        "uformer_plus_sqi_branch_ratio4p00",
    ]
    if args.variant_set == "default":
        return base
    if args.variant_set == "heavy":
        return heavy
    if args.variant_set == "all":
        return list(dict.fromkeys(base + heavy))
    raise ValueError(args.variant_set)


def train_and_eval(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    state_path = out_root / STATE_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(state_path, status="training_heads", updated_at=now_iso())
    data = load_prepared(args)
    ptb_meta = data["ptb_meta"]
    but_meta = data["but_meta"]
    y_ptb = ptb_meta["y"].to_numpy(dtype=np.int64)
    split_ptb = ptb_meta["split"].astype(str).to_numpy()
    y_but = but_meta["y"].to_numpy(dtype=np.int64)
    split_but = but_meta["split"].astype(str).to_numpy()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    variants = torch_variants(args)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        update_state(state_path, status="training_heads", current=variant, updated_at=now_iso())
        X_ptb_raw, X_but_raw, dims = variant_features(data, variant)
        scaler = fit_standardizer(X_ptb_raw, split_ptb == "train")
        X_ptb = apply_standardizer(X_ptb_raw, scaler)
        X_but = apply_standardizer(X_but_raw, scaler)
        model = make_torch_model(variant, dims, args)
        model, train_meta = train_torch_head(variant, X_ptb, y_ptb, split_ptb, model, args, device)
        probs_ptb = predict_torch_probs(model, X_ptb, device)
        probs_but = predict_torch_probs(model, X_but, device)
        row = eval_model_on_ptb_but(variant, probs_ptb, probs_but, y_ptb, split_ptb, y_but, split_but)
        row["model_type"] = "torch_mlp_head"
        row["feature_dims"] = dims
        row["train_meta"] = train_meta
        rows.append(row)
        write_json(out_root / "runs" / variant / "result.json", row)
        write_json(out_root / "runs" / variant / "train_log.json", train_meta["train_log"])
        torch.save({"state_dict": model.state_dict(), "variant": variant, "dims": dims, "args": vars(args)}, out_root / "runs" / variant / "head.pt")
        export_predictions(
            out_root / "runs" / variant / "but_predictions.csv",
            but_meta,
            probs_but,
            row["but_best_macro"],
            split_but,
        )
        append_jsonl(out_root / SUMMARY_NAME, compact_result(row))
    # Add two fast linear controls so we can separate representation value from
    # PyTorch-head capacity.
    for variant in ("sqi_only_7feat", "uformer_plus_sqi_concat"):
        name = f"{variant}_logreg"
        update_state(state_path, status="training_heads", current=name, updated_at=now_iso())
        X_ptb_raw, X_but_raw, dims = variant_features(data, variant)
        clf = fit_sklearn_logreg(X_ptb_raw, y_ptb, split_ptb, C=1.0)
        probs_ptb = clf.predict_proba(X_ptb_raw).astype(np.float32)
        probs_but = clf.predict_proba(X_but_raw).astype(np.float32)
        row = eval_model_on_ptb_but(name, probs_ptb, probs_but, y_ptb, split_ptb, y_but, split_but)
        row["model_type"] = "sklearn_logreg_balanced"
        row["feature_dims"] = dims
        rows.append(row)
        write_json(out_root / "runs" / name / "result.json", row)
        append_jsonl(out_root / SUMMARY_NAME, compact_result(row))
    write_report(args, rows)
    best = best_overall(rows)
    export_but_galleries(data["X_but"], but_meta, best["probs_but"], best["best_macro"], out_root / "visuals" / "best_macro_but")
    update_state(state_path, status="complete", updated_at=now_iso(), best=compact_result(best["row"]))
    return {"status": "complete", "best": compact_result(best["row"]), "n_variants": len(rows)}


def compact_result(row: dict[str, Any]) -> dict[str, Any]:
    rec_raw = row["but_raw_test"]["recall_good_medium_bad"]
    best = row["but_best_macro"]
    rec_best = best["test_report"]["recall_good_medium_bad"]
    return {
        "variant": row["variant"],
        "model_type": row.get("model_type"),
        "ptb_test": row["ptb_reports"]["test"],
        "but_raw_test": row["but_raw_test"],
        "but_best_macro_policy": best["policy"],
        "but_best_macro_test": best["test_report"],
        "but_raw_recalls": rec_raw,
        "but_best_macro_recalls": rec_best,
        "feature_dims": row.get("feature_dims", {}),
        "calibration_source": row.get("calibration_source"),
    }


def best_overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_row = max(rows, key=lambda r: (r["but_best_macro"]["test_report"]["macro_f1"], r["but_best_macro"]["test_report"]["balanced_acc"], r["but_best_macro"]["test_report"]["acc"]))
    # Re-load probabilities from the saved result is intentionally avoided; the
    # caller has them only during train_and_eval.  Recompute for the best row.
    args = argparse.Namespace(**best_row.get("_args", {})) if "_args" in best_row else None
    raise RuntimeError("best_overall should be called through best_overall_runtime")


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    compact = [compact_result(r) for r in rows]
    ranked = sorted(compact, key=lambda r: (r["but_best_macro_test"]["macro_f1"], r["but_best_macro_test"]["balanced_acc"], r["but_best_macro_test"]["acc"]), reverse=True)
    write_json(report_root / "sqi_fusion_summary.json", {"rows": compact, "ranked_by_but_macro": ranked})
    lines = [
        "# BUT 10s PTB-Trained Seven-SQI Fusion",
        "",
        "This experiment freezes the Uformer denoiser/features, trains only PTB classifier heads, and evaluates transfer on formal BUT 10s P1.  Seven SQI features are computed from noisy ECG only.",
        "",
        "Calibration for BUT is validation-only.  Test never selects thresholds.",
        "",
        "| rank | variant | model | PTB acc | PTB recalls G/M/B | BUT raw acc/bal/macro | BUT raw recalls G/M/B | best val-cal policy | BUT cal acc/bal/macro | BUT cal recalls G/M/B |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(ranked, start=1):
        ptb = row["ptb_test"]
        raw = row["but_raw_test"]
        cal = row["but_best_macro_test"]
        lines.append(
            f"| {i} | {row['variant']} | {row.get('model_type')} | {ptb['acc']:.4f} | "
            f"{fmt_rec(ptb['recall_good_medium_bad'])} | "
            f"{raw['acc']:.4f}/{raw['balanced_acc']:.4f}/{raw['macro_f1']:.4f} | {fmt_rec(raw['recall_good_medium_bad'])} | "
            f"{row['but_best_macro_policy']} | {cal['acc']:.4f}/{cal['balanced_acc']:.4f}/{cal['macro_f1']:.4f} | {fmt_rec(cal['recall_good_medium_bad'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- If concat/gated variants beat `uformer_no_sqi` on BUT without PTB collapse, seven SQI is useful cross-domain quality evidence.",
            "- If `sqi_only_7feat` is strong, BUT quality labels are partly explainable by classical SQI statistics.",
            "- If SQI improves bad but damages medium/good, it remains an analysis branch rather than a mainline head.",
        ]
    )
    (report_root / "sqi_fusion_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_rec(values: list[float]) -> str:
    return "/".join(f"{float(v):.3f}" for v in values)


def train_and_eval_with_best(args: argparse.Namespace) -> dict[str, Any]:
    """Wrapper that keeps best probabilities available for gallery export."""
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    state_path = out_root / STATE_NAME
    data = load_prepared(args)
    ptb_meta = data["ptb_meta"]
    but_meta = data["but_meta"]
    y_ptb = ptb_meta["y"].to_numpy(dtype=np.int64)
    split_ptb = ptb_meta["split"].astype(str).to_numpy()
    y_but = but_meta["y"].to_numpy(dtype=np.int64)
    split_but = but_meta["split"].astype(str).to_numpy()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    probs_by_variant: dict[str, np.ndarray] = {}
    variants = torch_variants(args)
    update_state(state_path, status="training_heads", updated_at=now_iso())
    for variant in variants:
        update_state(state_path, status="training_heads", current=variant, updated_at=now_iso())
        X_ptb_raw, X_but_raw, dims = variant_features(data, variant)
        scaler = fit_standardizer(X_ptb_raw, split_ptb == "train")
        X_ptb = apply_standardizer(X_ptb_raw, scaler)
        X_but = apply_standardizer(X_but_raw, scaler)
        model = make_torch_model(variant, dims, args)
        model, train_meta = train_torch_head(variant, X_ptb, y_ptb, split_ptb, model, args, device)
        probs_ptb = predict_torch_probs(model, X_ptb, device)
        probs_but = predict_torch_probs(model, X_but, device)
        row = eval_model_on_ptb_but(variant, probs_ptb, probs_but, y_ptb, split_ptb, y_but, split_but)
        row["model_type"] = "torch_mlp_head"
        row["feature_dims"] = dims
        row["train_meta"] = train_meta
        rows.append(row)
        probs_by_variant[variant] = probs_but
        write_json(out_root / "runs" / variant / "result.json", row)
        write_json(out_root / "runs" / variant / "train_log.json", train_meta["train_log"])
        torch.save({"state_dict": model.state_dict(), "variant": variant, "dims": dims, "args": vars(args)}, out_root / "runs" / variant / "head.pt")
        append_jsonl(out_root / SUMMARY_NAME, compact_result(row))
    for variant in ("sqi_only_7feat", "uformer_plus_sqi_concat"):
        name = f"{variant}_logreg"
        update_state(state_path, status="training_heads", current=name, updated_at=now_iso())
        X_ptb_raw, X_but_raw, dims = variant_features(data, variant)
        clf = fit_sklearn_logreg(X_ptb_raw, y_ptb, split_ptb, C=1.0)
        probs_ptb = clf.predict_proba(X_ptb_raw).astype(np.float32)
        probs_but = clf.predict_proba(X_but_raw).astype(np.float32)
        row = eval_model_on_ptb_but(name, probs_ptb, probs_but, y_ptb, split_ptb, y_but, split_but)
        row["model_type"] = "sklearn_logreg_balanced"
        row["feature_dims"] = dims
        rows.append(row)
        probs_by_variant[name] = probs_but
        write_json(out_root / "runs" / name / "result.json", row)
        append_jsonl(out_root / SUMMARY_NAME, compact_result(row))
    write_report(args, rows)
    ranked = sorted(rows, key=lambda r: (r["but_best_macro"]["test_report"]["macro_f1"], r["but_best_macro"]["test_report"]["balanced_acc"], r["but_best_macro"]["test_report"]["acc"]), reverse=True)
    best = ranked[0]
    export_but_galleries(data["X_but"], but_meta, probs_by_variant[best["variant"]], best["but_best_macro"], out_root / "visuals" / "best_macro_but")
    export_predictions(out_root / "best_macro_but_predictions.csv", but_meta, probs_by_variant[best["variant"]], best["but_best_macro"], split_but)
    runtime = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "finished_at": now_iso(),
    }
    write_json(out_root / "runtime_hardware.json", runtime)
    update_state(state_path, status="complete", updated_at=now_iso(), best=compact_result(best), runtime_hardware=runtime)
    return {"status": "complete", "best": compact_result(best), "n_variants": len(rows)}


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    args.max_sqi_rows = max(48, int(args.max_sqi_rows or 0))
    audit = prepare_features(args)
    payload = {"status": "smoke_ok", "feature_audit": audit}
    write_json(Path(args.out_root) / "smoke.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        return smoke(args)
    if args.stage in {"prepare_features", "all"}:
        prepare_features(args)
    if args.stage in {"train_eval", "all"}:
        return train_and_eval_with_best(args)
    if args.stage == "report":
        rows = [r for r in (read_json(p) for p in (Path(args.out_root) / "runs").glob("*/result.json"))]
        write_report(args, rows)
        return {"status": "report_complete", "rows": len(rows)}
    raise ValueError(args.stage)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PTB seven-SQI fusion heads and evaluate on BUT 10s P1.")
    parser.add_argument("--stage", choices=("smoke", "prepare_features", "train_eval", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--ptb_sqi_control_dir", default=str(DEFAULT_PTB_SQI_CONTROL))
    parser.add_argument("--but_protocol_dir", default=str(DEFAULT_BUT_PROTOCOL_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MAINLINE_CKPT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--head_batch_size", type=int, default=512)
    parser.add_argument("--epochs_head", type=int, default=70)
    parser.add_argument("--lr_head", type=float, default=7e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--class_weight", default="1,1.40,1.70")
    parser.add_argument("--variant_set", choices=("default", "heavy", "all"), default="default")
    parser.add_argument("--force_sqi", action="store_true")
    parser.add_argument("--force_uformer", action="store_true")
    parser.add_argument("--max_sqi_rows", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        payload = run(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
