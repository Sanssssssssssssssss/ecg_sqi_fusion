from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
MORPH_EXP = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid"
MORPH_TRAIN = MORPH_EXP / "morph_train.py"
DEFAULT_SOURCE = MORPH_EXP / "data" / "med6p25_badgap7_badcm0p75"

CLASS_NAMES = ["good", "medium", "bad"]


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


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        groups = max(1, min(8, out_ch // 4))
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock1D(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.transpose(1, 2)
        h = self.ln1(t)
        a, _ = self.attn(h, h, h, need_weights=False)
        t = t + self.drop(a)
        t = t + self.mlp(self.ln2(t))
        return t.transpose(1, 2)


class ConformerBlock1D(nn.Module):
    def __init__(self, dim: int, heads: int = 4, kernel: int = 15, dropout: float = 0.05):
        super().__init__()
        self.ff1 = nn.Sequential(nn.Conv1d(dim, dim * 2, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(dim * 2, dim, 1))
        self.tr = TransformerBlock1D(dim, heads=heads, mlp_ratio=2, dropout=dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim),
            nn.GroupNorm(max(1, min(8, dim // 4)), dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(nn.Conv1d(dim, dim * 2, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(dim * 2, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = self.tr(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ConvTransUNetLite(nn.Module):
    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.e1 = ConvBlock(1, 32)
        self.d1 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.e2 = ConvBlock(64, 64)
        self.d2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.e3 = ConvBlock(128, 128)
        self.tr = nn.Sequential(TransformerBlock1D(128, 4, dropout=dropout), TransformerBlock1D(128, 4, dropout=dropout))
        self.u2 = UpBlock(128, 64, 64)
        self.u1 = UpBlock(64, 32, 32)
        self.out = nn.Conv1d(32, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.e1(x)
        s2 = self.e2(F.gelu(self.d1(s1)))
        b = self.e3(F.gelu(self.d2(s2)))
        b = self.tr(b)
        y = self.u2(b, s2)
        y = self.u1(y, s1)
        noise_hat = self.out(y)
        return {"noise_hat": noise_hat, "features": [s1, s2, b], "bottleneck": b}


class UFormer1DHier(nn.Module):
    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.e1 = ConvBlock(1, 32, kernel=9)
        self.d1 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(ConvBlock(64, 64), TransformerBlock1D(64, 4, dropout=dropout))
        self.d2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.e3 = nn.Sequential(ConvBlock(128, 128), TransformerBlock1D(128, 4, dropout=dropout))
        self.d3 = nn.Conv1d(128, 160, kernel_size=4, stride=2, padding=1)
        self.b = nn.Sequential(ConvBlock(160, 160), TransformerBlock1D(160, 5, dropout=dropout), TransformerBlock1D(160, 5, dropout=dropout))
        self.u3 = UpBlock(160, 128, 96)
        self.u2 = UpBlock(96, 64, 64)
        self.u1 = UpBlock(64, 32, 32)
        self.out = nn.Conv1d(32, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.e1(x)
        s2 = self.e2(F.gelu(self.d1(s1)))
        s3 = self.e3(F.gelu(self.d2(s2)))
        b = self.b(F.gelu(self.d3(s3)))
        y = self.u3(b, s3)
        y = self.u2(y, s2)
        y = self.u1(y, s1)
        noise_hat = self.out(y)
        return {"noise_hat": noise_hat, "features": [s1, s2, s3, b], "bottleneck": b}


class ConformerUNet1D(nn.Module):
    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.e1 = ConvBlock(1, 32, kernel=11)
        self.d1 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(ConvBlock(64, 64), ConformerBlock1D(64, 4, dropout=dropout))
        self.d2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.e3 = nn.Sequential(ConvBlock(128, 128), ConformerBlock1D(128, 4, dropout=dropout), ConformerBlock1D(128, 4, dropout=dropout))
        self.u2 = UpBlock(128, 64, 64)
        self.u1 = UpBlock(64, 32, 32)
        self.out = nn.Conv1d(32, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.e1(x)
        s2 = self.e2(F.gelu(self.d1(s1)))
        b = self.e3(F.gelu(self.d2(s2)))
        y = self.u2(b, s2)
        y = self.u1(y, s1)
        noise_hat = self.out(y)
        return {"noise_hat": noise_hat, "features": [s1, s2, b], "bottleneck": b}


def build_denoiser(model_type: str, dropout: float) -> nn.Module:
    if model_type == "conv_transunet_lite":
        return ConvTransUNetLite(dropout=dropout)
    if model_type == "uformer1d_hier":
        return UFormer1DHier(dropout=dropout)
    if model_type == "conformer_unet1d":
        return ConformerUNet1D(dropout=dropout)
    raise ValueError(f"unknown model_type={model_type}")


def feature_vector(feature_set: str, noisy: torch.Tensor, denoise: torch.Tensor, features: list[torch.Tensor], bottleneck: torch.Tensor) -> torch.Tensor:
    residual = torch.abs(noisy - denoise)
    summaries = {
        "summary_only": torch.cat([tensor_summary(noisy), tensor_summary(denoise), tensor_summary(residual)], dim=1),
        "residual_summary_only": tensor_summary(residual),
        "bottleneck_only": pooled_channels(bottleneck),
    }
    if feature_set in summaries:
        return summaries[feature_set]
    if feature_set != "full_tokens":
        raise ValueError(f"unknown feature_set={feature_set}")
    pooled = [pooled_channels(x) for x in features]
    return torch.cat([*pooled, summaries["summary_only"]], dim=1)


class SQIMLPHead(nn.Module):
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


class TransUNetSQIModel(nn.Module):
    def __init__(self, denoiser: nn.Module, head: SQIMLPHead | None, noise_scale: float, feature_set: str, detach_features: bool):
        super().__init__()
        self.denoiser = denoiser
        self.head = head
        self.noise_scale = float(noise_scale)
        self.feature_set = feature_set
        self.detach_features = bool(detach_features)

    def forward(self, noisy: torch.Tensor) -> dict[str, torch.Tensor | None]:
        out = self.denoiser(noisy)
        noise_hat = out["noise_hat"]
        denoise = noisy - self.noise_scale * noise_hat
        feat = feature_vector(self.feature_set, noisy, denoise, out["features"], out["bottleneck"])
        if self.detach_features:
            feat = feat.detach()
        logits = self.head(feat) if self.head is not None else None
        return {"noise_hat": noise_hat, "denoise": denoise, "features": feat, "logits": logits}


def cls_report(y: np.ndarray, logits: np.ndarray | None) -> dict[str, Any]:
    if logits is None:
        return {"acc": None, "recall_good_medium_bad": [None, None, None], "confusion_3x3": None}
    pred = np.argmax(logits, axis=1)
    mat = np.zeros((3, 3), dtype=int)
    rec: list[float] = []
    for a, b in zip(y, pred):
        mat[int(a), int(b)] += 1
    for c in range(3):
        denom = max(1, int(np.sum(y == c)))
        rec.append(float(np.sum((y == c) & (pred == c)) / denom))
    return {"acc": float(np.mean(pred == y)), "recall_good_medium_bad": rec, "confusion_3x3": mat.tolist()}


@torch.no_grad()
def collect_outputs(model: TransUNetSQIModel, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in ["noisy", "clean", "denoise_pred", "y", "logits", "idx", "ecg_id", "snr_db", "original_snr_db", "local_mask", "critical_mask", "qrs_mask", "tst_mask"]}
    noise_kind_all: list[np.ndarray] = []
    placement_all: list[np.ndarray] = []
    for batch in loader:
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        out = model(noisy)
        denoise = out["denoise"]
        chunks["noisy"].append(noisy.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["clean"].append(batch["clean"].squeeze(1).numpy().astype(np.float32))
        chunks["denoise_pred"].append(denoise.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["y"].append(batch["y"].numpy().astype(np.int64))
        logits = out["logits"]
        if logits is None:
            chunks["logits"].append(np.zeros((int(noisy.shape[0]), 3), dtype=np.float32))
        else:
            chunks["logits"].append(logits.cpu().numpy().astype(np.float32))
        for key in ["idx", "ecg_id"]:
            chunks[key].append(batch[key].numpy().astype(np.int64))
        for key in ["snr_db", "original_snr_db", "local_mask", "critical_mask", "qrs_mask", "tst_mask"]:
            chunks[key].append(batch[key].numpy().astype(np.float32))
        noise_kind_all.append(np.asarray(batch["noise_kind"], dtype=str))
        placement_all.append(np.asarray(batch["placement"], dtype=str))
    arrays = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    arrays["noise_kind"] = np.concatenate(noise_kind_all, axis=0)
    arrays["placement"] = np.concatenate(placement_all, axis=0)
    logits = arrays["logits"]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-8)
    arrays["prob_denoise"] = probs.astype(np.float32)
    arrays["y_pred_denoise"] = np.argmax(arrays["logits"], axis=1).astype(np.int64)
    arrays["y_pred_base"] = arrays["y_pred_denoise"]
    arrays["prob_noisy"] = arrays["prob_denoise"]
    return arrays


def select_positions(arrays: dict[str, np.ndarray], mode: str, max_rows: int = 12) -> list[int]:
    y = arrays["y"].astype(np.int64)
    noisy = arrays["noisy"]
    clean = arrays["clean"]
    denoise = arrays["denoise_pred"]
    noise_mse = np.mean((noisy - clean) ** 2, axis=1)
    pred_mse = np.mean((denoise - clean) ** 2, axis=1)
    local = arrays.get("critical_mask", np.zeros_like(clean)) + arrays.get("qrs_mask", np.zeros_like(clean)) + arrays.get("tst_mask", np.zeros_like(clean))
    local_mass = np.mean(local, axis=1)
    chosen: list[int] = []

    def add(items) -> None:
        for item in list(items):
            pos = int(item)
            if pos not in chosen:
                chosen.append(pos)
            if len(chosen) >= max_rows:
                break

    if mode == "balanced":
        for c in [0, 1, 2]:
            idx = np.where(y == c)[0]
            add(idx[np.argsort(noise_mse[idx])[::-1]][:4])
    elif mode == "hard_bad":
        idx = np.where(y >= 1)[0]
        add(idx[np.argsort(noise_mse[idx])[::-1]][:max_rows])
    elif mode == "good_safety":
        idx = np.where(y == 0)[0]
        add(idx[np.argsort((pred_mse - noise_mse)[idx])[::-1]][:max_rows])
    elif mode == "worst_residual":
        add(np.argsort(pred_mse)[::-1][:max_rows])
    elif mode == "qrs_tst_focus":
        add(np.argsort(local_mass)[::-1][:max_rows])
    return chosen[:max_rows]


def export_fixed_gallery(arrays: dict[str, np.ndarray], positions: list[int], out_png: Path, title: str) -> None:
    if not positions:
        return
    t = np.arange(arrays["clean"].shape[1], dtype=np.float32) / 125.0
    cols = [("clean", arrays["clean"]), ("before/noisy", arrays["noisy"]), ("after/denoise", arrays["denoise_pred"])]
    fig, axes = plt.subplots(len(positions), 3, figsize=(12.2, max(5.8, 1.25 * len(positions))), sharex=True)
    if len(positions) == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, pos in enumerate(positions):
        row_min = min(float(arr[pos].min()) for _, arr in cols)
        row_max = max(float(arr[pos].max()) for _, arr in cols)
        pad = 0.06 * (row_max - row_min + 1e-8)
        y = int(arrays["y"][pos])
        pred = int(arrays["y_pred_denoise"][pos])
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
                    f"ecg={int(arrays['ecg_id'][pos])} snr={float(arrays['snr_db'][pos]):.2f}\n"
                    f"{arrays['noise_kind'][pos]} {arrays['placement'][pos]}\n"
                    f"pred={CLASS_NAMES[pred]}",
                    fontsize=7.2,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=58,
                )
            else:
                ax.set_yticklabels([])
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def evaluate(model: TransUNetSQIModel, loader: DataLoader, device: torch.device, morph: Any, out_dir: Path | None) -> dict[str, Any]:
    arrays = collect_outputs(model, loader, device)
    y = arrays["y"].astype(np.int64)
    report = cls_report(y, arrays["logits"] if model.head is not None else None)
    masks = {"local": arrays["local_mask"], "critical": arrays["critical_mask"], "qrs": arrays["qrs_mask"], "tst": arrays["tst_mask"]}
    metrics = morph.build_metrics(arrays["clean"], arrays["noisy"], arrays["denoise_pred"], y, arrays["noise_kind"], arrays["placement"], masks)
    if out_dir is not None:
        (out_dir / "denoise_eval").mkdir(parents=True, exist_ok=True)
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "test_report.json", report)
        write_json(out_dir / "denoise_eval" / "denoise_metrics.json", metrics)
        np.savez_compressed(
            out_dir / "denoise_eval" / "test_denoise_outputs.npz",
            noisy=arrays["noisy"],
            clean=arrays["clean"],
            denoise_pred=arrays["denoise_pred"],
            y=y,
            y_pred=arrays["y_pred_denoise"],
            idx=arrays["idx"],
            ecg_id=arrays["ecg_id"],
            snr_db=arrays["snr_db"],
            noise_kind=arrays["noise_kind"],
            placement=arrays["placement"],
        )
        morph.export_gallery(arrays, out_dir / "debug" / "denoise_compact_test.png", hard_only=False, title="Transformer-denoiser compact gallery")
        morph.export_gallery(arrays, out_dir / "debug" / "denoise_hard_test.png", hard_only=True, title="Transformer-denoiser hard gallery")
        for mode in ["balanced", "hard_bad", "good_safety", "worst_residual", "qrs_tst_focus"]:
            export_fixed_gallery(arrays, select_positions(arrays, mode), out_dir / "debug" / f"{mode}_gallery.png", f"{mode} gallery")
        (out_dir / "visual_review.md").write_text(
            "\n".join(
                [
                    "# Visual Review",
                    "",
                    "Automatic first-pass visual audit. Inspect the fixed galleries for good overfiltering, QRS/T distortion, baseline drift left, HF noise left, and morph damage not repaired.",
                    "",
                    f"- acc: {report.get('acc')}",
                    f"- recall_good_medium_bad: {report.get('recall_good_medium_bad')}",
                    f"- denoise_score: {metrics['overall'].get('denoise_score')}",
                    f"- mse_ratio_pred_over_noisy: {metrics['overall'].get('mse_ratio_pred_over_noisy')}",
                    f"- snr_improve_db_mean: {metrics['overall'].get('snr_improve_db_mean')}",
                    "",
                    "## Galleries",
                    "- debug/balanced_gallery.png",
                    "- debug/hard_bad_gallery.png",
                    "- debug/good_safety_gallery.png",
                    "- debug/worst_residual_gallery.png",
                    "- debug/qrs_tst_focus_gallery.png",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return {"report": report, "metrics": metrics}


def make_loaders(morph: Any, source: Path, batch_size: int, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader, Any, Any, Any]:
    train_ds = morph.ECGDenoiseDataset(source, "train")
    val_ds = morph.ECGDenoiseDataset(source, "val")
    test_ds = morph.ECGDenoiseDataset(source, "test")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda"),
        DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"),
        DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"),
        train_ds,
        val_ds,
        test_ds,
    )


def load_checkpoint(model: TransUNetSQIModel, path: str, device: torch.device) -> dict[str, Any]:
    if str(path).lower() in {"", "none", "null"}:
        return {"loaded": False}
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return {"loaded": False, "missing": str(ckpt_path)}
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("denoiser_state", ckpt))
    missing, unexpected = model.denoiser.load_state_dict(state, strict=False)
    return {"loaded": True, "checkpoint": str(ckpt_path), "missing": len(missing), "unexpected": len(unexpected)}


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    morph = import_module(MORPH_TRAIN, "e311_morph_train_transunet1d")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_loaders(morph, Path(args.source_artifact_dir), int(args.batch_size), device)
    print(f"device={device}", flush=True)
    print(f"splits train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    denoiser = build_denoiser(str(args.model_type), float(args.dropout)).to(device)
    model = TransUNetSQIModel(denoiser, None, float(args.noise_scale), str(args.feature_set), bool(args.detach_encoder_features)).to(device)
    load_report = load_checkpoint(model, str(args.init_checkpoint), device)
    print(f"checkpoint_load={load_report}", flush=True)
    if bool(args.freeze_denoiser):
        for p in model.denoiser.parameters():
            p.requires_grad_(False)
        model.denoiser.eval()

    if args.stage in {"sqi_head", "joint_finetune"}:
        with torch.no_grad():
            sample = next(iter(train_loader))["noisy"].to(device=device, dtype=torch.float32)
            feat_dim = int(model(sample)["features"].shape[1])
        model.head = SQIMLPHead(feat_dim, int(args.hidden_dim), float(args.dropout)).to(device)
        print(f"sqi_head feat_dim={feat_dim}", flush=True)

    if args.stage == "sqi_head" and not bool(args.freeze_denoiser) and bool(args.detach_encoder_features):
        print("sqi_head detach mode: CE will not update denoiser through encoder features", flush=True)

    params: list[dict[str, Any]] = []
    den_params = [p for p in model.denoiser.parameters() if p.requires_grad]
    if den_params:
        params.append({"params": den_params, "lr": float(args.lr_denoiser)})
    if model.head is not None:
        params.append({"params": model.head.parameters(), "lr": float(args.lr_head)})
    if not params:
        raise RuntimeError("No trainable parameters")
    opt = torch.optim.AdamW(params, weight_decay=float(args.weight_decay))
    class_weight = torch.tensor([float(v) for v in str(args.class_weight).split(",")], dtype=torch.float32, device=device)

    log: list[dict[str, Any]] = []
    best_key: tuple[float, ...] = (-1e9,)
    best_state: dict[str, Any] | None = None
    for epoch in range(int(args.epochs)):
        model.train()
        if bool(args.freeze_denoiser):
            model.denoiser.eval()
        sums = {"total": 0.0, "den": 0.0, "cls": 0.0}
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
            l_den, _ = morph.morph_loss(out["denoise"], out["noise_hat"], noisy, clean, y, point_weight, sample_weight, str(args.loss_variant))
            l_cls = torch.zeros((), device=device)
            if out["logits"] is not None and float(args.lambda_cls) > 0.0:
                l_cls = F.cross_entropy(out["logits"], y, weight=class_weight)
            loss = float(args.lambda_den) * l_den + float(args.lambda_cls) * l_cls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            bs = int(noisy.shape[0])
            n += bs
            sums["total"] += float(loss.detach().cpu().item()) * bs
            sums["den"] += float(l_den.detach().cpu().item()) * bs
            sums["cls"] += float(l_cls.detach().cpu().item()) * bs
        val = evaluate(model, val_loader, device, morph, None)
        report = val["report"]
        metrics = val["metrics"]["overall"]
        acc = report.get("acc")
        den_score = float(metrics["denoise_score"])
        key = (den_score, float(metrics["snr_improve_db_mean"])) if acc is None else (float(acc), den_score)
        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": v / max(1, n) for k, v in sums.items()},
            "val_acc": acc,
            "val_recall_good_medium_bad": report.get("recall_good_medium_bad"),
            "val_denoise_score": den_score,
            "val_snr_gain": metrics["snr_improve_db_mean"],
            "val_mse_ratio": metrics["mse_ratio_pred_over_noisy"],
        }
        log.append(row)
        write_json(out_dir / "train_log.json", log)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={row['train_total']:.4f} "
            f"val_acc={acc} den={den_score:.3f} snr+={row['val_snr_gain']:.3f}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_state = {"epoch": epoch + 1, "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "key": key}
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    result = evaluate(model, test_loader, device, morph, out_dir)
    torch.save({"model_state": model.denoiser.state_dict(), "full_model_state": model.state_dict(), "args": vars(args), "best_state_meta": best_state}, out_dir / "ckpt_best_transunet1d.pt")
    summary = {
        "status": "completed",
        "stage": args.stage,
        "model_type": args.model_type,
        "args": vars(args),
        "split_counts": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "checkpoint_load": load_report,
        "best_epoch": None if best_state is None else best_state["epoch"],
        "test_report": result["report"],
        "denoise_metrics": result["metrics"]["overall"],
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False)[:2400], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment-only 1D TransUNet/Uformer/Conformer denoise + SQI audit.")
    parser.add_argument("--stage", choices=("denoise_pretrain", "sqi_head", "joint_finetune"), default="denoise_pretrain")
    parser.add_argument("--model_type", choices=("conv_transunet_lite", "uformer1d_hier", "conformer_unet1d"), required=True)
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_checkpoint", default="none")
    parser.add_argument("--feature_set", choices=("full_tokens", "bottleneck_only", "summary_only", "residual_summary_only"), default="full_tokens")
    parser.add_argument("--freeze_denoiser", action="store_true")
    parser.add_argument("--detach_encoder_features", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--loss_variant", default="morph_soft_guard")
    parser.add_argument("--noise_scale", type=float, default=0.955)
    parser.add_argument("--lambda_den", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.0)
    parser.add_argument("--class_weight", default="1,1.40,1.70")
    parser.add_argument("--lr_denoiser", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
