from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from src.transformer_pipeline.models.mtl_transformer import MTLTransformerConfig, MTLTransformerPTBXL
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.transformer_pipeline.models.mtl_transformer import MTLTransformerConfig, MTLTransformerPTBXL
    from src.utils.paths import project_root


class NoisyCleanDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray) -> None:
        self.noisy = noisy.astype(np.float32)
        self.clean = clean.astype(np.float32)

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x_noisy": torch.from_numpy(self.noisy[i][None, :]),
            "x_clean": torch.from_numpy(self.clean[i][None, :]),
        }


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer_e6_local_counterfactual")
    out_dir = _path(
        params.get("out_dir"),
        artifact_dir / "pretrain" / "e7_masked_denoise",
    )
    seed = int(params.get("seed", 0))
    epochs = int(params.get("epochs", 8))
    batch_size = int(params.get("batch_size", 64))
    lr = float(params.get("lr", 8e-5))
    weight_decay = float(params.get("weight_decay", 0.03))
    dropout = float(params.get("dropout", 0.05))
    mask_ratio = float(params.get("mask_ratio", 0.35))
    lambda_mask = float(params.get("lambda_mask", 1.0))
    lambda_full = float(params.get("lambda_full", 0.5))
    num_workers = int(params.get("num_workers", 0))
    init_checkpoint = str(params.get("init_checkpoint", ""))
    dry_run = bool(params.get("dry_run", False))
    force = bool(params.get("force", False))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_last = out_dir / "ckpt_last.pt"
    out_best = out_dir / "ckpt_best_val.pt"
    out_log = out_dir / "pretrain_log.json"
    out_summary = out_dir / "pretrain_summary.json"
    outputs = [out_last, out_best, out_log, out_summary]
    if all(p.exists() for p in outputs) and not force:
        print(f"E7 pretrain outputs exist: {_rel(out_dir, root)}")
        return {"step": "e7_masked_denoise_pretrain", "skipped": True, "outputs": [str(p) for p in outputs]}

    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    clean, noisy, labels = load_arrays(artifact_dir)
    train_ds = make_split_dataset(noisy, clean, labels, "train")
    val_ds = make_split_dataset(noisy, clean, labels, "val")
    print(f"[train] n={len(train_ds)}")
    print(f"[val] n={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    cfg = MTLTransformerConfig(dropout=dropout)
    model = MTLTransformerPTBXL(cfg).to(device)
    if init_checkpoint:
        ckpt_path = _path(init_checkpoint, root / init_checkpoint)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"initialized from {_rel(ckpt_path, root)} | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"init missing keys: {list(missing)[:8]}")
        if unexpected:
            print(f"init unexpected keys: {list(unexpected)[:8]}")

    if dry_run:
        batch = next(iter(train_loader))
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        mask = make_patch_mask(x_noisy.shape[0], cfg, mask_ratio, device)
        x_in = x_noisy.masked_fill(mask[:, None, :], 0.0)
        with torch.no_grad():
            y_denoise = model(x_in)[0]
        if tuple(y_denoise.shape) != tuple(x_noisy.shape):
            raise RuntimeError(f"denoise shape mismatch: {tuple(y_denoise.shape)} != {tuple(x_noisy.shape)}")
        print("[dry-run] E7 masked denoise pretrain forward OK")
        return {"step": "e7_masked_denoise_pretrain", "skipped": True, "dry_run": True, "outputs": []}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=max(lr * 0.05, 1e-6))
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, cfg, mask_ratio, lambda_mask, lambda_full
        )
        val_metrics = run_epoch(
            model, val_loader, device, None, cfg, mask_ratio, lambda_mask, lambda_full
        )
        scheduler.step()

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"train L={train_metrics['loss']:.6f} mask={train_metrics['masked_mse']:.6f} full={train_metrics['full_mse']:.6f} | "
            f"val L={val_metrics['loss']:.6f} mask={val_metrics['masked_mse']:.6f} full={val_metrics['full_mse']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "hyperparams": {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "mask_ratio": mask_ratio,
                "lambda_mask": lambda_mask,
                "lambda_full": lambda_full,
                "init_checkpoint": init_checkpoint,
            },
        }
        torch.save(ckpt, out_last)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            torch.save(ckpt, out_best)

    out_log.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "last_val_loss": history[-1]["val"]["loss"] if history else None,
        "artifact_dir": _rel(artifact_dir, root),
        "out_dir": _rel(out_dir, root),
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_val_loss={best_val:.6f} epoch={best_epoch}")
    print(f"saved: {_rel(out_best, root)}")
    return {"step": "e7_masked_denoise_pretrain", "skipped": False, "outputs": [str(p) for p in outputs]}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cfg: MTLTransformerConfig,
    mask_ratio: float,
    lambda_mask: float,
    lambda_full: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    n = 0
    loss_sum = 0.0
    mask_sum = 0.0
    full_sum = 0.0
    for batch in loader:
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        x_clean = batch["x_clean"].to(device=device, dtype=torch.float32)
        mask = make_patch_mask(x_noisy.shape[0], cfg, mask_ratio, device)
        x_in = x_noisy.masked_fill(mask[:, None, :], 0.0)

        with torch.set_grad_enabled(train_mode):
            y_denoise = model(x_in)[0]
            err = (y_denoise - x_clean) ** 2
            full_mse = err.mean()
            mask_f = mask[:, None, :].to(dtype=err.dtype)
            masked_mse = (err * mask_f).sum() / mask_f.sum().clamp_min(1.0)
            loss = lambda_mask * masked_mse + lambda_full * full_mse
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bsz = int(x_noisy.shape[0])
        n += bsz
        loss_sum += float(loss.detach().cpu()) * bsz
        mask_sum += float(masked_mse.detach().cpu()) * bsz
        full_sum += float(full_mse.detach().cpu()) * bsz

    den = max(1, n)
    return {"loss": loss_sum / den, "masked_mse": mask_sum / den, "full_mse": full_sum / den}


def make_patch_mask(batch_size: int, cfg: MTLTransformerConfig, mask_ratio: float, device: torch.device) -> torch.Tensor:
    token_mask = torch.rand((batch_size, cfg.L), device=device) < mask_ratio
    mask = torch.zeros((batch_size, cfg.T), device=device, dtype=torch.bool)
    for i in range(cfg.L):
        if token_mask[:, i].any():
            start = i * cfg.stride
            mask[:, start : start + cfg.patch_size] |= token_mask[:, i : i + 1]
    return mask


def load_arrays(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    clean_path = artifact_dir / "datasets" / "synth_10s_125hz_clean.npz"
    noisy_path = artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz"
    labels_path = artifact_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        labels_path = artifact_dir / "datasets" / "synth_10s_125hz_labels.csv"
    clean = np.load(clean_path)["X_clean"].astype(np.float32)
    noisy = np.load(noisy_path)["X_noisy"].astype(np.float32)
    labels = pd.read_csv(labels_path)
    if not (clean.shape[0] == noisy.shape[0] == len(labels)):
        raise ValueError("array/label row count mismatch")
    return clean, noisy, labels


def make_split_dataset(noisy: np.ndarray, clean: np.ndarray, labels: pd.DataFrame, split: str) -> NoisyCleanDataset:
    idx = labels.index[labels["split"].astype(str).to_numpy() == split].to_numpy(dtype=np.int64)
    return NoisyCleanDataset(noisy[idx], clean[idx])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    run(vars(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7 masked denoising pretraining for the PTB-XL transformer.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_e6_local_counterfactual")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.35)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--lambda_full", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
