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
import torch.nn.functional as F
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


Y_TO_INT = {"good": 0, "medium": 1, "bad": 2}
ORDERED = ("good", "medium", "bad")


class TripletSeverityDataset(Dataset):
    def __init__(
        self,
        noisy: np.ndarray,
        labels: pd.DataFrame,
        split: str,
        max_groups: int | None = None,
        groups: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self.noisy = noisy.astype(np.float32)
        if groups is not None:
            self.groups = groups[:max_groups] if max_groups is not None else groups
            if not self.groups:
                raise ValueError(f"empty triplet groups for split={split}")
            return

        df = labels[labels["split"].astype(str) == split].copy()
        group_cols = ["seg_id", "variant_id", "noise_kind", "sample_source"]
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(f"labels missing triplet grouping columns: {missing}")

        groups: list[tuple[int, int, int]] = []
        for _, g in df.groupby(group_cols, sort=False):
            by_class = {str(row.y_class): int(row.idx) for row in g.itertuples(index=False)}
            if all(name in by_class for name in ORDERED):
                groups.append(tuple(by_class[name] for name in ORDERED))
        if max_groups is not None:
            groups = groups[:max_groups]
        if not groups:
            raise ValueError(f"no complete good/medium/bad triplets found for split={split}")
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.groups[i]
        x = np.stack([self.noisy[j] for j in idx], axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(x[:, None, :]),
            "y": torch.tensor([0, 1, 2], dtype=torch.long),
        }


class ProjectionHeads(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 64) -> None:
        super().__init__()
        self.quality = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, emb_dim),
        )
        self.morphology = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, emb_dim),
        )


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer_e3_triplet_k1")
    out_dir = _path(
        params.get("out_dir"),
        artifact_dir / "pretrain" / "e8_contrastive_severity",
    )
    seed = int(params.get("seed", 0))
    epochs = int(params.get("epochs", 8))
    batch_groups = int(params.get("batch_groups", params.get("batch_size", 32)))
    lr = float(params.get("lr", 6e-5))
    weight_decay = float(params.get("weight_decay", 0.03))
    dropout = float(params.get("dropout", 0.05))
    temperature = float(params.get("temperature", 0.12))
    rank_margin = float(params.get("rank_margin", 0.35))
    lambda_ce = float(params.get("lambda_ce", 1.0))
    lambda_contrastive = float(params.get("lambda_contrastive", 0.5))
    lambda_rank = float(params.get("lambda_rank", 0.5))
    lambda_morph = float(params.get("lambda_morph", 0.2))
    num_workers = int(params.get("num_workers", 0))
    max_train_groups = _optional_int(params.get("max_train_groups"))
    max_val_groups = _optional_int(params.get("max_val_groups"))
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
        print(f"E8 pretrain outputs exist: {_rel(out_dir, root)}")
        return {"step": "e8_contrastive_severity_pretrain", "skipped": True, "outputs": [str(p) for p in outputs]}

    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    noisy, labels = load_arrays(artifact_dir)
    train_ds, val_ds, val_source = make_train_val_datasets(
        noisy,
        labels,
        max_train_groups=max_train_groups,
        max_val_groups=max_val_groups,
    )
    print(f"[train triplets] n={len(train_ds)}")
    print(f"[val triplets] n={len(val_ds)} source={val_source}")

    train_loader = DataLoader(train_ds, batch_size=batch_groups, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_groups, shuffle=False, num_workers=num_workers)

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

    projector = ProjectionHeads(model.cls_fc.in_features).to(device)
    if dry_run:
        batch = next(iter(train_loader))
        metrics = run_batch(model, projector, batch, device, temperature, rank_margin)
        print(f"[dry-run] E8 contrastive forward OK loss={metrics['loss']:.6f}")
        return {"step": "e8_contrastive_severity_pretrain", "skipped": True, "dry_run": True, "outputs": []}

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=max(lr * 0.05, 1e-6))

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        train_metrics = run_epoch(
            model, projector, train_loader, device, optimizer, temperature, rank_margin,
            lambda_ce, lambda_contrastive, lambda_rank, lambda_morph,
        )
        val_metrics = run_epoch(
            model, projector, val_loader, device, None, temperature, rank_margin,
            lambda_ce, lambda_contrastive, lambda_rank, lambda_morph,
        )
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"train L={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} rank={train_metrics['rank']:.4f} | "
            f"val L={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} rank={val_metrics['rank']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "projector_state": projector.state_dict(),
            "hyperparams": {
                "seed": seed,
                "epochs": epochs,
                "batch_groups": batch_groups,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "temperature": temperature,
                "rank_margin": rank_margin,
                "lambda_ce": lambda_ce,
                "lambda_contrastive": lambda_contrastive,
                "lambda_rank": lambda_rank,
                "lambda_morph": lambda_morph,
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
        "last_val_acc": history[-1]["val"]["acc"] if history else None,
        "artifact_dir": _rel(artifact_dir, root),
        "out_dir": _rel(out_dir, root),
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_val_loss={best_val:.6f} epoch={best_epoch}")
    print(f"saved: {_rel(out_best, root)}")
    return {"step": "e8_contrastive_severity_pretrain", "skipped": False, "outputs": [str(p) for p in outputs]}


def run_epoch(
    model: nn.Module,
    projector: ProjectionHeads,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    temperature: float,
    rank_margin: float,
    lambda_ce: float,
    lambda_contrastive: float,
    lambda_rank: float,
    lambda_morph: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    projector.train(train_mode)
    sums = {"loss": 0.0, "ce": 0.0, "contrastive": 0.0, "rank": 0.0, "morph": 0.0, "acc": 0.0}
    n = 0
    for batch in loader:
        with torch.set_grad_enabled(train_mode):
            parts = run_batch(model, projector, batch, device, temperature, rank_margin)
            loss = (
                lambda_ce * parts["ce"]
                + lambda_contrastive * parts["contrastive"]
                + lambda_rank * parts["rank"]
                + lambda_morph * parts["morph"]
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bsz = int(batch["x"].shape[0] * batch["x"].shape[1])
        n += bsz
        sums["loss"] += float(loss.detach().cpu()) * bsz
        for key in ("ce", "contrastive", "rank", "morph", "acc"):
            value = parts[key]
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            sums[key] += float(value) * bsz

    den = max(1, n)
    return {key: value / den for key, value in sums.items()}


def run_batch(
    model: MTLTransformerPTBXL,
    projector: ProjectionHeads,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    temperature: float,
    rank_margin: float,
) -> dict[str, torch.Tensor | float]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.long)
    groups, views = x.shape[0], x.shape[1]
    x_flat = x.reshape(groups * views, x.shape[2], x.shape[3])
    y_flat = y.reshape(groups * views)

    features = model.forward_features(x_flat)
    pooled = features["pooled"]
    logits = model.cls_fc(pooled)
    quality_emb = F.normalize(projector.quality(pooled), dim=1)
    morph_emb = F.normalize(projector.morphology(pooled), dim=1)

    ce = nn.CrossEntropyLoss()(logits, y_flat)
    contrastive = supervised_contrastive_loss(quality_emb, y_flat, temperature)
    rank = ordinal_rank_loss(logits.reshape(groups, views, 3), rank_margin)
    morph = morphology_alignment_loss(morph_emb.reshape(groups, views, -1))
    acc = (torch.argmax(logits, dim=1) == y_flat).float().mean()
    total = ce + contrastive + rank + morph
    return {"loss": total, "ce": ce, "contrastive": contrastive, "rank": rank, "morph": morph, "acc": acc}


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = emb @ emb.T / temperature
    logits_mask = ~torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)
    positive = (labels[:, None] == labels[None, :]) & logits_mask
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * logits_mask.to(sim.dtype)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = positive.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return torch.zeros((), device=emb.device, dtype=emb.dtype)
    mean_log_prob = (log_prob * positive.to(log_prob.dtype)).sum(dim=1) / pos_count.clamp_min(1)
    return -mean_log_prob[valid].mean()


def ordinal_rank_loss(group_logits: torch.Tensor, margin: float) -> torch.Tensor:
    probs = torch.softmax(group_logits, dim=2)
    severity = torch.tensor([0.0, 1.0, 2.0], device=group_logits.device, dtype=group_logits.dtype)
    scores = torch.sum(probs * severity, dim=2)
    good = scores[:, 0]
    medium = scores[:, 1]
    bad = scores[:, 2]
    return F.relu(margin - (medium - good)).mean() + F.relu(margin - (bad - medium)).mean()


def morphology_alignment_loss(group_emb: torch.Tensor) -> torch.Tensor:
    good = group_emb[:, 0, :]
    medium = group_emb[:, 1, :]
    bad = group_emb[:, 2, :]
    return (
        (1.0 - F.cosine_similarity(good, medium, dim=1)).mean()
        + (1.0 - F.cosine_similarity(medium, bad, dim=1)).mean()
        + (1.0 - F.cosine_similarity(good, bad, dim=1)).mean()
    ) / 3.0


def load_arrays(artifact_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    noisy_path = artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz"
    labels_path = artifact_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        labels_path = artifact_dir / "datasets" / "synth_10s_125hz_labels.csv"
    noisy = np.load(noisy_path)["X_noisy"].astype(np.float32)
    labels = pd.read_csv(labels_path)
    if "idx" not in labels.columns:
        labels["idx"] = np.arange(len(labels), dtype=int)
    if noisy.shape[0] != len(labels):
        raise ValueError("array/label row count mismatch")
    labels["y_int"] = labels["y_class"].astype(str).map(Y_TO_INT)
    if labels["y_int"].isna().any():
        raise ValueError("labels contain unknown y_class")
    return noisy, labels


def make_train_val_datasets(
    noisy: np.ndarray,
    labels: pd.DataFrame,
    max_train_groups: int | None,
    max_val_groups: int | None,
) -> tuple[TripletSeverityDataset, TripletSeverityDataset, str]:
    all_train = TripletSeverityDataset(noisy, labels, "train", max_groups=None)
    try:
        val = TripletSeverityDataset(noisy, labels, "val", max_groups=max_val_groups)
        train = TripletSeverityDataset(noisy, labels, "train", max_groups=max_train_groups)
        return train, val, "val"
    except ValueError:
        groups = list(all_train.groups)
        val_n = max_val_groups if max_val_groups is not None else max(1, min(1000, len(groups) // 10))
        val_n = max(1, min(val_n, len(groups) - 1))
        train_groups = groups[:-val_n]
        val_groups = groups[-val_n:]
        if max_train_groups is not None:
            train_groups = train_groups[:max_train_groups]
        train = TripletSeverityDataset(noisy, labels, "train", groups=train_groups)
        val = TripletSeverityDataset(noisy, labels, "train_holdout", groups=val_groups)
        return train, val, "train_holdout"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


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
    parser = argparse.ArgumentParser(description="E8 triplet contrastive severity pretraining.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_e3_triplet_k1")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_groups", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--rank_margin", type=float, default=0.35)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_contrastive", type=float, default=0.5)
    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--lambda_morph", type=float, default=0.2)
    parser.add_argument("--max_train_groups", type=int)
    parser.add_argument("--max_val_groups", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
