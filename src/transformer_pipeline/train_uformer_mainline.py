"""Train the E3.11f Uformer full-token detached mainline.

Default recipe:

1. Denoiser pretrain from random init:
   uformer1d_hier / morph_soft_guard / noise_scale=0.9 / epochs=10.
2. Detached classifier head:
   load the best denoiser, read full-token features, detach CE from denoiser,
   continue a small denoise loss, train the SQI MLP head for 8 epochs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.transformer_pipeline.e311_uformer_data import E311DenoiseDataset, audit_to_json, split_audit
from src.transformer_pipeline.e311_uformer_eval import evaluate_and_export, write_json
from src.transformer_pipeline.e311_uformer_losses import morph_loss
from src.transformer_pipeline.models.uformer1d import Uformer1DConfig, UformerDenoiseSQIModel


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid" / "data" / "med6p25_badgap7_badcm0p75"
DEFAULT_OUTPUT = ROOT / "outputs" / "mainline" / "e311_uformer_full_tokens_detach_seed0"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def make_loaders(source: Path, batch_size: int, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_ds = E311DenoiseDataset(source, "train")
    val_ds = E311DenoiseDataset(source, "val")
    test_ds = E311DenoiseDataset(source, "test")
    kwargs = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **kwargs),
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )


def build_model(args: argparse.Namespace, with_head: bool, device: torch.device) -> UformerDenoiseSQIModel:
    cfg = Uformer1DConfig(
        noise_scale=float(args.noise_scale),
        dropout=float(args.dropout),
        feature_set=str(args.feature_set),
        detach_encoder_features=bool(args.detach_encoder_features),
        head_hidden_dim=int(args.hidden_dim),
        denoiser_width=float(args.denoiser_width),
    )
    model = UformerDenoiseSQIModel(cfg).to(device)
    if with_head:
        model.attach_head(model.infer_feature_dim(device=device))
        model.head = model.head.to(device)
    return model


def load_denoiser(model: UformerDenoiseSQIModel, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("denoiser_state", ckpt.get("model_state", ckpt))
    missing, unexpected = model.denoiser.load_state_dict(state, strict=False)
    return {"checkpoint": str(checkpoint), "missing": len(missing), "unexpected": len(unexpected)}


def classification_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    soft_y: torch.Tensor,
    cls_sample_weight: torch.Tensor,
    class_weight: torch.Tensor | None,
    variant: str,
) -> torch.Tensor:
    if variant == "hard_ce":
        return F.cross_entropy(logits, y, weight=class_weight)
    if variant == "sample_weighted_ce":
        per_sample = F.cross_entropy(logits, y, weight=class_weight, reduction="none")
    else:
        logp = F.log_softmax(logits, dim=1)
        soft = soft_y.to(device=logits.device, dtype=torch.float32)
        if class_weight is not None:
            soft = soft * class_weight.view(1, -1)
        per_sample = -(soft * logp).sum(dim=1)
    if variant in {"sample_weighted_ce", "soft_sample_weighted_ce"}:
        w = cls_sample_weight.to(device=logits.device, dtype=torch.float32).clamp_min(0.0)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-6)
    return per_sample.mean()


def train_epoch(
    model: UformerDenoiseSQIModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    class_weight: torch.Tensor | None,
    lambda_den: float,
    lambda_cls: float,
    max_train_batches: int = 0,
) -> dict[str, float]:
    model.train()
    sums = {"total": 0.0, "den": 0.0, "cls": 0.0}
    n = 0
    for batch_idx, batch in enumerate(loader):
        if max_train_batches > 0 and batch_idx >= max_train_batches:
            break
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        clean = batch["clean"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.long)
        soft_y = batch["soft_y"].to(device=device, dtype=torch.float32)
        point_weight = batch["point_weight"].to(device=device, dtype=torch.float32)
        sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
        cls_sample_weight = batch["cls_sample_weight"].to(device=device, dtype=torch.float32)
        out = model(noisy)
        l_den, _ = morph_loss(out["denoise"], out["noise_hat"], noisy, clean, y, point_weight, sample_weight, str(args.loss_variant))
        l_cls = torch.zeros((), device=device)
        if out["logits"] is not None and float(lambda_cls) > 0.0:
            l_cls = classification_loss(out["logits"], y, soft_y, cls_sample_weight, class_weight, str(args.cls_loss_variant))
        loss = float(lambda_den) * l_den + float(lambda_cls) * l_cls
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        bs = int(noisy.shape[0])
        n += bs
        sums["total"] += float(loss.detach().cpu().item()) * bs
        sums["den"] += float(l_den.detach().cpu().item()) * bs
        sums["cls"] += float(l_cls.detach().cpu().item()) * bs
    denom = max(1, n)
    return {k: v / denom for k, v in sums.items()}


def save_train_curve(log: list[dict[str, Any]], out_png: Path) -> None:
    if not log:
        return
    epochs = [int(row["epoch"]) for row in log]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(epochs, [row.get("val_acc") or 0.0 for row in log], marker="o")
    axes[0].set_title("val acc")
    axes[1].plot(epochs, [row.get("val_denoise_score") or 0.0 for row in log], marker="o")
    axes[1].set_title("val denoise score")
    axes[2].plot(epochs, [row.get("val_snr_gain") or 0.0 for row in log], marker="o")
    axes[2].set_title("val SNR gain")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_stage(args: argparse.Namespace, stage: str, out_dir: Path, init_checkpoint: Path | None = None) -> dict[str, Any]:
    seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = Path(args.source_artifact_dir)
    batch_size = int(args.batch_size_stage1 if stage == "denoise_pretrain" else args.batch_size_stage2)
    train_loader, val_loader, test_loader, split_counts = make_loaders(source, batch_size, device)
    with_head = stage == "classifier_head"
    model = build_model(args, with_head=with_head, device=device)
    load_report: dict[str, Any] = {"loaded": False}
    if init_checkpoint is not None:
        load_report = {"loaded": True, **load_denoiser(model, init_checkpoint, device)}
    if with_head:
        print(f"classifier feature_dim={model.infer_feature_dim(device=device)} detach={args.detach_encoder_features}", flush=True)
    params: list[dict[str, Any]] = []
    den_params = [p for p in model.denoiser.parameters() if p.requires_grad]
    if den_params:
        params.append({"params": den_params, "lr": float(args.lr_stage1 if stage == "denoise_pretrain" else args.lr_denoiser)})
    if model.head is not None:
        params.append({"params": model.head.parameters(), "lr": float(args.lr_head)})
    optimizer = torch.optim.AdamW(params, weight_decay=float(args.weight_decay))
    class_weight = None
    if model.head is not None:
        class_weight = torch.tensor(parse_float_list(str(args.class_weight)), dtype=torch.float32, device=device)
    log: list[dict[str, Any]] = []
    best_key: tuple[float, ...] = (-1e9,)
    best_state: dict[str, Any] | None = None
    epochs = int(args.epochs_stage1 if stage == "denoise_pretrain" else args.epochs_stage2)
    for epoch in range(epochs):
        lambda_den = float(args.lambda_den_stage1 if stage == "denoise_pretrain" else args.lambda_den_stage2)
        lambda_cls = 0.0 if stage == "denoise_pretrain" else float(args.lambda_cls)
        train_stats = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args,
            class_weight,
            lambda_den,
            lambda_cls,
            int(args.max_train_batches),
        )
        val = evaluate_and_export(model, val_loader, device, None)
        report = val["report"]
        metrics = val["metrics"]["overall"]
        acc = report.get("acc")
        denoise_score = float(metrics["denoise_score"])
        key = (denoise_score, float(metrics["snr_improve_db_mean"])) if not with_head else (float(acc), denoise_score)
        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_stats.items()},
            "val_acc": acc,
            "val_recall_good_medium_bad": report.get("recall_good_medium_bad"),
            "val_denoise_score": denoise_score,
            "val_snr_gain": metrics["snr_improve_db_mean"],
            "val_mse_ratio": metrics["mse_ratio_pred_over_noisy"],
        }
        log.append(row)
        write_json(out_dir / "train_log.json", log)
        print(
            f"{stage} epoch={epoch + 1}/{epochs} loss={row['train_total']:.4f} "
            f"val_acc={acc} den={denoise_score:.3f} snr+={row['val_snr_gain']:.3f}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_state = {
                "epoch": epoch + 1,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "denoiser_state": {k: v.detach().cpu().clone() for k, v in model.denoiser.state_dict().items()},
                "key": key,
            }
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    result = evaluate_and_export(model, test_loader, device, out_dir)
    save_train_curve(log, out_dir / "visuals" / "train_curves.png")
    checkpoint_name = "ckpt_best_stage1_denoiser.pt" if stage == "denoise_pretrain" else "ckpt_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "denoiser_state": model.denoiser.state_dict(),
            "config": model.cfg.__dict__,
            "args": vars(args),
            "stage": stage,
            "best_state_meta": None if best_state is None else {"epoch": best_state["epoch"], "key": best_state["key"]},
        },
        out_dir / checkpoint_name,
    )
    summary = {
        "status": "completed",
        "stage": stage,
        "architecture": "uformer1d_hier_residual_denoiser_detached_full_tokens_head",
        "split_counts": split_counts,
        "checkpoint_load": load_report,
        "lambda_den": float(args.lambda_den_stage1 if stage == "denoise_pretrain" else args.lambda_den_stage2),
        "lambda_cls": 0.0 if stage == "denoise_pretrain" else float(args.lambda_cls),
        "best_epoch": None if best_state is None else best_state["epoch"],
        "test_report": result["report"],
        "denoise_metrics": result["metrics"]["overall"],
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _, split_counts = make_loaders(Path(args.source_artifact_dir), int(args.batch_size_stage1), device)
    model = build_model(args, with_head=True, device=device)
    batch = next(iter(train_loader))
    noisy = batch["noisy"].to(device=device, dtype=torch.float32)
    out = model(noisy)
    payload = {
        "device": str(device),
        "split_counts": split_counts,
        "noisy_shape": list(noisy.shape),
        "noise_hat_shape": list(out["noise_hat"].shape),
        "denoise_shape": list(out["denoise"].shape),
        "feature_shape": list(out["features"].shape),
        "logits_shape": list(out["logits"].shape),
    }
    write_json(Path(args.output_dir) / "dry_run.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = split_audit(Path(args.source_artifact_dir))
    write_json(out_dir / "split_audit.json", audit_to_json(audit))
    stage1_dir = out_dir / "stage1_denoiser_pretrain"
    stage1 = run_stage(args, "denoise_pretrain", stage1_dir)
    stage1_ckpt = stage1_dir / "ckpt_best_stage1_denoiser.pt"
    stage2 = run_stage(args, "classifier_head", out_dir, stage1_ckpt)
    summary = {"stage1": stage1, "stage2": stage2, "source_artifact_dir": str(args.source_artifact_dir)}
    write_json(out_dir / "mainline_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train E3.11f Uformer detached full-token SQI mainline.")
    parser.add_argument("--stage", choices=("all", "denoise_pretrain", "classifier_head", "split_audit", "dry_run"), default="all")
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--feature_set", choices=("full_tokens", "bottleneck_only", "summary_only", "residual_summary_only"), default="full_tokens")
    parser.add_argument("--detach_encoder_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss_variant", choices=("morph_soft_guard", "morph_soft_identity"), default="morph_soft_guard")
    parser.add_argument("--noise_scale", type=float, default=0.9)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--denoiser_width", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs_stage1", type=int, default=10)
    parser.add_argument("--epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--lr_stage1", type=float, default=2e-4)
    parser.add_argument("--lr_denoiser", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=7e-4)
    parser.add_argument("--lambda_den_stage1", type=float, default=1.0)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--cls_loss_variant", choices=("hard_ce", "sample_weighted_ce", "soft_ce", "soft_sample_weighted_ce"), default="hard_ce")
    parser.add_argument("--class_weight", default="1,1.40,1.70")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    if args.stage == "split_audit":
        write_json(out_dir / "split_audit.json", audit_to_json(split_audit(Path(args.source_artifact_dir))))
        print(json.dumps(audit_to_json(split_audit(Path(args.source_artifact_dir))), ensure_ascii=False, indent=2), flush=True)
    elif args.stage == "dry_run":
        dry_run(args)
    elif args.stage == "denoise_pretrain":
        run_stage(args, "denoise_pretrain", out_dir)
    elif args.stage == "classifier_head":
        if not args.init_checkpoint:
            raise SystemExit("--init_checkpoint is required for classifier_head")
        run_stage(args, "classifier_head", out_dir, Path(args.init_checkpoint))
    else:
        run_all(args)


if __name__ == "__main__":
    main()
