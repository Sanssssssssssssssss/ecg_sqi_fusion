from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).absolute()
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else next(p for p in THIS_FILE.parents if (p / "src").exists())
EXP_ROOT = ROOT / "outputs" / "experiment" / "e311_sqi_denoise_classifier_grid"
FULL_JOINT = EXP_ROOT / "full_joint_train.py"
MORPH_EXP = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid"
MORPH_TRAIN = MORPH_EXP / "morph_train.py"
DEFAULT_SOURCE = MORPH_EXP / "data" / "med6p25_badgap7_badcm0p75"
DEFAULT_DENOISER = MORPH_EXP / "baseline" / "baseline_best_med6p25_badgap7_scale0p955" / "checkpoints" / "ckpt_best_denoise.pt"
DEFAULT_CLASSIFIER = (
    ROOT
    / "outputs"
    / "experiment"
    / "e311_denoise_strong_grid"
    / "artifact"
    / "models"
    / "promote_strong_noise_clean_multiscale_deriv_multiscale_patch_ld5p0_ep8_ld12p5_ep14"
    / "ckpt_best_val.pt"
)

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


def cls_report(y: np.ndarray, logits: np.ndarray, prefix: str = "") -> dict[str, Any]:
    pred = np.argmax(logits, axis=1)
    mat = np.zeros((3, 3), dtype=int)
    rec: list[float] = []
    prec: list[float] = []
    f1: list[float] = []
    for a, b in zip(y, pred):
        mat[int(a), int(b)] += 1
    for c in range(3):
        tp = float(np.sum((y == c) & (pred == c)))
        fn = float(np.sum((y == c) & (pred != c)))
        fp = float(np.sum((y != c) & (pred == c)))
        r = tp / max(1.0, tp + fn)
        p = tp / max(1.0, tp + fp)
        rec.append(r)
        prec.append(p)
        f1.append(2.0 * p * r / max(1e-12, p + r))
    return {
        f"{prefix}acc": float(np.mean(pred == y)),
        f"{prefix}confusion_3x3": mat.tolist(),
        f"{prefix}recall_good_medium_bad": rec,
        f"{prefix}precision_good_medium_bad": prec,
        f"{prefix}f1_good_medium_bad": f1,
    }


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


def residual_unet_forward_with_features(denoiser: nn.Module, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return denoise, noise_hat, and encoder feature vector for the residual_unet denoiser."""
    net = getattr(denoiser, "net", denoiser)
    if not all(hasattr(net, name) for name in ("e1", "d1", "e2", "d2", "e3", "d3", "mid", "u3", "u2", "u1", "out")):
        denoise, noise_hat = denoiser(noisy)
        residual = torch.abs(noisy - denoise)
        feats = torch.cat([tensor_summary(noisy), tensor_summary(denoise), tensor_summary(residual)], dim=1)
        return denoise, noise_hat, feats

    s1 = net.e1(noisy)
    s2 = net.e2(F.gelu(net.d1(s1)))
    s3 = net.e3(F.gelu(net.d2(s2)))
    b = net.mid(F.gelu(net.d3(s3)))
    x = net._upsample(b, s3)
    x = net.u3(torch.cat([x, s3], dim=1))
    x = net._upsample(x, s2)
    x = net.u2(torch.cat([x, s2], dim=1))
    x = net._upsample(x, s1)
    x = net.u1(torch.cat([x, s1], dim=1))
    noise_hat = net.out(x)
    denoise = noisy - noise_hat
    residual = torch.abs(noisy - denoise)
    feats = torch.cat(
        [
            pooled_channels(s1),
            pooled_channels(s2),
            pooled_channels(s3),
            pooled_channels(b),
            tensor_summary(noisy),
            tensor_summary(denoise),
            tensor_summary(residual),
        ],
        dim=1,
    )
    return denoise, noise_hat, feats


class EncoderClassifierHead(nn.Module):
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


class EncoderDeltaHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, gated: bool):
        super().__init__()
        self.gated = gated
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.gate = nn.Linear(hidden_dim, 1) if gated else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.net(x)
        delta = self.delta(h)
        gate = torch.sigmoid(self.gate(h)) if self.gate is not None else None
        if gate is not None:
            delta = delta * gate
        return delta, gate


class EncoderAuditModel(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        encoder_head: EncoderClassifierHead | None,
        transformer_classifier: nn.Module | None,
        encoder_delta: EncoderDeltaHead | None,
        variant: str,
        noise_scale: float,
        delta_scale: float,
        detach_encoder_features: bool,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.encoder_head = encoder_head
        self.transformer_classifier = transformer_classifier
        self.encoder_delta = encoder_delta
        self.variant = variant
        self.noise_scale = float(noise_scale)
        self.delta_scale = float(delta_scale)
        self.detach_encoder_features = bool(detach_encoder_features)

    def forward(self, noisy: torch.Tensor) -> dict[str, torch.Tensor | None]:
        denoise_raw, noise_hat, enc_feat = residual_unet_forward_with_features(self.denoiser, noisy)
        denoise = noisy - self.noise_scale * noise_hat
        feat = enc_feat.detach() if self.detach_encoder_features else enc_feat
        gate = None
        delta = None
        snr_pred = None
        local_pred = None
        if self.variant == "encoder_head_only":
            assert self.encoder_head is not None
            logits = self.encoder_head(feat)
            base_logits = logits
        elif self.variant == "dual_branch":
            assert self.transformer_classifier is not None and self.encoder_delta is not None
            out = self.transformer_classifier(denoise)
            base_logits = out[2]
            snr_pred = out[3] if len(out) > 3 else None
            local_pred = out[4] if len(out) > 4 else None
            delta, gate = self.encoder_delta(feat)
            logits = base_logits + self.delta_scale * delta
        else:
            raise ValueError(f"unknown variant: {self.variant}")
        return {
            "denoise": denoise,
            "noise_hat": noise_hat,
            "base_logits": base_logits,
            "logits": logits,
            "encoder_features": enc_feat,
            "delta": delta,
            "gate": gate,
            "snr_pred": snr_pred,
            "local_pred": local_pred,
        }


def severity_from_snr(snr_db: torch.Tensor) -> torch.Tensor:
    return torch.clamp((13.0 - snr_db.float()) / 14.5, 0.0, 1.0)


@torch.no_grad()
def collect_outputs(model: EncoderAuditModel, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in [
        "noisy",
        "clean",
        "denoise_pred",
        "y",
        "base_logits",
        "logits",
        "prob_denoise",
        "prob_base",
        "idx",
        "ecg_id",
        "snr_db",
        "original_snr_db",
        "local_mask",
        "critical_mask",
        "qrs_mask",
        "tst_mask",
        "encoder_delta",
        "encoder_gate",
    ]}
    noise_kind_all: list[np.ndarray] = []
    placement_all: list[np.ndarray] = []
    for batch in loader:
        noisy = batch["noisy"].to(device=device, dtype=torch.float32)
        out = model(noisy)
        logits = out["logits"]
        base_logits = out["base_logits"]
        denoise = out["denoise"]
        chunks["noisy"].append(noisy.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["clean"].append(batch["clean"].squeeze(1).numpy().astype(np.float32))
        chunks["denoise_pred"].append(denoise.squeeze(1).cpu().numpy().astype(np.float32))
        chunks["y"].append(batch["y"].numpy().astype(np.int64))
        chunks["base_logits"].append(base_logits.cpu().numpy().astype(np.float32))
        chunks["logits"].append(logits.cpu().numpy().astype(np.float32))
        chunks["prob_base"].append(torch.softmax(base_logits, dim=1).cpu().numpy().astype(np.float32))
        chunks["prob_denoise"].append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
        bs = int(noisy.shape[0])
        if out["delta"] is not None:
            chunks["encoder_delta"].append(out["delta"].cpu().numpy().astype(np.float32))
        else:
            chunks["encoder_delta"].append(np.zeros((bs, 3), dtype=np.float32))
        if out["gate"] is not None:
            chunks["encoder_gate"].append(out["gate"].cpu().numpy().astype(np.float32))
        else:
            chunks["encoder_gate"].append(np.full((bs, 1), np.nan, dtype=np.float32))
        for key in ["idx", "ecg_id"]:
            chunks[key].append(batch[key].numpy().astype(np.int64))
        for key in ["snr_db", "original_snr_db", "local_mask", "critical_mask", "qrs_mask", "tst_mask"]:
            chunks[key].append(batch[key].numpy().astype(np.float32))
        noise_kind_all.append(np.asarray(batch["noise_kind"], dtype=str))
        placement_all.append(np.asarray(batch["placement"], dtype=str))
    arrays = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    arrays["noise_kind"] = np.concatenate(noise_kind_all, axis=0)
    arrays["placement"] = np.concatenate(placement_all, axis=0)
    arrays["y_pred_denoise"] = np.argmax(arrays["logits"], axis=1).astype(np.int64)
    arrays["y_pred_base"] = np.argmax(arrays["base_logits"], axis=1).astype(np.int64)
    arrays["y_pred_noisy"] = arrays["y_pred_base"]
    return arrays


def evaluate(model: EncoderAuditModel, loader: DataLoader, device: torch.device, morph: Any, out_dir: Path | None = None) -> dict[str, Any]:
    arrays = collect_outputs(model, loader, device)
    y = arrays["y"].astype(np.int64)
    report = {
        **cls_report(y, arrays["base_logits"], "base_"),
        **cls_report(y, arrays["logits"], "final_"),
    }
    report["corrected_by_encoder_delta"] = int(np.sum((arrays["y_pred_base"] != y) & (arrays["y_pred_denoise"] == y)))
    report["harmed_by_encoder_delta"] = int(np.sum((arrays["y_pred_base"] == y) & (arrays["y_pred_denoise"] != y)))
    report["encoder_delta_abs_mean"] = float(np.mean(np.abs(arrays["encoder_delta"])))
    report["encoder_delta_abs_p95"] = float(np.percentile(np.abs(arrays["encoder_delta"]), 95))
    if np.isfinite(arrays["encoder_gate"]).any():
        report["encoder_gate_mean"] = float(np.nanmean(arrays["encoder_gate"]))
        report["encoder_gate_p95"] = float(np.nanpercentile(arrays["encoder_gate"], 95))
    masks = {
        "local": arrays["local_mask"],
        "critical": arrays["critical_mask"],
        "qrs": arrays["qrs_mask"],
        "tst": arrays["tst_mask"],
    }
    metrics = morph.build_metrics(arrays["clean"], arrays["noisy"], arrays["denoise_pred"], y, arrays["noise_kind"], arrays["placement"], masks)
    if out_dir is not None:
        (out_dir / "denoise_eval").mkdir(parents=True, exist_ok=True)
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "denoise_eval" / "test_denoise_outputs.npz",
            noisy=arrays["noisy"],
            clean=arrays["clean"],
            denoise_pred=arrays["denoise_pred"],
            y=y,
            y_pred=arrays["y_pred_denoise"],
            y_pred_base=arrays["y_pred_base"],
            prob=arrays["prob_denoise"],
            prob_base=arrays["prob_base"],
            encoder_delta=arrays["encoder_delta"],
            encoder_gate=arrays["encoder_gate"],
            idx=arrays["idx"],
            ecg_id=arrays["ecg_id"],
            snr_db=arrays["snr_db"],
            original_snr_db=arrays["original_snr_db"],
            noise_kind=arrays["noise_kind"],
            placement=arrays["placement"],
        )
        write_json(out_dir / "test_report.json", report)
        write_json(out_dir / "denoise_eval" / "denoise_metrics.json", metrics)
        morph.export_gallery(arrays, out_dir / "debug" / "denoise_compact_test.png", hard_only=False, title="encoder-head audit: compact SNR-marked gallery")
        morph.export_gallery(arrays, out_dir / "debug" / "denoise_hard_test.png", hard_only=True, title="encoder-head audit: hard SNR-marked gallery")
    return {"report": report, "metrics": metrics}


def make_model(args: argparse.Namespace, morph: Any, full_joint: Any, train_loader: DataLoader, device: torch.device) -> EncoderAuditModel:
    denoiser = full_joint.load_denoiser(morph, Path(args.init_denoiser_checkpoint), device)
    transformer = None
    encoder_head = None
    encoder_delta = None
    with torch.no_grad():
        sample = next(iter(train_loader))
        noisy = sample["noisy"].to(device=device, dtype=torch.float32)
        _, _, feat = residual_unet_forward_with_features(denoiser, noisy)
        feat_dim = int(feat.shape[1])
    if args.variant == "encoder_head_only":
        encoder_head = EncoderClassifierHead(feat_dim, int(args.hidden_dim), float(args.dropout)).to(device)
    elif args.variant == "dual_branch":
        transformer = full_joint.build_trainable_classifier(morph, Path(args.init_classifier_checkpoint), device, dropout=float(args.transformer_dropout))
        encoder_delta = EncoderDeltaHead(feat_dim, int(args.hidden_dim), float(args.dropout), bool(args.gated_delta)).to(device)
    else:
        raise ValueError(f"unknown variant: {args.variant}")
    print(f"variant={args.variant} feat_dim={feat_dim} detach_encoder_features={bool(args.detach_encoder_features)}", flush=True)
    return EncoderAuditModel(
        denoiser=denoiser,
        encoder_head=encoder_head,
        transformer_classifier=transformer,
        encoder_delta=encoder_delta,
        variant=args.variant,
        noise_scale=float(args.noise_scale),
        delta_scale=float(args.delta_scale),
        detach_encoder_features=bool(args.detach_encoder_features),
    ).to(device)


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    morph = import_module(MORPH_TRAIN, "e311_morph_train_encoder_head_audit")
    full_joint = import_module(FULL_JOINT, "e311_full_joint_for_encoder_head_audit")

    source = Path(args.source_artifact_dir)
    train_ds = morph.ECGDenoiseDataset(source, "train")
    val_ds = morph.ECGDenoiseDataset(source, "val")
    test_ds = morph.ECGDenoiseDataset(source, "test")
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"splits train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = make_model(args, morph, full_joint, train_loader, device)
    class_weight = torch.tensor([float(x) for x in str(args.class_weight).split(",")], dtype=torch.float32, device=device)
    params: list[dict[str, Any]] = [{"params": model.denoiser.parameters(), "lr": float(args.lr_denoiser)}]
    if model.encoder_head is not None:
        params.append({"params": model.encoder_head.parameters(), "lr": float(args.lr_head)})
    if model.transformer_classifier is not None:
        params.append({"params": model.transformer_classifier.parameters(), "lr": float(args.lr_classifier)})
    if model.encoder_delta is not None:
        params.append({"params": model.encoder_delta.parameters(), "lr": float(args.lr_head)})
    optimizer = torch.optim.AdamW(params, weight_decay=float(args.weight_decay))

    log: list[dict[str, Any]] = []
    best_key: tuple[float, ...] = (-1e9,)
    best_state: dict[str, Any] | None = None
    for epoch in range(int(args.epochs)):
        model.train()
        sums = {"total": 0.0, "cls": 0.0, "den": 0.0, "snr": 0.0, "local": 0.0}
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
            l_cls = F.cross_entropy(out["logits"], y, weight=class_weight)
            l_den, _dbg = morph.morph_loss(out["denoise"], out["noise_hat"], noisy, clean, y, point_weight, sample_weight, str(args.loss_variant))
            l_snr = torch.zeros((), device=device)
            l_local = torch.zeros((), device=device)
            if out["snr_pred"] is not None and float(args.lambda_snr) > 0.0:
                severity = severity_from_snr(batch["snr_db"].to(device=device, dtype=torch.float32))
                l_snr = F.smooth_l1_loss(torch.sigmoid(out["snr_pred"]), severity)
            if out["local_pred"] is not None and float(args.lambda_local) > 0.0:
                local_mask = batch["local_mask"].to(device=device, dtype=torch.float32).unsqueeze(1)
                l_local = F.binary_cross_entropy_with_logits(out["local_pred"], local_mask)
            loss = (
                float(args.lambda_cls) * l_cls
                + float(args.lambda_den) * l_den
                + float(args.lambda_snr) * l_snr
                + float(args.lambda_local) * l_local
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bs = int(noisy.shape[0])
            n += bs
            sums["total"] += float(loss.detach().cpu().item()) * bs
            sums["cls"] += float(l_cls.detach().cpu().item()) * bs
            sums["den"] += float(l_den.detach().cpu().item()) * bs
            sums["snr"] += float(l_snr.detach().cpu().item()) * bs
            sums["local"] += float(l_local.detach().cpu().item()) * bs
        val_eval = evaluate(model, val_loader, device, morph, None)
        val_report = val_eval["report"]
        val_metrics = val_eval["metrics"]["overall"]
        acc = float(val_report["final_acc"])
        good, medium, bad = [float(x) for x in val_report["final_recall_good_medium_bad"]]
        denoise_score = float(val_metrics["denoise_score"])
        key = (
            acc - 0.02 * max(0.0, 2.75 - denoise_score) - 0.5 * max(0.0, 0.94 - good) - 0.5 * max(0.0, 0.94 - medium),
            acc,
            bad,
            medium,
            denoise_score,
        )
        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": v / max(1, n) for k, v in sums.items()},
            "val_base_acc": val_report["base_acc"],
            "val_final_acc": acc,
            "val_good_recall": good,
            "val_medium_recall": medium,
            "val_bad_recall": bad,
            "val_denoise_score": denoise_score,
            "val_mse_ratio": val_metrics["mse_ratio_pred_over_noisy"],
            "val_snr_gain": val_metrics["snr_improve_db_mean"],
        }
        log.append(row)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={row['train_total']:.4f} "
            f"val_acc={acc:.5f} good={good:.5f} med={medium:.5f} bad={bad:.5f} "
            f"den={denoise_score:.3f} snr+={row['val_snr_gain']:.3f}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_state = {
                "epoch": epoch + 1,
                "key": key,
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    test_eval = evaluate(model, test_loader, device, morph, out_dir)
    write_json(out_dir / "train_log.json", log)
    torch.save({"model_state": model.state_dict(), "args": vars(args), "best_state_meta": best_state}, out_dir / "ckpt_best_encoder_audit.pt")
    summary = {
        "status": "completed",
        "variant": args.variant,
        "args": vars(args),
        "best_epoch": None if best_state is None else best_state["epoch"],
        "test_report": test_eval["report"],
        "denoise_metrics": test_eval["metrics"]["overall"],
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False)[:2500], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment-only denoiser encoder classifier audit.")
    parser.add_argument("--variant", choices=("encoder_head_only", "dual_branch"), required=True)
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_denoiser_checkpoint", default=str(DEFAULT_DENOISER))
    parser.add_argument("--init_classifier_checkpoint", default=str(DEFAULT_CLASSIFIER))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--transformer_dropout", type=float, default=0.10)
    parser.add_argument("--gated_delta", action="store_true")
    parser.add_argument("--detach_encoder_features", action="store_true")
    parser.add_argument("--loss_variant", default="morph_soft_guard")
    parser.add_argument("--noise_scale", type=float, default=0.955)
    parser.add_argument("--delta_scale", type=float, default=0.04)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--lambda_den", type=float, default=10.0)
    parser.add_argument("--lambda_snr", type=float, default=0.03)
    parser.add_argument("--lambda_local", type=float, default=0.005)
    parser.add_argument("--class_weight", default="1,1.40,1.70")
    parser.add_argument("--lr_denoiser", type=float, default=4e-5)
    parser.add_argument("--lr_classifier", type=float, default=4e-6)
    parser.add_argument("--lr_head", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
