from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import (
    CLASS_TO_INT,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_INIT_CHECKPOINT,
    DEFAULT_OUT_ROOT,
    INT_TO_CLASS,
    NOISE_TO_INT,
)
from .data import E311Dataset, load_bundle, save_split_info
from .model import ResearchConfig, ResearchSQITransformer
from .recipes import RECIPES, get_recipe, list_recipes


class UncertaintyWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_den = nn.Parameter(torch.tensor(0.0))
        self.log_var_lvl = nn.Parameter(torch.tensor(0.0))

    def forward(self, cls: torch.Tensor, den: torch.Tensor, lvl: torch.Tensor) -> torch.Tensor:
        return (
            torch.exp(-self.log_var_cls) * cls
            + self.log_var_cls
            + torch.exp(-self.log_var_den) * den
            + self.log_var_den
            + torch.exp(-self.log_var_lvl) * lvl
            + self.log_var_lvl
        )


class SAM:
    def __init__(self, params: list[nn.Parameter], base_optimizer: torch.optim.Optimizer, rho: float) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.state: dict[nn.Parameter, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None]),
            p=2,
        )
        scale = self.rho / (grad_norm + 1e-12)
        self.state = {}
        for p in self.params:
            if p.grad is None:
                continue
            e_w = p.grad * scale.to(p.device)
            p.add_(e_w)
            self.state[p] = e_w
        self.base_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self) -> None:
        for p, e_w in self.state.items():
            p.sub_(e_w)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad(set_to_none=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def make_model(recipe: dict[str, Any], sample: dict[str, torch.Tensor]) -> ResearchSQITransformer:
    cfg = ResearchConfig(
        in_ch=int(sample["x_noisy"].shape[0]),
        dropout=float(recipe["dropout"]),
        use_positional_embedding=bool(recipe["use_positional_embedding"]),
        head_type=str(recipe["head_type"]),
        multiscale=str(recipe["multiscale"]),
        use_snr_head=bool(recipe["use_snr_head"]),
        use_ordinal_head=bool(recipe["use_ordinal_head"]),
    )
    return ResearchSQITransformer(cfg)


def partial_load(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"checkpoint": str(checkpoint_path), "loaded": 0, "skipped": [], "missing_file": True}
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    src = ckpt.get("model_state", ckpt)
    dst = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, Any]] = []
    for key, val in src.items():
        if key in dst and tuple(dst[key].shape) == tuple(val.shape):
            loadable[key] = val
        else:
            skipped.append({"key": key, "shape": list(val.shape) if hasattr(val, "shape") else None})
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded": len(loadable),
        "skipped": skipped[:200],
        "skipped_count": len(skipped),
        "missing_after_partial": list(missing)[:200],
        "missing_count": len(missing),
        "unexpected_after_partial": list(unexpected),
        "checkpoint_hyperparams": ckpt.get("hyperparams", {}),
    }


def ordinal_target(y: torch.Tensor) -> torch.Tensor:
    return torch.stack([(y > 0).float(), (y > 1).float()], dim=1)


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def class_loss(logits: torch.Tensor, y: torch.Tensor, recipe: dict[str, Any]) -> torch.Tensor:
    if recipe["loss_type"] == "focal":
        return focal_loss(logits, y, float(recipe["focal_gamma"]))
    return F.cross_entropy(logits, y, label_smoothing=float(recipe["label_smoothing"]))


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_pa = F.log_softmax(logits_a, dim=1)
    log_pb = F.log_softmax(logits_b, dim=1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    return 0.5 * (F.kl_div(log_pa, pb, reduction="batchmean") + F.kl_div(log_pb, pa, reduction="batchmean"))


def denoise_weights(batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor], recipe: dict[str, Any]) -> torch.Tensor:
    x_clean = batch["x_clean"]
    gate = str(recipe["denoise_gate"])
    if gate == "none":
        return torch.ones_like(x_clean)
    if gate == "qrs_mask":
        return 1.0 + 7.0 * batch["qrs_mask"].unsqueeze(1)
    if gate == "level_weight":
        return 1.0 + 4.0 * batch["level"].unsqueeze(1)
    if gate == "abs_topq_wavelet":
        detail = torch.zeros_like(x_clean)
        detail[:, :, 1:] = torch.abs(x_clean[:, :, 1:] - x_clean[:, :, :-1])
        q = torch.quantile(detail.flatten(1), 0.70, dim=1).view(-1, 1, 1)
        return 1.0 + 5.0 * (detail >= q).float()
    raise ValueError(f"unknown denoise_gate={gate!r}")


def component_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    recipe: dict[str, Any],
) -> dict[str, torch.Tensor]:
    y = batch["y"]
    cls = class_loss(out["logits"], y, recipe)
    den_w = denoise_weights(batch, out, recipe)
    den = (((out["denoise"] - batch["x_clean"]) ** 2) * den_w).sum() / den_w.sum().clamp_min(1.0)
    level_target = batch["level"].unsqueeze(1)
    valid = batch["valid_rr"].view(-1, 1, 1)
    level_raw = ((torch.sigmoid(out["level"]) - level_target) ** 2).mean(dim=(1, 2))
    lvl = (level_raw * batch["valid_rr"]).sum() / batch["valid_rr"].sum().clamp_min(1.0)
    snr = torch.tensor(0.0, device=y.device)
    if recipe["use_snr_head"] and "snr_hat" in out:
        snr = F.smooth_l1_loss(out["snr_hat"], batch["snr_norm"])
    ordinal = torch.tensor(0.0, device=y.device)
    if recipe["use_ordinal_head"] and "ordinal_logits" in out:
        ordinal = F.binary_cross_entropy_with_logits(out["ordinal_logits"], ordinal_target(y))
    local = torch.tensor(0.0, device=y.device)
    if recipe["head_type"] in {"local_quality_v2", "sqi_local"}:
        # Lightweight auxiliary: predicted level should be higher where synthetic
        # local quality mask says contamination exists. This is only in the
        # isolated experiment and is intentionally weak.
        target = batch["local_mask"].unsqueeze(1)
        local = F.binary_cross_entropy_with_logits(out["level"], target)
    return {"cls": cls, "denoise": den, "level": lvl, "snr": snr, "ordinal": ordinal, "local": local}


def total_loss(
    comps: dict[str, torch.Tensor],
    recipe: dict[str, Any],
    uw: UncertaintyWeights | None,
    epoch: int,
) -> torch.Tensor:
    cls = float(recipe["cls_weight"]) * comps["cls"]
    den = float(recipe["denoise_weight"]) * comps["denoise"]
    lvl = float(recipe["level_weight"]) * comps["level"]
    if uw is not None and bool(recipe["uncertainty"]) and epoch >= int(recipe.get("uncertainty_start_epoch", 6)):
        loss = uw(cls, den, lvl)
    else:
        loss = cls + den + lvl
    loss = loss + float(recipe["snr_weight"]) * comps["snr"]
    loss = loss + float(recipe["ordinal_weight"]) * comps["ordinal"]
    loss = loss + 0.05 * comps["local"]
    return loss


def shared_grad_norms(model: ResearchSQITransformer, comps: dict[str, torch.Tensor], recipe: dict[str, Any]) -> dict[str, float]:
    params = model.shared_parameters()
    out: dict[str, float] = {}
    keys = {
        "grad_cls": comps["cls"],
        "grad_denoise": float(recipe["denoise_weight"]) * comps["denoise"],
        "grad_level": float(recipe["level_weight"]) * comps["level"],
    }
    for name, loss in keys.items():
        if not loss.requires_grad or float(loss.detach().abs().item()) == 0.0:
            out[name] = 0.0
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        sq = [g.detach().pow(2).sum() for g in grads if g is not None]
        out[name] = float(torch.sqrt(torch.stack(sq).sum()).item()) if sq else 0.0
    return out


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_one_epoch(
    *,
    model: ResearchSQITransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    sam: SAM | None,
    recipe: dict[str, Any],
    device: torch.device,
    epoch: int,
    uw: UncertaintyWeights | None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(train_mode)
    if uw is not None:
        uw.train(train_mode)
    totals = {k: 0.0 for k in ["loss", "cls", "denoise", "level", "snr", "ordinal", "local"]}
    n_seen = 0
    correct = 0
    grad_norm: dict[str, float] | None = None

    for bi, batch_cpu in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = move_batch(batch_cpu, device)
        bsz = int(batch["y"].shape[0])

        def forward_loss() -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            out = model(batch["x_noisy"])
            comps = component_losses(out, batch, recipe)
            loss = total_loss(comps, recipe, uw, epoch)
            if train_mode and float(recipe["rdrop_alpha"]) > 0.0:
                out2 = model(batch["x_noisy"])
                comps2 = component_losses(out2, batch, recipe)
                loss2 = total_loss(comps2, recipe, uw, epoch)
                loss = 0.5 * (loss + loss2) + float(recipe["rdrop_alpha"]) * symmetric_kl(out["logits"], out2["logits"])
            return loss, comps, out

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss, comps, out = forward_loss()
            if bi == 0:
                grad_norm = shared_grad_norms(model, comps, recipe)
            loss.backward()
            if sam is not None:
                sam.first_step()
                loss2, _, _ = forward_loss()
                loss2.backward()
                sam.second_step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
        else:
            with torch.no_grad():
                loss, comps, out = forward_loss()

        pred = out["logits"].argmax(dim=1)
        correct += int((pred == batch["y"]).sum().item())
        n_seen += bsz
        totals["loss"] += float(loss.detach().item()) * bsz
        for key in comps:
            totals[key] += float(comps[key].detach().item()) * bsz

    denom = max(1, n_seen)
    metrics = {key: val / denom for key, val in totals.items()}
    metrics["acc"] = correct / denom
    metrics["n"] = n_seen
    if grad_norm is not None:
        metrics.update(grad_norm)
    return metrics


@torch.no_grad()
def evaluate(
    *,
    model: ResearchSQITransformer,
    loader: DataLoader,
    recipe: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    n_all: list[np.ndarray] = []
    den_mse: dict[int, list[float]] = {0: [], 1: [], 2: []}
    lvl_mse: dict[int, list[float]] = {0: [], 1: [], 2: []}

    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        out = model(batch["x_noisy"])
        pred = out["logits"].argmax(dim=1)
        y_all.append(batch["y"].detach().cpu().numpy())
        p_all.append(pred.detach().cpu().numpy())
        n_all.append(batch["noise_type"].detach().cpu().numpy())
        den_each = ((out["denoise"] - batch["x_clean"]) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        lvl_each = ((torch.sigmoid(out["level"]) - batch["level"].unsqueeze(1)) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        for yy, dm, lm in zip(batch["y"].detach().cpu().numpy(), den_each, lvl_each, strict=False):
            den_mse[int(yy)].append(float(dm))
            lvl_mse[int(yy)].append(float(lm))

    y = np.concatenate(y_all) if y_all else np.empty((0,), dtype=np.int64)
    pred = np.concatenate(p_all) if p_all else np.empty((0,), dtype=np.int64)
    noise = np.concatenate(n_all) if n_all else np.empty((0,), dtype=np.int64)
    cm = np.zeros((3, 3), dtype=np.int64)
    for yy, pp in zip(y, pred, strict=False):
        cm[int(yy), int(pp)] += 1
    recall = {}
    for i, name in INT_TO_CLASS.items():
        recall[name] = float(cm[i, i] / max(1, cm[i].sum()))

    inv_noise = {v: k for k, v in NOISE_TO_INT.items()}
    per_noise: dict[str, Any] = {}
    for ni in sorted(set(noise.tolist())):
        m = noise == ni
        c = np.zeros((3, 3), dtype=np.int64)
        for yy, pp in zip(y[m], pred[m], strict=False):
            c[int(yy), int(pp)] += 1
        per_noise[inv_noise.get(int(ni), str(ni))] = {
            "n": int(m.sum()),
            "acc": float((y[m] == pred[m]).mean()) if m.any() else 0.0,
            "recall": {INT_TO_CLASS[i]: float(c[i, i] / max(1, c[i].sum())) for i in range(3)},
        }

    return {
        "n": int(y.shape[0]),
        "acc": float((y == pred).mean()) if y.size else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class_recall": recall,
        "per_noise_kind": per_noise,
        "denoise_mse_by_class": {INT_TO_CLASS[k]: float(np.mean(v)) if v else 0.0 for k, v in den_mse.items()},
        "level_mse_by_class": {INT_TO_CLASS[k]: float(np.mean(v)) if v else 0.0 for k, v in lvl_mse.items()},
    }


def make_loaders(bundle: Any, batch_size: int, seed: int) -> dict[str, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        "train": DataLoader(bundle.datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, generator=g),
        "val": DataLoader(bundle.datasets["val"], batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True),
        "test": DataLoader(bundle.datasets["test"], batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True),
    }


def train_recipe(args: argparse.Namespace, recipe: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.out_root) / recipe["group"] / recipe["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(recipe["seed"]))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    bundle = load_bundle(
        Path(args.artifact_dir),
        out_root=Path(args.out_root),
        level_target_mode=str(recipe["level_target_mode"]),
        input_mode=str(recipe["input_mode"]),
        force_level_cache=bool(args.force_level_cache),
    )
    save_split_info(out_dir / "split_info.json", bundle.split_info)
    loaders = make_loaders(bundle, int(recipe["batch_size"]), int(recipe["seed"]))
    sample = bundle.datasets["train"][0]
    model = make_model(recipe, sample).to(device)
    warm = partial_load(model, Path(args.init_checkpoint))

    uw = UncertaintyWeights().to(device) if bool(recipe["uncertainty"]) else None
    params = list(model.parameters()) + (list(uw.parameters()) if uw is not None else [])
    opt = torch.optim.AdamW(params, lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    sam = SAM(list(model.parameters()), opt, float(recipe["sam_rho"])) if float(recipe["sam_rho"]) > 0.0 else None

    config = {"recipe": recipe, "artifact_dir": str(args.artifact_dir), "init_checkpoint": str(args.init_checkpoint), "device": str(device)}
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "warmstart_report.json", warm)

    if args.dry_run:
        metrics = run_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=opt,
            sam=sam,
            recipe=recipe,
            device=device,
            epoch=1,
            uw=uw,
            max_batches=1,
        )
        val = run_one_epoch(
            model=model,
            loader=loaders["val"],
            optimizer=None,
            sam=None,
            recipe=recipe,
            device=device,
            epoch=1,
            uw=uw,
            max_batches=1,
        )
        result = {"dry_train": metrics, "dry_val": val, "warmstart": warm, "split_info": bundle.split_info}
        write_json(out_dir / "dry_run_report.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    best_val = -math.inf
    history: list[dict[str, Any]] = []
    best_path = out_dir / "ckpt_best_val_acc.pt"
    max_train_batches = args.max_train_batches if args.max_train_batches and args.max_train_batches > 0 else None
    for epoch in range(1, int(recipe["epochs"]) + 1):
        tr = run_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=opt,
            sam=sam,
            recipe=recipe,
            device=device,
            epoch=epoch,
            uw=uw,
            max_batches=max_train_batches,
        )
        va = run_one_epoch(
            model=model,
            loader=loaders["val"],
            optimizer=None,
            sam=None,
            recipe=recipe,
            device=device,
            epoch=epoch,
            uw=uw,
        )
        row = {"epoch": epoch, "train": tr, "val": va}
        if uw is not None:
            row["uncertainty"] = {k: float(v.detach().cpu().item()) for k, v in uw.state_dict().items()}
        history.append(row)
        write_json(out_dir / "train_log.json", history)
        if va["acc"] > best_val:
            best_val = float(va["acc"])
            torch.save({"model_state": model.state_dict(), "recipe": recipe, "epoch": epoch, "val": va}, best_path)
        print(
            f"[{recipe['name']}] epoch {epoch:02d} train_acc={tr['acc']:.4f} "
            f"val_acc={va['acc']:.4f} val_loss={va['loss']:.4f}",
            flush=True,
        )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model=model, loader=loaders["test"], recipe=recipe, device=device)
    report = {
        "recipe": recipe,
        "best_epoch": int(ckpt["epoch"]),
        "best_val": ckpt["val"],
        "test": test,
        "warmstart": warm,
        "split_info": bundle.split_info,
    }
    write_json(out_dir / "test_report.json", report)
    print(json.dumps({"name": recipe["name"], "test": test, "best_epoch": int(ckpt["epoch"])}, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated E3.11 SQI research trainer")
    parser.add_argument("--group", choices=sorted(RECIPES))
    parser.add_argument("--task_id", type=int)
    parser.add_argument("--recipe")
    parser.add_argument("--artifact_dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--init_checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--device")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_level_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print(json.dumps({g: [r["name"] for r in rows] for g, rows in RECIPES.items()}, indent=2, sort_keys=True))
        return
    if not args.group:
        raise SystemExit("--group is required unless --list is used")
    recipe = get_recipe(group=args.group, task_id=args.task_id, name=args.recipe)
    train_recipe(args, recipe)


if __name__ == "__main__":
    main()
