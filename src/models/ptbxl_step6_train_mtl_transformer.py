from __future__ import annotations

# --------- Imports ---------
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import pywt
except Exception:
    pywt = None

try:
    from src.utils.paths import project_root
    from src.models.mtl_transformer import MTLTransformerConfig, MTLTransformerPTBXL
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root
    from src.models.mtl_transformer import MTLTransformerConfig, MTLTransformerPTBXL


# --------- Fixed Hyperparams (edit here) ---------
SEED = 0
BATCH_SIZE = 16
LR = 3e-5
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
PIN_MEMORY = False

EPOCHS = 12
E_CLS = 12
E_DENOISE = 0
E_LEVEL = 0
E_UNCERT = 0

ALPHA = 2.0
WAVELET = "db6"
WAVELET_LEVEL = 4
WAVELET_Q = 0.65


# --------- Paths ---------
ROOT = project_root()
IN_NOISY = ROOT / "artifact1" / "datasets" / "synth_10s_125hz_noisy.npz"
IN_CLEAN = ROOT / "artifact1" / "datasets" / "synth_10s_125hz_clean.npz"
IN_LEVEL = ROOT / "artifact1" / "datasets" / "synth_10s_125hz_noise_level.npz"
IN_LABELS = ROOT / "artifact1" / "datasets" / "synth_10s_125hz_labels_with_level.csv"

OUT_DIR = ROOT / "artifact1" / "models" / "mtl_transformer_seed0_step6"
OUT_LAST = OUT_DIR / "ckpt_last.pt"
OUT_BEST = OUT_DIR / "ckpt_best_val.pt"
OUT_LOG = OUT_DIR / "train_log.json"
OUT_TEST = OUT_DIR / "test_report.json"


# --------- Utils ---------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_y_class(v: str) -> int:
    s = str(v).strip().lower()
    if s == "good":
        return 0
    if s == "medium":
        return 1
    if s == "bad":
        return 2
    raise ValueError(f"Unknown y_class: {v}")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    den = torch.sum(mask)
    if float(den.detach().cpu().item()) <= 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return torch.sum(x * mask) / den


def phase_name(epoch: int) -> str:
    b0 = E_CLS
    c0 = E_CLS + E_DENOISE
    d0 = E_CLS + E_DENOISE + E_LEVEL
    e0 = max(d0, EPOCHS - E_UNCERT) if E_UNCERT > 0 else EPOCHS + 1
    if epoch < b0:
        return "A_cls_only"
    if epoch < c0:
        return "B_add_denoise"
    if epoch < d0:
        return "C_add_level"
    if epoch >= e0:
        return "E_uncertainty_joint"
    return "D_joint"


def active_losses(phase: str) -> tuple[bool, bool, bool, bool]:
    if phase == "A_cls_only":
        return True, False, False, False
    if phase == "B_add_denoise":
        return True, True, False, False
    if phase == "C_add_level":
        return True, True, True, False
    if phase == "D_joint":
        return True, True, True, False
    return True, True, True, True


def safe_state_dict(model: nn.Module) -> dict[str, Any]:
    m = model.module if isinstance(model, nn.DataParallel) else model
    return m.state_dict()


# --------- Dataset / Dataloader ---------
class PTBXLMTLDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray, p: np.ndarray, valid_rr: np.ndarray, y: np.ndarray):
        self.noisy = noisy.astype(np.float32)
        self.clean = clean.astype(np.float32)
        self.p = p.astype(np.float32)
        self.valid_rr = (valid_rr.astype(np.float32) > 0.5).astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x_noisy": torch.from_numpy(self.noisy[i][None, :]),   # (1,1250)
            "x_clean": torch.from_numpy(self.clean[i][None, :]),   # (1,1250)
            "p": torch.from_numpy(self.p[i]),                      # (1250,)
            "valid_rr": torch.tensor(self.valid_rr[i], dtype=torch.float32),
            "y": torch.tensor(self.y[i], dtype=torch.long),
        }


def build_split_arrays() -> tuple[dict[str, PTBXLMTLDataset], dict[str, dict[str, Any]]]:
    X_noisy = np.load(IN_NOISY)["X_noisy"].astype(np.float32)
    X_clean = np.load(IN_CLEAN)["X_clean"].astype(np.float32)
    z = np.load(IN_LEVEL)
    P = z["P"].astype(np.float32)
    valid_rr = z["valid_rr"].astype(np.uint8)
    df = pd.read_csv(IN_LABELS)

    required = {"idx", "split", "y_class", "noise_kind", "snr_db", "seg_id", "ecg_id"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in labels: {sorted(miss)}")

    n = len(df)
    if not (X_noisy.shape[0] == X_clean.shape[0] == P.shape[0] == valid_rr.shape[0] == n):
        raise ValueError("Array/CSV row count mismatch")

    idx = df["idx"].to_numpy(dtype=np.int64)
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError("idx out of range")
    y_int = np.array([map_y_class(v) for v in df["y_class"].tolist()], dtype=np.int64)

    datasets: dict[str, PTBXLMTLDataset] = {}
    split_info: dict[str, dict[str, Any]] = {}
    for sp in ["train", "val", "test"]:
        m = (df["split"].astype(str).to_numpy() == sp)
        sp_idx = idx[m]
        order = np.argsort(sp_idx)
        sp_idx = sp_idx[order]
        sp_y = y_int[m][order]

        datasets[sp] = PTBXLMTLDataset(
            noisy=X_noisy[sp_idx],
            clean=X_clean[sp_idx],
            p=P[sp_idx],
            valid_rr=valid_rr[sp_idx],
            y=sp_y,
        )
        vc = pd.Series(sp_y).value_counts().sort_index().to_dict()
        split_info[sp] = {
            "n": int(len(sp_idx)),
            "y_dist_int": {int(k): int(v) for k, v in vc.items()},
            "y_dist_str": {
                "good(0)": int(vc.get(0, 0)),
                "medium(1)": int(vc.get(1, 0)),
                "bad(2)": int(vc.get(2, 0)),
            },
        }
    return datasets, split_info


# --------- Model ---------
class UncertaintyWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_sigma_cls = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.log_sigma_denoise = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.log_sigma_level = nn.Parameter(torch.zeros((), dtype=torch.float32))


def build_model(device: torch.device) -> tuple[nn.Module, UncertaintyWeights]:
    cfg = MTLTransformerConfig()
    model = MTLTransformerPTBXL(cfg).to(device=device, dtype=torch.float32)
    uw = UncertaintyWeights().to(device=device, dtype=torch.float32)
    return model, uw


# --------- Loss Functions ---------
_warned_wavelet = False


def wavelet_gate_batch(clean_batch: torch.Tensor, y_int: torch.Tensor) -> torch.Tensor:
    # [ASSUMPTION] wavelet mask only for good samples; medium no extra weight; bad excluded by sample mask.
    global _warned_wavelet
    b, _, t = clean_batch.shape
    gate = torch.zeros((b, t), device=clean_batch.device, dtype=torch.float32)
    if pywt is None:
        if not _warned_wavelet:
            print("WARNING: PyWavelets not installed, wavelet weighting disabled.")
            _warned_wavelet = True
        return gate

    clean_np = clean_batch.detach().cpu().numpy()
    y_np = y_int.detach().cpu().numpy()
    for i in range(b):
        if int(y_np[i]) != 0:
            continue
        x = clean_np[i, 0].astype(np.float64, copy=False)
        try:
            coeffs = pywt.wavedec(x, wavelet=WAVELET, level=WAVELET_LEVEL)
            detail = np.zeros_like(coeffs[0])
            rec_list = [np.zeros_like(c) for c in coeffs]
            # [ASSUMPTION] use level-4 detail (D4) as morphology cue.
            rec_list[-WAVELET_LEVEL] = coeffs[-WAVELET_LEVEL]
            detail = pywt.waverec(rec_list, wavelet=WAVELET)[:t]
            a = np.abs(detail)
            thr = float(np.quantile(a, WAVELET_Q))
            g = (a > thr).astype(np.float32)
            gate[i] = torch.from_numpy(g).to(device=clean_batch.device, dtype=torch.float32)
        except Exception:
            if not _warned_wavelet:
                print("WARNING: wavelet decomposition failed on some samples, fallback gate=0.")
                _warned_wavelet = True
    return gate


def compute_losses(
    y_denoise: torch.Tensor,
    y_level: torch.Tensor,
    logits: torch.Tensor,
    clean: torch.Tensor,
    p_target: torch.Tensor,
    valid_rr: torch.Tensor,
    y_int: torch.Tensor,
    use_cls: bool,
    use_den: bool,
    use_lvl: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.zeros((), device=clean.device, dtype=torch.float32)
    l_cls = z
    l_denoise = z
    l_level = z

    if use_cls:
        ce = nn.CrossEntropyLoss()
        l_cls = ce(logits, y_int)

    if use_den:
        pred_d = y_denoise.squeeze(1)
        tgt_d = clean.squeeze(1)
        mse_t = (pred_d - tgt_d) ** 2

        gate = wavelet_gate_batch(clean, y_int)
        w = torch.ones_like(mse_t, dtype=torch.float32, device=mse_t.device)
        w = w + (ALPHA - 1.0) * gate
        per_sample_d = torch.mean(mse_t * w, dim=1)

        mask_d = (y_int != 2).float()  # bad not used in denoise
        l_denoise = masked_mean(per_sample_d, mask_d)

    if use_lvl:
        p_pred = torch.sigmoid(y_level.squeeze(1))
        per_sample_l = torch.mean((p_pred - p_target) ** 2, dim=1)
        l_level = masked_mean(per_sample_l, valid_rr.float())
    return l_cls, l_denoise, l_level


def combine_losses(
    l_cls: torch.Tensor,
    l_denoise: torch.Tensor,
    l_level: torch.Tensor,
    phase: str,
    uw: UncertaintyWeights,
) -> torch.Tensor:
    use_cls, use_den, use_lvl, use_uncert = active_losses(phase)
    l1 = l_cls if use_cls else torch.zeros_like(l_cls)
    l2 = l_denoise if use_den else torch.zeros_like(l_cls)
    l3 = l_level if use_lvl else torch.zeros_like(l_cls)
    if not use_uncert:
        return l1 + l2 + l3
    # [ASSUMPTION] Kendall homoscedastic uncertainty weighting.
    s1 = uw.log_sigma_cls
    s2 = uw.log_sigma_denoise
    s3 = uw.log_sigma_level
    return torch.exp(-s1) * l1 + s1 + torch.exp(-s2) * l2 + s2 + torch.exp(-s3) * l3 + s3


# --------- Train Loop ---------
def run_epoch(
    model: nn.Module,
    uw: UncertaintyWeights,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    phase: str,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    uw.train(train_mode)

    n = 0
    correct = 0
    sum_total = 0.0
    sum_cls = 0.0
    sum_den = 0.0
    sum_lvl = 0.0

    iterator = tqdm(
        loader,
        desc=f"{phase} {'train' if train_mode else 'val'}",
        leave=False,
    )
    for batch in iterator:
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        x_clean = batch["x_clean"].to(device=device, dtype=torch.float32)
        p_t = batch["p"].to(device=device, dtype=torch.float32)
        valid_rr = batch["valid_rr"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.long)

        with torch.set_grad_enabled(train_mode):
            out = model(x_noisy)
            # [ASSUMPTION] adapt if model returns other container/order.
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                y_denoise, y_level, logits = out[0], out[1], out[2]
            else:
                raise RuntimeError("Model output must provide y_denoise, y_level, logits")

            use_cls, use_den, use_lvl, _ = active_losses(phase)
            l_cls, l_den, l_lvl = compute_losses(
                y_denoise, y_level, logits, x_clean, p_t, valid_rr, y, use_cls, use_den, use_lvl
            )
            l_total = combine_losses(l_cls, l_den, l_lvl, phase, uw)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                l_total.backward()
                optimizer.step()

        if train_mode:
            iterator.set_postfix({
                "L": f"{float(l_total.detach().cpu()):.3f}",
                "C": f"{float(l_cls.detach().cpu()):.3f}",
                "D": f"{float(l_den.detach().cpu()):.3f}",
                "Lv": f"{float(l_lvl.detach().cpu()):.3f}",
            })

        bsz = int(x_noisy.shape[0])
        n += bsz
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().detach().cpu().item())
        sum_total += float(l_total.detach().cpu().item()) * bsz
        sum_cls += float(l_cls.detach().cpu().item()) * bsz
        sum_den += float(l_den.detach().cpu().item()) * bsz
        sum_lvl += float(l_lvl.detach().cpu().item()) * bsz

    if n == 0:
        return {"total": 0.0, "cls": 0.0, "denoise": 0.0, "level": 0.0, "acc": 0.0}
    return {
        "total": sum_total / n,
        "cls": sum_cls / n,
        "denoise": sum_den / n,
        "level": sum_lvl / n,
        "acc": float(correct) / float(n),
    }


# --------- Eval ---------
@torch.no_grad()
def eval_test_report(model: nn.Module, uw: UncertaintyWeights, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    uw.eval()
    phase = "D_joint"
    metrics = run_epoch(model, uw, loader, device, optimizer=None, phase=phase)

    cm = np.zeros((3, 3), dtype=np.int64)
    for batch in loader:
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.long)
        out = model(x_noisy)
        y_denoise, y_level, logits = out[0], out[1], out[2]
        pred = torch.argmax(logits, dim=1)
        yt = y.detach().cpu().numpy()
        yp = pred.detach().cpu().numpy()
        for a, b in zip(yt.tolist(), yp.tolist()):
            cm[int(a), int(b)] += 1

    return {
        "acc": metrics["acc"],
        "loss_total": metrics["total"],
        "loss_cls": metrics["cls"],
        "loss_denoise": metrics["denoise"],
        "loss_level": metrics["level"],
        "confusion_matrix_3x3": cm.tolist(),
    }


# --------- Save Artifacts ---------
def main() -> None:
    seed_all(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    datasets, split_info = build_split_arrays()
    for sp in ["train", "val", "test"]:
        print(f"[{sp}] n={split_info[sp]['n']} y_dist={split_info[sp]['y_dist_str']}")

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model, uw = build_model(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(uw.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    hp = {
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,
        "E_CLS": E_CLS,
        "E_DENOISE": E_DENOISE,
        "E_LEVEL": E_LEVEL,
        "E_UNCERT": E_UNCERT,
        "ALPHA": ALPHA,
        "WAVELET": WAVELET,
        "WAVELET_LEVEL": WAVELET_LEVEL,
        "WAVELET_Q": WAVELET_Q,
    }

    history: list[dict[str, Any]] = []
    best_val = float("inf")

    for epoch in range(EPOCHS):
        phase = phase_name(epoch)
        tr = run_epoch(model, uw, train_loader, device, optimizer=optimizer, phase=phase)
        va = run_epoch(model, uw, val_loader, device, optimizer=None, phase=phase)

        row = {
            "epoch": epoch,
            "phase": phase,
            "train": tr,
            "val": va,
            "log_sigma": {
                "cls": float(uw.log_sigma_cls.detach().cpu().item()),
                "denoise": float(uw.log_sigma_denoise.detach().cpu().item()),
                "level": float(uw.log_sigma_level.detach().cpu().item()),
            },
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] phase={phase} | "
            f"train acc={tr['acc']:.4f} val acc={va['acc']:.4f} | "
            f"train L={tr['total']:.4f} (cls={tr['cls']:.4f}, den={tr['denoise']:.4f}, lvl={tr['level']:.4f}) | "
            f"val L={va['total']:.4f}"
        )
        if phase == "E_uncertainty_joint":
            print(
                f"log_sigma cls/den/level="
                f"{float(uw.log_sigma_cls.detach().cpu().item()):.4f}/"
                f"{float(uw.log_sigma_denoise.detach().cpu().item()):.4f}/"
                f"{float(uw.log_sigma_level.detach().cpu().item()):.4f}"
            )

        ckpt = {
            "epoch": epoch,
            "model_state": safe_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "hyperparams": hp,
        }
        torch.save(ckpt, OUT_LAST)

        # [ASSUMPTION] best checkpoint is only comparable in joint phases.
        if phase in ("D_joint", "E_uncertainty_joint"):
            if va["total"] < best_val:
                best_val = va["total"]
                torch.save(ckpt, OUT_BEST)

    with OUT_LOG.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if not OUT_BEST.exists():
        shutil.copyfile(OUT_LAST, OUT_BEST)

    test_report = eval_test_report(model, uw, test_loader, device)
    with OUT_TEST.open("w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    print(f"saved: {OUT_LAST}")
    print(f"saved: {OUT_BEST}")
    print(f"saved: {OUT_LOG}")
    print(f"saved: {OUT_TEST}")
    print("NEXT INPUT: artifact1/models/mtl_transformer_seed0_step6/ckpt_best_val.pt")


if __name__ == "__main__":
    main()
