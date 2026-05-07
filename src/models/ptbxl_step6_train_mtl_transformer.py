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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
BATCH_SIZE = 32
WEIGHT_DECAY = 0.03
NUM_WORKERS = 0
PIN_MEMORY = False

# ---- LR scheduler ----
LR = 6e-5
LR_ETA_MIN = 4e-6  # cosine lowest lr (after epochs decay)

EPOCHS = 20
E_CLS = 0
E_DENOISE = 15
E_LEVEL = 5
E_UNCERT = 0

# ---- denoise on bad curriculum ---- (focus on this affect the system)
BAD_DEN_W_MAX = 0.2     # final weight of bad samples denoise loss（0.2~0.5）
BAD_DEN_W_WARMUP_EPOCHS = 10  # epochs take linearly increase the bad weights from 0 to the maximum

ALPHA = 8.0
LAMBDA_CLS = 10.0
LAMBDA_DEN = 120.0
LAMBDA_LVL = 1.0
WAVELET = "db6"
WAVELET_LEVEL = 4
WAVELET_Q = 0.65


# --------- Early Stop ---------
EARLYSTOP_PATIENCE = 4
EARLYSTOP_MIN_DELTA = 5e-3
EARLYSTOP_START_EPOCH = E_CLS + 3   # start from denoise
EARLYSTOP_PHASES = {"B_add_denoise", "D_joint", "E_uncertainty_joint"}


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

# global curriculum weight (updated each epoch in main)
CUR_BAD_DEN_W = 0.0


def wavelet_gate_batch(clean_batch: torch.Tensor, y_int: torch.Tensor) -> torch.Tensor:
    """
    Gate:
      - only for "good" samples (y==0) [ASSUMPTION consistent with paper wording]
      - compute a(t) = reconstructed wavelet detail at level WAVELET_LEVEL
      - gate(t) = 1{ a(t) > 0 }
    Returns: gate shape (B,T) float32 in {0,1}
    """
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
        # only "good"
        if int(y_np[i]) != 0:
            continue

        x = clean_np[i, 0].astype(np.float64, copy=False)
        try:
            # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD1]
            coeffs = pywt.wavedec(x, wavelet=WAVELET, level=WAVELET_LEVEL, mode="symmetric")

            # reconstruct ONLY cD_L back to time domain as a(t)
            rec = [np.zeros_like(c) for c in coeffs]
            rec[1] = coeffs[1]  # keep cD_L
            a_t = pywt.waverec(rec, wavelet=WAVELET, mode="symmetric")[:t]

            g = (a_t > 0).astype(np.float32)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    z = torch.zeros((), device=clean.device, dtype=torch.float32)
    l_cls = z
    l_denoise = z
    l_level = z
    dbg: dict[str, float] = {}

    if use_cls:
        ce = nn.CrossEntropyLoss()
        l_cls = ce(logits, y_int)

    if use_den:
        pred_d = y_denoise.squeeze(1)  # (B,T)
        tgt_d = clean.squeeze(1)       # (B,T)
        l_base_t = (pred_d - tgt_d) ** 2  # L_base(t)
        per_sample_base = torch.mean(l_base_t, dim=1)

        # paper gate: g(t)=1{a(t)>0} for good only
        gate = wavelet_gate_batch(clean, y_int)  # (B,T) in {0,1}

        # Eq.(6): L(t)= alpha*L_base(t) if a(t)>0 else L_base(t)
        w = 1.0 + (ALPHA - 1.0) * gate
        per_sample_d = torch.mean(l_base_t * w, dim=1)  # mean over t

        # Eq.(4): y~ = 0 if Bad else 1   (mask on samples)
        # mask_d = (y_int != 2).float()  # bad excluded (Important)
        # l_denoise = masked_mean(per_sample_d, mask_d)
        # --- include bad with small (curriculum) weight ---
        # weight per sample: good/medium=1.0, bad=CUR_BAD_DEN_W
        mask_d = torch.ones_like(y_int, dtype=torch.float32)
        if CUR_BAD_DEN_W <= 0.0:
            mask_d = (y_int != 2).float()
        else:
            bad_w = torch.tensor(CUR_BAD_DEN_W, device=mask_d.device, dtype=mask_d.dtype)
            mask_d = mask_d * torch.where(y_int == 2, bad_w, torch.tensor(1.0, device=mask_d.device, dtype=mask_d.dtype))

        l_denoise = masked_mean(per_sample_d, mask_d)

        # ---- debug stats (batch-aggregated) ----
        with torch.no_grad():
            # fraction of samples participating in denoise (not bad)
            dbg["den_sample_frac"] = float(mask_d.mean().detach().cpu().item())

            # among GOOD samples only, what fraction of time points are gated
            good_mask = (y_int == 0).float()  # (B,)
            denom_good = float(good_mask.sum().detach().cpu().item())
            if denom_good > 0:
                gated_points = (gate * good_mask[:, None]).sum()
                total_points = (good_mask.sum() * gate.shape[1])
                dbg["wavelet_gate_frac_good"] = float((gated_points / (total_points + 1e-12)).detach().cpu().item())
            else:
                dbg["wavelet_gate_frac_good"] = 0.0

            # overall gate fraction (mostly for sanity; will be small because non-good are zeros)
            dbg["wavelet_gate_frac_all"] = float(gate.mean().detach().cpu().item())

            # mean weight actually applied
            dbg["wavelet_w_mean"] = float(w.mean().detach().cpu().item())

            # denoise losses (raw)
            dbg["den_l_base_mean"] = float(masked_mean(per_sample_base, mask_d).detach().cpu().item())
            dbg["den_l_weighted_mean"] = float(masked_mean(per_sample_d, mask_d).detach().cpu().item())

    if use_lvl:
        p_pred = torch.sigmoid(y_level.squeeze(1))
        per_sample_l = torch.mean((p_pred - p_target) ** 2, dim=1)
        l_level = masked_mean(per_sample_l, valid_rr.float())

    return l_cls, l_denoise, l_level, dbg


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
        return LAMBDA_CLS * l1 + LAMBDA_DEN * l2 + LAMBDA_LVL * l3
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
    dbg_sum: dict[str, float] = {}
    dbg_n = 0

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
            l_cls, l_den, l_lvl, dbg = compute_losses(
                y_denoise, y_level, logits, x_clean, p_t, valid_rr, y, use_cls, use_den, use_lvl
            )
            l_total = combine_losses(l_cls, l_den, l_lvl, phase, uw)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                l_total.backward()
                optimizer.step()

        if train_mode:
            post = {
                "L": f"{float(l_total.detach().cpu()):.3f}",
                "C": f"{float(l_cls.detach().cpu()):.3f}",
                "D": f"{float(l_den.detach().cpu()):.3f}",
                "Lv": f"{float(l_lvl.detach().cpu()):.3f}",
            }
            if use_den and dbg:
                post.update({
                    "g%": f"{dbg.get('wavelet_gate_frac_good', 0.0):.2f}",
                    "w": f"{dbg.get('wavelet_w_mean', 1.0):.2f}",
                })
            iterator.set_postfix(post)

        bsz = int(x_noisy.shape[0])
        n += bsz
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().detach().cpu().item())
        sum_total += float(l_total.detach().cpu().item()) * bsz
        sum_cls += float(l_cls.detach().cpu().item()) * bsz
        sum_den += float(l_den.detach().cpu().item()) * bsz
        sum_lvl += float(l_lvl.detach().cpu().item()) * bsz
        if dbg:
            dbg_n += 1
            for k, v in dbg.items():
                dbg_sum[k] = dbg_sum.get(k, 0.0) + float(v)

    if n == 0:
        out0 = {"total": 0.0, "cls": 0.0, "denoise": 0.0, "level": 0.0, "acc": 0.0}
        if dbg_n > 0:
            for k in dbg_sum:
                dbg_sum[k] /= float(dbg_n)
        out0.update(dbg_sum)
        return out0

    if dbg_n > 0:
        for k in dbg_sum:
            dbg_sum[k] /= float(dbg_n)

    out = {
        "total": sum_total / n,
        "cls": sum_cls / n,
        "denoise": sum_den / n,
        "level": sum_lvl / n,
        "acc": float(correct) / float(n),
    }
    out.update(dbg_sum)
    return out


# --------- Eval ---------

@torch.no_grad()
def eval_split_details(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()

    cm = np.zeros((3, 3), dtype=np.int64)  # rows=true, cols=pred
    n_true = np.zeros(3, dtype=np.int64)
    n_correct = np.zeros(3, dtype=np.int64)

    # prob stats: for each true class, collect p(true), and also p(pred)
    p_true_sum = np.zeros(3, dtype=np.float64)
    p_pred_sum = np.zeros(3, dtype=np.float64)

    for batch in loader:
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.long)

        out = model(x_noisy)
        y_denoise, y_level, logits = out[0], out[1], out[2]

        probs = torch.softmax(logits, dim=1)          # (B,3)
        pred = torch.argmax(probs, dim=1)             # (B,)

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        for i in range(len(y_np)):
            yt = int(y_np[i])
            yp = int(pred_np[i])
            cm[yt, yp] += 1
            n_true[yt] += 1
            if yt == yp:
                n_correct[yt] += 1
            p_true_sum[yt] += float(probs_np[i, yt])
            p_pred_sum[yt] += float(probs_np[i, yp])

    # per-class acc + mean probs
    per_class = {}
    name = {0: "good", 1: "medium", 2: "bad"}
    for c in range(3):
        denom = max(1, int(n_true[c]))
        row = cm[c, :].astype(np.float64)
        row_sum = float(max(1, int(row.sum())))
        per_class[name[c]] = {
            "n": int(n_true[c]),
            "acc": float(n_correct[c]) / float(denom),
            "mean_p_trueclass": float(p_true_sum[c]) / float(denom),
            "mean_p_predclass": float(p_pred_sum[c]) / float(denom),
            # helpful: where this class goes when it's wrong
            "confusion_row": row.tolist(),
            "confusion_row_frac": (row / row_sum).tolist(),
        }

    #特别关心：medium 被分到哪里
    medium_row = cm[1, :]
    medium_to = {
        "to_good": int(medium_row[0]),
        "to_medium": int(medium_row[1]),
        "to_bad": int(medium_row[2]),
    }

    overall_acc = float(np.trace(cm)) / float(max(1, int(cm.sum())))
    return {
        "overall_acc": overall_acc,
        "confusion_matrix_3x3": cm.tolist(),
        "per_class": per_class,
        "medium_breakdown": medium_to,
    }


@torch.no_grad()
def eval_test_report(model: nn.Module, uw: UncertaintyWeights, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    uw.eval()
    phase = phase_name(EPOCHS - 1)  # follow training final phase (A/B/C/D/E)
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

    den_by_class = denoise_metrics_by_class(model, loader, device)

    return {
        "acc": metrics["acc"],
        "loss_total": metrics["total"],
        "loss_cls": metrics["cls"],
        "loss_denoise": metrics["denoise"],
        "loss_level": metrics["level"],
        "confusion_matrix_3x3": cm.tolist(),
        "denoise_metrics_by_class": den_by_class,
    }


@torch.no_grad()
def denoise_metrics_by_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Compute denoise objective metrics by TRUE class:
      - MSE(noisy, clean), MSE(pred, clean)
      - SNR(clean/noise) for noisy and pred, and improvement (pred - noisy)
    """
    model.eval()

    sums = {0: {}, 1: {}, 2: {}}
    counts = {0: 0, 1: 0, 2: 0}

    def acc(c: int, key: str, val: float) -> None:
        sums[c][key] = sums[c].get(key, 0.0) + float(val)

    for batch in loader:
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)  # (B,1,T)
        x_clean = batch["x_clean"].to(device=device, dtype=torch.float32)  # (B,1,T)
        y = batch["y"].to(device=device, dtype=torch.long)                 # (B,)

        out = model(x_noisy)
        y_denoise = out[0]  # (B,1,T)

        xn = x_noisy.squeeze(1)   # (B,T)
        xc = x_clean.squeeze(1)   # (B,T)
        xp = y_denoise.squeeze(1) # (B,T)

        sig_pow = torch.mean(xc * xc, dim=1)
        mse_noisy = torch.mean((xn - xc) ** 2, dim=1)
        mse_pred = torch.mean((xp - xc) ** 2, dim=1)

        snr_noisy = 10.0 * torch.log10((sig_pow + eps) / (mse_noisy + eps))
        snr_pred = 10.0 * torch.log10((sig_pow + eps) / (mse_pred + eps))
        snr_improve = snr_pred - snr_noisy

        y_np = y.detach().cpu().numpy()
        mse_noisy_np = mse_noisy.detach().cpu().numpy()
        mse_pred_np = mse_pred.detach().cpu().numpy()
        snr_noisy_np = snr_noisy.detach().cpu().numpy()
        snr_pred_np = snr_pred.detach().cpu().numpy()
        snr_improve_np = snr_improve.detach().cpu().numpy()

        for i in range(len(y_np)):
            c = int(y_np[i])
            counts[c] += 1
            acc(c, "mse_noisy_vs_clean_sum", float(mse_noisy_np[i]))
            acc(c, "mse_pred_vs_clean_sum", float(mse_pred_np[i]))
            acc(c, "snr_noisy_db_sum", float(snr_noisy_np[i]))
            acc(c, "snr_pred_db_sum", float(snr_pred_np[i]))
            acc(c, "snr_improve_db_sum", float(snr_improve_np[i]))

    names = {0: "good", 1: "medium", 2: "bad"}
    out_metrics: dict[str, Any] = {}
    for c in [0, 1, 2]:
        n = counts[c]
        if n <= 0:
            out_metrics[names[c]] = {"n": 0}
            continue
        out_metrics[names[c]] = {
            "n": int(n),
            "mse_noisy_vs_clean_mean": sums[c]["mse_noisy_vs_clean_sum"] / n,
            "mse_pred_vs_clean_mean": sums[c]["mse_pred_vs_clean_sum"] / n,
            "snr_noisy_db_mean": sums[c]["snr_noisy_db_sum"] / n,
            "snr_pred_db_mean": sums[c]["snr_pred_db_sum"] / n,
            "snr_improve_db_mean": sums[c]["snr_improve_db_sum"] / n,
        }

    return out_metrics


@torch.no_grad()
def export_denoise_examples_by_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_png: Path,
    k_per_class: int = 5,
    fs: int = 125,
) -> None:
    model.eval()

    xs_noisy = {0: [], 1: [], 2: []}
    xs_clean = {0: [], 1: [], 2: []}
    xs_pred  = {0: [], 1: [], 2: []}
    ps_true  = {0: [], 1: [], 2: []}
    ps_pred  = {0: [], 1: [], 2: []}
    y_pred_c = {0: [], 1: [], 2: []}

    # -------- collect k examples per true class --------
    for batch in loader:
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        x_clean = batch["x_clean"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.long)

        out = model(x_noisy)
        y_denoise, _, logits = out[0], out[1], out[2]

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        xn = x_noisy.squeeze(1).detach().cpu().numpy()
        xc = x_clean.squeeze(1).detach().cpu().numpy()
        xd = y_denoise.squeeze(1).detach().cpu().numpy()
        yy = y.detach().cpu().numpy()
        pp = probs.detach().cpu().numpy()
        pr = pred.detach().cpu().numpy()

        for i in range(len(yy)):
            c = int(yy[i])
            if len(xs_noisy[c]) >= k_per_class:
                continue

            xs_noisy[c].append(xn[i])
            xs_clean[c].append(xc[i])
            xs_pred[c].append(xd[i])

            p_true = float(pp[i, c])
            p_pr = int(pr[i])
            p_pred = float(pp[i, p_pr])
            ps_true[c].append(p_true)
            ps_pred[c].append(p_pred)
            y_pred_c[c].append(p_pr)

        if all(len(xs_noisy[c]) >= k_per_class for c in [0, 1, 2]):
            break

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # -------- layout: (class x signaltype) rows, k columns --------
    # rows: good(noisy,pred,clean), medium(noisy,pred,clean), bad(noisy,pred,clean)
    names = {0: "good", 1: "medium", 2: "bad"}
    sig_names = ["noisy", "denoise_pred", "clean"]

    # determine T
    t_len = None
    for c in [0, 1, 2]:
        if xs_noisy[c]:
            t_len = len(xs_noisy[c][0])
            break
    if t_len is None:
        print(f"Skip denoise examples (no samples): {out_png}")
        return

    t = (np.arange(t_len) / float(fs)).astype(np.float32)

    ncols = k_per_class
    nrows = 3 * 3  # 3 classes * 3 signal types

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.6 * ncols, 1.7 * nrows),
        sharex=True,
        sharey=False,
    )

    # handle k=1 shape
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    def row_index(c: int, s: int) -> int:
        # c in {0,1,2}, s in {0,1,2}
        return c * 3 + s

    # global formatting
    for r in range(nrows):
        for j in range(ncols):
            ax = axes[r, j]
            ax.grid(True, alpha=0.15)
            if j != 0:
                ax.set_yticklabels([])

    # -------- plot --------
    for c in [0, 1, 2]:
        for j in range(k_per_class):
            if j >= len(xs_noisy[c]):
                # blank this column for all 3 rows of this class
                for s in range(3):
                    axes[row_index(c, s), j].axis("off")
                continue

            xn = xs_noisy[c][j]
            xd = xs_pred[c][j]
            xc = xs_clean[c][j]

            # same y-lim for the three rows of THIS sample (easy comparison)
            y_min = float(min(xn.min(), xd.min(), xc.min()))
            y_max = float(max(xn.max(), xd.max(), xc.max()))
            pad = 0.05 * (y_max - y_min + 1e-8)
            y0, y1 = y_min - pad, y_max + pad

            yp = y_pred_c[c][j]
            title = (
                f"{names[c]} #{j+1} | pred={names.get(yp, yp)}\n"
                f"p(true)={ps_true[c][j]:.3f}  p(pred)={ps_pred[c][j]:.3f}"
            )

            # (1) noisy
            ax0 = axes[row_index(c, 0), j]
            ax0.plot(t, xn, linewidth=1.0)
            ax0.set_ylim(y0, y1)
            ax0.set_title(title, fontsize=9)

            # (2) denoise_pred
            ax1 = axes[row_index(c, 1), j]
            ax1.plot(t, xd, linewidth=1.0)
            ax1.set_ylim(y0, y1)

            # (3) clean
            ax2 = axes[row_index(c, 2), j]
            ax2.plot(t, xc, linewidth=1.0, linestyle="--", alpha=0.95)
            ax2.set_ylim(y0, y1)

        # left-side labels for each row group
        axes[row_index(c, 0), 0].set_ylabel(f"{names[c]}\nnoisy", fontsize=10)
        axes[row_index(c, 1), 0].set_ylabel(f"{names[c]}\npred", fontsize=10)
        axes[row_index(c, 2), 0].set_ylabel(f"{names[c]}\nclean", fontsize=10)

    for j in range(ncols):
        axes[-1, j].set_xlabel("time (s)")

    fig.suptitle(
        "Denoise examples by TRUE class (columns = samples; rows = noisy / pred / clean)",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved denoise examples: {out_png}")


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
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(uw.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR_ETA_MIN,
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
        "LAMBDA_CLS": LAMBDA_CLS,
        "LAMBDA_DEN": LAMBDA_DEN,
        "LAMBDA_LVL": LAMBDA_LVL,
    }

    history: list[dict[str, Any]] = []
    best_val = float("inf")

    for epoch in range(EPOCHS):
        global CUR_BAD_DEN_W
        if phase_name(epoch) == "B_add_denoise":
            # linearly ramp bad weight 0 -> BAD_DEN_W_MAX during B phase
            t = max(0.0, min(1.0, float(epoch - E_CLS) / float(max(1, BAD_DEN_W_WARMUP_EPOCHS))))
            CUR_BAD_DEN_W = float(BAD_DEN_W_MAX * t)
        else:
            CUR_BAD_DEN_W = 0.0

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
            f"train L={tr['total']:.4f} (cls={LAMBDA_CLS * tr['cls']:.4f}, den={LAMBDA_DEN * tr['denoise']:.4f}, lvl={LAMBDA_LVL * tr['level']:.4f}) | "
            f"val L={va['total']:.4f} (cls={LAMBDA_CLS * va['cls']:.4f}, den={LAMBDA_DEN * va['denoise']:.4f}, lvl={LAMBDA_LVL * va['level']:.4f})"
        )
        cur_lr = scheduler.get_last_lr()[0]
        print(f"LR={cur_lr:.2e} | CUR_BAD_DEN_W={CUR_BAD_DEN_W:.3f}")
        if phase in ("B_add_denoise", "C_add_level", "D_joint", "E_uncertainty_joint"):
            print(
                f"WaveletDbg: den_sample_frac={va.get('den_sample_frac', 0.0):.3f} | "
                f"gate_frac_good={va.get('wavelet_gate_frac_good', 0.0):.3f} | "
                f"w_mean={va.get('wavelet_w_mean', 0.0):.3f} | "
                f"baseMSE={va.get('den_l_base_mean', 0.0):.6f} | "
                f"wMSE={va.get('den_l_weighted_mean', 0.0):.6f}"
            )
        if phase == "E_uncertainty_joint":
            print(
                f"log_sigma cls/den/level="
                f"{float(uw.log_sigma_cls.detach().cpu().item()):.4f}/"
                f"{float(uw.log_sigma_denoise.detach().cpu().item()):.4f}/"
                f"{float(uw.log_sigma_level.detach().cpu().item()):.4f}"
            )

        val_detail = eval_split_details(model, val_loader, device)

        pc = val_detail["per_class"]
        mb = val_detail["medium_breakdown"]
        gb = val_detail["per_class"]["good"]["confusion_row"]
        print(
            f"VAL per-class acc: "
            f"good={pc['good']['acc']:.3f} (n={pc['good']['n']}, p_true={pc['good']['mean_p_trueclass']:.3f}) | "
            f"medium={pc['medium']['acc']:.3f} (n={pc['medium']['n']}, p_true={pc['medium']['mean_p_trueclass']:.3f}) | "
            f"bad={pc['bad']['acc']:.3f} (n={pc['bad']['n']}, p_true={pc['bad']['mean_p_trueclass']:.3f})"
        )
        print(
            f"VAL medium breakdown: to_good={mb['to_good']} to_medium={mb['to_medium']} to_bad={mb['to_bad']}"
        )
        print(f"VAL good breakdown: to_good={gb[0]} to_medium={gb[1]} to_bad={gb[2]}")

        scheduler.step()

        ckpt = {
                "epoch": epoch,
                "model_state": safe_state_dict(model),
                "uw_state": uw.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "hyperparams": hp,
        }
        torch.save(ckpt, OUT_LAST)

        # [ASSUMPTION] best checkpoint is only comparable in joint phases.
        if phase in ("B_add_denoise", "D_joint", "E_uncertainty_joint"):
            if va["total"] < best_val:
                best_val = va["total"]
                torch.save(ckpt, OUT_BEST)

    with OUT_LOG.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if not OUT_BEST.exists():
        shutil.copyfile(OUT_LAST, OUT_BEST)

    # Evaluate/export using best validation checkpoint, not the last epoch state.
    ckpt_best = torch.load(OUT_BEST, map_location=device)
    model.load_state_dict(ckpt_best["model_state"], strict=True)
    if "uw_state" in ckpt_best:
        uw.load_state_dict(ckpt_best["uw_state"], strict=True)

    test_report = eval_test_report(model, uw, test_loader, device)
    with OUT_TEST.open("w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    export_denoise_examples_by_class(
        model, test_loader, device,
        OUT_DIR / "debug" / "denoise_examples_test.png",
        k_per_class=5,
    )
    export_denoise_examples_by_class(
        model, val_loader, device,
        OUT_DIR / "debug" / "denoise_examples_val.png",
        k_per_class=5,
    )

    print(f"saved: {OUT_LAST}")
    print(f"saved: {OUT_BEST}")
    print(f"saved: {OUT_LOG}")
    print(f"saved: {OUT_TEST}")
    print("NEXT INPUT: artifact1/models/mtl_transformer_seed0_step6/ckpt_best_val.pt")


if __name__ == "__main__":
    main()
