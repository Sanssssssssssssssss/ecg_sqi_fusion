from __future__ import annotations

# --------- Imports ---------
import argparse
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
    from src.transformer_pipeline.models.mtl_transformer import MTLTransformerConfig, MTLTransformerPTBXL
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root
    from src.transformer_pipeline.models.mtl_transformer import MTLTransformerConfig, MTLTransformerPTBXL


# --------- Fixed Hyperparams (edit here) ---------
SEED = 0
BATCH_SIZE = 32
WEIGHT_DECAY = 0.03
NUM_WORKERS = 0
PIN_MEMORY = False
VERBOSE = False
MODEL_DROPOUT = MTLTransformerConfig().dropout
CLS_POOL = MTLTransformerConfig().cls_pool
INPUT_MODE = "raw"
USE_ORDINAL_HEAD = False
USE_SNR_HEAD = False
USE_LOCAL_MASK_HEAD = False
USE_NOISE_TYPE_HEAD = False
USE_TEACHER_DISTILL = False
USE_SQI_HEAD = False

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
LAMBDA_ORD = 0.5
LAMBDA_SNR = 0.3
LAMBDA_LOCAL_MASK = 0.5
LAMBDA_NOISE_TYPE = 0.2
LAMBDA_TEACHER = 0.05
LAMBDA_SQI = 0.1
TEACHER_TEMPERATURE = 1.0
LABEL_SMOOTHING = 0.0
CLASS_WEIGHT_GOOD = 1.0
CLASS_WEIGHT_MEDIUM = 1.0
CLASS_WEIGHT_BAD = 1.0
WAVELET = "db6"
WAVELET_LEVEL = 4
WAVELET_Q = 0.65


# --------- Early Stop ---------
EARLYSTOP_PATIENCE = 4
EARLYSTOP_MIN_DELTA = 5e-3
EARLYSTOP_START_EPOCH = E_CLS + 3   # start from denoise
EARLYSTOP_PHASES = {"B_add_denoise", "D_joint", "E_uncertainty_joint"}
EARLYSTOP_ENABLED = False
SELECT_BEST_BY = "val_acc"
UNCERTAINTY_MODE = "kendall"
INIT_CHECKPOINT = ""
TEACHER_TARGETS = ""


# --------- Paths ---------
ROOT = project_root()
IN_NOISY = ROOT / "outputs/transformer" / "datasets" / "synth_10s_125hz_noisy.npz"
IN_CLEAN = ROOT / "outputs/transformer" / "datasets" / "synth_10s_125hz_clean.npz"
IN_LEVEL = ROOT / "outputs/transformer" / "datasets" / "synth_10s_125hz_noise_level.npz"
IN_LABELS = ROOT / "outputs/transformer" / "datasets" / "synth_10s_125hz_labels_with_level.csv"
IN_LOCAL_MASK = ROOT / "outputs/transformer" / "datasets" / "synth_10s_125hz_local_mask.npz"

OUT_DIR = ROOT / "outputs/transformer" / "models" / "mtl_transformer_seed0_step6"
OUT_LAST = OUT_DIR / "ckpt_last.pt"
OUT_BEST = OUT_DIR / "ckpt_best_val.pt"
OUT_BEST_ACC = OUT_DIR / "ckpt_best_val_acc.pt"
OUT_BEST_LOSS = OUT_DIR / "ckpt_best_val_loss.pt"
OUT_LOG = OUT_DIR / "train_log.json"
OUT_TEST = OUT_DIR / "test_report.json"
OUT_PROBE = OUT_DIR / "probe_summary.json"


def configure_from_params(params: dict[str, Any]) -> None:
    global SEED, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, EPOCHS, VERBOSE
    global WEIGHT_DECAY, LR, LR_ETA_MIN, MODEL_DROPOUT, CLS_POOL, INPUT_MODE
    global USE_ORDINAL_HEAD, USE_SNR_HEAD, USE_LOCAL_MASK_HEAD, USE_NOISE_TYPE_HEAD
    global USE_TEACHER_DISTILL, USE_SQI_HEAD
    global E_CLS, E_DENOISE, E_LEVEL, E_UNCERT
    global BAD_DEN_W_MAX, BAD_DEN_W_WARMUP_EPOCHS
    global LAMBDA_CLS, LAMBDA_DEN, LAMBDA_LVL, LAMBDA_ORD, LAMBDA_SNR
    global LAMBDA_LOCAL_MASK, LAMBDA_NOISE_TYPE
    global LAMBDA_TEACHER, LAMBDA_SQI, TEACHER_TEMPERATURE
    global LABEL_SMOOTHING, CLASS_WEIGHT_GOOD, CLASS_WEIGHT_MEDIUM, CLASS_WEIGHT_BAD
    global EARLYSTOP_ENABLED, EARLYSTOP_PATIENCE, EARLYSTOP_MIN_DELTA, EARLYSTOP_START_EPOCH
    global SELECT_BEST_BY, UNCERTAINTY_MODE, INIT_CHECKPOINT, TEACHER_TARGETS
    global IN_NOISY, IN_CLEAN, IN_LEVEL, IN_LABELS, IN_LOCAL_MASK
    global OUT_DIR, OUT_LAST, OUT_BEST, OUT_BEST_ACC, OUT_BEST_LOSS, OUT_LOG, OUT_TEST, OUT_PROBE

    if params.get("seed") is not None:
        SEED = int(params["seed"])
    if params.get("batch_size") is not None:
        BATCH_SIZE = int(params["batch_size"])
    if params.get("num_workers") is not None:
        NUM_WORKERS = int(params["num_workers"])
    if params.get("pin_memory") is not None:
        PIN_MEMORY = bool(params["pin_memory"])
    if params.get("epochs") is not None:
        EPOCHS = int(params["epochs"])
    if params.get("weight_decay") is not None:
        WEIGHT_DECAY = float(params["weight_decay"])
    if params.get("lr") is not None:
        LR = float(params["lr"])
    if params.get("lr_eta_min") is not None:
        LR_ETA_MIN = float(params["lr_eta_min"])
    if params.get("dropout") is not None:
        MODEL_DROPOUT = float(params["dropout"])
    if params.get("cls_pool") is not None:
        CLS_POOL = str(params["cls_pool"])
        if CLS_POOL not in {"decoder", "encoder", "both"}:
            raise ValueError("cls_pool must be 'decoder', 'encoder', or 'both'")
    if params.get("input_mode") is not None:
        INPUT_MODE = str(params["input_mode"])
        if INPUT_MODE not in {"raw", "robust", "raw_robust"}:
            raise ValueError("input_mode must be 'raw', 'robust', or 'raw_robust'")
    if params.get("ordinal_head") is not None:
        USE_ORDINAL_HEAD = bool(params["ordinal_head"])
    if params.get("snr_head") is not None:
        USE_SNR_HEAD = bool(params["snr_head"])
    if params.get("local_mask_head") is not None:
        USE_LOCAL_MASK_HEAD = bool(params["local_mask_head"])
    if params.get("noise_type_head") is not None:
        USE_NOISE_TYPE_HEAD = bool(params["noise_type_head"])
    if params.get("teacher_distill") is not None:
        USE_TEACHER_DISTILL = bool(params["teacher_distill"])
    if params.get("sqi_head") is not None:
        USE_SQI_HEAD = bool(params["sqi_head"])
    if params.get("e_cls") is not None:
        E_CLS = int(params["e_cls"])
    if params.get("e_denoise") is not None:
        E_DENOISE = int(params["e_denoise"])
    if params.get("e_level") is not None:
        E_LEVEL = int(params["e_level"])
    if params.get("e_uncert") is not None:
        E_UNCERT = int(params["e_uncert"])
    if params.get("bad_den_w_max") is not None:
        BAD_DEN_W_MAX = float(params["bad_den_w_max"])
    if params.get("bad_den_w_warmup_epochs") is not None:
        BAD_DEN_W_WARMUP_EPOCHS = int(params["bad_den_w_warmup_epochs"])
    if params.get("lambda_cls") is not None:
        LAMBDA_CLS = float(params["lambda_cls"])
    if params.get("lambda_den") is not None:
        LAMBDA_DEN = float(params["lambda_den"])
    if params.get("lambda_lvl") is not None:
        LAMBDA_LVL = float(params["lambda_lvl"])
    if params.get("lambda_ord") is not None:
        LAMBDA_ORD = float(params["lambda_ord"])
    if params.get("lambda_snr") is not None:
        LAMBDA_SNR = float(params["lambda_snr"])
    if params.get("lambda_local_mask") is not None:
        LAMBDA_LOCAL_MASK = float(params["lambda_local_mask"])
    if params.get("lambda_noise_type") is not None:
        LAMBDA_NOISE_TYPE = float(params["lambda_noise_type"])
    if params.get("lambda_teacher") is not None:
        LAMBDA_TEACHER = float(params["lambda_teacher"])
    if params.get("lambda_sqi") is not None:
        LAMBDA_SQI = float(params["lambda_sqi"])
    if params.get("teacher_temperature") is not None:
        TEACHER_TEMPERATURE = float(params["teacher_temperature"])
    if params.get("label_smoothing") is not None:
        LABEL_SMOOTHING = float(params["label_smoothing"])
    if params.get("class_weight_good") is not None:
        CLASS_WEIGHT_GOOD = float(params["class_weight_good"])
    if params.get("class_weight_medium") is not None:
        CLASS_WEIGHT_MEDIUM = float(params["class_weight_medium"])
    if params.get("class_weight_bad") is not None:
        CLASS_WEIGHT_BAD = float(params["class_weight_bad"])
    if params.get("early_stop") is not None:
        EARLYSTOP_ENABLED = bool(params["early_stop"])
    if params.get("earlystop_patience") is not None:
        EARLYSTOP_PATIENCE = int(params["earlystop_patience"])
    if params.get("earlystop_min_delta") is not None:
        EARLYSTOP_MIN_DELTA = float(params["earlystop_min_delta"])
    if params.get("earlystop_start_epoch") is not None:
        EARLYSTOP_START_EPOCH = int(params["earlystop_start_epoch"])
    if params.get("select_best_by") is not None:
        SELECT_BEST_BY = str(params["select_best_by"])
        if SELECT_BEST_BY not in {"val_acc", "val_loss"}:
            raise ValueError("select_best_by must be 'val_acc' or 'val_loss'")
    if params.get("uncertainty_mode") is not None:
        UNCERTAINTY_MODE = str(params["uncertainty_mode"])
        if UNCERTAINTY_MODE not in {"kendall", "fixed"}:
            raise ValueError("uncertainty_mode must be 'kendall' or 'fixed'")
    if params.get("init_checkpoint") is not None:
        INIT_CHECKPOINT = str(params["init_checkpoint"])
    if params.get("teacher_targets") is not None:
        TEACHER_TARGETS = str(params["teacher_targets"])
    VERBOSE = bool(params.get("verbose", False))

    art: Path | None = None
    artifact_dir = params.get("artifact_dir")
    if artifact_dir:
        art = Path(str(artifact_dir))
        if not art.is_absolute():
            art = ROOT / art
        IN_NOISY = art / "datasets" / "synth_10s_125hz_noisy.npz"
        IN_CLEAN = art / "datasets" / "synth_10s_125hz_clean.npz"
        IN_LEVEL = art / "datasets" / "synth_10s_125hz_noise_level.npz"
        IN_LABELS = art / "datasets" / "synth_10s_125hz_labels_with_level.csv"
        IN_LOCAL_MASK = art / "datasets" / "synth_10s_125hz_local_mask.npz"

    experiment_name = params.get("experiment_name")
    model_dir = params.get("model_dir")
    if model_dir:
        OUT_DIR = Path(str(model_dir))
        if not OUT_DIR.is_absolute():
            OUT_DIR = ROOT / OUT_DIR
        OUT_LAST = OUT_DIR / "ckpt_last.pt"
        OUT_BEST = OUT_DIR / "ckpt_best_val.pt"
        OUT_BEST_ACC = OUT_DIR / "ckpt_best_val_acc.pt"
        OUT_BEST_LOSS = OUT_DIR / "ckpt_best_val_loss.pt"
        OUT_LOG = OUT_DIR / "train_log.json"
        OUT_TEST = OUT_DIR / "test_report.json"
        OUT_PROBE = OUT_DIR / "probe_summary.json"
    elif art is not None and experiment_name:
        OUT_DIR = art / "models" / str(experiment_name)
        OUT_LAST = OUT_DIR / "ckpt_last.pt"
        OUT_BEST = OUT_DIR / "ckpt_best_val.pt"
        OUT_BEST_ACC = OUT_DIR / "ckpt_best_val_acc.pt"
        OUT_BEST_LOSS = OUT_DIR / "ckpt_best_val_loss.pt"
        OUT_LOG = OUT_DIR / "train_log.json"
        OUT_TEST = OUT_DIR / "test_report.json"
    elif art is not None:
        OUT_DIR = art / "models" / f"mtl_transformer_seed{SEED}_step6"
        OUT_LAST = OUT_DIR / "ckpt_last.pt"
        OUT_BEST = OUT_DIR / "ckpt_best_val.pt"
        OUT_BEST_ACC = OUT_DIR / "ckpt_best_val_acc.pt"
        OUT_BEST_LOSS = OUT_DIR / "ckpt_best_val_loss.pt"
        OUT_LOG = OUT_DIR / "train_log.json"
        OUT_TEST = OUT_DIR / "test_report.json"
        OUT_PROBE = OUT_DIR / "probe_summary.json"

    OUT_PROBE = OUT_DIR / "probe_summary.json"


def output_paths() -> list[Path]:
    return [
        OUT_LAST,
        OUT_BEST,
        OUT_BEST_ACC,
        OUT_BEST_LOSS,
        OUT_LOG,
        OUT_TEST,
        OUT_PROBE,
        OUT_DIR / "debug" / "training_curves.png",
        OUT_DIR / "debug" / "denoise_examples_test.png",
        OUT_DIR / "debug" / "denoise_examples_val.png",
    ]


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
    def __init__(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        p: np.ndarray,
        valid_rr: np.ndarray,
        y: np.ndarray,
        snr_db: np.ndarray,
        local_mask: np.ndarray,
        noise_type: np.ndarray,
        teacher_probs: np.ndarray,
        sqi_target: np.ndarray,
        input_mode: str,
    ):
        self.noisy = noisy.astype(np.float32)
        self.clean = clean.astype(np.float32)
        self.p = p.astype(np.float32)
        self.valid_rr = (valid_rr.astype(np.float32) > 0.5).astype(np.float32)
        self.y = y.astype(np.int64)
        self.snr_db = snr_db.astype(np.float32)
        self.local_mask = local_mask.astype(np.float32)
        self.noise_type = noise_type.astype(np.int64)
        self.teacher_probs = teacher_probs.astype(np.float32)
        self.sqi_target = sqi_target.astype(np.float32)
        self.input_mode = input_mode

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        x = make_input_channels(self.noisy[i], self.input_mode)
        return {
            "x_noisy": torch.from_numpy(x),                        # (C,1250)
            "x_clean": torch.from_numpy(self.clean[i][None, :]),   # (1,1250)
            "p": torch.from_numpy(self.p[i]),                      # (1250,)
            "valid_rr": torch.tensor(self.valid_rr[i], dtype=torch.float32),
            "y": torch.tensor(self.y[i], dtype=torch.long),
            "ordinal": torch.from_numpy(ordinal_target_np(int(self.y[i]))),
            "snr_norm": torch.tensor(normalize_snr_db(float(self.snr_db[i])), dtype=torch.float32),
            "local_mask": torch.from_numpy(self.local_mask[i]),
            "noise_type": torch.tensor(self.noise_type[i], dtype=torch.long),
            "teacher_probs": torch.from_numpy(self.teacher_probs[i]),
            "sqi_target": torch.from_numpy(self.sqi_target[i]),
        }


def robust_normalize_1d(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    scale = float(q75 - q25)
    if scale < 1e-6:
        scale = float(np.std(x)) + 1e-6
    y = (x.astype(np.float32) - med) / scale
    return np.clip(y, -10.0, 10.0).astype(np.float32)


def make_input_channels(x: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "raw":
        return x.astype(np.float32, copy=False)[None, :]
    robust = robust_normalize_1d(x)
    if input_mode == "robust":
        return robust[None, :]
    return np.stack([x.astype(np.float32, copy=False), robust], axis=0)


def raw_signal_channel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected input tensor shape (B,C,T), got {tuple(x.shape)}")
    return x[:, 0, :]


def ordinal_target_np(y_int: int) -> np.ndarray:
    if y_int == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    if y_int == 1:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([1.0, 1.0], dtype=np.float32)


def normalize_snr_db(snr_db: float) -> float:
    return float((snr_db - 7.0) / 13.0)


NOISE_TYPE_TO_INT = {"em": 0, "ma": 1, "bw": 2, "mix": 3}
SQI_TARGET_COLUMNS = ("I__iSQI", "I__bSQI", "I__pSQI", "I__sSQI", "I__kSQI", "I__fSQI", "I__basSQI")


def map_noise_type(v: object) -> int:
    s = str(v).strip().lower()
    if s not in NOISE_TYPE_TO_INT:
        if USE_NOISE_TYPE_HEAD:
            raise ValueError(f"noise_type_head supports only {sorted(NOISE_TYPE_TO_INT)}, got {s!r}")
        return 0
    return NOISE_TYPE_TO_INT[s]


def build_split_arrays() -> tuple[dict[str, PTBXLMTLDataset], dict[str, dict[str, Any]]]:
    X_noisy = np.load(IN_NOISY)["X_noisy"].astype(np.float32)
    X_clean = np.load(IN_CLEAN)["X_clean"].astype(np.float32)
    z = np.load(IN_LEVEL)
    P = z["P"].astype(np.float32)
    valid_rr = z["valid_rr"].astype(np.uint8)
    df = pd.read_csv(IN_LABELS)
    if USE_LOCAL_MASK_HEAD:
        if not IN_LOCAL_MASK.exists():
            raise FileNotFoundError(f"local_mask_head requested but missing: {IN_LOCAL_MASK}")
        m_npz = np.load(IN_LOCAL_MASK)
        M = m_npz["M"].astype(np.float32)
    else:
        M = np.zeros_like(P, dtype=np.float32)
    teacher_probs = np.zeros((len(df), 3), dtype=np.float32)
    sqi_target = np.zeros((len(df), len(SQI_TARGET_COLUMNS)), dtype=np.float32)
    if USE_TEACHER_DISTILL or USE_SQI_HEAD:
        if not TEACHER_TARGETS:
            raise ValueError("teacher_distill/sqi_head requested but --teacher_targets is empty")
        teacher_path = Path(TEACHER_TARGETS)
        if not teacher_path.is_absolute():
            teacher_path = ROOT / teacher_path
        teacher = pd.read_csv(teacher_path)
        required_teacher = {"idx", "prob_good", "prob_medium", "prob_bad", *SQI_TARGET_COLUMNS}
        missing_teacher = required_teacher - set(teacher.columns)
        if missing_teacher:
            raise ValueError(f"teacher targets missing columns: {sorted(missing_teacher)}")
        teacher = teacher.sort_values("idx").reset_index(drop=True)
        if len(teacher) != len(df) or not np.array_equal(teacher["idx"].to_numpy(dtype=np.int64), np.arange(len(df))):
            raise ValueError("teacher targets must contain one row per dataset idx, sorted from 0..n-1")
        teacher_probs = teacher[["prob_good", "prob_medium", "prob_bad"]].to_numpy(dtype=np.float32)
        teacher_probs = np.clip(teacher_probs, 1e-6, 1.0)
        teacher_probs = teacher_probs / teacher_probs.sum(axis=1, keepdims=True)
        sqi_target = teacher[list(SQI_TARGET_COLUMNS)].to_numpy(dtype=np.float32)

    required = {"idx", "split", "y_class", "noise_kind", "snr_db", "seg_id", "ecg_id"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in labels: {sorted(miss)}")

    n = len(df)
    if not (X_noisy.shape[0] == X_clean.shape[0] == P.shape[0] == valid_rr.shape[0] == M.shape[0] == n):
        raise ValueError("Array/CSV row count mismatch")

    idx = df["idx"].to_numpy(dtype=np.int64)
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError("idx out of range")
    y_int = np.array([map_y_class(v) for v in df["y_class"].tolist()], dtype=np.int64)
    noise_type_int = np.array([map_noise_type(v) for v in df["noise_kind"].tolist()], dtype=np.int64)

    datasets: dict[str, PTBXLMTLDataset] = {}
    split_info: dict[str, dict[str, Any]] = {}
    for sp in ["train", "val", "test"]:
        m = (df["split"].astype(str).to_numpy() == sp)
        sp_idx = idx[m]
        order = np.argsort(sp_idx)
        sp_idx = sp_idx[order]
        sp_y = y_int[m][order]
        sp_snr = df["snr_db"].to_numpy(dtype=np.float32)[m][order]
        sp_mask = M[sp_idx]
        sp_noise_type = noise_type_int[m][order]
        sp_teacher = teacher_probs[sp_idx]
        sp_sqi = sqi_target[sp_idx]

        datasets[sp] = PTBXLMTLDataset(
            noisy=X_noisy[sp_idx],
            clean=X_clean[sp_idx],
            p=P[sp_idx],
            valid_rr=valid_rr[sp_idx],
            y=sp_y,
            snr_db=sp_snr,
            local_mask=sp_mask,
            noise_type=sp_noise_type,
            teacher_probs=sp_teacher,
            sqi_target=sp_sqi,
            input_mode=INPUT_MODE,
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


def uncertainty_weight_snapshot(uw: UncertaintyWeights) -> dict[str, float]:
    with torch.no_grad():
        return {
            "cls": float(torch.exp(-uw.log_sigma_cls).detach().cpu().item()),
            "denoise": float(torch.exp(-uw.log_sigma_denoise).detach().cpu().item()),
            "level": float(torch.exp(-uw.log_sigma_level).detach().cpu().item()),
        }


def build_model(device: torch.device) -> tuple[nn.Module, UncertaintyWeights]:
    in_ch = 2 if INPUT_MODE == "raw_robust" else 1
    cfg = MTLTransformerConfig(
        in_ch=in_ch,
        dropout=MODEL_DROPOUT,
        cls_pool=CLS_POOL,
        use_ordinal_head=USE_ORDINAL_HEAD,
        use_snr_head=USE_SNR_HEAD,
        use_local_mask_head=USE_LOCAL_MASK_HEAD,
        use_noise_type_head=USE_NOISE_TYPE_HEAD,
        noise_type_classes=len(NOISE_TYPE_TO_INT),
        use_sqi_head=USE_SQI_HEAD,
        sqi_dim=len(SQI_TARGET_COLUMNS),
    )
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
    ordinal_logits: torch.Tensor | None,
    snr_hat: torch.Tensor | None,
    local_mask_logits: torch.Tensor | None,
    noise_type_logits: torch.Tensor | None,
    sqi_hat: torch.Tensor | None,
    clean: torch.Tensor,
    p_target: torch.Tensor,
    valid_rr: torch.Tensor,
    y_int: torch.Tensor,
    ordinal_target: torch.Tensor,
    snr_target: torch.Tensor,
    local_mask_target: torch.Tensor,
    noise_type_target: torch.Tensor,
    teacher_probs: torch.Tensor,
    sqi_target: torch.Tensor,
    use_cls: bool,
    use_den: bool,
    use_lvl: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    z = torch.zeros((), device=clean.device, dtype=torch.float32)
    l_cls = z
    l_denoise = z
    l_level = z
    l_ord = z
    l_snr = z
    l_local = z
    l_noise_type = z
    l_teacher = z
    l_sqi = z
    dbg: dict[str, float] = {}

    if use_cls:
        cls_weight = None
        weights = (CLASS_WEIGHT_GOOD, CLASS_WEIGHT_MEDIUM, CLASS_WEIGHT_BAD)
        if any(abs(float(w) - 1.0) > 1e-12 for w in weights):
            cls_weight = torch.tensor(weights, device=logits.device, dtype=logits.dtype)
        ce = nn.CrossEntropyLoss(weight=cls_weight, label_smoothing=LABEL_SMOOTHING)
        l_cls = ce(logits, y_int)
        if USE_ORDINAL_HEAD and ordinal_logits is not None:
            l_ord = nn.BCEWithLogitsLoss()(ordinal_logits, ordinal_target)
        if USE_SNR_HEAD and snr_hat is not None:
            l_snr = nn.SmoothL1Loss(beta=0.2)(snr_hat, snr_target)
        if USE_NOISE_TYPE_HEAD and noise_type_logits is not None:
            l_noise_type = nn.CrossEntropyLoss()(noise_type_logits, noise_type_target)
        if USE_TEACHER_DISTILL:
            t = max(1e-6, float(TEACHER_TEMPERATURE))
            logp = torch.log_softmax(logits / t, dim=1)
            l_teacher = nn.KLDivLoss(reduction="batchmean")(logp, teacher_probs) * (t * t)
        if USE_SQI_HEAD and sqi_hat is not None:
            l_sqi = nn.SmoothL1Loss(beta=0.1)(sqi_hat, sqi_target)

    if USE_LOCAL_MASK_HEAD and local_mask_logits is not None:
        l_local = nn.BCEWithLogitsLoss()(local_mask_logits.squeeze(1), local_mask_target)

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

    return l_cls, l_denoise, l_level, l_ord, l_snr, l_local, l_noise_type, l_teacher, l_sqi, dbg


def combine_losses(
    l_cls: torch.Tensor,
    l_denoise: torch.Tensor,
    l_level: torch.Tensor,
    l_ord: torch.Tensor,
    l_snr: torch.Tensor,
    l_local: torch.Tensor,
    l_noise_type: torch.Tensor,
    l_teacher: torch.Tensor,
    l_sqi: torch.Tensor,
    phase: str,
    uw: UncertaintyWeights,
) -> torch.Tensor:
    use_cls, use_den, use_lvl, use_uncert = active_losses(phase)
    l1 = l_cls if use_cls else torch.zeros_like(l_cls)
    l2 = l_denoise if use_den else torch.zeros_like(l_cls)
    l3 = l_level if use_lvl else torch.zeros_like(l_cls)
    if not use_uncert or UNCERTAINTY_MODE == "fixed":
        return (
            LAMBDA_CLS * l1 + LAMBDA_DEN * l2 + LAMBDA_LVL * l3
            + LAMBDA_ORD * l_ord + LAMBDA_SNR * l_snr
            + LAMBDA_LOCAL_MASK * l_local + LAMBDA_NOISE_TYPE * l_noise_type
            + LAMBDA_TEACHER * l_teacher + LAMBDA_SQI * l_sqi
        )
    # [ASSUMPTION] Kendall homoscedastic uncertainty weighting.
    s1 = uw.log_sigma_cls
    s2 = uw.log_sigma_denoise
    s3 = uw.log_sigma_level
    return (
        torch.exp(-s1) * l1 + s1 + torch.exp(-s2) * l2 + s2 + torch.exp(-s3) * l3 + s3
        + LAMBDA_ORD * l_ord + LAMBDA_SNR * l_snr
        + LAMBDA_LOCAL_MASK * l_local + LAMBDA_NOISE_TYPE * l_noise_type
        + LAMBDA_TEACHER * l_teacher + LAMBDA_SQI * l_sqi
    )


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
    sum_ord = 0.0
    sum_snr = 0.0
    sum_local = 0.0
    sum_noise_type = 0.0
    sum_teacher = 0.0
    sum_sqi = 0.0
    dbg_sum: dict[str, float] = {}
    dbg_n = 0

    iterator = tqdm(
        loader,
        desc=f"{phase} {'train' if train_mode else 'val'}",
        leave=False,
        disable=(not VERBOSE or not sys.stderr.isatty()),
    )
    for batch in iterator:
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        x_clean = batch["x_clean"].to(device=device, dtype=torch.float32)
        p_t = batch["p"].to(device=device, dtype=torch.float32)
        valid_rr = batch["valid_rr"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.long)
        ordinal_target = batch["ordinal"].to(device=device, dtype=torch.float32)
        snr_target = batch["snr_norm"].to(device=device, dtype=torch.float32)
        local_mask_target = batch["local_mask"].to(device=device, dtype=torch.float32)
        noise_type_target = batch["noise_type"].to(device=device, dtype=torch.long)
        teacher_probs = batch["teacher_probs"].to(device=device, dtype=torch.float32)
        sqi_target = batch["sqi_target"].to(device=device, dtype=torch.float32)

        with torch.set_grad_enabled(train_mode):
            out = model(x_noisy)
            # [ASSUMPTION] adapt if model returns other container/order.
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                y_denoise, y_level, logits = out[0], out[1], out[2]
                extra_i = 3
                ordinal_logits = out[extra_i] if USE_ORDINAL_HEAD and len(out) > extra_i else None
                extra_i += 1 if USE_ORDINAL_HEAD else 0
                snr_hat = out[extra_i] if USE_SNR_HEAD and len(out) > extra_i else None
                extra_i += 1 if USE_SNR_HEAD else 0
                local_mask_logits = out[extra_i] if USE_LOCAL_MASK_HEAD and len(out) > extra_i else None
                extra_i += 1 if USE_LOCAL_MASK_HEAD else 0
                noise_type_logits = out[extra_i] if USE_NOISE_TYPE_HEAD and len(out) > extra_i else None
                extra_i += 1 if USE_NOISE_TYPE_HEAD else 0
                sqi_hat = out[extra_i] if USE_SQI_HEAD and len(out) > extra_i else None
            else:
                raise RuntimeError("Model output must provide y_denoise, y_level, logits")

            use_cls, use_den, use_lvl, _ = active_losses(phase)
            l_cls, l_den, l_lvl, l_ord, l_snr, l_local, l_noise_type, l_teacher, l_sqi, dbg = compute_losses(
                y_denoise, y_level, logits, ordinal_logits, snr_hat, local_mask_logits, noise_type_logits, sqi_hat,
                x_clean, p_t, valid_rr, y, ordinal_target, snr_target, local_mask_target, noise_type_target,
                teacher_probs, sqi_target,
                use_cls, use_den, use_lvl
            )
            l_total = combine_losses(l_cls, l_den, l_lvl, l_ord, l_snr, l_local, l_noise_type, l_teacher, l_sqi, phase, uw)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                l_total.backward()
                optimizer.step()

        if train_mode and VERBOSE and sys.stderr.isatty():
            post = {
                "L": f"{float(l_total.detach().cpu()):.3f}",
                "C": f"{float(l_cls.detach().cpu()):.3f}",
                "D": f"{float(l_den.detach().cpu()):.3f}",
                "Lv": f"{float(l_lvl.detach().cpu()):.3f}",
            }
            if USE_ORDINAL_HEAD:
                post["O"] = f"{float(l_ord.detach().cpu()):.3f}"
            if USE_SNR_HEAD:
                post["S"] = f"{float(l_snr.detach().cpu()):.3f}"
            if USE_LOCAL_MASK_HEAD:
                post["M"] = f"{float(l_local.detach().cpu()):.3f}"
            if USE_NOISE_TYPE_HEAD:
                post["N"] = f"{float(l_noise_type.detach().cpu()):.3f}"
            if USE_TEACHER_DISTILL:
                post["T"] = f"{float(l_teacher.detach().cpu()):.3f}"
            if USE_SQI_HEAD:
                post["Q"] = f"{float(l_sqi.detach().cpu()):.3f}"
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
        sum_ord += float(l_ord.detach().cpu().item()) * bsz
        sum_snr += float(l_snr.detach().cpu().item()) * bsz
        sum_local += float(l_local.detach().cpu().item()) * bsz
        sum_noise_type += float(l_noise_type.detach().cpu().item()) * bsz
        sum_teacher += float(l_teacher.detach().cpu().item()) * bsz
        sum_sqi += float(l_sqi.detach().cpu().item()) * bsz
        if dbg:
            dbg_n += 1
            for k, v in dbg.items():
                dbg_sum[k] = dbg_sum.get(k, 0.0) + float(v)

    if n == 0:
        out0 = {
            "total": 0.0, "cls": 0.0, "denoise": 0.0, "level": 0.0,
            "ordinal": 0.0, "snr": 0.0, "local_mask": 0.0, "noise_type": 0.0,
            "teacher": 0.0, "sqi": 0.0, "acc": 0.0,
        }
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
        "ordinal": sum_ord / n,
        "snr": sum_snr / n,
        "local_mask": sum_local / n,
        "noise_type": sum_noise_type / n,
        "teacher": sum_teacher / n,
        "sqi": sum_sqi / n,
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

        xn = raw_signal_channel(x_noisy)  # (B,T)
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

        xn = raw_signal_channel(x_noisy).detach().cpu().numpy()
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


def export_training_curves(history: list[dict[str, Any]], out_png: Path) -> None:
    if not history:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    train_acc = [float(row["train"]["acc"]) for row in history]
    val_acc = [float(row["val"]["acc"]) for row in history]
    train_loss = [float(row["train"]["total"]) for row in history]
    val_loss = [float(row["val"]["total"]) for row in history]

    val_detail = [row.get("val_detail", {}).get("per_class", {}) for row in history]
    cls_names = ["good", "medium", "bad"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(epochs, train_acc, marker="o", label="train")
    ax.plot(epochs, val_acc, marker="o", label="val")
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("acc")
    ax.grid(True, alpha=0.2)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, train_loss, marker="o", label="train")
    ax.plot(epochs, val_loss, marker="o", label="val")
    ax.set_title("Total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.2)
    ax.legend()

    ax = axes[1, 0]
    for name in cls_names:
        values = []
        for item in val_detail:
            values.append(float(item.get(name, {}).get("acc", np.nan)))
        ax.plot(epochs, values, marker="o", label=name)
    ax.set_title("Validation per-class accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("acc")
    ax.grid(True, alpha=0.2)
    ax.legend()

    ax = axes[1, 1]
    den = [float(row["val"].get("den_l_weighted_mean", np.nan)) for row in history]
    lvl = [float(row["val"].get("level", np.nan)) for row in history]
    ax.plot(epochs, den, marker="o", label="val denoise weighted mse")
    ax.plot(epochs, lvl, marker="o", label="val level mse")
    ax.set_title("Auxiliary probes")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_probe_summary(history: list[dict[str, Any]], test_report: dict[str, Any]) -> dict[str, Any]:
    if not history:
        return {"test_report": test_report}

    best_acc_row = max(history, key=lambda row: float(row.get("val_detail", {}).get("overall_acc", row["val"]["acc"])))
    best_loss_row = min(history, key=lambda row: float(row["val"]["total"]))
    last = history[-1]

    def compact_epoch(row: dict[str, Any]) -> dict[str, Any]:
        val_detail = row.get("val_detail", {})
        return {
            "epoch": int(row["epoch"]),
            "phase": row["phase"],
            "train_acc": float(row["train"]["acc"]),
            "val_acc": float(val_detail.get("overall_acc", row["val"]["acc"])),
            "val_loss": float(row["val"]["total"]),
            "val_per_class": val_detail.get("per_class", {}),
            "val_confusion_matrix_3x3": val_detail.get("confusion_matrix_3x3"),
            "log_sigma": row.get("log_sigma", {}),
            "uncert_weight": row.get("uncert_weight", {}),
        }

    hp = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lr_eta_min": LR_ETA_MIN,
        "weight_decay": WEIGHT_DECAY,
        "dropout": MODEL_DROPOUT,
        "cls_pool": CLS_POOL,
        "input_mode": INPUT_MODE,
        "ordinal_head": USE_ORDINAL_HEAD,
        "snr_head": USE_SNR_HEAD,
        "local_mask_head": USE_LOCAL_MASK_HEAD,
        "noise_type_head": USE_NOISE_TYPE_HEAD,
        "teacher_distill": USE_TEACHER_DISTILL,
        "sqi_head": USE_SQI_HEAD,
        "e_cls": E_CLS,
        "e_denoise": E_DENOISE,
        "e_level": E_LEVEL,
        "e_uncert": E_UNCERT,
        "bad_den_w_max": BAD_DEN_W_MAX,
        "bad_den_w_warmup_epochs": BAD_DEN_W_WARMUP_EPOCHS,
        "lambda_cls": LAMBDA_CLS,
        "lambda_den": LAMBDA_DEN,
        "lambda_lvl": LAMBDA_LVL,
        "lambda_ord": LAMBDA_ORD,
        "lambda_snr": LAMBDA_SNR,
        "lambda_local_mask": LAMBDA_LOCAL_MASK,
        "lambda_noise_type": LAMBDA_NOISE_TYPE,
        "lambda_teacher": LAMBDA_TEACHER,
        "lambda_sqi": LAMBDA_SQI,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "label_smoothing": LABEL_SMOOTHING,
        "class_weight_good": CLASS_WEIGHT_GOOD,
        "class_weight_medium": CLASS_WEIGHT_MEDIUM,
        "class_weight_bad": CLASS_WEIGHT_BAD,
        "alpha": ALPHA,
        "wavelet": WAVELET,
        "wavelet_level": WAVELET_LEVEL,
        "wavelet_q": WAVELET_Q,
        "select_best_by": SELECT_BEST_BY,
        "early_stop": EARLYSTOP_ENABLED,
        "uncertainty_weighting": E_UNCERT > 0,
        "uncertainty_mode": UNCERTAINTY_MODE,
        "init_checkpoint": INIT_CHECKPOINT,
        "teacher_targets": TEACHER_TARGETS,
    }

    last_val_acc = float(last.get("val_detail", {}).get("overall_acc", last["val"]["acc"]))
    last_train_acc = float(last["train"]["acc"])
    return {
        "hyperparams": hp,
        "best_val_acc_epoch": compact_epoch(best_acc_row),
        "best_val_loss_epoch": compact_epoch(best_loss_row),
        "last_epoch": compact_epoch(last),
        "last_train_val_acc_gap": last_train_acc - last_val_acc,
        "test_acc": test_report.get("acc"),
        "test_confusion_matrix_3x3": test_report.get("confusion_matrix_3x3"),
        "test_denoise_metrics_by_class": test_report.get("denoise_metrics_by_class", {}),
    }


# --------- Save Artifacts ---------
def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    configure_from_params(params)
    dry_run = bool(params.get("dry_run", False))

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
    if INIT_CHECKPOINT:
        ckpt_path = Path(INIT_CHECKPOINT)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        ckpt_init = torch.load(ckpt_path, map_location=device)
        state = ckpt_init.get("model_state", ckpt_init)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"initialized model from {ckpt_path} | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"init missing keys: {list(missing)[:8]}")
        if unexpected:
            print(f"init unexpected keys: {list(unexpected)[:8]}")

    if dry_run:
        batch = next(iter(train_loader))
        x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out = model(x_noisy)
            y_denoise, y_level, logits = out[0], out[1], out[2]
        expected = (x_noisy.shape[0], 1, MTLTransformerConfig().T)
        if tuple(y_denoise.shape) != expected:
            raise RuntimeError(f"dry-run denoise shape mismatch: {tuple(y_denoise.shape)} != {expected}")
        if tuple(y_level.shape) != expected:
            raise RuntimeError(f"dry-run level shape mismatch: {tuple(y_level.shape)} != {expected}")
        if tuple(logits.shape) != (x_noisy.shape[0], 3):
            raise RuntimeError(f"dry-run logits shape mismatch: {tuple(logits.shape)}")
        extra_i = 3
        if USE_ORDINAL_HEAD and tuple(out[extra_i].shape) != (x_noisy.shape[0], 2):
            raise RuntimeError(f"dry-run ordinal shape mismatch: {tuple(out[extra_i].shape)}")
        extra_i += 1 if USE_ORDINAL_HEAD else 0
        if USE_SNR_HEAD:
            if tuple(out[extra_i].shape) != (x_noisy.shape[0],):
                raise RuntimeError(f"dry-run snr shape mismatch: {tuple(out[extra_i].shape)}")
        extra_i += 1 if USE_SNR_HEAD else 0
        if USE_LOCAL_MASK_HEAD and tuple(out[extra_i].shape) != expected:
            raise RuntimeError(f"dry-run local mask shape mismatch: {tuple(out[extra_i].shape)}")
        extra_i += 1 if USE_LOCAL_MASK_HEAD else 0
        if USE_NOISE_TYPE_HEAD and tuple(out[extra_i].shape) != (x_noisy.shape[0], len(NOISE_TYPE_TO_INT)):
            raise RuntimeError(f"dry-run noise type shape mismatch: {tuple(out[extra_i].shape)}")
        extra_i += 1 if USE_NOISE_TYPE_HEAD else 0
        if USE_SQI_HEAD and tuple(out[extra_i].shape) != (x_noisy.shape[0], len(SQI_TARGET_COLUMNS)):
            raise RuntimeError(f"dry-run SQI shape mismatch: {tuple(out[extra_i].shape)}")
        print("[dry-run] transformer train inputs and one forward pass OK")
        return {"step": "train", "skipped": True, "dry_run": True, "outputs": []}

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
        "MODEL_DROPOUT": MODEL_DROPOUT,
        "CLS_POOL": CLS_POOL,
        "INPUT_MODE": INPUT_MODE,
        "USE_ORDINAL_HEAD": USE_ORDINAL_HEAD,
        "USE_SNR_HEAD": USE_SNR_HEAD,
        "USE_LOCAL_MASK_HEAD": USE_LOCAL_MASK_HEAD,
        "USE_NOISE_TYPE_HEAD": USE_NOISE_TYPE_HEAD,
        "USE_TEACHER_DISTILL": USE_TEACHER_DISTILL,
        "USE_SQI_HEAD": USE_SQI_HEAD,
        "E_CLS": E_CLS,
        "E_DENOISE": E_DENOISE,
        "E_LEVEL": E_LEVEL,
        "E_UNCERT": E_UNCERT,
        "BAD_DEN_W_MAX": BAD_DEN_W_MAX,
        "BAD_DEN_W_WARMUP_EPOCHS": BAD_DEN_W_WARMUP_EPOCHS,
        "ALPHA": ALPHA,
        "WAVELET": WAVELET,
        "WAVELET_LEVEL": WAVELET_LEVEL,
        "WAVELET_Q": WAVELET_Q,
        "LAMBDA_CLS": LAMBDA_CLS,
        "LAMBDA_DEN": LAMBDA_DEN,
        "LAMBDA_LVL": LAMBDA_LVL,
        "LAMBDA_ORD": LAMBDA_ORD,
        "LAMBDA_SNR": LAMBDA_SNR,
        "LAMBDA_LOCAL_MASK": LAMBDA_LOCAL_MASK,
        "LAMBDA_NOISE_TYPE": LAMBDA_NOISE_TYPE,
        "LAMBDA_TEACHER": LAMBDA_TEACHER,
        "LAMBDA_SQI": LAMBDA_SQI,
        "TEACHER_TEMPERATURE": TEACHER_TEMPERATURE,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "CLASS_WEIGHT_GOOD": CLASS_WEIGHT_GOOD,
        "CLASS_WEIGHT_MEDIUM": CLASS_WEIGHT_MEDIUM,
        "CLASS_WEIGHT_BAD": CLASS_WEIGHT_BAD,
        "SELECT_BEST_BY": SELECT_BEST_BY,
        "UNCERTAINTY_MODE": UNCERTAINTY_MODE,
        "INIT_CHECKPOINT": INIT_CHECKPOINT,
        "TEACHER_TARGETS": TEACHER_TARGETS,
    }

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_epoch_loss = -1
    best_epoch_acc = -1
    early_stop_best_score = float("-inf")
    early_stop_bad_epochs = 0

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

        print(
            f"[epoch {epoch:03d}] phase={phase} | "
            f"train acc={tr['acc']:.4f} val acc={va['acc']:.4f} | "
            f"train L={tr['total']:.4f} (cls={LAMBDA_CLS * tr['cls']:.4f}, den={LAMBDA_DEN * tr['denoise']:.4f}, "
            f"lvl={LAMBDA_LVL * tr['level']:.4f}, ord={LAMBDA_ORD * tr['ordinal']:.4f}, "
            f"snr={LAMBDA_SNR * tr['snr']:.4f}, mask={LAMBDA_LOCAL_MASK * tr['local_mask']:.4f}, "
            f"ntype={LAMBDA_NOISE_TYPE * tr['noise_type']:.4f}, teach={LAMBDA_TEACHER * tr['teacher']:.4f}, "
            f"sqi={LAMBDA_SQI * tr['sqi']:.4f}) | "
            f"val L={va['total']:.4f} (cls={LAMBDA_CLS * va['cls']:.4f}, den={LAMBDA_DEN * va['denoise']:.4f}, "
            f"lvl={LAMBDA_LVL * va['level']:.4f}, ord={LAMBDA_ORD * va['ordinal']:.4f}, "
            f"snr={LAMBDA_SNR * va['snr']:.4f}, mask={LAMBDA_LOCAL_MASK * va['local_mask']:.4f}, "
            f"ntype={LAMBDA_NOISE_TYPE * va['noise_type']:.4f}, teach={LAMBDA_TEACHER * va['teacher']:.4f}, "
            f"sqi={LAMBDA_SQI * va['sqi']:.4f})"
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
        if phase == "E_uncertainty_joint" and UNCERTAINTY_MODE == "kendall":
            uw_eff = uncertainty_weight_snapshot(uw)
            print(
                f"log_sigma cls/den/level="
                f"{float(uw.log_sigma_cls.detach().cpu().item()):.4f}/"
                f"{float(uw.log_sigma_denoise.detach().cpu().item()):.4f}/"
                f"{float(uw.log_sigma_level.detach().cpu().item()):.4f}"
            )
            print(
                f"uncert_weight cls/den/level="
                f"{uw_eff['cls']:.3f}/{uw_eff['denoise']:.3f}/{uw_eff['level']:.3f}"
            )
        elif phase == "E_uncertainty_joint":
            print("uncert_weight mode=fixed explicit lambda weights")

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

        row = {
            "epoch": epoch,
            "phase": phase,
            "train": tr,
            "val": va,
            "val_detail": val_detail,
            "log_sigma": {
                "cls": float(uw.log_sigma_cls.detach().cpu().item()),
                "denoise": float(uw.log_sigma_denoise.detach().cpu().item()),
                "level": float(uw.log_sigma_level.detach().cpu().item()),
            },
            "uncert_weight": uncertainty_weight_snapshot(uw),
        }
        history.append(row)

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

        val_acc = float(val_detail["overall_acc"])
        val_loss = float(va["total"])
        prev_best_val_loss = best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_loss = epoch
            torch.save(ckpt, OUT_BEST_LOSS)
        if (val_acc > best_val_acc + 1e-12) or (
            abs(val_acc - best_val_acc) <= 1e-12 and val_loss < prev_best_val_loss
        ):
            best_val_acc = val_acc
            best_epoch_acc = epoch
            torch.save(ckpt, OUT_BEST_ACC)

        if EARLYSTOP_ENABLED and epoch >= EARLYSTOP_START_EPOCH and phase in EARLYSTOP_PHASES:
            early_score = val_acc if SELECT_BEST_BY == "val_acc" else -val_loss
            if early_score > early_stop_best_score + EARLYSTOP_MIN_DELTA:
                early_stop_best_score = early_score
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
            if early_stop_bad_epochs >= EARLYSTOP_PATIENCE:
                print(
                    f"Early stop: no {SELECT_BEST_BY} improvement for "
                    f"{EARLYSTOP_PATIENCE} epochs after epoch {EARLYSTOP_START_EPOCH}."
                )
                break

    with OUT_LOG.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    selected_best = OUT_BEST_ACC if SELECT_BEST_BY == "val_acc" else OUT_BEST_LOSS
    if not selected_best.exists():
        selected_best = OUT_LAST
    shutil.copyfile(selected_best, OUT_BEST)
    print(
        f"best selection: {SELECT_BEST_BY} | "
        f"best_val_acc={best_val_acc:.4f} epoch={best_epoch_acc} | "
        f"best_val_loss={best_val_loss:.4f} epoch={best_epoch_loss}"
    )

    # Evaluate/export using best validation checkpoint, not the last epoch state.
    ckpt_best = torch.load(OUT_BEST, map_location=device)
    model.load_state_dict(ckpt_best["model_state"], strict=True)
    if "uw_state" in ckpt_best:
        uw.load_state_dict(ckpt_best["uw_state"], strict=True)

    test_report = eval_test_report(model, uw, test_loader, device)
    with OUT_TEST.open("w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    curves_png = OUT_DIR / "debug" / "training_curves.png"
    export_training_curves(history, curves_png)

    probe_summary = build_probe_summary(history, test_report)
    with OUT_PROBE.open("w", encoding="utf-8") as f:
        json.dump(probe_summary, f, ensure_ascii=False, indent=2)

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
    print(f"saved: {OUT_BEST_ACC}")
    print(f"saved: {OUT_BEST_LOSS}")
    print(f"saved: {OUT_LOG}")
    print(f"saved: {OUT_TEST}")
    print(f"saved: {OUT_PROBE}")
    print(f"saved: {curves_png}")
    return {"step": "train", "skipped": False, "outputs": [str(p) for p in output_paths()]}


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PTB-XL multi-task transformer.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_eta_min", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--cls_pool", choices=("decoder", "encoder", "both"))
    parser.add_argument("--input_mode", choices=("raw", "robust", "raw_robust"))
    parser.add_argument("--ordinal_head", action="store_true")
    parser.add_argument("--snr_head", action="store_true")
    parser.add_argument("--local_mask_head", action="store_true")
    parser.add_argument("--noise_type_head", action="store_true")
    parser.add_argument("--teacher_distill", action="store_true")
    parser.add_argument("--sqi_head", action="store_true")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--teacher_targets", default="")
    parser.add_argument("--e_cls", type=int)
    parser.add_argument("--e_denoise", type=int)
    parser.add_argument("--e_level", type=int)
    parser.add_argument("--e_uncert", type=int)
    parser.add_argument("--bad_den_w_max", type=float)
    parser.add_argument("--bad_den_w_warmup_epochs", type=int)
    parser.add_argument("--lambda_cls", type=float)
    parser.add_argument("--lambda_den", type=float)
    parser.add_argument("--lambda_lvl", type=float)
    parser.add_argument("--lambda_ord", type=float)
    parser.add_argument("--lambda_snr", type=float)
    parser.add_argument("--lambda_local_mask", type=float)
    parser.add_argument("--lambda_noise_type", type=float)
    parser.add_argument("--lambda_teacher", type=float)
    parser.add_argument("--lambda_sqi", type=float)
    parser.add_argument("--teacher_temperature", type=float)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--class_weight_good", type=float)
    parser.add_argument("--class_weight_medium", type=float)
    parser.add_argument("--class_weight_bad", type=float)
    parser.add_argument("--uncertainty_mode", choices=("kendall", "fixed"))
    parser.add_argument("--select_best_by", choices=("val_acc", "val_loss"))
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--earlystop_patience", type=int)
    parser.add_argument("--earlystop_min_delta", type=float)
    parser.add_argument("--earlystop_start_epoch", type=int)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
