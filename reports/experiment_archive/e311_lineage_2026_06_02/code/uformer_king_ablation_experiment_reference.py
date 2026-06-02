from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


THIS_FILE = Path(__file__).absolute()
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else next(p for p in THIS_FILE.parents if (p / "src").exists())
BASE_PATH = ROOT / "outputs" / "experiment" / "e311_sqi_denoise_classifier_grid" / "transunet1d_audit.py"
DEFAULT_SOURCE = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid" / "data" / "med6p25_badgap7_badcm0p75"


def import_base():
    spec = importlib.util.spec_from_file_location("e311_transunet1d_base_for_ablation", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e311_transunet1d_base_for_ablation"] = module
    spec.loader.exec_module(module)
    return module


base = import_base()
ORIG_BUILD_DENOISER = base.build_denoiser


class UpNoSkipBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = base.ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return self.block(x)


class UFormerNoSkip1D(nn.Module):
    """Uformer encoder with decoder skips removed; tests whether U-shaped skip locality is necessary."""

    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.e1 = base.ConvBlock(1, 32, kernel=9)
        self.d1 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(base.ConvBlock(64, 64), base.TransformerBlock1D(64, 4, dropout=dropout))
        self.d2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.e3 = nn.Sequential(base.ConvBlock(128, 128), base.TransformerBlock1D(128, 4, dropout=dropout))
        self.d3 = nn.Conv1d(128, 160, kernel_size=4, stride=2, padding=1)
        self.b = nn.Sequential(
            base.ConvBlock(160, 160),
            base.TransformerBlock1D(160, 5, dropout=dropout),
            base.TransformerBlock1D(160, 5, dropout=dropout),
        )
        self.u3 = UpNoSkipBlock(160, 96)
        self.u2 = UpNoSkipBlock(96, 64)
        self.u1 = UpNoSkipBlock(64, 32)
        self.out = nn.Conv1d(32, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.e1(x)
        s2 = self.e2(F.gelu(self.d1(s1)))
        s3 = self.e3(F.gelu(self.d2(s2)))
        b = self.b(F.gelu(self.d3(s3)))
        y = self.u3(b, s3.shape[-1])
        y = self.u2(y, s2.shape[-1])
        y = self.u1(y, s1.shape[-1])
        return {"noise_hat": self.out(y), "features": [s1, s2, s3, b], "bottleneck": b}


class UFormerNoTransformer1D(nn.Module):
    """Conv-only U-shaped control with matched scales; tests whether Transformer blocks add useful representation."""

    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.e1 = base.ConvBlock(1, 32, kernel=9)
        self.d1 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.e2 = base.ConvBlock(64, 64)
        self.d2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.e3 = base.ConvBlock(128, 128)
        self.d3 = nn.Conv1d(128, 160, kernel_size=4, stride=2, padding=1)
        self.b = base.ConvBlock(160, 160)
        self.u3 = base.UpBlock(160, 128, 96)
        self.u2 = base.UpBlock(96, 64, 64)
        self.u1 = base.UpBlock(64, 32, 32)
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
        return {"noise_hat": self.out(y), "features": [s1, s2, s3, b], "bottleneck": b}


class UFormerPatchStem1D(nn.Module):
    """Patch-projection stem control; tests whether the richer local conv stem is needed for ECG morphology."""

    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.d1 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(base.TransformerBlock1D(64, 4, dropout=dropout), base.ConvBlock(64, 64))
        self.d2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.e3 = nn.Sequential(base.TransformerBlock1D(128, 4, dropout=dropout), base.ConvBlock(128, 128))
        self.d3 = nn.Conv1d(128, 160, kernel_size=4, stride=2, padding=1)
        self.b = nn.Sequential(
            base.TransformerBlock1D(160, 5, dropout=dropout),
            base.TransformerBlock1D(160, 5, dropout=dropout),
            base.ConvBlock(160, 160),
        )
        self.u3 = base.UpBlock(160, 128, 96)
        self.u2 = base.UpBlock(96, 64, 64)
        self.u1 = base.UpBlock(64, 32, 32)
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
        return {"noise_hat": self.out(y), "features": [s1, s2, s3, b], "bottleneck": b}


def build_denoiser(model_type: str, dropout: float) -> nn.Module:
    if model_type == "uformer_no_skip":
        return UFormerNoSkip1D(dropout=dropout)
    if model_type == "uformer_no_transformer":
        return UFormerNoTransformer1D(dropout=dropout)
    if model_type == "uformer_patch_stem":
        return UFormerPatchStem1D(dropout=dropout)
    return ORIG_BUILD_DENOISER(model_type, dropout)


def main() -> None:
    base.build_denoiser = build_denoiser
    parser = argparse.ArgumentParser(description="Experiment-only Uformer king mechanism ablation audit.")
    parser.add_argument("--stage", choices=("denoise_pretrain", "sqi_head", "joint_finetune"), default="denoise_pretrain")
    parser.add_argument(
        "--model_type",
        choices=("uformer1d_hier", "uformer_no_skip", "uformer_no_transformer", "uformer_patch_stem"),
        required=True,
    )
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_checkpoint", default="none")
    parser.add_argument("--feature_set", choices=("full_tokens", "bottleneck_only", "summary_only", "residual_summary_only"), default="full_tokens")
    parser.add_argument("--freeze_denoiser", action="store_true")
    parser.add_argument("--detach_encoder_features", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--loss_variant", default="morph_soft_guard")
    parser.add_argument("--noise_scale", type=float, default=0.9)
    parser.add_argument("--lambda_den", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.0)
    parser.add_argument("--class_weight", default="1,1.40,1.70")
    parser.add_argument("--lr_denoiser", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()
    base.train(args)


if __name__ == "__main__":
    main()
