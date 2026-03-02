# src/experiment/ptbxl_check_model_forward.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root

from src.models.mtl_transformer import MTLTransformerPTBXL, MTLTransformerConfig


def main() -> None:
    root = project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MTLTransformerConfig(
        fs=125,
        window_sec=10,
        T=1250,
        patch_size=20,
        stride=10,
        L=124,
        conv3_padding=10,  # critical closure knob
        dropout=0.0,
    )
    model = MTLTransformerPTBXL(cfg).to(device)
    model.eval()

    print("=== Model created ===")
    print(model.__class__.__name__)
    print(f"device={device}")
    print(f"cfg={cfg}")

    # ---------------- Dummy forward ----------------
    x = torch.randn(2, 1, cfg.T, device=device)
    with torch.no_grad():
        y_denoise, y_level, logits = model(x)

    print("\n=== Dummy forward shapes ===")
    print(f"x         : {tuple(x.shape)}")
    print(f"y_denoise : {tuple(y_denoise.shape)}  (expect (B,1,1250))")
    print(f"y_level   : {tuple(y_level.shape)}    (expect (B,1,1250))")
    print(f"logits    : {tuple(logits.shape)}     (expect (B,3))")

    # ---------------- Real sample forward (optional) ----------------
    npz_noisy = root / "artifact1" / "datasets" / "synth_10s_125hz_noisy.npz"
    if npz_noisy.exists():
        Xn = np.load(npz_noisy)["X_noisy"].astype(np.float32)
        x0 = torch.from_numpy(Xn[0]).to(device).view(1, 1, cfg.T)
        with torch.no_grad():
            y_denoise0, y_level0, logits0 = model(x0)

        print("\n=== Real sample forward (idx=0) ===")
        print(f"x0        : {tuple(x0.shape)}")
        print(f"y_denoise0 : {tuple(y_denoise0.shape)}")
        print(f"y_level0   : {tuple(y_level0.shape)}")
        print(f"logits0    : {tuple(logits0.shape)}")

        # sanity: finite
        assert torch.isfinite(y_denoise0).all()
        assert torch.isfinite(y_level0).all()
        assert torch.isfinite(logits0).all()
        print("finite check: OK")
    else:
        print("\n[SKIP] No real dataset found at artifact1/datasets/synth_10s_125hz_noisy.npz")

    print("\nNEXT: if L mismatch happens, adjust cfg.conv3_padding (or revisit conv3 dilation/padding).")


if __name__ == "__main__":
    main()