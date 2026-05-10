# Transformer model forward check.
from __future__ import annotations

import argparse
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

from src.transformer_pipeline.models.mtl_transformer import MTLTransformerPTBXL, MTLTransformerConfig


def run(params: dict | None = None) -> dict:
    params = params or {}
    root = project_root()
    artifact_dir = Path(str(params.get("artifact_dir") or (root / "outputs/transformer")))
    if not artifact_dir.is_absolute():
        artifact_dir = root / artifact_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = float(params.get("dropout", MTLTransformerConfig().dropout))
    cls_pool = str(params.get("cls_pool", MTLTransformerConfig().cls_pool))
    if cls_pool not in {"decoder", "encoder", "both"}:
        raise ValueError("cls_pool must be 'decoder', 'encoder', or 'both'")

    cfg = MTLTransformerConfig(
        fs=125,
        window_sec=10,
        T=1250,
        patch_size=20,
        stride=10,
        L=124,
        conv3_padding=10,  # critical closure knob
        dropout=dropout,
        cls_pool=cls_pool,
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
    npz_noisy = artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz"
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
        print(f"\n[SKIP] No real dataset found at {npz_noisy}")

    return {"step": "forward_check", "skipped": False, "outputs": []}


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a transformer model forward sanity check.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--cls_pool", choices=("decoder", "encoder", "both"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
