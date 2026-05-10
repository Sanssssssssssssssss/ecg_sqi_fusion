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
    dropout = float(_value_or_default(params.get("dropout"), MTLTransformerConfig().dropout))
    cls_pool = str(_value_or_default(params.get("cls_pool"), MTLTransformerConfig().cls_pool))
    if cls_pool not in {"decoder", "encoder", "both"}:
        raise ValueError("cls_pool must be 'decoder', 'encoder', or 'both'")
    input_mode = str(_value_or_default(params.get("input_mode"), "raw"))
    if input_mode not in {"raw", "robust", "raw_robust"}:
        raise ValueError("input_mode must be 'raw', 'robust', or 'raw_robust'")
    in_ch = 2 if input_mode == "raw_robust" else 1
    use_ordinal_head = bool(params.get("ordinal_head", False))
    use_snr_head = bool(params.get("snr_head", False))
    use_local_mask_head = bool(params.get("local_mask_head", False))
    use_noise_type_head = bool(params.get("noise_type_head", False))
    use_sqi_head = bool(params.get("sqi_head", False))

    cfg = MTLTransformerConfig(
        fs=125,
        window_sec=10,
        in_ch=in_ch,
        T=1250,
        patch_size=20,
        stride=10,
        L=124,
        conv3_padding=10,  # critical closure knob
        dropout=dropout,
        cls_pool=cls_pool,
        use_ordinal_head=use_ordinal_head,
        use_snr_head=use_snr_head,
        use_local_mask_head=use_local_mask_head,
        use_noise_type_head=use_noise_type_head,
        use_sqi_head=use_sqi_head,
    )
    model = MTLTransformerPTBXL(cfg).to(device)
    model.eval()

    print("=== Model created ===")
    print(model.__class__.__name__)
    print(f"device={device}")
    print(f"cfg={cfg}")

    # ---------------- Dummy forward ----------------
    x = torch.randn(2, cfg.in_ch, cfg.T, device=device)
    with torch.no_grad():
        out = model(x)
        y_denoise, y_level, logits = out[0], out[1], out[2]

    print("\n=== Dummy forward shapes ===")
    print(f"x         : {tuple(x.shape)}")
    print(f"y_denoise : {tuple(y_denoise.shape)}  (expect (B,1,1250))")
    print(f"y_level   : {tuple(y_level.shape)}    (expect (B,1,1250))")
    print(f"logits    : {tuple(logits.shape)}     (expect (B,3))")
    _print_extra_shapes(out, cfg)

    # ---------------- Real sample forward (optional) ----------------
    npz_noisy = artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz"
    if npz_noisy.exists():
        Xn = np.load(npz_noisy)["X_noisy"].astype(np.float32)
        raw = torch.from_numpy(Xn[0]).to(device).view(1, 1, cfg.T)
        if cfg.in_ch == 2:
            robust = (raw - raw.median()) / (torch.quantile(raw, 0.75) - torch.quantile(raw, 0.25) + 1e-6)
            x0 = torch.cat([raw, robust.clamp(-10.0, 10.0)], dim=1)
        else:
            x0 = raw
        with torch.no_grad():
            out0 = model(x0)
            y_denoise0, y_level0, logits0 = out0[0], out0[1], out0[2]

        print("\n=== Real sample forward (idx=0) ===")
        print(f"x0        : {tuple(x0.shape)}")
        print(f"y_denoise0 : {tuple(y_denoise0.shape)}")
        print(f"y_level0   : {tuple(y_level0.shape)}")
        print(f"logits0    : {tuple(logits0.shape)}")
        _print_extra_shapes(out0, cfg)

        # sanity: finite
        assert torch.isfinite(y_denoise0).all()
        assert torch.isfinite(y_level0).all()
        assert torch.isfinite(logits0).all()
        print("finite check: OK")
    else:
        print(f"\n[SKIP] No real dataset found at {npz_noisy}")

    return {"step": "forward_check", "skipped": False, "outputs": []}


def _print_extra_shapes(out: tuple[torch.Tensor, ...], cfg: MTLTransformerConfig) -> None:
    extra_i = 3
    if cfg.use_ordinal_head:
        print(f"ordinal   : {tuple(out[extra_i].shape)}     (expect (B,2))")
        extra_i += 1
    if cfg.use_snr_head:
        print(f"snr_hat   : {tuple(out[extra_i].shape)}       (expect (B,))")
        extra_i += 1
    if cfg.use_local_mask_head:
        print(f"local_mask: {tuple(out[extra_i].shape)}  (expect (B,1,1250))")
        extra_i += 1
    if cfg.use_noise_type_head:
        print(f"noise_type: {tuple(out[extra_i].shape)}     (expect (B,4))")
        extra_i += 1
    if cfg.use_sqi_head:
        print(f"sqi_hat   : {tuple(out[extra_i].shape)}     (expect (B,7))")


def _value_or_default(value: object, default: object) -> object:
    return default if value is None or value == "" else value


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a transformer model forward sanity check.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--cls_pool", choices=("decoder", "encoder", "both"))
    parser.add_argument("--input_mode", choices=("raw", "robust", "raw_robust"))
    parser.add_argument("--ordinal_head", action="store_true")
    parser.add_argument("--snr_head", action="store_true")
    parser.add_argument("--local_mask_head", action="store_true")
    parser.add_argument("--noise_type_head", action="store_true")
    parser.add_argument("--sqi_head", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
