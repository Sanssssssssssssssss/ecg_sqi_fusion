from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import predict_records
from .models import MODEL_NAMES, export_inference_bundles, get_predictor, verify_inference_bundles


def main() -> None:
    """Parse CLI arguments and run prediction, export, or bundle verification."""

    p = argparse.ArgumentParser(description="Inference-only ECG SQI segment classifier.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pred = sub.add_parser("predict")
    pred.add_argument("--model", required=True, choices=MODEL_NAMES)
    pred.add_argument("--input", required=True)
    pred.add_argument("--fs", required=True, type=float)
    pred.add_argument("--out", required=True)
    pred.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    exp = sub.add_parser("export-inference-bundle")
    exp.add_argument("--out", default=None)

    sub.add_parser("verify-bundles")

    args = p.parse_args()
    if args.cmd == "predict":
        verify_inference_bundles()
        predictor = get_predictor(args.model, device=args.device)
        summary = predict_records(
            input_path=Path(args.input),
            out_dir=Path(args.out),
            fs=float(args.fs),
            predictor=predictor,
        )
        print(json.dumps(summary, indent=2))
    elif args.cmd == "export-inference-bundle":
        out = Path(args.out) if args.out else None
        print(json.dumps(export_inference_bundles(out), indent=2))
    elif args.cmd == "verify-bundles":
        print(json.dumps(verify_inference_bundles(), indent=2))
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
