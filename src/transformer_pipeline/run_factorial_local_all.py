from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_pipeline.evaluate_factorial_local import run as evaluate_factorial
from src.transformer_pipeline.noise.make_rr_noise_level import run as make_noise_level
from src.transformer_pipeline.noise.synthesize_factorial_local_quality import run as synthesize_factorial


def main() -> None:
    args = parse_args()
    params = vars(args)
    synthesize_factorial(params)
    make_noise_level(params)
    evaluate_factorial(
        {
            "artifact_dir": args.artifact_dir,
            "out_dir": str(Path(args.artifact_dir) / "validation"),
            "sqi_summary": args.sqi_summary,
            "sqi_predictions": args.sqi_predictions,
            "model_dir": args.model_dir,
            "batch_size": args.batch_size,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Factorial Local Quality Benchmark data and validation steps.")
    parser.add_argument("--artifact_dir", default="outputs/transformer_factorial_local")
    parser.add_argument("--source_artifact_dir", default="outputs/transformer")
    parser.add_argument("--nstdb_root", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_clean", type=int, default=240)
    parser.add_argument("--max_val_clean", type=int, default=60)
    parser.add_argument("--max_test_clean", type=int, default=60)
    parser.add_argument("--noise_kinds", default="em,ma,bw,mix")
    parser.add_argument("--snr_profiles", default="-2,4,8,12,20")
    parser.add_argument("--heldout_noise_kind", default="ma")
    parser.add_argument("--heldout_snr_profiles", default="8")
    parser.add_argument("--sqi_summary", default="")
    parser.add_argument("--sqi_predictions", default="")
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
