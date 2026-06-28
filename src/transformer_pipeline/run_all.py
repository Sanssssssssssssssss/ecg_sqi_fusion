from __future__ import annotations

import argparse

from src.transformer_pipeline.config import TransformerPipelineConfig
from src.transformer_pipeline.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official v116/E31 Transformer pipeline.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--train", choices=["none", "E4", "E24", "E31", "both", "all"], default="none")
    parser.add_argument("--artifacts-dir", default="outputs/transformer/v116_e31")
    parser.add_argument("--seed", type=int, default=20260876)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = TransformerPipelineConfig.build(artifacts_dir=args.artifacts_dir, seed=args.seed, force=args.force)
    run_pipeline(cfg, run=args.run, train=args.train)


if __name__ == "__main__":
    main()
