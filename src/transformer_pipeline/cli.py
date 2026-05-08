from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_pipeline.config import TransformerPipelineConfig
from src.transformer_pipeline.runner import ensure_dirs, fresh_artifacts, run_pipeline, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PTB-XL transformer pipeline.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="delete generated transformer artifacts and rerun")
    parser.add_argument("--dry-run", action="store_true", help="for train: load data and run one forward pass, no training")
    parser.add_argument("--only", default="", help="comma-separated step names, e.g. forward_check,train")
    parser.add_argument("--artifact_dir", default="artifact1")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    cfg = TransformerPipelineConfig.build(
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    if args.fresh:
        fresh_artifacts(cfg.artifact_dir)
    ensure_dirs(cfg.artifact_dir)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    summary = run_pipeline(cfg, only=only)
    out = write_run_summary(summary, cfg)
    logging.getLogger(__name__).info("Run summary written: %s", out)


if __name__ == "__main__":
    main()
