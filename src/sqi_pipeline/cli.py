from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sqi_pipeline.config import SQIPipelineConfig
from src.sqi_pipeline.runner import ensure_dirs, fresh_artifacts, run_pipeline, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the classical ECG SQI pipeline.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="delete generated SQI artifacts and rerun")
    parser.add_argument("--force", action="store_true", help="force each step to rerun")
    parser.add_argument("--only", default="", help="comma-separated step names, e.g. manifest_raw,record84")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    cfg = SQIPipelineConfig.build(
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        verbose=args.verbose,
        force=args.force,
    )

    if args.fresh:
        fresh_artifacts(cfg.artifacts_dir)
    ensure_dirs(cfg.artifacts_dir)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    summary = run_pipeline(cfg, only=only)
    out = write_run_summary(summary, cfg)
    logging.getLogger(__name__).info("Run summary written: %s", out)


if __name__ == "__main__":
    main()
