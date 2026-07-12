from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sqi_pipeline.config import SQIPipelineConfig
from src.sqi_pipeline.runner import ensure_dirs, format_summary_table, fresh_artifacts, run_pipeline, write_run_summary
from src.utils.data_downloads import ensure_sqi_raw_data


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the classical SQI pipeline.

    Returns:
        Parsed profile, output, seed, stage-selection, and logging options.
    """

    parser = argparse.ArgumentParser(description="Run the classical ECG SQI pipeline.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="delete generated SQI artifacts and rerun")
    parser.add_argument("--force", action="store_true", help="force each step to rerun")
    parser.add_argument("--only", default="", help="comma-separated step names, e.g. manifest_raw,record84")
    parser.add_argument("--artifacts_dir", default=None)
    parser.add_argument("--profile", choices=["baseline", "paper_aligned"], default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure line-buffered project logging.

    Args:
        verbose: Use debug level when true, otherwise information level.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message="'penalty' was deprecated.*",
        category=FutureWarning,
        module="sklearn.linear_model._logistic",
    )


def main() -> None:
    """Resolve data and configuration, execute stages, and print a summary."""

    args = parse_args()
    setup_logging(args.verbose)
    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        artifacts_dir = "outputs/sqi_paper_aligned" if args.profile == "paper_aligned" else "outputs/sqi"

    cfg = SQIPipelineConfig.build(
        artifacts_dir=artifacts_dir,
        profile=args.profile,
        seed=args.seed,
        verbose=args.verbose,
        force=args.force,
    )

    if args.fresh:
        fresh_artifacts(cfg.artifacts_dir)
    ensure_sqi_raw_data(cfg.root)
    ensure_dirs(cfg.artifacts_dir)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    summary = run_pipeline(cfg, only=only)
    out = write_run_summary(summary, cfg)
    logging.getLogger(__name__).info("Run summary written: %s", out)
    print(format_summary_table(summary, title="SQI pipeline summary"))


if __name__ == "__main__":
    main()
