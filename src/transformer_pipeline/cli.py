from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_pipeline.config import TransformerPipelineConfig
from src.transformer_pipeline import runner
from src.transformer_pipeline.data_v1_gapfill import audit, build, plot, report, split, train_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v116/E31 Transformer SQI pipeline.")
    parser.add_argument("--artifacts-dir", default="outputs/transformer/v116_e31")
    parser.add_argument("--seed", type=int, default=20260876)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit")
    sub.add_parser("plot")
    sub.add_parser("report")

    extract = sub.add_parser("extract-but")
    extract.add_argument("--run", action="store_true")

    build_p = sub.add_parser("build-v116")
    build_p.add_argument("--run", action="store_true")

    split_p = sub.add_parser("split")
    split_p.add_argument("--run", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--model", choices=["E4", "E24", "E31", "both", "all"], default="E31")
    train.add_argument("--run", action="store_true")

    pipe = sub.add_parser("pipeline")
    pipe.add_argument("--run", action="store_true")
    pipe.add_argument("--train", choices=["none", "E4", "E24", "E31", "both", "all"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    cfg = TransformerPipelineConfig.build(
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        force=args.force,
        verbose=args.verbose,
    )

    if args.cmd == "extract-but":
        runner.ensure_dirs(cfg)
        runner.run_extract_but(cfg, run=args.run)
    elif args.cmd == "build-v116":
        build.main(run=args.run)
    elif args.cmd == "split":
        split.main(run=args.run)
    elif args.cmd == "audit":
        audit.main()
    elif args.cmd == "plot":
        plot.main()
    elif args.cmd == "report":
        report.main()
    elif args.cmd == "train":
        train_check.main(model=args.model, run=args.run)
    elif args.cmd == "pipeline":
        runner.run_pipeline(cfg, run=args.run, train=args.train)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
