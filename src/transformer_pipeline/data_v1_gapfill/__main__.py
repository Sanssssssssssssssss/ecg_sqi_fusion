from __future__ import annotations

import argparse

from . import audit, build, pipeline, plot, split, train_check


def main() -> None:
    parser = argparse.ArgumentParser(description="V116 data-v1 gap-fill pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--run", action="store_true")
    build_p.set_defaults(func=lambda args: build.main(run=args.run))

    split_p = sub.add_parser("split")
    split_p.add_argument("--run", action="store_true")
    split_p.set_defaults(func=lambda args: split.main(run=args.run))

    sub.add_parser("audit").set_defaults(func=lambda _args: audit.main())
    sub.add_parser("plot").set_defaults(func=lambda _args: plot.main())

    train_p = sub.add_parser("train-check")
    train_p.add_argument("--model", choices=["E4", "E24", "both"], default="both")
    train_p.add_argument("--run", action="store_true")
    train_p.set_defaults(func=lambda args: train_check.main(model=args.model, run=args.run))

    pipe_p = sub.add_parser("pipeline")
    pipe_p.add_argument("--run", action="store_true")
    pipe_p.add_argument("--train", choices=["none", "E4", "E24", "both"], default="none")
    pipe_p.set_defaults(func=lambda args: pipeline.main(run=args.run, train=args.train))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
