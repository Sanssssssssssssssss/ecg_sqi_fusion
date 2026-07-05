from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.data_downloads import ensure_sqi_raw_data, ensure_transformer_raw_data
from src.utils.paths import project_root

from .common import FROZEN_OUT, OUT_DEFAULT, Paths


def _paths(args: argparse.Namespace) -> Paths:
    out = Path(args.out)
    if not out.is_absolute():
        out = OUT_DEFAULT.parent / out if str(out) != str(OUT_DEFAULT) else OUT_DEFAULT
    return Paths(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 4 raw evidence experiments.")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--allow-frozen", action="store_true", help="allow writing to the frozen-final evidence directory")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ptb", type=int, default=2400)
    p.add_argument("--device", default="cuda")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in [
        "audit",
        "seta-build",
        "seta-sqi",
        "seta-repair",
        "seta-models",
        "but-models",
        "but-boundary-audit",
        "but-query-patching",
        "but-time-local-transplant",
        "but-architecture-ablation",
        "but-local-counterfactuals",
        "figures",
        "report",
        "audit-report",
        "pipeline",
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("--run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = _paths(args)
    if args.run and paths.out.resolve() == FROZEN_OUT.resolve() and not args.allow_frozen:
        raise SystemExit(f"refusing to write frozen evidence without --allow-frozen: {FROZEN_OUT}")
    if args.run:
        root = project_root()
        ensure_sqi_raw_data(root)
        ensure_transformer_raw_data(root)
    if args.cmd == "seta-build":
        from . import seta_build

        seta_build.run(paths, execute=args.run, force=args.force, seed=args.seed, max_ptb=args.max_ptb)
    elif args.cmd == "seta-sqi":
        from . import seta_sqi

        seta_sqi.run(paths, execute=args.run, force=args.force, seed=args.seed, max_ptb=args.max_ptb)
    elif args.cmd == "audit":
        from . import audit

        audit.run(paths, execute=args.run)
    elif args.cmd == "seta-repair":
        from . import repair

        repair.run(paths, execute=args.run, force=args.force)
    elif args.cmd == "seta-models":
        from . import seta_models

        seta_models.run(paths, execute=args.run, force=args.force, device=args.device)
    elif args.cmd == "but-models":
        from . import but_models

        but_models.run(paths, execute=args.run, force=args.force, device=args.device)
    elif args.cmd == "but-boundary-audit":
        from . import but_boundary_audit

        but_boundary_audit.run(paths, execute=args.run, force=args.force, device=args.device)
    elif args.cmd == "but-query-patching":
        from . import but_query_patching

        but_query_patching.run(paths, execute=args.run, force=args.force, device=args.device)
    elif args.cmd == "but-time-local-transplant":
        from . import but_time_local_transplant

        but_time_local_transplant.run(paths, execute=args.run, force=args.force, device=args.device)
    elif args.cmd == "but-architecture-ablation":
        from . import but_architecture_ablation

        but_architecture_ablation.run(paths, execute=args.run, force=args.force, device=args.device)
    elif args.cmd == "but-local-counterfactuals":
        from . import but_local_counterfactuals

        but_local_counterfactuals.run(paths, execute=args.run, force=args.force, device=args.device)
    elif args.cmd == "figures":
        from . import figures

        figures.run(paths, execute=args.run)
    elif args.cmd == "report":
        from . import report

        report.run(paths, execute=args.run)
    elif args.cmd == "audit-report":
        from . import audit_report

        audit_report.run(paths, execute=args.run)
    elif args.cmd == "pipeline":
        if not args.run:
            from .common import dry

            dry("pipeline", paths)
            return
        from . import audit, audit_report, but_boundary_audit, but_models, but_query_patching, figures, repair, report, seta_build, seta_models, seta_sqi

        seta_build.run(paths, execute=True, force=args.force, seed=args.seed, max_ptb=args.max_ptb)
        seta_sqi.run(paths, execute=True, force=args.force, seed=args.seed, max_ptb=args.max_ptb)
        audit.run(paths, execute=True)
        repair.run(paths, execute=True, force=args.force)
        seta_models.run(paths, execute=True, force=args.force, device=args.device)
        but_models.run(paths, execute=True, force=args.force, device=args.device)
        but_boundary_audit.run(paths, execute=True, force=args.force, device=args.device)
        but_query_patching.run(paths, execute=True, force=args.force, device=args.device)
        figures.run(paths, execute=True)
        report.run(paths, execute=True)
        audit_report.run(paths, execute=True)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
