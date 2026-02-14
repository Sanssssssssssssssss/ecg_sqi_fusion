# run_all.py
from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("run_all")


# ---------------------------
# Utilities
# ---------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def fresh_artifacts(artifacts_dir: str = "artifacts") -> None:
    """
    Remove generated artifact subfolders (safe-by-default).
    Adjust the list if your repo uses different names.
    """
    root = Path(artifacts_dir)
    if not root.exists():
        logger.info("artifacts dir not found: %s (skip fresh)", root)
        return

    # whitelist of generated subfolders
    targets = [
        "manifests",
        "splits",
        "cases_500",
        "resampled_125",  # legacy name if still exists
        "qrs",
        "features",
        "models",
        "eval",
    ]

    for name in targets:
        p = root / name
        if p.exists():
            logger.info("fresh: removing %s", p)
            shutil.rmtree(p)


def ensure_dirs(artifacts_dir: str = "artifacts") -> None:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)


def dump_run_summary(summary: dict, artifacts_dir: str = "artifacts", seed: int = 0) -> Path:
    out = Path(artifacts_dir) / "config" / f"run_summary_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out


# ---------------------------
# Step registry
# ---------------------------
@dataclass(frozen=True)
class StepSpec:
    name: str
    module: str
    func: str = "run"
    enabled: bool = True


def load_step_callable(spec: StepSpec) -> Callable[..., dict]:
    mod = importlib.import_module(spec.module)
    fn = getattr(mod, spec.func, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{spec.module}.{spec.func} not found or not callable")
    return fn


# ⚠️ 这里你按“现有脚本的真实 module 路径”填（先填你已经改完的那几个）
# 例如你的 split 文件在 src/data/make_split_seta.py，
# 那么 module 就是 "src.data.make_split_seta"（前提是 src 是 package，含 __init__.py）
STEPS: list[StepSpec] = [
    StepSpec("manifest_raw", "src.data.make_manifest_raw", enabled=True),
    # StepSpec("split", "src.data.make_split_seta", enabled=False),
    # StepSpec("balanced_noise", "src.data.make_balanced_noisy_cases", enabled=False),
    # StepSpec("resample_125", "src.data.resample_125", enabled=False),
    # StepSpec("qrs_cache", "src.data.qrs_cache", enabled=False),
    # StepSpec("record84", "src.features.make_record84", enabled=False),
    # StepSpec("norm_ks", "src.features.norm_record84_ks", enabled=False),
    # StepSpec("lm_mlp", "src.models.lm_mlp", enabled=False),
]


# ---------------------------
# Pipeline
# ---------------------------
def run_pipeline(params: dict[str, Any], *, only: list[str] | None = None) -> dict[str, Any]:
    """
    params: light-weight global dict, passed into each step's run()
    only: optional allowlist of step names to run
    """
    summary: dict[str, Any] = {
        "seed": params.get("seed"),
        "artifacts_dir": params.get("artifacts_dir", "artifacts"),
        "steps": [],
    }

    allow = set(only) if only else None

    for spec in STEPS:
        if not spec.enabled:
            continue
        if allow is not None and spec.name not in allow:
            logger.info("skip step (not in --only): %s", spec.name)
            continue

        logger.info("==== STEP: %s ====", spec.name)
        fn = load_step_callable(spec)

        # 约定：每个 step 的 run(params) 返回 dict，至少包含:
        # {"outputs": [...], "skipped": bool}
        out = fn(params)
        if not isinstance(out, dict):
            raise TypeError(f"{spec.name}: run() must return dict, got {type(out)}")

        summary["steps"].append(
            {
                "name": spec.name,
                "module": spec.module,
                "skipped": bool(out.get("skipped", False)),
                "outputs_n": len(out.get("outputs", []) or []),
                "outputs": out.get("outputs", []),
                "meta": {k: v for k, v in out.items() if k not in {"outputs"}},
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECG SQI Fusion - run all pipeline steps (lightweight)")
    p.add_argument("--artifacts_dir", default="artifacts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fresh", action="store_true", help="delete generated artifacts and rerun")
    p.add_argument("--force", action="store_true", help="force rerun each step (step.run should respect it)")
    p.add_argument(
        "--only",
        default="",
        help="comma-separated step names to run, e.g. split,resample_125",
    )

    # 你项目里关键路径/超参可以先放在这里（后面再逐步挪到 params）
    p.add_argument("--seta_root", default="")
    p.add_argument("--nstdb_root", default="")
    p.add_argument("--beat_match_tol_ms", type=int, default=150)
    p.add_argument("--snr_db", type=float, default=-6.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.fresh:
        fresh_artifacts(args.artifacts_dir)
    ensure_dirs(args.artifacts_dir)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

    # 轻量 params：先够用就行，后续每个 step 需要什么再加
    params: dict[str, Any] = {
        "seed": args.seed,
        "artifacts_dir": args.artifacts_dir,
        "verbose": args.verbose,
        "force": args.force,
        "seta_root": args.seta_root,
        "nstdb_root": args.nstdb_root,
        "beat_match_tol_ms": args.beat_match_tol_ms,
        "snr_db": args.snr_db,
        "leads": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
        "fs": {"raw": 500, "noise": 360, "work": 125},
        "half_policy": {"train": "first", "val": "first", "test": "second"},
    }

    summary = run_pipeline(params, only=only)
    out = dump_run_summary(summary, artifacts_dir=args.artifacts_dir, seed=args.seed)
    logger.info("Run summary written: %s", out)


if __name__ == "__main__":
    main()
