# src/run_all.py
from __future__ import annotations

import argparse
import os
import importlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import project_root

logger = logging.getLogger("run_all")

# ---------- Pretty step header ----------
try:
    # Windows terminals work best with colorama
    from colorama import init as _cinit, Fore, Style
    _cinit()
    _HAS_COLOR = True
except Exception:
    _HAS_COLOR = False
    Fore = Style = None  # type: ignore


def print_step_header(step_name: str) -> None:
    line = "=" * 90
    title = f" STEP: {step_name} "
    # center title in the line
    pad = max(0, (len(line) - len(title)) // 2)
    header = f"{'=' * pad}{title}{'=' * (len(line) - pad - len(title))}"

    if _HAS_COLOR:
        # cyan header, bold-ish style
        msg = f"\n{Fore.CYAN}{line}\n{header}\n{line}{Style.RESET_ALL}\n"
    else:
        msg = f"\n{line}\n{header}\n{line}\n"

    # Logged using logger (maintaining consistency with log files/summary)
    logger.info(msg.rstrip("\n"))



# ===========================
# User-editable params override
# 你想换数据集/路径，就改这里（平时不用动）
# ===========================
PARAMS_OVERRIDE: dict[str, Any] = {
    # Example:
    # "challenge_root": r"D:\datasets\physionet\challenge-2011",
    # "nstdb_root": r"D:\datasets\physionet\nstdb",
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def fresh_artifacts(artifacts_dir: str | Path = "artifacts") -> None:
    root = Path(artifacts_dir)
    if not root.exists():
        logger.info("artifacts dir not found: %s (skip fresh)", root)
        return

    targets = [
        "manifests",
        "splits",
        "cases_500",
        "cases_125",
        "resampled_125",  # legacy
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


def ensure_dirs(artifacts_dir: str | Path = "artifacts") -> None:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)


def dump_run_summary(summary: dict, artifacts_dir: str | Path = "artifacts", seed: int = 0) -> Path:
    out = Path(artifacts_dir) / "config" / f"run_summary_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out


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

STEPS: list[StepSpec] = [
    StepSpec("manifest_raw", "src.data.make_manifest_raw", enabled=True),
    StepSpec("split_seta", "src.data.make_split_seta", enabled=True),
    StepSpec("balanced_noise", "src.noise.make_balanced_noisy_cases", enabled=True),
    StepSpec("resample_125", "src.preprocess.resample_125", enabled=True),
    StepSpec("qrs_cache", "src.qrs.run_qrs_cache", enabled=True),
    StepSpec("record84", "src.features.make_record84", enabled=True),
    StepSpec("norm_record84_ks", "src.features.norm_record84_ks", enabled=True),
    StepSpec("lm_mlp_search", "src.models.lm_mlp_search", enabled=True),
    StepSpec("svm_tables", "src.models.svm_tables", enabled=True),
    StepSpec("logreg_baseline", "src.models.logreg_baseline", enabled=True),
    StepSpec("gnb_baseline", "src.models.gnb_baseline", enabled=True),
]


def build_default_params(*, verbose: bool, force: bool, artifacts_dir: str | Path = "artifacts") -> dict[str, Any]:
    """
    ✅ By default, “just run it and it works”:
    Automatically derives data paths from project_root()
    To switch datasets: modify PARAMS_OVERRIDE
    """
    root = project_root()
    artifacts_dir = Path(artifacts_dir)

    params: dict[str, Any] = {
        "seed": 0,
        "verbose": verbose,
        "force": force,
        "artifacts_dir": str(artifacts_dir),

        # dataset roots (default)
        "challenge_root": str(root / "data" / "physionet" / "challenge-2011"),
        "nstdb_root": str(root / "data" / "physionet" / "nstdb"),

        # global constants you already fixed in project
        "leads": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
        "fs": {"raw": 500, "noise": 360, "work": 125},
        "half_policy": {"train": "first", "val": "first", "test": "second"},
        "beat_match_tol_ms": 150,
        "snr_db": -6.0,
    }

    params["steps"] = {
        "lm_mlp_search": {
            "tables": True,
            "tables_mode": "fixedJ",    # "searchJ"
            "J_fixed": 12,              # fixedJ mode
        },
        # "svm_tables": {...}
    }


    # Apply user override (lightweight “parameterization” without CLI pain)
    params.update(PARAMS_OVERRIDE)
    return params


def run_pipeline(params: dict[str, Any], *, only: list[str] | None = None) -> dict[str, Any]:
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

        print_step_header(spec.name)
        fn = load_step_callable(spec)

        step_params = dict(params)  # shallow copy
        step_over = (params.get("steps") or {}).get(spec.name, {})
        if isinstance(step_over, dict):
            step_params.update(step_over)

        out = fn(step_params)

        if not isinstance(out, dict):
            raise TypeError(f"{spec.name}: run() must return dict, got {type(out)}")

        summary["steps"].append(
            {
                "name": spec.name,
                "module": spec.module,
                "skipped": bool(out.get("skipped", False)),
                "outputs": out.get("outputs", []),
                "meta": {k: v for k, v in out.items() if k not in {"outputs"}},
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECG SQI Fusion - run all steps (simple)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fresh", action="store_true", help="delete generated artifacts and rerun")
    p.add_argument("--force", action="store_true", help="force rerun each step (step.run should respect it)")
    p.add_argument("--only", default="", help="comma-separated step names, e.g. manifest_raw,split_seta")
    p.add_argument("--artifacts_dir", default="artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.fresh:
        fresh_artifacts(args.artifacts_dir)
    ensure_dirs(args.artifacts_dir)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

    params = build_default_params(
        verbose=args.verbose,
        force=args.force,
        artifacts_dir=args.artifacts_dir,
    )

    summary = run_pipeline(params, only=only)
    out = dump_run_summary(summary, artifacts_dir=args.artifacts_dir, seed=int(params.get("seed", 0)))
    logger.info("Run summary written: %s", out)


if __name__ == "__main__":
    main()
