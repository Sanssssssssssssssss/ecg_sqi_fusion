from __future__ import annotations

import logging
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.transformer_pipeline.config import TransformerPipelineConfig
from src.transformer_pipeline.data import but_source
from src.transformer_pipeline.data_v1_gapfill import audit, build, plot, report, split, train_check
from src.utils.data_downloads import ensure_transformer_raw_data
from src.utils.pipeline_summary import format_summary_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    name: str
    skipped: bool
    meta: dict[str, Any]
    outputs: tuple[Path, ...] = ()


def ensure_dirs(cfg: TransformerPipelineConfig) -> None:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "config").mkdir(parents=True, exist_ok=True)


def run_extract_but(cfg: TransformerPipelineConfig, *, run: bool) -> StepResult:
    if not run:
        print("python -m src.transformer_pipeline.cli extract-but --run")
        return StepResult("extract_but", True, {"dry_run": True})
    ensure_transformer_raw_data(cfg.root)
    out = but_source.run(cfg)
    return StepResult("extract_but", False, {"candidate_gap5": out["source"]["candidate_gap5_rows"]})


def run_clean_smoke(cfg: TransformerPipelineConfig, *, run: bool) -> StepResult:
    if not run:
        print("python -m src.transformer_pipeline.cli clean-smoke --run")
        return StepResult("clean_smoke", True, {"dry_run": True})
    ensure_transformer_raw_data(cfg.root)
    out = but_source.clean_smoke(cfg)
    print(json.dumps(out, indent=2, sort_keys=True))
    return StepResult(
        "clean_smoke",
        False,
        {
            "processed_shape": out["processed"].get("signals_shape"),
            "support_exact": out["source"].get("historical_support_exact"),
            "support_warning": out["source"].get("support_warning", ""),
        },
    )


def run_build_v116(*, run: bool) -> StepResult:
    build.main(run=run)
    return StepResult("build_v116", not run, {"dry_run": not run})


def run_split(*, run: bool) -> StepResult:
    split.main(run=run)
    return StepResult("split", not run, {"dry_run": not run})


def run_audit(*, run: bool) -> StepResult:
    if not run:
        print("python -m src.transformer_pipeline.cli audit")
        return StepResult("audit", True, {"dry_run": True})
    out = audit.payload()
    audit.validate(out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return StepResult("audit", False, {"rows": out["protocol_rows"], "train": out["train_class_counts"]})


def run_plot(*, run: bool) -> StepResult:
    if not run:
        print("python -m src.transformer_pipeline.cli plot")
        return StepResult("plot", True, {"dry_run": True})
    plot.main()
    return StepResult("plot", False, {})


def run_report(*, run: bool) -> StepResult:
    if not run:
        print("python -m src.transformer_pipeline.cli report")
        return StepResult("report", True, {"dry_run": True})
    report.main()
    return StepResult("report", False, {})


def run_train(model: str, *, run: bool) -> StepResult:
    train_check.main(model=model, run=run)
    return StepResult("train", not run, {"model": model, "dry_run": not run})


def run_pipeline(cfg: TransformerPipelineConfig, *, run: bool, train: str = "none") -> dict[str, Any]:
    ensure_dirs(cfg)
    steps: list[StepResult] = []
    for fn in (
        lambda: run_extract_but(cfg, run=run),
        lambda: run_build_v116(run=run),
        lambda: run_split(run=run),
        lambda: run_audit(run=run),
        lambda: run_plot(run=run),
        lambda: run_report(run=run),
    ):
        start = time.perf_counter()
        result = fn()
        result.meta["duration_sec"] = time.perf_counter() - start
        steps.append(result)

    if train != "none":
        start = time.perf_counter()
        result = run_train(train, run=run)
        result.meta["duration_sec"] = time.perf_counter() - start
        steps.append(result)

    summary = {
        "seed": cfg.seed,
        "artifacts_dir": _rel(cfg.artifacts_dir, cfg.root),
        "steps": [
            {
                "name": step.name,
                "skipped": step.skipped,
                "outputs": [_rel(p, cfg.root) for p in step.outputs],
                "meta": step.meta,
            }
            for step in steps
        ],
    }
    print(format_summary_table(summary, title="Transformer pipeline summary"))
    return summary


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
