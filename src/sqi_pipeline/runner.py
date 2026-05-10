from __future__ import annotations

import importlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.sqi_pipeline.config import SQIPipelineConfig
from src.utils.pipeline_summary import format_summary_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepSpec:
    name: str
    module: str
    func: str = "run"


STEPS: tuple[StepSpec, ...] = (
    StepSpec("manifest_raw", "src.sqi_pipeline.data.make_manifest_raw"),
    StepSpec("split_seta", "src.sqi_pipeline.data.make_split_seta"),
    StepSpec("balanced_noise", "src.sqi_pipeline.noise.make_balanced_noisy_cases"),
    StepSpec("resample_125", "src.sqi_pipeline.preprocess.resample_125"),
    StepSpec("qrs_cache", "src.sqi_pipeline.qrs.run_qrs_cache"),
    StepSpec("record84", "src.sqi_pipeline.features.make_record84"),
    StepSpec("norm_record84_ks", "src.sqi_pipeline.features.norm_record84_ks"),
    StepSpec("lm_mlp_search", "src.sqi_pipeline.models.lm_mlp_search"),
    StepSpec("svm_tables", "src.sqi_pipeline.models.svm_tables"),
    StepSpec("logreg_baseline", "src.sqi_pipeline.models.logreg_baseline"),
    StepSpec("gnb_baseline", "src.sqi_pipeline.models.gnb_baseline"),
)
STEP_NAMES = tuple(spec.name for spec in STEPS)


def fresh_artifacts(artifacts_dir: Path) -> None:
    targets = [
        "manifests",
        "splits",
        "cases_500",
        "resampled_125",
        "qrs",
        "features",
        "models",
        "eval",
        "qc",
        "report_figs",
        "paper_tables",
        "relabel_stats",
        "mitbih",
    ]
    for name in targets:
        path = artifacts_dir / name
        if path.exists():
            logger.info("fresh: removing %s", _display_path(path))
            shutil.rmtree(path)


def ensure_dirs(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "config").mkdir(parents=True, exist_ok=True)


def load_step_callable(spec: StepSpec) -> Callable[..., dict[str, Any]]:
    module = importlib.import_module(spec.module)
    fn = getattr(module, spec.func, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{spec.module}.{spec.func} not found or not callable")
    return fn


def step_params(cfg: SQIPipelineConfig, step_name: str) -> dict[str, Any]:
    art = cfg.artifacts_dir
    params = cfg.base_params()

    split_seed = art / "splits" / f"split_seta_seed{cfg.seed}.csv"
    split_balanced = art / "splits" / f"split_seta_seed{cfg.seed}_balanced.csv"
    features_dir = art / "features"
    record84 = features_dir / "record84.parquet"
    record84_norm = features_dir / "record84_norm.parquet"

    per_step: dict[str, dict[str, Any]] = {
        "split_seta": {
            "manifest_csv": str(art / "manifests" / "manifest_challenge2011_seta.csv"),
            "out_csv": str(split_seed),
            "qc_png": str(art / "qc" / f"split_seta_seed{cfg.seed}_label_counts.png"),
        },
        "balanced_noise": {
            "split_csv": str(split_seed),
            "out_split_csv": str(split_balanced),
            "set_a_dir": str(cfg.set_a_dir),
            "nstdb_dir": str(cfg.nstdb_root),
            "cases_500_dir": str(art / "cases_500"),
        },
        "resample_125": {
            "split_csv": str(split_balanced),
            "cases_500_dir": str(art / "cases_500"),
            "out_dir": str(art / "resampled_125"),
        },
        "qrs_cache": {
            "split_csv": str(split_balanced),
            "resampled_dir": str(art / "resampled_125"),
            "out_dir": str(art / "qrs"),
        },
        "record84": {
            "split_csv": str(split_balanced),
            "resampled_dir": str(art / "resampled_125"),
            "qrs_dir": str(art / "qrs"),
            "out_dir": str(features_dir),
        },
        "norm_record84_ks": {
            "split_csv": str(split_balanced),
            "in_parquet": str(record84),
            "out_dir": str(features_dir),
            "out_stats": str(features_dir / f"norm_stats_seed{cfg.seed}.json"),
            "out_parquet": str(record84_norm),
        },
        "lm_mlp_search": {
            "features_parquet": str(record84_norm),
            "split_csv": str(split_balanced),
            "out_dir": str(art / "models" / "lm_mlp"),
            "tables": True,
            "tables_mode": "fixedJ",
            "J_fixed": 12,
        },
        "svm_tables": {
            "features_parquet": str(record84_norm),
            "split_csv": str(split_balanced),
            "out_dir": str(art / "models" / "svm"),
        },
        "logreg_baseline": {
            "features_parquet": str(record84_norm),
            "split_csv": str(split_balanced),
            "out_dir": str(art / "models" / "baselines" / "logreg"),
        },
        "gnb_baseline": {
            "features_parquet": str(record84_norm),
            "split_csv": str(split_balanced),
            "out_dir": str(art / "models" / "baselines" / "gnb"),
        },
    }
    params.update(per_step.get(step_name, {}))
    return params


def run_pipeline(cfg: SQIPipelineConfig, *, only: list[str] | None = None) -> dict[str, Any]:
    allowed = set(only) if only else None
    if allowed:
        unknown = sorted(allowed - set(STEP_NAMES))
        if unknown:
            raise ValueError(f"unknown step(s): {', '.join(unknown)}")

    summary: dict[str, Any] = {
        "seed": cfg.seed,
        "artifacts_dir": _rel_path(cfg.artifacts_dir, cfg.root),
        "steps": [],
    }

    for spec in STEPS:
        if allowed is not None and spec.name not in allowed:
            continue

        _log_step(spec.name)
        fn = load_step_callable(spec)
        start = time.perf_counter()
        out = fn(step_params(cfg, spec.name))
        if not isinstance(out, dict):
            raise TypeError(f"{spec.name}: run() must return dict, got {type(out)}")
        duration_sec = time.perf_counter() - start
        meta = {k: v for k, v in out.items() if k not in {"outputs"}}
        meta["duration_sec"] = duration_sec

        summary["steps"].append(
            {
                "name": spec.name,
                "module": spec.module,
                "skipped": bool(out.get("skipped", False)),
                "outputs": [_rel_path(Path(str(p)), cfg.root) for p in out.get("outputs", [])],
                "meta": meta,
            }
        )

    return summary


def write_run_summary(summary: dict[str, Any], cfg: SQIPipelineConfig) -> Path:
    out = cfg.artifacts_dir / "config" / f"run_summary_seed{cfg.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _log_step(step_name: str) -> None:
    line = "=" * 90
    title = f" STEP: {step_name} "
    pad = max(0, (len(line) - len(title)) // 2)
    logger.info("\n%s\n%s%s%s\n%s", line, "=" * pad, title, "=" * (len(line) - pad - len(title)), line)


def _rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _display_path(path: Path) -> str:
    return _rel_path(path, Path.cwd())
