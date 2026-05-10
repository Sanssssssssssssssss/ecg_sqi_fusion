from __future__ import annotations

import importlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.transformer_pipeline.config import TransformerPipelineConfig
from src.utils.pipeline_summary import format_summary_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepSpec:
    name: str
    module: str
    outputs: tuple[str, ...]
    func: str = "run"


STEPS: tuple[StepSpec, ...] = (
    StepSpec("filter_lead_i", "src.transformer_pipeline.data.filter_lead_i", ("ptbxl/ptbxl_database_no_lead_I.csv",)),
    StepSpec("manifest", "src.transformer_pipeline.data.make_manifest_lead_i", ("manifests/ptbxl_leadI_manifest.csv",)),
    StepSpec(
        "segments",
        "src.transformer_pipeline.preprocess.make_segments_10s_125hz",
        ("segments/ptbxl_leadI_x_10s_125hz.npz", "segments/ptbxl_leadI_segments_10s_125hz.csv"),
    ),
    StepSpec("split", "src.transformer_pipeline.data.make_clean_split", ("splits/ptbxl_leadI_clean_10s_125hz_split.csv",)),
    StepSpec(
        "synthesize_noise",
        "src.transformer_pipeline.noise.synthesize_snr_dataset",
        (
            "datasets/synth_10s_125hz_clean.npz",
            "datasets/synth_10s_125hz_noisy.npz",
            "datasets/synth_10s_125hz_labels.csv",
        ),
    ),
    StepSpec(
        "noise_level",
        "src.transformer_pipeline.noise.make_rr_noise_level",
        ("datasets/synth_10s_125hz_noise_level.npz", "datasets/synth_10s_125hz_labels_with_level.csv"),
    ),
    StepSpec("forward_check", "src.transformer_pipeline.diagnostics.check_model_forward", ()),
    StepSpec(
        "train",
        "src.transformer_pipeline.train",
        (
            "models/mtl_transformer_seed0_step6/ckpt_last.pt",
            "models/mtl_transformer_seed0_step6/ckpt_best_val.pt",
            "models/mtl_transformer_seed0_step6/train_log.json",
            "models/mtl_transformer_seed0_step6/test_report.json",
        ),
    ),
    StepSpec(
        "evaluate",
        "src.transformer_pipeline.evaluate",
        (
            "models/mtl_transformer_seed0_step6/eval_best/test_report_best.json",
            "models/mtl_transformer_seed0_step6/eval_best/val_report_best.json",
        ),
    ),
)
STEP_NAMES = tuple(step.name for step in STEPS)
PREPROCESS_STEP_NAMES = (
    "filter_lead_i",
    "manifest",
    "segments",
    "split",
    "synthesize_noise",
    "noise_level",
)
MODEL_STEP_NAMES = (
    "forward_check",
    "train",
    "evaluate",
)
STAGE_STEPS = {
    "all": STEP_NAMES,
    "preprocess": PREPROCESS_STEP_NAMES,
    "model": MODEL_STEP_NAMES,
}


def fresh_artifacts(artifact_dir: Path, *, stage: str = "all") -> None:
    if stage == "preprocess":
        targets = ["ptbxl", "manifests", "segments", "splits", "datasets", "figs_for_noise_plot"]
    elif stage == "model":
        targets = ["models"]
    else:
        targets = ["ptbxl", "manifests", "segments", "splits", "datasets", "models", "figs_for_noise_plot"]
    for name in targets:
        path = artifact_dir / name
        if path.exists():
            logger.info("fresh: removing %s", _display_path(path))
            shutil.rmtree(path)


def ensure_dirs(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "config").mkdir(parents=True, exist_ok=True)


def run_pipeline(
    cfg: TransformerPipelineConfig,
    *,
    only: list[str] | None = None,
    stage: str = "all",
) -> dict[str, Any]:
    if stage not in STAGE_STEPS:
        raise ValueError(f"unknown stage: {stage}")
    stage_names = set(STAGE_STEPS[stage])
    allowed = set(only) if only else None
    if allowed:
        unknown = sorted(allowed - set(STEP_NAMES))
        if unknown:
            raise ValueError(f"unknown step(s): {', '.join(unknown)}")

    summary: dict[str, Any] = {
        "seed": cfg.seed,
        "artifact_dir": _rel_path(cfg.artifact_dir, cfg.root),
        "stage": stage,
        "dry_run": cfg.dry_run,
        "steps": [],
    }
    params = cfg.base_params()

    for spec in STEPS:
        if spec.name not in stage_names:
            continue
        if allowed is not None and spec.name not in allowed:
            continue
        _log_step(spec.name)
        fn = load_step_callable(spec)
        start = time.perf_counter()
        out = fn(params)
        duration_sec = time.perf_counter() - start
        if not isinstance(out, dict):
            raise TypeError(f"{spec.name}: expected dict output, got {type(out)}")
        outputs = out["outputs"] if "outputs" in out else [str(cfg.artifact_dir / rel) for rel in spec.outputs]
        meta = {k: v for k, v in out.items() if k not in {"outputs"}}
        meta["duration_sec"] = duration_sec
        summary["steps"].append(
            {
                "name": spec.name,
                "module": spec.module,
                "skipped": bool(out.get("skipped", False)),
                "outputs": [_rel_path(Path(str(path)), cfg.root) for path in outputs],
                "meta": meta,
            }
        )

    return summary


def write_run_summary(summary: dict[str, Any], cfg: TransformerPipelineConfig) -> Path:
    out = cfg.artifact_dir / "config" / f"transformer_run_summary_seed{cfg.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_step_callable(spec: StepSpec) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module = importlib.import_module(spec.module)
    fn = getattr(module, spec.func, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{spec.module}.{spec.func} not found or not callable")
    return fn


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
