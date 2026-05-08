from __future__ import annotations

import importlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.transformer_pipeline.config import TransformerPipelineConfig

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


def fresh_artifacts(artifact_dir: Path) -> None:
    targets = ["ptbxl", "manifests", "segments", "splits", "datasets", "models", "figs_for_noise_plot"]
    for name in targets:
        path = artifact_dir / name
        if path.exists():
            logger.info("fresh: removing %s", _display_path(path))
            shutil.rmtree(path)


def ensure_dirs(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "config").mkdir(parents=True, exist_ok=True)


def run_pipeline(cfg: TransformerPipelineConfig, *, only: list[str] | None = None) -> dict[str, Any]:
    allowed = set(only) if only else None
    if allowed:
        unknown = sorted(allowed - set(STEP_NAMES))
        if unknown:
            raise ValueError(f"unknown step(s): {', '.join(unknown)}")

    summary: dict[str, Any] = {
        "seed": cfg.seed,
        "artifact_dir": _rel_path(cfg.artifact_dir, cfg.root),
        "dry_run": cfg.dry_run,
        "steps": [],
    }
    params = cfg.base_params()

    for spec in STEPS:
        if allowed is not None and spec.name not in allowed:
            continue
        logger.info("STEP: %s", spec.name)
        out = _run_step(spec, params)
        outputs = out["outputs"] if "outputs" in out else [str(cfg.artifact_dir / rel) for rel in spec.outputs]
        summary["steps"].append(
            {
                "name": spec.name,
                "module": spec.module,
                "skipped": bool(out.get("skipped", False)),
                "outputs": [_rel_path(Path(str(path)), cfg.root) for path in outputs],
                "meta": {k: v for k, v in out.items() if k not in {"outputs"}},
            }
        )

    return summary


def write_run_summary(summary: dict[str, Any], cfg: TransformerPipelineConfig) -> Path:
    out = cfg.artifact_dir / "config" / f"transformer_run_summary_seed{cfg.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _run_step(spec: StepSpec, params: dict[str, Any]) -> dict[str, Any]:
    module = importlib.import_module(spec.module)
    fn = getattr(module, spec.func, None)
    if callable(fn):
        out = fn(params)
    else:
        main = getattr(module, "main", None)
        if not callable(main):
            raise AttributeError(f"{spec.module} has no callable run() or main()")
        main()
        out = {"step": spec.name, "skipped": False}
    if not isinstance(out, dict):
        raise TypeError(f"{spec.name}: expected dict output, got {type(out)}")
    return out


def _rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _display_path(path: Path) -> str:
    return _rel_path(path, Path.cwd())
