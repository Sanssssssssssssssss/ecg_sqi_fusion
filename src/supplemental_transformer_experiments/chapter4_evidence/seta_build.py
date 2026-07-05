from __future__ import annotations

from typing import Any

from src.supplemental_transformer_experiments.sqi12_gapfill import run as sqi12

from .common import Paths, dry, ensure_dirs


def run(paths: Paths, *, execute: bool, force: bool, seed: int = 0, max_ptb: int = 2400) -> dict[str, Any]:
    if not execute:
        dry("seta-build", paths)
        return {"step": "seta-build", "skipped": True}
    ensure_dirs(paths)
    sp = sqi12.Paths(paths.seta)
    sqi12.cmd_manifest(sp, run=True, force=force)
    sqi12.cmd_split(sp, run=True, force=force)
    sqi12.cmd_build(sp, run=True, force=force, seed=seed, max_ptb=max_ptb)
    sqi12.cmd_audit(sp)
    return {"step": "seta-build", "skipped": False, "out": str(paths.seta)}

