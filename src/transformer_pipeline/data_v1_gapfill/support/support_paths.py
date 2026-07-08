from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Cannot locate project root")


ROOT = project_root()
RUN_TAG = "v116_e31"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


OUT_ROOT = _env_path("ECG_V116_ARTIFACTS_DIR", ROOT / "outputs" / "transformer" / RUN_TAG)
REPORT_ROOT = OUT_ROOT / "reports"
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
SCRIPT_DIR = Path(__file__).resolve().parent
