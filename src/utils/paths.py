# src/utils/paths.py
from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    """Return the repository root containing ``pyproject.toml``.

    Returns:
        Absolute repository-root path.

    Raises:
        RuntimeError: If no project marker is found above this module.
    """

    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate project root (missing pyproject.toml or .git)")
