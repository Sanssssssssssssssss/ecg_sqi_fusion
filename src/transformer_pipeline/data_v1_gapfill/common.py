from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.utils.paths import project_root


ROOT = project_root()
RUN_TAG = "v116_e31"
POLICY = "v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876"
SPLIT_ALIAS = "v116gap_smc_k1_s20260876"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


ARTIFACTS = _env_path("ECG_V116_ARTIFACTS_DIR", ROOT / "outputs" / "transformer" / RUN_TAG)
ANALYSIS = ARTIFACTS / "analysis" / "good_medium_geometry_repair"
REPORT_ANALYSIS = ARTIFACTS / "reports" / "analysis" / "good_medium_geometry_repair"
SUPPORT = Path(__file__).resolve().parent / "support"


def py() -> str:
    exe = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(exe if exe.exists() else Path(sys.executable))


def protocol_dir() -> Path:
    return ANALYSIS / "clean_but_protocols" / POLICY


def split_dir() -> Path:
    return ANALYSIS / "event_factorized_sqi_conformer" / "rh_splits" / SPLIT_ALIAS / "fold0"


def report_dir() -> Path:
    return REPORT_ANALYSIS / "v116_native_budget_repair" / "s20260876"


def unbuffer_python_command(cmd: list[str]) -> list[str]:
    if not cmd:
        return cmd
    exe = Path(str(cmd[0])).name.lower()
    if not exe.startswith("python"):
        return cmd
    rest = [str(part) for part in cmd[1:]]
    if rest[:1] == ["-u"]:
        return [str(cmd[0]), *rest]
    return [str(cmd[0]), "-u", *rest]


def run_or_print(cmd: list[str], *, run: bool) -> None:
    cmd = unbuffer_python_command([str(part) for part in cmd])
    print(subprocess.list2cmdline(cmd))
    if run:
        env = os.environ.copy()
        env.setdefault("PYTHONFAULTHANDLER", "1")
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)
