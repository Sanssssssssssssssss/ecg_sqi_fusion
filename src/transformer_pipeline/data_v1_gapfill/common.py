from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.utils.paths import project_root


ROOT = project_root()
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
POLICY = "v116_gapfill_dual_goodorig_nm99_ms10_smc_s20260876"
SPLIT_ALIAS = "v116_gapfill_dual_goodorig_nm99__k1_s20260876"
ANALYSIS = ROOT / "outputs" / "external_benchmarks" / RUN_TAG / "analysis" / "good_medium_geometry_repair"
SUPPORT = Path(__file__).resolve().parent / "support"


def py() -> str:
    exe = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(exe if exe.exists() else Path(sys.executable))


def protocol_dir() -> Path:
    return ANALYSIS / "clean_but_protocols" / POLICY


def split_dir() -> Path:
    return ANALYSIS / "event_factorized_sqi_conformer" / "rh_splits" / SPLIT_ALIAS / "fold0"


def run_or_print(cmd: list[str], *, run: bool) -> None:
    print(subprocess.list2cmdline([str(part) for part in cmd]))
    if run:
        subprocess.run(cmd, cwd=ROOT, check=True)
