from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "transformer" / "supplemental"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight supplemental transformer diagnostics.")
    parser.add_argument("task", choices=["audit", "plot", "train-dry"], default="audit", nargs="?")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.task == "audit":
        cmd = [sys.executable, "-m", "src.transformer_pipeline.cli", "audit"]
    elif args.task == "plot":
        cmd = [sys.executable, "-m", "src.transformer_pipeline.cli", "plot"]
    else:
        cmd = [sys.executable, "-m", "src.transformer_pipeline.cli", "train", "--model", "E31"]
    print(subprocess.list2cmdline(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
