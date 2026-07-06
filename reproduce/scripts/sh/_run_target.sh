#!/usr/bin/env bash
set -euo pipefail

target="${1:?target required}"
shift

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"

venv="$root/reproduce/work/.venv"
py="$venv/bin/python"
if [ ! -x "$py" ] && [ -x "$venv/Scripts/python.exe" ]; then
  py="$venv/Scripts/python.exe"
fi
if [ ! -x "$py" ]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$venv"
  else
    python -m venv "$venv"
  fi
  py="$venv/bin/python"
  if [ ! -x "$py" ] && [ -x "$venv/Scripts/python.exe" ]; then
    py="$venv/Scripts/python.exe"
  fi
fi

marker="$venv/.ecg_sqi_fusion_installed"
if [ ! -f "$marker" ]; then
  if "$py" -c "import src, pandas, numpy" >/dev/null 2>&1; then
    date -Iseconds > "$marker"
  else
    "$py" -m pip install --disable-pip-version-check -e .
    date -Iseconds > "$marker"
  fi
fi
"$py" reproduce/run_reproduce.py --target "$target" "$@"
