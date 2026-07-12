#!/usr/bin/env bash
set -euo pipefail

target="${1:?target required}"
shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

if command -v python3 >/dev/null 2>&1; then
  py=(python3)
elif command -v python >/dev/null 2>&1; then
  py=(python)
elif command -v py >/dev/null 2>&1; then
  py=(py -3)
else
  echo "python, python3, or py not found" >&2
  exit 127
fi

"${py[@]}" run_reproduce.py --target "$target" "$@"
