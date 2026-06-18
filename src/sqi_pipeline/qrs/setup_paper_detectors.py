from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from src.utils.paths import project_root


WFDB_WQRS_URL = "https://physionet.org/files/wfdb/10.7.0/app/wqrs.c"
ECGKIT_BASE = "https://physionet.org/files/ecgkit/1.0/common/eplimited/src"
EPLIMITED_FILES = [
    "eplimited.c",
    "qrsdet.h",
    "qrsdet.c",
    "qrsdet2.c",
    "qrsfilt.c",
    "bdac.c",
    "bdac.h",
    "analbeat.c",
    "analbeat.h",
    "match.c",
    "match.h",
    "rythmchk.c",
    "rythmchk.h",
    "classify.c",
    "postclas.c",
    "postclas.h",
    "noisechk.c",
    "inputs.h",
]


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)


def _find_exe(name: str, bin_dir: Path) -> str | None:
    candidates = [
        bin_dir / f"{name}.exe",
        bin_dir / name,
        shutil.which(name),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(str(c))
        if p.exists() and p.is_file():
            q = p if p.is_absolute() else Path.cwd() / p
            return str(q)
    return None


def write_setup_note(out_dir: Path, manifest: dict) -> Path:
    note = out_dir / "paper_qrs_detector_setup.md"
    text = f"""# Paper QRS Detector Setup

This directory is the local tool cache for the paper-aligned SQI profile.

Expected executables:

- `{out_dir / 'bin' / 'wqrs.exe'}` or environment variable `SQI_WQRS_EXE`
- `{out_dir / 'bin' / 'eplimited.exe'}` or environment variable `SQI_EPLIMITED_EXE`

Important source facts:

- `wqrs.c` is the WFDB C application used by the paper's `wqrs` detector and must be linked with the WFDB C library.
- `eplimited.c` and its companion EP Limited files also use the WFDB C library for record I/O and annotation output.
- The Python pipeline writes temporary WFDB records, calls these executables per lead, then reads `.wqrs` and `.epl` annotations back through `wfdb.rdann`.

Current executable discovery:

```json
{json.dumps(manifest.get('executables', {}), indent=2)}
```

Downloaded source manifest:

```json
{json.dumps(manifest.get('sources', {}), indent=2)}
```
"""
    note.write_text(text, encoding="utf-8")
    return note


def _ensure_wfdb_loader_alias(bin_dir: Path) -> None:
    dll = bin_dir / "wfdb-10.5.dll"
    alias = bin_dir / "wfdb-10.5"
    if dll.exists() and dll.is_file() and not alias.exists():
        shutil.copy2(dll, alias)


def run(out_dir: Path, *, download_sources: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = out_dir / "src"
    bin_dir = out_dir / "bin"
    include_dir = out_dir / "include"
    bin_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "mem.h").write_text("#pragma once\n#include <memory.h>\n#include <string.h>\n", encoding="ascii")
    _ensure_wfdb_loader_alias(bin_dir)

    sources: dict[str, str] = {}
    if download_sources:
        wqrs_out = src_dir / "wfdb" / "app" / "wqrs.c"
        _download(WFDB_WQRS_URL, wqrs_out)
        sources["wqrs.c"] = str(wqrs_out)

        for name in EPLIMITED_FILES:
            out = src_dir / "ecgkit" / "eplimited" / name
            try:
                _download(f"{ECGKIT_BASE}/{name}", out)
                sources[name] = str(out)
            except Exception as exc:
                sources[name] = f"download_failed: {type(exc).__name__}: {exc}"

    manifest = {
        "sources": sources,
        "source_urls": {
            "wqrs.c": WFDB_WQRS_URL,
            "eplimited_base": ECGKIT_BASE,
        },
        "executables": {
            "wqrs": _find_exe("wqrs", bin_dir),
            "eplimited": _find_exe("eplimited", bin_dir),
            "bin_dir": str(bin_dir),
            "env": {
                "SQI_WQRS_EXE": "absolute path to wqrs executable",
                "SQI_EPLIMITED_EXE": "absolute path to eplimited executable",
            },
        },
    }
    manifest_path = out_dir / "paper_qrs_detector_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    note_path = write_setup_note(out_dir, manifest)
    return {"manifest": str(manifest_path), "note": str(note_path), **manifest}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare local source cache for paper QRS detector executables.")
    p.add_argument("--out_dir", default="outputs/sqi_paper_aligned/qrs/tools")
    p.add_argument("--no_download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    result = run(out_dir, download_sources=not args.no_download)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
