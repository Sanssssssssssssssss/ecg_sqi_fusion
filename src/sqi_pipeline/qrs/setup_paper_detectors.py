from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.utils.paths import project_root

from wfdb_qrs_kit.exceptions import DetectorNotFoundError
from wfdb_qrs_kit.install import detector_status, install_detectors
from wfdb_qrs_kit.install import find_executable


def _find_exe(name: str, executables: dict[str, str | None]) -> str | None:
    value = executables.get(name)
    if value and Path(value).exists():
        return str(Path(value).resolve())
    try:
        return str(find_executable(name, cache_dir=None))
    except DetectorNotFoundError:
        pass
    return shutil.which(name) or shutil.which(f"{name}.exe")


def _has_detector_pair(path: Path) -> bool:
    names = ["wqrs.exe", "eplimited.exe"] if os.name == "nt" else ["wqrs", "eplimited"]
    if all((path / name).is_file() for name in names):
        return True
    return (path / "wqrs").is_file() and (path / "eplimited").is_file()


def _candidate_bin_dirs(root: Path, out_dir: Path) -> list[Path]:
    env = [key for key in ["WFDB_QRS_KIT_BIN_DIR", "SQI_QRS_BIN_DIR"] if key]
    raw: list[Path] = []
    for key in env:
        value = os.environ.get(key)
        if value:
            raw.append(Path(value))
    raw.append(out_dir / "bin")
    for parent in [root, *root.parents]:
        raw.extend(
            [
                parent / "wfdb-qrs-kit" / "detector-cache" / "bin",
                parent / "wfdb-qrs-kit" / "detector-cache-wsl" / "bin",
            ]
        )
    raw.append(root / "outputs" / "sqi_paper_aligned" / "qrs" / "tools" / "bin")
    seen: set[str] = set()
    out: list[Path] = []
    for path in raw:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _auto_bin_dir(root: Path, out_dir: Path) -> Path | None:
    for path in _candidate_bin_dirs(root, out_dir):
        if path.is_dir() and _has_detector_pair(path):
            return path
    return None


def write_setup_note(out_dir: Path, manifest: dict[str, Any]) -> Path:
    note = out_dir / "paper_qrs_detector_setup.md"
    text = f"""# Paper QRS Detector Setup

This directory is the local tool cache for the paper-aligned SQI profile.

Detector management is delegated to `wfdb-qrs-kit`.

Expected executables:

- `wqrs` via `WFDB_QRS_KIT_WQRS_EXE`, `SQI_WQRS_EXE`, this cache, user cache, or PATH
- `eplimited` via `WFDB_QRS_KIT_EPLIMITED_EXE`, `SQI_EPLIMITED_EXE`, this cache, user cache, or PATH

Current detector status:

```json
{json.dumps(manifest.get('status', {}), indent=2)}
```

Install manifest:

```json
{json.dumps(manifest.get('install_manifest', {}), indent=2)}
```
"""
    note.write_text(text, encoding="utf-8")
    return note


def run(
    out_dir: Path,
    *,
    from_bin_dir: str | Path | None = None,
    download_sources: bool = True,
    compile_sources: bool = False,
    compiler: str = "auto",
    wfdb_include: str | Path | None = None,
    wfdb_lib: str | Path | None = None,
    require_executables: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = project_root()
    existing_bin_dir = out_dir / "bin"
    if _has_detector_pair(existing_bin_dir):
        auto_from_bin_dir = existing_bin_dir
        install_manifest_data: dict[str, Any] = {
            "skipped": True,
            "reason": "existing detector cache is complete",
            "bin_dir": str(existing_bin_dir),
        }
    else:
        auto_from_bin_dir = Path(from_bin_dir) if from_bin_dir else _auto_bin_dir(root, out_dir)
        install_manifest = install_detectors(
            cache_dir=out_dir,
            from_bin_dir=auto_from_bin_dir,
            download=download_sources,
            compile=compile_sources,
            compiler=compiler,
            wfdb_include=wfdb_include,
            wfdb_lib=wfdb_lib,
        )
        install_manifest_data = json.loads(install_manifest.to_json())
    status = detector_status(out_dir)
    manifest: dict[str, Any] = {
        "manager": "wfdb-qrs-kit",
        "cache_dir": str(out_dir),
        "auto_from_bin_dir": str(auto_from_bin_dir) if auto_from_bin_dir else None,
        "status": status,
        "install_manifest": install_manifest_data,
        "executables": {
            "wqrs": _find_exe("wqrs", status.get("executables", {})),
            "eplimited": _find_exe("eplimited", status.get("executables", {})),
        },
    }
    manifest_path = out_dir / "paper_qrs_detector_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    note_path = write_setup_note(out_dir, manifest)
    if require_executables and not all(manifest["executables"].values()):
        raise FileNotFoundError(
            "paper QRS detectors are not installed. Use --from-bin-dir with existing "
            "wqrs/eplimited binaries, or use --compile with a WFDB C development environment. "
            f"See {note_path}."
        )
    return {"manifest": str(manifest_path), "note": str(note_path), **manifest}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare paper QRS detectors through wfdb-qrs-kit.")
    p.add_argument("--out_dir", default="outputs/sqi_paper_aligned/qrs/tools")
    p.add_argument("--from-bin-dir", default=None)
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--compile", action="store_true", dest="compile_sources")
    p.add_argument("--compiler", choices=["auto", "gcc", "cl"], default="auto")
    p.add_argument("--wfdb-include", default=None)
    p.add_argument("--wfdb-lib", default=None)
    p.add_argument("--require", action="store_true", help="fail if wqrs/eplimited are not discoverable")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    result = run(
        out_dir,
        from_bin_dir=args.from_bin_dir,
        download_sources=not args.no_download,
        compile_sources=args.compile_sources,
        compiler=args.compiler,
        wfdb_include=args.wfdb_include,
        wfdb_lib=args.wfdb_lib,
        require_executables=args.require,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
