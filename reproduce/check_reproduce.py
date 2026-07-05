from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORK_DEFAULT = ROOT / "reproduce" / "work"
FROZEN = ROOT / "outputs" / "transformer" / "supplemental" / "chapter4_evidence_frozen_final"
OPTIONAL_ARCHIVE_PREFIXES = (
    "outputs/sqi_paper_aligned_ch4_rerun",
    "outputs/sqi_supplemental/ch4_rerun",
)
SKIPPED_OPTIONAL_REFERENCES: set[str] = set()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm_report_path(text: str) -> Path | None:
    raw = text.strip().strip("`'\".,;)")
    if any(ch.isspace() for ch in raw):
        return None
    raw = raw.replace("\\", "/")
    pos = raw.find("outputs/")
    if pos >= 0:
        rel = raw[pos:]
        if rel.startswith(OPTIONAL_ARCHIVE_PREFIXES):
            SKIPPED_OPTIONAL_REFERENCES.add(rel)
            return None
        return ROOT / rel
    if raw.startswith("reproduce/"):
        return ROOT / raw
    return None


def _readable_artifact(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_dir():
        return (any(path.iterdir()), "dir ok" if any(path.iterdir()) else "empty dir")
    if path.stat().st_size <= 0:
        return False, "empty file"
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return False, "csv has no header"
        elif suffix in {".md", ".txt", ".log"}:
            path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return False, f"unreadable: {exc}"
    return True, "ok"


def _markdown_paths() -> set[Path]:
    found: set[Path] = set()
    for md in sorted((FROZEN / "reports").glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        for match in re.findall(r"`([^`]+)`", text):
            path = _norm_report_path(match)
            if path is not None:
                found.add(path)
        for match in re.findall(r"(outputs[\\/][^\s`|)]+)", text):
            path = _norm_report_path(match)
            if path is not None:
                found.add(path)
    return found


def _known_artifacts() -> set[Path]:
    paths: set[Path] = set()
    for sub, patterns in {
        "reports": ("*.md", "*.json", "*.csv"),
        "tables": ("*.csv", "*.json"),
        "figures/source_data": ("*.csv", "*.json"),
    }.items():
        base = FROZEN / sub
        for pattern in patterns:
            paths.update(base.glob(pattern))
    fig_index = FROZEN / "reports" / "figure_index.json"
    if fig_index.exists():
        data = json.loads(fig_index.read_text(encoding="utf-8"))
        for value in data.values():
            path = _norm_report_path(str(value))
            if path is not None:
                paths.add(path)
    paths.update(_markdown_paths())
    return paths


def artifact_check(work: Path) -> dict[str, Any]:
    SKIPPED_OPTIONAL_REFERENCES.clear()
    results = []
    missing = []
    unreadable = []
    for path in sorted(_known_artifacts(), key=lambda p: _rel(p)):
        ok, note = _readable_artifact(path)
        row = {"path": _rel(path), "status": "ok" if ok else "fail", "note": note}
        results.append(row)
        if not ok and note == "missing":
            missing.append(row)
        elif not ok:
            unreadable.append(row)
    out = {
        "name": "artifact",
        "status": "pass" if not missing and not unreadable else "fail",
        "checked": len(results),
        "missing": missing,
        "unreadable": unreadable,
        "warnings": [
            f"skipped optional historical reference: {path}"
            for path in sorted(SKIPPED_OPTIONAL_REFERENCES)
        ],
        "results": results,
    }
    _write_json(work / "artifact_check.json", out)
    return out


def _run_command(cmd: list[str], work: Path, name: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    stdout = work / f"{name}_stdout.log"
    stderr = work / f"{name}_stderr.log"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=err, text=True)
    return {
        "cmd": subprocess.list2cmdline(cmd),
        "returncode": proc.returncode,
        "stdout": _rel(stdout),
        "stderr": _rel(stderr),
    }


def _git_status() -> str:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout


def _safe_rmtree(path: Path, work: Path) -> None:
    resolved = path.resolve()
    if not str(resolved).startswith(str(work.resolve())):
        raise RuntimeError(f"refusing to delete outside reproduce work: {path}")
    if path.exists():
        shutil.rmtree(path)


def chapter4_render(work: Path) -> dict[str, Any]:
    target = work / "chapter4_evidence_copy"
    _safe_rmtree(target, work)
    shutil.copytree(FROZEN, target)
    before = _git_status()
    commands = [
        _run_command(
            [
                sys.executable,
                "-m",
                "src.supplemental_transformer_experiments.chapter4_evidence.run",
                "--out",
                str(target.resolve()),
                "report",
                "--run",
            ],
            work,
            "chapter4_report",
        ),
        _run_command(
            [
                sys.executable,
                "-m",
                "src.supplemental_transformer_experiments.chapter4_evidence.run",
                "--out",
                str(target.resolve()),
                "audit-report",
                "--run",
            ],
            work,
            "chapter4_audit_report",
        ),
    ]
    after = _git_status()
    changed_main_tree = before != after
    failures = [c for c in commands if c["returncode"] != 0]
    out = {
        "name": "chapter4-render",
        "status": "pass" if not failures and not changed_main_tree else "fail",
        "copy_root": _rel(target),
        "commands": commands,
        "main_tree_changed": changed_main_tree,
    }
    _write_json(work / "chapter4_render.json", out)
    return out


def public_smoke(work: Path) -> dict[str, Any]:
    out_dir = work / "transformer_clean_smoke"
    env = os.environ.copy()
    env["ECG_DISABLE_LOCAL_ARCHIVE"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "src.transformer_pipeline.cli",
        "--artifacts-dir",
        str(out_dir.resolve()),
        "clean-smoke",
        "--run",
    ]
    command = _run_command(cmd, work, "public_smoke", env=env)
    summary_path = out_dir / "source" / "clean_smoke_summary.json"
    summary: dict[str, Any] = {}
    warnings: list[str] = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("source", {}).get("historical_support_exact") is False:
            warnings.append("historical_support_exact=false")
    else:
        warnings.append("missing clean_smoke_summary.json")
    out = {
        "name": "public-smoke",
        "status": "pass" if command["returncode"] == 0 and summary_path.exists() else "fail",
        "command": command,
        "summary": _rel(summary_path),
        "warnings": warnings,
        "reproduction_scope": summary.get("reproduction_scope", ""),
    }
    _write_json(work / "public_smoke.json", out)
    return out


def write_summary(work: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    status = "pass" if all(r["status"] == "pass" for r in results) else "fail"
    summary = {"status": status, "results": results}
    _write_json(work / "reproduce_summary.json", summary)
    lines = ["# Reproduce Summary", "", f"status: `{status}`", ""]
    for result in results:
        lines.append(f"- `{result['name']}`: `{result['status']}`")
        for warning in result.get("warnings", []):
            lines.append(f"  - warning: {warning}")
    (work / "reproduce_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce readiness checks.")
    parser.add_argument("mode", choices=["artifact", "chapter4-render", "public-smoke", "all"])
    parser.add_argument("--work", default=str(WORK_DEFAULT))
    args = parser.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.mode == "artifact":
        results = [artifact_check(work)]
    elif args.mode == "chapter4-render":
        results = [chapter4_render(work)]
    elif args.mode == "public-smoke":
        results = [public_smoke(work)]
    else:
        results = [artifact_check(work), chapter4_render(work), public_smoke(work)]
    summary = write_summary(work, results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
