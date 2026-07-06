from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = ROOT / "reproduce"
WORK_DEFAULT = REPRODUCE / "work"
EXPECTED = REPRODUCE / "expected_outputs.json"
FROZEN = ROOT / "outputs" / "transformer" / "supplemental" / "chapter4_evidence_frozen_final"
V116_ROOT = ROOT / "outputs" / "transformer" / "v116_e31"
PY = Path(sys.executable)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    cmd: list[str]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_info(path: Path) -> tuple[str, int, list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return "csv_no_header", 0, []
            rows = sum(1 for _ in reader)
        return "ok", rows, header
    except Exception as exc:
        return f"unreadable:{exc}", 0, []


def _artifact_info(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "exists": path.exists(),
        "size": 0,
        "read_status": "missing",
        "row_count": "",
        "columns": "",
        "sha256": "",
    }
    if not path.exists():
        return row
    if path.is_dir():
        row.update({"size": "", "read_status": "dir_ok" if any(path.iterdir()) else "empty_dir"})
        return row
    size = path.stat().st_size
    row["size"] = int(size)
    if size <= 0:
        row["read_status"] = "empty"
        return row
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            status, n, cols = _read_csv_info(path)
            row.update({"read_status": status, "row_count": n, "columns": "|".join(cols)})
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            row["read_status"] = "ok"
        elif suffix == ".png":
            with path.open("rb") as f:
                row["read_status"] = "ok" if f.read(8) == b"\x89PNG\r\n\x1a\n" else "bad_png_signature"
        elif suffix in {".md", ".txt", ".log", ".svg"}:
            path.read_text(encoding="utf-8", errors="replace")
            row["read_status"] = "ok"
        else:
            row["read_status"] = "ok"
        row["sha256"] = _sha256(path)
    except Exception as exc:
        row["read_status"] = f"unreadable:{exc}"
    return row


def _float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def _numeric_diff(reference: Path, generated: Path, tolerance: float) -> str:
    if not (reference.exists() and generated.exists()) or reference.suffix.lower() != ".csv" or generated.suffix.lower() != ".csv":
        return "not_applicable"
    try:
        with reference.open("r", encoding="utf-8-sig", newline="") as a, generated.open("r", encoding="utf-8-sig", newline="") as b:
            ra = list(csv.DictReader(a))
            rb = list(csv.DictReader(b))
        if len(ra) != len(rb):
            return f"row_count_diff:{len(ra)}!={len(rb)}"
        if not ra:
            return "ok"
        cols = [c for c in ra[0].keys() if c in rb[0]]
        max_diff = 0.0
        seen = 0
        for left, right in zip(ra, rb):
            for col in cols:
                av = _float(left.get(col, ""))
                bv = _float(right.get(col, ""))
                if av is None or bv is None:
                    continue
                seen += 1
                max_diff = max(max_diff, abs(av - bv))
        if seen == 0:
            return "no_numeric_columns"
        return "ok" if max_diff <= tolerance else f"max_abs_diff={max_diff:.6g}"
    except Exception as exc:
        return f"diff_error:{exc}"


def _run(cmd: CommandSpec, run_dir: Path, env: dict[str, str], *, dry_run: bool, timeout: int | None) -> dict[str, Any]:
    stdout = run_dir / "logs" / f"{cmd.name}.stdout.log"
    stderr = run_dir / "logs" / f"{cmd.name}.stderr.log"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if dry_run:
        stdout.write_text(subprocess.list2cmdline(cmd.cmd) + "\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        code = 0
    else:
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            try:
                proc = subprocess.run(cmd.cmd, cwd=ROOT, env=env, stdout=out, stderr=err, text=True, timeout=timeout)
                code = int(proc.returncode)
            except subprocess.TimeoutExpired as exc:
                err.write(f"\nTIMEOUT after {timeout}s: {exc}\n")
                code = 124
    return {
        "name": cmd.name,
        "cmd": subprocess.list2cmdline([str(x) for x in cmd.cmd]),
        "returncode": code,
        "duration_sec": round(time.perf_counter() - started, 3),
        "stdout": _rel(stdout),
        "stderr": _rel(stderr),
    }


def _load_expected() -> dict[str, Any]:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def _target_paths(run_dir: Path, target: str) -> dict[str, Path]:
    paper = run_dir / "sqi_paper_aligned"
    if target == "conformer-cinc2011":
        paper = run_dir / "sqi12_gapfill" / "sqi_full_rerun_clean"
    return {
        "paper": paper,
        "paper_reports": run_dir / "reports" / "sqi_paper_aligned",
        "but_sqi": run_dir / "but_sqi_baseline",
        "v116": run_dir / "transformer_v116",
        "chapter4": run_dir / "chapter4_evidence",
        "sqi_supp": run_dir / "sqi_supplemental",
        "sqi_supp_reports": run_dir / "reports" / "sqi_supplemental",
    }


def _base_env(work: Path, paths: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ECG_DATA_ROOT"] = str((work / "data").resolve())
    env["ECG_TRANSFORMER_ARTIFACTS"] = str(paths["v116"].resolve())
    env["ECG_E31_PREDICTION_ROOT"] = str(V116_ROOT.resolve())
    env["ECG_SQI_PAPER_ALIGNED_ROOT"] = str(paths["paper"].resolve())
    env["ECG_SQI_SUPPLEMENTAL_ROOT"] = str(paths["sqi_supp"].resolve())
    return env


def _cmds(target: str, run_dir: Path, *, force: bool, device: str, gpu_train: bool) -> list[CommandSpec]:
    p = _target_paths(run_dir, target)
    force_arg = ["--force"] if force else []
    sqi = [
        str(PY),
        "-m",
        "src.sqi_pipeline.run_all",
        "--profile",
        "paper_aligned",
        "--fresh",
        "--artifacts_dir",
        str(p["paper"]),
        "--seed",
        "0",
        *force_arg,
    ]
    transformer = [
        str(PY),
        "-m",
        "src.transformer_pipeline.run_all",
        "--run",
        "--train",
        "E31" if gpu_train else "none",
        "--artifacts-dir",
        str(p["v116"]),
        "--seed",
        "20260876",
        *force_arg,
    ]
    c4 = [str(PY), "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", str(p["chapter4"]), "--device", device, *force_arg]

    if target == "baseline-cinc2011":
        return [
            CommandSpec("sqi_paper_aligned", sqi),
            CommandSpec(
                "sqi_table_trends",
                [
                    str(PY),
                    "-m",
                    "src.sqi_pipeline.diagnostics.compare_paper_tables",
                    "--artifacts_dir",
                    str(p["paper"]),
                    "--out_dir",
                    str(p["paper_reports"] / "table_trend_comparison"),
                    "--seed",
                    "0",
                ],
            ),
            CommandSpec(
                "sqi_paper_tables",
                [
                    str(PY),
                    "-m",
                    "src.sqi_pipeline.tools.make_paper_tables",
                    "--artifacts_dir",
                    str(p["paper"]),
                    "--out_dir",
                    str(p["paper"] / "paper_tables"),
                    "--seed",
                    "0",
                ],
            ),
        ]
    if target == "baseline-but":
        return [
            CommandSpec("transformer_v116_public_rebuild", transformer),
            CommandSpec(
                "but_sqi_baseline",
                [
                    str(PY),
                    "-m",
                    "src.supplemental_transformer_experiments.but_sqi_baseline.run",
                    "--out",
                    str(p["but_sqi"]),
                    "--device",
                    device,
                    *force_arg,
                    "pipeline",
                    "--run",
                ],
            ),
        ]
    if target == "conformer-cinc2011":
        return [
            CommandSpec("sqi_paper_aligned_for_seta_figures", sqi),
            CommandSpec("seta_build", [*c4, "seta-build", "--run"]),
            CommandSpec("seta_sqi", [*c4, "seta-sqi", "--run"]),
            CommandSpec("protocol_audit", [*c4, "audit", "--run"]),
            CommandSpec("seta_repair", [*c4, "seta-repair", "--run"]),
            CommandSpec("seta_models", [*c4, "seta-models", "--run"]),
            CommandSpec("seta_figures", [*c4, "figures", "--run"]),
        ]
    if target == "conformer-but":
        return [
            CommandSpec("transformer_v116_public_rebuild", transformer),
            CommandSpec("but_models", [*c4, "but-models", "--run"]),
            CommandSpec("but_boundary_audit", [*c4, "but-boundary-audit", "--run"]),
            CommandSpec("but_query_patching", [*c4, "but-query-patching", "--run"]),
            CommandSpec("but_figures", [*c4, "figures", "--run"]),
        ]
    if target == "sqi-supplemental":
        return [
            CommandSpec("sqi_paper_aligned", sqi),
            CommandSpec(
                "sqi_supplemental",
                [
                    str(PY),
                    "-m",
                    "src.supplemental_sqi_experiments.run",
                    "diagnose-existing",
                    "--artifacts-dir",
                    str(p["paper"]),
                    "--out-dir",
                    str(p["sqi_supp"]),
                    "--report-dir",
                    str(p["sqi_supp_reports"]),
                    "--seed",
                    "0",
                    *force_arg,
                ],
            ),
        ]
    if target == "transformer-supplemental":
        return [
            CommandSpec("transformer_v116_public_rebuild", transformer),
            CommandSpec("but_models", [*c4, "but-models", "--run"]),
            CommandSpec("but_boundary_audit", [*c4, "but-boundary-audit", "--run"]),
            CommandSpec("but_query_patching", [*c4, "but-query-patching", "--run"]),
            CommandSpec("but_time_local_transplant", [*c4, "but-time-local-transplant", "--run"]),
            CommandSpec("but_architecture_ablation", [*c4, "but-architecture-ablation", "--run"]),
            CommandSpec("but_local_counterfactuals", [*c4, "but-local-counterfactuals", "--run"]),
            CommandSpec("transformer_supplemental_figures", [*c4, "figures", "--run"]),
        ]
    raise SystemExit(f"unknown target: {target}")


def _reference_rows(target: str, run_dir: Path, expected: dict[str, Any]) -> list[dict[str, str]]:
    spec = expected["targets"][target]
    rows: list[dict[str, str]] = []
    for item in spec.get("required", []):
        rows.append(
            {
                "id": item["id"],
                "report_section": item.get("report_section", ""),
                "table_or_figure": item.get("table_or_figure", item["id"]),
                "reference_path": item.get("reference", ""),
                "generated_path": item["generated"],
                "required": "1",
            }
        )
    for glob_spec in spec.get("reference_globs", []):
        base = ROOT / glob_spec["base"]
        for path in sorted(base.glob(glob_spec["pattern"])):
            if path.is_dir():
                continue
            rel = path.relative_to(base).as_posix()
            rows.append(
                {
                    "id": f"{glob_spec['id_prefix']}:{rel}",
                    "report_section": glob_spec.get("report_section", ""),
                    "table_or_figure": rel,
                    "reference_path": _rel(path),
                    "generated_path": f"{glob_spec['generated_base'].rstrip('/')}/{rel}",
                    "required": "1" if glob_spec.get("required", True) else "0",
                }
            )
    return rows


def _audit(target: str, run_dir: Path, expected: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = expected["targets"][target]
    tol = float(expected.get("numeric_tolerance", 0.02))
    p = _target_paths(run_dir, target)
    matrix: list[dict[str, Any]] = []
    checksums: list[dict[str, Any]] = []
    for row in _reference_rows(target, run_dir, expected):
        gen = run_dir / row["generated_path"]
        ref = ROOT / row["reference_path"] if row["reference_path"] else Path()
        info = _artifact_info(gen)
        ref_info = _artifact_info(ref) if row["reference_path"] else {}
        schema = "not_applicable"
        if row["reference_path"] and ref.exists() and gen.exists() and ref.suffix.lower() == ".csv" and gen.suffix.lower() == ".csv":
            schema = "ok" if ref_info.get("columns") == info.get("columns") else "column_diff"
        numeric = _numeric_diff(ref, gen, tol) if row["reference_path"] else "not_applicable"
        required = row["required"] == "1"
        ok = bool(info["exists"]) and info["read_status"] in {"ok", "dir_ok"}
        status = "pass" if ok and (not required or schema != "column_diff") else ("missing" if not info["exists"] else "fail")
        matrix.append(
            {
                "target": target,
                "dataset": spec["dataset"],
                "model_family": spec["model_family"],
                "report_section": row["report_section"],
                "table_or_figure": row["table_or_figure"],
                "reference_path": row["reference_path"],
                "generated_path": _rel(gen),
                "status": status,
                "row_count": info.get("row_count", ""),
                "schema_status": schema,
                "numeric_diff_status": numeric,
                "notes": info["read_status"],
            }
        )
        if gen.exists() and gen.is_file():
            checksums.append({"target": target, "path": _rel(gen), "size": info["size"], "sha256": info["sha256"]})
    for extra_root in [p["paper"], p["paper_reports"], p["but_sqi"], p["chapter4"], p["sqi_supp"], p["sqi_supp_reports"]]:
        if extra_root.exists():
            for path in sorted(extra_root.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".png", ".pdf", ".svg", ".tiff", ".parquet", ".npz"}:
                    info = _artifact_info(path)
                    if info["sha256"]:
                        checksums.append({"target": target, "path": _rel(path), "size": info["size"], "sha256": info["sha256"]})
    return matrix, checksums


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip()


def run_target(args: argparse.Namespace) -> dict[str, Any]:
    expected = _load_expected()
    target = args.target
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(args.work).resolve()
    run_dir = work / target / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = _target_paths(run_dir, target)
    env = _base_env(work, paths)
    if args.gpu_train:
        try:
            import torch

            if not torch.cuda.is_available():
                (run_dir / "GPU_REQUIRED").write_text("CUDA is not available on this machine.\n", encoding="utf-8")
        except Exception:
            (run_dir / "GPU_REQUIRED").write_text("PyTorch/CUDA probe failed.\n", encoding="utf-8")
    commands = _cmds(target, run_dir, force=args.force, device=args.device, gpu_train=args.gpu_train)
    results = [_run(cmd, run_dir, env, dry_run=args.dry_run, timeout=args.timeout_sec or None) for cmd in commands]
    matrix, checksums = _audit(target, run_dir, expected)
    _write_csv(run_dir / "audit_matrix.csv", matrix)
    _write_csv(run_dir / "artifact_checksums.csv", checksums)
    failures = [r for r in results if r["returncode"] != 0]
    missing = [] if args.dry_run else [r for r in matrix if r["status"] in {"missing", "fail"} and r["notes"] != "not_applicable"]
    status = "pass" if not failures and not missing else "fail"
    summary = {
        "target": target,
        "status": status,
        "run_id": run_id,
        "run_dir": _rel(run_dir),
        "dry_run": bool(args.dry_run),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_policy": "gpu-train" if args.gpu_train else "cpu-lane",
        "commands": results,
        "audit_rows": len(matrix),
        "audit_failures": len(missing),
        "checksum_rows": len(checksums),
        "env": {k: env[k] for k in ["ECG_DATA_ROOT", "ECG_TRANSFORMER_ARTIFACTS", "ECG_E31_PREDICTION_ROOT", "ECG_SQI_PAPER_ALIGNED_ROOT", "ECG_SQI_SUPPLEMENTAL_ROOT"]},
    }
    _json(run_dir / "summary.json", summary)
    lines = [
        f"# Reproduction Summary: {target}",
        "",
        f"- status: `{status}`",
        f"- run_id: `{run_id}`",
        f"- run_dir: `{_rel(run_dir)}`",
        f"- audit failures: `{len(missing)}`",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"- `{r['name']}`: `{r['returncode']}` ({r['duration_sec']} s)" for r in results)
    if missing:
        lines.extend(["", "## Audit Failures", ""])
        lines.extend(f"- `{r['table_or_figure']}` -> `{r['generated_path']}`: {r['notes']}" for r in missing[:100])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    expected = _load_expected()
    parser = argparse.ArgumentParser(description="Run one full report reproduction target.")
    parser.add_argument("--target", required=True, choices=sorted(expected["targets"]))
    parser.add_argument("--work", default=str(WORK_DEFAULT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-train", action="store_true")
    args = parser.parse_args()
    summary = run_target(args)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
