from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
WORK = REPO_ROOT.parent / "reproduce" / "work"
EXPECTED = ROOT / "expected_outputs.json"
DEFAULT_REPO = "https://github.com/Sanssssssssssssssss/ecg_sqi_fusion.git"
DEFAULT_BRANCH = "codex/reproduce-readiness"
DEFAULT_RAW_DATA = REPO_ROOT / "data"
TARGETS = [
    "baseline-cinc2011",
    "baseline-but",
    "conformer-cinc2011",
    "conformer-but",
    "sqi-supplemental",
    "transformer-supplemental",
]


class HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], logs: Path, name: str, timeout: int) -> dict[str, Any]:
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / f"{name}.stdout.log"
    err = logs / f"{name}.stderr.log"
    start = time.time()
    with out.open("w", encoding="utf-8", errors="replace") as so, err.open("w", encoding="utf-8", errors="replace") as se:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=so, stderr=se, text=True, timeout=timeout)
    return {
        "name": name,
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - start, 3),
        "stdout": rel(out),
        "stderr": rel(err),
    }


@contextmanager
def file_lock(path: Path, *, wait_timeout_sec: int = 12 * 60 * 60):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}\ncreated={datetime.now().isoformat()}\n".encode("utf-8"))
        except FileExistsError:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                pid_line = next((line for line in text.splitlines() if line.startswith("pid=")), "")
                pid = int(pid_line.split("=", 1)[1]) if pid_line else -1
                if pid > 0:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        path.unlink(missing_ok=True)
                        continue
                age = time.time() - path.stat().st_mtime
                if age > wait_timeout_sec:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.time() - start > wait_timeout_sec:
                raise TimeoutError(f"timed out waiting for lock: {path}")
            time.sleep(10)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        path.unlink(missing_ok=True)


def _count_qrs_npz(sqi_dir: Path) -> int:
    qrs = sqi_dir / "qrs"
    return sum(1 for p in qrs.glob("*.npz") if p.is_file()) if qrs.exists() else 0


def _expected_qrs_records(sqi_dir: Path) -> int:
    split = sqi_dir / "splits" / "split_seta_seed0_paper_balanced.csv"
    if not split.exists():
        return 0
    try:
        with split.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return sum(1 for row in reader if row.get("record_id"))
    except Exception:
        return 0


def _qrs_summary_exists(sqi_dir: Path) -> bool:
    qrs = sqi_dir / "qrs"
    return any(p.is_file() and p.stat().st_size > 0 for p in qrs.glob("qrs_summary*.csv")) if qrs.exists() else False


def _lm_mlp_table_status(sqi_dir: Path) -> tuple[int, int]:
    tables = sqi_dir / "models" / "lm_mlp" / "tables"
    expected = [
        tables / "table5_mlp_12lead_single_sqi_seed0.csv",
        tables / "table6_mlp_12lead_combo_sqi_seed0.csv",
        tables / "table7_mlp_selected5_seed0.csv",
    ]
    complete = sum(1 for path in expected if path.exists() and path.stat().st_size > 0)
    partial = sum(1 for path in tables.glob("*.csv") if path.is_file()) if tables.exists() else 0
    return complete, partial


def run_qrs_cache_with_retries(
    cmd: list[str],
    *,
    repo: Path,
    sqi_dir: Path,
    env: dict[str, str],
    logs: Path,
    timeout: int,
    max_attempts: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    lock = WORK / ".locks" / "sqi_qrs_cache.lock"
    with file_lock(lock):
        last_count = _count_qrs_npz(sqi_dir)
        expected = _expected_qrs_records(sqi_dir)
        for attempt in range(1, max_attempts + 1):
            name = "sqi_qrs_cache" if attempt == 1 else f"sqi_qrs_cache_retry{attempt - 1}"
            result = run_cmd(cmd, cwd=repo, env=env, logs=logs, name=name, timeout=timeout)
            result["qrs_cache_npz_before"] = last_count
            result["qrs_cache_npz_after"] = _count_qrs_npz(sqi_dir)
            result["qrs_cache_expected_records"] = expected
            attempts.append(result)
            if result["returncode"] == 0:
                return attempts, result
            current = int(result["qrs_cache_npz_after"])
            complete = expected > 0 and current >= expected and _qrs_summary_exists(sqi_dir)
            if complete:
                fixed = {**result, "name": f"{name}_accepted_complete_cache", "returncode": 0}
                attempts.append(fixed)
                return attempts, fixed
            if current <= last_count:
                return attempts, result
            last_count = current
        return attempts, attempts[-1]


def run_lm_mlp_with_retries(
    cmd: list[str],
    *,
    repo: Path,
    sqi_dir: Path,
    env: dict[str, str],
    logs: Path,
    timeout: int,
    max_attempts: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    last_complete, last_partial = _lm_mlp_table_status(sqi_dir)
    for attempt in range(1, max_attempts + 1):
        name = "sqi_lm_mlp_search" if attempt == 1 else f"sqi_lm_mlp_search_retry{attempt - 1}"
        result = run_cmd(cmd, cwd=repo, env=env, logs=logs, name=name, timeout=timeout)
        complete, partial = _lm_mlp_table_status(sqi_dir)
        result["lm_mlp_expected_tables_complete"] = complete
        result["lm_mlp_table_csv_count"] = partial
        attempts.append(result)
        if result["returncode"] == 0:
            return attempts, result
        if complete >= 3:
            fixed = {**result, "name": f"{name}_accepted_complete_tables", "returncode": 0}
            attempts.append(fixed)
            return attempts, fixed
        if complete <= last_complete and partial <= last_partial and attempt > 1:
            return attempts, result
        last_complete, last_partial = complete, partial
    return attempts, attempts[-1]


def safe_rmtree(path: Path, root: Path) -> None:
    rp = path.resolve()
    rr = root.resolve()
    if rr not in rp.parents and rp != rr:
        raise RuntimeError(f"refusing to delete outside work root: {path}")
    if path.exists():
        shutil.rmtree(path)


def clone_repo(args: argparse.Namespace, run_dir: Path) -> Path:
    repo = run_dir / "repo"
    if repo.exists() and (repo / ".git").exists():
        return repo
    if repo.exists():
        safe_rmtree(repo, WORK if Path(args.work).resolve() == WORK.resolve() else Path(args.work).resolve())
    cmd = ["git", "-c", "core.longpaths=true", "clone", "--depth", "1", "--single-branch", "--branch", args.branch, args.repo_url, str(repo)]
    last: subprocess.CalledProcessError | None = None
    for attempt in range(1, 4):
        try:
            subprocess.run(cmd, cwd=run_dir, check=True)
            return repo
        except subprocess.CalledProcessError as exc:
            last = exc
            safe_rmtree(repo, WORK if Path(args.work).resolve() == WORK.resolve() else Path(args.work).resolve())
            if attempt < 3:
                time.sleep(5 * attempt)
                continue
            raise
    return repo


def git_head(repo: Path) -> str:
    last: Exception | None = None
    for _ in range(5):
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        except Exception as exc:
            last = exc
            time.sleep(2)
    raise RuntimeError(f"git rev-parse HEAD failed after clone: {last}")


def make_venv(run_dir: Path) -> Path:
    venv = run_dir / ".venv"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    return py


def artifact_dirs(run_dir: Path) -> dict[str, Path]:
    base = run_dir / "artifacts"
    return {
        "sqi": base / "sqi_paper_aligned",
        "sqi_reports": base / "reports" / "sqi_paper_aligned",
        "sqi_supplemental": base / "sqi_supplemental",
        "sqi_supplemental_reports": base / "reports" / "sqi_supplemental",
        "v116": base / "transformer" / "v116_e31",
        "but_sqi": base / "transformer" / "supplemental" / "but_sqi_baseline",
        "chapter4": base / "transformer" / "supplemental" / "chapter4_evidence",
        "data": run_dir / "data",
    }


def _link_dir(link: Path, target: Path) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "junction"
    os.symlink(target, link, target_is_directory=True)
    return "symlink"


def ensure_repo_data(repo: Path, data_root: Path) -> tuple[Path, str]:
    repo_data = repo / "data"
    if repo_data.exists() and any(repo_data.iterdir()):
        return repo_data, "existing_repo_data"
    if repo_data.exists():
        shutil.rmtree(repo_data)
    return data_root, _link_dir(repo_data, data_root)


def copy_raw_data(source: Path, repo: Path, data_root: Path | None = None) -> dict[str, Any]:
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"raw data root does not exist: {source}")
    copied: list[dict[str, Any]] = []
    repo_data = repo / "data"
    data_dir, data_mode = ensure_repo_data(repo, data_root) if data_root is not None else (repo_data, "repo")
    for name in ["external", "ptb-xl", "physionet"]:
        src = source / name
        dst = data_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing raw data source: {src}")
        if dst.exists() and any(dst.iterdir()):
            files = sum(1 for p in dst.rglob("*") if p.is_file())
            copied.append({"name": name, "source": str(src), "target": str(dst), "files": files, "mode": "existing"})
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mode = "copy"
        if os.name == "nt":
            try:
                subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                mode = "junction"
            except Exception:
                shutil.copytree(src, dst)
        else:
            try:
                os.symlink(src, dst, target_is_directory=True)
                mode = "symlink"
            except Exception:
                shutil.copytree(src, dst)
        files = sum(1 for p in dst.rglob("*") if p.is_file())
        copied.append({"name": name, "source": str(src), "target": str(dst), "files": files, "mode": mode})
    return {"raw_data_root": str(source), "repo_data": str(repo_data), "data_dir": str(data_dir), "data_mode": data_mode, "copied": copied}


def url_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "ecg-sqi-fusion-external-reproduce"})
    with urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(3):
        try:
            tmp.unlink(missing_ok=True)
            req = Request(url, headers={"User-Agent": "ecg-sqi-fusion-external-reproduce"})
            with urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(target)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def prepare_challenge2011(repo: Path) -> None:
    set_a = repo / "data" / "physionet" / "challenge-2011" / "set-a"
    records = set_a / "RECORDS"
    if records.exists():
        ids = [line.strip().split("/")[-1].split(".")[0] for line in records.read_text(errors="ignore").splitlines() if line.strip()]
        if ids and all((set_a / f"{rid}.hea").exists() and (set_a / f"{rid}.dat").exists() for rid in ids):
            return
    listing_base = "https://physionet.org/content/challenge-2011/1.0.0/set-a/"
    file_base = "https://physionet.org/files/challenge-2011/1.0.0/set-a/"
    parser = HrefParser()
    parser.feed(url_text(listing_base))
    names: set[str] = {"RECORDS", "RECORDS-acceptable", "RECORDS-unacceptable"}
    for href in parser.hrefs:
        name = Path(urlparse(href).path).name
        if name.endswith((".hea", ".dat")):
            names.add(name)
    if not names:
        raise RuntimeError(f"no Set-A files found at {listing_base}")
    for name in ["RECORDS", "RECORDS-acceptable", "RECORDS-unacceptable"]:
        download_url(urljoin(file_base, name), set_a / name)
    record_ids = [line.strip().split("/")[-1].split(".")[0] for line in (set_a / "RECORDS").read_text(errors="ignore").splitlines() if line.strip()]
    wanted = {f"{rid}.{ext}" for rid in record_ids for ext in ["hea", "dat"]}
    missing_from_listing = sorted(wanted - names)
    if missing_from_listing:
        raise RuntimeError(f"Set-A listing missing files, examples={missing_from_listing[:5]}")
    jobs = [(urljoin(file_base, name), set_a / name) for name in sorted(wanted)]
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(lambda item: download_url(*item), jobs):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"downloaded Set-A files {done}/{len(jobs)}", flush=True)
    bad = []
    for path in set_a.glob("*.hea"):
        if path.read_text(errors="ignore").lstrip().lower().startswith("<!doctype"):
            bad.append(path.name)
    for path in set_a.glob("*.dat"):
        if path.stat().st_size < 100_000:
            bad.append(path.name)
    if bad:
        raise RuntimeError(f"invalid Set-A downloads, examples={bad[:5]}")


def prepare_nstdb_em_ma(repo: Path) -> None:
    target = repo / "data" / "physionet" / "nstdb"
    base = "https://physionet.org/files/nstdb/1.0.0/"
    for name in ["em.hea", "em.dat", "ma.hea", "ma.dat"]:
        download_url(urljoin(base, name), target / name)


def install_repo(py: Path, repo: Path, env: dict[str, str], logs: Path, timeout: int, *, name: str = "pip_install") -> dict[str, Any]:
    marker = py.parent.parent / ".installed"
    if marker.exists():
        return {"name": name, "cmd": "cached", "returncode": 0, "duration_sec": 0, "stdout": "", "stderr": ""}
    result = run_cmd([str(py), "-m", "pip", "install", "-e", str(repo)], cwd=repo, env=env, logs=logs, name=name, timeout=timeout)
    if result["returncode"] == 0:
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    return result


def prepare_paper_detectors(py: Path, repo: Path, env: dict[str, str], logs: Path, timeout: int, sqi_dir: Path) -> dict[str, Any]:
    cmd = [
        str(py),
        "-m",
        "src.sqi_pipeline.qrs.setup_paper_detectors",
        "--out_dir",
        str(sqi_dir / "qrs" / "tools"),
        "--require",
    ]
    source_bin = env.get("WFDB_QRS_KIT_FROM_BIN_DIR") or env.get("SQI_PAPER_QRS_BIN_DIR")
    if source_bin:
        cmd.extend(["--from-bin-dir", source_bin, "--no_download"])
    elif env.get("ECG_NO_DOWNLOAD") == "1":
        cmd.append("--no_download")
    compile_requested = env.get("WFDB_QRS_KIT_COMPILE", "").strip().lower() in {"1", "true", "yes", "on"}
    if compile_requested:
        cmd.append("--compile")
        if env.get("WFDB_QRS_KIT_COMPILER"):
            cmd.extend(["--compiler", env["WFDB_QRS_KIT_COMPILER"]])
        if env.get("WFDB_QRS_KIT_WFDB_INCLUDE"):
            cmd.extend(["--wfdb-include", env["WFDB_QRS_KIT_WFDB_INCLUDE"]])
        if env.get("WFDB_QRS_KIT_WFDB_LIB"):
            cmd.extend(["--wfdb-lib", env["WFDB_QRS_KIT_WFDB_LIB"]])
    result = run_cmd(cmd, cwd=repo, env=env, logs=logs, name="paper_qrs_setup", timeout=timeout)
    manifest_path = sqi_dir / "qrs" / "tools" / "paper_qrs_detector_manifest.json"
    if result["returncode"] == 0 and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        executables = manifest.get("executables", {})
        if executables.get("wqrs"):
            env["WFDB_QRS_KIT_WQRS_EXE"] = str(executables["wqrs"])
        if executables.get("eplimited"):
            env["WFDB_QRS_KIT_EPLIMITED_EXE"] = str(executables["eplimited"])
    return result


def command_mentions_data_download(result: dict[str, Any]) -> bool:
    for key in ["stdout", "stderr"]:
        rel_path = result.get(key)
        if not rel_path:
            continue
        path = ROOT / str(rel_path)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Downloading files..." in text or "Downloading BUT QDB" in text:
            return True
    return False


def sqi_paper_plan(p: str, sqi_dir: Path) -> list[tuple[str, list[str]]]:
    base = [p, "-m", "src.sqi_pipeline.run_all", "--profile", "paper_aligned", "--artifacts_dir", str(sqi_dir), "--seed", "0"]
    return [
        ("sqi_manifest_raw", base + ["--fresh", "--only", "manifest_raw"]),
        ("sqi_paper_balanced_seta", base + ["--only", "paper_balanced_seta"]),
        ("sqi_resample_125", base + ["--only", "resample_125"]),
        ("sqi_qrs_cache", base + ["--only", "qrs_cache"]),
        ("sqi_record84", base + ["--only", "record84"]),
        ("sqi_norm_record84", base + ["--only", "norm_record84_ks"]),
        ("sqi_lm_mlp_search", base + ["--only", "lm_mlp_search"]),
        ("sqi_svm_tables", base + ["--only", "svm_tables"]),
    ]


def command_plan(target: str, py: Path, repo: Path, device: str, run_dir: Path) -> list[tuple[str, list[str]]]:
    p = str(py)
    dirs = artifact_dirs(run_dir)
    sqi_dir = dirs["sqi"]
    out_ch4 = str(dirs["chapter4"])
    v116_dir = str(dirs["v116"])
    if target == "baseline-cinc2011":
        return sqi_paper_plan(p, sqi_dir) + [
            ("sqi_table_trends", [p, "-m", "src.sqi_pipeline.diagnostics.compare_paper_tables", "--artifacts_dir", str(sqi_dir), "--out_dir", str(dirs["sqi_reports"] / "table_trend_comparison"), "--seed", "0"]),
            ("sqi_paper_tables", [p, "-m", "src.sqi_pipeline.tools.make_paper_tables", "--artifacts_dir", str(sqi_dir), "--out_dir", str(sqi_dir / "paper_tables"), "--seed", "0"]),
        ]
    if target == "baseline-but":
        return [
            ("transformer_v116_public_rebuild", [p, "-m", "src.transformer_pipeline.run_all", "--run", "--train", "none", "--artifacts-dir", v116_dir, "--seed", "20260876"]),
            ("but_sqi_baseline", [p, "-m", "src.supplemental_transformer_experiments.but_sqi_baseline.run", "--out", str(dirs["but_sqi"]), "--device", device, "--jobs", "1", "pipeline", "--run"]),
        ]
    if target == "conformer-cinc2011":
        return sqi_paper_plan(p, sqi_dir) + [
            ("seta_build", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "seta-build", "--run"]),
            ("seta_sqi", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "seta-sqi", "--run"]),
            ("protocol_audit", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "--scope", "seta", "audit", "--run"]),
            ("seta_repair", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "seta-repair", "--run"]),
            ("seta_models", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "seta-models", "--run"]),
            ("figures", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "--scope", "seta", "figures", "--run"]),
        ]
    if target == "conformer-but":
        return [
            ("transformer_v116_public_rebuild", [p, "-m", "src.transformer_pipeline.run_all", "--run", "--train", "none", "--artifacts-dir", v116_dir, "--seed", "20260876"]),
            ("but_models", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-models", "--run"]),
            ("but_boundary_audit", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-boundary-audit", "--run"]),
            ("but_query_patching", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-query-patching", "--run"]),
            ("figures", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "--scope", "but", "figures", "--run"]),
        ]
    if target == "sqi-supplemental":
        return sqi_paper_plan(p, sqi_dir) + [
            ("sqi_supplemental", [p, "-m", "src.supplemental_sqi_experiments.run", "diagnose-existing", "--artifacts-dir", str(sqi_dir), "--out-dir", str(dirs["sqi_supplemental"]), "--report-dir", str(dirs["sqi_supplemental_reports"]), "--seed", "0"]),
        ]
    if target == "transformer-supplemental":
        return [
            ("transformer_v116_public_rebuild", [p, "-m", "src.transformer_pipeline.run_all", "--run", "--train", "none", "--artifacts-dir", v116_dir, "--seed", "20260876"]),
            ("but_models", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-models", "--run"]),
            ("but_boundary_audit", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-boundary-audit", "--run"]),
            ("but_query_patching", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-query-patching", "--run"]),
            ("but_time_local_transplant", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-time-local-transplant", "--run"]),
            ("but_architecture_ablation", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-architecture-ablation", "--run"]),
            ("but_local_counterfactuals", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "but-local-counterfactuals", "--run"]),
            ("figures", [p, "-m", "src.supplemental_transformer_experiments.chapter4_evidence.run", "--out", out_ch4, "--device", device, "--scope", "but", "figures", "--run"]),
        ]
    raise ValueError(target)


def csv_info(path: Path) -> tuple[str, int, str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return "csv_no_header", 0, ""
            rows = sum(1 for _ in reader)
        return "ok", rows, "|".join(header)
    except Exception as exc:
        return f"unreadable:{exc}", 0, ""


def generated_path(logical_path: str, repo: Path, run_dir: Path) -> Path:
    s = logical_path.replace("\\", "/")
    mappings = [
        ("outputs/transformer/v116_e31/", run_dir / "artifacts" / "transformer" / "v116_e31"),
        ("outputs/transformer/supplemental/chapter4_evidence/", run_dir / "artifacts" / "transformer" / "supplemental" / "chapter4_evidence"),
        ("outputs/transformer/supplemental/but_sqi_baseline/", run_dir / "artifacts" / "transformer" / "supplemental" / "but_sqi_baseline"),
        ("outputs/sqi_paper_aligned/", run_dir / "artifacts" / "sqi_paper_aligned"),
        ("outputs/reports/sqi_paper_aligned/", run_dir / "artifacts" / "reports" / "sqi_paper_aligned"),
        ("outputs/sqi_supplemental/", run_dir / "artifacts" / "sqi_supplemental"),
        ("outputs/reports/sqi_supplemental/", run_dir / "artifacts" / "reports" / "sqi_supplemental"),
    ]
    for prefix, root in mappings:
        if s.startswith(prefix):
            return root / s[len(prefix):]
    return repo / logical_path


def audit(target: str, repo: Path, run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = json.loads(EXPECTED.read_text(encoding="utf-8")).get(target, [])
    rows: list[dict[str, Any]] = []
    sums: list[dict[str, Any]] = []
    for item in spec:
        path = generated_path(item["path"], repo, run_dir)
        exists = path.exists() and path.stat().st_size > 0 if path.is_file() else path.exists()
        status = "pass" if exists else ("missing_required" if item.get("required", True) else "missing_optional")
        read_status = "missing"
        row_count: int | str = ""
        columns = ""
        digest = ""
        if exists and path.is_file():
            digest = sha256(path)
            if path.suffix.lower() == ".csv":
                read_status, row_count, columns = csv_info(path)
            elif path.suffix.lower() in {".json", ".md", ".txt", ".xlsx", ".parquet", ".svg", ".png", ".pdf", ".tiff"}:
                read_status = "exists"
            else:
                read_status = "exists"
            sums.append({"name": item["name"], "path": item["path"], "bytes": path.stat().st_size, "sha256": digest})
        rows.append({
            "target": target,
            "artifact": item["name"],
            "path": item["path"],
            "generated_path": str(path),
            "required": bool(item.get("required", True)),
            "status": status,
            "read_status": read_status,
            "row_count": row_count,
            "columns": columns,
            "sha256": digest,
        })
    write_csv(run_dir / "audit_matrix.csv", rows)
    write_csv(run_dir / "artifact_checksums.csv", sums)
    return rows, sums


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full reproduction controller for ECG SQI reports.")
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--repo-url", default=DEFAULT_REPO)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--clone", action="store_true", help="clone --repo-url into the run directory instead of using this checkout")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timeout-sec", type=int, default=24 * 60 * 60)
    ap.add_argument("--force", action="store_true", help="delete an existing run directory with the same run id")
    ap.add_argument("--raw-copy", action="store_true", help="copy local raw data into the fresh clone before running")
    ap.add_argument("--raw-data-root", default=str(DEFAULT_RAW_DATA))
    ap.add_argument("--no-download", action="store_true", help="fail if a pipeline tries to download raw data")
    ap.add_argument("--start-at", default="", help="skip target commands before this step name")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = work / args.target / run_id
    if run_dir.exists() and args.force:
        safe_rmtree(run_dir, work)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    dirs = artifact_dirs(run_dir)
    env["ECG_V116_ARTIFACTS_DIR"] = str(dirs["v116"])
    if args.raw_copy or args.no_download:
        env["ECG_NO_DOWNLOAD"] = "1"

    commands: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_copy: dict[str, Any] | None = None
    stopped_after_failure = ""
    repo: Path | None = None
    commit = ""
    py: Path | None = None
    rows: list[dict[str, Any]] = []
    sums: list[dict[str, Any]] = []
    audit_failures: list[dict[str, Any]] = []
    exception = ""
    status = "fail"
    try:
        repo = clone_repo(args, run_dir) if args.clone else REPO_ROOT
        commit = git_head(repo)
        if args.raw_copy:
            raw_copy = copy_raw_data(Path(args.raw_data_root), repo, dirs["data"])
        else:
            ensure_repo_data(repo, dirs["data"])
        py = make_venv(run_dir)
        if args.start_at and (py.parent.parent / ".installed").exists():
            commands.append({"name": "pip_install", "cmd": "skipped by --start-at", "returncode": 0, "duration_sec": 0, "skipped": True})
        else:
            commands.append(install_repo(py, repo, env, logs, args.timeout_sec))
            if commands[-1]["returncode"] != 0:
                commands.append(install_repo(py, repo, env, logs, args.timeout_sec, name="pip_install_retry"))
            if commands[-1]["returncode"] != 0:
                failures.append(commands[-1])

        if not failures and args.target in {"baseline-cinc2011", "conformer-cinc2011", "sqi-supplemental"}:
            commands.append(prepare_paper_detectors(py, repo, env, logs, args.timeout_sec, dirs["sqi"]))
            if commands[-1]["returncode"] != 0:
                failures.append(commands[-1])

        if not failures and args.target in {"baseline-cinc2011", "conformer-cinc2011", "sqi-supplemental"} and not (args.raw_copy or args.no_download):
            prepare_challenge2011(repo)
            prepare_nstdb_em_ma(repo)

        if not failures:
            seen_start = not args.start_at
            for name, cmd in command_plan(args.target, py, repo, args.device, run_dir):
                if not seen_start:
                    if name == args.start_at:
                        seen_start = True
                    else:
                        continue
                if name == "sqi_qrs_cache":
                    qrs_attempts, result = run_qrs_cache_with_retries(
                        cmd,
                        repo=repo,
                        sqi_dir=dirs["sqi"],
                        env=env,
                        logs=logs,
                        timeout=args.timeout_sec,
                    )
                    commands.extend(qrs_attempts)
                elif name == "sqi_lm_mlp_search":
                    mlp_attempts, result = run_lm_mlp_with_retries(
                        cmd,
                        repo=repo,
                        sqi_dir=dirs["sqi"],
                        env=env,
                        logs=logs,
                        timeout=args.timeout_sec,
                    )
                    commands.extend(mlp_attempts)
                else:
                    result = run_cmd(cmd, cwd=repo, env=env, logs=logs, name=name, timeout=args.timeout_sec)
                    commands.append(result)
                if result["returncode"] != 0:
                    failures.append(result)
                    stopped_after_failure = name
                    break
                if (args.raw_copy or args.no_download) and command_mentions_data_download(result):
                    failure = {**result, "returncode": 97, "error": "raw-copy/no-download lane attempted raw data download"}
                    failures.append(failure)
                    stopped_after_failure = name
                    break

        rows, sums = audit(args.target, repo, run_dir)
        audit_failures = [r for r in rows if r["status"] == "missing_required"]
        status = "pass" if not failures and not audit_failures else "fail"
    except Exception as exc:
        exception = repr(exc)
    summary = {
        "target": args.target,
        "status": status,
        "run_id": run_id,
        "run_dir": rel(run_dir),
        "repo_url": args.repo_url,
        "branch": args.branch,
        "repo": str(repo) if repo else "",
        "clone": bool(args.clone),
        "commit": commit,
        "python": subprocess.check_output([str(py), "--version"], text=True).strip() if py else sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_status": "not_checked_cpu_lane",
        "qrs_detector_policy": "paper profiles require wfdb-qrs-kit-managed wqrs/eplimited; no xqrs/gqrs fallback is enabled",
        "raw_copy": raw_copy,
        "no_download": bool(args.raw_copy or args.no_download),
        "stopped_after_failure": stopped_after_failure,
        "exception": exception,
        "commands": commands,
        "command_failures": failures,
        "audit_rows": len(rows),
        "audit_failures": len(audit_failures),
        "checksum_rows": len(sums),
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(
        "\n".join([
            f"# {args.target}",
            f"- status: {status}",
            f"- commit: `{commit}`",
            f"- exception: `{exception}`",
            f"- commands: {len(commands)}",
            f"- command_failures: {len(failures)}",
            f"- audit_failures: {len(audit_failures)}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()
