from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import wfdb


CHALLENGE_2011_SETA_FILES = ("RECORDS", "RECORDS-acceptable", "RECORDS-unacceptable")
CHALLENGE_2011_BASE_URL = "https://physionet.org/files/challenge-2011/1.0.0/set-a/"
BUTQDB_BASE_URL = "https://physionet.org/files/butqdb/1.0.0/"


def _nonempty(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def _marker(target: Path) -> Path:
    return target / ".download_complete.json"


def _write_marker(target: Path, slug: str, checked: list[str]) -> None:
    _marker(target).write_text(
        json.dumps({"slug": slug, "checked": checked}, indent=2),
        encoding="utf-8",
    )


def _download_url(url: str, target: Path, *, attempts: int = 3) -> None:
    if _nonempty(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "ecg-sqi-fusion"})
            with urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(target)
            return
        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(10, 2 * attempt))
    raise RuntimeError(f"failed to download {url} after {attempts} attempts") from last_exc


def _downloads_disabled() -> bool:
    return os.environ.get("ECG_NO_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def _require_download_allowed(slug: str, target: Path) -> None:
    if _downloads_disabled():
        raise FileNotFoundError(f"{slug} raw data incomplete at {target}; ECG_NO_DOWNLOAD=1")


def _generic_complete(target: Path, required: tuple[str, ...]) -> bool:
    if not target.exists():
        return False
    for name in required:
        path = target / name
        if not _nonempty(path):
            return False
        if path.is_dir() and not any(path.iterdir()):
            return False
    return True


def _read_record_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(errors="ignore").splitlines():
        value = line.strip()
        if not value:
            continue
        rid = Path(value.replace("\\", "/").split("/")[-1]).stem
        if rid and rid not in seen:
            seen.add(rid)
            ids.append(rid)
    return ids


def _challenge_seta_missing(target: Path) -> list[Path]:
    set_a = target / "set-a"
    missing: list[Path] = []
    for name in CHALLENGE_2011_SETA_FILES:
        path = set_a / name
        if not _nonempty(path):
            missing.append(path)

    record_ids = _read_record_ids(set_a / "RECORDS")
    if not record_ids:
        return missing or [set_a / "RECORDS"]

    for rid in record_ids:
        for suffix in (".hea", ".dat"):
            path = set_a / f"{rid}{suffix}"
            if not _nonempty(path):
                missing.append(path)
    return missing


def _download_challenge_2011_seta(target: Path) -> None:
    set_a = target / "set-a"
    set_a.mkdir(parents=True, exist_ok=True)
    for name in CHALLENGE_2011_SETA_FILES:
        _download_url(urljoin(CHALLENGE_2011_BASE_URL, name), set_a / name)

    record_ids = _read_record_ids(set_a / "RECORDS")
    for i, rid in enumerate(record_ids, start=1):
        if not (_nonempty(set_a / f"{rid}.hea") and _nonempty(set_a / f"{rid}.dat")):
            print(f"Downloading Challenge 2011 Set-A record {i}/{len(record_ids)}: {rid}", flush=True)
            _download_url(urljoin(CHALLENGE_2011_BASE_URL, f"{rid}.hea"), set_a / f"{rid}.hea")
            _download_url(urljoin(CHALLENGE_2011_BASE_URL, f"{rid}.dat"), set_a / f"{rid}.dat")


def _but_record_ids(target: Path) -> list[str]:
    records = target / "RECORDS"
    if records.exists():
        ids: set[str] = set()
        for value in _read_record_ids(records):
            if value.endswith(("_ECG", "_ACC")):
                ids.add(value.rsplit("_", 1)[0])
        if ids:
            return sorted(ids)
    dirs = [p.name for p in target.iterdir() if p.is_dir()] if target.exists() else []
    return sorted([name for name in dirs if name[:1].isdigit()])


def _but_missing(target: Path) -> list[Path]:
    ids = _but_record_ids(target)
    if not ids:
        return [target / "RECORDS"]
    missing: list[Path] = []
    root_required = ["RECORDS", "SHA256SUMS.txt", "subject-info.csv"]
    for name in root_required:
        path = target / name
        if not _nonempty(path):
            missing.append(path)
    for rid in ids:
        rec_dir = target / rid
        for suffix in ["ACC.hea", "ACC.dat", "ECG.hea", "ECG.dat", "ANN.csv"]:
            path = rec_dir / f"{rid}_{suffix}"
            if not _nonempty(path):
                missing.append(path)
    return missing


def _but_complete(target: Path) -> bool:
    return target.exists() and not _but_missing(target)


def _download_butqdb(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ["RECORDS", "SHA256SUMS.txt", "subject-info.csv", "ANNOTATORS", "LICENSE.txt"]:
        _download_url(urljoin(BUTQDB_BASE_URL, name), target / name)

    ids = _but_record_ids(target)
    for i, rid in enumerate(ids, start=1):
        print(f"Downloading BUT QDB record {i}/{len(ids)}: {rid}", flush=True)
        for suffix in ["ACC.hea", "ACC.dat", "ECG.hea", "ECG.dat", "ANN.csv"]:
            name = f"{rid}_{suffix}"
            _download_url(urljoin(BUTQDB_BASE_URL, f"{rid}/{name}"), target / rid / name)


def _ptbxl_complete(target: Path) -> bool:
    csv_path = target / "ptbxl_database.csv"
    records = target / "records100"
    if not (_nonempty(csv_path) and records.exists()):
        return False
    hea = list(records.rglob("*.hea"))
    dat = list(records.rglob("*.dat"))
    return len(hea) >= 100 and len(dat) >= 100


def ensure_wfdb_database(slug: str, target: Path, required: tuple[str, ...]) -> None:
    target = Path(target)
    if slug == "challenge-2011" and target.exists() and not _challenge_seta_missing(target):
        _write_marker(target, slug, ["set-a"])
        return
    if slug == "butqdb" and _but_complete(target):
        _write_marker(target, slug, ["butqdb_records"])
        return
    if slug == "ptb-xl" and _ptbxl_complete(target):
        _write_marker(target, slug, ["ptbxl_database.csv", "records100"])
        return
    if slug not in {"challenge-2011", "butqdb", "ptb-xl"} and _generic_complete(target, required):
        _write_marker(target, slug, list(required))
        return
    _require_download_allowed(slug, target)
    target.mkdir(parents=True, exist_ok=True)
    if slug == "challenge-2011":
        _download_challenge_2011_seta(target)
        missing = _challenge_seta_missing(target)
        if missing:
            examples = ", ".join(str(p) for p in missing[:5])
            raise FileNotFoundError(
                f"Challenge 2011 Set-A download incomplete; missing {len(missing)} file(s), "
                f"examples: {examples}"
            )
        _write_marker(target, slug, ["set-a"])
        return
    if slug == "butqdb":
        _download_butqdb(target)
        missing = _but_missing(target)
        if missing:
            examples = ", ".join(str(p) for p in missing[:5])
            raise FileNotFoundError(f"BUT QDB download incomplete; missing {len(missing)} file(s), examples: {examples}")
        _write_marker(target, slug, ["butqdb_records"])
        return
    wfdb.dl_database(slug, dl_dir=str(target))
    if slug == "ptb-xl" and not _ptbxl_complete(target):
        raise FileNotFoundError(f"PTB-XL download incomplete at {target}")
    if slug not in {"challenge-2011", "ptb-xl"} and not _generic_complete(target, required):
        raise FileNotFoundError(f"{slug} download incomplete at {target}")
    _write_marker(target, slug, list(required))


def ensure_sqi_raw_data(root: Path) -> None:
    physionet = Path(root) / "data" / "physionet"
    ensure_wfdb_database("challenge-2011", physionet / "challenge-2011", ("set-a",))
    ensure_wfdb_database("nstdb", physionet / "nstdb", ("em.hea", "em.dat", "ma.hea", "ma.dat"))


def ensure_ptbxl_raw_data(root: Path) -> None:
    root = Path(root)
    ensure_wfdb_database("ptb-xl", root / "data" / "ptb-xl", ("ptbxl_database.csv", "records100"))


def ensure_transformer_raw_data(root: Path) -> None:
    root = Path(root)
    ensure_wfdb_database("butqdb", root / "data" / "external" / "butqdb_1_0_0", ("100001",))
    ensure_ptbxl_raw_data(root)
