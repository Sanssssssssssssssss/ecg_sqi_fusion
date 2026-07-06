from __future__ import annotations

import os
from pathlib import Path

import wfdb


def data_root(root: Path) -> Path:
    override = os.environ.get("ECG_DATA_ROOT")
    return Path(override).expanduser().resolve() if override else Path(root) / "data"


def ensure_wfdb_database(slug: str, target: Path, required: tuple[str, ...]) -> None:
    target = Path(target)
    if all((target / name).exists() for name in required):
        return
    target.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(slug, dl_dir=str(target))


def ensure_sqi_raw_data(root: Path) -> None:
    physionet = data_root(root) / "physionet"
    ensure_wfdb_database("challenge-2011", physionet / "challenge-2011", ("set-a",))
    ensure_wfdb_database("nstdb", physionet / "nstdb", ("em.hea", "ma.hea"))


def ensure_ptbxl_raw_data(root: Path) -> None:
    ensure_wfdb_database("ptb-xl", data_root(root) / "ptb-xl", ("ptbxl_database.csv", "records100"))


def ensure_transformer_raw_data(root: Path) -> None:
    base = data_root(root)
    ensure_wfdb_database("butqdb", base / "external" / "butqdb_1_0_0", ("100001",))
    ensure_wfdb_database("ptb-xl", base / "ptb-xl", ("ptbxl_database.csv", "records100"))
