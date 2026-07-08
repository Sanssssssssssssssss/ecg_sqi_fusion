from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import data_downloads


def _write_complete_seta(root: Path, records: tuple[str, ...] = ("1002603", "1002867")) -> None:
    set_a = root / "set-a"
    set_a.mkdir(parents=True)
    (set_a / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
    (set_a / "RECORDS-acceptable").write_text(records[0] + "\n", encoding="utf-8")
    (set_a / "RECORDS-unacceptable").write_text(records[-1] + "\n", encoding="utf-8")
    for rid in records:
        (set_a / f"{rid}.hea").write_text("header\n", encoding="utf-8")
        (set_a / f"{rid}.dat").write_bytes(b"data")


def test_challenge_seta_complete_skips_download(tmp_path, monkeypatch):
    challenge_root = tmp_path / "challenge-2011"
    _write_complete_seta(challenge_root)

    def forbidden_download(*args, **kwargs):
        raise AssertionError("complete Challenge Set-A should not download")

    monkeypatch.setattr(data_downloads.wfdb, "dl_database", forbidden_download)

    data_downloads.ensure_wfdb_database("challenge-2011", challenge_root, ("set-a",))

    assert (challenge_root / ".download_complete.json").exists()


def test_challenge_seta_partial_fails_when_downloads_disabled(tmp_path, monkeypatch):
    challenge_root = tmp_path / "challenge-2011"
    set_a = challenge_root / "set-a"
    set_a.mkdir(parents=True)
    (set_a / "RECORDS").write_text("1002603\n", encoding="utf-8")
    (set_a / "RECORDS-acceptable").write_text("1002603\n", encoding="utf-8")
    (set_a / "RECORDS-unacceptable").write_text("", encoding="utf-8")
    (set_a / "1002603.hea").write_text("header\n", encoding="utf-8")

    monkeypatch.setenv("ECG_NO_DOWNLOAD", "1")

    with pytest.raises(FileNotFoundError, match="challenge-2011 raw data incomplete"):
        data_downloads.ensure_wfdb_database("challenge-2011", challenge_root, ("set-a",))


def test_challenge_seta_missing_downloads_only_missing_records(tmp_path, monkeypatch):
    challenge_root = tmp_path / "challenge-2011"

    def fake_download_url(url: str, target: Path) -> None:
        if target.name == "RECORDS":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("1002603\n", encoding="utf-8")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("1002603\n", encoding="utf-8")

    def forbidden_download(*args, **kwargs):
        raise AssertionError("Challenge Set-A uses direct file downloads")

    monkeypatch.setattr(data_downloads, "_download_url", fake_download_url)
    monkeypatch.setattr(data_downloads.wfdb, "dl_database", forbidden_download)

    data_downloads.ensure_wfdb_database("challenge-2011", challenge_root, ("set-a",))

    assert (challenge_root / "set-a" / "1002603.hea").exists()
    assert (challenge_root / "set-a" / "1002603.dat").exists()
    assert (challenge_root / ".download_complete.json").exists()


def test_sqi_raw_data_skips_download_when_complete(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _write_complete_seta(root / "data" / "physionet" / "challenge-2011")
    nstdb = root / "data" / "physionet" / "nstdb"
    nstdb.mkdir(parents=True)
    for name in ("em.hea", "em.dat", "ma.hea", "ma.dat"):
        (nstdb / name).write_bytes(b"data")

    def forbidden_download(*args, **kwargs):
        raise AssertionError("complete SQI raw data should not download")

    monkeypatch.setattr(data_downloads.wfdb, "dl_database", forbidden_download)

    data_downloads.ensure_sqi_raw_data(root)

    assert (nstdb / ".download_complete.json").exists()


def test_butqdb_download_uses_direct_files(tmp_path, monkeypatch):
    root = tmp_path / "butqdb"

    def fake_download_url(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.name == "RECORDS":
            target.write_text("100001/100001_ACC\n100001/100001_ECG\n", encoding="utf-8")
        else:
            target.write_bytes(b"data")

    def forbidden_wfdb_download(*args, **kwargs):
        raise AssertionError("BUT QDB download should not use wfdb.dl_database")

    monkeypatch.setattr(data_downloads, "_download_url", fake_download_url)
    monkeypatch.setattr(data_downloads.wfdb, "dl_database", forbidden_wfdb_download)

    data_downloads.ensure_wfdb_database("butqdb", root, ("100001",))

    for suffix in ("ACC.hea", "ACC.dat", "ECG.hea", "ECG.dat", "ANN.csv"):
        assert (root / "100001" / f"100001_{suffix}").exists()
    assert (root / ".download_complete.json").exists()
