import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi
from src.utils import data_downloads


def test_but_raw_data_incomplete_fails_when_downloads_disabled(tmp_path, monkeypatch):
    root = tmp_path / "butqdb"
    rec = root / "100001"
    rec.mkdir(parents=True)
    (root / "RECORDS").write_text("100001/100001_ACC\n100001/100001_ECG\n", encoding="utf-8")
    (rec / "100001_ACC.hea").write_text("header\n", encoding="utf-8")

    monkeypatch.setenv("ECG_NO_DOWNLOAD", "1")
    with pytest.raises(FileNotFoundError):
        data_downloads.ensure_wfdb_database("butqdb", root, ("100001",))


def test_but_sqi_prepare_preflights_protocol_before_writing_split(tmp_path, monkeypatch):
    missing_protocol = tmp_path / "missing_protocol"
    missing_split = tmp_path / "missing_split"
    monkeypatch.setattr(but_sqi, "protocol_dir", lambda: missing_protocol)
    monkeypatch.setattr(but_sqi, "split_dir", lambda: missing_split)

    paths = but_sqi.Paths(tmp_path / "out")
    with pytest.raises(SystemExit):
        but_sqi.cmd_prepare(paths, run=True, force=False)

    assert not paths.split_csv.exists()
