import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transformer_pipeline.data_v1_gapfill.support import build_v114_but_style_residual_hybrid as v114
from src.transformer_pipeline.data_v1_gapfill.support import run_v116_native_budget_repair as v116
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


def test_v114_normalize_frame_adds_split_before_feature_recompute(monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_recompute(frame, signals, label):
        seen["split"] = frame["split"].astype(str).tolist()
        return frame

    monkeypatch.setattr(v114.V81, "recompute_protocol_features", fake_recompute)
    monkeypatch.setattr(v114.V81, "ensure_subtype", lambda frame: frame)

    frame = pd.DataFrame({"y": [0, 1], "class_name": ["good", "medium"]})
    signals = np.zeros((2, 16), dtype=np.float32)

    out, _ = v114.normalize_frame(frame, signals, "missing split regression")

    assert seen["split"] == ["train", "train"]
    assert out["split"].astype(str).tolist() == ["train", "train"]


def test_v116_gap_fill_downsamples_oversized_public_native_pool():
    rows = []
    signals = []
    for cls in v114.CLASS_ORDER:
        for i in range(5):
            rows.append({"class_name": cls, "split": "train", "_row_pos": len(rows), "source_idx": len(rows)})
            signals.append(np.full(8, len(rows), dtype=np.float32))
    native = pd.DataFrame(rows)
    x = np.stack(signals).astype(np.float32)
    empty = native.iloc[[]].copy()
    empty_x = x[:0]

    selected, selected_x, trace = v116.select_gap_fill(
        but_train=native,
        pools={"native": native, "clean": empty, "native_morph": empty, "residual": empty},
        pool_xs={"native": x, "clean": empty_x, "native_morph": empty_x, "residual": empty_x},
        final_per_class=3,
        clean_cap=0.0,
        native_morph_min_frac=0.0,
        native_morph_selection="random",
        features=[],
        seed=123,
        swaps=0,
        support_rows=0,
        device="cpu",
        rff_dim=16,
    )

    assert len(selected) == 9
    assert selected_x.shape == (9, 8)
    assert selected["class_name"].value_counts().to_dict() == {"good": 3, "medium": 3, "bad": 3}
    assert set(trace["gap_fill_component"]) == {"original_but_downsample"}
