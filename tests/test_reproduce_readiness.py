import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transformer_pipeline.data_v1_gapfill.support import build_v114_but_style_residual_hybrid as v114
from src.transformer_pipeline.data_v1_gapfill.support import run_event_factorized_sqi_conformer as event_factorized
from src.transformer_pipeline.data_v1_gapfill.support import run_v116_native_budget_repair as v116
from src.transformer_pipeline.data_v1_gapfill import audit as gapfill_audit
from src.transformer_pipeline.data_v1_gapfill import common as gapfill_common
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


def test_v116_gap_fill_reallocates_when_residual_pool_is_short(monkeypatch):
    def fake_select_component(**kwargs):
        n = int(kwargs["final_n"])
        pool = kwargs["pool_cls"].iloc[:n].copy()
        x = kwargs["pool_x"][:n]
        trace = pd.DataFrame(
            [
                {
                    "class_name_quota": kwargs["cls"],
                    "gap_fill_component": kwargs["label"],
                    "selected_n": n,
                    "pool_n": len(kwargs["pool_cls"]),
                }
            ]
        )
        return pool, x, trace

    monkeypatch.setattr(v116, "select_component", fake_select_component)

    rows = []
    signals = []
    for cls, count in {"good": 5, "medium": 2, "bad": 2}.items():
        for _ in range(count):
            rows.append({"class_name": cls, "split": "train", "_row_pos": len(rows), "source_idx": len(rows)})
            signals.append(np.full(8, len(rows), dtype=np.float32))
    native = pd.DataFrame(rows)
    native_x = np.stack(signals).astype(np.float32)

    def pool_frame(cls: str, count: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"class_name": cls, "split": "train", "_row_pos": i, "source_idx": i}
                for i in range(count)
            ]
        )

    morph = pd.concat([pool_frame("medium", 4), pool_frame("bad", 4)], ignore_index=True)
    residual = pd.concat([pool_frame("medium", 1), pool_frame("bad", 1)], ignore_index=True)
    clean = native.iloc[[]].copy()

    selected, _, trace = v116.select_gap_fill(
        but_train=native,
        pools={"native": native, "clean": clean, "native_morph": morph, "residual": residual},
        pool_xs={
            "native": native_x,
            "clean": native_x[:0],
            "native_morph": np.zeros((len(morph), 8), dtype=np.float32),
            "residual": np.zeros((len(residual), 8), dtype=np.float32),
        },
        final_per_class=5,
        clean_cap=0.0,
        native_morph_min_frac=0.4,
        native_morph_selection="smc",
        features=[],
        seed=123,
        swaps=0,
        support_rows=0,
        device="cpu",
        rff_dim=16,
    )

    assert selected["class_name"].value_counts().to_dict() == {"good": 5, "medium": 5, "bad": 5}
    realloc = trace.loc[trace["gap_fill_component"].astype(str).eq("dynamic_shortage_reallocation")]
    assert len(realloc) == 2
    assert set(realloc["ptb_morph_n"].astype(int)) == {1}


def test_gapfill_subprocess_python_is_unbuffered():
    assert gapfill_common.unbuffer_python_command(["python", "script.py"]) == ["python", "-u", "script.py"]
    assert gapfill_common.unbuffer_python_command(["python", "-u", "script.py"]) == ["python", "-u", "script.py"]
    assert gapfill_common.unbuffer_python_command(["cmd", "/c", "echo", "ok"]) == ["cmd", "/c", "echo", "ok"]


def test_v116_original_split_keeps_extra_public_rows_in_train():
    rows = []
    for cls, counts in event_factorized.V116_ORIGINAL_SPLIT_COUNTS.items():
        total = int(sum(counts.values()))
        if cls == "medium":
            total += 51
        rows.extend({"class_name": cls} for _ in range(total))
    frame = pd.DataFrame(rows)

    split = event_factorized.v116_original_split(frame, seed=20260876)
    medium = split.loc[frame["class_name"].astype(str).eq("medium")].value_counts().to_dict()

    assert medium["val"] == event_factorized.V116_ORIGINAL_SPLIT_COUNTS["medium"]["val"]
    assert medium["test"] == event_factorized.V116_ORIGINAL_SPLIT_COUNTS["medium"]["test"]
    assert medium["train"] == event_factorized.V116_ORIGINAL_SPLIT_COUNTS["medium"]["train"] + 51


def test_gapfill_audit_accepts_public_fallback_original_surplus():
    out = {
        "protocol_class_counts": gapfill_audit.EXPECTED_PROTOCOL,
        "protocol_rows": 31590,
        "original_but_class_counts": {"bad": 5156, "good": 10530, "medium": 9212},
        "original_but_rows": 24898,
        "train_class_counts": gapfill_audit.EXPECTED_TRAIN,
        "val_test_generated_rows": 0,
        "train_generated_donor_split_problems": 0,
        "allowed_candidate_types": gapfill_audit.EXPECTED_TYPES,
        "missing_class_rows": 0,
        "missing_idx_rows": 0,
        "raw_alias_max_abs_delta": {"raw_rms_vs_rms": 0.0, "raw_ptp_vs_ptp": 0.0, "raw_diff_vs_non_qrs_diff": 0.0},
    }

    gapfill_audit.validate(out)

    assert out["expected_warnings"] == ["public fallback original_but rows exceed frozen v116 counts"]
