from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ecg_sqi_inference import __main__ as cli
from src.ecg_sqi_inference import models
from src.ecg_sqi_inference.core import (
    as_samples_by_lead,
    iter_input_files,
    predict_records,
    read_record,
    resample_signal,
    segment_signal,
)
from src.ecg_sqi_inference.models import get_predictor


class DummyPredictor:
    """Return deterministic usable labels for service-contract tests."""

    name = "dummy"
    n_leads = 1

    def predict(self, segments: np.ndarray) -> pd.DataFrame:
        """Return one fixed prediction row for each segment."""

        return pd.DataFrame(
            {
                "raw_class": ["acceptable"] * len(segments),
                "display_class": ["usable"] * len(segments),
                "prob_acceptable": np.ones(len(segments)),
            }
        )


def test_lead_count_validation_rejects_wrong_shape():
    with pytest.raises(ValueError, match="single-lead"):
        as_samples_by_lead(np.zeros((1250, 12), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="12-lead"):
        as_samples_by_lead(np.zeros(1250, dtype=np.float32), 12)


def test_lead_count_orients_supported_shapes():
    single = np.zeros((1, 1250), dtype=np.float64)
    twelve = np.zeros((12, 1250), dtype=np.float64)

    assert as_samples_by_lead(single, 1).shape == (1250, 1)
    assert as_samples_by_lead(twelve, 12).shape == (1250, 12)
    assert as_samples_by_lead(twelve, 12).dtype == np.float32
    with pytest.raises(ValueError, match="unsupported lead count"):
        as_samples_by_lead(single, 2)


def test_read_record_supports_numpy_csv_and_mocked_wfdb(tmp_path, monkeypatch):
    np.save(tmp_path / "array.npy", np.arange(6).reshape(3, 2))
    np.savez(tmp_path / "archive.npz", other=np.arange(4))
    pd.DataFrame({"label": ["a", "b"], "ecg": [1, 2], "ignored_numeric": [3, 4]}).to_csv(
        tmp_path / "table.csv", index=False
    )
    (tmp_path / "wave.hea").write_text("mock", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "wfdb",
        SimpleNamespace(rdsamp=lambda stem: (np.ones((5, 2)), {"stem": stem})),
    )

    assert read_record(tmp_path / "array.npy").signal.shape == (3, 2)
    assert read_record(tmp_path / "archive.npz").signal.shape == (4,)
    assert read_record(tmp_path / "table.csv").signal.shape == (2, 1)
    wfdb_record = read_record(tmp_path / "wave.hea")
    assert wfdb_record.signal.shape == (5, 2)
    assert wfdb_record.signal.dtype == np.float32


def test_read_record_rejects_unsupported_or_invalid_arrays(tmp_path):
    (tmp_path / "signal.txt").write_text("1", encoding="utf-8")
    np.save(tmp_path / "cube.npy", np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="unsupported input type"):
        read_record(tmp_path / "signal.txt")
    with pytest.raises(ValueError, match="expected 1D or 2D"):
        read_record(tmp_path / "cube.npy")


def test_read_record_rejects_pickled_npz_object_arrays(tmp_path):
    path = tmp_path / "object.npz"
    np.savez(path, signal=np.array([{"unsafe": True}], dtype=object))

    with pytest.raises(ValueError, match="allow_pickle=False"):
        read_record(path)


def test_iter_input_files_is_recursive_sorted_and_filtered(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    for path in [tmp_path / "z.npy", nested / "a.csv", tmp_path / "ignored.txt"]:
        path.write_text("", encoding="utf-8")

    found = iter_input_files(tmp_path)

    assert found == sorted([nested / "a.csv", tmp_path / "z.npy"])
    assert iter_input_files(found[0]) == [found[0]]
    with pytest.raises(FileNotFoundError):
        iter_input_files(tmp_path / "missing")


def test_segment_signal_drops_short_tail():
    signal = np.zeros((125 * 25, 1), dtype=np.float32)

    segments, dropped = segment_signal(signal)

    assert segments.shape == (2, 1250, 1)
    assert dropped == pytest.approx(5.0)


def test_resample_signal_changes_sample_count_and_validates_frequency():
    signal = np.zeros((2500, 2), dtype=np.float64)

    result = resample_signal(signal, fs=250)

    assert result.shape == (1250, 2)
    assert result.dtype == np.float32
    with pytest.raises(ValueError, match="positive"):
        resample_signal(signal, fs=0)


def test_segment_signal_reports_all_of_a_short_record_as_dropped():
    segments, dropped = segment_signal(np.zeros((625, 1), dtype=np.float32))

    assert segments.shape == (0, 1250, 1)
    assert dropped == pytest.approx(5.0)


def test_predict_records_writes_per_record_and_index(tmp_path):
    inp = tmp_path / "inputs"
    out = tmp_path / "out"
    inp.mkdir()
    np.savez(inp / "r1.npz", signal=np.zeros((500 * 25,), dtype=np.float32))
    np.savez(inp / "r2.npz", signal=np.zeros((500 * 10,), dtype=np.float32))

    summary = predict_records(input_path=inp, out_dir=out, fs=500, predictor=DummyPredictor())

    assert summary["records_ok"] == 2
    assert summary["segments"] == 3
    assert (out / "r1_segments.csv").exists()
    assert (out / "r2_segments.csv").exists()
    assert json.loads((out / "run_summary.json").read_text(encoding="utf-8")) == summary
    all_segments = pd.read_csv(out / "all_segments.csv")
    assert all_segments[["record_id", "segment_index", "start_sec", "end_sec"]].to_dict("records") == [
        {"record_id": "r1", "segment_index": 0, "start_sec": 0.0, "end_sec": 10.0},
        {"record_id": "r1", "segment_index": 1, "start_sec": 10.0, "end_sec": 20.0},
        {"record_id": "r2", "segment_index": 0, "start_sec": 0.0, "end_sec": 10.0},
    ]


def test_predict_records_exits_nonzero_on_any_bad_file(tmp_path):
    inp = tmp_path / "inputs"
    out = tmp_path / "out"
    inp.mkdir()
    np.savez(inp / "good.npz", signal=np.zeros((1250,), dtype=np.float32))
    np.savez(inp / "bad.npz", signal=np.zeros((1250, 12), dtype=np.float32))

    with pytest.raises(SystemExit):
        predict_records(input_path=inp, out_dir=out, fs=125, predictor=DummyPredictor())

    assert (out / "good_segments.csv").exists()
    assert "bad.npz" in (out / "run_summary.json").read_text(encoding="utf-8")


def test_predict_records_rejects_duplicate_stems_before_writing_outputs(tmp_path):
    inp = tmp_path / "inputs"
    out = tmp_path / "out"
    (inp / "first").mkdir(parents=True)
    (inp / "second").mkdir()
    np.save(inp / "first" / "same.npy", np.zeros(1250, dtype=np.float32))
    np.save(inp / "second" / "same.npy", np.zeros(1250, dtype=np.float32))

    with pytest.raises(ValueError, match=r"duplicate record_id\(s\): same"):
        predict_records(input_path=inp, out_dir=out, fs=125, predictor=DummyPredictor())

    assert not (out / "same_segments.csv").exists()
    assert not (out / "all_segments.csv").exists()
    assert not (out / "run_summary.json").exists()


def test_predict_records_ignores_nested_output_csvs_on_second_run(tmp_path):
    inp = tmp_path / "inputs"
    out = inp / "results"
    inp.mkdir()
    np.save(inp / "record.npy", np.zeros(1250, dtype=np.float32))

    first = predict_records(input_path=inp, out_dir=out, fs=125, predictor=DummyPredictor())
    (out / "previous_segments.csv").write_text("not,an,ecg\n", encoding="utf-8")
    second = predict_records(input_path=inp, out_dir=out, fs=125, predictor=DummyPredictor())

    assert first["records_ok"] == second["records_ok"] == 1
    assert second["records_failed"] == 0
    assert second["segments"] == 1
    assert [row["record_id"] for row in second["records"]] == ["record"]
    assert len(pd.read_csv(out / "all_segments.csv")) == 1
    assert (out / "previous_segments.csv").read_text(encoding="utf-8") == "not,an,ecg\n"


def test_cli_predict_wires_arguments_and_prints_summary(tmp_path, monkeypatch, capsys):
    predictor = DummyPredictor()
    seen = {}
    events = []
    expected = {"records_ok": 1, "records_failed": 0, "segments": 2}

    def fake_predict_records(**kwargs):
        events.append("predict")
        seen.update(kwargs)
        return expected

    def fake_get_predictor(model, device):
        events.append("load")
        seen.update(model=model, device=device)
        return predictor

    monkeypatch.setattr(cli, "verify_inference_bundles", lambda: events.append("verify"))
    monkeypatch.setattr(cli, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(cli, "predict_records", fake_predict_records)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ecg-sqi",
            "predict",
            "--model",
            "singlelead-rbfsvm",
            "--input",
            str(tmp_path / "input.npy"),
            "--fs",
            "500",
            "--out",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ],
    )

    cli.main()

    assert events == ["verify", "load", "predict"]
    assert seen == {
        "model": "singlelead-rbfsvm",
        "device": "cpu",
        "input_path": tmp_path / "input.npy",
        "out_dir": tmp_path / "out",
        "fs": 500.0,
        "predictor": predictor,
    }
    assert json.loads(capsys.readouterr().out) == expected


def test_cli_verify_bundles_prints_verification_summary(monkeypatch, capsys):
    expected = {"status": "ok", "schema": 1, "models": {"singlelead-conformer": ["model.pt"]}}
    monkeypatch.setattr(cli, "verify_inference_bundles", lambda: expected)
    monkeypatch.setattr(sys, "argv", ["ecg-sqi", "verify-bundles"])

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected


def test_get_predictor_selects_public_models(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(models, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        models,
        "Conformer12Predictor",
        lambda path, device: calls.append(("conformer", path, device)) or "conformer",
    )
    monkeypatch.setattr(
        models,
        "Conformer1Predictor",
        lambda path, device: calls.append(("conformer1", path, device)) or "conformer1",
    )
    monkeypatch.setattr(
        models,
        "RBFSVMBundlePredictor",
        lambda path, name, leads: calls.append(("rbfsvm", path, name, leads)) or name,
    )

    assert models.get_predictor("12lead-conformer", device="cuda") == "conformer"
    assert models.get_predictor("singlelead-conformer", device="cuda") == "conformer1"
    assert models.get_predictor("12lead-rbfsvm") == "12lead-rbfsvm"
    assert models.get_predictor("singlelead-rbfsvm") == "singlelead-rbfsvm"
    assert calls == [
        ("conformer", tmp_path / "pretrained/chapter4/seta_e31_leadwise_shared", "cuda"),
        ("conformer1", tmp_path / "pretrained/inference/singlelead-conformer", "cuda"),
        ("rbfsvm", tmp_path / "pretrained/inference/12lead-rbfsvm", "12lead-rbfsvm", 12),
        ("rbfsvm", tmp_path / "pretrained/inference/singlelead-rbfsvm", "singlelead-rbfsvm", 1),
    ]


def test_get_predictor_rejects_unknown_or_missing_bundle(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unknown model"):
        get_predictor("missing")

    monkeypatch.setattr(models, "project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="profile.json"):
        models.get_predictor("12lead-rbfsvm")


def test_export_inference_bundles_omits_unavailable_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "project_root", lambda: tmp_path / "empty-repository")
    out = tmp_path / "bundles"

    assert models.export_inference_bundles(out) == {}
    assert out.is_dir()


def test_verify_inference_bundles_accepts_matching_artifacts(tmp_path, monkeypatch):
    profile = tmp_path / "pretrained/inference/demo/profile.json"
    artifact = tmp_path / "pretrained/inference/demo/model.bin"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}", encoding="utf-8")
    artifact.write_bytes(b"frozen model")
    manifest = {
        "schema": 1,
        "models": {
            "demo": {
                "profile": "pretrained/inference/demo/profile.json",
                "artifacts": {
                    "pretrained/inference/demo/model.bin": hashlib.sha256(b"frozen model").hexdigest()
                },
            }
        },
    }
    (tmp_path / "pretrained/inference/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(models, "project_root", lambda: tmp_path)

    assert models.verify_inference_bundles() == {
        "status": "ok",
        "schema": 1,
        "models": {"demo": ["pretrained/inference/demo/model.bin"]},
    }


def test_verify_inference_bundles_rejects_missing_artifact(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "pretrained/inference"
    profile = manifest_dir / "demo/profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}", encoding="utf-8")
    manifest = {
        "schema": 1,
        "models": {
            "demo": {
                "profile": "pretrained/inference/demo/profile.json",
                "artifacts": {"pretrained/inference/demo/missing.bin": "0" * 64},
            }
        },
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(models, "project_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="missing.bin"):
        models.verify_inference_bundles()


def test_verify_inference_bundles_rejects_hash_mismatch(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "pretrained/inference"
    artifact = manifest_dir / "demo/model.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"changed model")
    manifest = {
        "schema": 1,
        "models": {
            "demo": {
                "artifacts": {"pretrained/inference/demo/model.bin": "0" * 64},
            }
        },
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(models, "project_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="hash mismatch: pretrained/inference/demo/model.bin"):
        models.verify_inference_bundles()


def test_singlelead_conformer_predicts_three_class_probabilities():
    seen = {}

    def make_channels(raw, stats):
        seen["raw_shape"] = raw.shape
        seen["stats"] = stats
        return np.zeros((len(raw), 8, 1250), dtype=np.float32)

    class FakeModel:
        def __call__(self, channels):
            seen["channel_shape"] = tuple(channels.shape)
            return {"probs": models.torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])}

    predictor = models.Conformer1Predictor.__new__(models.Conformer1Predictor)
    predictor._channel_stats = object()
    predictor._cfg = {"width": 16}
    predictor._device = models.torch.device("cpu")
    predictor._gm = SimpleNamespace(
        ACTIVE_CFG=None,
        EVT=SimpleNamespace(DUAL=SimpleNamespace(make_dualview_channels=make_channels)),
    )
    predictor._model = FakeModel()

    result = predictor.predict(np.zeros((2, 1250, 1), dtype=np.float32))

    assert result.shape == (2, 5)
    assert result.columns.tolist() == ["raw_class", "display_class", "prob_good", "prob_medium", "prob_bad"]
    assert result["raw_class"].tolist() == ["good", "bad"]
    assert result[["prob_good", "prob_medium", "prob_bad"]].sum(axis=1).to_numpy() == pytest.approx([1, 1])
    assert seen["raw_shape"] == (2, 1250)
    assert seen["channel_shape"] == (2, 8, 1250)
    assert predictor._gm.ACTIVE_CFG == predictor._cfg


def test_rbfsvm_predictor_maps_binary_probabilities_to_public_labels(monkeypatch):
    predictor = models.RBFSVMBundlePredictor.__new__(models.RBFSVMBundlePredictor)
    predictor._feature_columns = ["f"]
    predictor._profile = {"classes": ["unacceptable", "acceptable"], "poor_threshold": 0.6}
    predictor._classes = predictor._profile["classes"]
    predictor.n_leads = 12
    predictor._model = SimpleNamespace(
        classes_=np.array(["unacceptable", "acceptable"]),
        predict_proba=lambda x: np.array([[0.75, 0.25], [0.2, 0.8]]),
    )
    monkeypatch.setattr(models, "feature_frame", lambda segments, leads, profile: pd.DataFrame({"f": [1, 2]}))

    result = predictor.predict(np.zeros((2, 1250, 12), dtype=np.float32))

    assert result["raw_class"].tolist() == ["unacceptable", "acceptable"]
    assert result["display_class"].tolist() == ["unusable", "usable"]
    assert result["prob_acceptable"].tolist() == [0.25, 0.8]
    assert result["prob_unacceptable"].tolist() == pytest.approx([0.75, 0.2])


def test_rbfsvm_predictor_loads_profile_and_joblib_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    profile = {"classes": ["good", "medium", "bad"], "feature_columns": ["f"], "norm_stats": {}}
    (bundle / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    estimator = object()
    seen = {}
    monkeypatch.setitem(
        sys.modules,
        "joblib",
        SimpleNamespace(load=lambda path: seen.setdefault(str(path), {"estimator": estimator})),
    )

    predictor = models.RBFSVMBundlePredictor(bundle, "singlelead-rbfsvm", 1)

    assert predictor._profile == profile
    assert predictor._model is estimator
    assert seen == {str(bundle / "model.joblib"): {"estimator": estimator}}


def test_rbfsvm_predictor_maps_three_class_probabilities(monkeypatch):
    predictor = models.RBFSVMBundlePredictor.__new__(models.RBFSVMBundlePredictor)
    predictor._feature_columns = ["f"]
    predictor._profile = {"classes": ["good", "medium", "bad"]}
    predictor._classes = predictor._profile["classes"]
    predictor.n_leads = 1
    predictor._model = SimpleNamespace(
        classes_=np.array(["good", "medium", "bad"]),
        predict_proba=lambda x: np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
    )
    monkeypatch.setattr(models, "feature_frame", lambda segments, leads, profile: pd.DataFrame({"f": [1, 2]}))

    result = predictor.predict(np.zeros((2, 1250, 1), dtype=np.float32))

    assert result.to_dict("list") == {
        "raw_class": ["good", "bad"],
        "display_class": ["good", "bad"],
        "prob_good": [0.8, 0.1],
        "prob_medium": [0.1, 0.2],
        "prob_bad": [0.1, 0.7],
    }


def test_feature_frame_normalizes_single_lead_features(monkeypatch):
    fake_module = SimpleNamespace(
        _compute_one=lambda task: (task[0], {"single__sqi": 5.0}, None, None)
    )
    monkeypatch.setitem(
        sys.modules,
        "src.supplemental_transformer_experiments.but_sqi_baseline.run",
        fake_module,
    )
    profile = {
        "norm_stats": {
            "columns": ["single__sqi"],
            "median_train": {"single__sqi": 1.0},
            "std_train": {"single__sqi": 2.0},
        }
    }

    result = models.feature_frame(np.zeros((2, 1250, 1), dtype=np.float32), 1, profile)

    assert result["single__sqi"].tolist() == [2.0, 2.0]


def test_current_but_conformer_checkpoint_can_predict_one_segment():
    bundle = Path("pretrained/inference/singlelead-conformer")
    profile_path = bundle / "profile.json"
    if not profile_path.exists():
        pytest.skip("single-lead Conformer inference profile not present")
    checkpoint = Path(json.loads(profile_path.read_text(encoding="utf-8"))["checkpoint"])
    if not checkpoint.exists():
        pytest.skip("current BUT Conformer checkpoint not present")

    predictor = models.Conformer1Predictor(bundle)
    out = predictor.predict(np.zeros((1, 1250, 1), dtype=np.float32))

    assert list(out.columns) == ["raw_class", "display_class", "prob_good", "prob_medium", "prob_bad"]
    assert len(out) == 1


def test_singlelead_rbfsvm_outputs_probabilities():
    bundle = Path("pretrained/inference/singlelead-rbfsvm")
    if not (bundle / "model.joblib").exists():
        pytest.skip("single-lead RBF-SVM inference bundle not exported")

    predictor = get_predictor("singlelead-rbfsvm")
    out = predictor.predict(np.zeros((1, 1250, 1), dtype=np.float32))

    assert {"prob_good", "prob_medium", "prob_bad"}.issubset(out.columns)
    assert float(out[["prob_good", "prob_medium", "prob_bad"]].iloc[0].sum()) == pytest.approx(1.0)
