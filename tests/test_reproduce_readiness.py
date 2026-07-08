import sys
import os
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transformer_pipeline.data_v1_gapfill.support import build_v114_but_style_residual_hybrid as v114
from src.transformer_pipeline.data_v1_gapfill.support import run_event_factorized_sqi_conformer as event_factorized
from src.transformer_pipeline.data_v1_gapfill.support import run_v116_native_budget_repair as v116
from src.transformer_pipeline.data_v1_gapfill import audit as gapfill_audit
from src.transformer_pipeline.data_v1_gapfill import common as gapfill_common
from src.transformer_pipeline.data import but_source
from src.sqi_pipeline.qrs import setup_paper_detectors
from src.sqi_pipeline.qrs import run_qrs_cache
from src.sqi_pipeline.qrs.paper_detectors import PaperQRSExecutables
from src.sqi_pipeline.models import lm_mlp_search
from src.supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi
from src.supplemental_transformer_experiments.chapter4_evidence import audit as chapter4_audit
from src.supplemental_transformer_experiments.chapter4_evidence import but_models
from src.supplemental_transformer_experiments.chapter4_evidence import run as chapter4_run
from src.supplemental_transformer_experiments.chapter4_evidence import seta_sqi
from src.supplemental_transformer_experiments.chapter4_evidence import seta_models
from src.supplemental_transformer_experiments.chapter4_evidence.common import Paths as Chapter4Paths
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


def test_v116_gapfill_policy_alias_stays_short_for_fresh_clone_paths():
    alias = event_factorized.policy_alias("v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876")

    assert alias == "v116gap_smc"
    assert len(alias) < 16
    assert gapfill_common.SPLIT_ALIAS == f"{alias}_k1_s20260876"


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


def test_qrs_setup_auto_discovers_detector_cache(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    sibling_cache = tmp_path / "wfdb-qrs-kit" / "detector-cache" / "bin"
    sibling_cache.mkdir(parents=True)
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""
    (sibling_cache / f"wqrs{exe_suffix}").write_bytes(b"wqrs")
    (sibling_cache / f"eplimited{exe_suffix}").write_bytes(b"eplimited")

    monkeypatch.setattr(setup_paper_detectors, "project_root", lambda: root)

    assert setup_paper_detectors._auto_bin_dir(root, root / "outputs" / "tools") == sibling_cache


def test_qrs_setup_reads_documented_from_bin_env(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    bin_dir = tmp_path / "detectors"
    bin_dir.mkdir()
    exe_suffix = ".exe" if os.name == "nt" else ""
    (bin_dir / f"wqrs{exe_suffix}").write_bytes(b"wqrs")
    (bin_dir / f"eplimited{exe_suffix}").write_bytes(b"eplimited")

    monkeypatch.setenv("WFDB_QRS_KIT_FROM_BIN_DIR", str(bin_dir))

    assert setup_paper_detectors._auto_bin_dir(root, root / "outputs" / "tools") == bin_dir


def test_qrs_setup_reuses_complete_local_cache(tmp_path, monkeypatch):
    out = tmp_path / "qrs" / "tools"
    bin_dir = out / "bin"
    bin_dir.mkdir(parents=True)
    exe_suffix = ".exe" if os.name == "nt" else ""
    wqrs = bin_dir / f"wqrs{exe_suffix}"
    eplimited = bin_dir / f"eplimited{exe_suffix}"
    wqrs.write_bytes(b"wqrs")
    eplimited.write_bytes(b"eplimited")

    def forbidden_install(*args, **kwargs):
        raise AssertionError("complete local detector cache should be reused")

    monkeypatch.setattr(setup_paper_detectors, "project_root", lambda: tmp_path)
    monkeypatch.setattr(setup_paper_detectors, "install_detectors", forbidden_install)
    monkeypatch.setattr(
        setup_paper_detectors,
        "detector_status",
        lambda cache_dir: {"executables": {"wqrs": str(wqrs), "eplimited": str(eplimited)}},
    )

    result = setup_paper_detectors.run(out, require_executables=True)

    assert result["install_manifest"]["skipped"] is True
    assert Path(result["executables"]["wqrs"]) == wqrs.resolve()
    assert Path(result["executables"]["eplimited"]) == eplimited.resolve()


def test_qrs_setup_cli_summary_omits_source_urls():
    summary = setup_paper_detectors._cli_summary(
        {
            "manifest": "manifest.json",
            "note": "note.md",
            "manager": "wfdb-qrs-kit",
            "cache_dir": "cache",
            "auto_from_bin_dir": "bin",
            "executables": {"wqrs": "wqrs.exe", "eplimited": "eplimited.exe"},
            "install_manifest": {
                "sources": [{"url": "https://physionet.org/files/wfdb/10.7.0/app/wqrs.c"}],
            },
        }
    )

    text = json.dumps(summary)
    assert "physionet.org" not in text
    assert summary["install_manifest"]["source_count"] == 1


def test_paper_qrs_cache_auto_runs_detector_setup(tmp_path, monkeypatch):
    split_csv = tmp_path / "split.csv"
    split_csv.write_text("record_id\nr1\n", encoding="utf-8")
    resampled = tmp_path / "resampled"
    resampled.mkdir()
    np.savez(
        resampled / "r1.npz",
        sig_125=np.zeros((32, 12), dtype=np.float32),
        fs=np.array(125),
        leads=np.array(run_qrs_cache.LEADS_12, dtype=object),
    )
    out = tmp_path / "qrs"
    calls: list[Path] = []

    def fake_setup(path, **kwargs):
        calls.append(Path(path))
        assert kwargs["require_executables"] is True
        return {}

    monkeypatch.setattr(run_qrs_cache.setup_paper_detectors, "run", fake_setup)
    monkeypatch.setattr(
        run_qrs_cache,
        "resolve_paper_qrs_executables",
        lambda params, work_dir: PaperQRSExecutables(Path("wqrs"), Path("eplimited")),
    )
    monkeypatch.setattr(
        run_qrs_cache,
        "run_paper_qrs_12lead",
        lambda **kwargs: ([np.array([8]) for _ in run_qrs_cache.LEADS_12], [np.array([9]) for _ in run_qrs_cache.LEADS_12]),
    )

    run_qrs_cache.run(
        {
            "split_csv": str(split_csv),
            "resampled_dir": str(resampled),
            "out_dir": str(out),
            "detector_profile": "paper",
            "force": True,
        }
    )

    assert calls == [out / "tools"]
    assert (out / "r1.npz").exists()


def test_but_source_audit_reads_long_windows_paths(tmp_path):
    long_root = tmp_path / ("a" * 80) / ("b" * 80) / ("c" * 80)
    analysis = long_root / "analysis" / "good_medium_geometry_repair"
    protocol = analysis / "clean_but_protocols" / "margin_ge_5s_drop_outlier" / "original_region_atlas.csv"
    support = (
        analysis
        / "clean_but_protocols"
        / but_source.SUPPORT_POLICY
        / "original_region_atlas.csv"
    )
    assets = long_root / "source" / "cleanbut_support_assets.json"
    for path in [protocol, support, assets]:
        os.makedirs(but_source.long_path(path.parent), exist_ok=True)

    frame = pd.DataFrame({"class_name": ["good", "medium", "bad"]})
    frame.to_csv(but_source.long_path(protocol), index=False)
    frame.to_csv(but_source.long_path(support), index=False)
    with open(but_source.long_path(assets), "w", encoding="utf-8") as f:
        f.write("{}")

    cfg = SimpleNamespace(analysis_dir=analysis, artifacts_dir=long_root)
    out = but_source.audit_source(cfg)

    assert out["candidate_gap5_rows"] == 3
    assert out["support_pool_rows"] == 3
    assert out["historical_support_exact"] is False


def test_chapter4_out_path_does_not_duplicate_outputs_prefix(monkeypatch):
    root = Path(r"C:\repo") if os.name == "nt" else Path("/repo")
    monkeypatch.setattr(chapter4_run, "project_root", lambda: root)
    monkeypatch.setattr(chapter4_run, "OUT_DEFAULT", root / "outputs" / "transformer" / "supplemental" / "chapter4_evidence_work")

    args = SimpleNamespace(out="outputs/transformer/supplemental/chapter4_evidence")
    paths = chapter4_run._paths(args)

    assert paths.out == root / "outputs" / "transformer" / "supplemental" / "chapter4_evidence"


def test_chapter4_seta_audit_scope_does_not_require_but(monkeypatch, tmp_path):
    calls: list[str] = []

    monkeypatch.setattr(chapter4_audit, "_seta_audit", lambda paths: {"split_counts": {"train_acceptable": 1}})

    def forbidden_but():
        calls.append("but")
        raise AssertionError("BUT audit should not run in Set-A scope")

    monkeypatch.setattr(chapter4_audit, "_but_audit", forbidden_but)

    paths = Chapter4Paths(tmp_path / "chapter4")
    out = chapter4_audit.run(paths, execute=True, scope="seta")

    assert calls == []
    assert out["scope"] == "seta"
    assert out["outputs"]["seta"]["split_counts"] == {"train_acceptable": 1}
    assert out["outputs"]["scope"] == "seta"
    assert "but" not in out["outputs"]


def test_seta_sqi_features_ready_accepts_leadwise_qrs_summary(tmp_path):
    root = tmp_path / "arm"
    (root / "splits").mkdir(parents=True)
    (root / "qrs").mkdir()
    (root / "features").mkdir()
    pd.DataFrame({"record_id": ["r00000", "r00001"]}).to_csv(root / "splits" / "split.csv", index=False)
    pd.DataFrame(
        [{"record_id": rid} for rid in ["r00000", "r00001"] for _ in seta_sqi.LEADS_12]
    ).to_csv(root / "qrs" / "qrs_summary.csv", index=False)
    pd.DataFrame(
        [{"record_id": rid, **{f"lead{i}__sSQI": 0.0 for i in range(84)}} for rid in ["r00000", "r00001"]]
    ).to_parquet(root / "features" / "record84_norm.parquet", index=False)
    (root / "features" / "record84.parquet").write_bytes(b"exists")

    assert seta_sqi._features_ready(root)


def test_seta_sqi_seed_original_qrs_counts_existing_cache(tmp_path):
    native = tmp_path / "native"
    arm = tmp_path / "fixed"
    (native / "qrs").mkdir(parents=True)
    (arm / "qrs").mkdir(parents=True)
    (arm / "splits").mkdir()
    pd.DataFrame(
        {
            "record_id": ["r00000", "r00001", "r00002"],
            "is_augmented": [0, 0, 1],
        }
    ).to_csv(arm / "splits" / "split.csv", index=False)
    (arm / "qrs" / "r00000.npz").write_bytes(b"already here")
    (native / "qrs" / "r00001.npz").write_bytes(b"copy me")

    assert seta_sqi._seed_original_qrs(arm, native) == 2
    assert (arm / "qrs" / "r00001.npz").read_bytes() == b"copy me"


def test_lm_mlp_tables_required_for_tables_skip(tmp_path):
    paths = lm_mlp_search.Paths(
        root=tmp_path,
        features_parquet=tmp_path / "record84_norm.parquet",
        split_csv=tmp_path / "split.csv",
        out_dir=tmp_path / "lm_mlp",
        models_dir=tmp_path / "lm_mlp" / "models",
        tables_dir=tmp_path / "lm_mlp" / "tables",
        roc_dir=tmp_path / "lm_mlp" / "roc",
        probs_dir=tmp_path / "lm_mlp" / "probs",
        maxacc_dir=tmp_path / "lm_mlp" / "maxacc",
    )
    paths.mkdirs()
    seed = 0
    (paths.models_dir / f"search_J_results_seed{seed}.csv").write_text("J,val_acc\n2,0.5\n", encoding="utf-8")
    (paths.models_dir / f"best_J_seed{seed}.json").write_text("{}", encoding="utf-8")
    (paths.models_dir / f"model_84-6-1_seed{seed}.pkl").write_bytes(b"model")
    (paths.out_dir / f"lm_mlp_test_metrics_seed{seed}.json").write_text("{}", encoding="utf-8")

    assert lm_mlp_search.outputs_exist(paths, seed, tables=False)
    assert not lm_mlp_search.outputs_exist(paths, seed, tables=True)

    for table in lm_mlp_search._expected_table_outputs(paths, seed):
        table.write_text("metric,value\nacc,1\n", encoding="utf-8")

    assert lm_mlp_search.outputs_exist(paths, seed, tables=True)


def test_seta_models_runs_svm_before_reading_svm_rows(tmp_path, monkeypatch):
    paths = Chapter4Paths(tmp_path / "chapter4")
    calls: list[str] = []

    def row(model: str) -> dict[str, str]:
        return {"run_id": model, "construction": "smc_gapfill", "model": model}

    monkeypatch.setattr(seta_models, "ARMS", ("smc_gapfill",))
    monkeypatch.setattr(seta_models, "_row_from_source_only_svm", lambda arm, root: row("source-only"))

    def fake_run_svm(root, out, *, force):
        calls.append("run-svm")
        out.mkdir(parents=True, exist_ok=True)

    def fake_row_from_svm(arm, model, out):
        assert "run-svm" in calls
        calls.append(f"read-{model}")
        return row(model)

    monkeypatch.setattr(seta_models, "_run_svm", fake_run_svm)
    monkeypatch.setattr(seta_models, "_row_from_svm", fake_row_from_svm)
    monkeypatch.setattr(seta_models, "_run_mlp", lambda root, out, *, force, device: calls.append("run-mlp"))
    monkeypatch.setattr(seta_models, "_row_from_mlp", lambda out: row("mlp"))
    monkeypatch.setattr(seta_models, "_row_from_conformer", lambda paths, *, force, device: row("conformer"))

    out = seta_models.run(paths, execute=True, force=False, device="cpu")

    assert calls[0] == "run-svm"
    assert out["outputs"]["model_comparison"][0]["model"] == "SQI SVM-RBF selected5"
    assert (paths.tables / "seta_repaired_model_comparison.csv").exists()


def test_seta_conformer_cpu_uses_pretrained_instead_of_training(tmp_path, monkeypatch):
    paths = Chapter4Paths(tmp_path / "chapter4")
    checkpoint_dir = tmp_path / "pretrained" / "seta_e31_leadwise_shared"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_model.pt").write_bytes(b"checkpoint")
    (checkpoint_dir / "history.csv").write_text("epoch,val_acc\n1,1.0\n", encoding="utf-8")
    calls: list[tuple[str, Path, str]] = []

    def fake_predict(sp, *, run, checkpoint_dir, device):
        calls.append(("predict", Path(checkpoint_dir), device))
        sp.models.mkdir(parents=True, exist_ok=True)
        (sp.models / "metrics.json").write_text("{}", encoding="utf-8")
        (sp.models / "predictions.csv").write_text("split,y,prob_acceptable,pred_fixed\n", encoding="utf-8")

    def forbidden_train(*args, **kwargs):
        raise AssertionError("CPU reproduction must not retrain Set-A Conformer by default")

    val_met = {
        "acc": 1.0,
        "macro_f1": 1.0,
        "acceptable_recall": 1.0,
        "original_unacceptable_recall": 1.0,
    }
    test_met = {
        "threshold": 0.5,
        "acc": 1.0,
        "auc": 1.0,
        "acceptable_recall": 1.0,
        "original_unacceptable_recall": 1.0,
        "confusion": {"tn": 1, "fp": 0, "fn": 0, "tp": 1},
    }

    monkeypatch.delenv("ECG_ALLOW_CPU_CONFORMER_TRAIN", raising=False)
    monkeypatch.setattr(seta_models, "SETA_E31_DIR", checkpoint_dir)
    monkeypatch.setattr(seta_models.sqi12, "cmd_predict_from_checkpoint", fake_predict)
    monkeypatch.setattr(seta_models.sqi12, "cmd_train", forbidden_train)
    monkeypatch.setattr(seta_models, "_acceptability_preserving_threshold", lambda path: (0.5, val_met, test_met))
    monkeypatch.setattr(seta_models, "_fixed_original_bad_recall", lambda path: {})

    row = seta_models._row_from_conformer(paths, force=True, device="cpu")

    assert calls == [("predict", checkpoint_dir, "cpu")]
    assert row["run_id"] == "seta_smc_gapfill_e31style_waveform"


def test_but_e31_predictions_falls_back_to_pretrained(tmp_path, monkeypatch):
    pretrained = tmp_path / "but_e31_query_mean_fused_conformer"
    pretrained.mkdir(parents=True)
    np.savez(pretrained / "test_predictions.npz", y=np.array([0]), probs=np.array([[1.0, 0.0, 0.0]]))

    monkeypatch.setattr(but_models, "E31_QUERY_MEAN_ARTIFACT", "missing_artifact_for_test")
    monkeypatch.setattr(but_models, "BUT_E31_QUERY_MEAN_DIR", pretrained)

    assert but_models._find_e31_predictions() == (pretrained / "test_predictions.npz").resolve()
