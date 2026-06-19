"""Evaluate saved waveform-student checkpoints on bucketed original BUT.

External-only diagnostic. This script loads PTB-trained waveform-only student
checkpoints and emits original-BUT report-only buckets plus feature recovery.
It does not train, tune thresholds on original, or use original rows for model
selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"
NODE_ID = "N17043_gm_probe"

DEFAULT_CANDIDATES = [
    "featurefirst_top20_rawbeat_artifact_auxctx_dual_balanced_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_veto_finetune_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_veto_margin_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_jointartifact_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_jointartifact_badrecall_a050",
    "featurefirst_top20_rawbeat_artifact_auxctx_router_event_mediumrecall_a050",
    "predfailaxis_sqiquery_subject111_impulsebad_dual_p20",
    "predtop14balanced_sqiquery_subject111_impulsebad_dual_p20",
    "wavecomp_sqiquery_p20",
    "wavecomp_sqiquery_v5_p20_badguard",
    "p20_sqiquery_primctx_v5_badguard",
    "detbase_tokens_qfeatbin_badguard",
    "detbase_rrtokens_qfeatbin_badguard",
]


def load_student_module() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_geometry_student", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-name", type=str, default="waveform_student_checkpoint_eval")
    args = parser.parse_args()

    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    student = load_student_module()
    out_dir = REPORT_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for name in names:
        ckpt_path = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search" / name / "ckpt_best.pt"
        if not ckpt_path.exists():
            print(f"skip missing {name}: {ckpt_path}", flush=True)
            continue
        try:
            cand_rows, cand_feature_rows = evaluate_checkpoint(student, name, ckpt_path, args)
        except Exception as exc:
            print(f"skip failed {name}: {type(exc).__name__}: {exc}", flush=True)
            rows.append(
                {
                    "candidate": name,
                    "run_label": "eval_existing",
                    "bucket": "load_failed",
                    "n": 0,
                    "acc": np.nan,
                    "macro_f1": np.nan,
                    "good_recall": np.nan,
                    "medium_recall": np.nan,
                    "bad_recall": np.nan,
                    "good_precision": np.nan,
                    "medium_precision": np.nan,
                    "bad_precision": np.nan,
                    "good_to_medium": np.nan,
                    "medium_to_good": np.nan,
                    "bad_to_medium": np.nan,
                    "confusion_3x3": str(exc),
                    "teacher_mae": np.nan,
                    "teacher_core_mae": np.nan,
                }
            )
            continue
        rows.extend(cand_rows)
        feature_rows.extend(cand_feature_rows)
        pd.DataFrame(rows).to_csv(out_dir / "metrics.partial.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(out_dir / "feature_recovery.partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    features = pd.DataFrame(feature_rows)
    metrics_path = out_dir / "metrics.csv"
    features_path = out_dir / "feature_recovery.csv"
    summary_path = out_dir / "summary.json"
    metrics.to_csv(metrics_path, index=False)
    features.to_csv(features_path, index=False)
    payload = {
        "input_contract": "waveform checkpoint only; original BUT report-only",
        "candidates": names,
        "metrics_csv": str(metrics_path),
        "feature_recovery_csv": str(features_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not metrics.empty:
        show = (
            metrics[metrics["bucket"].eq("original_test_all_10s+")]
            .sort_values("acc", ascending=False)
            .head(30)
        )
        print(show.to_csv(index=False), flush=True)


def evaluate_checkpoint(
    student: Any,
    name: str,
    ckpt_path: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"evaluating {name}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    base_train_ds = student.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    norm = student.ARCH.FeatureNorm(mean=base_train_ds.mean, std=base_train_ds.std)
    val_ds = student.ARCH.SyntheticWaveDataset("val", str(cfg["channels"]))
    original_ds = student.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    pin = device.type == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    original_loader = DataLoader(
        original_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    model = student.GeometryStudent(cfg, int(base_train_ds.x.shape[1])).to(device)
    configure_optional_state(student, model, cfg, base_train_ds, ckpt)
    model.load_state_dict(ckpt["model_state"], strict=True)
    val_payload = student.eval_loader(model, val_loader, device)
    threshold = student.ARCH.calibrate_bad_threshold(val_ds.y, val_payload.probs)
    original_payload = student.eval_loader(model, original_loader, device)
    rows: list[dict[str, Any]] = []
    for bucket in [
        "original_test_all_10s+",
        "original_all_10s+",
        "bad_core_nearboundary",
        "bad_outlier_stress",
    ]:
        rows.append(
            student.metric_row(
                name,
                "eval_existing",
                bucket,
                student.ARCH.bucket_report(original_ds, original_payload.probs, bucket),
            )
        )
        rows.append(
            student.metric_row(
                f"{name}_badcal",
                "eval_existing",
                bucket,
                student.ARCH.bucket_report(original_ds, original_payload.probs, bucket, threshold),
            )
        )
    feature_rows = student.feature_recovery_rows(
        name,
        "eval_existing",
        "original_all_10s+",
        original_payload,
        original_ds.aux,
    )
    return rows, feature_rows


def configure_optional_state(student: Any, model: torch.nn.Module, cfg: dict[str, Any], base_train_ds: Any, ckpt: dict[str, Any]) -> None:
    if getattr(model, "use_teacher_atlas", False):
        if ckpt.get("teacher_prototypes") is not None:
            model.set_teacher_prototypes(np.asarray(ckpt["teacher_prototypes"], dtype=np.float32))
        else:
            prototypes = student.build_teacher_prototypes(
                base_train_ds.aux,
                base_train_ds.y,
                int(cfg["teacher_prototypes_per_class"]),
                str(cfg.get("teacher_prototype_mode", "pc1_quantile")),
            )
            model.set_teacher_prototypes(prototypes)
    if getattr(model, "use_waveform_atlas", False):
        if ckpt.get("waveform_atlas_prototypes") is not None:
            model.set_waveform_atlas(
                np.asarray(ckpt["waveform_atlas_prototypes"], dtype=np.float32),
                np.asarray(ckpt["waveform_atlas_mean"], dtype=np.float32),
                np.asarray(ckpt["waveform_atlas_std"], dtype=np.float32),
            )
        else:
            wave_proto, wave_mean, wave_std = student.build_waveform_atlas(
                base_train_ds,
                int(cfg["waveform_atlas_prototypes_per_class"]),
                str(cfg.get("waveform_atlas_mode", "farthest")),
            )
            model.set_waveform_atlas(wave_proto, wave_mean, wave_std)


if __name__ == "__main__":
    main()
