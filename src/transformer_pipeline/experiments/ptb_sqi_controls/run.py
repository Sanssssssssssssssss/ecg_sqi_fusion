"""Run three-class SQI feature baselines on the current E3.11f PTB artifact.

This wrapper intentionally keeps the compatibility layer thin.  It reuses the
existing PTB SQI multiclass adapter and writes a controls-specific output/report
layout, while leaving ``src/sqi_pipeline`` untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.transformer_pipeline import sqi_ml_multiclass


ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid" / "data" / "med6p25_badgap7_badcm0p75"
DEFAULT_OUT = ROOT / "outputs" / "controls" / "e311f_ptb_sqi_three_class"
DEFAULT_REPORT = ROOT / "reports" / "controls" / "e311f_ptb_sqi_three_class"
DEFAULT_MAINLINE = ROOT / "outputs" / "mainline" / "e311_uformer_full_tokens_detach_seed0"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def model_report(model_name: str, metrics: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    test = metrics["test"]
    report = {
        "model": model_name,
        "task": "three_class_good_medium_bad",
        "feature_set": "seven_inference_time_sqi_features",
        "acc": test["acc"],
        "balanced_acc": test["balanced_acc"],
        "macro_f1": test["macro_f1"],
        "recall_good_medium_bad": [
            test["per_class_recall"]["good"],
            test["per_class_recall"]["medium"],
            test["per_class_recall"]["bad"],
        ],
        "confusion_3x3": test["confusion_matrix_3x3"],
        "notes": [
            "Three-class compatibility control for PTB/E3.11f.",
            "Inputs are noisy ECG-derived SQI features only; clean/masks/morph scores are not model features.",
        ],
    }
    write_json(out_dir / "test_report.json", report)
    return report


def compare_mainline(mainline_dir: Path) -> dict[str, Any] | None:
    report_path = mainline_dir / "test_report.json"
    metrics_path = mainline_dir / "denoise_eval" / "denoise_metrics.json"
    if not report_path.exists():
        return None
    report = read_json(report_path)
    out: dict[str, Any] = {
        "acc": report.get("acc"),
        "recall_good_medium_bad": report.get("recall_good_medium_bad"),
        "confusion_3x3": report.get("confusion_3x3"),
    }
    if metrics_path.exists():
        metrics = read_json(metrics_path)["overall"]
        out["denoise_score"] = metrics.get("denoise_score")
        out["snr_improve_db_mean"] = metrics.get("snr_improve_db_mean")
    return out


def split_audit(split_csv: Path, feature_parquet: Path) -> dict[str, Any]:
    split = pd.read_csv(split_csv)
    feat = pd.read_parquet(feature_parquet)
    label_cols = [c for c in ("y", "y_class", "label", "target") if c in feat.columns]
    feature_cols = [c for c in feat.columns if c not in {"record_id", *label_cols}]
    forbidden = [
        c
        for c in feature_cols
        if any(token in c.lower() for token in ("clean", "mask", "morph", "damage", "label", "target", "critical"))
    ]
    return {
        "split_counts": {str(k): int(v) for k, v in split["split"].value_counts().sort_index().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in split["y_class"].value_counts().sort_index().to_dict().items()},
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "label_columns_not_features": label_cols,
        "forbidden_feature_columns": forbidden,
        "feature_policy": "SQI features only, normalized with train split statistics by reused adapter.",
    }


def render_report(summary: dict[str, Any], normalized: dict[str, Any]) -> str:
    svm = normalized["models"]["sqi_svm_rbf_three_class"]
    mlp = normalized["models"]["sqi_mlp_three_class"]
    mainline = normalized.get("uformer_mainline")

    lines = [
        "# E3.11f PTB SQI Three-Class Controls",
        "",
        "This is a standalone control suite. It uses traditional SQI features on the current E3.11f PTB artifact and does not modify `src/sqi_pipeline`.",
        "",
        "Task: `good / medium / bad` three-class compatibility control.",
        "",
        "## Data",
        "",
        f"- source artifact: `{normalized['source_artifact_dir']}`",
        f"- split counts: `{normalized['data_audit']['split_counts']}`",
        f"- class counts: `{normalized['data_audit']['class_counts']}`",
        f"- feature columns: `{normalized['data_audit']['feature_columns']}`",
        f"- label columns not used as features: `{normalized['data_audit']['label_columns_not_features']}`",
        f"- forbidden feature columns detected: `{normalized['data_audit']['forbidden_feature_columns']}`",
        "",
        "## Results",
        "",
        "| Model | Features | Test Acc | Balanced Acc | Macro F1 | Good R | Medium R | Bad R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| SQI SVM-RBF | 7 SQI | {svm['acc']:.4f} | {svm['balanced_acc']:.4f} | {svm['macro_f1']:.4f} | {svm['recall_good_medium_bad'][0]:.4f} | {svm['recall_good_medium_bad'][1]:.4f} | {svm['recall_good_medium_bad'][2]:.4f} |",
        f"| SQI MLP | 7 SQI | {mlp['acc']:.4f} | {mlp['balanced_acc']:.4f} | {mlp['macro_f1']:.4f} | {mlp['recall_good_medium_bad'][0]:.4f} | {mlp['recall_good_medium_bad'][1]:.4f} | {mlp['recall_good_medium_bad'][2]:.4f} |",
    ]
    if mainline:
        rec = mainline.get("recall_good_medium_bad") or [None, None, None]
        lines.extend(
            [
                f"| Uformer mainline | waveform + detached Uformer tokens | {mainline['acc']:.4f} | - | - | {rec[0]:.4f} | {rec[1]:.4f} | {rec[2]:.4f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These are traditional SQI-feature three-class compatibility controls, not the original binary SQI paper task.",
            "- They are intentionally kept separate from the mainline and from previous experiment lineage.",
            "- Poor or imbalanced performance is still useful evidence: it shows what handcrafted SQI features can and cannot explain on this synthetic PTB three-class task.",
            "",
            "## Artifacts",
            "",
            f"- normalized summary: `{normalized['normalized_summary_json']}`",
            f"- raw adapter summary: `{summary.get('artifact_dir')}/three_class_summary.json`",
            "- per-model reports and predictions live under `outputs/controls/e311f_ptb_sqi_three_class/models/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_artifact_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    result = sqi_ml_multiclass.run(
        {
            "transformer_artifact_dir": str(source),
            "out_dir": str(out_dir),
            "seed": int(args.seed),
            "force": bool(args.force),
            "verbose": bool(args.verbose),
        }
    )
    summary_path = out_dir / "three_class_summary.json"
    summary = read_json(summary_path)

    svm_dir = out_dir / "models" / "svm"
    mlp_dir = out_dir / "models" / "mlp"
    svm_report = model_report("sqi_svm_rbf_three_class", summary["metrics"]["svm_rbf"], svm_dir)
    mlp_report = model_report("sqi_mlp_three_class", summary["metrics"]["mlp"], mlp_dir)

    normalized = {
        "control_name": "e311f_ptb_sqi_three_class",
        "seed": int(args.seed),
        "task": "three_class_good_medium_bad",
        "source_artifact_dir": str(source),
        "output_dir": str(out_dir),
        "report_dir": str(report_dir),
        "implementation_note": "Thin wrapper around src.transformer_pipeline.sqi_ml_multiclass; src/sqi_pipeline is read-only.",
        "data_audit": split_audit(out_dir / "splits" / f"transformer_three_class_seed{args.seed}.csv", out_dir / "features" / "record7_norm.parquet"),
        "models": {
            "sqi_svm_rbf_three_class": svm_report,
            "sqi_mlp_three_class": mlp_report,
        },
        "uformer_mainline": compare_mainline(Path(args.mainline_dir)),
        "raw_adapter_result": result,
    }
    normalized["normalized_summary_json"] = str(out_dir / "summary.json")
    write_json(out_dir / "summary.json", normalized)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "README.md").write_text(render_report(summary, normalized), encoding="utf-8")
    write_json(report_dir / "summary.json", normalized)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3.11f PTB SQI three-class controls.")
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--report_dir", default=str(DEFAULT_REPORT))
    parser.add_argument("--mainline_dir", default=str(DEFAULT_MAINLINE))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    normalized = run(parse_args())
    svm = normalized["models"]["sqi_svm_rbf_three_class"]
    mlp = normalized["models"]["sqi_mlp_three_class"]
    print(
        "E3.11f PTB SQI controls | "
        f"SVM acc={svm['acc']:.4f} medium={svm['recall_good_medium_bad'][1]:.4f} | "
        f"MLP acc={mlp['acc']:.4f} medium={mlp['recall_good_medium_bad'][1]:.4f}"
    )


if __name__ == "__main__":
    main()
