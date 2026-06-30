from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .common import Paths, dry, ensure_dirs, git_commit, read_json, rel, run_date, table_to_md


def _read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run(paths: Paths, *, execute: bool) -> dict[str, Any]:
    if not execute:
        dry("report", paths)
        return {"step": "report", "skipped": True}
    ensure_dirs(paths)
    audit = read_json(paths.audit_json)
    figure_index = read_json(paths.reports / "figure_index.json") if (paths.reports / "figure_index.json").exists() else {}
    manifest = pd.DataFrame(
        [
            {"Field": "git commit", "Value": git_commit()},
            {"Field": "run date", "Value": run_date()},
            {"Field": "random seed", "Value": "0"},
            {"Field": "data policy", "Value": "split first; train-only repair; validation/test original only"},
            {"Field": "Set-A protocol path", "Value": rel(paths.seta / "data" / "protocol_gapfill.csv")},
            {"Field": "BUT protocol path", "Value": audit["but"]["protocol_path"]},
            {"Field": "output root", "Value": rel(paths.out)},
            {"Field": "code command", "Value": "python -m supplemental_transformer_experiments.chapter4_evidence.run pipeline --run"},
            {"Field": "config file", "Value": "CLI defaults; seed=0; Python/matplotlib figures"},
            {"Field": "checkpoint path", "Value": "Set-A: chapter4 output; BUT E31: frozen v116 test_predictions.npz"},
        ]
    )
    protocol_rows = []
    for key, value in audit["seta"]["split_counts"].items():
        split, cls = key.split("_", 1)
        protocol_rows.append(
            {
                "Dataset": "Set-A",
                "Split": split,
                "Class": cls,
                "Original rows": value,
                "Generated rows": 0,
                "Total rows": value,
                "Generated in val/test": 0,
            }
        )
    for key, value in audit["but"]["split_counts"].items():
        split, cls = key.split("_", 1)
        generated = 0 if split in {"val", "test"} else ""
        protocol_rows.append(
            {
                "Dataset": "BUT",
                "Split": split,
                "Class": cls,
                "Original rows": "",
                "Generated rows": generated,
                "Total rows": value,
                "Generated in val/test": audit["but"]["val_test_generated_rows"] if split in {"val", "test"} else "",
            }
        )
    protocol_table = pd.DataFrame(protocol_rows)
    repair_metrics = _read_csv(paths.tables / "seta_distribution_repair_metrics.csv")
    transfer = _read_csv(paths.tables / "seta_source_transfer.csv")
    paired = _read_csv(paths.tables / "seta_paired_mmd_calibration.csv")
    seta_construction = _read_csv(paths.tables / "seta_construction_effect_models.csv")
    seta_models = _read_csv(paths.tables / "seta_repaired_model_comparison.csv")
    but_models = _read_csv(paths.tables / "but_model_comparison.csv")
    figs = pd.DataFrame(
        [
            {
                "Figure": name,
                "File path": rel(__import__("pathlib").Path(path)),
                "Source data": rel(paths.source_data),
                "Conclusion role": "raw evidence",
            }
            for name, path in figure_index.items()
        ]
    )
    lines = [
        "# Chapter 4 Raw Results Report",
        "",
        "This is a raw experiment report, not manuscript prose.",
        "",
        "## 1. Run Manifest",
        "",
        table_to_md(manifest),
        "",
        "## 2. Protocol And Split Audit",
        "",
        table_to_md(protocol_table),
        "",
        "## 3. Data Repair Audit",
        "",
        "### Distribution Metrics",
        "",
        table_to_md(repair_metrics),
        "",
        "### Source-Transfer Metrics",
        "",
        table_to_md(transfer),
        "",
        "### Paired MMD Calibration",
        "",
        table_to_md(paired),
        "",
        "## 4. Set-A Model Comparison",
        "",
        "### Construction Effect",
        "",
        table_to_md(seta_construction),
        "",
        "### Repaired Setup Model Comparison",
        "",
        table_to_md(seta_models),
        "",
        "## 5. BUT Model Comparison",
        "",
        table_to_md(but_models),
        "",
        "## 6. Figure Index",
        "",
        table_to_md(figs),
        "",
        "## 7. Candidate Diagnostics For Observation Section",
        "",
        table_to_md(
            pd.DataFrame(
                [
                    {"Diagnostic": "source sensitivity", "Trigger": "C2ST high or source imbalance remains", "Needed output": "score by source, embedding C2ST", "Decision": "defer until raw evidence readout"},
                    {"Diagnostic": "shortcut check", "Trigger": "generated source remains separable", "Needed output": "provenance classifier", "Decision": "defer"},
                    {"Diagnostic": "local evidence maps", "Trigger": "waveform model improves boundary recall", "Needed output": "ECG + local map overlays", "Decision": "defer"},
                    {"Diagnostic": "input ablation", "Trigger": "need to explain waveform gain", "Needed output": "channel ablation table", "Decision": "defer"},
                    {"Diagnostic": "calibration", "Trigger": "models close or threshold-sensitive", "Needed output": "ECE, reliability, threshold sweep", "Decision": "defer"},
                ]
            )
        ),
        "",
    ]
    paths.report_md.write_text("\n".join(lines), encoding="utf-8")
    print(paths.report_md)
    return {"step": "report", "skipped": False, "output": str(paths.report_md)}

