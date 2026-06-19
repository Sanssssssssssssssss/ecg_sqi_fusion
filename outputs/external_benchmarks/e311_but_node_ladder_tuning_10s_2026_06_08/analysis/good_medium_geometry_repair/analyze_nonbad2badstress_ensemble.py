"""Report-only bad-specialist ensemble diagnostic for waveform students.

This script never trains and never uses original BUT for model selection.  It
asks a narrow diagnostic question: can a nonbad->bad-stress waveform specialist
recover original-test bad-stress windows if it is allowed to override a stronger
good/medium waveform model only at very high bad confidence?
"""

from __future__ import annotations

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
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
RUN_ROOT = OUT_ROOT / "runs" / "waveform_geometry_student" / "N17043_gm_probe" / "search"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"

MAIN_CANDIDATES = {
    "predtop20_boundary_pretrain": RUN_ROOT / "predtop20_sqiquery_boundary_pretrain" / "ckpt_best.pt",
    "predtop23_hardfeat_pretrain": RUN_ROOT / "predtop23_sqiquery_hardfeat_pretrain" / "ckpt_best.pt",
    "qrsbank_top14_focus_patch": RUN_ROOT / "qrsbank_top14_focus_patch" / "ckpt_best.pt",
    "qrsbank_top14_focus_stattoken": RUN_ROOT / "qrsbank_top14_focus_stattoken" / "ckpt_best.pt",
}
SPECIALISTS = {
    "predtop20_badguardlite": RUN_ROOT / "predtop20_sqiquery_badguardlite_pretrain" / "ckpt_best.pt",
    "predtop20_lowfreqbad_stress": RUN_ROOT / "predtop20_sqiquery_lowfreqbad_stress_pretrain" / "ckpt_best.pt",
    "nonbad2bad_balanced": RUN_ROOT / "sqiquery_nonbad2badstress_balanced" / "ckpt_best.pt",
    "nonbad2bad_stress": RUN_ROOT / "sqiquery_nonbad2badstress_stress" / "ckpt_best.pt",
    "eventqrs_badguard": RUN_ROOT / "predtop20_eventqrs_sqiquery_badguard_pretrain" / "ckpt_best.pt",
    "eventqrs_intermitbad_balanced": RUN_ROOT / "predtop20_eventqrs_intermitbad_balanced_pretrain" / "ckpt_best.pt",
    "eventqrs_intermitbad_stress": RUN_ROOT / "predtop20_eventqrs_intermitbad_stress_pretrain" / "ckpt_best.pt",
    "eventqrs_intermitbad_lowaux": RUN_ROOT / "predtop20_eventqrs_intermitbad_lowaux_pretrain" / "ckpt_best.pt",
    "eventqrs_stresshead_lowaux_specific": RUN_ROOT / "predtop20_eventqrs_stresshead_lowaux_specific_pretrain" / "ckpt_best.pt",
    "sqiquery_stresshead_lowaux_specific": RUN_ROOT / "predtop20_sqiquery_stresshead_lowaux_specific_pretrain" / "ckpt_best.pt",
    "eventqrs_longcontact_stresshead": RUN_ROOT / "predtop20_eventqrs_longcontact_stresshead_pretrain" / "ckpt_best.pt",
}
BUCKETS = [
    "original_test_all_10s+",
    "original_all_10s+",
    "original_test_main_without_bad_stress",
    "bad_core_nearboundary",
    "bad_outlier_stress",
]
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


def load_student_module():
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(mod: Any, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["candidate_config"]
    train_ds = mod.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    model = mod.GeometryStudent(cfg, int(train_ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = mod.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    return model, cfg, norm, ckpt


@torch.no_grad()
def predict_probs(mod: Any, model: torch.nn.Module, ds: Any, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x, mask_ratio=0.0)
        probs.append(torch.softmax(out["logits"], dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def onehot_from_pred(pred: np.ndarray) -> np.ndarray:
    out = np.zeros((len(pred), 3), dtype=np.float32)
    out[np.arange(len(pred)), pred.astype(np.int64)] = 1.0
    return out


def metric_rows(mod: Any, ds: Any, candidate: str, probs_or_onehot: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        rep = mod.ARCH.bucket_report(ds, probs_or_onehot, bucket)
        row = {"candidate": candidate, "bucket": bucket}
        row.update(rep)
        row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
        rows.append(row)
    return rows


def scan_override(
    mod: Any,
    ds: Any,
    main_name: str,
    specialist_name: str,
    main_probs: np.ndarray,
    spec_probs: np.ndarray,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    main_pred = main_probs.argmax(axis=1).astype(np.int64)
    spec_bad = spec_probs[:, CLASS_TO_INT["bad"]]
    spec_nonbad = spec_probs[:, :2].max(axis=1)

    # Baselines for comparison.
    rows.extend(metric_rows(mod, ds, f"{main_name}__raw", main_probs))
    rows.extend(metric_rows(mod, ds, f"{specialist_name}__raw", spec_probs))

    for threshold in np.round(np.linspace(0.50, 0.995, 100), 3):
        for margin in [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
            override = (spec_bad >= threshold) & ((spec_bad - spec_nonbad) >= margin)
            pred = main_pred.copy()
            pred[override] = CLASS_TO_INT["bad"]
            onehot = onehot_from_pred(pred)
            cand = f"{main_name}+{specialist_name}_badgate_t{threshold:.3f}_m{margin:.2f}"
            for row in metric_rows(mod, ds, cand, onehot):
                row["threshold"] = float(threshold)
                row["margin"] = float(margin)
                row["overrides"] = int(override.sum())
                rows.append(row)
            detail_rows.append(
                {
                    "candidate": cand,
                    "threshold": float(threshold),
                    "margin": float(margin),
                    "overrides": int(override.sum()),
                    "test_overrides": int((override & (ds.split == "test")).sum()),
                    "test_bad_stress_overrides": int(
                        (
                            override
                            & (ds.split == "test")
                            & (ds.y == CLASS_TO_INT["bad"])
                            & (ds.region == "outlier_low_confidence")
                        ).sum()
                    ),
                    "test_nonbad_overrides": int(
                        (override & (ds.split == "test") & (ds.y != CLASS_TO_INT["bad"])).sum()
                    ),
                }
            )
    return rows, pd.DataFrame(detail_rows)


def best_table(metrics: pd.DataFrame) -> pd.DataFrame:
    piv = metrics.pivot_table(
        index="candidate",
        columns="bucket",
        values=["acc", "good_recall", "medium_recall", "bad_recall"],
        aggfunc="first",
    )
    flat = pd.DataFrame(index=piv.index)
    for metric, bucket in piv.columns:
        flat[f"{bucket}__{metric}"] = piv[(metric, bucket)]
    flat = flat.reset_index()
    # Prefer original_test accuracy, then bad-stress recall, while requiring no
    # total collapse. This is diagnostic ranking only, not model selection.
    for col in [
        "original_test_all_10s+__acc",
        "original_test_all_10s+__good_recall",
        "original_test_all_10s+__medium_recall",
        "original_test_all_10s+__bad_recall",
        "bad_outlier_stress__bad_recall",
        "bad_core_nearboundary__bad_recall",
    ]:
        if col not in flat:
            flat[col] = np.nan
    flat["diagnostic_score"] = (
        flat["original_test_all_10s+__acc"].fillna(0.0)
        + 0.30 * flat["bad_outlier_stress__bad_recall"].fillna(0.0)
        + 0.20 * flat["bad_core_nearboundary__bad_recall"].fillna(0.0)
        - 0.25 * np.maximum(0.0, 0.85 - flat["original_test_all_10s+__good_recall"].fillna(0.0))
        - 0.25 * np.maximum(0.0, 0.75 - flat["original_test_all_10s+__medium_recall"].fillna(0.0))
    )
    return flat.sort_values("diagnostic_score", ascending=False)


def render_report(best: pd.DataFrame, out_csv: Path, detail_csv: Path) -> str:
    view_cols = [
        "candidate",
        "diagnostic_score",
        "original_test_all_10s+__acc",
        "original_test_all_10s+__good_recall",
        "original_test_all_10s+__medium_recall",
        "original_test_all_10s+__bad_recall",
        "bad_core_nearboundary__bad_recall",
        "bad_outlier_stress__bad_recall",
    ]
    lines = [
        "# Nonbad-to-Bad Stress Specialist Ensemble Diagnostic",
        "",
        "This is report-only. Original BUT is not used for training, threshold selection, or promotion.",
        "",
        "## Top Diagnostic Rows",
        "",
        "| Candidate | Score | Original Test Acc | Good R | Medium R | Bad R | Bad Core R | Bad Stress R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.head(20)[view_cols].iterrows():
        lines.append(
            f"| `{row['candidate']}` | {float(row['diagnostic_score']):.4f} | "
            f"{float(row['original_test_all_10s+__acc']):.4f} | "
            f"{float(row['original_test_all_10s+__good_recall']):.4f} | "
            f"{float(row['original_test_all_10s+__medium_recall']):.4f} | "
            f"{float(row['original_test_all_10s+__bad_recall']):.4f} | "
            f"{float(row['bad_core_nearboundary__bad_recall']):.4f} | "
            f"{float(row['bad_outlier_stress__bad_recall']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "- If no high-threshold override improves original-test accuracy and bad-stress recall together, the bad-stress specialist is not composable with the current good/medium model.",
            "- A useful next architecture should learn a separable bad-stress representation while preserving non-bad specificity, not merely increase bad probability globally.",
            f"- Full metrics: `{out_csv}`",
            f"- Override counts: `{detail_csv}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    mod = load_student_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512 if device.type == "cuda" else 128
    all_rows: list[dict[str, Any]] = []
    all_details: list[pd.DataFrame] = []

    loaded_main: dict[str, tuple[Any, np.ndarray, Any]] = {}
    for main_name, main_ckpt in MAIN_CANDIDATES.items():
        if not main_ckpt.exists():
            continue
        model, cfg, norm, _ = load_model(mod, main_ckpt, device)
        ds = mod.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
        probs = predict_probs(mod, model, ds, device, batch_size)
        loaded_main[main_name] = (ds, probs, cfg)

    loaded_spec: dict[str, np.ndarray] = {}
    # All current candidates use robust3; use the first main dataset for length.
    first_ds = next(iter(loaded_main.values()))[0]
    for spec_name, spec_ckpt in SPECIALISTS.items():
        if not spec_ckpt.exists():
            continue
        model, cfg, norm, _ = load_model(mod, spec_ckpt, device)
        spec_ds = mod.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
        if len(spec_ds.y) != len(first_ds.y) or not np.array_equal(spec_ds.y, first_ds.y):
            raise RuntimeError(f"dataset mismatch for {spec_name}")
        loaded_spec[spec_name] = predict_probs(mod, model, spec_ds, device, batch_size)

    for main_name, (ds, main_probs, _cfg) in loaded_main.items():
        for spec_name, spec_probs in loaded_spec.items():
            rows, details = scan_override(mod, ds, main_name, spec_name, main_probs, spec_probs)
            all_rows.extend(rows)
            details["main_candidate"] = main_name
            details["specialist_candidate"] = spec_name
            all_details.append(details)

    metrics = pd.DataFrame(all_rows)
    details = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    out_csv = ANALYSIS_DIR / "waveform_nonbad2badstress_ensemble_diagnostic.csv"
    detail_csv = ANALYSIS_DIR / "waveform_nonbad2badstress_ensemble_override_counts.csv"
    report_path = ANALYSIS_DIR / "waveform_nonbad2badstress_ensemble_report.md"
    metrics.to_csv(out_csv, index=False)
    details.to_csv(detail_csv, index=False)
    best = best_table(metrics)
    best.to_csv(ANALYSIS_DIR / "waveform_nonbad2badstress_ensemble_best.csv", index=False)
    report_path.write_text(render_report(best, out_csv, detail_csv), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {report_path}")
    print(best.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
