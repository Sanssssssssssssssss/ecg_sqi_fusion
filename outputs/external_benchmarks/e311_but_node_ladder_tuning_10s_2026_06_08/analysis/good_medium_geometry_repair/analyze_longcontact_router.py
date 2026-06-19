"""Narrow report-only router scan for new bad-stress waveform specialists.

This is intentionally smaller than analyze_nonbad2badstress_ensemble.py. It
does not train and does not use original BUT for selection; it only diagnoses
whether the new stress specialists can compose with the strongest waveform
good/medium model at high bad confidence.
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
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
BUCKETS = [
    "original_test_all_10s+",
    "original_all_10s+",
    "original_test_main_without_bad_stress",
    "bad_core_nearboundary",
    "bad_outlier_stress",
]


def load_student_module() -> Any:
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
    ds = mod.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    return model, cfg, ds


@torch.no_grad()
def predict_probs(model: torch.nn.Module, ds: Any, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x, mask_ratio=0.0)
        probs.append(torch.softmax(out["logits"], dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def onehot(pred: np.ndarray) -> np.ndarray:
    out = np.zeros((len(pred), 3), dtype=np.float32)
    out[np.arange(len(pred)), pred.astype(np.int64)] = 1.0
    return out


def bucket_rows(mod: Any, ds: Any, candidate: str, probs: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        rep = mod.ARCH.bucket_report(ds, probs, bucket)
        row = {"candidate": candidate, "bucket": bucket}
        row.update(rep)
        row["confusion_3x3"] = json.dumps(row["confusion_3x3"])
        rows.append(row)
    return rows


def main() -> None:
    mod = load_student_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512 if device.type == "cuda" else 128
    main_ckpt = RUN_ROOT / "predtop20_sqiquery_boundary_pretrain" / "ckpt_best.pt"
    specialist_ckpts = {
        "eventqrs_stresshead": RUN_ROOT / "predtop20_eventqrs_stresshead_lowaux_specific_pretrain" / "ckpt_best.pt",
        "sqiquery_stresshead": RUN_ROOT / "predtop20_sqiquery_stresshead_lowaux_specific_pretrain" / "ckpt_best.pt",
        "eventqrs_longcontact": RUN_ROOT / "predtop20_eventqrs_longcontact_stresshead_pretrain" / "ckpt_best.pt",
    }
    main_model, _main_cfg, ds = load_model(mod, main_ckpt, device)
    main_probs = predict_probs(main_model, ds, device, batch_size)
    main_pred = main_probs.argmax(axis=1).astype(np.int64)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    rows.extend(bucket_rows(mod, ds, "predtop20_boundary_pretrain__raw", main_probs))
    thresholds = np.array([0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.97], dtype=np.float32)
    margins = np.array([-0.20, -0.10, 0.0, 0.10, 0.20, 0.35, 0.50], dtype=np.float32)
    for spec_name, ckpt in specialist_ckpts.items():
        if not ckpt.exists():
            continue
        spec_model, _cfg, spec_ds = load_model(mod, ckpt, device)
        if len(spec_ds.y) != len(ds.y) or not np.array_equal(spec_ds.y, ds.y):
            raise RuntimeError(f"dataset mismatch for {spec_name}")
        spec_probs = predict_probs(spec_model, spec_ds, device, batch_size)
        rows.extend(bucket_rows(mod, ds, f"{spec_name}__raw", spec_probs))
        spec_bad = spec_probs[:, CLASS_TO_INT["bad"]]
        spec_nonbad = spec_probs[:, :2].max(axis=1)
        for threshold in thresholds:
            for margin in margins:
                override = (spec_bad >= float(threshold)) & ((spec_bad - spec_nonbad) >= float(margin))
                pred = main_pred.copy()
                pred[override] = CLASS_TO_INT["bad"]
                candidate = f"predtop20_boundary+{spec_name}_t{threshold:.2f}_m{margin:.2f}"
                rows.extend(bucket_rows(mod, ds, candidate, onehot(pred)))
                details.append(
                    {
                        "candidate": candidate,
                        "threshold": float(threshold),
                        "margin": float(margin),
                        "overrides": int(override.sum()),
                        "test_nonbad_overrides": int((override & (ds.split == "test") & (ds.y != CLASS_TO_INT["bad"])).sum()),
                        "test_bad_stress_overrides": int(
                            (override & (ds.split == "test") & (ds.y == CLASS_TO_INT["bad"]) & (ds.region == "outlier_low_confidence")).sum()
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    pivot = metrics.pivot_table(index="candidate", columns="bucket", values=["acc", "good_recall", "medium_recall", "bad_recall"], aggfunc="first")
    flat = pd.DataFrame(index=pivot.index)
    for metric, bucket in pivot.columns:
        flat[f"{bucket}__{metric}"] = pivot[(metric, bucket)]
    flat = flat.reset_index()
    flat["score"] = (
        flat["original_test_all_10s+__acc"].fillna(0)
        + 0.25 * flat["bad_outlier_stress__bad_recall"].fillna(0)
        + 0.15 * flat["bad_core_nearboundary__bad_recall"].fillna(0)
        - 0.30 * np.maximum(0, 0.84 - flat["original_test_all_10s+__good_recall"].fillna(0))
        - 0.30 * np.maximum(0, 0.72 - flat["original_test_all_10s+__medium_recall"].fillna(0))
    )
    flat = flat.sort_values("score", ascending=False)
    metrics_path = ANALYSIS_DIR / "waveform_longcontact_router_metrics.csv"
    best_path = ANALYSIS_DIR / "waveform_longcontact_router_best.csv"
    details_path = ANALYSIS_DIR / "waveform_longcontact_router_overrides.csv"
    metrics.to_csv(metrics_path, index=False)
    flat.to_csv(best_path, index=False)
    pd.DataFrame(details).to_csv(details_path, index=False)
    print(flat.head(20).to_string(index=False))
    print(f"wrote {metrics_path}")
    print(f"wrote {best_path}")


if __name__ == "__main__":
    main()
