"""Report-only BUT bucket evaluation for saved gated-bad waveform checkpoints.

This script is experiment-only. It loads already-trained waveform-only
checkpoints, rebuilds the same synthetic training normalization/prototype
objects, and evaluates original BUT buckets. It does not train and does not use
BUT for selection.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(r"E:\GPTProject2\ecg")
SCRIPT = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_node_ladder_tuning_10s_2026_06_08"
    / "analysis"
    / "good_medium_geometry_repair"
    / "run_waveform_geometry_student.py"
)
OUT_DIR = SCRIPT.parent
NODE_ID = "N17043_gm_probe"
CANDIDATES = ["gated_bad_statfed_stablemix", "gated_bad_multiscale_stablemix"]


def load_student_module() -> Any:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_candidate(module: Any, name: str) -> list[dict[str, Any]]:
    cfg = dict(module.CANDIDATES[name])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_train_ds = module.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    val_ds = module.ARCH.SyntheticWaveDataset("val", str(cfg["channels"]))
    norm = module.ARCH.FeatureNorm(mean=base_train_ds.mean, std=base_train_ds.std)
    original_ds = module.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    val_loader = DataLoader(val_ds, batch_size=192, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    original_loader = DataLoader(original_ds, batch_size=192, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = module.GeometryStudent(cfg, int(base_train_ds.x.shape[1])).to(device)
    if getattr(model, "use_teacher_atlas", False):
        prototypes = module.build_teacher_prototypes(
            base_train_ds.aux,
            base_train_ds.y,
            int(cfg["teacher_prototypes_per_class"]),
            str(cfg.get("teacher_prototype_mode", "pc1_quantile")),
        )
        model.set_teacher_prototypes(prototypes)
    if getattr(model, "use_waveform_atlas", False):
        wave_proto, wave_mean, wave_std = module.build_waveform_atlas(
            base_train_ds,
            int(cfg["waveform_atlas_prototypes_per_class"]),
            str(cfg.get("waveform_atlas_mode", "farthest")),
        )
        model.set_waveform_atlas(wave_proto, wave_mean, wave_std)

    ckpt_path = module.OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search" / name / "ckpt_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_payload = module.eval_loader(model, val_loader, device)
    threshold = module.ARCH.calibrate_bad_threshold(val_ds.y, val_payload.probs)
    original_payload = module.eval_loader(model, original_loader, device)
    pred_badcal = module.ARCH.apply_bad_threshold(original_payload.probs, threshold)

    atlas = pd.read_csv(module.ARCH.ORIGINAL_ATLAS)
    class_names = {v: k for k, v in module.CLASS_TO_INT.items()}
    pred_rows = atlas.copy()
    pred_rows["pred"] = [class_names[int(v)] for v in original_payload.pred]
    pred_rows["pred_badcal"] = [class_names[int(v)] for v in pred_badcal]
    pred_rows["prob_good"] = original_payload.probs[:, module.CLASS_TO_INT["good"]]
    pred_rows["prob_medium"] = original_payload.probs[:, module.CLASS_TO_INT["medium"]]
    pred_rows["prob_bad"] = original_payload.probs[:, module.CLASS_TO_INT["bad"]]
    pred_rows["correct"] = pred_rows["pred"].astype(str) == pred_rows["class_name"].astype(str)
    pred_rows["correct_badcal"] = pred_rows["pred_badcal"].astype(str) == pred_rows["class_name"].astype(str)
    pred_rows.to_csv(module.OUT_ROOT / "analysis" / "good_medium_geometry_repair" / f"{name}_original_predictions.csv", index=False)

    rows: list[dict[str, Any]] = []
    rows.append({"candidate": name, "bucket": "synthetic_val", **val_payload.report})
    for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
        rows.append({"candidate": name, "bucket": bucket, **module.ARCH.bucket_report(original_ds, original_payload.probs, bucket)})
        rows.append({"candidate": f"{name}_badcal", "bucket": bucket, **module.ARCH.bucket_report(original_ds, original_payload.probs, bucket, threshold)})

    run_dir = module.OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search" / name
    pd.DataFrame(module.feature_recovery_rows(name, "search", "original_all_10s+", original_payload, original_ds.aux)).to_csv(
        run_dir / "feature_teacher_recovery_original.csv",
        index=False,
    )
    return rows


def main() -> None:
    module = load_student_module()
    rows: list[dict[str, Any]] = []
    for name in CANDIDATES:
        rows.extend(evaluate_candidate(module, name))
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "gated_bad_stablemix_original_report.csv"
    out_json = OUT_DIR / "gated_bad_stablemix_original_report.json"
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(df[["candidate", "bucket", "acc", "macro_f1", "good_recall", "medium_recall", "bad_recall"]].to_string(index=False))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
