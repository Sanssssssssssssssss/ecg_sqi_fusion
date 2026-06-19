from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler


THIS = Path(__file__)
ANALYSIS_DIR = THIS.parent
OUT_ROOT = ANALYSIS_DIR.parent.parent
REPORT_DIR = Path(
    str(ANALYSIS_DIR).replace(
        "\\outputs\\external_benchmarks\\",
        "\\reports\\external_benchmarks\\",
    )
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
BANKS = ["qrs_enhanced", "qrs_enhanced_v2", "qrs_stress_v3"]


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", THIS.with_name("run_waveform_geometry_student.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run_waveform_geometry_student.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa = a[mask] - a[mask].mean()
    bb = b[mask] - b[mask].mean()
    denom = math.sqrt(float((aa * aa).sum() * (bb * bb).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((aa * bb).sum() / denom)


def collect_stats(mod: Any, ds: Any, bank: str = "qrs_enhanced", batch_size: int = 512) -> np.ndarray:
    rows: list[np.ndarray] = []
    x = np.asarray(ds.x, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size])
            rows.append(mod.primitive_waveform_stats(xb, bank).cpu().numpy())
    return np.concatenate(rows, axis=0)


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "good_recall": float(rec[0]),
        "medium_recall": float(rec[1]),
        "bad_recall": float(rec[2]),
        "confusion_3x3": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> None:
    mod = load_runner()
    channels = "robust3"
    train_ds = mod.ARCH.SyntheticWaveDataset("train", channels)
    test_ds = mod.ARCH.SyntheticWaveDataset("test", channels)
    norm = mod.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = mod.ARCH.OriginalWaveDataset(norm, channels)

    target_names = []
    for name in list(mod.TOP14_FEATURE_COLUMNS) + list(mod.HARD_FEATURE_COLUMNS):
        if name in mod.FEATURE_COLUMNS and name not in target_names:
            target_names.append(name)
    target_idx = [mod.FEATURE_COLUMNS.index(name) for name in target_names]

    y_train_aux = np.asarray(train_ds.aux[:, target_idx], dtype=np.float32)
    y_test_aux = np.asarray(test_ds.aux[:, target_idx], dtype=np.float32)
    y_orig_aux = np.asarray(original_ds.aux[:, target_idx], dtype=np.float32)

    rows = []
    class_rows = []
    for bank in BANKS:
        print(f"collecting primitive stats bank={bank}", flush=True)
        x_train = collect_stats(mod, train_ds, bank=bank)
        x_test = collect_stats(mod, test_ds, bank=bank)
        x_orig = collect_stats(mod, original_ds, bank=bank)
        scaler = StandardScaler()
        x_train_z = scaler.fit_transform(x_train)
        x_test_z = scaler.transform(x_test)
        x_orig_z = scaler.transform(x_orig)

        print(f"fitting primitive feature regressor bank={bank}", flush=True)
        reg = ExtraTreesRegressor(
            n_estimators=240,
            max_features=0.55,
            min_samples_leaf=3,
            random_state=20260971,
            n_jobs=-1,
        )
        reg.fit(x_train_z, y_train_aux)
        pred_test = reg.predict(x_test_z)
        pred_orig = reg.predict(x_orig_z)

        for j, name in enumerate(target_names):
            for split, pred, target in [
                ("synthetic_test", pred_test, y_test_aux),
                ("original_test_all_10s+", pred_orig, y_orig_aux),
            ]:
                rows.append(
                    {
                        "bank": bank,
                        "split": split,
                        "feature": name,
                        "corr": corr(pred[:, j], target[:, j]),
                        "mae_z": float(np.mean(np.abs(pred[:, j] - target[:, j]))),
                        "is_top14": name in mod.TOP14_FEATURE_COLUMNS,
                        "is_hard": name in mod.HARD_FEATURE_COLUMNS,
                    }
                )

        print(f"fitting primitive class classifier bank={bank}", flush=True)
        clf = ExtraTreesClassifier(
            n_estimators=320,
            max_features=0.55,
            min_samples_leaf=2,
            class_weight={0: 1.0, 1: 1.35, 2: 3.2},
            random_state=20260972,
            n_jobs=-1,
        )
        clf.fit(x_train_z, np.asarray(train_ds.y, dtype=int))
        for split, xz, y in [
            ("synthetic_test", x_test_z, np.asarray(test_ds.y, dtype=int)),
            ("original_test_all_10s+", x_orig_z, np.asarray(original_ds.y, dtype=int)),
        ]:
            pred = clf.predict(xz)
            item = report_metrics(y, pred)
            item["split"] = split
            item["model"] = "primitive_extra_trees"
            item["bank"] = bank
            class_rows.append(item)
    feature_df = pd.DataFrame(rows)
    feature_path = ANALYSIS_DIR / "waveform_primitive_hard_feature_learnability.csv"
    feature_df.to_csv(feature_path, index=False)
    feature_df.to_csv(REPORT_DIR / feature_path.name, index=False)

    class_df = pd.DataFrame(class_rows)
    class_path = ANALYSIS_DIR / "waveform_primitive_class_learnability.csv"
    class_df.to_csv(class_path, index=False)
    class_df.to_csv(REPORT_DIR / class_path.name, index=False)

    summary = {
        "created_by": str(THIS),
        "input_contract": "waveform primitive stats computed from ECG only; diagnostic, not final model",
        "banks": BANKS,
        "target_features": target_names,
        "feature_csv": str(feature_path),
        "class_csv": str(class_path),
        "class_rows": class_rows,
    }
    summary_path = ANALYSIS_DIR / "waveform_primitive_learnability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / summary_path.name).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Waveform Primitive Hard-Feature Learnability",
        "",
        "Diagnostic-only check: can waveform-derived primitive stats predict hard SQI/geometry targets?",
        "",
        "## Classifier",
        "",
    ]
    for row in class_rows:
        lines.append(
            f"- {row['bank']} / {row['split']}: acc={row['acc']:.6f}, macro_f1={row['macro_f1']:.6f}, "
            f"good/medium/bad={row['good_recall']:.3f}/{row['medium_recall']:.3f}/{row['bad_recall']:.3f}"
        )
    lines.extend(["", "## Hard Feature Recovery", ""])
    view = feature_df[feature_df["split"] == "synthetic_test"].sort_values(["bank", "corr"], ascending=[True, False])
    lines.append("| bank | feature | corr | mae_z | top14 | hard |")
    lines.append("|---|---|---:|---:|---|---|")
    for _, row in view.iterrows():
        lines.append(
            f"| {row['bank']} | {row['feature']} | {row['corr']:.4f} | {row['mae_z']:.4f} | {row['is_top14']} | {row['is_hard']} |"
        )
    report_path = ANALYSIS_DIR / "waveform_primitive_learnability_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / report_path.name).write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
