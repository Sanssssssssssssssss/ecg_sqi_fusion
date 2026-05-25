from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.transformer_pipeline.audit_medium_errors import _load_transformer_dataset, _predict_transformer


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = Path(
    os.environ.get("E310_ARTIFACT_DIR", ROOT / "outputs/transformer_e310_smooth_morph_mild_snr")
).expanduser()
MODEL_ROOT = ARTIFACT_DIR / "models"
OUT_DIR = ARTIFACT_DIR / "calibration_ensemble"
REPORT = ROOT / "reports/transformer_e310_calibration_ensemble_report.md"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
CLASS_NAMES = ("good", "medium", "bad")

MODEL_RUNS = [
    ("M2 warm-start + SNR head", "e310_m2_d1warm_snr005"),
    ("M3 low denoise", "e310_m3_d1warm_snr005_lowden"),
    ("R1 M2 seed 1", "e310_r1_m2_seed1"),
    ("R1 M2 seed 2", "e310_r1_m2_seed2"),
    ("R1 M2 seed 3", "e310_r1_m2_seed3"),
    ("R1 SNR lambda 0.02", "e310_r1_snr002"),
    ("R1 SNR lambda 0.075", "e310_r1_snr0075"),
    ("R1 medium weight 1.03", "e310_r1_medium103"),
    ("R1 medium weight 1.05", "e310_r1_medium105"),
    ("R1 label smoothing 0.005", "e310_r1_label_smoothing005"),
]

ENSEMBLES = [
    (
        "ensemble M2 seeds",
        ["M2 warm-start + SNR head", "R1 M2 seed 1", "R1 M2 seed 2", "R1 M2 seed 3"],
    ),
    (
        "ensemble SNR family",
        ["M2 warm-start + SNR head", "R1 SNR lambda 0.02", "R1 SNR lambda 0.075"],
    ),
    (
        "ensemble denoise family",
        ["M2 warm-start + SNR head", "M3 low denoise"],
    ),
    (
        "ensemble top5 mixed",
        [
            "M2 warm-start + SNR head",
            "M3 low denoise",
            "R1 M2 seed 1",
            "R1 M2 seed 3",
            "R1 SNR lambda 0.075",
        ],
    ),
    (
        "ensemble all R1 selected",
        [label for label, _run in MODEL_RUNS],
    ),
]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _summary(y: np.ndarray, probs: np.ndarray, offsets: np.ndarray | None = None) -> dict[str, Any]:
    scores = np.log(np.clip(probs, 1e-8, 1.0))
    if offsets is not None:
        scores = scores + offsets[None, :]
    pred = np.argmax(scores, axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]).astype(int)
    recalls = {}
    for i, name in enumerate(CLASS_NAMES):
        total = int(cm[i].sum())
        recalls[name] = float(cm[i, i] / total) if total else 0.0
    return {
        "acc": float(accuracy_score(y, pred)),
        "recall": recalls,
        "confusion_matrix_3x3": cm.tolist(),
    }


def _fit_offsets(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, Any]:
    logp = np.log(np.clip(p_val, 1e-8, 1.0))
    grid = np.round(np.arange(-0.35, 0.351, 0.01), 2)
    best: tuple[float, float, float, np.ndarray] | None = None
    for bg in grid:
        for bm in grid:
            offsets = np.array([bg, bm, 0.0], dtype=np.float64)
            pred = np.argmax(logp + offsets[None, :], axis=1)
            cm = confusion_matrix(y_val, pred, labels=[0, 1, 2]).astype(float)
            acc = float(np.trace(cm) / max(1.0, cm.sum()))
            recalls = np.divide(np.diag(cm), np.maximum(1.0, cm.sum(axis=1)))
            bal = float(np.mean(recalls))
            norm = float(abs(bg) + abs(bm))
            key = (acc, bal, -norm)
            if best is None or key > best[:3]:
                best = (acc, bal, -norm, offsets)
    assert best is not None
    offsets = best[3]
    return {
        "offsets": offsets,
        "offset_good": float(offsets[0]),
        "offset_medium": float(offsets[1]),
        "offset_bad": float(offsets[2]),
        "val_acc_objective": float(best[0]),
        "val_balanced_acc_objective": float(best[1]),
    }


def _load_probs(label: str, run_name: str, labels: pd.DataFrame, x_noisy: np.ndarray, batch_size: int) -> np.ndarray:
    out_path = OUT_DIR / "probs" / f"{run_name}.npz"
    if out_path.exists():
        z = np.load(out_path)
        return z["probs"].astype(np.float32)

    model_dir = MODEL_ROOT / run_name
    if not (model_dir / "ckpt_best_val.pt").exists():
        raise FileNotFoundError(f"missing checkpoint for {label}: {model_dir}")
    probs = _predict_transformer(
        artifact_dir=ARTIFACT_DIR,
        model_dir=model_dir,
        labels=labels,
        x_noisy=x_noisy,
        seed=0,
        batch_size=batch_size,
        verbose=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, probs=probs)
    return probs


def _row(name: str, item: dict[str, Any]) -> str:
    raw = item["test_raw"]
    cal = item["test_calibrated"]
    r = cal["recall"]
    return (
        f"| {name} | {raw['acc']:.4f} | {cal['acc']:.4f} | "
        f"{r['good']:.4f} | {r['medium']:.4f} | {r['bad']:.4f} | "
        f"`{item['offsets']}` | `{cal['confusion_matrix_3x3']}` |"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_all, x_noisy_all, _x_clean, _valid_rr = _load_transformer_dataset(ARTIFACT_DIR)
    split_all = labels_all["split"].astype(str).to_numpy()
    keep = np.isin(split_all, ["val", "test"])
    labels = labels_all.loc[keep].reset_index(drop=True)
    x_noisy = x_noisy_all[keep]
    split = labels["split"].astype(str).to_numpy()
    y = labels["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    val_mask = split == "val"
    test_mask = split == "test"

    probs_by_label: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    for label, run_name in MODEL_RUNS:
        probs = _load_probs(label, run_name, labels, x_noisy, batch_size=512)
        probs_by_label[label] = probs
        fit = _fit_offsets(y[val_mask], probs[val_mask])
        offsets = fit["offsets"]
        results[label] = {
            "kind": "single",
            "members": [label],
            "offsets": [float(v) for v in offsets.tolist()],
            "val_raw": _summary(y[val_mask], probs[val_mask]),
            "val_calibrated": _summary(y[val_mask], probs[val_mask], offsets),
            "test_raw": _summary(y[test_mask], probs[test_mask]),
            "test_calibrated": _summary(y[test_mask], probs[test_mask], offsets),
            "fit": {k: v for k, v in fit.items() if k != "offsets"},
        }

    for label, members in ENSEMBLES:
        missing = [m for m in members if m not in probs_by_label]
        if missing:
            raise KeyError(f"ensemble {label} missing members: {missing}")
        probs = np.mean([probs_by_label[m] for m in members], axis=0)
        fit = _fit_offsets(y[val_mask], probs[val_mask])
        offsets = fit["offsets"]
        results[label] = {
            "kind": "ensemble",
            "members": members,
            "offsets": [float(v) for v in offsets.tolist()],
            "val_raw": _summary(y[val_mask], probs[val_mask]),
            "val_calibrated": _summary(y[val_mask], probs[val_mask], offsets),
            "test_raw": _summary(y[test_mask], probs[test_mask]),
            "test_calibrated": _summary(y[test_mask], probs[test_mask], offsets),
            "fit": {k: v for k, v in fit.items() if k != "offsets"},
        }

    best_raw = max(results.items(), key=lambda kv: kv[1]["test_raw"]["acc"])
    best_cal = max(results.items(), key=lambda kv: kv[1]["test_calibrated"]["acc"])
    summary = {
        "artifact_dir": str(ARTIFACT_DIR),
        "rows": {"val": int(val_mask.sum()), "test": int(test_mask.sum())},
        "best_raw": {"name": best_raw[0], "acc": best_raw[1]["test_raw"]["acc"]},
        "best_calibrated": {"name": best_cal[0], "acc": best_cal[1]["test_calibrated"]["acc"]},
        "results": results,
    }
    (OUT_DIR / "calibration_ensemble_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# E3.10 Calibration And Ensemble",
        "",
        "Purpose: diagnose whether the current E3.10 visual benchmark is limited by model variance or by the data boundary.",
        "The logit offsets are fit on validation only and then applied once to test.",
        "",
        f"Rows: val={int(val_mask.sum())}, test={int(test_mask.sum())}.",
        f"Best raw: `{best_raw[0]}` = `{best_raw[1]['test_raw']['acc']:.4f}`.",
        f"Best calibrated/ensemble: `{best_cal[0]}` = `{best_cal[1]['test_calibrated']['acc']:.4f}`.",
        "",
        "| Run | Raw Test Acc | Calibrated Test Acc | Good Recall | Medium Recall | Bad Recall | Offsets [g,m,b] | Calibrated Confusion Matrix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for label, _run_name in MODEL_RUNS:
        lines.append(_row(label, results[label]))
    for label, _members in ENSEMBLES:
        lines.append(_row(label, results[label]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- M2 remains the single-model anchor unless calibration or ensemble beats it on test.",
            "- A useful ensemble gain would mean model variance still matters; no gain means the remaining errors are mostly label/data boundary.",
            "- This is a diagnostic pass only: it does not change model structure, data generation, or training code.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {OUT_DIR / 'calibration_ensemble_summary.json'}")


if __name__ == "__main__":
    main()
