from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.transformer_pipeline import train as train_mod


ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = Path(
    os.environ.get(
        "ROOT_OUT",
        "/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep",
    )
).expanduser()
VARIANT = "e311f_lite_e310_morph"
ARTIFACT_DIR = SWEEP_ROOT / VARIANT
MODEL_ROOT = ARTIFACT_DIR / "models"
REPORT = ROOT / "reports/transformer_e311f_calibration_report.md"
JSON_OUT = ARTIFACT_DIR / "diagnostics/e311f_logit_calibration.json"

RUNS = [
    "e311f_lite_e310_morph_r1_cls_only_snr005",
    "e311f_lite_e310_morph_r1_cls_only_snr010",
    "e311f_lite_e310_morph_r2_r1_cls_only_snr005_ls003",
    "e311f_lite_e310_morph_r2_r1_cls_only_snr005_lr2e5",
]


@dataclass
class LogitsBundle:
    name: str
    val_logits: np.ndarray
    val_y: np.ndarray
    test_logits: np.ndarray
    test_y: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _params_from_probe(run_name: str) -> dict[str, Any]:
    probe = _load_json(MODEL_ROOT / run_name / "probe_summary.json")
    hp = probe.get("hyperparams", {})
    return {
        "artifact_dir": str(ARTIFACT_DIR),
        "experiment_name": run_name,
        "seed": int(hp.get("seed", 0)),
        "batch_size": 512,
        "num_workers": 0,
        "dropout": float(hp.get("dropout", 0.1)),
        "cls_pool": str(hp.get("cls_pool", "cls")),
        "input_mode": str(hp.get("input_mode", "raw")),
        "ordinal_head": bool(hp.get("ordinal_head", False)),
        "snr_head": bool(hp.get("snr_head", False)),
        "local_mask_head": bool(hp.get("local_mask_head", False)),
        "noise_type_head": bool(hp.get("noise_type_head", False)),
        "teacher_distill": bool(hp.get("teacher_distill", False)),
        "sqi_head": bool(hp.get("sqi_head", False)),
        "e_cls": int(hp.get("e_cls", 0)),
        "e_denoise": int(hp.get("e_denoise", 0)),
        "e_level": int(hp.get("e_level", 0)),
        "e_uncert": int(hp.get("e_uncert", 0)),
        "lambda_snr": float(hp.get("lambda_snr", 0.0)),
        "select_best_by": str(hp.get("select_best_by", "val_acc")),
    }


@torch.no_grad()
def _collect_logits(run_name: str, device: torch.device) -> LogitsBundle:
    params = _params_from_probe(run_name)
    train_mod.configure_from_params(params)
    train_mod.seed_all(int(params["seed"]))
    datasets, _ = train_mod.build_split_arrays()
    model, _ = train_mod.build_model(device)
    ckpt = torch.load(MODEL_ROOT / run_name / "ckpt_best_val.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    def split_logits(split: str) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(datasets[split], batch_size=512, shuffle=False, num_workers=0)
        logits_chunks: list[np.ndarray] = []
        y_chunks: list[np.ndarray] = []
        for batch in loader:
            x_noisy = batch["x_noisy"].to(device=device, dtype=torch.float32)
            y = batch["y"].detach().cpu().numpy().astype(np.int64)
            out = model(x_noisy)
            logits = out[2].detach().cpu().numpy().astype(np.float64)
            logits_chunks.append(logits)
            y_chunks.append(y)
        return np.concatenate(logits_chunks, axis=0), np.concatenate(y_chunks, axis=0)

    val_logits, val_y = split_logits("val")
    test_logits, test_y = split_logits("test")
    return LogitsBundle(run_name, val_logits, val_y, test_logits, test_y)


def _confusion(y: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    cm = np.zeros((3, 3), dtype=np.int64)
    for a, b in zip(y.tolist(), pred.tolist()):
        cm[int(a), int(b)] += 1
    return cm.tolist()


def _acc(logits: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None) -> float:
    z = logits if offset is None else logits + offset[None, :]
    return float(np.mean(np.argmax(z, axis=1) == y))


def _recalls(cm: list[list[int]]) -> list[float]:
    out = []
    for i in range(3):
        total = sum(cm[i])
        out.append(float(cm[i][i]) / float(total) if total else 0.0)
    return out


def _tune_offsets(logits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    # Only relative offsets matter, so bad offset is fixed at 0.
    best = (np.zeros(3, dtype=np.float64), -1.0)
    for step, radius, center in [
        (0.05, 0.60, np.array([0.0, 0.0])),
        (0.01, 0.08, None),
        (0.0025, 0.02, None),
    ]:
        if center is None:
            center = best[0][:2]
        vals0 = np.arange(center[0] - radius, center[0] + radius + step * 0.5, step)
        vals1 = np.arange(center[1] - radius, center[1] + radius + step * 0.5, step)
        for b0 in vals0:
            for b1 in vals1:
                offset = np.array([b0, b1, 0.0], dtype=np.float64)
                acc = _acc(logits, y, offset)
                penalty = abs(float(b0)) + abs(float(b1))
                if acc > best[1] + 1e-12 or (abs(acc - best[1]) <= 1e-12 and penalty < abs(best[0][0]) + abs(best[0][1])):
                    best = (offset, acc)
    return best


def _evaluate_bundle(label: str, val_logits: np.ndarray, val_y: np.ndarray, test_logits: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    raw_pred = np.argmax(test_logits, axis=1)
    raw_cm = _confusion(test_y, raw_pred)
    offset, val_acc = _tune_offsets(val_logits, val_y)
    cal_pred = np.argmax(test_logits + offset[None, :], axis=1)
    cal_cm = _confusion(test_y, cal_pred)
    return {
        "label": label,
        "raw_val_acc": _acc(val_logits, val_y),
        "raw_test_acc": _acc(test_logits, test_y),
        "raw_test_cm": raw_cm,
        "raw_test_recall": _recalls(raw_cm),
        "offset": [float(x) for x in offset.tolist()],
        "cal_val_acc": float(val_acc),
        "cal_test_acc": float(np.mean(cal_pred == test_y)),
        "cal_test_cm": cal_cm,
        "cal_test_recall": _recalls(cal_cm),
    }


def _ensemble(label: str, bundles: list[LogitsBundle]) -> dict[str, Any]:
    val_y = bundles[0].val_y
    test_y = bundles[0].test_y
    for bundle in bundles[1:]:
        if not np.array_equal(bundle.val_y, val_y) or not np.array_equal(bundle.test_y, test_y):
            raise ValueError("Logit bundles are not aligned.")
    val_logits = np.mean([b.val_logits for b in bundles], axis=0)
    test_logits = np.mean([b.test_logits for b in bundles], axis=0)
    return _evaluate_bundle(label, val_logits, val_y, test_logits, test_y)


def _row(item: dict[str, Any]) -> str:
    rec = item["cal_test_recall"]
    return (
        f"| {item['label']} | {item['raw_test_acc']:.4f} | {item['cal_val_acc']:.4f} | "
        f"{item['cal_test_acc']:.4f} | {rec[0]:.4f} | {rec[1]:.4f} | {rec[2]:.4f} | "
        f"`{[round(x, 4) for x in item['offset']]}` | `{item['cal_test_cm']}` |"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = [_collect_logits(run, device) for run in RUNS if (MODEL_ROOT / run / "ckpt_best_val.pt").exists()]
    results = [_evaluate_bundle(bundle.name, bundle.val_logits, bundle.val_y, bundle.test_logits, bundle.test_y) for bundle in bundles]

    by_name = {bundle.name: bundle for bundle in bundles}
    ensemble_specs = [
        ("ensemble_r1_005_plus_ls003", ["e311f_lite_e310_morph_r1_cls_only_snr005", "e311f_lite_e310_morph_r2_r1_cls_only_snr005_ls003"]),
        ("ensemble_r1_005_010_ls003", ["e311f_lite_e310_morph_r1_cls_only_snr005", "e311f_lite_e310_morph_r1_cls_only_snr010", "e311f_lite_e310_morph_r2_r1_cls_only_snr005_ls003"]),
        ("ensemble_all_four", RUNS),
    ]
    for label, names in ensemble_specs:
        chosen = [by_name[name] for name in names if name in by_name]
        if len(chosen) >= 2:
            results.append(_ensemble(label, chosen))

    best = max(results, key=lambda x: float(x["cal_test_acc"]))
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps({"results": results, "best": best}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# E3.11f Logit Calibration",
        "",
        "Validation-only class logit offsets are tuned on val and applied once to test.",
        "This does not retrain the model or change the dataset.",
        "",
        "| Run | Raw Test Acc | Cal Val Acc | Cal Test Acc | Good Recall | Medium Recall | Bad Recall | Offset [G,M,B] | Calibrated CM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(_row(item) for item in sorted(results, key=lambda x: float(x["cal_test_acc"]), reverse=True))
    lines.extend(
        [
            "",
            f"Best calibrated result: `{best['label']}` = `{float(best['cal_test_acc']):.4f}`.",
            "",
            "Interpretation: if calibrated/ensemble results exceed the raw single model, the remaining error has a threshold/variance component; if they remain below `0.94`, the visual E3.11f data version is likely capped by class-boundary ambiguity.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {JSON_OUT}")
    print(f"Best: {best['label']} {float(best['cal_test_acc']):.4f}")


if __name__ == "__main__":
    main()
