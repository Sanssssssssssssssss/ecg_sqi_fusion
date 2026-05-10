from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import project_root


def main() -> None:
    args = parse_args()
    root = project_root()
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = root / artifact_dir

    summary = collect_summary(root, artifact_dir, seed=args.seed)
    if args.write:
        out = Path(args.write)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(out.relative_to(root).as_posix())
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PTB-XL transformer artifacts.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", default="")
    return parser.parse_args()


def collect_summary(root: Path, artifact_dir: Path, *, seed: int = 0) -> dict[str, Any]:
    labels = pd.read_csv(artifact_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv")
    clean = np.load(artifact_dir / "datasets" / "synth_10s_125hz_clean.npz")["X_clean"]
    noisy = np.load(artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"]
    z_level = np.load(artifact_dir / "datasets" / "synth_10s_125hz_noise_level.npz")
    p = z_level["P"]
    valid_rr = z_level["valid_rr"]

    _assert(clean.shape == noisy.shape, f"clean/noisy shape mismatch: {clean.shape} vs {noisy.shape}")
    _assert(clean.shape == p.shape, f"clean/P shape mismatch: {clean.shape} vs {p.shape}")
    _assert(clean.shape[0] == len(labels) == valid_rr.shape[0], "dataset row count mismatch")
    _assert(clean.shape[1] == 1250, f"expected 1250 samples, got {clean.shape[1]}")

    model_dir = artifact_dir / "models" / f"mtl_transformer_seed{seed}_step6"
    run_summary = artifact_dir / "config" / f"transformer_run_summary_seed{seed}.json"
    test_report = _read_json_if_exists(model_dir / "test_report.json")

    outputs = [model_dir / "ckpt_best_val.pt", model_dir / "ckpt_last.pt", model_dir / "train_log.json", model_dir / "test_report.json"]
    missing = [_rel(path, root) for path in outputs if not path.exists()]
    run_summary_outputs = _summary_output_contract(run_summary, root)

    return {
        "dataset": {
            "clean_shape": list(clean.shape),
            "noisy_shape": list(noisy.shape),
            "noise_level_shape": list(p.shape),
            "labels_shape": list(labels.shape),
            "split_counts": _counts(labels["split"]),
            "y_class_counts_by_split": {
                split: _counts(labels.loc[labels["split"] == split, "y_class"])
                for split in ["train", "val", "test"]
            },
            "valid_rr": {
                "total": int(valid_rr.sum()),
                "n": int(valid_rr.shape[0]),
                "frac": float(valid_rr.mean()) if valid_rr.shape[0] else 0.0,
            },
            "p_mean": float(p.mean()),
            "p_gt_0_5_frac": float((p > 0.5).mean()),
        },
        "model": {
            "dir": _rel(model_dir, root),
            "missing_outputs": missing,
            "test_report": test_report,
        },
        "run_summary": run_summary_outputs,
    }


def _summary_output_contract(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": _rel(path, root), "exists": False, "absolute_outputs": [], "missing_outputs": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    absolute: list[str] = []
    missing: list[str] = []
    for step in data.get("steps", []):
        for out in step.get("outputs", []):
            text = str(out)
            if Path(text).is_absolute() or ":" in text[:3]:
                absolute.append(text)
            p = Path(text)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                missing.append(text)
    return {
        "path": _rel(path, root),
        "exists": True,
        "absolute_outputs": absolute,
        "missing_outputs": missing,
    }


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().to_dict().items()}


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _assert(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


if __name__ == "__main__":
    main()
