from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.paths import project_root

logger = logging.getLogger(__name__)

SQI_COLUMNS = ("I__iSQI", "I__bSQI", "I__pSQI", "I__sSQI", "I__kSQI", "I__fSQI", "I__basSQI")
PROB_COLUMNS = ("prob_good", "prob_medium", "prob_bad")


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)
    root = project_root()
    sqi_dir = _path(params.get("sqi_ml_dir"), root / "outputs/transformer_e6_local_counterfactual_sqi_ml")
    out_csv = _path(params.get("out_csv"), sqi_dir / "teacher_targets" / "mlp_teacher_targets.csv")
    force = bool(params.get("force", False))
    if out_csv.exists() and not force:
        logger.info("teacher targets exist -> skip: %s", _display(out_csv, root))
        return {"step": "export_sqi_teacher_targets", "skipped": True, "outputs": [_display(out_csv, root)]}

    split_csv = sqi_dir / "splits" / "transformer_three_class_seed0.csv"
    feat_parquet = sqi_dir / "features" / "record7_norm.parquet"
    model_path = sqi_dir / "models" / "mlp" / "mlp_three_class_seed0.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"MLP teacher not found: {model_path}")

    split = pd.read_csv(split_csv)
    feat = pd.read_parquet(feat_parquet)
    split["record_id"] = split["record_id"].astype(str)
    feat["record_id"] = feat["record_id"].astype(str)
    df = split.merge(feat[["record_id", *SQI_COLUMNS]], on="record_id", how="inner")
    if len(df) != len(split):
        raise ValueError(f"feature/label merge mismatch: {len(df)} != {len(split)}")

    payload = joblib.load(model_path)
    model = payload["model"]
    probs = model.predict_proba(df[list(SQI_COLUMNS)].to_numpy(dtype=np.float64))
    if probs.shape[1] != 3:
        raise ValueError(f"expected teacher probabilities shape (*,3), got {probs.shape}")

    out = df[["source_idx", "record_id", "split", "y_class", *SQI_COLUMNS]].copy()
    out = out.rename(columns={"source_idx": "idx"})
    for j, col in enumerate(PROB_COLUMNS):
        out[col] = probs[:, j].astype(np.float32)
    out = out[["idx", "record_id", "split", "y_class", *PROB_COLUMNS, *SQI_COLUMNS]]
    out = out.sort_values("idx").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info("wrote teacher targets: %s rows=%d", _display(out_csv, root), len(out))
    return {"step": "export_sqi_teacher_targets", "skipped": False, "outputs": [_display(out_csv, root)], "rows": int(len(out))}


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SQI-ML teacher probabilities and SQI targets.")
    parser.add_argument("--sqi_ml_dir", default="outputs/transformer_e6_local_counterfactual_sqi_ml")
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
