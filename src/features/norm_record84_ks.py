from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd

from src.utils.paths import project_root

logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# =======================
# Fixed config
# =======================
SEED = 0
STD_EPS = 1e-8   # avoid std=0
# =======================

def _outputs_exist(out_stats: Path, out_parquet: Path) -> bool:
    def ok(p: Path) -> bool:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    return ok(out_stats) and ok(out_parquet)

def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - seed: int (default SEED)
      - split_csv: str (default artifacts/splits/split_seta_seed0_balanced.csv)
      - in_parquet: str (default artifacts/features/record84.parquet)
      - out_dir: str (default artifacts/features)
      - out_stats: str (default {out_dir}/norm_stats_seed{seed}.json)
      - out_parquet: str (default {out_dir}/record84_norm.parquet)
      - std_eps: float (default STD_EPS)

    Returns dict with outputs list and skipped bool.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()

    # allow light overrides but keep defaults identical
    global SEED, STD_EPS
    if "seed" in params and params["seed"] is not None:
        SEED = int(params["seed"])
    if "std_eps" in params and params["std_eps"] is not None:
        STD_EPS = float(params["std_eps"])

    split_csv = params.get("split_csv") or (root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv")
    in_parquet = params.get("in_parquet") or (root / "artifacts" / "features" / "record84.parquet")
    out_dir = params.get("out_dir") or (root / "artifacts" / "features")

    split_csv = Path(str(split_csv))
    in_parquet = Path(str(in_parquet))
    out_dir = Path(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stats = params.get("out_stats")
    if not out_stats:
        out_stats = out_dir / f"norm_stats_seed{SEED}.json"
    else:
        out_stats = Path(str(out_stats))

    out_parquet = params.get("out_parquet")
    if not out_parquet:
        out_parquet = out_dir / "record84_norm.parquet"
    else:
        out_parquet = Path(str(out_parquet))

    if (not force) and _outputs_exist(Path(out_stats), Path(out_parquet)):
        logger.info("norm_record84_ks: outputs exist -> skip (set force=True to rerun)")
        return {"step": "norm_record84_ks", "skipped": True, "outputs": [str(out_stats), str(out_parquet)]}

    logger.info("split_csv: %s", split_csv)
    logger.info("in_features: %s", in_parquet)
    logger.info("out_stats: %s", out_stats)
    logger.info("out_features: %s", out_parquet)
    logger.info("rule: (x - median_train) / std_train, only for *__(sSQI|kSQI) columns")
    logger.info("STD_EPS=%s", STD_EPS)

    df_split = pd.read_csv(split_csv)
    df_feat = pd.read_parquet(in_parquet)

    df_split["record_id"] = df_split["record_id"].astype(str)
    df_feat["record_id"] = df_feat["record_id"].astype(str)

    df = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="left")
    if df["split"].isna().any():
        miss = df[df["split"].isna()]["record_id"].head(10).tolist()
        raise ValueError(f"Missing split info for some record_id, examples: {miss}")

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_test = int((df["split"] == "test").sum())
    logger.info("rows: train=%d, val=%d, test=%d, total=%d", n_train, n_val, n_test, len(df))

    ks_cols = [c for c in df.columns if c.endswith("__kSQI")]
    ss_cols = [c for c in df.columns if c.endswith("__sSQI")]
    target_cols = ks_cols + ss_cols
    if not target_cols:
        raise ValueError("No __kSQI/__sSQI columns found in record84.parquet")

    df_train = df.loc[df["split"] == "train", target_cols]
    med = df_train.median(axis=0, skipna=True)
    std = df_train.std(axis=0, ddof=1, skipna=True)

    std_safe = std.copy()
    tiny = std_safe.abs() < STD_EPS
    n_tiny = int(tiny.sum())
    if n_tiny > 0:
        logger.warning("tiny std columns: %d -> clamp to 1.0 (examples below)", n_tiny)
        ex = std_safe[tiny].head(10)
        for k, v in ex.items():
            logger.warning("  %s: std=%s", k, v)
        std_safe[tiny] = 1.0

    df_norm = df.copy()
    df_norm[target_cols] = (df_norm[target_cols] - med) / std_safe

    train_norm = df_norm.loc[df_norm["split"] == "train", target_cols]
    med_after = train_norm.median(axis=0, skipna=True)
    max_abs_med = float(np.max(np.abs(med_after.to_numpy())))
    logger.info("sanity: max |median_train_after| over normalized cols = %.6f", max_abs_med)
    if max_abs_med > 1e-6:
        logger.warning("train median after norm not near 0 (unexpected but may happen with NaNs).")

    stats = {
        "seed": SEED,
        "method": "(x - median_train) / std_train",
        "std_eps": STD_EPS,
        "columns": target_cols,
        "median_train": {k: float(v) for k, v in med.items()},
        "std_train": {k: float(v) for k, v in std_safe.items()},
    }
    Path(out_stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("wrote stats -> %s", out_stats)

    df_out = df_norm.drop(columns=["split"])
    df_out.to_parquet(Path(out_parquet), index=False)
    logger.info("wrote normalized features -> %s rows=%d cols=%d", out_parquet, len(df_out), df_out.shape[1])

    # keep your preview behavior (do not change functionality)
    sample_cols = ["II__sSQI", "II__kSQI"]
    sample_cols = [c for c in sample_cols if c in df_out.columns]
    if sample_cols:
        logger.info("preview (first 2 rows) columns=%s", sample_cols)
        # still prints table; keep identical
        print(df_out[["record_id", "y"] + sample_cols].head(2).to_string(index=False))

    return {"step": "norm_record84_ks", "skipped": False, "outputs": [str(out_stats), str(out_parquet)]}

def main() -> None:
    params = {
        "verbose": False,
        "force": False,
        # "seed": 0,
        # "std_eps": STD_EPS,
    }
    run(params)

if __name__ == "__main__":
    main()
