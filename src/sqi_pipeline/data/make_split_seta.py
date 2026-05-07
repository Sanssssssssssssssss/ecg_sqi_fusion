# src/data/make_split_seta.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import project_root

logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    split_ratio: tuple[float, float, float] = (0.70, 0.15, 0.15)


LABEL_MAP = {
    "acceptable": +1,
    "unacceptable": -1,
}


def stratified_split_indices(y: np.ndarray, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """
    Record-level stratified split (train/val/test) by y.
    Returns indices (positions) for each split.
    """
    rng = np.random.default_rng(cfg.seed)
    train_r, val_r, test_r = cfg.split_ratio
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"split_ratio must sum to 1.0, got {cfg.split_ratio}")

    idx_train, idx_val, idx_test = [], [], []

    for cls in sorted(np.unique(y)):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(train_r * n))
        n_val = int(round(val_r * n))
        # remainder -> test
        idx_train.extend(cls_idx[:n_train])
        idx_val.extend(cls_idx[n_train:n_train + n_val])
        idx_test.extend(cls_idx[n_train + n_val:])

    return {
        "train": np.array(idx_train, dtype=int),
        "val": np.array(idx_val, dtype=int),
        "test": np.array(idx_test, dtype=int),
    }


def plot_label_counts(df_split: pd.DataFrame, out_png: Path) -> None:
    table = (
        df_split.groupby(["split", "y"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    ax = table.plot(kind="bar")
    ax.set_title("Label counts by split (y=+1 acceptable, y=-1 unacceptable)")
    ax.set_xlabel("split")
    ax.set_ylabel("count")
    ax.legend(title="y")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _outputs_exist(out_path: Path, qc_png: Path) -> bool:
    def ok(p: Path) -> bool:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    return ok(out_path) and ok(qc_png)


def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - seed: int (default 0)
      - split_ratio: tuple/list of 3 floats (default (0.70,0.15,0.15))
      - artifacts_dir: str (default project_root()/artifacts)
      - manifest_csv: str (default artifacts/manifests/manifest_challenge2011_seta.csv)
      - out_csv: str (default artifacts/splits/split_seta_seed{seed}.csv)
      - qc_png: str (default artifacts/report_figs/split_seta_seed{seed}_label_counts.png)

    Returns dict with outputs list and skipped bool.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()
    seed = int(params.get("seed", 0))
    split_ratio = params.get("split_ratio", (0.70, 0.15, 0.15))
    cfg = SplitConfig(seed=seed, split_ratio=tuple(split_ratio))

    artifacts_dir = params.get("artifacts_dir")
    if not artifacts_dir:
        artifacts_dir = root / "artifacts"
    else:
        artifacts_dir = Path(str(artifacts_dir))

    manifest_path = params.get("manifest_csv")
    if not manifest_path:
        manifest_path = Path(artifacts_dir) / "manifests" / "manifest_challenge2011_seta.csv"
    else:
        manifest_path = Path(str(manifest_path))

    out_path = params.get("out_csv")
    if not out_path:
        out_path = Path(artifacts_dir) / "splits" / f"split_seta_seed{seed}.csv"
    else:
        out_path = Path(str(out_path))

    qc_png = params.get("qc_png")
    if not qc_png:
        qc_png = Path(artifacts_dir) / "report_figs" / f"split_seta_seed{seed}_label_counts.png"
    else:
        qc_png = Path(str(qc_png))

    if (not force) and _outputs_exist(out_path, qc_png):
        logger.info("split_seta: outputs exist -> skip (set force=True to rerun)")
        return {"step": "split_seta", "skipped": True, "outputs": [str(out_path), str(qc_png)]}

    logger.info("Reading manifest: %s", manifest_path)
    df = pd.read_csv(manifest_path)

    # --- sanity: required columns from your manifest script ---
    required = ["record_id", "quality_record"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"manifest missing column '{c}', got columns={list(df.columns)}")

    logger.info("Manifest rows: %d", len(df))
    logger.info("Quality counts (manifest): %s", df["quality_record"].value_counts(dropna=False).to_dict())

    # --- filter unknown and map labels ---
    df_f = df[df["quality_record"].isin(["acceptable", "unacceptable"])].copy()
    dropped = len(df) - len(df_f)
    logger.info("Filtered to acceptable/unacceptable: %d rows (dropped %d unknown/other)", len(df_f), dropped)

    df_f["y"] = df_f["quality_record"].map(LABEL_MAP).astype(int)

    # --- stratified split ---
    y = df_f["y"].to_numpy()
    split_idx = stratified_split_indices(y, cfg)

    df_f["split"] = ""
    for name, idx in split_idx.items():
        df_f.iloc[idx, df_f.columns.get_loc("split")] = name

    # --- build output ---
    out_df = pd.DataFrame(
        {
            "record_id": df_f["record_id"].astype(str),
            "y": df_f["y"].astype(int),
            "split": df_f["split"].astype(str),
            "seed": cfg.seed,
        }
    )

    # --- QC: label counts ---
    qc_table = out_df.groupby(["split", "y"]).size().unstack(fill_value=0).sort_index()
    logger.info("QC label counts by split: %s", qc_table.to_dict())

    # --- QC: record_id disjoint ---
    r_train = set(out_df.loc[out_df["split"] == "train", "record_id"])
    r_val = set(out_df.loc[out_df["split"] == "val", "record_id"])
    r_test = set(out_df.loc[out_df["split"] == "test", "record_id"])
    inter_tv = r_train & r_val
    inter_tt = r_train & r_test
    inter_vt = r_val & r_test
    logger.info("QC record_id intersections: train∩val=%d train∩test=%d val∩test=%d",
                len(inter_tv), len(inter_tt), len(inter_vt))
    if inter_tv or inter_tt or inter_vt:
        raise AssertionError("Record leakage detected across splits!")

    # --- save outputs ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Saved split CSV -> %s (rows=%d)", out_path, len(out_df))

    plot_label_counts(out_df, qc_png)
    logger.info("Saved QC plot -> %s", qc_png)

    logger.info("QC split sizes: %s", out_df["split"].value_counts().to_dict())

    return {"step": "split_seta", "skipped": False, "outputs": [str(out_path), str(qc_png)]}


def main() -> None:
    root = project_root()
    params = {
        "seed": 0,
        "split_ratio": (0.70, 0.15, 0.15),
        "artifacts_dir": str(root / "artifacts"),
        "manifest_csv": str(root / "artifacts" / "manifests" / "manifest_challenge2011_seta.csv"),
        "out_csv": str(root / "artifacts" / "splits" / "split_seta_seed0.csv"),
        "qc_png": str(root / "artifacts" / "report_figs" / "split_seta_seed0_label_counts.png"),
        "verbose": False,
        "force": False,
    }
    run(params)


if __name__ == "__main__":
    main()
