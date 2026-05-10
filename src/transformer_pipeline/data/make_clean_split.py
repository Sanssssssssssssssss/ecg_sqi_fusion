from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root

logger = logging.getLogger(__name__)


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    _setup_logging(bool(params.get("verbose", False)))
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer")
    seed = int(params.get("seed", 0))
    force = bool(params.get("force", False))

    in_csv = artifact_dir / "segments" / "ptbxl_leadI_segments_10s_125hz.csv"
    out_csv = artifact_dir / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not force:
        logger.info("split: output exists -> skip (set --force to rerun)")
        return {"step": "split", "skipped": True, "outputs": [str(out_csv)]}
    if not in_csv.exists():
        raise FileNotFoundError(f"Input segments CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if "ecg_id" not in df.columns:
        raise ValueError("Input CSV must contain ecg_id")

    ids = df["ecg_id"].dropna().drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    ids = ids[rng.permutation(len(ids))]

    n = len(ids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])

    def _split(ecg_id: object) -> str:
        if ecg_id in train_ids:
            return "train"
        if ecg_id in val_ids:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out["ecg_id"].map(_split)
    keep = [c for c in ["seg_id", "ecg_id", "npz_index", "split"] if c in out.columns]
    out = out[keep] if keep else out
    out.to_csv(out_csv, index=False)

    counts = out["split"].value_counts()
    train_n = int(counts.get("train", 0))
    val_n = int(counts.get("val", 0))
    test_n = int(counts.get("test", 0))
    logger.info("train segments: %d", train_n)
    logger.info("val segments: %d", val_n)
    logger.info("test segments: %d", test_n)
    logger.info("Output: %s", _display(out_csv, root))
    return {
        "step": "split",
        "skipped": False,
        "outputs": [str(out_csv)],
        "train_segments": train_n,
        "val_segments": val_n,
        "test_segments": test_n,
    }


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split PTB-XL Lead I segments by ECG id.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
