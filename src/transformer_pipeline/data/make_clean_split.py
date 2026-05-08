from __future__ import annotations

import sys
from pathlib import Path

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


def main() -> None:
    root = project_root()
    in_csv = root / "artifact1" / "segments" / "ptbxl_leadI_segments_10s_125hz.csv"
    out_csv = root / "artifact1" / "splits" / "ptbxl_leadI_clean_10s_125hz_split.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "ecg_id" not in df.columns:
        raise ValueError("Input CSV must contain ecg_id")

    ids = df["ecg_id"].dropna().drop_duplicates().to_numpy()
    rng = np.random.default_rng(0)
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
    print(f"train segments: {int(counts.get('train', 0))}")
    print(f"val segments: {int(counts.get('val', 0))}")
    print(f"test segments: {int(counts.get('test', 0))}")
    print(f"Output: {out_csv}")
    print("NEXT INPUT: artifact1/splits/ptbxl_leadI_clean_10s_125hz_split.csv")


if __name__ == "__main__":
    main()
