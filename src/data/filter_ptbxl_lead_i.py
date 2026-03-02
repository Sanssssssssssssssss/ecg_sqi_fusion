from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    # Support direct execution: python src/data/filter_ptbxl_lead_i.py
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root

TARGET_COLUMNS = [
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
]

# Match lead I as an independent token (avoid matching II/III/aVL-like words).
LEAD_I_PATTERN = re.compile(r"(?<![A-Z])I(?![A-Z])")


def has_lead_i(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(LEAD_I_PATTERN.search(str(value).upper()))


def resolve_input_csv(root: Path) -> Path:
    candidates = [
        root / "src" / "data" / "ptb-xl" / "ptbxl_database.csv",
        root / "data" / "ptb-xl" / "ptbxl_database.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find ptbxl_database.csv in: {candidates}")


def save_csv_with_fallback(df: pd.DataFrame, out_csv: Path) -> Path:
    try:
        df.to_csv(out_csv, index=False)
        return out_csv
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = out_csv.with_name(f"{out_csv.stem}_{ts}{out_csv.suffix}")
        df.to_csv(alt, index=False)
        return alt


def main() -> None:
    root = project_root()
    csv_path = resolve_input_csv(root)

    df = pd.read_csv(csv_path)

    missing_columns = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    contains_lead_i = df[TARGET_COLUMNS].apply(lambda col: col.map(has_lead_i)).any(axis=1)
    normal_df = df[~contains_lead_i].copy()

    out_dir = root / "artifact1" / "ptbxl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ptbxl_database_no_lead_I.csv"
    saved_csv = save_csv_with_fallback(normal_df, out_csv)

    print(f"Input CSV: {csv_path}")
    print(f"Total samples: {len(df)}")
    print(f"Samples containing lead I in target columns: {int(contains_lead_i.sum())}")
    print(f"Normal samples (without lead I): {len(normal_df)}")
    print(f"Saved filtered CSV: {saved_csv}")


if __name__ == "__main__":
    main()
