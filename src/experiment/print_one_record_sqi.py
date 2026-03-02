# src/tool/print_one_record_sqi.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "   nan"
        return f"{float(x):7.{nd}f}"
    except Exception:
        return "   nan"


def print_sample_block(i: int, rid: str, y: int, row: Dict[str, Any], leads: list[str]) -> None:
    """
    Print 12-lead SQIs for one record in a readable aligned block.
    row: dict-like, keys like 'II__bSQI'
    """
    header = f"sample {i:03d} {rid}: y={y}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    print(
        f"{'lead':>4} | "
        f"{'iSQI':>7} {'bSQI':>7} {'pSQI':>7} {'sSQI':>7} {'kSQI':>7} {'fSQI':>7} {'bas':>7}"
    )
    print("-" * 4 + "-+-" + "-" * (7 * 7 + 6))

    for ld in leads:
        iSQI = row.get(f"{ld}__iSQI", np.nan)
        bSQI = row.get(f"{ld}__bSQI", np.nan)
        pSQI = row.get(f"{ld}__pSQI", np.nan)
        sSQI = row.get(f"{ld}__sSQI", np.nan)
        kSQI = row.get(f"{ld}__kSQI", np.nan)
        fSQI = row.get(f"{ld}__fSQI", np.nan)
        bas  = row.get(f"{ld}__basSQI", np.nan)

        print(
            f"{ld:>4} | "
            f"{_fmt(iSQI)} {_fmt(bSQI)} {_fmt(pSQI)} {_fmt(sSQI)} {_fmt(kSQI)} {_fmt(fSQI)} {_fmt(bas)}"
        )


def main() -> None:
    # =========================
    # EDIT HERE
    # =========================
    PARQUET = Path(r"artifacts/relabel_stats/features/record84.parquet")
    RID = "1273536"   # <<< 改成你要看的 record_id
    RID_COL = "record_id"
    Y_COL = "y"
    # =========================

    if not PARQUET.exists():
        raise FileNotFoundError(f"Missing parquet: {PARQUET}")

    df = pd.read_parquet(PARQUET)
    if RID_COL not in df.columns:
        raise KeyError(f"Missing col {RID_COL!r} in {PARQUET.name}")
    if Y_COL not in df.columns:
        raise KeyError(f"Missing col {Y_COL!r} in {PARQUET.name}")

    sub = df[df[RID_COL].astype(str) == str(RID)]
    if len(sub) == 0:
        # help you debug: maybe rid is int in parquet
        raise ValueError(f"record_id={RID!r} not found in {PARQUET}")

    r = sub.iloc[0].to_dict()
    y = int(r.get(Y_COL, 0))

    print_sample_block(i=0, rid=str(RID), y=y, row=r, leads=LEADS_12)


if __name__ == "__main__":
    main()