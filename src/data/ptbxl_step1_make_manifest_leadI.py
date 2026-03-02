from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import wfdb

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root


def _norm_rel_path(v: object) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    return s.replace("\\", "/")


def _pick_record_path(row: pd.Series, data_root: Path) -> str | None:
    hr = _norm_rel_path(row.get("filename_hr"))
    lr = _norm_rel_path(row.get("filename_lr"))

    for rel in [hr, lr]:
        if not rel:
            continue
        hea = data_root / f"{rel}.hea"
        if hea.exists():
            return rel
    return None


def main() -> None:
    root = project_root()

    input_csv = root / "artifact1" / "ptbxl" / "ptbxl_database_no_lead_I.csv"
    data_root = root / "data" / "ptb-xl"
    out_csv = root / "artifact1" / "manifests" / "ptbxl_leadI_manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = ["ecg_id", "filename_lr", "filename_hr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    total = len(df)
    ok = 0
    fail = 0
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        ecg_id = row["ecg_id"]
        record_path = _pick_record_path(row, data_root)
        if not record_path:
            fail += 1
            continue

        try:
            h = wfdb.rdheader(str(data_root / record_path))
        except Exception:
            fail += 1
            continue

        rows.append(
            {
                "ecg_id": ecg_id,
                "record_path": record_path,
                "fs_raw": float(h.fs),
                "sig_len": int(h.sig_len),
                "source": "ptbxl",
                "lead": "I",
            }
        )
        ok += 1

    out_df = pd.DataFrame(
        rows,
        columns=["ecg_id", "record_path", "fs_raw", "sig_len", "source", "lead"],
    )
    out_df.to_csv(out_csv, index=False)

    print(f"Total rows: {total}")
    print(f"Success: {ok}")
    print(f"Failed: {fail}")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()
