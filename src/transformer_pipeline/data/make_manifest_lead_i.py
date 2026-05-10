from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)


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


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    _setup_logging(bool(params.get("verbose", False)))
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer")
    data_root = _path(params.get("ptbxl_root"), root / "data" / "ptb-xl")
    force = bool(params.get("force", False))

    input_csv = artifact_dir / "ptbxl" / "ptbxl_database_no_lead_I.csv"
    out_csv = artifact_dir / "manifests" / "ptbxl_leadI_manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not force:
        logger.info("manifest: output exists -> skip (set --force to rerun)")
        return {"step": "manifest", "skipped": True, "outputs": [str(out_csv)]}
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

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if params.get("verbose") and i % 2000 == 0:
            logger.info("manifest progress: %d/%d", i, total)
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

    logger.info("Total rows: %d", total)
    logger.info("Success: %d", ok)
    logger.info("Failed: %d", fail)
    logger.info("Output: %s", _display(out_csv, root))
    return {
        "step": "manifest",
        "skipped": False,
        "outputs": [str(out_csv)],
        "total": int(total),
        "ok": int(ok),
        "fail": int(fail),
    }


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PTB-XL Lead I manifest.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--ptbxl_root", default="")
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
