from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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

TARGET_COLUMNS = [
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
]
DEFAULT_EXCLUDE_IDS = Path("config/ptbxl_clean_source_exclude_ecg_ids.txt")

# Match lead I as an independent token, avoiding II/III/aVL-like words.
LEAD_I_PATTERN = re.compile(r"(?<![A-Z])I(?![A-Z])")


def has_lead_i(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(LEAD_I_PATTERN.search(str(value).upper()))


def has_noise_annotation(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "[]", "nan", "none"}


def resolve_input_csv(root: Path, ptbxl_root: Path) -> Path:
    candidates = [
        ptbxl_root / "ptbxl_database.csv",
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


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    _setup_logging(bool(params.get("verbose", False)))
    root = project_root()
    artifact_dir = _artifact_dir(root, params)
    ptbxl_root = _path(params.get("ptbxl_root"), root / "data" / "ptb-xl")
    force = bool(params.get("force", False))
    lead_i_only = bool(params.get("lead_i_only", False))
    exclude_ecg_ids_file = _path(params.get("exclude_ecg_ids_file"), root / DEFAULT_EXCLUDE_IDS)

    out_dir = artifact_dir / "ptbxl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ptbxl_database_no_lead_I.csv"
    if out_csv.exists() and not force:
        logger.info("filter_lead_i: output exists -> skip (set --force to rerun)")
        return {"step": "filter_lead_i", "skipped": True, "outputs": [str(out_csv)]}

    csv_path = resolve_input_csv(root, ptbxl_root)
    df = pd.read_csv(csv_path)

    missing_columns = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    contains_lead_i = df[TARGET_COLUMNS].apply(lambda col: col.map(has_lead_i)).any(axis=1)
    contains_any_noise = df[TARGET_COLUMNS].apply(lambda col: col.map(has_noise_annotation)).any(axis=1)
    manual_exclude_ids = _read_ecg_id_list(exclude_ecg_ids_file)
    manual_exclude = df["ecg_id"].isin(manual_exclude_ids) if manual_exclude_ids else pd.Series(False, index=df.index)
    reject_mask = contains_lead_i if lead_i_only else contains_any_noise
    reject_mask = reject_mask | manual_exclude
    normal_df = df[~reject_mask].copy()
    saved_csv = save_csv_with_fallback(normal_df, out_csv)

    logger.info("Input CSV: %s", _display(csv_path, root))
    logger.info("Total samples: %d", len(df))
    logger.info("Samples containing lead I in target columns: %d", int(contains_lead_i.sum()))
    logger.info("Samples containing any lead noise/electrode annotation: %d", int(contains_any_noise.sum()))
    logger.info("Manual clean-source exclusions: %d", int(manual_exclude.sum()))
    if manual_exclude_ids:
        logger.info("Manual exclusion file: %s", _display(exclude_ecg_ids_file, root))
    logger.info("Filter mode: %s", "lead_i_only" if lead_i_only else "strict_any_lead")
    logger.info("Clean samples kept: %d", len(normal_df))
    logger.info("Saved filtered CSV: %s", _display(saved_csv, root))
    return {
        "step": "filter_lead_i",
        "skipped": False,
        "outputs": [str(saved_csv)],
        "total": int(len(df)),
        "rows": int(len(normal_df)),
        "lead_i_noise_rows": int(contains_lead_i.sum()),
        "any_noise_rows": int(contains_any_noise.sum()),
        "manual_exclude_rows": int(manual_exclude.sum()),
        "manual_exclude_file": str(exclude_ecg_ids_file) if manual_exclude_ids else "",
        "filter_mode": "lead_i_only" if lead_i_only else "strict_any_lead",
    }


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter PTB-XL metadata rows with noise/electrode annotations.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--ptbxl_root", default="")
    parser.add_argument("--exclude_ecg_ids_file", default="", help="optional newline/comma separated ECG IDs to exclude from the clean source pool")
    parser.add_argument("--lead_i_only", action="store_true", help="legacy mode: reject only rows where the target columns mention lead I")
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


def _artifact_dir(root: Path, params: dict[str, Any]) -> Path:
    return _path(params.get("artifact_dir"), root / "outputs/transformer")


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _read_ecg_id_list(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            ids.add(int(token))
    return ids


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
