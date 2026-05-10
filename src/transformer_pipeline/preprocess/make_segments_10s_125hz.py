from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample, resample_poly

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root

logger = logging.getLogger(__name__)

FS_TARGET = 125
SEG_SEC = 10
SEG_SAMPLES = FS_TARGET * SEG_SEC
NPZ_KEY = "X"


def find_lead_i(sig_names: list[str]) -> int | None:
    for i, name in enumerate(sig_names):
        if str(name).strip().upper() == "I":
            return i
    return None


def resample_to_125(x: np.ndarray, fs_raw: float) -> np.ndarray:
    if int(round(fs_raw)) == FS_TARGET:
        return x.astype(np.float32)
    fs_int = int(round(fs_raw))
    if abs(fs_raw - fs_int) < 1e-8 and fs_int > 0:
        g = math.gcd(FS_TARGET, fs_int)
        up, down = FS_TARGET // g, fs_int // g
        y = resample_poly(x, up=up, down=down)
    else:
        n_out = int(round(len(x) * FS_TARGET / fs_raw))
        y = resample(x, n_out)
    return y.astype(np.float32)


def is_flatline_segment(seg: np.ndarray) -> bool:
    return float(np.std(seg)) < 1e-4 or float(np.max(seg) - np.min(seg)) < 1e-3


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    _setup_logging(verbose)
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer")
    data_root = _path(params.get("ptbxl_root"), root / "data" / "ptb-xl")
    force = bool(params.get("force", False))

    in_csv = artifact_dir / "manifests" / "ptbxl_leadI_manifest.csv"
    out_dir = artifact_dir / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "ptbxl_leadI_x_10s_125hz.npz"
    out_idx = out_dir / "ptbxl_leadI_segments_10s_125hz.csv"

    if out_npz.exists() and out_idx.exists() and not force:
        logger.info("segments: outputs exist -> skip (set --force to rerun)")
        return {"step": "segments", "skipped": True, "outputs": [str(out_npz), str(out_idx)]}
    if not in_csv.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_csv}")

    df = pd.read_csv(in_csv)
    x_list: list[np.ndarray] = []
    idx_rows: list[dict[str, object]] = []
    read_records = 0
    filtered = 0
    skipped = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        read_records += 1
        if verbose and i % 1000 == 0:
            logger.info("segments progress: %d/%d", i, len(df))

        try:
            rec_rel = str(row.record_path).replace("\\", "/")
            sig, fields = wfdb.rdsamp(str(data_root / rec_rel))
            fs_raw = float(fields["fs"])
            sig_names = list(fields.get("sig_name", []))

            lead_i = find_lead_i(sig_names)
            if lead_i is None:
                skipped += 1
                continue

            x = sig[:, lead_i]
            x125 = resample_to_125(x, fs_raw)
            n_seg = len(x125) // SEG_SAMPLES
            if n_seg <= 0:
                skipped += 1
                continue

            for s in range(n_seg):
                seg = x125[s * SEG_SAMPLES : (s + 1) * SEG_SAMPLES]
                if is_flatline_segment(seg):
                    filtered += 1
                    continue
                npz_index = len(x_list)
                x_list.append(seg.astype(np.float32, copy=False))
                idx_rows.append(
                    {
                        "seg_id": len(idx_rows),
                        "ecg_id": row.ecg_id,
                        "start_sec": float(s * SEG_SEC),
                        "fs_target": FS_TARGET,
                        "n_samples": SEG_SAMPLES,
                        "npz_key": NPZ_KEY,
                        "npz_index": npz_index,
                    }
                )
        except Exception as exc:
            skipped += 1
            if verbose:
                logger.debug("skip record %s: %s", getattr(row, "record_path", "?"), exc)

    X = np.stack(x_list, axis=0) if x_list else np.empty((0, SEG_SAMPLES), dtype=np.float32)
    np.savez(out_npz, **{NPZ_KEY: X})
    pd.DataFrame(
        idx_rows,
        columns=["seg_id", "ecg_id", "start_sec", "fs_target", "n_samples", "npz_key", "npz_index"],
    ).to_csv(out_idx, index=False)

    logger.info("Read records: %d", read_records)
    logger.info("Generated segments: %d", len(idx_rows))
    logger.info("Filtered segments: %d", filtered)
    logger.info("Skipped records: %d", skipped)
    logger.info("Output NPZ: %s", _display(out_npz, root))
    logger.info("Output CSV: %s", _display(out_idx, root))
    return {
        "step": "segments",
        "skipped": False,
        "outputs": [str(out_npz), str(out_idx)],
        "read_records": int(read_records),
        "generated_segments": int(len(idx_rows)),
        "filtered_segments": int(filtered),
        "skipped_records": int(skipped),
    }


def main() -> None:
    args = _parse_args()
    run(vars(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make PTB-XL Lead I 10 s segments at 125 Hz.")
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
