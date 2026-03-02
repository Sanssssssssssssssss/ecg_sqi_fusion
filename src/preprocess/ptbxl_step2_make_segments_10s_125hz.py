from __future__ import annotations

import math
import sys
from pathlib import Path

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


FS_TARGET = 125
SEG_SEC = 10
SEG_SAMPLES = FS_TARGET * SEG_SEC  # 1250
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


def main() -> None:
    root = project_root()
    in_csv = root / "artifact1" / "manifests" / "ptbxl_leadI_manifest.csv"
    data_root = root / "data" / "ptb-xl"
    out_dir = root / "artifact1" / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "ptbxl_leadI_x_10s_125hz.npz"
    out_idx = out_dir / "ptbxl_leadI_segments_10s_125hz.csv"

    df = pd.read_csv(in_csv)

    x_list: list[np.ndarray] = []
    idx_rows: list[dict[str, object]] = []
    read_records = 0
    filtered = 0
    skipped = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        read_records += 1
        if i % 1000 == 0:
            print(f"Processed records: {i}/{len(df)}")

        try:
            rec_rel = str(row.record_path).replace("\\", "/")
            sig, fields = wfdb.rdsamp(str(data_root / rec_rel))
            fs_raw = float(fields["fs"])
            sig_names = list(fields.get("sig_name", []))

            lead_i = find_lead_i(sig_names)
            if lead_i is None:
                skipped += 1
                continue

            x = sig[:, lead_i] # type: ignore
            x125 = resample_to_125(x, fs_raw)
            n_seg = len(x125) // SEG_SAMPLES
            if n_seg <= 0:
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
        except Exception:
            skipped += 1

    X = np.stack(x_list, axis=0) if x_list else np.empty((0, SEG_SAMPLES), dtype=np.float32)
    np.savez_compressed(out_npz, **{NPZ_KEY: X})
    pd.DataFrame(
        idx_rows,
        columns=["seg_id", "ecg_id", "start_sec", "fs_target", "n_samples", "npz_key", "npz_index"],
    ).to_csv(out_idx, index=False)

    print(f"Read records: {read_records}")
    print(f"Generated segments: {len(idx_rows)}")
    print(f"Filtered segments: {filtered}")
    print(f"Output NPZ: {out_npz}")
    print(f"Output CSV: {out_idx}")
    print("NEXT INPUT: artifact1/segments/ptbxl_leadI_segments_10s_125hz.csv and artifact1/segments/ptbxl_leadI_x_10s_125hz.npz")


if __name__ == "__main__":
    main()
