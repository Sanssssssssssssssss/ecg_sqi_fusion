# src/preprocess/resample_seta_125.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

from scipy.signal import decimate

from src.utils.paths import project_root


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# parameter
SEED = 0
FS_IN = 500
FS_OUT = 125
DECIM_Q = FS_IN // FS_OUT  # 4
assert FS_IN % FS_OUT == 0 and DECIM_Q == 4


def ensure_lead_order(sig: np.ndarray, sig_name: list[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(sig_name)}
    missing = [l for l in LEADS_12 if l not in name_to_idx]
    if missing:
        raise ValueError(f"Missing leads: {missing}. Available={sig_name}")
    idxs = [name_to_idx[l] for l in LEADS_12]
    return sig[:, idxs]


def load_record_500(record_id: str, data_dir: Path) -> tuple[np.ndarray, float, list[str]]:
    rec_path = data_dir / record_id
    rec = wfdb.rdrecord(str(rec_path), physical=True)
    sig = rec.p_signal  # (N, n_sig) physical units
    fs = float(rec.fs)
    sig_name = list(rec.sig_name) if rec.sig_name is not None else [f"ch{i}" for i in range(sig.shape[1])]
    return sig, fs, sig_name


def resample_500_to_125(sig12_500: np.ndarray) -> np.ndarray:
    """
    Anti-aliasing + decimation:
      scipy.signal.decimate(q=4, ftype='fir', zero_phase=True)
    Applied per lead (axis=0).
    """
    # decimate along time axis
    sig125 = decimate(sig12_500, q=DECIM_Q, axis=0, ftype="fir", zero_phase=True)
    return sig125.astype(np.float32)


def main() -> None:
    root = project_root()

    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    data_dir = root / "data" / "physionet" / "challenge-2011" / "set-a"
    out_dir = root / "artifacts" / "resampled_125"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Task 1.3] split_csv = {split_csv}")
    df = pd.read_csv(split_csv)

    record_ids = df["record_id"].astype(str).tolist()
    print(f"Total records in split CSV: {len(record_ids)}")
    print(f"Resample: {FS_IN}Hz -> {FS_OUT}Hz (q={DECIM_Q}), anti-alias: FIR + zero_phase")

    n_ok = 0
    n_skip = 0

    for i, rid in enumerate(record_ids, start=1):
        out_path = out_dir / f"{rid}.npz"
        if out_path.exists():
            n_skip += 1
            if (i <= 5) or (i % 200 == 0):
                print(f"[skip] {rid} exists")
            continue

        sig, fs, sig_name = load_record_500(rid, data_dir)
        if int(round(fs)) != FS_IN:
            raise ValueError(f"{rid}: expected fs={FS_IN}, got fs={fs}")

        sig12 = ensure_lead_order(sig, sig_name)  # (N, 12)
        sig125 = resample_500_to_125(sig12)

        # save to npz
        np.savez_compressed(
            out_path,
            record_id=rid,
            fs=np.array(FS_OUT, dtype=np.int32),
            leads=np.array(LEADS_12, dtype=object),
            sig_125=sig125,   # shape (N/4, 12)
        )

        n_ok += 1
        if (i <= 5) or (i % 200 == 0):
            print(f"[ok] {rid}: {sig12.shape} -> {sig125.shape} saved {out_path.name}")

    print("\n=== DONE Task 1.3 ===")
    print(f"saved new: {n_ok}")
    print(f"skipped (already existed): {n_skip}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
