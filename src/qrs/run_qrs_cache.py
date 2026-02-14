# src/qrs/run_qrs_cache.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.paths import project_root
from src.qrs.detectors import run_xqrs, run_gqrs


# --------- configuration ---------
SEED = 0
FS = 125
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

DETECTOR1 = "xqrs"
DETECTOR2 = "gqrs"

# The tolerance is write in npz, it can be changed in sqi
BEAT_MATCH_TOL_MS = 150
# --------------------------------------------------


def load_sig125(record_id: str, resampled_dir: Path) -> tuple[np.ndarray, int, list[str]]:
    z = np.load(resampled_dir / f"{record_id}.npz", allow_pickle=True)
    sig = z["sig_125"].astype(np.float32)
    fs = int(z["fs"])
    leads = list(z["leads"].tolist())
    return sig, fs, leads


def main() -> None:
    root = project_root()
    # split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    split_csv = root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv"
    resampled_dir = root / "artifacts" / "resampled_125"
    out_dir = root / "artifacts" / "qrs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    record_ids = df["record_id"].astype(str).tolist()

    print(f"split_csv: {split_csv}")
    print(f"resampled_125: {resampled_dir}")
    print(f"out qrs dir: {out_dir}")
    print(f"fs={FS}, detectors=({DETECTOR1},{DETECTOR2}), beat_match_tol_ms={BEAT_MATCH_TOL_MS}")
    print(f"records={len(record_ids)}")

    n_ok, n_skip, n_fail = 0, 0, 0

    for i, rid in enumerate(record_ids, start=1):
        out_npz = out_dir / f"{rid}.npz"
        if out_npz.exists():
            n_skip += 1
            if i <= 5 or i % 200 == 0:
                print(f"[skip] {rid}")
            continue

        try:
            sig12, fs, leads = load_sig125(rid, resampled_dir)
            if fs != FS:
                raise ValueError(f"{rid}: expected fs={FS}, got {fs}")
            if leads != LEADS_12:
                raise ValueError(f"{rid}: lead order mismatch")

            r1_list, r2_list = [], []
            for li in range(sig12.shape[1]):
                x = sig12[:, li]
                r1_list.append(run_xqrs(x, fs=FS))
                r2_list.append(run_gqrs(x, fs=FS))

            np.savez_compressed(
                out_npz,
                record_id=rid,
                fs=np.array(FS, dtype=np.int32),
                leads=np.array(LEADS_12, dtype=object),
                detector1=np.array(DETECTOR1, dtype=object),
                detector2=np.array(DETECTOR2, dtype=object),
                beat_match_tol_ms=np.array(BEAT_MATCH_TOL_MS, dtype=np.int32),
                rpeaks_1=np.array(r1_list, dtype=object),
                rpeaks_2=np.array(r2_list, dtype=object),
            )

            n_ok += 1
            if i <= 5 or i % 200 == 0:
                print(f"[ok] {rid} -> {out_npz.name}")

        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {rid}: {type(e).__name__}: {e}")

    print("\n=== DONE cache qrs ===")
    print(f"saved : {n_ok}")
    print(f"skipped: {n_skip}")
    print(f"failed : {n_fail}")


if __name__ == "__main__":
    main()
