# src/noise/make_balanced_noisy_cases.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd
import wfdb

import matplotlib
matplotlib.use("Agg")  # safe for headless

from scipy.signal import resample_poly

from src.utils.paths import project_root
from src.noise.pca import make_3_orthogonal_from_2
from src.noise.dower import dower_3_to_12, LEADS_12


# ================== CONFIG (edit here) ==================
SEED = 0

CASE_SEC = 10.0

# STRICT: add noise at 500Hz ONLY (no downsample here)
FS_ECG = 500
N_SAMPLES = int(FS_ECG * CASE_SEC)  # 5000

# paper uses -6 dB
SNR_DB = -6.0

# paper: em + ma
NOISE_TYPES = ["em", "ma"]

# NSTDB input format
FS_NOISE_IN = 360
N_NOISE_LEADS = 2

# input split (original set-a split)
SPLIT_CSV = "artifacts/splits/split_seta_seed0.csv"

# clean ECG source (500Hz) from PhysioNet WFDB set-a
SET_A_DIR = "data/physionet/challenge-2011/set-a"

# output unified 500Hz cases: clean_500 + noisy_500
CASES_500_DIR = "artifacts/cases_500"

# output new split (balanced)
OUT_SPLIT_CSV = "artifacts/splits/split_seta_seed0_balanced.csv"

# NSTDB location
NSTDB_DIR = "data/physionet/nstdb"  # expects em.dat/em.hea, ma.dat/ma.hea

# leakage control: use first half for train/val, second half for test
HALF_POLICY = {"train": "first", "val": "first", "test": "second"}

# cap for debugging (None = full)
MAX_NEW_PER_SPLIT = None  # e.g. 50

# write an audit csv (recommended). Set False if you truly hate it :)
WRITE_AUDIT_CSV = True
# ========================================================

logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def ensure_lead_order(sig: np.ndarray, sig_name: list[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(sig_name)}
    idxs = [name_to_idx[l] for l in LEADS_12]
    return sig[:, idxs]


def load_clean_500_from_wfdb(record_id: str, set_a_dir: Path) -> np.ndarray:
    """
    Load 10s 12-lead ECG at 500Hz from WFDB set-a.
    Return shape (5000, 12) float64.
    """
    rec = wfdb.rdrecord(str(set_a_dir / record_id), physical=True)
    fs = int(round(rec.fs))
    if fs != FS_ECG:
        raise ValueError(f"{record_id}: expected fs={FS_ECG}, got {rec.fs}")

    sig = ensure_lead_order(np.asarray(rec.p_signal, dtype=np.float64), list(rec.sig_name))

    if sig.shape[0] < N_SAMPLES:
        raise ValueError(f"{record_id}: too short: {sig.shape[0]} < {N_SAMPLES}")

    sig = sig[:N_SAMPLES, :]
    if sig.shape != (N_SAMPLES, 12):
        raise ValueError(f"{record_id}: expected {(N_SAMPLES, 12)}, got {sig.shape}")

    return sig


def read_nstdb(noise_type: str, nstdb_dir: Path) -> np.ndarray:
    """
    Return raw NSTDB signal at 360Hz, physical units.
    Shape: (T, 2)
    """
    rec = wfdb.rdrecord(str(nstdb_dir / noise_type), physical=True)
    if int(round(rec.fs)) != FS_NOISE_IN:
        raise ValueError(f"{noise_type}: expected fs={FS_NOISE_IN}, got {rec.fs}")

    sig = np.asarray(rec.p_signal, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != N_NOISE_LEADS:
        raise ValueError(f"{noise_type}: expected 2 leads, got {sig.shape}")

    return sig


def choose_noise_segment_500(
    noise360: np.ndarray,
    rng: np.random.Generator,
    half: str,
) -> tuple[np.ndarray, int]:
    """
    Choose a 10s segment from first/second half of NSTDB (360Hz),
    then resample to 500Hz and return (5000, 2).
    """
    n_in = noise360.shape[0]
    mid = n_in // 2

    if half == "first":
        lo, hi = 0, mid
    elif half == "second":
        lo, hi = mid, n_in
    else:
        raise ValueError("half must be 'first' or 'second'")

    seg_len_in = int(round(CASE_SEC * FS_NOISE_IN))  # 3600
    if hi - lo <= seg_len_in + 1:
        raise ValueError(f"NSTDB half too short: available={hi-lo}, need={seg_len_in}")

    start = int(rng.integers(lo, hi - seg_len_in))
    seg360 = noise360[start : start + seg_len_in, :]  # (3600, 2)

    # resample 360 -> 500 : 500/360 = 25/18
    seg500 = resample_poly(seg360, up=25, down=18, axis=0).astype(np.float64)

    # align to exactly 5000
    if seg500.shape[0] < N_SAMPLES:
        pad = N_SAMPLES - seg500.shape[0]
        seg500 = np.pad(seg500, ((0, pad), (0, 0)), mode="edge")
    seg500 = seg500[:N_SAMPLES, :]
    return seg500, start


def scale_noise_to_snr(clean12: np.ndarray, noise12: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Per-lead scaling to achieve target SNR:
      y = x + a*v
      a = sqrt( 10^(-S/10) * Px/Pv )
    """
    factor = 10.0 ** (-snr_db / 10.0)
    eps = 1e-12

    scaled = np.zeros_like(noise12)
    for li in range(12):
        Px = float(np.mean(clean12[:, li] ** 2)) + eps
        Pv = float(np.mean(noise12[:, li] ** 2)) + eps
        a = np.sqrt(factor * (Px / Pv))
        scaled[:, li] = a * noise12[:, li]
    return scaled


def make_noisy_case_500(
    clean500_12: np.ndarray,
    noise2_500: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
) -> tuple[np.ndarray, dict]:
    """
    2-lead noise -> PCA third -> Dower 12lead -> scale to SNR -> add to clean (all at 500Hz)
    Returns (noisy500_12, meta)
    """
    noise3 = make_3_orthogonal_from_2(noise2_500, rng=rng)  # (N,3)
    noise12 = dower_3_to_12(noise3)                         # (N,12)

    noise12_scaled = scale_noise_to_snr(clean500_12, noise12, snr_db=snr_db)
    noisy500 = clean500_12 + noise12_scaled

    meta = {
        "snr_db": float(snr_db),
        "px_mean_per_lead": [float(np.mean(clean500_12[:, i] ** 2)) for i in range(12)],
        "pv_mean_per_lead_before_scale": [float(np.mean(noise12[:, i] ** 2)) for i in range(12)],
        "pv_mean_per_lead_after_scale": [float(np.mean(noise12_scaled[:, i] ** 2)) for i in range(12)],
    }
    return noisy500, meta


def write_case_500_npz(out_path: Path, sig500: np.ndarray, meta: dict) -> None:
    """
    Store as npz with key 'sig_500'
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sig_500=sig500.astype(np.float32),
        fs=np.int32(FS_ECG),
        leads=np.array(LEADS_12, dtype=object),
        # keep meta as a tiny string to avoid you hating json files on disk
        meta_str=str(meta),
    )

def cache_all_clean_cases(df_split: pd.DataFrame, root: Path) -> None:
    """
    Ensure *ALL original clean* records in split_csv are cached into cases_500/{record_id}.npz.

    This makes downstream steps simple: resample/qrs/features always read from cases_500.
    """
    set_a_dir = root / SET_A_DIR
    cases500_dir = root / CASES_500_DIR
    cases500_dir.mkdir(parents=True, exist_ok=True)

    # split_csv may contain augmented ids later; here df_split is the ORIGINAL split (no aug)
    record_ids = df_split["record_id"].astype(str).tolist()

    n_ok = 0
    n_skip = 0
    for i, rid in enumerate(record_ids, start=1):
        out_path = cases500_dir / f"{rid}.npz"
        if out_path.exists():
            n_skip += 1
            continue

        clean500 = load_clean_500_from_wfdb(rid, set_a_dir)
        meta = {"record_id": rid, "kind": "clean", "fs": FS_ECG}
        write_case_500_npz(out_path, clean500, meta=meta)
        n_ok += 1

        if (i <= 5) or (i % 200 == 0):
            logger.info("[cache clean] ok %s -> %s", rid, out_path.name)

    logger.info("[cache clean] done. new=%d, skipped(existing)=%d, total=%d", n_ok, n_skip, len(record_ids))



def balance_one_split(
    df_split: pd.DataFrame,
    split_name: str,
    nstdb_signals: dict[str, np.ndarray],
    rng: np.random.Generator,
    root: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Return:
      new_rows_for_split_csv, audit_rows
    """
    set_a_dir = root / SET_A_DIR
    cases500_dir = root / CASES_500_DIR

    df_s = df_split[df_split["split"] == split_name].copy()
    df_s["record_id"] = df_s["record_id"].astype(str)

    good = df_s[df_s["y"] == 1]["record_id"].tolist()
    bad = df_s[df_s["y"] == -1]["record_id"].tolist()

    n_good = len(good)
    n_bad = len(bad)

    target_bad = n_good
    need = max(0, target_bad - n_bad)

    if need == 0:
        logger.info("[%s] already balanced: good=%d, bad=%d", split_name, n_good, n_bad)
        return [], []

    # Each chosen clean case can generate len(NOISE_TYPES) noisy cases
    k = int(np.ceil(need / len(NOISE_TYPES)))
    k = min(k, n_good)
    if MAX_NEW_PER_SPLIT is not None:
        k = min(k, MAX_NEW_PER_SPLIT)

    chosen = rng.choice(good, size=k, replace=False).tolist()

    logger.info("[%s] good=%d, bad=%d -> need %d new bad", split_name, n_good, n_bad, need)
    logger.info("[%s] choose %d clean cases, generate %d noisy (cap to need)",split_name, k, len(NOISE_TYPES) * k)

    half = HALF_POLICY.get(split_name, "first")

    new_rows: list[dict] = []
    audit: list[dict] = []

    made = 0
    for rid in chosen:
        # cases_500 already contains ALL clean records (cached in main)
        clean500 = load_clean_500_from_wfdb(rid, set_a_dir)

        for noise_type in NOISE_TYPES:
            if made >= need:
                break

            noise360 = nstdb_signals[noise_type]
            seg2_500, start360 = choose_noise_segment_500(noise360, rng=rng, half=half)

            noisy500, meta_extra = make_noisy_case_500(clean500, seg2_500, rng=rng, snr_db=SNR_DB)

            new_id = f"{rid}__{noise_type}__snr{int(SNR_DB)}__seed{SEED}__{made:05d}"
            out500 = cases500_dir / f"{new_id}.npz"

            meta = {
                "new_record_id": new_id,
                "source_record_id": rid,
                "split": split_name,
                "y": -1,
                "noise_type": noise_type,
                "snr_db": float(SNR_DB),
                "noise_half": half,
                "noise_start_360": int(start360),
                "fs_ecg": FS_ECG,
                "fs_noise_in": FS_NOISE_IN,
                "kind": "noisy",
                **meta_extra,
            }

            write_case_500_npz(out500, noisy500, meta=meta)

            new_rows.append({
                "record_id": new_id,
                "y": -1,
                "split": split_name,
                "is_augmented": 1,
                "source_record_id": rid,
                "noise_type": noise_type,
                "snr_db": float(SNR_DB),
            })

            audit.append({
                "record_id": new_id,
                "split": split_name,
                "source_record_id": rid,
                "noise_type": noise_type,
                "snr_db": float(SNR_DB),
                "noise_half": half,
                "noise_start_360": int(start360),
                "out_npz_500": str(out500),
            })

            made += 1

        if made >= need:
            break

    return new_rows, audit

def _outputs_exist(root: Path) -> bool:
    """
    Minimal skip check:
      - balanced split csv exists and non-empty
      - cases_500 dir exists
    (We do NOT try to verify every npz for stability.)
    """
    out_csv = root / OUT_SPLIT_CSV
    cases_dir = root / CASES_500_DIR

    if not (out_csv.exists() and out_csv.is_file() and out_csv.stat().st_size > 0):
        return False
    if not cases_dir.exists():
        return False
    return True

def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.
    Minimal parameterization, defaults keep current behavior.

    params (optional):
      - verbose: bool
      - force: bool
      - seed: int
      - snr_db: float
      - split_csv: str (relative to project root or absolute)
      - out_split_csv: str
      - cases_500_dir: str
      - set_a_dir: str
      - nstdb_dir: str
      - max_new_per_split: int|None
      - write_audit_csv: bool

    Returns: {"step": "...", "skipped": bool, "outputs": [...]}
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()

    # ---- optional overrides (keep original constants as defaults) ----
    global SEED, SNR_DB, SPLIT_CSV, OUT_SPLIT_CSV, CASES_500_DIR, SET_A_DIR, NSTDB_DIR, MAX_NEW_PER_SPLIT, WRITE_AUDIT_CSV
    if "seed" in params and params["seed"] is not None:
        SEED = int(params["seed"])
    if "snr_db" in params and params["snr_db"] is not None:
        SNR_DB = float(params["snr_db"])
    if "split_csv" in params and params["split_csv"]:
        SPLIT_CSV = str(params["split_csv"])
    if "out_split_csv" in params and params["out_split_csv"]:
        OUT_SPLIT_CSV = str(params["out_split_csv"])
    if "cases_500_dir" in params and params["cases_500_dir"]:
        CASES_500_DIR = str(params["cases_500_dir"])
    if "set_a_dir" in params and params["set_a_dir"]:
        SET_A_DIR = str(params["set_a_dir"])
    if "nstdb_dir" in params and params["nstdb_dir"]:
        NSTDB_DIR = str(params["nstdb_dir"])
    if "max_new_per_split" in params:
        MAX_NEW_PER_SPLIT = params["max_new_per_split"]
    if "write_audit_csv" in params and params["write_audit_csv"] is not None:
        WRITE_AUDIT_CSV = bool(params["write_audit_csv"])

    if (not force) and _outputs_exist(root):
        out_csv = root / OUT_SPLIT_CSV
        logger.info("balanced_noise: outputs exist -> skip (set force=True to rerun)")
        outs = [str(out_csv)]
        if WRITE_AUDIT_CSV:
            outs.append(str(out_csv.with_suffix(".audit.csv")))
        return {"step": "balanced_noise", "skipped": True, "outputs": outs}

    # ---- resolve paths ----
    split_csv = root / SPLIT_CSV
    nstdb_dir = root / NSTDB_DIR

    logger.info("split_csv: %s", split_csv)
    logger.info("set-a dir (500Hz): %s", root / SET_A_DIR)
    logger.info("cases_500 out: %s", root / CASES_500_DIR)
    logger.info("nstdb_dir: %s", nstdb_dir)
    logger.info("STRICT: add noise at %dHz (no downsample here)", FS_ECG)
    logger.info("snr_db=%.1f, seed=%d, case=%.1fs (N=%d)", SNR_DB, SEED, CASE_SEC, N_SAMPLES)
    logger.info("noise types=%s, NSTDB fs=%d, half_policy=%s", NOISE_TYPES, FS_NOISE_IN, HALF_POLICY)

    df = pd.read_csv(split_csv)

    # 0) cache ALL original clean records into cases_500 first (your requirement)
    logger.info("[Step0] caching all original clean 500Hz records into cases_500 ...")
    cache_all_clean_cases(df, root)

    df["record_id"] = df["record_id"].astype(str)
    df["y"] = df["y"].astype(int)
    if "split" not in df.columns:
        raise ValueError("split_csv must have a 'split' column")

    rng = np.random.default_rng(SEED)

    # load NSTDB once
    nstdb_signals = {t: read_nstdb(t, nstdb_dir) for t in NOISE_TYPES}

    all_new_rows: list[dict] = []
    all_audit: list[dict] = []

    for split_name in ["train", "test"]:
        new_rows, audit = balance_one_split(df, split_name, nstdb_signals, rng, root)
        all_new_rows.extend(new_rows)
        all_audit.extend(audit)

    # build balanced split csv
    df_out = df.copy()
    for col, default in [
        ("is_augmented", 0),
        ("source_record_id", ""),
        ("noise_type", ""),
        ("snr_db", np.nan),
    ]:
        if col not in df_out.columns:
            df_out[col] = default

    df_new = pd.DataFrame(all_new_rows)
    df_bal = pd.concat([df_out, df_new], ignore_index=True)

    out_csv = root / OUT_SPLIT_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_csv(out_csv, index=False)
    logger.info("[saved] balanced split -> %s rows=%d (new=%d)", out_csv, len(df_bal), len(df_new))

    outputs = [str(out_csv)]

    if WRITE_AUDIT_CSV:
        audit_csv = out_csv.with_suffix(".audit.csv")
        pd.DataFrame(all_audit).to_csv(audit_csv, index=False)
        logger.info("[saved] audit -> %s rows=%d", audit_csv, len(all_audit))
        outputs.append(str(audit_csv))

    # summary
    for split_name in ["train", "test"]:
        d = df_bal[df_bal["split"] == split_name]
        n_good = int((d["y"] == 1).sum())
        n_bad = int((d["y"] == -1).sum())
        n_aug = int((d.get("is_augmented", 0) == 1).sum()) if len(d) else 0
        logger.info("[%s] after balance: good=%d, bad=%d, total=%d, augmented=%d",
                    split_name, n_good, n_bad, len(d), n_aug)

    return {"step": "balanced_noise", "skipped": False, "outputs": outputs}


def main() -> None:
    # keep single-file runnable; defaults identical to old behavior
    params = {
        "verbose": False,
        "force": False,
    }
    run(params)


if __name__ == "__main__":
    main()
