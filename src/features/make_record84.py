# src/features/make_record84.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd

from src.utils.paths import project_root
from src.features.sqi import (
    sqi_pSQI, sqi_basSQI, sqi_sSQI, sqi_kSQI, sqi_fSQI,
    sqi_bSQI_li2008_global,
    sqi_iSQI_li2008_global_per_lead,
)

logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# =======================
# Fixed configuration
# =======================
FS = 125
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Welch parameters (fixed)
WELCH_KW = dict(
    fmax=40.0,
    window="hann",
    nperseg=256,
    noverlap=128,
    detrend="constant",
)

FLATLINE_EPS = 1e-4
PRINT_N = 5
PRINT_ALL_LEADS = True

# iSQI uses ONE detector across all leads
# qrs npz: rpeaks_1 = xqrs, rpeaks_2 = gqrs
ISQI_USE = "r1"   # "r1" or "r2"
# =======================


def load_sig125(record_id: str, resampled_dir: Path) -> np.ndarray:
    z = np.load(resampled_dir / f"{record_id}.npz", allow_pickle=True)
    sig = z["sig_125"].astype(np.float32)
    fs = int(z["fs"])
    leads = list(z["leads"].tolist())
    if fs != FS:
        raise ValueError(f"{record_id}: expected fs={FS}, got fs={fs}")
    if leads != LEADS_12:
        raise ValueError(f"{record_id}: lead order mismatch: {leads}")
    return sig


def load_qrs(record_id: str, qrs_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    z = np.load(qrs_dir / f"{record_id}.npz", allow_pickle=True)
    fs = int(z["fs"])
    if fs != FS:
        raise ValueError(f"{record_id}: qrs fs mismatch {fs}")
    r1 = [np.asarray(x, dtype=int) for x in z["rpeaks_1"].tolist()]
    r2 = [np.asarray(x, dtype=int) for x in z["rpeaks_2"].tolist()]
    tol_ms = int(z["beat_match_tol_ms"])
    return r1, r2, tol_ms

def print_sample_block(i: int, rid: str, y: int, row: dict, leads: list[str]) -> None:
    """
    Print 12-lead SQIs for one record in a readable aligned block.
    """
    # for every lead print 7 SQI
    header = f"sample {i:03d} {rid}: y={y}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # header
    print(
        f"{'lead':>4} | "
        f"{'iSQI':>6} {'bSQI':>6} {'pSQI':>6} {'sSQI':>7} {'kSQI':>7} {'fSQI':>6} {'bas':>6}"
    )
    print("-" * 72)

    for ld in leads:
        print(
            f"{ld:>4} | "
            f"{row[f'{ld}__iSQI']:6.3f} "
            f"{row[f'{ld}__bSQI']:6.3f} "
            f"{row[f'{ld}__pSQI']:6.3f} "
            f"{row[f'{ld}__sSQI']:7.3f} "
            f"{row[f'{ld}__kSQI']:7.3f} "
            f"{row[f'{ld}__fSQI']:6.3f} "
            f"{row[f'{ld}__basSQI']:6.3f}"
        )

def _outputs_exist(out_dir: Path) -> bool:
    """
    Minimal skip check: record84.parquet exists and non-empty.
    (lead7.parquet is optional but we include it if you want.)
    """
    p1 = out_dir / "record84.parquet"
    p2 = out_dir / "lead7.parquet"
    if not (p1.exists() and p1.is_file() and p1.stat().st_size > 0):
        return False
    if not (p2.exists() and p2.is_file() and p2.stat().st_size > 0):
        return False
    return True

def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - split_csv: str (default artifacts/splits/split_seta_seed0_balanced.csv)
      - resampled_dir: str (default artifacts/resampled_125)
      - qrs_dir: str (default artifacts/qrs)
      - out_dir: str (default artifacts/features)
      - print_n: int (default PRINT_N)
      - print_all_leads: bool (default PRINT_ALL_LEADS)
      - isqi_use: str ("r1" or "r2", default ISQI_USE)

    Returns dict with outputs list and skipped bool.
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()

    split_csv = params.get("split_csv") or (root / "artifacts" / "splits" / "split_seta_seed0_balanced.csv")
    resampled_dir = params.get("resampled_dir") or (root / "artifacts" / "resampled_125")
    qrs_dir = params.get("qrs_dir") or (root / "artifacts" / "qrs")
    out_dir = params.get("out_dir") or (root / "artifacts" / "features")

    split_csv = Path(str(split_csv))
    resampled_dir = Path(str(resampled_dir))
    qrs_dir = Path(str(qrs_dir))
    out_dir = Path(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional overrides (keep defaults)
    global PRINT_N, PRINT_ALL_LEADS, ISQI_USE
    if "print_n" in params and params["print_n"] is not None:
        PRINT_N = int(params["print_n"])
    if "print_all_leads" in params and params["print_all_leads"] is not None:
        PRINT_ALL_LEADS = bool(params["print_all_leads"])
    if "isqi_use" in params and params["isqi_use"] is not None:
        ISQI_USE = str(params["isqi_use"])

    if (not force) and _outputs_exist(out_dir):
        logger.info("record84: outputs exist -> skip (set force=True to rerun)")
        return {
            "step": "record84",
            "skipped": True,
            "outputs": [str(out_dir / "record84.parquet"), str(out_dir / "lead7.parquet")],
        }

    logger.info("split_csv: %s", split_csv)
    logger.info("resampled_125: %s", resampled_dir)
    logger.info("qrs cache: %s", qrs_dir)
    logger.info("Welch fixed: %s", WELCH_KW)
    logger.info("flatline_eps=%s, iSQI_use=%s", FLATLINE_EPS, ISQI_USE)
    logger.info("bSQI/iSQI = Li2008 GLOBAL (full 10s segment), no window, no median")

    df_split = pd.read_csv(split_csv)
    record_ids = df_split["record_id"].astype(str).tolist()
    id_to_y = dict(zip(df_split["record_id"].astype(str), df_split["y"].astype(int)))

    # feature names
    feature_names: list[str] = []
    for lead in LEADS_12:
        feature_names.extend([
            f"{lead}__iSQI",
            f"{lead}__bSQI",
            f"{lead}__pSQI",
            f"{lead}__sSQI",
            f"{lead}__kSQI",
            f"{lead}__fSQI",
            f"{lead}__basSQI",
        ])

    feat_rows: list[dict] = []
    lead_rows: list[dict] = []
    nan_count = {k: 0 for k in feature_names}

    tol_ms_seen = None

    for i, rid in enumerate(record_ids, start=1):
        y = int(id_to_y[rid])
        sig12 = load_sig125(rid, resampled_dir)
        r1_all, r2_all, tol_ms = load_qrs(rid, qrs_dir)

        tol_samp = int(round(tol_ms * FS / 1000.0))
        if tol_ms_seen is None:
            tol_ms_seen = tol_ms
            logger.info("beat_match_tol_ms (from qrs npz) = %dms -> %d samples", tol_ms_seen, tol_samp)

        # iSQI (Li2008 global): use one detector across leads
        r_for_isqi = r1_all if ISQI_USE == "r1" else r2_all
        iSQI_list = sqi_iSQI_li2008_global_per_lead(r_for_isqi, tol_samp)

        feats: list[float] = []
        for li, lead in enumerate(LEADS_12):
            x = sig12[:, li]

            iSQI = float(iSQI_list[li])
            bSQI = float(sqi_bSQI_li2008_global(r1_all[li], r2_all[li], tol_samp))

            pSQI = float(sqi_pSQI(x, fs=FS, welch_kwargs=WELCH_KW))
            sSQI = float(sqi_sSQI(x))
            kSQI = float(sqi_kSQI(x))
            fSQI = float(sqi_fSQI(x, flatline_eps=FLATLINE_EPS))
            basSQI = float(sqi_basSQI(x, fs=FS, welch_kwargs=WELCH_KW))

            feats.extend([iSQI, bSQI, pSQI, sSQI, kSQI, fSQI, basSQI])

            lead_rows.append({
                "record_id": rid, "y": y, "lead": lead,
                "iSQI": iSQI, "bSQI": bSQI, "pSQI": pSQI,
                "sSQI": sSQI, "kSQI": kSQI, "fSQI": fSQI, "basSQI": basSQI
            })

        # NaN/inf check
        for name, val in zip(feature_names, feats):
            if not np.isfinite(val):
                nan_count[name] += 1

        row = {"record_id": rid, "y": y}
        row.update({n: float(v) for n, v in zip(feature_names, feats)})
        feat_rows.append(row)

        # keep your original printing behavior (do not change functionality)
        if i <= PRINT_N and PRINT_ALL_LEADS:
            print_sample_block(i, rid, y, row, LEADS_12)
        if i % 200 == 0:
            logger.info("processed %d/%d", i, len(record_ids))

    df_record84 = pd.DataFrame(feat_rows)
    df_lead7 = pd.DataFrame(lead_rows)

    out_record84 = out_dir / "record84.parquet"
    out_lead7 = out_dir / "lead7.parquet"

    df_record84.to_parquet(out_record84, index=False)
    df_lead7.to_parquet(out_lead7, index=False)

    logger.info("=== Saved features ===")
    logger.info("record84 -> %s rows=%d cols=%d", out_record84, len(df_record84), df_record84.shape[1])
    logger.info("lead7    -> %s rows=%d cols=%d", out_lead7, len(df_lead7), df_lead7.shape[1])

    nan_any = {k: v for k, v in nan_count.items() if v > 0}
    if nan_any:
        top = sorted(nan_any.items(), key=lambda kv: kv[1], reverse=True)[:20]
        logger.warning("NaN/inf features found. Top20:")
        for k, v in top:
            logger.warning("  %s: %d", k, v)
    else:
        logger.info("NaN check: OK (no NaN/inf in any feature)")

    return {
        "step": "record84",
        "skipped": False,
        "outputs": [str(out_record84), str(out_lead7)],
    }

def main() -> None:
    params = {
        "verbose": False,
        "force": False,
        # 可选：调小打印
        "print_n": 5,
        "print_all_leads": True,
        # "isqi_use": "r1",
    }
    run(params)

if __name__ == "__main__":
    main()
