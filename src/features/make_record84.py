# src/features/make_record84.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.paths import project_root
from src.features.sqi import (
    sqi_pSQI, sqi_basSQI, sqi_sSQI, sqi_kSQI, sqi_fSQI,
    sqi_bSQI_li2008_global,
    sqi_iSQI_li2008_global_per_lead,
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
PRINT_N = 100
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



def main() -> None:
    root = project_root()

    split_csv = root / "artifacts" / "splits" / "split_seta_seed0.csv"
    resampled_dir = root / "artifacts" / "resampled_125"
    qrs_dir = root / "artifacts" / "qrs"
    out_dir = root / "artifacts" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"split_csv: {split_csv}")
    print(f"resampled_125: {resampled_dir}")
    print(f"qrs cache: {qrs_dir}")
    print(f"Welch fixed: {WELCH_KW}")
    print(f"flatline_eps={FLATLINE_EPS}, iSQI_use={ISQI_USE}")
    print(f"bSQI/iSQI = Li2008 GLOBAL (full 10s segment), no window, no median")

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
            print(f"beat_match_tol_ms (from qrs npz) = {tol_ms_seen}ms -> {tol_samp} samples")

        # iSQI (Li2008 global): use one detector across leads
        r_for_isqi = r1_all if ISQI_USE == "r1" else r2_all
        iSQI_list = sqi_iSQI_li2008_global_per_lead(r_for_isqi, tol_samp)

        feats: list[float] = []
        for li, lead in enumerate(LEADS_12):
            x = sig12[:, li]

            iSQI = float(iSQI_list[li])

            # bSQI (Li2008 global): use BOTH detectors on same lead (symmetric union ratio)
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

        if i <= PRINT_N and PRINT_ALL_LEADS:
            print_sample_block(i, rid, y, row, LEADS_12)
        if i % 200 == 0:
            print(f"processed {i}/{len(record_ids)}")

    df_record84 = pd.DataFrame(feat_rows)
    df_lead7 = pd.DataFrame(lead_rows)

    out_record84 = out_dir / "record84.parquet"
    out_lead7 = out_dir / "lead7.parquet"

    df_record84.to_parquet(out_record84, index=False)
    df_lead7.to_parquet(out_lead7, index=False)

    print("\n=== Saved features ===")
    print(f"record84 -> {out_record84} rows={len(df_record84)} cols={df_record84.shape[1]}")
    print(f"lead7    -> {out_lead7} rows={len(df_lead7)} cols={df_lead7.shape[1]}")

    nan_any = {k: v for k, v in nan_count.items() if v > 0}
    if nan_any:
        top = sorted(nan_any.items(), key=lambda kv: kv[1], reverse=True)[:20]
        print("\n[WARN] NaN/inf features found. Top20:")
        for k, v in top:
            print(f"  {k}: {v}")
    else:
        print("\nNaN check: OK (no NaN/inf in any feature)")


if __name__ == "__main__":
    main()
