"""Build a raw PTB-XL carrier protocol for distribution-first synthesis.

The v75-v77 audits showed that the previous BASE_PROTOCOL was a replay/mixture
bank rather than raw PTB morphology.  This helper creates an external-only
protocol from PTB-XL records100 lead I/II so later subtype generators can add
interpretable artifacts on top of real PTB ECG carriers.

The class labels in this carrier protocol are pool roles only; the raw carrier
itself is not a quality-labeled dataset.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, ROOT, RUN_TAG  # noqa: E402
PTBXL_DIR = ROOT / "data" / "ptb-xl"
V37_BUILDER = SCRIPT_DIR / "build_ptb_v37_subtype_balanced_distribution.py"
OUT_DIR = ANALYSIS_DIR / "raw_ptbxl_carrier_protocols"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
NOISE_NOTE_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def clean_noise_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in NOISE_NOTE_COLS:
        if col not in df.columns:
            continue
        vals = df[col].fillna("").astype(str).str.strip()
        mask &= vals.eq("")
    return mask


def normalize_lead_name(lead: str) -> str:
    value = str(lead or "I").strip().upper()
    aliases = {
        "1": "I",
        "LEAD1": "I",
        "LEAD I": "I",
        "LEADI": "I",
        "2": "II",
        "LEAD2": "II",
        "LEAD II": "II",
        "LEADII": "II",
    }
    return aliases.get(value, value)


def read_lead_lr(filename_lr: str, lead: str) -> np.ndarray:
    rec = PTBXL_DIR / str(filename_lr)
    sig, fields = wfdb.rdsamp(str(rec))
    names = [str(x).upper() for x in fields.get("sig_name", [])]
    lead_name = normalize_lead_name(lead)
    if lead_name in names:
        lead_idx = names.index(lead_name)
    else:
        # PTB-XL order starts with I, II.  Keep a deterministic fallback but
        # record it through the protocol summary so the audit remains explicit.
        lead_idx = 0 if lead_name == "I" else min(1, sig.shape[1] - 1)
    x = np.asarray(sig[:, lead_idx], dtype=np.float32)
    fs = int(round(float(fields.get("fs", 100))))
    if fs != 125:
        # PTB-XL records100 are 100 Hz, so 5/4 gives the 1250-sample 10 s
        # contract used by the rest of this experiment.
        if fs == 100:
            x = resample_poly(x, 5, 4).astype(np.float32)
        else:
            target_n = int(round(len(x) * 125.0 / float(fs)))
            # Use rational-like resampling through scipy for unusual fs.
            x = resample_poly(x, target_n, len(x)).astype(np.float32)
    if len(x) < 1250:
        x = np.pad(x, (0, 1250 - len(x)), mode="edge")
    elif len(x) > 1250:
        start = (len(x) - 1250) // 2
        x = x[start : start + 1250]
    x = x - float(np.nanmedian(x))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(int(args.seed))
    lead_name = normalize_lead_name(args.lead)
    db = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    if args.clean_noise_notes:
        db = db.loc[clean_noise_mask(db)].copy()
    db = db.dropna(subset=["filename_lr"]).copy()
    db = db.sort_values(["strat_fold", "ecg_id"]).reset_index(drop=True)
    if int(args.max_records) > 0 and len(db) > int(args.max_records):
        # Stratified fold sampling keeps morphology and acquisition mix broad
        # without making every quick distribution experiment enormous.
        take: list[int] = []
        per_fold = max(1, int(np.ceil(int(args.max_records) / max(1, db["strat_fold"].nunique()))))
        for _, sub in db.groupby("strat_fold", sort=True):
            idx = sub.index.to_numpy()
            if len(idx) > per_fold:
                idx = rng.choice(idx, size=per_fold, replace=False)
            take.extend([int(i) for i in idx])
        if len(take) > int(args.max_records):
            take = rng.choice(np.asarray(take), size=int(args.max_records), replace=False).astype(int).tolist()
        db = db.loc[sorted(take)].reset_index(drop=True)

    print(f"{now()} reading {len(db)} PTB-XL records100 lead-{lead_name} carriers", flush=True)
    carriers: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []
    for i, row in db.iterrows():
        try:
            x = read_lead_lr(str(row["filename_lr"]), lead_name)
        except Exception as exc:  # pragma: no cover - data dependent
            print(f"{now()} skip ecg_id={row.get('ecg_id')} read failed: {exc}", flush=True)
            continue
        carriers.append(x)
        kept_rows.append(row.to_dict())
        if (len(carriers) % 500) == 0:
            print(f"{now()} read {len(carriers)} carriers", flush=True)
    if not carriers:
        raise RuntimeError("no PTB-XL carriers were read")

    base_x = np.stack(carriers).astype(np.float32)
    base_meta = pd.DataFrame(kept_rows).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    global_i = 0
    for cls, y in CLASS_TO_INT.items():
        for src_i, meta in base_meta.iterrows():
            fold = int(meta.get("strat_fold", 1))
            split = "test" if fold == 10 else ("val" if fold == 9 else "train")
            row = {
                "idx": global_i,
                "source_idx": int(meta.get("ecg_id", src_i)),
                "split": split,
                "y": int(y),
                "class_name": cls,
                "record_id": f"ptbxl_{int(meta.get('ecg_id', src_i))}",
                "subject_id": f"ptbxl_patient_{int(float(meta.get('patient_id', -1))) if pd.notna(meta.get('patient_id', np.nan)) else -1}",
                "original_region": "raw_ptbxl_clean_carrier",
                "ambiguous_type": "raw_ptbxl_clean_carrier",
                "display_subtype": "raw_ptbxl_clean_carrier",
                "subtype_id": "raw_ptbxl_clean_carrier",
                "subtype_for_split": "raw_ptbxl_clean_carrier",
                "clean_policy": "raw_ptbxl_noise_note_clean_carrier",
                "filename_lr": str(meta.get("filename_lr", "")),
                "ptbxl_ecg_id": int(meta.get("ecg_id", src_i)),
                "ptbxl_strat_fold": fold,
                "ptbxl_noise_note_clean": bool(args.clean_noise_notes),
                "ptbxl_lead": lead_name,
            }
            rows.append(row)
            signals.append(base_x[src_i].astype(np.float32))
            global_i += 1

    frame = pd.DataFrame(rows)
    X = np.stack(signals).astype(np.float32)

    print(f"{now()} recomputing waveform feature columns for {len(frame)} carrier-role rows", flush=True)
    v37 = load_module("v37_for_raw_ptbxl_carrier", V37_BUILDER)
    feature_frame = v37.recompute_full_features(X, frame.copy())
    for col in feature_frame.columns:
        if col not in frame.columns or col in {"idx", "source_idx", "split", "y", "class_name", "record_id", "subject_id"}:
            continue
        frame[col] = feature_frame[col].to_numpy()
    for col in feature_frame.columns:
        if col not in frame.columns:
            frame[col] = feature_frame[col].to_numpy()

    out = OUT_DIR / str(args.name)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(long_path(out / "original_region_atlas.csv"), index=False)
    meta = frame[["idx", "source_idx", "split", "y", "class_name", "record_id", "subject_id", "filename_lr", "ptbxl_ecg_id"]].copy()
    meta.to_csv(long_path(out / "metadata.csv"), index=False)
    np.savez_compressed(long_path(out / "signals.npz"), X=X)
    summary = {
        "created_at": now(),
        "name": str(args.name),
        "source": f"PTB-XL records100 lead {lead_name} raw carrier",
        "lead": lead_name,
        "raw_carrier_count": int(len(base_x)),
        "role_rows": int(len(frame)),
        "class_counts": frame["class_name"].value_counts().to_dict(),
        "split_counts": frame.groupby(["split", "class_name"]).size().reset_index(name="n").to_dict(orient="records"),
        "clean_noise_notes": bool(args.clean_noise_notes),
        "noise_note_columns": NOISE_NOTE_COLS,
        "max_records": int(args.max_records),
        "seed": int(args.seed),
        "contract": "class labels are generation pool roles only; raw PTB carrier has no quality labels",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{now()} wrote {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680")
    parser.add_argument("--lead", type=str, default="I", help="PTB-XL lead name to use as carrier, e.g. I or II.")
    parser.add_argument("--max-records", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=20260679)
    parser.add_argument("--clean-noise-notes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()

