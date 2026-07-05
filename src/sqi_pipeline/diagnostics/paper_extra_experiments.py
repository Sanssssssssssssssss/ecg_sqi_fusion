from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.sqi_pipeline.features.make_record84 import LEADS_12, WELCH_KW
from src.sqi_pipeline.features.sqi import (
    band_power,
    sqi_bSQI_paper_wqrs_eplimited,
    sqi_fSQI,
    sqi_kSQI,
    sqi_sSQI,
    sqi_iSQI_paper_all_leads_per_lead,
    welch_psd,
)
from src.sqi_pipeline.models.lm_mlp import LMConfig, LMMLP
from src.sqi_pipeline.qrs.paper_detectors import (
    PaperQRSExecutables,
    resolve_paper_qrs_executables,
    run_eplimited_multilead,
    run_paper_qrs_12lead,
    run_paper_qrs_multilead,
    run_wqrs_multilead,
)
from src.utils.paths import project_root


FS_OUT = 125
WINDOW_SEC = 10
BEAT_MATCH_TOL_MS = 150
EPLIMITED_WARMUP_SEC = 8.0
SELECTED5 = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]
PAPER_TABLE8_MS = {
    "kSQI": 0.33,
    "sSQI": 0.29,
    "pSQI/basSQI shared PSD": 1.92,
    "fSQI": 0.07,
    "P&T/eplimited": 2.46,
    "wqrs": 33.18,
    "SVM predict": 0.10,
    "MLP predict": 0.001,
}
C_LIST = (1.0, 2.0, 8.0, 25.0, 32.0, 128.0, 512.0)
GAMMA_LIST = (2.0**-11, 2.0**-9, 2.0**-7, 2.0**-5, 2.0**-3, 0.14, 0.5, 0.7, 1.0, 1.5, 2.0)


@dataclass
class SingleLeadProxy:
    svm: Pipeline
    svm_threshold: float
    mlp: LMMLP
    mlp_scaler: StandardScaler
    mlp_threshold: float
    feature_cols: list[str]


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _max_acc_threshold(y01: np.ndarray, p: np.ndarray, n_grid: int = 2001) -> dict[str, Any]:
    y = y01.astype(int).ravel()
    p = p.astype(float).ravel()
    best = {"threshold": 0.5, "acc": -1.0, "se": np.nan, "sp": np.nan, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for threshold in np.linspace(0.0, 1.0, int(n_grid)):
        pred = (p > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        acc = (tn + tp) / max(1, tn + fp + fn + tp)
        if acc > best["acc"]:
            best = {
                "threshold": float(threshold),
                "acc": float(acc),
                "se": float(tp / max(1, tp + fn)),
                "sp": float(tn / max(1, tn + fp)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
    return best


def _metrics(y01: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = y01.astype(int).ravel()
    pred = (p.astype(float).ravel() > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = float("nan")
    return {
        "Ac": float((tn + tp) / max(1, tn + fp + fn + tp)),
        "Se": float(tp / max(1, tp + fn)),
        "Sp": float(tn / max(1, tn + fp)),
        "AUC": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _summary_ms(df: pd.DataFrame, group_cols: list[str], value_col: str = "elapsed_ms") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        vals = g[value_col].astype(float).to_numpy()
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: v for c, v in zip(group_cols, key)}
        row.update(
            {
                "n": int(len(vals)),
                "mean_ms": float(np.mean(vals)),
                "median_ms": float(np.median(vals)),
                "p95_ms": float(np.percentile(vals, 95)),
                "std_ms": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min_ms": float(np.min(vals)),
                "max_ms": float(np.max(vals)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_md_value(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.3f}"
    return str(x)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt_md_value(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _shared_psd_features(x: np.ndarray) -> tuple[float, float]:
    f, p = welch_psd(x, fs=FS_OUT, **WELCH_KW)
    psqi = band_power(f, p, 5.0, 15.0) / (band_power(f, p, 5.0, 40.0) + 1e-12)
    bassqi = 1.0 - band_power(f, p, 0.0, 1.0) / (band_power(f, p, 0.0, 40.0) + 1e-12)
    return float(psqi), float(bassqi)


def _singlelead_features(x: np.ndarray, wqrs: np.ndarray, eplimited: np.ndarray) -> dict[str, float]:
    p_sqi, bas_sqi = _shared_psd_features(x)
    tol_samp = int(round(BEAT_MATCH_TOL_MS * FS_OUT / 1000.0))
    return {
        "bSQI": float(sqi_bSQI_paper_wqrs_eplimited(wqrs, eplimited, tol_samp)),
        "basSQI": bas_sqi,
        "kSQI": float(sqi_kSQI(x)),
        "sSQI": float(sqi_sSQI(x)),
        "fSQI": float(sqi_fSQI(x)),
        "pSQI": p_sqi,
    }


def _record84_from_qrs(sig12: np.ndarray, wqrs_all: list[np.ndarray], epl_all: list[np.ndarray]) -> dict[str, float]:
    tol_samp = int(round(BEAT_MATCH_TOL_MS * FS_OUT / 1000.0))
    i_sqi = sqi_iSQI_paper_all_leads_per_lead(wqrs_all, tol_samp)
    row: dict[str, float] = {}
    for i, lead in enumerate(LEADS_12):
        x = sig12[:, i]
        single = _singlelead_features(x, wqrs_all[i], epl_all[i])
        row[f"{lead}__iSQI"] = float(i_sqi[i])
        row[f"{lead}__bSQI"] = single["bSQI"]
        row[f"{lead}__pSQI"] = single["pSQI"]
        row[f"{lead}__sSQI"] = single["sSQI"]
        row[f"{lead}__kSQI"] = single["kSQI"]
        row[f"{lead}__fSQI"] = single["fSQI"]
        row[f"{lead}__basSQI"] = single["basSQI"]
    return row


def _normalize_record84_row(row: dict[str, float], norm_stats: dict[str, Any]) -> dict[str, float]:
    out = dict(row)
    for c in norm_stats.get("columns", []):
        if c in out:
            med = float(norm_stats["median_train"][c])
            std = float(norm_stats["std_train"][c])
            out[c] = float((out[c] - med) / (std if abs(std) > 1e-12 else 1.0))
    return out


def _selected5_record_cols() -> list[str]:
    return [f"{lead}__{sqi}" for lead in LEADS_12 for sqi in SELECTED5]


def ensure_mitbih(mitbih_dir: Path) -> list[str]:
    mitbih_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(p.stem for p in mitbih_dir.glob("*.hea"))
    if not records:
        wfdb.dl_database(
            "mitdb",
            dl_dir=str(mitbih_dir),
            records="all",
            annotators=["atr"],
            keep_subdirs=False,
            overwrite=False,
        )
        records = sorted(p.stem for p in mitbih_dir.glob("*.hea"))
    if not records:
        raise RuntimeError(f"MIT-BIH download/listing produced no records under {mitbih_dir}")
    return records


def _resample_to_125(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if int(round(fs)) == FS_OUT:
        return x.astype(np.float32)
    up = FS_OUT
    down = int(round(fs))
    g = math.gcd(up, down)
    y = resample_poly(x, up // g, down // g, axis=0)
    return y.astype(np.float32)


def _load_challenge_lead_proxy_data(artifacts_dir: Path) -> pd.DataFrame:
    lead7 = pd.read_parquet(artifacts_dir / "features" / "lead7.parquet")
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv")
    split = split[["record_id", "split"]].copy()
    split["record_id"] = split["record_id"].astype(str)
    lead7["record_id"] = lead7["record_id"].astype(str)
    df = lead7.merge(split, on="record_id", how="left")
    if df["split"].isna().any():
        raise ValueError("lead7 rows missing split assignment")
    df["y01"] = (df["y"].astype(int) == 1).astype(int)
    if not np.isfinite(df[SELECTED5].to_numpy(dtype=float)).all():
        raise ValueError("non-finite values in challenge lead-level selected-five features")
    return df


def train_singlelead_proxy(artifacts_dir: Path, out_dir: Path, seed: int) -> SingleLeadProxy:
    df = _load_challenge_lead_proxy_data(artifacts_dir)
    tr = df["split"] == "train"
    va = df["split"] == "val"
    te = df["split"] == "test"

    Xtr = df.loc[tr, SELECTED5].to_numpy(dtype=float)
    ytr = df.loc[tr, "y01"].to_numpy(dtype=int)
    Xva = df.loc[va, SELECTED5].to_numpy(dtype=float)
    yva = df.loc[va, "y01"].to_numpy(dtype=int)
    Xte = df.loc[te, SELECTED5].to_numpy(dtype=float)
    yte = df.loc[te, "y01"].to_numpy(dtype=int)

    # Use the selected-five RBF parameters from the paper-aligned Table 7 run.
    # This keeps the weak-label proxy lightweight and avoids treating it as a
    # separately optimized paper result.
    best = {"C": 25.0, "gamma": 0.03125}
    probe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], probability=True, random_state=seed)),
        ]
    )
    probe.fit(Xtr, ytr)
    best["threshold"] = _max_acc_threshold(yva, probe.predict_proba(Xva)[:, 1].astype(float))

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], probability=True, random_state=seed)),
        ]
    )
    Xtrv = np.concatenate([Xtr, Xva], axis=0)
    ytrv = np.concatenate([ytr, yva], axis=0)
    svm.fit(Xtrv, ytrv)
    p_svm_test = svm.predict_proba(Xte)[:, 1].astype(float)
    svm_threshold = float(best["threshold"]["threshold"])

    mlp_scaler = StandardScaler().fit(Xtr)
    Xtr_s = mlp_scaler.transform(Xtr)
    Xva_s = mlp_scaler.transform(Xva)
    Xte_s = mlp_scaler.transform(Xte)
    device = torch.device("cpu")
    dtype = torch.float64
    mlp = LMMLP(J=5, D=len(SELECTED5), device=device, dtype=dtype, seed=seed)
    mlp.fit_lm(
        torch.tensor(Xtr_s, dtype=dtype),
        torch.tensor(ytr.astype(float), dtype=dtype),
        LMConfig(epochs_max=100),
        X_val=torch.tensor(Xva_s, dtype=dtype),
        y_val=torch.tensor(yva.astype(float), dtype=dtype),
        model_select_metric="val_auc",
        patience=15,
        threshold=0.5,
    )
    p_mlp_val = mlp.predict_proba(torch.tensor(Xva_s, dtype=dtype))
    mlp_thr = _max_acc_threshold(yva, p_mlp_val)
    p_mlp_test = mlp.predict_proba(torch.tensor(Xte_s, dtype=dtype))

    rows = [
        {
            "model": "singlelead_weak_svm",
            "n_train_leads": int(len(ytr)),
            "n_val_leads": int(len(yva)),
            "n_test_leads": int(len(yte)),
            "C": best["C"],
            "gamma": best["gamma"],
            "threshold": svm_threshold,
            **{f"test_{k}": v for k, v in _metrics(yte, p_svm_test, svm_threshold).items()},
        },
        {
            "model": "singlelead_weak_lm_mlp_J5",
            "n_train_leads": int(len(ytr)),
            "n_val_leads": int(len(yva)),
            "n_test_leads": int(len(yte)),
            "C": np.nan,
            "gamma": np.nan,
            "threshold": float(mlp_thr["threshold"]),
            **{f"test_{k}": v for k, v in _metrics(yte, p_mlp_test, float(mlp_thr["threshold"])).items()},
        },
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "challenge_singlelead_proxy_metrics.csv", index=False)
    _json_dump(
        out_dir / "challenge_singlelead_proxy_config.json",
        {"feature_cols": SELECTED5, "label_source": "record-level Challenge Set-a weak label copied to each lead"},
    )
    return SingleLeadProxy(svm=svm, svm_threshold=svm_threshold, mlp=mlp, mlp_scaler=mlp_scaler, mlp_threshold=float(mlp_thr["threshold"]), feature_cols=list(SELECTED5))


def build_mitbih_features(
    *,
    mitbih_dir: Path,
    out_dir: Path,
    executables: PaperQRSExecutables,
    qrs_work_dir: Path,
    max_records: int | None,
    max_windows_per_record: int | None,
) -> pd.DataFrame:
    records = ensure_mitbih(mitbih_dir)
    if max_records is not None:
        records = records[: int(max_records)]
    parts_dir = out_dir / "mitbih_feature_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    window_samples = FS_OUT * WINDOW_SEC

    manifest_rows = []
    for rec_i, record_id in enumerate(records, start=1):
        part_path = parts_dir / f"{record_id}.csv"
        if part_path.exists() and part_path.stat().st_size > 0:
            part = pd.read_csv(part_path, usecols=["record_id", "window_idx", "lead"])
            manifest_rows.append({"record_id": record_id, "status": "cached", "rows": int(len(part))})
            continue

        rec = wfdb.rdrecord(str(mitbih_dir / record_id))
        sig = np.asarray(rec.p_signal, dtype=np.float64)
        if sig.ndim != 2:
            raise ValueError(f"{record_id}: expected 2D p_signal, got {sig.shape}")
        sig125 = _resample_to_125(sig, float(rec.fs))
        n_windows = int(sig125.shape[0] // window_samples)
        if max_windows_per_record is not None:
            n_windows = min(n_windows, int(max_windows_per_record))
        leads = [str(x) for x in rec.sig_name]

        rows: list[dict[str, Any]] = []
        for wi in range(n_windows):
            start = wi * window_samples
            xw = sig125[start : start + window_samples, :]
            wqrs_all, epl_all = run_paper_qrs_multilead(
                record_id=f"mitbih_{record_id}_{wi:04d}",
                sig=xw,
                fs=FS_OUT,
                leads=leads,
                executables=executables,
                work_dir=qrs_work_dir,
                eplimited_warmup_sec=EPLIMITED_WARMUP_SEC,
            )
            for li, lead in enumerate(leads):
                feats = _singlelead_features(xw[:, li], wqrs_all[li], epl_all[li])
                rows.append(
                    {
                        "record_id": record_id,
                        "window_idx": wi,
                        "lead": lead,
                        "start_s": float(wi * WINDOW_SEC),
                        "fs": FS_OUT,
                        "n_wqrs": int(len(wqrs_all[li])),
                        "n_eplimited": int(len(epl_all[li])),
                        **{k: feats[k] for k in ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "pSQI"]},
                    }
                )
        pd.DataFrame(rows).to_csv(part_path, index=False)
        manifest_rows.append({"record_id": record_id, "status": "computed", "rows": int(len(rows))})
        print(f"[MIT-BIH] {rec_i}/{len(records)} {record_id}: rows={len(rows)}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "mitbih_feature_manifest.csv", index=False)
    frames = [pd.read_csv(p) for p in sorted(parts_dir.glob("*.csv")) if p.stem in set(records)]
    if not frames:
        raise RuntimeError("No MIT-BIH feature parts available")
    features = pd.concat(frames, ignore_index=True)
    features.to_csv(out_dir / "mitbih_window_features.csv", index=False)
    return features


def predict_mitbih(features: pd.DataFrame, proxy: SingleLeadProxy, out_dir: Path) -> dict[str, Any]:
    X = features[proxy.feature_cols].to_numpy(dtype=float)
    p_svm = proxy.svm.predict_proba(X)[:, 1].astype(float)
    p_mlp = proxy.mlp.predict_proba(torch.tensor(proxy.mlp_scaler.transform(X), dtype=torch.float64))

    out = features.copy()
    out["p_accept_svm"] = p_svm
    out["accept_svm"] = (p_svm > proxy.svm_threshold).astype(int)
    out["p_accept_mlp"] = p_mlp
    out["accept_mlp"] = (p_mlp > proxy.mlp_threshold).astype(int)
    out["assumed_quality_label"] = "acceptable_proxy"
    out.to_csv(out_dir / "mitbih_window_predictions.csv", index=False)

    summary_rows = []
    for model, accept_col, prob_col, thr in [
        ("singlelead_weak_svm", "accept_svm", "p_accept_svm", proxy.svm_threshold),
        ("singlelead_weak_lm_mlp_J5", "accept_mlp", "p_accept_mlp", proxy.mlp_threshold),
    ]:
        summary_rows.append(
            {
                "model": model,
                "threshold": float(thr),
                "n_windows_leads": int(len(out)),
                "n_records": int(out["record_id"].nunique()),
                "n_leads": int(out[["record_id", "lead"]].drop_duplicates().shape[0]),
                "acceptance_rate": float(out[accept_col].mean()),
                "false_rejection_rate_proxy": float(1.0 - out[accept_col].mean()),
                "mean_p_accept": float(out[prob_col].mean()),
                "median_p_accept": float(out[prob_col].median()),
            }
        )
    overall = pd.DataFrame(summary_rows)
    overall.to_csv(out_dir / "mitbih_transfer_overall_summary.csv", index=False)

    per_record = (
        out.groupby("record_id")
        .agg(
            n_windows_leads=("accept_svm", "size"),
            svm_acceptance_rate=("accept_svm", "mean"),
            mlp_acceptance_rate=("accept_mlp", "mean"),
            mean_p_accept_svm=("p_accept_svm", "mean"),
            mean_p_accept_mlp=("p_accept_mlp", "mean"),
        )
        .reset_index()
    )
    per_record.to_csv(out_dir / "mitbih_transfer_per_record_summary.csv", index=False)
    return {"overall": overall.to_dict(orient="records"), "per_record_rows": int(len(per_record))}


def _fit_12lead_svm_for_benchmark(artifacts_dir: Path, seed: int) -> Pipeline:
    feat = pd.read_parquet(artifacts_dir / "features" / "record84_norm.parquet")
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv", usecols=["record_id", "split"])
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    df = feat.merge(split, on="record_id", how="left")
    cols = _selected5_record_cols()
    mask = df["split"].isin(["train", "val"])
    X = df.loc[mask, cols].to_numpy(dtype=float)
    y = (df.loc[mask, "y"].astype(int) == 1).astype(int).to_numpy()
    table7 = pd.read_csv(artifacts_dir / "models" / "svm" / "table7_svm_selected5_seed0.csv")
    C = float(table7.iloc[0]["best_C"])
    gamma = float(table7.iloc[0]["best_gamma"])
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _load_84_mlp(artifacts_dir: Path) -> tuple[LMMLP, list[str]]:
    model_dir = artifacts_dir / "models" / "lm_mlp" / "models"
    candidates = sorted(model_dir.glob("model_84-*-1_seed0.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No 84-feature MLP model found under {model_dir}")
    with open(candidates[0], "rb") as f:
        payload = pickle.load(f)
    model = LMMLP.from_pickle_dict(payload["model"], device=torch.device("cpu"), dtype=torch.float64)
    return model, list(payload["feature_columns"])


def benchmark(
    *,
    artifacts_dir: Path,
    out_dir: Path,
    executables: PaperQRSExecutables,
    proxy: SingleLeadProxy,
    seed: int,
    n_records: int,
) -> None:
    split = pd.read_csv(artifacts_dir / "splits" / "split_seta_seed0_paper_balanced.csv")
    clean = split[~split["record_id"].astype(str).str.contains("__paper_", regex=False)].copy()
    rng = np.random.default_rng(seed)
    ids = clean["record_id"].astype(str).to_numpy()
    choose = rng.choice(ids, size=min(int(n_records), len(ids)), replace=False)
    resampled_dir = artifacts_dir / "resampled_125"
    qrs_work_dir = out_dir / "qrs_tmp"
    qrs_work_dir.mkdir(parents=True, exist_ok=True)
    norm_stats = json.loads((artifacts_dir / "features" / "norm_stats_seed0.json").read_text(encoding="utf-8"))
    svm12 = _fit_12lead_svm_for_benchmark(artifacts_dir, seed)
    mlp84, mlp84_cols = _load_84_mlp(artifacts_dir)

    component_rows: list[dict[str, Any]] = []
    end_rows: list[dict[str, Any]] = []
    for i, rid in enumerate(choose, start=1):
        z = np.load(resampled_dir / f"{rid}.npz", allow_pickle=True)
        sig12 = z["sig_125"].astype(np.float64)
        lead_idx = 1 if sig12.shape[1] > 1 else 0
        x = sig12[:, lead_idx]
        one_sig = x[:, None]
        one_lead = [LEADS_12[lead_idx]]

        t0 = time.perf_counter(); _ = sqi_kSQI(x); component_rows.append({"record_id": rid, "scope": "per_lead", "component": "kSQI", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})
        t0 = time.perf_counter(); _ = sqi_sSQI(x); component_rows.append({"record_id": rid, "scope": "per_lead", "component": "sSQI", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})
        t0 = time.perf_counter(); _ = sqi_fSQI(x); component_rows.append({"record_id": rid, "scope": "per_lead", "component": "fSQI", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})
        t0 = time.perf_counter(); _ = _shared_psd_features(x); component_rows.append({"record_id": rid, "scope": "per_lead", "component": "shared_pSQI_basSQI_PSD", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})

        t0 = time.perf_counter()
        wqrs = run_wqrs_multilead(record_id=f"bench_{rid}_wqrs", sig=one_sig, fs=FS_OUT, leads=one_lead, executable=executables.wqrs, work_dir=qrs_work_dir)[0]
        component_rows.append({"record_id": rid, "scope": "per_lead", "component": "wqrs", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})
        t0 = time.perf_counter()
        epl = run_eplimited_multilead(record_id=f"bench_{rid}_epl", sig=one_sig, fs=FS_OUT, leads=one_lead, executable=executables.eplimited, work_dir=qrs_work_dir, eplimited_warmup_sec=EPLIMITED_WARMUP_SEC)[0]
        component_rows.append({"record_id": rid, "scope": "per_lead", "component": "eplimited_PandT", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})

        feat = _singlelead_features(x, wqrs, epl)
        X1 = np.asarray([[feat[c] for c in proxy.feature_cols]], dtype=float)
        t0 = time.perf_counter(); _ = proxy.svm.predict_proba(X1); component_rows.append({"record_id": rid, "scope": "per_lead", "component": "SVM_predict", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})
        X1s = proxy.mlp_scaler.transform(X1)
        t0 = time.perf_counter(); _ = proxy.mlp.predict_proba(torch.tensor(X1s, dtype=torch.float64)); component_rows.append({"record_id": rid, "scope": "per_lead", "component": "MLP_predict", "elapsed_ms": (time.perf_counter() - t0) * 1000.0})

        total_start = time.perf_counter()
        qrs_start = time.perf_counter()
        w12, e12 = run_paper_qrs_12lead(
            record_id=f"bench12_{rid}",
            sig12=sig12,
            fs=FS_OUT,
            leads=LEADS_12,
            executables=executables,
            work_dir=qrs_work_dir,
            eplimited_warmup_sec=EPLIMITED_WARMUP_SEC,
        )
        qrs_ms = (time.perf_counter() - qrs_start) * 1000.0
        feat_start = time.perf_counter()
        row84 = _normalize_record84_row(_record84_from_qrs(sig12, w12, e12), norm_stats)
        feat_ms = (time.perf_counter() - feat_start) * 1000.0
        svm_start = time.perf_counter()
        _ = svm12.predict_proba(np.asarray([[row84[c] for c in _selected5_record_cols()]], dtype=float))
        svm_ms = (time.perf_counter() - svm_start) * 1000.0
        mlp_start = time.perf_counter()
        _ = mlp84.predict_proba(torch.tensor(np.asarray([[row84[c] for c in mlp84_cols]], dtype=float), dtype=torch.float64))
        mlp_ms = (time.perf_counter() - mlp_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        end_rows.append(
            {
                "record_id": rid,
                "scope": "full_12lead",
                "qrs_ms": qrs_ms,
                "feature84_ms": feat_ms,
                "svm_predict_ms": svm_ms,
                "mlp_predict_ms": mlp_ms,
                "total_ms": total_ms,
            }
        )
        if i % 20 == 0:
            print(f"[benchmark] {i}/{len(choose)}")

    comp = pd.DataFrame(component_rows)
    comp.to_csv(out_dir / "component_timing_rows.csv", index=False)
    _summary_ms(comp, ["scope", "component"]).to_csv(out_dir / "component_timing_summary.csv", index=False)
    end = pd.DataFrame(end_rows)
    end.to_csv(out_dir / "end_to_end_timing_rows.csv", index=False)
    end_long = end.melt(id_vars=["record_id", "scope"], value_vars=["qrs_ms", "feature84_ms", "svm_predict_ms", "mlp_predict_ms", "total_ms"], var_name="component", value_name="elapsed_ms")
    _summary_ms(end_long, ["scope", "component"]).to_csv(out_dir / "end_to_end_timing_summary.csv", index=False)
    pd.DataFrame([{"component": k, "paper_ms": v} for k, v in PAPER_TABLE8_MS.items()]).to_csv(out_dir / "paper_table8_reference.csv", index=False)
    _json_dump(
        out_dir / "benchmark_manifest.json",
        {
            "seed": seed,
            "n_records": int(len(choose)),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "note": "Paper Table 8 is single-lead Matlab timing; this benchmark reports both per-lead and full 12-lead Python/WFDB-wrapper timing.",
        },
    )


def write_report(extra_dir: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    challenge = pd.read_csv(extra_dir / "challenge_singlelead_proxy_metrics.csv")
    mit_overall = pd.read_csv(extra_dir / "mitbih_transfer_overall_summary.csv")
    comp = pd.read_csv(extra_dir / "component_timing_summary.csv")
    end = pd.read_csv(extra_dir / "end_to_end_timing_summary.csv")
    paper = pd.read_csv(extra_dir / "paper_table8_reference.csv")
    manifest = pd.read_csv(extra_dir / "mitbih_feature_manifest.csv")

    def _first_value(df: pd.DataFrame, key_col: str, key: str, value_col: str) -> float:
        vals = df.loc[df[key_col].eq(key), value_col]
        return float(vals.iloc[0]) if len(vals) else float("nan")

    svm_accept = _first_value(mit_overall, "model", "singlelead_weak_svm", "acceptance_rate")
    mlp_accept = _first_value(mit_overall, "model", "singlelead_weak_lm_mlp_J5", "acceptance_rate")
    wqrs_mean = _first_value(comp, "component", "wqrs", "mean_ms")
    full_qrs_mean = _first_value(end, "component", "qrs_ms", "mean_ms")
    full_total_mean = _first_value(end, "component", "total_ms", "mean_ms")

    lines = [
        "# Paper Extra Experiments",
        "",
        "## Window-Length Experiment",
        "",
        "Skipped for this rerun. Clifford et al. describe the 5-10 s window-length test as being performed on single-lead data. This repository does not have the paper's single-lead manual labels, so generating a window curve would be a weak proxy rather than a faithful replication.",
        "",
        "## MIT-BIH Transfer Proxy",
        "",
        "MIT-BIH has arrhythmia annotations but no signal-quality labels. We therefore treat MIT-BIH windows as acceptable-proxy examples and report acceptance / proxy false-rejection rate, not paper-exact accuracy.",
        "",
        f"- MIT-BIH records processed: {int(manifest['record_id'].nunique())}",
        f"- MIT-BIH feature rows: {int(manifest['rows'].sum())}",
        "",
        "Challenge weak-label proxy performance:",
        "",
        _markdown_table(challenge),
        "",
        "MIT-BIH transfer summary:",
        "",
        _markdown_table(mit_overall),
        "",
        f"Directional comparison: the paper reports approximately 93% accuracy for its single-lead MIT-BIH transfer experiment. This rerun is weaker by design: it accepts {svm_accept:.1%} (SVM) and {mlp_accept:.1%} (MLP) of MIT-BIH lead-windows under a Set-a-trained proxy, so it should be read as a false-rejection/generalization signal rather than paper-exact accuracy.",
        "",
        "## Execution-Time Benchmark",
        "",
        "Paper Table 8 reference:",
        "",
        _markdown_table(paper),
        "",
        "This run, per-lead components:",
        "",
        _markdown_table(comp),
        "",
        "This run, full 12-lead end-to-end:",
        "",
        _markdown_table(end),
        "",
        f"Timing trend: wqrs is still the slowest per-lead detector component ({wqrs_mean:.1f} ms mean), and the full 12-lead path is dominated by QRS ({full_qrs_mean:.1f} ms of {full_total_mean:.1f} ms mean total). The qualitative ordering matches Table 8, while absolute times include Python, temporary WFDB record writing, and external executable launch overhead.",
        "",
        "## Limitations",
        "",
        "- MIT-BIH transfer is a weak-label rhythm/generalization proxy, not a replication of the paper's single-lead classifier with single-lead quality labels.",
        "- Timing includes Python overhead, WFDB temp-record writing, and C executable launch overhead; it is therefore expected to be slower than paper Table 8 Matlab component timings in some places and not hardware-comparable.",
        "- The paper's single-lead 5-10 s window experiment remains intentionally unimplemented in this repo until single-lead labels are available.",
    ]
    out = report_dir / "paper_extra_experiments.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in [
        "challenge_singlelead_proxy_metrics.csv",
        "mitbih_transfer_overall_summary.csv",
        "mitbih_transfer_per_record_summary.csv",
        "component_timing_summary.csv",
        "end_to_end_timing_summary.csv",
        "paper_table8_reference.csv",
    ]:
        src = extra_dir / name
        if src.exists():
            (report_dir / name).write_bytes(src.read_bytes())
    return out


def run(args: argparse.Namespace) -> Path:
    root = project_root()
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = root / artifacts_dir
    extra_dir = Path(args.out_dir)
    if not extra_dir.is_absolute():
        extra_dir = root / extra_dir
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = root / report_dir
    mitbih_dir = Path(args.mitbih_dir)
    if not mitbih_dir.is_absolute():
        mitbih_dir = root / mitbih_dir

    extra_dir.mkdir(parents=True, exist_ok=True)
    qrs_work_dir = extra_dir / "qrs_tmp"
    executables = resolve_paper_qrs_executables({}, artifacts_dir / "qrs")
    proxy = train_singlelead_proxy(artifacts_dir, extra_dir, int(args.seed))
    mit_features = build_mitbih_features(
        mitbih_dir=mitbih_dir,
        out_dir=extra_dir,
        executables=executables,
        qrs_work_dir=qrs_work_dir,
        max_records=args.max_mit_records,
        max_windows_per_record=args.max_windows_per_record,
    )
    predict_mitbih(mit_features, proxy, extra_dir)
    benchmark(
        artifacts_dir=artifacts_dir,
        out_dir=extra_dir,
        executables=executables,
        proxy=proxy,
        seed=int(args.seed),
        n_records=int(args.benchmark_records),
    )
    _json_dump(
        extra_dir / "extra_experiments_manifest.json",
        {
            "artifacts_dir": str(artifacts_dir),
            "mitbih_dir": str(mitbih_dir),
            "out_dir": str(extra_dir),
            "report_dir": str(report_dir),
            "window_length_experiment": "skipped_single_lead_labels_unavailable",
        },
    )
    return write_report(extra_dir, report_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper-aligned extra SQI experiments.")
    p.add_argument("--artifacts_dir", default="outputs/sqi_paper_aligned")
    p.add_argument("--out_dir", default="outputs/sqi_paper_aligned/extra_experiments")
    p.add_argument("--report_dir", default="outputs/reports/sqi_paper_aligned/extra_experiments")
    p.add_argument("--mitbih_dir", default="data/physionet/mit-bih")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_mit_records", type=int, default=None)
    p.add_argument("--max_windows_per_record", type=int, default=None)
    p.add_argument("--benchmark_records", type=int, default=100)
    return p.parse_args()


def main() -> None:
    report = run(parse_args())
    print(report)


if __name__ == "__main__":
    main()
