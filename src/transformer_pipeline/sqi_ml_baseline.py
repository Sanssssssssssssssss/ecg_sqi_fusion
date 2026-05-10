from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.sqi_pipeline.features.sqi import (
    sqi_basSQI,
    sqi_bSQI_li2008_global,
    sqi_fSQI,
    sqi_iSQI_li2008_global_per_lead,
    sqi_kSQI,
    sqi_pSQI,
    sqi_sSQI,
)
from src.sqi_pipeline.models.lm_mlp import LMConfig, LMMLP
from src.sqi_pipeline.models.svm_rbf import SVMConfig, SVMRBF
from src.sqi_pipeline.qrs.detectors import run_gqrs, run_xqrs
from src.utils.paths import project_root

logger = logging.getLogger(__name__)

FS = 125
WIN_N = 1250
LEAD = "I"
LEADS_SINGLE = [LEAD]
SQI_LIST = ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]
COMBOS = [
    ("Pairs", ["iSQI", "basSQI"]),
    ("Triplets", ["bSQI", "basSQI", "pSQI"]),
    ("Quadruplets", ["bSQI", "basSQI", "kSQI", "sSQI"]),
    ("Quintuplets", ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]),
    ("Sextuplets", ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI", "iSQI"]),
    ("All SQI", ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]),
]
WELCH_KW = dict(fmax=40.0, window="hann", nperseg=256, noverlap=128, detrend="constant")
FLATLINE_EPS = 1e-4
BEAT_MATCH_TOL_MS = 150
STD_EPS = 1e-8


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()
    transformer_dir = _path(params.get("transformer_artifact_dir"), root / "outputs" / "transformer")
    out_dir = _path(params.get("out_dir"), root / "outputs" / "transformer_sqi_ml")
    seed = int(params.get("seed", 0))
    label_mode = str(params.get("label_mode", "good_bad"))
    if label_mode not in {"good_bad", "good_vs_not_good"}:
        raise ValueError("label_mode must be 'good_bad' or 'good_vs_not_good'")

    paths = _paths(out_dir, seed)
    paths["root"].mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    split = build_split_from_transformer_labels(
        transformer_dir=transformer_dir,
        out_csv=paths["split"],
        seed=seed,
        label_mode=label_mode,
        force=force,
    )
    export_single_lead_resampled_cache(
        transformer_dir=transformer_dir,
        split=split,
        out_dir=paths["resampled"],
        force=force,
    )
    build_qrs_cache(split=split, resampled_dir=paths["resampled"], out_dir=paths["qrs"], force=force)
    features = build_single_lead_features(
        split=split,
        resampled_dir=paths["resampled"],
        qrs_dir=paths["qrs"],
        out_dir=paths["features"],
        force=force,
    )
    norm = normalize_single_lead_features(
        split_csv=paths["split"],
        in_parquet=paths["record7"],
        out_parquet=paths["record7_norm"],
        out_stats=paths["norm_stats"],
        force=force,
    )
    svm = train_svm_single_lead(
        features_parquet=paths["record7_norm"],
        split_csv=paths["split"],
        out_dir=paths["svm"],
        seed=seed,
        force=force,
    )
    lm_mlp = train_lm_mlp_single_lead(
        features_parquet=paths["record7_norm"],
        split_csv=paths["split"],
        out_dir=paths["lm_mlp"],
        seed=seed,
        force=force,
    )

    summary = build_comparison_summary(
        root=root,
        out_dir=out_dir,
        transformer_dir=transformer_dir,
        seed=seed,
        label_mode=label_mode,
        split=split,
        features=features,
        norm=norm,
        svm=svm,
        lm_mlp=lm_mlp,
        duration_sec=time.perf_counter() - start,
    )
    summary_path = paths["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = paths["summary_md"]
    md_path.write_text(render_summary_markdown(summary), encoding="utf-8")

    logger.info("wrote summary json: %s", _rel(summary_path, root))
    logger.info("wrote summary md: %s", _rel(md_path, root))
    return {
        "step": "transformer_sqi_ml_baseline",
        "skipped": False,
        "outputs": [_rel(summary_path, root), _rel(md_path, root)],
        "duration_sec": summary["duration_sec"],
    }


def build_split_from_transformer_labels(
    *,
    transformer_dir: Path,
    out_csv: Path,
    seed: int,
    label_mode: str,
    force: bool,
) -> pd.DataFrame:
    if out_csv.exists() and not force:
        logger.info("split exists -> skip: %s", out_csv)
        return pd.read_csv(out_csv)

    labels_path = transformer_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    labels = pd.read_csv(labels_path)
    required = {"idx", "seg_id", "ecg_id", "split", "y_class", "snr_db", "noise_kind"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"transformer labels missing columns: {sorted(missing)}")

    if label_mode == "good_bad":
        df = labels[labels["y_class"].isin(["good", "bad"])].copy()
        df["y"] = df["y_class"].map({"good": 1, "bad": -1}).astype(int)
        dropped_medium = int((labels["y_class"] == "medium").sum())
    else:
        df = labels.copy()
        df["y"] = np.where(df["y_class"].astype(str) == "good", 1, -1).astype(int)
        dropped_medium = 0

    df = df.sort_values("idx").reset_index(drop=True)
    df["record_id"] = [f"tx_{int(i):05d}" for i in df["idx"]]
    df["seed"] = int(seed)
    df["source_idx"] = df["idx"].astype(int)

    out = df[
        [
            "record_id",
            "y",
            "split",
            "seed",
            "source_idx",
            "y_class",
            "snr_db",
            "noise_kind",
            "seg_id",
            "ecg_id",
        ]
    ].copy()
    out.attrs["dropped_medium"] = dropped_medium
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    logger.info("split: %s rows=%d label_mode=%s dropped_medium=%d", out_csv, len(out), label_mode, dropped_medium)
    logger.info("split counts: %s", out["split"].value_counts().sort_index().to_dict())
    logger.info("label counts: %s", out["y"].value_counts().sort_index().to_dict())
    return out


def export_single_lead_resampled_cache(
    *,
    transformer_dir: Path,
    split: pd.DataFrame,
    out_dir: Path,
    force: bool,
) -> None:
    done_marker = out_dir / "_complete.json"
    if done_marker.exists() and not force:
        logger.info("single-lead cache exists -> skip: %s", out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    noisy = np.load(transformer_dir / "datasets" / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    if noisy.ndim != 2 or noisy.shape[1] != WIN_N:
        raise ValueError(f"expected noisy shape (N,{WIN_N}), got {noisy.shape}")

    n_ok = 0
    for row in split.itertuples(index=False):
        rid = str(row.record_id)
        idx = int(row.source_idx)
        out_npz = out_dir / f"{rid}.npz"
        if out_npz.exists() and not force:
            n_ok += 1
            continue
        x = noisy[idx].astype(np.float32, copy=False)
        np.savez(
            out_npz,
            record_id=rid,
            source_idx=np.array(idx, dtype=np.int32),
            fs=np.array(FS, dtype=np.int32),
            leads=np.array(LEADS_SINGLE, dtype=object),
            sig_125=x[:, None].astype(np.float32),
        )
        n_ok += 1
        if n_ok <= 5 or n_ok % 2000 == 0:
            logger.info("resampled adapter: %d/%d %s", n_ok, len(split), rid)

    done_marker.write_text(json.dumps({"rows": int(len(split)), "fs": FS, "lead": LEAD}, indent=2), encoding="utf-8")
    logger.info("single-lead cache done: rows=%d out=%s", len(split), out_dir)


def build_qrs_cache(*, split: pd.DataFrame, resampled_dir: Path, out_dir: Path, force: bool) -> None:
    done_marker = out_dir / "_complete.json"
    if done_marker.exists() and not force:
        logger.info("qrs cache exists -> skip: %s", out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    for row in split.itertuples(index=False):
        rid = str(row.record_id)
        out_npz = out_dir / f"{rid}.npz"
        if out_npz.exists() and not force:
            n_ok += 1
            continue
        try:
            z = np.load(resampled_dir / f"{rid}.npz", allow_pickle=True)
            x = z["sig_125"].astype(np.float32)[:, 0]
            r1 = run_xqrs(x, fs=FS)
            r2 = run_gqrs(x, fs=FS)
            np.savez(
                out_npz,
                record_id=rid,
                fs=np.array(FS, dtype=np.int32),
                leads=np.array(LEADS_SINGLE, dtype=object),
                detector1=np.array("xqrs", dtype=object),
                detector2=np.array("gqrs", dtype=object),
                beat_match_tol_ms=np.array(BEAT_MATCH_TOL_MS, dtype=np.int32),
                rpeaks_1=np.array([r1], dtype=object),
                rpeaks_2=np.array([r2], dtype=object),
            )
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            logger.warning("qrs failed for %s: %s: %s", rid, type(exc).__name__, exc)
        if (n_ok + n_fail) <= 5 or (n_ok + n_fail) % 2000 == 0:
            logger.info("qrs: %d/%d ok=%d fail=%d", n_ok + n_fail, len(split), n_ok, n_fail)

    done_marker.write_text(json.dumps({"rows": int(len(split)), "ok": n_ok, "failed": n_fail}, indent=2), encoding="utf-8")
    if n_fail:
        raise RuntimeError(f"qrs cache had {n_fail} failures")
    logger.info("qrs cache done: rows=%d", n_ok)


def build_single_lead_features(
    *,
    split: pd.DataFrame,
    resampled_dir: Path,
    qrs_dir: Path,
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    record7 = out_dir / "record7.parquet"
    lead7 = out_dir / "lead7.parquet"
    if record7.exists() and lead7.exists() and not force:
        logger.info("features exist -> skip: %s", out_dir)
        return _feature_summary(record7, lead7)

    out_dir.mkdir(parents=True, exist_ok=True)
    tol_samp = int(round(BEAT_MATCH_TOL_MS * FS / 1000.0))
    rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []

    for i, row in enumerate(split.itertuples(index=False), start=1):
        rid = str(row.record_id)
        y = int(row.y)
        z_sig = np.load(resampled_dir / f"{rid}.npz", allow_pickle=True)
        x = z_sig["sig_125"].astype(np.float32)[:, 0]
        z_qrs = np.load(qrs_dir / f"{rid}.npz", allow_pickle=True)
        r1 = np.asarray(z_qrs["rpeaks_1"].tolist()[0], dtype=int)
        r2 = np.asarray(z_qrs["rpeaks_2"].tolist()[0], dtype=int)

        i_sqi = float(sqi_iSQI_li2008_global_per_lead([r1], tol_samp)[0])
        b_sqi = float(sqi_bSQI_li2008_global(r1, r2, tol_samp))
        p_sqi = float(sqi_pSQI(x, fs=FS, welch_kwargs=WELCH_KW))
        s_sqi = float(sqi_sSQI(x))
        k_sqi = float(sqi_kSQI(x))
        f_sqi = float(sqi_fSQI(x, flatline_eps=FLATLINE_EPS))
        bas_sqi = float(sqi_basSQI(x, fs=FS, welch_kwargs=WELCH_KW))

        feat = {
            "record_id": rid,
            "y": y,
            f"{LEAD}__iSQI": i_sqi,
            f"{LEAD}__bSQI": b_sqi,
            f"{LEAD}__pSQI": p_sqi,
            f"{LEAD}__sSQI": s_sqi,
            f"{LEAD}__kSQI": k_sqi,
            f"{LEAD}__fSQI": f_sqi,
            f"{LEAD}__basSQI": bas_sqi,
        }
        rows.append(feat)
        lead_rows.append(
            {
                "record_id": rid,
                "y": y,
                "lead": LEAD,
                "iSQI": i_sqi,
                "bSQI": b_sqi,
                "pSQI": p_sqi,
                "sSQI": s_sqi,
                "kSQI": k_sqi,
                "fSQI": f_sqi,
                "basSQI": bas_sqi,
            }
        )
        if i <= 5 or i % 2000 == 0:
            logger.info("features: %d/%d %s", i, len(split), rid)

    df_record = pd.DataFrame(rows)
    df_lead = pd.DataFrame(lead_rows)
    df_record.to_parquet(record7, index=False)
    df_lead.to_parquet(lead7, index=False)
    logger.info("features done: record7=%s shape=%s", record7, df_record.shape)
    return _feature_summary(record7, lead7)


def normalize_single_lead_features(
    *,
    split_csv: Path,
    in_parquet: Path,
    out_parquet: Path,
    out_stats: Path,
    force: bool,
) -> dict[str, Any]:
    if out_parquet.exists() and out_stats.exists() and not force:
        logger.info("normalized features exist -> skip: %s", out_parquet)
        return _frame_summary(out_parquet)

    split = pd.read_csv(split_csv)
    feat = pd.read_parquet(in_parquet)
    split["record_id"] = split["record_id"].astype(str)
    feat["record_id"] = feat["record_id"].astype(str)
    df = feat.merge(split[["record_id", "split"]], on="record_id", how="left")
    if df["split"].isna().any():
        raise ValueError("missing split rows during normalization")

    target_cols = [c for c in df.columns if c.endswith("__kSQI") or c.endswith("__sSQI")]
    train = df.loc[df["split"] == "train", target_cols]
    med = train.median(axis=0, skipna=True)
    std = train.std(axis=0, ddof=1, skipna=True)
    std_safe = std.mask(std.abs() < STD_EPS, 1.0)

    df_norm = df.copy()
    df_norm[target_cols] = (df_norm[target_cols] - med) / std_safe
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_norm.drop(columns=["split"]).to_parquet(out_parquet, index=False)
    out_stats.write_text(
        json.dumps(
            {
                "method": "(x - median_train) / std_train",
                "std_eps": STD_EPS,
                "columns": target_cols,
                "median_train": {k: float(v) for k, v in med.items()},
                "std_train": {k: float(v) for k, v in std_safe.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("normalized features done: %s", out_parquet)
    return _frame_summary(out_parquet)


def train_svm_single_lead(
    *,
    features_parquet: Path,
    split_csv: Path,
    out_dir: Path,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    metrics_path = out_dir / f"svm_single_lead_metrics_seed{seed}.json"
    if metrics_path.exists() and not force:
        logger.info("svm metrics exist -> skip: %s", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_xy(features_parquet, split_csv)
    cfg = SVMConfig(seed=seed, cv_folds=5, use_standard_scaler=False)
    model = SVMRBF(cfg)
    grid = model.fit_gridsearch(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        threshold=0.5,
        select_metric="val_acc",
        log_each=False,
    )
    p_train = model.predict_proba(data["X_train"])
    p_val = model.predict_proba(data["X_val"])
    p_test = model.predict_proba(data["X_test"])

    all7 = {
        "feature_set": "all7",
        "n_features": int(data["X_train"].shape[1]),
        "best_params": grid["best_params"],
        "best_val_acc": float(grid["best_val_acc"]),
        "best_val_auc": float(grid["best_val_auc"]),
        "train_time_s": float(grid["train_time_s"]),
        "train": compute_metrics(data["y_train"], p_train, 0.5),
        "val": compute_metrics(data["y_val"], p_val, 0.5),
        "test": compute_metrics(data["y_test"], p_test, 0.5),
        "test_maxacc": find_maxacc_threshold(data["y_test"], p_test),
    }

    joblib.dump(
        {"model": model.best_estimator_, "feature_columns": data["feat_cols"], "metrics": all7},
        out_dir / f"svm_single_lead_all7_seed{seed}.joblib",
    )
    np.savez_compressed(out_dir / f"svm_single_lead_probs_seed{seed}.npz", y01_test=data["y_test"], p_test=p_test)

    tables = build_svm_single_lead_tables(data, cfg, grid["best_params"], out_dir, seed)
    result = {"all7": all7, "tables": tables}
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("svm done: test_acc=%.4f auc=%.4f", all7["test"]["acc"], all7["test"]["auc"])
    return result


def build_svm_single_lead_tables(
    data: dict[str, Any],
    cfg: SVMConfig,
    best_params: dict[str, Any],
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rows_single: list[dict[str, Any]] = []
    rows_combo: list[dict[str, Any]] = []
    c_star = float(best_params["svc__C"])
    gamma_star = float(best_params["svc__gamma"])

    for sqi in SQI_LIST:
        rows_single.append(_fit_svm_feature_subset(data, cfg, [f"{LEAD}__{sqi}"], c_star, gamma_star, sqi))
    for group, sqis in COMBOS:
        cols = [f"{LEAD}__{s}" for s in sqis]
        rows_combo.append(_fit_svm_feature_subset(data, cfg, cols, c_star, gamma_star, group))

    table_single = out_dir / f"table5_single_lead_single_sqi_seed{seed}.csv"
    table_combo = out_dir / f"table6_single_lead_combo_sqi_seed{seed}.csv"
    pd.DataFrame(rows_single).to_csv(table_single, index=False)
    pd.DataFrame(rows_combo).to_csv(table_combo, index=False)
    return {
        "table5": str(table_single),
        "table6": str(table_combo),
        "best_single": _best_row(rows_single),
        "best_combo": _best_row(rows_combo),
    }


def _fit_svm_feature_subset(
    data: dict[str, Any],
    cfg: SVMConfig,
    cols: list[str],
    c_star: float,
    gamma_star: float,
    name: str,
) -> dict[str, Any]:
    idx = [data["feat_cols"].index(c) for c in cols]
    model = SVMRBF(cfg)
    model.fit_fixed(data["X_train"][:, idx], data["y_train"], C=c_star, gamma=gamma_star)
    p_test = model.predict_proba(data["X_test"][:, idx])
    met = compute_metrics(data["y_test"], p_test, 0.5)
    maxacc = find_maxacc_threshold(data["y_test"], p_test)
    return {
        "name": name,
        "features": ",".join(cols),
        "n_features": len(cols),
        "Ac_test": met["acc"],
        "Se_test": met["se"],
        "Sp_test": met["sp"],
        "AUC_test": met["auc"],
        "maxAcc_test": maxacc["acc"],
        "maxAcc_thr_test": maxacc["threshold"],
    }


def train_lm_mlp_single_lead(
    *,
    features_parquet: Path,
    split_csv: Path,
    out_dir: Path,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    metrics_path = out_dir / f"lm_mlp_single_lead_metrics_seed{seed}.json"
    if metrics_path.exists() and not force:
        logger.info("lm_mlp metrics exist -> skip: %s", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_xy(features_parquet, split_csv)
    device = torch.device("cpu")
    dtype = torch.float64
    lm_cfg = LMConfig()
    threshold = 0.7
    j_range = tuple(range(2, 18))

    Xtr = torch.tensor(data["X_train"], device=device, dtype=dtype)
    ytr = torch.tensor(data["y_train"], device=device, dtype=dtype)
    Xva = torch.tensor(data["X_val"], device=device, dtype=dtype)
    yva = torch.tensor(data["y_val"], device=device, dtype=dtype)
    Xte = torch.tensor(data["X_test"], device=device, dtype=dtype)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    t0 = time.perf_counter()
    for j in j_range:
        model = LMMLP(J=j, D=data["X_train"].shape[1], device=device, dtype=dtype, seed=seed)
        train_summary = model.fit_lm(
            X_train=Xtr,
            y_train=ytr,
            cfg=lm_cfg,
            X_val=Xva,
            y_val=yva,
            model_select_metric="val_acc",
            patience=15,
            threshold=threshold,
        )
        p_val = model.predict_proba(Xva)
        val_metrics = compute_metrics(data["y_val"], p_val, threshold)
        row = {
            "J": int(j),
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
            "epochs_used": int(train_summary["epochs_used"]),
            "final_error": float(train_summary["final_error"]),
            "final_grad": float(train_summary["final_grad"]),
            "stop_reason": str(train_summary["stop_reason"]),
        }
        rows.append(row)
        if best is None or (row["val_acc"], row["val_auc"], -row["J"]) > (best["val_acc"], best["val_auc"], -best["J"]):
            best = row
        logger.info("lm_mlp J=%02d val_acc=%.4f val_auc=%.4f", j, row["val_acc"], row["val_auc"])

    assert best is not None
    best_j = int(best["J"])
    final = LMMLP(J=best_j, D=data["X_train"].shape[1], device=device, dtype=dtype, seed=seed)
    final_summary = final.fit_lm(
        X_train=Xtr,
        y_train=ytr,
        cfg=lm_cfg,
        X_val=Xva,
        y_val=yva,
        model_select_metric="val_acc",
        patience=15,
        threshold=threshold,
    )
    p_train = final.predict_proba(Xtr)
    p_val = final.predict_proba(Xva)
    p_test = final.predict_proba(Xte)
    elapsed = time.perf_counter() - t0

    pd.DataFrame(rows).to_csv(out_dir / f"search_J_results_seed{seed}.csv", index=False)
    with (out_dir / f"model_7-{best_j}-1_seed{seed}.pkl").open("wb") as f:
        pickle.dump(
            {
                "seed": seed,
                "J": best_j,
                "D": data["X_train"].shape[1],
                "feature_columns": data["feat_cols"],
                "threshold": threshold,
                "model": final.to_pickle_dict(),
            },
            f,
        )
    np.savez_compressed(out_dir / f"lm_mlp_single_lead_probs_seed{seed}.npz", y01_test=data["y_test"], p_test=p_test)

    result = {
        "feature_set": "all7",
        "n_features": int(data["X_train"].shape[1]),
        "device": "cpu",
        "dtype": str(dtype),
        "J": best_j,
        "threshold_fixed": threshold,
        "train_time_s": float(elapsed),
        "search": rows,
        "train_stop": final_summary,
        "train": compute_metrics(data["y_train"], p_train, threshold),
        "val": compute_metrics(data["y_val"], p_val, threshold),
        "test": compute_metrics(data["y_test"], p_test, threshold),
        "test_maxacc": find_maxacc_threshold(data["y_test"], p_test),
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("lm_mlp done: J=%d test_acc=%.4f auc=%.4f", best_j, result["test"]["acc"], result["test"]["auc"])
    return result


def load_xy(features_parquet: Path, split_csv: Path) -> dict[str, Any]:
    feat = pd.read_parquet(features_parquet)
    split = pd.read_csv(split_csv)
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    df = feat.merge(split[["record_id", "split"]], on="record_id", how="inner")
    drop_cols = {"record_id", "y", "split"}
    feat_cols = sorted(c for c in df.columns if c not in drop_cols)
    x = df[feat_cols].to_numpy(dtype=np.float64)
    y = (df["y"].astype(int).to_numpy() == 1).astype(np.int32)
    split_arr = df["split"].astype(str).to_numpy()
    out = {
        "feat_cols": feat_cols,
        "X_train": x[split_arr == "train"],
        "y_train": y[split_arr == "train"],
        "X_val": x[split_arr == "val"],
        "y_val": y[split_arr == "val"],
        "X_test": x[split_arr == "test"],
        "y_test": y[split_arr == "test"],
    }
    return out


def build_comparison_summary(
    *,
    root: Path,
    out_dir: Path,
    transformer_dir: Path,
    seed: int,
    label_mode: str,
    split: pd.DataFrame,
    features: dict[str, Any],
    norm: dict[str, Any],
    svm: dict[str, Any],
    lm_mlp: dict[str, Any],
    duration_sec: float,
) -> dict[str, Any]:
    main_lm = _read_json_if_exists(root / "outputs" / "sqi" / "models" / "lm_mlp" / f"lm_mlp_test_metrics_seed{seed}.json")
    main_svm_table6 = root / "outputs" / "sqi" / "models" / "svm" / f"table6_12lead_combo_sqi_seed{seed}.csv"
    main_svm_all = None
    if main_svm_table6.exists():
        df_svm = pd.read_csv(main_svm_table6)
        m = df_svm[df_svm["Group"].astype(str) == "All SQI"]
        if len(m):
            row = m.iloc[0].to_dict()
            main_svm_all = {k: _json_float(v) for k, v in row.items()}

    return {
        "seed": seed,
        "artifact_dir": _rel(out_dir, root),
        "source_transformer_artifact_dir": _rel(transformer_dir, root),
        "label_mode": label_mode,
        "duration_sec": float(duration_sec),
        "flow_contract": {
            "same_as_sqi_main": [
                "binary y convention: +1 good/acceptable, -1 bad/unacceptable",
                "split column is respected; no random re-split inside models",
                "fs=125 and 10 second windows at model/SQI feature stage",
                "QRS detectors: WFDB XQRS and GQRS",
                "beat_match_tol_ms=150",
                "SQI formulas: bSQI, iSQI, pSQI, sSQI, kSQI, fSQI, basSQI",
                "Welch parameters for pSQI/basSQI",
                "normalization only for sSQI/kSQI using train median and train std",
                "LM-MLP Levenberg-Marquardt training on CPU",
                "RBF SVM validation-set grid selection",
            ],
            "intentional_differences": [
                "source data is transformer synthetic PTB-XL Lead I instead of Challenge 2011 Set A",
                "lead_count=1 instead of 12",
                "transformer segments are already 125 Hz, so the sidecar writes a resampled-cache adapter rather than decimating 500 Hz ECG",
                "label_mode=good_bad drops transformer medium samples to match the binary ML task",
                "single-lead iSQI has no second lead and is therefore the existing Li2008 implementation's constant 0.0 output",
            ],
        },
        "data": {
            "split_shape": list(split.shape),
            "split_counts": _counts(split["split"]),
            "label_counts": _counts(split["y"]),
            "y_class_counts": _counts(split["y_class"]),
        },
        "features": {
            "record7": features["record"],
            "lead7": features["lead"],
            "record7_norm": norm,
        },
        "metrics": {
            "sidecar_lm_mlp": lm_mlp,
            "sidecar_svm": svm,
            "main_sqi_lm_mlp": main_lm,
            "main_sqi_svm_all_sqi": main_svm_all,
        },
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lm = summary["metrics"]["sidecar_lm_mlp"]
    svm = summary["metrics"]["sidecar_svm"]["all7"]
    main_lm = summary["metrics"].get("main_sqi_lm_mlp") or {}
    main_svm = summary["metrics"].get("main_sqi_svm_all_sqi") or {}
    return "\n".join(
        [
            "# Transformer Dataset SQI-ML Sidecar",
            "",
            f"Artifact dir: `{summary['artifact_dir']}`",
            f"Label mode: `{summary['label_mode']}`",
            "",
            "## Flow Check",
            "",
            "This sidecar keeps the traditional SQI/ML preprocessing and model logic, but swaps the input to transformer synthetic PTB-XL Lead I data.",
            "",
            "Intentional differences:",
            "",
            *[f"- {item}" for item in summary["flow_contract"]["intentional_differences"]],
            "",
            "## Data",
            "",
            f"- split shape: `{summary['data']['split_shape']}`",
            f"- split counts: `{summary['data']['split_counts']}`",
            f"- label counts: `{summary['data']['label_counts']}`",
            "",
            "## Results",
            "",
            "| Model | Dataset | Features | Test Acc | Test AUC |",
            "| --- | --- | --- | ---: | ---: |",
            f"| LM-MLP | transformer Lead I sidecar | 7 SQI | {lm['test']['acc']:.4f} | {lm['test']['auc']:.4f} |",
            f"| SVM-RBF | transformer Lead I sidecar | 7 SQI | {svm['test']['acc']:.4f} | {svm['test']['auc']:.4f} |",
            f"| LM-MLP | original SQI main | 84 SQI | {_main_lm_acc(main_lm):.4f} | {_main_lm_auc(main_lm):.4f} |",
            f"| SVM-RBF | original SQI main | 84 SQI | {float(main_svm.get('Ac_test', float('nan'))):.4f} | {float(main_svm.get('AUC_test', float('nan'))):.4f} |",
            "",
            "## Key Files",
            "",
            "- `splits/transformer_good_bad_seed0.csv`",
            "- `features/record7.parquet`",
            "- `features/record7_norm.parquet`",
            "- `models/lm_mlp/lm_mlp_single_lead_metrics_seed0.json`",
            "- `models/svm/svm_single_lead_metrics_seed0.json`",
            "- `comparison_summary.json`",
        ]
    )


def compute_metrics(y01: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y01 = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    pred = (p > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y01, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    se = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    try:
        auc = float(roc_auc_score(y01, p))
    except Exception:
        auc = float("nan")
    return {
        "acc": float(acc),
        "se": float(se),
        "sp": float(sp),
        "auc": auc,
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_maxacc_threshold(y01: np.ndarray, p: np.ndarray, n_grid: int = 2001) -> dict[str, Any]:
    y = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    best: dict[str, Any] = {"threshold": 0.5, "acc": -1.0, "se": 0.0, "sp": 0.0}
    for t in np.linspace(0.0, 1.0, int(n_grid)):
        met = compute_metrics(y, p, float(t))
        if met["acc"] > best["acc"]:
            best = {
                "threshold": float(t),
                "acc": met["acc"],
                "se": met["se"],
                "sp": met["sp"],
                **met["confusion_matrix"],
            }
    return best


def _paths(out_dir: Path, seed: int) -> dict[str, Path]:
    return {
        "root": out_dir,
        "split": out_dir / "splits" / f"transformer_good_bad_seed{seed}.csv",
        "resampled": out_dir / "resampled_125",
        "qrs": out_dir / "qrs",
        "features": out_dir / "features",
        "record7": out_dir / "features" / "record7.parquet",
        "lead7": out_dir / "features" / "lead7.parquet",
        "record7_norm": out_dir / "features" / "record7_norm.parquet",
        "norm_stats": out_dir / "features" / f"norm_stats_seed{seed}.json",
        "models": out_dir / "models",
        "svm": out_dir / "models" / "svm",
        "lm_mlp": out_dir / "models" / "lm_mlp",
        "summary_json": out_dir / "comparison_summary.json",
        "summary_md": out_dir / "comparison_summary.md",
    }


def _frame_summary(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    return {"path": str(path), "shape": list(df.shape), "columns": list(df.columns)}


def _feature_summary(record7: Path, lead7: Path) -> dict[str, Any]:
    return {"record": _frame_summary(record7), "lead": _frame_summary(lead7)}


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().to_dict().items()}


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row.get("Ac_test", float("-inf"))))


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _main_lm_acc(data: dict[str, Any]) -> float:
    return float((data.get("test_metrics_fixed") or {}).get("acc", float("nan")))


def _main_lm_auc(data: dict[str, Any]) -> float:
    return float((data.get("test_metrics_fixed") or {}).get("auc", float("nan")))


def _json_float(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    return path if path.is_absolute() else project_root() / path


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = parse_args()
    run(vars(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional SQI+ML baselines on transformer Lead I data.")
    parser.add_argument("--transformer_artifact_dir", default="outputs/transformer")
    parser.add_argument("--out_dir", default="outputs/transformer_sqi_ml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label_mode", choices=("good_bad", "good_vs_not_good"), default="good_bad")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
