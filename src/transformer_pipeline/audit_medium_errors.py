from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.utils.paths import project_root
    from src.transformer_pipeline import train as train_mod
    from src.transformer_pipeline.noise.make_rr_noise_level import detect_r_peaks
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root
    from src.transformer_pipeline import train as train_mod
    from src.transformer_pipeline.noise.make_rr_noise_level import detect_r_peaks


CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CLASS_ORDER = ["good", "medium", "bad"]
FS = 125
EPS = 1e-12


def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    root = project_root()
    artifact_dir = _path(params.get("artifact_dir"), root / "outputs/transformer", root)
    model_dir = _path(
        params.get("model_dir"),
        root / "outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26",
        root,
    )
    out_dir = _path(params.get("out_dir"), artifact_dir / "medium_error_audit" / model_dir.name, root)
    force = bool(params.get("force", False))
    verbose = bool(params.get("verbose", False))
    seed = int(params.get("seed", 0))
    batch_size = int(params.get("batch_size", 512))
    splits = _parse_splits(params.get("splits"))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "medium_error_audit.md"
    out_json = out_dir / "medium_error_summary.json"
    if out_md.exists() and out_json.exists() and not force:
        print(f"audit outputs exist -> skip: {_display(out_md, root)}")
        return {"step": "medium_error_audit", "skipped": True, "outputs": [str(out_md), str(out_json)]}

    labels, x_noisy, x_clean, valid_rr = _load_transformer_dataset(artifact_dir)
    if splits:
        mask = labels["split"].astype(str).isin(splits).to_numpy()
        labels = labels.loc[mask].reset_index(drop=True)
        x_noisy = x_noisy[mask]
        x_clean = x_clean[mask]
        valid_rr = valid_rr[mask]
    if verbose:
        print(f"loaded dataset: rows={len(labels)} X={x_noisy.shape}")

    y_true = labels["y_class"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    probs = _predict_transformer(
        artifact_dir=artifact_dir,
        model_dir=model_dir,
        labels=labels,
        x_noisy=x_noisy,
        seed=seed,
        batch_size=batch_size,
        verbose=verbose,
    )
    pred = np.argmax(probs, axis=1).astype(np.int64)

    features = _sample_audit_features(x_clean=x_clean, x_noisy=x_noisy, verbose=verbose)
    measured_snr = _measured_snr(x_clean, x_noisy)

    pred_df = labels.copy()
    pred_df["y_int"] = y_true
    pred_df["pred_int"] = pred
    pred_df["pred_class"] = [INT_TO_CLASS[int(v)] for v in pred]
    pred_df["p_good"] = probs[:, 0]
    pred_df["p_medium"] = probs[:, 1]
    pred_df["p_bad"] = probs[:, 2]
    top2 = np.sort(probs, axis=1)[:, -2:]
    pred_df["top2_margin"] = top2[:, 1] - top2[:, 0]
    pred_df["p_good_minus_medium"] = pred_df["p_good"] - pred_df["p_medium"]
    pred_df["p_bad_minus_medium"] = pred_df["p_bad"] - pred_df["p_medium"]
    pred_df["snr_measured"] = measured_snr
    pred_df["snr_error"] = measured_snr - pred_df["snr_db"].to_numpy(dtype=np.float64)
    pred_df["valid_rr"] = valid_rr.astype(np.uint8)
    for name, values in features.items():
        pred_df[name] = values

    for split in ["val", "test"]:
        split_df = pred_df[pred_df["split"].astype(str) == split].copy()
        split_df.to_csv(out_dir / f"predictions_{split}.csv", index=False)
        medium_errors = split_df[(split_df["y_class"].astype(str) == "medium") & (split_df["pred_class"] != "medium")]
        medium_errors.to_csv(out_dir / f"medium_errors_{split}.csv", index=False)

    oracle_summary = _oracle_summary(pred_df)
    baseline_summary = _noisy_feature_baselines(pred_df, x_noisy=x_noisy, seed=seed, verbose=verbose)
    transformer_summary = _transformer_summary(pred_df)
    medium_summary = {
        split: _summarize_medium_split(pred_df[pred_df["split"].astype(str) == split].copy())
        for split in ["val", "test"]
    }

    summary = {
        "artifact_dir": _display(artifact_dir, root),
        "model_dir": _display(model_dir, root),
        "checkpoint": _display(model_dir / "ckpt_best_val.pt", root),
        "rows": int(len(pred_df)),
        "transformer": transformer_summary,
        "measured_snr_oracle": oracle_summary,
        "noisy_feature_baselines": baseline_summary,
        "medium_error_audit": medium_summary,
        "outputs": {
            "report": _display(out_md, root),
            "summary_json": _display(out_json, root),
            "predictions_val": _display(out_dir / "predictions_val.csv", root),
            "predictions_test": _display(out_dir / "predictions_test.csv", root),
            "medium_errors_val": _display(out_dir / "medium_errors_val.csv", root),
            "medium_errors_test": _display(out_dir / "medium_errors_test.csv", root),
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, ensure_ascii=False, indent=2)
    out_md.write_text(_render_markdown(summary), encoding="utf-8")

    print(f"saved: {_display(out_md, root)}")
    print(f"saved: {_display(out_json, root)}")
    return {"step": "medium_error_audit", "skipped": False, "outputs": [str(out_md), str(out_json)]}


def _load_transformer_dataset(artifact_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    labels_path = artifact_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    noisy_path = artifact_dir / "datasets" / "synth_10s_125hz_noisy.npz"
    clean_path = artifact_dir / "datasets" / "synth_10s_125hz_clean.npz"
    level_path = artifact_dir / "datasets" / "synth_10s_125hz_noise_level.npz"

    labels = pd.read_csv(labels_path).sort_values("idx").reset_index(drop=True)
    x_noisy = np.load(noisy_path)["X_noisy"].astype(np.float32)
    x_clean = np.load(clean_path)["X_clean"].astype(np.float32)
    z_level = np.load(level_path)
    valid_rr = z_level["valid_rr"].astype(np.uint8)

    if len(labels) != len(x_noisy) or x_noisy.shape != x_clean.shape or len(valid_rr) != len(labels):
        raise ValueError(
            "Transformer artifact row mismatch: "
            f"labels={len(labels)} noisy={x_noisy.shape} clean={x_clean.shape} valid_rr={len(valid_rr)}"
        )
    if not np.array_equal(labels["idx"].to_numpy(dtype=np.int64), np.arange(len(labels), dtype=np.int64)):
        raise ValueError("labels must be sorted by idx and aligned with array rows")
    return labels, x_noisy, x_clean, valid_rr


def _predict_transformer(
    artifact_dir: Path,
    model_dir: Path,
    labels: pd.DataFrame,
    x_noisy: np.ndarray,
    seed: int,
    batch_size: int,
    verbose: bool,
) -> np.ndarray:
    hp = _load_model_hyperparams(model_dir)
    params: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "model_dir": str(model_dir),
        "seed": seed,
        "batch_size": batch_size,
        "verbose": False,
    }
    for key in [
        "epochs",
        "dropout",
        "cls_pool",
        "use_positional_embedding",
        "input_mode",
        "ordinal_head",
        "snr_head",
        "e_cls",
        "e_denoise",
        "e_level",
        "e_uncert",
        "bad_den_w_max",
        "bad_den_w_warmup_epochs",
        "lambda_cls",
        "lambda_den",
        "lambda_lvl",
        "lambda_ord",
        "lambda_snr",
        "label_smoothing",
        "class_weight_good",
        "class_weight_medium",
        "class_weight_bad",
        "select_best_by",
        "uncertainty_mode",
    ]:
        if key in hp:
            params[key] = hp[key]

    train_mod.configure_from_params(params)
    train_mod.seed_all(seed)
    device = torch.device("cpu")
    model, _uw = train_mod.build_model(device)

    ckpt_path = model_dir / "ckpt_best_val.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    probs = np.zeros((len(labels), 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            end = min(len(labels), start + batch_size)
            if verbose and start % (batch_size * 10) == 0:
                print(f"predict transformer: {start}/{len(labels)}")
            x_np = np.stack(
                [train_mod.make_input_channels(row, train_mod.INPUT_MODE) for row in x_noisy[start:end]],
                axis=0,
            )
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            logits = model(x)[2]
            probs[start:end] = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return probs


def _load_model_hyperparams(model_dir: Path) -> dict[str, Any]:
    probe_path = model_dir / "probe_summary.json"
    if probe_path.exists():
        with probe_path.open("r", encoding="utf-8") as f:
            probe = json.load(f)
        return dict(probe.get("hyperparams", {}))
    ckpt_path = model_dir / "ckpt_best_val.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hp = ckpt.get("hyperparams", {})
        return {
            "epochs": hp.get("EPOCHS"),
            "dropout": hp.get("MODEL_DROPOUT"),
            "cls_pool": hp.get("CLS_POOL"),
            "use_positional_embedding": hp.get("USE_POSITIONAL_EMBEDDING"),
            "input_mode": hp.get("INPUT_MODE"),
            "ordinal_head": hp.get("USE_ORDINAL_HEAD"),
            "snr_head": hp.get("USE_SNR_HEAD"),
            "e_cls": hp.get("E_CLS"),
            "e_denoise": hp.get("E_DENOISE"),
            "e_level": hp.get("E_LEVEL"),
            "e_uncert": hp.get("E_UNCERT"),
            "lambda_cls": hp.get("LAMBDA_CLS"),
            "lambda_den": hp.get("LAMBDA_DEN"),
            "lambda_lvl": hp.get("LAMBDA_LVL"),
            "lambda_ord": hp.get("LAMBDA_ORD"),
            "lambda_snr": hp.get("LAMBDA_SNR"),
        }
    return {}


def _sample_audit_features(x_clean: np.ndarray, x_noisy: np.ndarray, verbose: bool) -> dict[str, np.ndarray]:
    clean_rms = np.sqrt(np.mean(np.square(x_clean.astype(np.float64)), axis=1))
    noisy_rms = np.sqrt(np.mean(np.square(x_noisy.astype(np.float64)), axis=1))
    noise = x_noisy.astype(np.float64) - x_clean.astype(np.float64)
    noise_rms = np.sqrt(np.mean(np.square(noise), axis=1))
    clean_ptp = np.ptp(x_clean, axis=1).astype(np.float64)
    clean_abs_p95 = np.percentile(np.abs(x_clean), 95, axis=1)

    clean_peak_count, clean_qrs_amp, clean_rr_cv = _qrs_summary(x_clean, verbose=verbose, label="clean")
    noisy_peak_count, noisy_qrs_amp, noisy_rr_cv = _qrs_summary(x_noisy, verbose=verbose, label="noisy")
    return {
        "clean_rms": clean_rms,
        "noisy_rms": noisy_rms,
        "noise_rms": noise_rms,
        "clean_ptp": clean_ptp,
        "clean_abs_p95": clean_abs_p95,
        "clean_peak_count": clean_peak_count,
        "clean_qrs_amp_median": clean_qrs_amp,
        "clean_rr_cv": clean_rr_cv,
        "noisy_peak_count": noisy_peak_count,
        "noisy_qrs_amp_median": noisy_qrs_amp,
        "noisy_rr_cv": noisy_rr_cv,
        "noisy_rpeak_failed": ((noisy_peak_count < 4) | ~np.isfinite(noisy_rr_cv)).astype(np.uint8),
    }


def _qrs_summary(x: np.ndarray, verbose: bool, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    count = np.zeros(n, dtype=np.float64)
    amp = np.full(n, np.nan, dtype=np.float64)
    rr_cv = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if verbose and (i + 1) % 5000 == 0:
            print(f"qrs {label}: {i + 1}/{n}")
        try:
            peaks = detect_r_peaks(x[i], FS)
        except Exception:
            peaks = np.array([], dtype=int)
        count[i] = float(len(peaks))
        if len(peaks) > 0:
            amp[i] = float(np.median(np.abs(x[i, peaks])))
        if len(peaks) > 2:
            rr = np.diff(peaks).astype(np.float64)
            mean_rr = float(np.mean(rr))
            if mean_rr > 0:
                rr_cv[i] = float(np.std(rr) / mean_rr)
    return count, amp, rr_cv


def _measured_snr(x_clean: np.ndarray, x_noisy: np.ndarray) -> np.ndarray:
    signal_power = np.mean(np.square(x_clean.astype(np.float64)), axis=1)
    noise_power = np.mean(np.square(x_noisy.astype(np.float64) - x_clean.astype(np.float64)), axis=1)
    return 10.0 * np.log10((signal_power + EPS) / (noise_power + EPS))


def _oracle_summary(df: pd.DataFrame) -> dict[str, Any]:
    pred = np.where(df["snr_measured"].to_numpy(dtype=np.float64) >= 16.0, 0, np.where(df["snr_measured"] >= 2.0, 1, 2))
    y = df["y_int"].to_numpy(dtype=np.int64)
    out: dict[str, Any] = {
        "rule": "good if measured_snr>=16; medium if 2<=measured_snr<16; bad if measured_snr<2",
        "snr_error": _describe(df["snr_error"]),
    }
    for split in ["all", "train", "val", "test"]:
        mask = np.ones(len(df), dtype=bool) if split == "all" else df["split"].astype(str).to_numpy() == split
        out[split] = _classification_summary(y[mask], pred[mask])
    return out


def _noisy_feature_baselines(df: pd.DataFrame, x_noisy: np.ndarray, seed: int, verbose: bool) -> dict[str, Any]:
    y = df["y_int"].to_numpy(dtype=np.int64)
    split = df["split"].astype(str).to_numpy()
    feature_df = _build_noisy_feature_frame(df, x_noisy)
    x = feature_df.to_numpy(dtype=np.float64)
    train_mask = split == "train"
    if int(train_mask.sum()) == 0:
        return {
            "skipped": "No train rows in this audit subset.",
            "feature_columns": list(feature_df.columns),
        }

    models = {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.06,
                l2_regularization=0.02,
                random_state=seed,
            ),
        ),
    }

    out: dict[str, Any] = {"feature_columns": list(feature_df.columns)}
    for name, model in models.items():
        if verbose:
            print(f"fit noisy-only baseline: {name}")
        model.fit(x[train_mask], y[train_mask])
        item: dict[str, Any] = {}
        for split_name in ["train", "val", "test"]:
            mask = split == split_name
            pred = model.predict(x[mask]).astype(np.int64)
            item[split_name] = _classification_summary(y[mask], pred)
        out[name] = item
    return out


def _build_noisy_feature_frame(df: pd.DataFrame, x_noisy: np.ndarray) -> pd.DataFrame:
    abs_x = np.abs(x_noisy)
    dx = np.diff(x_noisy, axis=1)
    f, pxx = welch(x_noisy, fs=FS, nperseg=256, noverlap=128, axis=1)
    total_power = np.trapezoid(pxx, f, axis=1) + EPS

    def band(lo: float, hi: float) -> np.ndarray:
        m = (f >= lo) & (f < hi)
        if not np.any(m):
            return np.zeros(x_noisy.shape[0], dtype=np.float64)
        return np.trapezoid(pxx[:, m], f[m], axis=1)

    low_0_1 = band(0.0, 1.0)
    base_0_5 = band(0.0, 0.5)
    qrs_5_15 = band(5.0, 15.0)
    high_15_40 = band(15.0, 40.0)
    mid_1_5 = band(1.0, 5.0)

    out = pd.DataFrame({
        "mean": np.mean(x_noisy, axis=1),
        "std": np.std(x_noisy, axis=1),
        "rms": np.sqrt(np.mean(np.square(x_noisy.astype(np.float64)), axis=1)),
        "abs_mean": np.mean(abs_x, axis=1),
        "abs_p95": np.percentile(abs_x, 95, axis=1),
        "ptp": np.ptp(x_noisy, axis=1),
        "skew": skew(x_noisy, axis=1, bias=False),
        "kurtosis": kurtosis(x_noisy, axis=1, fisher=True, bias=False),
        "diff_std": np.std(dx, axis=1),
        "diff_rms": np.sqrt(np.mean(np.square(dx.astype(np.float64)), axis=1)),
        "zero_cross_rate": np.mean(np.diff(np.signbit(x_noisy), axis=1), axis=1),
        "band_0_1": low_0_1 / total_power,
        "band_0_5": base_0_5 / total_power,
        "band_1_5": mid_1_5 / total_power,
        "band_5_15": qrs_5_15 / total_power,
        "band_15_40": high_15_40 / total_power,
        "psqi_5_15_over_5_40": qrs_5_15 / (band(5.0, 40.0) + EPS),
        "bas_sqi_0_1_over_0_40": low_0_1 / (band(0.0, 40.0) + EPS),
        "noisy_peak_count": df["noisy_peak_count"].to_numpy(dtype=np.float64),
        "noisy_qrs_amp_median": df["noisy_qrs_amp_median"].to_numpy(dtype=np.float64),
        "noisy_rr_cv": df["noisy_rr_cv"].to_numpy(dtype=np.float64),
        "valid_rr": df["valid_rr"].to_numpy(dtype=np.float64),
    })
    return out

def _transformer_summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    y = df["y_int"].to_numpy(dtype=np.int64)
    pred = df["pred_int"].to_numpy(dtype=np.int64)
    split = df["split"].astype(str).to_numpy()
    for split_name in ["val", "test"]:
        mask = split == split_name
        out[split_name] = _classification_summary(y[mask], pred[mask])
    return out


def _summarize_medium_split(split_df: pd.DataFrame) -> dict[str, Any]:
    medium = split_df[split_df["y_class"].astype(str) == "medium"].copy()
    groups = {
        "all_medium": medium,
        "correct_medium": medium[medium["pred_class"] == "medium"],
        "medium_to_good": medium[medium["pred_class"] == "good"],
        "medium_to_bad": medium[medium["pred_class"] == "bad"],
    }
    out = {
        "medium_n": int(len(medium)),
        "counts": {name: int(len(g)) for name, g in groups.items()},
        "groups": {name: _summarize_group(g, medium) for name, g in groups.items()},
    }
    return out


def _summarize_group(group: pd.DataFrame, medium_ref: pd.DataFrame) -> dict[str, Any]:
    n_ref = max(1, int(len(medium_ref)))
    out: dict[str, Any] = {
        "n": int(len(group)),
        "fraction_of_medium": float(len(group) / n_ref),
    }
    if len(group) == 0:
        return out

    for col in [
        "snr_db",
        "snr_measured",
        "snr_error",
        "clean_rms",
        "clean_abs_p95",
        "clean_ptp",
        "clean_peak_count",
        "clean_qrs_amp_median",
        "clean_rr_cv",
        "noisy_peak_count",
        "noisy_qrs_amp_median",
        "noisy_rr_cv",
        "p_good",
        "p_medium",
        "p_bad",
        "top2_margin",
        "p_good_minus_medium",
        "p_bad_minus_medium",
    ]:
        out[col] = _describe(group[col])

    out["snr_db_bins"] = _bin_counts(group["snr_db"])
    out["snr_measured_bins"] = _bin_counts(group["snr_measured"])
    out["noise_kind"] = _value_counts(group["noise_kind"])
    out["valid_rr_zero_frac"] = float((group["valid_rr"].to_numpy(dtype=np.float64) == 0).mean())
    out["noisy_rpeak_failed_frac"] = float((group["noisy_rpeak_failed"].to_numpy(dtype=np.float64) > 0).mean())
    out["p_medium_below_0_40_frac"] = float((group["p_medium"].to_numpy(dtype=np.float64) < 0.40).mean())
    out["p_good_medium_close_0_10_frac"] = float((np.abs(group["p_good_minus_medium"].to_numpy(dtype=np.float64)) <= 0.10).mean())
    out["p_bad_medium_close_0_10_frac"] = float((np.abs(group["p_bad_minus_medium"].to_numpy(dtype=np.float64)) <= 0.10).mean())
    out["clean_rms_outlier_frac_vs_medium"] = _outlier_frac(group["clean_rms"], medium_ref["clean_rms"])
    out["clean_qrs_amp_outlier_frac_vs_medium"] = _outlier_frac(
        group["clean_qrs_amp_median"], medium_ref["clean_qrs_amp_median"]
    )
    return out


def _classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "n": 0,
            "acc": 0.0,
            "confusion_matrix_3x3": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "medium": {"n": 0, "recall": 0.0, "to_good": 0, "to_medium": 0, "to_bad": 0},
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    medium_row = cm[1].astype(int).tolist()
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "confusion_matrix_3x3": cm.astype(int).tolist(),
        "medium": {
            "n": int(cm[1].sum()),
            "recall": float(cm[1, 1] / max(1, cm[1].sum())),
            "to_good": int(medium_row[0]),
            "to_medium": int(medium_row[1]),
            "to_bad": int(medium_row[2]),
        },
    }


def _describe(values: pd.Series | np.ndarray) -> dict[str, Any]:
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def _bin_counts(values: pd.Series) -> dict[str, dict[str, float | int]]:
    labels = ["<2", "2-3", "3-4", "4-5", "5-6", ">=6"]
    bins = [-np.inf, 2.0, 3.0, 4.0, 5.0, 6.0, np.inf]
    cat = pd.cut(values.astype(float), bins=bins, labels=labels, right=False, include_lowest=True)
    counts = cat.value_counts().reindex(labels, fill_value=0)
    total = max(1, int(counts.sum()))
    return {str(k): {"n": int(v), "frac": float(v / total)} for k, v in counts.items()}


def _value_counts(values: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = values.astype(str).value_counts()
    total = max(1, int(counts.sum()))
    return {str(k): {"n": int(v), "frac": float(v / total)} for k, v in counts.items()}


def _outlier_frac(group: pd.Series, ref: pd.Series) -> float:
    g = pd.Series(group).replace([np.inf, -np.inf], np.nan).dropna()
    r = pd.Series(ref).replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) == 0 or len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    z = np.abs((g.to_numpy(dtype=np.float64) - float(r.mean())) / sd)
    return float(np.mean(z > 2.0))


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Transformer Medium Error Audit")
    lines.append("")
    lines.append(f"- artifact_dir: `{summary['artifact_dir']}`")
    lines.append(f"- model_dir: `{summary['model_dir']}`")
    lines.append(f"- checkpoint: `{summary['checkpoint']}`")
    lines.append(f"- rows: {summary['rows']}")
    lines.append("")

    lines.append("## Sanity Baselines")
    oracle = summary["measured_snr_oracle"]
    lines.append(
        f"- measured-SNR oracle acc: all={_pct(oracle['all']['acc'])}, "
        f"val={_pct(oracle['val']['acc'])}, test={_pct(oracle['test']['acc'])}"
    )
    snr_err = oracle["snr_error"]
    lines.append(
        f"- measured SNR minus label snr_db: mean={_f(snr_err.get('mean'))}, "
        f"median={_f(snr_err.get('median'))}, max={_f(snr_err.get('max'))}, min={_f(snr_err.get('min'))}"
    )
    lines.append("")
    baselines = summary["noisy_feature_baselines"]
    if baselines.get("skipped"):
        lines.append(f"- noisy-only baselines skipped: {baselines['skipped']}")
    else:
        for name in ["logreg", "hist_gradient_boosting"]:
            item = baselines[name]
            lines.append(
                f"- noisy-only `{name}` acc: train={_pct(item['train']['acc'])}, "
                f"val={_pct(item['val']['acc'])}, test={_pct(item['test']['acc'])}; "
                f"test medium recall={_pct(item['test']['medium']['recall'])}"
            )
    lines.append("")

    lines.append("## Transformer Accuracy")
    for split in ["val", "test"]:
        item = summary["transformer"][split]
        med = item["medium"]
        lines.append(
            f"- {split}: acc={_pct(item['acc'])}; medium recall={_pct(med['recall'])}; "
            f"medium -> good={med['to_good']}, medium -> bad={med['to_bad']}"
        )
    lines.append("")

    lines.append("## Medium Error Slices")
    for split in ["val", "test"]:
        lines.append(f"### {split}")
        med = summary["medium_error_audit"][split]
        counts = med["counts"]
        lines.append(
            f"- counts: all={counts['all_medium']}, correct={counts['correct_medium']}, "
            f"to_good={counts['medium_to_good']}, to_bad={counts['medium_to_bad']}"
        )
        for group_name in ["medium_to_good", "medium_to_bad", "correct_medium"]:
            g = med["groups"][group_name]
            if g["n"] == 0:
                lines.append(f"- {group_name}: n=0")
                continue
            top_noise = _top_count(g["noise_kind"])
            lines.append(
                f"- {group_name}: n={g['n']} ({_pct(g['fraction_of_medium'])}); "
                f"snr_db median={_f(g['snr_db']['median'])}, q25-q75={_f(g['snr_db']['q25'])}-{_f(g['snr_db']['q75'])}; "
                f"measured_snr median={_f(g['snr_measured']['median'])}; "
                f"top noise={top_noise}; valid_rr=0 {_pct(g['valid_rr_zero_frac'])}; "
                f"rpeak_fail={_pct(g['noisy_rpeak_failed_frac'])}"
            )
            lines.append(
                f"  probs: p_good={_f(g['p_good']['median'])}, p_medium={_f(g['p_medium']['median'])}, "
                f"p_bad={_f(g['p_bad']['median'])}, top2_margin={_f(g['top2_margin']['median'])}; "
                f"p_medium<0.40 {_pct(g['p_medium_below_0_40_frac'])}; "
                f"|p_good-p_medium|<=0.10 {_pct(g['p_good_medium_close_0_10_frac'])}; "
                f"|p_bad-p_medium|<=0.10 {_pct(g['p_bad_medium_close_0_10_frac'])}"
            )
            lines.append(
                f"  morphology: clean_rms median={_f(g['clean_rms']['median'])}, "
                f"clean_qrs_amp median={_f(g['clean_qrs_amp_median']['median'])}, "
                f"clean_rms outlier={_pct(g['clean_rms_outlier_frac_vs_medium'])}, "
                f"clean_qrs_amp outlier={_pct(g['clean_qrs_amp_outlier_frac_vs_medium'])}"
            )
        lines.append("")

    lines.append("## Read")
    lines.extend(_auto_read(summary))
    lines.append("")
    lines.append("## Files")
    for name, path in summary["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _auto_read(summary: dict[str, Any]) -> list[str]:
    out: list[str] = []
    oracle_acc = min(
        float(summary["measured_snr_oracle"]["val"]["acc"]),
        float(summary["measured_snr_oracle"]["test"]["acc"]),
    )
    baselines = summary["noisy_feature_baselines"]
    if oracle_acc >= 0.999:
        out.append("- measured-SNR oracle is essentially perfect, so the SNR label generation is internally consistent.")
    else:
        out.append("- measured-SNR oracle is not near-perfect; inspect generation/label thresholds before model tuning.")

    if baselines.get("skipped"):
        out.append("- noisy-only baselines were skipped for this split-only audit.")
    else:
        best_noisy_test = max(
            float(baselines["logreg"]["test"]["acc"]),
            float(baselines["hist_gradient_boosting"]["test"]["acc"]),
        )
        if best_noisy_test >= 0.95:
            out.append("- a noisy-only feature baseline reaches 0.95+ test accuracy, so the current transformer is missing usable explicit noise cues.")
        else:
            out.append("- noisy-only baselines also stay below 0.95, which points to a harder medium separability limit in the current dataset.")

    for split in ["val", "test"]:
        groups = summary["medium_error_audit"][split]["groups"]
        to_good = groups["medium_to_good"]
        to_bad = groups["medium_to_bad"]
        if to_good["n"] and to_bad["n"]:
            good_snr = float(to_good["snr_db"]["median"])
            bad_snr = float(to_bad["snr_db"]["median"])
            if good_snr >= 4.8 and bad_snr <= 3.2:
                out.append(
                    f"- {split}: medium->good clusters near the high-SNR edge and medium->bad near the low-SNR edge; "
                    "this looks like boundary/ordinal behavior."
                )
            else:
                out.append(
                    f"- {split}: medium error SNR medians are not cleanly split by both boundaries; check noise kind and morphology slices."
                )
    return out


def _pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{100.0 * float(v):.2f}%"


def _f(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{float(v):.4f}"


def _top_count(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    key = max(counts, key=lambda k: counts[k]["n"])
    return f"{key} ({counts[key]['n']}, {_pct(counts[key]['frac'])})"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _path(value: Any, default: Path, root: Path) -> Path:
    if value is None or str(value) == "":
        return default
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def _parse_splits(value: Any) -> set[str]:
    if value is None or str(value).strip() == "":
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit transformer medium-class errors without retraining.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--model_dir", default="outputs/transformer/models/tune09_fixed_d05_cls22_den25_bad02_ls02_mw115_e26")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--splits", default="", help="Optional comma-separated split subset, e.g. val,test.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(vars(args))


if __name__ == "__main__":
    main()
