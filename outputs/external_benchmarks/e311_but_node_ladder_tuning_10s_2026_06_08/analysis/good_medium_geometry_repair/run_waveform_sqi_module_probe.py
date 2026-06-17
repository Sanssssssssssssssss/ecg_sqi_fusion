"""Module-first waveform SQI probe.

This is an analysis-only experiment.  It does not use 47-column SQI/geometry
features as inference inputs for a final model.  Instead it asks a narrower
question:

    Which ECG-quality submodules are recoverable from the waveform?

The probe splits waveform-derived evidence into physiologic/SQI groups:

- qrs_rr: QRS visibility, detector-agreement surrogates, event/RR-like tokens.
- baseline_band: baseline wander/step and band-power proxies.
- local_artifact: sparse reset/dropout/contact-loss evidence.
- stable_all: all waveform-computable primitive groups.

It then fits small diagnostic classifiers/regressors on PTB synthetic train
only and evaluates synthetic/node plus original BUT report-only buckets.  A
strong result here is not a formal final model; it is a module viability check
for the next beat/RR-aware Transformer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch


RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
ARCH_SCRIPT = ANALYSIS_DIR / "run_waveform_architecture_search.py"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"


KEY_TARGETS = [
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "sqi_basSQI",
    "flatline_ratio",
    "non_qrs_diff_p95",
    "pca_margin",
    "pc1",
    "pc2",
    "pc3",
    "boundary_confidence",
    "knn_label_purity",
]

FEATURE_TAXONOMY = {
    "stable_waveform_sqi_morph_rr": [
        "qrs_visibility",
        "qrs_band_ratio",
        "qrs_prom_p90",
        "template_corr",
        "detector_agreement",
        "baseline_step",
        "flatline_ratio",
        "contact_loss_win_ratio",
        "non_qrs_rms_ratio",
        "non_qrs_diff_p95",
        "diff_abs_p95",
        "band_15_30",
        "band_30_45",
        "rms",
        "std",
        "mean_abs",
        "ptp_p99_p01",
        "amplitude_entropy",
        "low_amp_ratio",
        "sqi_iSQI",
        "sqi_bSQI",
        "sqi_pSQI",
        "sqi_sSQI",
        "sqi_kSQI",
        "sqi_fSQI",
        "sqi_basSQI",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "zero_crossing_rate",
        "diff_zero_crossing_rate",
        "sample_entropy_proxy",
        "higuchi_fd_proxy",
        "wavelet_e0",
        "wavelet_e1",
        "wavelet_e2",
        "wavelet_e3",
        "wavelet_e4",
        "fatal_or_score",
    ],
    "weak_target_distribution_proxy": ["pc1", "pc3", "pca_margin"],
    "atlas_label_geometry_not_waveform_fact": ["pc2", "pc4", "boundary_confidence", "region_confidence", "knn_label_purity"],
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCH = load_module(ARCH_SCRIPT, "probe_arch")
STUDENT = load_module(STUDENT_SCRIPT, "probe_student")
FEATURE_COLUMNS = list(ARCH.FEATURE_COLUMNS)
KEY_IDXS = [FEATURE_COLUMNS.index(c) for c in KEY_TARGETS if c in FEATURE_COLUMNS]
KEY_COLS = [c for c in KEY_TARGETS if c in FEATURE_COLUMNS]


@dataclass
class SplitData:
    name: str
    x: np.ndarray
    aux: np.ndarray
    y: np.ndarray
    split: np.ndarray | None = None
    region: np.ndarray | None = None


def _collect_dataset(ds: Any) -> SplitData:
    split = getattr(ds, "split", None)
    region = getattr(ds, "region", None)
    return SplitData(
        name=ds.__class__.__name__,
        x=np.asarray(ds.x, dtype=np.float32),
        aux=np.asarray(ds.aux, dtype=np.float32),
        y=np.asarray(ds.y, dtype=np.int64),
        split=np.asarray(split) if split is not None else None,
        region=np.asarray(region) if region is not None else None,
    )


def load_data(channel_mode: str) -> dict[str, SplitData]:
    train = ARCH.SyntheticWaveDataset("train", channel_mode)
    norm = ARCH.FeatureNorm(mean=train.mean, std=train.std)
    return {
        "synthetic_train": _collect_dataset(train),
        "synthetic_val": _collect_dataset(ARCH.SyntheticWaveDataset("val", channel_mode)),
        "synthetic_test": _collect_dataset(ARCH.SyntheticWaveDataset("test", channel_mode)),
        "original_all": _collect_dataset(ARCH.OriginalWaveDataset(norm, channel_mode)),
    }


def _batch_stats(x: np.ndarray, fn: Any, batch_size: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32))
            rows.append(fn(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def event_rr_summary(x: torch.Tensor) -> torch.Tensor:
    """Compact RR/event summary from top QRS-like event windows."""

    feat = STUDENT.qrs_event_patch_features(x, event_count=18, window=101)
    # [B, C, K, F] -> per-channel mean/std/max/min and adjacent RR-like spacing.
    mean = feat.mean(dim=2)
    std = feat.std(dim=2)
    mx = feat.amax(dim=2)
    mn = feat.amin(dim=2)
    pos = feat[..., -1]
    pos_sorted = torch.sort(pos, dim=2).values
    spacing = pos_sorted[:, :, 1:] - pos_sorted[:, :, :-1]
    rr_mean = spacing.mean(dim=2, keepdim=True)
    rr_std = spacing.std(dim=2, keepdim=True)
    rr_min = spacing.amin(dim=2, keepdim=True)
    rr_max = spacing.amax(dim=2, keepdim=True)
    rr_cv = rr_std / (rr_mean.abs() + 1e-6)
    rr = torch.cat([rr_mean, rr_std, rr_min, rr_max, rr_cv], dim=2)
    return torch.cat([mean.flatten(1), std.flatten(1), mx.flatten(1), mn.flatten(1), rr.flatten(1)], dim=1)


def build_module_features(
    data: dict[str, SplitData], batch_size: int, include_full_bank: bool
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    module_parts = {
        "qrs_rr": [
            ("qrs_visibility_bank", STUDENT.qrs_visibility_bank_stats),
            ("qrs_detector_agreement", STUDENT.qrs_detector_agreement_stats),
            ("event_rr_summary", event_rr_summary),
        ],
        "baseline_band": [
            ("baseline_frequency", STUDENT.baseline_frequency_stats),
            ("waveform_stress", STUDENT.waveform_stress_stats),
        ],
        "local_artifact": [
            ("sparse_extreme_event", STUDENT.sparse_extreme_event_stats),
            ("waveform_stress", STUDENT.waveform_stress_stats),
        ],
        "morph_detail": [
            ("waveform_stats_v2", STUDENT.waveform_stats_v2),
            ("qrs_detail", STUDENT.qrs_detail_stats),
        ],
    }
    features: dict[str, dict[str, np.ndarray]] = {name: {} for name in data}
    dims: dict[str, int] = {}
    for split_name, split in data.items():
        cache: dict[str, np.ndarray] = {}
        for group, parts in module_parts.items():
            arrays = []
            for part_name, fn in parts:
                key = f"{group}:{part_name}"
                if key not in cache:
                    cache[key] = _batch_stats(split.x, fn, batch_size)
                arrays.append(cache[key])
            features[split_name][group] = np.concatenate(arrays, axis=1)
        features[split_name]["stable_all"] = np.concatenate(
            [
                features[split_name]["qrs_rr"],
                features[split_name]["baseline_band"],
                features[split_name]["local_artifact"],
                features[split_name]["morph_detail"],
            ],
            axis=1,
        )
        if include_full_bank:
            # Full primitive bank is useful as an upper diagnostic for waveform-computable stats,
            # but it is too heavy for every fast iteration.
            features[split_name]["qrs_stress_v5_all"] = _batch_stats(
                split.x, lambda xb: STUDENT.primitive_waveform_stats(xb, "qrs_stress_v5"), batch_size
            )
    for group in features["synthetic_train"]:
        dims[group] = int(features["synthetic_train"][group].shape[1])
    return features, dims


def metric_report(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, Any]:
    return ARCH.GEOM.metric_report(y.astype(np.int64), pred.astype(np.int64), prob)


def bucket_indices(split: SplitData, bucket: str) -> np.ndarray:
    if split.split is None or split.region is None:
        return np.arange(len(split.y))
    if bucket == "original_test_all_10s+":
        return np.where(split.split.astype(str) == "test")[0]
    if bucket == "original_all_10s+":
        return np.arange(len(split.y))
    if bucket == "bad_core_nearboundary":
        mask = (split.split.astype(str) == "test") & (split.y == 2) & (split.region.astype(str) != "outlier_low_confidence")
        return np.where(mask)[0]
    if bucket == "bad_outlier_stress":
        mask = (split.split.astype(str) == "test") & (split.y == 2) & (split.region.astype(str) == "outlier_low_confidence")
        return np.where(mask)[0]
    raise ValueError(f"unknown bucket {bucket}")


def append_metric(rows: list[dict[str, Any]], model: str, group: str, bucket: str, y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> None:
    rep = metric_report(y, pred, prob)
    rows.append(
        {
            "model": model,
            "feature_group": group,
            "bucket": bucket,
            "n": int(len(y)),
            "acc": rep["acc"],
            "macro_f1": rep["macro_f1"],
            "good_recall": rep["good_recall"],
            "medium_recall": rep["medium_recall"],
            "bad_recall": rep["bad_recall"],
            "good_to_medium": rep["good_to_medium"],
            "medium_to_good": rep["medium_to_good"],
            "bad_to_medium": rep["bad_to_medium"],
            "confusion_3x3": json.dumps(rep["confusion_3x3"]),
        }
    )


def run_classification_probe(features: dict[str, dict[str, np.ndarray]], data: dict[str, SplitData], fast: bool) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows: list[dict[str, Any]] = []
    groups = list(features["synthetic_train"].keys())
    models = {
        "hgb": lambda seed: HistGradientBoostingClassifier(
            max_iter=80 if fast else 180,
            learning_rate=0.055,
            max_leaf_nodes=31,
            l2_regularization=0.025,
            random_state=seed,
        ),
        "extratrees": lambda seed: ExtraTreesClassifier(
            n_estimators=72 if fast else 192,
            max_depth=12 if fast else 18,
            min_samples_leaf=2,
            max_features=0.45,
            class_weight={0: 1.0, 1: 1.25, 2: 2.6},
            random_state=seed,
            n_jobs=-1,
        ),
    }
    for group in groups:
        x_train = features["synthetic_train"][group]
        y_train = data["synthetic_train"].y
        for model_name, factory in models.items():
            clf = make_pipeline(StandardScaler(), factory(20261700 + len(group) + len(model_name)))
            clf.fit(x_train, y_train)
            for split_name in ["synthetic_val", "synthetic_test"]:
                prob = clf.predict_proba(features[split_name][group])
                pred = prob.argmax(axis=1)
                append_metric(rows, model_name, group, split_name, data[split_name].y, pred, prob)
            orig = data["original_all"]
            orig_prob = clf.predict_proba(features["original_all"][group])
            orig_pred = orig_prob.argmax(axis=1)
            for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
                idx = bucket_indices(orig, bucket)
                append_metric(rows, model_name, group, bucket, orig.y[idx], orig_pred[idx], orig_prob[idx])
    return pd.DataFrame(rows)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_recovery_probe(features: dict[str, dict[str, np.ndarray]], data: dict[str, SplitData], fast: bool) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows: list[dict[str, Any]] = []
    y_train = data["synthetic_train"].aux[:, KEY_IDXS]
    for group in features["synthetic_train"]:
        reg = make_pipeline(
            StandardScaler(),
            ExtraTreesRegressor(
                n_estimators=64 if fast else 160,
                max_depth=12 if fast else 18,
                min_samples_leaf=2,
                max_features=0.50,
                random_state=20261737 + len(group),
                n_jobs=-1,
            ),
        )
        reg.fit(features["synthetic_train"][group], y_train)
        for split_name in ["synthetic_val", "synthetic_test", "original_all"]:
            pred = reg.predict(features[split_name][group])
            truth = data[split_name].aux[:, KEY_IDXS]
            for j, col in enumerate(KEY_COLS):
                rows.append(
                    {
                        "feature_group": group,
                        "split": split_name,
                        "target": col,
                        "corr": _corr(pred[:, j], truth[:, j]),
                        "mae_z": float(np.nanmean(np.abs(pred[:, j] - truth[:, j]))),
                    }
                )
    return pd.DataFrame(rows)


def write_taxonomy() -> pd.DataFrame:
    rows = []
    assigned = {}
    for group, cols in FEATURE_TAXONOMY.items():
        for col in cols:
            assigned[col] = group
            rows.append({"feature": col, "taxonomy": group})
    for col in FEATURE_COLUMNS:
        if col not in assigned:
            rows.append({"feature": col, "taxonomy": "unassigned_review"})
    return pd.DataFrame(rows)


def render_report(metrics: pd.DataFrame, recovery: pd.DataFrame, dims: dict[str, int], taxonomy: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Waveform SQI Module Probe")
    lines.append("")
    lines.append(f"Created: {now()}")
    lines.append("")
    lines.append("This is a module-first diagnostic. It uses PTB synthetic train only for fitting diagnostic probes; original BUT remains report-only.")
    lines.append("")
    lines.append("## Feature Taxonomy")
    lines.append("")
    for tax, frame in taxonomy.groupby("taxonomy"):
        lines.append(f"- `{tax}`: {len(frame)} features")
    lines.append("")
    lines.append("## Module Feature Dimensions")
    lines.append("")
    for group, dim in dims.items():
        lines.append(f"- `{group}`: {dim} waveform-derived dimensions")
    lines.append("")
    lines.append("## Best Original-Test Classification Probes")
    lines.append("")
    view = metrics[metrics["bucket"].eq("original_test_all_10s+")].sort_values("acc", ascending=False).head(12)
    lines.append("| model | group | acc | good | medium | bad |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in view.iterrows():
        lines.append(
            f"| `{row['model']}` | `{row['feature_group']}` | {row['acc']:.6f} | {row['good_recall']:.3f} | {row['medium_recall']:.3f} | {row['bad_recall']:.3f} |"
        )
    lines.append("")
    lines.append("## Bad Stress Buckets")
    lines.append("")
    bad = metrics[metrics["bucket"].isin(["bad_core_nearboundary", "bad_outlier_stress"])].sort_values(["bucket", "acc"], ascending=[True, False]).groupby("bucket").head(8)
    lines.append("| bucket | model | group | acc / bad recall |")
    lines.append("|---|---|---|---:|")
    for _, row in bad.iterrows():
        lines.append(f"| `{row['bucket']}` | `{row['model']}` | `{row['feature_group']}` | {row['bad_recall']:.6f} |")
    lines.append("")
    lines.append("## Feature Recovery Snapshot")
    lines.append("")
    if recovery.empty:
        lines.append("_Feature recovery skipped for this run._")
    else:
        snap = recovery[recovery["split"].eq("synthetic_test")].pivot_table(index="target", columns="feature_group", values="corr", aggfunc="max")
        snap = snap.round(3).reset_index()
        cols = list(snap.columns)
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for _, row in snap.iterrows():
            lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in cols) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If `qrs_rr` does not recover `qrs_visibility` / `detector_agreement`, the next Transformer needs true event/RR sequence tokens rather than global QRS summaries.")
    lines.append("- If `baseline_band` does not recover `baseline_step` / `sqi_basSQI`, fixed filterbank/baseline token streams should become first-class inputs.")
    lines.append("- `boundary_confidence`, `knn_label_purity`, and much of `pc2` are atlas/label geometry diagnostics, not single-waveform facts; poor recovery there should not drive architecture capacity.")
    lines.append("- The formal model must still be waveform-only. These probes are a map for what the waveform encoder must learn, not a final classifier claim.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-mode", default="robust3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fast", action="store_true", help="Run a smaller first-pass probe.")
    parser.add_argument("--include-full-bank", action="store_true", help="Also compute qrs_stress_v5_all primitive bank.")
    parser.add_argument("--render-existing", action="store_true", help="Render report from existing CSV outputs.")
    parser.add_argument("--skip-recovery", action="store_true", help="Skip regression feature-recovery probes.")
    parser.add_argument("--tag", default="waveform_sqi_module_probe", help="Output file prefix.")
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prefix = str(args.tag)
    metrics_path = ANALYSIS_DIR / f"{prefix}_classification_metrics.csv"
    recovery_path = ANALYSIS_DIR / f"{prefix}_feature_recovery.csv"
    taxonomy_path = ANALYSIS_DIR / f"{prefix}_feature_taxonomy.csv"
    summary_path = ANALYSIS_DIR / f"{prefix}_summary.json"
    report_path = REPORT_DIR / f"{prefix}_report.md"

    if args.render_existing:
        metrics = pd.read_csv(metrics_path)
        recovery = pd.read_csv(recovery_path)
        taxonomy = pd.read_csv(taxonomy_path)
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            dims = {str(k): int(v) for k, v in payload.get("module_dims", {}).items()}
        else:
            dims = {}
        report = render_report(metrics, recovery, dims, taxonomy)
        report_path.write_text(report, encoding="utf-8")
        print(report)
        return

    data = load_data(args.channel_mode)
    features, dims = build_module_features(data, args.batch_size, include_full_bank=bool(args.include_full_bank))
    metrics = run_classification_probe(features, data, fast=bool(args.fast))
    if args.skip_recovery:
        recovery = pd.DataFrame(columns=["feature_group", "split", "target", "corr", "mae_z"])
    else:
        recovery = run_recovery_probe(features, data, fast=bool(args.fast))
    taxonomy = write_taxonomy()

    metrics.to_csv(metrics_path, index=False)
    recovery.to_csv(recovery_path, index=False)
    taxonomy.to_csv(taxonomy_path, index=False)
    payload = {
        "created_at": now(),
        "channel_mode": args.channel_mode,
        "fast": bool(args.fast),
        "include_full_bank": bool(args.include_full_bank),
        "skip_recovery": bool(args.skip_recovery),
        "original_used_for_training": False,
        "module_dims": dims,
        "metrics_csv": str(metrics_path),
        "feature_recovery_csv": str(recovery_path),
        "taxonomy_csv": str(taxonomy_path),
        "report": str(report_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(metrics, recovery, dims, taxonomy)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
