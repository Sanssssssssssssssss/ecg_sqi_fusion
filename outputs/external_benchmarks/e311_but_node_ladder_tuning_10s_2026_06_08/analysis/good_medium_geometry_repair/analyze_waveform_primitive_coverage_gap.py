"""Waveform-computable primitive coverage gap audit.

This external-only diagnostic asks a narrower question than accuracy:

Do BUT original-test mistakes live outside the PTB synthetic training geometry
when both are represented only by waveform-computable primitive statistics?

If yes, more architecture is unlikely to fix the error without expanding the
synthetic generator.  If no, the architecture/loss is failing to use available
waveform evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path.cwd()
ANALYSIS_DIR = Path(__file__).parent
RUNNER_PATH = ANALYSIS_DIR / "run_waveform_geometry_student.py"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
AUDIT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "original_candidate_error_audit"


def load_runner():
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def primitive_matrix(runner, ds, bank: str, batch_size: int) -> np.ndarray:
    pieces = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            x = torch.from_numpy(ds.x[start : start + batch_size]).float()
            stats = runner.primitive_waveform_stats(x, bank).cpu().numpy().astype(np.float32)
            pieces.append(stats)
    return np.concatenate(pieces, axis=0)


def knn_min_distance(query: np.ndarray, ref: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    ref_norm = np.sum(ref * ref, axis=1, keepdims=True).T
    mins = np.empty(query.shape[0], dtype=np.float32)
    idxs = np.empty(query.shape[0], dtype=np.int64)
    for start in range(0, query.shape[0], chunk):
        q = query[start : start + chunk]
        d = np.sum(q * q, axis=1, keepdims=True) + ref_norm - 2.0 * (q @ ref.T)
        d = np.maximum(d, 0.0)
        idx = np.argmin(d, axis=1)
        mins[start : start + len(q)] = np.sqrt(d[np.arange(len(q)), idx]).astype(np.float32)
        idxs[start : start + len(q)] = idx.astype(np.int64)
    return mins, idxs


def robust_standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    med = np.nanmedian(train, axis=0)
    q1 = np.nanpercentile(train, 25, axis=0)
    q3 = np.nanpercentile(train, 75, axis=0)
    scale = np.where((q3 - q1) > 1e-6, q3 - q1, np.nanstd(train, axis=0) + 1e-6)
    out = [np.nan_to_num((train - med) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)]
    for arr in others:
        out.append(np.nan_to_num((arr - med) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    return tuple(out)


def summarize(work: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in work.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(by, keys)}
        row.update(
            {
                "n": int(len(sub)),
                "err_rate": float((~sub["correct_raw"].astype(bool)).mean()) if len(sub) else np.nan,
                "nn_dist_median": float(sub["nn_dist"].median()) if len(sub) else np.nan,
                "nn_dist_p90": float(sub["nn_dist"].quantile(0.90)) if len(sub) else np.nan,
                "same_class_nn_rate": float(sub["same_class_nn"].mean()) if len(sub) else np.nan,
                "same_region_nn_rate": float(sub["same_region_nn"].mean()) if len(sub) else np.nan,
                "pred_top": str(sub["pred_raw"].value_counts().idxmax()) if len(sub) else "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["err_rate", "n"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="predtop20_sqiquery_subject111_impulsebad_dual_p20")
    parser.add_argument("--bank", default="qrs_stress_v5")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    runner = load_runner()
    syn_train = runner.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = runner.ARCH.FeatureNorm(mean=syn_train.mean, std=syn_train.std)
    original = runner.ARCH.OriginalWaveDataset(norm, "robust3")

    syn_stats = primitive_matrix(runner, syn_train, args.bank, args.batch_size)
    orig_stats = primitive_matrix(runner, original, args.bank, args.batch_size)
    syn_z, orig_z = robust_standardize(syn_stats, orig_stats)

    nn_dist, nn_idx = knn_min_distance(orig_z, syn_z, chunk=args.chunk)
    syn_labels = np.asarray([runner.INT_TO_CLASS[int(v)] for v in syn_train.y], dtype=object)
    syn_regions = syn_train.frame.get("block", pd.Series("", index=syn_train.frame.index)).astype(str).to_numpy()

    pred_path = AUDIT_DIR / args.candidate / "original_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    pred = pd.read_csv(pred_path).reset_index(drop=True)
    if len(pred) != len(original.y):
        raise RuntimeError(f"prediction/original length mismatch: {len(pred)} vs {len(original.y)}")

    work = pred.copy()
    work["nn_dist"] = nn_dist
    work["nearest_syn_label"] = syn_labels[nn_idx]
    work["nearest_syn_region"] = syn_regions[nn_idx]
    work["same_class_nn"] = work["nearest_syn_label"].astype(str).eq(work["true"].astype(str))
    work["same_region_nn"] = work["nearest_syn_region"].astype(str).eq(work["region"].astype(str))

    out_dir = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "waveform_primitive_coverage_gap"
    out_dir.mkdir(parents=True, exist_ok=True)
    work.to_csv(out_dir / f"{args.candidate}_{args.bank}_rows.csv", index=False)
    by_true_region = summarize(work[work["is_test"].astype(bool)], ["true", "region"])
    by_true_region.to_csv(out_dir / f"{args.candidate}_{args.bank}_by_true_region.csv", index=False)
    by_error = summarize(work[work["is_test"].astype(bool)], ["true", "pred_raw", "region"])
    by_error.to_csv(out_dir / f"{args.candidate}_{args.bank}_by_error.csv", index=False)

    test = work[work["is_test"].astype(bool)].copy()
    correct = test[test["correct_raw"].astype(bool)]
    wrong = test[~test["correct_raw"].astype(bool)]
    summary = {
        "candidate": args.candidate,
        "bank": args.bank,
        "n_synthetic_train": int(len(syn_train)),
        "n_original_test": int(len(test)),
        "correct_nn_dist_median": float(correct["nn_dist"].median()),
        "wrong_nn_dist_median": float(wrong["nn_dist"].median()),
        "correct_same_class_nn_rate": float(correct["same_class_nn"].mean()),
        "wrong_same_class_nn_rate": float(wrong["same_class_nn"].mean()),
        "bad_outlier_wrong_nn_dist_median": float(
            test[(test["is_bad_outlier"].astype(bool)) & (~test["correct_raw"].astype(bool))]["nn_dist"].median()
        ),
        "medium_to_good_nn_dist_median": float(
            test[(test["true"] == "medium") & (test["pred_raw"] == "good")]["nn_dist"].median()
        ),
        "good_to_medium_nn_dist_median": float(
            test[(test["true"] == "good") & (test["pred_raw"] == "medium")]["nn_dist"].median()
        ),
    }
    (out_dir / f"{args.candidate}_{args.bank}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Waveform Primitive Coverage Gap",
        "",
        "Report-only diagnostic. It compares BUT original-test rows with PTB synthetic train rows in waveform-computable primitive-stat space.",
        "",
        f"Candidate: `{args.candidate}`",
        f"Primitive bank: `{args.bank}`",
        "",
        "## Headline",
        "",
        f"- correct median NN distance: `{summary['correct_nn_dist_median']:.4f}`",
        f"- wrong median NN distance: `{summary['wrong_nn_dist_median']:.4f}`",
        f"- correct same-class nearest-neighbor rate: `{summary['correct_same_class_nn_rate']:.3f}`",
        f"- wrong same-class nearest-neighbor rate: `{summary['wrong_same_class_nn_rate']:.3f}`",
        f"- bad-outlier wrong median NN distance: `{summary['bad_outlier_wrong_nn_dist_median']:.4f}`",
        f"- medium->good median NN distance: `{summary['medium_to_good_nn_dist_median']:.4f}`",
        f"- good->medium median NN distance: `{summary['good_to_medium_nn_dist_median']:.4f}`",
        "",
        "## By True/Region",
        "",
        by_true_region.head(16).to_csv(index=False),
        "",
        "## By Error Type",
        "",
        by_error.head(20).to_csv(index=False),
        "",
    ]
    (out_dir / "waveform_primitive_coverage_gap_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
