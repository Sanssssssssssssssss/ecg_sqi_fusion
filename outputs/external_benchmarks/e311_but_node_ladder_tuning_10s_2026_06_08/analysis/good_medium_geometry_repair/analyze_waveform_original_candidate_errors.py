"""Original-BUT bucket error audit for waveform-only geometry students.

External-only analysis.  Loads saved waveform-only checkpoints and writes
report-only diagnostics for BUT buckets.  Original BUT is not used for training
or model selection by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# Use the active workspace root instead of resolving through archived output
# junctions/symlinks. Some historical run paths resolve into ecg_keep archives.
ROOT = Path.cwd()
ANALYSIS_DIR = Path(__file__).parent
RUNNER_PATH = ANALYSIS_DIR / "run_waveform_geometry_student.py"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
NODE_ID = "N17043_gm_probe"


def load_runner():
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_dir(candidate: str, run_label: str) -> Path:
    return OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / run_label / candidate


def class_name(runner, value: int) -> str:
    return runner.INT_TO_CLASS[int(value)]


def evaluate_candidate(runner, candidate: str, run_label: str, batch_size: int) -> dict:
    run_dir = candidate_dir(candidate, run_label)
    ckpt_path = run_dir / "ckpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["candidate_config"]
    if hasattr(runner, "configure_arch_variant"):
        runner.configure_arch_variant(str(cfg.get("variant_id") or ""))

    base_train = runner.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    norm = runner.ARCH.FeatureNorm(mean=base_train.mean, std=base_train.std)
    original_ds = runner.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    loader = DataLoader(original_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = runner.GeometryStudent(cfg, int(original_ds.x.shape[1])).to(device)
    load_checkpoint_state(model, ckpt["model_state"])
    payload = runner.eval_loader(model, loader, device)
    threshold = float(ckpt.get("bad_threshold_trainval", 0.0))
    pred_badcal = runner.ARCH.apply_bad_threshold(payload.probs, threshold)

    pred_modes = {"raw": payload.pred, "badcal": pred_badcal}
    summary_rows = []
    for mode, pred in pred_modes.items():
        for bucket in ["original_test_all_10s+", "original_all_10s+", "bad_core_nearboundary", "bad_outlier_stress"]:
            mask = runner.ARCH.bucket_mask(original_ds, bucket)
            rep = runner.GEOM.metric_report(original_ds.y[mask], pred[mask], payload.probs[mask])
            summary_rows.append({"candidate": candidate, "mode": mode, "bucket": bucket, **rep})

    feature_cols = [
        "pc1",
        "pc2",
        "pc3",
        "pca_margin",
        "boundary_confidence",
        "knn_label_purity",
        "qrs_visibility",
        "detector_agreement",
        "baseline_step",
        "sqi_basSQI",
        "flatline_ratio",
        "non_qrs_diff_p95",
        "amplitude_entropy",
        "low_amp_ratio",
        "mean_abs",
        "ptp_p99_p01",
        "band_15_30",
        "band_30_45",
    ]
    feature_idx = {col: runner.FEATURE_COLUMNS.index(col) for col in feature_cols if col in runner.FEATURE_COLUMNS}
    rows = []
    for i in range(len(original_ds.y)):
        y = int(original_ds.y[i])
        raw = int(payload.pred[i])
        badcal = int(pred_badcal[i])
        row = {
            "row": i,
            "split": str(original_ds.split[i]),
            "region": str(original_ds.region[i]),
            "true": class_name(runner, y),
            "pred_raw": class_name(runner, raw),
            "pred_badcal": class_name(runner, badcal),
            "correct_raw": y == raw,
            "correct_badcal": y == badcal,
            "prob_good": float(payload.probs[i, 0]),
            "prob_medium": float(payload.probs[i, 1]),
            "prob_bad": float(payload.probs[i, 2]),
            "is_test": str(original_ds.split[i]) == "test",
            "is_bad_core": str(original_ds.split[i]) == "test" and y == runner.CLASS_TO_INT["bad"] and str(original_ds.region[i]) != "outlier_low_confidence",
            "is_bad_outlier": str(original_ds.split[i]) == "test" and y == runner.CLASS_TO_INT["bad"] and str(original_ds.region[i]) == "outlier_low_confidence",
        }
        for col, idx in feature_idx.items():
            row[f"target_{col}"] = float(original_ds.aux[i, idx])
            row[f"pred_{col}"] = float(payload.aux_pred[i, idx])
            row[f"err_{col}"] = float(payload.aux_pred[i, idx] - original_ds.aux[i, idx])
        rows.append(row)
    pred_df = pd.DataFrame(rows)

    out_dir = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "original_candidate_error_audit" / candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "bucket_summary.csv", index=False)
    pred_df.to_csv(out_dir / "original_predictions.csv", index=False)
    summarize_feature_gaps(pred_df, feature_idx, out_dir / "error_feature_gap.csv")

    for name, query in {
        "original_test_errors_raw": "is_test and not correct_raw",
        "bad_outlier_errors_raw": "is_bad_outlier and not correct_raw",
        "bad_core_errors_raw": "is_bad_core and not correct_raw",
        "good_to_medium_raw": "is_test and true == 'good' and pred_raw == 'medium'",
        "medium_to_good_raw": "is_test and true == 'medium' and pred_raw == 'good'",
        "nonbad_to_bad_raw": "is_test and true != 'bad' and pred_raw == 'bad'",
    }.items():
        try:
            pred_df.query(query).to_csv(out_dir / f"{name}.csv", index=False)
        except OSError:
            pass

    try:
        plot_error_panels(runner, original_ds, pred_df, out_dir / "original_error_waveform_panels.png", candidate)
    except OSError as exc:
        try:
            (out_dir / "panel_skip.txt").write_text(
                f"Skipped waveform panel render: {type(exc).__name__}: {exc}",
                encoding="utf-8",
            )
        except OSError:
            pass

    report = render_report(candidate, summary_rows, pred_df, out_dir)
    try:
        (out_dir / "audit_report.md").write_text(report, encoding="utf-8")
    except OSError:
        pass
    return {"candidate": candidate, "out_dir": str(out_dir), "summary": summary_rows}


def summarize_feature_gaps(pred_df: pd.DataFrame, feature_idx: dict[str, int], out_path: Path) -> None:
    groups = {
        "test_correct": "is_test and correct_raw",
        "test_errors": "is_test and not correct_raw",
        "good_correct": "is_test and true == 'good' and correct_raw",
        "medium_correct": "is_test and true == 'medium' and correct_raw",
        "bad_correct": "is_test and true == 'bad' and correct_raw",
        "good_to_medium": "is_test and true == 'good' and pred_raw == 'medium'",
        "medium_to_good": "is_test and true == 'medium' and pred_raw == 'good'",
        "bad_outlier_error": "is_bad_outlier and not correct_raw",
        "bad_core_error": "is_bad_core and not correct_raw",
        "nonbad_to_bad": "is_test and true != 'bad' and pred_raw == 'bad'",
    }
    rows = []
    for group, query in groups.items():
        try:
            view = pred_df.query(query)
        except Exception:
            continue
        if view.empty:
            continue
        for col in feature_idx:
            rows.append(
                {
                    "group": group,
                    "feature": col,
                    "n": int(len(view)),
                    "target_mean_z": float(pd.to_numeric(view[f"target_{col}"], errors="coerce").mean()),
                    "pred_mean_z": float(pd.to_numeric(view[f"pred_{col}"], errors="coerce").mean()),
                    "error_mean_z": float(pd.to_numeric(view[f"err_{col}"], errors="coerce").mean()),
                    "target_p10_z": float(pd.to_numeric(view[f"target_{col}"], errors="coerce").quantile(0.10)),
                    "target_p90_z": float(pd.to_numeric(view[f"target_{col}"], errors="coerce").quantile(0.90)),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def load_checkpoint_state(model: torch.nn.Module, state: dict) -> None:
    """Load historical waveform-student checkpoints for report-only audits.

    Some old artifact-head checkpoints predate optional submodules added to the
    experiment runner.  For these diagnostics we load all shape-compatible
    weights and leave newly added heads at initialization; this keeps old
    checkpoints evaluable without rewriting them.
    """
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    relevant_missing = [
        key
        for key in missing
        if not (
            key.startswith("artifact_bad_expert.local_event_head")
            or key.startswith("teacher_prototypes")
            or key.startswith("waveform_atlas")
        )
    ]
    if skipped or relevant_missing or unexpected:
        print(
            "checkpoint compatible load "
            f"copied={len(compatible)} skipped={len(skipped)} "
            f"missing={len(missing)} relevant_missing={len(relevant_missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )


def plot_error_panels(runner, ds, pred_df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    selections = []
    for label, query in [
        ("bad outlier -> nonbad", "is_bad_outlier and pred_raw != 'bad'"),
        ("bad core -> nonbad", "is_bad_core and pred_raw != 'bad'"),
        ("good -> medium", "is_test and true == 'good' and pred_raw == 'medium'"),
        ("medium -> good", "is_test and true == 'medium' and pred_raw == 'good'"),
    ]:
        sub = pred_df.query(query).head(3)
        for _, row in sub.iterrows():
            selections.append((label, int(row["row"])))
    if not selections:
        path.with_suffix(".txt").write_text("no selected errors", encoding="utf-8")
        return
    cols = 3
    rows = int(math.ceil(len(selections) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.7 * rows), dpi=160, squeeze=False)
    t = np.arange(ds.x.shape[-1])
    colors = ["#252525", "#3182bd", "#de2d26"]
    for ax, (label, idx) in zip(axes.ravel(), selections):
        x = ds.x[idx]
        scale = np.nanpercentile(np.abs(x), 95) + 1e-6
        for ch in range(x.shape[0]):
            ax.plot(t, x[ch] / scale + ch * 2.2, lw=0.75, color=colors[ch % len(colors)], alpha=0.85)
        y = class_name(runner, int(ds.y[idx]))
        pred = str(pred_df.loc[pred_df["row"] == idx, "pred_raw"].iloc[0])
        prob_bad = float(pred_df.loc[pred_df["row"] == idx, "prob_bad"].iloc[0])
        ax.set_title(f"{label}: {y}->{pred}, p_bad={prob_bad:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(selections) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_report(candidate: str, summary_rows: list[dict], pred_df: pd.DataFrame, out_dir: Path) -> str:
    lines = [f"# Original Candidate Error Audit: {candidate}", ""]
    lines += ["Original BUT is report-only; no training or selection uses these rows.", ""]
    lines += ["## Buckets", ""]
    lines += ["| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(
            f"| {row['mode']} | {row['bucket']} | {float(row['acc']):.6f} | "
            f"{float(row['good_recall']):.6f} | {float(row['medium_recall']):.6f} | {float(row['bad_recall']):.6f} | "
            f"{int(row['good_to_medium'])} | {int(row['medium_to_good'])} | {int(row['bad_to_medium'])} |"
        )
    lines += ["", "## Error Counts", ""]
    for label, query in [
        ("test errors raw", "is_test and not correct_raw"),
        ("bad outlier errors raw", "is_bad_outlier and not correct_raw"),
        ("bad core errors raw", "is_bad_core and not correct_raw"),
        ("good->medium raw", "is_test and true == 'good' and pred_raw == 'medium'"),
        ("medium->good raw", "is_test and true == 'medium' and pred_raw == 'good'"),
        ("nonbad->bad raw", "is_test and true != 'bad' and pred_raw == 'bad'"),
    ]:
        lines.append(f"- {label}: {len(pred_df.query(query))}")
    lines += ["", f"![error panels]({(out_dir / 'original_error_waveform_panels.png').as_posix()})", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="predtop20_sqiquery_subject111_dual_stress_pretrain,predtop20_sqiquery_subject111_mixedbad_dual_p28",
    )
    parser.add_argument("--run-label", default="search")
    parser.add_argument("--batch-size", type=int, default=384)
    args = parser.parse_args()
    runner = load_runner()
    results = []
    for candidate in [x.strip() for x in args.candidates.split(",") if x.strip()]:
        results.append(evaluate_candidate(runner, candidate, args.run_label, args.batch_size))
    out_path = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "original_candidate_error_audit" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
