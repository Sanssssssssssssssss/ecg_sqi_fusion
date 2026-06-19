"""Report-only bad-score frontier for waveform Transformer checkpoints.

This script never trains and never selects on original BUT.  It answers one
diagnostic question: do waveform-only checkpoints rank original bad-stress
windows highly enough that calibration could recover them, or is the bad signal
itself missing from the model score?
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path.cwd().resolve()
RUNNER = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair" / "run_waveform_geometry_student.py"
OUT_DIR = RUNNER.parent
RUN_ROOT = ROOT / "outputs" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "runs" / "waveform_geometry_student" / "N17043_gm_probe" / "search"
REPORT_DIR = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair"

CANDIDATES = [
    "gated_bad_multiscale_stablemix",
    "primitive_badexpert_multiscale",
    "qrsbank_badexpert_balanced",
    "qrsbank_badexpert_badguard",
    "qrsbank_badstress_hardneg_balanced",
    "qrsbank_badstress_hardneg_wide",
    "qrsbank_stattoken_balanced",
    "qrsbank_stattoken_badguard",
    "qrsbank_highamp_patch_balanced",
    "qrsbank_highamp_stattoken_badguard",
    "qrsbank_highamp_mix_patch_guard",
    "qrsbank_highamp_mix_stattoken_guard",
    "qrsbank_predmargin_badguard_patch",
    "qrsbank_predmargin_badguard_stattoken",
    "qrsbank_highamp_mix_patch_stressselect",
    "qrsbank_top14_focus_patch",
    "qrsbank_top14_focus_stattoken",
    "sqiquery_top14_boundary",
    "sqiquery_top14_badguard",
    "sqiquery_auxprimitive_boundary",
    "sqiquery_auxprimitive_badguard",
]


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score).astype(float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    pos_rank_sum = float(ranks[y].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def metric_row(mod: Any, name: str, threshold: float, y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = mod.ARCH.apply_bad_threshold(probs, threshold)
    rep = mod.GEOM.metric_report(y, pred, probs)
    return {
        "candidate": name,
        "threshold": threshold,
        "acc": rep["acc"],
        "macro_f1": rep["macro_f1"],
        "good_recall": rep["good_recall"],
        "medium_recall": rep["medium_recall"],
        "bad_recall": rep["bad_recall"],
        "good_to_bad": rep.get("good_to_bad", 0),
        "medium_to_bad": rep.get("medium_to_bad", 0),
        "bad_to_good": rep.get("bad_to_good", 0),
        "bad_to_medium": rep.get("bad_to_medium", 0),
    }


@torch.no_grad()
def eval_candidate(mod: Any, name: str, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    ckpt_path = RUN_ROOT / name / "ckpt_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt["candidate_config"])
    base_train = mod.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    val_ds = mod.ARCH.SyntheticWaveDataset("val", str(cfg["channels"]))
    norm = mod.ARCH.FeatureNorm(mean=base_train.mean, std=base_train.std)
    original_ds = mod.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    model = mod.GeometryStudent(cfg, int(base_train.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"])
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    original_loader = DataLoader(original_ds, batch_size=256, shuffle=False, num_workers=0)
    val_payload = mod.eval_loader(model, val_loader, device)
    original_payload = mod.eval_loader(model, original_loader, device)
    trainval_threshold = mod.ARCH.calibrate_bad_threshold(val_ds.y, val_payload.probs)

    test_mask = original_ds.split == "test"
    y = original_ds.y[test_mask]
    probs = original_payload.probs[test_mask]
    region = original_ds.region[test_mask]
    bad_score = probs[:, mod.CLASS_TO_INT["bad"]]
    y_bad = y == mod.CLASS_TO_INT["bad"]
    bad_outlier = y_bad & (region == "outlier_low_confidence")
    bad_core = y_bad & (region != "outlier_low_confidence")
    nonbad = ~y_bad
    rows = []
    for threshold in np.round(np.linspace(0.01, 0.95, 95), 2):
        row = metric_row(mod, name, float(threshold), y, probs)
        pred = mod.ARCH.apply_bad_threshold(probs, float(threshold))
        row["bad_core_recall"] = float(np.mean(pred[bad_core] == mod.CLASS_TO_INT["bad"])) if bad_core.any() else float("nan")
        row["bad_outlier_recall"] = float(np.mean(pred[bad_outlier] == mod.CLASS_TO_INT["bad"])) if bad_outlier.any() else float("nan")
        row["nonbad_false_bad_rate"] = float(np.mean(pred[nonbad] == mod.CLASS_TO_INT["bad"])) if nonbad.any() else float("nan")
        rows.append(row)

    score_rows = []
    for label, mask in [
        ("test_bad_all", y_bad),
        ("test_bad_core", bad_core),
        ("test_bad_outlier", bad_outlier),
        ("test_nonbad", nonbad),
        ("test_good", y == mod.CLASS_TO_INT["good"]),
        ("test_medium", y == mod.CLASS_TO_INT["medium"]),
    ]:
        if not mask.any():
            continue
        values = bad_score[mask]
        score_rows.append(
            {
                "candidate": name,
                "bucket": label,
                "n": int(mask.sum()),
                "bad_score_mean": float(values.mean()),
                "bad_score_p50": float(np.quantile(values, 0.50)),
                "bad_score_p75": float(np.quantile(values, 0.75)),
                "bad_score_p90": float(np.quantile(values, 0.90)),
                "bad_score_p95": float(np.quantile(values, 0.95)),
                "bad_score_max": float(values.max()),
            }
        )
    score_rows.append(
        {
            "candidate": name,
            "bucket": "auc_bad_vs_nonbad",
            "n": int(len(y)),
            "bad_score_mean": auc_rank(y_bad, bad_score),
            "bad_score_p50": auc_rank(bad_outlier, bad_score),
            "bad_score_p75": float("nan"),
            "bad_score_p90": float("nan"),
            "bad_score_p95": float("nan"),
            "bad_score_max": float("nan"),
        }
    )
    pred_rows = pd.DataFrame(
        {
            "candidate": name,
            "orig_row": np.where(test_mask)[0],
            "true": [mod.INT_TO_CLASS[int(v)] for v in y],
            "region": region,
            "prob_good": probs[:, 0],
            "prob_medium": probs[:, 1],
            "prob_bad": probs[:, 2],
            "pred": [mod.INT_TO_CLASS[int(v)] for v in probs.argmax(axis=1)],
            "pred_trainval_badcal": [mod.INT_TO_CLASS[int(v)] for v in mod.ARCH.apply_bad_threshold(probs, trainval_threshold)],
        }
    )
    return rows, score_rows, pred_rows


def main() -> None:
    mod = load_runner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_frontier: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    for name in CANDIDATES:
        if not (RUN_ROOT / name / "ckpt_best.pt").exists():
            continue
        frontier, scores, preds = eval_candidate(mod, name, device)
        all_frontier.extend(frontier)
        all_scores.extend(scores)
        all_preds.append(preds)
    frontier_df = pd.DataFrame(all_frontier)
    scores_df = pd.DataFrame(all_scores)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    frontier_path = OUT_DIR / "waveform_bad_score_frontier.csv"
    scores_path = OUT_DIR / "waveform_bad_score_distributions.csv"
    preds_path = OUT_DIR / "waveform_bad_score_original_test_predictions.csv"
    frontier_df.to_csv(frontier_path, index=False)
    scores_df.to_csv(scores_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    for path in [frontier_path, scores_path, preds_path]:
        target = REPORT_DIR / path.name
        target.write_bytes(path.read_bytes())

    best = frontier_df.sort_values(["acc", "bad_recall"], ascending=False).groupby("candidate").head(5)
    best_bad = frontier_df.sort_values(["bad_outlier_recall", "acc"], ascending=False).groupby("candidate").head(5)
    lines = [
        "# Waveform Bad Score Frontier",
        "",
        "Report-only diagnostic. Original BUT is not used for training or checkpoint selection.",
        "",
        "## Best Overall Thresholds",
        "",
        markdown_table(best),
        "",
        "## Best Bad-Outlier Thresholds",
        "",
        markdown_table(best_bad),
        "",
        "## Bad Score Distributions",
        "",
        markdown_table(scores_df),
        "",
        f"- Frontier CSV: `{frontier_path}`",
        f"- Score distribution CSV: `{scores_path}`",
        f"- Prediction CSV: `{preds_path}`",
        "",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "waveform_bad_score_frontier_report.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "waveform_bad_score_frontier_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
