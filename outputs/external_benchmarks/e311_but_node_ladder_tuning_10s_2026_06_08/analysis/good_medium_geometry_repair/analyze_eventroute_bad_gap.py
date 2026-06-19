"""Analyze original BUT bad-stress failures for waveform event-route students.

This is external-only diagnostic code.  It does not train, select thresholds on
original BUT, or write outside the current experiment analysis/report folders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
RUNNER_PATH = ANALYSIS_DIR / "run_waveform_geometry_student.py"
NODE_ID = "N17043_gm_probe"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
EVENT_NAMES = ["flat_dropout", "baseline_jump", "impulse_curvature", "detail_burst", "asym_reset"]
HARD_AUX = [
    "qrs_visibility",
    "detector_agreement",
    "baseline_step",
    "sqi_basSQI",
    "flatline_ratio",
    "pc1",
    "pc2",
    "pc3",
    "pca_margin",
    "boundary_confidence",
    "knn_label_purity",
    "non_qrs_diff_p95",
]


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("waveform_geometry_student_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def collect_original_outputs(wgs: Any, candidate: str, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = OUT_ROOT / "runs" / "waveform_geometry_student" / NODE_ID / "search" / candidate / "ckpt_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config") or ckpt["candidate_config"]
    train_ds = wgs.ARCH.SyntheticWaveDataset("train", str(cfg["channels"]))
    norm = wgs.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    ds = wgs.ARCH.OriginalWaveDataset(norm, str(cfg["channels"]))
    atlas = pd.read_csv(wgs.ARCH.ORIGINAL_ATLAS)
    if len(atlas) != len(ds):
        raise ValueError(f"atlas rows {len(atlas)} != original dataset rows {len(ds)}")
    model = wgs.GeometryStudent(cfg, int(ds.x.shape[1])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    offset = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].cpu().numpy().astype(int)
        out = model(x, mask_ratio=0.0)
        logits = out["logits"].detach()
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1).astype(int)
        bad_threshold = float(ckpt.get("bad_threshold_trainval", ckpt.get("meta", {}).get("bad_threshold_trainval", 0.5)))
        pred_badcal = pred.copy()
        pred_badcal[probs[:, CLASS_TO_INT["bad"]] >= bad_threshold] = CLASS_TO_INT["bad"]

        def sigmoid_np(name: str) -> np.ndarray:
            value = out.get(name)
            if value is None:
                return np.full(len(y), np.nan, dtype=np.float32)
            return torch.sigmoid(value.detach()).cpu().numpy().astype(np.float32)

        bad_sig = sigmoid_np("bad_logit")
        artifact_bad_sig = sigmoid_np("artifact_bad_logit")
        artifact_veto_sig = sigmoid_np("artifact_veto_logit")
        router_sig = sigmoid_np("bad_router_logit")
        primitive_bad_sig = sigmoid_np("primitive_bad_logit")
        stress_sig = sigmoid_np("stress_logit")
        aux_pred = out["aux_pred"].detach().cpu().numpy()
        aux_target = batch["aux"].cpu().numpy()

        def event_reduce(value: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray]:
            if value is None:
                empty = np.full((len(y), len(EVENT_NAMES)), np.nan, dtype=np.float32)
                return empty, empty
            arr = value.detach().cpu().numpy()
            if arr.ndim == 4:
                reduce_axes = (1, 2)
            elif arr.ndim == 3:
                reduce_axes = (1,)
            elif arr.ndim == 2:
                return arr, arr
            else:
                empty = np.full((len(y), len(EVENT_NAMES)), np.nan, dtype=np.float32)
                return empty, empty
            return arr.max(axis=reduce_axes), arr.mean(axis=reduce_axes)

        event_logits = out.get("artifact_event_logits")
        event_targets = out.get("artifact_event_targets")
        if event_logits is not None:
            event_prob_max, event_prob_mean = event_reduce(torch.sigmoid(event_logits.detach()))
        else:
            event_prob_max = np.full((len(y), len(EVENT_NAMES)), np.nan, dtype=np.float32)
            event_prob_mean = np.full((len(y), len(EVENT_NAMES)), np.nan, dtype=np.float32)
        if event_targets is not None:
            event_target_max, event_target_mean = event_reduce(event_targets.detach())
        else:
            event_target_max = np.full((len(y), len(EVENT_NAMES)), np.nan, dtype=np.float32)
            event_target_mean = np.full((len(y), len(EVENT_NAMES)), np.nan, dtype=np.float32)

        feature_index = {name: i for i, name in enumerate(wgs.FEATURE_COLUMNS)}
        for j in range(len(y)):
            global_i = offset + j
            row = {
                "candidate": candidate,
                "row": global_i,
                "split": str(ds.split[global_i]),
                "region": str(ds.region[global_i]),
                "true": INT_TO_CLASS[int(y[j])],
                "pred_raw": INT_TO_CLASS[int(pred[j])],
                "pred_badcal": INT_TO_CLASS[int(pred_badcal[j])],
                "prob_good": float(probs[j, 0]),
                "prob_medium": float(probs[j, 1]),
                "prob_bad": float(probs[j, 2]),
                "bad_threshold_trainval": bad_threshold,
                "bad_sigmoid": float(bad_sig[j]),
                "artifact_bad_sigmoid": float(artifact_bad_sig[j]),
                "artifact_veto_sigmoid": float(artifact_veto_sig[j]),
                "bad_router_sigmoid": float(router_sig[j]),
                "primitive_bad_sigmoid": float(primitive_bad_sig[j]),
                "stress_sigmoid": float(stress_sig[j]),
            }
            for col in [
                "record_id",
                "record",
                "patient_id",
                "segment_id",
                "window_start",
                "window_end",
                "class_name",
                "original_region",
            ]:
                if col in atlas.columns:
                    row[col] = atlas.iloc[global_i][col]
            for name_i, event_name in enumerate(EVENT_NAMES):
                row[f"event_pred_max_{event_name}"] = float(event_prob_max[j, name_i])
                row[f"event_pred_mean_{event_name}"] = float(event_prob_mean[j, name_i])
                row[f"event_target_max_{event_name}"] = float(event_target_max[j, name_i])
                row[f"event_target_mean_{event_name}"] = float(event_target_mean[j, name_i])
            for feat in HARD_AUX:
                if feat in feature_index:
                    idx = feature_index[feat]
                    row[f"aux_target_z_{feat}"] = float(aux_target[j, idx])
                    row[f"aux_pred_z_{feat}"] = float(aux_pred[j, idx])
            rows.append(row)
        offset += len(y)
    return pd.DataFrame(rows), {"checkpoint": str(ckpt_path), "config": cfg}


def bucket_name(row: pd.Series) -> str:
    if row["split"] == "test" and row["true"] == "bad" and row["region"] == "outlier_low_confidence":
        return "bad_outlier_stress"
    if row["split"] == "test" and row["true"] == "bad":
        return "bad_core_nearboundary"
    if row["split"] == "test" and row["true"] == "medium" and row["pred_badcal"] == "bad":
        return "medium_false_badcal"
    if row["split"] == "test" and row["true"] == "medium" and row["pred_raw"] == "medium":
        return "medium_correct_raw"
    if row["split"] == "test" and row["true"] == "good" and row["pred_raw"] == "good":
        return "good_correct_raw"
    if row["split"] == "test":
        return "other_test"
    return "non_test"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diagnostic_bucket"] = df.apply(bucket_name, axis=1)
    numeric = [
        "prob_bad",
        "bad_sigmoid",
        "artifact_bad_sigmoid",
        "artifact_veto_sigmoid",
        "bad_router_sigmoid",
        "primitive_bad_sigmoid",
        "stress_sigmoid",
    ]
    numeric += [c for c in df.columns if c.startswith("event_pred_max_") or c.startswith("event_target_max_")]
    numeric += [c for c in df.columns if c.startswith("aux_target_z_") or c.startswith("aux_pred_z_")]
    rows = []
    for bucket, g in df.groupby("diagnostic_bucket", dropna=False):
        row: dict[str, Any] = {
            "diagnostic_bucket": bucket,
            "n": int(len(g)),
            "raw_bad_rate": float((g["pred_raw"] == "bad").mean()) if len(g) else np.nan,
            "badcal_bad_rate": float((g["pred_badcal"] == "bad").mean()) if len(g) else np.nan,
            "raw_correct_rate": float((g["pred_raw"] == g["true"]).mean()) if len(g) else np.nan,
            "badcal_correct_rate": float((g["pred_badcal"] == g["true"]).mean()) if len(g) else np.nan,
        }
        for col in numeric:
            if col in g.columns:
                row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
                row[f"{col}_p90"] = float(pd.to_numeric(g[col], errors="coerce").quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("diagnostic_bucket")


def plot_bucket_examples(df: pd.DataFrame, wgs: Any, candidate: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        out_path.with_suffix(".txt").write_text(f"matplotlib unavailable: {exc}", encoding="utf-8")
        return
    train_ds = wgs.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = wgs.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    ds = wgs.ARCH.OriginalWaveDataset(norm, "robust3")
    df = df.copy()
    df["diagnostic_bucket"] = df.apply(bucket_name, axis=1)
    buckets = [
        "bad_outlier_stress",
        "bad_core_nearboundary",
        "medium_false_badcal",
        "medium_correct_raw",
    ]
    fig, axes = plt.subplots(len(buckets), 3, figsize=(12, 8), dpi=160, squeeze=False)
    rng = np.random.default_rng(20260618)
    for r, bucket in enumerate(buckets):
        sub = df[df["diagnostic_bucket"] == bucket]
        if bucket == "bad_outlier_stress":
            sub = sub.sort_values("prob_bad", ascending=False)
        elif bucket == "medium_false_badcal":
            sub = sub.sort_values("prob_bad", ascending=False)
        elif bucket == "bad_core_nearboundary":
            sub = sub.sort_values("prob_bad", ascending=True)
        if len(sub) > 3 and bucket in {"medium_correct_raw"}:
            take = sub.iloc[rng.choice(len(sub), size=3, replace=False)]
        else:
            take = sub.head(3)
        for c, (_, item) in enumerate(take.iterrows()):
            ax = axes[r, c]
            x = ds.x[int(item["row"]), 0]
            ax.plot(x, color="#222222", lw=0.9)
            ax.set_title(
                f"{bucket}\n{item['true']}->{item['pred_raw']} bad={item['prob_bad']:.2f} route={item['bad_router_sigmoid']:.2f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        for c in range(len(take), 3):
            axes[r, c].axis("off")
    fig.suptitle(f"{candidate}: original BUT bad-gap waveform examples", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        default="featurefirst_top20_qrsbase_eventroute_recall_a050",
        help="candidate name under runs/waveform_geometry_student/N17043_gm_probe/search",
    )
    parser.add_argument("--batch-size", type=int, default=192)
    args = parser.parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    wgs = load_runner()
    df, meta = collect_original_outputs(wgs, args.candidate, args.batch_size)
    df["diagnostic_bucket"] = df.apply(bucket_name, axis=1)
    rows_path = ANALYSIS_DIR / f"{args.candidate}_bad_gap_original_rows.csv"
    summary_path = ANALYSIS_DIR / f"{args.candidate}_bad_gap_bucket_summary.csv"
    fig_path = ANALYSIS_DIR / f"{args.candidate}_bad_gap_waveforms.png"
    report_copy = REPORT_DIR / summary_path.name
    df.to_csv(rows_path, index=False)
    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(report_copy, index=False)
    plot_bucket_examples(df, wgs, args.candidate, fig_path)
    payload = {
        "candidate": args.candidate,
        "checkpoint": meta["checkpoint"],
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
        "waveform_png": str(fig_path),
        "original_used_for_training_or_selection": False,
    }
    json_path = ANALYSIS_DIR / f"{args.candidate}_bad_gap_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
