"""Compare current-best vs event-contract bad-stress transfer.

External-only diagnostic.  It loads waveform-only checkpoints, evaluates the
held-out original BUT atlas report-only, and asks which waveform primitive
families separate bad outliers that the current best catches from those the
event-contract variant loses.
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


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"


def load_student() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student_compare", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STUDENT = load_student()
ARCH = STUDENT.ARCH
LABELS = {0: "good", 1: "medium", 2: "bad"}


def primitive_column_names(bank: str, dim: int) -> list[str]:
    per_ch = STUDENT.primitive_bank_per_channel(bank)
    if dim % per_ch != 0:
        return [f"{bank}_{i:03d}" for i in range(dim)]
    groups = [
        ("atlas", 55),
        ("qrs_visibility_bank", 19),
        ("detector_agreement_bank", 16),
        ("baseline_frequency_bank", 24),
        ("stress_bank", 24),
        ("sparse_event_bank", 32),
    ]
    names: list[str] = []
    channels = dim // per_ch
    for ch in range(channels):
        pos = 0
        for group, width in groups:
            if pos >= per_ch:
                break
            use = min(width, per_ch - pos)
            names.extend(f"ch{ch}_{group}_{j:02d}" for j in range(use))
            pos += use
        while pos < per_ch:
            names.append(f"ch{ch}_extra_{pos:02d}")
            pos += 1
    return names[:dim]


@torch.no_grad()
def collect_primitives(ds: Any, bank: str, batch_size: int = 256) -> np.ndarray:
    chunks: list[np.ndarray] = []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        stats = STUDENT.primitive_waveform_stats(batch["x"].to(dtype=torch.float32), bank)
        chunks.append(stats.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def eval_candidate(name: str, ds: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    ckpt_path = OUT_ROOT / "runs" / "waveform_geometry_student" / STUDENT.NODE_ID / "search" / name / "ckpt_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt["candidate_config"])
    model = STUDENT.GeometryStudent(cfg, int(ds.x.shape[1])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"{name} unexpected state keys: {unexpected[:5]}")
    if missing:
        print(f"{name}: missing state keys {len(missing)}", flush=True)
    loader = DataLoader(ds, batch_size=192, shuffle=False, num_workers=0)
    payload = STUDENT.eval_loader(model, loader, device)
    threshold = float(ckpt.get("bad_threshold_trainval", np.nan))
    pred_badcal = ARCH.apply_bad_threshold(payload.probs, threshold) if np.isfinite(threshold) else payload.pred
    return payload.probs, pred_badcal.astype(np.int64), threshold


def summarize_sets(frame: pd.DataFrame, prim_z: np.ndarray, prim_names: list[str], sets: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    feature_rows = []
    for set_name, mask in sets.items():
        sub = frame.loc[mask]
        rows.append(
            {
                "set": set_name,
                "n": int(mask.sum()),
                "good": int((sub["true"] == "good").sum()),
                "medium": int((sub["true"] == "medium").sum()),
                "bad": int((sub["true"] == "bad").sum()),
                "current_prob_bad_mean": float(sub["current_prob_bad"].mean()) if len(sub) else np.nan,
                "event_prob_bad_mean": float(sub["event_prob_bad"].mean()) if len(sub) else np.nan,
                "current_prob_bad_p90": float(sub["current_prob_bad"].quantile(0.90)) if len(sub) else np.nan,
                "event_prob_bad_p90": float(sub["event_prob_bad"].quantile(0.90)) if len(sub) else np.nan,
            }
        )
    base = sets.get("current_hit_event_miss")
    ref = sets.get("both_miss")
    if base is not None and ref is not None and base.sum() and ref.sum():
        diff = prim_z[base].mean(axis=0) - prim_z[ref].mean(axis=0)
        order = np.argsort(-np.abs(diff))[:30]
        for idx in order:
            feature_rows.append(
                {
                    "contrast": "current_hit_event_miss_minus_both_miss",
                    "primitive": prim_names[int(idx)],
                    "mean_z_current_hit_event_miss": float(prim_z[base, idx].mean()),
                    "mean_z_both_miss": float(prim_z[ref, idx].mean()),
                    "delta_z": float(diff[idx]),
                }
            )
    hit = sets.get("both_hit")
    if hit is not None and ref is not None and hit.sum() and ref.sum():
        diff = prim_z[hit].mean(axis=0) - prim_z[ref].mean(axis=0)
        order = np.argsort(-np.abs(diff))[:30]
        for idx in order:
            feature_rows.append(
                {
                    "contrast": "both_hit_minus_both_miss",
                    "primitive": prim_names[int(idx)],
                    "mean_z_both_hit": float(prim_z[hit, idx].mean()),
                    "mean_z_both_miss": float(prim_z[ref, idx].mean()),
                    "delta_z": float(diff[idx]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows) if max_rows is not None else df
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    current = "featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050"
    event = "featurefirst_top20_qrsbase_dualcoreout_eventcontract_recall_a050"
    bank = "qrs_stress_v5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ARCH.SyntheticWaveDataset("train", "robust3")
    norm = ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = ARCH.OriginalWaveDataset(norm, "robust3")
    atlas = pd.read_csv(ARCH.ORIGINAL_ATLAS).reset_index(drop=True)
    current_probs, current_pred, current_thr = eval_candidate(current, original_ds, device)
    event_probs, event_pred, event_thr = eval_candidate(event, original_ds, device)
    prim = collect_primitives(original_ds, bank)
    prim_mean = np.nanmean(prim, axis=0, keepdims=True)
    prim_std = np.nanstd(prim, axis=0, keepdims=True)
    prim_std = np.where(prim_std > 1e-6, prim_std, 1.0)
    prim_z = np.nan_to_num((prim - prim_mean) / prim_std, nan=0.0, posinf=0.0, neginf=0.0)
    prim_names = primitive_column_names(bank, prim.shape[1])

    frame = pd.DataFrame(
        {
            "row": np.arange(len(original_ds.y)),
            "idx": atlas["idx"].to_numpy() if "idx" in atlas.columns else np.arange(len(original_ds.y)),
            "split": original_ds.split,
            "region": original_ds.region,
            "true": [LABELS[int(x)] for x in original_ds.y],
            "current_pred": [LABELS[int(x)] for x in current_pred],
            "event_pred": [LABELS[int(x)] for x in event_pred],
            "current_prob_good": current_probs[:, 0],
            "current_prob_medium": current_probs[:, 1],
            "current_prob_bad": current_probs[:, 2],
            "event_prob_good": event_probs[:, 0],
            "event_prob_medium": event_probs[:, 1],
            "event_prob_bad": event_probs[:, 2],
        }
    )
    test_bad_stress = ARCH.bucket_mask(original_ds, "bad_outlier_stress") & (original_ds.split == "test")
    test_nonbad = (original_ds.split == "test") & (original_ds.y != 2)
    sets = {
        "both_hit": test_bad_stress & (current_pred == 2) & (event_pred == 2),
        "current_hit_event_miss": test_bad_stress & (current_pred == 2) & (event_pred != 2),
        "event_hit_current_miss": test_bad_stress & (current_pred != 2) & (event_pred == 2),
        "both_miss": test_bad_stress & (current_pred != 2) & (event_pred != 2),
        "current_false_bad_nonbad": test_nonbad & (current_pred == 2),
        "event_false_bad_nonbad": test_nonbad & (event_pred == 2),
    }
    set_summary, feature_gap = summarize_sets(frame, prim_z, prim_names, sets)
    out_prefix = ANALYSIS_DIR / "eventcontract_bad_gap_compare"
    frame.loc[test_bad_stress | sets["current_false_bad_nonbad"] | sets["event_false_bad_nonbad"]].to_csv(
        out_prefix.with_suffix(".rows.csv"), index=False
    )
    set_summary.to_csv(out_prefix.with_suffix(".sets.csv"), index=False)
    feature_gap.to_csv(out_prefix.with_suffix(".primitive_gaps.csv"), index=False)
    payload = {
        "current": current,
        "event": event,
        "bank": bank,
        "current_threshold": current_thr,
        "event_threshold": event_thr,
        "sets": set_summary.to_dict(orient="records"),
        "top_gaps": feature_gap.head(20).to_dict(orient="records"),
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Event-Contract Bad Gap Compare",
        "",
        f"- Current: `{current}` badcal threshold `{current_thr:.6f}`",
        f"- Event-contract: `{event}` badcal threshold `{event_thr:.6f}`",
        f"- Primitive bank: `{bank}`",
        "",
        "## Set Counts",
        "",
        md_table(set_summary),
        "",
        "## Top Primitive Gaps",
        "",
        md_table(feature_gap, max_rows=20),
        "",
    ]
    md = "\n".join(lines)
    out_prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    (REPORT_DIR / out_prefix.with_suffix(".md").name).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
