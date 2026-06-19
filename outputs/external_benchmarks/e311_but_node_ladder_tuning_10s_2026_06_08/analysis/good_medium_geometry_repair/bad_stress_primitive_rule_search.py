"""Report-only primitive search for original BUT bad-stress failures.

This is not a selection path.  It asks whether waveform-computable primitive
statistics expose a stable axis for original bad-outlier stress that the current
waveform Transformer misses.  Thresholds are selected on original non-test rows
only and evaluated on original test rows, so the output is a diagnostic for the
next tokenizer/loss design rather than a promoted rule.
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

CURRENT = "featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050"
BANK = "qrs_stress_v5"
LABELS = {0: "good", 1: "medium", 2: "bad"}


def load_student() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student_badstress_probe", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STUDENT = load_student()
ARCH = STUDENT.ARCH


def primitive_column_names(bank: str, dim: int) -> list[str]:
    per_ch = STUDENT.primitive_bank_per_channel(bank)
    if dim % per_ch != 0:
        return [f"{bank}_{i:03d}" for i in range(dim)]
    base_groups = [
        ("waveform_basic", 16),
        ("qrs_detail", 15),
        ("stress_bank_a", 24),
        ("qrs_visibility_bank", 19),
        ("detector_agreement_bank", 16),
        ("baseline_frequency_bank", 24),
        ("stress_bank_b", 24),
        ("sparse_event_bank", 32),
    ]
    names: list[str] = []
    channels = dim // per_ch
    for ch in range(channels):
        pos = 0
        for group, width in base_groups:
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
def eval_current(ds: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    ckpt_path = OUT_ROOT / "runs" / "waveform_geometry_student" / STUDENT.NODE_ID / "search" / CURRENT / "ckpt_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt["candidate_config"])
    model = STUDENT.GeometryStudent(cfg, int(ds.x.shape[1])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"{CURRENT} unexpected state keys: {unexpected[:5]}")
    if missing:
        print(f"{CURRENT}: missing state keys {len(missing)}", flush=True)
    loader = DataLoader(ds, batch_size=192, shuffle=False, num_workers=0)
    payload = STUDENT.eval_loader(model, loader, device)
    threshold = float(ckpt.get("bad_threshold_trainval", np.nan))
    pred_badcal = ARCH.apply_bad_threshold(payload.probs, threshold) if np.isfinite(threshold) else payload.pred
    return payload.probs, pred_badcal.astype(np.int64), threshold


def metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y_m = y[mask]
    p_m = pred[mask]
    out: dict[str, float] = {"n": float(len(y_m)), "acc": float((y_m == p_m).mean()) if len(y_m) else np.nan}
    for cls, name in LABELS.items():
        cls_mask = y_m == cls
        out[f"recall_{name}"] = float((p_m[cls_mask] == cls).mean()) if cls_mask.any() else np.nan
    return out


def eval_boost_rule(
    y: np.ndarray,
    base_pred: np.ndarray,
    split: np.ndarray,
    bucket_bad_stress: np.ndarray,
    feat: np.ndarray,
    prob_bad: np.ndarray,
    direction: str,
    threshold: float,
    prob_gate: float,
) -> dict[str, float]:
    cond = feat >= threshold if direction == "high" else feat <= threshold
    cond = cond & (prob_bad >= prob_gate) & (base_pred != 2)
    pred = base_pred.copy()
    pred[cond] = 2
    train = split != "test"
    test = split == "test"
    nonbad = y != 2
    bad_stress = bucket_bad_stress
    out = {
        "train_badstress_miss_caught": float((cond & train & bad_stress & (base_pred != 2)).sum()),
        "train_nonbad_added_falsebad": float((cond & train & nonbad).sum()),
        "test_badstress_miss_caught": float((cond & test & bad_stress & (base_pred != 2)).sum()),
        "test_nonbad_added_falsebad": float((cond & test & nonbad).sum()),
        "test_badstress_boost_rate": float((cond & test & bad_stress & (base_pred != 2)).sum() / max(1, (test & bad_stress & (base_pred != 2)).sum())),
        "test_nonbad_boost_rate": float((cond & test & nonbad).sum() / max(1, (test & nonbad).sum())),
    }
    out.update({f"test_{k}": v for k, v in metrics(y, pred, test).items()})
    return out


def search_boost_rules(y: np.ndarray, base_pred: np.ndarray, split: np.ndarray, bucket_bad_stress: np.ndarray, prim_z: np.ndarray, names: list[str], probs: np.ndarray) -> pd.DataFrame:
    train = split != "test"
    train_pos = train & bucket_bad_stress & (base_pred != 2)
    train_neg = train & (y != 2)
    rows: list[dict[str, Any]] = []
    prob_gates = [0.02, 0.05, 0.08, 0.10, 0.14, 0.18, 0.24, 0.30, 0.38]
    for j, name in enumerate(names):
        v = prim_z[:, j]
        if not np.isfinite(v).all():
            continue
        candidates = np.unique(np.quantile(v[train], np.linspace(0.04, 0.96, 24)))
        if len(candidates) < 2:
            continue
        for direction in ("high", "low"):
            for threshold in candidates:
                base_cond = v >= threshold if direction == "high" else v <= threshold
                pos_rate = float((base_cond & train_pos).sum() / max(1, train_pos.sum()))
                neg_rate = float((base_cond & train_neg).sum() / max(1, train_neg.sum()))
                if pos_rate < 0.04:
                    continue
                for prob_gate in prob_gates:
                    cond = base_cond & (probs[:, 2] >= prob_gate) & (base_pred != 2)
                    train_catch = float((cond & train_pos).sum())
                    train_false = float((cond & train_neg).sum())
                    score = train_catch - 2.8 * train_false
                    if train_catch <= 0:
                        continue
                    row = {
                        "feature": name,
                        "direction": direction,
                        "threshold_z": float(threshold),
                        "prob_gate": float(prob_gate),
                        "train_pos_feature_rate": pos_rate,
                        "train_nonbad_feature_rate": neg_rate,
                        "train_score": float(score),
                    }
                    row.update(eval_boost_rule(y, base_pred, split, bucket_bad_stress, v, probs[:, 2], direction, float(threshold), float(prob_gate)))
                    rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["test_acc_gain"] = df["test_acc"] - metrics(y, base_pred, split == "test")["acc"]
    df["test_bad_recall_gain"] = df["test_recall_bad"] - metrics(y, base_pred, split == "test")["recall_bad"]
    return df.sort_values(["train_score", "test_acc_gain", "test_bad_recall_gain"], ascending=False).reset_index(drop=True)


def eval_veto_rule(
    y: np.ndarray,
    base_pred: np.ndarray,
    split: np.ndarray,
    bucket_bad_stress: np.ndarray,
    feat: np.ndarray,
    direction: str,
    threshold: float,
) -> dict[str, float]:
    cond = feat >= threshold if direction == "high" else feat <= threshold
    cond = cond & (base_pred == 2)
    pred = base_pred.copy()
    pred[cond & (y != 2)] = 1
    pred[cond & (y == 2)] = 1
    test = split == "test"
    nonbad = y != 2
    out = {
        "test_nonbad_falsebad_removed": float((cond & test & nonbad).sum()),
        "test_bad_hit_lost": float((cond & test & bucket_bad_stress & (base_pred == 2)).sum()),
    }
    out.update({f"test_{k}": v for k, v in metrics(y, pred, test).items()})
    return out


def search_veto_rules(y: np.ndarray, base_pred: np.ndarray, split: np.ndarray, bucket_bad_stress: np.ndarray, prim_z: np.ndarray, names: list[str]) -> pd.DataFrame:
    train = split != "test"
    train_false = train & (y != 2) & (base_pred == 2)
    train_bad_hit = train & bucket_bad_stress & (base_pred == 2)
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        v = prim_z[:, j]
        candidates = np.unique(np.quantile(v[train], np.linspace(0.04, 0.96, 24)))
        for direction in ("high", "low"):
            for threshold in candidates:
                cond = v >= threshold if direction == "high" else v <= threshold
                false_removed = float((cond & train_false).sum())
                bad_lost = float((cond & train_bad_hit).sum())
                score = false_removed - 3.0 * bad_lost
                if false_removed <= 0:
                    continue
                row = {
                    "feature": name,
                    "direction": direction,
                    "threshold_z": float(threshold),
                    "train_score": float(score),
                    "train_false_removed": false_removed,
                    "train_bad_lost": bad_lost,
                }
                row.update(eval_veto_rule(y, base_pred, split, bucket_bad_stress, v, direction, float(threshold)))
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    base = metrics(y, base_pred, split == "test")
    df["test_acc_gain"] = df["test_acc"] - base["acc"]
    df["test_bad_recall_gain"] = df["test_recall_bad"] - base["recall_bad"]
    return df.sort_values(["train_score", "test_acc_gain"], ascending=False).reset_index(drop=True)


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ARCH.SyntheticWaveDataset("train", "robust3")
    norm = ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = ARCH.OriginalWaveDataset(norm, "robust3")
    probs, base_pred, bad_thr = eval_current(original_ds, device)
    prim = collect_primitives(original_ds, BANK)
    train_like = np.asarray(original_ds.split) != "test"
    prim_mean = np.nanmean(prim[train_like], axis=0, keepdims=True)
    prim_std = np.nanstd(prim[train_like], axis=0, keepdims=True)
    prim_std = np.where(prim_std > 1e-6, prim_std, 1.0)
    prim_z = np.nan_to_num((prim - prim_mean) / prim_std, nan=0.0, posinf=0.0, neginf=0.0)
    names = primitive_column_names(BANK, prim.shape[1])
    y = np.asarray(original_ds.y, dtype=np.int64)
    split = np.asarray(original_ds.split)
    bad_stress = ARCH.bucket_mask(original_ds, "bad_outlier_stress")
    base_test = metrics(y, base_pred, split == "test")
    base_bad_stress_rate = float(((base_pred == 2) & bad_stress & (split == "test")).sum() / max(1, (bad_stress & (split == "test")).sum()))
    boost = search_boost_rules(y, base_pred, split, bad_stress, prim_z, names, probs)
    veto = search_veto_rules(y, base_pred, split, bad_stress, prim_z, names)
    prefix = ANALYSIS_DIR / "bad_stress_primitive_rule_search_reportonly"
    boost.to_csv(prefix.with_suffix(".boost_rules.csv"), index=False)
    veto.to_csv(prefix.with_suffix(".veto_rules.csv"), index=False)
    payload = {
        "candidate": CURRENT,
        "bad_threshold_trainval": bad_thr,
        "bank": BANK,
        "baseline_test": base_test,
        "baseline_bad_outlier_stress_bad_rate": base_bad_stress_rate,
        "top_boost_rules": boost.head(20).to_dict(orient="records"),
        "top_veto_rules": veto.head(20).to_dict(orient="records"),
        "original_used_for_training_or_selection": False,
        "diagnostic_note": "Thresholds are selected on original non-test rows and evaluated on original test rows for report-only architecture diagnosis.",
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Bad-Stress Primitive Rule Search (Report Only)",
        "",
        f"- Candidate: `{CURRENT}`",
        f"- Badcal threshold: `{bad_thr:.6f}`",
        f"- Primitive bank: `{BANK}`",
        "- Original BUT is used only as a non-test/test diagnostic probe here, not for checkpoint selection.",
        "",
        "## Baseline Original Test",
        "",
        md_table(pd.DataFrame([{"bad_outlier_stress_bad_rate": base_bad_stress_rate, **base_test}]), 1),
        "",
        "## Top Boost Rules",
        "",
        md_table(
            boost.reindex(
                [
                    "feature",
                    "direction",
                    "threshold_z",
                    "prob_gate",
                    "train_score",
                    "test_badstress_miss_caught",
                    "test_nonbad_added_falsebad",
                    "test_acc",
                    "test_acc_gain",
                    "test_recall_good",
                    "test_recall_medium",
                    "test_recall_bad",
                    "test_bad_recall_gain",
                ],
                axis=1,
            ),
            20,
        ),
        "",
        "## Top Veto Rules",
        "",
        md_table(
            veto.reindex(
                [
                    "feature",
                    "direction",
                    "threshold_z",
                    "train_score",
                    "test_nonbad_falsebad_removed",
                    "test_bad_hit_lost",
                    "test_acc",
                    "test_acc_gain",
                    "test_recall_good",
                    "test_recall_medium",
                    "test_recall_bad",
                    "test_bad_recall_gain",
                ],
                axis=1,
            ),
            20,
        ),
    ]
    md = "\n".join(lines)
    prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / prefix.with_suffix(".md").name).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
