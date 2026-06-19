"""Report-only waveform Transformer ensemble probe.

This is an external-only diagnostic.  It reads saved original-BUT report
predictions from PTB-trained waveform-only candidates and tests fixed
probability averages.  It never trains on BUT and does not use original rows for
formal selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08"
AUDIT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "original_candidate_error_audit"
OUT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair" / "waveform_ensemble_probe_v2"
LABELS = ["good", "medium", "bad"]


CANDIDATES = [
    "predtop20_sqiquery_subject111_shift_stress_pretrain",
    "featurefirst_top20_shift_stress_a050",
    "featurefirst_top20_hardrec_a050",
    "featurefirst_top20_hardrec_qrsfocus_a050",
    "featurefirst_top20_hardrec_qrslite_a050",
    "featurefirst_top20_hardrec_badlite_a050",
    "featurefirst_top20_hardrec_goodguard_badlite_a050",
    "featurefirst_top20_shift_stress_a050_seed18",
    "featurefirst_top20_shift_stress_a050_seed19",
    "featurefirst_top20_shift_stress_a050_seed20",
    "featurefirst_top20_shift_stress_a050_seed21",
    "predtop20_sqiquery_subject111_impulsebad_dual_p20",
    "predtop20_sqiquery_subject111_burstdrop_dual_p26",
    "predtop20_sqiquery_subject111_sparseevent_v5_badonly",
    "predtop20_sqiquery_qrsstressv3_stress_pretrain",
    "predtop20_sqiquery_qrsstressv3_pretrain",
    "predtop20_sqiquery_thresholdtree_badguard_pretrain",
    "predtop20_sqiquery_primtree_stress_teacher_pretrain",
]


ENSEMBLES = {
    "equal_top2_old_qrsstress": [
        "predtop20_sqiquery_subject111_shift_stress_pretrain",
        "predtop20_sqiquery_qrsstressv3_stress_pretrain",
    ],
    "equal_top2_old_featurefirst": [
        "predtop20_sqiquery_subject111_shift_stress_pretrain",
        "featurefirst_top20_shift_stress_a050",
    ],
    "equal_featurefirst_sparsebad": [
        "featurefirst_top20_shift_stress_a050",
        "predtop20_sqiquery_subject111_sparseevent_v5_badonly",
    ],
    "equal_old_featurefirst_sparsebad": [
        "predtop20_sqiquery_subject111_shift_stress_pretrain",
        "featurefirst_top20_shift_stress_a050",
        "predtop20_sqiquery_subject111_sparseevent_v5_badonly",
    ],
    "equal_hardrec_badlite": [
        "featurefirst_top20_hardrec_a050",
        "featurefirst_top20_hardrec_badlite_a050",
    ],
    "equal_hardrec_qrslite": [
        "featurefirst_top20_hardrec_a050",
        "featurefirst_top20_hardrec_qrslite_a050",
    ],
    "equal_hardrec_goodguard_badlite": [
        "featurefirst_top20_hardrec_a050",
        "featurefirst_top20_hardrec_goodguard_badlite_a050",
    ],
    "equal_hardrec_badlite_qrsfocus": [
        "featurefirst_top20_hardrec_a050",
        "featurefirst_top20_hardrec_badlite_a050",
        "featurefirst_top20_hardrec_qrsfocus_a050",
    ],
    "equal_hardrec_all_badlite": [
        "featurefirst_top20_hardrec_a050",
        "featurefirst_top20_hardrec_qrslite_a050",
        "featurefirst_top20_hardrec_badlite_a050",
        "featurefirst_top20_hardrec_goodguard_badlite_a050",
    ],
    "equal_featurefirst_a050_seeds": [
        "featurefirst_top20_shift_stress_a050",
        "featurefirst_top20_shift_stress_a050_seed18",
        "featurefirst_top20_shift_stress_a050_seed19",
        "featurefirst_top20_shift_stress_a050_seed20",
        "featurefirst_top20_shift_stress_a050_seed21",
    ],
    "equal_old_plus_featurefirst_seeds": [
        "predtop20_sqiquery_subject111_shift_stress_pretrain",
        "featurefirst_top20_shift_stress_a050",
        "featurefirst_top20_shift_stress_a050_seed18",
        "featurefirst_top20_shift_stress_a050_seed19",
        "featurefirst_top20_shift_stress_a050_seed20",
        "featurefirst_top20_shift_stress_a050_seed21",
    ],
    "equal_top3_old_p20_qrsstress": [
        "predtop20_sqiquery_subject111_shift_stress_pretrain",
        "predtop20_sqiquery_subject111_impulsebad_dual_p20",
        "predtop20_sqiquery_qrsstressv3_stress_pretrain",
    ],
    "equal_top4_old_p20_burstdrop_sparse": [
        "predtop20_sqiquery_subject111_shift_stress_pretrain",
        "predtop20_sqiquery_subject111_impulsebad_dual_p20",
        "predtop20_sqiquery_subject111_burstdrop_dual_p26",
        "predtop20_sqiquery_subject111_sparseevent_v5_badonly",
    ],
    "equal_all_structural": CANDIDATES,
    "equal_old_plus_structural": [
        "predtop20_sqiquery_subject111_shift_stress_pretrain",
        "predtop20_sqiquery_subject111_impulsebad_dual_p20",
        "predtop20_sqiquery_qrsstressv3_stress_pretrain",
        "predtop20_sqiquery_thresholdtree_badguard_pretrain",
        "predtop20_sqiquery_primtree_stress_teacher_pretrain",
    ],
}


def load_candidate(name: str) -> pd.DataFrame:
    path = AUDIT_DIR / name / "original_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    keep = ["row", "split", "region", "true", "is_test", "is_bad_core", "is_bad_outlier"]
    prob = ["prob_good", "prob_medium", "prob_bad"]
    return df[keep + prob].rename(columns={c: f"{name}__{c}" for c in prob})


def join_predictions(candidates: list[str]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for name in candidates:
        df = load_candidate(name)
        if base is None:
            base = df
        else:
            base = base.merge(df, on=["row", "split", "region", "true", "is_test", "is_bad_core", "is_bad_outlier"], how="inner")
    assert base is not None
    return base


def metrics(df: pd.DataFrame, pred: np.ndarray, name: str, bucket: str) -> dict[str, float | int | str]:
    true = df["true"].to_numpy()
    out: dict[str, float | int | str] = {
        "name": name,
        "bucket": bucket,
        "n": int(len(df)),
        "acc": float(np.mean(pred == true)) if len(df) else float("nan"),
    }
    for label in LABELS:
        m = true == label
        out[f"{label}_recall"] = float(np.mean(pred[m] == label)) if np.any(m) else float("nan")
    out["good_to_medium"] = int(np.sum((true == "good") & (pred == "medium")))
    out["medium_to_good"] = int(np.sum((true == "medium") & (pred == "good")))
    out["bad_to_good"] = int(np.sum((true == "bad") & (pred == "good")))
    out["bad_to_medium"] = int(np.sum((true == "bad") & (pred == "medium")))
    return out


def ensemble_probs(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    mats = []
    for name in candidates:
        mats.append(df[[f"{name}__prob_good", f"{name}__prob_medium", f"{name}__prob_bad"]].to_numpy(dtype=float))
    return np.mean(np.stack(mats, axis=0), axis=0)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render_report(rows: pd.DataFrame) -> str:
    test_rows = rows[rows["bucket"].eq("original_test_all_10s+")].sort_values("acc", ascending=False)
    bad_rows = rows[rows["bucket"].isin(["bad_core_nearboundary", "bad_outlier_stress"])].sort_values(
        ["bucket", "acc"], ascending=[True, False]
    )
    lines = [
        "# Waveform Ensemble Probe V2",
        "",
        "Report-only diagnostic.  All members are PTB-trained waveform-only Transformer candidates.",
        "The ensembles below are fixed probability averages; original BUT is not used for training.",
        "",
        "## Original Test 10s+",
        "",
        markdown_table(test_rows),
        "",
        "## Bad Buckets",
        "",
        markdown_table(bad_rows),
        "",
        "## Interpretation",
        "",
        "- If fixed averaging beats every member, the candidates contain complementary waveform evidence.",
        "- If it does not, the current waveform-only family is representation/coverage limited rather than merely checkpoint-limited.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = join_predictions(CANDIDATES)
    rows = []
    buckets = {
        "original_test_all_10s+": df["is_test"].astype(bool),
        "bad_core_nearboundary": df["is_test"].astype(bool) & df["is_bad_core"].astype(bool),
        "bad_outlier_stress": df["is_test"].astype(bool) & df["is_bad_outlier"].astype(bool),
    }
    for name in CANDIDATES:
        pred = np.array(LABELS, dtype=object)[df[[f"{name}__prob_good", f"{name}__prob_medium", f"{name}__prob_bad"]].to_numpy(float).argmax(axis=1)]
        for bucket, mask in buckets.items():
            rows.append(metrics(df[mask].copy(), pred[mask.to_numpy()], name, bucket))
    for ens_name, members in ENSEMBLES.items():
        probs = ensemble_probs(df, members)
        pred = np.array(LABELS, dtype=object)[probs.argmax(axis=1)]
        for bucket, mask in buckets.items():
            rows.append(metrics(df[mask].copy(), pred[mask.to_numpy()], ens_name, bucket))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "waveform_ensemble_probe_v2_metrics.csv", index=False)
    (OUT_DIR / "waveform_ensemble_probe_v2_summary.json").write_text(
        json.dumps({"candidates": CANDIDATES, "ensembles": ENSEMBLES}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "waveform_ensemble_probe_v2_report.md").write_text(render_report(out), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
