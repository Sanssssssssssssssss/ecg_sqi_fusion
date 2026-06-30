from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.utils.paths import project_root


ROOT = project_root()
OUT_DEFAULT = ROOT / "outputs" / "transformer" / "supplemental" / "chapter4_evidence"
SEED = 0
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SYNTH_QUOTA = {"ptb12_morph": 211, "seta_native_morph": 153, "noise_style": 19}


@dataclass(frozen=True)
class Paths:
    out: Path

    @property
    def seta(self) -> Path:
        return self.out / "seta_gapfill"

    @property
    def seta_arms(self) -> Path:
        return self.out / "seta_arms"

    @property
    def seta_models(self) -> Path:
        return self.out / "seta_models"

    @property
    def but(self) -> Path:
        return self.out / "but_v116"

    @property
    def tables(self) -> Path:
        return self.out / "tables"

    @property
    def reports(self) -> Path:
        return self.out / "reports"

    @property
    def figures(self) -> Path:
        return self.out / "figures"

    @property
    def source_data(self) -> Path:
        return self.figures / "source_data"

    @property
    def audit_json(self) -> Path:
        return self.reports / "protocol_audit.json"

    @property
    def repair_json(self) -> Path:
        return self.reports / "seta_repair_audit.json"

    @property
    def seta_models_json(self) -> Path:
        return self.reports / "seta_model_summary.json"

    @property
    def but_models_json(self) -> Path:
        return self.reports / "but_model_summary.json"

    @property
    def report_md(self) -> Path:
        return self.reports / "chapter4_raw_results_report.md"


def ensure_dirs(paths: Paths) -> None:
    for path in [
        paths.out,
        paths.seta,
        paths.seta_arms,
        paths.seta_models,
        paths.but,
        paths.tables,
        paths.reports,
        paths.figures,
        paths.source_data,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_date() -> str:
    return datetime.now().isoformat(timespec="seconds")


def dry(step: str, paths: Paths) -> None:
    print(
        json.dumps(
            {
                "step": step,
                "out": str(paths.out),
                "seta": str(paths.seta),
                "seta_arms": str(paths.seta_arms),
                "but": str(paths.but),
                "reports": str(paths.reports),
                "figures": str(paths.figures),
            },
            indent=2,
        )
    )


def stable_record_ids(n: int) -> list[str]:
    return [f"r{i:05d}" for i in range(int(n))]


def feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {
        "row_idx",
        "record_id",
        "source_record_id",
        "split",
        "quality_record",
        "label",
        "y",
        "generated",
        "candidate_type",
        "donor_split",
        "ptb_ecg_id",
        "ptb_patient_id",
        "style_anchor_record_id",
        "pool_index",
    }
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def binary_metrics(y01: np.ndarray, score: np.ndarray | None = None, pred: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y01, dtype=int).ravel()
    if pred is None:
        if score is None:
            raise ValueError("score or pred required")
        pred = (np.asarray(score, dtype=np.float64).ravel() > 0.5).astype(int)
    pred = np.asarray(pred, dtype=int).ravel()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "acceptable_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "original_unacceptable_recall": float(tn / max(1, tn + fp)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if score is not None:
        try:
            out["auc"] = float(roc_auc_score(y, np.asarray(score, dtype=np.float64).ravel()))
        except Exception:
            out["auc"] = float("nan")
    return out


def multiclass_summary(y: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict[str, Any]:
    yy = np.asarray(y, dtype=int)
    pred = np.asarray(probs).argmax(axis=1)
    rec = recall_score(yy, pred, labels=list(range(len(labels))), average=None, zero_division=0)
    cm = confusion_matrix(yy, pred, labels=list(range(len(labels))))
    out = {
        "acc": float(accuracy_score(yy, pred)),
        "macro_f1": float(f1_score(yy, pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(yy, pred)),
        "confusion": cm.astype(int).tolist(),
    }
    for name, value in zip(labels, rec):
        out[f"{name}_recall"] = float(value)
    return out


def table_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for c in work.columns:
        if pd.api.types.is_float_dtype(work[c]):
            work[c] = work[c].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    cols = [str(c) for c in work.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in work.columns) + " |")
    return "\n".join(lines)
