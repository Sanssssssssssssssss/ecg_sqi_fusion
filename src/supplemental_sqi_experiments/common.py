from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.utils.paths import project_root as _project_root
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SQIS = ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]
SELECTED_FIVE = ["bSQI", "basSQI", "kSQI", "sSQI", "fSQI"]
GROUP_ORDER = ["original acceptable", "original unacceptable", "synthetic em", "synthetic ma"]


@dataclass(frozen=True)
class SplitArrays:
    df: pd.DataFrame
    feature_cols: list[str]
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def project_root() -> Path:
    return _project_root()


def resolve_rooted(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_table(df: pd.DataFrame, csv_path: str | Path, *, md_path: str | Path | None = None) -> None:
    csv = Path(csv_path)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    if md_path is not None:
        md = Path(md_path)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(_to_markdown_no_optional_deps(df), encoding="utf-8")


def _to_markdown_no_optional_deps(df: pd.DataFrame, *, max_rows: int = 500) -> str:
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    rows = []
    for _, r in view.iterrows():
        rows.append([_format_md_cell(r[c]) for c in view.columns])
    widths = [
        max([len(cols[i])] + [len(row[i]) for row in rows])
        for i in range(len(cols))
    ]
    header = "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in rows]
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n\nShowing first {max_rows} of {len(df)} rows.\n"
    return "\n".join([header, sep, *body]) + suffix + "\n"


def _format_md_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.6g}"
        return str(value)
    return str(value).replace("|", "\\|")


def feature_cols_for_sqis(sqis: Sequence[str], *, order: str = "lead_sqi") -> list[str]:
    unknown = [s for s in sqis if s not in SQIS]
    if unknown:
        raise ValueError(f"Unknown SQI(s): {unknown}")
    if order == "lead_sqi":
        return [f"{lead}__{sqi}" for lead in LEADS_12 for sqi in sqis]
    if order == "sqi_lead":
        return [f"{lead}__{sqi}" for sqi in sqis for lead in LEADS_12]
    raise ValueError("order must be 'lead_sqi' or 'sqi_lead'")


def sample_group(row: pd.Series) -> str:
    y = int(row["y"])
    is_aug = int(row.get("is_augmented", 0))
    noise_type = str(row.get("noise_type", "")).strip().lower()
    if y == 1 and is_aug == 0:
        return "original acceptable"
    if y == -1 and is_aug == 0:
        return "original unacceptable"
    if y == -1 and is_aug == 1 and noise_type == "em":
        return "synthetic em"
    if y == -1 and is_aug == 1 and noise_type == "ma":
        return "synthetic ma"
    return "other"


def load_split_frame(
    artifacts_dir: str | Path,
    *,
    normalized: bool = True,
    split_csv: str | Path | None = None,
) -> pd.DataFrame:
    art = Path(artifacts_dir)
    feature_name = "record84_norm.parquet" if normalized else "record84.parquet"
    feat_path = art / "features" / feature_name
    if split_csv is None:
        split_path = art / "splits" / "split_seta_seed0_paper_balanced.csv"
        if not split_path.exists():
            candidates = sorted((art / "splits").glob("*paper_balanced.csv"))
            if not candidates:
                raise FileNotFoundError(f"No paper-balanced split CSV found under {art / 'splits'}")
            split_path = candidates[0]
    else:
        split_path = Path(split_csv)

    feat = pd.read_parquet(feat_path)
    split = pd.read_csv(split_path)
    feat["record_id"] = feat["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    if "source_record_id" in split.columns:
        split["source_record_id"] = split["source_record_id"].astype(str)
    else:
        split["source_record_id"] = split["record_id"].astype(str)
    keep = [
        c
        for c in [
            "record_id",
            "split",
            "seed",
            "quality_record",
            "is_augmented",
            "source_record_id",
            "noise_type",
            "snr_db",
        ]
        if c in split.columns
    ]
    df = feat.merge(split[keep], on="record_id", how="inner", validate="one_to_one")
    if len(df) != len(feat):
        raise RuntimeError(f"Feature/split merge lost rows: features={len(feat)} merged={len(df)}")
    df["y01"] = df["y"].astype(int).eq(1).astype(np.int32)
    df["sample_group"] = df.apply(sample_group, axis=1)
    bad = sorted(df.loc[df["sample_group"].eq("other"), "record_id"].astype(str).head(10).tolist())
    if bad:
        raise RuntimeError(f"Unexpected sample groups, example record IDs: {bad}")
    return df


def split_arrays(df: pd.DataFrame, sqis: Sequence[str] | None = None) -> SplitArrays:
    cols = feature_cols_for_sqis(sqis or SQIS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:10]}")
    split = df["split"].astype(str).to_numpy()
    return SplitArrays(
        df=df,
        feature_cols=cols,
        train=split == "train",
        val=split == "val",
        test=split == "test",
    )


def binary_metrics(y01: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y01, dtype=int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    pred = (s > float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    total = max(1, tn + fp + fn + tp)
    out: dict[str, Any] = {
        "Ac": float((tn + tp) / total),
        "Se": float(tn / max(1, tn + fp)),
        "Sp": float(tp / max(1, tp + fn)),
        "acceptable_recall": float(tp / max(1, tp + fn)),
        "unacceptable_recall": float(tn / max(1, tn + fp)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(total),
    }
    if len(np.unique(y)) < 2:
        out["AUC"] = float("nan")
    else:
        try:
            out["AUC"] = float(roc_auc_score(y, s))
        except Exception:
            out["AUC"] = float("nan")
    return out


def max_accuracy_threshold(y01: np.ndarray, score: np.ndarray, *, n_grid: int = 2001) -> dict[str, Any]:
    y = np.asarray(y01, dtype=int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    best: dict[str, Any] = {"threshold": 0.5, "Ac": -1.0, "Se": -1.0}
    for threshold in np.linspace(0.0, 1.0, int(n_grid)):
        pred = (s > float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        total = max(1, tn + fp + fn + tp)
        ac = float((tn + tp) / total)
        acceptable_recall = float(tp / max(1, tp + fn))
        unacceptable_recall = float(tn / max(1, tn + fp))
        met = {
            "threshold": float(threshold),
            "Ac": ac,
            "Se": unacceptable_recall,
            "Sp": acceptable_recall,
            "acceptable_recall": acceptable_recall,
            "unacceptable_recall": unacceptable_recall,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "n": int(total),
        }
        if (
            met["Ac"] > best["Ac"]
            or (met["Ac"] == best["Ac"] and met["Se"] > best.get("Se", -np.inf))
        ):
            best = dict(met)
    if len(np.unique(y)) < 2:
        best["AUC"] = float("nan")
    else:
        best["AUC"] = float(roc_auc_score(y, s))
    return best


def threshold_curve(y01: np.ndarray, score: np.ndarray, *, n_grid: int = 501) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.0, 1.0, int(n_grid)):
        met = binary_metrics(y01, score, float(threshold))
        rows.append({"threshold": threshold, **met})
    return pd.DataFrame(rows)


def make_svm(*, C: float, gamma: float, seed: int, probability: bool = True) -> Pipeline:
    return Pipeline(
        [
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=float(C),
                    gamma=float(gamma),
                    probability=bool(probability),
                    class_weight=None,
                    random_state=int(seed),
                ),
            )
        ]
    )


def fit_fixed_svm(X: np.ndarray, y01: np.ndarray, *, C: float, gamma: float, seed: int) -> Pipeline:
    model = make_svm(C=C, gamma=gamma, seed=seed, probability=True)
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(y01, dtype=int).ravel())
    return model


def predict_score(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(X, dtype=np.float64))[:, 1].astype(np.float64)


def source_bootstrap_ci(
    df: pd.DataFrame,
    score_col: str,
    *,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    y = df["y01"].to_numpy(dtype=int)
    score = df[score_col].to_numpy(dtype=float)
    source = df["source_record_id"].astype(str).to_numpy()
    groups = np.array(sorted(pd.unique(source)))
    group_to_idx = {g: np.flatnonzero(source == g) for g in groups}
    rng = np.random.default_rng(seed)
    observed = binary_metrics(y, score, threshold)
    draws: dict[str, list[float]] = {k: [] for k in ["Ac", "Se", "Sp", "AUC"]}
    for _ in range(int(n_boot)):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in chosen])
        yy = y[idx]
        ss = score[idx]
        if len(np.unique(yy)) < 2:
            continue
        met = binary_metrics(yy, ss, threshold)
        for k in draws:
            draws[k].append(float(met[k]))
    rows = []
    for metric in draws:
        vals = np.asarray(draws[metric], dtype=float)
        if len(vals) == 0:
            lo = hi = float("nan")
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "estimate": float(observed[metric]),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_bootstrap_valid": int(len(vals)),
                "bootstrap_unit": "source_record_id",
            }
        )
    return pd.DataFrame(rows)


def subgroup_rows(df: pd.DataFrame, score_col: str, *, threshold: float, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        g = df[df["sample_group"].eq(group)]
        if g.empty:
            continue
        score = g[score_col].to_numpy(dtype=float)
        y = g["y01"].to_numpy(dtype=int)
        pred_accept = score > float(threshold)
        rows.append(
            {
                "model": model_name,
                "sample_group": group,
                "n": int(len(g)),
                "label_acceptable": int(y[0]) if len(np.unique(y)) == 1 else -1,
                "acceptance_rate": float(np.mean(pred_accept)),
                "rejection_rate": float(np.mean(~pred_accept)),
                "score_mean": float(np.mean(score)),
                "score_median": float(np.median(score)),
                "score_q10": float(np.quantile(score, 0.10)),
                "score_q90": float(np.quantile(score, 0.90)),
                "threshold": float(threshold),
            }
        )

    accept = df[df["sample_group"].eq("original acceptable")]
    for poor_group in ["original unacceptable", "synthetic em", "synthetic ma"]:
        poor = df[df["sample_group"].eq(poor_group)]
        pair = pd.concat([accept, poor], ignore_index=True)
        if len(pair) and pair["y01"].nunique() == 2:
            try:
                auc = float(roc_auc_score(pair["y01"].to_numpy(dtype=int), pair[score_col].to_numpy(dtype=float)))
            except Exception:
                auc = float("nan")
            rows.append(
                {
                    "model": model_name,
                    "sample_group": f"acceptable vs {poor_group}",
                    "n": int(len(pair)),
                    "label_acceptable": -1,
                    "acceptance_rate": float("nan"),
                    "rejection_rate": float("nan"),
                    "score_mean": float("nan"),
                    "score_median": float("nan"),
                    "score_q10": float("nan"),
                    "score_q90": float("nan"),
                    "threshold": float(threshold),
                    "AUC_pairwise": auc,
                }
            )
    return rows


def calibration_summary(y01: np.ndarray, score: np.ndarray, *, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    y = np.asarray(y01, dtype=int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    frac_pos, mean_pred = calibration_curve(y, s, n_bins=n_bins, strategy="quantile")
    brier = float(brier_score_loss(y, s))
    df = pd.DataFrame({"mean_score": mean_pred, "observed_fraction": frac_pos})
    return df, brier


def validate_integrity(df: pd.DataFrame) -> dict[str, Any]:
    y_counts = df["y"].astype(int).value_counts().to_dict()
    source_split = df[["source_record_id", "split"]].drop_duplicates().groupby("source_record_id")["split"].nunique()
    leaked_sources = source_split[source_split > 1].index.astype(str).tolist()
    feature_cols = [c for c in df.columns if "__" in c]
    finite = bool(np.isfinite(df[feature_cols].to_numpy(dtype=float)).all())
    return {
        "n_rows": int(len(df)),
        "n_sources": int(df["source_record_id"].nunique()),
        "y_counts": {str(k): int(v) for k, v in y_counts.items()},
        "sample_group_counts": {str(k): int(v) for k, v in df["sample_group"].value_counts().items()},
        "source_group_leakage_n": int(len(leaked_sources)),
        "source_group_leakage_examples": leaked_sources[:20],
        "n_feature_cols": int(len(feature_cols)),
        "feature_cols_all_finite": finite,
    }
