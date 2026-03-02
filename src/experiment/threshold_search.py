# # src/analysis/manual_threshold_eval.py
# from __future__ import annotations

# import json
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Literal, Optional

# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix

# from src.utils.paths import project_root

# LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
#             "V1", "V2", "V3", "V4", "V5", "V6"]

# SQI_LIST = ["iSQI", "bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI"]


# # =========================
# # 你只改这里：PARAMS
# # =========================
# PARAMS = {
#     # input/output
#     "in_parquet": "artifacts/relabel_stats/features/record84.parquet",
#     "out_dir": "artifacts/relabel_stats/manual_threshold_eval/run_manual",

#     # evaluation focus
#     "rank_note": "Target ~97% on set-a (note: depends on label noise).",

#     # HARD: 任意 lead 触发就直接死（record=unacc）
#     # op: "<" or ">" or "abs>" (for abs threshold)
#     "hard_rules": [
#         # basSQI very low => dead
#         {"sqi": "basSQI", "op": "<=", "thr": 0},
#         # pSQI == 0 often indicates missing/degenerate => dead (use <= eps)
#         {"sqi": "pSQI", "op": "<=", "thr": 0},
#         # fSQI high (flatline) => dead
#         {"sqi": "fSQI", "op": ">=", "thr": 0.4},
#         # kSQI huge => dead
#         {"sqi": "kSQI", "op": ">=", "thr": 900.0},
#         # sSQI extreme => dead
#         {"sqi": "sSQI", "op": "abs>=", "thr": 25.0},
#     ],

#     # SOFT: 统计“坏导联数 bad_leads”，再做 k-of-12
#     # 每条 soft rule 可以是 global（所有lead同一阈值）或 per_lead（给 dict）
#     "soft_rules": [
#         # 示例：bSQI 太小 -> lead bad（你可以改或删）
#         {"sqi": "basSQI", "op": "<=", "thr": 0.30},
#         # {"sqi": "bSQI", "op": "<=", "thr": 0.1, "scope": "global"},
#         # 示例：iSQI 太小 -> lead bad
#         {"sqi": "iSQI", "op": "<=", "thr": 0.2, "scope": "global"},
#         {"sqi": "kSQI", "op": ">=", "thr": 600.0},

#     ],

#     # k-of-12：soft 的 bad_leads 达到多少才判 unacc
#     "k_soft": 3,

#     # 输出多少个错判样本（按类型）
#     "n_show_each": 8,

#     # 可选：只评估某个 split（如果 parquet 有 split 列）
#     "split_filter": None,  # "train"/"val"/"test"/None
# }


# # =========================
# # utils
# # =========================
# def _apply_op(x: np.ndarray, op: str, thr: float) -> np.ndarray:
#     if op in ("<", "<="):
#         return x <= thr if op == "<=" else x < thr
#     if op in (">", ">="):
#         return x >= thr if op == ">=" else x > thr
#     if op == "abs>=":
#         return np.abs(x) >= thr
#     if op == "abs>":
#         return np.abs(x) > thr
#     raise ValueError(f"Bad op: {op}")


# def _conf_unacc_positive(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
#     # unacc (-1) is positive
#     yt = (y_true == -1).astype(int)
#     yp = (y_pred == -1).astype(int)
#     tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()

#     acc = (tp + tn) / max(1, (tp + tn + fp + fn))
#     se_bad = tp / max(1, (tp + fn))
#     sp_good = tn / max(1, (tn + fp))
#     bacc = 0.5 * (se_bad + sp_good)
#     return dict(acc=acc, bacc=bacc, se_bad=se_bad, sp_good=sp_good, TP=tp, FP=fp, FN=fn, TN=tn)


# def _badleads_quantiles(bad_counts: np.ndarray) -> dict:
#     qs = [0.50, 0.75, 0.90, 0.95, 1.00]
#     out = {f"q{int(q*100):02d}": float(np.quantile(bad_counts, q)) for q in qs}
#     return out


# def _to_jsonable(obj):
#     """Recursively convert numpy/pandas scalars/arrays to stdlib JSON-safe types."""
#     if isinstance(obj, dict):
#         return {str(k): _to_jsonable(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_to_jsonable(v) for v in obj]
#     if isinstance(obj, np.ndarray):
#         return [_to_jsonable(v) for v in obj.tolist()]
#     if isinstance(obj, (np.integer,)):
#         return int(obj)
#     if isinstance(obj, (np.floating,)):
#         x = float(obj)
#         return x if np.isfinite(x) else None
#     if isinstance(obj, (np.bool_,)):
#         return bool(obj)
#     if isinstance(obj, Path):
#         return str(obj)
#     return obj


# def print_sample_block(i: int, rid: str, y: int, row: dict, leads: list[str]) -> None:
#     def _fmt(v, w=7, nd=3):
#         try:
#             if v is None or (isinstance(v, float) and np.isnan(v)):
#                 return " " * (w - 1) + "-"
#             return f"{float(v):{w}.{nd}f}"
#         except Exception:
#             return " " * (w - 1) + "?"

#     header = f"sample {i:03d} {rid}: y={y}"
#     print("\n" + "=" * len(header))
#     print(header)
#     print("=" * len(header))
#     print(f"{'lead':>4} | {'iSQI':>7} {'bSQI':>7} {'pSQI':>7} {'sSQI':>7} {'kSQI':>7} {'fSQI':>7} {'bas':>7}")
#     print("-----+--------------------------------------------------------")

#     for ld in leads:
#         vals = {
#             "iSQI": row.get(f"{ld}__iSQI", np.nan),
#             "bSQI": row.get(f"{ld}__bSQI", np.nan),
#             "pSQI": row.get(f"{ld}__pSQI", np.nan),
#             "sSQI": row.get(f"{ld}__sSQI", np.nan),
#             "kSQI": row.get(f"{ld}__kSQI", np.nan),
#             "fSQI": row.get(f"{ld}__fSQI", np.nan),
#             "basSQI": row.get(f"{ld}__basSQI", np.nan),
#         }
#         print(
#             f"{ld:>4} | "
#             f"{_fmt(vals['iSQI'])} {_fmt(vals['bSQI'])} {_fmt(vals['pSQI'])} {_fmt(vals['sSQI'])} "
#             f"{_fmt(vals['kSQI'])} {_fmt(vals['fSQI'])} {_fmt(vals['basSQI'])}"
#         )


# # =========================
# # core eval
# # =========================
# def _lead_bad_from_rules(df: pd.DataFrame, rules: list[dict]) -> np.ndarray:
#     """
#     Return (N, 12) boolean lead_bad triggered by OR over provided rules.
#     Each rule is applied per-lead.
#     """
#     n = len(df)
#     lead_bad = np.zeros((n, len(LEADS_12)), dtype=bool)

#     for j, lead in enumerate(LEADS_12):
#         bad_j = np.zeros(n, dtype=bool)
#         for r in rules:
#             sqi = r["sqi"]
#             op = r["op"]
#             thr = float(r["thr"])
#             col = f"{lead}__{sqi}"
#             if col not in df.columns:
#                 continue
#             x = df[col].to_numpy(float)
#             bad_j |= (np.isfinite(x) & _apply_op(x, op, thr))
#         lead_bad[:, j] = bad_j

#     return lead_bad


# def predict(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, dict]:
#     hard_rules = params["hard_rules"]
#     soft_rules = params["soft_rules"]
#     k_soft = int(params["k_soft"])

#     hard_lead_bad = _lead_bad_from_rules(df, hard_rules)
#     soft_lead_bad = _lead_bad_from_rules(df, soft_rules)

#     hard_record_bad = hard_lead_bad.any(axis=1)              # hard: any lead triggers => dead
#     soft_bad_counts = soft_lead_bad.sum(axis=1)              # soft: count bad leads
#     soft_record_bad = soft_bad_counts >= k_soft

#     pred_unacc = hard_record_bad | soft_record_bad
#     y_pred = np.where(pred_unacc, -1, 1)

#     debug = {
#         "hard_bad_counts": hard_lead_bad.sum(axis=1),
#         "soft_bad_counts": soft_bad_counts,
#         "hard_record_bad_rate": float(hard_record_bad.mean()),
#         "soft_record_bad_rate": float(soft_record_bad.mean()),
#     }
#     return y_pred, debug


# def run(params: dict) -> None:
#     root = project_root()
#     in_path = root / params["in_parquet"]
#     out_root = root / params["out_dir"]
#     out_root.mkdir(parents=True, exist_ok=True)

#     df = pd.read_parquet(in_path)
#     df = df[df["y"].isin([1, -1])].reset_index(drop=True)

#     if params.get("split_filter") is not None and "split" in df.columns:
#         df = df[df["split"].astype(str) == str(params["split_filter"])].reset_index(drop=True)

#     y_true = df["y"].to_numpy(int)
#     n_acc = int((y_true == 1).sum())
#     n_unacc = int((y_true == -1).sum())

#     y_pred, dbg = predict(df, params)
#     m = _conf_unacc_positive(y_true, y_pred)

#     # bad lead stats
#     q_hard = _badleads_quantiles(dbg["hard_bad_counts"])
#     q_soft = _badleads_quantiles(dbg["soft_bad_counts"])

#     # save config + summary
#     (out_root / "params.json").write_text(
#         json.dumps(_to_jsonable(params), indent=2),
#         encoding="utf-8",
#     )
#     summary = {
#         "n": len(df),
#         "n_acc": n_acc,
#         "n_unacc": n_unacc,
#         "metrics": m,
#         "debug": dbg,
#         "hard_badlead_quantiles": q_hard,
#         "soft_badlead_quantiles": q_soft,
#     }
#     (out_root / "summary.json").write_text(
#         json.dumps(_to_jsonable(summary), indent=2),
#         encoding="utf-8",
#     )

#     print("\n=== MANUAL THRESHOLD EVAL ===")
#     print("data:", in_path)
#     print(f"df: {df.shape} (acc={n_acc}, unacc={n_unacc})")
#     print("k_soft:", params["k_soft"])
#     print("metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
#     print("hard bad-leads quantiles:", q_hard)
#     print("soft bad-leads quantiles:", q_soft)
#     print("hard_record_bad_rate:", round(dbg["hard_record_bad_rate"], 4),
#           "soft_record_bad_rate:", round(dbg["soft_record_bad_rate"], 4))
#     print("outputs ->", out_root)

#     # collect errors
#     fp_idx = np.where((y_true == 1) & (y_pred == -1))[0]
#     fn_idx = np.where((y_true == -1) & (y_pred == 1))[0]
#     tp_idx = np.where((y_true == -1) & (y_pred == -1))[0]
#     tn_idx = np.where((y_true == 1) & (y_pred == 1))[0]

#     def _dump_cases(name: str, idx: np.ndarray, n_show: int):
#         print(f"\n=== {name} (n={len(idx)}) ===")
#         if len(idx) == 0:
#             return
#         take = idx[: min(n_show, len(idx))]
#         for i, k in enumerate(take, start=1):
#             r = df.iloc[int(k)]
#             rid = str(r.get("record_id", "NA"))
#             y = int(r["y"])
#             print_sample_block(i=i, rid=rid, y=y, row=r.to_dict(), leads=LEADS_12)

#     n_show = int(params.get("n_show_each", 8))
#     _dump_cases("FP (false alarm): true acc, predicted unacc", fp_idx, n_show)
#     _dump_cases("FN (miss): true unacc, predicted acc", fn_idx, n_show)
#     _dump_cases("TP (hit): true unacc, predicted unacc", tp_idx, min(3, n_show))
#     _dump_cases("TN (correct ok): true acc, predicted acc", tn_idx, min(3, n_show))


# def main() -> None:
#     run(PARAMS)


# if __name__ == "__main__":
#     main()


# src/analysis/manual_threshold_eval.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.utils.paths import project_root

logger = logging.getLogger(__name__)

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"]

SQI_LIST = ["iSQI", "bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI"]


# =========================
# PARAMS (FN-first tuning + FN waveform dump)
# =========================
PARAMS = {
    "in_parquet": "artifacts/relabel_stats/features/record84.parquet",
    "out_dir": "artifacts/relabel_stats/manual_threshold_eval/run_manual",
    "rank_note": "Target ~97% on set-a (note: depends on label noise).",

    # HARD: any lead triggers => record unacc
    "hard_rules": [
        {"sqi": "basSQI", "op": "<=", "thr": 0.1},
        {"sqi": "pSQI", "op": "<=", "thr": 0},       # missing/degenerate pSQI
        {"sqi": "fSQI", "op": ">=", "thr": 0.4},     # flatline-ish
        {"sqi": "kSQI", "op": ">=", "thr": 900.0},   # crazy kurtosis
        {"sqi": "sSQI", "op": "abs>=", "thr": 25.0}, # extreme skew
    ],

    # NEW: degenerate lead (all-zero-ish) => HARD
    # This avoids using bSQI==0 as hard, but still catches "all zeros / dead lead"
    "degenerate_hard": {
        "eps": 0.1,
        "require_all": ["basSQI", "bSQI"],  # all <= eps => degenerate
    },

    # SOFT (single-rule hits per lead): used for k-of-12
    # NOTE: bSQI removed from single soft because it's too "dominant"
    "soft_rules": [
        {"sqi": "pSQI", "op": "<=", "thr": 1e-6}, 
        {"sqi": "basSQI", "op": "<=", "thr": 0.30},
        {"sqi": "iSQI", "op": "<=", "thr": 0.3, "scope": "global"},
        {"sqi": "kSQI", "op": ">=", "thr": 600.0},
        # {"sqi": "bSQI", "op": "<=", "thr": 0.1}
    ],

    # NEW: joint soft rule (needs two signals) to treat bSQI as "evidence", not a dictator
    # lead is bad if (bSQI is low) AND (iSQI is low OR basSQI is low)
    "joint_soft_rules": [
        {
            "if_all": [
                {"sqi": "bSQI", "op": "<=", "thr": 0.04},
            ],
            "and_any": [
                {"sqi": "iSQI", "op": "<=", "thr": 0.20},
            ],
        }
    ],

    # k-of-12 for soft bad leads
    "k_soft": 2,

    # NEW: strong soft (single lead multi-hit) => record unacc
    # if any lead triggers >= m soft-hits (including joint hit counts as 1) => unacc
    "strong_soft": {
        "m_hits": 2,
    },

    "n_show_each": 8,
    "split_filter": None,  # "train"/"val"/"test"/None

    # -------------------------
    # NEW: Dump FN waveforms
    # -------------------------
    "waveform": {
        "enabled": True,
        # Set this to where your cached 12-lead signals live.
        # Examples you can use:
        #   "artifacts/resample_125/signals"  (if you saved per record_id)
        #   "artifacts/preprocess/resample_125" ...
        "signal_dir": "artifacts/relabel_stats/resampled_125",

        # file format: "npy" or "npz" or "parquet"
        "fmt": "npz",

        # how to build filename from record_id
        # e.g. "{record_id}.npy" or "{record_id}.npz"
        "pattern": "{record_id}.npz",

        # sampling rate for x-axis (seconds)
        "fs": 125,

        # optional: if npz, which key to use
        "npz_key": "sig_125",
    },
}


# =========================
# helpers
# =========================
def _apply_op(x: np.ndarray, op: str, thr: float) -> np.ndarray:
    if op == "<":
        return x < thr
    if op == "<=":
        return x <= thr
    if op == ">":
        return x > thr
    if op == ">=":
        return x >= thr
    if op == "abs>":
        return np.abs(x) > thr
    if op == "abs>=":
        return np.abs(x) >= thr
    raise ValueError(f"Bad op: {op}")


def _to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _conf_unacc_positive(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = (y_true == -1).astype(int)  # unacc is positive
    yp = (y_pred == -1).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    se_bad = tp / max(1, (tp + fn))
    sp_good = tn / max(1, (tn + fp))
    bacc = 0.5 * (se_bad + sp_good)
    return dict(acc=acc, bacc=bacc, se_bad=se_bad, sp_good=sp_good, TP=tp, FP=fp, FN=fn, TN=tn)


def _badleads_quantiles(bad_counts: np.ndarray) -> dict:
    qs = [0.50, 0.75, 0.90, 0.95, 1.00]
    return {f"q{int(q*100):02d}": float(np.quantile(bad_counts, q)) for q in qs}


def _lead_mask_from_rule(df: pd.DataFrame, lead: str, rule: dict) -> np.ndarray:
    col = f"{lead}__{rule['sqi']}"
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    x = df[col].to_numpy(float)
    thr = float(rule["thr"])
    op = str(rule["op"])
    return np.isfinite(x) & _apply_op(x, op, thr)


def _lead_bad_from_rules(df: pd.DataFrame, rules: list[dict]) -> np.ndarray:
    n = len(df)
    lead_bad = np.zeros((n, len(LEADS_12)), dtype=bool)
    for j, lead in enumerate(LEADS_12):
        bad_j = np.zeros(n, dtype=bool)
        for r in rules:
            bad_j |= _lead_mask_from_rule(df, lead, r)
        lead_bad[:, j] = bad_j
    return lead_bad


def _degenerate_hard_mask(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    eps = float(cfg.get("eps", 1e-6))
    req = list(cfg.get("require_all", []))
    if not req:
        return np.zeros(len(df), dtype=bool)

    n = len(df)
    lead_deg = np.zeros((n, len(LEADS_12)), dtype=bool)
    for j, lead in enumerate(LEADS_12):
        m = np.ones(n, dtype=bool)
        for sqi in req:
            col = f"{lead}__{sqi}"
            if col not in df.columns:
                m &= False
                continue
            x = df[col].to_numpy(float)
            m &= np.isfinite(x) & (x <= eps)
        lead_deg[:, j] = m
    return lead_deg.any(axis=1) # type: ignore


def _joint_soft_lead_hits(df: pd.DataFrame, joint_rules: list[dict]) -> np.ndarray:
    """
    Return (N,12) boolean: joint rule triggers per lead (OR over joint_rules)
    """
    n = len(df)
    out = np.zeros((n, len(LEADS_12)), dtype=bool)
    if not joint_rules:
        return out

    for j, lead in enumerate(LEADS_12):
        hit = np.zeros(n, dtype=bool)
        for jr in joint_rules:
            if_all = jr.get("if_all", [])
            and_any = jr.get("and_any", [])

            m_all = np.ones(n, dtype=bool)
            for r in if_all:
                m_all &= _lead_mask_from_rule(df, lead, r)

            m_any = np.zeros(n, dtype=bool)
            for r in and_any:
                m_any |= _lead_mask_from_rule(df, lead, r)

            hit |= (m_all & m_any)
        out[:, j] = hit
    return out


def predict(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, dict]:
    hard_rules = params.get("hard_rules", [])
    soft_rules = params.get("soft_rules", [])
    joint_soft_rules = params.get("joint_soft_rules", [])
    k_soft = int(params.get("k_soft", 3))
    m_hits = int(params.get("strong_soft", {}).get("m_hits", 999999))

    hard_lead_bad = _lead_bad_from_rules(df, hard_rules)
    soft_single_bad = _lead_bad_from_rules(df, soft_rules)
    soft_joint_bad = _joint_soft_lead_hits(df, joint_soft_rules)

    # soft lead_bad for k-of-12: single OR joint
    soft_lead_bad = soft_single_bad | soft_joint_bad

    # strong-soft: count hits per lead (single hits per lead + joint hit counts as 1)
    # Here we approximate "single hits per lead" as: number of soft_single rules satisfied
    # (so strong-soft is sensitive to multiple independent abnormalities).
    n = len(df)
    single_hitcount = np.zeros((n, len(LEADS_12)), dtype=int)
    for j, lead in enumerate(LEADS_12):
        c = np.zeros(n, dtype=int)
        for r in soft_rules:
            c += _lead_mask_from_rule(df, lead, r).astype(int)
        single_hitcount[:, j] = c
    strong_hitcount = single_hitcount + soft_joint_bad.astype(int)
    strong_soft_record_bad = (strong_hitcount.max(axis=1) >= m_hits)

    deg_hard = _degenerate_hard_mask(df, params.get("degenerate_hard", {}))

    hard_record_bad = hard_lead_bad.any(axis=1) | deg_hard
    soft_bad_counts = soft_lead_bad.sum(axis=1)
    soft_record_bad = soft_bad_counts >= k_soft

    pred_unacc = hard_record_bad | soft_record_bad | strong_soft_record_bad
    y_pred = np.where(pred_unacc, -1, 1)

    debug = {
        "hard_bad_counts": hard_lead_bad.sum(axis=1),
        "soft_bad_counts": soft_bad_counts,
        "hard_record_bad_rate": float(hard_record_bad.mean()),
        "soft_record_bad_rate": float(soft_record_bad.mean()),
        "strong_soft_rate": float(strong_soft_record_bad.mean()),
        "degenerate_hard_rate": float(deg_hard.mean()),
        "strong_soft_hitcount_max_q": _badleads_quantiles(strong_hitcount.max(axis=1)),
    }
    return y_pred, debug


# =========================
# NEW: waveform dump for FN
# =========================
def _pick_npz_array(z: np.lib.npyio.NpzFile, prefer_key: str | None):
    """
    Pick waveform array from an NPZ WITHOUT touching object arrays (e.g., 'leads').

    Key idea:
    - Never iterate and read z[k] for all keys (would trigger object arrays).
    - Only read from a small whitelist of likely waveform keys.
    """
    keys = set(z.files)

    # 1) user preferred key (best)
    if prefer_key:
        if prefer_key not in keys:
            raise KeyError(f"prefer_key='{prefer_key}' not found in npz keys={sorted(keys)}")
        return prefer_key, z[prefer_key]

    # 2) whitelist of common waveform keys (extend if needed)
    # IMPORTANT: put your actual cache key first
    candidates = [
        "sig_125",   # <- your resample_125 saves this
        "x", "X",
        "sig", "signal", "ecg", "data",
        "arr_0",
        "sig_500",
    ]
    for k in candidates:
        if k in keys:
            return k, z[k]

    # 3) fail loudly (safer than probing keys and touching object arrays)
    raise KeyError(
        "Cannot find waveform array in npz. "
        f"Available keys={sorted(keys)}. "
        "Set PARAMS['waveform']['npz_key'] to the correct key (e.g., 'sig_125')."
    )


def _load_waveform(record_id: str, wf_cfg: dict):
    root = project_root()
    signal_dir = root / str(wf_cfg.get("signal_dir", ""))
    fmt = str(wf_cfg.get("fmt", "npz")).lower()
    pattern = str(wf_cfg.get("pattern", "{record_id}.npz"))
    fs = float(wf_cfg.get("fs", 125))
    prefer_key = wf_cfg.get("npz_key", None)

    path = signal_dir / pattern.format(record_id=record_id)
    if not path.exists():
        return None

    if fmt == "npz":
        # Critical: keep allow_pickle=False, but we must NOT touch object arrays.
        with np.load(path, allow_pickle=False) as z:
            k_used, x = _pick_npz_array(z, prefer_key)
            x = np.asarray(x)
    elif fmt == "npy":
        k_used, x = "npy", np.load(path)
        x = np.asarray(x)
    else:
        raise ValueError(f"Bad waveform fmt: {fmt}")

    if x.ndim != 2:
        raise ValueError(f"Waveform must be 2D, got {x.shape} at {path}")

    # normalize to (12, T)
    if x.shape[0] == 12:
        x12 = x
    elif x.shape[1] == 12:
        x12 = x.T
    else:
        raise ValueError(f"Waveform must have a 12 dimension, got {x.shape} at {path}")

    meta = {"path": str(path), "fs": fs, "npz_key_used": k_used, "shape": tuple(x12.shape)}
    return x12.astype(float), meta


def _plot_12lead(record_id: str, x12: np.ndarray, fs: float, out_png: Path, title_extra: str = "") -> None:
    T = x12.shape[1]
    t = np.arange(T) / max(fs, 1e-9)

    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for i, ld in enumerate(LEADS_12):
        ax = axes[i]
        ax.plot(t, x12[i])
        ax.set_title(ld)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"FN waveform: {record_id} {title_extra}".strip())
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run(params: dict) -> None:
    root = project_root()
    in_path = root / params["in_parquet"]
    out_root = root / params["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    df = df[df["y"].isin([1, -1])].reset_index(drop=True)

    if params.get("split_filter") is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == str(params["split_filter"])].reset_index(drop=True)

    y_true = df["y"].to_numpy(int)
    n_acc = int((y_true == 1).sum())
    n_unacc = int((y_true == -1).sum())

    y_pred, dbg = predict(df, params)
    m = _conf_unacc_positive(y_true, y_pred)

    q_hard = _badleads_quantiles(dbg["hard_bad_counts"])
    q_soft = _badleads_quantiles(dbg["soft_bad_counts"])

    (out_root / "params.json").write_text(json.dumps(_to_jsonable(params), indent=2), encoding="utf-8")
    summary = {
        "n": len(df),
        "n_acc": n_acc,
        "n_unacc": n_unacc,
        "metrics": m,
        "debug": dbg,
        "hard_badlead_quantiles": q_hard,
        "soft_badlead_quantiles": q_soft,
    }
    (out_root / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")

    print("\n=== MANUAL THRESHOLD EVAL ===")
    print("data:", in_path)
    print(f"df: {df.shape} (acc={n_acc}, unacc={n_unacc})")
    print("k_soft:", params["k_soft"])
    print("metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    print("hard bad-leads quantiles:", q_hard)
    print("soft bad-leads quantiles:", q_soft)
    print("hard_record_bad_rate:", round(dbg["hard_record_bad_rate"], 4),
          "soft_record_bad_rate:", round(dbg["soft_record_bad_rate"], 4),
          "strong_soft_rate:", round(dbg["strong_soft_rate"], 4),
          "degenerate_hard_rate:", round(dbg["degenerate_hard_rate"], 4))
    print("strong_soft_hitcount_max_q:", dbg.get("strong_soft_hitcount_max_q"))
    print("outputs ->", out_root)

    # indices
    fp_idx = np.where((y_true == 1) & (y_pred == -1))[0]
    fn_idx = np.where((y_true == -1) & (y_pred == 1))[0]
    tp_idx = np.where((y_true == -1) & (y_pred == -1))[0]
    tn_idx = np.where((y_true == 1) & (y_pred == 1))[0]

    def _fmt_row_block(i: int, rid: str, y: int, row: dict) -> None:
        def _fmt(v, w=7, nd=3):
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return " " * (w - 1) + "-"
                return f"{float(v):{w}.{nd}f}"
            except Exception:
                return " " * (w - 1) + "?"

        header = f"sample {i:03d} {rid}: y={y}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        print(f"{'lead':>4} | {'iSQI':>7} {'bSQI':>7} {'pSQI':>7} {'sSQI':>7} {'kSQI':>7} {'fSQI':>7} {'bas':>7}")
        print("-----+--------------------------------------------------------")
        for ld in LEADS_12:
            vals = {s: row.get(f"{ld}__{s}", np.nan) for s in SQI_LIST}
            print(
                f"{ld:>4} | "
                f"{_fmt(vals['iSQI'])} {_fmt(vals['bSQI'])} {_fmt(vals['pSQI'])} {_fmt(vals['sSQI'])} "
                f"{_fmt(vals['kSQI'])} {_fmt(vals['fSQI'])} {_fmt(vals['basSQI'])}"
            )

    def _dump_cases(name: str, idx: np.ndarray, n_show: int):
        print(f"\n=== {name} (n={len(idx)}) ===")
        if len(idx) == 0:
            return
        take = idx[: min(n_show, len(idx))]
        for i, k in enumerate(take, start=1):
            r = df.iloc[int(k)]
            rid = str(r.get("record_id", "NA"))
            y = int(r["y"])
            _fmt_row_block(i=i, rid=rid, y=y, row=r.to_dict())

    n_show = int(params.get("n_show_each", 8))
    _dump_cases("FP (false alarm): true acc, predicted unacc", fp_idx, n_show)
    _dump_cases("FN (miss): true unacc, predicted acc", fn_idx, n_show)
    _dump_cases("TP (hit): true unacc, predicted unacc", tp_idx, min(3, n_show))
    _dump_cases("TN (correct ok): true acc, predicted acc", tn_idx, min(3, n_show))

    # NEW: dump FN waveforms
    wf_cfg = params.get("waveform", {})
    if wf_cfg.get("enabled", False):
        fn_dir = out_root / "fn_waveforms"
        ok = 0
        miss = 0
        for k in fn_idx:
            r = df.iloc[int(k)]
            rid = str(r.get("record_id", "NA"))
            loaded = _load_waveform(rid, wf_cfg)
            if loaded is None:
                miss += 1
                continue
            x12, meta = loaded
            out_png = fn_dir / f"{rid}.png"
            title_extra = f"(y=-1 pred=+1) src={Path(meta['path']).name}"
            _plot_12lead(rid, x12, fs=float(meta["fs"]), out_png=out_png, title_extra=title_extra)
            ok += 1

        print(f"\n[FN waveform dump] saved={ok} missing_waveform={miss} -> {fn_dir}")
        if ok == 0:
            print("  -> You need to set PARAMS['waveform']['signal_dir'/'fmt'/'pattern'] to match your cached signals.")


def main() -> None:
    run(PARAMS)


if __name__ == "__main__":
    main()

