# src/experiment/baseline_sqi.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# =========================
# Config
# =========================
@dataclass(frozen=True)
class BaselineCfg:
    in_parquet: Path
    out_dir: Path

    # waveform cache
    signal_dir: Path
    fmt: str = "npz"               # "npz" | "npy" (npz recommended)
    pattern: str = "{record_id}.npz"
    npz_key: str = "sig_125"
    fs: float = 125.0

    # signal preproc
    robust_norm: bool = True
    mad_eps: float = 1e-6

    # baseline estimation
    baseline_win_s: float = 2.0     # moving average window (seconds)
    eps: float = 1e-8

    # printing / ranking
    n_show_each: int = 10

    # columns
    record_id_col: str = "record_id"
    y_col: str = "y"                # expect +1 good, -1 bad
    pred_col: Optional[str] = "pred" # if present, we will print FN

    # optional thresholds -> for hit_count (you can tune later)
    thr: Dict[str, float] = None     # type: ignore # e.g. {"BR_max": 0.8, "JS_max": 5.0}


# =========================
# IO
# =========================
def _read_parquet(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing parquet: {p}")
    df = pd.read_parquet(p)
    if len(df) == 0:
        raise ValueError(f"Empty parquet: {p}")
    return df


def _load_signal(cfg: BaselineCfg, record_id: Any) -> np.ndarray:
    fp = cfg.signal_dir / cfg.pattern.format(record_id=record_id)
    if not fp.exists():
        raise FileNotFoundError(f"Missing signal file: {fp}")

    if cfg.fmt.lower() == "npz":
        z = np.load(fp, allow_pickle=False)
        if cfg.npz_key not in z:
            raise KeyError(f"npz_key={cfg.npz_key!r} not in {fp.name}. keys={list(z.keys())}")
        sig = z[cfg.npz_key]
    elif cfg.fmt.lower() == "npy":
        sig = np.load(fp, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported fmt={cfg.fmt!r}. Use 'npz' or 'npy'.")

    sig = np.asarray(sig)

    # Accept [12, T] or [T, 12]
    if sig.ndim != 2:
        raise ValueError(f"Signal must be 2D, got shape={sig.shape} in {fp.name}")
    if sig.shape[0] == 12:
        return sig.astype(np.float64, copy=False)
    if sig.shape[1] == 12:
        return sig.T.astype(np.float64, copy=False)

    raise ValueError(f"Signal must have a 12-lead axis. got shape={sig.shape} in {fp.name}")


# =========================
# Core metrics
# =========================
def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _robust_norm(x: np.ndarray, eps: float) -> np.ndarray:
    med = np.median(x)
    mad = _mad(x)
    return (x - med) / (mad + eps)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    # centered moving average; for small win we can do reflect padding
    if win <= 1:
        return x.copy()
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    k = np.ones(win, dtype=np.float64) / float(win)
    y = np.convolve(xpad, k, mode="valid")
    # y length equals x length if padding is correct
    if len(y) != len(x):
        y = y[: len(x)]
    return y


def compute_lead_metrics(
    x: np.ndarray,
    fs: float,
    baseline_win_s: float,
    robust_norm: bool,
    mad_eps: float,
    eps: float,
) -> Dict[str, float]:
    """
    x: 1D lead signal (T,)
    returns: BR, BRng, BSE, JS
    """
    x = np.asarray(x, dtype=np.float64)
    if robust_norm:
        x = _robust_norm(x, mad_eps)

    win = max(1, int(round(baseline_win_s * fs)))
    b = _moving_average(x, win=win)
    r = x - b

    # Baseline Ratio (slow drift relative to residual)
    br = float(np.sqrt(np.mean(b * b)) / (np.sqrt(np.mean(r * r)) + eps))

    # Baseline range (robust amplitude of baseline)
    brng = float(np.quantile(b, 0.95) - np.quantile(b, 0.05))

    # Baseline slope energy
    db = np.diff(b)
    bse = float(np.sqrt(np.mean(db * db)) + 0.0)

    # Jump score: extreme instantaneous change in original signal
    dx = np.diff(x)
    js = float(np.quantile(np.abs(dx), 0.999))  # strong pop detector

    return {"BR": br, "BRng": brng, "BSE": bse, "JS": js}


def _topk_mean(a: np.ndarray, k: int) -> float:
    a = np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return float("nan")
    k = max(1, min(k, len(a)))
    idx = np.argsort(a)[-k:]
    return float(np.mean(a[idx]))


def compute_record_metrics(cfg: BaselineCfg, sig12: np.ndarray) -> Tuple[pd.Series, Dict[str, Dict[str, float]]]:
    """
    sig12: [12, T]
    returns:
      - record-level series (columns like BR_max, BR_top3_mean, ...)
      - per-lead dict: metric -> {lead_name: value}
    """
    per_lead: Dict[str, List[float]] = {"BR": [], "BRng": [], "BSE": [], "JS": []}
    per_lead_named: Dict[str, Dict[str, float]] = {k: {} for k in per_lead.keys()}

    for i, lead in enumerate(LEADS_12):
        m = compute_lead_metrics(
            sig12[i],
            fs=cfg.fs,
            baseline_win_s=cfg.baseline_win_s,
            robust_norm=cfg.robust_norm,
            mad_eps=cfg.mad_eps,
            eps=cfg.eps,
        )
        for k, v in m.items():
            per_lead[k].append(v)
            per_lead_named[k][lead] = v

    out: Dict[str, float] = {}
    for k, arr in per_lead.items():
        a = np.asarray(arr, dtype=np.float64)
        out[f"{k}_max"] = float(np.max(a))
        out[f"{k}_top3_mean"] = _topk_mean(a, 3)

    # Optional hit-counts if you provide thresholds
    if cfg.thr:
        for name, thr in cfg.thr.items():
            # name could be "BR_max" (record-level) OR "BR" (per-lead)
            if name in out:
                out[f"{name}_hit"] = float(out[name] >= thr)
            elif name in per_lead:
                a = np.asarray(per_lead[name], dtype=np.float64)
                out[f"{name}_hitcount"] = float(np.sum(a >= thr))
            else:
                # ignore unknown keys silently (so you can iterate fast)
                pass

    return pd.Series(out), per_lead_named


# =========================
# Ranking + printing
# =========================
def _worst_leads(per_lead_named: Dict[str, Dict[str, float]], key: str, topk: int = 3) -> List[Tuple[str, float]]:
    d = per_lead_named.get(key, {})
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    return items[:topk]


def _print_block(title: str, df: pd.DataFrame, cols: List[str], n: int) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    show = df.head(n)
    if len(show) == 0:
        print("(empty)")
        return
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(show[cols].to_string(index=False))


def run(cfg: BaselineCfg) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_parquet(cfg.in_parquet)

    need_cols = [cfg.record_id_col, cfg.y_col]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column {c!r} in {cfg.in_parquet.name}. columns={list(df.columns)[:20]}...")

    has_pred = cfg.pred_col is not None and cfg.pred_col in df.columns

    rows = []
    per_lead_cache: Dict[Any, Dict[str, Dict[str, float]]] = {}

    # iterate records
    for rid, y in zip(df[cfg.record_id_col].tolist(), df[cfg.y_col].tolist()):
        try:
            sig12 = _load_signal(cfg, rid)
        except Exception as e:
            # keep going but record error
            rows.append({cfg.record_id_col: rid, cfg.y_col: y, "err": str(e)})
            continue

        rec_metrics, per_lead_named = compute_record_metrics(cfg, sig12)
        per_lead_cache[rid] = per_lead_named

        row = {cfg.record_id_col: rid, cfg.y_col: y, "err": ""}
        if has_pred:
            row[cfg.pred_col] = df.loc[df[cfg.record_id_col] == rid, cfg.pred_col].iloc[0]
        row.update(rec_metrics.to_dict())
        rows.append(row)

    out = pd.DataFrame(rows)

    # save raw metrics
    out_csv = cfg.out_dir / "baseline_metrics.csv"
    out.to_csv(out_csv, index=False)

    # save cfg for audit
    cfg_json = cfg.out_dir / "baseline_cfg.json"
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}, f, indent=2)

    # Prepare ranking columns
    metric_cols = [c for c in out.columns if c.endswith("_max") or c.endswith("_top3_mean")]
    metric_cols = [c for c in metric_cols if c.split("_")[0] in {"BR", "BRng", "BSE", "JS"}]

    # choose a primary score for sorting (you can change quickly)
    # I recommend BRng_max + JS_max as "baseline issues + pops"
    if "BRng_max" in out.columns and "JS_max" in out.columns:
        out["score_primary"] = out["BRng_max"].fillna(0) + out["JS_max"].fillna(0)
    elif "BR_max" in out.columns:
        out["score_primary"] = out["BR_max"].fillna(0)
    else:
        out["score_primary"] = 0.0

    # Groups
    good = out[(out[cfg.y_col] == 1) & (out["err"] == "")].copy()
    bad = out[(out[cfg.y_col] == -1) & (out["err"] == "")].copy()

    # Print columns
    base_cols = [cfg.record_id_col, cfg.y_col]
    if has_pred:
        base_cols += [cfg.pred_col]
    cols_show = base_cols + ["score_primary"] + sorted(metric_cols)

    # 1) good best (lowest score)
    _print_block(
        title=f"GOOD (y=+1) best baseline (lowest score_primary)  n={len(good)}",
        df=good.sort_values("score_primary", ascending=True),
        cols=cols_show,
        n=cfg.n_show_each,
    )

    # 2) good worst (highest score) -> possible label noise or hidden issues
    _print_block(
        title=f"GOOD (y=+1) worst baseline (highest score_primary) n={len(good)}",
        df=good.sort_values("score_primary", ascending=False),
        cols=cols_show,
        n=cfg.n_show_each,
    )

    # 3) bad worst (highest score) -> should look obviously bad
    _print_block(
        title=f"BAD (y=-1) worst baseline (highest score_primary)  n={len(bad)}",
        df=bad.sort_values("score_primary", ascending=False),
        cols=cols_show,
        n=cfg.n_show_each,
    )

    # 4) FN worst if pred exists: y=-1 pred=+1
    if has_pred:
        fn = out[(out[cfg.y_col] == -1) & (out[cfg.pred_col] == 1) & (out["err"] == "")].copy()
        _print_block(
            title=f"FN (y=-1 pred=+1) worst baseline (highest score_primary) n={len(fn)}",
            df=fn.sort_values("score_primary", ascending=False),
            cols=cols_show,
            n=cfg.n_show_each,
        )

    # Also: print top-3 worst leads for a few worst FN/bad
    def _print_worst_leads_for(df_ranked: pd.DataFrame, title: str, n: int = 5) -> None:
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)
        for rid in df_ranked.head(n)[cfg.record_id_col].tolist():
            per = per_lead_cache.get(rid)
            if per is None:
                continue
            w_brng = _worst_leads(per, "BRng", topk=3)
            w_js = _worst_leads(per, "JS", topk=3)
            print(f"record_id={rid}  worst_BRng={w_brng}  worst_JS={w_js}")

    _print_worst_leads_for(bad.sort_values("score_primary", ascending=False), "Worst BAD: top leads (BRng, JS)", n=5)
    if has_pred:
        fn = out[(out[cfg.y_col] == -1) & (out[cfg.pred_col] == 1) & (out["err"] == "")].copy()
        _print_worst_leads_for(fn.sort_values("score_primary", ascending=False), "Worst FN: top leads (BRng, JS)", n=10)

    print("\nSaved ->", out_csv)
    print("Saved ->", cfg_json)


# =========================
# CLI example
# =========================
if __name__ == "__main__":
    # You can hardcode exactly your PARAMS here for now (fast iteration)
    cfg = BaselineCfg(
        in_parquet=Path(r"artifacts/relabel_stats/features/record84.parquet"),
        out_dir=Path(r"artifacts/relabel_stats/baseline_eval/run0"),
        signal_dir=Path(r"artifacts/relabel_stats/resampled_125"),
        fmt="npz",
        pattern="{record_id}.npz",
        npz_key="sig_125",
        fs=125.0,
        baseline_win_s=2.0,
        n_show_each=10,
        # if you later want hitcount:
        # thr={"BRng_max": 0.8, "JS_max": 6.0, "BR": 0.7}
        thr=None,
        pred_col="pred",  # if not present in parquet, it will auto-disable FN printing
    )
    run(cfg)