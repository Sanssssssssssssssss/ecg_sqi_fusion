#!/usr/bin/env python
"""Dual-view waveform separability audit for generated-vs-original rows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent


def import_local(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DUAL = import_local("dualview_for_auc", HERE / "run_clean_but_dualview_hier_transformer.py")
REPORT_ROOT = HERE.parents[4] / "reports" / "external_benchmarks" / "e311_but_node_ladder_tuning_10s_2026_06_08" / "analysis" / "good_medium_geometry_repair" / "dual_generated_auc"


def lp(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def protocol_path(policy: str) -> Path:
    path = Path(policy)
    if path.exists():
        return path
    return DUAL.PROTOCOL_ROOT / policy


def generated_label(frame: pd.DataFrame) -> np.ndarray:
    ctype = frame.get("v116_candidate_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    generated = pd.to_numeric(frame.get("v116_generated", pd.Series(0, index=frame.index)), errors="coerce").fillna(0).to_numpy() > 0
    original = ctype.eq("original_but").to_numpy()
    generated = generated | ctype.isin(["but_native_morph", "ptb_morph", "clean_style"]).to_numpy()
    return np.where(original, 0, np.where(generated, 1, -1)).astype(int)


def dual_summary(signals: np.ndarray) -> np.ndarray:
    stats = DUAL.compute_channel_stats(signals)
    dual = DUAL.make_dualview_channels(signals, stats).astype(np.float32)
    qs = np.percentile(dual, [5, 25, 50, 75, 95], axis=2).transpose(1, 2, 0).reshape(len(dual), -1)
    mean = dual.mean(axis=2)
    std = dual.std(axis=2)
    abs_mean = np.abs(dual).mean(axis=2)
    diff_std = np.diff(dual, axis=2).std(axis=2)
    return np.hstack([mean, std, abs_mean, diff_std, qs]).astype(np.float32)


def take_balanced(rows: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_per_label: int) -> tuple[np.ndarray, np.ndarray]:
    chosen: list[int] = []
    for label in [0, 1]:
        idx = np.flatnonzero(y == label)
        n = min(len(idx), int(max_per_label))
        if n:
            chosen.extend(rng.choice(idx, size=n, replace=False).tolist())
    chosen_arr = np.asarray(chosen, dtype=int)
    rng.shuffle(chosen_arr)
    return rows[chosen_arr], y[chosen_arr]


def score_scope(
    *,
    scope: str,
    frame: pd.DataFrame,
    signals: np.ndarray,
    rng: np.random.Generator,
    max_per_label: int,
) -> dict[str, Any]:
    y_all = generated_label(frame)
    valid = y_all >= 0
    sub = frame.loc[valid].copy()
    y = y_all[valid]
    rows = sub["idx"].astype(int).to_numpy()
    rows, y = take_balanced(rows, y, rng, int(max_per_label))
    if len(np.unique(y)) < 2 or min(np.bincount(y, minlength=2)) < 6:
        return {"scope": scope, "status": "skip", "rows": int(len(y)), "original_n": int((y == 0).sum()), "generated_n": int((y == 1).sum())}
    x = dual_summary(signals[rows])
    tr, te = train_test_split(np.arange(len(y)), test_size=0.35, random_state=int(rng.integers(1, 2**31 - 1)), stratify=y)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="lbfgs"),
    )
    clf.fit(x[tr], y[tr])
    prob = clf.predict_proba(x[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y[te], prob))
    sym_auc = float(max(auc, 1.0 - auc))
    return {
        "scope": scope,
        "status": "ok",
        "rows": int(len(y)),
        "original_n": int((y == 0).sum()),
        "generated_n": int((y == 1).sum()),
        "auc": auc,
        "sym_auc": sym_auc,
        "acc": float(accuracy_score(y[te], pred)),
        "ideal_pass": bool(sym_auc <= 0.55) if scope == "pooled_medium_bad_class_balanced" else bool(sym_auc <= 0.58),
    }


def pooled_medium_bad(frame: pd.DataFrame, rng: np.random.Generator, max_per_cell: int) -> pd.DataFrame:
    labels = generated_label(frame)
    tmp = frame.copy()
    tmp["_generated_label"] = labels
    parts: list[pd.DataFrame] = []
    for cls in ["medium", "bad"]:
        for label in [0, 1]:
            cell = tmp.loc[tmp["class_name"].astype(str).eq(cls) & tmp["_generated_label"].eq(label)].copy()
            if len(cell) == 0:
                continue
            n = min(len(cell), int(max_per_cell))
            parts.append(cell.iloc[rng.choice(np.arange(len(cell)), size=n, replace=False)].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, default=20260876)
    parser.add_argument("--max-per-label", type=int, default=5000)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    source = protocol_path(str(args.policy))
    frame = pd.read_csv(source / "original_region_atlas.csv")
    signals = np.load(source / "signals.npz")["X"].astype(np.float32)
    if signals.ndim == 3:
        signals = signals[:, 0, :]

    rows: list[dict[str, Any]] = []
    for cls in ["medium", "bad"]:
        sub = frame.loc[frame["class_name"].astype(str).eq(cls)].copy()
        rows.append(score_scope(scope=cls, frame=sub, signals=signals, rng=rng, max_per_label=int(args.max_per_label)))
    pooled = pooled_medium_bad(frame, rng, max(1, int(args.max_per_label) // 2))
    rows.append(score_scope(scope="pooled_medium_bad_class_balanced", frame=pooled, signals=signals, rng=rng, max_per_label=int(args.max_per_label)))

    out = pd.DataFrame(rows)
    out_dir = Path(args.out_dir) if args.out_dir else REPORT_ROOT / source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(lp(out_dir / "dual_generated_auc.csv"), index=False)
    print(json.dumps({"protocol": str(source), "report": str(out_dir), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
