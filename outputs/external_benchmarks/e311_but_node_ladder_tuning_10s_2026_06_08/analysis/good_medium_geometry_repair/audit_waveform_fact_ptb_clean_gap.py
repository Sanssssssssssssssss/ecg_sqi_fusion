"""Audit PTB synthetic waveform-fact features against curated clean BUT.

This diagnostic intentionally compares features computed from the final PTB
noisy waveform, not manifest-aligned target geometry. It answers whether the
synthetic waveform distribution itself is aligned with the clean BUT protocol.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
PTB_SCRIPT = ANALYSIS_DIR / "run_ptb_bad_alignment_cross_dataset.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PTB = load_module(PTB_SCRIPT, "ptb_bad_alignment_for_wavefact_gap")
DUAL = PTB.DUAL
FEATURE_COLUMNS = list(PTB.FORMAL_FEATURE_COLUMNS)
CLASS_ORDER = ["good", "medium", "bad"]


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64)[np.isfinite(a)])
    b = np.sort(np.asarray(b, dtype=np.float64)[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([a, b])))
    ca = np.searchsorted(a, values, side="right") / float(len(a))
    cb = np.searchsorted(b, values, side="right") / float(len(b))
    return float(np.max(np.abs(ca - cb)))


def summarize_gap(src: pd.DataFrame, tgt: pd.DataFrame, src_name: str, tgt_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cls in CLASS_ORDER:
        s_cls = src[src["y_class"].astype(str).eq(cls)]
        t_cls = tgt[tgt["y_class"].astype(str).eq(cls)]
        if len(s_cls) < 8 or len(t_cls) < 8:
            continue
        for feat in FEATURE_COLUMNS:
            s = pd.to_numeric(s_cls[feat], errors="coerce").to_numpy(dtype=np.float64)
            t = pd.to_numeric(t_cls[feat], errors="coerce").to_numpy(dtype=np.float64)
            s_med = float(np.nanmedian(s))
            t_med = float(np.nanmedian(t))
            rows.append(
                {
                    "source": src_name,
                    "target": tgt_name,
                    "class": cls,
                    "feature": feat,
                    "ks": ks_stat(s, t),
                    "ptb_median": s_med,
                    "clean_median": t_med,
                    "median_delta_clean_minus_ptb": t_med - s_med,
                    "ptb_q10": float(np.nanpercentile(s, 10)),
                    "ptb_q90": float(np.nanpercentile(s, 90)),
                    "clean_q10": float(np.nanpercentile(t, 10)),
                    "clean_q90": float(np.nanpercentile(t, 90)),
                    "ptb_n": int(np.isfinite(s).sum()),
                    "clean_n": int(np.isfinite(t).sum()),
                }
            )
    return pd.DataFrame(rows)


def load_ptb_wavefacts(split: str | None) -> pd.DataFrame:
    labels = pd.read_csv(PTB.ARCH.DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    features = PTB.load_ptb_feature_matrix("waveform_primitives")
    feat_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    df = pd.concat([labels.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    if split is not None:
        df = df[df["split"].astype(str).eq(split)].copy()
    return df.reset_index(drop=True)


def load_clean(policy: str, split: str | None) -> pd.DataFrame:
    protocol_dir = DUAL.PROTOCOL_ROOT / policy
    atlas = pd.read_csv(DUAL.PROTOCOL_ROOT / policy / "original_region_atlas.csv")
    for col in FEATURE_COLUMNS:
        if col not in atlas.columns:
            atlas[col] = 0.0
    atlas = atlas.rename(columns={"class_name": "y_class"})
    cache_path = protocol_dir / "tabular_features_waveform_primitives.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cols = [str(x) for x in cached["feature_columns"].tolist()]
        if cols == FEATURE_COLUMNS:
            features = cached["features"].astype(np.float32)
        else:
            features = None
    else:
        features = None
    if features is None:
        npz = np.load(protocol_dir / "signals.npz")
        key = "X" if "X" in npz.files else npz.files[0]
        signals = np.asarray(npz[key], dtype=np.float32)
        features = PTB.compute_waveform_formal_features(signals)
        np.savez_compressed(
            cache_path,
            features=features,
            feature_columns=np.asarray(FEATURE_COLUMNS, dtype=object),
            source="computed_from_clean_but_signals",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
    for col in FEATURE_COLUMNS:
        atlas[col] = features[:, FEATURE_COLUMNS.index(col)]
    if split is not None:
        atlas = atlas[atlas["split"].astype(str).eq(split)].copy()
    return atlas.reset_index(drop=True)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{float(val):.4g}" if isinstance(val, (float, np.floating)) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    policy = "margin_ge_5s_drop_outlier"
    out_dir = ANALYSIS_DIR / "waveform_fact_ptb_clean_gap"
    report_dir = REPORT_DIR / "waveform_fact_ptb_clean_gap"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    ptb_train = load_ptb_wavefacts("train")
    ptb_test = load_ptb_wavefacts("test")
    clean_train = load_clean(policy, "train")
    clean_test = load_clean(policy, "test")

    gap = pd.concat(
        [
            summarize_gap(ptb_train, clean_train, "ptb_train_wavefact", f"clean_{policy}_train"),
            summarize_gap(ptb_train, clean_test, "ptb_train_wavefact", f"clean_{policy}_test"),
            summarize_gap(ptb_test, clean_test, "ptb_test_wavefact", f"clean_{policy}_test"),
        ],
        ignore_index=True,
    )
    csv_path = out_dir / "waveform_fact_ptb_clean_feature_gaps.csv"
    summary_path = out_dir / "waveform_fact_ptb_clean_gap_summary.json"
    report_path = out_dir / "waveform_fact_ptb_clean_gap_report.md"
    gap.to_csv(csv_path, index=False)

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": policy,
        "feature_source": "PTB features recomputed from final noisy waveform",
        "csv": str(csv_path),
        "report": str(report_dir / report_path.name),
        "counts": {
            "ptb_train": ptb_train["y_class"].value_counts().to_dict(),
            "ptb_test": ptb_test["y_class"].value_counts().to_dict(),
            "clean_train": clean_train["y_class"].value_counts().to_dict(),
            "clean_test": clean_test["y_class"].value_counts().to_dict(),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# PTB Waveform-Fact vs Clean BUT Feature Gap",
        "",
        "PTB features here are recomputed from `synth_10s_125hz_noisy.npz`; they are not manifest-aligned geometry targets.",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(payload["counts"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    for target in [f"clean_{policy}_train", f"clean_{policy}_test"]:
        lines.extend([f"## Top Gaps: PTB Train vs {target}", ""])
        sub = gap[(gap["source"].eq("ptb_train_wavefact")) & (gap["target"].eq(target))].copy()
        for cls in CLASS_ORDER:
            top = sub[sub["class"].eq(cls)].sort_values("ks", ascending=False).head(12)
            lines.extend([f"### {cls}", ""])
            lines.append(
                md_table(
                    top,
                    [
                        "feature",
                        "ks",
                        "ptb_median",
                        "clean_median",
                        "median_delta_clean_minus_ptb",
                        "ptb_q10",
                        "ptb_q90",
                        "clean_q10",
                        "clean_q90",
                    ],
                )
            )
            lines.append("")
    lines.extend(["", f"CSV: `{csv_path}`"])
    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    (report_dir / report_path.name).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
