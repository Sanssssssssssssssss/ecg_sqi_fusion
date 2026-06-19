"""Analyze strict pure-cross errors for PTB<->clean-BUT waveform checkpoints.

This is read-only.  It does not train, tune thresholds, or use BUT for PTB
model selection.  It writes feature-gap tables and compact waveform panels so
the next experiment can target the actual cross-domain failure shell.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"E:\GPTProject2\ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
PTB_ALIGN_SCRIPT = ANALYSIS_DIR / "run_ptb_bad_alignment_cross_dataset.py"

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ALIGN = load_module("ptb_align_for_pure_cross_gap", PTB_ALIGN_SCRIPT)
DUAL = ALIGN.DUAL
FORMAL_FEATURE_COLUMNS = list(DUAL.FORMAL_FEATURE_COLUMNS)

KEY_COLUMNS = [
    "qrs_visibility",
    "detector_agreement",
    "qrs_band_ratio",
    "template_corr",
    "baseline_step",
    "flatline_ratio",
    "contact_loss_win_ratio",
    "non_qrs_rms_ratio",
    "non_qrs_diff_p95",
    "diff_abs_p95",
    "amplitude_entropy",
    "low_amp_ratio",
    "sqi_iSQI",
    "sqi_bSQI",
    "sqi_pSQI",
    "sqi_sSQI",
    "sqi_kSQI",
    "sqi_fSQI",
    "sqi_basSQI",
    "rms",
    "std",
    "mean_abs",
    "ptp_p99_p01",
    "band_15_30",
    "band_30_45",
    "wavelet_e4",
]
KEY_COLUMNS = [c for c in KEY_COLUMNS if c in FORMAL_FEATURE_COLUMNS]


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError(f"not an experiment checkpoint: {path}")
    return ckpt


def make_model(ckpt: dict[str, Any], in_ch: int, aux_dim: int) -> torch.nn.Module:
    cfg = dict(ckpt["candidate_config"])
    state = ckpt["model_state"]
    if any(str(k).startswith("base.") for k in state):
        model = ALIGN.StressAwareHierTransformer(cfg, in_ch, aux_dim)
    else:
        model = DUAL.HierTransformer(cfg, in_ch, aux_dim)
    model.load_state_dict(state)
    return model


def checkpoint_norm(ckpt: dict[str, Any]) -> tuple[Any, Any]:
    channel_stats = DUAL.ChannelStats(**ckpt["channel_stats"])
    feature_norm = DUAL.FeatureNorm(
        mean=np.asarray(ckpt["feature_train_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_train_std"], dtype=np.float32),
    )
    return channel_stats, feature_norm


def ptb_dataset(split: str, ckpt: dict[str, Any]):
    channel_stats, feature_norm = checkpoint_norm(ckpt)
    return ALIGN.BadAlignedSyntheticDualviewDataset(
        split,
        "none",
        0.0,
        0.0,
        0.0,
        0.0,
        channel_stats,
        feature_norm,
        False,
        "",
        0.0,
        20260619,
    )


def clean_dataset(policy: str, split: str, ckpt: dict[str, Any]):
    channel_stats, feature_norm = checkpoint_norm(ckpt)
    return DUAL.CleanButDataset(DUAL.PROTOCOL_ROOT / policy, split, channel_stats, feature_norm)


@torch.no_grad()
def predict_probs(ckpt: dict[str, Any], ds, batch_size: int, device: torch.device) -> np.ndarray:
    model = make_model(ckpt, int(ds.x.shape[1]), len(FORMAL_FEATURE_COLUMNS)).to(device)
    model.eval()
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    probs: list[np.ndarray] = []
    for batch in loader:
        out = model(batch["x"].to(device=device, dtype=torch.float32))
        probs.append(torch.softmax(out["logits"], dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def ensemble_probs(ckpts: list[dict[str, Any]], mode: str, policy: str, split: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summed: np.ndarray | None = None
    y_ref: np.ndarray | None = None
    frame_ref: pd.DataFrame | None = None
    signals_ref: np.ndarray | None = None
    for ckpt in ckpts:
        if mode == "ptb":
            ds = ptb_dataset(split, ckpt)
            frame = ds.frame.copy()
            raw_all = np.load(ALIGN.ARCH.DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
            signals = raw_all[frame["idx"].to_numpy(dtype=np.int64)]
            feat_npz = np.load(ALIGN.ARCH.DATA_DIR / "tabular_features.npz")
            features_all = feat_npz["features"].astype(np.float32)[:, DUAL.FORMAL_FEATURE_IDX]
            feature_rows = features_all[frame["idx"].to_numpy(dtype=np.int64)]
            for j, col in enumerate(FORMAL_FEATURE_COLUMNS):
                if col not in frame.columns:
                    frame[col] = feature_rows[:, j]
        else:
            ds = clean_dataset(policy, split, ckpt)
            frame = ds.frame.copy()
            npz = np.load(DUAL.PROTOCOL_ROOT / policy / "signals.npz")
            sig_key = "X" if "X" in npz.files else npz.files[0]
            raw = npz[sig_key].astype(np.float32)
            if raw.ndim == 3:
                raw = raw[:, 0, :]
            signals = raw[frame["idx"].to_numpy(dtype=np.int64)]
        probs = predict_probs(ckpt, ds, batch_size, device)
        summed = probs if summed is None else summed + probs
        if y_ref is None:
            y_ref = ds.y.copy()
            frame_ref = frame
            signals_ref = signals
        elif not np.array_equal(y_ref, ds.y):
            raise RuntimeError(f"label order mismatch for {mode} {policy} {split}")
    assert summed is not None and y_ref is not None and frame_ref is not None and signals_ref is not None
    return summed / float(len(ckpts)), y_ref, frame_ref, signals_ref


def robust_raw_features(signals: np.ndarray) -> pd.DataFrame:
    x = np.asarray(signals, dtype=np.float32)
    med = np.median(x, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    iqr = np.maximum((q75 - q25) / 1.349, 1e-6)
    z = (x - med) / iqr
    dz = np.diff(z, axis=1, prepend=z[:, :1])
    rows = {
        "raw_rms": np.sqrt(np.mean(x * x, axis=1)),
        "raw_std": np.std(x, axis=1),
        "raw_mean_abs": np.mean(np.abs(x), axis=1),
        "raw_ptp_p99_p01": np.percentile(x, 99, axis=1) - np.percentile(x, 1, axis=1),
        "raw_z_rms": np.sqrt(np.mean(z * z, axis=1)),
        "raw_dz_p95": np.percentile(np.abs(dz), 95, axis=1),
        "raw_flatline_ratio": np.mean(np.abs(dz) < 1e-3, axis=1),
        "raw_low_amp_ratio": np.mean(np.abs(z) < 0.05, axis=1),
        "raw_clip_like_ratio": np.mean(np.abs(z) > 7.5, axis=1),
    }
    return pd.DataFrame(rows)


def add_prediction_columns(frame: pd.DataFrame, y: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    pred = probs.argmax(axis=1)
    out["true"] = [INT_TO_CLASS[int(v)] for v in y]
    out["pred"] = [INT_TO_CLASS[int(v)] for v in pred]
    for i, label in INT_TO_CLASS.items():
        out[f"prob_{label}"] = probs[:, i]
    out["correct"] = pred == y
    out["error_type"] = "correct"
    out.loc[(y == 0) & (pred == 1), "error_type"] = "good_to_medium"
    out.loc[(y == 1) & (pred == 0), "error_type"] = "medium_to_good"
    out.loc[(y == 2) & (pred != 2), "error_type"] = "bad_to_nonbad"
    out.loc[(y != 2) & (pred == 2), "error_type"] = "nonbad_to_bad"
    out.loc[(y == 0) & (pred == 2), "error_type"] = "good_to_bad"
    out.loc[(y == 1) & (pred == 2), "error_type"] = "medium_to_bad"
    return out


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([a, b])))
    ca = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def feature_gap(rows: pd.DataFrame, cols: list[str], group_a: str, group_b: str, label: str) -> pd.DataFrame:
    a = rows.loc[rows["error_type"].eq(group_a)]
    b = rows.loc[rows["error_type"].eq(group_b)]
    out: list[dict[str, Any]] = []
    for col in cols:
        if col not in rows.columns:
            continue
        av = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=np.float64)
        bv = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=np.float64)
        av = av[np.isfinite(av)]
        bv = bv[np.isfinite(bv)]
        if len(av) < 3 or len(bv) < 3:
            continue
        out.append(
            {
                "comparison": label,
                "feature": col,
                "n_a": int(len(av)),
                "n_b": int(len(bv)),
                "median_a": float(np.median(av)),
                "median_b": float(np.median(bv)),
                "mean_a": float(np.mean(av)),
                "mean_b": float(np.mean(bv)),
                "delta_median_a_minus_b": float(np.median(av) - np.median(bv)),
                "ks": ks_stat(av, bv),
            }
        )
    if not out:
        return pd.DataFrame(
            columns=[
                "comparison",
                "feature",
                "n_a",
                "n_b",
                "median_a",
                "median_b",
                "mean_a",
                "mean_b",
                "delta_median_a_minus_b",
                "ks",
            ]
        )
    return pd.DataFrame(out).sort_values(["ks", "feature"], ascending=[False, True])


def plot_waveforms(rows: pd.DataFrame, signals: np.ndarray, title: str, out_path: Path, max_per_group: int = 4) -> None:
    groups = ["good_to_medium", "medium_to_good", "bad_to_nonbad", "correct"]
    available = [g for g in groups if rows["error_type"].eq(g).any()]
    if not available:
        return
    fig, axes = plt.subplots(len(available), max_per_group, figsize=(3.4 * max_per_group, 2.1 * len(available)), squeeze=False)
    t = np.arange(signals.shape[1]) / 125.0
    for r, group in enumerate(available):
        subset = rows[rows["error_type"].eq(group)].copy()
        subset["confidence"] = subset[[c for c in subset.columns if c.startswith("prob_")]].max(axis=1)
        take = subset.sort_values("confidence", ascending=False).head(max_per_group)
        for c in range(max_per_group):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(take):
                continue
            row = take.iloc[c]
            sig = signals[int(row["_row_pos"])]
            sig = sig.astype(np.float32)
            z = (sig - np.median(sig)) / max(float(np.std(sig)), 1e-6)
            ax.plot(t, z, lw=0.8, color="#2f5d8c")
            ax.set_title(f"{group}\n{row['true']}->{row['pred']} p={row[[k for k in row.index if str(k).startswith('prob_')]].max():.2f}", fontsize=8)
            ax.set_xlim(0, float(t[-1]))
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_target(name: str, mode: str, checkpoints: list[Path], policy: str, split: str, batch_size: int, out_dir: Path, report_dir: Path) -> dict[str, Any]:
    ckpts = [load_checkpoint(p) for p in checkpoints]
    probs, y, frame, signals = ensemble_probs(ckpts, mode, policy, split, batch_size)
    rows = add_prediction_columns(frame, y, probs)
    rows["_row_pos"] = np.arange(len(rows))
    raw = robust_raw_features(signals)
    for col in raw.columns:
        rows[col] = raw[col].to_numpy()

    rep = DUAL.GEOM.metric_report(y, probs.argmax(axis=1), probs)
    metric = {k: v for k, v in rep.items() if k != "confusion_3x3"}
    metric["confusion_3x3"] = json.dumps(rep["confusion_3x3"], ensure_ascii=False)
    metric.update({"name": name, "mode": mode, "policy": policy, "split": split})

    compare_specs = [
        ("good_to_medium", "correct", "good_to_medium_vs_all_correct"),
        ("medium_to_good", "correct", "medium_to_good_vs_all_correct"),
        ("good_to_medium", "medium_to_good", "gm_error_direction"),
        ("bad_to_nonbad", "correct", "bad_to_nonbad_vs_all_correct"),
    ]
    feature_cols = [c for c in KEY_COLUMNS if c in rows.columns] + [c for c in raw.columns if c in rows.columns]
    gap_tables = []
    for a, b, label in compare_specs:
        gap = feature_gap(rows, feature_cols, a, b, label)
        if not gap.empty:
            gap_tables.append(gap)
    gaps = pd.concat(gap_tables, ignore_index=True) if gap_tables else pd.DataFrame()

    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:120]
    rows_path = out_dir / f"{safe}_rows.csv"
    gap_path = out_dir / f"{safe}_feature_gaps.csv"
    metric_path = out_dir / f"{safe}_metrics.json"
    fig_path = out_dir / f"{safe}_waveform_errors.png"
    rows.to_csv(rows_path, index=False)
    gaps.to_csv(gap_path, index=False)
    metric_path.write_text(json.dumps(metric, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    plot_waveforms(rows, signals, f"{name}: pure-cross error waveforms", fig_path)

    return {
        "name": name,
        "mode": mode,
        "policy": policy,
        "split": split,
        "checkpoints": [str(p) for p in checkpoints],
        "metrics": metric,
        "rows_csv": str(rows_path),
        "feature_gaps_csv": str(gap_path),
        "waveform_png": str(fig_path),
    }


def render_report(payloads: list[dict[str, Any]]) -> str:
    lines = [
        "# Pure-Cross Error Gap Analysis",
        "",
        "Strict read-only analysis for PTB-only -> clean-BUT and clean-BUT-only -> PTB.",
        "Joint/replay and rule artifacts are intentionally excluded.",
        "",
        "| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for payload in payloads:
        m = payload["metrics"]
        lines.append(
            f"| {payload['name']} | {float(m['acc']):.6f} | {float(m['macro_f1']):.6f} | "
            f"{float(m['good_recall']):.6f} | {float(m['medium_recall']):.6f} | {float(m['bad_recall']):.6f} | "
            f"{int(m['good_to_medium'])} | {int(m['medium_to_good'])} | {int(m['bad_to_medium'])} |"
        )
    lines.extend(["", "## Outputs", ""])
    for payload in payloads:
        lines.extend(
            [
                f"### {payload['name']}",
                f"- Rows: `{payload['rows_csv']}`",
                f"- Feature gaps: `{payload['feature_gaps_csv']}`",
                f"- Waveforms: `{payload['waveform_png']}`",
                "",
            ]
        )
    lines.append("Interpretation should focus on stable feature gaps, not on one-off split luck.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ptb-to-but-checkpoint", type=Path, required=True)
    parser.add_argument("--but-to-ptb-checkpoints", type=str, required=True, help="semicolon-separated clean-BUT checkpoint paths")
    parser.add_argument("--clean-policy", type=str, default="margin_ge_5s_drop_outlier")
    parser.add_argument("--clean-split", type=str, default="test")
    parser.add_argument("--ptb-split", type=str, default="test")
    args = parser.parse_args()

    out_dir = ANALYSIS_DIR / "pure_cross_error_gap_analysis"
    report_dir = REPORT_DIR / "pure_cross_error_gap_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    payloads = [
        run_target(
            name="ptb_train_to_clean_but_test",
            mode="clean",
            checkpoints=[args.ptb_to_but_checkpoint],
            policy=args.clean_policy,
            split=args.clean_split,
            batch_size=args.batch_size,
            out_dir=out_dir,
            report_dir=report_dir,
        ),
        run_target(
            name="clean_but_train_to_ptb_test_ensemble",
            mode="ptb",
            checkpoints=[Path(p.strip()) for p in args.but_to_ptb_checkpoints.split(";") if p.strip()],
            policy=args.clean_policy,
            split=args.ptb_split,
            batch_size=args.batch_size,
            out_dir=out_dir,
            report_dir=report_dir,
        ),
    ]
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "payloads": payloads,
    }
    summary_path = out_dir / "pure_cross_error_gap_summary.json"
    report_path = out_dir / "pure_cross_error_gap_report.md"
    report_copy = report_dir / report_path.name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    report = render_report(payloads)
    report_path.write_text(report, encoding="utf-8")
    report_copy.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
