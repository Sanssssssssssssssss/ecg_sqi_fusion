"""Build a PTB bad bank matched to BUT bad waveform distribution.

This is an external-only distribution diagnostic/generator.  It does not train a
model and does not overwrite any synthetic variant.  The goal is to replace
hand-picked bad augmentation with a feature-distribution matching step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
SYNTH_DATA_DIR = (
    OUT_ROOT
    / "synthetic_variants"
    / "nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6"
    / "datasets"
)
BUT_SIGNALS = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_protocol_adaptation_2026_06_03"
    / "protocols"
    / "p1_current_10s_center"
    / "signals.npz"
)
BUT_ATLAS = OUT_ROOT / "original_region_boundary" / "original_region_atlas.csv"
OUT_DIR = ANALYSIS_DIR / "ptb_bad_distribution_match"
REPORT_OUT_DIR = REPORT_DIR / "ptb_bad_distribution_match"


FEATURE_NAMES = [
    "z_flat_015",
    "diff_flat_015",
    "diff_contact_015",
    "diff_lowamp_050",
    "z_zcr",
    "diff_abs",
    "z_mean_abs",
    "z_diff_abs",
    "baseline_ptp",
    "baseline_mean_abs",
    "baseline_std",
    "baseline_rms",
    "baseline_zcr",
    "baseline_lowamp_050",
    "baseline_contact_015",
    "diff_std",
    "diff_rms",
    "z_diff_rms",
    "z_diff_top10",
    "dropout_longest_run",
    "reset_edge_top5",
    "qrs_like_peak_count",
    "detail_to_baseline_ratio",
]


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(max(3, window))
    filt = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, filt, mode="valid")[: x.shape[0]].astype(np.float32)


def zcr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))


def longest_run(mask: np.ndarray) -> float:
    best = 0
    cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best) / float(max(1, mask.size))


def waveform_features(signals: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in np.asarray(signals, dtype=np.float32):
        x = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        med = float(np.median(x))
        iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
        scale = iqr if iqr > 1e-5 else float(np.std(x) + 1e-5)
        z = (x - med) / scale
        dz = np.diff(z, prepend=z[0])
        baseline = moving_average(z, 251)
        detail = z - moving_average(z, 31)
        abs_dz = np.abs(dz)
        abs_z = np.abs(z)
        abs_base = np.abs(baseline)
        low_diff = abs_dz < 0.050
        reset_top = float(np.mean(np.sort(abs_dz)[-5:])) if abs_dz.size >= 5 else float(np.max(abs_dz))
        peak_count = float(np.sum((abs_z[1:-1] > abs_z[:-2]) & (abs_z[1:-1] > abs_z[2:]) & (abs_z[1:-1] > 1.25)))
        base_rms = float(np.sqrt(np.mean(baseline * baseline)))
        detail_rms = float(np.sqrt(np.mean(detail * detail)))
        rows.append(
            [
                float(np.mean(abs_z < 0.15)),
                float(np.mean(abs_dz < 0.015)),
                float(np.mean(abs_dz < 0.015)),
                float(np.mean(low_diff)),
                zcr(z),
                float(np.mean(abs_dz)),
                float(np.mean(abs_z)),
                float(np.mean(abs_dz)),
                float(np.ptp(baseline)),
                float(np.mean(abs_base)),
                float(np.std(baseline)),
                base_rms,
                zcr(baseline),
                float(np.mean(abs_base < 0.50)),
                float(np.mean(abs_base < 0.15)),
                float(np.std(dz)),
                float(np.sqrt(np.mean(dz * dz))),
                float(np.sqrt(np.mean(dz * dz))),
                float(np.mean(np.sort(abs_dz)[-max(1, abs_dz.size // 10) :])),
                longest_run(low_diff),
                reset_top,
                peak_count / max(1.0, x.size / 125.0),
                detail_rms / (base_rms + 1e-6),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def augment_candidate(x: np.ndarray, rng: np.random.Generator, mode: str, strength: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[0]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    s = float(strength)
    if mode == "identity":
        return y
    if mode in {"baseline", "mixed", "stress"}:
        drift = float(rng.uniform(0.35, 1.65) * s)
        curve = float(rng.uniform(0.10, 0.75) * s)
        y += drift * (0.52 * t + 0.32 * np.sin(np.pi * t + float(rng.uniform(-np.pi, np.pi)))) + curve * (t * t - 0.35)
    if mode in {"dropout", "mixed", "stress"}:
        chunks = int(rng.integers(1, 5 if mode != "stress" else 7))
        for _ in range(chunks):
            width = int(rng.integers(max(30, n // 30), max(60, int(n * rng.uniform(0.18, 0.62)))))
            start = int(rng.integers(0, max(1, n - width)))
            end = start + width
            if rng.random() < 0.55:
                level = float(rng.uniform(-0.10, 0.10))
                y[start:end] = level + rng.normal(0.0, 0.005 * s, size=width).astype(np.float32)
            else:
                y[start:end] *= float(rng.uniform(0.01, 0.22))
    if mode in {"smooth", "mixed", "stress"}:
        kernel = int(rng.choice([21, 31, 51, 71]))
        filt = np.ones(kernel, dtype=np.float32) / float(kernel)
        smooth = np.convolve(np.pad(y, (kernel // 2, kernel // 2), mode="edge"), filt, mode="valid")[:n]
        mix = float(rng.uniform(0.25, 0.72) * min(1.0, 0.60 + 0.16 * s))
        y = (1.0 - mix) * y + mix * smooth
    if mode in {"reset", "mixed", "stress"}:
        for _ in range(int(rng.integers(2, 8))):
            center = int(rng.integers(8, max(9, n - 8)))
            width = int(rng.integers(4, 18))
            lo = max(0, center - width)
            hi = min(n, center + width + 1)
            local = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
            spike = np.exp(-16.0 * local * local).astype(np.float32)
            y[lo:hi] += float(rng.uniform(0.10, 0.70) * s) * spike * (1.0 if rng.random() < 0.5 else -1.0)
    if mode in {"detail", "highzcr", "rough", "mixed", "stress"}:
        # Record111-like bad is not just drift/dropout: it often has dense
        # small sign changes and local derivative energy.  Add controlled
        # alternating detail so the matcher can learn that distribution.
        hf_scale = float(rng.uniform(0.018, 0.095) * s)
        alt = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        alt = np.convolve(np.pad(alt, (1, 1), mode="edge"), np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="valid")[:n]
        y += hf_scale * alt
        bursts = int(rng.integers(3, 12 if mode != "highzcr" else 18))
        for _ in range(bursts):
            width = int(rng.integers(6, 42 if mode != "highzcr" else 30))
            start = int(rng.integers(0, max(1, n - width)))
            local = np.arange(width, dtype=np.float32)
            freq = float(rng.uniform(0.20, 0.48))
            burst = np.sin(2.0 * np.pi * freq * local + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
            taper = np.hanning(max(3, width)).astype(np.float32)
            amp = float(rng.uniform(0.035, 0.20) * s)
            y[start : start + width] += amp * burst * taper
    if mode in {"rough", "stress"} and rng.random() < 0.75:
        # Short slope flips emulate unstable electrode contact without turning
        # every row into a broad bad stress holdout.
        for _ in range(int(rng.integers(2, 7))):
            start = int(rng.integers(0, max(1, n - 80)))
            width = int(rng.integers(20, 95))
            end = min(n, start + width)
            ramp = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
            y[start:end] += float(rng.uniform(-0.18, 0.18) * s) * ramp
    if mode != "identity" and rng.random() < 0.45:
        y += rng.normal(0.0, float(rng.uniform(0.003, 0.020) * s), size=n).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.sort(a.astype(np.float64))
    y = np.sort(b.astype(np.float64))
    vals = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, vals, side="right") / max(1, len(x))
    cdf_y = np.searchsorted(y, vals, side="right") / max(1, len(y))
    return float(np.max(np.abs(cdf_x - cdf_y)))


def distribution_table(target: np.ndarray, base: np.ndarray, matched: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(FEATURE_NAMES):
        for group, arr in [("base_synthetic_bad", base), ("matched_synthetic_bad", matched)]:
            rows.append(
                {
                    "feature": name,
                    "group": group,
                    "ks_vs_but_bad": ks_stat(target[:, i], arr[:, i]),
                    "target_median": float(np.median(target[:, i])),
                    "group_median": float(np.median(arr[:, i])),
                    "target_q10": float(np.quantile(target[:, i], 0.10)),
                    "target_q90": float(np.quantile(target[:, i], 0.90)),
                    "group_q10": float(np.quantile(arr[:, i], 0.10)),
                    "group_q90": float(np.quantile(arr[:, i], 0.90)),
                }
            )
    return pd.DataFrame(rows)


def greedy_match(cand: np.ndarray, target: np.ndarray, select_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target_idx = rng.choice(np.arange(len(target)), size=select_count, replace=len(target) < select_count)
    order = rng.permutation(target_idx)
    used = np.zeros(len(cand), dtype=bool)
    selected: list[int] = []
    for ti in order:
        diff = cand - target[ti][None, :]
        dist = np.mean(diff * diff, axis=1)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if not np.isfinite(dist[j]):
            break
        used[j] = True
        selected.append(j)
    return np.asarray(selected, dtype=np.int64)


def pca2(z: np.ndarray) -> np.ndarray:
    mu = z.mean(axis=0, keepdims=True)
    sd = z.std(axis=0, keepdims=True) + 1e-6
    x = (z - mu) / sd
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def plot_pca(target: np.ndarray, base: np.ndarray, matched: np.ndarray, out_path: Path) -> None:
    all_x = np.concatenate([target, base, matched], axis=0)
    coords = pca2(all_x)
    n_t, n_b = len(target), len(base)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    ax.scatter(coords[:n_t, 0], coords[:n_t, 1], s=8, alpha=0.35, label="BUT bad target", color="#d62728")
    ax.scatter(coords[n_t : n_t + n_b, 0], coords[n_t : n_t + n_b, 1], s=8, alpha=0.25, label="PTB bad base", color="#7f7f7f")
    ax.scatter(coords[n_t + n_b :, 0], coords[n_t + n_b :, 1], s=9, alpha=0.45, label="PTB bad matched", color="#1f77b4")
    ax.set_title("Bad waveform primitive distribution: BUT target vs PTB base/matched")
    ax.set_xlabel("primitive PC1")
    ax.set_ylabel("primitive PC2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_waveforms(signals: np.ndarray, manifest: pd.DataFrame, out_path: Path) -> None:
    take = np.arange(min(12, len(signals)))
    fig, axes = plt.subplots(3, 4, figsize=(12, 6), dpi=160, sharex=True)
    t = np.arange(signals.shape[-1]) / 125.0
    for ax, idx in zip(axes.ravel(), take):
        ax.plot(t, signals[idx], lw=0.75, color="#1f77b4")
        ax.set_title(str(manifest.iloc[idx]["mode"]), fontsize=8)
        ax.set_yticks([])
        ax.grid(alpha=0.12)
    fig.suptitle("Selected PTB bad distribution-matched waveforms")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-per-bad", type=int, default=8)
    parser.add_argument("--select-count", type=int, default=1816)
    parser.add_argument("--target", type=str, default="but_bad_all", choices=["but_bad_all", "but_bad_test"])
    parser.add_argument("--seed", type=int, default=2026061907)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(SYNTH_DATA_DIR / "synth_10s_125hz_labels_with_level.csv").sort_values("idx").reset_index(drop=True)
    synth = np.load(SYNTH_DATA_DIR / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    bad_idx = labels.loc[labels["y_class"].astype(str).eq("bad"), "idx"].to_numpy(dtype=np.int64)
    base_bad = synth[bad_idx]

    atlas = pd.read_csv(BUT_ATLAS)
    but_x = np.load(BUT_SIGNALS)["X"][:, 0, :].astype(np.float32)
    if args.target == "but_bad_test":
        target_mask = atlas["class_name"].astype(str).eq("bad") & atlas["split"].astype(str).eq("test")
    else:
        target_mask = atlas["class_name"].astype(str).eq("bad")
    target_idx = atlas.loc[target_mask, "idx"].to_numpy(dtype=np.int64)
    target_bad = but_x[target_idx]

    rng = np.random.default_rng(args.seed)
    cand_signals: list[np.ndarray] = []
    cand_rows: list[dict[str, Any]] = []
    modes = ["identity", "baseline", "dropout", "smooth", "reset", "detail", "highzcr", "rough", "mixed", "stress"]
    strengths = [0.55, 0.85, 1.15, 1.45]
    for row_no, src_idx in enumerate(bad_idx):
        raw = synth[src_idx]
        cand_signals.append(raw.copy())
        cand_rows.append({"source_idx": int(src_idx), "mode": "identity", "strength": 0.0})
        for _ in range(int(args.candidate_per_bad)):
            mode = str(rng.choice(modes[1:]))
            strength = float(rng.choice(strengths) * rng.uniform(0.85, 1.15))
            cand_signals.append(augment_candidate(raw, rng, mode, strength))
            cand_rows.append({"source_idx": int(src_idx), "mode": mode, "strength": strength})
    cand = np.stack(cand_signals).astype(np.float32)
    cand_manifest = pd.DataFrame(cand_rows)

    target_feat = waveform_features(target_bad)
    base_feat = waveform_features(base_bad)
    cand_feat = waveform_features(cand)
    center = np.median(target_feat, axis=0)
    scale = np.percentile(target_feat, 75, axis=0) - np.percentile(target_feat, 25, axis=0)
    scale = np.where(scale > 1e-6, scale, np.std(target_feat, axis=0) + 1e-6)
    cand_z = (cand_feat - center[None, :]) / scale[None, :]
    target_z = (target_feat - center[None, :]) / scale[None, :]
    selected = greedy_match(cand_z, target_z, int(args.select_count), args.seed + 19)
    matched = cand[selected]
    matched_feat = cand_feat[selected]
    matched_manifest = cand_manifest.iloc[selected].copy().reset_index(drop=True)
    matched_manifest.insert(0, "matched_idx", np.arange(len(matched_manifest)))

    dist = distribution_table(target_feat, base_feat, matched_feat)
    summary = {
        "target": args.target,
        "target_count": int(len(target_bad)),
        "base_bad_count": int(len(base_bad)),
        "candidate_count": int(len(cand)),
        "selected_count": int(len(matched)),
        "base_mean_ks": float(dist.loc[dist["group"].eq("base_synthetic_bad"), "ks_vs_but_bad"].mean()),
        "matched_mean_ks": float(dist.loc[dist["group"].eq("matched_synthetic_bad"), "ks_vs_but_bad"].mean()),
        "base_top10_mean_ks": float(dist.loc[dist["group"].eq("base_synthetic_bad")].nlargest(10, "ks_vs_but_bad")["ks_vs_but_bad"].mean()),
        "matched_top10_mean_ks": float(dist.loc[dist["group"].eq("matched_synthetic_bad")].nlargest(10, "ks_vs_but_bad")["ks_vs_but_bad"].mean()),
        "selected_modes": matched_manifest["mode"].value_counts().to_dict(),
    }

    np.savez_compressed(OUT_DIR / "ptb_bad_distribution_matched_signals.npz", X=matched)
    np.savez_compressed(OUT_DIR / "ptb_bad_distribution_candidate_features.npz", features=cand_feat, selected=selected)
    matched_manifest.to_csv(OUT_DIR / "ptb_bad_distribution_matched_manifest.csv", index=False)
    dist.to_csv(OUT_DIR / "ptb_bad_distribution_match_feature_ks.csv", index=False)
    (OUT_DIR / "ptb_bad_distribution_match_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pca_path = REPORT_OUT_DIR / "ptb_bad_distribution_match_pca.png"
    wave_path = REPORT_OUT_DIR / "ptb_bad_distribution_match_waveforms.png"
    plot_pca(target_feat, base_feat, matched_feat, pca_path)
    plot_waveforms(matched, matched_manifest, wave_path)

    report_lines = [
        "# PTB Bad Distribution Match",
        "",
        "External-only distribution diagnostic/generator.  It builds a PTB bad bank selected in waveform primitive space to match BUT bad target distribution.",
        "",
        "## Summary",
        "",
        f"- Target: `{args.target}`",
        f"- BUT target bad rows: `{summary['target_count']}`",
        f"- Base PTB bad rows: `{summary['base_bad_count']}`",
        f"- Candidate PTB bad rows: `{summary['candidate_count']}`",
        f"- Selected matched rows: `{summary['selected_count']}`",
        f"- Mean KS base -> BUT bad: `{summary['base_mean_ks']:.4f}`",
        f"- Mean KS matched -> BUT bad: `{summary['matched_mean_ks']:.4f}`",
        f"- Top10 KS base -> BUT bad: `{summary['base_top10_mean_ks']:.4f}`",
        f"- Top10 KS matched -> BUT bad: `{summary['matched_top10_mean_ks']:.4f}`",
        f"- Selected modes: `{summary['selected_modes']}`",
        "",
        "## Figures",
        "",
        f"![PCA]({pca_path})",
        "",
        f"![Waveforms]({wave_path})",
        "",
        "## Files",
        "",
        f"- Matched signals: `{OUT_DIR / 'ptb_bad_distribution_matched_signals.npz'}`",
        f"- Manifest: `{OUT_DIR / 'ptb_bad_distribution_matched_manifest.csv'}`",
        f"- Feature KS: `{OUT_DIR / 'ptb_bad_distribution_match_feature_ks.csv'}`",
    ]
    report = "\n".join(report_lines) + "\n"
    (OUT_DIR / "ptb_bad_distribution_match_report.md").write_text(report, encoding="utf-8")
    (REPORT_OUT_DIR / "ptb_bad_distribution_match_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
