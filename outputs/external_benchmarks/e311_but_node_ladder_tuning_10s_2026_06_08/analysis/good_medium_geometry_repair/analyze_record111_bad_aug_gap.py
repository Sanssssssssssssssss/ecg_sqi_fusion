"""Compare original-test bad-stress rows with PTB bad augmentations.

This is an external-only diagnostic. It does not train or select a model.
It asks whether our synthetic bad augmentations occupy the same waveform-derived
primitive space as the held-out BUT record-111 bad outlier stress bucket.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
STUDENT_SCRIPT = ANALYSIS_DIR / "run_waveform_geometry_student.py"
CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}


def load_student_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_waveform_geometry_student", STUDENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {STUDENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_rows(x: np.ndarray, idx: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros((0,) + x.shape[1:], dtype=np.float32)
    take = rng.choice(idx, size=min(count, len(idx)), replace=False)
    return x[take].astype(np.float32, copy=True)


def augment_rows(mod: Any, rows: np.ndarray, mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for row in rows:
        y = CLASS_TO_INT["bad"]
        if mode == "contact_bad":
            aug = mod.augment_bad_intermittent_contact_qrs(row.copy(), rng, 1.70)
        elif mode == "record111_motion_bad":
            aug = mod.augment_bad_record111_motion(row.copy(), rng, 2.55)
        elif mode == "record111_burstdrop_bad":
            aug = mod.augment_bad_record111_burst_dropout(row.copy(), rng, 1.90)
        elif mode == "highamp_bad":
            aug = mod.augment_bad_highamp_baseline_detail(row.copy(), rng, 0.85)
        else:
            aug = row.copy()
        out.append(np.nan_to_num(aug, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    return np.stack(out, axis=0) if out else rows[:0]


@torch.no_grad()
def primitive_stats(mod: Any, rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    chunks = []
    for start in range(0, len(rows), 256):
        x = torch.from_numpy(rows[start : start + 256]).float()
        chunks.append(mod.primitive_waveform_stats(x, "qrs_stress_v4").cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    z = (x - mu) / sd
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    comp = vt[:2]
    return z @ comp.T, mu.reshape(-1), sd.reshape(-1)


def top_gap_table(stats: dict[str, np.ndarray], names: list[str], target: str) -> pd.DataFrame:
    target_rows = stats[target]
    rows = []
    for name in names:
        if name == target:
            continue
        other = stats[name]
        if len(target_rows) == 0 or len(other) == 0:
            continue
        tq = np.quantile(target_rows, [0.25, 0.5, 0.75], axis=0)
        oq = np.quantile(other, [0.25, 0.5, 0.75], axis=0)
        pooled = np.nanstd(np.concatenate([target_rows, other], axis=0), axis=0) + 1e-6
        median_gap = (oq[1] - tq[1]) / pooled
        iqr_gap = ((oq[2] - oq[0]) - (tq[2] - tq[0])) / pooled
        score = np.abs(median_gap) + 0.35 * np.abs(iqr_gap)
        for idx in np.argsort(score)[-20:][::-1]:
            rows.append(
                {
                    "comparison": name,
                    "primitive_idx": int(idx),
                    "target_median": float(tq[1, idx]),
                    "other_median": float(oq[1, idx]),
                    "median_gap_sd": float(median_gap[idx]),
                    "iqr_gap_sd": float(iqr_gap[idx]),
                    "gap_score": float(score[idx]),
                }
            )
    return pd.DataFrame(rows)


def plot_pca(coords: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    colors = {
        "original_test_bad_stress": "#d62728",
        "synthetic_bad_plain": "#7f7f7f",
        "contact_bad_aug": "#1f77b4",
        "record111_motion_bad_aug": "#ff7f0e",
        "record111_burstdrop_bad_aug": "#2ca02c",
        "highamp_bad_aug": "#9467bd",
    }
    for label in sorted(set(labels)):
        mask = np.array(labels) == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.55, label=label, color=colors.get(label))
    ax.set_title("Bad-stress primitive PCA: BUT target vs PTB augmentations")
    ax.set_xlabel("primitive PC1")
    ax.set_ylabel("primitive PC2")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_waveforms(groups: dict[str, np.ndarray], out_path: Path) -> None:
    show_groups = [
        "original_test_bad_stress",
        "synthetic_bad_plain",
        "contact_bad_aug",
        "record111_motion_bad_aug",
        "record111_burstdrop_bad_aug",
    ]
    fig, axes = plt.subplots(len(show_groups), 3, figsize=(11, 8), dpi=160, sharex=True)
    t = None
    for r, group in enumerate(show_groups):
        rows = groups[group]
        if len(rows) == 0:
            continue
        if t is None:
            t = np.arange(rows.shape[-1])
        for c in range(3):
            ax = axes[r, c]
            row = rows[min(c, len(rows) - 1)]
            for ch in range(min(3, row.shape[0])):
                ax.plot(t, row[ch] + 2.5 * ch, linewidth=0.8)
            ax.set_title(group if c == 0 else f"example {c+1}", fontsize=8)
            ax.set_yticks([])
            ax.grid(alpha=0.12)
    fig.suptitle("Waveform examples: held-out bad-stress vs generated bad variants", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_student_module()
    train_ds = mod.ARCH.SyntheticWaveDataset("train", "robust3")
    norm = mod.ARCH.FeatureNorm(mean=train_ds.mean, std=train_ds.std)
    original_ds = mod.ARCH.OriginalWaveDataset(norm, "robust3")

    orig_bad_stress_idx = np.where(
        (original_ds.split == "test")
        & (original_ds.y == CLASS_TO_INT["bad"])
        & (original_ds.region == "outlier_low_confidence")
    )[0]
    synth_bad_idx = np.where(train_ds.y == CLASS_TO_INT["bad"])[0]

    base_bad = sample_rows(train_ds.x, synth_bad_idx, 320, 20261101)
    groups = {
        "original_test_bad_stress": sample_rows(original_ds.x, orig_bad_stress_idx, 320, 20261102),
        "synthetic_bad_plain": base_bad,
        "contact_bad_aug": augment_rows(mod, base_bad, "contact_bad", 20261103),
        "record111_motion_bad_aug": augment_rows(mod, base_bad, "record111_motion_bad", 20261104),
        "record111_burstdrop_bad_aug": augment_rows(mod, base_bad, "record111_burstdrop_bad", 20261105),
        "highamp_bad_aug": augment_rows(mod, base_bad, "highamp_bad", 20261106),
    }
    stats = {name: primitive_stats(mod, rows) for name, rows in groups.items()}
    all_stats = np.concatenate(list(stats.values()), axis=0)
    labels = []
    for name, values in stats.items():
        labels.extend([name] * len(values))
    coords, _, _ = pca2(all_stats)
    pca_path = REPORT_DIR / "record111_bad_aug_primitive_pca.png"
    wave_path = REPORT_DIR / "record111_bad_aug_waveform_examples.png"
    plot_pca(coords, labels, pca_path)
    plot_waveforms(groups, wave_path)

    gap = top_gap_table(stats, list(stats.keys()), "original_test_bad_stress")
    gap_path = REPORT_DIR / "record111_bad_aug_primitive_gap.csv"
    gap.to_csv(gap_path, index=False)
    out_gap = ANALYSIS_DIR / gap_path.name
    gap.to_csv(out_gap, index=False)

    payload = {
        "original_test_bad_stress_count": int(len(orig_bad_stress_idx)),
        "synthetic_bad_sampled": int(len(base_bad)),
        "pca_figure": str(pca_path),
        "waveform_figure": str(wave_path),
        "gap_table": str(gap_path),
        "groups": {k: int(len(v)) for k, v in groups.items()},
    }
    summary_path = REPORT_DIR / "record111_bad_aug_gap_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ANALYSIS_DIR / summary_path.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# Record111 Bad Augmentation Gap",
        "",
        "This diagnostic compares held-out original-test bad outlier stress rows against PTB bad augmentations in waveform-derived primitive space.",
        "",
        f"- Original test bad-stress rows: `{payload['original_test_bad_stress_count']}`",
        f"- Sampled synthetic bad rows: `{payload['synthetic_bad_sampled']}`",
        f"- PCA figure: `{pca_path}`",
        f"- Waveform examples: `{wave_path}`",
        f"- Gap table: `{gap_path}`",
    ]
    report_path = REPORT_DIR / "record111_bad_aug_gap_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    (ANALYSIS_DIR / report_path.name).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
