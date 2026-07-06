"""Build PTB bad-subtype-balanced synthetic protocol v12.

The current PTB bad bank is heavily dominated by contact/reset-like artifacts.
This builder keeps the existing v11 good/medium rows, replaces bad rows with
waveform-generated bad subtype blocks, recomputes waveform-computable SQI
features, and writes a protocol compatible with EventFactorizedSQIConformer.

No BUT waveform is copied into the training protocol.  BUT is used only to set
the target bad-subtype proportions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402

BASE_PROTOCOL = ANALYSIS_DIR / "event_xds_aligned_v11_buttrain_style_replay" / "protocol_ptb_buttrain_aligned_pc3000_s20260620"
BUT_REFERENCE_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_keep_outlier"
OUT_DIR = ANALYSIS_DIR / "event_xds_aligned_v12_bad_subtype_balanced"
REPORT_OUT_DIR = REPORT_DIR / "event_xds_aligned_v12_bad_subtype_balanced"
PROTOCOL_NAME = "protocol_ptb_bad_subtype_balanced_v12_pc3000_s20260621"
VERSION_LABEL = "v12"

SUBTYPE_SCRIPT = SCRIPT_DIR / "run_but_subtype_aux_experiment.py"
FEATURE_SCRIPT = SCRIPT_DIR / "waveform_feature_extractor.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUB = load_module(SUBTYPE_SCRIPT, "subtype_tools_for_ptb_v12")
FEATURES = load_module(FEATURE_SCRIPT, "waveform_features_for_ptb_v12")

CLASS_TO_INT = {"good": 0, "medium": 1, "bad": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
BAD_SUBTYPES = [
    "bad_dense_right_island",
    "bad_detector_template_disagree",
    "bad_baseline_wander_lowfreq",
    "bad_contact_reset_flatline",
    "bad_low_qrs_visibility",
    "bad_highfreq_detail_noise",
    "bad_other_boundary",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def robust_norm(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    med = float(np.median(y))
    scale = float(np.percentile(y, 95) - np.percentile(y, 5))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = float(np.std(y) + 1e-4)
    return ((y - med) / scale).astype(np.float32)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win) | 1)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(np.pad(x, (win // 2, win // 2), mode="edge"), kernel, mode="valid")[: len(x)].astype(np.float32)


def sparse_pulses(n: int, rng: np.random.Generator, count: int, amp_range: tuple[float, float], width_range: tuple[int, int]) -> np.ndarray:
    y = np.zeros(n, dtype=np.float32)
    for _ in range(int(count)):
        width = int(rng.integers(width_range[0], width_range[1] + 1))
        center = int(rng.integers(width + 1, max(width + 2, n - width - 1)))
        lo, hi = max(0, center - width), min(n, center + width + 1)
        local = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
        pulse = np.exp(-10.0 * local * local).astype(np.float32)
        y[lo:hi] += float(rng.uniform(*amp_range)) * pulse * (1.0 if rng.random() < 0.5 else -1.0)
    return y


def add_bursts(n: int, rng: np.random.Generator, count: int, amp: tuple[float, float], width: tuple[int, int], freq: tuple[float, float]) -> np.ndarray:
    y = np.zeros(n, dtype=np.float32)
    for _ in range(int(count)):
        w = int(rng.integers(width[0], width[1] + 1))
        start = int(rng.integers(0, max(1, n - w)))
        local = np.arange(w, dtype=np.float32)
        burst = np.sin(2.0 * np.pi * float(rng.uniform(*freq)) * local + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        taper = np.hanning(max(3, w)).astype(np.float32)
        y[start : start + w] += float(rng.uniform(*amp)) * burst * taper
    return y


def generate_bad_waveform(base: np.ndarray, subtype: str, rng: np.random.Generator, variant: str = "v12") -> np.ndarray:
    x = robust_norm(base)
    n = len(x)
    t = np.linspace(0.0, 10.0, n, dtype=np.float32)
    centered = np.linspace(-1.0, 1.0, n, dtype=np.float32)

    if variant == "v13_softmatch":
        # Softer, BUT-like bad: keep the mechanisms interpretable but avoid the
        # v12 failure mode where synthetic bad became a high-amplitude artifact
        # detector far outside the BUT bad feature range.
        if subtype == "bad_dense_right_island":
            y = float(rng.uniform(0.08, 0.22)) * x
            y += add_bursts(n, rng, int(rng.integers(12, 28)), (0.025, 0.095), (10, 42), (0.35, 0.49))
            y += add_bursts(n, rng, int(rng.integers(4, 10)), (0.015, 0.055), (18, 65), (0.12, 0.28))
            y += float(rng.uniform(0.05, 0.20)) * np.sin(
                2.0 * np.pi * float(rng.uniform(0.08, 0.32)) * t + float(rng.uniform(-np.pi, np.pi))
            ).astype(np.float32)
            y += rng.normal(0.0, float(rng.uniform(0.010, 0.040)), n).astype(np.float32)
            for _ in range(int(rng.integers(1, 4))):
                w = int(rng.integers(20, 85))
                s = int(rng.integers(0, max(1, n - w)))
                y[s : s + w] *= float(rng.uniform(0.35, 0.85))

        elif subtype == "bad_detector_template_disagree":
            y = float(rng.uniform(0.12, 0.28)) * x
            y += add_bursts(n, rng, int(rng.integers(5, 14)), (0.025, 0.105), (8, 32), (0.18, 0.42))
            y += sparse_pulses(n, rng, int(rng.integers(4, 12)), (0.05, 0.26), (3, 12))
            if rng.random() < 0.75:
                y = float(rng.uniform(0.55, 0.78)) * y + float(rng.uniform(0.22, 0.45)) * moving_average(
                    y, int(rng.choice([7, 11, 15, 21]))
                )
            y += rng.normal(0.0, float(rng.uniform(0.006, 0.026)), n).astype(np.float32)

        elif subtype == "bad_baseline_wander_lowfreq":
            drift = float(rng.uniform(0.16, 0.75)) * (
                0.45 * centered
                + 0.55 * np.sin(
                    2.0 * np.pi * float(rng.uniform(0.05, 0.22)) * t + float(rng.uniform(-np.pi, np.pi))
                ).astype(np.float32)
            )
            curve = float(rng.uniform(0.04, 0.24)) * (centered * centered - 0.30)
            y = float(rng.uniform(0.12, 0.30)) * x + drift + curve
            y += add_bursts(n, rng, int(rng.integers(1, 4)), (0.015, 0.070), (12, 42), (0.16, 0.34))

        elif subtype == "bad_contact_reset_flatline":
            y = float(rng.uniform(0.12, 0.30)) * x
            for _ in range(int(rng.integers(1, 4))):
                w = int(rng.integers(max(28, n // 35), max(60, n // 5)))
                s = int(rng.integers(0, max(1, n - w)))
                if rng.random() < 0.55:
                    y[s : s + w] = float(rng.uniform(-0.08, 0.08)) + rng.normal(0.0, 0.006, w).astype(np.float32)
                else:
                    y[s : s + w] *= float(rng.uniform(0.06, 0.35))
            y += sparse_pulses(n, rng, int(rng.integers(1, 5)), (0.10, 0.45), (4, 18))
            y += rng.normal(0.0, float(rng.uniform(0.004, 0.020)), n).astype(np.float32)

        elif subtype == "bad_low_qrs_visibility":
            smooth = moving_average(x, int(rng.choice([31, 45, 61])))
            y = float(rng.uniform(0.06, 0.22)) * smooth
            y += float(rng.uniform(0.08, 0.26)) * np.sin(
                2.0 * np.pi * float(rng.uniform(0.08, 0.28)) * t + float(rng.uniform(-np.pi, np.pi))
            ).astype(np.float32)
            y += add_bursts(n, rng, int(rng.integers(2, 7)), (0.012, 0.055), (14, 46), (0.22, 0.45))
            for _ in range(int(rng.integers(2, 7))):
                w = int(rng.integers(16, 52))
                s = int(rng.integers(0, max(1, n - w)))
                y[s : s + w] *= float(rng.uniform(0.08, 0.42))
            y += rng.normal(0.0, float(rng.uniform(0.004, 0.018)), n).astype(np.float32)

        elif subtype == "bad_highfreq_detail_noise":
            y = float(rng.uniform(0.08, 0.20)) * x
            y += add_bursts(n, rng, int(rng.integers(8, 20)), (0.035, 0.125), (8, 36), (0.33, 0.49))
            y += rng.normal(0.0, float(rng.uniform(0.010, 0.035)), n).astype(np.float32)

        elif subtype == "bad_other_boundary":
            modes = [
                "bad_baseline_wander_lowfreq",
                "bad_contact_reset_flatline",
                "bad_low_qrs_visibility",
                "bad_highfreq_detail_noise",
                "bad_detector_template_disagree",
            ]
            a, b = rng.choice(modes, size=2, replace=False)
            ya = generate_bad_waveform(base, str(a), rng, variant=variant)
            yb = generate_bad_waveform(base, str(b), rng, variant=variant)
            mix = float(rng.uniform(0.35, 0.65))
            y = mix * ya + (1.0 - mix) * yb

        else:
            raise ValueError(f"unknown subtype {subtype}")

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        p = float(np.percentile(np.abs(y), 99))
        if p > 1e-4:
            y = y / max(1.0, p / 1.05)
        return y.astype(np.float32)

    if subtype == "bad_dense_right_island":
        carrier = float(rng.uniform(18.0, 28.0))
        osc = np.sin(2.0 * np.pi * carrier * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        osc += float(rng.uniform(0.12, 0.35)) * np.sin(2.0 * np.pi * float(rng.uniform(9.0, 16.0)) * t).astype(np.float32)
        env = 0.80 + 0.20 * np.sin(2.0 * np.pi * float(rng.uniform(0.08, 0.28)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        y = float(rng.uniform(0.02, 0.12)) * x + float(rng.uniform(0.55, 0.95)) * osc * env
        for _ in range(int(rng.integers(2, 7))):
            w = int(rng.integers(18, 70))
            s = int(rng.integers(0, max(1, n - w)))
            y[s : s + w] *= float(rng.uniform(0.10, 0.65))
        y += rng.normal(0.0, float(rng.uniform(0.004, 0.020)), n).astype(np.float32)

    elif subtype == "bad_detector_template_disagree":
        y = 0.22 * x
        y += add_bursts(n, rng, int(rng.integers(8, 22)), (0.05, 0.18), (5, 26), (0.24, 0.49))
        y += sparse_pulses(n, rng, int(rng.integers(9, 22)), (0.08, 0.45), (3, 10))
        y += float(rng.uniform(0.03, 0.12)) * rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        if rng.random() < 0.65:
            y = 0.75 * y + 0.25 * moving_average(y, int(rng.choice([9, 15, 21])))

    elif subtype == "bad_baseline_wander_lowfreq":
        drift = float(rng.uniform(0.45, 1.80)) * (
            0.55 * centered
            + 0.35 * np.sin(2.0 * np.pi * float(rng.uniform(0.07, 0.24)) * t + float(rng.uniform(-np.pi, np.pi))).astype(np.float32)
        )
        curve = float(rng.uniform(0.10, 0.70)) * (centered * centered - 0.35)
        y = 0.28 * x + drift + curve
        y += add_bursts(n, rng, int(rng.integers(1, 5)), (0.03, 0.12), (8, 36), (0.18, 0.42))

    elif subtype == "bad_contact_reset_flatline":
        y = 0.35 * x
        for _ in range(int(rng.integers(2, 6))):
            w = int(rng.integers(max(40, n // 25), max(80, n // 3)))
            s = int(rng.integers(0, max(1, n - w)))
            if rng.random() < 0.65:
                y[s : s + w] = float(rng.uniform(-0.10, 0.10)) + rng.normal(0.0, 0.004, w).astype(np.float32)
            else:
                y[s : s + w] *= float(rng.uniform(0.02, 0.20))
        y += sparse_pulses(n, rng, int(rng.integers(2, 8)), (0.20, 0.90), (3, 16))
        if rng.random() < 0.50:
            s = int(rng.integers(n // 8, 7 * n // 8))
            y[s:] += float(rng.uniform(-0.8, 0.8))

    elif subtype == "bad_low_qrs_visibility":
        smooth = moving_average(x, int(rng.choice([41, 61, 81])))
        y = float(rng.uniform(0.35, 0.70)) * smooth
        for _ in range(int(rng.integers(4, 10))):
            w = int(rng.integers(18, 55))
            s = int(rng.integers(0, max(1, n - w)))
            y[s : s + w] *= float(rng.uniform(0.04, 0.30))
        y += float(rng.uniform(0.10, 0.45)) * np.sin(2.0 * np.pi * float(rng.uniform(0.10, 0.35)) * t).astype(np.float32)
        y += rng.normal(0.0, float(rng.uniform(0.004, 0.018)), n).astype(np.float32)

    elif subtype == "bad_highfreq_detail_noise":
        y = 0.18 * x
        y += add_bursts(n, rng, int(rng.integers(12, 28)), (0.10, 0.34), (5, 30), (0.28, 0.50))
        alt = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        alt = np.convolve(np.pad(alt, (1, 1), mode="edge"), np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="valid")[:n]
        y += float(rng.uniform(0.05, 0.18)) * alt
        y += rng.normal(0.0, float(rng.uniform(0.010, 0.040)), n).astype(np.float32)

    elif subtype == "bad_other_boundary":
        modes = [
            "bad_baseline_wander_lowfreq",
            "bad_contact_reset_flatline",
            "bad_low_qrs_visibility",
            "bad_highfreq_detail_noise",
            "bad_detector_template_disagree",
        ]
        a, b = rng.choice(modes, size=2, replace=False)
        ya = generate_bad_waveform(base, str(a), rng, variant=variant)
        yb = generate_bad_waveform(base, str(b), rng, variant=variant)
        mix = float(rng.uniform(0.35, 0.65))
        y = mix * ya + (1.0 - mix) * yb

    else:
        raise ValueError(f"unknown subtype {subtype}")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Keep amplitude in a plausible range for the existing dual-view pipeline.
    p = float(np.percentile(np.abs(y), 99))
    if p > 1e-4:
        y = y / max(1.0, p / 1.25)
    return y.astype(np.float32)


def split_counts(total: int, rng: np.random.Generator) -> dict[str, int]:
    train = int(round(0.70 * total))
    val = int(round(0.15 * total))
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def target_bad_counts(but_frame: pd.DataFrame, total_bad: int) -> dict[str, int]:
    labels = [SUB.SUBTYPES[i] for i in SUB.subtype_labels(but_frame)]
    work = but_frame.assign(_subtype=labels)
    bad_counts = work.loc[work["class_name"].astype(str).eq("bad"), "_subtype"].value_counts()
    raw = {name: float(bad_counts.get(name, 0)) for name in BAD_SUBTYPES}
    denom = sum(raw.values()) or 1.0
    counts = {name: int(round(total_bad * raw[name] / denom)) for name in BAD_SUBTYPES}
    diff = total_bad - sum(counts.values())
    order = sorted(BAD_SUBTYPES, key=lambda n: raw[n], reverse=True)
    for i in range(abs(diff)):
        counts[order[i % len(order)]] += 1 if diff > 0 else -1
    return counts


def recompute_features(signals: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["class_name"].astype(str).map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    if "split" in frame.columns:
        train_mask = frame["split"].astype(str).eq("train").to_numpy()
    else:
        train_mask = np.ones(len(frame), dtype=bool)
    primitives = FEATURES.compute_primitives(signals)
    state = FEATURES.fit_feature_state(primitives, y, train_mask)
    features = FEATURES.assemble_features(primitives, state)
    out = frame.copy()
    for col in FEATURES.FEATURE_COLUMNS:
        out[col] = features[col].to_numpy(dtype=np.float32)
    return out


def make_protocol(seed: int, total_bad: int, variant: str) -> tuple[Path, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    out_protocol = OUT_DIR / PROTOCOL_NAME
    out_protocol.mkdir(parents=True, exist_ok=True)

    base_atlas = pd.read_csv(BASE_PROTOCOL / "original_region_atlas.csv")
    base_x = np.load(BASE_PROTOCOL / "signals.npz")["X"].astype(np.float32)
    if base_x.ndim == 3:
        base_x = base_x[:, 0, :]
    but = pd.read_csv(BUT_REFERENCE_PROTOCOL / "original_region_atlas.csv")

    keep_nonbad = base_atlas.loc[base_atlas["class_name"].astype(str).isin(["good", "medium"])].copy()
    nonbad_x = base_x[keep_nonbad["idx"].astype(int).to_numpy()]
    source_pool_idx = base_atlas.index.to_numpy()
    # Use every class as a morphology seed so bad generation is not tied to the
    # previous contact-heavy bad bank.
    source_pool_x = base_x[source_pool_idx]

    target_counts = target_bad_counts(but, total_bad)
    bad_rows: list[pd.Series] = []
    bad_signals: list[np.ndarray] = []
    subtype_rows: list[dict[str, Any]] = []
    base_bad_template = base_atlas.loc[base_atlas["class_name"].astype(str).eq("bad")].iloc[0].copy()
    global_bad_i = 0
    for subtype, count in target_counts.items():
        per_split = split_counts(int(count), rng)
        for split, n in per_split.items():
            for _ in range(int(n)):
                src_i = int(rng.integers(0, len(source_pool_x)))
                y = generate_bad_waveform(source_pool_x[src_i], subtype, rng, variant=variant)
                row = base_bad_template.copy()
                row["split"] = split
                row["class_name"] = "bad"
                row["y"] = 2
                row["source_idx"] = int(10_000_000 + global_bad_i)
                row["record_id"] = f"ptbgen_{subtype}"
                row["subject_id"] = f"ptbgen_{subtype}"
                row["original_region"] = subtype
                row["ambiguous_type"] = subtype
                row["clean_policy"] = f"ptb_bad_subtype_balanced_{VERSION_LABEL}"
                row["target_bad_subtype"] = subtype
                row["generated_bad_mode"] = subtype
                row["base_pool_idx"] = int(src_i)
                bad_rows.append(row)
                bad_signals.append(y)
                subtype_rows.append({"subtype": subtype, "split": split, "n": 1})
                global_bad_i += 1

    new_x = np.concatenate([nonbad_x, np.stack(bad_signals, axis=0)], axis=0).astype(np.float32)
    new_frame = pd.concat([keep_nonbad, pd.DataFrame(bad_rows)], ignore_index=True)
    new_frame["idx"] = np.arange(len(new_frame), dtype=np.int64)
    new_frame["y"] = new_frame["class_name"].astype(str).map(CLASS_TO_INT).astype(int)
    new_frame = recompute_features(new_x, new_frame)
    subtype_labels = [SUB.SUBTYPES[i] for i in SUB.subtype_labels(new_frame)]
    new_frame["computed_subtype"] = subtype_labels
    new_frame.loc[new_frame["class_name"].astype(str).eq("bad"), "computed_subtype"] = new_frame.loc[
        new_frame["class_name"].astype(str).eq("bad"), "target_bad_subtype"
    ].astype(str)

    np.savez_compressed(out_protocol / "signals.npz", X=new_x[:, None, :].astype(np.float32))
    new_frame.to_csv(out_protocol / "original_region_atlas.csv", index=False)
    new_frame[["idx", "source_idx", "split", "class_name", "y", "record_id", "subject_id", "target_bad_subtype", "generated_bad_mode"]].to_csv(
        out_protocol / "metadata.csv", index=False
    )
    split_summary = new_frame.groupby(["split", "class_name"], dropna=False).size().reset_index(name="n")
    split_summary.to_csv(out_protocol / "split_counts.csv", index=False)
    subtype_summary = new_frame.groupby(["split", "class_name", "computed_subtype"], dropna=False).size().reset_index(name="n")
    subtype_summary.to_csv(out_protocol / "subtype_counts.csv", index=False)
    pd.DataFrame(subtype_rows).groupby(["subtype", "split"], dropna=False).size().reset_index(name="n").to_csv(
        out_protocol / "generated_bad_subtype_plan.csv", index=False
    )
    summary = {
        "created_at": now(),
        "base_protocol": str(BASE_PROTOCOL),
        "but_reference_protocol": str(BUT_REFERENCE_PROTOCOL),
        "protocol": str(out_protocol),
        "seed": int(seed),
        "rows": int(len(new_frame)),
        "total_bad": int(total_bad),
        "variant": str(variant),
        "target_bad_counts": target_counts,
        "input_contract": "waveform synthetic protocol; BUT used only for subtype proportions, no BUT waveform copied",
    }
    (out_protocol / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_protocol, new_frame


def plot_outputs(protocol: Path, frame: pd.DataFrame) -> None:
    x = np.load(protocol / "signals.npz")["X"][:, 0, :]
    subtype_counts = pd.read_csv(protocol / "subtype_counts.csv")
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    bad_counts = subtype_counts.loc[subtype_counts["class_name"].eq("bad")].groupby("computed_subtype")["n"].sum().sort_values(ascending=False)
    bad_counts.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_title(f"PTB {VERSION_LABEL} generated bad subtype counts")
    ax.set_ylabel("windows")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(REPORT_OUT_DIR / f"ptb_{VERSION_LABEL}_bad_subtype_counts.png", dpi=220)
    plt.close(fig)

    bad = frame.loc[frame["class_name"].astype(str).eq("bad")].copy()
    fig, axes = plt.subplots(len(BAD_SUBTYPES), 4, figsize=(18, 18), sharex=True, constrained_layout=True)
    fs = 125.0
    rng = np.random.default_rng(20260621)
    for r, subtype in enumerate(BAD_SUBTYPES):
        sub = bad.loc[bad["target_bad_subtype"].astype(str).eq(subtype)]
        if sub.empty:
            continue
        take = sub.sample(n=min(4, len(sub)), random_state=int(rng.integers(0, 1_000_000)))
        for c, (_, row) in enumerate(take.iterrows()):
            ax = axes[r, c]
            sig = x[int(row["idx"])]
            t = np.arange(len(sig)) / fs
            smooth = moving_average(sig, 41)
            ax.plot(t, sig, color="#a7adb5", lw=0.45, alpha=0.75)
            ax.plot(t, smooth, color="black", lw=0.9)
            ax.set_title(f"{subtype}\n{row['split']}", fontsize=8)
            ax.set_xlim(0, 10)
            ax.grid(alpha=0.12)
        axes[r, 0].set_ylabel(subtype.replace("bad_", "").replace("_", "\n"), fontsize=8)
    fig.suptitle(f"PTB {VERSION_LABEL} generated bad subtype waveform examples", fontsize=15)
    fig.savefig(REPORT_OUT_DIR / f"ptb_{VERSION_LABEL}_bad_subtype_waveforms.png", dpi=220)
    plt.close(fig)

    feature_cols = ["baseline_step", "qrs_visibility", "qrs_band_ratio", "detector_agreement", "non_qrs_diff_p95", "band_30_45"]
    but = pd.read_csv(BUT_REFERENCE_PROTOCOL / "original_region_atlas.csv")
    but_bad = but.loc[but["class_name"].astype(str).eq("bad")].copy()
    ptb_bad = bad.copy()
    rows = []
    for col in feature_cols:
        for group, df in [("BUT keep bad", but_bad), (f"PTB {VERSION_LABEL} bad", ptb_bad)]:
            rows.append({"feature": col, "group": group, "median": float(pd.to_numeric(df[col], errors="coerce").median())})
    pd.DataFrame(rows).to_csv(REPORT_OUT_DIR / f"ptb_{VERSION_LABEL}_vs_but_bad_feature_medians.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, col in zip(axes.ravel(), feature_cols):
        data = [
            pd.to_numeric(but_bad[col], errors="coerce").dropna().to_numpy(),
            pd.to_numeric(ptb_bad[col], errors="coerce").dropna().to_numpy(),
        ]
        ax.boxplot(data, tick_labels=["BUT bad", f"PTB {VERSION_LABEL} bad"], showfliers=False)
        ax.set_title(col)
        ax.grid(alpha=0.2)
    fig.suptitle(f"BUT bad vs PTB {VERSION_LABEL} bad: interpretable feature distribution sanity", fontsize=15)
    fig.savefig(REPORT_OUT_DIR / f"ptb_{VERSION_LABEL}_vs_but_bad_feature_distributions.png", dpi=220)
    plt.close(fig)


def write_report(protocol: Path, frame: pd.DataFrame) -> None:
    subtype_counts = pd.read_csv(protocol / "subtype_counts.csv")
    split_counts = pd.read_csv(protocol / "split_counts.csv")
    def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
        view = df.head(max_rows)
        cols = list(view.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        f"# PTB Bad Subtype-Balanced Synthetic Protocol {VERSION_LABEL}",
        "",
        f"- Generated: {now()}",
        f"- Protocol: `{protocol}`",
        "- Good/medium rows: inherited from v11 PTB aligned protocol.",
        "- Bad rows: regenerated from PTB waveform seeds only, using interpretable subtype modes.",
        "- BUT usage: subtype proportions and feature sanity target only; no BUT waveform is copied.",
        "",
        "## Split Counts",
        "",
        md_table(split_counts),
        "",
        "## Subtype Counts",
        "",
        md_table(subtype_counts),
        "",
        "## Figures",
        "",
        f"- Bad subtype counts: `{REPORT_OUT_DIR / f'ptb_{VERSION_LABEL}_bad_subtype_counts.png'}`",
        f"- Bad waveform examples: `{REPORT_OUT_DIR / f'ptb_{VERSION_LABEL}_bad_subtype_waveforms.png'}`",
        f"- BUT/PTB feature distributions: `{REPORT_OUT_DIR / f'ptb_{VERSION_LABEL}_vs_but_bad_feature_distributions.png'}`",
    ]
    (REPORT_OUT_DIR / f"ptb_bad_subtype_balanced_{VERSION_LABEL}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    global OUT_DIR, REPORT_OUT_DIR, PROTOCOL_NAME, VERSION_LABEL
    variant = str(args.variant)
    VERSION_LABEL = "v13" if variant == "v13_softmatch" else "v12"
    if VERSION_LABEL == "v13":
        OUT_DIR = ANALYSIS_DIR / "event_xds_aligned_v13_bad_subtype_softmatch"
        REPORT_OUT_DIR = REPORT_DIR / "event_xds_aligned_v13_bad_subtype_softmatch"
        PROTOCOL_NAME = f"protocol_ptb_bad_subtype_softmatch_v13_pc{int(args.total_bad)}_s{int(args.seed)}"
    ensure_dirs()
    protocol, frame = make_protocol(seed=int(args.seed), total_bad=int(args.total_bad), variant=variant)
    plot_outputs(protocol, frame)
    write_report(protocol, frame)
    print(protocol, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--total-bad", type=int, default=3000)
    parser.add_argument("--variant", type=str, default="v12", choices=["v12", "v13_softmatch"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

