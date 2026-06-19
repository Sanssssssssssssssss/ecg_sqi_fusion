"""Plot fixed-10s cleaned BUT waveform panels by policy/class/region."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\GPTProject2\ecg")
RUN_TAG = "e311_but_node_ladder_tuning_10s_2026_06_08"
OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
ANALYSIS_DIR = OUT_ROOT / "analysis" / "good_medium_geometry_repair"
REPORT_DIR = REPORT_ROOT / "analysis" / "good_medium_geometry_repair"
PROTOCOL_ROOT = ANALYSIS_DIR / "clean_but_protocols"
OUT_DIR = REPORT_DIR / "clean_but_protocol_visuals"


def robust_scale(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    q75 = np.percentile(x, 75)
    q25 = np.percentile(x, 25)
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < 1e-5:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < 1e-5:
        scale = 1.0
    return np.clip((x - med) / scale, -6, 6)


def select_rows(atlas: pd.DataFrame, label: str, region: str | None, n: int, seed: int) -> pd.DataFrame:
    sub = atlas[atlas["class_name"].astype(str).eq(label)].copy()
    if region is not None:
        if region == "bad_core":
            sub = sub[sub["original_region"].astype(str).ne("outlier_low_confidence")]
        else:
            sub = sub[sub["original_region"].astype(str).eq(region)]
    if sub.empty:
        return sub
    return sub.sample(n=min(n, len(sub)), random_state=seed).sort_values("idx")


def plot_policy(policy: str, n: int, seed: int) -> dict[str, str]:
    protocol_dir = PROTOCOL_ROOT / policy
    atlas = pd.read_csv(protocol_dir / "original_region_atlas.csv")
    x = np.load(protocol_dir / "signals.npz")["X"].astype(np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    panels = [
        ("good", "clean_core", "good / clean core"),
        ("good", "good_medium_overlap", "good / overlap"),
        ("medium", "clean_core", "medium / clean core"),
        ("medium", "good_medium_overlap", "medium / overlap"),
        ("bad", "bad_core", "bad / core-near-boundary"),
        ("bad", "outlier_low_confidence", "bad / outlier stress"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    time = np.arange(x.shape[1]) / 125.0
    summary_rows = []
    for ax, (label, region, title) in zip(axes, panels):
        rows = select_rows(atlas, label, region, n, seed)
        if rows.empty:
            ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_yticks([])
            continue
        offsets = np.linspace(0, max(0, len(rows) - 1) * 2.4, len(rows))
        for offset, (_, row) in zip(offsets, rows.iterrows()):
            idx = int(row["idx"])
            y = robust_scale(x[idx]) + offset
            ax.plot(time, y, lw=0.9, alpha=0.88)
            summary_rows.append(
                {
                    "policy": policy,
                    "label": label,
                    "region": region,
                    "idx": idx,
                    "source_idx": int(row.get("source_idx", -1)),
                    "record_id": str(row.get("record_id", "")),
                    "split": str(row.get("split", "")),
                }
            )
        ax.set_title(f"{title} ({len(rows)} examples)", loc="left", fontsize=10)
        ax.set_yticks([])
        ax.grid(alpha=0.18, axis="x")
    axes[-1].set_xlabel("seconds")
    fig.suptitle(f"Cleaned fixed-10s BUT examples: {policy}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{policy}_waveform_examples.png"
    pdf = OUT_DIR / f"{policy}_waveform_examples.pdf"
    csv = OUT_DIR / f"{policy}_waveform_examples_rows.csv"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(csv, index=False)
    return {"policy": policy, "png": str(png), "pdf": str(pdf), "rows": str(csv)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=str, default="margin_ge_5s_drop_outlier,margin_ge_5s_keep_outlier")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()
    outputs = [plot_policy(p.strip(), args.n, args.seed) for p in args.policies.split(",") if p.strip()]
    report = OUT_DIR / "clean_but_protocol_visuals_report.md"
    lines = [
        "# Clean BUT Protocol Waveform Visuals",
        "",
        "Fixed 10s windows only. Each trace is robust-scaled for visual comparison; labels/regions come from the materialized protocol atlas.",
        "",
    ]
    for item in outputs:
        lines.append(f"## {item['policy']}")
        lines.append("")
        lines.append(f"![{item['policy']}]({item['png']})")
        lines.append("")
        lines.append(f"Rows: `{item['rows']}`")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    for item in outputs:
        print(item["png"])


if __name__ == "__main__":
    main()
