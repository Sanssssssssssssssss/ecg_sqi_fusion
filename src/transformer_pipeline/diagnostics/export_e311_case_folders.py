from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root


CLASS_ORDER = ("good", "medium", "bad")
NOISE_ORDER = ("em", "ma", "mix")
VARIANTS = (
    ("E3.11b", "e311b_snr_gap_e310_morph"),
    ("E3.11c", "e311c_snr_gap_relaxed_morph"),
    ("E3.11d", "e311d_snr_primary_good_guard"),
    ("E3.11e", "e311e_snr_only_visual"),
    ("E3.11f", "e311f_lite_e310_morph"),
    ("E3.11g", "e311g_lite_snr_primary"),
)
TARGET_SCORE = {"good": 0.06, "medium": 0.31, "bad": 0.55}


def main() -> None:
    args = parse_args()
    root = project_root()
    sweep_root = resolve(root, args.root_out)
    out_dir = resolve(root, args.out_dir)
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for short, variant in VARIANTS:
        artifact_dir = sweep_root / variant
        if not artifact_dir.exists():
            print(f"[skip] missing {artifact_dir}", file=sys.stderr)
            continue
        selected = export_variant_cases(
            short=short,
            variant=variant,
            artifact_dir=artifact_dir,
            out_dir=out_dir,
            split=args.split,
            examples_per_cell=args.examples_per_cell,
            csv_points=args.csv_points,
        )
        all_rows.extend(selected)

    selected_csv = out_dir / "selected_cases.csv"
    write_selected_csv(selected_csv, all_rows)
    write_index_html(out_dir / "index.html", all_rows, out_dir)
    write_readme(out_dir / "README.md", all_rows)
    print(f"cases: {len(all_rows)}")
    print(f"out_dir: {out_dir}")
    print(f"selected_csv: {selected_csv}")
    print(f"index: {out_dir / 'index.html'}")


def export_variant_cases(
    *,
    short: str,
    variant: str,
    artifact_dir: Path,
    out_dir: Path,
    split: str,
    examples_per_cell: int,
    csv_points: bool,
) -> list[dict[str, Any]]:
    dataset_dir = artifact_dir / "datasets"
    labels_path = dataset_dir / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        labels_path = dataset_dir / "synth_10s_125hz_labels.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels["split"].astype(str) == split].copy()

    clean = np.load(dataset_dir / "synth_10s_125hz_clean.npz")["X_clean"].astype(np.float32)
    noisy = np.load(dataset_dir / "synth_10s_125hz_noisy.npz")["X_noisy"].astype(np.float32)
    masks = np.load(dataset_dir / "synth_10s_125hz_local_mask.npz")
    inj_mask = masks["M"].astype(np.float32)
    critical_mask = masks["critical_mask"].astype(np.float32)
    qrs_mask = masks["qrs_mask"].astype(np.float32)
    tst_mask = masks["tst_mask"].astype(np.float32)
    noise_level_path = dataset_dir / "synth_10s_125hz_noise_level.npz"
    noise_level = np.load(noise_level_path)["P"].astype(np.float32) if noise_level_path.exists() else None

    rows: list[dict[str, Any]] = []
    for y_class in CLASS_ORDER:
        for noise_kind in NOISE_ORDER:
            selected = select_examples(labels, y_class=y_class, noise_kind=noise_kind, k=examples_per_cell)
            for case_i, row in enumerate(selected.itertuples(index=False), start=1):
                idx = int(row.idx)
                case_id = f"{variant}_{y_class}_{noise_kind}_{case_i:02d}_idx{idx:05d}"
                case_dir = out_dir / "by_variant" / variant / y_class / noise_kind
                case_dir.mkdir(parents=True, exist_ok=True)
                png_path = case_dir / f"{case_id}.png"
                csv_path = case_dir / f"{case_id}.csv" if csv_points else None
                meta_path = case_dir / f"{case_id}.json"
                plot_case(
                    row=row,
                    short=short,
                    variant=variant,
                    clean=clean[idx],
                    noisy=noisy[idx],
                    inj_mask=inj_mask[idx],
                    critical_mask=critical_mask[idx],
                    qrs_mask=qrs_mask[idx],
                    tst_mask=tst_mask[idx],
                    noise_level=None if noise_level is None else noise_level[idx],
                    out_png=png_path,
                )
                if csv_path is not None:
                    write_case_csv(
                        csv_path,
                        clean=clean[idx],
                        noisy=noisy[idx],
                        inj_mask=inj_mask[idx],
                        critical_mask=critical_mask[idx],
                        qrs_mask=qrs_mask[idx],
                        tst_mask=tst_mask[idx],
                        noise_level=None if noise_level is None else noise_level[idx],
                        fs=int(getattr(row, "fs", 125)),
                    )
                meta = row._asdict()
                meta.update(
                    {
                        "variant_short": short,
                        "variant": variant,
                        "case_id": case_id,
                        "png": png_path.as_posix(),
                        "csv": csv_path.as_posix() if csv_path is not None else "",
                        "meta_json": meta_path.as_posix(),
                    }
                )
                meta_path.write_text(json.dumps(json_safe(meta), indent=2), encoding="utf-8")
                rows.append(meta)
    return rows


def select_examples(labels: pd.DataFrame, *, y_class: str, noise_kind: str, k: int) -> pd.DataFrame:
    sub = labels[(labels["y_class"].astype(str) == y_class) & (labels["noise_kind"].astype(str) == noise_kind)].copy()
    if sub.empty:
        return sub
    target = TARGET_SCORE[y_class]
    sub["_target_dist"] = (sub["smooth_morph_score"].astype(float) - target).abs()
    sub["_snr"] = sub["measured_snr_db"].astype(float)
    if y_class == "good":
        sub = sub.sort_values(["_target_dist", "global_noise_score", "idx"], ascending=[True, True, True])
    elif y_class == "medium":
        sub = sub.sort_values(["_target_dist", "global_noise_score", "idx"], ascending=[True, False, True])
    else:
        sub = sub.sort_values(["_target_dist", "global_noise_score", "idx"], ascending=[True, False, True])
    return sub.head(k).drop(columns=["_target_dist", "_snr"])


def plot_case(
    *,
    row: Any,
    short: str,
    variant: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    inj_mask: np.ndarray,
    critical_mask: np.ndarray,
    qrs_mask: np.ndarray,
    tst_mask: np.ndarray,
    noise_level: np.ndarray | None,
    out_png: Path,
) -> None:
    fs = int(getattr(row, "fs", 125))
    t = np.arange(len(noisy), dtype=np.float32) / float(fs)
    residual = noisy - clean

    fig, axes = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2, 1.0, 1.0]})
    ax = axes[0]
    shade_mask(ax, t, critical_mask, color="#fde68a", alpha=0.35, label="critical")
    shade_mask(ax, t, inj_mask, color="#fecaca", alpha=0.25, label="injected")
    ax.plot(t, noisy, color="#2563eb", linewidth=1.0, label="noisy ECG")
    ax.plot(t, clean, color="#111827", linewidth=0.8, linestyle="--", alpha=0.75, label="clean ECG")
    ax.set_ylabel("Lead I")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, residual, color="#dc2626", linewidth=0.9, label="noisy - clean")
    ax.axhline(0, color="#6b7280", linewidth=0.7)
    ax.set_ylabel("residual")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(t, inj_mask, color="#ef4444", linewidth=1.0, label="injected mask")
    ax.plot(t, critical_mask * 0.85, color="#f59e0b", linewidth=1.0, label="critical mask")
    ax.plot(t, qrs_mask * 0.65, color="#7c3aed", linewidth=1.0, label="QRS")
    ax.plot(t, tst_mask * 0.45, color="#059669", linewidth=1.0, label="T-ST")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("masks")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[3]
    if noise_level is not None:
        ax.plot(t, noise_level, color="#0f766e", linewidth=1.0, label="noise level P")
    ax.plot(t, np.abs(residual) / (np.max(np.abs(residual)) + 1e-8), color="#fb923c", linewidth=0.7, alpha=0.8, label="|residual| norm")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("level")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    title = (
        f"{short} real ECG case | {row.y_class.upper()} | {row.noise_kind} | {row.placement} | "
        f"idx={int(row.idx)} group={int(row.counterfactual_group)} ecg={int(row.ecg_id)}\n"
        f"subtype={row.label_subtype} | SNR={float(row.measured_snr_db):.2f} dB | "
        f"smooth={float(row.smooth_morph_score):.3f} | global={float(row.global_noise_score):.3f} | "
        f"qrs={float(row.qrs_nprd):.3f} | tst={float(row.tst_nprd):.3f} | corr={float(row.beat_corr):.3f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def shade_mask(ax: plt.Axes, t: np.ndarray, mask: np.ndarray, *, color: str, alpha: float, label: str) -> None:
    above = mask > 0.5
    if not np.any(above):
        return
    start = None
    used_label = False
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(above) - 1):
            end = i if not flag else i + 1
            ax.axvspan(t[start], t[min(end - 1, len(t) - 1)], color=color, alpha=alpha, label=label if not used_label else None)
            used_label = True
            start = None


def write_case_csv(
    path: Path,
    *,
    clean: np.ndarray,
    noisy: np.ndarray,
    inj_mask: np.ndarray,
    critical_mask: np.ndarray,
    qrs_mask: np.ndarray,
    tst_mask: np.ndarray,
    noise_level: np.ndarray | None,
    fs: int,
) -> None:
    t = np.arange(len(noisy), dtype=np.float32) / float(fs)
    residual = noisy - clean
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "clean", "noisy", "residual", "injected_mask", "critical_mask", "qrs_mask", "tst_mask", "noise_level"])
        for i in range(len(noisy)):
            writer.writerow(
                [
                    f"{float(t[i]):.6f}",
                    f"{float(clean[i]):.8g}",
                    f"{float(noisy[i]):.8g}",
                    f"{float(residual[i]):.8g}",
                    f"{float(inj_mask[i]):.6g}",
                    f"{float(critical_mask[i]):.6g}",
                    f"{float(qrs_mask[i]):.6g}",
                    f"{float(tst_mask[i]):.6g}",
                    "" if noise_level is None else f"{float(noise_level[i]):.6g}",
                ]
            )


def write_selected_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = [
        "case_id",
        "variant_short",
        "variant",
        "idx",
        "split",
        "y_class",
        "label_subtype",
        "noise_kind",
        "placement",
        "counterfactual_group",
        "ecg_id",
        "measured_snr_db",
        "smooth_morph_score",
        "global_noise_score",
        "qrs_nprd",
        "tst_nprd",
        "beat_corr",
        "max_beat_nprd",
        "png",
        "csv",
    ]
    df = pd.DataFrame(rows)
    df[[c for c in cols if c in df.columns]].to_csv(path, index=False)


def write_index_html(path: Path, rows: list[dict[str, Any]], out_dir: Path) -> None:
    def rel(p: str) -> str:
        return Path(p).resolve().relative_to(out_dir.resolve()).as_posix()

    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>E3.11 Real ECG Case Folders</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}h1,h2{margin:16px 0 8px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}.card{background:white;border:1px solid #d1d5db;padding:10px}.card img{width:100%;height:auto}.meta{font-size:12px;color:#4b5563;line-height:1.35}</style>",
        "<h1>E3.11 Real ECG Case Folders</h1>",
        "<p>Each card is one real generated Lead-I ECG sample: clean/noisy/residual/masks/noise level.</p>",
    ]
    for variant in sorted({str(r["variant"]) for r in rows}):
        lines.append(f"<h2>{variant}</h2>")
        for y_class in CLASS_ORDER:
            sub = [r for r in rows if str(r["variant"]) == variant and str(r["y_class"]) == y_class]
            if not sub:
                continue
            lines.append(f"<h3>{y_class}</h3><div class='grid'>")
            for r in sub:
                lines.append(
                    "<div class='card'>"
                    f"<a href='{rel(str(r['png']))}'><img src='{rel(str(r['png']))}'></a>"
                    f"<div class='meta'><b>{r['case_id']}</b><br>"
                    f"{r['noise_kind']} / {r['placement']} / SNR={float(r['measured_snr_db']):.2f}<br>"
                    f"smooth={float(r['smooth_morph_score']):.3f}, global={float(r['global_noise_score']):.3f}, "
                    f"qrs={float(r['qrs_nprd']):.3f}, tst={float(r['tst_nprd']):.3f}, corr={float(r['beat_corr']):.3f}</div>"
                    "</div>"
                )
            lines.append("</div>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    variants = len({str(r["variant"]) for r in rows})
    text = f"""# E3.11 Real ECG Case Folders

This folder contains individual ECG case plots exported from the generated E3.11 sweep datasets.

- Variants: {variants}
- Cases: {len(rows)}
- Folder layout: `by_variant/<variant>/<good|medium|bad>/<noise_kind>/`
- Each `.png` contains clean ECG, noisy ECG, residual, injected/critical/QRS/T-ST masks, and noise level.
- Matching `.csv` files contain the time-series values for each plotted case.
- `selected_cases.csv` is the metadata index for all exported cases.
- `index.html` is a browser-friendly gallery.
"""
    path.write_text(text, encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export individual real ECG case plots for E3.11 sweep variants.")
    parser.add_argument(
        "--root_out",
        default=os.environ.get("ROOT_OUT", "outputs/transformer_e311_margin_snr_sweep"),
        help="Sweep output root containing e311* variant folders.",
    )
    parser.add_argument(
        "--out_dir",
        default=os.environ.get("CASE_OUT", "outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders"),
        help="Output directory for case folders.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples_per_cell", type=int, default=3, help="Cases per class/noise-kind cell for each variant.")
    parser.add_argument("--csv_points", action="store_true", help="Also export per-sample time-series CSV files.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
