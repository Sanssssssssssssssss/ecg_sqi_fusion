# src/experiment/ptbxl_viz_examples_by_class_noise.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.utils.paths import project_root
except ModuleNotFoundError:
    this_file = Path(__file__).resolve()
    root = this_file.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import project_root


SEED = 0
FS = 125
WIN_SEC = 10
N = FS * WIN_SEC  # 1250

Y_CLASSES = ["good", "medium", "bad"]
NOISE_KINDS = ["em", "ma", "mix"]
K = 3  # examples per (y_class, noise_kind)


def pick_examples(df: pd.DataFrame, y: str, nk: str, k: int, rng: np.random.Generator) -> pd.DataFrame:
    sub = df[(df["y_class"] == y) & (df["noise_kind"] == nk)].copy()
    if len(sub) == 0:
        return sub
    if len(sub) <= k:
        return sub
    return sub.sample(n=k, random_state=int(rng.integers(0, 1_000_000_000)))


def main() -> None:
    root = project_root()
    lbl_path = root / "artifact1" / "datasets" / "synth_10s_125hz_labels.csv"
    clean_npz = root / "artifact1" / "datasets" / "synth_10s_125hz_clean.npz"
    noisy_npz = root / "artifact1" / "datasets" / "synth_10s_125hz_noisy.npz"

    out_dir = root / "artifact1" / "figs_for_noise_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "ptbxl_examples_by_class_noise.png"
    out_pdf = out_dir / "ptbxl_examples_by_class_noise.pdf"

    df = pd.read_csv(lbl_path)
    Xc = np.load(clean_npz)["X_clean"].astype(np.float32)
    Xn = np.load(noisy_npz)["X_noisy"].astype(np.float32)

    assert len(df) == Xc.shape[0] == Xn.shape[0], "labels rows must match npz rows"
    assert Xc.shape[1] == N and Xn.shape[1] == N, f"expected length {N}"

    rng = np.random.default_rng(SEED)
    t = np.arange(N) / FS

    nrows = len(Y_CLASSES) * len(NOISE_KINDS)  # 9
    ncols = K  # 3

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 1.9 * nrows),
        sharex=True,
        sharey=False,
    )
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, None]

    row = 0
    missing_groups: list[str] = []

    for y in Y_CLASSES:
        for nk in NOISE_KINDS:
            chosen = pick_examples(df, y, nk, K, rng)
            if len(chosen) == 0:
                missing_groups.append(f"{y}/{nk}")
                for c in range(ncols):
                    ax = axes[row, c]
                    ax.axis("off")
                    ax.set_title(f"(missing) y_class={y}, noise={nk}")
                row += 1
                continue

            chosen = chosen.reset_index(drop=True)
            if len(chosen) < K:
                last = chosen.iloc[[-1]].copy()
                chosen = pd.concat([chosen, *([last] * (K - len(chosen)))], ignore_index=True)

            for c in range(ncols):
                ax = axes[row, c]
                r = chosen.iloc[c]
                idx = int(r["idx"])
                xc = Xc[idx]
                xn = Xn[idx]

                # Add labels for legend
                ln_noisy, = ax.plot(t, xn, linewidth=1.2, label="Noisy")
                ln_clean, = ax.plot(t, xc, linestyle="--", linewidth=1.0, alpha=0.7, label="Clean")

                split = str(r.get("split", ""))
                snr = float(r.get("snr_db", float("nan")))
                seg_id = r.get("seg_id", "")
                ecg_id = r.get("ecg_id", "")

                title = (
                    f"[{split}] y_class={y} noise={nk} SNR={snr:.1f} dB\n"
                    f"idx={idx} seg_id={seg_id} ecg_id={ecg_id}"
                )
                ax.set_title(title, fontsize=9)

                # Only show legend once per row (leftmost subplot) to avoid clutter
                if c == 0:
                    ax.legend(loc="upper right", fontsize=8, frameon=True)

                if c == 0:
                    ax.set_ylabel("ampl.")
                if row == nrows - 1:
                    ax.set_xlabel("time (s)")

            row += 1

    fig.suptitle(
        f"PTB-XL Lead I | 10s windows @ {FS} Hz | {K} examples per (y_class, noise_kind)\n"
        f"solid=Noisy, dashed=Clean",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    if missing_groups:
        print(f"Warning: missing groups: {missing_groups}")


if __name__ == "__main__":
    main()