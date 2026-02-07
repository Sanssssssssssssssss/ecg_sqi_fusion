# src/features/qc_record84.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from src.utils.paths import project_root


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SQI_TYPES = ["iSQI", "bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI"]


def main() -> None:
    root = project_root()
    feat_path = root / "artifacts" / "features" / "record84.parquet"
    qc_dir = root / "artifacts" / "qc" / "features"
    qc_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Task 1.5 QC] read: {feat_path}")
    df = pd.read_parquet(feat_path)
    print(f"[Task 1.5 QC] rows={len(df)} cols={df.shape[1]}")

    # 7 histograms (pool all leads for each SQI type)
    for t in SQI_TYPES:
        cols = [f"{lead}__{t}" for lead in LEADS_12]
        good = df[df["y"] == 1][cols].to_numpy().ravel()
        bad = df[df["y"] == -1][cols].to_numpy().ravel()

        good = good[np.isfinite(good)]
        bad = bad[np.isfinite(bad)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(good, bins=60, alpha=0.6, label="y=+1 (acceptable)")
        ax.hist(bad, bins=60, alpha=0.6, label="y=-1 (unacceptable)")
        ax.set_title(f"{t} distribution (all leads pooled)")
        ax.set_xlabel(t)
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(qc_dir / f"hist_{t}.png", dpi=200)
        plt.close(fig)

    print(f"[Task 1.5 QC] saved 7 histograms -> {qc_dir}")

    # Single-feature AUC (84)
    feature_names = []
    for lead in LEADS_12:
        for t in SQI_TYPES:
            feature_names.append(f"{lead}__{t}")

    y01 = (df["y"].to_numpy() == 1).astype(int)

    auc_rows = []
    for name in feature_names:
        x = df[name].to_numpy()
        m = np.isfinite(x)
        if m.sum() < 10:
            auc = np.nan
        else:
            try:
                auc = float(roc_auc_score(y01[m], x[m]))
            except Exception:
                auc = np.nan
        auc_rows.append({"feature": name, "auc": auc})

    df_auc = pd.DataFrame(auc_rows).sort_values("auc", ascending=False)
    out_csv = qc_dir / "single_feature_auc.csv"
    df_auc.to_csv(out_csv, index=False)

    top20 = df_auc.dropna().head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["auc"])
    ax.set_title("Top-20 single-feature AUC")
    ax.set_xlabel("AUC")
    fig.tight_layout()
    out_png = qc_dir / "single_feature_auc_top20.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[Task 1.5 QC] saved AUC csv -> {out_csv}")
    print(f"[Task 1.5 QC] saved top20 plot -> {out_png}")

    print("\nTop 5 AUC features:")
    print(df_auc.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
