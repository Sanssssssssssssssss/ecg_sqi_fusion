# src/models/task_1_7_lm_mlp_search.py
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import torch

from src.utils.paths import project_root
from src.models.lm_mlp import LMMLP, LMConfig


# =======================
# Fixed config (no CLI)
# =======================
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

J_RANGE = list(range(2, 18))  # 2..17
MODEL_SELECT_METRIC = "val_acc"  # paper uses val accuracy; also store AUC

# Data
FEATURES_PARQUET = "record84_norm.parquet"   # from Task 1.6
SPLIT_CSV = "split_seta_seed0.csv"

# Training config (paper)
EPOCHS_MAX = 100
STOP_ERROR = 1e-5
STOP_GRAD = 1e-5

# Val early stop for final model (your requirement: train with val preventing overfit)
FINAL_PATIENCE = 15
THRESH = 0.5  # label uses 0/1; predict proba >0.5 -> 1

# LM damping behavior (stable defaults)
MU_INIT = 1e-3
MU_DEC = 0.1
MU_INC = 10.0
MAX_DAMP_TRIES = 10
# =======================


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_Xy_split(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feat_path = root / "artifacts" / "features" / FEATURES_PARQUET
    split_path = root / "artifacts" / "splits" / SPLIT_CSV

    print(f"features:  {feat_path}")
    print(f"split_csv:  {split_path}")

    df_feat = pd.read_parquet(feat_path)
    df_split = pd.read_csv(split_path)

    # unify record_id type to avoid merge error
    df_feat["record_id"] = df_feat["record_id"].astype(str)
    df_split["record_id"] = df_split["record_id"].astype(str)

    df = df_feat.merge(df_split[["record_id", "split"]], on="record_id", how="inner")

    # label mapping: y in parquet is ±1. we convert to 0/1 here globally for this task.
    y_pm1 = df["y"].astype(int).to_numpy()
    y01 = (y_pm1 == 1).astype(np.int32)

    # feature columns: exclude record_id, y, split
    drop_cols = {"record_id", "y", "split"}
    feat_cols = sorted([c for c in df.columns if c not in drop_cols])

    X = df[feat_cols].to_numpy(dtype=np.float64)
    split = df["split"].astype(str).to_numpy()

    if X.shape[1] != 84:
        raise ValueError(f"Expected 84 features, got {X.shape[1]}")

    print(f"X shape: {X.shape} (#features={X.shape[1]})")
    print(f"splits: train={(split=='train').sum()} val={(split=='val').sum()} test={(split=='test').sum()}")
    print(f"labels(train): good={int(y01[split=='train'].sum())} bad={int((split=='train').sum()-y01[split=='train'].sum())}")

    return X, y01, split, feat_cols


def metrics_from_probs(y01: np.ndarray, p: np.ndarray) -> dict:
    y01 = y01.astype(int).ravel()
    p = p.astype(np.float64).ravel()
    pred = (p > THRESH).astype(int)

    cm = confusion_matrix(y01, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    se = tp / max(1, (tp + fn))
    sp = tn / max(1, (tn + fp))

    try:
        auc = float(roc_auc_score(y01, p))
    except Exception:
        auc = float("nan")

    return {
        "acc": float(acc),
        "se": float(se),
        "sp": float(sp),
        "auc": float(auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_val_curve(df_res: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_res["J"], df_res["val_acc"], marker="o", label="val_acc")
    ax.plot(df_res["J"], df_res["val_auc"], marker="o", label="val_auc")
    ax.set_xlabel("Hidden units J")
    ax.set_ylabel("Metric")
    ax.set_title("LM-MLP architecture search (84–J–1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_roc(y01: np.ndarray, p: np.ndarray, out_png: Path) -> None:
    fpr, tpr, _ = roc_curve(y01.astype(int), p.astype(np.float64))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (test)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_confmat(y01: np.ndarray, p: np.ndarray, out_png: Path) -> None:
    pred = (p > THRESH).astype(int)
    cm = confusion_matrix(y01.astype(int), pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["bad(0)", "good(1)"])
    ax.set_yticks([0, 1], labels=["bad(0)", "good(1)"])
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    seed_all(SEED)
    root = project_root()
    print(f"device: {DEVICE}")

    # outputs
    model_dir = root / "artifacts" / "models" / "lm_mlp"
    eval_dir = root / "artifacts" / "eval"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    X, y01, split, feat_cols = load_Xy_split(root)

    Xtr = X[split == "train"]
    ytr = y01[split == "train"]
    Xva = X[split == "val"]
    yva = y01[split == "val"]
    Xte = X[split == "test"]
    yte = y01[split == "test"]

    # torch tensors (full-batch LM)
    Xtr_t = torch.tensor(Xtr, device=DEVICE, dtype=DTYPE)
    ytr_t = torch.tensor(ytr, device=DEVICE, dtype=DTYPE)
    Xva_t = torch.tensor(Xva, device=DEVICE, dtype=DTYPE)
    yva_t = torch.tensor(yva, device=DEVICE, dtype=DTYPE)
    Xte_t = torch.tensor(Xte, device=DEVICE, dtype=DTYPE)
    yte_t = torch.tensor(yte, device=DEVICE, dtype=DTYPE)

    cfg = LMConfig(
        epochs_max=EPOCHS_MAX,
        stop_error=STOP_ERROR,
        stop_grad=STOP_GRAD,
        mu_init=MU_INIT,
        mu_dec=MU_DEC,
        mu_inc=MU_INC,
        max_damping_tries=MAX_DAMP_TRIES,
    )

    # -------------------------
    # 1) search J=2..17 by val
    # -------------------------
    results = []
    stop_counts: dict[str, int] = {}

    print("\n[Task 1.7] LM-MLP search J=2..17")
    for J in J_RANGE:
        t0 = time.time()
        m = LMMLP(J=J, device=DEVICE, dtype=DTYPE, seed=SEED)

        train_summary = m.fit_lm(
            X_train=Xtr_t,
            y_train=ytr_t,
            cfg=cfg,
            X_val=Xva_t,
            y_val=yva_t,
            model_select_metric=MODEL_SELECT_METRIC,
            patience=FINAL_PATIENCE,
            threshold=THRESH,
        )

        p_val = m.predict_proba(Xva_t)
        met_val = metrics_from_probs(yva, p_val)

        row = {
            "J": int(J),
            "epochs_used": int(train_summary["epochs_used"]),
            "final_error": float(train_summary["final_error"]),
            "final_grad": float(train_summary["final_grad"]),
            "stop_reason": str(train_summary["stop_reason"]),
            "val_acc": float(met_val["acc"]),
            "val_auc": float(met_val["auc"]),
            "train_time_s": float(time.time() - t0),
        }
        results.append(row)
        stop_counts[row["stop_reason"]] = stop_counts.get(row["stop_reason"], 0) + 1

        print(
            f"  J={J:02d} | val_acc={row['val_acc']:.4f} val_auc={row['val_auc']:.4f} | "
            f"epochs={row['epochs_used']} err={row['final_error']:.3e} grad={row['final_grad']:.3e} "
            f"stop={row['stop_reason']}"
        )

    df_res = pd.DataFrame(results).sort_values("J").reset_index(drop=True)

    out_csv = model_dir / "search_J_results_seed0.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")

    out_curve = model_dir / "val_metric_vs_J_seed0.png"
    plot_val_curve(df_res, out_curve)
    print(f"[saved] {out_curve}")

    # tie-break: best val_metric, then val_auc, then smaller J
    metric_col = "val_acc" if MODEL_SELECT_METRIC == "val_acc" else "val_auc"
    df_rank = df_res.sort_values(by=[metric_col, "val_auc", "J"], ascending=[False, False, True])
    best_row = df_rank.iloc[0].to_dict()
    bestJ = int(best_row["J"])

    best_json = model_dir / "best_J_seed0.json"
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "device": str(DEVICE),
                "dtype": str(DTYPE),
                "J_star": bestJ,
                "model_select_metric": MODEL_SELECT_METRIC,
                "tie_break": "val_metric desc, then val_auc desc, then smaller J",
                "stop_counts": stop_counts,
                "lm_config": {
                    "epochs_max": EPOCHS_MAX,
                    "stop_error": STOP_ERROR,
                    "stop_grad": STOP_GRAD,
                    "mu_init": MU_INIT,
                    "mu_dec": MU_DEC,
                    "mu_inc": MU_INC,
                    "max_damping_tries": MAX_DAMP_TRIES,
                    "final_patience": FINAL_PATIENCE,
                    "threshold": THRESH,
                },
            },
            f,
            indent=2,
        )
    print(f"[saved] {best_json}")

    print("\nStop-reason counts:", stop_counts)
    print(f"Best J*={bestJ} by {MODEL_SELECT_METRIC}")

    # ----------------------------------------
    # 2) final model: train with val to prevent overfit
    #    test evaluated ONCE
    # ----------------------------------------
    seed_all(SEED)
    t0 = time.time()
    m_final = LMMLP(J=bestJ, device=DEVICE, dtype=DTYPE, seed=SEED)

    train_summary_final = m_final.fit_lm(
        X_train=Xtr_t,
        y_train=ytr_t,
        cfg=cfg,
        X_val=Xva_t,
        y_val=yva_t,
        model_select_metric=MODEL_SELECT_METRIC,
        patience=FINAL_PATIENCE,
        threshold=THRESH,
    )
    train_time = float(time.time() - t0)

    # test once
    p_test = m_final.predict_proba(Xte_t)
    met_test = metrics_from_probs(yte, p_test)

    # save model pickle
    out_model = model_dir / f"model_84-{bestJ}-1_seed0.pkl"
    with open(out_model, "wb") as f:
        payload = {
            "seed": SEED,
            "J": bestJ,
            "feature_columns": feat_cols,  # keep for reproducibility
            "threshold": THRESH,
            "model": m_final.to_pickle_dict(),
        }
        pickle.dump(payload, f)
    print(f"[saved] {out_model}")

    # save test metrics json
    out_metrics = eval_dir / "lm_mlp_test_metrics_seed0.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "device": str(DEVICE),
                "J": bestJ,
                "threshold": THRESH,
                "train_time_s": train_time,
                "train_stop": train_summary_final,
                "test_metrics": met_test,
            },
            f,
            indent=2,
        )
    print(f"[saved] {out_metrics}")

    out_roc = eval_dir / "lm_mlp_roc_seed0.png"
    plot_roc(yte, p_test, out_roc)
    print(f"[saved] {out_roc}")

    out_cm = eval_dir / "lm_mlp_confmat_seed0.png"
    plot_confmat(yte, p_test, out_cm)
    print(f"[saved] {out_cm}")

    print("\n=== FINAL TEST (once) ===")
    print(f"J*={bestJ} | acc={met_test['acc']:.4f} se={met_test['se']:.4f} sp={met_test['sp']:.4f} auc={met_test['auc']:.4f}")
    print("confmat:", met_test["confusion_matrix"])


if __name__ == "__main__":
    main()
